"""Coding-agent read tool from ``core/tools/read.ts``."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from pi_agent_core.harness.utils.truncate import (
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size,
    truncate_head, utf8_byte_length,
)
from pi_agent_core.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_ai.abort import AbortSignal
from pi_ai.types import ImageContent, TextContent

from .definition import ToolDefinition, ToolExecutionContext, wrap_tool_definition
from .image import (
    ProcessImageOptions, detect_supported_image_mime_type_from_file, process_image,
)
from .path_utils import resolve_read_path_async

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


class _ReadPromptContribution(TypedDict):
    snippet: str
    guidelines: list[str]


READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION: _ReadPromptContribution = {
    "snippet": "Read file contents",
    "guidelines": ["Use read to examine files instead of cat or sed."],
}


@dataclass
class ReadOperations:
    read_file: Callable[[str], Awaitable[bytes]]
    access: Callable[[str], Awaitable[None]]
    detect_image_mime_type: Callable[[str], Awaitable[str | None]] | None = None


@dataclass
class ReadToolOptions:
    auto_resize_images: bool = True
    operations: ReadOperations | None = None


async def _local_read_file(path: str) -> bytes:
    return await asyncio.to_thread(Path(path).read_bytes)


async def _local_access(path: str) -> None:
    if not await asyncio.to_thread(os.access, path, os.R_OK):
        raise PermissionError(f"File is not readable: {path}")


_LOCAL_OPERATIONS = ReadOperations(
    _local_read_file, _local_access, detect_supported_image_mime_type_from_file,
)


def _truncation_details(value: TruncationResult) -> dict[str, object]:
    return {"truncation": {
        "content": value.content,
        "truncated": value.truncated,
        "truncatedBy": value.truncated_by,
        "totalLines": value.total_lines,
        "totalBytes": value.total_bytes,
        "outputLines": value.output_lines,
        "outputBytes": value.output_bytes,
        "lastLinePartial": value.last_line_partial,
        "firstLineExceedsLimit": value.first_line_exceeds_limit,
        "maxLines": value.max_lines,
        "maxBytes": value.max_bytes,
    }}


def create_read_tool_definition(cwd: str, options: ReadToolOptions | None = None) -> ToolDefinition:
    resize = options.auto_resize_images if options is not None else True
    operations = options.operations if options is not None and options.operations is not None else _LOCAL_OPERATIONS

    async def execute(
        _tool_call_id: str,
        params: object,
        signal: AbortSignal | None = None,
        _on_update: AgentToolUpdateCallback | None = None,
        context: ToolExecutionContext | None = None,
    ) -> AgentToolResult:
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")
        data = cast(Mapping[str, object], params)
        path = cast(str, data["path"])
        offset = cast(int | float | None, data.get("offset"))
        limit = cast(int | float | None, data.get("limit"))
        active_cwd = context.cwd if context is not None and context.cwd else cwd

        async def read() -> AgentToolResult:
            absolute_path = await resolve_read_path_async(path, active_cwd)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            await operations.access(absolute_path)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            mime_type = (await operations.detect_image_mime_type(absolute_path)
                         if operations.detect_image_mime_type is not None else None)
            note = None
            if context is not None and context.model is not None and "image" not in context.model.input:
                note = "[Current model does not support images. The image will be omitted from this request.]"
            if mime_type:
                image_bytes = await operations.read_file(absolute_path)
                processed = await process_image(image_bytes, mime_type, ProcessImageOptions(auto_resize_images=resize))
                if not processed.ok:
                    text = f"Read image file [{mime_type}]\n{processed.message}"
                    if note:
                        text += f"\n{note}"
                    return AgentToolResult(content=[TextContent(text=text)])
                text = f"Read image file [{processed.mime_type}]"
                if processed.hints:
                    text += "\n" + "\n".join(processed.hints)
                if note:
                    text += f"\n{note}"
                return AgentToolResult(content=[
                    TextContent(text=text),
                    ImageContent(data=processed.data, mime_type=processed.mime_type),
                ])
            buffer = await operations.read_file(absolute_path)
            content = buffer.decode("utf-8", errors="replace")
            all_lines = content.split("\n")
            start = max(0, int(offset - 1)) if offset else 0
            start_display = start + 1
            if start >= len(all_lines):
                raise RuntimeError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")
            user_limited_lines: int | None = None
            if limit is not None:
                end = min(start + int(limit), len(all_lines))
                selected = "\n".join(all_lines[start:end])
                user_limited_lines = end - start
            else:
                selected = "\n".join(all_lines[start:])
            truncation = truncate_head(selected)
            details: dict[str, object] | None = None
            if truncation.first_line_exceeds_limit:
                first_size = format_size(utf8_byte_length(all_lines[start]))
                output = (
                    f"[Line {start_display} is {first_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                    f"Use bash: sed -n '{start_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
                )
                details = _truncation_details(truncation)
            elif truncation.truncated:
                end_display = start_display + truncation.output_lines - 1
                next_offset = end_display + 1
                output = truncation.content
                if truncation.truncated_by == "lines":
                    output += (
                        f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                        f"Use offset={next_offset} to continue.]"
                    )
                else:
                    output += (
                        f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                    )
                details = _truncation_details(truncation)
            elif user_limited_lines is not None and start + user_limited_lines < len(all_lines):
                remaining = len(all_lines) - start - user_limited_lines
                next_offset = start + user_limited_lines + 1
                output = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            else:
                output = truncation.content
            return AgentToolResult(content=[TextContent(text=output)], details=details)

        task = asyncio.get_running_loop().create_task(read())
        if signal is None:
            return await task
        aborted: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def on_abort() -> None:
            if not aborted.done():
                aborted.set_result(None)

        signal.add_event_listener("abort", on_abort, once=True)
        if signal.aborted:
            on_abort()
        try:
            done, _pending = await asyncio.wait((task, aborted), return_when=asyncio.FIRST_COMPLETED)
            if aborted in done:
                task.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
                raise RuntimeError("Operation aborted")
            return await task
        finally:
            signal.remove_event_listener("abort", on_abort)
            if not aborted.done():
                aborted.cancel()

    return ToolDefinition(
        name="read", label="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            f"Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} "
            f"lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. "
            "When you need the full file, continue with offset until complete."
        ),
        parameters=READ_SCHEMA,
        constrained_sampling={"type": "json_schema", "strict": "prefer"},
        prompt_snippet=READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        prompt_guidelines=list(READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
        execute=execute,
    )


def create_read_tool(cwd: str, options: ReadToolOptions | None = None) -> AgentTool:
    return wrap_tool_definition(create_read_tool_definition(cwd, options))


__all__ = [
    "READ_SCHEMA", "READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION", "ReadOperations",
    "ReadToolOptions", "create_read_tool_definition", "create_read_tool",
]
