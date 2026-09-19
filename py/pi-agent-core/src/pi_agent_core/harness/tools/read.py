"""Read tool ported from ``harness/tools/read.ts``."""

from __future__ import annotations

from typing import Any, Optional

from pi_ai.types import ImageContent, TextContent
from ...types import AgentToolResult
from ..._chord.context import Context
from ..types import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateOptions,
    ExecutionToolContext,
    get_or_throw,
)
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head
from .image import detect_supported_image_mime_type, encode_base64
from .path_utils import resolve_read_tool_path

__all__ = ["create_read_tool", "ReadToolOptions", "ReadToolDetails"]

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


class ReadToolOptions:
    """Options for the read tool factory."""

    def __init__(self, auto_resize_images: bool = True, image_processor: Any = None) -> None:
        self.auto_resize_images = auto_resize_images
        self.image_processor = image_processor


def create_read_tool(options: Optional[ReadToolOptions] = None) -> AgentHarnessTool:
    options = options or ReadToolOptions()

    async def execute(
        _tool_call_id: str,
        params: Any,
        _on_update: Any,
        tool_context: ExecutionToolContext,
        _invocation: Optional[AgentHarnessToolInvocation],
        context: Context,
    ) -> AgentToolResult:
        env = tool_context.env
        path = params["path"]
        offset = params.get("offset")
        limit = params.get("limit")
        absolute_path = await resolve_read_tool_path(env, path, context)
        data: bytes = get_or_throw(await env.read_binary_file(absolute_path, context))
        mime_type = detect_supported_image_mime_type(data)
        if mime_type:
            if options.image_processor is not None:
                processed = options.image_processor(
                    data,
                    mime_type,
                    {"autoResizeImages": options.auto_resize_images},
                    context,
                )
                if asyncio_iscoroutine(processed):
                    processed = await processed
                if not processed.get("ok"):
                    return AgentToolResult(
                        content=[TextContent(text=f"Read image file [{mime_type}]\n{processed.get('message', '')}")]
                    )
                hints = (
                    f"\n{'\n'.join(processed.get('hints') or [])}"
                    if processed.get("hints")
                    else ""
                )
                return AgentToolResult(
                    content=[
                        TextContent(text=f"Read image file [{processed.get('mimeType')}]{hints}"),
                        ImageContent(data=processed.get("data", ""), mime_type=processed.get("mimeType", "")),
                    ]
                )
            if mime_type == "image/bmp":
                return AgentToolResult(
                    content=[
                        TextContent(
                            text=(
                                "Read image file [image/bmp]\n"
                                "[Image omitted: configure an imageProcessor to convert BMP images.]"
                            )
                        )
                    ]
                )
            return AgentToolResult(
                content=[
                    TextContent(text=f"Read image file [{mime_type}]"),
                    ImageContent(data=encode_base64(data), mime_type=mime_type),
                ]
            )

        text_content = data.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, (offset - 1)) if offset else 0
        start_line_display = start_line + 1
        if start_line >= len(all_lines):
            raise RuntimeError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")

        user_limited_lines: Optional[int] = None
        if limit is not None:
            end_line = min(start_line + int(limit), len(all_lines))
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected_content)
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8", errors="replace")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            details = {"truncation": _truncation_json(truncation)}
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
            details = {"truncation": _truncation_json(truncation)}
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            details = None
        else:
            output_text = truncation.content
            details = None

        return AgentToolResult(content=[TextContent(text=output_text)], details=details)

    return AgentHarnessTool(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            "Images are sent as attachments. For text files, output is truncated to "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            "Use offset/limit for large files. When you need the full file, continue with offset until complete."
        ),
        parameters=READ_SCHEMA,
        execute=execute,
    )


def _truncation_json(truncation: Any) -> dict:
    return {
        "truncated": truncation.truncated,
        "truncatedBy": truncation.truncated_by,
        "totalLines": truncation.total_lines,
        "totalBytes": truncation.total_bytes,
        "outputLines": truncation.output_lines,
        "outputBytes": truncation.output_bytes,
        "lastLinePartial": truncation.last_line_partial,
        "firstLineExceedsLimit": truncation.first_line_exceeds_limit,
        "maxLines": truncation.max_lines,
        "maxBytes": truncation.max_bytes,
    }


def asyncio_iscoroutine(value: Any) -> bool:
    import asyncio

    return asyncio.iscoroutine(value) or asyncio.isfuture(value)
