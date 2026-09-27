"""Coding-agent edit tool from ``core/tools/edit.ts``."""

from __future__ import annotations

import asyncio
import errno
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from pi_agent_core.harness.tools.edit_diff import (
    Edit, apply_edits_to_normalized_content, detect_line_ending,
    generate_diff_string, generate_unified_patch, normalize_to_lf,
    restore_line_endings, strip_bom,
)
from pi_agent_core.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_ai.abort import AbortSignal
from pi_ai.types import TextContent

from .definition import ToolDefinition, ToolExecutionContext, wrap_tool_definition
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_to_cwd

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {
            "type": "array",
            "description": (
                "One or more targeted replacements. Each edit is matched against the original file, "
                "not incrementally. Do not include overlapping or nested edits. If two changes touch "
                "the same block or nearby lines, merge them into one edit instead."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "oldText": {
                        "type": "string",
                        "description": (
                            "Exact text for one targeted replacement. It must be unique in the original file "
                            "and must not overlap with any other edits[].oldText in the same call."
                        ),
                    },
                    "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                },
                "required": ["oldText", "newText"],
            },
        },
    },
    "required": ["path", "edits"],
}


class _EditPromptContribution(TypedDict):
    snippet: str
    guidelines: list[str]


EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION: _EditPromptContribution = {
    "snippet": "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
    "guidelines": [
        "Use edit for precise changes (edits[].oldText must match exactly)",
        "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
    ],
}


@dataclass
class EditOperations:
    read_file: Callable[[str], Awaitable[bytes]]
    write_file: Callable[[str, str], Awaitable[None]]
    access: Callable[[str], Awaitable[None]]


@dataclass
class EditToolOptions:
    operations: EditOperations | None = None


async def _local_read_file(path: str) -> bytes:
    return await asyncio.to_thread(Path(path).read_bytes)


async def _local_write_file(path: str, content: str) -> None:
    write = asyncio.create_task(asyncio.to_thread(Path(path).write_text, content, encoding="utf-8"))
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        await write
        raise


async def _local_access(path: str) -> None:
    await asyncio.to_thread(os.stat, path)
    if not await asyncio.to_thread(os.access, path, os.R_OK | os.W_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)


_LOCAL_OPERATIONS = EditOperations(_local_read_file, _local_write_file, _local_access)


def _is_single_edit(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("oldText"), str)
        and isinstance(value.get("newText"), str)
    )


def prepare_edit_arguments(input_value: object) -> object:
    """Accept JSON-string, singleton, and legacy edit arguments from models."""
    if not isinstance(input_value, dict):
        return input_value
    args: dict[str, object] = dict(input_value)
    raw_edits = args.get("edits")
    if isinstance(raw_edits, str):
        try:
            parsed: object = json.loads(raw_edits)
        except (ValueError, TypeError):
            pass
        else:
            if isinstance(parsed, list):
                args["edits"] = parsed
            elif _is_single_edit(parsed):
                args["edits"] = [parsed]
    elif _is_single_edit(raw_edits):
        args["edits"] = [raw_edits]

    old_text, new_text = args.get("oldText"), args.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = args.get("edits")
        combined = list(edits) if isinstance(edits, list) else []
        combined.append({"oldText": old_text, "newText": new_text})
        args.pop("oldText")
        args.pop("newText")
        args["edits"] = combined
    return args


def create_edit_tool_definition(cwd: str, options: EditToolOptions | None = None) -> ToolDefinition:
    operations = options.operations if options is not None and options.operations is not None else _LOCAL_OPERATIONS

    async def execute(
        _tool_call_id: str,
        params: object,
        signal: AbortSignal | None = None,
        _on_update: AgentToolUpdateCallback | None = None,
        context: ToolExecutionContext | None = None,
    ) -> AgentToolResult:
        data = cast(Mapping[str, object], params)
        raw_edits = data.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            raise RuntimeError("Edit tool input is invalid. edits must contain at least one replacement.")
        path = cast(str, data["path"])
        edits = [
            Edit(old_text=cast(str, cast(Mapping[str, object], item)["oldText"]),
                 new_text=cast(str, cast(Mapping[str, object], item)["newText"]))
            for item in raw_edits
        ]
        absolute_path = resolve_to_cwd(path, context.cwd if context is not None and context.cwd else cwd)

        async def mutate() -> AgentToolResult:
            def throw_if_aborted() -> None:
                if signal is not None and signal.aborted:
                    raise RuntimeError("Operation aborted")

            throw_if_aborted()
            try:
                await operations.access(absolute_path)
            except Exception as error:
                throw_if_aborted()
                if isinstance(error, OSError) and error.errno is not None:
                    message = f"Error code: {errno.errorcode.get(error.errno, error.errno)}"
                else:
                    message = str(error)
                raise RuntimeError(f"Could not edit file: {path}. {message}.") from error
            throw_if_aborted()

            buffer = await operations.read_file(absolute_path)
            raw_content = buffer.decode("utf-8", errors="replace")
            throw_if_aborted()
            bom, content = strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            throw_if_aborted()

            final_content = bom + restore_line_endings(applied.new_content, original_ending)
            await operations.write_file(absolute_path, final_content)
            throw_if_aborted()

            diff, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
            patch = generate_unified_patch(path, applied.base_content, applied.new_content)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details={"diff": diff, "patch": patch, "firstChangedLine": first_changed_line},
            )

        return await with_file_mutation_queue(absolute_path, mutate)

    return ToolDefinition(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby "
            "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
            "unchanged regions just to connect distant changes."
        ),
        parameters=EDIT_SCHEMA,
        constrained_sampling={"type": "json_schema", "strict": "prefer"},
        prompt_snippet=EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        prompt_guidelines=list(EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
        prepare_arguments=prepare_edit_arguments,
        execute=execute,
    )


def create_edit_tool(cwd: str, options: EditToolOptions | None = None) -> AgentTool:
    return wrap_tool_definition(create_edit_tool_definition(cwd, options))


__all__ = [
    "EDIT_SCHEMA", "EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION", "EditOperations",
    "EditToolOptions", "prepare_edit_arguments", "create_edit_tool_definition",
    "create_edit_tool",
]
