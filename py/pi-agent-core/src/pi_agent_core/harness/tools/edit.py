"""Edit tool ported from ``harness/tools/edit.ts``."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from pi_ai.types import TextContent
from ...types import AgentToolResult
from ..._chord.context import Context
from ..types import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    ExecutionToolContext,
    FileError,
    get_or_throw,
)
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path

__all__ = ["create_edit_tool", "prepare_edit_arguments"]

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {
            "type": "array",
            "description": (
                "One or more targeted replacements. Each edit is matched against the original file, not "
                "incrementally. Do not include overlapping or nested edits. If two changes touch the same "
                "block or nearby lines, merge them into one edit instead."
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


def _is_single_edit_input(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("oldText"), str) and isinstance(value.get("newText"), str)


def prepare_edit_arguments(value: Any) -> Any:
    """Normalize edit arguments: stringified edits, single-edit forms, legacy top-level oldText/newText."""
    if not isinstance(value, dict):
        return value
    args = dict(value)
    if isinstance(args.get("edits"), str):
        try:
            parsed = json.loads(args["edits"])
            if isinstance(parsed, list):
                args["edits"] = parsed
            elif _is_single_edit_input(parsed):
                args["edits"] = [parsed]
        except json.JSONDecodeError:
            pass
    elif _is_single_edit_input(args.get("edits")):
        args["edits"] = [args["edits"]]

    if isinstance(args.get("oldText"), str) and isinstance(args.get("newText"), str):
        edits = list(args.get("edits") or [])
        edits.append({"oldText": args.pop("oldText"), "newText": args.pop("newText")})
        args["edits"] = edits
    return args


def _validate_edit_input(value: Any) -> tuple:
    edits = getattr(value, "edits", None) if not isinstance(value, dict) else value.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise RuntimeError("Edit tool input is invalid. edits must contain at least one replacement.")
    return value["path"], [Edit(old_text=e["oldText"], new_text=e["newText"]) for e in edits]


def _edit_access_error(path: str, error: FileError) -> RuntimeError:
    wrapped = RuntimeError(f"Could not edit file: {path}. Error code: {error.code}.")
    wrapped.__cause__ = error
    return wrapped


def create_edit_tool() -> AgentHarnessTool:
    async def execute(
        _tool_call_id: str,
        params: Any,
        _on_update: Any,
        tool_context: ExecutionToolContext,
        _invocation: Optional[AgentHarnessToolInvocation],
        context: Context,
    ) -> AgentToolResult:
        env = tool_context.env
        path, edits = _validate_edit_input(params)

        async def _mutate() -> AgentToolResult:
            if context.signal is not None and context.signal.aborted:
                raise RuntimeError("Operation aborted")
            absolute_path = await resolve_tool_path(env, path, context)
            info = await env.file_info(absolute_path, context)
            if not info.ok:
                raise _edit_access_error(path, info.error)
            if info.value.kind not in ("file", "symlink"):
                raise RuntimeError(f"Could not edit file: {path}. Path is not a file.")

            read_result = await env.read_text_file(absolute_path, context)
            if not read_result.ok:
                raise _edit_access_error(path, read_result.error)
            if context.signal is not None and context.signal.aborted:
                raise RuntimeError("Operation aborted")

            bom, content = strip_bom(read_result.value)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            if context.signal is not None and context.signal.aborted:
                raise RuntimeError("Operation aborted")

            final_content = bom + restore_line_endings(applied.new_content, original_ending)
            write_result = await env.write_file(absolute_path, final_content, context)
            if not write_result.ok:
                raise _edit_access_error(path, write_result.error)
            if context.signal is not None and context.signal.aborted:
                raise RuntimeError("Operation aborted")

            diff_text, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details={
                    "diff": diff_text,
                    "patch": generate_unified_patch(path, applied.base_content, applied.new_content),
                    "firstChangedLine": first_changed_line,
                },
            )

        return await with_file_mutation_queue(env, path, _mutate, context)

    return AgentHarnessTool(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby "
            "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
            "unchanged regions just to connect distant changes."
        ),
        parameters=EDIT_SCHEMA,
        prepare_arguments=prepare_edit_arguments,
        execute=execute,
    )
