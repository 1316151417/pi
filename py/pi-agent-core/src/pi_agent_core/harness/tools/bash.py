"""Bash tool ported from ``harness/tools/bash.ts``."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Optional

from ..._pi_ai.types import TextContent
from ...types import AgentToolResult
from ..._chord.context import Context
from ..types import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateOptions,
    ExecutionToolContext,
)
from ..utils.output_capture import apply_shell_output_update
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size

__all__ = ["create_bash_tool", "BashToolOptions", "BashExecution"]

MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
BASH_CHECKPOINT_INTERVAL_MS = 2_000

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}


class BashExecution:
    def __init__(self, command: str, cwd: str, env: dict, inherit_env: bool) -> None:
        self.command = command
        self.cwd = cwd
        self.env = env
        self.inherit_env = inherit_env


class BashToolOptions:
    def __init__(
        self,
        command_prefix: Optional[str] = None,
        prepare: Optional[Callable] = None,
    ) -> None:
        self.command_prefix = command_prefix
        self.prepare = prepare


def _validate_timeout(timeout: Any) -> None:
    if timeout is None:
        return
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise RuntimeError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


def create_bash_tool(options: Optional[BashToolOptions] = None) -> AgentHarnessTool:
    options = options or BashToolOptions()

    async def execute(
        _tool_call_id: str,
        params: Any,
        on_update: Any,
        tool_context: ExecutionToolContext,
        _invocation: Optional[AgentHarnessToolInvocation],
        context: Context,
    ) -> AgentToolResult:
        command = params["command"]
        timeout = params.get("timeout")
        _validate_timeout(timeout)
        env = tool_context.env
        execution = BashExecution(
            command=f"{options.command_prefix}\n{command}" if options.command_prefix else command,
            cwd=env.cwd,
            env={},
            inherit_env=True,
        )
        if options.prepare is not None:
            prepared = options.prepare(execution, tool_context, context)
            import asyncio

            if asyncio.iscoroutine(prepared):
                await prepared

        view: Optional[Any] = None
        last_checkpoint_at = time.monotonic() * 1000
        last_checkpoint: Optional[str] = None
        accepting_updates = True

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        def _on_exec_update(update: Any, _context: Context) -> None:
            nonlocal view, last_checkpoint_at, last_checkpoint
            if not accepting_updates:
                return
            view = apply_shell_output_update(view, update)
            snapshot = AgentToolResult(
                content=[TextContent(text=view.text)],
                details={
                    "truncation": _truncation_to_json(view.truncation) if view.truncation.truncated else None,
                    "fullOutputPath": view.spill_path,
                },
            )
            now = time.monotonic() * 1000
            encoded = json.dumps(_snapshot_to_json(snapshot), default=str)
            is_checkpoint = (
                now - last_checkpoint_at >= BASH_CHECKPOINT_INTERVAL_MS and encoded != last_checkpoint
            )
            if on_update is not None:
                on_update(snapshot, AgentHarnessToolUpdateOptions(checkpoint=True) if is_checkpoint else None)
            if is_checkpoint:
                last_checkpoint_at = now
                last_checkpoint = encoded

        from ..types import ShellExecOptions, ShellOutputCaptureOptions, ShellOutputLimits

        exec_options = ShellExecOptions(
            cwd=execution.cwd,
            env=dict(execution.env),
            inherit_env=execution.inherit_env,
            timeout=timeout,
            capture=ShellOutputCaptureOptions(
                limits=ShellOutputLimits(max_bytes=DEFAULT_MAX_BYTES, max_lines=DEFAULT_MAX_LINES, retain="tail"),
                spill=True,
            ),
            on_update=_on_exec_update,
        )
        result = await env.exec(execution.command, exec_options, context)
        accepting_updates = False

        output_text = view.text if view is not None else ""
        if result.ok:
            truncation = result.value.truncation
            spill_path = result.value.spill_path
            last_line_bytes = result.value.last_line_bytes
        else:
            truncation = view.truncation if view is not None else None
            spill_path = view.spill_path if view is not None else None
            last_line_bytes = view.last_line_bytes if view is not None else None

        details: Optional[dict] = None
        if truncation is not None and truncation.truncated:
            details = {
                "truncation": _truncation_to_json(truncation),
                "fullOutputPath": spill_path,
            }
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line_size = format_size(last_line_bytes if last_line_bytes is not None else truncation.output_bytes)
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} "
                    f"(line is {last_line_size}). Full output: {spill_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. Full output: {spill_path}]"
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {spill_path}]"
                )

        if not result.ok:
            error_code = getattr(result.error, "code", "unknown")
            if error_code == "timeout":
                status = f"Command timed out after {timeout} seconds"
            elif error_code == "aborted":
                status = "Command aborted"
            else:
                status = getattr(result.error, "message", str(result.error))
            raise RuntimeError(f"{output_text}\n\n{status}" if output_text else status) from result.error

        if result.value.exit_code != 0:
            raise RuntimeError(
                f"{output_text}\n\nCommand exited with code {result.value.exit_code}"
                if output_text
                else f"Command exited with code {result.value.exit_code}"
            )

        return AgentToolResult(content=[TextContent(text=output_text or "(no output)")], details=details)

    return AgentHarnessTool(
        name="bash",
        label="bash",
        description=(
            "Execute a bash command in the current working directory. Returns combined stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds."
        ),
        parameters=BASH_SCHEMA,
        execute=execute,
    )


def _truncation_to_json(truncation: Any) -> dict:
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


def _snapshot_to_json(snapshot: AgentToolResult) -> dict:
    return {
        "content": [{"type": getattr(b, "type", ""), "text": getattr(b, "text", "")} for b in snapshot.content],
        "details": snapshot.details,
    }
