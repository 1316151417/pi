"""Streaming bash tool from ``core/tools/bash.ts``."""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

from pi_agent_core.harness.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size
from pi_agent_core.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_ai.abort import AbortSignal
from pi_ai.types import TextContent

from pi_coding_agent.shell import ShellConfig, get_shell_config, get_shell_env, kill_process_tree

from .definition import ToolDefinition, ToolExecutionContext, wrap_tool_definition
from .output_accumulator import OutputAccumulator, OutputAccumulatorOptions, OutputSnapshot

MAX_TIMEOUT_MS = 2_147_483_647
BASH_UPDATE_THROTTLE_MS = 100

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}


class _BashPromptContribution(TypedDict):
    snippet: str
    guidelines: list[str]


BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION: _BashPromptContribution = {
    "snippet": "Execute bash commands (ls, grep, find, etc.)",
    "guidelines": ["You can inspect PI_* environment variables for current model and session details."],
}


def _resolve_timeout_seconds(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("Invalid timeout: must be a finite number of seconds")
    if timeout * 1000 > MAX_TIMEOUT_MS:
        raise RuntimeError(f"Invalid timeout: maximum is {MAX_TIMEOUT_MS / 1000} seconds")
    return timeout


@dataclass
class BashExecOptions:
    on_data: Callable[[bytes], None]
    signal: AbortSignal | None = None
    timeout: float | None = None
    env: dict[str, str] | None = None


@dataclass
class BashExecResult:
    exit_code: int | None


@dataclass
class BashOperations:
    exec: Callable[[str, str, BashExecOptions], Awaitable[BashExecResult]]


def create_local_shell_operations(
    shell_name: str, resolve_shell_config: Callable[[], ShellConfig],
) -> BashOperations:
    async def execute(command: str, cwd: str, options: BashExecOptions) -> BashExecResult:
        timeout = _resolve_timeout_seconds(options.timeout)
        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("aborted")
        shell_config = resolve_shell_config()
        if not os.path.exists(cwd):
            raise RuntimeError(f"Working directory does not exist: {cwd}\nCannot execute {shell_name} commands.")

        command_from_stdin = shell_config.command_transport == "stdin"
        args = shell_config.args if command_from_stdin else [*shell_config.args, command]
        process = await asyncio.create_subprocess_exec(
            shell_config.shell, *args, cwd=cwd,
            env=options.env if options.env is not None else get_shell_env(),
            stdin=asyncio.subprocess.PIPE if command_from_stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        if command_from_stdin and process.stdin is not None:
            process.stdin.write(command.encode("utf-8"))
            try:
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.stdin.close()

        activity = asyncio.Event()

        async def drain(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while chunk := await stream.read(65536):
                options.on_data(chunk)
                activity.set()

        readers = [asyncio.create_task(drain(process.stdout)), asyncio.create_task(drain(process.stderr))]
        timed_out = False
        timer: asyncio.TimerHandle | None = None

        def abort() -> None:
            if process.pid is not None:
                kill_process_tree(process.pid)

        def on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            abort()

        if timeout is not None:
            timer = asyncio.get_running_loop().call_later(timeout, on_timeout)
        if options.signal is not None:
            options.signal.add_event_listener("abort", abort, once=True)
            if options.signal.aborted:
                abort()

        try:
            await process.wait()
            # A detached child may inherit the pipes after the shell exits. Keep
            # reading while data arrives, then release quiet inherited handles.
            while not all(reader.done() for reader in readers):
                activity.clear()
                try:
                    await asyncio.wait_for(activity.wait(), 0.1)
                except TimeoutError:
                    break
            for reader in readers:
                if reader.done():
                    reader.result()
            if options.signal is not None and options.signal.aborted:
                raise RuntimeError("aborted")
            if timed_out:
                raise RuntimeError(f"timeout:{options.timeout}")
            code = process.returncode
            if code is None:
                return BashExecResult(1)
            if code < 0:
                return BashExecResult(128 + abs(code))
            return BashExecResult(code)
        finally:
            if timer is not None:
                timer.cancel()
            if options.signal is not None:
                options.signal.remove_event_listener("abort", abort)
            if process.returncode is None:
                abort()
                await process.wait()
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    return BashOperations(execute)


def create_local_bash_operations(*, shell_path: str | None = None) -> BashOperations:
    return create_local_shell_operations("bash", lambda: get_shell_config(shell_path))


@dataclass
class BashSpawnContext:
    command: str
    cwd: str
    env: dict[str, str]


type BashSpawnHook = Callable[[BashSpawnContext], BashSpawnContext]


@dataclass
class BashToolOptions:
    operations: BashOperations | None = None
    command_prefix: str | None = None
    shell_path: str | None = None
    expose_session_environment: bool = True
    spawn_hook: BashSpawnHook | None = None


@dataclass
class ShellToolConfig:
    name: str
    label: str
    shell_name: str
    prompt: str
    prompt_snippet: str
    prompt_guidelines: list[str]
    temp_file_prefix: str


def _resolve_spawn_context(
    command: str, cwd: str, hook: BashSpawnHook | None,
    expose_session_environment: bool, context: ToolExecutionContext | None,
) -> BashSpawnContext:
    env = get_shell_env()
    for name in ("PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL"):
        env.pop(name, None)
    if expose_session_environment and context is not None:
        if context.session_id:
            env["PI_SESSION_ID"] = context.session_id
        if context.session_file:
            env["PI_SESSION_FILE"] = context.session_file
        if context.model is not None:
            env["PI_PROVIDER"] = context.model.provider
            env["PI_MODEL"] = context.model.id
        if context.thinking_level:
            env["PI_REASONING_LEVEL"] = context.thinking_level
    base = BashSpawnContext(command, cwd, env)
    return hook(base) if hook is not None else base


def _truncation_details(value: TruncationResult) -> dict[str, object]:
    return {
        "content": value.content, "truncated": value.truncated,
        "truncatedBy": value.truncated_by, "totalLines": value.total_lines,
        "totalBytes": value.total_bytes, "outputLines": value.output_lines,
        "outputBytes": value.output_bytes, "lastLinePartial": value.last_line_partial,
        "firstLineExceedsLimit": value.first_line_exceeds_limit,
        "maxLines": value.max_lines, "maxBytes": value.max_bytes,
    }


def create_shell_tool_definition(
    cwd: str, config: ShellToolConfig, options: BashToolOptions | None = None,
) -> ToolDefinition:
    chosen = options or BashToolOptions()
    operations = chosen.operations or create_local_bash_operations(shell_path=chosen.shell_path)

    async def execute(
        _tool_call_id: str, params: object,
        signal: AbortSignal | None = None,
        on_update: AgentToolUpdateCallback | None = None,
        context: ToolExecutionContext | None = None,
    ) -> AgentToolResult:
        data = cast(Mapping[str, object], params)
        command = cast(str, data["command"])
        timeout = cast(float | None, data.get("timeout"))
        resolved_command = f"{chosen.command_prefix}\n{command}" if chosen.command_prefix else command
        spawn_context = _resolve_spawn_context(
            resolved_command, context.cwd if context is not None and context.cwd else cwd,
            chosen.spawn_hook, chosen.expose_session_environment, context,
        )
        output = OutputAccumulator(OutputAccumulatorOptions(temp_file_prefix=config.temp_file_prefix))
        accepting_output = True
        update_timer: asyncio.TimerHandle | None = None
        update_dirty = False
        last_update_at = 0.0

        def details_for(snapshot: OutputSnapshot) -> dict[str, object] | None:
            if not snapshot.truncation.truncated and snapshot.full_output_path is None:
                return None
            details: dict[str, object] = {}
            if snapshot.truncation.truncated:
                details["truncation"] = _truncation_details(snapshot.truncation)
            if snapshot.full_output_path is not None:
                details["fullOutputPath"] = snapshot.full_output_path
            return details

        def emit_output_update() -> None:
            nonlocal update_dirty, last_update_at
            if on_update is None or not update_dirty:
                return
            update_dirty = False
            last_update_at = time.monotonic()
            snapshot = output.snapshot(persist_if_truncated=True)
            on_update(AgentToolResult(
                content=[TextContent(text=snapshot.content)], details=details_for(snapshot),
            ))

        def clear_update_timer() -> None:
            nonlocal update_timer
            if update_timer is not None:
                update_timer.cancel()
                update_timer = None

        def schedule_output_update() -> None:
            nonlocal update_dirty, update_timer
            if on_update is None:
                return
            update_dirty = True
            delay = BASH_UPDATE_THROTTLE_MS / 1000 - (time.monotonic() - last_update_at)
            if delay <= 0:
                clear_update_timer()
                emit_output_update()
            elif update_timer is None:
                def flush_update() -> None:
                    nonlocal update_timer
                    update_timer = None
                    emit_output_update()

                update_timer = asyncio.get_running_loop().call_later(delay, flush_update)

        if on_update is not None:
            on_update(AgentToolResult(content=[]))

        def handle_data(chunk: bytes) -> None:
            if accepting_output:
                output.append(chunk)
                schedule_output_update()

        def finish_output() -> OutputSnapshot:
            nonlocal accepting_output
            accepting_output = False
            output.finish()
            clear_update_timer()
            emit_output_update()
            snapshot = output.snapshot(persist_if_truncated=True)
            output.close_temp_file()
            return snapshot

        def format_output(snapshot: OutputSnapshot, empty_text: str = "(no output)") -> tuple[str, dict[str, object] | None]:
            truncation = snapshot.truncation
            result_text = snapshot.content or empty_text
            details = details_for(snapshot) if truncation.truncated else None
            if truncation.truncated:
                start = truncation.total_lines - truncation.output_lines + 1
                end = truncation.total_lines
                if truncation.last_line_partial:
                    size = format_size(output.get_last_line_bytes())
                    result_text += (
                        f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end} "
                        f"(line is {size}). Full output: {snapshot.full_output_path}]"
                    )
                elif truncation.truncated_by == "lines":
                    result_text += f"\n\n[Showing lines {start}-{end} of {truncation.total_lines}. Full output: {snapshot.full_output_path}]"
                else:
                    result_text += (
                        f"\n\n[Showing lines {start}-{end} of {truncation.total_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {snapshot.full_output_path}]"
                    )
            return result_text, details

        def append_status(text: str, status: str) -> str:
            return f"{text}\n\n{status}" if text else status

        try:
            try:
                result = await operations.exec(
                    spawn_context.command, spawn_context.cwd,
                    BashExecOptions(handle_data, signal, timeout, spawn_context.env),
                )
            except Exception as error:
                snapshot = finish_output()
                text, _ = format_output(snapshot, "")
                if str(error) == "aborted":
                    raise RuntimeError(append_status(text, "Command aborted")) from error
                if str(error).startswith("timeout:"):
                    timeout_seconds = str(error).split(":", 1)[1]
                    raise RuntimeError(append_status(text, f"Command timed out after {timeout_seconds} seconds")) from error
                raise
            snapshot = finish_output()
            text, details = format_output(snapshot)
            if result.exit_code is None:
                raise RuntimeError(append_status(text, "Command terminated without an exit code"))
            if result.exit_code != 0:
                raise RuntimeError(append_status(text, f"Command exited with code {result.exit_code}"))
            return AgentToolResult(content=[TextContent(text=text)], details=details)
        finally:
            clear_update_timer()
            output.close_temp_file()

    return ToolDefinition(
        name=config.name, label=config.label,
        description=(
            f"Execute a {config.shell_name} command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds."
        ),
        parameters=BASH_SCHEMA, constrained_sampling={"type": "json_schema", "strict": "prefer"},
        prompt_snippet=config.prompt_snippet,
        prompt_guidelines=list(config.prompt_guidelines) if chosen.expose_session_environment else [],
        execute=execute,
    )


_BASH_TOOL_CONFIG = ShellToolConfig(
    name="bash", label="bash", shell_name="bash", prompt="$",
    prompt_snippet=BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
    prompt_guidelines=BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"],
    temp_file_prefix="pi-bash",
)


def create_bash_tool_definition(cwd: str, options: BashToolOptions | None = None) -> ToolDefinition:
    return create_shell_tool_definition(cwd, _BASH_TOOL_CONFIG, options)


def create_bash_tool(cwd: str, options: BashToolOptions | None = None) -> AgentTool:
    return wrap_tool_definition(create_bash_tool_definition(cwd, options))


__all__ = [
    "BASH_SCHEMA", "BASH_UPDATE_THROTTLE_MS", "BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION",
    "BashExecOptions", "BashExecResult", "BashOperations", "BashSpawnContext",
    "BashSpawnHook", "BashToolOptions", "ShellToolConfig", "create_local_shell_operations",
    "create_local_bash_operations", "create_shell_tool_definition",
    "create_bash_tool_definition", "create_bash_tool",
]
