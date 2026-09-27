"""Shell capture compatibility collector ported from ``harness/utils/shell-output.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..._chord.context import Context
from ..result import Result, err, ok
from ..types import (
    ExecutionEnv,
    ExecutionError,
    ShellExecOptions,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputView,
)
from .output_capture import apply_shell_output_update, sanitize_shell_output
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail

__all__ = [
    "ShellCaptureProgress",
    "ShellCaptureOptions",
    "ShellCaptureResult",
    "execute_shell_with_capture",
    "sanitize_binary_output",
]


@dataclass
class ShellCaptureProgress:
    output: str
    truncation: TruncationResult
    last_line_bytes: int = 0
    full_output_path: Optional[str] = None


@dataclass
class ShellCaptureOptions:
    """Options for :func:`execute_shell_with_capture`."""

    cwd: Optional[str] = None
    env: Optional[dict] = None
    inherit_env: bool = True
    timeout: Optional[float] = None
    stdin: Optional[str] = None
    on_chunk: Optional[Callable[[str, Callable[[], ShellCaptureProgress], Context], None]] = None
    #: Return shell execution failures with captured output instead of a failed Result.
    return_execution_errors: bool = False


@dataclass
class ShellCaptureResult(ShellCaptureProgress):
    exit_code: Optional[int] = None
    cancelled: bool = False
    truncated: bool = False
    execution_error: Optional[ExecutionError] = None


def _progress_from(output: ShellOutputView) -> ShellCaptureProgress:
    truncation = TruncationResult(
        content=output.text,
        truncated=output.truncation.truncated,
        truncated_by=output.truncation.truncated_by,
        total_lines=output.truncation.total_lines,
        total_bytes=output.truncation.total_bytes,
        output_lines=output.truncation.output_lines,
        output_bytes=output.truncation.output_bytes,
        last_line_partial=output.truncation.last_line_partial,
        first_line_exceeds_limit=output.truncation.first_line_exceeds_limit,
        max_lines=output.truncation.max_lines,
        max_bytes=output.truncation.max_bytes,
    )
    return ShellCaptureProgress(
        output=output.text,
        truncation=truncation,
        full_output_path=output.spill_path,
        last_line_bytes=output.last_line_bytes or 0,
    )


async def execute_shell_with_capture(
    env: ExecutionEnv,
    command: str,
    options: Optional[ShellCaptureOptions],
    context: Context,
) -> Result:
    """Run a shell command collecting one bounded final view.

    Source-side capture, adaptive publication, and spilling remain owned by the
    execution environment.
    """
    holder: dict = {"output": None}
    options = options or ShellCaptureOptions()

    def _on_update(update, update_context) -> None:
        previous = holder["output"]
        current = apply_shell_output_update(previous, update)
        holder["output"] = current
        if update.kind in ("append", "slide"):
            chunk = update.text
        elif update.kind == "replace" and previous is None:
            chunk = current.text
        else:
            chunk = None
        # A metadata-only update and a post-cap replacement contain no new
        # incremental chunk; reporting their complete view would duplicate bytes.
        if chunk and options.on_chunk is not None:
            options.on_chunk(chunk, lambda: _progress_from(holder["output"]), update_context)

    result = await env.exec(
        command,
        ShellExecOptions(
            cwd=options.cwd,
            env=options.env,
            inherit_env=options.inherit_env,
            timeout=options.timeout,
            stdin=options.stdin,
            capture=ShellOutputCaptureOptions(
                limits=ShellOutputLimits(
                    max_bytes=DEFAULT_MAX_BYTES, max_lines=DEFAULT_MAX_LINES, retain="tail"
                ),
                spill=True,
            ),
            on_update=_on_update,
        ),
        context,
    )

    if holder["output"] is None:
        from ..types import ShellOutputTruncation

        empty = truncate_tail("")
        holder["output"] = ShellOutputView(
            text="",
            truncation=ShellOutputTruncation(
                truncated=empty.truncated,
                truncated_by=empty.truncated_by,
                total_lines=empty.total_lines,
                total_bytes=empty.total_bytes,
                output_lines=empty.output_lines,
                output_bytes=empty.output_bytes,
                last_line_partial=empty.last_line_partial,
                first_line_exceeds_limit=empty.first_line_exceeds_limit,
                max_lines=empty.max_lines,
                max_bytes=empty.max_bytes,
            ),
        )

    progress = _progress_from(holder["output"])

    if not result.ok:
        aborted = result.error.code == "aborted" or (context.abort_signal is not None and context.abort_signal.aborted)
        if aborted:
            return ok(
                ShellCaptureResult(
                    output=progress.output,
                    truncation=progress.truncation,
                    full_output_path=progress.full_output_path,
                    last_line_bytes=progress.last_line_bytes,
                    exit_code=None,
                    cancelled=True,
                    truncated=progress.truncation.truncated,
                )
            )
        if options.return_execution_errors:
            return ok(
                ShellCaptureResult(
                    output=progress.output,
                    truncation=progress.truncation,
                    full_output_path=progress.full_output_path,
                    last_line_bytes=progress.last_line_bytes,
                    exit_code=None,
                    cancelled=False,
                    truncated=progress.truncation.truncated,
                    execution_error=result.error,
                )
            )
        return err(result.error)

    return ok(
        ShellCaptureResult(
            output=progress.output,
            truncation=progress.truncation,
            full_output_path=progress.full_output_path,
            last_line_bytes=progress.last_line_bytes,
            exit_code=result.value.exit_code,
            cancelled=False,
            truncated=result.value.truncation.truncated,
        )
    )


#: Alias mirroring the TS re-export.
sanitize_binary_output = sanitize_shell_output
