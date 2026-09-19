"""Bash tool ported from ``harness/pico3/bash.ts``.

Runs a shell command; output is piped to the kernel, bounds come from ``output``.
"""

from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..._chord.context import Context
from .types import ToolDeclaration, ToolResult

__all__ = ["bash_tool", "BashTool", "BASH_PARAMETERS"]


#: JSON-Schema stand-in for the TypeScript ``Type.Object`` parameters schema.
BASH_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "cwd": {"type": "string"},
    },
    "required": ["command"],
    "additionalProperties": False,
}


@dataclass
class BashTool:
    """The built-in ``bash`` tool declaration."""

    name: str = "bash"
    description: str = "Run a shell command"
    parameters: Dict[str, Any] = None  # type: ignore[assignment]
    replay: str = "unsafe"
    output: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = dict(BASH_PARAMETERS)
        if self.output is None:
            self.output = {}

    async def execute(self, args: Dict[str, Any], api: Any, ctx: Context) -> ToolResult:
        command = args.get("command")
        cwd = args.get("cwd")
        started = int(time.time() * 1000)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def pump(stream: Any) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return
                api.stream(chunk)

        signal = ctx.signal

        async def kill_on_abort() -> None:
            if signal is None:
                return
            await signal.wait()
            if process.returncode is None:
                process.kill()

        killer = asyncio.ensure_future(kill_on_abort())
        try:
            await asyncio.gather(pump(process.stdout), pump(process.stderr))
            status = await process.wait()
        finally:
            killer.cancel()
        code: Optional[int] = status if status >= 0 else None
        signal_name: Optional[str] = None
        if status < 0:
            try:
                signal_name = signal.Signals(-status).name
            except ValueError:  # pragma: no cover - unknown signal number
                signal_name = str(-status)
        return ToolResult(
            is_error=code != 0,
            details={
                "exitCode": code,
                "signal": signal_name,
                "ms": int(time.time() * 1000) - started,
            },
        )


def bash_tool(output: Optional[Dict[str, Any]] = None) -> BashTool:
    """Run a shell command. Output is piped to the kernel; bounds come from ``output``."""
    return BashTool(output=dict(output or {}))


_ = ToolDeclaration
