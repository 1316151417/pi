"""Plugin task kind ported from ``harness/pico3/kinds/plugin.ts``.

Runs one plugin handler and resolves with its stored result. The whole point of
this kind is that a plugin invocation is a *task*, so it gets a task id, an
owner, and an abort path for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...._chord.context import Context
from pi_ai.types import JsonValue
from ..types import Completion, Kind, Step, Task, to_stored
from .task_api import task_api

__all__ = ["PluginInput", "PluginCheckpoint", "PluginFailure", "plugin", "plugin_kind"]


@dataclass
class PluginInput:
    handler: str = ""
    input: JsonValue = None

    def to_json(self) -> JsonObject:
        return {"handler": self.handler, "input": self.input}


@dataclass
class PluginCheckpoint:
    phase: str = "started"


@dataclass
class PluginFailure:
    reason: str = "missing_handler"  # "missing_handler" | "threw"
    detail: str = ""

    def to_json(self) -> JsonObject:
        return {"reason": self.reason, "detail": self.detail}


async def _run(task: Task, runtime: Any, ctx: Context) -> Step:
    input_record = task.input if isinstance(task.input, dict) else {}
    handler_name = input_record.get("handler")
    handler = runtime.plugins.get(handler_name)
    if handler is None:
        return Step(
            done=lambda _tx=None, _task=None: Completion(
                status="failed",
                failure=PluginFailure(reason="missing_handler", detail=handler_name),
            )
        )
    try:
        value = handler(input_record.get("input"), task_api(task, runtime), ctx)
        if hasattr(value, "__await__"):
            value = await value
        result = to_stored(value)
        return Step(
            done=lambda _tx=None, _task=None: Completion(status="completed", result=result)
        )
    except Exception as error:  # noqa: BLE001 - reported as a task failure
        if ctx.abort_signal is not None and ctx.abort_signal.aborted:
            raise
        return Step(
            done=lambda _tx=None, _task=None, error=error: Completion(
                status="failed", failure=PluginFailure(reason="threw", detail=str(error))
            )
        )


async def _initial(task: Task, runtime: Any, ctx: Context) -> Step:
    await runtime.commit(
        lambda tx, current=None, _ctx=None: tx.checkpoint({"phase": "started"}), ctx
    )
    return await _run(task, runtime, ctx)


async def _started(task: Task, runtime: Any, ctx: Context) -> Step:
    return await _run(task, runtime, ctx)


async def _abort(_task: Task, _runtime: Any, _ctx: Context) -> Any:
    return lambda *_args: None


plugin_kind = Kind(
    name="pi.plugin",
    inflight=["started"],
    phases={"started": _started},
    initial=_initial,
    abort=_abort,
)

#: Frozen declaration, mirroring ``Object.freeze(pluginKind)``.
plugin = plugin_kind


_ = Dict, Optional
