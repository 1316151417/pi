"""Drive dispatcher ported from ``harness/runtime/drive.ts``.

This package doubles as the TS ``drive.ts`` module: Python cannot keep a
``drive.py`` module and a ``drive/`` package side by side, so the dispatcher
lives here and the procedures live in sibling modules.

Drives one installed pass through direct durable procedures until settlement or
a durable wait.
"""

from __future__ import annotations

from typing import Any

from ...execution.effect_gate import AbortRequested
from ...session.session import SessionInvariantError
from .checkpoint import run_checkpoint, start_run
from .deferred import run_deferred
from .generation import run_generation
from .reconcile import reconcile_operation
from .recovery import recover_assistant_generation
from .structural import (
    commit_navigation,
    recover_structural_generation,
    run_structural_decision,
    run_structural_generation,
    run_structural_retry_wait,
)
from .tools import run_tools
from ..types import Drive, ProcedureResult

__all__ = ["drive_operation", "current_operation"]


def current_operation(lane: Any, drive: Drive) -> Any:
    """The current operation owned by an installed drive, or a loud failure."""
    operation = lane.state.operation
    if operation is None or operation.meta.operation_id != drive.operation_id:
        raise SessionInvariantError(
            f"Drive {drive.operation_id} has no matching current operation"
        )
    return operation


async def drive_operation(lane: Any, drive: Drive) -> dict:
    """Drive one installed pass through direct durable procedures until settlement or a durable wait."""
    operation = current_operation(lane, drive)
    if operation.state.control.status == "running":
        try:
            await lane.hooks.run_with_gate(
                "before_drive",
                {
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "operation": operation.meta.intent.kind,
                },
                drive.gate,
                drive.context,
            )
        except AbortRequested as error:
            await error.cancellation

    while True:
        operation = current_operation(lane, drive)
        state = operation.state
        try:
            if state.control.status == "cancel_requested":
                result = await reconcile_operation(lane, drive)
            else:
                result = await _dispatch(lane, drive, state)
        except AbortRequested as error:
            await error.cancellation
            result = ProcedureResult(kind="continue")

        if result.kind == "settled":
            return {"kind": "settled", "outcome": result.outcome}
        if result.kind == "waiting":
            return result.outcome
        next_state = current_operation(lane, drive).state
        if (
            next_state is state
            and next_state.control.status != "cancel_requested"
        ):
            raise SessionInvariantError(
                f"Drive procedure made no progress from {getattr(state, 'at', None)}"
            )


async def _dispatch(lane: Any, drive: Drive, state: Any) -> ProcedureResult:
    at = state.at
    if at == "starting":
        return await start_run(lane, drive, state)
    if at == "checkpoint":
        return await run_checkpoint(lane, drive, state)
    if at in ("assistant.ready", "assistant.retry_wait"):
        return await run_generation(lane, drive, state)
    if at == "assistant.effect_pending":
        return await recover_assistant_generation(lane, drive, state)
    if at == "tools":
        return await run_tools(lane, drive, state)
    if at in ("deferred.suspended", "deferred.effect_pending"):
        return await run_deferred(lane, drive, state)
    if at == "summary.deciding":
        return await run_structural_decision(lane, drive, state)
    if at == "summary.ready":
        return await run_structural_generation(lane, drive, state)
    if at == "summary.effect_pending":
        return await recover_structural_generation(lane, drive, state)
    if at == "summary.retry_wait":
        return await run_structural_retry_wait(lane, drive, state)
    if at == "navigation.ready_to_commit":
        return await commit_navigation(lane, drive, state)
    raise SessionInvariantError(f"Unknown operation state {at!r}")
