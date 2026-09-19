"""Cancellation reconciliation ported from ``harness/runtime/drive/reconcile.ts``.

Advances one cancelled durable leaf to a terminal record without starting new
ordinary work, best-effort cancelling remote deferred responses on the way.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from ...context import get_telemetry_context
from ...session.session import SessionInvariantError
from ..types import FinishDecision, LaneReturn, ProcedureResult
from .deferred import read_deferred_source_handle
from .recovery import recover_cancelled_assistant_effect
from .terminal import operation_cleanup_writes, operation_result_record

__all__ = ["reconcile_operation", "settled_cancellation"]


async def _cancel_deferred_best_effort(lane: Any, drive: Any, deferred: Any, handle: Any) -> None:
    identity = deferred.configuration.model
    model = lane.models.get_model(
        _opt(identity, "provider"), _opt(identity, "model_id", "modelId")
    )
    if model is None:
        return
    options = deferred.stream_options
    try:
        await lane.models.cancel_deferred(
            model,
            handle,
            {
                "signal": drive.close_signal,
                "telemetryContext": get_telemetry_context(drive.context),
                "timeoutMs": _opt(options, "timeout_ms", "timeoutMs"),
                "maxRetries": _opt(options, "max_retries", "maxRetries"),
                "maxRetryDelayMs": _opt(options, "max_retry_delay_ms", "maxRetryDelayMs"),
                "headers": _opt(options, "headers"),
            },
        )
    except Exception:  # noqa: BLE001
        # Remote cancellation is best-effort; durable local reconciliation must continue.
        pass


async def _read_deferred_handle(lane: Any, drive: Any, deferred: Any) -> Any:
    async def _plan(_state: Any, _current: Any, _meta: Any, reader: Any) -> Any:
        return LaneReturn(
            result=await read_deferred_source_handle(reader, deferred, drive.context)
        )

    return await lane.settle_operation(deferred, _plan, drive.context)


async def _publish_aborted_terminal(lane: Any, drive: Any, capability: Any) -> ProcedureResult:
    async def _plan(state: Any, current: Any, meta: Any, reader: Any) -> Any:
        if current.control.status != "cancel_requested":
            raise SessionInvariantError(
                "Cancellation reconciliation requires cancelled durable control"
            )
        record = operation_result_record(meta, "aborted", state.tip_id)
        cleanup = await operation_cleanup_writes(
            reader, drive.operation_id, current, drive.context
        )
        events: List[Any] = []
        if meta.intent.kind == "run":
            if current.at in (
                "summary.deciding",
                "summary.ready",
                "summary.effect_pending",
                "summary.retry_wait",
            ):
                if (
                    current.task.boundary.kind != "resume_checkpoint"
                    or current.task.reason is None
                ):
                    raise SessionInvariantError(
                        "Cancelled run summary has an invalid result boundary"
                    )
                events.append(
                    {
                        "type": "compaction_end",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "reason": current.task.reason,
                        "status": "aborted",
                        "endedAt": record.ended_at,
                    }
                )
            events.append(
                {
                    "type": "run_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "status": "aborted",
                    "fromTipId": meta.source_tip_id,
                    "tipId": state.tip_id,
                    "endedAt": record.ended_at,
                }
            )
        elif meta.intent.kind == "compaction":
            events.append(
                {
                    "type": "compaction_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "reason": "manual",
                    "status": "aborted",
                    "endedAt": record.ended_at,
                }
            )
        else:
            events.append(
                {
                    "type": "navigation_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "status": "aborted",
                    "fromTipId": meta.source_tip_id,
                    "tipId": state.tip_id,
                    "endedAt": record.ended_at,
                }
            )
        return FinishDecision(
            writes=cleanup,
            record=record,
            materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
            events=lambda _commit: events,
        )

    return await lane.settle_operation(capability, _plan, drive.context)


async def reconcile_operation(lane: Any, drive: Any) -> ProcedureResult:
    """Advance one cancelled durable leaf without starting new ordinary work."""
    from .tools import run_tools

    operation = lane.state.operation
    if operation is None or operation.meta.operation_id != drive.operation_id:
        raise SessionInvariantError(
            f"Drive {drive.operation_id} has no matching operation to reconcile"
        )
    if operation.state.control.status != "cancel_requested":
        raise SessionInvariantError(f"Operation {drive.operation_id} is not cancelled")
    # Reconciliation has nothing left to await: the cancellation already happened
    # in a previous process, so the gate gets an already-settled cancellation.
    drive.begin_abort(settled_cancellation())
    drive.signal_abort()

    state = operation.state
    at = state.at
    if at == "assistant.effect_pending":
        return await recover_cancelled_assistant_effect(lane, drive, state)
    if at == "tools":
        return await run_tools(lane, drive, state)
    if at == "deferred.suspended":
        handle = await _read_deferred_handle(lane, drive, state)
        await _cancel_deferred_best_effort(lane, drive, state, handle)
        return await _publish_aborted_terminal(lane, drive, state)
    if at == "deferred.effect_pending":
        handle = await _read_deferred_handle(lane, drive, state)
        await _cancel_deferred_best_effort(lane, drive, state, handle)
        return await recover_cancelled_assistant_effect(lane, drive, state)
    if at in (
        "starting",
        "checkpoint",
        "assistant.ready",
        "assistant.retry_wait",
        "summary.deciding",
        "summary.ready",
        "summary.effect_pending",
        "summary.retry_wait",
        "navigation.ready_to_commit",
    ):
        return await _publish_aborted_terminal(lane, drive, state)
    raise SessionInvariantError(f"Unknown operation state {at!r}")


def settled_cancellation() -> "asyncio.Future":
    """A cancellation awaitable that is already complete."""
    future = asyncio.get_running_loop().create_future()
    future.set_result(None)
    return future


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping or an attribute."""
    if value is None:
        return None
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
