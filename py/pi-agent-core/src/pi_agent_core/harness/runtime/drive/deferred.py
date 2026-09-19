"""Deferred polling procedures ported from ``harness/runtime/drive/deferred.ts``.

Reads a suspended response's durable source handle, prepares one poll under the
pass's permit budget, and classifies the poll response.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ...._pi_ai.types import DeferredHandle
from ...context import get_telemetry_context, with_abort_signal
from ...execution.assistant import consume_assistant_stream
from ...hooks import apply_stream_options_patch
from ...session.session import SessionInvariantError
from ...session.types import (
    DeferredEffectPendingOperation,
    OperationError,
    operation_scope_of,
)
from ...session.values import delete_list, pending_assistant_frames
from ..types import CommitDecision, ContinueOperationResult, ProcedureResult
from .response import open_assistant_response, publish_configuration_failure, publish_response

__all__ = [
    "PreparedDeferredPoll",
    "read_deferred_source_handle",
    "run_deferred",
    "run_deferred_suspended",
    "recover_deferred_poll",
]


class PreparedDeferredPoll:
    """One deferred poll ready to be executed."""

    def __init__(
        self, source: DeferredHandle, model: Any, poll: int, stream_options: Any
    ) -> None:
        self.kind = "ready"
        self.source = source
        self.model = model
        self.poll = poll
        self.stream_options = stream_options


def _configuration_error(identity: Any) -> OperationError:
    return OperationError(
        code="model_unavailable",
        message="The configured model is unavailable in this process",
        details=identity,
    )


async def read_deferred_source_handle(reader: Any, deferred: Any, context: Any) -> DeferredHandle:
    """The validated provider handle recorded on one deferred source entry."""
    source = (await reader.get_entries([deferred.source_entry_id], context)).get(
        deferred.source_entry_id
    )
    message = getattr(source, "message", None)
    if (
        getattr(source, "type", None) != "message"
        or getattr(message, "role", None) != "assistant"
        or getattr(message, "stop_reason", None) != "deferred"
        or getattr(message, "deferred", None) is None
    ):
        raise SessionInvariantError(
            f"Deferred source {deferred.source_entry_id} is missing its assistant handle"
        )
    handle = message.deferred
    identity = deferred.configuration.model
    if (
        len(handle.id) == 0
        or handle.provider != _opt(identity, "provider")
        or handle.model_id != _opt(identity, "model_id", "modelId")
        or handle.api != getattr(message, "api", None)
    ):
        raise SessionInvariantError(
            f"Deferred source {deferred.source_entry_id} has an invalid handle"
        )
    return handle


async def _read_source_handle(lane: Any, drive: Any, deferred: Any) -> ContinueOperationResult:
    async def _plan(_state: Any, _current: Any, _meta: Any, reader: Any) -> Any:
        from ..types import LaneReturn

        return LaneReturn(
            result=await read_deferred_source_handle(reader, deferred, drive.context)
        )

    return await lane.continue_operation(deferred, _plan, drive.context)


async def prepare_deferred_poll(lane: Any, drive: Any, expected: Any) -> Any:
    source = await _read_source_handle(lane, drive, expected)
    if source.kind == "cancel_requested":
        return source
    if drive.deferred_permits == 0:
        return _Bag({"kind": "waiting", "source": source.value})

    identity = expected.configuration.model
    model = lane.models.get_model(
        _opt(identity, "provider"), _opt(identity, "model_id", "modelId")
    )
    if model is None:
        return _Bag({"kind": "configuration_failure"})
    base_options = _replace_option(expected.stream_options, "deferred", False)
    poll = expected.poll + 1 if expected.at == "deferred.suspended" else expected.poll
    before_request = await lane.hooks.run_with_gate(
        "before_request",
        {
            "lane": lane.name,
            "runId": drive.operation_id,
            "model": model,
            "step": "deferred",
            "attempt": poll,
            "streamOptions": base_options,
        },
        drive.gate,
        drive.context,
    )
    patch = None if before_request is None else _opt(before_request, "stream_options", "streamOptions")
    resolved = (
        base_options if patch is None else apply_stream_options_patch(base_options, patch)
    )
    return PreparedDeferredPoll(
        source=source.value,
        model=model,
        poll=poll,
        stream_options=_replace_option(resolved, "deferred", False),
    )


async def publish_poll_intent(
    lane: Any, drive: Any, deferred: Any, prepared: PreparedDeferredPoll, recovery: bool
) -> ContinueOperationResult:
    at = _now()
    next_fields = {
        "step_id": deferred.step_id,
        "source_entry_id": deferred.source_entry_id,
        "poll": prepared.poll,
        "response_entry_id": lane.session.id_generator.next(at),
        "usage_id": lane.session.id_generator.next(at),
        "configuration": deferred.configuration,
        "stream_options": deferred.stream_options,
    }

    def _plan(_state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        next_state = DeferredEffectPendingOperation(
            **operation_scope_of(current), **next_fields
        )

        def _materialize(_commit: Any) -> Any:
            drive.deferred_permits -= 1
            return next_state

        return CommitDecision(
            writes=(
                [
                    delete_list(
                        pending_assistant_frames(
                            drive.operation_id, deferred.response_entry_id
                        )
                    )
                ]
                if deferred.at == "deferred.effect_pending"
                else []
            ),
            operation_state=next_state,
            materialize=_materialize,
            events=lambda _commit: [
                {
                    "type": "run_resume",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    **({"recovery": True} if recovery else {}),
                },
                {
                    "type": "turn_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "turnId": f"{next_fields['step_id']}:poll:{next_fields['poll']}",
                    **({"recovery": True} if recovery else {}),
                },
            ],
        )

    return await lane.continue_operation(deferred, _plan, drive.context)


async def perform_deferred_poll(
    lane: Any,
    drive: Any,
    prepared: PreparedDeferredPoll,
    intent: DeferredEffectPendingOperation,
    recovery: bool,
) -> Any:
    response = open_assistant_response(lane, drive, intent.response_entry_id, recovery)
    admitted = with_abort_signal(drive.gate.signal, drive.context)
    options = prepared.stream_options
    metadata_box: dict = {}

    async def _on_payload(payload: Any, request_model: Any) -> Any:
        result = await lane.hooks.run_with_gate(
            "before_payload",
            {
                "lane": lane.name,
                "runId": drive.operation_id,
                "model": request_model,
                "payload": payload,
            },
            drive.gate,
            drive.context,
        )
        return None if result is None else _opt(result, "payload")

    async def _on_response(response_payload: Any, _model: Any = None) -> None:
        metadata_box["metadata"] = {
            "status": _opt(response_payload, "status"),
            "headers": _opt(response_payload, "headers"),
        }

    def _start_poll() -> Any:
        return lane.models.stream_deferred(
            prepared.model,
            prepared.source,
            {
                "wait": 0,
                "signal": admitted.signal,
                "telemetryContext": get_telemetry_context(admitted),
                "timeoutMs": _opt(options, "timeout_ms", "timeoutMs"),
                "maxRetries": _opt(options, "max_retries", "maxRetries"),
                "maxRetryDelayMs": _opt(options, "max_retry_delay_ms", "maxRetryDelayMs"),
                "headers": _opt(options, "headers"),
                "onPayload": _on_payload,
                "onResponse": _on_response,
            },
        )

    stream = drive.gate.admit(_start_poll)

    async def _after_response(message: Any, context: Any) -> Any:
        return await response.after_response(message, metadata_box.get("metadata") or {}, context)

    try:
        return await consume_assistant_stream(
            stream, response.observer, _after_response, drive.context
        )
    finally:
        await response.close()


async def _poll_deferred(lane: Any, drive: Any, expected: Any, recovery: bool) -> ProcedureResult:
    prepared = await prepare_deferred_poll(lane, drive, expected)
    if prepared.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    if prepared.kind == "waiting":
        return ProcedureResult(
            kind="waiting",
            outcome={
                "kind": "waiting",
                "operationId": drive.operation_id,
                "reason": "deferred",
                "deferred": prepared.source,
            },
        )
    if prepared.kind == "configuration_failure":
        return await publish_configuration_failure(
            lane, drive, expected, _configuration_error(expected.configuration.model)
        )

    intent = await publish_poll_intent(lane, drive, expected, prepared, recovery)
    if intent.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    response = await perform_deferred_poll(lane, drive, prepared, intent.value, recovery)
    return await publish_response(
        lane, drive, intent.value, response, {"recovery": True} if recovery else {}
    )


async def run_deferred_suspended(lane: Any, drive: Any, deferred: Any) -> ProcedureResult:
    """Poll one durably suspended deferred response when this pass carries a permit."""
    return await _poll_deferred(lane, drive, deferred, False)


async def recover_deferred_poll(lane: Any, drive: Any, deferred: Any) -> ProcedureResult:
    """Replace one orphaned unknown-outcome poll under fresh ids when a permit exists."""
    return await _poll_deferred(lane, drive, deferred, True)


async def run_deferred(lane: Any, drive: Any, deferred: Any) -> ProcedureResult:
    """Advance or report the wait for one deferred run phase."""
    if deferred.at == "deferred.suspended":
        return await run_deferred_suspended(lane, drive, deferred)
    return await recover_deferred_poll(lane, drive, deferred)


def _now() -> int:
    return int(time.time() * 1000)


def _replace_option(options: Any, name: str, value: Any) -> Any:
    """Return a copy of a stream-options object with one field overridden."""
    if options is None:
        return None
    if isinstance(options, dict):
        return {**options, name: value}
    return options.__class__(**{**vars(options), name: value})


class _Bag:
    """Attribute view over a plain mapping."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping or an attribute."""
    if value is None:
        return None
    if isinstance(value, _Bag):
        value = value._values
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
