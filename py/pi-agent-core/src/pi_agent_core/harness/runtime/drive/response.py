"""Assistant response lifecycle ported from ``harness/runtime/drive/response.ts``.

Opens the streaming response observer used by generation, and classifies and
atomically settles one assistant-generation or deferred-poll response.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from ...._pi_ai.assistant_message_frame import AssistantMessageFrameEncoder
from ...._pi_ai.overflow import is_context_overflow, is_recoverable_length
from ...._pi_ai.types import AssistantMessage
from ...session.commit import insert_entry, insert_usage
from ...session.session import SessionInvariantError
from ...session.types import (
    AssistantRetryWaitOperation,
    PendingEntry,
    CheckpointOperation,
    Continuation,
    DeferredSuspendedOperation,
    MessageEntry,
    OperationError,
    ResultBoundary,
    SummaryDecidingOperation,
    SummaryTask,
    ToolBatch,
    ToolCall,
    UsageRow,
)
from ...session.values import (
    branch_tip,
    delete_list,
    operation_preparation,
    pending_assistant_frames,
    set_value,
)
from ...utils.retry import is_retryable_assistant_error, retry_delay_ms
from ..progress import open_frame_progress
from ..types import CommitDecision, FinishDecision, ProcedureResult
from .retry import retry_not_before
from .structural import prepare_overflow_compaction
from .terminal import operation_cleanup_writes, operation_result_record
from ...execution.assistant import AssistantResponseMetadata

__all__ = [
    "AssistantResponseLifecycle",
    "open_assistant_response",
    "publish_configuration_failure",
    "publish_response",
]


@dataclass
class AssistantResponseLifecycle:
    """The streaming observer for one in-flight assistant response."""

    observer: Any
    after_response: Callable[..., Any]
    close: Callable[..., Any]


def open_assistant_response(
    lane: Any,
    drive: Any,
    response_entry_id: str,
    recovery: bool = False,
) -> AssistantResponseLifecycle:
    """Open the frame-backed response observer for one assistant generation."""
    progress = open_frame_progress(lane, drive, response_entry_id)
    frame_encoder = AssistantMessageFrameEncoder()
    event_context: dict = {"lane": lane.name, "runId": drive.operation_id}
    if recovery:
        event_context["recovery"] = True

    async def _close() -> None:
        progress.seal()
        await progress.drain()

    class _Observer:
        async def start(self, message: Any, event: Any, context: Any) -> None:
            frame = frame_encoder.encode(event)
            if frame is not None:
                progress.write(frame)
            await lane.emit_batch(
                [{"type": "message_start", **event_context, "message": message}], context
            )

        async def update(self, message: Any, event: Any, context: Any) -> None:
            frame = frame_encoder.encode(event)
            if frame is not None:
                progress.write(frame)
            payload = {
                "type": "message_update",
                **event_context,
                "message": message,
                "event": event,
            }
            if frame is not None:
                payload["frame"] = frame
            await lane.emit_batch([payload], context)

        async def end(self, message: Any, context: Any) -> None:
            await lane.emit_batch(
                [
                    {
                        "type": "message_end",
                        **event_context,
                        "message": message,
                        "entryId": response_entry_id,
                    }
                ],
                context,
            )

    async def _after_response(message: Any, metadata: Any, context: Any) -> Any:
        await _close()
        result = await lane.hooks.run_with_gate(
            "after_response",
            {
                "lane": lane.name,
                "runId": drive.operation_id,
                **_metadata_fields(metadata),
                "message": message,
            },
            drive.gate,
            context,
        )
        resolved = None if result is None else getattr(result, "message", None)
        return message if resolved is None else resolved

    return AssistantResponseLifecycle(
        observer=_Observer(), after_response=_after_response, close=_close
    )


async def publish_configuration_failure(
    lane: Any, drive: Any, capability: Any, error: OperationError
) -> ProcedureResult:
    """Publish a non-retryable request-configuration failure before reserving response ids."""

    async def _plan(state: Any, current: Any, meta: Any, reader: Any) -> Any:
        if state.tip_id is None:
            raise SessionInvariantError("Failed run has no Branch tip")
        record = operation_result_record(meta, "failed", state.tip_id, error)
        cleanup = await operation_cleanup_writes(
            reader, drive.operation_id, current, drive.context
        )

        def _events(_commit: Any) -> List[Any]:
            return [
                {
                    "type": "run_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "status": "failed",
                    "error": error,
                    "fromTipId": meta.source_tip_id,
                    "tipId": state.tip_id,
                    "endedAt": record.ended_at,
                }
            ]

        return FinishDecision(
            writes=cleanup,
            record=record,
            materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
            events=_events,
        )

    result = await lane.continue_operation(capability, _plan, drive.context)
    if result.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    return result.value


def uuid_v7_timestamp(entry_id: str) -> int:
    """Recover the epoch-ms timestamp encoded in a reserved UUIDv7 id."""
    try:
        timestamp = int(entry_id[:8] + entry_id[9:13], 16)
    except ValueError as error:
        raise SessionInvariantError(f"Invalid reserved UUIDv7 {entry_id}") from error
    return timestamp


def _metadata_fields(metadata: Any) -> dict:
    """Spread the captured HTTP response metadata into a hook payload."""
    if metadata is None:
        return {}
    resolved = (
        {"status": metadata.status, "headers": metadata.headers}
        if isinstance(metadata, AssistantResponseMetadata)
        else dict(metadata)
    )
    return {key: value for key, value in resolved.items() if value is not None}


def _provider_error(source: str, message: Any) -> OperationError:
    label = "Assistant" if source == "assistant" else "Deferred"
    return OperationError(
        code="assistant_error",
        message=message.error_message
        or f"{label} request ended with {message.stop_reason}",
    )


def _normalize_error(message: Any, error_message: str) -> Any:
    resolved = copy.copy(message)
    resolved.stop_reason = "error"
    resolved.error_message = error_message
    return resolved


def _normalize_aborted(source: str, message: Any) -> Any:
    label = "Assistant" if source == "assistant" else "Deferred"
    resolved = copy.copy(message)
    resolved.stop_reason = "aborted"
    resolved.error_message = (
        message.error_message or f"{label} request was cancelled"
    )
    return resolved


def _deferred_handle_is_valid(message: Any, generation: Any) -> bool:
    handle = message.deferred
    identity = generation.generation_context.configuration.model
    provider = _field(identity, "provider", "provider")
    model_id = _field(identity, "model_id", "modelId")
    return (
        message.stop_reason == "deferred"
        and handle is not None
        and len(handle.id) != 0
        and handle.provider == provider
        and handle.model_id == model_id
        and handle.api == message.api
    )


async def publish_response(
    lane: Any,
    drive: Any,
    intent: Any,
    response: Any,
    options: Optional[dict] = None,
) -> ProcedureResult:
    """Classify and atomically settle one assistant-generation or deferred-poll response."""
    options = options or {}
    overflow = intent.at == "assistant.effect_pending" and (
        is_context_overflow(response, intent.context_window)
        or is_recoverable_length(response, intent.intended_output_limit)
    )
    overflow_preparation = (
        await prepare_overflow_compaction(lane, drive, intent)
        if overflow and not intent.generation_context.overflow_recovery_used
        else None
    )

    async def _plan(state: Any, current: Any, meta: Any, reader: Any) -> Any:
        source = "assistant" if current.at == "assistant.effect_pending" else "deferred"
        response_entry_id = intent.response_entry_id
        configuration = (
            current.generation_context.configuration
            if current.at == "assistant.effect_pending"
            else current.configuration
        )
        turn_id = (
            current.generation_context.step_id
            if current.at == "assistant.effect_pending"
            else f"{current.step_id}:poll:{current.poll}"
        )
        scope = _scope_of(current, response_entry_id)
        committed = response
        settled: Any = None
        failure: Optional[OperationError] = None

        if current.control.status == "cancel_requested":
            committed = _normalize_aborted(source, response)
            settled = CheckpointOperation(
                **scope,
                continuation=Continuation(kind="may_finish", include_final_assistant=True),
                trigger_entry_id=response_entry_id,
            )
        elif response.stop_reason == "aborted":
            label = "Assistant" if source == "assistant" else "Deferred"
            raise SessionInvariantError(
                f"{label} response is aborted while durable control is running"
            )
        elif current.at == "assistant.effect_pending" and overflow:
            committed = _normalize_error(
                response,
                response.error_message or "Assistant request exceeded the context window",
            )
            if current.generation_context.overflow_recovery_used or overflow_preparation is None:
                failure = _provider_error(source, committed)
            else:
                settled = SummaryDecidingOperation(
                    **scope,
                    task=SummaryTask(
                        task_id=overflow_preparation["taskId"],
                        reason="overflow",
                        boundary=ResultBoundary(
                            kind="resume_checkpoint",
                            resume_after=_checkpoint_data(
                                kind="need_assistant",
                                overflow_recovery_used=True,
                                trigger_entry_id=current.generation_context.trigger_entry_id,
                            ),
                        ),
                    ),
                )
        elif response.stop_reason == "deferred":
            if current.at == "assistant.effect_pending":
                if _deferred_handle_is_valid(response, current):
                    settled = DeferredSuspendedOperation(
                        **scope,
                        step_id=current.generation_context.step_id,
                        source_entry_id=response_entry_id,
                        poll=0,
                        configuration=configuration,
                        stream_options=current.generation_context.stream_options,
                    )
                else:
                    committed = _normalize_error(
                        response, "Provider returned an invalid deferred handle"
                    )
                    failure = _provider_error(source, committed)
            else:
                settled = DeferredSuspendedOperation(
                    **scope,
                    step_id=current.step_id,
                    source_entry_id=response_entry_id,
                    poll=current.poll,
                    configuration=configuration,
                    stream_options=current.stream_options,
                )
        elif response.stop_reason == "error":
            if (
                current.at == "assistant.effect_pending"
                and (options.get("recovery") is True or is_retryable_assistant_error(response))
                and current.attempt < current.generation_context.retry_policy.max_attempts
            ):
                settled = AssistantRetryWaitOperation(
                    **scope,
                    generation_context=current.generation_context,
                    next_attempt=current.attempt + 1,
                    not_before=retry_not_before(
                        current.generation_context.retry_policy, current.attempt
                    ),
                    error_message=response.error_message or "Assistant request failed",
                )
            else:
                failure = _provider_error(source, response)
        else:
            calls = [
                {"sourceIndex": index}
                for index, content in enumerate(response.content)
                if getattr(content, "type", None) == "toolCall"
            ]
            if calls:
                timestamp = uuid_v7_timestamp(response_entry_id)
                planned = [
                    ToolCall(
                        status="planned",
                        source_index=call["sourceIndex"],
                        result_entry_id=lane.session.id_generator.next(timestamp),
                    )
                    for call in calls
                ]
                from ...session.types import ToolsOperation

                settled = ToolsOperation(
                    **scope,
                    batch=ToolBatch(
                        assistant_entry_id=response_entry_id,
                        configuration=configuration,
                        turn_id=turn_id,
                        calls=planned,
                    ),
                )
            elif response.stop_reason == "toolUse":
                committed = _normalize_error(
                    response, "Provider reported tool use without any tool calls"
                )
                failure = _provider_error(source, committed)
            else:
                settled = CheckpointOperation(
                    **scope,
                    continuation=Continuation(kind="may_finish", include_final_assistant=True),
                    trigger_entry_id=response_entry_id,
                )

        response_entry = MessageEntry(
            id=response_entry_id, parent_id=state.tip_id, message=committed
        )
        usage_row = UsageRow(
            id=intent.usage_id,
            usage=committed.usage,
            entry_id=response_entry_id,
            adjustment=False,
        )
        if settled is None and failure is None:
            raise SessionInvariantError("Response settlement has no durable disposition")
        record = (
            None
            if failure is None
            else operation_result_record(meta, "failed", response_entry_id, failure)
        )
        cleanup = (
            []
            if record is None
            else await operation_cleanup_writes(reader, drive.operation_id, current, drive.context)
        )
        writes: List[Any] = [
            insert_entry(response_entry),
            insert_usage(usage_row),
            set_value(branch_tip(lane.name), response_entry_id),
        ]
        if record is None:
            writes.append(
                delete_list(pending_assistant_frames(drive.operation_id, response_entry_id))
            )
        else:
            writes.extend(cleanup)
        if (
            settled is not None
            and getattr(settled, "at", None) == "summary.deciding"
            and overflow_preparation is not None
        ):
            writes.append(
                set_value(
                    operation_preparation(drive.operation_id, overflow_preparation["taskId"]),
                    overflow_preparation["preparation"],
                )
            )

        def _events(commit: Any) -> List[Any]:
            materialized = copy.copy(response_entry)
            materialized.seq = commit.seqs[0]
            materialized.timestamp = commit.timestamp
            batch: List[Any] = [
                {"type": "entry_added", "lane": lane.name, "entry": materialized, **options},
                {
                    "type": "usage",
                    "lane": lane.name,
                    "row": {**usage_row.to_json(), "seq": commit.seqs[1]},
                    "totals": commit.stats.usage,
                },
            ]
            settled_at = None if settled is None else getattr(settled, "at", None)
            if current.at == "assistant.effect_pending":
                if (
                    options.get("recovery") is not True
                    and current.attempt > 1
                    and settled_at != "assistant.retry_wait"
                ):
                    success = committed.stop_reason not in ("error", "aborted")
                    retry_end: dict = {
                        "type": "retry_end",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "step": turn_id,
                        "attempt": current.attempt,
                        "success": success,
                    }
                    if not success:
                        retry_end["finalError"] = (
                            committed.error_message
                            or f"Assistant request ended with {committed.stop_reason}"
                        )
                    batch.append(retry_end)
                if options.get("recovery") is not True and settled_at == "assistant.retry_wait":
                    batch.append(
                        {
                            "type": "retry_scheduled",
                            "lane": lane.name,
                            "runId": drive.operation_id,
                            "step": turn_id,
                            "attempt": settled.next_attempt,
                            "maxAttempts": settled.generation_context.retry_policy.max_attempts,
                            "delayMs": retry_delay_ms(
                                current.generation_context.retry_policy, current.attempt
                            ),
                            "notBefore": settled.not_before,
                            "errorMessage": settled.error_message,
                        }
                    )
                if (
                    options.get("recovery") is not True
                    and settled_at not in ("tools", "assistant.retry_wait")
                ):
                    batch.append(
                        {
                            "type": "turn_end",
                            "lane": lane.name,
                            "runId": drive.operation_id,
                            "turnId": turn_id,
                            "message": committed,
                            "toolResults": [],
                        }
                    )
                if settled_at == "summary.deciding":
                    batch.append(
                        {
                            "type": "compaction_start",
                            "lane": lane.name,
                            "runId": drive.operation_id,
                            "reason": "overflow",
                            "startedAt": commit.timestamp,
                        }
                    )
            elif settled_at != "tools":
                batch.append(
                    {
                        "type": "turn_end",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "turnId": turn_id,
                        "message": committed,
                        "toolResults": [],
                        **options,
                    }
                )
            if (
                (current.at == "deferred.effect_pending" or options.get("recovery") is not True)
                and settled_at == "deferred.suspended"
                and committed.deferred is not None
            ):
                suspend: dict = {
                    "type": "run_suspend",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "reason": "deferred",
                    "deferred": committed.deferred,
                    "poll": settled.poll,
                }
                if current.at == "deferred.effect_pending":
                    suspend.update(options)
                batch.append(suspend)
            if record is not None and failure is not None:
                batch.append(
                    {
                        "type": "run_end",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "status": "failed",
                        "error": failure,
                        "fromTipId": meta.source_tip_id,
                        "tipId": response_entry_id,
                        "endedAt": record.ended_at,
                    }
                )
            return batch

        if record is not None:
            return FinishDecision(
                writes=writes,
                record=record,
                lane={"tipId": response_entry_id},
                materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
                events=_events,
            )
        if settled is None:
            raise SessionInvariantError("Response settlement is missing its next state")
        return CommitDecision(
            writes=writes,
            operation_state=settled,
            lane={"tipId": response_entry_id},
            materialize=lambda _commit: ProcedureResult(kind="continue"),
            events=_events,
        )

    return await lane.settle_operation(intent, _plan, drive.context)


def _scope_of(state: Any, latest_assistant_entry_id: str) -> dict:
    """The uniform operation scope with the response bound as latest assistant."""
    return {
        "control": state.control,
        "settings": state.settings,
        "latest_assistant_entry_id": latest_assistant_entry_id,
    }


def _checkpoint_data(kind: str, overflow_recovery_used: bool, trigger_entry_id: str) -> Any:
    from ...session.types import CheckpointData

    return CheckpointData(
        continuation=Continuation(
            kind=kind, overflow_recovery_used=overflow_recovery_used
        ),
        trigger_entry_id=trigger_entry_id,
    )


def _field(value: Any, snake: str, camel: str, default: Any = None) -> Any:
    """Read a durable field from a dataclass instance or a JSON-replayed mapping."""
    if value is None:
        return default
    if isinstance(value, dict):
        if camel in value:
            return value[camel]
        if snake in value:
            return value[snake]
        return default
    return getattr(value, snake, default)
