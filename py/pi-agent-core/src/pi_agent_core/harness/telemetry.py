"""Agent telemetry schemas and context-bound starters from ``harness/telemetry.ts``.

Core telemetry contracts, recording, and no-op behavior come from pi-telemetry.
The schema descriptions, attribute definitions, parents, and status policies
are the declarations in the TypeScript module; no runtime validation is added.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, cast, overload

from pi_telemetry import (
    AttributeValue, ExactTelemetryAttributes, InMemoryTelemetryContext,
    NOOP_TELEMETRY_CONTEXT, RecordedTelemetryEvent, RecordedTelemetrySpan,
    SchemaTelemetrySpan, SpanAttributes, SpanCallback, SpanError, SpanOptions,
    SpanStatus, TelemetryAttributeDefinition, TelemetryAttributeMetadata,
    TelemetryAttributeType, TelemetryContext, TelemetryEventAttributeDefinition,
    TelemetryEventDefinition, TelemetryParentDefinition, TelemetrySchemaDefinition,
    TelemetrySchemaSpanEndAttributes, TelemetrySchemaSpanEventAttributes,
    TelemetrySchemaSpanEventName, TelemetrySchemaSpanName,
    TelemetrySchemaSpanStartAttributes, TelemetrySchemaSpanUnion, TelemetrySpan,
    TelemetrySpanDefinition, TelemetrySpanStatusDefinition,
    TelemetryStartAttributeDefinition, TypedSpanStarter,
)
from pi_telemetry.noop import _NoopTelemetrySpan as NoopTelemetrySpan

from ._telemetry_types import (
    AgentTelemetrySpan, AiSpan, AiSpanAttributes, AiSpanEndAttributes,
    AiSpanEventAttributes, AiSpanEventName, AiSpanName, AiSpanStartAttributes,
    AiTelemetrySpan, AiRequestStartAttributes, CompactionSpanEndAttributes,
    CompactionSpanStartAttributes, CheckpointSpanStartAttributes,
    EmptySpanEndAttributes, EventHandlerSpanStartAttributes, HarnessEventType,
    HarnessSpan, HarnessSpanAttributes, HarnessSpanEndAttributes,
    HarnessSpanEventAttributes, HarnessSpanEventName, HarnessSpanName,
    HarnessSpanStartAttributes, HarnessTelemetrySpan, HookName,
    HookSpanEndAttributes, HookSpanStartAttributes, NavigationSpanEndAttributes,
    NavigationSpanStartAttributes, RunSpanEndAttributes, RunSpanStartAttributes,
    SessionWriteSpanEndAttributes, SessionWriteSpanStartAttributes,
    SleepSpanEndAttributes, SleepSpanStartAttributes, StepSpanEndAttributes,
    StepSpanStartAttributes, ToolSpanEndAttributes, ToolSpanStartAttributes,
    TurnSpanStartAttributes,
)
from .context import Context, get_telemetry_context, with_telemetry_context

# Preserve the previously public no-op names as aliases of the common backend.
NOOP_TELEMETRY_SPAN = cast(TelemetrySpan, NOOP_TELEMETRY_CONTEXT)

__all__ = [
    # pi-telemetry vocabulary
    "AttributeValue",
    "SpanAttributes",
    "SpanOptions",
    "SpanError",
    "SpanStatus",
    "SpanCallback",
    "TelemetryContext",
    "TelemetrySpan",
    "NoopTelemetrySpan",
    "NOOP_TELEMETRY_SPAN",
    "NOOP_TELEMETRY_CONTEXT",
    "RecordedTelemetryEvent",
    "RecordedTelemetrySpan",
    "InMemoryTelemetryContext",
    # schema vocabulary
    "TelemetryAttributeType",
    "TelemetryAttributeMetadata",
    "TelemetryAttributeDefinition",
    "TelemetryStartAttributeDefinition",
    "TelemetryEventAttributeDefinition",
    "TelemetryEventDefinition",
    "TelemetryParentDefinition",
    "TelemetrySpanStatusDefinition",
    "TelemetrySpanDefinition",
    "TelemetrySchemaDefinition",
    "ExactTelemetryAttributes",
    "SchemaTelemetrySpan",
    "TypedSpanStarter",
    # schemas and their vocabularies
    "HOOK_NAMES",
    "EVENT_TYPES",
    "AI_TELEMETRY_SCHEMA",
    "HARNESS_TELEMETRY_SCHEMA",
    "AGENT_TELEMETRY_SCHEMAS",
    # schema-derived aliases
    "AiSpanName",
    "AiSpanStartAttributes",
    "AiSpanEndAttributes",
    "AiSpanAttributes",
    "AiSpanEventName",
    "AiSpanEventAttributes",
    "AiTelemetrySpan",
    "AiSpan",
    "HarnessSpanName",
    "HarnessSpanStartAttributes",
    "HarnessSpanEndAttributes",
    "HarnessSpanAttributes",
    "HarnessSpanEventName",
    "HarnessSpanEventAttributes",
    "HarnessTelemetrySpan",
    "HarnessSpan",
    # span starters
    "start_ai_span",
    "start_harness_span",
]


__all__ += [
    "TelemetrySchemaSpanEndAttributes", "TelemetrySchemaSpanEventAttributes",
    "TelemetrySchemaSpanEventName", "TelemetrySchemaSpanName",
    "TelemetrySchemaSpanStartAttributes", "TelemetrySchemaSpanUnion",
]


def _parents(
    kind: Literal["any", "root_or_external", "spans"],
    spans: Sequence[str] | None = None,
) -> TelemetryParentDefinition:
    return TelemetryParentDefinition(kind=kind, spans=list(spans) if spans is not None else None)


def _start(
    type: TelemetryAttributeType,
    description: str,
    required: bool,
    *,
    cardinality: Literal["low", "high"] | None = None,
    values: Sequence[str] | None = None,
    element_values: Sequence[str] | None = None,
) -> TelemetryStartAttributeDefinition:
    return TelemetryStartAttributeDefinition(
        type=type, description=description, required=required, cardinality=cardinality,
        values=values, element_values=element_values,
    )


def _end(
    type: TelemetryAttributeType,
    description: str,
    *,
    cardinality: Literal["low", "high"] | None = None,
    values: Sequence[str] | None = None,
) -> TelemetryAttributeDefinition:
    return TelemetryAttributeDefinition(
        type=type, description=description, cardinality=cardinality, values=values,
    )


AI_TELEMETRY_SCHEMA = TelemetrySchemaDefinition(
    version=1,
    spans={
        "pi.ai.request": TelemetrySpanDefinition(
            description="One logical request to an AI provider",
            parents=_parents("any"),
            start_attributes={
                "pi.ai.operation": _start(
                    "string",
                    "Logical provider operation",
                    True,
                    values=("stream", "fetch_deferred", "cancel_deferred", "generate_images"),
                ),
                "pi.ai.provider": _start("string", "Selected provider id", True),
                "pi.ai.model": _start("string", "Requested model id", True),
                "pi.ai.api": _start("string", "Provider API id", True),
                "pi.ai.streaming": _start("boolean", "Whether this operation returns a stream", True),
                "pi.ai.deferred": _start(
                    "boolean",
                    "Whether the operation requests or participates in deferred execution",
                    False,
                ),
            },
            end_attributes={
                "pi.ai.response.model": _end("string", "Concrete response model"),
                "pi.ai.response.id": _end("string", "Provider response id", cardinality="high"),
                "pi.ai.response.stop_reason": _end(
                    "string",
                    "Normalized terminal response reason",
                    values=("stop", "length", "tool_use", "error", "aborted", "deferred"),
                ),
                "pi.ai.http.status_code": _end("number", "Final HTTP status"),
                "pi.ai.usage.input_tokens": _end("number", "Reported input tokens"),
                "pi.ai.usage.output_tokens": _end("number", "Reported output tokens"),
                "pi.ai.usage.cache_read_tokens": _end("number", "Reported cache-read tokens"),
                "pi.ai.usage.cache_write_tokens": _end("number", "Reported cache-write tokens"),
                "pi.ai.usage.reasoning_tokens": _end("number", "Reported reasoning tokens"),
                "pi.ai.usage.total_tokens": _end("number", "Reported total tokens"),
                "pi.ai.usage.cost": _end("number", "Reported total cost"),
                "pi.ai.stream.chunk_count": _end("number", "Streamed update chunk count"),
                "pi.ai.stream.time_to_first_chunk_ms": _end(
                    "number", "Elapsed milliseconds to first update chunk"
                ),
                "pi.ai.error.type": _end("string", "Provider or transport error class", cardinality="low"),
            },
            status=TelemetrySpanStatusDefinition(
                default="ok", error_when="The operation throws or returns an error result"
            ),
        ),
    },
)


# ---------------------------------------------------------------------------
# Harness schema
# ---------------------------------------------------------------------------

HOOK_NAMES: tuple[HookName, ...] = (
    "before_run",
    "before_drive",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
)

EVENT_TYPES: tuple[HarnessEventType, ...] = (
    "run_start",
    "run_resume",
    "run_suspend",
    "operation_abort",
    "run_end",
    "fault",
    "handler_error",
    "turn_start",
    "turn_end",
    "retry_scheduled",
    "retry_start",
    "retry_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_start",
    "tool_update",
    "tool_end",
    "entry_added",
    "queue_update",
    "value_update",
    "config_update",
    "compaction_start",
    "compaction_end",
    "navigation_start",
    "navigation_end",
    "lane_created",
    "usage",
)

_OPERATION_START_ATTRIBUTES: dict[str, TelemetryStartAttributeDefinition] = {
    "pi.session.id": _start("string", "Session id", True, cardinality="high"),
    "pi.lane.name": _start("string", "Lane name", True, cardinality="high"),
    "pi.operation.id": _start("string", "Durable operation id", True, cardinality="high"),
    "pi.operation.recovery": _start("boolean", "Whether this invocation resumes durable work", True),
}

_OPERATION_ERROR_ATTRIBUTES: dict[str, TelemetryAttributeDefinition] = {
    "pi.error.code": _end("string", "Stable operation error code", cardinality="low"),
    "pi.error.type": _end("string", "Low-cardinality operation error class", cardinality="low"),
}

HARNESS_TELEMETRY_SCHEMA = TelemetrySchemaDefinition(
    version=1,
    spans={
        "pi.harness.run": TelemetrySpanDefinition(
            description="One admitted in-process run invocation",
            parents=_parents("root_or_external"),
            start_attributes={
                **_OPERATION_START_ATTRIBUTES,
                "pi.operation.kind": _start("string", "Run operation kind", True, values=("run",)),
            },
            end_attributes={
                "pi.operation.outcome": _end(
                    "string",
                    "Run invocation outcome",
                    values=("completed", "aborted", "failed", "suspended"),
                ),
                **_OPERATION_ERROR_ATTRIBUTES,
            },
            status=TelemetrySpanStatusDefinition(default="ok", error_when="The run fails or throws"),
        ),
        "pi.harness.compaction": TelemetrySpanDefinition(
            description="One admitted in-process manual compaction invocation",
            parents=_parents("root_or_external"),
            start_attributes={
                **_OPERATION_START_ATTRIBUTES,
                "pi.operation.kind": _start("string", "Compaction operation kind", True, values=("compaction",)),
            },
            end_attributes={
                "pi.operation.outcome": _end(
                    "string",
                    "Compaction invocation outcome",
                    values=("completed", "declined", "aborted", "failed"),
                ),
                **_OPERATION_ERROR_ATTRIBUTES,
            },
            status=TelemetrySpanStatusDefinition(default="ok", error_when="The compaction fails or throws"),
        ),
        "pi.harness.navigation": TelemetrySpanDefinition(
            description="One admitted in-process navigation invocation",
            parents=_parents("root_or_external"),
            start_attributes={
                **_OPERATION_START_ATTRIBUTES,
                "pi.operation.kind": _start("string", "Navigation operation kind", True, values=("navigation",)),
            },
            end_attributes={
                "pi.operation.outcome": _end(
                    "string",
                    "Navigation invocation outcome",
                    values=("completed", "declined", "aborted", "failed"),
                ),
                **_OPERATION_ERROR_ATTRIBUTES,
            },
            status=TelemetrySpanStatusDefinition(default="ok", error_when="The navigation fails or throws"),
        ),
        "pi.harness.checkpoint": TelemetrySpanDefinition(
            description="One run checkpoint",
            parents=_parents("spans", ["pi.harness.run"]),
            start_attributes={
                "pi.lane.name": _start("string", "Lane name", True, cardinality="high"),
                "pi.operation.id": _start("string", "Durable operation id", True, cardinality="high"),
                "pi.checkpoint.kind": _start(
                    "string", "Checkpoint purpose", True, values=("normal", "abort_reconcile")
                ),
            },
            end_attributes={},
            status=TelemetrySpanStatusDefinition(default="ok", error_when="Checkpoint work throws"),
        ),
        "pi.harness.turn": TelemetrySpanDefinition(
            description="One assistant response and its tool batch",
            parents=_parents("spans", ["pi.harness.run"]),
            start_attributes={
                "pi.lane.name": _start("string", "Lane name", True, cardinality="high"),
                "pi.operation.id": _start("string", "Durable operation id", True, cardinality="high"),
                "pi.turn.id": _start("string", "Invocation-local turn id", True, cardinality="high"),
            },
            end_attributes={},
            status=TelemetrySpanStatusDefinition(default="ok", error_when="Turn work throws"),
        ),
        "pi.harness.step": TelemetrySpanDefinition(
            description="One durable retry attempt",
            parents=_parents(
                "spans",
                [
                    "pi.harness.turn",
                    "pi.harness.checkpoint",
                    "pi.harness.compaction",
                    "pi.harness.navigation",
                ],
            ),
            start_attributes={
                "pi.lane.name": _start("string", "Lane name", True, cardinality="high"),
                "pi.operation.id": _start("string", "Durable operation id", True, cardinality="high"),
                "pi.step.kind": _start(
                    "string", "Retryable step kind", True, values=("assistant", "compaction", "branch_summary")
                ),
                "pi.step.attempt": _start("number", "One-based durable attempt number", True),
                "pi.compaction.reason": _start(
                    "string", "Compaction trigger", False, values=("manual", "threshold", "overflow")
                ),
            },
            end_attributes={
                "pi.step.outcome": _end(
                    "string",
                    "Attempt outcome",
                    values=("succeeded", "retry", "failed", "aborted", "deferred", "overflow"),
                ),
            },
            status=TelemetrySpanStatusDefinition(default="ok", error_when="The attempt retries, fails, or throws"),
        ),
        "pi.harness.tool": TelemetrySpanDefinition(
            description="One raw phase-2 tool execution",
            parents=_parents("spans", ["pi.harness.turn", "pi.harness.run"]),
            start_attributes={
                "pi.lane.name": _start("string", "Lane name", True, cardinality="high"),
                "pi.operation.id": _start("string", "Durable operation id", True, cardinality="high"),
                "pi.turn.id": _start("string", "Invocation-local live turn id", False, cardinality="high"),
                "pi.tool.name": _start("string", "Tool name", True),
                "pi.tool.call_id": _start("string", "Tool call id", True, cardinality="high"),
                "pi.tool.replay": _start("string", "Declared replay policy", True, values=("never", "safe")),
                "pi.tool.recovery": _start("boolean", "Whether this is recovery execution", True),
            },
            end_attributes={
                "pi.tool.is_error": _end("boolean", "Whether raw phase-2 execution returned an error"),
            },
            status=TelemetrySpanStatusDefinition(
                default="ok", error_when="Raw phase-2 execution returns an error"
            ),
        ),
        "pi.harness.hook": TelemetrySpanDefinition(
            description="One registered hook handler invocation",
            parents=_parents("any"),
            start_attributes={
                "pi.lane.name": _start("string", "Lane name", True, cardinality="high"),
                "pi.operation.id": _start("string", "Durable operation id when accepted", False, cardinality="high"),
                "pi.hook.name": _start("string", "Hook name", True, values=HOOK_NAMES),
                "pi.hook.registration_id": _start(
                    "string", "Optional hook registration metadata", False
                ),
            },
            end_attributes={
                "pi.hook.outcome": _end(
                    "string", "Handler outcome", values=("completed", "skipped", "blocked", "failed")
                ),
            },
            status=TelemetrySpanStatusDefinition(default="ok", error_when="The handler throws"),
        ),
        "pi.harness.sleep": TelemetrySpanDefinition(
            description="One retry delay",
            parents=_parents(
                "spans",
                [
                    "pi.harness.run",
                    "pi.harness.compaction",
                    "pi.harness.navigation",
                    "pi.harness.turn",
                    "pi.harness.checkpoint",
                ],
            ),
            start_attributes={
                "pi.operation.id": _start("string", "Durable operation id", True, cardinality="high"),
                "pi.sleep.delay_ms": _start("number", "Requested delay in milliseconds", True),
            },
            end_attributes={
                "pi.sleep.outcome": _end("string", "Delay outcome", values=("elapsed", "aborted")),
            },
            status=TelemetrySpanStatusDefinition(default="ok", error_when="Sleep work throws"),
        ),
        "pi.harness.event_handler": TelemetrySpanDefinition(
            description="One passive event listener invocation",
            parents=_parents("any"),
            start_attributes={
                "pi.event.type": _start(
                    "string", "Delivered harness event type", True, cardinality="low", values=EVENT_TYPES
                ),
                "pi.lane.name": _start("string", "Lane name for lane-scoped events", False, cardinality="high"),
            },
            end_attributes={},
            status=TelemetrySpanStatusDefinition(default="ok", error_when="The listener throws"),
        ),
        "pi.session.write": TelemetrySpanDefinition(
            description="One committed session transaction",
            parents=_parents("any"),
            start_attributes={
                "pi.session.id": _start("string", "Session id", True, cardinality="high"),
                "pi.lane.name": _start("string", "Lane name when supplied by the caller", False, cardinality="high"),
                "pi.operation.id": _start(
                    "string", "Durable operation id when supplied by the caller", False, cardinality="high"
                ),
                "pi.session.item_count": _start("number", "Number of writes in the transaction", True),
                "pi.session.item_kinds": _start(
                    "string[]",
                    "Distinct write kinds in the transaction",
                    True,
                    element_values=("entry", "usage", "value", "list"),
                ),
            },
            end_attributes={
                "pi.session.first_seq": _end("number", "First committed sequence in the transaction"),
                "pi.session.last_seq": _end("number", "Last committed sequence in the transaction"),
            },
            status=TelemetrySpanStatusDefinition(default="ok", error_when="Storage rejects the transaction"),
        ),
    },
)

#: Combined typed span vocabulary for agent-owned AI-request and harness telemetry.
AGENT_TELEMETRY_SCHEMAS: tuple[TelemetrySchemaDefinition, ...] = (
    AI_TELEMETRY_SCHEMA,
    HARNESS_TELEMETRY_SCHEMA,
)

def start_ai_span[Result](
    name: AiSpanName,
    attributes: AiRequestStartAttributes,
    callback: Callable[[AiTelemetrySpan[AiSpanName], Context], Result | Awaitable[Result]],
    context: Context,
) -> Awaitable[Result]:
    """Start immediately beneath the explicit telemetry parent in ``context``."""
    return get_telemetry_context(context).start_span(
        SpanOptions(name=name, attributes=cast(SpanAttributes, attributes)),
        lambda span: callback(cast(AiTelemetrySpan[AiSpanName], span), with_telemetry_context(span, context)),
    )


@overload
def start_harness_span[Result](name: Literal["pi.harness.run"], attributes: RunSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[RunSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.compaction"], attributes: CompactionSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[CompactionSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.navigation"], attributes: NavigationSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[NavigationSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.checkpoint"], attributes: CheckpointSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[EmptySpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.turn"], attributes: TurnSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[EmptySpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.step"], attributes: StepSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[StepSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.tool"], attributes: ToolSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[ToolSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.hook"], attributes: HookSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[HookSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.sleep"], attributes: SleepSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[SleepSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.harness.event_handler"], attributes: EventHandlerSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[EmptySpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...
@overload
def start_harness_span[Result](name: Literal["pi.session.write"], attributes: SessionWriteSpanStartAttributes, callback: Callable[[AgentTelemetrySpan[SessionWriteSpanEndAttributes], Context], Result | Awaitable[Result]], context: Context) -> Awaitable[Result]: ...


def start_harness_span[Result](
    name: HarnessSpanName,
    attributes: HarnessSpanStartAttributes[HarnessSpanName],
    callback: Callable[..., Result | Awaitable[Result]],
    context: Context,
) -> Awaitable[Result]:
    """Start one schema span and pass its parent-bound context to the callback.

    Per-name callback types are expressed by the overloads. The implementation
    forwards the exact backend handle, result, and awaitable without wrapping.
    """
    return get_telemetry_context(context).start_span(
        SpanOptions(name=name, attributes=cast(SpanAttributes, attributes)),
        lambda span: callback(span, with_telemetry_context(span, context)),
    )
