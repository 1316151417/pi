"""Agent telemetry ported from ``harness/telemetry.ts``.

The TypeScript module re-exports its span vocabulary from
``@earendil-works/pi-telemetry`` and declares two runtime schemas: the AI
request schema and the harness schema. The Python port has no separate
pi-telemetry package yet, so the vocabulary this module uses (attribute values,
span and status shapes, the shared no-op context, and the in-memory recording
context) lives in the first two sections below; the schemas and the typed span
starters follow the TypeScript file one-to-one.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from .._chord.context import Context
from . import context as _context

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

# ---------------------------------------------------------------------------
# pi-telemetry vocabulary (``@earendil-works/pi-telemetry`` index.ts / noop.ts)
# ---------------------------------------------------------------------------

AttributeValue = Union[str, float, bool, List[str], List[float], List[bool]]
SpanAttributes = Dict[str, Optional[AttributeValue]]
SpanCallback = Callable[["TelemetrySpan"], Any]


@dataclass
class SpanError:
    """Structured error recorded on an unsuccessful span."""

    name: str = ""
    message: str = ""


@dataclass
class SpanStatus:
    """Span outcome. ``error`` is ignored unless ``status == "error"``."""

    status: str = "ok"  # "ok" | "error"
    error: Optional[SpanError] = None


@dataclass
class SpanOptions:
    name: str
    attributes: Optional[SpanAttributes] = None


class TelemetrySpan(Protocol):
    """One active telemetry span, which is itself a context for child spans."""

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Awaitable[Any]:
        """Run ``callback`` inside a child span and settle the child afterwards."""
        ...

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None: ...

    def set_attributes(self, attributes: SpanAttributes) -> None: ...

    def set_status(self, status: SpanStatus) -> None: ...


class TelemetryContext(Protocol):
    """Factory for top-level spans of one telemetry backend."""

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Awaitable[Any]: ...


async def _resolve(value: Any) -> Any:
    """Await callback results that are awaitable and pass through plain values."""
    if inspect.isawaitable(value):
        return await value
    return value


class NoopTelemetrySpan:
    """Span used when an application does not provide a telemetry backend.

    Mirrors the frozen ``noopTelemetrySpan`` from pi-telemetry ``noop.ts``:
    callbacks still run, nothing is recorded, and only the callback result
    matters.
    """

    __slots__ = ()

    async def start_span(self, options: SpanOptions, callback: SpanCallback) -> Any:
        return await _resolve(callback(self))

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None:
        return None

    def set_attributes(self, attributes: SpanAttributes) -> None:
        return None

    def set_status(self, status: SpanStatus) -> None:
        return None


NOOP_TELEMETRY_SPAN = NoopTelemetrySpan()

#: Shared telemetry context used when an application does not provide one.
NOOP_TELEMETRY_CONTEXT: TelemetryContext = NOOP_TELEMETRY_SPAN


# ---------------------------------------------------------------------------
# In-memory telemetry (``@earendil-works/pi-telemetry`` memory.ts)
# ---------------------------------------------------------------------------


@dataclass
class RecordedTelemetryEvent:
    """Detached snapshot of one span event."""

    name: str
    attributes: SpanAttributes = field(default_factory=dict)


@dataclass
class RecordedTelemetrySpan:
    """Detached snapshot of one recorded span."""

    id: int
    parent_id: Optional[int]
    name: str
    attributes: SpanAttributes = field(default_factory=dict)
    events: List[RecordedTelemetryEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=SpanStatus)
    settled: bool = False
    end_sequence: Optional[int] = None


@dataclass
class _MutableRecordedSpan:
    id: int
    parent_id: Optional[int]
    name: str
    attributes: SpanAttributes
    events: List[RecordedTelemetryEvent]
    status: SpanStatus
    explicit_status: bool
    settled: bool
    end_sequence: Optional[int] = None


@dataclass
class _InMemoryTelemetryState:
    spans: List[_MutableRecordedSpan] = field(default_factory=list)
    next_span_id: int = 1
    next_end_sequence: int = 1


def _copy_attribute_value(value: AttributeValue) -> AttributeValue:
    return list(value) if isinstance(value, list) else value


def _copy_attributes(attributes: Optional[SpanAttributes]) -> SpanAttributes:
    copy: SpanAttributes = {}
    if not attributes:
        return copy
    for name, value in attributes.items():
        if value is not None:
            copy[name] = _copy_attribute_value(value)
    return copy


def _merge_attributes(current: SpanAttributes, attributes: SpanAttributes) -> SpanAttributes:
    merged = _copy_attributes(current)
    for name, value in attributes.items():
        if value is not None:
            merged[name] = _copy_attribute_value(value)
    return merged


def _copy_status(status: SpanStatus) -> SpanStatus:
    if status.status == "ok":
        return SpanStatus(status="ok")
    if status.error is None:
        return SpanStatus(status="error")
    return SpanStatus(status="error", error=SpanError(name=status.error.name, message=status.error.message))


def _automatic_error_status(error: Any) -> SpanStatus:
    if isinstance(error, BaseException):
        name = getattr(error, "name", None) or type(error).__name__
        return SpanStatus(status="error", error=SpanError(name=name, message=str(error)))
    return SpanStatus(status="error")


def _settle_span(state: _InMemoryTelemetryState, span: _MutableRecordedSpan, failed: bool, error: Any = None) -> None:
    if span.settled:
        return
    if failed and not span.explicit_status:
        span.status = _automatic_error_status(error)
    span.settled = True
    span.end_sequence = state.next_end_sequence
    state.next_end_sequence += 1


def _create_span(
    state: _InMemoryTelemetryState,
    parent: Optional[_MutableRecordedSpan],
    options: SpanOptions,
) -> _MutableRecordedSpan:
    span = _MutableRecordedSpan(
        id=state.next_span_id,
        parent_id=parent.id if parent is not None else None,
        name=options.name,
        attributes=_copy_attributes(options.attributes),
        events=[],
        status=SpanStatus(status="ok"),
        explicit_status=False,
        settled=False,
    )
    state.next_span_id += 1
    return span


async def _start_in_memory_span(
    state: _InMemoryTelemetryState,
    parent: Optional[_MutableRecordedSpan],
    options: SpanOptions,
    callback: SpanCallback,
) -> Any:
    if parent is not None and parent.settled:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    recorded = _create_span(state, parent, options)
    state.spans.append(recorded)

    span = _InMemorySpan(state, recorded)

    try:
        result = callback(span)
    except BaseException as error:
        _settle_span(state, recorded, True, error)
        raise

    try:
        value = await _resolve(result)
    except BaseException as error:
        _settle_span(state, recorded, True, error)
        raise
    _settle_span(state, recorded, False)
    return value


class _InMemorySpan:
    """Live span handle writing into one in-memory recording state."""

    __slots__ = ("_state", "_span")

    def __init__(self, state: _InMemoryTelemetryState, span: _MutableRecordedSpan) -> None:
        self._state = state
        self._span = span

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Awaitable[Any]:
        return _start_in_memory_span(self._state, self._span, options, callback)

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None:
        if self._span.settled:
            return
        self._span.events.append(RecordedTelemetryEvent(name=name, attributes=_copy_attributes(attributes)))

    def set_attributes(self, attributes: SpanAttributes) -> None:
        if self._span.settled:
            return
        self._span.attributes = _merge_attributes(self._span.attributes, attributes)

    def set_status(self, status: SpanStatus) -> None:
        if self._span.settled:
            return
        self._span.status = _copy_status(status)
        self._span.explicit_status = True


class InMemoryTelemetryContext:
    """Backend-neutral recorder keeping spans in process memory.

    Create a fresh instance to isolate tests or independent recording scopes.
    """

    def __init__(self) -> None:
        self._state = _InMemoryTelemetryState()

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Awaitable[Any]:
        return _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> List[RecordedTelemetrySpan]:
        """Return detached snapshots in span-start order."""
        return [
            RecordedTelemetrySpan(
                id=span.id,
                parent_id=span.parent_id,
                name=span.name,
                attributes=_copy_attributes(span.attributes),
                events=[
                    RecordedTelemetryEvent(name=event.name, attributes=_copy_attributes(event.attributes))
                    for event in span.events
                ],
                status=_copy_status(span.status),
                settled=span.settled,
                end_sequence=span.end_sequence,
            )
            for span in self._state.spans
        ]


# ---------------------------------------------------------------------------
# Schema vocabulary (``@earendil-works/pi-telemetry`` index.ts)
# ---------------------------------------------------------------------------

TelemetryAttributeType = str  # "string" | "number" | "boolean" | "string[]" | "number[]" | "boolean[]"


@dataclass
class TelemetryAttributeMetadata:
    description: str = ""
    sensitive: bool = False
    cardinality: Optional[str] = None  # "low" | "high"


@dataclass
class TelemetryAttributeDefinition(TelemetryAttributeMetadata):
    type: TelemetryAttributeType = "string"
    values: Optional[Sequence[Any]] = None
    element_values: Optional[Sequence[Any]] = None
    examples: Optional[Sequence[Any]] = None


@dataclass
class TelemetryStartAttributeDefinition(TelemetryAttributeDefinition):
    required: bool = False


#: Event attributes share the required/optional shape of start attributes.
TelemetryEventAttributeDefinition = TelemetryStartAttributeDefinition


@dataclass
class TelemetryEventDefinition:
    description: str = ""
    attributes: Dict[str, TelemetryEventAttributeDefinition] = field(default_factory=dict)


@dataclass
class TelemetryParentDefinition:
    kind: str = "any"  # "any" | "root_or_external" | "spans"
    spans: Optional[List[str]] = None


@dataclass
class TelemetrySpanStatusDefinition:
    default: str = "ok"
    error_when: str = ""


@dataclass
class TelemetrySpanDefinition:
    description: str = ""
    parents: TelemetryParentDefinition = field(default_factory=TelemetryParentDefinition)
    start_attributes: Dict[str, TelemetryStartAttributeDefinition] = field(default_factory=dict)
    end_attributes: Dict[str, TelemetryAttributeDefinition] = field(default_factory=dict)
    events: Optional[Dict[str, TelemetryEventDefinition]] = None
    status: TelemetrySpanStatusDefinition = field(default_factory=TelemetrySpanStatusDefinition)


@dataclass
class TelemetrySchemaDefinition:
    version: int = 1
    spans: Dict[str, TelemetrySpanDefinition] = field(default_factory=dict)


#: Expected attributes of one span start call. The TypeScript ``ExactTelemetryAttributes``
#: rejects unknown keys at compile time; the Python port relies on the schema only.
ExactTelemetryAttributes = Dict[str, Any]
#: Typed span handed to schema callbacks (a span plus its schema-bound attribute names).
SchemaTelemetrySpan = TelemetrySpan
#: ``(name, attributes, callback) -> awaitable`` bound to one schema vocabulary.
TypedSpanStarter = Callable[..., Awaitable[Any]]


def _parents(kind: str, spans: Optional[Sequence[str]] = None) -> TelemetryParentDefinition:
    return TelemetryParentDefinition(kind=kind, spans=list(spans) if spans is not None else None)


def _start(
    type: TelemetryAttributeType, description: str, required: bool, **options: Any
) -> TelemetryStartAttributeDefinition:
    return TelemetryStartAttributeDefinition(type=type, description=description, required=required, **options)


def _end(type: TelemetryAttributeType, description: str, **options: Any) -> TelemetryAttributeDefinition:
    return TelemetryAttributeDefinition(type=type, description=description, **options)


# ---------------------------------------------------------------------------
# AI request schema
# ---------------------------------------------------------------------------

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

HOOK_NAMES: Tuple[str, ...] = (
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

EVENT_TYPES: Tuple[str, ...] = (
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

_OPERATION_START_ATTRIBUTES: Dict[str, TelemetryStartAttributeDefinition] = {
    "pi.session.id": _start("string", "Session id", True, cardinality="high"),
    "pi.lane.name": _start("string", "Lane name", True, cardinality="high"),
    "pi.operation.id": _start("string", "Durable operation id", True, cardinality="high"),
    "pi.operation.recovery": _start("boolean", "Whether this invocation resumes durable work", True),
}

_OPERATION_ERROR_ATTRIBUTES: Dict[str, TelemetryAttributeDefinition] = {
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
AGENT_TELEMETRY_SCHEMAS: Tuple[TelemetrySchemaDefinition, ...] = (
    AI_TELEMETRY_SCHEMA,
    HARNESS_TELEMETRY_SCHEMA,
)


# ---------------------------------------------------------------------------
# Schema-derived aliases
# ---------------------------------------------------------------------------

AiSpanName = str
AiSpanStartAttributes = Dict[str, Any]
AiSpanEndAttributes = Dict[str, Any]
AiSpanAttributes = Dict[str, Any]
AiSpanEventName = str
AiSpanEventAttributes = Dict[str, Any]
AiTelemetrySpan = TelemetrySpan
AiSpan = Dict[str, Any]

HarnessSpanName = str
HarnessSpanStartAttributes = Dict[str, Any]
HarnessSpanEndAttributes = Dict[str, Any]
HarnessSpanAttributes = Dict[str, Any]
HarnessSpanEventName = str
HarnessSpanEventAttributes = Dict[str, Any]
HarnessTelemetrySpan = TelemetrySpan
HarnessSpan = Dict[str, Any]


# ---------------------------------------------------------------------------
# Typed span starters
# ---------------------------------------------------------------------------


async def start_ai_span(
    name: AiSpanName,
    attributes: ExactTelemetryAttributes,
    callback: Callable[[AiTelemetrySpan, Context], Any],
    context: Context,
) -> Any:
    """Start one ``pi.ai.*`` span through the telemetry parent attached to ``context``."""

    def invoke(span: TelemetrySpan) -> Any:
        return callback(span, _context.with_telemetry_context(span, context))

    return await _context.get_telemetry_context(context).start_span(
        SpanOptions(name=name, attributes=attributes), invoke
    )


async def start_harness_span(
    name: HarnessSpanName,
    attributes: ExactTelemetryAttributes,
    callback: Callable[[HarnessTelemetrySpan, Context], Any],
    context: Context,
) -> Any:
    """Start one ``pi.harness.*`` span through the telemetry parent attached to ``context``."""

    def invoke(span: TelemetrySpan) -> Any:
        return callback(span, _context.with_telemetry_context(span, context))

    return await _context.get_telemetry_context(context).start_span(
        SpanOptions(name=name, attributes=attributes), invoke
    )
