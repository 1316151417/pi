"""Telemetry contracts and schema shapes from ``packages/telemetry/src/index.ts``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

type AttributeValue = str | int | float | bool | Sequence[str] | Sequence[int | float] | Sequence[bool]
type SpanAttributes = Mapping[str, AttributeValue | None]
type SpanCallback[Result] = Callable[[TelemetrySpan], Result | Awaitable[Result]]


@dataclass
class SpanOptions:
    name: str
    attributes: SpanAttributes | None = None


@dataclass
class SpanError:
    name: str
    message: str


@dataclass
class SpanStatus:
    status: Literal["ok", "error"] = "ok"
    error: SpanError | None = None


class TelemetryContext(Protocol):
    def start_span[Result](self, options: SpanOptions, callback: SpanCallback[Result]) -> Awaitable[Result]: ...


class TelemetrySpan(TelemetryContext, Protocol):
    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None: ...

    def set_attributes(self, attributes: SpanAttributes) -> None: ...

    def set_status(self, status: SpanStatus) -> None: ...


type TelemetryAttributeType = Literal["string", "number", "boolean", "string[]", "number[]", "boolean[]"]
type TelemetryScalarValues = Sequence[str] | Sequence[int | float] | Sequence[bool]
type TelemetryAttributeExamples = TelemetryScalarValues | Sequence[Sequence[str]] | Sequence[Sequence[int | float]] | Sequence[Sequence[bool]]


@dataclass(kw_only=True)
class TelemetryAttributeMetadata:
    description: str
    sensitive: bool | None = None
    cardinality: Literal["low", "high"] | None = None


@dataclass(kw_only=True)
class TelemetryAttributeDefinition(TelemetryAttributeMetadata):
    type: TelemetryAttributeType
    values: TelemetryScalarValues | None = None
    element_values: TelemetryScalarValues | None = None
    examples: TelemetryAttributeExamples | None = None


@dataclass(kw_only=True)
class TelemetryStartAttributeDefinition(TelemetryAttributeDefinition):
    required: bool


TelemetryEventAttributeDefinition = TelemetryStartAttributeDefinition


@dataclass
class TelemetryEventDefinition:
    description: str
    attributes: Mapping[str, TelemetryEventAttributeDefinition]


@dataclass
class TelemetryParentDefinition:
    kind: Literal["any", "root_or_external", "spans"]
    spans: Sequence[str] | None = None


@dataclass(kw_only=True)
class TelemetrySpanStatusDefinition:
    default: Literal["ok"] = "ok"
    error_when: str


@dataclass
class TelemetrySpanDefinition:
    description: str
    parents: TelemetryParentDefinition
    start_attributes: Mapping[str, TelemetryStartAttributeDefinition]
    end_attributes: Mapping[str, TelemetryAttributeDefinition]
    status: TelemetrySpanStatusDefinition
    events: Mapping[str, TelemetryEventDefinition] | None = None


@dataclass
class TelemetrySchemaDefinition:
    version: int
    spans: Mapping[str, TelemetrySpanDefinition]


# Python's static type system cannot derive literal-key overloads from schema
# values. These aliases retain the corresponding public structural vocabulary;
# neither the source helpers nor this port validate schemas at runtime.
type InferRequiredAndOptionalAttributes[Definitions] = SpanAttributes
type InferStartAttributes[Definitions] = SpanAttributes
type InferOptionalAttributes[Definitions] = SpanAttributes
type ExactTelemetryAttributes[Expected, Actual] = SpanAttributes
type InferEventAttributes[Definitions] = SpanAttributes
type TelemetrySchemaSpanName[Schema] = str
type TelemetrySchemaSpanStartAttributes[Schema, Name] = SpanAttributes
type TelemetrySchemaSpanEndAttributes[Schema, Name] = SpanAttributes
type TelemetrySchemaSpanEventName[Schema, Name] = str
type TelemetrySchemaSpanEventAttributes[Schema, Name, EventName] = SpanAttributes
type SchemaTelemetrySpan[Schema, Name] = TelemetrySpan


@dataclass
class TelemetrySchemaSpanUnion[Schema]:
    name: str
    start_attributes: SpanAttributes
    end_attributes: SpanAttributes
    events: Mapping[str, SpanAttributes]


type TypedSpanCallback[Result] = Callable[[TelemetrySpan, TypedSpanStarter], Result | Awaitable[Result]]


class TypedSpanStarter(Protocol):
    def __call__[Result](
        self,
        name: str,
        attributes: SpanAttributes,
        callback: TypedSpanCallback[Result],
    ) -> Awaitable[Result]: ...


@dataclass
class RecordedTelemetryEvent:
    """Detached event snapshot; mutating it cannot change recorded state."""

    name: str
    attributes: SpanAttributes = field(default_factory=dict)


@dataclass
class RecordedTelemetrySpan:
    """Detached span snapshot in span-start order."""

    id: int
    parent_id: int | None
    name: str
    attributes: SpanAttributes = field(default_factory=dict)
    events: list[RecordedTelemetryEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=SpanStatus)
    settled: bool = False
    end_sequence: int | None = None
