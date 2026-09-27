"""Identity schemas and explicit parent binding from ``telemetry/src/index.ts``."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence

from .types import (
    SpanAttributes,
    SpanOptions,
    TelemetryContext,
    TelemetrySchemaDefinition,
    TypedSpanCallback,
    TypedSpanStarter,
)


def define_telemetry_schema[Schema: TelemetrySchemaDefinition](schema: Schema) -> Schema:
    """Return the supplied schema unchanged, without inspecting or validating it."""
    return schema


def _bind_typed_span_starter(telemetry_context: TelemetryContext) -> TypedSpanStarter:
    def start_span[Result](
        name: str,
        attributes: SpanAttributes,
        callback: TypedSpanCallback[Result],
    ) -> Awaitable[Result]:
        return telemetry_context.start_span(
            SpanOptions(name=name, attributes=attributes),
            lambda span: callback(span, _bind_typed_span_starter(span)),
        )

    return start_span


def create_typed_span_starter(
    telemetry_context: TelemetryContext,
    schemas: Sequence[TelemetrySchemaDefinition],
) -> TypedSpanStarter:
    """Bind a starter to its parent; schema values are neither inspected nor retained."""
    return _bind_typed_span_starter(telemetry_context)
