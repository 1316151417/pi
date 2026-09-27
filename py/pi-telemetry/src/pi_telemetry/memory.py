"""Passive in-memory span recording from ``packages/telemetry/src/memory.ts``."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import cast

from ._promise import reject, schedule
from .noop import NOOP_TELEMETRY_CONTEXT
from .types import (
    AttributeValue,
    RecordedTelemetryEvent,
    RecordedTelemetrySpan,
    SpanAttributes,
    SpanCallback,
    SpanError,
    SpanOptions,
    SpanStatus,
)


@dataclass
class _MutableRecordedTelemetrySpan:
    id: int
    parent_id: int | None
    name: str
    attributes: SpanAttributes
    events: list[RecordedTelemetryEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=SpanStatus)
    explicit_status: bool = False
    settled: bool = False
    end_sequence: int | None = None


@dataclass
class _InMemoryTelemetryState:
    spans: list[_MutableRecordedTelemetrySpan] = field(default_factory=list)
    next_span_id: int = 1
    next_end_sequence: int = 1


def _copy_attribute_value(value: AttributeValue) -> AttributeValue:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return cast(AttributeValue, list(value))
    return value


def _copy_attributes(attributes: SpanAttributes | None = None) -> dict[str, AttributeValue | None]:
    copied: dict[str, AttributeValue | None] = {}
    if attributes is None:
        return copied
    for name, value in attributes.items():
        if value is not None:
            copied[name] = _copy_attribute_value(value)
    return copied


def _copy_status(status: SpanStatus) -> SpanStatus:
    if status.status == "ok":
        return SpanStatus(status="ok")
    if status.error is not None:
        return SpanStatus(status="error", error=SpanError(name=status.error.name, message=status.error.message))
    return SpanStatus(status="error")


def _settle_span(
    state: _InMemoryTelemetryState,
    span: _MutableRecordedTelemetrySpan,
    failed: bool,
    error: BaseException | None = None,
) -> None:
    if span.settled:
        return
    if failed and not span.explicit_status:
        status = SpanStatus(status="error")
        try:
            if isinstance(error, BaseException):
                status.error = SpanError(name=getattr(error, "name", type(error).__name__), message=str(error))
        except BaseException:
            # Inspecting an exception must not replace the original failure.
            pass
        span.status = status
    span.settled = True
    span.end_sequence = state.next_end_sequence
    state.next_end_sequence += 1


class _InMemoryTelemetrySpan:
    __slots__ = ("_state", "_recorded_span")

    def __init__(self, state: _InMemoryTelemetryState, recorded_span: _MutableRecordedTelemetrySpan) -> None:
        self._state = state
        self._recorded_span = recorded_span

    def start_span[Result](self, options: SpanOptions, callback: SpanCallback[Result]) -> Awaitable[Result]:
        return _start_in_memory_span(self._state, self._recorded_span, options, callback)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        if self._recorded_span.settled:
            return
        try:
            self._recorded_span.events.append(RecordedTelemetryEvent(name=name, attributes=_copy_attributes(attributes)))
        except BaseException:
            # Recording is passive, including unreadable or malformed payloads.
            pass

    def set_attributes(self, attributes: SpanAttributes) -> None:
        if self._recorded_span.settled:
            return
        try:
            merged = _copy_attributes(self._recorded_span.attributes)
            for name, value in attributes.items():
                if value is not None:
                    merged[name] = _copy_attribute_value(value)
            self._recorded_span.attributes = merged
        except BaseException:
            pass

    def set_status(self, status: SpanStatus) -> None:
        if self._recorded_span.settled:
            return
        try:
            self._recorded_span.status = _copy_status(status)
            self._recorded_span.explicit_status = True
        except BaseException:
            pass


def _start_in_memory_span[Result](
    state: _InMemoryTelemetryState,
    parent: _MutableRecordedTelemetrySpan | None,
    options: SpanOptions,
    callback: SpanCallback[Result],
) -> Awaitable[Result]:
    if parent is not None and parent.settled:
        return NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    try:
        name = options.name
        attributes = _copy_attributes(options.attributes)
        recorded_span = _MutableRecordedTelemetrySpan(
            id=state.next_span_id,
            parent_id=parent.id if parent is not None else None,
            name=name,
            attributes=attributes,
        )
        state.next_span_id += 1
        state.spans.append(recorded_span)
    except BaseException:
        return NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    try:
        result = callback(_InMemoryTelemetrySpan(state, recorded_span))
    except BaseException as error:
        _settle_span(state, recorded_span, True, error)
        return schedule(reject(error))

    async def finish() -> Result:
        try:
            value = await result if inspect.isawaitable(result) else result
        except BaseException as error:
            _settle_span(state, recorded_span, True, error)
            raise
        _settle_span(state, recorded_span, False)
        return value

    completion = schedule(finish())
    if isinstance(completion, asyncio.Task):
        # A task cancelled before its first execution never enters finish().
        def settle_cancelled(task: asyncio.Task[Result]) -> None:
            if task.cancelled() and not recorded_span.settled:
                if inspect.iscoroutine(result):
                    result.close()
                _settle_span(state, recorded_span, True, asyncio.CancelledError())

        completion.add_done_callback(settle_cancelled)
    return completion


class InMemoryTelemetryContext:
    """Backend-neutral reference context with deterministic process-local records."""

    def __init__(self) -> None:
        self._state = _InMemoryTelemetryState()

    def start_span[Result](self, options: SpanOptions, callback: SpanCallback[Result]) -> Awaitable[Result]:
        return _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> list[RecordedTelemetrySpan]:
        """Return detached snapshots, ordered by admission rather than completion."""
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
