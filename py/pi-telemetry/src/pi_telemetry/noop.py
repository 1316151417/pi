"""Shared inert telemetry context from ``packages/telemetry/src/noop.ts``."""

from __future__ import annotations

from collections.abc import Awaitable

from ._promise import reject, resolve, schedule
from .types import SpanAttributes, SpanCallback, SpanOptions, SpanStatus, TelemetryContext


class _NoopTelemetrySpan:
    __slots__ = ()

    def start_span[Result](self, options: SpanOptions, callback: SpanCallback[Result]) -> Awaitable[Result]:
        try:
            result = callback(self)
        except BaseException as error:
            return schedule(reject(error))
        return schedule(resolve(result))

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        pass

    def set_attributes(self, attributes: SpanAttributes) -> None:
        pass

    def set_status(self, status: SpanStatus) -> None:
        pass


NOOP_TELEMETRY_CONTEXT: TelemetryContext = _NoopTelemetrySpan()
