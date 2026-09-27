"""Telemetry-aware context helpers ported from ``harness/context.ts``.

The primitives share the independent ``pi_chord`` runtime. This module adds the
telemetry parent using the independent ``pi_telemetry`` contracts.
"""

from __future__ import annotations

from typing import cast

from pi_telemetry import NOOP_TELEMETRY_CONTEXT, TelemetryContext

from .._chord.context import (
    BACKGROUND_CONTEXT,
    TODO_CONTEXT,
    UNDEFINED,
    Context,
    ContextKey,
    await_with_context,
    create_context_key,
    with_abort_signal,
    with_cancel,
    with_context_value,
    without_abort_signal,
)

__all__ = [
    "await_with_context",
    "BACKGROUND_CONTEXT",
    "Context",
    "ContextKey",
    "create_context_key",
    "TODO_CONTEXT",
    "with_abort_signal",
    "with_cancel",
    "with_context_value",
    "without_abort_signal",
    "TELEMETRY_CONTEXT_KEY",
    "get_telemetry_context",
    "with_telemetry_context",
]

TELEMETRY_CONTEXT_KEY: ContextKey[TelemetryContext] = create_context_key("pi.telemetryContext")


def get_telemetry_context(context: Context) -> TelemetryContext:
    """Return the telemetry parent attached to a context, or the shared no-op parent."""
    attached = context.value(TELEMETRY_CONTEXT_KEY)
    if attached is UNDEFINED or attached is None:
        return NOOP_TELEMETRY_CONTEXT
    return cast(TelemetryContext, attached)


def with_telemetry_context(telemetry_context: TelemetryContext, context: Context) -> Context:
    """Derive a context whose telemetry children use the supplied parent or active span."""
    return with_context_value(TELEMETRY_CONTEXT_KEY, telemetry_context, context)
