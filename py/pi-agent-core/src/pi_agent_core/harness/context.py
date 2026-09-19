"""Telemetry-aware context helpers ported from ``harness/context.ts``.

The chord context primitives already live in :mod:`.._chord.context` and are
re-exported here; this module adds the telemetry parent that spans started
through a context derive from.
"""

from __future__ import annotations

from .._chord.context import (
    BACKGROUND_CONTEXT,
    TODO_CONTEXT,
    Context,
    ContextKey,
    await_with_context,
    create_context_key,
    with_abort_signal,
    with_cancel,
    with_context_value,
    without_abort_signal,
)
from . import telemetry as _telemetry

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

TELEMETRY_CONTEXT_KEY = create_context_key("pi.telemetryContext")


def get_telemetry_context(context: Context) -> "_telemetry.TelemetryContext":
    """Return the telemetry parent attached to a context, or the shared no-op parent."""
    attached = context.value(TELEMETRY_CONTEXT_KEY)
    return attached if attached is not None else _telemetry.NOOP_TELEMETRY_CONTEXT


def with_telemetry_context(telemetry_context: "_telemetry.TelemetryContext", context: Context) -> Context:
    """Derive a context whose telemetry children use the supplied parent or active span."""
    return with_context_value(TELEMETRY_CONTEXT_KEY, telemetry_context, context)
