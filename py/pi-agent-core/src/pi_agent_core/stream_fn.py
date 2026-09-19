"""Default stream function configuration ported from ``src/stream-fn.ts``."""

from __future__ import annotations

from typing import Optional

from .types import StreamFn

__all__ = ["set_default_stream_fn", "get_default_stream_fn"]

_default_stream_fn: Optional[StreamFn] = None


def set_default_stream_fn(stream_fn: Optional[StreamFn]) -> None:
    """Configure the fallback used by Agent and low-level loops when callers omit stream_fn."""
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn
