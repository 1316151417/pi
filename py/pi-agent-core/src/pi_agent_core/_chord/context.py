"""Minimal Python port of the ``@earendil-works/chord`` context primitives.

Chord models Go-style cancellation contexts: immutable value chains plus an
abort signal. Python coroutines run on one event loop, so ``await_with_context``
simply awaits; the context is still threaded through explicit parameters like
the TypeScript original.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from .._pi_ai.abort import AbortSignal

__all__ = [
    "ContextKey",
    "Context",
    "BACKGROUND_CONTEXT",
    "TODO_CONTEXT",
    "create_context_key",
    "with_context_value",
    "with_abort_signal",
    "without_abort_signal",
    "with_cancel",
    "await_with_context",
]

T = TypeVar("T")


@dataclass(frozen=True)
class ContextKey(Generic[T]):
    """Stable identity for one context value."""

    name: str


def create_context_key[T](name: str) -> ContextKey[T]:
    return ContextKey[T](name=name)


class Context:
    """Immutable chain of values plus an optional abort signal and cancel cause."""

    __slots__ = ("_values", "_signal", "_cancel_message")

    def __init__(
        self,
        values: Optional[Dict[ContextKey, Any]] = None,
        signal: Optional[AbortSignal] = None,
        cancel_message: Optional[str] = None,
    ) -> None:
        self._values: Dict[ContextKey, Any] = dict(values or {})
        self._signal = signal
        self._cancel_message = cancel_message

    def value(self, key: ContextKey[T]) -> Optional[T]:
        return self._values.get(key)

    @property
    def signal(self) -> Optional[AbortSignal]:
        return self._signal

    @property
    def cancel_message(self) -> Optional[str]:
        return self._cancel_message

    @property
    def cancelled(self) -> bool:
        return self._signal is not None and self._signal.aborted

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError(self._cancel_message or "context canceled")

    def _derive(
        self,
        values: Optional[Dict[ContextKey, Any]] = None,
        signal: Optional[AbortSignal] = None,
        keep_signal: bool = True,
        cancel_message: Optional[str] = None,
    ) -> "Context":
        merged = dict(self._values)
        if values:
            merged.update(values)
        return Context(
            values=merged,
            signal=self._signal if keep_signal and signal is None else signal,
            cancel_message=cancel_message if cancel_message is not None else self._cancel_message,
        )


#: Shared root context that is never canceled.
BACKGROUND_CONTEXT = Context()

#: Placeholder context for unimplemented paths (mirrors Go's context.TODO).
TODO_CONTEXT = Context()


def with_context_value(key: ContextKey[T], value: T, context: Context) -> Context:
    """Derive a context carrying ``key = value``."""
    return context._derive(values={key: value})


def with_abort_signal(signal: AbortSignal, context: Context) -> Context:
    """Derive a context whose cancellation follows ``signal``."""
    return context._derive(signal=signal, keep_signal=False)


def without_abort_signal(context: Context) -> Context:
    """Derive a context with cancellation removed."""
    return context._derive(signal=None, keep_signal=False, cancel_message=None)


def with_cancel(context: Context) -> tuple[Context, "Callable[[], None]"]:
    """Derive a context that can be canceled through the returned cancel callable."""
    from ..abort import AbortController

    controller = AbortController()
    derived = context._derive(signal=controller.signal, keep_signal=False)

    def cancel(message: Optional[str] = None) -> None:
        controller.abort(asyncio.CancelledError(message or "context canceled") if message else None)

    return derived, cancel


async def await_with_context(context: Context, awaitable: Any) -> Any:
    """Await ``awaitable`` bound to ``context``.

    Coroutines and futures are awaited directly; anything else is returned
    unchanged so synchronous hook results pass through.
    """
    if asyncio.iscoroutine(awaitable) or asyncio.isfuture(awaitable):
        return await awaitable
    return awaitable
