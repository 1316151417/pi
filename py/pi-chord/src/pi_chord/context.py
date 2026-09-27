"""Immutable context values and cancellation from ``chord/src/context``.

Cancellation rejects a waiter without cancelling the operation it observes.
The small abort runtime is standalone so Chord remains a dependency leaf.
Foreign signals implement ``AbortSignalLike``; no AI runtime is imported.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, NamedTuple, Protocol, TypeVar, cast
import sys
import weakref

from ._undefined import UNDEFINED, Undefined

__all__ = [
    "AbortError", "AbortSignalLike", "AbortSignal", "AbortController", "ContextKey", "Context",
    "BACKGROUND_CONTEXT", "TODO_CONTEXT", "create_context_key", "with_context_value",
    "with_abort_signal", "without_abort_signal", "with_cancel", "await_with_context",
    "CancelContext", "UNDEFINED", "Undefined",
]

T = TypeVar("T")


class AbortError(Exception):
    name = "AbortError"

    def __init__(self, message: str = "The operation was aborted") -> None:
        super().__init__(message)


class AbortSignalLike(Protocol):
    """The event-based signal contract shared with other Python runtimes."""

    @property
    def aborted(self) -> bool: ...

    @property
    def reason(self) -> object: ...

    def add_event_listener(
        self, event: str, callback: Callable[[], None], /, *, once: bool = False,
    ) -> None: ...

    def remove_event_listener(self, event: str, callback: Callable[[], None], /) -> None: ...

    def throw_if_aborted(self) -> None: ...


class AbortSignal:
    """An idempotent abort event carrying an arbitrary cancellation reason."""

    def __init__(self) -> None:
        self._aborted = False
        self._reason: object = UNDEFINED
        self._listeners: dict[Callable[[], None], bool] = {}
        self._sources: list[tuple[AbortSignalLike, Callable[[], None]]] = []

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> object:
        return self._reason

    def add_event_listener(self, event: str, callback: Callable[[], None], *, once: bool = False) -> None:
        if event == "abort" and callback not in self._listeners:
            self._listeners[callback] = once

    def remove_event_listener(self, event: str, callback: Callable[[], None]) -> None:
        if event == "abort":
            self._listeners.pop(callback, None)

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise _abort_error(self)

    def _abort(self, reason: object = UNDEFINED) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._reason = AbortError() if reason is UNDEFINED else reason
        for source, listener in self._sources:
            source.remove_event_listener("abort", listener)
        self._sources.clear()
        for listener, once in tuple(self._listeners.items()):
            if listener not in self._listeners:
                continue
            if once:
                self._listeners.pop(listener, None)
            try:
                listener()
            except BaseException as error:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    sys.excepthook(type(error), error, error.__traceback__)
                else:
                    loop.call_exception_handler({"message": "Exception in abort event listener", "exception": error})

    @classmethod
    def any(cls, signals: Iterable[AbortSignalLike]) -> AbortSignal:
        sources = tuple(signals)
        combined = cls()
        for signal in sources:
            if signal.aborted:
                combined._abort(signal.reason)
                return combined
        reference = weakref.ref(combined)
        for signal in sources:
            def propagate(source: AbortSignalLike = signal) -> None:
                target = reference()
                if target is not None:
                    target._abort(source.reason)

            combined._sources.append((signal, propagate))
            signal.add_event_listener("abort", propagate, once=True)
            if signal.aborted:
                propagate()
            if combined.aborted:
                signal.remove_event_listener("abort", propagate)
                break
        weakref.finalize(combined, _remove_source_listeners, combined._sources)
        return combined


def _remove_source_listeners(sources: list[tuple[AbortSignalLike, Callable[[], None]]]) -> None:
    for signal, listener in sources:
        signal.remove_event_listener("abort", listener)
    sources.clear()


class AbortController:
    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: object = UNDEFINED) -> None:
        self.signal._abort(reason)


@dataclass(frozen=True, eq=False)
class ContextKey(Generic[T]):
    description: str
    token: object = field(default_factory=object)


class Context:
    """A persistent value chain; keys compare their tokens by identity."""

    @property
    def abort_signal(self) -> AbortSignalLike | None:
        value = self.value(_ABORT_SIGNAL_CONTEXT_KEY)
        return None if value is UNDEFINED else cast(AbortSignalLike | None, value)

    def value(self, key: ContextKey[T]) -> T | Undefined:
        raise NotImplementedError

    def to_string(self) -> str:
        return str(self)


class _EmptyContext(Context):
    def __init__(self, name: str) -> None:
        self._name = name

    def value(self, key: ContextKey[T]) -> T | Undefined:
        return UNDEFINED

    def __str__(self) -> str:
        return self._name


class _ContextValue(Context, Generic[T]):
    def __init__(self, parent: Context, key: ContextKey[T], value: T) -> None:
        self._parent = parent
        self._key = key
        self._value = value

    def value(self, key: ContextKey[T]) -> T | Undefined:
        if key.token is self._key.token:
            return cast(T, self._value)
        return self._parent.value(key)

    def __str__(self) -> str:
        return f"{self._parent}.WithValue({self._key.description})"


_ABORT_SIGNAL_CONTEXT_KEY: ContextKey[AbortSignalLike | Undefined] = ContextKey("chord.abortSignal")
BACKGROUND_CONTEXT: Context = _EmptyContext("[Context BACKGROUND_CONTEXT]")
TODO_CONTEXT: Context = _EmptyContext("[Context TODO_CONTEXT]")


def create_context_key(description: str) -> ContextKey[T]:
    return ContextKey(description)


def with_context_value(key: ContextKey[T], value: T, parent: Context) -> Context:
    return _ContextValue(parent, key, value)


def with_abort_signal(signal: AbortSignalLike, context: Context) -> Context:
    parent = context.abort_signal
    combined = signal if parent is None else AbortSignal.any((parent, signal))
    return with_context_value(_ABORT_SIGNAL_CONTEXT_KEY, combined, context)


def without_abort_signal(context: Context) -> Context:
    return with_context_value(_ABORT_SIGNAL_CONTEXT_KEY, UNDEFINED, context)


class _Cancel(Protocol):
    def __call__(self, reason: object = UNDEFINED) -> None: ...


class CancelContext(NamedTuple):
    context: Context
    cancel: _Cancel


def with_cancel(context: Context) -> CancelContext:
    controller = AbortController()
    return CancelContext(with_abort_signal(controller.signal, context), controller.abort)


def _abort_error(signal: AbortSignalLike) -> BaseException:
    return signal.reason if isinstance(signal.reason, BaseException) else AbortError()


def await_with_context(promise: Awaitable[T], context: Context) -> Awaitable[T]:
    signal = context.abort_signal
    if signal is None:
        return promise
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[T] = loop.create_future()
    operation = asyncio.ensure_future(promise)

    def on_abort() -> None:
        signal.remove_event_listener("abort", on_abort)
        if not waiter.done():
            waiter.set_exception(_abort_error(signal))

    def settled(future: asyncio.Future[T]) -> None:
        signal.remove_event_listener("abort", on_abort)
        try:
            result = future.result()
        except BaseException as error:
            if not waiter.done():
                waiter.set_exception(error)
        else:
            if not waiter.done():
                waiter.set_result(result)

    # Python coroutines start here, whereas the source receives an already
    # running Promise. Observe its eventual exception even for a pre-aborted
    # context or a cancelled waiter, without cancelling that operation.
    operation.add_done_callback(settled)
    waiter.add_done_callback(lambda _waiter: signal.remove_event_listener("abort", on_abort))
    if signal.aborted:
        on_abort()
        return waiter
    signal.add_event_listener("abort", on_abort, once=True)
    if signal.aborted:
        on_abort()
    return waiter
