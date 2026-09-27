"""AbortController / AbortSignal mirroring the WHATWG DOM API used by pi."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import partial
from typing import TypeVar

__all__ = ["AbortError", "AbortController", "AbortSignal", "any_signal", "operation_signal", "race_with_abort_signal"]

T = TypeVar("T")
_UNSET = object()


class AbortError(Exception):
    name = "AbortError"


class AbortSignal:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: object = None
        self._listeners: dict[Callable[[], None], bool] = {}

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> object:
        return self._reason

    def _abort(self, reason: object = _UNSET) -> None:
        if not self._event.is_set():
            if reason is _UNSET:
                reason = AbortError("This operation was aborted")
            self._reason = reason
            self._event.set()
            for listener, once in list(self._listeners.items()):
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
                        logging.getLogger(__name__).exception("Exception in abort event listener")
                    else:
                        loop.call_exception_handler({"message": "Exception in abort event listener", "exception": error})

    def add_event_listener(self, event: str, listener: Callable[[], None], *, once: bool = False) -> None:
        if event == "abort" and listener not in self._listeners:
            self._listeners[listener] = once

    def remove_event_listener(self, event: str, listener: Callable[[], None]) -> None:
        if event == "abort":
            self._listeners.pop(listener, None)

    async def wait(self) -> None:
        """Await until the signal is aborted."""
        await self._event.wait()

    def throw_if_aborted(self) -> None:
        if self._event.is_set():
            raise self._reason if isinstance(self._reason, BaseException) else AbortError(str(self._reason))


class AbortController:
    def __init__(self) -> None:
        self._signal = AbortSignal()

    @property
    def signal(self) -> AbortSignal:
        return self._signal

    def abort(self, reason: object = _UNSET) -> None:
        self._signal._abort(reason)


def any_signal(*signals: AbortSignal | None) -> AbortSignal | None:
    """Return a signal that aborts when any of the given signals aborts."""
    live = [s for s in signals if s is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    combined = AbortSignal()

    def _on_abort(signal: AbortSignal) -> None:
        combined._abort(signal.reason)

    for live_signal in live:
        if live_signal.aborted:
            combined._abort(live_signal.reason)
            return combined
        live_signal.add_event_listener("abort", partial(_on_abort, live_signal), once=True)
    return combined


def operation_signal(signal: AbortSignal | None = None) -> AbortSignal:
    """Create an operation-local signal for public APIs with optional signals."""
    return signal if signal is not None else AbortController().signal


async def race_with_abort_signal(operation: Awaitable[T], signal: AbortSignal) -> T:
    """Stop awaiting on abort while observing the abandoned operation's result."""
    task = asyncio.ensure_future(operation)
    result: asyncio.Future[T] = asyncio.get_running_loop().create_future()

    def on_abort() -> None:
        if not result.done():
            reason = signal.reason
            result.set_exception(reason if isinstance(reason, BaseException) else AbortError(str(reason)))

    def on_done(completed: asyncio.Future[T]) -> None:
        try:
            value = completed.result()
        except BaseException as error:
            if not result.done():
                result.set_exception(error)
        else:
            if not result.done():
                result.set_result(value)

    task.add_done_callback(on_done)
    signal.add_event_listener("abort", on_abort, once=True)
    if signal.aborted:
        on_abort()
    try:
        return await result
    finally:
        signal.remove_event_listener("abort", on_abort)
