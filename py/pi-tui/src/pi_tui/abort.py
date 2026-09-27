"""Standalone DOM abort adapter used by cancellable terminal components."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

_UNSET = object()


class AbortError(Exception):
    name = "AbortError"


class AbortSignal:
    def __init__(self) -> None:
        self._aborted = False
        self._reason: object = None
        self._listeners: list[tuple[Callable[[], None], bool]] = []

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> object:
        return self._reason

    def add_event_listener(self, event: str, callback: Callable[[], None], *, once: bool = False) -> None:
        if event == "abort" and all(listener is not callback for listener, _ in self._listeners):
            self._listeners.append((callback, once))

    def remove_event_listener(self, event: str, callback: Callable[[], None]) -> None:
        if event != "abort":
            return
        for index, (listener, _) in enumerate(self._listeners):
            if listener is callback:
                del self._listeners[index]
                return

    def throw_if_aborted(self) -> None:
        if self._aborted:
            raise self._reason if isinstance(self._reason, BaseException) else AbortError(str(self._reason))

    async def wait(self) -> None:
        if self._aborted:
            return
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()

        def resolve() -> None:
            if not waiter.done():
                waiter.set_result(None)

        def on_abort() -> None:
            loop.call_soon_threadsafe(resolve)

        self.add_event_listener("abort", on_abort, once=True)
        if self._aborted:
            on_abort()
        try:
            await waiter
        finally:
            self.remove_event_listener("abort", on_abort)

    def _abort(self, reason: object = _UNSET) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._reason = AbortError("This operation was aborted") if reason is _UNSET else reason
        for callback, once in list(self._listeners):
            if all(listener is not callback for listener, _ in self._listeners):
                continue
            if once:
                self.remove_event_listener("abort", callback)
            try:
                callback()
            except BaseException as error:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    sys.excepthook(type(error), error, error.__traceback__)
                else:
                    loop.call_exception_handler({"message": "Exception in abort event listener", "exception": error})


class AbortController:
    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: object = _UNSET) -> None:
        self.signal._abort(reason)
