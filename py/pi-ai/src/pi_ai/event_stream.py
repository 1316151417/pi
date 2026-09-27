"""FIFO event streams from pi-ai ``src/utils/event-stream.ts``.

``result`` remains a stable ``asyncio.Future`` property: Python callers use
``await stream.result``. Futures and pending iterator pulls belong to their
asyncio loop. Cancelling a Python iterator pull closes that iterator and removes
its pending waiter. The result Future retains ordinary asyncio cancellation;
cancellation has no counterpart in JavaScript promises.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar, cast

from .types import AssistantMessage, AssistantMessageEvent

__all__ = ["EventStream", "AssistantMessageEventStream", "create_assistant_message_event_stream"]

T = TypeVar("T")
R = TypeVar("R")


class _Unset:
    __slots__ = ()


class _End:
    __slots__ = ()


_UNSET = _Unset()
_END = _End()


class EventStream(Generic[T, R]):
    """A shared event queue with FIFO delivery to waiting consumers.

    A completing push resolves the final result and delivers the event to one
    consumer. As in the source, other already-waiting consumers are released by
    ``end()``, not by that push. Buffered events remain readable after either
    form of completion. ``end()`` leaves the final result pending, whereas
    ``end(None)`` resolves it to ``None``.
    """

    def __init__(self, is_complete: Callable[[T], bool], extract_result: Callable[[T], R]) -> None:
        self._items: deque[T] = deque()
        self._waiting: deque[asyncio.Future[tuple[T] | _End]] = deque()
        self._done = False
        self._final_result: asyncio.Future[R] | None = None
        self._result_value: R | _Unset = _UNSET
        self._is_complete = is_complete
        self._extract_result = extract_result

    def _resolve_final_result(self, result: R) -> None:
        if self._result_value is _UNSET:
            self._result_value = result
            if self._final_result is not None and not self._final_result.done():
                self._final_result.set_result(result)

    def push(self, event: T) -> None:
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            self._resolve_final_result(self._extract_result(event))

        while self._waiting:
            waiter = self._waiting.popleft()
            if not waiter.cancelled():
                waiter.set_result((event,))
                return
        self._items.append(event)

    def end(self, result: R | _Unset = _UNSET) -> None:
        self._done = True
        if result is not _UNSET:
            self._resolve_final_result(cast(R, result))
        while self._waiting:
            waiter = self._waiting.popleft()
            if not waiter.cancelled():
                waiter.set_result(_END)

    def __aiter__(self) -> AsyncIterator[T]:
        return _EventStreamIterator(self)

    @property
    def result(self) -> asyncio.Future[R]:
        if self._final_result is None:
            self._final_result = asyncio.get_running_loop().create_future()
            if self._result_value is not _UNSET:
                self._final_result.set_result(cast(R, self._result_value))
        return self._final_result


class _EventStreamIterator(AsyncIterator[T], Generic[T, R]):
    def __init__(self, stream: EventStream[T, R]) -> None:
        self._stream = stream
        self._finished = False
        # JavaScript async generators queue overlapping next() calls. Serializing
        # each iterator preserves that behavior while consumers share the stream.
        self._next_lock = asyncio.Lock()

    def __aiter__(self) -> _EventStreamIterator[T, R]:
        return self

    async def __anext__(self) -> T:
        async with self._next_lock:
            if self._finished:
                raise StopAsyncIteration
            stream = self._stream
            if stream._items:
                return stream._items.popleft()
            if stream._done:
                self._finished = True
                raise StopAsyncIteration

            waiter: asyncio.Future[tuple[T] | _End] = asyncio.get_running_loop().create_future()
            stream._waiting.append(waiter)
            try:
                result = await waiter
            except BaseException:
                self._finished = True
                raise
            finally:
                # A cancelled pull must not absorb a later producer event.
                if waiter.cancelled():
                    try:
                        stream._waiting.remove(waiter)
                    except ValueError:
                        pass
            if result is _END:
                self._finished = True
                raise StopAsyncIteration
            return cast(tuple[T], result)[0]

    async def aclose(self) -> None:
        async with self._next_lock:
            self._finished = True


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self) -> None:
        super().__init__(
            lambda event: event.type in ("done", "error"),
            _extract_final_message,
        )


def _extract_final_message(event: AssistantMessageEvent) -> AssistantMessage:
    if event.type == "done":
        return cast(AssistantMessage, event.message)
    if event.type == "error":
        return cast(AssistantMessage, event.error)
    raise ValueError("Unexpected event type for final result")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    return AssistantMessageEventStream()
