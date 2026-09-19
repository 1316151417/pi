"""Async event stream ported from pi-ai ``src/utils/event-stream.ts``.

Python note: ``result`` is a property returning an awaitable ``asyncio.Future``,
so consumers write ``await stream.result`` where TypeScript writes
``await stream.result()``.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Generic, List, Optional, TypeVar

from .types import AssistantMessage, AssistantMessageEvent

__all__ = ["EventStream", "AssistantMessageEventStream", "create_assistant_message_event_stream"]

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    """Generic push/end async iterable with a final-result awaitable.

    Mirrors the TypeScript semantics: pushing after completion drops the event;
    consumers iterating the stream terminate after the completing event or a
    call to :meth:`end` even when no further events arrive.
    """

    def __init__(self, is_complete: Callable[[T], bool], extract_result: Callable[[T], R]) -> None:
        self._items: List[T] = []
        self._wakeup = asyncio.Event()
        self._done = False
        self._final_result: "Optional[asyncio.Future[R]]" = None
        self._is_complete = is_complete
        self._extract_result = extract_result

    def _get_final_result(self) -> "asyncio.Future[R]":
        if self._final_result is None:
            self._final_result = asyncio.get_running_loop().create_future()
        return self._final_result

    def push(self, event: T) -> None:
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            future = self._get_final_result()
            if not future.done():
                future.set_result(self._extract_result(event))
        self._items.append(event)
        self._wakeup.set()

    def end(self, result: Optional[R] = None) -> None:
        self._done = True
        if result is not None:
            future = self._get_final_result()
            if not future.done():
                future.set_result(result)
        self._wakeup.set()

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            while self._items:
                yield self._items.pop(0)
            if self._done:
                return
            await self._wakeup.wait()
            self._wakeup.clear()

    @property
    def result(self) -> "asyncio.Future[R]":
        return self._get_final_result()


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self) -> None:
        super().__init__(
            lambda event: event.type in ("done", "error"),
            _extract_final_message,
        )


def _extract_final_message(event: AssistantMessageEvent) -> AssistantMessage:
    if event.type == "done":
        assert event.message is not None
        return event.message
    if event.type == "error":
        assert event.error is not None
        return event.error
    raise ValueError("Unexpected event type for final result")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    return AssistantMessageEventStream()
