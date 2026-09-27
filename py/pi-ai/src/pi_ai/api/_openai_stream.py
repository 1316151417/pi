"""Native port of OpenAI 6.40.0 ``core/streaming`` and its line decoder.

Readable streams are Python async byte iterables. Their ``aclose`` method is
the counterpart of a JavaScript iterator's ``return``/ReadableStream.cancel.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from ..abort import AbortController
from ..types import UNDEFINED, Undefined
from ..utils._javascript import javascript_json_parse, javascript_json_stringify
from ._openai_errors import APIError, OpenAIError, is_abort_error, truthy

type ByteChunk = bytes | bytearray | memoryview | str | None | Undefined


def encode_utf8(value: str) -> bytes:
    return value.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace").encode("utf-8")


class StreamingResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    @property
    def body(self) -> AsyncIterable[ByteChunk] | None: ...

    async def cancel_body(self) -> None: ...


class LineDecoder:
    NEWLINE_CHARS = frozenset(("\n", "\r"))

    def __init__(self) -> None:
        self._buffer = b""
        self._carriage_return_index: int | None = None

    def decode(self, chunk: ByteChunk) -> list[str]:
        if chunk is None or chunk is UNDEFINED:
            return []
        self._buffer += encode_utf8(chunk) if isinstance(chunk, str) else bytes(chunk)
        lines: list[str] = []
        while True:
            found: tuple[int, int, bool] | None = None
            for index in range(self._carriage_return_index or 0, len(self._buffer)):
                if self._buffer[index] in (0x0A, 0x0D):
                    found = (index, index + 1, self._buffer[index] == 0x0D)
                    break
            if found is None:
                break
            preceding, after, carriage = found
            if carriage and self._carriage_return_index is None:
                self._carriage_return_index = after
                continue
            if self._carriage_return_index is not None and (after != self._carriage_return_index + 1 or carriage):
                lines.append(self._buffer[: self._carriage_return_index - 1].decode("utf-8-sig", errors="replace"))
                self._buffer = self._buffer[self._carriage_return_index :]
                self._carriage_return_index = None
                continue
            end = preceding - 1 if self._carriage_return_index is not None else preceding
            lines.append(self._buffer[:end].decode("utf-8-sig", errors="replace"))
            self._buffer = self._buffer[after:]
            self._carriage_return_index = None
        return lines

    def flush(self) -> list[str]:
        return self.decode("\n") if self._buffer else []


def find_double_newline_index(buffer: bytes | bytearray | memoryview) -> int:
    for index in range(len(buffer) - 1):
        if buffer[index] == 0x0A and buffer[index + 1] == 0x0A:
            return index + 2
        if buffer[index] == 0x0D and buffer[index + 1] == 0x0D:
            return index + 2
        if buffer[index] == 0x0D and buffer[index + 1] == 0x0A and index + 3 < len(buffer) and buffer[index + 2] == 0x0D and buffer[index + 3] == 0x0A:
            return index + 4
    return -1


@dataclass
class ServerSentEvent:
    event: str | None
    data: str
    raw: list[str]


class SSEDecoder:
    def __init__(self) -> None:
        self._event: str | None = None
        self._data: list[str] = []
        self._chunks: list[str] = []

    def decode(self, line: str) -> ServerSentEvent | None:
        if line.endswith("\r"):
            line = line[:-1]
        if not line:
            if not self._event and not self._data:
                return None
            event = ServerSentEvent(self._event, "\n".join(self._data), self._chunks)
            self._event = None
            self._data = []
            self._chunks = []
            return event
        self._chunks.append(line)
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None


async def _close_iterator(iterator: object, *, preserve_error: BaseException | None = None) -> None:
    try:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
    except BaseException:
        # AsyncIteratorClose preserves an existing throw, but a failed return
        # while breaking/returning is itself observable.
        if preserve_error is None or isinstance(preserve_error, GeneratorExit):
            raise


async def iter_sse_chunks(source: AsyncIterable[ByteChunk]) -> AsyncIterator[bytes]:
    buffered = b""
    iterator = aiter(source)
    exhausted = False
    try:
        async for chunk in iterator:
            if chunk is None or chunk is UNDEFINED:
                continue
            buffered += encode_utf8(chunk) if isinstance(chunk, str) else bytes(chunk)
            while (index := find_double_newline_index(buffered)) != -1:
                yield buffered[:index]
                buffered = buffered[index:]
        exhausted = True
        if buffered:
            yield buffered
    finally:
        if not exhausted:
            await _close_iterator(iterator, preserve_error=sys.exception())


async def iter_sse_messages(response: StreamingResponse, controller: AbortController) -> AsyncIterator[ServerSentEvent]:
    if response.body is None:
        controller.abort()
        raise OpenAIError("Attempted to iterate over a response with no body")
    sse_decoder, line_decoder = SSEDecoder(), LineDecoder()
    chunks = iter_sse_chunks(response.body)
    try:
        async for chunk in chunks:
            for line in line_decoder.decode(chunk):
                event = sse_decoder.decode(line)
                if event is not None:
                    yield event
        for line in line_decoder.flush():
            event = sse_decoder.decode(line)
            if event is not None:
                yield event
    finally:
        await _close_iterator(chunks, preserve_error=sys.exception())


@dataclass
class _IteratorResult[T]:
    done: bool
    value: T | None = None


class _TeeIterator[T]:
    def __init__(self, queue: deque[asyncio.Task[_IteratorResult[T]]], pull: Callable[[], asyncio.Task[_IteratorResult[T]]]) -> None:
        self._queue = queue
        self._pull = pull

    def __aiter__(self) -> _TeeIterator[T]:
        return self

    async def __anext__(self) -> T:
        if not self._queue:
            self._pull()
        result = await asyncio.shield(self._queue.popleft())
        if result.done:
            raise StopAsyncIteration
        return cast(T, result.value)


class _ReadableStream[T]:
    def __init__(self, iterator: AsyncIterator[T]) -> None:
        self._iterator = iterator

    def __aiter__(self) -> _ReadableStream[T]:
        return self

    async def __anext__(self) -> bytes:
        value = await anext(self._iterator)
        encoded = javascript_json_stringify(value)
        return encode_utf8((encoded if encoded is not None else "undefined") + "\n")

    async def aclose(self) -> None:
        await _close_iterator(self._iterator)


class _StreamIterator[T]:
    """Close an abandoned Python iterator as JS for-await's return would.

    Callers keeping an explicit iterator reference should await ``aclose`` when
    leaving early; Python for-await has no automatic iterator-return protocol.
    """

    def __init__(self, iterator: AsyncIterator[T], controller: AbortController) -> None:
        self._iterator = iterator
        self._controller = controller
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._done = False

    def __aiter__(self) -> _StreamIterator[T]:
        return self

    async def __anext__(self) -> T:
        self._started = True
        self._loop = asyncio.get_running_loop()
        try:
            return await anext(self._iterator)
        except BaseException:
            self._done = True
            raise

    async def aclose(self) -> None:
        try:
            await _close_iterator(self._iterator)
        finally:
            self._done = True

    def __del__(self) -> None:
        if self._started and not self._done and self._loop is not None and not self._loop.is_closed():
            self._controller.abort()
            self._loop.create_task(_close_iterator(self._iterator))


class OpenAIStream[T]:
    def __init__(self, iterator: Callable[[], AsyncIterator[T]], controller: AbortController) -> None:
        self._iterator = iterator
        self.controller = controller

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterator()

    @staticmethod
    def from_sse_response(
        response: StreamingResponse, controller: AbortController, *,
        synthesize_event_data: bool = False, on_close: Callable[[], None] | None = None,
        log_errors: bool = True,
    ) -> OpenAIStream[object]:
        consumed = False

        async def iterator() -> AsyncIterator[object]:
            nonlocal consumed
            if consumed:
                raise OpenAIError("Cannot iterate over a consumed stream, use `.tee()` to split the stream.")
            consumed = True
            done = False
            messages = iter_sse_messages(response, controller)
            try:
                async for sse in messages:
                    if done:
                        continue
                    if sse.data.startswith("[DONE]"):
                        done = True
                        continue
                    try:
                        data = javascript_json_parse(sse.data)
                    except (Exception, asyncio.CancelledError):
                        if log_errors or sse.event is not None and sse.event.startswith("thread."):
                            logging.getLogger(__name__).error("Could not parse message into JSON: %s", sse.data)
                            logging.getLogger(__name__).error("From chunk: %s", sse.raw)
                        raise
                    if sse.event is None or not sse.event.startswith("thread."):
                        error = data.get("error") if isinstance(data, Mapping) else None
                        if truthy(data) and truthy(error):
                            raise APIError(error=error, headers=response.headers)
                        yield {"event": sse.event, "data": data} if synthesize_event_data else data
                    else:
                        # The pinned source retains this branch even though an
                        # event named "error" does not begin with "thread.".
                        if sse.event == "error":
                            raise APIError(
                                error=data.get("error") if isinstance(data, Mapping) else None,
                                message=cast(str | None, data.get("message")) if isinstance(data, Mapping) else None,
                            )
                        yield {"event": sse.event, "data": data}
                done = True
            except (Exception, asyncio.CancelledError) as error:
                if is_abort_error(error):
                    return
                raise
            finally:
                try:
                    await _close_iterator(messages, preserve_error=sys.exception())
                finally:
                    if not done:
                        controller.abort()
                    if on_close is not None:
                        on_close()

        return OpenAIStream(lambda: _StreamIterator(iterator(), controller), controller)

    @staticmethod
    def from_readable_stream(source: AsyncIterable[ByteChunk], controller: AbortController) -> OpenAIStream[object]:
        consumed = False

        async def iterator() -> AsyncIterator[object]:
            nonlocal consumed
            if consumed:
                raise OpenAIError("Cannot iterate over a consumed stream, use `.tee()` to split the stream.")
            consumed = True
            done = False
            decoder = LineDecoder()
            chunks = aiter(source)
            exhausted = False
            try:
                async for chunk in chunks:
                    for line in decoder.decode(chunk):
                        if line:
                            yield javascript_json_parse(line)
                exhausted = True
                for line in decoder.flush():
                    if line:
                        yield javascript_json_parse(line)
                done = True
            except (Exception, asyncio.CancelledError) as error:
                if is_abort_error(error):
                    return
                raise
            finally:
                try:
                    if not exhausted:
                        await _close_iterator(chunks, preserve_error=sys.exception())
                finally:
                    if not done:
                        controller.abort()

        return OpenAIStream(lambda: _StreamIterator(iterator(), controller), controller)

    def tee(self) -> tuple[OpenAIStream[T], OpenAIStream[T]]:
        left: deque[asyncio.Task[_IteratorResult[T]]] = deque()
        right: deque[asyncio.Task[_IteratorResult[T]]] = deque()
        iterator = self._iterator()
        lock = asyncio.Lock()

        async def advance() -> _IteratorResult[T]:
            async with lock:
                try:
                    return _IteratorResult(done=False, value=await anext(iterator))
                except StopAsyncIteration:
                    return _IteratorResult(done=True)

        def pull() -> asyncio.Task[_IteratorResult[T]]:
            result = asyncio.create_task(advance())
            left.append(result)
            right.append(result)
            return result

        return (
            OpenAIStream(lambda: _TeeIterator(left, pull), self.controller),
            OpenAIStream(lambda: _TeeIterator(right, pull), self.controller),
        )

    def to_readable_stream(self) -> _ReadableStream[T]:
        return _ReadableStream(self._iterator())


__all__ = [
    "ByteChunk", "StreamingResponse", "LineDecoder", "ServerSentEvent", "SSEDecoder",
    "find_double_newline_index", "iter_sse_chunks", "iter_sse_messages", "OpenAIStream",
]
