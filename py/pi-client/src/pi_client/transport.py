"""The transport-neutral byte contract from ``transport.ts``."""

from collections.abc import Awaitable, Callable
from typing import Protocol


class ByteTransport(Protocol):
    def send(self, chunk: bytes) -> Awaitable[None]: ...

    def close(self) -> None: ...


class ByteTransportHandlers:
    def __init__(
        self,
        on_data: Callable[[bytes], None],
        on_close: Callable[[], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.on_data = on_data
        self.on_close = on_close
        self.on_error = on_error


type ByteTransportFactory = Callable[[ByteTransportHandlers], ByteTransport | Awaitable[ByteTransport]]
