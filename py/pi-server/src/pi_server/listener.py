"""Authorized byte listener contract from ``listener.ts``."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from .connection import ByteConnection, ByteConnectionHandler

type ByteConnectionAcceptor = Callable[[ByteConnection], ByteConnectionHandler]


class ServerListener(Protocol):
    def start(self, accept: ByteConnectionAcceptor) -> Awaitable[None]: ...

    def close(self) -> Awaitable[None]: ...
