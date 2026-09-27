"""Established byte connection and state from ``connection.ts``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pi_chord.context import AbortController
from pi_chord.services.state_codec import ServiceStateEncoder
from pi_protocol import ClientMessageDecoder, RpcTarget

from .types import MaybePromise, RoutedServerServiceAttachment


class ByteConnection(Protocol):
    @property
    def closed(self) -> bool: ...

    def send(self, chunk: bytes) -> Awaitable[None]: ...

    def close(self, final_chunk: bytes | None = None) -> MaybePromise[None]: ...


class ByteConnectionHandler:
    def __init__(
        self, on_data: Callable[[bytes], None], on_close: Callable[[], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.on_data = on_data
        self.on_close = on_close
        self.on_error = on_error


type ConnectionStage = Literal["awaitingHello", "handshaking", "ready", "closing", "closed"]


@dataclass(eq=False)
class ConnectionState:
    connection: ByteConnection
    decoder: ClientMessageDecoder
    handshake_timeout: asyncio.TimerHandle
    service_state_encoders: dict[str, ServiceStateEncoder] = field(default_factory=dict)
    stage: ConnectionStage = "awaitingHello"
    disconnected: bool = False
    handshake: asyncio.Task[None] | None = None
    server_services: RoutedServerServiceAttachment | None = None
    active_requests: dict[str, tuple[AbortController, RpcTarget]] = field(default_factory=dict)


def is_terminal_connection(state: ConnectionState) -> bool:
    return state.disconnected or state.stage in ("closing", "closed")
