"""Physical connection handshake and framed message delivery from ``connection.ts``."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from pi_protocol import (
    DEFAULT_MAX_FRAME_LENGTH, PROTOCOL_VERSION, FrameDecoderOptions, ProtocolValidationError,
    ServerHello, ServerMessage, ServerMessageDecoder, encode_client_message,
)

from .errors import DisconnectedError, ServerError, to_disconnected_error, to_error
from .promise import PromiseResolvers, create_promise_resolvers
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers
from .types import ConnectionState, ConnectionStateChange

MAX_UINT32 = 0xFFFFFFFF


@dataclass
class ConnectionOptions:
    transport_factory: ByteTransportFactory
    server_id: str
    on_handshake: Callable[[ServerHello], None]
    on_message: Callable[[ServerMessage], None]
    on_state_change: Callable[[ConnectionStateChange], None]
    max_frame_length: int | float | None = None


@dataclass
class _Lifecycle:
    state: ConnectionState
    id: int = 0
    decoder: ServerMessageDecoder | None = None
    transport: ByteTransport | None = None
    handshake: PromiseResolvers[ServerHello] | None = None


class Connection:
    def __init__(self, options: ConnectionOptions) -> None:
        self._options = options
        maximum = options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        if (type(maximum) not in (int, float) or not math.isfinite(maximum)
                or maximum != int(maximum) or not 0 < maximum <= MAX_UINT32):
            raise TypeError(f"Client maxFrameLength must be an integer between 1 and {MAX_UINT32}")
        self._max_frame_length = int(maximum)
        self._lifecycle = _Lifecycle("disconnected")
        self._sequence = 0

    @property
    def state(self) -> ConnectionState:
        return self._lifecycle.state

    @property
    def max_frame_length(self) -> int:
        return self._max_frame_length

    def connect(self) -> asyncio.Future[ServerHello]:
        if self._lifecycle.state != "disconnected":
            future: asyncio.Future[ServerHello] = asyncio.get_running_loop().create_future()
            future.set_exception(DisconnectedError(f"Client is already {self._lifecycle.state}"))
            return future
        self._sequence += 1
        handshake: PromiseResolvers[ServerHello] = create_promise_resolvers()
        connection_id = self._sequence
        self._lifecycle = _Lifecycle(
            "connecting", connection_id,
            ServerMessageDecoder(FrameDecoderOptions(max_frame_length=self._max_frame_length)),
            handshake=handshake,
        )
        self._options.on_state_change(ConnectionStateChange("connecting"))
        handlers = ByteTransportHandlers(
            lambda chunk: self._handle_data(connection_id, chunk),
            lambda: self._handle_close() if self._is_current(connection_id) else None,
            lambda error: self._fail_and_close(to_disconnected_error(error)) if self._is_current(connection_id) else None,
        )
        asyncio.get_running_loop().create_task(self._open_transport(connection_id, handlers))
        return handshake.promise

    def disconnect(self, reason: str | Exception = "Client disconnected") -> None:
        if self.state != "disconnected":
            self._fail_and_close(DisconnectedError(reason) if isinstance(reason, str) else reason)

    def fail(self, error: Exception) -> None:
        self._fail_and_close(error)

    def send(self, frame: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state != "connected" or lifecycle.transport is None:
            raise DisconnectedError()
        try:
            sending = lifecycle.transport.send(frame)
        except Exception as error:
            self._fail_and_close(to_disconnected_error(error))
            return

        async def observe() -> None:
            try:
                await sending
            except Exception as error:
                current = self._lifecycle
                if current.state != "disconnected" and current.transport is lifecycle.transport:
                    self._fail_and_close(to_disconnected_error(error))

        asyncio.get_running_loop().create_task(observe())

    async def _open_transport(self, connection_id: int, handlers: ByteTransportHandlers) -> None:
        try:
            created = self._options.transport_factory(handlers)
            transport = await created if inspect.isawaitable(created) else created
        except Exception as error:
            if self._is_current(connection_id):
                self._fail(to_disconnected_error(error))
            return
        lifecycle = self._lifecycle
        if lifecycle.state != "connecting" or lifecycle.id != connection_id:
            transport.close()
            return
        lifecycle.transport = transport
        try:
            await transport.send(encode_client_message(
                {"type": "hello", "version": PROTOCOL_VERSION},
                FrameDecoderOptions(max_frame_length=self._max_frame_length),
            ))
        except Exception as error:
            if self._is_current(connection_id):
                self._fail_and_close(to_disconnected_error(error))

    def _handle_data(self, connection_id: int, chunk: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected" or lifecycle.id != connection_id:
            return
        if lifecycle.state == "connecting" and lifecycle.transport is None:
            self._fail_and_close(ProtocolValidationError("Received server data before the client hello was sent"))
            return
        try:
            messages = cast(ServerMessageDecoder, lifecycle.decoder).push(chunk)
        except Exception as error:
            self._fail_and_close(to_error(error))
            return
        for message in messages:
            if self.state == "disconnected":
                return
            self._handle_message(message)

    def _handle_message(self, message: ServerMessage) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "connecting":
            if message["type"] == "hello_error":
                self._fail_and_close(ServerError(message["error"]))
                return
            if message["type"] != "hello":
                self._fail_and_close(ProtocolValidationError("Expected server hello as first message"))
                return
            if message["serverId"] != self._options.server_id:
                self._fail_and_close(ProtocolValidationError(
                    f"Connected server {json.dumps(message['serverId'], ensure_ascii=False)} does not match "
                    f"{json.dumps(self._options.server_id, ensure_ascii=False)}"
                ))
                return
            if lifecycle.transport is None:
                self._fail_and_close(ProtocolValidationError("Received server hello before the client hello was sent"))
                return
            connected = _Lifecycle(
                "connected", lifecycle.id, lifecycle.decoder, lifecycle.transport, lifecycle.handshake,
            )
            self._lifecycle = connected
            try:
                self._options.on_handshake(message)
            except Exception as error:
                if self._lifecycle is connected:
                    self._fail_and_close(to_error(error))
                return
            if self._lifecycle is not connected:
                return
            self._options.on_state_change(ConnectionStateChange("connected"))
            if self._lifecycle is not connected:
                return
            self._lifecycle = _Lifecycle("connected", connected.id, connected.decoder, connected.transport)
            cast(PromiseResolvers[ServerHello], lifecycle.handshake).resolve(message)
            return
        if lifecycle.state != "connected":
            return
        if message["type"] in ("hello", "hello_error"):
            self._fail_and_close(ProtocolValidationError("Unexpected handshake message"))
            return
        self._options.on_message(message)

    def _handle_close(self) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        error: Exception = DisconnectedError("Byte transport closed")
        try:
            cast(ServerMessageDecoder, lifecycle.decoder).end()
        except Exception as decoder_error:
            error = to_error(decoder_error)
        self._fail(error)

    def _fail_and_close(self, error: Exception) -> None:
        transport = self._lifecycle.transport
        self._fail(error)
        if transport is not None:
            transport.close()

    def _fail(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        self._lifecycle = _Lifecycle("disconnected")
        if lifecycle.handshake is not None:
            lifecycle.handshake.reject(error)
        self._options.on_state_change(ConnectionStateChange("disconnected", error))

    def _is_current(self, connection_id: int) -> bool:
        return self.state != "disconnected" and self._lifecycle.id == connection_id
