"""Transport-neutral remote Session server from ``server.ts``."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable
from typing import cast

from pi_chord.context import AbortController, AbortError, BACKGROUND_CONTEXT, Context, TODO_CONTEXT, with_abort_signal
from pi_chord.delta import UNDEFINED, Undefined
from pi_chord.services.errors import RemoteServiceError
from pi_chord.services.state_codec import create_service_state_encoder
from pi_chord.services.wire import (
    decode_service_control_call, parse_service_call, parse_service_subscription_snapshot,
)
from pi_chord.types import JsonValue, ServiceProviderUpdate
from pi_protocol import (
    ClientMessage, ClientMessageDecoder, DEFAULT_MAX_FRAME_LENGTH, FrameDecoderOptions,
    PROTOCOL_VERSION, ProtocolError, ProtocolValidationError, RequestEnvelope,
    RpcTarget, ServerMessage, encode_server_message, is_server_id,
    is_supported_protocol_version,
)

from .connection import ByteConnection, ByteConnectionHandler, ConnectionState, is_terminal_connection
from .errors import INTERNAL_SERVER_ERROR_MESSAGE, ServerError, WrongServerError
from .session_router import SessionRouter, SessionRouterOptions
from .types import ServerHost, ServerOptions, SessionMetadata

DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000
MAX_UINT32 = 0xFFFFFFFF
MAX_TIMER_DELAY_MS = 2_147_483_647


def _safe_integer(value: object, maximum: int) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and int(value) == value and 0 < value <= maximum)


def _same_target(left: RpcTarget, right: RpcTarget) -> bool:
    if left["serverId"] != right["serverId"]:
        return False
    if "sessionId" not in left or "sessionId" not in right:
        return "sessionId" not in left and "sessionId" not in right
    return (left["sessionId"] == right["sessionId"]
            and left["attachmentId"] == right["attachmentId"])


async def _await_value[T](value: T | Awaitable[T]) -> T:
    return await value if inspect.isawaitable(value) else value


class _Presentation[TMetadata: SessionMetadata]:
    def __init__(self, sessions: SessionRouter[TMetadata], state: ConnectionState) -> None:
        self._sessions = sessions
        self._state = state

    async def attach_session(self, session_id: str, context: Context) -> None:
        await self._sessions.attach_client(self._state, session_id, context)

    async def detach_session(self, context: Context) -> None:
        await self._sessions.detach_client(self._state, context)

    async def prepare_session_removal(self, session_id: str, context: Context) -> None:
        await self._sessions.remove_session(session_id, context)


class Server[TMetadata: SessionMetadata]:
    def __init__(self, host: ServerHost[TMetadata], options: ServerOptions) -> None:
        if not isinstance(options.listeners, (list, tuple)):
            raise TypeError("Server listeners must be an array")
        if not is_server_id(options.server_id):
            raise TypeError("serverId must be a canonical lowercase UUIDv4")
        max_frame_length = options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        timeout_ms = options.handshake_timeout_ms if options.handshake_timeout_ms is not None else DEFAULT_HANDSHAKE_TIMEOUT_MS
        if not _safe_integer(max_frame_length, MAX_UINT32):
            raise TypeError(f"Server maxFrameLength must be an integer between 1 and {MAX_UINT32}")
        if not _safe_integer(timeout_ms, MAX_TIMER_DELAY_MS):
            raise TypeError(f"Server handshakeTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")
        self.server_id = options.server_id
        self._host = host
        self._listeners = options.listeners
        self._max_frame_length = int(max_frame_length)
        self._handshake_timeout_ms = int(timeout_ms)
        self._on_connection_count_changed = options.on_connection_count_changed
        self._on_error = options.on_error
        self._connections: set[ConnectionState] = set()
        self._closing = False
        self._started = False
        self._start_promise: asyncio.Task[Server[TMetadata]] | None = None
        self._close_promise: asyncio.Task[None] | None = None
        self._closed: asyncio.Future[None] | None = None
        self._sessions = SessionRouter(SessionRouterOptions(
            host=host,
            server_id=self.server_id,
            is_closing=lambda: self._closing,
            publish_attachment=self._publish_attachment,
            report_error=self._report_error,
        ))

    @property
    def closed(self) -> asyncio.Future[None]:
        if self._closed is None:
            self._closed = asyncio.get_running_loop().create_future()
            self._closed.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        return self._closed

    def start(self) -> asyncio.Task[Server[TMetadata]] | asyncio.Future[Server[TMetadata]]:
        loop = asyncio.get_running_loop()
        if self._started or self._start_promise is not None or self._closing:
            result: asyncio.Future[Server[TMetadata]] = loop.create_future()
            result.set_exception(RuntimeError(
                "Server is already started" if self._started else
                "Server is already starting" if self._start_promise is not None else
                "Server is closing or closed"
            ))
            return result
        self._start_promise = loop.create_task(self._start_internal())
        return self._start_promise

    async def _start_internal(self) -> Server[TMetadata]:
        started = []
        try:
            for listener in self._listeners:
                await listener.start(self.accept)
                started.append(listener)
            self._started = True
            return self
        except Exception as error:
            self._closing = True
            failures = [value for value in await asyncio.gather(
                *(listener.close() for listener in started), return_exceptions=True,
            ) if isinstance(value, Exception)]
            try:
                await self._close_server_state()
            except Exception as cleanup_error:
                failures.append(cleanup_error)
            if failures:
                failure = ExceptionGroup("Server startup and cleanup failed", [error, *failures])
                self._settle_closed(failure)
                raise failure
            self._settle_closed()
            raise
        finally:
            self._start_promise = None

    def accept(self, connection: ByteConnection) -> ByteConnectionHandler:
        if self._closing:
            asyncio.get_running_loop().create_task(self._close_connection(connection))
            return ByteConnectionHandler(lambda _chunk: None, lambda: None, self._report_error)
        loop = asyncio.get_running_loop()
        state_ref: list[ConnectionState] = []

        def timeout() -> None:
            loop.create_task(self._fail_protocol(state_ref[0], {
                "code": "invalid_request", "message": "Handshake timeout",
            }))

        timer = loop.call_later(self._handshake_timeout_ms / 1000, timeout)
        state = ConnectionState(connection, ClientMessageDecoder(
            FrameDecoderOptions(max_frame_length=self._max_frame_length)), timer)
        state_ref.append(state)
        self._connections.add(state)
        self._notify_connection_count_changed()

        def on_error(error: Exception) -> None:
            self._report_error(error)

            async def close() -> None:
                await self._close_connection(connection)
                self._disconnect(state)
            loop.create_task(close())

        return ByteConnectionHandler(
            lambda chunk: self._receive(state, chunk),
            lambda: self._transport_closed(state), on_error,
        )

    def close(self) -> asyncio.Task[None]:
        if self._close_promise is None:
            self._closing = True
            self._close_promise = asyncio.get_running_loop().create_task(self._close_internal())
        return self._close_promise

    async def _close_internal(self) -> None:
        starting = self._start_promise
        if starting is not None:
            try:
                await starting
            except Exception:
                pass
        errors = [value for value in await asyncio.gather(
            *(listener.close() for listener in self._listeners), return_exceptions=True,
        ) if isinstance(value, Exception)]
        try:
            await self._close_server_state()
        except Exception as error:
            errors.append(error)
        self._started = False
        if errors:
            failure = errors[0] if len(errors) == 1 else ExceptionGroup("Server shutdown failed", errors)
            self._settle_closed(failure)
            raise failure
        self._settle_closed()

    def _receive(self, state: ConnectionState, chunk: bytes) -> None:
        if is_terminal_connection(state):
            return
        try:
            messages = state.decoder.push(chunk)
        except Exception as error:
            asyncio.get_running_loop().create_task(self._fail_protocol(state, self._to_protocol_error(error)))
            return
        for message in messages:
            if is_terminal_connection(state):
                return
            self._dispatch_message(state, message)

    def _dispatch_message(self, state: ConnectionState, message: ClientMessage) -> None:
        loop = asyncio.get_running_loop()
        if state.stage == "awaitingHello":
            if message["type"] != "hello":
                loop.create_task(self._fail_protocol(state, {
                    "code": "invalid_request", "message": "The first client message must be hello",
                }))
                return
            state.stage = "handshaking"

            async def handshake() -> None:
                try:
                    await self._finish_handshake(state, message)
                except Exception as error:
                    await self._fail_protocol(state, self._to_protocol_error(error))

            state.handshake = loop.create_task(handshake())
            return
        if message["type"] == "hello":
            loop.create_task(self._fail_protocol(state, {
                "code": "invalid_request", "message": "hello may only be sent as the first message",
            }))
            return
        if state.stage == "ready":
            if message["type"] == "cancel":
                self._handle_cancel(state, message)
            else:
                loop.create_task(self._handle_request(state, message))
        elif state.stage == "handshaking" and state.handshake is not None:
            async def after_handshake() -> None:
                await state.handshake
                if state.stage != "ready" or state.disconnected:
                    return
                if message["type"] == "cancel":
                    self._handle_cancel(state, message)
                else:
                    loop.create_task(self._handle_request(state, message))
            loop.create_task(after_handshake())

    async def _finish_handshake(self, state: ConnectionState, hello: ClientMessage) -> None:
        if not is_supported_protocol_version(hello["version"]):
            await self._fail_protocol(state, {
                "code": "version",
                "message": f"Unsupported protocol version {hello['version']}; expected {PROTOCOL_VERSION}",
            })
            return
        if self._closing or state.disconnected or state.stage != "handshaking" or state.connection.closed:
            return
        services = await _await_value(self._host.server_services.attach_client(
            _Presentation(self._sessions, state), TODO_CONTEXT,
        ))
        if self._closing or state.disconnected or state.stage != "handshaking" or state.connection.closed:
            await _await_value(services.release(TODO_CONTEXT))
            return
        state.server_services = services
        sent = await self._send_message(state, {
            "type": "hello", "version": PROTOCOL_VERSION, "serverId": self.server_id,
        })
        if sent and not state.disconnected and state.stage == "handshaking":
            state.stage = "ready"
            state.handshake_timeout.cancel()

    def _handle_cancel(self, state: ConnectionState, envelope: ClientMessage) -> None:
        if envelope["target"]["serverId"] != self.server_id:
            return
        active = state.active_requests.get(envelope["id"])
        if active is not None and _same_target(active[1], envelope["target"]):
            active[0].abort(AbortError("RPC request cancelled"))

    async def _handle_request(self, state: ConnectionState, envelope: RequestEnvelope) -> None:
        request_id = envelope["id"]
        if request_id in state.active_requests:
            await self._send_message(state, {"type": "response", "id": request_id, "ok": False,
                "error": {"code": "invalid_request", "message": "Request ID is already active"}})
            return
        try:
            call = parse_service_call(envelope["call"])
        except Exception:
            await self._send_message(state, {"type": "response", "id": request_id, "ok": False,
                "error": {"code": "invalid_request", "message": "Invalid service call"}})
            return
        controller = AbortController()
        active = (controller, envelope["target"])
        state.active_requests[request_id] = active
        context = with_abort_signal(controller.signal, TODO_CONTEXT)
        control = decode_service_control_call(call)
        subscribing = control if control is not None and control["type"] == "subscribe" else None
        pending_updates: list[ServiceProviderUpdate] = []
        subscription_ready = subscribing is None
        installed_encoder = False
        responded = False

        async def publish(subscription_id: str, update: ServiceProviderUpdate, _context: Context) -> None:
            if subscribing is not None and subscription_id == subscribing["subscriptionId"] and not subscription_ready:
                pending_updates.append(update)
                return
            await self._send_service_update(state, subscription_id, update)

        try:
            if envelope["target"]["serverId"] != self.server_id:
                raise WrongServerError()
            if subscribing is not None and subscribing["subscriptionId"] in state.service_state_encoders:
                raise ProtocolValidationError(f"Duplicate service subscription {subscribing['subscriptionId']}")
            result: JsonValue | Undefined
            if "sessionId" in envelope["target"]:
                result = await self._sessions.execute_service_call(call, envelope["target"], state, publish, context)
            elif state.server_services is not None:
                result = await state.server_services.invoke_service(call, publish, context)
            else:
                raise ProtocolValidationError(f"Unknown service member {call['serviceId']}.{call['member']}")
            if subscribing is not None:
                if result is UNDEFINED:
                    raise ProtocolValidationError("Service subscription did not return a snapshot")
                encoder = create_service_state_encoder()
                result = encoder.encode_snapshot(parse_service_subscription_snapshot(result))
                state.service_state_encoders[subscribing["subscriptionId"]] = encoder
                installed_encoder = True
            elif control is not None and control["type"] == "unsubscribe":
                state.service_state_encoders.pop(control["subscriptionId"], None)
            response: ServerMessage = {"type": "response", "id": request_id, "ok": True}
            if result is not UNDEFINED:
                response["result"] = result
            await self._send_message(state, response)
            responded = True
            if subscribing is not None:
                while pending_updates:
                    await self._send_service_update(state, subscribing["subscriptionId"], pending_updates.pop(0))
                subscription_ready = True
        except Exception as error:
            if subscribing is not None and installed_encoder and not responded:
                state.service_state_encoders.pop(subscribing["subscriptionId"], None)
            if responded:
                self._report_error(error)
                await self._close_connection(state.connection)
                self._disconnect(state)
            else:
                await self._send_message(state, {"type": "response", "id": request_id, "ok": False,
                    "error": {"code": "cancelled", "message": "RPC request cancelled"}
                    if controller.signal.aborted else self._to_protocol_error(error)})
        finally:
            if state.active_requests.get(request_id) is active:
                del state.active_requests[request_id]

    def _transport_closed(self, state: ConnectionState) -> None:
        if not state.disconnected and state.stage != "closing":
            try:
                state.decoder.end()
            except Exception as error:
                self._report_error(error)
        self._disconnect(state)

    def _disconnect(self, state: ConnectionState) -> None:
        if state.disconnected:
            return
        state.disconnected = True
        state.stage = "closed"
        state.handshake_timeout.cancel()
        for controller, _target in state.active_requests.values():
            controller.abort(RuntimeError("Client disconnected"))
        state.active_requests.clear()
        state.service_state_encoders.clear()
        if state in self._connections:
            self._connections.remove(state)
            self._notify_connection_count_changed()
        services = state.server_services
        state.server_services = None

        async def cleanup() -> None:
            results = await asyncio.gather(
                self._sessions.disconnect(state, TODO_CONTEXT),
                _await_value(services.release(TODO_CONTEXT)) if services is not None else asyncio.sleep(0),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    self._report_error(result)
        asyncio.get_running_loop().create_task(cleanup())

    async def _send_service_update(
        self, state: ConnectionState, subscription_id: str, update: ServiceProviderUpdate,
    ) -> None:
        encoder = state.service_state_encoders.get(subscription_id)
        if encoder is None:
            return
        await self._send_message(state, {"type": "service_update", "subscriptionId": subscription_id,
            "update": encoder.encode_update(update)})

    async def _send_message(self, state: ConnectionState, message: ServerMessage) -> bool:
        if state.disconnected or state.connection.closed:
            return False
        try:
            frame = encode_server_message(message, FrameDecoderOptions(max_frame_length=self._max_frame_length))
            await state.connection.send(frame)
            return True
        except Exception as error:
            self._report_error(error)
            await self._close_connection(state.connection)
            self._disconnect(state)
            return False

    async def _fail_protocol(self, state: ConnectionState, error: ProtocolError) -> None:
        if state.disconnected or state.stage in ("closing", "closed"):
            return
        state.stage = "closing"
        state.handshake_timeout.cancel()
        try:
            final_frame = encode_server_message({"type": "hello_error", "error": error},
                FrameDecoderOptions(max_frame_length=self._max_frame_length))
        except Exception as encode_error:
            self._report_error(encode_error)
            final_frame = None
        await self._close_connection(state.connection, final_frame)
        self._disconnect(state)

    async def _close_server_state(self) -> None:
        connections = list(self._connections)
        for state in connections:
            state.stage = "closing"
            state.handshake_timeout.cancel()
        await asyncio.gather(*(self._close_connection(state.connection) for state in connections))
        for state in connections:
            self._disconnect(state)
        await self._sessions.close(BACKGROUND_CONTEXT)
        self._connections.clear()

    async def _close_connection(self, connection: ByteConnection, final_chunk: bytes | None = None) -> None:
        try:
            await _await_value(connection.close(final_chunk))
        except Exception as error:
            self._report_error(error)

    def _to_protocol_error(self, error: object) -> ProtocolError:
        if isinstance(error, (ServerError, RemoteServiceError)):
            return {"code": error.code, "message": str(error)}
        if isinstance(error, ProtocolValidationError):
            return {"code": "invalid_request", "message": str(error)}
        self._report_error(error)
        return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}

    async def _publish_attachment(self, client: object, attachment, _context) -> None:
        await self._send_message(cast(ConnectionState, client), {
            "type": "attachment", "attachment": attachment,
        })

    def _notify_connection_count_changed(self) -> None:
        if self._on_connection_count_changed is not None:
            try:
                self._on_connection_count_changed(len(self._connections))
            except Exception as error:
                self._report_error(error)

    def _report_error(self, error: object) -> None:
        if self._on_error is not None:
            try:
                self._on_error(error if isinstance(error, Exception) else RuntimeError(str(error)))
            except Exception:
                pass

    def _settle_closed(self, error: Exception | None = None) -> None:
        closed = self.closed
        if closed.done():
            return
        if error is None:
            closed.set_result(None)
        else:
            closed.set_exception(error)


__all__ = ["Server"]
