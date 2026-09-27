"""Transport-neutral routed client from ``client.ts``."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from pi_chord.context import AbortError, AbortSignalLike, BACKGROUND_CONTEXT, Context
from pi_chord.delta import UNDEFINED, Undefined
from pi_chord.services.state_codec import ServiceStateDecoder, create_service_state_decoder
from pi_chord.services.wire import (
    create_service_catalogue_call, create_service_subscribe_call,
    create_service_unsubscribe_call, parse_service_call,
    parse_service_catalogue, parse_wire_service_provider_update,
    parse_wire_service_subscription_snapshot,
)
from pi_chord.types import (
    JsonValue, ServiceCall, ServiceCatalogueEntry, ServiceMode,
    ServiceProviderUpdate, ServiceSubscriptionSnapshot,
)
from pi_protocol import (
    FrameDecoderOptions, ProtocolValidationError, RpcTarget, ServerHello,
    ServerMessage, SessionTarget, encode_client_message, is_server_id,
)

from .connection import Connection, ConnectionOptions
from .errors import ClientDisposedError, DisconnectedError, ServerError, to_error
from .promise import create_promise_resolvers
from .types import (
    AttachmentChangeListener, ClientOptions, ConnectionState, ConnectionStateChange,
    Unsubscribe,
)

type ServiceResult = JsonValue | Undefined
type ServiceUpdateListener = Callable[[ServiceProviderUpdate], None | Awaitable[None]]


@dataclass
class _PendingRequest:
    resolve: Callable[[ServiceResult], None]
    reject: Callable[[Exception], None]
    cleanup: Callable[[], None]


@dataclass
class _ActiveServiceListener:
    target: RpcTarget
    listener: ServiceUpdateListener
    decoder: ServiceStateDecoder = field(default_factory=create_service_state_decoder)
    queued_wire_updates: list[JsonValue] = field(default_factory=list)
    queued: list[ServiceProviderUpdate] = field(default_factory=list)
    delivery_tail: asyncio.Future[None] | None = None
    hydrated: bool = False
    ready: bool = False


class _ListenerSet[T]:
    """JS Set identity and live traversal, including remove/add during delivery."""

    def __init__(self) -> None:
        self._items: list[T | None] = []

    def add(self, listener: T) -> None:
        if not any(item is listener for item in self._items):
            self._items.append(listener)

    def discard(self, listener: T) -> None:
        for index, item in enumerate(self._items):
            if item is listener:
                self._items[index] = None

    def clear(self) -> None:
        self._items.clear()

    def __iter__(self):
        index = 0
        while index < len(self._items):
            item = self._items[index]
            index += 1
            if item is not None:
                yield item


class _ClientServiceSubscription:
    def __init__(
        self, client: Client, subscription_id: str, target: RpcTarget,
        snapshot: ServiceSubscriptionSnapshot, active: _ActiveServiceListener,
    ) -> None:
        self.id = subscription_id
        self.target = target
        self.snapshot = snapshot
        self._client = client
        self._active = active
        self._disposed = False

    def start(self) -> None:
        if self._disposed or self._active.ready:
            return
        self._active.ready = True
        queued = self._active.queued[:]
        self._active.queued.clear()
        for update in queued:
            self._client._deliver_service_update(self._active, update)

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        if self._client._service_listeners.get(self.id) is self._active:
            del self._client._service_listeners[self.id]
        try:
            if self._client.connected and self._client._target_is_current(self.target):
                await self._client._request(self.target, create_service_unsubscribe_call(self.id))
            if self._active.delivery_tail is not None:
                await self._active.delivery_tail
        finally:
            self._active.queued_wire_updates.clear()
            self._active.queued.clear()


class Client:
    def __init__(self, options: ClientOptions) -> None:
        if not is_server_id(options.server_id):
            raise TypeError("serverId must be a canonical lowercase UUIDv4")
        self._options = options
        self._connection = Connection(ConnectionOptions(
            transport_factory=options.transport_factory,
            server_id=options.server_id,
            max_frame_length=options.max_frame_length,
            on_handshake=self._on_handshake,
            on_message=self._handle_message,
            on_state_change=self._handle_connection_state_change,
        ))
        self._pending_requests: dict[str, _PendingRequest] = {}
        self._connection_state_listeners: _ListenerSet[Callable[[ConnectionStateChange], None]] = _ListenerSet()
        self._attachment_listeners: _ListenerSet[AttachmentChangeListener] = _ListenerSet()
        self._service_listeners: dict[str, _ActiveServiceListener] = {}
        self._request_sequence = 0
        self._service_subscription_sequence = 0
        self._hello: ServerHello | None = None
        self._attachment: SessionTarget | None = None
        self._disposed = False
        self._dispose_promise: asyncio.Future[None] | None = None

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection.state

    @property
    def connected(self) -> bool:
        return self.connection_state == "connected"

    @property
    def server_id(self) -> str:
        return self._options.server_id

    @property
    def hello(self) -> ServerHello | None:
        return self._hello

    @property
    def attachment(self) -> SessionTarget | None:
        return self._attachment

    def connect(self: Client | ClientOptions) -> asyncio.Future[ServerHello] | asyncio.Task[Client]:
        """``Client.connect(options)`` and ``client.connect()`` share the TS name."""
        if isinstance(self, ClientOptions):
            async def create() -> Client:
                client = Client(self)
                try:
                    await client.connect()
                    return client
                except Exception:
                    await client.dispose()
                    raise
            return asyncio.get_running_loop().create_task(create())
        client = cast(Client, self)
        if client._disposed:
            rejected: asyncio.Future[ServerHello] = asyncio.get_running_loop().create_future()
            rejected.set_exception(ClientDisposedError())
            return rejected
        client._hello = None
        return client._connection.connect()

    def reconnect(self) -> asyncio.Future[ServerHello]:
        return cast(asyncio.Future[ServerHello], self.connect())

    def disconnect(self, reason: str = "Client disconnected") -> None:
        self._connection.disconnect(reason)

    def on_connection_state_change(self, listener: Callable[[ConnectionStateChange], None]) -> Unsubscribe:
        self._assert_not_disposed()
        self._connection_state_listeners.add(listener)
        return lambda: self._connection_state_listeners.discard(listener)

    def on_attachment_change(self, listener: AttachmentChangeListener) -> Unsubscribe:
        self._assert_not_disposed()
        self._attachment_listeners.add(listener)
        return lambda: self._attachment_listeners.discard(listener)

    def request(
        self, target: RpcTarget, call: ServiceCall, signal: AbortSignalLike | None = None,
    ) -> asyncio.Future[ServiceResult]:
        return self._request(target, call, signal)

    async def service_catalogue(
        self, target: RpcTarget, signal: AbortSignalLike | None = None,
    ) -> Sequence[ServiceCatalogueEntry]:
        result = await self._request(target, create_service_catalogue_call(), signal)
        try:
            return parse_service_catalogue(result)
        except Exception as error:
            validation_error = ProtocolValidationError(str(error))
            self._connection.fail(validation_error)
            raise validation_error from error

    async def subscribe_service(
        self,
        target: RpcTarget,
        service_id: str,
        mode: ServiceMode,
        listener: ServiceUpdateListener,
        signal: AbortSignalLike | None = None,
    ) -> _ClientServiceSubscription:
        self._service_subscription_sequence += 1
        subscription_id = f"service-{self._service_subscription_sequence}"
        active = _ActiveServiceListener(target, listener)
        self._service_listeners[subscription_id] = active

        def hydrate(result: ServiceResult) -> ServiceSubscriptionSnapshot:
            snapshot = active.decoder.decode_snapshot(parse_wire_service_subscription_snapshot(result))
            active.hydrated = True
            updates = active.queued_wire_updates[:]
            active.queued_wire_updates.clear()
            for update in updates:
                active.queued.append(active.decoder.decode_update(parse_wire_service_provider_update(update)))
            return snapshot

        try:
            snapshot = await self._request(
                target, create_service_subscribe_call(subscription_id, service_id, mode), signal,
                hydrate,
            )
        except Exception:
            if self._service_listeners.get(subscription_id) is active:
                del self._service_listeners[subscription_id]
            raise
        if self._service_listeners.get(subscription_id) is not active:
            raise DisconnectedError()
        return _ClientServiceSubscription(self, subscription_id, target, snapshot, active)

    def _request[T](
        self,
        target: RpcTarget,
        call: ServiceCall,
        signal: AbortSignalLike | None = None,
        transform: Callable[[ServiceResult], T] | None = None,
    ) -> asyncio.Future[T]:
        loop = asyncio.get_running_loop()
        if self._disposed or not self.connected or signal is not None and signal.aborted:
            rejected: asyncio.Future[T] = loop.create_future()
            rejected.set_exception(
                ClientDisposedError() if self._disposed else DisconnectedError() if not self.connected
                else _abort_error(signal)
            )
            return rejected
        self._request_sequence += 1
        request_id = f"request-{self._request_sequence}"
        promise = create_promise_resolvers()
        sent = False
        aborted = False

        def send_cancel() -> None:
            if not sent or not self.connected:
                return
            try:
                self._connection.send(encode_client_message(
                    {"type": "cancel", "id": request_id, "target": target},
                    FrameDecoderOptions(max_frame_length=self._connection.max_frame_length),
                ))
            except Exception as error:
                self._connection.fail(to_error(error))

        def on_abort() -> None:
            nonlocal aborted
            if aborted:
                return
            aborted = True
            promise.reject(_abort_error(signal))
            send_cancel()

        if signal is not None:
            signal.add_event_listener("abort", on_abort, once=True)

        def resolve(result: ServiceResult) -> None:
            try:
                promise.resolve(cast(T, result) if transform is None else transform(result))
            except Exception as error:
                validation_error = ProtocolValidationError(str(error))
                self._connection.fail(validation_error)
                promise.reject(validation_error)

        self._pending_requests[request_id] = _PendingRequest(
            resolve, promise.reject,
            lambda: signal.remove_event_listener("abort", on_abort) if signal is not None else None,
        )
        try:
            frame = encode_client_message(
                {"type": "request", "id": request_id, "target": target, "call": parse_service_call(call)},
                FrameDecoderOptions(max_frame_length=self._connection.max_frame_length),
            )
        except Exception as error:
            pending = self._take_pending_request(request_id)
            if pending is not None:
                pending.reject(to_error(error))
            return promise.promise
        self._connection.send(frame)
        sent = True
        if aborted:
            send_cancel()
        return promise.promise

    def _on_handshake(self, hello: ServerHello) -> None:
        self._hello = hello

    def _handle_message(self, message: ServerMessage) -> None:
        kind = message["type"]
        if kind == "attachment":
            attachment = message["attachment"]
            if attachment is not None and attachment["serverId"] != self.server_id:
                self._connection.fail(ProtocolValidationError("Attachment update belongs to another server"))
                return
            self._set_attachment(attachment)
            return
        if kind == "service_update":
            active = self._service_listeners.get(message["subscriptionId"])
            if active is None:
                return
            if not active.hydrated:
                active.queued_wire_updates.append(message["update"])
                return
            try:
                update = active.decoder.decode_update(parse_wire_service_provider_update(message["update"]))
            except Exception as error:
                self._connection.fail(ProtocolValidationError(str(error)))
                return
            if active.ready:
                self._deliver_service_update(active, update)
            else:
                active.queued.append(update)
            return
        if kind != "response":
            return
        pending = self._take_pending_request(message["id"])
        if pending is None:
            self._connection.fail(ProtocolValidationError("Response has no matching request"))
            return
        if not message["ok"]:
            pending.reject(ServerError(message["error"]))
            return
        pending.resolve(message.get("result", UNDEFINED))

    def _handle_connection_state_change(self, change: ConnectionStateChange) -> None:
        if change.state == "disconnected":
            self._hello = None
            self._set_attachment(None)
            self._reject_pending_requests(change.error if change.error is not None else DisconnectedError())
            self._service_listeners.clear()
        for listener in self._connection_state_listeners:
            try:
                listener(change)
            except Exception as error:
                self._report_listener_error(error)

    def _take_pending_request(self, request_id: str) -> _PendingRequest | None:
        pending = self._pending_requests.pop(request_id, None)
        if pending is not None:
            pending.cleanup()
        return pending

    def _reject_pending_requests(self, error: Exception) -> None:
        requests = list(self._pending_requests.values())
        self._pending_requests.clear()
        for pending in requests:
            pending.cleanup()
            pending.reject(error)

    def dispose(self) -> asyncio.Future[None]:
        if self._dispose_promise is not None:
            return self._dispose_promise
        self._disposed = True
        promise: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._dispose_promise = promise
        error = ClientDisposedError()
        self._reject_pending_requests(error)
        self._connection.disconnect(error)
        self._hello = None
        self._set_attachment(None)
        self._connection_state_listeners.clear()
        self._attachment_listeners.clear()
        self._service_listeners.clear()
        promise.set_result(None)
        return promise

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.dispose()

    def _set_attachment(self, attachment: SessionTarget | None) -> None:
        previous = self._attachment
        if _attachment_key(previous) == _attachment_key(attachment):
            return
        self._attachment = attachment
        for listener in self._attachment_listeners:
            try:
                listener(attachment)
            except Exception as error:
                self._report_listener_error(error)

    def _deliver_service_update(self, active: _ActiveServiceListener, update: ServiceProviderUpdate) -> None:
        previous = active.delivery_tail

        async def deliver() -> None:
            if previous is not None:
                await previous
            try:
                result = active.listener(update)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                self._report_listener_error(error)

        active.delivery_tail = asyncio.get_running_loop().create_task(deliver())

    def _target_is_current(self, target: RpcTarget) -> bool:
        if "sessionId" not in target:
            return self._hello is not None and self._hello["serverId"] == target["serverId"]
        return _attachment_key(self._attachment) == _attachment_key(target)

    def _assert_not_disposed(self) -> None:
        if self._disposed:
            raise ClientDisposedError()

    def _report_listener_error(self, error: object) -> None:
        handler = self._options.on_listener_error
        if handler is None:
            return
        try:
            handler(to_error(error))
        except Exception:
            pass


def _attachment_key(target: SessionTarget | None) -> tuple[str, str, str] | None:
    return None if target is None else (
        target["serverId"], target["sessionId"], target["attachmentId"],
    )


def _abort_error(signal: AbortSignalLike | None) -> Exception:
    reason = signal.reason if signal is not None else None
    return reason if isinstance(reason, Exception) else AbortError()


class _TransportSubscription:
    def __init__(self, subscription: _ClientServiceSubscription) -> None:
        self.snapshot = subscription.snapshot
        self._subscription = subscription

    def activate(self) -> None:
        self._subscription.start()

    def close(self, _context: Context | None = None) -> Awaitable[None]:
        return self._subscription.dispose()


class _ClientServiceTransport:
    def __init__(self, client: Client, get_target: Callable[[], RpcTarget | None]) -> None:
        self._client = client
        self._get_target = get_target

    def _target(self) -> RpcTarget:
        target = self._get_target()
        if target is None:
            raise RuntimeError("Remote service target is unavailable")
        return target

    async def invoke(self, call: ServiceCall, context: Context) -> ServiceResult:
        return await self._client.request(self._target(), call, context.abort_signal)

    async def subscribe(
        self,
        service_id: str,
        mode: ServiceMode,
        listener: Callable[[ServiceProviderUpdate, Context], None | Awaitable[None]],
        context: Context,
    ) -> _TransportSubscription:
        subscription = await self._client.subscribe_service(
            self._target(), service_id, mode,
            lambda update: listener(update, BACKGROUND_CONTEXT), context.abort_signal,
        )
        return _TransportSubscription(subscription)


def create_client_service_transport(
    client: Client, get_target: Callable[[], RpcTarget | None],
) -> _ClientServiceTransport:
    return _ClientServiceTransport(client, get_target)


__all__ = ["Client", "create_client_service_transport"]
