"""Public client contracts from ``types.ts``."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pi_chord.types import ServiceProviderUpdate, ServiceSubscriptionSnapshot
from pi_protocol.protocol import RpcTarget, SessionTarget

from .transport import ByteTransportFactory

type ConnectionState = Literal["disconnected", "connecting", "connected"]
type Unsubscribe = Callable[[], None]
type ListenerErrorHandler = Callable[[Exception], None]
type AttachmentChangeListener = Callable[[SessionTarget | None], None]


@dataclass(frozen=True)
class ConnectionStateChange:
    state: ConnectionState
    error: Exception | None = None


class ServiceSubscription(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def target(self) -> RpcTarget: ...

    @property
    def snapshot(self) -> ServiceSubscriptionSnapshot: ...

    def start(self) -> None: ...

    def dispose(self) -> Awaitable[None]: ...


@dataclass
class ClientOptions:
    transport_factory: ByteTransportFactory
    server_id: str
    max_frame_length: int | float | None = None
    on_listener_error: ListenerErrorHandler | None = None
