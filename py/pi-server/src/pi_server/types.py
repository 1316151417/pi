"""Routed host capabilities from ``types.ts``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pi_chord.context import Context
from pi_chord.delta import Undefined
from pi_chord.types import JsonValue, ServiceCall, ServiceProviderUpdate

if TYPE_CHECKING:
    from .listener import ServerListener

type MaybePromise[T] = T | Awaitable[T]
type PublishUpdate = Callable[[str, ServiceProviderUpdate, Context], MaybePromise[None]]


@dataclass
class ServerOptions:
    listeners: Sequence[ServerListener]
    server_id: str
    max_frame_length: int | float | None = None
    handshake_timeout_ms: int | float | None = None
    on_connection_count_changed: Callable[[int], None] | None = None
    on_error: Callable[[Exception], None] | None = None


class RoutedSessionAttachment(Protocol):
    def invoke_service(
        self, call: ServiceCall, publish: PublishUpdate, context: Context,
    ) -> Awaitable[JsonValue | Undefined]: ...

    def release(self, context: Context) -> MaybePromise[None]: ...


class RoutedServerPresentation(Protocol):
    def attach_session(self, session_id: str, context: Context) -> Awaitable[None]: ...

    def detach_session(self, context: Context) -> Awaitable[None]: ...

    def prepare_session_removal(self, session_id: str, context: Context) -> Awaitable[None]: ...


class RoutedServerServiceAttachment(Protocol):
    def invoke_service(
        self, call: ServiceCall, publish: PublishUpdate, context: Context,
    ) -> Awaitable[JsonValue | Undefined]: ...

    def release(self, context: Context) -> MaybePromise[None]: ...


class RoutedServerServiceHost(Protocol):
    def attach_client(
        self, presentation: RoutedServerPresentation, context: Context,
    ) -> MaybePromise[RoutedServerServiceAttachment]: ...


class RoutedSessionHandle(Protocol):
    @property
    def terminated(self) -> Awaitable[Exception | None] | None: ...

    def attach_client(self, context: Context) -> MaybePromise[RoutedSessionAttachment]: ...

    def close(self, context: Context) -> Awaitable[None]: ...


class SessionMetadata(Protocol):
    id: str


class ServerHost[TMetadata: SessionMetadata](Protocol):
    @property
    def server_services(self) -> RoutedServerServiceHost: ...

    def resolve_session(self, session_id: str, context: Context) -> Awaitable[TMetadata]: ...

    def open_session(self, metadata: TMetadata, context: Context) -> Awaitable[RoutedSessionHandle]: ...
