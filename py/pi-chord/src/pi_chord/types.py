"""Chord contracts; transport records retain their TypeScript JSON field names."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

from .context import Context, ContextKey

if TYPE_CHECKING:
    from .delta import Op, Undefined
    from .services.provider import RemoteServiceProvider

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type ServiceMode = Literal["singleton", "keyed"]
type Disposal = Callable[[], None | Awaitable[None]]


class ReplicatedStateDelivery(TypedDict):
    kind: Literal["hydrate", "update"]
    sequence: int


class ReplicatedState[T](Protocol):
    @property
    def value(self) -> T | Undefined: ...

    def subscribe(
        self, listener: Callable[[T, Context, ReplicatedStateDelivery], None]
    ) -> Callable[[], None]: ...


class MutableReplicatedState[T](ReplicatedState[T], Protocol):
    @property
    def value(self) -> T: ...

    @property
    def state(self) -> T: ...

    def publish(self, context: Context) -> None: ...


@dataclass(frozen=True)
class Service[T]:
    id: str
    local: bool = False


class ServiceSpawner[T](Protocol):
    def spawn(self, key: str, implementation: T) -> Callable[[], None]: ...


class RemoteServices(Protocol):
    def use[T](self, service: Service[T]) -> T: ...

    def observe[T](
        self, service: Service[T], handler: Callable[[T, Context], None | Awaitable[None]]
    ) -> Callable[[], None]: ...

    async def ready(self, context: Context) -> None: ...

    async def dispose(self, context: Context) -> None: ...


class ServiceCatalogueEntry(TypedDict):
    serviceId: str
    mode: ServiceMode


class ServiceInstanceAddress(TypedDict):
    key: str
    generation: int


class ServiceMethodSnapshot(TypedDict):
    name: str
    kind: Literal["method"]


class ServiceStateSnapshot(TypedDict):
    name: str
    kind: Literal["state"]
    sequence: int
    ops: Sequence[Op]


type ServiceMemberSnapshot = ServiceMethodSnapshot | ServiceStateSnapshot


class ServiceInstanceSnapshot(TypedDict):
    instance: NotRequired[ServiceInstanceAddress]
    members: Sequence[ServiceMemberSnapshot]


class ServiceSubscriptionSnapshot(TypedDict):
    serviceId: str
    mode: ServiceMode
    instances: Sequence[ServiceInstanceSnapshot]


class ServiceStateUpdate(TypedDict):
    type: Literal["state"]
    instance: NotRequired[ServiceInstanceAddress]
    member: str
    sequence: int
    ops: Sequence[Op]


class ServiceUnavailableUpdate(TypedDict):
    type: Literal["unavailable"]


class ServiceReplacedUpdate(TypedDict):
    type: Literal["replaced"]
    snapshot: ServiceInstanceSnapshot


class ServiceSpawnedUpdate(TypedDict):
    type: Literal["spawned"]
    instance: ServiceInstanceSnapshot


class ServiceClosedUpdate(TypedDict):
    type: Literal["closed"]
    instance: ServiceInstanceAddress


type ServiceProviderUpdate = (
    ServiceStateUpdate | ServiceUnavailableUpdate | ServiceReplacedUpdate
    | ServiceSpawnedUpdate | ServiceClosedUpdate
)


class ServiceCall(TypedDict):
    serviceId: str
    instance: NotRequired[ServiceInstanceAddress]
    member: str
    args: Sequence[JsonValue]


class ServiceSubscription(Protocol):
    @property
    def snapshot(self) -> ServiceSubscriptionSnapshot: ...

    def activate(self) -> None: ...

    def close(self, context: Context | None = None) -> None | Awaitable[None]: ...


class RemoteServiceTransport(Protocol):
    async def invoke(self, call: ServiceCall, context: Context) -> JsonValue | Undefined: ...

    async def subscribe(
        self, service_id: str, mode: ServiceMode,
        listener: Callable[[ServiceProviderUpdate, Context], None], context: Context,
    ) -> ServiceSubscription: ...


class ServiceIdentity(Protocol):
    @property
    def id(self) -> str: ...


@dataclass(frozen=True)
class RemoteServiceBindingOptions:
    services: Sequence[ServiceIdentity]
    transport: RemoteServiceTransport
    bound: bool = True
    on_error: Callable[[Exception], None] | None = None
    assert_access: Callable[[], None] | None = None


class RemoteServiceBinding(RemoteServices, Protocol):
    async def rebind(self, bound: bool, context: Context) -> None: ...


class FacetEnvironment(Protocol):
    def use[T](self, service: Service[T]) -> T: ...

    def observe[T](
        self, service: Service[T], handler: Callable[[T, Context], None | Awaitable[None]]
    ) -> None: ...

    def provide[T](self, service: Service[T], implementation: T) -> None: ...

    def provide_many[T](self, service: Service[T]) -> ServiceSpawner[T]: ...

    def replicated_state[T](self, initial: T) -> MutableReplicatedState[T]: ...

    def own(self, disposal: Disposal) -> None: ...

    def on_activate(self, callback: Disposal) -> None: ...

    def on_deactivate(self, callback: Disposal) -> None: ...


@dataclass(frozen=True)
class Facet:
    id: str
    setup: Callable[[FacetEnvironment], None]


@dataclass(frozen=True)
class RemoteServiceSourceOptions:
    services: Sequence[ServiceIdentity]
    assert_access: Callable[[], None]
    on_error: Callable[[Exception], None]


class RemoteServiceSource(Protocol):
    @property
    def accepts_unavailable_services(self) -> bool: ...

    async def catalogue(self, context: Context) -> Sequence[ServiceCatalogueEntry]: ...

    def open(self, options: RemoteServiceSourceOptions) -> RemoteServices: ...


@dataclass(frozen=True)
class FacetOptions:
    facets: Sequence[Facet]
    service_sources: Sequence[RemoteServiceSource] = ()
    on_error: Callable[[Exception], None] | None = None


class FacetHost(Protocol):
    @property
    def services(self) -> RemoteServiceProvider: ...

    async def reload(self, facets: Sequence[Facet]) -> None: ...

    async def dispose(self) -> None: ...


class LoadedFacets(Protocol):
    @property
    def facets(self) -> Sequence[Facet]: ...

    async def dispose(self) -> None: ...


class FacetLoader(Protocol):
    async def load(self) -> LoadedFacets: ...
