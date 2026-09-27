"""Remote provider admission, snapshots, and updates from ``services/provider.ts``.

Implementations expose their own data members: dictionary entries or instance
attributes. As with Object.keys in the source, inherited class methods are not
published implicitly. Store bound methods as dictionary entries when needed.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from icu import Collator

from ..context import Context
from ..delta import UNDEFINED, Op, Undefined
from ..types import (
    JsonValue,
    Service,
    ServiceCall,
    ServiceCatalogueEntry,
    ServiceIdentity,
    ServiceInstanceAddress,
    ServiceInstanceSnapshot,
    ServiceMemberSnapshot,
    ServiceMode,
    ServiceProviderUpdate,
    ServiceStateUpdate,
    ServiceSubscriptionSnapshot,
)
from .errors import RemoteServiceError
from .state import service_delivery_context
from .state_internals import ReplicatedStateInternals, get_replicated_state_internals
from .wire import decode_service_control_call

type ServiceUpdatePublisher = Callable[[str, ServiceProviderUpdate, Context], None | Awaitable[None]]
type _ServiceMemberKind = Literal["method", "state"]


class RemoteServiceEndpoint(Protocol):
    async def invoke(
        self, call: ServiceCall, publish: ServiceUpdatePublisher, context: Context,
    ) -> JsonValue | Undefined: ...

    def dispose(self) -> None: ...


@dataclass(frozen=True)
class ServiceProviderDefinition:
    service: ServiceIdentity
    mode: ServiceMode


@dataclass(frozen=True)
class _MethodMember:
    method: Callable[..., object]
    kind: Literal["method"] = "method"


@dataclass(frozen=True)
class _StateMember:
    state: ReplicatedStateInternals
    kind: Literal["state"] = "state"


type _InstanceMember = _MethodMember | _StateMember


@dataclass
class _ClassifiedImplementation:
    implementation: object
    members: dict[str, _InstanceMember]


@dataclass(eq=False)
class _ProviderInstance:
    implementation: object
    members: Mapping[str, _InstanceMember]
    address: ServiceInstanceAddress | None
    remove_member_listeners: list[Callable[[], None]] = field(default_factory=list)
    active: bool = True


@dataclass(eq=False)
class _ProviderSubscriber:
    listener: Callable[[ServiceProviderUpdate, Context], None]
    buffer: list[tuple[ServiceProviderUpdate, Context]] = field(default_factory=list)
    active: bool = False
    terminated: bool = False
    closed: bool = False


@dataclass
class _ServiceRegistration:
    service_id: str
    mode: ServiceMode
    singleton: _ProviderInstance | None = None
    singleton_shape: dict[str, _ServiceMemberKind] | None = None
    instances: dict[str, _ProviderInstance] = field(default_factory=dict)
    generations: dict[str, int] = field(default_factory=dict)
    subscribers: dict[_ProviderSubscriber, None] = field(default_factory=dict)


class _FrozenCatalogueEntry(dict[str, object]):
    """Keep catalogue entries JSON-serializable while enforcing Object.freeze."""

    def __setitem__(self, key: str, value: object) -> None:
        raise TypeError("Remote service catalogue is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("Remote service catalogue is immutable")

    def clear(self) -> None:
        raise TypeError("Remote service catalogue is immutable")

    def pop(self, key: str, *default: object) -> object:
        raise TypeError("Remote service catalogue is immutable")

    def popitem(self) -> tuple[str, object]:
        raise TypeError("Remote service catalogue is immutable")

    def setdefault(self, key: str, default: object = None) -> object:
        raise TypeError("Remote service catalogue is immutable")

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Remote service catalogue is immutable")

    def __ior__(self, other: object) -> _FrozenCatalogueEntry:
        raise TypeError("Remote service catalogue is immutable")


class _ProviderSubscription:
    def __init__(
        self, registration: _ServiceRegistration, subscriber: _ProviderSubscriber,
        snapshot: ServiceSubscriptionSnapshot,
    ) -> None:
        self._registration = registration
        self._subscriber = subscriber
        self.snapshot = snapshot

    def activate(self) -> None:
        subscriber = self._subscriber
        if subscriber.closed or subscriber.active:
            return
        subscriber.active = True
        errors: list[BaseException] = []
        buffered = subscriber.buffer[:]
        subscriber.buffer.clear()
        try:
            for update, context in buffered:
                try:
                    subscriber.listener(update, context)
                except BaseException as error:
                    errors.append(error)
        finally:
            if subscriber.terminated:
                subscriber.closed = True
        _throw_collected_errors(errors, "Failed to activate remote service subscription")

    def close(self, context: Context | None = None) -> None:
        subscriber = self._subscriber
        if subscriber.closed:
            return
        subscriber.closed = True
        subscriber.buffer.clear()
        self._registration.subscribers.pop(subscriber, None)


class RemoteServiceProvider:
    def __init__(self, entries: Sequence[ServiceProviderDefinition | ServiceIdentity]) -> None:
        definitions = [
            entry if isinstance(entry, ServiceProviderDefinition) else ServiceProviderDefinition(entry, "singleton")
            for entry in entries
        ]
        for definition in definitions:
            if getattr(definition.service, "local", False) is True:
                raise TypeError(f"Local service {definition.service.id} cannot be published remotely")
        ids = [definition.service.id for definition in definitions]
        if len(set(ids)) != len(ids):
            raise TypeError("Remote service catalogue contains duplicate IDs")
        self._catalogue = tuple(
            cast(ServiceCatalogueEntry, _FrozenCatalogueEntry(serviceId=definition.service.id, mode=definition.mode))
            for definition in definitions
        )
        self._registrations = {
            definition.service.id: _ServiceRegistration(definition.service.id, definition.mode)
            for definition in definitions
        }
        self._disposed = False

    @property
    def catalogue(self) -> Sequence[ServiceCatalogueEntry]:
        return self._catalogue

    def provide[T](self, service: Service[T], implementation: T) -> None:
        self._assert_active()
        self._assert_remotable(service)
        self._assert_allowed(service.id)
        registration = self._registration(service.id, "singleton")
        if registration.singleton is not None:
            raise RemoteServiceError("service_mode_mismatch", f"Remote service {service.id} already has a provider")
        classified = _classify_remote_service_implementation(registration.service_id, implementation)
        shape = {name: member.kind for name, member in classified.members.items()}
        self._assert_singleton_shape(registration, shape)
        registration.singleton = self._create_instance(registration, classified, None)
        registration.singleton_shape = shape

    def withdraw[T](self, service: Service[T]) -> None:
        self._assert_active()
        self._assert_remotable(service)
        self._assert_allowed(service.id)
        registration = self._registration(service.id, "singleton")
        previous = registration.singleton
        if previous is None:
            return
        previous.active = False
        for remove in previous.remove_member_listeners:
            remove()
        registration.singleton = None
        self._emit(registration, {"type": "unavailable"})

    def validate_replacement[T](self, service: Service[T], implementation: T) -> None:
        self._assert_active()
        self._assert_remotable(service)
        self._assert_allowed(service.id)
        registration = self._registration(service.id, "singleton")
        classified = _classify_remote_service_implementation(registration.service_id, implementation)
        self._assert_singleton_shape(registration, {name: member.kind for name, member in classified.members.items()})

    def replace[T](self, service: Service[T], implementation: T) -> None:
        self._assert_active()
        self._assert_remotable(service)
        self._assert_allowed(service.id)
        registration = self._registration(service.id, "singleton")
        classified = _classify_remote_service_implementation(registration.service_id, implementation)
        shape = {name: member.kind for name, member in classified.members.items()}
        self._assert_singleton_shape(registration, shape)
        replacement = self._create_instance(registration, classified, None)
        previous = registration.singleton
        if previous is not None:
            previous.active = False
            for remove in previous.remove_member_listeners:
                remove()
        registration.singleton = replacement
        registration.singleton_shape = shape
        self._emit(registration, {"type": "replaced", "snapshot": self._snapshot_instance(replacement)})

    def use[T](self, service: Service[T]) -> T:
        self._assert_active()
        self._assert_remotable(service)
        self._assert_allowed(service.id)
        registration = self._registrations.get(service.id)
        if registration is None or registration.mode != "singleton" or registration.singleton is None:
            raise RemoteServiceError("service_not_found", f"Remote service {service.id} has no local provider")
        return cast(T, registration.singleton.implementation)

    def spawn[T](self, service: Service[T], key: str, implementation: T) -> Callable[[], None]:
        self._assert_active()
        self._assert_remotable(service)
        self._assert_allowed(service.id)
        if len(key) == 0:
            raise TypeError("Remote service instance key must not be empty")
        registration = self._registration(service.id, "keyed")
        if key in registration.instances:
            raise RemoteServiceError(
                "service_mode_mismatch", f"Remote service {service.id} already has a live instance with key {key}",
            )
        generation = registration.generations.get(key, 0) + 1
        registration.generations[key] = generation
        address: ServiceInstanceAddress = {"key": key, "generation": generation}
        classified = _classify_remote_service_implementation(registration.service_id, implementation)
        instance = self._create_instance(registration, classified, address)
        registration.instances[key] = instance
        self._emit(registration, {"type": "spawned", "instance": self._snapshot_instance(instance)})
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            if registration.instances.get(key) is not instance:
                return
            instance.active = False
            for remove in instance.remove_member_listeners:
                remove()
            del registration.instances[key]
            self._emit(registration, {"type": "closed", "instance": address})

        return close

    async def invoke(self, call: ServiceCall, context: Context) -> JsonValue | Undefined:
        self._assert_active()
        service_id = call["serviceId"]
        self._assert_allowed(service_id)
        registration = self._registrations.get(service_id)
        if registration is None:
            raise RemoteServiceError("service_not_found", f"Unknown remote service {service_id}")
        instance = self._resolve_instance(registration, call.get("instance"))
        member = instance.members.get(call["member"])
        if member is None:
            raise RemoteServiceError(
                "service_member_not_found", f"Unknown remote service member {service_id}.{call['member']}",
            )
        if member.kind != "method":
            raise RemoteServiceError(
                "service_member_mismatch", f"Remote service member {service_id}.{call['member']} is not a method",
            )
        result = member.method(*call["args"], context)
        if inspect.isawaitable(result):
            result = await result
        return cast(JsonValue | Undefined, result)

    def subscribe(
        self, service_id: str, mode: ServiceMode,
        listener: Callable[[ServiceProviderUpdate, Context], None],
    ) -> _ProviderSubscription:
        self._assert_active()
        self._assert_allowed(service_id)
        registration = self._registration(service_id, mode)
        if registration.mode == "singleton" and registration.singleton is None:
            raise RemoteServiceError("service_not_found", f"Remote service {service_id} has no provider")
        subscriber = _ProviderSubscriber(listener)
        self._publish_pending(registration)
        registration.subscribers[subscriber] = None
        return _ProviderSubscription(registration, subscriber, self._snapshot(registration))

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        errors: list[BaseException] = []
        for registration in self._registrations.values():
            singleton = registration.singleton
            if singleton is not None:
                singleton.active = False
                for remove in singleton.remove_member_listeners:
                    remove()
                registration.singleton = None
                try:
                    self._emit(registration, {"type": "unavailable"})
                except BaseException as error:
                    errors.append(error)
            for key, instance in tuple(registration.instances.items()):
                instance.active = False
                for remove in instance.remove_member_listeners:
                    remove()
                registration.instances.pop(key, None)
                try:
                    self._emit(registration, {"type": "closed", "instance": cast(ServiceInstanceAddress, instance.address)})
                except BaseException as error:
                    errors.append(error)
            for subscriber in registration.subscribers:
                if subscriber.active:
                    subscriber.closed = True
                    subscriber.buffer.clear()
                else:
                    subscriber.terminated = True
            registration.subscribers.clear()
        self._registrations.clear()
        _throw_collected_errors(errors, "Failed to dispose remote service provider")

    def _registration(self, service_id: str, mode: ServiceMode) -> _ServiceRegistration:
        registration = self._registrations.get(service_id)
        if registration is None:
            raise RemoteServiceError("service_not_found", f"Unknown remote service {service_id}")
        if registration.mode != mode:
            raise RemoteServiceError(
                "service_mode_mismatch", f"Remote service {service_id} is {registration.mode}, not {mode}",
            )
        return registration

    def _create_instance(
        self, registration: _ServiceRegistration, classified: _ClassifiedImplementation,
        address: ServiceInstanceAddress | None,
    ) -> _ProviderInstance:
        instance = _ProviderInstance(classified.implementation, classified.members, address)
        for name, member in classified.members.items():
            if member.kind != "state":
                continue

            def publish(ops: Sequence[Op], sequence: int, context: Context, member_name: str = name) -> None:
                if not instance.active:
                    return
                update: ServiceStateUpdate = {
                    "type": "state", "member": member_name, "sequence": sequence,
                    "ops": [cast(Op, list(op) if op[0] == "r" else [op[0], list(op[1]), *op[2:]]) for op in ops],
                }
                if address is not None:
                    update["instance"] = address
                self._emit(registration, update, context)

            instance.remove_member_listeners.append(member.state.subscribe(publish))
        return instance

    def _assert_singleton_shape(
        self, registration: _ServiceRegistration, replacement: dict[str, _ServiceMemberKind],
    ) -> None:
        current = registration.singleton_shape
        if current is None or current == replacement:
            return
        raise RemoteServiceError(
            "service_member_mismatch",
            f"Remote service {registration.service_id} replacement must preserve its member shape",
        )

    def _resolve_instance(
        self, registration: _ServiceRegistration, address: ServiceInstanceAddress | None,
    ) -> _ProviderInstance:
        if address is UNDEFINED:
            address = None
        if registration.mode == "singleton":
            if address is not None:
                raise RemoteServiceError("service_mode_mismatch", f"Remote service {registration.service_id} is singleton")
            if registration.singleton is None:
                raise RemoteServiceError("service_not_found", f"Remote service {registration.service_id} has no provider")
            return registration.singleton
        if address is None:
            raise RemoteServiceError("service_mode_mismatch", f"Remote service {registration.service_id} is keyed")
        instance = registration.instances.get(address["key"])
        if instance is None:
            raise RemoteServiceError(
                "service_instance_not_found",
                f"Remote service {registration.service_id} has no instance {address['key']}",
            )
        if instance.address is None or instance.address["generation"] != address["generation"]:
            raise RemoteServiceError(
                "service_stale_instance", f"Remote service {registration.service_id} instance {address['key']} is stale",
            )
        return instance

    def _publish_pending(self, registration: _ServiceRegistration) -> None:
        context = service_delivery_context()
        if registration.mode == "singleton":
            instances = [] if registration.singleton is None else [registration.singleton]
            for instance in instances:
                for member in instance.members.values():
                    if member.kind == "state":
                        member.state.publish(context)
            return
        visited: set[_ProviderInstance] = set()
        while True:
            instance = next((candidate for candidate in registration.instances.values() if candidate not in visited), None)
            if instance is None:
                break
            visited.add(instance)
            for member in instance.members.values():
                if member.kind == "state":
                    member.state.publish(context)

    def _snapshot(self, registration: _ServiceRegistration) -> ServiceSubscriptionSnapshot:
        if registration.mode == "singleton":
            instances = [] if registration.singleton is None else [self._snapshot_instance(registration.singleton)]
        else:
            collator = Collator.createInstance()
            ordered = sorted(
                registration.instances.values(),
                key=lambda instance: collator.getSortKey(cast(ServiceInstanceAddress, instance.address)["key"]),
            )
            instances = [self._snapshot_instance(instance) for instance in ordered]
        return {"serviceId": registration.service_id, "mode": registration.mode, "instances": instances}

    def _snapshot_instance(self, instance: _ProviderInstance) -> ServiceInstanceSnapshot:
        members: list[ServiceMemberSnapshot] = []
        for name, member in instance.members.items():
            if member.kind == "method":
                members.append({"name": name, "kind": "method"})
            else:
                members.append({
                    "name": name, "kind": "state", "sequence": member.state.sequence,
                    "ops": [cast(Op, ["r", member.state.value])],
                })
        snapshot: ServiceInstanceSnapshot = {"members": members}
        if instance.address is not None:
            snapshot["instance"] = instance.address
        return snapshot

    def _emit(
        self, registration: _ServiceRegistration, update: ServiceProviderUpdate,
        context: Context | None = None,
    ) -> None:
        if not registration.subscribers:
            return
        delivery_context = context if context is not None else service_delivery_context()
        errors: list[BaseException] = []
        # JavaScript Set iteration observes subscriptions added by earlier listeners.
        visited: set[_ProviderSubscriber] = set()
        while True:
            subscriber = next((candidate for candidate in registration.subscribers if candidate not in visited), None)
            if subscriber is None:
                break
            visited.add(subscriber)
            if subscriber.closed:
                continue
            if not subscriber.active:
                subscriber.buffer.append((update, delivery_context))
                continue
            try:
                subscriber.listener(update, delivery_context)
            except BaseException as error:
                errors.append(error)
        _throw_collected_errors(errors, f"Failed to publish remote service {registration.service_id} update")

    def _assert_remotable[T](self, service: Service[T]) -> None:
        if service.local:
            raise RemoteServiceError("service_not_allowed", f"Service {service.id} is process-local")

    def _assert_allowed(self, service_id: str) -> None:
        if service_id not in self._registrations:
            raise RemoteServiceError("service_not_allowed", f"Remote service {service_id} is not allowlisted")

    def _assert_active(self) -> None:
        if self._disposed:
            raise RuntimeError("Remote service provider is disposed")


class _RemoteServiceEndpoint:
    def __init__(self, provider: RemoteServiceProvider) -> None:
        self._provider = provider
        self._subscriptions: dict[str, _ProviderSubscription] = {}
        self._disposed = False
        self._publications: set[asyncio.Task[None]] = set()

    async def invoke(
        self, call: ServiceCall, publish: ServiceUpdatePublisher, context: Context,
    ) -> JsonValue | Undefined:
        if self._disposed:
            raise RuntimeError("Remote service endpoint is disposed")
        control = decode_service_control_call(call)
        if control is not None and control["type"] == "catalogue":
            return [{"serviceId": entry["serviceId"], "mode": entry["mode"]} for entry in self._provider.catalogue]
        if control is not None and control["type"] == "subscribe":
            subscription_id = control["subscriptionId"]
            if subscription_id in self._subscriptions:
                raise RuntimeError("Service subscription ID is already active")

            def on_update(update: ServiceProviderUpdate, update_context: Context) -> None:
                result = publish(subscription_id, update, update_context)
                if inspect.isawaitable(result):
                    async def ignore_rejection() -> None:
                        try:
                            await result
                        except BaseException:
                            pass

                    pending = asyncio.get_running_loop().create_task(ignore_rejection())
                    self._publications.add(pending)
                    pending.add_done_callback(self._publications.discard)

            subscription = self._provider.subscribe(control["serviceId"], control["mode"], on_update)
            self._subscriptions[subscription_id] = subscription
            subscription.activate()
            return cast(JsonValue, subscription.snapshot)
        if control is not None and control["type"] == "unsubscribe":
            subscription_id = control["subscriptionId"]
            subscription = self._subscriptions.get(subscription_id)
            if subscription is None:
                raise RuntimeError("Service subscription was not found")
            subscription.close()
            del self._subscriptions[subscription_id]
            return UNDEFINED
        return await self._provider.invoke(call, context)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for subscription in self._subscriptions.values():
            subscription.close()
        self._subscriptions.clear()


def create_remote_service_endpoint(provider: RemoteServiceProvider) -> RemoteServiceEndpoint:
    return _RemoteServiceEndpoint(provider)


def validate_remote_service_implementation(service_id: str, implementation: object) -> None:
    _classify_remote_service_implementation(service_id, implementation)


def _classify_remote_service_implementation(service_id: str, implementation: object) -> _ClassifiedImplementation:
    if (implementation is None or isinstance(implementation, (str, bytes, bool, int, float, complex, list, tuple))
            or callable(implementation)):
        raise TypeError(f"Remote service {service_id} implementation must be an object")
    if isinstance(implementation, Mapping):
        properties = cast(Mapping[str, object], implementation)
    else:
        try:
            properties = vars(implementation)
        except TypeError:
            properties = {}
    members: dict[str, _InstanceMember] = {}
    # Array.sort compares UTF-16 code units, independently of the current locale.
    for name in sorted(properties, key=lambda value: value.encode("utf-16-be", errors="surrogatepass")):
        value = properties[name]
        if isinstance(value, property):
            raise TypeError(f"Remote service member {service_id}.{name} must be a data property")
        if callable(value):
            members[name] = _MethodMember(value)
            continue
        state = get_replicated_state_internals(value)
        if state is not None:
            members[name] = _StateMember(state)
            continue
        raise TypeError(f"Remote service member {service_id}.{name} is not remotely exposable")
    if not members:
        raise TypeError(f"Remote service {service_id} has no members")
    return _ClassifiedImplementation(implementation, members)


def _throw_collected_errors(errors: Sequence[BaseException], message: str) -> None:
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise BaseExceptionGroup(message, list(errors))
