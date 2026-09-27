"""Remote service handles and binding lifecycle from ``services/consumer.ts``."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ..context import BACKGROUND_CONTEXT, Context, await_with_context
from ..delta import UNDEFINED, Op, Undefined
from ..types import (
    JsonValue,
    RemoteServiceBindingOptions,
    RemoteServiceTransport,
    ReplicatedStateDelivery,
    Service,
    ServiceCall,
    ServiceInstanceAddress,
    ServiceInstanceSnapshot,
    ServiceMemberSnapshot,
    ServiceMode,
    ServiceProviderUpdate,
    ServiceSubscription,
)
from .errors import RemoteServiceError
from .handle import ServiceSlot
from .instances import InstanceDirectory
from .state import ReplicatedStateReplica, service_delivery_context

__all__ = ["RemoteServiceBindingImpl"]

type ErrorReporter = Callable[[Exception], None]
type ServiceMemberKind = Literal["method", "state"]
type StateListener = Callable[[JsonValue, Context, ReplicatedStateDelivery], None]


class _RemoteServiceMember:
    def __init__(self, slot: _MemberSlot) -> None:
        self._slot = slot

    @property
    def value(self) -> JsonValue | Undefined:
        self._slot.assert_access()
        self._slot.expect("state")
        return self._slot.state.value

    def subscribe(self, listener: StateListener) -> Callable[[], None]:
        self._slot.assert_access()
        if not callable(listener):
            raise TypeError("Replicated state subscription listener must be a function")
        self._slot.expect("state")
        return self._slot.state.subscribe(listener)

    def __call__(self, *args: object) -> Awaitable[JsonValue | Undefined]:
        return self._slot.call(args)

    def __getattr__(self, property: str) -> Undefined:
        if property.startswith("__") and property.endswith("__"):
            raise AttributeError(property)
        return UNDEFINED

    def __getitem__(self, property: str) -> object:
        if property == "value":
            return self.value
        if property == "subscribe":
            return self.subscribe
        return UNDEFINED


class _MemberSlot:
    def __init__(
        self,
        service_id: str,
        member: str,
        invoke: Callable[[Sequence[JsonValue], Context], Awaitable[JsonValue | Undefined]],
        is_active: Callable[[], bool],
        assert_access: Callable[[], None],
        report_error: ErrorReporter,
    ) -> None:
        self._service_id = service_id
        self._member = member
        self._invoke = invoke
        self._is_active = is_active
        self.assert_access = assert_access
        self.state = ReplicatedStateReplica[JsonValue](report_error)
        self.value = _RemoteServiceMember(self)
        self._kind: ServiceMemberKind | None = None
        self._expected_kind: ServiceMemberKind | None = None

    def set_description(self, kind: ServiceMemberKind) -> None:
        if self._kind is not None and self._kind != kind:
            raise RuntimeError(f"Remote service member {self._service_id}.{self._member} changed kind")
        self._kind = kind
        if self._expected_kind is not None and self._expected_kind != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._service_id}.{self._member} is {kind}, not {self._expected_kind}",
            )

    def hydrate(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        self.set_description("state")
        self.state.hydrate(sequence, ops, context)

    def update(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        self.set_description("state")
        self.state.update(sequence, ops, context)

    def clear(self) -> None:
        self.state.clear()

    def expect(self, kind: ServiceMemberKind) -> None:
        if self._expected_kind is not None and self._expected_kind != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._service_id}.{self._member} was used as two different kinds",
            )
        self._expected_kind = kind
        if self._kind is not None and self._kind != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._service_id}.{self._member} is {self._kind}, not {kind}",
            )

    def call(self, args: Sequence[object]) -> Awaitable[JsonValue | Undefined]:
        self.assert_access()
        self.expect("method")
        if not self._is_active():
            error = RemoteServiceError(
                "service_stale_instance", f"Remote service {self._service_id} binding is closed"
            )
            return _rejected(error)
        context = args[-1] if args else None
        if context is None or not callable(getattr(context, "value", None)) or not callable(getattr(context, "to_string", None)):
            error = RemoteServiceError(
                "service_invalid_value",
                f"Remote service method {self._service_id}.{self._member} requires a trailing Context",
            )
            return _rejected(error)
        return self._invoke(cast(Sequence[JsonValue], args[:-1]), cast(Context, context))


class _ServiceProxy:
    def __init__(self, facade: _ServiceFacade) -> None:
        self._facade = facade

    def __getattr__(self, member: str) -> _RemoteServiceMember:
        if member.startswith("__") and member.endswith("__"):
            raise AttributeError(member)
        return self._facade.slot(member).value

    def __getitem__(self, member: str) -> _RemoteServiceMember:
        return self._facade.slot(member).value


class _ServiceFacade:
    def __init__(
        self,
        service_id: str,
        address: ServiceInstanceAddress | None,
        transport: RemoteServiceTransport,
        is_active: Callable[[], bool],
        assert_access: Callable[[], None],
        report_error: ErrorReporter,
    ) -> None:
        self._service_id = service_id
        self._address = address
        self._transport = transport
        self._is_active = is_active
        self._assert_access = assert_access
        self._report_error = report_error
        self._slots: dict[str, _MemberSlot] = {}
        self._descriptions: dict[str, ServiceMemberKind] = {}
        self.proxy = _ServiceProxy(self)

    def install(self, snapshot: ServiceInstanceSnapshot, context: Context) -> None:
        if not _same_address(snapshot.get("instance"), self._address):
            raise RuntimeError("Remote service snapshot has the wrong address")
        members: dict[str, ServiceMemberSnapshot] = {}
        for member in snapshot["members"]:
            if len(member["name"]) == 0 or member["name"] in members:
                raise RuntimeError("Remote service has invalid member descriptions")
            members[member["name"]] = member
        for name in self._slots:
            if name not in members:
                raise RemoteServiceError(
                    "service_member_not_found", f"Unknown remote service member {self._service_id}.{name}"
                )
        self._descriptions.clear()
        for name, member in members.items():
            self._descriptions[name] = member["kind"]
        for member in members.values():
            slot = self._slots.get(member["name"])
            if member["kind"] == "state":
                (slot if slot is not None else self.slot(member["name"])).hydrate(
                    member["sequence"], member["ops"], context
                )
            elif slot is not None:
                slot.set_description(member["kind"])

    def update(self, member: str, sequence: int, ops: Sequence[Op], context: Context) -> None:
        if self._descriptions.get(member) != "state":
            raise RuntimeError(f"Remote service update targets non-state member {self._service_id}.{member}")
        self.slot(member).update(sequence, ops, context)

    def clear(self) -> None:
        for slot in self._slots.values():
            slot.clear()

    def slot(self, member: str) -> _MemberSlot:
        slot = self._slots.get(member)
        if slot is not None:
            return slot

        def invoke(args: Sequence[JsonValue], context: Context) -> Awaitable[JsonValue | Undefined]:
            call: ServiceCall = {"serviceId": self._service_id, "member": member, "args": list(args)}
            if self._address is not None:
                call["instance"] = self._address
            return self._transport.invoke(call, context)

        slot = _MemberSlot(
            self._service_id, member, invoke, self._is_active, self._assert_access, self._report_error
        )
        kind = self._descriptions.get(member)
        if kind is not None:
            slot.set_description(kind)
        self._slots[member] = slot
        return slot


@dataclass
class _SingletonBinding:
    facade: _ServiceFacade
    subscription: ServiceSubscription | None = None
    starting: asyncio.Task[None] | None = None
    active: bool = True
    revision: int = 0


@dataclass
class _KeyedInstance:
    key: str
    generation: int
    service: object
    facade: _ServiceFacade
    deactivate: Callable[[], None]


class _KeyedBinding[T]:
    def __init__(
        self,
        service: Service[T],
        transport: RemoteServiceTransport,
        report_error: ErrorReporter,
        assert_access: Callable[[], None],
        on_empty: Callable[[], None],
        bound: bool,
    ) -> None:
        self._service = service
        self._transport = transport
        self._report_error = report_error
        self._assert_access = assert_access
        self._on_empty = on_empty
        self._bound = bound
        self._instances = InstanceDirectory[_KeyedInstance](ready=False, on_error=report_error)
        self._subscription: ServiceSubscription | None = None
        self._starting: asyncio.Task[None] | None = None
        self._closed = False
        self._revision = 0

    def observe(self, handler: Callable[[T, Context], None | Awaitable[None]]) -> Callable[[], None]:
        if self._closed:
            raise RuntimeError("Remote keyed service binding is closed")
        stopped = False

        def observe_instance(service: object, context: Context) -> None | Awaitable[None]:
            slot = ServiceSlot(self._service.id, True)
            slot.bind(service)

            def assert_access() -> None:
                self._assert_access()
                if stopped or (context.abort_signal is not None and context.abort_signal.aborted):
                    raise RemoteServiceError(
                        "service_stale_instance", f"Remote service {self._service.id} observation is closed"
                    )

            return handler(cast(T, slot.view(assert_access)), context)

        stop = self._instances.observe(observe_instance)
        if self._bound and self._starting is None:
            revision = self._revision
            self._starting = asyncio.get_running_loop().create_task(self._start(revision))

            def report_starting_error(starting: asyncio.Task[None]) -> None:
                try:
                    starting.result()
                except BaseException as error:
                    if not self._closed and self._revision == revision and self._bound:
                        self._report_error(_to_error(error))

            self._starting.add_done_callback(report_starting_error)

        def close_observation() -> None:
            nonlocal stopped
            if stopped:
                return
            stopped = True
            stop()
            if self._instances.observer_count == 0:
                self._on_empty()

        return close_observation

    async def rebind(self, bound: bool, context: Context) -> None:
        if self._closed:
            return
        self._bound = bound
        self._revision += 1
        revision = self._revision
        await self._reset(context, False)
        if self._closed or self._revision != revision or self._bound != bound:
            return
        if bound and self._instances.observer_count > 0:
            self._starting = asyncio.get_running_loop().create_task(self._start(revision))
            await asyncio.shield(self._starting)

    async def ready(self) -> None:
        if self._starting is not None:
            await asyncio.shield(self._starting)

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        self._revision += 1
        await self._reset(context, True)
        self._instances.dispose()

    async def _reset(self, context: Context, wait_for_starting: bool) -> None:
        self._instances.reset()
        starting = self._starting
        self._starting = None
        subscription = self._subscription
        self._subscription = None
        pending: list[Awaitable[object]] = []
        if wait_for_starting and starting is not None:
            pending.append(_ignore_rejection(starting))
        if subscription is not None:
            pending.append(_close_subscription(subscription, context))
        await asyncio.gather(*pending)

    async def _start(self, revision: int) -> None:
        def update(update: ServiceProviderUpdate, context: Context) -> None:
            if self._revision == revision:
                self._update(update, context)

        subscription = await self._transport.subscribe(self._service.id, "keyed", update, BACKGROUND_CONTEXT)
        if self._closed or not self._bound or self._revision != revision:
            await _close_subscription(subscription, BACKGROUND_CONTEXT)
            return
        self._subscription = subscription
        snapshot = subscription.snapshot
        if snapshot["mode"] != "keyed" or snapshot["serviceId"] != self._service.id:
            raise RuntimeError(f"Remote service {self._service.id} returned the wrong keyed snapshot")
        for instance in snapshot["instances"]:
            self._spawn(instance, service_delivery_context())
        subscription.activate()
        self._instances.ready()

    def _update(self, update: ServiceProviderUpdate, context: Context) -> None:
        if self._closed:
            return
        try:
            if update["type"] in ("unavailable", "replaced"):
                raise RuntimeError("Keyed service received a singleton lifecycle update")
            if update["type"] == "spawned":
                self._spawn(update["instance"], context)
            elif update["type"] == "closed":
                address = update["instance"]
                instance = self._instances.get(address["key"])
                if instance is not None and instance.generation == address["generation"]:
                    self._instances.remove(instance)
            elif update["type"] == "state":
                address = update.get("instance")
                if address is None or address is UNDEFINED:
                    raise RuntimeError("Keyed state update has no instance address")
                instance = self._instances.get(address["key"])
                if instance is None or instance.generation != address["generation"]:
                    return
                instance.facade.update(update["member"], update["sequence"], update["ops"], context)
        except Exception as error:
            self._report_error(error)

    def _spawn(self, snapshot: ServiceInstanceSnapshot, context: Context) -> None:
        address = snapshot.get("instance")
        if address is None or address is UNDEFINED:
            raise RuntimeError("Keyed service instance snapshot has no address")
        active = True
        facade = _ServiceFacade(
            self._service.id, address, self._transport,
            lambda: active and not self._closed, self._assert_access, self._report_error,
        )
        facade.install(snapshot, context)

        def deactivate() -> None:
            nonlocal active
            active = False
            facade.clear()

        self._instances.replace(_KeyedInstance(
            key=address["key"], generation=address["generation"],
            service=facade.proxy, facade=facade, deactivate=deactivate,
        ))


class RemoteServiceBindingImpl:
    def __init__(self, options: RemoteServiceBindingOptions) -> None:
        self._transport = options.transport
        ids = [service.id for service in options.services]
        if len(set(ids)) != len(ids):
            raise TypeError("Remote service binding has duplicate service IDs")
        self._allowlist = set(ids)
        self._report_error = options.on_error if options.on_error is not None else lambda error: None
        self._assert_access = options.assert_access if options.assert_access is not None else lambda: None
        self._bound = options.bound
        self._modes: dict[str, ServiceMode] = {}
        self._singletons: dict[str, _SingletonBinding] = {}
        self._keyed: dict[str, _KeyedBinding[object]] = {}
        self._readiness_revision = 0
        self._binding_transition: asyncio.Task[None] | None = None
        self._disposed = False
        self._pending_closes: set[asyncio.Task[None]] = set()

    def use[T](self, service: Service[T]) -> T:
        self._assert_remotable(service)
        self._assert_available(service.id, "singleton")
        existing = self._singletons.get(service.id)
        if existing is not None:
            return cast(T, existing.facade.proxy)
        facade = _ServiceFacade(
            service.id, None, self._transport,
            lambda: binding.active and not self._disposed and self._bound,
            self._assert_handle_access, self._report_error,
        )
        binding = _SingletonBinding(facade)
        self._singletons[service.id] = binding
        self._readiness_revision += 1
        if self._bound:
            revision = binding.revision
            binding.starting = asyncio.get_running_loop().create_task(self._start_singleton(service.id, binding, revision))

            def report_starting_error(starting: asyncio.Task[None]) -> None:
                try:
                    starting.result()
                except BaseException as error:
                    if binding.active and binding.revision == revision and not self._disposed and self._bound:
                        self._report_error(_to_error(error))

            binding.starting.add_done_callback(report_starting_error)
        return cast(T, binding.facade.proxy)

    def observe[T](
        self, service: Service[T], handler: Callable[[T, Context], None | Awaitable[None]],
    ) -> Callable[[], None]:
        self._assert_remotable(service)
        self._assert_available(service.id, "keyed")
        binding = cast(_KeyedBinding[T] | None, self._keyed.get(service.id))
        if binding is None:
            def on_empty() -> None:
                if self._keyed.get(service.id) is not binding:
                    return
                del self._keyed[service.id]
                self._readiness_revision += 1
                closing = asyncio.get_running_loop().create_task(cast(_KeyedBinding[T], binding).close(BACKGROUND_CONTEXT))
                self._pending_closes.add(closing)

                def report_close_error(completion: asyncio.Task[None]) -> None:
                    self._pending_closes.discard(completion)
                    try:
                        completion.result()
                    except BaseException as error:
                        self._report_error(_to_error(error))

                closing.add_done_callback(report_close_error)

            binding = _KeyedBinding(
                service, self._transport, self._report_error, self._assert_handle_access,
                on_empty, self._bound,
            )
            self._keyed[service.id] = cast(_KeyedBinding[object], binding)
            self._readiness_revision += 1
        return binding.observe(handler)

    async def ready(self, context: Context) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        while True:
            revision = self._readiness_revision
            starts: list[Awaitable[None]] = []
            if self._binding_transition is not None:
                starts.append(self._binding_transition)
            starts.extend(binding.starting for binding in self._singletons.values() if binding.starting is not None)
            starts.extend(binding.ready() for binding in self._keyed.values())
            await await_with_context(asyncio.gather(*starts), context)
            if self._disposed:
                raise RuntimeError("Remote service binding is disposed")
            if revision == self._readiness_revision:
                return

    async def rebind(self, bound: bool, context: Context) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        self._bound = bound
        self._readiness_revision += 1
        transitions: list[asyncio.Task[None]] = []
        for service_id, binding in self._singletons.items():
            binding.revision += 1
            binding.facade.clear()
            subscription = binding.subscription
            binding.subscription = None
            binding.starting = asyncio.get_running_loop().create_task(self._restart_singleton(
                service_id, binding, binding.revision, bound, subscription, context,
            ))
            transitions.append(binding.starting)
        for keyed in self._keyed.values():
            transitions.append(asyncio.get_running_loop().create_task(keyed.rebind(bound, context)))

        async def complete() -> None:
            results = await asyncio.gather(*transitions, return_exceptions=True)
            errors = [_to_error(result) for result in results if isinstance(result, BaseException)]
            if errors:
                raise ExceptionGroup("Failed to rebind services", errors)

        self._binding_transition = asyncio.get_running_loop().create_task(complete())
        await asyncio.shield(self._binding_transition)

    async def dispose(self, context: Context) -> None:
        if self._disposed:
            return
        self._disposed = True
        closes: list[Awaitable[None]] = []
        for binding in self._singletons.values():
            binding.active = False
            binding.facade.clear()
            if binding.starting is not None:
                closes.append(_ignore_rejection(binding.starting))
            if binding.subscription is not None:
                closes.append(_close_subscription(binding.subscription, context))
        for keyed in self._keyed.values():
            closes.append(keyed.close(context))
        self._singletons.clear()
        self._keyed.clear()
        results = await asyncio.gather(*closes, return_exceptions=True)
        errors = [_to_error(result) for result in results if isinstance(result, BaseException)]
        if errors:
            raise ExceptionGroup("Failed to dispose services", errors)

    async def _restart_singleton(
        self, service_id: str, binding: _SingletonBinding, revision: int, bound: bool,
        subscription: ServiceSubscription | None, context: Context,
    ) -> None:
        if subscription is not None:
            await _close_subscription(subscription, context)
        if bound:
            await self._start_singleton(service_id, binding, revision)

    async def _start_singleton(self, service_id: str, binding: _SingletonBinding, revision: int) -> None:
        def update(update: ServiceProviderUpdate, context: Context) -> None:
            if not binding.active or binding.revision != revision:
                return
            try:
                if update["type"] == "unavailable":
                    binding.facade.clear()
                elif update["type"] == "replaced":
                    if update["snapshot"].get("instance", UNDEFINED) is not UNDEFINED:
                        raise RuntimeError("Singleton replacement has an instance address")
                    binding.facade.install(update["snapshot"], context)
                elif update["type"] == "state" and update.get("instance", UNDEFINED) is UNDEFINED:
                    binding.facade.update(update["member"], update["sequence"], update["ops"], context)
            except Exception as error:
                self._report_error(error)

        subscription = await self._transport.subscribe(service_id, "singleton", update, BACKGROUND_CONTEXT)
        if not binding.active or self._disposed or not self._bound or binding.revision != revision:
            await _close_subscription(subscription, BACKGROUND_CONTEXT)
            return
        binding.subscription = subscription
        snapshot = subscription.snapshot
        if snapshot["mode"] != "singleton" or snapshot["serviceId"] != service_id or len(snapshot["instances"]) != 1:
            raise RuntimeError(f"Remote service {service_id} returned an invalid singleton snapshot")
        binding.facade.install(snapshot["instances"][0], service_delivery_context())
        subscription.activate()

    def _assert_handle_access(self) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        self._assert_access()

    def _assert_remotable[T](self, service: Service[T]) -> None:
        if service.local:
            raise RemoteServiceError("service_not_allowed", f"Service {service.id} is process-local")

    def _assert_available(self, service_id: str, mode: ServiceMode) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        if service_id not in self._allowlist:
            raise RemoteServiceError("service_not_allowed", f"Remote service {service_id} is not allowlisted")
        existing = self._modes.get(service_id)
        if existing is not None and existing != mode:
            raise RemoteServiceError(
                "service_mode_mismatch", f"Remote service {service_id} is already used as {existing}"
            )
        self._modes[service_id] = mode


async def _rejected(error: Exception) -> JsonValue | Undefined:
    raise error


async def _ignore_rejection(operation: Awaitable[object]) -> None:
    try:
        await asyncio.shield(operation)
    except BaseException:
        pass


async def _close_subscription(subscription: ServiceSubscription, context: Context) -> None:
    result = subscription.close(context)
    if inspect.isawaitable(result):
        await result


def _same_address(left: ServiceInstanceAddress | None, right: ServiceInstanceAddress | None) -> bool:
    if left is UNDEFINED:
        left = None
    if right is UNDEFINED:
        right = None
    if left is None or right is None:
        return left is right
    return left["key"] == right["key"] and left["generation"] == right["generation"]


def _to_error(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else RuntimeError(str(error))
