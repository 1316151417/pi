"""Facet dependency and lifecycle kernel, ported from facets/host.ts."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from ..context import BACKGROUND_CONTEXT, Context
from ..services.consumer import RemoteServiceBindingImpl
from ..services.handle import ServiceSlot
from ..services.instances import InstanceDirectory
from ..services.loopback import create_loopback_service_transport
from ..services.provider import RemoteServiceProvider, ServiceProviderDefinition, validate_remote_service_implementation
from ..services.state import MutableReplicatedStateImpl
from ..types import (
    Disposal, Facet, FacetOptions, RemoteServiceBindingOptions, RemoteServiceSource,
    RemoteServiceSourceOptions, RemoteServices, Service, ServiceIdentity, ServiceMode,
)

type LifecycleState = Literal["setting_up", "prepared", "active", "disposing", "dead"]
type GenerationPhase = Literal[
    "setup", "assembling", "connecting", "activating", "active", "reloading", "disposing", "dead",
]
type Observer = Callable[[object, Context], None | Awaitable[None]]


@dataclass(frozen=True)
class FacetServiceReference:
    service_id: str
    service: Service[object]
    mode: ServiceMode


@dataclass(frozen=True)
class ExternalService:
    service: Service[object]
    mode: ServiceMode
    source: RemoteServiceSource


class FacetLifecycle:
    def __init__(self, id: str) -> None:
        self.id = id
        self._effects: list[Disposal] = []
        self._observations: list[Callable[[], Disposal]] = []
        self._activate: list[Disposal] = []
        self._state: LifecycleState = "setting_up"
        self._service_access = False

    def assert_setting_up(self, operation: str) -> None:
        if self._state != "setting_up":
            raise RuntimeError(f"Facet {self.id} can {operation} only during setup")

    def assert_running(self, operation: str) -> None:
        if self._state not in ("setting_up", "active"):
            raise RuntimeError(f"Facet {self.id} cannot {operation} while {self._state}")

    def assert_active(self, operation: str) -> None:
        if self._state != "active":
            raise RuntimeError(f"Facet {self.id} can {operation} only while active")

    def assert_service_access(self) -> None:
        if not self._service_access:
            raise RuntimeError(f"Facet {self.id} service handles cannot be used while {self._state}")

    def revoke(self) -> None:
        self._service_access = False

    def own(self, disposal: Disposal) -> None:
        self.assert_running("own resources")
        self._effects.append(disposal)

    def observe(self, start: Callable[[], Disposal]) -> None:
        self.assert_setting_up("observe services")
        self._observations.append(start)

    def on_activate(self, callback: Disposal) -> None:
        self.assert_setting_up("register activation callbacks")
        self._activate.append(callback)

    def prepared(self) -> None:
        self.assert_setting_up("finish setup")
        self._state = "prepared"

    async def activate(self) -> None:
        if self._state != "prepared":
            raise RuntimeError(f"Facet {self.id} is not prepared")
        self._state = "active"
        self._service_access = True
        for start in self._observations:
            self._effects.append(start())
        for callback in self._activate:
            await _await_result(callback())

    async def dispose(self) -> None:
        if self._state == "dead":
            return
        self._state = "disposing"
        effects, self._effects = self._effects, []
        errors: list[BaseException] = []
        for effect in reversed(effects):
            try:
                await _await_result(effect())
            except (Exception, asyncio.CancelledError) as error:
                errors.append(error)
        self._observations.clear()
        self._activate.clear()
        self._service_access = False
        self._state = "dead"
        _raise_errors(errors, f"Failed to dispose facet {self.id}")


@dataclass
class FacetRuntime:
    facet_id: str
    lifecycle: FacetLifecycle
    requires: list[FacetServiceReference] = field(default_factory=list)
    provides: list[FacetServiceReference] = field(default_factory=list)
    provisions: list[FacetProvision] = field(default_factory=list)
    singleton_views: dict[str, object] = field(default_factory=dict)


class KeyedServiceSource(Protocol):
    def observe(self, service: Service[object], handler: Observer) -> Callable[[], None]: ...


@dataclass
class _LocalInstance:
    key: str
    generation: int
    service: object

    def deactivate(self) -> None:
        pass


@dataclass
class _LocalKeyedRegistration:
    directory: InstanceDirectory[_LocalInstance]
    generations: dict[str, int] = field(default_factory=dict)


class LocalKeyedServiceRegistry:
    def __init__(self, services: Sequence[ServiceIdentity], on_error: Callable[[Exception], None]) -> None:
        ids = [service.id for service in services]
        if len(set(ids)) != len(ids):
            raise TypeError("Local keyed service registry has duplicate IDs")
        self._registrations = {
            service_id: _LocalKeyedRegistration(InstanceDirectory(ready=True, on_error=on_error))
            for service_id in ids
        }
        self._disposed = False

    def spawn(self, service: Service[object], key: str, implementation: object) -> Callable[[], None]:
        self._assert_active()
        if not key:
            raise TypeError("Local service instance key must not be empty")
        if not _is_implementation(implementation):
            raise TypeError(f"Local service {service.id} implementation must be an object")
        registration = self._registration(service.id)
        if registration.directory.get(key) is not None:
            raise RuntimeError(f"Local service {service.id} already has a live instance with key {key}")
        generation = registration.generations.get(key, 0) + 1
        registration.generations[key] = generation
        instance = _LocalInstance(key, generation, implementation)
        registration.directory.insert(instance)
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            registration.directory.remove(instance)

        return close

    def observe(self, service: Service[object], handler: Observer) -> Callable[[], None]:
        self._assert_active()
        return self._registration(service.id).directory.observe(handler)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for registration in self._registrations.values():
            registration.directory.dispose()
        self._registrations.clear()

    def _registration(self, service_id: str) -> _LocalKeyedRegistration:
        registration = self._registrations.get(service_id)
        if registration is None:
            raise RuntimeError(f"Local keyed service {service_id} is not registered")
        return registration

    def _assert_active(self) -> None:
        if self._disposed:
            raise RuntimeError("Local keyed service registry is disposed")


class HostServiceSlots:
    def __init__(self) -> None:
        self._singletons: dict[str, ServiceSlot] = {}
        self._keyed_sources: dict[str, KeyedServiceSource] = {}

    def get_singleton[T](self, service: Service[T], assert_access: Callable[[], None]) -> T:
        slot = self._singletons.get(service.id)
        if slot is None:
            slot = ServiceSlot(service.id, not service.local)
            self._singletons[service.id] = slot
        return cast(T, slot.view(assert_access))

    def has_singleton(self, service_id: str) -> bool:
        return service_id in self._singletons

    def observe(self, service: Service[object], assert_access: Callable[[], None], handler: Observer) -> Disposal:
        source = self._keyed_sources.get(service.id)
        if source is None:
            raise RuntimeError(f"Service {service.id} is disconnected")
        stopped = False

        def observe_target(target: object, context: Context) -> None | Awaitable[None]:
            slot = ServiceSlot(service.id, not service.local)
            slot.bind(target)

            def assert_live() -> None:
                assert_access()
                if stopped or (context.abort_signal is not None and context.abort_signal.aborted):
                    raise RuntimeError(f"Keyed service {service.id} observation is closed")

            return handler(slot.view(assert_live), context)

        stop = source.observe(service, observe_target)

        def close() -> None:
            nonlocal stopped
            if stopped:
                return
            stopped = True
            stop()

        return close

    def bind_singleton(self, service_id: str, target: object) -> None:
        slot = self._singletons.get(service_id)
        if slot is not None:
            slot.bind(target)

    def bind_keyed(self, service_id: str, services: KeyedServiceSource) -> None:
        self._keyed_sources[service_id] = services

    def dispose(self) -> None:
        for slot in self._singletons.values():
            slot.unbind()
        self._singletons.clear()
        self._keyed_sources.clear()


@dataclass(eq=False)
class _StagedInstance:
    key: str
    implementation: object
    release: Callable[[], None] | None = None


class StagedServiceSpawner:
    def __init__(self, lifecycle: FacetLifecycle, validate: Callable[[str, object], None]) -> None:
        self._lifecycle = lifecycle
        self._validate = validate
        self._instances: dict[str, _StagedInstance] = {}
        self._installer: Callable[[str, object], Callable[[], None]] | None = None

    def connect(self, installer: Callable[[str, object], Callable[[], None]]) -> None:
        if self._installer is not None:
            raise RuntimeError("Facet service provider is already connected")
        self._installer = installer
        visited: set[_StagedInstance] = set()
        while True:
            instance = next((candidate for candidate in self._instances.values() if candidate not in visited), None)
            if instance is None:
                break
            visited.add(instance)
            instance.release = installer(instance.key, instance.implementation)

    def spawn(self, key: str, implementation: object) -> Callable[[], None]:
        self._lifecycle.assert_active("spawn service instances")
        self._validate(key, implementation)
        if key in self._instances:
            raise RuntimeError(f"Facet service already has a live instance with key {key}")
        instance = _StagedInstance(key, implementation)
        self._instances[key] = instance
        if self._installer is not None:
            instance.release = self._installer(key, implementation)

        def close() -> None:
            if self._instances.get(key) is not instance:
                return
            del self._instances[key]
            if instance.release is not None:
                instance.release()

        self._lifecycle.own(close)
        return close


@dataclass(frozen=True)
class _SingletonProvision:
    service: Service[object]
    implementation: object
    kind: Literal["singleton"] = field(default="singleton", init=False)


@dataclass(frozen=True)
class _KeyedProvision:
    service: Service[object]
    spawner: StagedServiceSpawner
    kind: Literal["keyed"] = field(default="keyed", init=False)

    def connect_local(self, registry: LocalKeyedServiceRegistry) -> None:
        self.spawner.connect(lambda key, implementation: registry.spawn(self.service, key, implementation))

    def connect_remote(self, provider: RemoteServiceProvider) -> None:
        self.spawner.connect(lambda key, implementation: provider.spawn(self.service, key, implementation))


type FacetProvision = _SingletonProvision | _KeyedProvision


class _Environment:
    def __init__(self, runtime: FacetRuntime, slots: HostServiceSlots) -> None:
        self._runtime = runtime
        self._slots = slots

    def provide(self, service: Service[object], implementation: object) -> None:
        self._runtime.lifecycle.assert_setting_up("provide services")
        if not _is_implementation(implementation):
            raise TypeError(f"Service {service.id} implementation must be an object")
        _record_service_reference(self._runtime.provides, service, "singleton")
        self._runtime.provisions.append(_SingletonProvision(service, implementation))

    def provide_many(self, service: Service[object]) -> StagedServiceSpawner:
        lifecycle = self._runtime.lifecycle
        lifecycle.assert_setting_up("provide service instances")
        _record_service_reference(self._runtime.provides, service, "keyed")

        def validate(key: str, implementation: object) -> None:
            if not key:
                raise TypeError("Facet service instance key must not be empty")
            if not _is_implementation(implementation):
                raise TypeError(f"Facet service {service.id} implementation must be an object")
            if not service.local:
                validate_remote_service_implementation(service.id, implementation)

        instances = StagedServiceSpawner(lifecycle, validate)
        self._runtime.provisions.append(_KeyedProvision(service, instances))
        return instances

    def use[T](self, service: Service[T]) -> T:
        runtime = self._runtime
        runtime.lifecycle.assert_setting_up("acquire services")
        _record_service_reference(runtime.requires, service, "singleton")
        if service.id not in runtime.singleton_views:
            runtime.singleton_views[service.id] = self._slots.get_singleton(service, runtime.lifecycle.assert_service_access)
        return cast(T, runtime.singleton_views[service.id])

    def observe(self, service: Service[object], handler: Observer) -> None:
        lifecycle = self._runtime.lifecycle
        lifecycle.assert_setting_up("observe services")
        _record_service_reference(self._runtime.requires, service, "keyed")
        lifecycle.observe(lambda: self._slots.observe(service, lifecycle.assert_service_access, handler))

    def replicated_state[T](self, initial: T) -> MutableReplicatedStateImpl[T]:
        self._runtime.lifecycle.assert_running("create replicated state")
        return MutableReplicatedStateImpl(initial)

    def own(self, disposal: Disposal) -> None:
        self._runtime.lifecycle.own(disposal)

    def on_activate(self, callback: Disposal) -> None:
        self._runtime.lifecycle.on_activate(callback)

    def on_deactivate(self, callback: Disposal) -> None:
        self._runtime.lifecycle.own(callback)


class FacetKernel:
    def __init__(self, options: FacetOptions) -> None:
        ids = [facet.id for facet in options.facets]
        if any(not id for id in ids):
            raise RuntimeError("Facet ID must not be empty")
        if len(set(ids)) != len(ids):
            raise RuntimeError("Facet IDs must be unique within a generation")
        self._initial_facets = options.facets
        self._service_sources = options.service_sources
        self._on_error = options.on_error or (lambda error: None)
        self._facets: dict[str, FacetRuntime] = {}
        self._service_slots = HostServiceSlots()
        self._source_bindings: dict[int, tuple[RemoteServiceSource, RemoteServices]] = {}
        self._activation_order: list[str] = []
        self._provider: RemoteServiceProvider | None = None
        self._internal_services: RemoteServices | None = None
        self._local_keyed_services: LocalKeyedServiceRegistry | None = None
        self._phase: GenerationPhase = "setup"

    @property
    def provider(self) -> RemoteServiceProvider:
        if self._provider is None:
            raise RuntimeError("Facet service provider is not assembled")
        return self._provider

    def _setup_facet(self, facet: Facet, record: FacetRuntime) -> None:
        result = facet.setup(_Environment(record, self._service_slots))
        if inspect.isawaitable(result):
            pending = asyncio.ensure_future(result)
            pending.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            raise RuntimeError(f"Facet {facet.id} setup must be synchronous")
        record.lifecycle.prepared()

    async def activate(self) -> None:
        records: list[FacetRuntime] = []
        try:
            for facet in self._initial_facets:
                record = FacetRuntime(facet.id, FacetLifecycle(facet.id))
                self._facets[facet.id] = record
                self._setup_facet(facet, record)
                records.append(record)
            self._phase = "assembling"
            external_services = await self._resolve_external_services(records)
            self._activation_order = _validate_facets(records, external_services)
            self._assemble_providers()
            self._bind_services(external_services)
            self._phase = "connecting"
            bindings = [binding for _, binding in self._source_bindings.values()]
            bindings.append(self._internal_service_binding)
            await asyncio.gather(*(services.ready(BACKGROUND_CONTEXT) for services in bindings))
            self._phase = "activating"
            for id in self._activation_order:
                await self._facets[id].lifecycle.activate()
            self._phase = "active"
        except (Exception, asyncio.CancelledError) as error:
            cleanup_errors = await self._terminate()
            if cleanup_errors:
                raise BaseExceptionGroup("Facet generation startup and cleanup failed", [error, *cleanup_errors])
            raise

    async def reload(self, facets: Sequence[Facet]) -> None:
        if self._phase != "active":
            raise RuntimeError(f"Facet host cannot reload while {self._phase}")
        ids = [facet.id for facet in facets]
        if any(not id for id in ids):
            raise RuntimeError("Facet ID must not be empty")
        if len(set(ids)) != len(ids):
            raise RuntimeError("Reloaded facet IDs must be unique")
        for id in ids:
            if id not in self._facets:
                raise RuntimeError(f"Facet {id} is not active")
        self._phase = "reloading"
        staged: list[FacetRuntime] = []
        candidates: list[FacetRuntime] = []
        try:
            for facet in facets:
                record = FacetRuntime(facet.id, FacetLifecycle(facet.id))
                staged.append(record)
                self._setup_facet(facet, record)
                previous = self._facets[facet.id]
                if not (_same_references(previous.requires, record.requires)
                        and _same_references(previous.provides, record.provides)):
                    raise RuntimeError(f"Reloaded facet {facet.id} must preserve its service requirements and provisions")
                self._validate_replacement_provisions(record.provisions)
                candidates.append(record)
        except (Exception, asyncio.CancelledError) as error:
            cleanup_errors = await _dispose_facet_records(list(reversed(staged)))
            if cleanup_errors:
                abort_errors = await self._abort()
                raise BaseExceptionGroup("Facet reload setup and cleanup failed", [error, *cleanup_errors, *abort_errors])
            self._phase = "active"
            raise

        replacements = {record.facet_id: record for record in candidates}
        candidate_order = [replacements[id] for id in self._activation_order if id in replacements]
        try:
            for candidate in candidate_order:
                await candidate.lifecycle.activate()
            for candidate in candidate_order:
                self._validate_replacement_provisions(candidate.provisions)
        except (Exception, asyncio.CancelledError) as error:
            cleanup_errors = await _dispose_facet_records(list(reversed(candidate_order)))
            if cleanup_errors:
                abort_errors = await self._abort()
                raise BaseExceptionGroup("Facet reload activation and cleanup failed", [error, *cleanup_errors, *abort_errors])
            self._phase = "active"
            raise

        previous_records = [self._facets[candidate.facet_id] for candidate in candidate_order]
        for candidate in candidate_order:
            self._facets[candidate.facet_id] = candidate
        try:
            for candidate in candidate_order:
                for provision in candidate.provisions:
                    if isinstance(provision, _SingletonProvision):
                        if provision.service.local:
                            self._service_slots.bind_singleton(provision.service.id, provision.implementation)
                        else:
                            self.provider.replace(provision.service, provision.implementation)
            _raise_errors(await _dispose_facet_records(list(reversed(previous_records))), "Failed to retire replaced facets")
            for candidate in candidate_order:
                for provision in candidate.provisions:
                    if isinstance(provision, _KeyedProvision):
                        if provision.service.local:
                            provision.connect_local(self._local_keyed_registry)
                        else:
                            provision.connect_remote(self.provider)
        except (Exception, asyncio.CancelledError) as error:
            abort_errors = await self._abort(previous_records)
            raise BaseExceptionGroup("Facet reload failed after cutover", [error, *abort_errors])
        self._phase = "active"

    async def dispose(self) -> None:
        if self._phase == "dead":
            return
        if self._phase != "active":
            raise RuntimeError(f"Facet host cannot be disposed while {self._phase}")
        _raise_errors(await self._terminate(), "Failed to dispose facet generation")

    def _validate_replacement_provisions(self, provisions: Sequence[FacetProvision]) -> None:
        for provision in provisions:
            if isinstance(provision, _SingletonProvision) and not provision.service.local:
                self.provider.validate_replacement(provision.service, provision.implementation)

    async def _resolve_external_services(self, records: Sequence[FacetRuntime]) -> dict[str, ExternalService]:
        catalogues = await asyncio.gather(*(source.catalogue(BACKGROUND_CONTEXT) for source in self._service_sources))
        offered: dict[str, tuple[ServiceMode, RemoteServiceSource]] = {}
        for source, entries in zip(self._service_sources, catalogues):
            for entry in entries:
                service_id = entry["serviceId"]
                if service_id in offered:
                    raise RuntimeError(f"Facet host service {service_id} is offered by more than one source")
                offered[service_id] = (entry["mode"], source)
        local = {provision.service_id for record in records for provision in record.provides}
        external: dict[str, ExternalService] = {}
        for record in records:
            for requirement in record.requires:
                service_id = requirement.service_id
                if service_id in local or service_id in external:
                    continue
                source_info = offered.get(service_id)
                if source_info is None:
                    deferred = [source for source in self._service_sources if source.accepts_unavailable_services]
                    if len(deferred) > 1:
                        raise RuntimeError(f"Facet host service {service_id} has more than one deferred source")
                    if len(deferred) == 1:
                        source_info = (requirement.mode, deferred[0])
                if source_info is not None:
                    external[service_id] = ExternalService(requirement.service, source_info[0], source_info[1])
        by_source: dict[int, tuple[RemoteServiceSource, list[str]]] = {}
        for service_id, external_service in external.items():
            source = external_service.source
            if id(source) not in by_source:
                by_source[id(source)] = (source, [])
            by_source[id(source)][1].append(service_id)
        for identity, (source, service_ids) in by_source.items():
            self._source_bindings[identity] = (source, source.open(RemoteServiceSourceOptions(
                services=[Service(service_id) for service_id in service_ids],
                assert_access=self._assert_service_target_access, on_error=self._on_error,
            )))
        return external

    def _assemble_providers(self) -> None:
        provisions = self._provisions()
        remote = [provision for provision in provisions if not provision.service.local]
        provider = RemoteServiceProvider([ServiceProviderDefinition(provision.service, provision.kind) for provision in remote])
        internal = RemoteServiceBindingImpl(RemoteServiceBindingOptions(
            services=[provision.service for provision in remote],
            transport=create_loopback_service_transport(provider),
            assert_access=self._assert_service_target_access, on_error=self._on_error,
        ))
        local_keyed = LocalKeyedServiceRegistry([
            provision.service for provision in provisions
            if isinstance(provision, _KeyedProvision) and provision.service.local
        ], self._on_error)
        self._provider = provider
        self._internal_services = internal
        self._local_keyed_services = local_keyed
        for provision in provisions:
            if isinstance(provision, _SingletonProvision):
                if not provision.service.local:
                    provider.provide(provision.service, provision.implementation)
            elif provision.service.local:
                provision.connect_local(local_keyed)
            else:
                provision.connect_remote(provider)

    def _bind_services(self, external_services: dict[str, ExternalService]) -> None:
        for provision in self._provisions():
            if isinstance(provision, _SingletonProvision):
                if self._service_slots.has_singleton(provision.service.id):
                    target = (provision.implementation if provision.service.local
                              else self._internal_service_binding.use(provision.service))
                    self._service_slots.bind_singleton(provision.service.id, target)
            else:
                self._service_slots.bind_keyed(provision.service.id,
                    self._local_keyed_registry if provision.service.local else self._internal_service_binding)
        for service_id, external in external_services.items():
            binding = self._source_bindings.get(id(external.source))
            if binding is None:
                raise RuntimeError(f"Service source for {service_id} is not open")
            services = binding[1]
            if external.mode == "singleton":
                self._service_slots.bind_singleton(service_id, services.use(external.service))
            else:
                self._service_slots.bind_keyed(service_id, services)

    def _provisions(self) -> list[FacetProvision]:
        return [provision for record in self._facets.values() for provision in record.provisions]

    @property
    def _local_keyed_registry(self) -> LocalKeyedServiceRegistry:
        if self._local_keyed_services is None:
            raise RuntimeError("Facet keyed services are not assembled")
        return self._local_keyed_services

    @property
    def _internal_service_binding(self) -> RemoteServices:
        if self._internal_services is None:
            raise RuntimeError("Facet remote services are not assembled")
        return self._internal_services

    async def _dispose_service_bindings(self) -> list[BaseException]:
        bindings = [binding for _, binding in self._source_bindings.values()]
        self._source_bindings.clear()
        if self._internal_services is not None:
            bindings.append(self._internal_services)
        self._internal_services = None
        results = await asyncio.gather(*(services.dispose(BACKGROUND_CONTEXT) for services in bindings), return_exceptions=True)
        return [result for result in results if isinstance(result, BaseException)]

    def _assert_service_target_access(self) -> None:
        if self._phase not in ("activating", "active", "reloading", "disposing"):
            raise RuntimeError(f"Facet service targets cannot be used during {self._phase}")

    async def _abort(self, extra_records: Sequence[FacetRuntime] = ()) -> list[BaseException]:
        for record in self._facets.values():
            record.lifecycle.revoke()
        for record in extra_records:
            record.lifecycle.revoke()
        return await self._terminate(extra_records)

    async def _terminate(self, extra_records: Sequence[FacetRuntime] = ()) -> list[BaseException]:
        self._phase = "disposing"
        errors = await self._dispose_lifecycles()
        errors.extend(await _dispose_facet_records(list(reversed(extra_records))))
        try:
            if self._local_keyed_services is not None:
                self._local_keyed_services.dispose()
        except (Exception, asyncio.CancelledError) as error:
            errors.append(error)
        self._local_keyed_services = None
        errors.extend(await self._dispose_service_bindings())
        try:
            self._service_slots.dispose()
        except (Exception, asyncio.CancelledError) as error:
            errors.append(error)
        try:
            if self._provider is not None:
                self._provider.dispose()
        except (Exception, asyncio.CancelledError) as error:
            errors.append(error)
        self._phase = "dead"
        return errors

    async def _dispose_lifecycles(self) -> list[BaseException]:
        errors: list[BaseException] = []
        order = list(reversed(self._activation_order or self._facets))
        for id in order:
            record = self._facets.pop(id, None)
            if record is not None:
                try:
                    await record.lifecycle.dispose()
                except (Exception, asyncio.CancelledError) as error:
                    errors.append(error)
        return errors


async def _dispose_facet_records(records: Sequence[FacetRuntime]) -> list[BaseException]:
    errors: list[BaseException] = []
    for record in records:
        try:
            await record.lifecycle.dispose()
        except (Exception, asyncio.CancelledError) as error:
            errors.append(error)
    return errors


def _validate_facets(records: Sequence[FacetRuntime], external_services: dict[str, ExternalService]) -> list[str]:
    providers: dict[str, tuple[str | None, ServiceMode]] = {
        service_id: (None, service.mode) for service_id, service in external_services.items()
    }
    for record in records:
        for provision in record.provides:
            existing = providers.get(provision.service_id)
            if existing is not None:
                facet_id, mode = existing
                if mode != provision.mode:
                    raise RuntimeError(f"Service {provision.service_id} is provided as both singleton and keyed")
                if facet_id is None:
                    raise RuntimeError(f"Service {provision.service_id} is provided by both the host and {record.facet_id}")
                raise RuntimeError(f"Service {provision.service_id} is provided by both {facet_id} and {record.facet_id}")
            providers[provision.service_id] = (record.facet_id, provision.mode)
    dependencies: dict[str, dict[str, None]] = {record.facet_id: {} for record in records}
    dependents: dict[str, dict[str, None]] = {record.facet_id: {} for record in records}
    for record in records:
        for requirement in record.requires:
            provider = providers.get(requirement.service_id)
            if provider is None:
                raise RuntimeError(f"Facet {record.facet_id} requires local/{requirement.service_id}/{requirement.mode}, but no facet provides it")
            facet_id, mode = provider
            if mode != requirement.mode:
                raise RuntimeError(f"Facet {record.facet_id} requires {requirement.service_id} as {requirement.mode}, but {facet_id or 'the host'} provides it as {mode}")
            if facet_id is None or facet_id == record.facet_id:
                continue
            dependencies[record.facet_id][facet_id] = None
            dependents[facet_id][record.facet_id] = None
    remaining = {id: len(values) for id, values in dependencies.items()}
    ready = deque(record.facet_id for record in records if remaining[record.facet_id] == 0)
    order: list[str] = []
    while ready:
        id = ready.popleft()
        order.append(id)
        for dependent in dependents[id]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
    if len(order) != len(records):
        cycle = [record.facet_id for record in records if remaining[record.facet_id] > 0]
        raise RuntimeError(f"Facet dependency cycle: {', '.join(cycle)}")
    return order


def _same_references(left: Sequence[FacetServiceReference], right: Sequence[FacetServiceReference]) -> bool:
    return len(left) == len(right) and all(
        any(other.service_id == reference.service_id and other.mode == reference.mode for other in right)
        for reference in left
    )


async def _await_result(value: None | Awaitable[None]) -> None:
    if inspect.isawaitable(value):
        await value


def _raise_errors(errors: list[BaseException], message: str) -> None:
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup(message, errors)


def _is_implementation(value: object) -> bool:
    return value is not None and not callable(value) and not isinstance(
        value, (list, tuple, str, bytes, bool, int, float, complex),
    )


def _record_service_reference(target: list[FacetServiceReference], service: Service[object], mode: ServiceMode) -> None:
    if not any(reference.service_id == service.id and reference.mode == mode for reference in target):
        target.append(FacetServiceReference(service.id, service, mode))
