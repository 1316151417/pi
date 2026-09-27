"""Provider collections, transactional model refresh and authenticated requests."""

from __future__ import annotations

import asyncio
import copy
import inspect
import os
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, MutableSequence, Sequence
from dataclasses import dataclass, fields, replace
from typing import cast, overload

from ._values import UNDEFINED, Undefined
from .abort import AbortController, AbortError, AbortSignal, operation_signal, race_with_abort_signal
from .api.lazy import lazy_stream
from .auth.context import default_provider_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.helpers import env_api_key_auth
from .auth.resolve import AuthResolutionOverrides, ModelsError, ModelsErrorCode, resolve_provider_auth
from .auth.types import (
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthContext, AuthInteraction,
    AuthOperationOptions, AuthResult, AuthType, Credential, CredentialStore, ModelAuth, OAuthAuth,
    OAuthCredential, ProviderAuth as AuthConfiguration, ProviderAuthInteraction,
)
from .event_stream import AssistantMessageEventStream
from .models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions
from .transcript import normalize_context
from .types import (
    AssistantMessage, Context, Cost, DeferredCancelFunction, DeferredCancelOptions, DeferredFetchFunction,
    DeferredFetchOptions, DeferredHandle, Model, ModelCostRates, ModelThinkingLevel, ProviderHeaders, ProviderRequestOptions,
    ProviderStreams, SimpleStreamFunction, SimpleStreamOptions, StreamFunction, StreamOptions,
    TranscriptContext, Usage,
)
from .utils._javascript import javascript_object_keys
from .utils.abort_signals import combine_abort_signals


@dataclass
class ModelsPublication:
    """Omitted persistence keeps storage unchanged; explicit None deletes it."""

    persist: ModelsStoreEntry | None | Undefined = UNDEFINED
    update: Callable[[], None] | None = None


@dataclass
class RefreshModelsContext:
    publish: Callable[[ModelsPublication], Awaitable[bool]]
    allow_network: bool
    signal: AbortSignal
    credential: Credential | None = None
    stored: ModelsStoreEntry | None = None
    force: bool | None = None


@dataclass
class ModelsRefreshOptions:
    allow_network: bool | None = None
    providers: Sequence[str] | None = None
    force: bool | None = None
    signal: AbortSignal | None = None


@dataclass
class ModelsRefreshResult:
    aborted: bool
    errors: Mapping[str, BaseException]


type HeaderTransform = Callable[[ProviderHeaders], ProviderHeaders | Awaitable[ProviderHeaders]]


@dataclass
class ModelsRequestTransforms:
    transform_headers: HeaderTransform | None = None


@dataclass
class ModelsApiStreamOptions(StreamOptions):
    transform_headers: HeaderTransform | None = None


@dataclass
class ModelsSimpleStreamOptions(SimpleStreamOptions):
    transform_headers: HeaderTransform | None = None


@dataclass
class ModelsDeferredFetchOptions(DeferredFetchOptions):
    transform_headers: HeaderTransform | None = None


@dataclass
class ModelsDeferredCancelOptions(DeferredCancelOptions):
    transform_headers: HeaderTransform | None = None


@dataclass
class CreateModelsOptions:
    credentials: CredentialStore | None = None
    models_store: ModelsStore | None = None
    auth_context: AuthContext | None = None


type FetchModelsFunction = Callable[[RefreshModelsContext], Awaitable[Sequence[Model]]]
type FilterModelsFunction = Callable[[Sequence[Model], Credential | None], Sequence[Model] | None]
type RefreshModelsFunction = Callable[[RefreshModelsContext], Awaitable[None]]


@dataclass
class CreateProviderOptions:
    id: str
    auth: AuthConfiguration
    models: Sequence[Model]
    api: ProviderStreams | Mapping[str, ProviderStreams | None]
    name: str | None = None
    base_url: str | None = None
    headers: ProviderHeaders | None = None
    fetch_models: FetchModelsFunction | None = None
    filter_models: FilterModelsFunction | None = None


class ProviderAuth(AuthConfiguration):
    """The existing Python environment/key constructor, backed by full auth."""

    def __init__(
        self, env_var: str | None = None, explicit_key: str | None = None, *,
        api_key: ApiKeyAuth | None = None, oauth: OAuthAuth | None = None,
    ) -> None:
        self.env_var = env_var
        self.explicit_key = explicit_key
        if api_key is None:
            handler = env_api_key_auth("API key", [env_var] if env_var is not None else [])

            async def resolve(input: ApiKeyAuthInput) -> AuthResult | None:
                input.signal.throw_if_aborted()
                if input.credential is not None and input.credential.key is not None:
                    return AuthResult(
                        auth=ModelAuth(api_key=input.credential.key), env=input.credential.env,
                        source="stored credential",
                    )
                if self.explicit_key is not None:
                    return AuthResult(auth=ModelAuth(api_key=self.explicit_key), source="explicit key")
                if self.env_var is not None:
                    value = await input.ctx.env(self.env_var)
                    input.signal.throw_if_aborted()
                    if value is not None:
                        return AuthResult(auth=ModelAuth(api_key=value), source=self.env_var)
                return None

            api_key = ApiKeyAuth(name=handler.name, resolve=resolve, login=handler.login)
        super().__init__(api_key=api_key, oauth=oauth)

    def resolve(self) -> str | None:
        """Retain direct ambient-key access for existing Python callers."""
        if self.explicit_key is not None:
            return self.explicit_key
        return os.environ.get(self.env_var) if self.env_var is not None else None


class Provider:
    """Concrete provider with a static baseline and a published dynamic overlay."""

    def __init__(
        self, id: str, name: str | None = None, models: Sequence[Model] = (),
        stream: StreamFunction | None = None, stream_simple: SimpleStreamFunction | None = None,
        base_url: str | None = None, headers: ProviderHeaders | None = None,
        api_key_env_var: str | None = None, api_key: str | None = None,
        fetch_deferred: DeferredFetchFunction | None = None,
        cancel_deferred: DeferredCancelFunction | None = None, *,
        auth: AuthConfiguration | None = None,
        api: ProviderStreams | Mapping[str, ProviderStreams | None] | None = None,
        fetch_models: FetchModelsFunction | None = None, filter_models: FilterModelsFunction | None = None,
    ) -> None:
        self.id = id
        self.name = name if name is not None else id
        self.base_url = base_url
        self.headers = headers
        self.auth = auth if auth is not None else ProviderAuth(api_key_env_var, api_key)
        self._baseline_models: Sequence[Model] = models if api is not None else list(models)
        self._dynamic_models: Sequence[Model] = []
        self._fetch_models = fetch_models
        self.refresh_models: RefreshModelsFunction | None = self._refresh_models if fetch_models is not None else None
        self.filter_models = filter_models
        self.fetch_deferred: DeferredFetchFunction | None = None
        self.cancel_deferred: DeferredCancelFunction | None = None
        if api is None:
            if stream is None:
                raise TypeError("Provider requires an api or stream implementation")
            api = ProviderStreams(
                stream=stream, stream_simple=stream_simple if stream_simple is not None else stream,
                fetch_deferred=fetch_deferred, cancel_deferred=cancel_deferred,
            )
        self._single = cast(ProviderStreams, api) if callable(getattr(api, "stream", None)) else None
        self._by_api = None if self._single is not None else cast(Mapping[str, ProviderStreams | None], api)
        implementations = (
            [self._single] if self._single is not None
            else [entry for entry in cast(Mapping[str, ProviderStreams | None], self._by_api).values() if entry is not None]
        )
        if any(entry.fetch_deferred is not None for entry in implementations):
            self.fetch_deferred = self._fetch_deferred
        if any(entry.cancel_deferred is not None for entry in implementations):
            self.cancel_deferred = self._cancel_deferred

    def get_models(self) -> list[Model]:
        merged = list(self._baseline_models)
        for model in self._dynamic_models:
            index = next((i for i, entry in enumerate(merged) if entry.id == model.id), -1)
            if index >= 0:
                merged[index] = model
            else:
                merged.append(model)
        return merged

    def add_model(self, model: Model) -> Model:
        """Register a Python caller's custom model in the static baseline."""
        if isinstance(self._baseline_models, MutableSequence):
            self._baseline_models.append(model)
        else:
            self._baseline_models = [*self._baseline_models, model]
        return model

    async def _refresh_models(self, context: RefreshModelsContext) -> None:
        if context.stored is not None:
            restored = [model for model in context.stored.models if model.provider == self.id]

            def restore() -> None:
                self._dynamic_models = restored

            if not await context.publish(ModelsPublication(update=restore)):
                return
        if not context.allow_network or context.signal.aborted:
            return
        refreshed = await cast(FetchModelsFunction, self._fetch_models)(context)
        if context.signal.aborted:
            return

        def update() -> None:
            self._dynamic_models = refreshed

        await context.publish(ModelsPublication(
            persist=ModelsStoreEntry(models=refreshed, checked_at=time.time_ns() // 1_000_000), update=update,
        ))

    def _api_for(self, model: Model) -> ProviderStreams | None:
        return self._single if self._single is not None else cast(Mapping[str, ProviderStreams | None], self._by_api).get(model.api)

    def _dispatch(
        self, model: Model, run: Callable[[ProviderStreams], AssistantMessageEventStream],
    ) -> AssistantMessageEventStream:
        streams = self._api_for(model)
        if streams is None:
            async def missing() -> AssistantMessageEventStream:
                raise ModelsError("stream", f'Provider {self.id} has no API implementation for "{model.api}"')

            return lazy_stream(model, missing)
        return run(streams)

    def stream(
        self, model: Model, context: TranscriptContext, options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._dispatch(model, lambda streams: streams.stream(model, context, options))

    def stream_simple(
        self, model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self._dispatch(model, lambda streams: streams.stream_simple(model, context, options))

    def _fetch_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def setup() -> AssistantMessageEventStream:
            implementation = self._api_for(model)
            if implementation is None or implementation.fetch_deferred is None:
                raise ModelsError("provider", f'Provider {self.id} does not support deferred responses for "{model.api}"')
            return implementation.fetch_deferred(model, handle, options)

        return lazy_stream(model, setup)

    async def _cancel_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredCancelOptions | None = None,
    ) -> None:
        implementation = self._api_for(model)
        if implementation is None or implementation.cancel_deferred is None:
            raise ModelsError("provider", f'Provider {self.id} cannot cancel deferred responses for "{model.api}"')
        await implementation.cancel_deferred(model, handle, options)


@overload
def create_provider(id: CreateProviderOptions) -> Provider: ...


@overload
def create_provider(
    id: str, name: str | None = None, models: Sequence[Model] = (), stream: StreamFunction | None = None,
    *, stream_simple: SimpleStreamFunction | None = None, base_url: str | None = None,
    headers: ProviderHeaders | None = None, api_key_env_var: str | None = None, api_key: str | None = None,
    fetch_deferred: DeferredFetchFunction | None = None, cancel_deferred: DeferredCancelFunction | None = None,
    auth: AuthConfiguration | None = None, api: ProviderStreams | Mapping[str, ProviderStreams | None] | None = None,
    fetch_models: FetchModelsFunction | None = None, filter_models: FilterModelsFunction | None = None,
) -> Provider: ...


def create_provider(
    id: str | CreateProviderOptions, name: str | None = None, models: Sequence[Model] = (),
    stream: StreamFunction | None = None, *, stream_simple: SimpleStreamFunction | None = None,
    base_url: str | None = None, headers: ProviderHeaders | None = None,
    api_key_env_var: str | None = None, api_key: str | None = None,
    fetch_deferred: DeferredFetchFunction | None = None, cancel_deferred: DeferredCancelFunction | None = None,
    auth: AuthConfiguration | None = None, api: ProviderStreams | Mapping[str, ProviderStreams | None] | None = None,
    fetch_models: FetchModelsFunction | None = None, filter_models: FilterModelsFunction | None = None,
) -> Provider:
    if isinstance(id, CreateProviderOptions):
        return Provider(
            id=id.id, name=id.name, models=id.models, base_url=id.base_url, headers=id.headers,
            auth=id.auth, api=id.api, fetch_models=id.fetch_models, filter_models=id.filter_models,
        )
    return Provider(
        id=id, name=name, models=models, stream=stream, stream_simple=stream_simple, base_url=base_url,
        headers=headers, api_key_env_var=api_key_env_var, api_key=api_key, fetch_deferred=fetch_deferred,
        cancel_deferred=cancel_deferred, auth=auth, api=api, fetch_models=fetch_models, filter_models=filter_models,
    )


@dataclass
class _ProviderEntry:
    key: str
    value: Provider
    active: bool = True


class _ProviderMap:
    """Map iteration stays live when provider callbacks mutate the registry."""

    def __init__(self) -> None:
        self._entries: dict[str, _ProviderEntry] = {}
        self._order: list[_ProviderEntry] = []
        self._iterators = 0

    def get(self, key: str) -> Provider | None:
        entry = self._entries.get(key)
        return entry.value if entry is not None else None

    def set(self, key: str, value: Provider) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            entry.value = value
        else:
            entry = _ProviderEntry(key, value)
            self._entries[key] = entry
            self._order.append(entry)

    def delete(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            entry.active = False
            if not self._iterators:
                self._order = list(self._entries.values())

    def clear(self) -> None:
        for entry in self._entries.values():
            entry.active = False
        self._entries.clear()
        if not self._iterators:
            self._order.clear()

    def keys(self) -> list[str]:
        return list(self._entries)

    def values(self) -> Iterator[Provider]:
        self._iterators += 1
        try:
            index = 0
            while index < len(self._order):
                entry = self._order[index]
                index += 1
                if entry.active:
                    yield entry.value
        finally:
            self._iterators -= 1
            if not self._iterators:
                self._order = list(self._entries.values())


def _merge_headers(base: ProviderHeaders | None, override: ProviderHeaders | None) -> ProviderHeaders | None:
    if base is None and override is None:
        return None
    merged = dict(base) if base is not None else {}
    if override is not None:
        for name in javascript_object_keys(override):
            lower_name = name.lower()
            for existing_name in list(merged):
                if existing_name.lower() == lower_name:
                    del merged[existing_name]
            merged[name] = override[name]
    return merged


class Models:
    def __init__(self, options: CreateModelsOptions | None = None) -> None:
        options = options if options is not None else CreateModelsOptions()
        self._providers = _ProviderMap()
        self._credentials = options.credentials if options.credentials is not None else InMemoryCredentialStore()
        self._models_store = options.models_store if options.models_store is not None else InMemoryModelsStore()
        self._auth_context = options.auth_context if options.auth_context is not None else default_provider_auth_context()
        self._refresh_generations: dict[str, int] = {}
        self._refresh_controllers: dict[str, AbortController] = {}
        self._publication_chains: dict[str, asyncio.Task[None]] = {}

    def set_provider(self, provider: Provider) -> None:
        self._supersede_provider_refresh(provider.id)
        self._providers.set(provider.id, provider)

    def delete_provider(self, id: str) -> None:
        self._supersede_provider_refresh(id)
        self._providers.delete(id)

    def remove_provider(self, provider_id: str) -> None:
        self.delete_provider(provider_id)

    def clear_providers(self) -> None:
        for provider_id in dict.fromkeys([*self._providers.keys(), *self._refresh_controllers]):
            self._supersede_provider_refresh(provider_id)
        self._providers.clear()

    def get_providers(self) -> list[Provider]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> Provider | None:
        return self._providers.get(id)

    def get_models(self, provider: str | None = None) -> Sequence[Model]:
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None:
                return []
            try:
                return entry.get_models()
            except (Exception, asyncio.CancelledError):
                return []
        models: list[Model] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_models())
            except (Exception, asyncio.CancelledError):
                pass
        return models

    def get_model(self, provider: str, id: str) -> Model | None:
        return next((model for model in self.get_models(provider) if model.id == id), None)

    def _supersede_provider_refresh(self, provider_id: str) -> int:
        generation = self._refresh_generations.get(provider_id, 0) + 1
        self._refresh_generations[provider_id] = generation
        previous = self._refresh_controllers.pop(provider_id, None)
        if previous is not None:
            previous.abort()
        return generation

    def _publish_provider_models(
        self, provider_id: str, generation: int, signal: AbortSignal, publication: ModelsPublication,
    ) -> Awaitable[bool]:
        previous = self._publication_chains.get(provider_id)

        async def run() -> bool:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except (Exception, asyncio.CancelledError):
                    pass
            if signal.aborted or self._refresh_generations.get(provider_id) != generation:
                return False
            if publication.persist is None:
                await self._models_store.delete(provider_id, ModelsStoreOperationOptions(signal=signal))
            elif not isinstance(publication.persist, Undefined):
                await self._models_store.write(
                    provider_id, copy.deepcopy(publication.persist), ModelsStoreOperationOptions(signal=signal),
                )
            if signal.aborted or self._refresh_generations.get(provider_id) != generation:
                return False
            if publication.update is not None:
                publication.update()
            return True

        queued = asyncio.get_running_loop().create_task(run())

        async def observe() -> None:
            try:
                await asyncio.shield(queued)
            except (Exception, asyncio.CancelledError):
                pass

        tail = asyncio.get_running_loop().create_task(observe())
        self._publication_chains[provider_id] = tail

        def cleanup(completed: asyncio.Task[None]) -> None:
            if self._publication_chains.get(provider_id) is completed:
                del self._publication_chains[provider_id]

        tail.add_done_callback(cleanup)
        return race_with_abort_signal(queued, signal)

    async def _run_provider_refresh_phase(
        self, provider: Provider, credential: Credential | None, allow_network: bool, force: bool | None,
        generation: int, signal: AbortSignal,
    ) -> None:
        stored = await self._models_store.read(provider.id, ModelsStoreOperationOptions(signal=signal))
        await cast(RefreshModelsFunction, getattr(provider, "refresh_models"))(RefreshModelsContext(
            credential=credential, stored=copy.deepcopy(stored) if stored is not None else None,
            publish=lambda publication: self._publish_provider_models(provider.id, generation, signal, publication),
            allow_network=allow_network, force=force if allow_network else None, signal=signal,
        ))

    async def refresh(self, options: ModelsRefreshOptions | None = None) -> ModelsRefreshResult:
        options = options if options is not None else ModelsRefreshOptions()
        allow_network = options.allow_network if options.allow_network is not None else True
        caller_signal = operation_signal(options.signal)
        errors: dict[str, BaseException] = {}
        if caller_signal.aborted:
            return ModelsRefreshResult(aborted=True, errors=errors)
        selected = set(options.providers) if options.providers is not None else None
        refreshable = [
            provider for provider in self._providers.values()
            if getattr(provider, "refresh_models", None) is not None and (selected is None or provider.id in selected)
        ]

        async def refresh_provider(provider: Provider, generation: int, controller: AbortController) -> None:
            combined = combine_abort_signals((caller_signal, controller.signal))
            signal = cast(AbortSignal, combined.signal)

            async def operation() -> None:
                credential: Credential | None = None
                credential_error: BaseException | None = None
                try:
                    credential = await self._read_credential(provider.id, signal)
                except (Exception, asyncio.CancelledError) as error:
                    credential_error = error
                await self._run_provider_refresh_phase(provider, credential, False, None, generation, signal)
                if credential_error is not None:
                    raise credential_error
                if not allow_network or signal.aborted:
                    return
                credential = await self._resolve_refresh_credential(provider, credential, signal)
                if credential is None:
                    return
                await self._run_provider_refresh_phase(provider, credential, True, options.force, generation, signal)

            try:
                await race_with_abort_signal(operation(), signal)
            except (Exception, asyncio.CancelledError) as error:
                if not signal.aborted:
                    errors[provider.id] = error
            finally:
                if self._refresh_controllers.get(provider.id) is controller:
                    del self._refresh_controllers[provider.id]
                combined.cleanup()

        pending: list[asyncio.Task[None]] = []
        for provider in refreshable:
            generation = self._supersede_provider_refresh(provider.id)
            controller = AbortController()
            self._refresh_controllers[provider.id] = controller
            pending.append(asyncio.get_running_loop().create_task(refresh_provider(provider, generation, controller)))
        try:
            await race_with_abort_signal(asyncio.gather(*pending), caller_signal)
        except (Exception, asyncio.CancelledError):
            if not caller_signal.aborted:
                raise
        return ModelsRefreshResult(aborted=caller_signal.aborted, errors=dict(errors))

    async def _resolve_refresh_credential(
        self, provider: Provider, stored: Credential | None, signal: AbortSignal,
    ) -> Credential | None:
        if isinstance(stored, OAuthCredential):
            oauth = provider.auth.oauth
            if oauth is None:
                return None
            if time.time_ns() // 1_000_000 < stored.expires:
                return stored
            if signal.aborted:
                return None

            async def refresh(current: Credential | None) -> Credential | None:
                if not isinstance(current, OAuthCredential) or time.time_ns() // 1_000_000 < current.expires:
                    return None
                return await oauth.refresh(current, signal)

            post = await self._credentials.modify(provider.id, refresh, AuthOperationOptions(signal=signal))
            return post if isinstance(post, OAuthCredential) else None
        api_key = provider.auth.api_key
        if api_key is None:
            return None
        result = await api_key.resolve(ApiKeyAuthInput(
            ctx=self._auth_context, credential=stored if isinstance(stored, ApiKeyCredential) else None, signal=signal,
        ))
        return ApiKeyCredential(key=result.auth.api_key, env=result.env) if result is not None else None

    async def _read_credential(self, provider_id: str, signal: AbortSignal) -> Credential | None:
        try:
            return await self._credentials.read(provider_id, AuthOperationOptions(signal=signal))
        except (Exception, asyncio.CancelledError) as error:
            raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error) from error

    async def _check_provider_auth(
        self, provider: Provider, credential: Credential | None, signal: AbortSignal,
    ) -> AuthCheck | None:
        if isinstance(credential, OAuthCredential):
            return AuthCheck(type="oauth", source="OAuth") if provider.auth.oauth is not None else None
        api_key = provider.auth.api_key
        if api_key is None:
            return None
        if api_key.check is not None:
            try:
                return await api_key.check(ApiKeyAuthInput(
                    ctx=self._auth_context, credential=credential if isinstance(credential, ApiKeyCredential) else None,
                    signal=signal,
                ))
            except (Exception, asyncio.CancelledError) as error:
                raise ModelsError("auth", f"API key auth check failed for provider {provider.id}", cause=error) from error
        resolution = await resolve_provider_auth(
            provider, self._credentials, self._auth_context, AuthResolutionOverrides(signal=signal),
        )
        return AuthCheck(type="api_key", source=resolution.source) if resolution is not None else None

    def check_auth(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[AuthCheck | None]:
        signal = operation_signal(options.signal if options is not None else None)

        async def check() -> AuthCheck | None:
            signal.throw_if_aborted()
            provider = self._providers.get(provider_id)
            if provider is None:
                return None
            return await self._check_provider_auth(provider, await self._read_credential(provider_id, signal), signal)

        return race_with_abort_signal(check(), signal)

    async def is_configured(self, provider_id: str) -> bool:
        return await self.check_auth(provider_id) is not None

    def get_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None,
    ) -> Awaitable[list[Model]]:
        signal = operation_signal(options.signal if options is not None else None)

        async def available() -> list[Model]:
            signal.throw_if_aborted()
            if provider_id:
                provider = self._providers.get(provider_id)
                providers = [provider] if provider is not None else []
            else:
                providers = self.get_providers()

            async def check(provider: Provider) -> tuple[Provider, Credential | None, AuthCheck | None]:
                credential = await self._read_credential(provider.id, signal)
                return provider, credential, await self._check_provider_auth(provider, credential, signal)

            checks = await asyncio.gather(*(check(provider) for provider in providers))
            available_models: list[Model] = []
            for provider, credential, auth in checks:
                if auth is None:
                    continue
                models = provider.get_models()
                filter_models = getattr(provider, "filter_models", None)
                filtered = filter_models(models, credential) if filter_models is not None else None
                available_models.extend(filtered if filtered is not None else models)
            return available_models

        return race_with_abort_signal(available(), signal)

    async def get_auth(
        self, provider_or_model: str | Model, overrides: AuthResolutionOverrides | None = None,
    ) -> AuthResult | None:
        signal = operation_signal(overrides.signal if overrides is not None else None)
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        resolved_overrides = replace(overrides, signal=signal) if overrides is not None else AuthResolutionOverrides(signal=signal)
        result = await resolve_provider_auth(provider, self._credentials, self._auth_context, resolved_overrides)
        if result is None or isinstance(provider_or_model, str) or provider_or_model.headers is None:
            return result
        return replace(result, auth=replace(result.auth, headers=_merge_headers(result.auth.headers, provider_or_model.headers)))

    async def login(self, provider_id: str, type: AuthType, interaction: AuthInteraction) -> Credential:
        signal = operation_signal(interaction.signal)
        signal.throw_if_aborted()
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        method = provider.auth.oauth if type == "oauth" else provider.auth.api_key
        if method is None or method.login is None:
            raise ModelsError("auth", f"{provider.name} does not support {type} login")
        credential = await race_with_abort_signal(method.login(ProviderAuthInteraction(
            prompt=interaction.prompt, notify=interaction.notify, signal=signal,
        )), signal)
        mutation_started = False
        loop = asyncio.get_running_loop()
        started: asyncio.Future[None] = loop.create_future()

        async def store(_current: Credential | None) -> Credential:
            nonlocal mutation_started
            mutation_started = True
            if not started.done():
                started.set_result(None)
            return credential

        mutation = asyncio.ensure_future(self._credentials.modify(provider_id, store, AuthOperationOptions(signal=signal)))
        barrier: asyncio.Future[None] = loop.create_future()

        def mutation_done(completed: asyncio.Future[Credential | None]) -> None:
            try:
                completed.result()
            except BaseException as error:
                if not barrier.done():
                    barrier.set_exception(error)
            else:
                if not barrier.done():
                    barrier.set_result(None)

        def start_done(_completed: asyncio.Future[None]) -> None:
            if not barrier.done():
                barrier.set_result(None)

        def on_abort() -> None:
            if not mutation_started and not barrier.done():
                reason = signal.reason
                barrier.set_exception(reason if isinstance(reason, BaseException) else AbortError(str(reason)))

        mutation.add_done_callback(mutation_done)
        started.add_done_callback(start_done)
        signal.add_event_listener("abort", on_abort, once=True)
        if signal.aborted:
            on_abort()
        try:
            await barrier
            signal.remove_event_listener("abort", on_abort)
            await asyncio.shield(mutation)
        except (Exception, asyncio.CancelledError) as error:
            signal.throw_if_aborted()
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error) from error
        finally:
            signal.remove_event_listener("abort", on_abort)
        return credential

    async def logout(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        signal = operation_signal(options.signal if options is not None else None)
        signal.throw_if_aborted()
        try:
            await self._credentials.delete(provider_id, AuthOperationOptions(signal=signal))
        except (Exception, asyncio.CancelledError) as error:
            signal.throw_if_aborted()
            raise ModelsError("auth", f"Credential store delete failed for {provider_id}", cause=error) from error

    def _require_provider(self, model: Model) -> Provider:
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    async def _apply_auth[TOptions: ProviderRequestOptions[Model]](
        self, model: Model, options: TOptions | Mapping[str, object] | None, cls: type[TOptions],
    ) -> tuple[Model, TOptions]:
        self._require_provider(model)
        normalized = normalize_stream_options(options, cls)
        if isinstance(options, Mapping):
            # JS object spread preserves caller extensions, including transforms.
            for key, value in options.items():
                setattr(normalized, _snake_case(key), value)
        resolution = await self.get_auth(model, AuthResolutionOverrides(
            api_key=getattr(normalized, "api_key", None), env=getattr(normalized, "env", None),
            signal=getattr(normalized, "signal", None),
        ))
        if resolution is None:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")
        auth = resolution.auth
        explicit_key = getattr(normalized, "api_key", None)
        api_key = explicit_key if explicit_key is not None else auth.api_key
        headers = _merge_headers(auth.headers, getattr(normalized, "headers", None))
        transform = cast(HeaderTransform | None, getattr(normalized, "transform_headers", None))
        if transform is not None:
            transformed = transform(headers if headers is not None else {})
            headers = await transformed if inspect.isawaitable(transformed) else transformed
        options_env = getattr(normalized, "env", None)
        env = (
            {**(resolution.env if resolution.env is not None else {}), **(options_env if options_env is not None else {})}
            if resolution.env is not None or options_env is not None else None
        )
        request_model = model
        if auth.base_url:
            request_model = copy.copy(model)
            request_model.base_url = auth.base_url
        if normalized is None:
            request_options = cls()
        elif isinstance(normalized, (ModelsApiStreamOptions, ModelsSimpleStreamOptions, ModelsDeferredFetchOptions, ModelsDeferredCancelOptions)):
            request_options = cls()
            for key, value in vars(normalized).items():
                if key != "transform_headers":
                    setattr(request_options, key, value)
        else:
            request_options = copy.copy(normalized)
            if "transform_headers" in getattr(request_options, "__dict__", {}):
                delattr(request_options, "transform_headers")
        request_options.api_key = api_key
        request_options.headers = headers
        request_options.env = env
        return request_model, request_options

    def stream(
        self, model: Model, context: Context | TranscriptContext,
        options: StreamOptions | Mapping[str, object] | None = None,
    ) -> AssistantMessageEventStream:
        transcript = context if isinstance(context, TranscriptContext) else normalize_context(context)

        async def setup() -> AssistantMessageEventStream:
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(model, options, StreamOptions)
            return provider.stream(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete(
        self, model: Model, context: Context | TranscriptContext,
        options: StreamOptions | Mapping[str, object] | None = None,
    ) -> AssistantMessage:
        return await self.stream(model, context, options).result

    def stream_simple(
        self, model: Model, context: Context | TranscriptContext,
        options: SimpleStreamOptions | Mapping[str, object] | None = None,
    ) -> AssistantMessageEventStream:
        transcript = context if isinstance(context, TranscriptContext) else normalize_context(context)

        async def setup() -> AssistantMessageEventStream:
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(model, options, SimpleStreamOptions)
            return provider.stream_simple(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete_simple(
        self, model: Model, context: Context | TranscriptContext,
        options: SimpleStreamOptions | Mapping[str, object] | None = None,
    ) -> AssistantMessage:
        return await self.stream_simple(model, context, options).result

    def stream_deferred(
        self, model: Model, handle: DeferredHandle,
        options: DeferredFetchOptions | Mapping[str, object] | None = None,
    ) -> AssistantMessageEventStream:
        async def setup() -> AssistantMessageEventStream:
            provider = self._require_provider(model)
            if getattr(provider, "fetch_deferred", None) is None:
                raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
            request_model, request_options = await self._apply_auth(model, options, DeferredFetchOptions)
            return provider.fetch_deferred(request_model, handle, request_options)

        return lazy_stream(model, setup)

    async def fetch_deferred(
        self, model: Model, handle: DeferredHandle,
        options: DeferredFetchOptions | Mapping[str, object] | None = None,
    ) -> AssistantMessage:
        return await self.stream_deferred(model, handle, options).result

    async def cancel_deferred(
        self, model: Model, handle: DeferredHandle,
        options: DeferredCancelOptions | Mapping[str, object] | None = None,
    ) -> None:
        provider = self._require_provider(model)
        if getattr(provider, "cancel_deferred", None) is None:
            raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
        request_model, request_options = await self._apply_auth(model, options, DeferredCancelOptions)
        await provider.cancel_deferred(request_model, handle, request_options)


MutableModels = Models


def create_models(options: CreateModelsOptions | None = None) -> Models:
    return Models(options)


def has_api(model: Model, api: str) -> bool:
    return model.api == api


def calculate_cost(model: Model, usage: Usage) -> Cost:
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates: ModelCostRates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers if model.cost.tiers is not None else []:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above
    long_write = usage.cache_write_1h if usage.cache_write_1h is not None else 0
    short_write = usage.cache_write - long_write
    usage.cost.input = rates.input / 1_000_000 * usage.input
    usage.cost.output = rates.output / 1_000_000 * usage.output
    usage.cost.cache_read = rates.cache_read / 1_000_000 * usage.cache_read
    usage.cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


_EXTENDED_THINKING_LEVELS: list[ModelThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    if not model.reasoning:
        return ["off"]
    mapping = model.thinking_level_map
    return [
        level for level in _EXTENDED_THINKING_LEVELS
        if not (mapping is not None and level in mapping and mapping[level] is None)
        and (level not in ("xhigh", "max") or mapping is not None and level in mapping)
    ]


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    if level not in _EXTENDED_THINKING_LEVELS:
        return available[0] if available else "off"
    requested_index = _EXTENDED_THINKING_LEVELS.index(level)
    for candidate in _EXTENDED_THINKING_LEVELS[requested_index:]:
        if candidate in available:
            return candidate
    for candidate in reversed(_EXTENDED_THINKING_LEVELS[:requested_index]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    return a is not None and b is not None and a.id == b.id and a.provider == b.provider


def _snake_case(value: str) -> str:
    return "".join("_" + character.lower() if character.isupper() else character for character in value)


@overload
def normalize_stream_options(
    options: SimpleStreamOptions | Mapping[str, object] | None,
) -> SimpleStreamOptions | None: ...


@overload
def normalize_stream_options[TOptions: ProviderRequestOptions[Model]](
    options: TOptions | Mapping[str, object] | None, cls: type[TOptions],
) -> TOptions | None: ...


def normalize_stream_options(
    options: ProviderRequestOptions[Model] | Mapping[str, object] | None,
    cls: type[ProviderRequestOptions[Model]] = SimpleStreamOptions,
) -> ProviderRequestOptions[Model] | None:
    """Normalize existing Python mapping options; preserve foreign option carriers."""
    if options is None or not isinstance(options, Mapping):
        return cast(ProviderRequestOptions[Model] | None, options)
    known = {field.name for field in fields(cls)}
    normalized = cls()
    for key, value in options.items():
        snake = _snake_case(key)
        if snake in known:
            setattr(normalized, snake, value)
    return normalized


def with_api_key[TOptions: ProviderRequestOptions[Model]](
    options: TOptions | Mapping[str, object] | None, api_key: str | None, cls: type[TOptions],
) -> TOptions:
    """Retain option extensions when attaching a key to an existing carrier."""
    if options is None:
        return cls(api_key=api_key)
    normalized = cast(TOptions, normalize_stream_options(options, cls))
    if isinstance(normalized, cls) and normalized.api_key is not None:
        return normalized
    resolved = copy.copy(normalized)
    resolved.api_key = api_key
    return resolved


__all__ = [
    "ModelsError", "ModelsErrorCode", "ModelsPublication", "RefreshModelsContext", "ModelsRefreshOptions",
    "ModelsRefreshResult", "ModelsRequestTransforms", "ModelsApiStreamOptions", "ModelsSimpleStreamOptions",
    "ModelsDeferredFetchOptions", "ModelsDeferredCancelOptions", "CreateModelsOptions", "CreateProviderOptions",
    "ProviderAuth", "Provider", "Models", "MutableModels", "create_models", "create_provider", "has_api",
    "calculate_cost", "get_supported_thinking_levels", "clamp_thinking_level", "models_are_equal",
    "normalize_stream_options", "with_api_key",
]
