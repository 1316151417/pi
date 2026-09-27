"""Image provider collection and auth application from ``images-models.ts``."""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

from .auth.context import default_provider_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.resolve import AuthResolutionOverrides, ModelsError, resolve_provider_auth
from .auth.types import AuthResult, ProviderAuth
from .models import CreateModelsOptions
from .types import (
    AssistantImages, ImagesContext, ImagesFunction, ImagesModel, ImagesOptions,
    ProviderImages,
)


@dataclass
class ImagesProvider:
    id: str
    name: str
    auth: ProviderAuth
    get_models: Callable[[], Sequence[ImagesModel]]
    generate_images: ImagesFunction
    refresh_models: Callable[[], Awaitable[None]] | None = None


@dataclass
class CreateImagesProviderOptions:
    id: str
    auth: ProviderAuth
    models: Sequence[ImagesModel]
    api: ProviderImages
    name: str | None = None
    refresh_models: Callable[[], Awaitable[Sequence[ImagesModel]]] | None = None


@dataclass
class _ProviderEntry:
    key: str
    value: ImagesProvider
    active: bool = True


class _ProviderMap:
    """Maintain live JS Map iteration through callback insertions/deletions."""

    def __init__(self) -> None:
        self._entries: dict[str, _ProviderEntry] = {}
        self._order: list[_ProviderEntry] = []
        self._iterators = 0

    def get(self, key: str) -> ImagesProvider | None:
        entry = self._entries.get(key)
        return entry.value if entry is not None else None

    def set(self, key: str, value: ImagesProvider) -> None:
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

    def values(self) -> Iterator[ImagesProvider]:
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


async def _await_promise[T](operation: Awaitable[T] | T) -> T:
    if isinstance(operation, Coroutine):
        pending = asyncio.Task(operation, loop=asyncio.get_running_loop(), eager_start=True)
    elif inspect.isawaitable(operation):
        pending = asyncio.ensure_future(operation)
    else:
        await asyncio.sleep(0)
        return operation
    if pending.done():
        await asyncio.sleep(0)
    return await asyncio.shield(pending)


class ImagesModels:
    def __init__(self, options: CreateModelsOptions | None = None) -> None:
        self._providers = _ProviderMap()
        self._credentials = options.credentials if options is not None and options.credentials is not None else InMemoryCredentialStore()
        self._auth_context = options.auth_context if options is not None and options.auth_context is not None else default_provider_auth_context()

    def get_providers(self) -> Sequence[ImagesProvider]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> ImagesProvider | None:
        return self._providers.get(id)

    def get_models(self, provider: str | None = None) -> Sequence[ImagesModel]:
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None:
                return []
            try:
                return entry.get_models()
            except (Exception, asyncio.CancelledError):
                return []
        models: list[ImagesModel] = []
        for entry in self._providers.values():
            try:
                # JS evaluates the spread before push, so an iterator failure
                # cannot partially append the failed provider's list.
                models.extend(list(entry.get_models()))
            except (Exception, asyncio.CancelledError):
                pass
        return models

    def get_model(self, provider: str, id: str) -> ImagesModel | None:
        return next((model for model in self.get_models(provider) if model.id == id), None)

    async def refresh(self, provider: str | None = None) -> None:
        if provider is not None:
            entry = self._providers.get(provider)
            if entry is None or entry.refresh_models is None:
                return
            try:
                await _await_promise(entry.refresh_models())
            except (Exception, asyncio.CancelledError) as error:
                if isinstance(error, ModelsError):
                    raise
                raise ModelsError("model_source", f"Model refresh failed for {provider}", cause=error) from error
            return

        async def refresh_entry(entry: ImagesProvider) -> None:
            if entry.refresh_models is not None:
                await _await_promise(entry.refresh_models())

        pending: list[asyncio.Task[None]] = []
        for entry in self._providers.values():
            pending.append(asyncio.Task(refresh_entry(entry), loop=asyncio.get_running_loop(), eager_start=True))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        else:
            await asyncio.sleep(0)

    async def get_auth(
        self, provider_or_model: str | ImagesModel, overrides: AuthResolutionOverrides | None = None,
    ) -> AuthResult | None:
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        return await _await_promise(resolve_provider_auth(provider, self._credentials, self._auth_context, overrides))

    async def generate_images(
        self, model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None,
    ) -> AssistantImages:
        delegated: Awaitable[AssistantImages]
        try:
            provider = self._providers.get(model.provider)
            if provider is None:
                raise ModelsError("provider", f"Unknown provider: {model.provider}")
            resolution = await _await_promise(self.get_auth(model, AuthResolutionOverrides(
                api_key=options.api_key if options is not None else None,
                env=options.env if options is not None else None,
                signal=options.signal if options is not None else None,
            )))
            auth = resolution.auth if resolution is not None else None
            if auth is None:
                delegated = provider.generate_images(model, context, options)
            else:
                request_model = copy.copy(model) if auth.base_url else model
                if auth.base_url:
                    request_model.base_url = auth.base_url
                api_key = options.api_key if options is not None and options.api_key is not None else auth.api_key
                option_headers = options.headers if options is not None else None
                headers = {**(auth.headers or {}), **(option_headers or {})} if auth.headers is not None or option_headers is not None else None
                resolved_env = cast(AuthResult, resolution).env
                option_env = options.env if options is not None else None
                env = {**(resolved_env or {}), **(option_env or {})} if resolved_env is not None or option_env is not None else None
                request_options = copy.copy(options) if options is not None else ImagesOptions()
                request_options.api_key = api_key
                request_options.headers = headers
                request_options.env = env
                return await _await_promise(provider.generate_images(request_model, context, request_options))
        except (Exception, asyncio.CancelledError) as error:
            return AssistantImages(
                api=model.api, provider=model.provider, model=model.id, output=[], stop_reason="error",
                error_message=str(error), timestamp=time.time_ns() // 1_000_000,
            )
        # The source returns the provider promise without awaiting in its try
        # when no auth is resolved. Preserve the resulting rejection boundary.
        return await _await_promise(delegated)


class MutableImagesModels(ImagesModels):
    def set_provider(self, provider: ImagesProvider) -> None:
        self._providers.set(provider.id, provider)

    def delete_provider(self, id: str) -> None:
        self._providers.delete(id)

    def clear_providers(self) -> None:
        self._providers.clear()


def create_images_models(options: CreateModelsOptions | None = None) -> MutableImagesModels:
    return MutableImagesModels(options)


def create_images_provider(input: CreateImagesProviderOptions) -> ImagesProvider:
    models = input.models
    inflight_refresh: asyncio.Task[None] | None = None
    fetch_models = input.refresh_models

    def get_models() -> Sequence[ImagesModel]:
        return models

    def refresh_models() -> Awaitable[None]:
        nonlocal inflight_refresh
        if inflight_refresh is None:
            async def run() -> None:
                nonlocal models, inflight_refresh
                try:
                    models = await _await_promise(cast(Callable[[], Awaitable[Sequence[ImagesModel]]], fetch_models)())
                finally:
                    inflight_refresh = None

            inflight_refresh = asyncio.Task(run(), loop=asyncio.get_running_loop(), eager_start=True)
        return inflight_refresh

    def generate_images(model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None) -> Awaitable[AssistantImages]:
        return input.api.generate_images(model, context, options)

    return ImagesProvider(
        id=input.id, name=input.name if input.name is not None else input.id, auth=input.auth,
        get_models=get_models, generate_images=generate_images,
        refresh_models=refresh_models if fetch_models is not None else None,
    )


__all__ = [
    "ImagesProvider", "ImagesModels", "MutableImagesModels", "CreateImagesProviderOptions",
    "create_images_models", "create_images_provider",
]
