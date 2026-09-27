"""The global stream registry and dispatch functions from ``compat.ts``."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from .._json_runtime import JS_WHITESPACE
from .._random import random_base36
from ..api.anthropic_messages_lazy import anthropic_messages_api
from ..api.azure_openai_responses_lazy import azure_openai_responses_api
from ..api.google_generative_ai_lazy import google_generative_ai_api
from ..api.openai_completions_lazy import openai_completions_api
from ..api.openai_responses_lazy import openai_responses_api
from ..api.pi_messages_lazy import pi_messages_api
from ..env_api_keys import get_env_api_key
from ..event_stream import AssistantMessageEventStream
from ..models import Provider
from ..providers.all import builtin_models, get_builtin_model, get_builtin_models, get_builtin_providers
from ..providers.faux import (
    FauxProviderHandle, FauxProviderRegistration, FauxResponseStep,
    RegisterFauxProviderOptions, create_faux_core,
)
from ..transcript import normalize_context
from ..types import (
    AssistantMessage, Context, Model, SimpleStreamFunction, SimpleStreamOptions,
    StreamFunction, StreamOptions, TranscriptContext,
)

get_model = get_builtin_model
get_models = get_builtin_models
get_providers = get_builtin_providers
type ApiStreamFunction = StreamFunction
type ApiStreamSimpleFunction = SimpleStreamFunction


@dataclass
class ApiProvider:
    api: str
    stream: StreamFunction
    stream_simple: SimpleStreamFunction


@dataclass
class _RegisteredApiProvider:
    provider: ApiProvider
    source_id: str | None = None


_registry: dict[str, _RegisteredApiProvider] = {}


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    api, stream_function, simple_function = provider.api, provider.stream, provider.stream_simple

    def wrapped_stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
        if model.api != api:
            raise RuntimeError(f"Mismatched api: {model.api} expected {api}")
        return stream_function(model, context, options)

    def wrapped_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
        if model.api != api:
            raise RuntimeError(f"Mismatched api: {model.api} expected {api}")
        return simple_function(model, context, options)

    _registry[api] = _RegisteredApiProvider(ApiProvider(api, wrapped_stream, wrapped_simple), source_id)


def get_api_provider(api: str) -> ApiProvider | None:
    registered = _registry.get(api)
    return registered.provider if registered is not None else None


def get_api_providers() -> list[ApiProvider]:
    return [entry.provider for entry in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    for api, entry in list(_registry.items()):
        if entry.source_id == source_id:
            del _registry[api]


class _FauxRegistration(FauxProviderRegistration):
    def __init__(self, core: FauxProviderHandle, source_id: str) -> None:
        self._core = core
        self.api, self.models, self.state = core.api, core.models, core.state
        self.unregister = lambda: unregister_api_providers(source_id)

    @overload
    def get_model(self, model_id: None = None) -> Model: ...

    @overload
    def get_model(self, model_id: str) -> Model | None: ...

    def get_model(self, model_id: str | None = None) -> Model | None:
        return self._core.get_model(model_id)

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        self._core.set_responses(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        self._core.append_responses(responses)

    def get_pending_response_count(self) -> int:
        return self._core.get_pending_response_count()


def register_faux_provider(options: RegisterFauxProviderOptions | None = None) -> FauxProviderRegistration:
    core, streams = create_faux_core(options if options is not None else RegisterFauxProviderOptions())
    source_id = "faux-provider-" + random_base36()[:8]
    register_api_provider(ApiProvider(core.api, streams["stream"], streams["stream_simple"]), source_id)
    return _FauxRegistration(core, source_id)


_BUILTIN_APIS = [
    ("anthropic-messages", anthropic_messages_api()),
    ("openai-completions", openai_completions_api()),
    ("openai-responses", openai_responses_api()),
    ("azure-openai-responses", azure_openai_responses_api()),
    ("google-generative-ai", google_generative_ai_api()),
    ("pi-messages", pi_messages_api()),
]
_builtin_api_provider_instances: dict[str, ApiProvider | None] = {}


def register_built_in_api_providers() -> None:
    for api, streams in _BUILTIN_APIS:
        if get_api_provider(api) is None:
            register_api_provider(ApiProvider(api, streams.stream, streams.stream_simple))
        _builtin_api_provider_instances[api] = get_api_provider(api)


def reset_api_providers() -> None:
    _registry.clear()
    _builtin_api_provider_instances.clear()
    register_built_in_api_providers()


register_built_in_api_providers()
_compat_models = builtin_models()


def _has_explicit_api_key(api_key: str | None) -> bool:
    return isinstance(api_key, str) and bool(api_key.strip(JS_WHITESPACE))


def _with_env_api_key[T: StreamOptions](model: Model, options: T | None, default: Callable[[], T]) -> T | None:
    if _has_explicit_api_key(options.api_key if options is not None else None):
        return options
    api_key = get_env_api_key(model.provider, options.env if options is not None else None)
    if not api_key or api_key == "<authenticated>":
        return options
    updated = copy.copy(options) if options is not None else default()
    updated.api_key = api_key
    return updated


def _has_resolved_cloudflare_auth(options: StreamOptions | None) -> bool:
    return options is not None and (
        _has_explicit_api_key(options.api_key)
        or options.headers is not None and isinstance(options.headers.get("cf-aig-authorization"), str)
    )


def _get_builtin_provider_for_model(model: Model) -> Provider | None:
    if get_api_provider(model.api) is not _builtin_api_provider_instances.get(model.api):
        return None
    provider = _compat_models.get_provider(model.provider)
    return provider if provider is not None and any(candidate.api == model.api for candidate in provider.get_models()) else None


def _resolve_api_provider(api: str) -> ApiProvider:
    provider = get_api_provider(api)
    if provider is None:
        raise RuntimeError(f"No API provider registered for api: {api}")
    return provider


def stream(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    transcript = normalize_context(context)
    builtin = _get_builtin_provider_for_model(model)
    if builtin is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _compat_models.stream(model, transcript, options)
        return builtin.stream(model, transcript, _with_env_api_key(model, options, StreamOptions))
    return _resolve_api_provider(model.api).stream(model, transcript, _with_env_api_key(model, options, StreamOptions))


async def complete(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
    return await stream(model, context, options).result


def stream_simple(model: Model, context: Context, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
    transcript = normalize_context(context)
    builtin = _get_builtin_provider_for_model(model)
    if builtin is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _compat_models.stream_simple(model, transcript, options)
        return builtin.stream_simple(model, transcript, _with_env_api_key(model, options, SimpleStreamOptions))
    return _resolve_api_provider(model.api).stream_simple(model, transcript, _with_env_api_key(model, options, SimpleStreamOptions))


async def complete_simple(model: Model, context: Context, options: SimpleStreamOptions | None = None) -> AssistantMessage:
    return await stream_simple(model, context, options).result


__all__ = [
    "get_model", "get_models", "get_providers", "ApiStreamFunction", "ApiStreamSimpleFunction", "ApiProvider",
    "register_api_provider", "get_api_provider", "get_api_providers", "unregister_api_providers", "register_faux_provider",
    "register_built_in_api_providers", "reset_api_providers", "stream", "complete", "stream_simple", "complete_simple",
]
