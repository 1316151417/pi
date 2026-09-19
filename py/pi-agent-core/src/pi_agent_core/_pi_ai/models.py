"""Models registry ported from pi-ai ``src/models-store.ts`` / ``src/models.ts``.

Runtime collection of providers plus auth application and stream convenience.
Providers own stream behavior; :class:`Models` resolves auth and delegates each
request to the provider that owns the model.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .event_stream import AssistantMessageEventStream
from .types import (
    AssistantMessage,
    Context,
    DeferredHandle,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from .transcript import normalize_context

__all__ = ["Provider", "Models", "create_models", "calculate_cost", "normalize_stream_options"]


@dataclass
class ProviderAuth:
    """API-key auth semantics; ``resolve`` reports whether the provider is configured."""

    env_var: Optional[str] = None
    explicit_key: Optional[str] = None

    def resolve(self) -> Optional[str]:
        if self.explicit_key is not None:
            return self.explicit_key
        if self.env_var is not None:
            return os.environ.get(self.env_var)
        return None


class Provider:
    """Concrete runtime unit owning metadata, auth, model list, and stream behavior."""

    def __init__(
        self,
        id: str,
        name: str,
        models: Sequence[Model],
        stream: Callable[..., AssistantMessageEventStream],
        stream_simple: Optional[Callable[..., AssistantMessageEventStream]] = None,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, Optional[str]]] = None,
        api_key_env_var: Optional[str] = None,
        api_key: Optional[str] = None,
        fetch_deferred: Optional[Callable[..., AssistantMessageEventStream]] = None,
        cancel_deferred: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.id = id
        self.name = name
        self._models = list(models)
        self._stream = stream
        self._stream_simple = stream_simple or stream
        self.fetch_deferred = fetch_deferred
        self.cancel_deferred = cancel_deferred
        self.base_url = base_url
        self.headers = headers
        self.auth = ProviderAuth(env_var=api_key_env_var, explicit_key=api_key)

    def get_models(self) -> List[Model]:
        return list(self._models)

    def stream(
        self,
        model: Model,
        context: TranscriptContext,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return self._stream(model, context, options)

    def stream_simple(
        self,
        model: Model,
        context: TranscriptContext,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return self._stream_simple(model, context, options)


def create_provider(
    id: str,
    name: str,
    models: Sequence[Model],
    stream: Callable[..., AssistantMessageEventStream],
    **kwargs: Any,
) -> Provider:
    return Provider(id=id, name=name, models=models, stream=stream, **kwargs)


def calculate_cost(model: Model, usage: Usage) -> None:
    """Populate ``usage.cost`` from the model's cost rates ($/million tokens)."""
    per_million = 1_000_000
    usage.cost.input = usage.input * model.cost.input / per_million
    usage.cost.output = usage.output * model.cost.output / per_million
    usage.cost.cache_read = usage.cache_read * model.cost.cache_read / per_million
    usage.cost.cache_write = usage.cache_write * model.cost.cache_write / per_million
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )


class Models:
    """Runtime collection of providers plus auth application and stream convenience."""

    def __init__(self) -> None:
        self._providers: Dict[str, Provider] = {}

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        self._providers[provider.id] = provider

    def remove_provider(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def get_providers(self) -> List[Provider]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> Optional[Provider]:
        return self._providers.get(id)

    # ------------------------------------------------------------------
    # Model lookup
    # ------------------------------------------------------------------

    def get_models(self, provider: Optional[str] = None) -> List[Model]:
        if provider is not None:
            found = self._providers.get(provider)
            if found is None:
                return []
            try:
                return found.get_models()
            except Exception:  # noqa: BLE001 - best-effort like the TS store
                return []
        models: List[Model] = []
        for candidate in self._providers.values():
            try:
                models.extend(candidate.get_models())
            except Exception:  # noqa: BLE001
                continue
        return models

    def get_model(self, provider: str, id: str) -> Optional[Model]:
        return next((m for m in self.get_models(provider) if m.id == id), None)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def get_auth(self, provider_id: str) -> Optional[str]:
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        return provider.auth.resolve()

    def is_configured(self, provider_id: str) -> bool:
        return self.get_auth(provider_id) is not None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        model: Model,
        context: Any,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        provider = self._providers.get(model.provider)
        if provider is None:
            raise RuntimeError(f"No provider registered for {model.provider!r}")
        options = normalize_stream_options(options, StreamOptions)
        api_key = options.api_key if options and options.api_key else provider.auth.resolve()
        resolved_options = options
        if api_key is not None and (options is None or options.api_key is None):
            resolved_options = (
                dataclasses.replace(options, api_key=api_key) if options else StreamOptions(api_key=api_key)
            )
        normalized = context if isinstance(context, TranscriptContext) else normalize_context(context)
        return provider.stream(model, normalized, resolved_options)

    def stream_simple(
        self,
        model: Model,
        context: Any,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        provider = self._providers.get(model.provider)
        if provider is None:
            raise RuntimeError(f"No provider registered for {model.provider!r}")
        options = normalize_stream_options(options)
        api_key = options.api_key if options and options.api_key else provider.auth.resolve()
        resolved_options = options
        if api_key is not None and (options is None or options.api_key is None):
            resolved_options = (
                SimpleStreamOptions(**{**vars(options or SimpleStreamOptions()), "api_key": api_key})
            )
        normalized = context if isinstance(context, TranscriptContext) else normalize_context(context)
        return provider.stream_simple(model, normalized, resolved_options)

    def stream_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        """Resume one provider-deferred response under its handle."""
        provider = self._providers.get(model.provider)
        if provider is None:
            raise RuntimeError(f"No provider registered for {model.provider!r}")
        if provider.fetch_deferred is None:
            raise RuntimeError(
                f"Provider {model.provider} does not support deferred responses"
            )
        options = normalize_stream_options(options)
        api_key = options.api_key if options and options.api_key else provider.auth.resolve()
        resolved_options = options
        if api_key is not None and (options is None or options.api_key is None):
            resolved_options = SimpleStreamOptions(
                **{**vars(options or SimpleStreamOptions()), "api_key": api_key}
            )
        return provider.fetch_deferred(model, handle, resolved_options)

    def complete(
        self,
        model: Model,
        context: Any,
        options: Optional[StreamOptions] = None,
    ) -> "AssistantMessage":
        return self.stream(model, context, options).result()

    def complete_simple(
        self,
        model: Model,
        context: Any,
        options: Optional[SimpleStreamOptions] = None,
    ) -> "AssistantMessage":
        return self.stream_simple(model, context, options).result()

    def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessage:
        return self.stream_deferred(model, handle, options).result()

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: Optional[SimpleStreamOptions] = None,
    ) -> None:
        """Ask the provider to abandon one deferred response."""
        provider = self._providers.get(model.provider)
        if provider is None:
            raise RuntimeError(f"No provider registered for {model.provider!r}")
        if provider.cancel_deferred is None:
            raise RuntimeError(f"API cannot cancel deferred responses")
        options = normalize_stream_options(options)
        api_key = options.api_key if options and options.api_key else provider.auth.resolve()
        resolved_options = options
        if api_key is not None and (options is None or options.api_key is None):
            resolved_options = SimpleStreamOptions(
                **{**vars(options or SimpleStreamOptions()), "api_key": api_key}
            )
        await provider.cancel_deferred(model, handle, resolved_options)


def normalize_stream_options(options: Any, cls: Any = SimpleStreamOptions) -> Any:
    """Coerce a plain option mapping into the typed options object the providers read.

    The TypeScript API takes structural option objects; Python providers read
    dataclass fields, so the registry is the one place that bridges the two.
    Unknown keys are dropped rather than raising, mirroring TS structural typing.
    """
    if options is None or not isinstance(options, dict):
        return options
    known = {field.name for field in dataclasses.fields(cls)}
    values: Dict[str, Any] = {}
    for key, value in options.items():
        snake = _snake_case(key)
        if snake in known:
            values[snake] = value
    return cls(**values)


def _snake_case(value: str) -> str:
    out = []
    for char in value:
        if char.isupper():
            out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out)


def create_models() -> Models:
    return Models()
