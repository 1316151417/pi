"""Three-protocol model runtime from ``core/model-runtime.ts``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pi_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pi_ai.api.openai_completions_lazy import openai_completions_api
from pi_ai.api.openai_responses_lazy import openai_responses_api
from pi_ai.auth.types import (
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck,
    AuthOperationOptions, AuthResult, Credential, CredentialInfo, ModelAuth,
    ProviderAuth,
)
from pi_ai.models import (
    CreateModelsOptions, Models, ModelsRefreshOptions, ModelsRefreshResult,
    Provider,
)
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.openai import openai_provider
from pi_ai.types import Context, Model, ProviderStreams, SimpleStreamOptions, TranscriptContext, model_from_json

from .auth_storage import AuthStorage, RuntimeCredentials
from .config import get_auth_path, get_models_path
from .model_config import ModelConfig
from .resolve_config_value import (
    get_missing_config_value_env_var_names, is_command_config_value,
    resolve_config_value_or_throw, resolve_headers_or_throw,
)

_PROTOCOLS: dict[str, ProviderStreams] = {
    "openai-completions": openai_completions_api(),
    "openai-responses": openai_responses_api(),
    "anthropic-messages": anthropic_messages_api(),
}


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _merge_headers(base: Mapping[str, str] | None, override: Mapping[str, str] | None) -> dict[str, str] | None:
    if base is None and override is None:
        return None
    result = dict(base or {})
    for name, value in (override or {}).items():
        for existing in list(result):
            if existing.lower() == name.lower():
                del result[existing]
        result[name] = value
    return result


def _model_from_definition(
    provider_id: str, definition: Mapping[str, object],
    config: Mapping[str, object], defaults: Model | None,
) -> Model:
    model_id = cast(str, definition["id"])
    api = definition.get("api") or config.get("api") or (defaults.api if defaults else None)
    if not isinstance(api, str):
        raise ValueError(f'Provider {provider_id}, model {model_id}: no "api" specified. Set at provider or model level.')
    if api not in _PROTOCOLS:
        raise ValueError(f"Provider {provider_id}, model {model_id}: unsupported API {api}")
    base_url = definition.get("baseUrl") or config.get("baseUrl") or (defaults.base_url if defaults else None)
    if not isinstance(base_url, str) or not base_url:
        raise ValueError(f'Provider {provider_id}: "baseUrl" is required when defining custom models.')
    for key in ("contextWindow", "maxTokens"):
        value = definition.get(key)
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError(f"Provider {provider_id}, model {model_id}: invalid {key}")
    compat = {**cast(dict[str, object], _thaw(config.get("compat") or {})),
              **cast(dict[str, object], _thaw(definition.get("compat") or {}))}
    raw: dict[str, object] = {
        "id": model_id, "name": definition.get("name") or model_id,
        "api": api, "provider": provider_id, "baseUrl": base_url,
        "reasoning": definition.get("reasoning", False),
        "input": _thaw(definition.get("input", ["text"])),
        "cost": _thaw(definition.get("cost", {
            "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
        })),
        "contextWindow": definition.get("contextWindow", 128000),
        "maxTokens": definition.get("maxTokens", 16384),
    }
    for key in ("thinkingLevelMap", "samplingParams"):
        if key in definition:
            raw[key] = _thaw(definition[key])
    if compat:
        raw["compat"] = compat
    return model_from_json(raw)


def _compose_models(provider_id: str, base: Provider | None, config: Mapping[str, object]) -> list[Model]:
    models = [] if base is None else [
        model_from_json({
            **model.to_json(),
            "baseUrl": config.get("baseUrl") or model.base_url,
            "compat": {**(model.compat or {}), **cast(dict[str, object], _thaw(config.get("compat") or {}))},
        })
        for model in base.get_models()
    ]
    definitions = config.get("models")
    for definition in definitions if isinstance(definitions, tuple) else []:
        if not isinstance(definition, Mapping):
            continue
        model_id = cast(str, definition["id"])
        defaults = next((model for model in models if model.id == model_id), None)
        if defaults is None:
            preferred_api = definition.get("api") or config.get("api")
            defaults = next((model for model in models if model.api == preferred_api), None)
        if defaults is None:
            defaults = next((model for model in models if model.api == "openai-completions"), None)
        if defaults is None and models:
            defaults = models[0]
        replacement = _model_from_definition(provider_id, definition, config, defaults)
        index = next((index for index, model in enumerate(models) if model.id == model_id), None)
        if index is None:
            models.append(replacement)
        else:
            models[index] = replacement
    overrides = config.get("modelOverrides")
    if isinstance(overrides, Mapping):
        updated: list[Model] = []
        for model in models:
            override = overrides.get(model.id)
            if not isinstance(override, Mapping):
                updated.append(model)
                continue
            raw = model.to_json()
            for key, value in override.items():
                if key in ("cost", "thinkingLevelMap", "samplingParams", "compat") and isinstance(value, Mapping):
                    raw[key] = {**cast(dict[str, object], raw.get(key) or {}), **cast(dict[str, object], _thaw(value))}
                else:
                    raw[key] = _thaw(value)
            updated.append(model_from_json(raw))
        models = updated
    return models


def _compose_auth(provider_id: str, base: Provider | None, config: Mapping[str, object]) -> ProviderAuth:
    inherited = base.auth.api_key if base is not None else None
    oauth = base.auth.oauth if base is not None else None
    raw_key = config.get("apiKey")
    raw_headers = config.get("headers")
    auth_header = bool(config.get("authHeader", False))
    if inherited is None and raw_key is None and oauth is not None:
        return ProviderAuth(oauth=oauth)

    async def check(input_value: ApiKeyAuthInput) -> AuthCheck | None:
        input_value.signal.throw_if_aborted()
        if input_value.credential is not None:
            if inherited is not None and inherited.check is not None:
                return await inherited.check(input_value)
            if input_value.credential.key:
                return AuthCheck(type="api_key", source="stored credential")
            if inherited is not None:
                resolved = await inherited.resolve(input_value)
                return AuthCheck(type="api_key", source=resolved.source) if resolved else None
        if isinstance(raw_key, str):
            if is_command_config_value(raw_key):
                return AuthCheck(type="api_key", source="configured API key")
            for name in get_missing_config_value_env_var_names(raw_key):
                if await input_value.ctx.env(name) is None:
                    return None
            return AuthCheck(type="api_key", source="configured API key")
        if inherited is not None:
            if inherited.check is not None:
                return await inherited.check(input_value)
            resolved = await inherited.resolve(input_value)
            return AuthCheck(type="api_key", source=resolved.source) if resolved else None
        return None

    async def resolve(input_value: ApiKeyAuthInput) -> AuthResult | None:
        input_value.signal.throw_if_aborted()
        result: AuthResult | None = None
        if input_value.credential is not None:
            if inherited is not None:
                result = await inherited.resolve(input_value)
            elif input_value.credential.key:
                result = AuthResult(ModelAuth(api_key=input_value.credential.key),
                                    env=input_value.credential.env, source="stored credential")
        elif isinstance(raw_key, str):
            key = resolve_config_value_or_throw(raw_key, f'API key for provider "{provider_id}"')
            if inherited is not None:
                result = await inherited.resolve(ApiKeyAuthInput(
                    ctx=input_value.ctx, signal=input_value.signal,
                    credential=ApiKeyCredential(key=key),
                ))
            else:
                result = AuthResult(ModelAuth(api_key=key), source="configured API key")
        elif inherited is not None:
            result = await inherited.resolve(input_value)
        if result is None:
            return None
        headers = resolve_headers_or_throw(
            cast(Mapping[str, str] | None, raw_headers) if isinstance(raw_headers, Mapping) else None,
            f'provider "{provider_id}"', result.env,
        )
        merged = _merge_headers(result.auth.headers, headers)
        if auth_header:
            if not result.auth.api_key:
                raise RuntimeError("authHeader requires a resolved API key")
            merged = _merge_headers(merged, {"Authorization": f"Bearer {result.auth.api_key}"})
        return AuthResult(
            ModelAuth(api_key=result.auth.api_key, headers=merged, base_url=result.auth.base_url),
            env=result.env, source=result.source,
        )

    return ProviderAuth(
        api_key=ApiKeyAuth(name=inherited.name if inherited else "API key", resolve=resolve,
                           login=inherited.login if inherited else None, check=check),
        oauth=oauth,
    )


@dataclass
class CreateModelRuntimeOptions:
    credentials: AuthStorage | RuntimeCredentials | None = None
    auth_path: str | None = None
    models_path: str | None = None
    refresh_on_create: bool = True


class ModelRuntime:
    def __init__(
        self, credentials: RuntimeCredentials, config: ModelConfig,
        models_path: str | None,
    ) -> None:
        self.credentials = credentials
        self.config = config
        self.models_path = models_path
        self.models = Models(CreateModelsOptions(credentials=credentials))
        self._builtins = {provider.id: provider for provider in (anthropic_provider(), openai_provider())}
        self._composition_errors: dict[str, str] = {}
        self._available_snapshot: list[Model] = []
        self._rebuild_providers()

    @classmethod
    async def create(cls, options: CreateModelRuntimeOptions | None = None) -> ModelRuntime:
        chosen = options or CreateModelRuntimeOptions()
        credentials = chosen.credentials or AuthStorage.create(chosen.auth_path or get_auth_path())
        runtime_credentials = credentials if isinstance(credentials, RuntimeCredentials) else RuntimeCredentials(credentials)
        models_path = chosen.models_path if chosen.models_path is not None else get_models_path()
        config = await ModelConfig.load(models_path)
        runtime = cls(runtime_credentials, config, models_path)
        if chosen.refresh_on_create:
            await runtime.refresh()
        return runtime

    def _rebuild_providers(self) -> None:
        self.models.clear_providers()
        self._composition_errors.clear()
        for provider_id in dict.fromkeys([*self._builtins, *self.config.get_provider_ids()]):
            base = self._builtins.get(provider_id)
            config = self.config.get_provider(provider_id)
            if config is None:
                if base is not None:
                    self.models.set_provider(base)
                continue
            try:
                models = _compose_models(provider_id, base, config)
                if any(model.api not in _PROTOCOLS for model in models):
                    raise ValueError(f"Provider {provider_id}: unsupported API")
                provider = Provider(
                    id=provider_id, name=cast(str, config.get("name") or (base.name if base else provider_id)),
                    models=models, auth=_compose_auth(provider_id, base, config),
                    api={api: implementation for api, implementation in _PROTOCOLS.items()},
                    base_url=cast(str | None, config.get("baseUrl") or (base.base_url if base else None)),
                )
                self.models.set_provider(provider)
            except Exception as error:
                self._composition_errors[provider_id] = str(error)
                if base is not None:
                    self.models.set_provider(base)

    async def refresh(self, options: ModelsRefreshOptions | None = None) -> ModelsRefreshResult:
        self.config = await ModelConfig.load(self.models_path)
        self._rebuild_providers()
        chosen = options or ModelsRefreshOptions(allow_network=False)
        if chosen.allow_network is None:
            chosen.allow_network = False
        result = await self.models.refresh(chosen)
        self._available_snapshot = await self.models.get_available()
        return result

    def get_error(self) -> str | None:
        errors = [self.config.error] if self.config.error else []
        errors.extend(f'Provider "{name}": {error}' for name, error in self._composition_errors.items())
        return "\n\n".join(errors) if errors else None

    def get_models(self, provider_id: str | None = None) -> Sequence[Model]:
        return self.models.get_models(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        return self.models.get_model(provider_id, model_id)

    def get_available_snapshot(self) -> list[Model]:
        return list(self._available_snapshot)

    async def get_available(self, provider_id: str | None = None) -> list[Model]:
        available = await self.models.get_available(provider_id)
        if provider_id is None:
            self._available_snapshot = list(available)
        return list(available)

    async def is_configured(self, provider_id: str) -> bool:
        return await self.models.is_configured(provider_id)

    async def check_auth(self, provider_id: str) -> AuthCheck | None:
        return await self.models.check_auth(provider_id)

    async def get_auth(self, provider_or_model: str | Model) -> AuthResult | None:
        return await self.models.get_auth(provider_or_model)

    async def list_credentials(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        return await self.credentials.list(options)

    async def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        self.credentials.set_runtime_api_key(provider_id, api_key)
        self._available_snapshot = await self.models.get_available()

    async def remove_runtime_api_key(self, provider_id: str) -> None:
        self.credentials.remove_runtime_api_key(provider_id)
        self._available_snapshot = await self.models.get_available()

    def stream_simple(
        self, model: Model, context: Context | TranscriptContext,
        options: SimpleStreamOptions | Mapping[str, object] | None = None,
    ) -> AssistantMessageEventStream:
        return self.models.stream_simple(model, context, options)


__all__ = ["CreateModelRuntimeOptions", "ModelRuntime"]
