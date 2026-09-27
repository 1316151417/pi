"""Radius gateway configuration and model catalog from ``radius-config.ts``."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, cast

from ..abort import AbortSignal
from ..auth.oauth._common import JS_WHITESPACE
from ..auth.oauth._http import fetch, parse_url
from ..auth.types import OAuthCredential
from ..types import Model, model_from_json
from ..utils._javascript import utf16_length, utf16_slice

DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"


class RadiusGatewayModel(TypedDict):
    id: str
    name: str
    reasoning: bool
    thinkingLevelMap: NotRequired[dict[str, str | None]]
    input: list[Literal["text", "image"]]
    cost: dict[str, object]
    contextWindow: int | float
    maxTokens: int | float


@dataclass
class RadiusGatewayConfig:
    base_url: str
    models: list[RadiusGatewayModel]

    def to_json(self) -> dict[str, object]:
        return {"baseUrl": self.base_url, "models": self.models}


# Provider-specific credential fields use the common credential's wire-key
# extension map, so gatewayConfig retains its object identity until sanitized.
type RadiusOAuthCredential = OAuthCredential


def _is_radius_gateway_model(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        isinstance(value.get("id"), str) and isinstance(value.get("name"), str)
        and isinstance(value.get("reasoning"), bool) and isinstance(value.get("input"), list)
        and isinstance(value.get("cost"), Mapping)
        and type(value.get("contextWindow")) in (int, float)
        and type(value.get("maxTokens")) in (int, float)
    )


def _sanitize_radius_gateway_config(config: object) -> RadiusGatewayConfig | None:
    if isinstance(config, RadiusGatewayConfig):
        config = config.to_json()
    if not isinstance(config, Mapping):
        return None
    base_url, models = config.get("baseUrl"), config.get("models")
    if not isinstance(base_url, str) or not isinstance(models, list):
        return None
    return RadiusGatewayConfig(base_url, [cast(RadiusGatewayModel, dict(cast(Mapping[str, object], model))) for model in models if _is_radius_gateway_model(model)])


def normalize_radius_gateway_url(value: str) -> str:
    with_scheme = value if re.match(r"^https?://", value, re.IGNORECASE) else f"https://{value}"
    return re.sub(r"/+\Z", "", with_scheme)


def get_radius_credential_config(credential: OAuthCredential | None) -> RadiusGatewayConfig | None:
    return _sanitize_radius_gateway_config(credential.extra.get("gatewayConfig") if credential is not None else None)


def get_radius_models_from_config(provider_id: str, config: RadiusGatewayConfig) -> list[Model]:
    return [model_from_json({**model, "api": "pi-messages", "provider": provider_id, "baseUrl": config.base_url}) for model in config.models]


def get_radius_models(provider_id: str, credential: OAuthCredential | None) -> list[Model]:
    config = get_radius_credential_config(credential)
    return get_radius_models_from_config(provider_id, config) if config is not None else []


async def load_radius_gateway_config(gateway: str, api_key: str | None = None, signal: AbortSignal | None = None) -> RadiusGatewayConfig:
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    response = await fetch(parse_url("/v1/config", gateway).href, headers=headers, signal=signal)
    if not response.ok:
        body = (await response.text()).strip(JS_WHITESPACE)
        if utf16_length(body) > 512:
            body = utf16_slice(body, 512) + "…"
        raise RuntimeError(f"Could not load Radius config from {gateway}: {response.status}: {body}")
    config = _sanitize_radius_gateway_config(await response.json())
    if config is None:
        raise RuntimeError(f"Invalid Radius config from {gateway}")
    return config


__all__ = [
    "DEFAULT_RADIUS_GATEWAY", "RadiusGatewayModel", "RadiusGatewayConfig", "RadiusOAuthCredential",
    "normalize_radius_gateway_url", "get_radius_credential_config", "get_radius_models_from_config",
    "get_radius_models", "load_radius_gateway_config",
]
