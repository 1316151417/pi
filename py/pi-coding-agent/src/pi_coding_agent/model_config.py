"""Credential-blind ``models.json`` snapshot from ``core/model-config.ts``."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast

from .paths import normalize_path

_COMMENTS = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*')
_TRAILING_COMMAS = re.compile(r'"(?:\\.|[^"\\])*"|,(\s*[}\]])')


def strip_json_comments(content: str) -> str:
    without_comments = _COMMENTS.sub(lambda match: match.group() if match.group().startswith('"') else "", content)
    return _TRAILING_COMMAS.sub(
        lambda match: match.group(1) if match.group(1) is not None else match.group(),
        without_comments,
    )


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _validate_provider(provider_id: str, provider: object) -> list[str]:
    path = f"providers.{provider_id}"
    if not isinstance(provider, dict):
        return [f"{path}: Expected object"]
    errors: list[str] = []
    for key in ("name", "baseUrl", "apiKey", "api"):
        if key in provider and (not isinstance(provider[key], str) or not provider[key]):
            errors.append(f"{path}.{key}: Expected non-empty string")
    for key in ("headers", "compat", "modelOverrides"):
        if key in provider and not isinstance(provider[key], dict):
            errors.append(f"{path}.{key}: Expected object")
    if "authHeader" in provider and not isinstance(provider["authHeader"], bool):
        errors.append(f"{path}.authHeader: Expected boolean")
    if "oauth" in provider and provider["oauth"] != "radius":
        errors.append(f"{path}.oauth: Expected 'radius'")
    headers = provider.get("headers")
    if isinstance(headers, dict):
        for name, value in headers.items():
            if not isinstance(value, str):
                errors.append(f"{path}.headers.{name}: Expected string")
    models = provider.get("models")
    if models is not None:
        if not isinstance(models, list):
            errors.append(f"{path}.models: Expected array")
        else:
            for index, model in enumerate(models):
                model_path = f"{path}.models.{index}"
                if not isinstance(model, dict):
                    errors.append(f"{model_path}: Expected object")
                    continue
                if not isinstance(model.get("id"), str) or not model["id"]:
                    errors.append(f"{model_path}.id: Expected non-empty string")
                for key in ("name", "api", "baseUrl"):
                    if key in model and (not isinstance(model[key], str) or not model[key]):
                        errors.append(f"{model_path}.{key}: Expected non-empty string")
                for key in ("reasoning",):
                    if key in model and not isinstance(model[key], bool):
                        errors.append(f"{model_path}.{key}: Expected boolean")
                for key in ("contextWindow", "maxTokens"):
                    if key in model and (isinstance(model[key], bool) or not isinstance(model[key], (int, float))):
                        errors.append(f"{model_path}.{key}: Expected number")
                model_input = model.get("input")
                if model_input is not None and (
                    not isinstance(model_input, list)
                    or any(value not in ("text", "image") for value in model_input)
                ):
                    errors.append(f"{model_path}.input: Expected text/image array")
                for key in ("cost", "headers", "compat", "samplingParams", "thinkingLevelMap"):
                    if key in model and not isinstance(model[key], dict):
                        errors.append(f"{model_path}.{key}: Expected object")
    return errors


@dataclass(frozen=True)
class ModelConfig:
    providers: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    async def load(cls, models_json_path: str | None) -> ModelConfig:
        if not models_json_path:
            return cls()
        path = normalize_path(models_json_path)
        try:
            content = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
        except FileNotFoundError:
            return cls()
        except OSError as error:
            return cls(error=f"Failed to load models.json: {error}\n\nFile: {path}")
        try:
            parsed: object = json.loads(strip_json_comments(content.removeprefix("\ufeff")))
        except (ValueError, TypeError) as error:
            return cls(error=f"Failed to parse models.json: {error}\n\nFile: {path}")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("providers"), dict):
            return cls(error=f"Invalid models.json schema:\n  - providers: Expected object\n\nFile: {path}")
        raw_providers = cast(dict[str, object], parsed["providers"])
        errors = [
            error for provider_id, provider in raw_providers.items()
            for error in _validate_provider(provider_id, provider)
        ]
        if errors:
            return cls(error=f"Invalid models.json schema:\n" + "\n".join(f"  - {error}" for error in errors) + f"\n\nFile: {path}")
        copied = deepcopy(raw_providers)
        frozen = cast(Mapping[str, Mapping[str, object]], _freeze(copied))
        return cls(providers=frozen)

    def get_provider(self, provider_id: str) -> Mapping[str, object] | None:
        return self.providers.get(provider_id)

    def get_provider_ids(self) -> list[str]:
        return list(self.providers)

    def get_error(self) -> str | None:
        return self.error


__all__ = ["ModelConfig", "strip_json_comments"]
