"""Catalog access and the selected native provider factories.

Catalog reads retain all bundled metadata. Runtime factories cover the
representative providers selected for this port, not every catalog vendor.
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib.resources import files
from typing import cast

from ..images_models import ImagesProvider, MutableImagesModels, create_images_models
from ..models import CreateModelsOptions, MutableModels, Provider, create_models
from ..models_generated import MODELS
from ..types import Model
from .ant_ling import ant_ling_provider
from .anthropic import anthropic_provider
from .azure_openai_responses import azure_openai_responses_provider
from .baseten import baseten_provider
from .cerebras import cerebras_provider
from .deepseek import deepseek_provider
from .google import google_provider
from .groq import groq_provider
from .huggingface import huggingface_provider
from .moonshotai import moonshotai_provider
from .moonshotai_cn import moonshotai_cn_provider
from .nvidia import nvidia_provider
from .openai import openai_provider
from .openrouter_images import openrouter_images_provider
from .qwen_token_plan import qwen_token_plan_provider
from .qwen_token_plan_cn import qwen_token_plan_cn_provider
from .qwen_token_plan_individual import qwen_token_plan_individual_provider
from .together import together_provider

type BuiltinProvider = str
_MODEL_DATA_MANIFEST = cast(dict[str, object], json.loads(
    files("pi_ai.providers").joinpath("data", ".manifest.json").read_text(encoding="utf-8"),
))


def get_builtin_model(provider: str, model_id: str) -> Model | None:
    models = MODELS.get(provider)
    return models.get(model_id) if models is not None else None


def get_builtin_providers() -> list[str]:
    return list(MODELS)


def get_builtin_model_data_generated_at() -> int | None:
    value = _MODEL_DATA_MANIFEST.get("generatedAt")
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, OverflowError, OSError):
        return None


def get_builtin_models(provider: str) -> list[Model]:
    models = MODELS.get(provider)
    return list(models.values()) if models is not None else []


def builtin_providers() -> list[Provider]:
    """Construct fresh providers for the supported representative subset."""
    return [
        ant_ling_provider(), anthropic_provider(), azure_openai_responses_provider(),
        baseten_provider(), cerebras_provider(), deepseek_provider(), google_provider(),
        groq_provider(), huggingface_provider(), moonshotai_provider(), moonshotai_cn_provider(),
        nvidia_provider(), openai_provider(), qwen_token_plan_provider(), qwen_token_plan_cn_provider(),
        qwen_token_plan_individual_provider(), together_provider(),
    ]


def builtin_models(options: CreateModelsOptions | None = None) -> MutableModels:
    models = create_models(options)
    for provider in builtin_providers():
        models.set_provider(provider)
    return models


def builtin_images_providers() -> list[ImagesProvider]:
    return [openrouter_images_provider()]


def builtin_images_models(options: CreateModelsOptions | None = None) -> MutableImagesModels:
    models = create_images_models(options)
    for provider in builtin_images_providers():
        models.set_provider(provider)
    return models


__all__ = [
    "BuiltinProvider", "get_builtin_model", "get_builtin_providers", "get_builtin_model_data_generated_at",
    "get_builtin_models", "builtin_providers", "builtin_models", "builtin_images_providers", "builtin_images_models",
]
