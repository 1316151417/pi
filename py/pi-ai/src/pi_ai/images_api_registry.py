"""Image API adapter registry from ``images-api-registry.ts``."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass

from .types import AssistantImages, ImagesFunction, ImagesModel, ImagesContext, ImagesOptions

type ImagesApiFunction = ImagesFunction


@dataclass
class ImagesApiProvider:
    api: str
    generate_images: ImagesFunction


@dataclass
class _RegisteredImagesApiProvider:
    provider: ImagesApiProvider
    source_id: str | None = None


_registry: dict[str, _RegisteredImagesApiProvider] = {}


def register_images_api_provider(provider: ImagesApiProvider, source_id: str | None = None) -> None:
    api, generate = provider.api, provider.generate_images

    def wrapped(model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None) -> Awaitable[AssistantImages]:
        if model.api != api:
            raise RuntimeError(f"Mismatched api: {model.api} expected {api}")
        return generate(model, context, options)

    _registry[provider.api] = _RegisteredImagesApiProvider(ImagesApiProvider(provider.api, wrapped), source_id)


def get_images_api_provider(api: str) -> ImagesApiProvider | None:
    entry = _registry.get(api)
    return entry.provider if entry is not None else None


__all__ = ["ImagesApiFunction", "ImagesApiProvider", "register_images_api_provider", "get_images_api_provider"]
