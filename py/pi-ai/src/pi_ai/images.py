"""Image generation through the global API registry from ``images.ts``."""

from .providers.images import register_builtins as _register_builtins
from .images_api_registry import get_images_api_provider
from .types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions


async def generate_images(model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None) -> AssistantImages:
    provider = get_images_api_provider(model.api)
    if provider is None:
        raise RuntimeError(f"No API provider registered for api: {model.api}")
    return await provider.generate_images(model, context, options)


__all__ = ["generate_images"]
