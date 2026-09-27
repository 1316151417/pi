"""Native image API factory corresponding to ``openrouter-images.lazy.ts``."""

import asyncio

from ..types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions, ProviderImages
from . import openrouter_images


def openrouter_images_api() -> ProviderImages:
    async def generate_images(model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None) -> AssistantImages:
        # Static imports satisfy Python's module contract; retain the awaited
        # import's asynchronous boundary and read the module export on each call.
        await asyncio.sleep(0)
        return await openrouter_images.generate_images(model, context, options)

    return ProviderImages(generate_images=generate_images)


__all__ = ["openrouter_images_api"]
