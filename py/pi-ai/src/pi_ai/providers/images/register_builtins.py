"""Register the built-in image API, preserving the lazy wrapper error boundary."""

from __future__ import annotations

import asyncio
import time

from ...api import openrouter_images
from ...images_api_registry import ImagesApiProvider, register_images_api_provider
from ...types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions


async def generate_images_open_router(model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None) -> AssistantImages:
    try:
        await asyncio.sleep(0)
        return await openrouter_images.generate_images(model, context, options)
    except (Exception, asyncio.CancelledError) as error:
        return AssistantImages(
            api=model.api, provider=model.provider, model=model.id, output=[], stop_reason="error",
            error_message=str(error), timestamp=time.time_ns() // 1_000_000,
        )


def register_built_in_images_api_providers() -> None:
    register_images_api_provider(ImagesApiProvider(api="openrouter-images", generate_images=generate_images_open_router))


register_built_in_images_api_providers()

__all__ = ["generate_images_open_router", "register_built_in_images_api_providers"]
