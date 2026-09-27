"""Static image catalog access from ``image-models.ts``."""

from .image_models_generated import IMAGE_MODELS
from .types import ImagesModel

_image_model_registry = {provider: dict(models) for provider, models in IMAGE_MODELS.items()}


def get_image_model(provider: str, model_id: str) -> ImagesModel | None:
    models = _image_model_registry.get(provider)
    return models.get(model_id) if models is not None else None


def get_image_providers() -> list[str]:
    return list(_image_model_registry)


def get_image_models(provider: str) -> list[ImagesModel]:
    models = _image_model_registry.get(provider)
    return list(models.values()) if models is not None else []


__all__ = ["get_image_model", "get_image_providers", "get_image_models"]
