"""OpenRouter image provider from ``providers/openrouter-images.ts``."""

from ..api.openrouter_images_lazy import openrouter_images_api
from ..auth.helpers import LazyOAuthOptions, env_api_key_auth, lazy_oauth
from ..auth.oauth.load import load_open_router_oauth
from ..auth.types import ProviderAuth
from ..image_models_generated import IMAGE_MODELS
from ..images_models import CreateImagesProviderOptions, ImagesProvider, create_images_provider


def openrouter_images_provider() -> ImagesProvider:
    return create_images_provider(CreateImagesProviderOptions(
        id="openrouter", name="OpenRouter",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
            oauth=lazy_oauth(LazyOAuthOptions(
                name="OpenRouter OAuth", login_label="Sign in with OpenRouter", load=load_open_router_oauth,
            )),
        ),
        models=list(IMAGE_MODELS["openrouter"].values()), api=openrouter_images_api(),
    ))


__all__ = ["openrouter_images_provider"]
