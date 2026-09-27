"""Provider factory from ``providers/baseten.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .baseten_models import BASETEN_MODELS


def baseten_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="baseten", name="Baseten",
        base_url="https://inference.baseten.co/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Baseten API key", ["BASETEN_API_KEY"])),
        models=list(BASETEN_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["baseten_provider"]
