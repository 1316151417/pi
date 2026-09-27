"""Provider factory from ``providers/cerebras.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .cerebras_models import CEREBRAS_MODELS


def cerebras_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="cerebras", name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Cerebras API key", ["CEREBRAS_API_KEY"])),
        models=list(CEREBRAS_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["cerebras_provider"]
