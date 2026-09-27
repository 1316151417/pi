"""Provider factory from ``providers/openai.ts``."""

from ..api.openai_responses_lazy import openai_responses_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .openai_models import OPENAI_MODELS


def openai_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="openai", name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"])),
        models=list(OPENAI_MODELS.values()), api=openai_responses_api(),
    ))


__all__ = ["openai_provider"]
