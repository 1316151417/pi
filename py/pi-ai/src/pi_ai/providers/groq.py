"""Provider factory from ``providers/groq.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .groq_models import GROQ_MODELS


def groq_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="groq", name="Groq",
        base_url="https://api.groq.com/openai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Groq API key", ["GROQ_API_KEY"])),
        models=list(GROQ_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["groq_provider"]
