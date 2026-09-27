"""Provider factory from ``providers/together.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .together_models import TOGETHER_MODELS


def together_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="together", name="Together",
        base_url="https://api.together.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Together API key", ["TOGETHER_API_KEY"])),
        models=list(TOGETHER_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["together_provider"]
