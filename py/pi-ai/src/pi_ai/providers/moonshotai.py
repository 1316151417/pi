"""Provider factory from ``providers/moonshotai.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .moonshotai_models import MOONSHOTAI_MODELS


def moonshotai_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="moonshotai", name="Moonshot AI",
        base_url="https://api.moonshot.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
        models=list(MOONSHOTAI_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["moonshotai_provider"]
