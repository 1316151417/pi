"""Provider factory from ``providers/azure-openai-responses.ts``."""

from ..api.azure_openai_responses_lazy import azure_openai_responses_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .azure_openai_responses_models import AZURE_OPENAI_RESPONSES_MODELS


def azure_openai_responses_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="azure-openai-responses", name="Azure OpenAI",
        auth=ProviderAuth(api_key=env_api_key_auth("Azure OpenAI API key", ["AZURE_OPENAI_API_KEY"])),
        models=list(AZURE_OPENAI_RESPONSES_MODELS.values()), api=azure_openai_responses_api(),
    ))


__all__ = ["azure_openai_responses_provider"]
