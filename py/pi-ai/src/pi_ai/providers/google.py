"""Provider factory from ``providers/google.ts``."""

from ..api.google_generative_ai_lazy import google_generative_ai_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .google_models import GOOGLE_MODELS


def google_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="google", name="Google", base_url="https://generativelanguage.googleapis.com/v1beta",
        auth=ProviderAuth(api_key=env_api_key_auth("Gemini API key", ["GEMINI_API_KEY"])),
        models=list(GOOGLE_MODELS.values()), api=google_generative_ai_api(),
    ))


__all__ = ["google_provider"]
