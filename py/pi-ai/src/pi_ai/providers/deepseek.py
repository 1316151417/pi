"""Provider factory from ``providers/deepseek.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .deepseek_models import DEEPSEEK_MODELS


def deepseek_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="deepseek", name="DeepSeek",
        base_url="https://api.deepseek.com",
        auth=ProviderAuth(api_key=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])),
        models=list(DEEPSEEK_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["deepseek_provider"]
