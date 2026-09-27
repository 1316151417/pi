"""Provider factory from ``providers/huggingface.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .huggingface_models import HUGGINGFACE_MODELS


def huggingface_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="huggingface", name="Hugging Face",
        base_url="https://router.huggingface.co/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Hugging Face token", ["HF_TOKEN"])),
        models=list(HUGGINGFACE_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["huggingface_provider"]
