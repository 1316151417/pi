"""Provider factory from ``providers/nvidia.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .nvidia_models import NVIDIA_MODELS


def nvidia_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="nvidia", name="NVIDIA",
        base_url="https://integrate.api.nvidia.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("NVIDIA API key", ["NVIDIA_API_KEY"])),
        models=list(NVIDIA_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["nvidia_provider"]
