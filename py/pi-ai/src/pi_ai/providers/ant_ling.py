"""Provider factory from ``providers/ant-ling.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .ant_ling_models import ANT_LING_MODELS


def ant_ling_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="ant-ling", name="Ant Ling",
        base_url="https://api.ant-ling.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Ant Ling API key", ["ANT_LING_API_KEY"])),
        models=list(ANT_LING_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["ant_ling_provider"]
