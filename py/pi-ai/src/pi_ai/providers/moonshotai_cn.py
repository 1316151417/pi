"""Provider factory from ``providers/moonshotai-cn.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .moonshotai_cn_models import MOONSHOTAI_CN_MODELS


def moonshotai_cn_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="moonshotai-cn", name="Moonshot AI CN",
        base_url="https://api.moonshot.cn/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
        models=list(MOONSHOTAI_CN_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["moonshotai_cn_provider"]
