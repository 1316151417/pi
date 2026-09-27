"""Provider factory from ``providers/qwen-token-plan-individual.ts``."""

from ..api.openai_completions_lazy import openai_completions_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider
from .qwen_token_plan_individual_models import QWEN_TOKEN_PLAN_INDIVIDUAL_MODELS


def qwen_token_plan_individual_provider() -> Provider:
    return create_provider(CreateProviderOptions(
        id="qwen-token-plan-individual", name="Qwen Token Plan Individual",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Qwen Token Plan Individual API key", ["QWEN_TOKEN_PLAN_API_KEY"])),
        models=list(QWEN_TOKEN_PLAN_INDIVIDUAL_MODELS.values()), api=openai_completions_api(),
    ))


__all__ = ["qwen_token_plan_individual_provider"]
