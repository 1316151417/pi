"""Provider credential discovery from ``env-api-keys.ts``."""

from __future__ import annotations

from pathlib import Path

from .types import ProviderEnv
from .utils.provider_env import get_provider_env_value

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

_cached_vertex_adc_credentials_exists: bool | None = None
_API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "github-copilot": ("COPILOT_GITHUB_TOKEN",),
    "anthropic": (ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV),
    "ant-ling": ("ANT_LING_API_KEY",),
    "qwen-token-plan": ("QWEN_TOKEN_PLAN_API_KEY",),
    "qwen-token-plan-cn": ("QWEN_TOKEN_PLAN_CN_API_KEY",),
    "qwen-token-plan-individual": ("QWEN_TOKEN_PLAN_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "azure-openai-responses": ("AZURE_OPENAI_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "google-vertex": ("GOOGLE_CLOUD_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "radius": ("RADIUS_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "vercel-ai-gateway": ("AI_GATEWAY_API_KEY",),
    "zai": ("ZAI_API_KEY",),
    "zai-coding-cn": ("ZAI_CODING_CN_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "minimax-cn": ("MINIMAX_CN_API_KEY",),
    "moonshotai": ("MOONSHOT_API_KEY",),
    "moonshotai-cn": ("MOONSHOT_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "baseten": ("BASETEN_API_KEY",),
    "opencode": ("OPENCODE_API_KEY",),
    "opencode-go": ("OPENCODE_API_KEY",),
    "kimi-coding": ("KIMI_API_KEY",),
    "cloudflare-workers-ai": ("CLOUDFLARE_API_KEY",),
    "cloudflare-ai-gateway": ("CLOUDFLARE_API_KEY",),
    "xiaomi": ("XIAOMI_API_KEY",),
    "xiaomi-token-plan-cn": ("XIAOMI_TOKEN_PLAN_CN_API_KEY",),
    "xiaomi-token-plan-ams": ("XIAOMI_TOKEN_PLAN_AMS_API_KEY",),
    "xiaomi-token-plan-sgp": ("XIAOMI_TOKEN_PLAN_SGP_API_KEY",),
}


def _has_vertex_adc_credentials(env: ProviderEnv | None = None) -> bool:
    global _cached_vertex_adc_credentials_exists
    explicit_path = env.get("GOOGLE_APPLICATION_CREDENTIALS") if env is not None else None
    if explicit_path:
        return Path(explicit_path).exists()
    if _cached_vertex_adc_credentials_exists is None:
        credentials_path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
        path = Path(credentials_path) if credentials_path else Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        # Python has synchronous filesystem imports, so there is no Node/Bun
        # startup import race or browser-only permanent false cache.
        _cached_vertex_adc_credentials_exists = path.exists()
    return _cached_vertex_adc_credentials_exists


def find_env_keys(provider: str, env: ProviderEnv | None = None) -> list[str] | None:
    variables = _API_KEY_ENV_VARS.get(provider)
    if variables is None:
        return None
    return [name for name in variables if get_provider_env_value(name, env)] or None


def get_env_api_key(provider: str, env: ProviderEnv | None = None) -> str | None:
    env_keys = find_env_keys(provider, env)
    if env_keys:
        key = next((name for name in env_keys if name != ANTHROPIC_AUTH_TOKEN_ENV), None) if provider == "anthropic" else env_keys[0]
        if key is not None:
            return get_provider_env_value(key, env)
    if provider == "google-vertex":
        has_credentials = _has_vertex_adc_credentials(env)
        has_project = get_provider_env_value("GOOGLE_CLOUD_PROJECT", env) or get_provider_env_value("GCLOUD_PROJECT", env)
        has_location = get_provider_env_value("GOOGLE_CLOUD_LOCATION", env)
        if has_credentials and has_project and has_location:
            return "<authenticated>"
    if provider == "amazon-bedrock" and (
        get_provider_env_value("AWS_PROFILE", env)
        or (get_provider_env_value("AWS_ACCESS_KEY_ID", env) and get_provider_env_value("AWS_SECRET_ACCESS_KEY", env))
        or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env)
        or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env)
        or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env)
        or get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env)
    ):
        return "<authenticated>"
    return None


__all__ = [
    "ANTHROPIC_AUTH_TOKEN_ENV", "ANTHROPIC_OAUTH_TOKEN_ENV", "ANTHROPIC_API_KEY_ENV",
    "find_env_keys", "get_env_api_key",
]
