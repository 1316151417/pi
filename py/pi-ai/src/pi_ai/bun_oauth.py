"""Register statically bundled native flows, corresponding to bun-oauth.ts."""

from .auth.oauth.anthropic import anthropic_oauth
from .auth.oauth.github_copilot import github_copilot_oauth
from .auth.oauth.kimi_coding import kimi_coding_oauth
from .auth.oauth.load import OAuthFlowLoaders, register_bundled_oauth_flow_loaders
from .auth.oauth.openai_codex import openai_codex_oauth
from .auth.oauth.openrouter import open_router_oauth
from .auth.oauth.radius import create_radius_oauth
from .auth.oauth.xai import xai_oauth


def register_bun_oauth_flows() -> None:
    register_bundled_oauth_flow_loaders(OAuthFlowLoaders(
        anthropic=lambda: anthropic_oauth,
        openai_codex=lambda: openai_codex_oauth,
        github_copilot=lambda: github_copilot_oauth,
        openrouter=lambda: open_router_oauth,
        kimi_coding=lambda: kimi_coding_oauth,
        xai=lambda: xai_oauth,
        radius=create_radius_oauth,
    ))


__all__ = ["register_bun_oauth_flows"]
