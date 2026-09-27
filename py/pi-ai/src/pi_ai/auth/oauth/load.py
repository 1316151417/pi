"""OAuth flow selection and standalone-loader overrides from oauth/load.ts.

Python imports the implementation modules statically. Registering a bundled
loader still overrides each subsequent lookup, matching the source API.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..types import OAuthAuth
from .anthropic import anthropic_oauth
from .github_copilot import github_copilot_oauth
from .kimi_coding import kimi_coding_oauth
from .openai_codex import openai_codex_oauth
from .openrouter import open_router_oauth
from .radius import RadiusOAuthOptions, create_radius_oauth
from .xai import xai_oauth

type OAuthFlowLoader = Callable[[], OAuthAuth | Awaitable[OAuthAuth]]


@dataclass
class OAuthFlowLoaders:
    anthropic: OAuthFlowLoader
    openai_codex: OAuthFlowLoader
    github_copilot: OAuthFlowLoader
    openrouter: OAuthFlowLoader
    kimi_coding: OAuthFlowLoader
    xai: OAuthFlowLoader
    radius: Callable[[RadiusOAuthOptions], OAuthAuth | Awaitable[OAuthAuth]]


_bundled_loaders: OAuthFlowLoaders | None = None


def register_bundled_oauth_flow_loaders(loaders: OAuthFlowLoaders) -> None:
    global _bundled_loaders
    _bundled_loaders = loaders


async def load_anthropic_oauth() -> OAuthAuth:
    value = _bundled_loaders.anthropic() if _bundled_loaders is not None else anthropic_oauth
    return await value if inspect.isawaitable(value) else value


async def load_openai_codex_oauth() -> OAuthAuth:
    value = _bundled_loaders.openai_codex() if _bundled_loaders is not None else openai_codex_oauth
    return await value if inspect.isawaitable(value) else value


async def load_github_copilot_oauth() -> OAuthAuth:
    value = _bundled_loaders.github_copilot() if _bundled_loaders is not None else github_copilot_oauth
    return await value if inspect.isawaitable(value) else value


async def load_open_router_oauth() -> OAuthAuth:
    value = _bundled_loaders.openrouter() if _bundled_loaders is not None else open_router_oauth
    return await value if inspect.isawaitable(value) else value


async def load_kimi_coding_oauth() -> OAuthAuth:
    value = _bundled_loaders.kimi_coding() if _bundled_loaders is not None else kimi_coding_oauth
    return await value if inspect.isawaitable(value) else value


async def load_xai_oauth() -> OAuthAuth:
    value = _bundled_loaders.xai() if _bundled_loaders is not None else xai_oauth
    return await value if inspect.isawaitable(value) else value


async def load_radius_oauth(options: RadiusOAuthOptions) -> OAuthAuth:
    value = _bundled_loaders.radius(options) if _bundled_loaders is not None else create_radius_oauth(options)
    return await value if inspect.isawaitable(value) else value


__all__ = [
    "OAuthFlowLoader", "OAuthFlowLoaders", "register_bundled_oauth_flow_loaders", "load_anthropic_oauth",
    "load_openai_codex_oauth", "load_github_copilot_oauth", "load_open_router_oauth", "load_kimi_coding_oauth",
    "load_xai_oauth", "load_radius_oauth",
]
