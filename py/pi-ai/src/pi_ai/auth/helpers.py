"""Standard API-key handlers and lazy OAuth loading from ``auth/helpers.ts``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ..abort import AbortSignal
from .types import (
    ApiKeyAuth,
    ApiKeyAuthInput,
    ApiKeyCredential,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
    SecretAuthPrompt,
)

__all__ = ["env_api_key_auth", "LazyOAuthOptions", "lazy_oauth"]


def env_api_key_auth(name: str, env_vars: Sequence[str]) -> ApiKeyAuth:
    async def login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        interaction.signal.throw_if_aborted()
        key = await interaction.prompt(SecretAuthPrompt(message=f"Enter {name}"))
        interaction.signal.throw_if_aborted()
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyAuthInput) -> AuthResult | None:
        input.signal.throw_if_aborted()
        credential = input.credential
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")
        for env_var in env_vars:
            value = await input.ctx.env(env_var)
            input.signal.throw_if_aborted()
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None

    return ApiKeyAuth(name=name, login=login, resolve=resolve)


@dataclass
class LazyOAuthOptions:
    name: str
    load: Callable[[], Awaitable[OAuthAuth]]
    is_subscription: bool | None = None
    login_label: str | None = None


def lazy_oauth(input: LazyOAuthOptions) -> OAuthAuth:
    promise: asyncio.Future[OAuthAuth] | None = None

    async def loaded() -> OAuthAuth:
        nonlocal promise
        if promise is None:
            promise = asyncio.ensure_future(input.load())
        return await asyncio.shield(promise)

    async def login(interaction: ProviderAuthInteraction) -> OAuthCredential:
        return await (await loaded()).login(interaction)

    async def refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
        return await (await loaded()).refresh(credential, signal)

    async def to_auth(credential: OAuthCredential) -> ModelAuth:
        return await (await loaded()).to_auth(credential)

    return OAuthAuth(
        name=input.name, is_subscription=input.is_subscription, login_label=input.login_label,
        login=login, refresh=refresh, to_auth=to_auth,
    )
