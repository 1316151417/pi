"""Cloudflare credential fields and interactive API-key login."""

from __future__ import annotations

from typing import Literal

from ..abort import AbortSignal
from ..auth.types import (
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthContext, AuthResult, ModelAuth,
    ProviderAuthInteraction, SecretAuthPrompt, TextAuthPrompt,
)

CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"
type CloudflareAuthKind = Literal["workers-ai", "ai-gateway"]


async def _resolve_value(
    name: str, ctx: AuthContext, credential: ApiKeyCredential | None, signal: AbortSignal,
) -> str | None:
    stored = None
    if credential is not None:
        stored = credential.key if name == CLOUDFLARE_API_KEY else (credential.env or {}).get(name)
    if stored is not None:
        return stored
    signal.throw_if_aborted()
    value = await ctx.env(name)
    signal.throw_if_aborted()
    return value


async def _resolve_cloudflare_env(kind: CloudflareAuthKind, request: ApiKeyAuthInput) -> AuthResult | None:
    api_key = await _resolve_value(CLOUDFLARE_API_KEY, request.ctx, request.credential, request.signal)
    account_id = await _resolve_value(CLOUDFLARE_ACCOUNT_ID, request.ctx, request.credential, request.signal)
    gateway_id = (
        await _resolve_value(CLOUDFLARE_GATEWAY_ID, request.ctx, request.credential, request.signal)
        if kind == "ai-gateway" else None
    )
    if not api_key or not account_id or (kind == "ai-gateway" and not gateway_id):
        return None
    env = {CLOUDFLARE_ACCOUNT_ID: account_id}
    if gateway_id:
        env[CLOUDFLARE_GATEWAY_ID] = gateway_id
    auth = (
        ModelAuth(headers={"cf-aig-authorization": f"Bearer {api_key}", "Authorization": None, "x-api-key": None})
        if kind == "ai-gateway" else ModelAuth(api_key=api_key)
    )
    return AuthResult(auth=auth, env=env, source="stored credential" if request.credential is not None else CLOUDFLARE_API_KEY)


def cloudflare_workers_ai_auth() -> ApiKeyAuth:
    async def login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        key = await interaction.prompt(SecretAuthPrompt(message="Enter Cloudflare API key"))
        account_id = await interaction.prompt(TextAuthPrompt(message="Enter Cloudflare account ID"))
        return ApiKeyCredential(key=key, env={CLOUDFLARE_ACCOUNT_ID: account_id})

    async def resolve(request: ApiKeyAuthInput) -> AuthResult | None:
        return await _resolve_cloudflare_env("workers-ai", request)

    return ApiKeyAuth(name="Cloudflare API key", login=login, resolve=resolve)


def cloudflare_ai_gateway_auth() -> ApiKeyAuth:
    async def login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        key = await interaction.prompt(SecretAuthPrompt(message="Enter Cloudflare API key"))
        account_id = await interaction.prompt(TextAuthPrompt(message="Enter Cloudflare account ID"))
        gateway_id = await interaction.prompt(TextAuthPrompt(message="Enter Cloudflare AI Gateway ID"))
        return ApiKeyCredential(key=key, env={CLOUDFLARE_ACCOUNT_ID: account_id, CLOUDFLARE_GATEWAY_ID: gateway_id})

    async def resolve(request: ApiKeyAuthInput) -> AuthResult | None:
        return await _resolve_cloudflare_env("ai-gateway", request)

    return ApiKeyAuth(name="Cloudflare API key", login=login, resolve=resolve)


__all__ = ["CloudflareAuthKind", "cloudflare_workers_ai_auth", "cloudflare_ai_gateway_auth"]
