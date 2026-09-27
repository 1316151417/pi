"""Auth precedence and locked OAuth refresh shared by model collections."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol, cast

from ..abort import AbortController, AbortSignal, operation_signal, race_with_abort_signal
from ..utils.abort_signals import combine_abort_signals
from ..utils.diagnostics import format_thrown_value
from .context import _JS_WHITESPACE
from .types import (
    ApiKeyAuth,
    ApiKeyAuthInput,
    ApiKeyCredential,
    AuthContext,
    AuthOperationOptions,
    AuthResult,
    Credential,
    CredentialStore,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)

if TYPE_CHECKING:
    from ..types import ProviderEnv

__all__ = ["ModelsErrorCode", "AuthResolutionOverrides", "ModelsError", "resolve_provider_auth"]

type ModelsErrorCode = Literal["model_source", "model_validation", "provider", "stream", "auth", "oauth"]

DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000
DEFAULT_OAUTH_REFRESH_TIMEOUT_MS = 15_000


@dataclass
class AuthResolutionOverrides:
    api_key: str | None = None
    env: ProviderEnv | None = None
    min_oauth_validity_ms: int | float | None = None
    signal: AbortSignal | None = None


class ModelsError(Exception):
    name = "ModelsError"

    def __init__(self, code: ModelsErrorCode, message: str, *, cause: object = None) -> None:
        if cause is not None:
            detail = format_thrown_value(cause).strip(_JS_WHITESPACE)
            if detail and detail not in message:
                message = f"{message}: {detail}"
        super().__init__(message)
        self.code = code
        self.cause = cause
        if isinstance(cause, BaseException):
            self.__cause__ = cause


class _AuthProvider(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def auth(self) -> ProviderAuth: ...


def resolve_provider_auth(
    provider: _AuthProvider,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None = None,
) -> Awaitable[AuthResult | None]:
    signal = operation_signal(overrides.signal if overrides is not None else None)
    return race_with_abort_signal(
        _resolve_provider_auth_with_signal(provider, credentials, auth_context, overrides, signal), signal,
    )


async def _resolve_provider_auth_with_signal(
    provider: _AuthProvider,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None,
    signal: AbortSignal,
) -> AuthResult | None:
    signal.throw_if_aborted()
    request_auth_context = (
        _OverlayAuthContext(auth_context, overrides.env)
        if overrides is not None and overrides.env is not None else auth_context
    )
    if overrides is not None and overrides.api_key is not None and provider.auth.api_key is not None:
        return await _resolve_api_key(
            request_auth_context, provider.auth.api_key, provider.id,
            ApiKeyCredential(key=overrides.api_key, env=overrides.env), signal,
        )
    stored = await _read_credential(credentials, provider.id, signal)
    if stored is not None:
        if stored.type == "oauth" and provider.auth.oauth is not None:
            return await _resolve_stored_oauth(
                credentials, provider.id, provider.auth.oauth, cast(OAuthCredential, stored), signal,
                overrides.min_oauth_validity_ms if overrides is not None else None,
            )
        if stored.type == "api_key" and provider.auth.api_key is not None:
            credential = cast(ApiKeyCredential, stored)
            if overrides is not None and overrides.env is not None:
                credential = replace(credential, env={**(credential.env or {}), **overrides.env})
            return await _resolve_api_key(request_auth_context, provider.auth.api_key, provider.id, credential, signal)
        return None
    if provider.auth.api_key is not None:
        return await _resolve_api_key(request_auth_context, provider.auth.api_key, provider.id, None, signal)
    return None


class _OverlayAuthContext:
    def __init__(self, base: AuthContext, env: ProviderEnv) -> None:
        self._base = base
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name) or await self._base.env(name)

    async def file_exists(self, path: str) -> bool:
        return await self._base.file_exists(path)


class _OAuthRefreshTimeoutError(TimeoutError):
    name = "TimeoutError"


async def _resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: OAuthCredential,
    signal: AbortSignal,
    min_oauth_validity_ms: int | float | None,
) -> AuthResult | None:
    requested_minimum = min_oauth_validity_ms if min_oauth_validity_ms is not None else 0
    minimum_validity_ms = (
        math.nan if isinstance(requested_minimum, float) and math.isnan(requested_minimum)
        else max(DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, requested_minimum)
    )

    def expires_soon(credential: OAuthCredential) -> bool:
        return time.time_ns() // 1_000_000 + minimum_validity_ms >= credential.expires

    credential = stored
    if expires_soon(credential):
        async def refresh(current: Credential | None) -> Credential | None:
            if current is None or current.type != "oauth":
                return None
            oauth_credential = cast(OAuthCredential, current)
            if not expires_soon(oauth_credential):
                return None
            timeout_controller = AbortController()
            timeout = asyncio.get_running_loop().call_later(
                DEFAULT_OAUTH_REFRESH_TIMEOUT_MS / 1000,
                timeout_controller.abort,
                _OAuthRefreshTimeoutError("The operation was aborted due to timeout"),
            )
            combined = combine_abort_signals((signal, timeout_controller.signal))
            try:
                return await oauth.refresh(oauth_credential, cast(AbortSignal, combined.signal))
            except (Exception, asyncio.CancelledError) as error:
                raise ModelsError("oauth", f"OAuth refresh failed for {provider_id}", cause=error) from error
            finally:
                timeout.cancel()
                combined.cleanup()

        try:
            post = await credentials.modify(provider_id, refresh, AuthOperationOptions(signal=signal))
        except ModelsError:
            raise
        except (Exception, asyncio.CancelledError) as error:
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error) from error
        if post is None or post.type != "oauth":
            return None
        credential = cast(OAuthCredential, post)
        if min_oauth_validity_ms is not None and expires_soon(credential):
            raise ModelsError("oauth", f"OAuth refresh returned a token that expires too soon for {provider_id}")
    try:
        return AuthResult(auth=await oauth.to_auth(credential), source="OAuth")
    except (Exception, asyncio.CancelledError) as error:
        raise ModelsError("oauth", f"OAuth auth derivation failed for {provider_id}", cause=error) from error


async def _resolve_api_key(
    auth_context: AuthContext, api_key: ApiKeyAuth, provider_id: str,
    credential: ApiKeyCredential | None, signal: AbortSignal,
) -> AuthResult | None:
    try:
        return await api_key.resolve(ApiKeyAuthInput(ctx=auth_context, credential=credential, signal=signal))
    except (Exception, asyncio.CancelledError) as error:
        raise ModelsError("auth", f"API key auth failed for provider {provider_id}", cause=error) from error


async def _read_credential(credentials: CredentialStore, provider_id: str, signal: AbortSignal) -> Credential | None:
    try:
        return await credentials.read(provider_id, AuthOperationOptions(signal=signal))
    except (Exception, asyncio.CancelledError) as error:
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error) from error
