"""Default credentials and token refresh from Anthropic JavaScript SDK 0.124.0.

This ports credential-chain, token-cache, identity-token, oidc-federation and
user-oauth. It performs no reads or requests until a resolver/provider runs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from .._javascript import javascript_json_parse, javascript_json_stringify, javascript_string, utf16_length
from .._json_runtime import JS_WHITESPACE, utf16_units
from .._values import UNDEFINED
from ..auth.oauth._common import js_number
from ..auth.oauth._http import fetch
from ..types import FetchFunction
from ._anthropic_credentials_config import (
    CREDENTIALS_FILE_VERSION, AnthropicConfig, get_credentials_path, load_config_with_source,
    read_env, read_utf8_file,
)
from ._anthropic_credential_types import (
    ADVISORY_REFRESH_BACKOFF_IN_SECONDS, ADVISORY_REFRESH_THRESHOLD_IN_SECONDS,
    FEDERATION_BETA_HEADER, MANDATORY_REFRESH_THRESHOLD_IN_SECONDS, OAUTH_API_BETA_HEADER,
    TOKEN_ENDPOINT, AccessToken, AccessTokenProvider, IdentityTokenProvider, TokenProviderOptions,
    TokenResponse, WorkloadIdentityError, check_credentials_file_safety, parse_token_response,
    property_value, redact_sensitive, require_secure_token_endpoint, truthy, write_credentials_file_atomic,
)
from ._anthropic_errors import AnthropicError


class TokenCache:
    def __init__(
        self, provider: AccessTokenProvider,
        on_advisory_refresh_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._provider = provider
        self._cached: AccessToken | None = None
        self._pending_refresh: asyncio.Task[AccessToken] | None = None
        self._next_force = False
        self._last_advisory_error = 0
        self._on_advisory_refresh_error = on_advisory_refresh_error
        self._background_observers: set[asyncio.Task[None]] = set()

    async def get_token(self) -> str:
        force = self._next_force
        self._next_force = False
        cached = self._cached
        if force or cached is None:
            return (await asyncio.shield(self._refresh(force))).token
        if cached.expires_at is None:
            return cached.token
        remaining = js_number(cached.expires_at) - time.time_ns() // 1_000_000_000
        if remaining > ADVISORY_REFRESH_THRESHOLD_IN_SECONDS:
            return cached.token
        if remaining > MANDATORY_REFRESH_THRESHOLD_IN_SECONDS:
            self._background_refresh()
            return cached.token
        return (await asyncio.shield(self._refresh())).token

    def invalidate(self) -> None:
        self._cached = None
        self._next_force = True

    def _refresh(self, force: bool = False) -> asyncio.Task[AccessToken]:
        if self._pending_refresh is not None and not force:
            return self._pending_refresh
        return self._do_refresh(force)

    def _background_refresh(self) -> None:
        if self._pending_refresh is not None:
            return
        if time.time_ns() // 1_000_000_000 - self._last_advisory_error < ADVISORY_REFRESH_BACKOFF_IN_SECONDS:
            return
        refresh = self._do_refresh()

        async def observe() -> None:
            try:
                await asyncio.shield(refresh)
            except (Exception, asyncio.CancelledError) as error:
                self._last_advisory_error = time.time_ns() // 1_000_000_000
                if self._on_advisory_refresh_error is not None:
                    self._on_advisory_refresh_error(error)

        task = asyncio.create_task(observe())
        self._background_observers.add(task)
        task.add_done_callback(self._background_observers.discard)

    def _do_refresh(self, force: bool = False) -> asyncio.Task[AccessToken]:
        # Invoke outside the observer, matching the source's synchronous
        # provider-call boundary before its returned promise is chained.
        pending = self._provider(TokenProviderOptions(force_refresh=True) if force else None)

        async def resolve() -> AccessToken:
            try:
                token = await pending
                self._cached = token
                return token
            finally:
                self._pending_refresh = None

        self._pending_refresh = asyncio.create_task(resolve())
        return self._pending_refresh


@dataclass
class ResolverOptions:
    base_url: str
    custom_fetch: FetchFunction | None = None
    user_agent: str | None = None
    on_cache_write_error: Callable[[BaseException], None] | None = None
    on_safety_warning: Callable[[str], None] | None = None


@dataclass
class CredentialResult:
    provider: AccessTokenProvider
    extra_headers: dict[str, str]
    base_url: str | None = None


@dataclass
class ResolvedCredentials:
    token_cache: TokenCache
    extra_headers: dict[str, str]
    base_url: str | None = None


@dataclass
class OIDCFederationConfig:
    identity_token_provider: IdentityTokenProvider
    federation_rule_id: str
    organization_id: str
    base_url: str
    custom_fetch: FetchFunction | None = None
    service_account_id: str | None = None
    workspace_id: str | None = None
    user_agent: str | None = None


@dataclass
class UserOAuthConfig:
    credentials_path: str
    base_url: str
    custom_fetch: FetchFunction | None = None
    client_id: str | None = None
    user_agent: str | None = None
    on_safety_warning: Callable[[str], None] | None = None


def identity_token_from_file(path: str) -> IdentityTokenProvider:
    if not path:
        raise AnthropicError("Identity token file path is empty")

    async def read() -> str:
        try:
            content = await read_utf8_file(path)
        except (Exception, asyncio.CancelledError) as error:
            raise AnthropicError(f"Failed to read identity token file at {path}: {error}") from error
        token = content.strip(JS_WHITESPACE)
        if not token:
            raise AnthropicError(f"Identity token file at {path} is empty")
        return token

    return read


def identity_token_from_value(token: str) -> IdentityTokenProvider:
    if not token:
        raise AnthropicError("Identity token value is empty")
    return lambda: token


def oidc_federation_provider(config: OIDCFederationConfig) -> AccessTokenProvider:
    async def exchange(_options: TokenProviderOptions | None = None) -> AccessToken:
        require_secure_token_endpoint(config.base_url)
        identity = config.identity_token_provider()
        jwt = await identity if inspect.isawaitable(identity) else identity
        length = utf16_length(jwt)
        if length > 16 * 1024:
            raise WorkloadIdentityError(f"Identity token is {math.ceil(length / 1024)} KiB, exceeds the 16 KiB assertion limit")
        body = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": jwt,
            "federation_rule_id": config.federation_rule_id, "organization_id": config.organization_id,
        }
        if config.service_account_id:
            body["service_account_id"] = config.service_account_id
        if config.workspace_id:
            body["workspace_id"] = config.workspace_id
        url = config.base_url + TOKEN_ENDPOINT
        try:
            response = cast(TokenResponse, await (config.custom_fetch or fetch)(
                url, method="POST", headers={
                    "Content-Type": "application/json",
                    "anthropic-beta": OAUTH_API_BETA_HEADER + "," + FEDERATION_BETA_HEADER,
                    "User-Agent": config.user_agent or "anthropic-sdk-python/0.124.0 oidcFederationProvider",
                }, body=javascript_json_stringify(body),
            ))
        except (Exception, asyncio.CancelledError) as error:
            raise WorkloadIdentityError(f"Failed to reach token endpoint {url}: {error}") from error
        request_id = next((value for key, value in response.headers.items() if key.lower() == "request-id"), None)
        if not response.ok:
            try:
                text = await response.text()
            except (Exception, asyncio.CancelledError):
                text = ""
            redacted = redact_sensitive(text)
            hint = ""
            if response.status == 401:
                middle = "" if config.workspace_id else "If your federation rule is scoped to multiple workspaces, set the ANTHROPIC_WORKSPACE_ID environment variable, the 'workspace_id' config key, or the `workspaceId` option. "
                hint = " Ensure your federation rule matches your identity token. " + middle + "View your authentication events in the Workload identity page of Claude Console for more details."
            request_info = f" (request-id {request_id})" if request_id else ""
            raise WorkloadIdentityError(
                f"Token exchange failed with status {response.status}{request_info}: {javascript_string(redacted)}{hint}",
                response.status, redacted, request_id,
            )
        data = await parse_token_response(response, request_id)
        expires_in = js_number(data.get("expires_in", UNDEFINED))
        if not math.isfinite(expires_in):
            raise WorkloadIdentityError(
                f"Token endpoint response missing required fields: {javascript_json_stringify(redact_sensitive(data))}",
                response.status, redact_sensitive(data), request_id,
            )
        return AccessToken(token=cast(str, data["access_token"]), expires_at=time.time_ns() // 1_000_000_000 + expires_in)

    return exchange


def user_oauth_provider(config: UserOAuthConfig) -> AccessTokenProvider:
    async def provide(options: TokenProviderOptions | None = None) -> AccessToken:
        await check_credentials_file_safety(config.credentials_path, config.on_safety_warning)
        try:
            raw = await read_utf8_file(config.credentials_path)
        except (Exception, asyncio.CancelledError) as error:
            raise WorkloadIdentityError(f"Credentials file not found at {config.credentials_path}: {error}") from error
        try:
            credentials = javascript_json_parse(raw)
        except (ValueError, TypeError) as error:
            raise WorkloadIdentityError(f"Credentials file at {config.credentials_path} is not valid JSON: {error}") from error
        access_token = property_value(credentials, "access_token")
        if not truthy(access_token):
            raise WorkloadIdentityError(f"Credentials file at {config.credentials_path} must include 'access_token'")
        expires_at = property_value(credentials, "expires_at")
        if (options is None or not options.force_refresh) and (
            expires_at is None or expires_at is UNDEFINED
            or time.time_ns() // 1_000_000_000 < js_number(expires_at) - MANDATORY_REFRESH_THRESHOLD_IN_SECONDS
        ):
            return AccessToken(token=cast(str, access_token), expires_at=None if expires_at is UNDEFINED else cast(float | None, expires_at))
        refresh_token = property_value(credentials, "refresh_token")
        if not config.client_id or not truthy(refresh_token):
            raise WorkloadIdentityError(
                f"Access token at {config.credentials_path} has expired and no refresh is available "
                f"(client_id {'set' if config.client_id else 'empty'}, refresh_token {'set' if truthy(refresh_token) else 'empty'})",
            )
        require_secure_token_endpoint(config.base_url)
        url = config.base_url + TOKEN_ENDPOINT
        try:
            response = cast(TokenResponse, await (config.custom_fetch or fetch)(
                url, method="POST", headers={
                    "Content-Type": "application/json", "anthropic-beta": OAUTH_API_BETA_HEADER,
                    "User-Agent": config.user_agent or "anthropic-sdk-python/0.124.0 userOAuthProvider",
                }, body=javascript_json_stringify({"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": config.client_id}),
            ))
        except (Exception, asyncio.CancelledError) as error:
            raise WorkloadIdentityError(f"User OAuth refresh failed to reach token endpoint: {error}") from error
        request_id = next((value for key, value in response.headers.items() if key.lower() == "request-id"), None)
        if not response.ok:
            try:
                text = await response.text()
            except (Exception, asyncio.CancelledError):
                text = ""
            raise WorkloadIdentityError(
                f"User OAuth refresh failed (HTTP {response.status}): {javascript_string(redact_sensitive(text))}",
                response.status, redact_sensitive(text), request_id,
            )
        data = await parse_token_response(response, request_id)
        expires_in = js_number(data.get("expires_in", UNDEFINED))
        if not math.isfinite(expires_in):
            raise WorkloadIdentityError(
                f"User OAuth refresh response missing or invalid expires_in: {javascript_json_stringify(redact_sensitive(data))}",
                response.status, redact_sensitive(data), request_id,
            )
        new_expires_at = time.time_ns() // 1_000_000_000 + expires_in
        new_refresh_token = data.get("refresh_token") if truthy(data.get("refresh_token")) else refresh_token
        await write_credentials_file_atomic(config.credentials_path, {
            **cast(dict[str, object], credentials), "version": CREDENTIALS_FILE_VERSION, "type": "oauth_token",
            "access_token": data["access_token"], "expires_at": new_expires_at, "refresh_token": new_refresh_token,
        })
        return AccessToken(token=cast(str, data["access_token"]), expires_at=new_expires_at)

    return provide


def _cached_exchange_provider(exchange: AccessTokenProvider, path: str, options: ResolverOptions) -> AccessTokenProvider:
    async def provide(provider_options: TokenProviderOptions | None = None) -> AccessToken:
        await check_credentials_file_safety(path, options.on_safety_warning)
        existing: object = None
        try:
            existing = javascript_json_parse(await read_utf8_file(path))
            token = property_value(existing, "access_token") if existing is not None else UNDEFINED
            if truthy(token) and (provider_options is None or not provider_options.force_refresh):
                expires_at = property_value(existing, "expires_at")
                if expires_at is None or expires_at is UNDEFINED or time.time_ns() // 1_000_000_000 < js_number(expires_at) - MANDATORY_REFRESH_THRESHOLD_IN_SECONDS:
                    return AccessToken(token=cast(str, token), expires_at=None if expires_at is UNDEFINED else cast(float | None, expires_at))
        except (Exception, asyncio.CancelledError) as error:
            if not isinstance(error, (FileNotFoundError, json.JSONDecodeError)) and options.on_cache_write_error is not None:
                options.on_cache_write_error(error)
        result = await exchange(provider_options)
        try:
            if isinstance(existing, Mapping):
                preserved = dict(existing)
            elif isinstance(existing, str):
                preserved = {str(index): character for index, character in enumerate(utf16_units(existing))}
            elif isinstance(existing, list):
                preserved = {str(index): value for index, value in enumerate(existing)}
            else:
                preserved = {}
            await write_credentials_file_atomic(path, {
                **preserved,
                "version": CREDENTIALS_FILE_VERSION, "type": "oauth_token",
                "access_token": result.token, "expires_at": result.expires_at,
            })
        except (Exception, asyncio.CancelledError) as error:
            if options.on_cache_write_error is not None:
                options.on_cache_write_error(error)
        return result

    return provide


def resolve_credentials_from_config(config: AnthropicConfig, options: ResolverOptions) -> CredentialResult:
    auth = cast(dict[str, object], config["authentication"])
    credentials_path = auth.get("credentials_path")
    effective_base = cast(str, config.get("base_url") if truthy(config.get("base_url")) else options.base_url).rstrip("/")
    kind = auth.get("type", UNDEFINED)
    if kind == "oidc_federation":
        identity_provider: IdentityTokenProvider | None = None
        identity = auth.get("identity_token", UNDEFINED)
        if truthy(identity):
            source = property_value(identity, "source")
            if source != "file":
                raise WorkloadIdentityError(f'identity_token.source "{javascript_string(source)}" is not supported by this SDK version (only "file")')
            path = property_value(identity, "path")
            if not truthy(path):
                raise WorkloadIdentityError('identity_token.source "file" requires a non-empty path')
            identity_provider = identity_token_from_file(cast(str, path))
        else:
            token_file = read_env("ANTHROPIC_IDENTITY_TOKEN_FILE")
            token_value = read_env("ANTHROPIC_IDENTITY_TOKEN")
            if token_file:
                identity_provider = identity_token_from_file(token_file)
            elif token_value:
                identity_provider = identity_token_from_value(token_value)
        if identity_provider is None:
            raise WorkloadIdentityError(
                "oidc_federation config requires an identity token (set authentication.identity_token, ANTHROPIC_IDENTITY_TOKEN_FILE, or ANTHROPIC_IDENTITY_TOKEN)",
            )
        if not truthy(auth.get("federation_rule_id")):
            raise WorkloadIdentityError("oidc_federation config requires 'federation_rule_id'. Set it in authentication.federation_rule_id in your profile, or via ANTHROPIC_FEDERATION_RULE_ID (profile takes precedence).")
        if not truthy(config.get("organization_id")):
            raise WorkloadIdentityError("oidc_federation config requires organization_id (set ANTHROPIC_ORGANIZATION_ID or config.organization_id)")
        provider = oidc_federation_provider(OIDCFederationConfig(
            identity_token_provider=identity_provider, federation_rule_id=cast(str, auth["federation_rule_id"]),
            organization_id=cast(str, config["organization_id"]), base_url=effective_base, custom_fetch=options.custom_fetch,
            service_account_id=cast(str | None, auth.get("service_account_id")), workspace_id=cast(str | None, config.get("workspace_id")),
            user_agent=options.user_agent,
        ))
        if truthy(credentials_path):
            provider = _cached_exchange_provider(provider, cast(str, credentials_path), options)
    elif kind == "user_oauth":
        if not truthy(credentials_path):
            raise WorkloadIdentityError("user_oauth config requires authentication.credentials_path (or load via a profile so it defaults to <config_dir>/credentials/<profile>.json)")
        provider = user_oauth_provider(UserOAuthConfig(
            credentials_path=cast(str, credentials_path), base_url=effective_base, custom_fetch=options.custom_fetch,
            client_id=cast(str | None, auth.get("client_id")), user_agent=options.user_agent,
            on_safety_warning=options.on_safety_warning,
        ))
    else:
        raise WorkloadIdentityError(f'authentication.type "{javascript_string(kind)}" is not a known authentication type')
    headers: dict[str, str] = {}
    if truthy(config.get("workspace_id")) and kind == "user_oauth":
        headers["anthropic-workspace-id"] = cast(str, config["workspace_id"])
    return CredentialResult(provider=provider, extra_headers=headers, base_url=cast(str | None, config.get("base_url") or None))


async def default_credentials(options: ResolverOptions, profile: str | None = None) -> CredentialResult | None:
    loaded = await load_config_with_source(profile)
    if loaded is None:
        return None
    config = loaded.config
    auth = cast(dict[str, object], config["authentication"])
    if not truthy(auth.get("credentials_path")) and loaded.from_file:
        config = {**config, "authentication": {**auth, "credentials_path": await get_credentials_path(config, profile) or UNDEFINED}}
    return resolve_credentials_from_config(config, options)


async def resolve_default_credentials(
    *, base_url: str, custom_fetch: FetchFunction | None = None,
    user_agent: str = "Anthropic/Python 0.124.0",
    on_cache_write_error: Callable[[BaseException], None] | None = None,
    on_safety_warning: Callable[[str], None] | None = None,
    on_advisory_refresh_error: Callable[[BaseException], None] | None = None,
) -> ResolvedCredentials | None:
    result = await default_credentials(ResolverOptions(
        base_url=base_url, custom_fetch=custom_fetch, user_agent=user_agent,
        on_cache_write_error=on_cache_write_error, on_safety_warning=on_safety_warning,
    ))
    if result is None:
        return None
    return ResolvedCredentials(
        token_cache=TokenCache(result.provider, on_advisory_refresh_error),
        extra_headers=result.extra_headers, base_url=result.base_url,
    )
