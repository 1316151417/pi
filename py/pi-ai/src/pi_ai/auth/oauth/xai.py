"""xAI OAuth device authorization and refresh from ``oauth/xai.ts``."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import cast

from ...abort import AbortSignal
from ..types import AuthDeviceCodeEvent, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from ._http import fetch, form_url_encode, parse_url
from .device_code import (
    OAuthDeviceCodeComplete, OAuthDeviceCodeFailed, OAuthDeviceCodePending, OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult, OAuthDeviceCodeSlowDown, poll_oauth_device_code_flow,
)

_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
_SCOPE = "openid profile email offline_access grok-cli:access api:access"
_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
_TOKEN_URL = "https://auth.x.ai/oauth2/token"
_REFRESH_SKEW_MS = 5 * 60 * 1000
_DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


@dataclass
class _HttpResponse:
    ok: bool
    status: int
    body: dict[str, object]


@dataclass
class _DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in_seconds: int | float
    verification_uri_complete: str | None = None
    interval_seconds: int | float | None = None


def _required_string(body: dict[str, object], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(body: dict[str, object], field: str) -> int | float:
    value = body.get(field)
    if type(value) not in (int, float) or not math.isfinite(cast(int | float, value)) or cast(int | float, value) <= 0:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return cast(int | float, value)


def _validate_verification_uri(raw: str) -> str:
    try:
        url = parse_url(raw)
    except ValueError:
        raise RuntimeError("Untrusted verification URI in xAI OAuth response") from None
    if url.protocol != "https:":
        raise RuntimeError("Untrusted verification URI in xAI OAuth response")
    return url.href


async def _post_form(url: str, fields: dict[str, str], signal: AbortSignal) -> _HttpResponse:
    try:
        response = await fetch(
            url, method="POST", headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            body=form_url_encode(fields), signal=signal,
        )
    except Exception:
        if signal.aborted:
            raise RuntimeError("Login cancelled") from None
        raise
    try:
        parsed = await response.json()
        body = cast(dict[str, object], parsed) if isinstance(parsed, dict) else {}
    except Exception:
        if signal.aborted:
            raise RuntimeError("Login cancelled") from None
        raise RuntimeError(f"xAI OAuth returned invalid JSON (HTTP {response.status})") from None
    return _HttpResponse(response.ok, response.status, body)


def _request_failure(action: str, response: _HttpResponse) -> RuntimeError:
    error = response.body.get("error")
    description = response.body.get("error_description")
    detail = ": ".join(value for value in (error, description) if isinstance(value, str) and value)
    return RuntimeError(f"xAI OAuth {action} failed (HTTP {response.status})" + (f": {detail}" if detail else ""))


def _parse_device_code(body: dict[str, object]) -> _DeviceCode:
    interval = body.get("interval")
    interval_seconds = cast(int | float, interval) if type(interval) in (int, float) and math.isfinite(cast(int | float, interval)) and cast(int | float, interval) > 0 else None
    complete = body.get("verification_uri_complete")
    verification_uri_complete = _validate_verification_uri(complete) if isinstance(complete, str) and complete else None
    return _DeviceCode(
        device_code=_required_string(body, "device_code"), user_code=_required_string(body, "user_code"),
        verification_uri=_validate_verification_uri(_required_string(body, "verification_uri")),
        verification_uri_complete=verification_uri_complete, interval_seconds=interval_seconds,
        expires_in_seconds=_positive_number(body, "expires_in"),
    )


def _credentials_from_token_response(body: dict[str, object], previous_refresh_token: str | None = None) -> OAuthCredential:
    access = _required_string(body, "access_token")
    refresh = previous_refresh_token if "refresh_token" not in body and previous_refresh_token else _required_string(body, "refresh_token")
    expires_in = _DEFAULT_TOKEN_LIFETIME_SECONDS if "expires_in" not in body else _positive_number(body, "expires_in")
    return OAuthCredential(
        access=access, refresh=refresh, expires=math.floor(time.time() * 1000) + expires_in * 1000 - _REFRESH_SKEW_MS,
    )


async def _request_device_code(signal: AbortSignal) -> _DeviceCode:
    response = await _post_form(_DEVICE_CODE_URL, {"client_id": _CLIENT_ID, "scope": _SCOPE, "referrer": "pi"}, signal)
    if not response.ok:
        raise _request_failure("device authorization", response)
    return _parse_device_code(response.body)


async def _poll_for_tokens(device: _DeviceCode, signal: AbortSignal) -> OAuthCredential:
    async def poll() -> OAuthDeviceCodePollResult[OAuthCredential]:
        response = await _post_form(_TOKEN_URL, {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code", "client_id": _CLIENT_ID,
            "device_code": device.device_code,
        }, signal)
        if response.ok:
            return OAuthDeviceCodeComplete(_credentials_from_token_response(response.body))
        error = response.body.get("error")
        if error == "authorization_pending":
            return OAuthDeviceCodePending()
        if error == "slow_down":
            interval = response.body.get("interval")
            return OAuthDeviceCodeSlowDown(cast(int | float, interval) if type(interval) in (int, float) else None)
        if error in ("access_denied", "authorization_denied"):
            return OAuthDeviceCodeFailed("xAI device authorization was denied")
        if error == "expired_token":
            return OAuthDeviceCodeFailed("xAI device code expired")
        return OAuthDeviceCodeFailed(str(_request_failure("device token polling", response)))

    return await poll_oauth_device_code_flow(OAuthDeviceCodePollOptions(
        poll=poll, signal=signal, interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds, wait_before_first_poll=True,
    ))


async def _login(interaction: ProviderAuthInteraction) -> OAuthCredential:
    device = await _request_device_code(interaction.signal)
    interaction.notify(AuthDeviceCodeEvent(
        user_code=device.user_code,
        verification_uri=device.verification_uri_complete if device.verification_uri_complete is not None else device.verification_uri,
        interval_seconds=device.interval_seconds, expires_in_seconds=device.expires_in_seconds,
    ))
    return await _poll_for_tokens(device, interaction.signal)


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    response = await _post_form(_TOKEN_URL, {
        "grant_type": "refresh_token", "client_id": _CLIENT_ID, "refresh_token": credential.refresh,
    }, signal)
    if not response.ok:
        raise _request_failure("token refresh", response)
    return _credentials_from_token_response(response.body, credential.refresh)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


xai_oauth = OAuthAuth(
    name="xAI (Grok/X subscription)", is_subscription=True, login_label="Sign in with SuperGrok or X Premium",
    login=_login, refresh=_refresh, to_auth=_to_auth,
)

__all__ = ["xai_oauth"]
