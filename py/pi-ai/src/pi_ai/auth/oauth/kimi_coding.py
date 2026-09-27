"""Kimi Code subscription OAuth from ``oauth/kimi-coding.ts``."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import cast

from ...abort import AbortSignal, any_signal
from ...utils._javascript import javascript_json_stringify
from ...utils.provider_env import get_provider_env_value
from ...utils.sleep import sleep
from ..types import AuthDeviceCodeEvent, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from ._http import OAuthHttpResponse, fetch, form_url_encode, timeout_signal, trusted_http_url
from .device_code import (
    OAuthDeviceCodeComplete, OAuthDeviceCodeFailed, OAuthDeviceCodePending, OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult, OAuthDeviceCodeSlowDown, poll_oauth_device_code_flow,
)

_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
_DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
_DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
_DEFAULT_POLL_INTERVAL_SECONDS = 5
_REQUEST_TIMEOUT_MS = 30 * 1000
_REFRESH_MAX_RETRIES = 3

type _JsonRecord = dict[str, object] | list[object]


@dataclass
class _DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    interval_seconds: int | float
    expires_in_seconds: int | float


@dataclass
class _TokenResponse:
    access: str
    refresh: str
    expires: int | float


def _get_oauth_host() -> str:
    override = get_provider_env_value("KIMI_CODE_OAUTH_HOST") or get_provider_env_value("KIMI_OAUTH_HOST")
    return (override or _DEFAULT_OAUTH_HOST).rstrip("/")


def _request_signal(signal: AbortSignal) -> AbortSignal:
    result = any_signal(timeout_signal(_REQUEST_TIMEOUT_MS), signal)
    assert result is not None
    return result


async def _read_json(response: OAuthHttpResponse) -> _JsonRecord | None:
    try:
        value = await response.json()
        return cast(_JsonRecord, value) if isinstance(value, (dict, list)) else None
    except Exception:
        return None


def _field(record: _JsonRecord | None, name: str) -> object:
    return record.get(name) if isinstance(record, dict) else None


async def _start_device_authorization(oauth_host: str, signal: AbortSignal) -> _DeviceAuthorization:
    response = await fetch(
        f"{oauth_host}/api/oauth/device_authorization", method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        body=form_url_encode({"client_id": _CLIENT_ID}), signal=_request_signal(signal),
    )
    if not response.ok:
        try:
            text = await response.text()
        except Exception:
            text = ""
        raise RuntimeError(f"Kimi Code device authorization failed with status {response.status}" + (f": {text}" if text else ""))
    data = await _read_json(response)
    device_code = _field(data, "device_code")
    user_code = _field(data, "user_code")
    verification_uri = _field(data, "verification_uri")
    complete = _field(data, "verification_uri_complete")
    if (
        not isinstance(device_code, str) or not isinstance(user_code, str)
        or not isinstance(verification_uri, str) or not isinstance(complete, str)
        or not trusted_http_url(complete) or not trusted_http_url(verification_uri)
    ):
        raise RuntimeError(f"Invalid Kimi Code device authorization response: {javascript_json_stringify(data)}")
    interval = _field(data, "interval")
    expires = _field(data, "expires_in")
    return _DeviceAuthorization(
        device_code=device_code, user_code=user_code, verification_uri=verification_uri, verification_uri_complete=complete,
        interval_seconds=cast(int | float, interval) if type(interval) in (int, float) and math.isfinite(cast(int | float, interval)) and cast(int | float, interval) > 0 else _DEFAULT_POLL_INTERVAL_SECONDS,
        expires_in_seconds=cast(int | float, expires) if type(expires) in (int, float) and math.isfinite(cast(int | float, expires)) and cast(int | float, expires) > 0 else _DEVICE_CODE_TIMEOUT_SECONDS,
    )


def _parse_token_response(data: _JsonRecord | None, operation: str) -> _TokenResponse:
    access = _field(data, "access_token")
    refresh = _field(data, "refresh_token")
    expires = _field(data, "expires_in")
    if (
        not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh
        or type(expires) not in (int, float) or not math.isfinite(cast(int | float, expires)) or cast(int | float, expires) <= 0
    ):
        raise RuntimeError(f"Kimi Code token {operation} response missing fields: {javascript_json_stringify(data)}")
    return _TokenResponse(access, refresh, math.floor(time.time() * 1000) + cast(int | float, expires) * 1000)


async def _poll_for_token(oauth_host: str, device: _DeviceAuthorization, signal: AbortSignal) -> _TokenResponse:
    async def poll() -> OAuthDeviceCodePollResult[_TokenResponse]:
        response = await fetch(
            f"{oauth_host}/api/oauth/token", method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            body=form_url_encode({
                "client_id": _CLIENT_ID, "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }), signal=_request_signal(signal),
        )
        if response.status >= 500:
            try:
                text = await response.text()
            except Exception:
                text = ""
            return OAuthDeviceCodeFailed(f"Kimi Code device token request failed with status {response.status}" + (f": {text}" if text else ""))
        data = await _read_json(response)
        if response.ok and isinstance(_field(data, "access_token"), str):
            try:
                return OAuthDeviceCodeComplete(_parse_token_response(data, "poll"))
            except Exception as error:
                return OAuthDeviceCodeFailed(str(error))
        error_code = _field(data, "error")
        raw_description = _field(data, "error_description")
        description = f": {raw_description}" if isinstance(raw_description, str) else ""
        if error_code == "authorization_pending":
            return OAuthDeviceCodePending()
        if error_code == "slow_down":
            interval = _field(data, "interval")
            return OAuthDeviceCodeSlowDown(cast(int | float, interval) if type(interval) in (int, float) and cast(int | float, interval) > 0 else None)
        if error_code == "expired_token":
            return OAuthDeviceCodeFailed("Kimi Code device authorization expired. Please restart login.")
        if error_code == "access_denied":
            return OAuthDeviceCodeFailed("Kimi Code login was denied.")
        return OAuthDeviceCodeFailed(
            f"Kimi Code device token request failed (status {response.status})"
            + (f": {error_code}{description}" if isinstance(error_code, str) else ""),
        )

    return await poll_oauth_device_code_flow(OAuthDeviceCodePollOptions(
        poll=poll, signal=signal, interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds, wait_before_first_poll=True,
    ))


async def _refresh_token(oauth_host: str, refresh: str, signal: AbortSignal) -> _TokenResponse:
    last_error: Exception | None = None
    for attempt in range(_REFRESH_MAX_RETRIES + 1):
        if attempt > 0:
            await sleep(1000 * 2 ** (attempt - 1), signal)
        if signal.aborted:
            raise RuntimeError("Kimi Code token refresh aborted")
        try:
            response = await fetch(
                f"{oauth_host}/api/oauth/token", method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                body=form_url_encode({"client_id": _CLIENT_ID, "grant_type": "refresh_token", "refresh_token": refresh}),
                signal=_request_signal(signal),
            )
        except Exception as error:
            last_error = error
            continue
        data = await _read_json(response)
        if response.ok:
            return _parse_token_response(data, "refresh")
        if response.status in (401, 403) or _field(data, "error") == "invalid_grant":
            raw_description = _field(data, "error_description")
            description = f": {raw_description}" if isinstance(raw_description, str) else ""
            raise RuntimeError(f"Kimi Code token refresh unauthorized (status {response.status}){description}")
        if (response.status == 429 or response.status >= 500) and attempt < _REFRESH_MAX_RETRIES:
            last_error = RuntimeError(f"Kimi Code token refresh failed with status {response.status}")
            continue
        text = javascript_json_stringify(data)
        raise RuntimeError(f"Kimi Code token refresh failed with status {response.status}" + (f": {text}" if text else ""))
    raise last_error if last_error is not None else RuntimeError("Kimi Code token refresh failed")


async def _login(interaction: ProviderAuthInteraction) -> OAuthCredential:
    host = _get_oauth_host()
    device = await _start_device_authorization(host, interaction.signal)
    interaction.notify(AuthDeviceCodeEvent(
        user_code=device.user_code, verification_uri=device.verification_uri_complete,
        interval_seconds=device.interval_seconds, expires_in_seconds=device.expires_in_seconds,
    ))
    token = await _poll_for_token(host, device, interaction.signal)
    return OAuthCredential(access=token.access, refresh=token.refresh, expires=token.expires)


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    token = await _refresh_token(_get_oauth_host(), credential.refresh, signal)
    return OAuthCredential(access=token.access, refresh=token.refresh, expires=token.expires)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(headers={"Authorization": f"Bearer {credential.access}"})


kimi_coding_oauth = OAuthAuth(
    name="Kimi Code (subscription)", is_subscription=True, login_label="Sign in with Kimi Code",
    login=_login, refresh=_refresh, to_auth=_to_auth,
)

__all__ = ["kimi_coding_oauth"]
