"""OpenAI Codex ChatGPT OAuth browser and device-code login."""

from __future__ import annotations

import asyncio
import base64
import math
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from ...abort import AbortController, AbortSignal
from ...utils._javascript import javascript_json_stringify
from ...utils.provider_env import get_provider_env_value
from ..types import (
    AuthDeviceCodeEvent, AuthSelectOption, AuthUrlEvent, ManualCodeAuthPrompt,
    OAuthAuth, OAuthCredential, ProviderAuthInteraction, SelectAuthPrompt,
)
from ._callback_server import CallbackRequest, CallbackResponse, CallbackServer
from ._common import (
    JS_WHITESPACE, ManualPrompt, await_promise, credential_to_auth, defer_operation, js_number,
    js_truthy, json_parse, parse_authorization_input, property_value, search_param,
)
from ._http import OAuthHttpResponse, fetch, form_url_encode, parse_url
from .device_code import (
    OAuthDeviceCodeComplete, OAuthDeviceCodeFailed, OAuthDeviceCodePending,
    OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, OAuthDeviceCodeSlowDown,
    poll_oauth_device_code_flow,
)
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_AUTH_BASE_URL = "https://auth.openai.com"
_AUTHORIZE_URL = f"{_AUTH_BASE_URL}/oauth/authorize"
_TOKEN_URL = f"{_AUTH_BASE_URL}/oauth/token"
_REDIRECT_URI = "http://localhost:1455/auth/callback"
_DEVICE_USER_CODE_URL = f"{_AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
_DEVICE_TOKEN_URL = f"{_AUTH_BASE_URL}/api/accounts/deviceauth/token"
_DEVICE_VERIFICATION_URI = f"{_AUTH_BASE_URL}/codex/device"
_DEVICE_REDIRECT_URI = f"{_AUTH_BASE_URL}/deviceauth/callback"
_DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
_BROWSER_LOGIN_METHOD = "browser"
_DEVICE_CODE_LOGIN_METHOD = "device_code"
_SCOPE = "openid profile email offline_access"
_JWT_CLAIM_PATH = "https://api.openai.com/auth"


@dataclass
class _OAuthToken:
    access: str
    refresh: str
    expires: int | float


@dataclass
class _DeviceAuthInfo:
    device_auth_id: str
    user_code: str
    interval_seconds: int | float


@dataclass
class _DeviceTokenSuccess:
    authorization_code: str
    code_verifier: str


def _decode_jwt(token: str) -> object:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = re.sub(r"[\t\n\f\r ]", "", parts[1])
        if len(payload) % 4 == 0:
            payload = payload[:-2] if payload.endswith("==") else payload[:-1] if payload.endswith("=") else payload
        if len(payload) % 4 == 1 or re.search(r"[^A-Za-z0-9+/]", payload):
            return None
        decoded = base64.b64decode(payload + "=" * (-len(payload) % 4), validate=True).decode("latin-1")
        return json_parse(decoded)
    except Exception:
        return None


async def _fetch_with_login_cancellation(url: str, *, headers: Mapping[str, str], body: str, signal: AbortSignal) -> OAuthHttpResponse:
    try:
        return await fetch(url, method="POST", headers=headers, body=body, signal=signal)
    except Exception:
        if signal.aborted:
            raise RuntimeError("Login cancelled") from None
        raise


async def _read_token_response(response: OAuthHttpResponse, operation: Literal["exchange", "refresh"]) -> _OAuthToken:
    if not response.ok:
        try:
            text = await response.text()
        except Exception:
            text = ""
        raise RuntimeError(f"OpenAI Codex token {operation} failed ({response.status}): {text or response.status_text}")
    data = await response.json()
    access = property_value(data, "access_token") if data is not None else None
    refresh = property_value(data, "refresh_token") if data is not None else None
    expires_in = property_value(data, "expires_in") if data is not None else None
    if not js_truthy(access) or not js_truthy(refresh) or type(expires_in) not in (int, float):
        raise RuntimeError(f"OpenAI Codex token {operation} response missing fields: {javascript_json_stringify(data)}")
    return _OAuthToken(cast(str, access), cast(str, refresh), time.time_ns() // 1_000_000 + cast(int | float, expires_in) * 1000)


async def _exchange_authorization_code(code: str, verifier: str, redirect_uri: str, signal: AbortSignal) -> _OAuthToken:
    response = await _fetch_with_login_cancellation(
        _TOKEN_URL, headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=form_url_encode({
            "grant_type": "authorization_code", "client_id": _CLIENT_ID, "code": code,
            "code_verifier": verifier, "redirect_uri": redirect_uri,
        }), signal=signal,
    )
    return await _read_token_response(response, "exchange")


async def _refresh_access_token(refresh_token: str, signal: AbortSignal) -> _OAuthToken:
    try:
        response = await fetch(
            _TOKEN_URL, method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=form_url_encode({"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": _CLIENT_ID}),
            signal=signal,
        )
    except Exception as error:
        raise RuntimeError(f"OpenAI Codex token refresh error: {error}") from None
    return await _read_token_response(response, "refresh")


async def _start_device_auth(signal: AbortSignal) -> _DeviceAuthInfo:
    response = await _fetch_with_login_cancellation(
        _DEVICE_USER_CODE_URL, headers={"Content-Type": "application/json"},
        body=cast(str, javascript_json_stringify({"client_id": _CLIENT_ID})), signal=signal,
    )
    if not response.ok:
        if response.status == 404:
            await response.cancel_body()
            raise RuntimeError("OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL.")
        try:
            body = await response.text()
        except Exception:
            body = ""
        raise RuntimeError(f"OpenAI Codex device code request failed with status {response.status}" + (f": {body}" if body else ""))
    data = await response.json()
    device_id = property_value(data, "device_auth_id") if data is not None else None
    user_code = property_value(data, "user_code") if data is not None else None
    interval = property_value(data, "interval") if data is not None else None
    interval_seconds = js_number(interval.strip(JS_WHITESPACE)) if isinstance(interval, str) else interval
    if not js_truthy(device_id) or not js_truthy(user_code) or type(interval_seconds) not in (int, float) or not math.isfinite(cast(int | float, interval_seconds)) or cast(int | float, interval_seconds) < 0:
        raise RuntimeError(f"Invalid OpenAI Codex device code response: {javascript_json_stringify(data)}")
    return _DeviceAuthInfo(cast(str, device_id), cast(str, user_code), cast(int | float, interval_seconds))


async def _poll_device_auth(device: _DeviceAuthInfo, signal: AbortSignal) -> _DeviceTokenSuccess:
    async def poll() -> OAuthDeviceCodePollResult[_DeviceTokenSuccess]:
        response = await _fetch_with_login_cancellation(
            _DEVICE_TOKEN_URL, headers={"Content-Type": "application/json"},
            body=cast(str, javascript_json_stringify({"device_auth_id": device.device_auth_id, "user_code": device.user_code})),
            signal=signal,
        )
        if response.ok:
            data = await response.json()
            code = property_value(data, "authorization_code") if data is not None else None
            verifier = property_value(data, "code_verifier") if data is not None else None
            if not js_truthy(code) or not js_truthy(verifier):
                return OAuthDeviceCodeFailed(f"Invalid OpenAI Codex device auth token response: {javascript_json_stringify(data)}")
            return OAuthDeviceCodeComplete(_DeviceTokenSuccess(cast(str, code), cast(str, verifier)))
        if response.status in (403, 404):
            await response.cancel_body()
            return OAuthDeviceCodePending()
        try:
            body = await response.text()
        except Exception:
            body = ""
        error_code: object = None
        try:
            data = json_parse(body)
            error = property_value(data, "error") if data is not None else None
            error_code = property_value(error, "code") if error is not None and isinstance(error, (Mapping, list)) else error
        except Exception:
            pass
        if error_code == "deviceauth_authorization_pending":
            return OAuthDeviceCodePending()
        if error_code == "slow_down":
            return OAuthDeviceCodeSlowDown()
        return OAuthDeviceCodeFailed(f"OpenAI Codex device auth failed with status {response.status}" + (f": {body}" if body else ""))

    return await poll_oauth_device_code_flow(OAuthDeviceCodePollOptions(
        poll=poll, interval_seconds=device.interval_seconds, expires_in_seconds=_DEVICE_CODE_TIMEOUT_SECONDS, signal=signal,
    ))


@dataclass
class _AuthorizationFlow:
    verifier: str
    state: str
    url: str


async def _create_authorization_flow(originator: str = "pi") -> _AuthorizationFlow:
    pkce = await generate_pkce()
    state = secrets.token_bytes(16).hex()
    url = parse_url(_AUTHORIZE_URL)
    url.search = form_url_encode({
        "response_type": "code", "client_id": _CLIENT_ID, "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPE, "code_challenge": pkce.challenge, "code_challenge_method": "S256",
        "state": state, "id_token_add_organizations": "true", "codex_cli_simplified_flow": "true", "originator": originator,
    })
    return _AuthorizationFlow(pkce.verifier, state, url.href)


@dataclass
class _Callback:
    close: Callable[[], None]
    code: asyncio.Future[str | None]

    def cancel_wait(self) -> None:
        if not self.code.done():
            self.code.set_result(None)

    async def wait_for_code(self) -> str | None:
        return await await_promise(self.code)


async def _start_local_oauth_server(state: str) -> _Callback:
    result: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()

    async def handle(request: CallbackRequest, response: CallbackResponse) -> None:
        try:
            url = parse_url(request.target or "", "http://localhost")
            if url.pathname != "/auth/callback":
                response.send(404, oauth_error_html("Callback route not found."))
                return
            if search_param(url.search, "state") != state:
                response.send(400, oauth_error_html("State mismatch."))
                return
            code = search_param(url.search, "code")
            if not code:
                response.send(400, oauth_error_html("Missing authorization code."))
                return
            response.send(200, oauth_success_html("OpenAI authentication completed. You can close this window."))
            if not result.done():
                result.set_result(code)
        except Exception:
            response.send(500, oauth_error_html("Internal error while processing OAuth callback."))

    try:
        server = await CallbackServer.listen(get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1", 1455, handle)
    except OSError:
        result.set_result(None)
        return _Callback(lambda: None, result)
    return _Callback(server.close, result)


def _get_account_id(access_token: str) -> str | None:
    payload = _decode_jwt(access_token)
    auth = property_value(payload, _JWT_CLAIM_PATH) if payload is not None else None
    account_id = property_value(auth, "chatgpt_account_id") if auth is not None else None
    return account_id if isinstance(account_id, str) and account_id else None


def _credentials_from_token(token: _OAuthToken) -> OAuthCredential:
    account_id = _get_account_id(token.access)
    if not account_id:
        raise RuntimeError("Failed to extract accountId from token")
    return OAuthCredential(access=token.access, refresh=token.refresh, expires=token.expires, extra={"accountId": account_id})


async def _exchange_authorization_code_for_credentials(code: str, verifier: str, redirect_uri: str, signal: AbortSignal) -> OAuthCredential:
    return _credentials_from_token(await _exchange_authorization_code(code, verifier, redirect_uri, signal))


async def _login_device_code(interaction: ProviderAuthInteraction) -> OAuthCredential:
    device = await _start_device_auth(interaction.signal)
    interaction.notify(AuthDeviceCodeEvent(device.user_code, _DEVICE_VERIFICATION_URI, device.interval_seconds, _DEVICE_CODE_TIMEOUT_SECONDS))
    code = await _poll_device_auth(device, interaction.signal)
    return await _exchange_authorization_code_for_credentials(code.authorization_code, code.code_verifier, _DEVICE_REDIRECT_URI, interaction.signal)


async def _login_browser(interaction: ProviderAuthInteraction) -> OAuthCredential:
    flow = await _create_authorization_flow()
    server = await _start_local_oauth_server(flow.state)
    manual_abort = AbortController()
    interaction.signal.add_event_listener("abort", server.cancel_wait, once=True)
    if interaction.signal.aborted:
        server.cancel_wait()
    code: str | None = None
    interaction.notify(AuthUrlEvent(flow.url, "A browser window should open. Complete login to finish."))
    try:
        manual = ManualPrompt(interaction, ManualCodeAuthPrompt(
            "Complete login in your browser, or paste the authorization code / redirect URL here:",
            placeholder=_REDIRECT_URI, signal=manual_abort.signal,
        ), server.cancel_wait)
        result = await server.wait_for_code()
        if manual.error is not None:
            raise manual.error
        if result:
            code = result
        elif manual.input:
            parsed = parse_authorization_input(manual.input)
            if parsed.state and parsed.state != flow.state:
                raise RuntimeError("State mismatch")
            code = parsed.code
        if not code:
            await manual.wait()
            if manual.error is not None:
                raise manual.error
            if manual.input:
                parsed = parse_authorization_input(manual.input)
                if parsed.state and parsed.state != flow.state:
                    raise RuntimeError("State mismatch")
                code = parsed.code
        if not code:
            raise RuntimeError("Missing authorization code")
        exchange = defer_operation(_exchange_authorization_code_for_credentials(code, flow.verifier, _REDIRECT_URI, interaction.signal))
    finally:
        interaction.signal.remove_event_listener("abort", server.cancel_wait)
        manual_abort.abort()
        server.close()
    return await exchange


async def _login(interaction: ProviderAuthInteraction) -> OAuthCredential:
    method = await interaction.prompt(SelectAuthPrompt("Select OpenAI Codex login method:", [
        AuthSelectOption(_BROWSER_LOGIN_METHOD, "Browser login (default)"),
        AuthSelectOption(_DEVICE_CODE_LOGIN_METHOD, "Device code login (headless)"),
    ]))
    if method == _DEVICE_CODE_LOGIN_METHOD:
        return await _login_device_code(interaction)
    if method != _BROWSER_LOGIN_METHOD:
        raise RuntimeError(f"Unknown OpenAI Codex login method: {method}")
    return await _login_browser(interaction)


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    return _credentials_from_token(await _refresh_access_token(credential.refresh, signal))


openai_codex_oauth = OAuthAuth(
    name="OpenAI (ChatGPT Plus/Pro)", is_subscription=True,
    login=_login, refresh=_refresh, to_auth=credential_to_auth,
)

__all__ = ["openai_codex_oauth"]
