"""Radius gateway browser and device-code OAuth flows."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from ...abort import AbortSignal
from ...providers.radius_config import normalize_radius_gateway_url
from ..types import (
    AuthDeviceCodeEvent, AuthProgressEvent, AuthSelectOption, AuthUrlEvent,
    OAuthAuth, OAuthCredential, ProviderAuthInteraction, SelectAuthPrompt,
)
from ._callback_server import CallbackRequest, CallbackResponse, CallbackServer
from ._common import await_promise, credential_to_auth, js_number, js_truthy, json_parse, property_value, search_param
from ._http import OAuthHttpResponse, fetch, form_url_encode, parse_url
from .device_code import (
    OAuthDeviceCodeComplete, OAuthDeviceCodeFailed, OAuthDeviceCodePending,
    OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, OAuthDeviceCodeSlowDown,
    poll_oauth_device_code_flow,
)
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

_CALLBACK_HOST = "127.0.0.1"
_CALLBACK_PORT = 1456
_CALLBACK_PATH = "/oauth/callback"
_REDIRECT_URI = f"http://{_CALLBACK_HOST}:{_CALLBACK_PORT}{_CALLBACK_PATH}"
_TOKEN_EXPIRY_SKEW_MS = 60_000
_LOGIN_METHOD_BROWSER = "browser"
_LOGIN_METHOD_DEVICE_CODE = "device-code"
_OAUTH_CLIENT_ID = "pi-gateway"
_OAUTH_SCOPE = "gateway offline_access"
_OAUTH_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


@dataclass
class _DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int | float
    interval: int | float | None = None


async def _load_radius_oauth_discovery(gateway: str, signal: AbortSignal) -> str:
    response = await fetch(parse_url("/v1/oauth", gateway).href, headers={"accept": "application/json"}, signal=signal)
    if not response.ok:
        raise RuntimeError(f"Could not load Radius OAuth config from {gateway}: {response.status} {await response.text()}")
    discovery = await response.json()
    endpoint = property_value(discovery, "authorizationEndpoint")
    if not isinstance(endpoint, str):
        raise RuntimeError(f"Invalid Radius OAuth config from {gateway}")
    return endpoint


class OAuthResponseError(RuntimeError):
    def __init__(self, status: int, oauth_error: str | None, description: str | None, message: str) -> None:
        detail = f"{oauth_error}: {description}" if oauth_error and description else oauth_error if oauth_error else description or str(status)
        super().__init__(f"{message}: {detail}")
        self.status = status
        self.oauth_error = oauth_error


async def _read_oauth_response_error(response: OAuthHttpResponse, message: str) -> OAuthResponseError:
    try:
        text = await response.text()
    except Exception:
        text = ""
    oauth_error: str | None = None
    description: str | None = None
    if text:
        try:
            data = json_parse(text)
            error = property_value(data, "error")
            detail = property_value(data, "error_description")
            oauth_error = error if isinstance(error, str) else None
            description = detail if isinstance(detail, str) else None
        except Exception:
            description = text
    return OAuthResponseError(response.status, oauth_error, description, message)


async def _request_oauth_token(gateway: str, body: Mapping[str, str], signal: AbortSignal) -> OAuthCredential:
    try:
        response = await fetch(
            parse_url("/v1/oauth/token", gateway).href, method="POST",
            headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
            body=form_url_encode(body), signal=signal,
        )
    except Exception:
        if signal.aborted:
            raise RuntimeError("Login cancelled") from None
        raise
    if not response.ok:
        raise await _read_oauth_response_error(response, "Radius OAuth token request failed")
    data = await response.json()
    access = property_value(data, "access_token")
    refresh = property_value(data, "refresh_token")
    now = time.time_ns() // 1_000_000
    expires_in = property_value(data, "expires_in")
    extra: dict[str, object] = {}
    if isinstance(data, Mapping) and "scope" in data:
        extra["scope"] = data["scope"]
    return OAuthCredential(
        access=cast(str, access), refresh=cast(str, refresh),
        expires=now + js_number(expires_in, missing=not isinstance(data, Mapping) or "expires_in" not in data) * 1000 - _TOKEN_EXPIRY_SKEW_MS,
        extra=extra,
    )


@dataclass
class _Callback:
    code: asyncio.Future[str | None]
    close: Callable[[], None]

    async def wait_for_code(self) -> str | None:
        return await await_promise(self.code)


async def _start_oauth_callback_server(expected_state: str, signal: AbortSignal) -> _Callback:
    result: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
    settled = False

    def finish(code: str | None) -> None:
        nonlocal settled
        if settled:
            return
        settled = True
        signal.remove_event_listener("abort", on_abort)
        if not result.done():
            result.set_result(code)

    def on_abort() -> None:
        finish(None)

    signal.add_event_listener("abort", on_abort, once=True)

    async def handle(request: CallbackRequest, response: CallbackResponse) -> None:
        url = parse_url(request.target or "/", _REDIRECT_URI)
        if url.pathname != _CALLBACK_PATH:
            response.send(404, oauth_error_html("Callback route not found."))
            return
        if search_param(url.search, "state") != expected_state:
            response.send(400, oauth_error_html("OAuth state mismatch."))
            return
        error = search_param(url.search, "error")
        if error:
            description = search_param(url.search, "error_description")
            response.send(400, oauth_error_html(description if description is not None else error))
            finish(None)
            return
        code = search_param(url.search, "code")
        if not code:
            response.send(400, oauth_error_html("Missing authorization code."))
            return
        response.send(200, oauth_success_html("Signed in to Radius. You may now close this page."))
        finish(code)

    try:
        server = await CallbackServer.listen(_CALLBACK_HOST, _CALLBACK_PORT, handle)
    except OSError:
        finish(None)
        return _Callback(result, lambda: None)

    def close() -> None:
        finish(None)
        server.close()

    return _Callback(result, close)


async def _login_with_browser(gateway: str, authorization_endpoint: str, interaction: ProviderAuthInteraction) -> OAuthCredential:
    pkce = await generate_pkce()
    state = str(uuid.uuid4())
    authorize_url = parse_url(authorization_endpoint)
    authorize_url.search = form_url_encode({
        "response_type": "code", "client_id": _OAUTH_CLIENT_ID, "redirect_uri": _REDIRECT_URI,
        "scope": _OAUTH_SCOPE, "code_challenge": pkce.challenge, "code_challenge_method": "S256",
        "handoff": "url", "state": state,
    })
    callback = await _start_oauth_callback_server(state, interaction.signal)
    interaction.notify(AuthProgressEvent(f"Listening for OAuth callback on {_REDIRECT_URI}"))
    interaction.notify(AuthUrlEvent(authorize_url.href, "Continue in your browser."))
    try:
        code = await callback.wait_for_code()
        if not code:
            if interaction.signal.aborted:
                raise RuntimeError("Login cancelled")
            raise RuntimeError("OAuth callback did not complete.")
        return await _request_oauth_token(gateway, {
            "grant_type": "authorization_code", "client_id": _OAUTH_CLIENT_ID,
            "redirect_uri": _REDIRECT_URI, "code": code, "code_verifier": pkce.verifier,
        }, interaction.signal)
    finally:
        callback.close()


async def _request_device_authorization(gateway: str, signal: AbortSignal) -> _DeviceAuthorization:
    try:
        response = await fetch(
            parse_url("/v1/oauth/device", gateway).href, method="POST",
            headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
            body=form_url_encode({"client_id": _OAUTH_CLIENT_ID, "scope": _OAUTH_SCOPE}), signal=signal,
        )
    except Exception:
        if signal.aborted:
            raise RuntimeError("Login cancelled") from None
        raise
    if not response.ok:
        raise await _read_oauth_response_error(response, "Radius OAuth device authorization failed")
    data = await response.json()
    device_code, user_code, verification_uri, expires_in = (property_value(data, key) for key in ("device_code", "user_code", "verification_uri", "expires_in"))
    if not all(js_truthy(value) for value in (device_code, user_code, verification_uri, expires_in)):
        raise RuntimeError("Radius OAuth device authorization response is missing required fields")
    return _DeviceAuthorization(
        cast(str, device_code), cast(str, user_code), cast(str, verification_uri),
        cast(int | float, expires_in), cast(int | float | None, property_value(data, "interval")),
    )


async def _login_with_device_code(gateway: str, interaction: ProviderAuthInteraction) -> OAuthCredential:
    device = await _request_device_authorization(gateway, interaction.signal)
    interaction.notify(AuthDeviceCodeEvent(device.user_code, device.verification_uri, device.interval, device.expires_in))

    async def poll() -> OAuthDeviceCodePollResult[OAuthCredential]:
        try:
            credential = await _request_oauth_token(gateway, {
                "grant_type": _OAUTH_DEVICE_CODE_GRANT_TYPE, "client_id": _OAUTH_CLIENT_ID, "device_code": device.device_code,
            }, interaction.signal)
            return OAuthDeviceCodeComplete(credential)
        except OAuthResponseError as error:
            if error.oauth_error == "authorization_pending":
                return OAuthDeviceCodePending()
            if error.oauth_error == "slow_down":
                return OAuthDeviceCodeSlowDown()
            if error.oauth_error == "expired_token":
                return OAuthDeviceCodeFailed("Device authorization expired.")
            if error.oauth_error == "access_denied":
                return OAuthDeviceCodeFailed("Device authorization was denied.")
            raise

    return await poll_oauth_device_code_flow(OAuthDeviceCodePollOptions(
        poll=poll, signal=interaction.signal, interval_seconds=device.interval, expires_in_seconds=device.expires_in,
    ))


@dataclass
class RadiusOAuthOptions:
    name: str
    gateway: str


def create_radius_oauth(options: RadiusOAuthOptions) -> OAuthAuth:
    gateway = normalize_radius_gateway_url(options.gateway)

    async def login(interaction: ProviderAuthInteraction) -> OAuthCredential:
        method = await interaction.prompt(SelectAuthPrompt(f"Sign in to {options.name}:", [
            AuthSelectOption(_LOGIN_METHOD_BROWSER, "Sign in with browser (recommended)"),
            AuthSelectOption(_LOGIN_METHOD_DEVICE_CODE, "Sign in with device code (when signing in from another device)"),
        ]))
        if method == _LOGIN_METHOD_DEVICE_CODE:
            return await _login_with_device_code(gateway, interaction)
        if method == _LOGIN_METHOD_BROWSER:
            endpoint = await _load_radius_oauth_discovery(gateway, interaction.signal)
            return await _login_with_browser(gateway, endpoint, interaction)
        raise RuntimeError(f"Unknown {options.name} sign-in method: {method}")

    async def refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
        return await _request_oauth_token(gateway, {
            "grant_type": "refresh_token", "client_id": _OAUTH_CLIENT_ID, "refresh_token": credential.refresh,
        }, signal)

    return OAuthAuth(name=options.name, login=login, refresh=refresh, to_auth=credential_to_auth)


__all__ = ["RadiusOAuthOptions", "create_radius_oauth"]
