"""Anthropic Claude Pro/Max OAuth with native loopback callback and PKCE."""

from __future__ import annotations

import asyncio
import base64
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from ...abort import AbortController, AbortSignal, any_signal
from ...utils._javascript import javascript_json_stringify, javascript_string
from ...utils.provider_env import get_provider_env_value
from ..types import AuthProgressEvent, AuthUrlEvent, ManualCodeAuthPrompt, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from ._callback_server import CallbackRequest, CallbackResponse, CallbackServer
from ._common import ManualPrompt, await_promise, credential_to_auth, defer_operation, js_number, json_parse, parse_authorization_input, property_value, search_param
from ._http import fetch, form_url_encode, parse_url, timeout_signal
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

_CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode("ascii")
_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_CALLBACK_HOST = get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"
_CALLBACK_PORT = 53692
_CALLBACK_PATH = "/callback"
_REDIRECT_URI = f"http://localhost:{_CALLBACK_PORT}{_CALLBACK_PATH}"
_SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"


@dataclass
class _Code:
    code: str
    state: str


@dataclass
class _Callback:
    server: CallbackServer
    code: asyncio.Future[_Code | None]

    def cancel_wait(self) -> None:
        if not self.code.done():
            self.code.set_result(None)

    async def wait_for_code(self) -> _Code | None:
        return await await_promise(self.code)


def _format_error_details(error: object) -> str:
    if not isinstance(error, BaseException):
        return javascript_string(error)
    details = [f"{getattr(error, 'name', type(error).__name__)}: {error}"]
    code = getattr(error, "code", None)
    if code:
        details.append(f"code={code}")
    if hasattr(error, "errno"):
        details.append(f"errno={javascript_string(getattr(error, 'errno'))}")
    cause = getattr(error, "cause", error.__cause__)
    if cause is not None:
        details.append(f"cause={_format_error_details(cause)}")
    stack = getattr(error, "stack", None)
    if stack is None and error.__traceback__ is not None:
        stack = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    if stack:
        details.append(f"stack={stack}")
    return "; ".join(details)


async def _start_callback_server(expected_state: str) -> _Callback:
    result: asyncio.Future[_Code | None] = asyncio.get_running_loop().create_future()

    async def handle(request: CallbackRequest, response: CallbackResponse) -> None:
        try:
            url = parse_url(request.target or "", "http://localhost")
            if url.pathname != _CALLBACK_PATH:
                response.send(404, oauth_error_html("Callback route not found."))
                return
            code, state, error = (search_param(url.search, key) for key in ("code", "state", "error"))
            if error:
                response.send(400, oauth_error_html("Anthropic authentication did not complete.", f"Error: {error}"))
                return
            if not code or not state:
                response.send(400, oauth_error_html("Missing code or state parameter."))
                return
            if state != expected_state:
                response.send(400, oauth_error_html("State mismatch."))
                return
            response.send(200, oauth_success_html("Anthropic authentication completed. You can close this window."))
            if not result.done():
                result.set_result(_Code(code, state))
        except Exception:
            response.send(500, "Internal error", content_type="text/plain; charset=utf-8")

    return _Callback(await CallbackServer.listen(_CALLBACK_HOST, _CALLBACK_PORT, handle), result)


async def _post_json(url: str, body: Mapping[str, str | int], signal: AbortSignal) -> str:
    response = await fetch(
        url, method="POST", headers={"Content-Type": "application/json", "Accept": "application/json"},
        body=javascript_json_stringify(dict(body)), signal=any_signal(signal, timeout_signal(30_000)),
    )
    response_body = await response.text()
    if not response.ok:
        raise RuntimeError(f"HTTP request failed. status={response.status}; url={url}; body={response_body}")
    return response_body


def _credential_from_token(data: object) -> OAuthCredential:
    refresh = property_value(data, "refresh_token")
    access = property_value(data, "access_token")
    now = time.time_ns() // 1_000_000
    expires = property_value(data, "expires_in")
    return OAuthCredential(
        refresh=cast(str, refresh), access=cast(str, access),
        expires=now + js_number(expires, missing=not isinstance(data, Mapping) or "expires_in" not in data) * 1000 - 5 * 60 * 1000,
    )


async def _exchange_authorization_code(code: str, state: str, verifier: str, redirect_uri: str, signal: AbortSignal) -> OAuthCredential:
    try:
        response_body = await _post_json(_TOKEN_URL, {
            "grant_type": "authorization_code", "client_id": _CLIENT_ID, "code": code,
            "state": state, "redirect_uri": redirect_uri, "code_verifier": verifier,
        }, signal)
    except Exception as error:
        raise RuntimeError(f"Token exchange request failed. url={_TOKEN_URL}; redirect_uri={redirect_uri}; response_type=authorization_code; details={_format_error_details(error)}") from None
    try:
        data = json_parse(response_body)
    except Exception as error:
        raise RuntimeError(f"Token exchange returned invalid JSON. url={_TOKEN_URL}; body={response_body}; details={_format_error_details(error)}") from None
    return _credential_from_token(data)


async def _login_anthropic(interaction: ProviderAuthInteraction) -> OAuthCredential:
    pkce = await generate_pkce()
    server = await _start_callback_server(pkce.verifier)
    manual_abort = AbortController()
    interaction.signal.add_event_listener("abort", server.cancel_wait, once=True)
    if interaction.signal.aborted:
        server.cancel_wait()
    code: str | None = None
    state: str | None = None
    try:
        params = form_url_encode({
            "code": "true", "client_id": _CLIENT_ID, "response_type": "code",
            "redirect_uri": _REDIRECT_URI, "scope": _SCOPES, "code_challenge": pkce.challenge,
            "code_challenge_method": "S256", "state": pkce.verifier,
        })
        interaction.notify(AuthUrlEvent(f"{_AUTHORIZE_URL}?{params}", "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."))
        manual = ManualPrompt(interaction, ManualCodeAuthPrompt(
            "Complete login in your browser, or paste the authorization code / redirect URL here:",
            placeholder=_REDIRECT_URI, signal=manual_abort.signal,
        ), server.cancel_wait)
        result = await server.wait_for_code()
        if manual.error is not None:
            raise manual.error
        if result is not None and result.code:
            code, state = result.code, result.state
        elif manual.input:
            parsed = parse_authorization_input(manual.input)
            if parsed.state and parsed.state != pkce.verifier:
                raise RuntimeError("OAuth state mismatch")
            code, state = parsed.code, parsed.state if parsed.state is not None else pkce.verifier
        if not code:
            await manual.wait()
            if manual.error is not None:
                raise manual.error
            if manual.input:
                parsed = parse_authorization_input(manual.input)
                if parsed.state and parsed.state != pkce.verifier:
                    raise RuntimeError("OAuth state mismatch")
                code, state = parsed.code, parsed.state if parsed.state is not None else pkce.verifier
        if not code:
            raise RuntimeError("Missing authorization code")
        if not state:
            raise RuntimeError("Missing OAuth state")
        interaction.notify(AuthProgressEvent("Exchanging authorization code for tokens..."))
        exchange = defer_operation(_exchange_authorization_code(code, state, pkce.verifier, _REDIRECT_URI, interaction.signal))
    finally:
        interaction.signal.remove_event_listener("abort", server.cancel_wait)
        manual_abort.abort()
        server.server.close()
    return await exchange


async def _refresh_anthropic_token(refresh_token: str, signal: AbortSignal) -> OAuthCredential:
    try:
        response_body = await _post_json(_TOKEN_URL, {
            "grant_type": "refresh_token", "client_id": _CLIENT_ID, "refresh_token": refresh_token,
        }, signal)
    except Exception as error:
        raise RuntimeError(f"Anthropic token refresh request failed. url={_TOKEN_URL}; details={_format_error_details(error)}") from None
    try:
        data = json_parse(response_body)
    except Exception as error:
        raise RuntimeError(f"Anthropic token refresh returned invalid JSON. url={_TOKEN_URL}; body={response_body}; details={_format_error_details(error)}") from None
    return _credential_from_token(data)


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    return await _refresh_anthropic_token(credential.refresh, signal)


anthropic_oauth = OAuthAuth(
    name="Anthropic (Claude Pro/Max)", is_subscription=True, login=_login_anthropic,
    refresh=_refresh, to_auth=credential_to_auth,
)

__all__ = ["anthropic_oauth"]
