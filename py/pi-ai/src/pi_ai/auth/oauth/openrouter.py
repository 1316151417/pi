"""OpenRouter OAuth PKCE, including callback/manual ownership of key exchange."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...abort import AbortController, AbortSignal
from ...utils._javascript import javascript_json_stringify
from ...utils.provider_env import get_provider_env_value
from ..types import AuthProgressEvent, AuthUrlEvent, ManualCodeAuthPrompt, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from ._callback_server import CallbackRequest, CallbackResponse, CallbackServer
from ._common import ManualPrompt, await_promise, credential_to_auth, parse_authorization_input, search_param
from ._http import fetch, form_url_encode, parse_url
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

_AUTHORIZE_URL = "https://openrouter.ai/auth"
_TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
_LOGIN_TIMEOUT_MS = 5 * 60 * 1000
_TOKEN_EXCHANGE_TIMEOUT_MS = 30_000


def _error_detail(body: Mapping[str, object]) -> str | None:
    for key in ("error_description", "message", "error"):
        if isinstance(body.get(key), str):
            return str(body[key])
    error = body.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
        return str(error["message"])
    return None


async def _exchange_authorization_code(code: str, verifier: str, signal: AbortSignal) -> OAuthCredential:
    if signal.aborted:
        raise RuntimeError("Login cancelled")
    controller = AbortController()

    def on_abort() -> None:
        controller.abort(signal.reason)

    signal.add_event_listener("abort", on_abort, once=True)
    timeout = asyncio.get_running_loop().call_later(
        _TOKEN_EXCHANGE_TIMEOUT_MS / 1000, controller.abort,
        RuntimeError("OpenRouter OAuth token exchange timed out"),
    )
    body: Mapping[str, object] = {}
    try:
        response = await fetch(
            _TOKEN_URL, method="POST", headers={"accept": "application/json", "content-type": "application/json"},
            body=javascript_json_stringify({"code": code, "code_verifier": verifier, "code_challenge_method": "S256"}),
            signal=controller.signal,
        )
        try:
            parsed = await response.json()
            if isinstance(parsed, Mapping):
                body = parsed
        except Exception:
            if response.ok:
                raise RuntimeError("OpenRouter OAuth returned invalid JSON") from None
    except Exception:
        if signal.aborted:
            raise RuntimeError("Login cancelled") from None
        if controller.signal.aborted:
            raise RuntimeError("OpenRouter OAuth token exchange timed out") from None
        raise
    finally:
        timeout.cancel()
        signal.remove_event_listener("abort", on_abort)
    if not response.ok:
        detail = _error_detail(body)
        raise RuntimeError(f"OpenRouter OAuth key exchange failed (HTTP {response.status})" + (f": {detail}" if detail else ""))
    key = body.get("key")
    if not isinstance(key, str) or not key:
        raise RuntimeError('OpenRouter OAuth response carries no "key"')
    return OAuthCredential(access=key, refresh="", expires=9007199254740991)


@dataclass
class _Callback:
    callback_url: str
    close: Callable[[], None]
    cancel_wait: Callable[[], None]
    credential: asyncio.Future[OAuthCredential | None]

    async def wait_for_credential(self) -> OAuthCredential | None:
        return await await_promise(self.credential)


async def _start_callback_server(callback_path: str, verifier: str, signal: AbortSignal) -> _Callback:
    if signal.aborted:
        raise RuntimeError("Login cancelled")
    host = get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"
    loop = asyncio.get_running_loop()
    credential: asyncio.Future[OAuthCredential | None] = loop.create_future()
    claimed = settled = False
    server: CallbackServer | None = None
    timeout: asyncio.TimerHandle | None = None
    on_abort: Callable[[], None] | None = None

    def close() -> None:
        if timeout is not None:
            timeout.cancel()
        if on_abort is not None:
            signal.remove_event_listener("abort", on_abort)
        if server is not None:
            server.close()

    def finish(value: OAuthCredential | None = None, error: BaseException | None = None) -> None:
        nonlocal settled
        if settled:
            return
        settled = True
        close()
        if not credential.done():
            if error is not None:
                credential.set_exception(error)
            else:
                credential.set_result(value)

    async def handle(request: CallbackRequest, response: CallbackResponse) -> None:
        nonlocal claimed
        url = parse_url(request.target or "/", f"http://{host}")
        if request.method != "GET" or url.pathname != callback_path:
            response.send(404, oauth_error_html("OAuth callback route not found."), no_store=True)
            return
        if claimed or settled:
            response.send(409, oauth_error_html("This OAuth callback has already been used."), no_store=True)
            return
        error = search_param(url.search, "error")
        if error:
            description = search_param(url.search, "error_description")
            if description is None:
                description = error
            response.send(400, oauth_error_html("OpenRouter authorization was denied.", description), no_store=True)
            finish(error=RuntimeError(f"OpenRouter authorization failed: {description}"))
            return
        code = search_param(url.search, "code")
        if not code:
            response.send(400, oauth_error_html("OpenRouter returned no authorization code."), no_store=True)
            return
        claimed = True
        try:
            result = await _exchange_authorization_code(code, verifier, signal)
            response.send(200, oauth_success_html("Signed in to OpenRouter. You may now close this page."), no_store=True)
            finish(result)
        except Exception as error:
            response.send(502, oauth_error_html("OpenRouter key exchange failed.", str(error)), no_store=True)
            finish(error=error)

    server = await CallbackServer.listen(host, 0, handle)

    def abort() -> None:
        finish(error=RuntimeError("Login cancelled"))

    on_abort = abort
    signal.add_event_listener("abort", on_abort, once=True)
    if signal.aborted:
        close()
        raise RuntimeError("Login cancelled")
    timeout = loop.call_later(_LOGIN_TIMEOUT_MS / 1000, lambda: finish(error=RuntimeError("OpenRouter OAuth login timed out")))
    port = server.port
    if port is None:
        close()
        raise RuntimeError("Could not determine the OpenRouter OAuth callback port")

    def cancel_wait() -> None:
        if not claimed:
            finish()

    return _Callback(f"http://{host}:{port}{callback_path}", close, cancel_wait, credential)


async def _login_openrouter(interaction: ProviderAuthInteraction) -> OAuthCredential:
    pkce = await generate_pkce()
    callback = await _start_callback_server(f"/oauth/callback/{uuid.uuid4()}", pkce.verifier, interaction.signal)
    manual_abort = AbortController()
    try:
        authorize_url = parse_url(_AUTHORIZE_URL)
        authorize_url.search = form_url_encode({
            "callback_url": callback.callback_url, "code_challenge": pkce.challenge, "code_challenge_method": "S256",
        })
        interaction.notify(AuthProgressEvent(f"Listening for OpenRouter OAuth callback on {callback.callback_url}"))
        interaction.notify(AuthUrlEvent(authorize_url.href, "Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here."))
        manual = ManualPrompt(interaction, ManualCodeAuthPrompt(
            "Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
            placeholder=callback.callback_url, signal=manual_abort.signal,
        ), callback.cancel_wait)
        credential = await callback.wait_for_credential()
        if manual.error is not None:
            raise manual.error
        if credential is not None:
            return credential
        await manual.wait()
        if manual.error is not None:
            raise manual.error
        code = parse_authorization_input(manual.input, hash_state=False).code if manual.input else None
        if not code:
            raise RuntimeError("Missing authorization code")
        interaction.notify(AuthProgressEvent("Exchanging authorization code for an API key..."))
        return await _exchange_authorization_code(code, pkce.verifier, interaction.signal)
    finally:
        manual_abort.abort()
        callback.close()


async def _refresh(credential: OAuthCredential, _signal: AbortSignal) -> OAuthCredential:
    return credential


open_router_oauth = OAuthAuth(
    name="OpenRouter OAuth", login_label="Sign in with OpenRouter", login=_login_openrouter,
    refresh=_refresh, to_auth=credential_to_auth,
)

__all__ = ["open_router_oauth"]
