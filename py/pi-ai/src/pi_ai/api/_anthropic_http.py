"""Native raw-response transport for Anthropic SDK 0.124.0's Messages path.

The public injection protocol follows ``client.beta.messages.create(...).
as_response()``. The HTTP implementation uses Python Fetch and identifies the
actual runtime in Stainless headers; request construction and errors follow the
locked JavaScript SDK. Pi disables SDK retries and applies its own retry policy.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import platform
import re
import sys
import weakref
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.parse import parse_qsl, quote

import httpx

from .._javascript import javascript_json_parse, javascript_json_stringify, javascript_object_keys, javascript_string
from .._json_runtime import JS_WHITESPACE
from .._values import UNDEFINED
from ..abort import AbortController, AbortSignal
from ..auth.oauth._http import fetch, parse_url
from ..types import FetchFunction
from ._anthropic_credentials import ResolvedCredentials, resolve_default_credentials
from ._anthropic_errors import AnthropicError, APIConnectionError, APIConnectionTimeoutError, APIError, APIUserAbortError, _truthy


class AnthropicResponse(Protocol):
    status: int
    ok: bool
    headers: Mapping[str, str]

    @property
    def body(self) -> AsyncIterable[bytes] | None: ...
    async def text(self) -> str: ...
    async def json(self) -> object: ...
    async def cancel_body(self) -> None: ...


@dataclass
class AnthropicRequestOptions:
    signal: AbortSignal | None = None
    timeout_ms: int | float | None = None
    max_retries: int = 0


class AnthropicRawResponsePromise(Protocol):
    def as_response(self) -> Awaitable[AnthropicResponse]: ...


class AnthropicMessages(Protocol):
    def create(self, params: Mapping[str, object], options: AnthropicRequestOptions) -> AnthropicRawResponsePromise: ...


class AnthropicBeta(Protocol):
    @property
    def messages(self) -> AnthropicMessages: ...


class AnthropicClient(Protocol):
    @property
    def beta(self) -> AnthropicBeta: ...


class _RawResponsePromise:
    def __init__(self, response: Awaitable[AnthropicResponse]) -> None:
        self._response = asyncio.ensure_future(response)

    def as_response(self) -> Awaitable[AnthropicResponse]:
        return asyncio.shield(self._response)


class _Messages:
    def __init__(self, client: AnthropicHttpClient) -> None:
        self._client = client

    def create(self, params: Mapping[str, object], options: AnthropicRequestOptions) -> AnthropicRawResponsePromise:
        modified = dict(params)
        if _truthy(modified.get("output_format", UNDEFINED)):
            output_config = modified.get("output_config")
            if isinstance(output_config, Mapping) and _truthy(output_config.get("format", UNDEFINED)):
                raise AnthropicError("Both output_format and output_config.format were provided. Please use only output_config.format (output_format is deprecated).")
            # Like object spread, null/undefined do not contribute properties.
            config = dict(output_config) if isinstance(output_config, Mapping) else {}
            config["format"] = modified.pop("output_format")
            modified["output_config"] = config
        betas = modified.pop("betas", UNDEFINED)
        user_profile_id = modified.pop("user_profile_id", UNDEFINED)
        workspace_id = modified.pop("workspace_id", UNDEFINED)
        model = modified.get("model", UNDEFINED)
        thinking = modified.get("thinking")
        if model in ("claude-mythos-preview", "claude-opus-4-6") and isinstance(thinking, Mapping) and thinking.get("type") == "enabled":
            logging.getLogger(__name__).warning(
                "Using Claude with %s and 'thinking.type=enabled' is deprecated. Use 'thinking.type=adaptive' instead which results in better model performance in our testing: https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking", model,
            )
        timeout = 600_000 if options.timeout_ms is None else options.timeout_ms
        if not _truthy(modified.get("stream", UNDEFINED)) and options.timeout_ms is None:
            # on_payload can mutate the original params in place without
            # returning a replacement, including switching streaming off.
            max_tokens = modified.get("max_tokens")
            max_nonstreaming = 8192 if model in (
                "claude-opus-4@20250514", "anthropic.claude-opus-4-1-20250805-v1:0", "claude-opus-4-1@20250805",
            ) else None
            if isinstance(max_tokens, (float, int)) and (
                3_600_000 * max_tokens / 128_000 > 600_000 or max_nonstreaming is not None and max_tokens > max_nonstreaming
            ):
                raise AnthropicError("Streaming is required for operations that may take longer than 10 minutes. See https://github.com/anthropics/anthropic-sdk-typescript#long-requests for more details")
        headers: dict[str, str] = {}
        if betas is not None and betas is not UNDEFINED:
            headers["anthropic-beta"] = javascript_string(betas)
        if user_profile_id is not None and user_profile_id is not UNDEFINED:
            headers["anthropic-user-profile-id"] = cast(str, user_profile_id)
        if workspace_id is not None and workspace_id is not UNDEFINED:
            headers["anthropic-workspace-id"] = cast(str, workspace_id)
        # SDK-helper symbols have no JSON representation. Python callers can
        # attach the analogous private attribute to helper-created dict objects.
        helpers: list[str] = []
        tools = modified.get("tools")
        messages = modified.get("messages")
        if _truthy(tools):
            for tool in cast(list[object], tools):
                tag = getattr(tool, "_anthropic_stainless_helper", None)
                if tag is not None and tag not in helpers:
                    helpers.append(tag)
        if _truthy(messages):
            for message in cast(list[object], messages):
                tag = getattr(message, "_anthropic_stainless_helper", None)
                if tag is not None and tag not in helpers:
                    helpers.append(tag)
                if message is None:
                    raise TypeError("Cannot read properties of null (reading 'content')")
                content = message.get("content") if isinstance(message, Mapping) else None
                if isinstance(content, list):
                    for block in content:
                        tag = getattr(block, "_anthropic_stainless_helper", None)
                        if tag is not None and tag not in helpers:
                            helpers.append(tag)
        if helpers:
            headers["x-stainless-helper"] = ", ".join(helpers)
        return _RawResponsePromise(self._client._post_message(modified, options, timeout, headers))


@dataclass
class _Beta:
    messages: AnthropicMessages


def _merge_headers(*sources: Mapping[str, str | None]) -> tuple[dict[str, str], set[str]]:
    headers: dict[str, str] = {}
    nulls: set[str] = set()
    for source in sources:
        for name, value in source.items():
            lower = name.lower()
            if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
                raise TypeError(f'Headers: "{name}" is an invalid header name.')
            if value is None:
                headers.pop(lower, None)
                nulls.add(lower)
                continue
            value = javascript_string(value).strip(" \t\r\n")
            if "\x00" in value or "\r" in value or "\n" in value:
                raise TypeError(f'Headers: "{value}" is an invalid header value.')
            value.encode("latin-1")
            if lower == "x-stainless-helper":
                tokens = [token.strip(JS_WHITESPACE) for token in headers.get(lower, "").split(",") if token.strip(JS_WHITESPACE)]
                for token in value.split(","):
                    token = token.strip(JS_WHITESPACE)
                    if token and token not in tokens:
                        tokens.append(token)
                value = ", ".join(tokens)
            headers[lower] = value
            nulls.discard(lower)
    return headers, nulls


class _BodyIterator:
    def __init__(self, owner: _TrackedBody) -> None:
        self._owner = owner
        self._iterator = aiter(owner._source)

    def __aiter__(self) -> _BodyIterator:
        return self

    async def __anext__(self) -> bytes:
        return await anext(self._iterator)


class _TrackedBody:
    def __init__(self, source: AsyncIterable[bytes], release_signal: Callable[[], None]) -> None:
        self._source = source
        self._finalizer = weakref.finalize(self, release_signal)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return _BodyIterator(self)


class _TrackedResponse:
    def __init__(self, source: AnthropicResponse, release_signal: Callable[[], None]) -> None:
        self._source = source
        self.status = source.status
        self.ok = source.ok
        self.headers = source.headers
        body = source.body
        self._body = _TrackedBody(body, release_signal) if body is not None else None
        if body is None:
            self._finalizer = weakref.finalize(self, release_signal)

    @property
    def body(self) -> AsyncIterable[bytes] | None:
        return self._body

    async def text(self) -> str:
        return await self._source.text()

    async def json(self) -> object:
        return await self._source.json()

    async def cancel_body(self) -> None:
        await self._source.cancel_body()


class AnthropicHttpClient:
    def __init__(
        self, *, api_key: str | None, auth_token: str | None, base_url: str | None = None,
        custom_fetch: FetchFunction | None = None, default_headers: Mapping[str, str | None] | None = None,
    ) -> None:
        self.api_key = api_key
        self.auth_token = auth_token
        env_base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip(JS_WHITESPACE) or None
        self.base_url = (base_url if base_url is not None else env_base_url) or "https://api.anthropic.com"
        self._base_url_explicit = bool(base_url if base_url is not None else env_base_url)
        self._custom_fetch = custom_fetch
        self._default_headers: dict[str, str | None] = {}
        for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").strip(JS_WHITESPACE).split("\n"):
            if ":" in line:
                name, value = line.split(":", 1)
                self._default_headers[name.strip(JS_WHITESPACE)] = value.strip(JS_WHITESPACE)
        self._default_headers.update(default_headers or {})
        self.beta = _Beta(_Messages(self))
        self._credentials: ResolvedCredentials | None = None
        self._credentials_error: BaseException | None = None
        self._credential_resolution: asyncio.Task[None] | None = None
        self._log_level = os.environ.get("ANTHROPIC_LOG", "").strip(JS_WHITESPACE) or "warn"
        if self._log_level not in ("off", "error", "warn", "info", "debug"):
            logging.getLogger(__name__).warning("process.env['ANTHROPIC_LOG'] was set to %s, expected one of %s",
                javascript_json_stringify(self._log_level), '["off","error","warn","info","debug"]')
            self._log_level = "warn"
        if api_key is None and auth_token is None:
            self._credential_resolution = asyncio.get_running_loop().create_task(self._resolve_credentials())

    async def _resolve_credentials(self) -> None:
        try:
            logger = logging.getLogger(__name__)
            self._credentials = await resolve_default_credentials(
                base_url=self.base_url, custom_fetch=self._custom_fetch, user_agent="Anthropic/Python 0.124.0",
                on_cache_write_error=lambda error: logger.debug("credential cache write failed (best-effort): %s", error) if self._log_level == "debug" else None,
                on_safety_warning=lambda message: logger.warning("%s", message) if self._log_level not in ("off", "error") else None,
                on_advisory_refresh_error=lambda error: logger.debug("advisory token refresh failed; serving cached token: %s", error) if self._log_level == "debug" else None,
            )
            if self._credentials is not None and self._credentials.base_url and not self._base_url_explicit:
                self.base_url = self._credentials.base_url.rstrip("/")
        except BaseException as error:
            self._credentials_error = error
        finally:
            self._credential_resolution = None

    async def _post_message(
        self, body: Mapping[str, object], options: AnthropicRequestOptions, timeout: int | float,
        request_headers: Mapping[str, str],
    ) -> AnthropicResponse:
        if self._credential_resolution is not None:
            await asyncio.shield(self._credential_resolution)
        path = "/v1/messages?beta=true"
        url = parse_url(self.base_url + (path[1:] if self.base_url.endswith("/") else path))
        query = dict(parse_qsl(url.search.removeprefix("?"), keep_blank_values=True))
        if query:
            url.search = "&".join(f"{quote(key, safe='~')}={quote(query[key], safe='~')}" for key in javascript_object_keys(query))
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout != math.trunc(timeout):
            raise AnthropicError("timeout must be an integer")
        if timeout < 0:
            raise AnthropicError("timeout must be a positive integer")
        raw_body = javascript_json_stringify(body)
        operating_system = {"darwin": "MacOS", "win32": "Windows", "linux": "Linux", "freebsd": "FreeBSD", "openbsd": "OpenBSD"}.get(sys.platform, f"Other:{sys.platform}" if sys.platform else "Unknown")
        machine = platform.machine()
        architecture = {"x86_64": "x64", "AMD64": "x64", "aarch64": "arm64", "arm64": "arm64", "arm": "arm", "x32": "x32"}.get(machine, f"other:{machine}" if machine else "unknown")
        defaults = {"accept": "application/json", "user-agent": "Anthropic/Python 0.124.0", "x-stainless-retry-count": "0",
            "x-stainless-lang": "python", "x-stainless-package-version": "0.124.0", "x-stainless-os": operating_system,
            "x-stainless-arch": architecture, "x-stainless-runtime": "python", "x-stainless-runtime-version": platform.python_version(),
            "anthropic-dangerous-direct-browser-access": "true", "anthropic-version": "2023-06-01"}
        if timeout:
            defaults["x-stainless-timeout"] = str(math.trunc(timeout / 1000))
        auth: dict[str, str] = {}
        used_token_cache = False
        if self._credentials_error is None:
            if self._credentials is not None and self.api_key is None:
                auth["authorization"] = f"Bearer {await self._credentials.token_cache.get_token()}"
                used_token_cache = True
            else:
                if self.api_key is not None:
                    auth["x-api-key"] = self.api_key
                if self.auth_token is not None:
                    auth["authorization"] = f"Bearer {self.auth_token}"
        headers, nulls = _merge_headers(defaults, auth, self._default_headers, {"content-type": "application/json"}, request_headers)
        if not (headers.get("x-api-key") or headers.get("authorization")):
            if self._credentials_error is not None:
                raise self._credentials_error
            if self._credentials is None and not nulls.intersection(("x-api-key", "authorization")):
                raise AnthropicError('Could not resolve authentication method. Expected one of apiKey, authToken, credentials, config, or profile to be set. Or for one of the "X-Api-Key" or "Authorization" headers to be explicitly omitted')
        signal = options.signal
        if signal is not None and signal.aborted:
            raise APIUserAbortError()
        controller = AbortController()

        def abort() -> None:
            controller.abort()

        def release_signal() -> None:
            if signal is not None:
                signal.remove_event_listener("abort", abort)

        if signal is not None:
            signal.add_event_listener("abort", abort, once=True)
        # prepareRequest adds token-derived headers after all header merges.
        if self._credentials is not None and self.api_key is None:
            for key, value in self._credentials.extra_headers.items():
                headers.setdefault(key.lower(), value)
            existing = [item.strip(JS_WHITESPACE) for item in headers.get("anthropic-beta", "").split(",")]
            if "oauth-2025-04-20" not in existing:
                headers["anthropic-beta"] = headers["anthropic-beta"] + ", oauth-2025-04-20" if "anthropic-beta" in headers else "oauth-2025-04-20"
        delay = 1 if timeout < 1 or timeout > 2_147_483_647 else timeout
        timer = asyncio.get_running_loop().call_later(delay / 1000, abort)
        transferred = False
        try:
            try:
                response = cast(AnthropicResponse, await (self._custom_fetch or fetch)(url.href,
                    method="POST", headers=httpx.Headers(headers, encoding="latin-1"), body=raw_body, signal=controller.signal))
            except (Exception, asyncio.CancelledError) as error:
                if signal is not None and signal.aborted:
                    raise APIUserAbortError() from error
                is_abort = getattr(error, "name", None) == "AbortError" or "FetchRequestCanceledException" in str(error)
                if is_abort or re.search(r"timed? ?out", str(error) + " " + str(getattr(error, "cause", "")), re.I):
                    raise APIConnectionTimeoutError() from error
                raise APIConnectionError(cause=error) from error
            finally:
                timer.cancel()
            if not response.ok:
                if response.status == 401 and used_token_cache and self._credentials is not None:
                    self._credentials.token_cache.invalidate()
                try:
                    error_text = await response.text()
                except (Exception, asyncio.CancelledError) as error:
                    error_text = str(error)
                try:
                    error_json = javascript_json_parse(error_text)
                except (ValueError, TypeError):
                    error_json = UNDEFINED
                raise APIError.generate(response.status, error_json, None if _truthy(error_json) else error_text, response.headers)
            tracked = _TrackedResponse(response, release_signal)
            transferred = True
            return tracked
        finally:
            if not transferred:
                release_signal()


__all__ = ["AnthropicClient", "AnthropicMessages", "AnthropicBeta", "AnthropicResponse", "AnthropicRawResponsePromise", "AnthropicRequestOptions", "AnthropicHttpClient"]
