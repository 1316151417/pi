"""Native HTTP and SSE transport matching the OpenAI JavaScript 6.40.0 SDK.

The request/error contract comes from that pinned SDK; actual I/O uses the
shared native Fetch adapter. Custom fetch functions accept its keyword arguments
and return the response protocol below. Platform headers identify Python.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import platform
import re
import sys
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast
from urllib.parse import parse_qsl, quote

import httpx

from ..abort import AbortController, AbortSignal
from ..auth.oauth._http import fetch, parse_url
from ..types import UNDEFINED, FetchFunction
from ..utils._javascript import javascript_json_parse, javascript_json_stringify, javascript_object_keys
from ._openai_errors import (
    APIConnectionError, APIConnectionTimeoutError, APIError, APIUserAbortError,
    AuthenticationError, BadRequestError, ConflictError, InternalServerError,
    NotFoundError, OpenAIError, PermissionDeniedError, RateLimitError, UnprocessableEntityError,
    is_abort_error, truthy as _truthy,
)
from ._openai_stream import ByteChunk, OpenAIStream, StreamingResponse, _close_iterator


class FetchResponse(StreamingResponse, Protocol):
    ok: bool
    headers: Mapping[str, str]

    async def text(self) -> str: ...
    async def json(self) -> object: ...
    async def cancel_body(self) -> None: ...


@dataclass
class OpenAIResponse:
    data: object
    response: FetchResponse


class _TrackedResponse:
    def __init__(self, response: FetchResponse, on_close: Callable[[], None]) -> None:
        self.status = response.status
        self.ok = response.ok
        self.headers = response.headers
        self._response = response
        self._on_close = on_close
        self._body_ready = False
        self._body: AsyncIterator[ByteChunk] | None = None

    @property
    def body(self) -> AsyncIterable[ByteChunk] | None:
        if not self._body_ready:
            source = self._response.body
            self._body = self._read_body(source) if source is not None else None
            self._body_ready = True
        return self._body

    async def _read_body(self, source: AsyncIterable[ByteChunk]) -> AsyncIterator[ByteChunk]:
        iterator = aiter(source)
        exhausted = False
        try:
            async for chunk in iterator:
                yield chunk
            exhausted = True
        finally:
            try:
                if not exhausted:
                    await _close_iterator(iterator, preserve_error=sys.exception())
            finally:
                self._on_close()

    async def text(self) -> str:
        try:
            return await self._response.text()
        finally:
            self._on_close()

    async def json(self) -> object:
        try:
            return await self._response.json()
        finally:
            self._on_close()

    async def cancel_body(self) -> None:
        try:
            await self._response.cancel_body()
        finally:
            self._on_close()


class OpenAIJsonObject(dict[str, object]):
    """Keep the SDK's non-enumerable request id out of the JSON object keys."""

    _request_id: str | None


def _normalize_headers(headers: dict[str, str]) -> None:
    for name, value in headers.items():
        if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
            raise TypeError(f'Headers: "{name}" is an invalid header name.')
        value = value.strip(" \t\r\n")
        if "\x00" in value or "\r" in value or "\n" in value:
            raise TypeError(f'Headers: "{value}" is an invalid header value.')
        value.encode("latin-1")
        headers[name] = value


class OpenAIHttpClient:
    def __init__(
        self, *, api_key: str, base_url: str | None = None,
        default_headers: Mapping[str, str | None] | None = None, custom_fetch: FetchFunction | None = None,
        default_query: Mapping[str, str] | None = None,
        auth_header: Literal["authorization", "api-key"] = "authorization",
        user_agent_name: str = "OpenAI",
    ) -> None:
        if not api_key and not os.environ.get("OPENAI_ADMIN_KEY"):
            raise OpenAIError(
                "Missing credentials. Please pass an `apiKey`, `workloadIdentity`, `adminAPIKey`, or set the `OPENAI_API_KEY` or `OPENAI_ADMIN_KEY` environment variable.",
            )
        self.api_key = api_key
        self.base_url = (base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")) or "https://api.openai.com/v1"
        self.organization = os.environ.get("OPENAI_ORG_ID")
        self.project = os.environ.get("OPENAI_PROJECT_ID")
        self.custom_fetch = custom_fetch
        self.default_query = default_query
        self.auth_header = auth_header
        self.user_agent_name = user_agent_name
        log_level = os.environ.get("OPENAI_LOG")
        if log_level and log_level not in ("off", "error", "warn", "info", "debug"):
            logging.getLogger(__name__).warning(
                "process.env['OPENAI_LOG'] was set to %s, expected one of %s",
                javascript_json_stringify(log_level), '["off","error","warn","info","debug"]',
            )
            log_level = None
        self.log_level = log_level or "warn"
        self.default_headers: dict[str, str | None] = {}
        custom_headers = os.environ.get("OPENAI_CUSTOM_HEADERS")
        if custom_headers:
            for line in custom_headers.split("\n"):
                if ":" in line:
                    name, value = line.split(":", 1)
                    self.default_headers[name.strip().lower()] = value.strip()
        if custom_headers:
            normalized = {name: value for name, value in self.default_headers.items() if value is not None}
            _normalize_headers(normalized)
            self.default_headers.update(normalized)
        if default_headers is not None:
            if custom_headers:
                normalized = {name.lower(): value for name, value in default_headers.items() if value is not None}
                _normalize_headers(normalized)
            self.default_headers.update((name.lower(), value) for name, value in default_headers.items())

    async def post(
        self, path: str, body: object, *, signal: AbortSignal | None = None,
        timeout_ms: int | float | None = None, stream: bool = False,
        synthesize_event_data: bool = False,
    ) -> OpenAIResponse:
        if timeout_ms is not None:
            if type(timeout_ms) not in (int, float) or not math.isfinite(timeout_ms) or timeout_ms != math.trunc(timeout_ms):
                raise OpenAIError("timeout must be an integer")
            if timeout_ms < 0:
                raise OpenAIError("timeout must be a positive integer")
        timeout = 600_000 if timeout_ms is None else timeout_ms
        url = parse_url(path if re.match(r"^[a-z][a-z0-9+.-]*:", path, re.I) else self.base_url + (
            path[1:] if self.base_url.endswith("/") and path.startswith("/") else path
        ))
        # The SDK round-trips existing query parameters through an object, so
        # duplicate names take their last value and spaces become %20.
        query = dict(parse_qsl(url.search.removeprefix("?"), keep_blank_values=True))
        if self.default_query is not None:
            query.update(self.default_query)
        if query:
            url.search = "&".join(f"{quote(key, safe='~')}={quote(query[key], safe='~')}" for key in javascript_object_keys(query))
        raw_body: str | bytes | None = None
        has_json_body = False
        if _truthy(body):
            if isinstance(body, (bytes, bytearray, memoryview)):
                raw_body = bytes(body)
            else:
                raw_body = javascript_json_stringify(body)
                has_json_body = True

        operating_system = {
            "darwin": "MacOS", "win32": "Windows", "freebsd": "FreeBSD", "openbsd": "OpenBSD", "linux": "Linux",
        }.get(sys.platform, f"Other:{sys.platform}" if sys.platform else "Unknown")
        machine = platform.machine()
        architecture = {"x86_64": "x64", "AMD64": "x64", "aarch64": "arm64", "arm64": "arm64", "arm": "arm", "x32": "x32"}.get(
            machine, f"other:{machine}" if machine else "unknown",
        )
        headers = {
            "accept": "application/json", "user-agent": f"{self.user_agent_name}/Python 6.40.0",
            "x-stainless-retry-count": "0", "x-stainless-lang": "python",
            "x-stainless-package-version": "6.40.0", "x-stainless-os": operating_system,
            "x-stainless-arch": architecture, "x-stainless-runtime": "python",
            "x-stainless-runtime-version": platform.python_version(),
            self.auth_header: self.api_key if self.auth_header == "api-key" else f"Bearer {self.api_key}",
        }
        # SDK adds the timeout header only when explicitly supplied by callers.
        if timeout_ms:
            headers["x-stainless-timeout"] = str(math.trunc(timeout_ms / 1000))
        if self.organization is not None:
            headers["openai-organization"] = self.organization
        if self.project is not None:
            headers["openai-project"] = self.project
        null_headers: set[str] = set()
        for name, value in self.default_headers.items():
            if value is None:
                _normalize_headers({name: ""})
                headers.pop(name, None)
                null_headers.add(name)
            else:
                headers[name] = value
        if has_json_body:
            headers["content-type"] = "application/json"
        _normalize_headers(headers)
        if not (headers.get("authorization") or headers.get("api-key") or null_headers.intersection(("authorization", "api-key"))):
            raise OpenAIError(
                'Could not resolve authentication method. Expected either apiKey or adminAPIKey to be set. Or for one of the "Authorization" or "api-key" headers to be explicitly omitted',
            )

        if signal is not None and signal.aborted:
            raise APIUserAbortError()
        controller = AbortController()

        def abort() -> None:
            controller.abort()

        if signal is not None:
            signal.add_event_listener("abort", abort, once=True)

        def release_signal() -> None:
            if signal is not None:
                signal.remove_event_listener("abort", abort)
            controller.signal.remove_event_listener("abort", release_signal)

        controller.signal.add_event_listener("abort", release_signal, once=True)
        streaming = False
        delay = 1 if timeout < 1 or timeout > 2_147_483_647 else timeout
        timer = asyncio.get_running_loop().call_later(delay / 1000, abort)
        try:
            try:
                response = cast(FetchResponse, await (self.custom_fetch or fetch)(
                    url.href, method="POST", headers=httpx.Headers(headers, encoding="latin-1"),
                    body=raw_body, signal=controller.signal,
                ))
            except (Exception, asyncio.CancelledError) as error:
                if signal is not None and signal.aborted:
                    raise APIUserAbortError() from error
                is_timeout = is_abort_error(error)
                is_timeout = is_timeout or re.search(r"timed? ?out", str(error) + " " + str(getattr(error, "cause", "")), re.I) is not None
                if is_timeout:
                    raise APIConnectionTimeoutError() from error
                raise APIConnectionError(cause=error) from error
            finally:
                # Like fetchWithTimeout, the timer covers only response headers;
                # the caller's abort signal still applies while consuming body.
                timer.cancel()
            if not response.ok:
                try:
                    error_text = await response.text()
                except (Exception, asyncio.CancelledError) as error:
                    error_text = str(error)
                try:
                    error_json = javascript_json_parse(error_text)
                except (ValueError, TypeError):
                    error_json = None
                detail = error_json.get("error") if isinstance(error_json, Mapping) else None
                status = response.status
                if not status:
                    raise APIConnectionError(message=error_text if not _truthy(error_json) and error_text else "Connection error.")
                error_class = {
                    400: BadRequestError, 401: AuthenticationError, 403: PermissionDeniedError,
                    404: NotFoundError, 409: ConflictError, 422: UnprocessableEntityError, 429: RateLimitError,
                }.get(status, InternalServerError if status >= 500 else APIError)
                raise error_class(status, detail, None if _truthy(error_json) else error_text, response.headers)
            if stream:
                response = _TrackedResponse(response, release_signal)
                data_stream = OpenAIStream.from_sse_response(
                    response, controller, synthesize_event_data=synthesize_event_data, on_close=release_signal,
                    log_errors=self.log_level != "off",
                )
                streaming = True
                return OpenAIResponse(data_stream, response)
            if response.status == 204:
                await response.cancel_body()
                data: object = None
            else:
                media_type = response.headers.get("content-type", "").split(";")[0].strip()
                if "application/json" in media_type or media_type.endswith("+json"):
                    if response.headers.get("content-length") == "0":
                        await response.cancel_body()
                        data = UNDEFINED
                    else:
                        data = await response.json()
                        if isinstance(data, dict):
                            wrapped = OpenAIJsonObject(cast(dict[str, object], data))
                            wrapped._request_id = response.headers.get("x-request-id")
                            data = wrapped
                else:
                    data = await response.text()
            return OpenAIResponse(data, response)
        finally:
            if not streaming:
                release_signal()


__all__ = [
    "FetchResponse", "OpenAIResponse", "OpenAIHttpClient", "OpenAIStream", "OpenAIError", "APIError", "APIUserAbortError",
    "APIConnectionError", "APIConnectionTimeoutError", "BadRequestError", "AuthenticationError",
    "PermissionDeniedError", "NotFoundError", "ConflictError", "UnprocessableEntityError",
    "RateLimitError", "InternalServerError",
]
