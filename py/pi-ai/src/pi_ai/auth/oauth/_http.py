"""Fetch-compatible HTTP, URL and form primitives shared by OAuth providers."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Mapping

import httpx
from ada_url import URL, URLSearchParams

from ...abort import AbortController, AbortSignal
from ...utils._javascript import javascript_object_keys


def _usv_string(value: str) -> str:
    scalar_text = value.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="surrogatepass")
    return re.sub(r"[\ud800-\udfff]", "\ufffd", scalar_text)


def parse_url(value: str, base: str | None = None) -> URL:
    return URL(_usv_string(value), _usv_string(base) if base is not None else None)


def normalize_url(value: str) -> str:
    return parse_url(value).href


def trusted_http_url(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        url = parse_url(value)
    except ValueError:
        return None
    return url.href if url.protocol in ("https:", "http:") else None


def form_url_encode(fields: Mapping[str, str]) -> str:
    parameters = URLSearchParams("")
    for key in javascript_object_keys(fields):
        parameters.append(_usv_string(key), _usv_string(fields[key]))
    return str(parameters)


class OAuthTimeoutError(TimeoutError):
    name = "TimeoutError"


def timeout_signal(milliseconds: int | float) -> AbortSignal:
    if type(milliseconds) not in (int, float):
        raise TypeError('The "delay" argument must be of type number')
    if not math.isfinite(milliseconds) or milliseconds < 0 or milliseconds > 9007199254740991 or milliseconds != math.trunc(milliseconds):
        raise ValueError('The value of "delay" is out of range')
    controller = AbortController()
    delay = 1 if milliseconds < 1 or milliseconds > 2147483647 else milliseconds
    asyncio.get_running_loop().call_later(
        delay / 1000, controller.abort, OAuthTimeoutError("The operation was aborted due to timeout"),
    )
    return controller.signal


async def _await_with_signal[T](operation: Awaitable[T], signal: AbortSignal | None) -> T:
    task = asyncio.ensure_future(operation)
    if signal is None:
        return await task
    loop = asyncio.get_running_loop()

    def on_abort() -> None:
        loop.call_soon_threadsafe(task.cancel)

    signal.add_event_listener("abort", on_abort, once=True)
    if signal.aborted:
        task.cancel()
    try:
        try:
            return await task
        except asyncio.CancelledError:
            signal.throw_if_aborted()
            raise
    finally:
        signal.remove_event_listener("abort", on_abort)


class OAuthHttpResponse:
    def __init__(self, response: httpx.Response, client: httpx.AsyncClient, signal: AbortSignal | None) -> None:
        self.status = response.status_code
        self.status_text = response.reason_phrase
        self.ok = 200 <= self.status <= 299
        self.headers = response.headers
        self._response = response
        self._client = client
        self._signal = signal
        self._used = False
        self._closed = False
        self._loop = asyncio.get_running_loop()
        self._close_task: asyncio.Task[None] | None = None
        self._body = None if self.status in (204, 205, 304) or response.request.method == "HEAD" else self.iter_bytes()
        if signal is not None:
            signal.add_event_listener("abort", self._on_abort, once=True)
            if signal.aborted:
                self._on_abort()

    def _on_abort(self) -> None:
        def close() -> None:
            if self._close_task is None:
                self._close_task = self._loop.create_task(self._close())

        self._loop.call_soon_threadsafe(close)

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._signal is not None:
            self._signal.remove_event_listener("abort", self._on_abort)
        try:
            await self._response.aclose()
        finally:
            await self._client.aclose()

    async def _read(self) -> bytes:
        if self._signal is not None:
            self._signal.throw_if_aborted()
        if self._used:
            raise TypeError("Body is unusable: Body has already been read")
        self._used = True
        try:
            try:
                content = await _await_with_signal(self._response.aread(), self._signal)
            except Exception:
                if self._signal is not None:
                    self._signal.throw_if_aborted()
                raise
            if self._signal is not None:
                self._signal.throw_if_aborted()
            return content
        finally:
            await self._close()

    async def text(self) -> str:
        return (await self._read()).decode("utf-8-sig", errors="replace")

    async def json(self) -> object:
        text = (await self._read()).decode("utf-8-sig", errors="replace")

        def invalid_constant(value: str) -> object:
            raise ValueError(f"Unexpected token {value} in JSON")

        # JS JSON.parse uses IEEE-754 numbers, including for integer literals.
        return json.loads(text, parse_int=float, parse_float=float, parse_constant=invalid_constant)

    @property
    def body(self) -> AsyncIterable[bytes] | None:
        return self._body

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        """Consume decoded HTTP bytes once, retaining abort until end-of-body."""
        if self._signal is not None:
            self._signal.throw_if_aborted()
        if self._used:
            raise TypeError("Body is unusable: Body has already been read")
        self._used = True
        iterator = self._response.aiter_bytes()
        try:
            while True:
                try:
                    chunk = await _await_with_signal(anext(iterator), self._signal)
                except StopAsyncIteration:
                    return
                except Exception:
                    if self._signal is not None:
                        self._signal.throw_if_aborted()
                    raise
                if self._signal is not None:
                    self._signal.throw_if_aborted()
                yield chunk
        finally:
            try:
                await iterator.aclose()
            finally:
                await self._close()

    async def cancel_body(self) -> None:
        self._used = True
        await self._close()


async def fetch(
    url: str, *, method: str = "GET", headers: Mapping[str, str] | None = None,
    body: str | bytes | None = None, signal: AbortSignal | None = None,
) -> OAuthHttpResponse:
    if signal is not None:
        signal.throw_if_aborted()
    parsed = parse_url(url)
    if parsed.username or parsed.password:
        raise TypeError("Request cannot be constructed from a URL that includes credentials")
    if method.upper() in ("GET", "HEAD") and body is not None:
        raise TypeError("Request with GET/HEAD method cannot have body")
    if parsed.protocol not in ("https:", "http:"):
        raise TypeError("fetch failed: unknown scheme")
    request_headers = httpx.Headers(headers)
    request_headers.setdefault("Accept", "*/*")
    request_headers.setdefault("Accept-Language", "*")
    request_headers.setdefault("Sec-Fetch-Mode", "cors")
    request_headers.setdefault("User-Agent", "node")
    if isinstance(body, str):
        request_headers.setdefault("Content-Type", "text/plain;charset=UTF-8")
        content: bytes | None = _usv_string(body).encode("utf-8")
    else:
        content = body
    # Follow redirects explicitly: HTTPX retains cookies and keeps Authorization
    # on HTTP-to-HTTPS redirects, while Node's fetch uses Fetch's origin rules.
    client = httpx.AsyncClient(timeout=None, follow_redirects=False)
    try:
        redirects = 0
        while True:
            client.cookies.clear()
            request = client.build_request(method, parsed.href, headers=request_headers, content=content)
            response = await _await_with_signal(client.send(request, stream=True), signal)
            location = response.headers.get("Location")
            if response.status_code not in (301, 302, 303, 307, 308) or location is None:
                return OAuthHttpResponse(response, client, signal)
            try:
                redirected = parse_url(location, parsed.href)
                if "#" not in location:
                    redirected.hash = parsed.hash
                if redirected.protocol not in ("https:", "http:"):
                    raise TypeError("fetch failed: redirect URL must use HTTP or HTTPS")
                if redirects == 20:
                    raise TypeError("fetch failed: redirect count exceeded")
                if redirected.username or redirected.password:
                    raise TypeError("fetch failed: redirect URL includes credentials")
                if (response.status_code in (301, 302) and method.upper() == "POST") or (
                    response.status_code == 303 and method.upper() not in ("GET", "HEAD")
                ):
                    method = "GET"
                    content = None
                    for name in ("Content-Encoding", "Content-Language", "Content-Location", "Content-Type", "Content-Length"):
                        request_headers.pop(name, None)
                if parsed.origin != redirected.origin:
                    for name in ("Authorization", "Proxy-Authorization", "Cookie", "Host"):
                        request_headers.pop(name, None)
                parsed = redirected
                redirects += 1
            finally:
                await response.aclose()
    except BaseException:
        await client.aclose()
        raise


__all__ = [
    "OAuthHttpResponse", "OAuthTimeoutError", "fetch", "form_url_encode", "parse_url", "normalize_url",
    "trusted_http_url", "timeout_signal",
]
