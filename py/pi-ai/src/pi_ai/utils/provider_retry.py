"""Provider transport retry ported from pi-ai ``src/utils/provider-retry.ts``.

Reproduces the retry behavior the OpenAI and Anthropic SDKs use, with two
improvements kept from the TypeScript original: the backoff sleep is
interruptible by the request AbortSignal, and a server-requested delay above
``max_retry_delay_ms`` fails immediately instead of blocking.

Callers must disable the underlying HTTP client's own retries (``maxRetries: 0``
in TS terms) and wrap the whole request with :func:`retry_provider_request`,
because the SDKs' built-in timers ignore the AbortSignal.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol, TypeGuard

from ..abort import AbortSignal
from ._javascript import javascript_string

__all__ = [
    "DEFAULT_MAX_RETRY_DELAY_MS",
    "ProviderHttpError",
    "ProviderRetryAborted",
    "is_retryable_provider_error",
    "retry_delay_ms",
    "retry_provider_request",
]

#: Default ceiling for server-requested delays; ``0`` disables the limit.
DEFAULT_MAX_RETRY_DELAY_MS = 60_000


class ProviderRetryAborted(Exception):
    """The request AbortSignal fired while waiting to retry."""

    def __init__(self) -> None:
        super().__init__("Request aborted")
        self.name = "AbortError"


class _HeaderLookup(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


type _Headers = Mapping[str, str] | _HeaderLookup


class _ProviderError(Protocol):
    status: int | float | None
    headers: _Headers | None


@dataclass
class ProviderHttpError(Exception):
    """One failed provider attempt, carrying the HTTP status and headers.

    ``status`` is ``None`` for transport-level failures (connection, timeout),
    mirroring the SDK errors whose status is undefined and which the policy
    treats as retryable.
    """

    status: int | float | None = None
    headers: _Headers | None = None
    message: str = ""

    def __post_init__(self) -> None:
        # A dataclass exception starts with no args, so str(error) would be empty;
        # the message must stay readable because callers surface it to the user.
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return self.message


def _is_provider_error(error: object) -> TypeGuard[_ProviderError]:
    if not isinstance(error, BaseException) or not hasattr(error, "status") or not hasattr(error, "headers"):
        return False
    status, headers = getattr(error, "status"), getattr(error, "headers")
    if status is not None and (not isinstance(status, (int, float)) or isinstance(status, bool)):
        return False
    if headers is None:
        return True
    if isinstance(headers, Mapping):
        return all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items())
    return callable(getattr(headers, "get", None))


def _header(headers: _Headers | None, name: str) -> str | None:
    if headers is None:
        return None
    if isinstance(headers, Mapping):
        values = [value.strip(" \t") for key, value in headers.items() if key.lower() == name]
        return ", ".join(values) if values else None
    return headers.get(name)


def is_retryable_provider_error(error: _ProviderError) -> bool:
    """Whether one failed attempt should be retried.

    Mirrors the pinned OpenAI/Anthropic SDK policy: the provider's explicit
    ``x-should-retry`` header wins outright; transport failures (no status)
    retry; otherwise only 408, 409, 429, and 5xx retry.
    """
    should_retry = _header(error.headers, "x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False
    if error.status is None:
        return True
    return error.status in (408, 409, 429) or error.status >= 500


def _validate_server_retry_delay_ms(
    delay_ms: float, max_retry_delay_ms: int | None, provider_error_message: str
) -> float:
    max_delay_ms = DEFAULT_MAX_RETRY_DELAY_MS if max_retry_delay_ms is None else max_retry_delay_ms
    if max_delay_ms > 0 and delay_ms > max_delay_ms:
        raise RuntimeError(
            f"Server requested {javascript_string(math.ceil(delay_ms / 1000) if math.isfinite(delay_ms) else delay_ms)}s retry delay "
            f"(max: {javascript_string(math.ceil(max_delay_ms / 1000))}s). {provider_error_message}"
        )
    return delay_ms


_PARSE_FLOAT = re.compile(r"^[+-]?(?:Infinity|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)")
_JS_WHITESPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def _parse_float(value: str) -> float:
    match = _PARSE_FLOAT.match(value.lstrip(_JS_WHITESPACE))
    return float(match.group()) if match is not None else math.nan


def retry_delay_ms(
    error: _ProviderError,
    retry_index: int,
    max_retry_delay_ms: int | None = None,
) -> float:
    """The delay before retry ``retry_index`` (0-based), honoring server hints.

    Order mirrors the TS policy: ``retry-after-ms`` (milliseconds), then
    ``retry-after`` (seconds or an HTTP date), then exponential backoff with
    jitter — ``min(0.5 * 2^i, 8s)`` scaled down by up to 25%.
    """
    message = getattr(error, "message", str(error))
    retry_after_ms = _header(error.headers, "retry-after-ms")
    if retry_after_ms:
        value = _parse_float(retry_after_ms)
        if not math.isnan(value):
            return _validate_server_retry_delay_ms(value, max_retry_delay_ms, message)

    retry_after = _header(error.headers, "retry-after")
    if retry_after:
        seconds = _parse_float(retry_after)
        delay = _parse_http_date_ms(retry_after) if math.isnan(seconds) else seconds * 1000
        return _validate_server_retry_delay_ms(delay, max_retry_delay_ms, message)

    exponential = 8000 if retry_index >= 4 else min(0.5 * (2.0 ** retry_index), 8) * 1000
    return exponential * (1 - random.random() * 0.25)


def _parse_http_date_ms(value: str) -> float:
    """Milliseconds from now until an HTTP-date ``retry-after`` value."""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return math.nan
    if parsed.tzinfo is None:
        parsed = parsed.astimezone() if "T" in value else parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() * 1000 - math.floor(datetime.now(timezone.utc).timestamp() * 1000)


async def retry_provider_request[T](
    request: Callable[[], Awaitable[T]],
    max_retries: int = 0,
    max_retry_delay_ms: int | None = None,
    signal: AbortSignal | None = None,
) -> T:
    """Run one request with transport-level retry.

    Each retry re-invokes ``request`` from scratch (a fresh SDK request in TS
    terms, so retry-count headers stay zero). Failures that are not retryable,
    or that exhaust ``max_retries``, raise; an abort during the backoff sleep
    raises :class:`ProviderRetryAborted`.
    """
    retries_remaining = max_retries
    while True:
        try:
            return await request()
        except BaseException as error:
            if signal is not None and signal.aborted:
                raise ProviderRetryAborted() from error
            if retries_remaining <= 0 or not _is_provider_error(error) or not is_retryable_provider_error(error):
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await _abortable_sleep(
                retry_delay_ms(error, retry_index, max_retry_delay_ms), signal
            )


async def _abortable_sleep(ms: float, signal: AbortSignal | None) -> None:
    """Sleep that resolves early into an abort error when the signal fires."""
    if signal is not None and signal.aborted:
        raise ProviderRetryAborted()
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[None] = loop.create_future()

    def on_abort() -> None:
        timeout.cancel()
        if not waiter.done():
            waiter.set_exception(ProviderRetryAborted())

    def on_timeout() -> None:
        if not waiter.done():
            waiter.set_result(None)

    delay = ms if math.isfinite(ms) and 1 <= ms <= 2_147_483_647 else 1
    timeout = loop.call_later(math.trunc(delay) / 1000, on_timeout)
    if signal is not None:
        signal.add_event_listener("abort", on_abort, once=True)
    try:
        await waiter
    finally:
        timeout.cancel()
        if signal is not None:
            signal.remove_event_listener("abort", on_abort)
