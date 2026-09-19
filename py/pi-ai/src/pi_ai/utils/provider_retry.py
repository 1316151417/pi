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
import random
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..abort import AbortSignal

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


@dataclass
class ProviderHttpError(Exception):
    """One failed provider attempt, carrying the HTTP status and headers.

    ``status`` is ``None`` for transport-level failures (connection, timeout),
    mirroring the SDK errors whose status is undefined and which the policy
    treats as retryable.
    """

    status: Optional[int] = None
    headers: Optional[dict] = None
    message: str = ""

    def __post_init__(self) -> None:
        # A dataclass exception starts with no args, so str(error) would be empty;
        # the message must stay readable because callers surface it to the user.
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return self.message


def is_retryable_provider_error(error: ProviderHttpError) -> bool:
    """Whether one failed attempt should be retried.

    Mirrors the pinned OpenAI/Anthropic SDK policy: the provider's explicit
    ``x-should-retry`` header wins outright; transport failures (no status)
    retry; otherwise only 408, 409, 429, and 5xx retry.
    """
    headers = error.headers or {}
    should_retry = headers.get("x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False
    if error.status is None:
        return True
    return error.status in (408, 409, 429) or error.status >= 500


def _validate_server_retry_delay_ms(
    delay_ms: float, max_retry_delay_ms: Optional[int], provider_error_message: str
) -> float:
    max_delay_ms = DEFAULT_MAX_RETRY_DELAY_MS if max_retry_delay_ms is None else max_retry_delay_ms
    if max_delay_ms > 0 and delay_ms > max_delay_ms:
        raise ProviderHttpError(
            status=None,
            headers=None,
            message=(
                f"Server requested {delay_ms / 1000:.0f}s retry delay "
                f"(max: {max_delay_ms / 1000:.0f}s). {provider_error_message}"
            ),
        )
    return delay_ms


def retry_delay_ms(
    error: ProviderHttpError,
    retry_index: int,
    max_retry_delay_ms: Optional[int] = None,
) -> float:
    """The delay before retry ``retry_index`` (0-based), honoring server hints.

    Order mirrors the TS policy: ``retry-after-ms`` (milliseconds), then
    ``retry-after`` (seconds or an HTTP date), then exponential backoff with
    jitter — ``min(0.5 * 2^i, 8s)`` scaled down by up to 25%.
    """
    headers = error.headers or {}
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            value = float(retry_after_ms)
        except ValueError:
            value = float("nan")
        if value == value:  # not NaN
            return _validate_server_retry_delay_ms(value, max_retry_delay_ms, error.message)

    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = float("nan")
        if seconds == seconds:
            delay = seconds * 1000
        else:
            delay = _parse_http_date_ms(retry_after)
        return _validate_server_retry_delay_ms(delay, max_retry_delay_ms, error.message)

    exponential = min(0.5 * (2 ** retry_index), 8) * 1000
    return exponential * (1 - random.random() * 0.25)


_HTTP_DATE = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), "
    r"\d{2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4} "
    r"\d{2}:\d{2}:\d{2} GMT$"
)


def _parse_http_date_ms(value: str) -> float:
    """Milliseconds from now until an HTTP-date ``retry-after`` value."""
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timezone

    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds() * 1000)


async def retry_provider_request(
    request: Callable[[], Awaitable[Any]],
    max_retries: int = 0,
    max_retry_delay_ms: Optional[int] = None,
    signal: Optional[AbortSignal] = None,
) -> Any:
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
        except ProviderHttpError as error:
            if signal is not None and signal.aborted:
                raise ProviderRetryAborted() from error
            if retries_remaining <= 0 or not is_retryable_provider_error(error):
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await _abortable_sleep(
                retry_delay_ms(error, retry_index, max_retry_delay_ms), signal
            )


async def _abortable_sleep(ms: float, signal: Optional[AbortSignal]) -> None:
    """Sleep that resolves early into an abort error when the signal fires."""
    if signal is not None and signal.aborted:
        raise ProviderRetryAborted()
    sleep_task = asyncio.ensure_future(asyncio.sleep(max(0.0, ms) / 1000))
    if signal is None:
        await sleep_task
        return
    wait_task = asyncio.ensure_future(signal.wait())
    done, pending = await asyncio.wait({wait_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if wait_task in done:
        raise ProviderRetryAborted()
