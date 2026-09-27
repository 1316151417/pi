"""Assistant-call retry ported from pi-ai ``src/utils/retry.ts``."""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from pi_ai.abort import AbortSignal
from pi_ai.types import AssistantMessage

__all__ = [
    "RetryPolicy",
    "RetryCallbacks",
    "DEFAULT_MAX_AGENT_RETRY_DELAY_MS",
    "retry_delay_ms",
    "retry_assistant_call",
    "is_retryable_assistant_error",
]

#: Subscription/account limits: not transient, so retrying wastes the user's time.
NON_RETRYABLE_PROVIDER_LIMIT_PATTERNS = [
    "GoUsageLimitError",
    "FreeUsageLimitError",
    "Monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
]

#: Transient provider load, HTTP status, and transport failures.
RETRYABLE_PROVIDER_ERROR_PATTERNS = [
    "overloaded",
    "currently experiencing high demand",
    "rate.?limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "520",
    "524",
    "service.?unavailable",
    "server.?error",
    "internal.?error",
    "provider.?returned.?error",
    "exceeded request buffer limit while retrying upstream",
    "network.?error",
    "connection.?error",
    "connection.?refused",
    "connection.?lost",
    "other side closed",
    "fetch failed",
    "getaddrinfo",
    "ENOTFOUND",
    "EAI_AGAIN",
    "upstream.?connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed? out",
    "timeout",
    "terminated",
    "websocket.?closed",
    "websocket.?error",
    "ended without",
    "stream ended before message_stop",
    "stream ended before a terminal response event",
    "http2 request did not get a response",
    "retry delay",
    "you can retry your request",
    "try your request again",
    "please retry your request",
    "ResourceExhausted",
]

_NON_RETRYABLE_RE = re.compile("|".join(NON_RETRYABLE_PROVIDER_LIMIT_PATTERNS), re.IGNORECASE)
_RETRYABLE_RE = re.compile("|".join(RETRYABLE_PROVIDER_ERROR_PATTERNS), re.IGNORECASE)

DEFAULT_MAX_AGENT_RETRY_DELAY_MS = 60_000


@dataclass
class RetryPolicy:
    """Bounded retry policy. ``max_retries`` excludes the initial call."""

    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 1000
    max_agent_delay_ms: int | None = DEFAULT_MAX_AGENT_RETRY_DELAY_MS


class _RetryFinished(Protocol):
    def __call__(self, success: bool, attempt: int, final_error: str | None = None) -> None | Awaitable[None]: ...


@dataclass
class RetryCallbacks:
    on_retry_scheduled: Callable[[int, int, int, str], None | Awaitable[None]] | None = None
    on_retry_attempt_start: Callable[[], None | Awaitable[None]] | None = None
    on_retry_finished: _RetryFinished | None = None


def retry_delay_ms(policy: RetryPolicy, attempt: int) -> int:
    maximum_safe_integer = 2**53 - 1
    try:
        delay = policy.base_delay_ms * (2.0 ** max(0, attempt - 1))
    except OverflowError:
        delay = math.inf
    safe_delay = (
        int(delay) if math.isfinite(delay) and delay.is_integer() and abs(delay) <= maximum_safe_integer
        else maximum_safe_integer
    )
    cap = DEFAULT_MAX_AGENT_RETRY_DELAY_MS if policy.max_agent_delay_ms is None else policy.max_agent_delay_ms
    return min(safe_delay, cap)


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    if message.stop_reason != "error" or not message.error_message:
        return False
    if _NON_RETRYABLE_RE.search(message.error_message):
        return False
    return bool(_RETRYABLE_RE.search(message.error_message))


class _RetrySleepAbort(Exception):
    def __init__(self) -> None:
        super().__init__("Aborted")


async def _sleep(ms: int, signal: AbortSignal | None) -> None:
    if signal is not None and signal.aborted:
        raise _RetrySleepAbort()
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[None] = loop.create_future()

    def on_abort() -> None:
        timeout.cancel()
        if not waiter.done():
            waiter.set_exception(_RetrySleepAbort())

    def on_timeout() -> None:
        if not waiter.done():
            waiter.set_result(None)

    delay = ms if 1 <= ms <= 2_147_483_647 else 1
    timeout = loop.call_later(math.trunc(delay) / 1000, on_timeout)
    if signal is not None:
        signal.add_event_listener("abort", on_abort, once=True)
    try:
        await waiter
    finally:
        timeout.cancel()
        if signal is not None:
            signal.remove_event_listener("abort", on_abort)


async def _maybe_await(value: None | Awaitable[None]) -> None:
    if inspect.isawaitable(value):
        await value


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    signal: AbortSignal | None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """Run one assistant-producing call with bounded retry on transient errors."""
    max_attempts = policy.max_retries if (policy is not None and policy.enabled) else 0

    attempt = 0
    last_retry: tuple[int, str] | None = None
    while True:
        response = await produce()

        if response.stop_reason == "aborted":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(False, last_retry[0]))
            return response

        if response.stop_reason != "error":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(True, last_retry[0]))
            return response

        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(
                    callbacks.on_retry_finished(False, last_retry[0], response.error_message)
                )
            return response

        attempt += 1
        last_retry = (attempt, response.error_message or "Unknown error")
        assert policy is not None
        delay = retry_delay_ms(policy, attempt)
        if callbacks is not None and callbacks.on_retry_scheduled:
            await _maybe_await(
                callbacks.on_retry_scheduled(attempt, max_attempts, delay, last_retry[1])
            )

        try:
            await _sleep(delay, signal)
        except BaseException as error:
            if callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(False, attempt, last_retry[1]))
            if isinstance(error, _RetrySleepAbort):
                aborted = copy.copy(response)
                aborted.stop_reason = "aborted"
                aborted.error_message = None
                return aborted
            raise

        if callbacks is not None and callbacks.on_retry_attempt_start:
            await _maybe_await(callbacks.on_retry_attempt_start())
