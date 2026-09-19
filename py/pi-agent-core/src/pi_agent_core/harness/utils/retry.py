"""Assistant-call retry ported from pi-ai ``src/utils/retry.ts``."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..._pi_ai.abort import AbortSignal
from ..._pi_ai.types import AssistantMessage

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
    "billing",
    "quota exceeded",
    "out of credits",
    "exceeded your current quota",
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
    "529",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway time-?out",
    "connection (?:error|reset|refused|closed)",
    "econnreset",
    "econnrefused",
    "etimedout",
    "socket hang up",
    "network error",
    "fetch failed",
    "request timed out",
    "timed? ?out",
    "temporarily unavailable",
    "try again",
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
    max_agent_delay_ms: int = DEFAULT_MAX_AGENT_RETRY_DELAY_MS


@dataclass
class RetryCallbacks:
    on_retry_scheduled: Optional[Callable[..., Any]] = None
    on_retry_attempt_start: Optional[Callable[..., Any]] = None
    on_retry_finished: Optional[Callable[..., Any]] = None


def retry_delay_ms(policy: RetryPolicy, attempt: int) -> int:
    delay = policy.base_delay_ms * (2 ** max(0, attempt - 1))
    return min(delay, policy.max_agent_delay_ms or DEFAULT_MAX_AGENT_RETRY_DELAY_MS)


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    if message.stop_reason != "error" or not message.error_message:
        return False
    if _NON_RETRYABLE_RE.search(message.error_message):
        return False
    return bool(_RETRYABLE_RE.search(message.error_message))


class _RetrySleepAbort(Exception):
    pass


async def _sleep(ms: int, signal: Optional[AbortSignal]) -> None:
    if signal is not None and signal.aborted:
        raise _RetrySleepAbort()
    try:
        if signal is None:
            await asyncio.sleep(ms / 1000)
            return
        wait_task = asyncio.ensure_future(signal.wait())
        sleep_task = asyncio.ensure_future(asyncio.sleep(ms / 1000))
        done, pending = await asyncio.wait(
            {wait_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if wait_task in done:
            raise _RetrySleepAbort()
    except asyncio.CancelledError:
        raise


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: Optional[RetryPolicy],
    signal: Optional[AbortSignal],
    callbacks: Optional[RetryCallbacks] = None,
) -> AssistantMessage:
    """Run one assistant-producing call with bounded retry on transient errors."""
    max_attempts = policy.max_retries if (policy is not None and policy.enabled) else 0

    attempt = 0
    last_retry: Optional[dict] = None
    while True:
        response = await produce()

        if response.stop_reason == "aborted":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(False, last_retry["attempt"]))
            return response

        if response.stop_reason != "error":
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(True, last_retry["attempt"]))
            return response

        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None and callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(
                    callbacks.on_retry_finished(False, last_retry["attempt"], response.error_message)
                )
            return response

        attempt += 1
        last_retry = {"attempt": attempt, "error_message": response.error_message or "Unknown error"}
        delay = retry_delay_ms(policy, attempt)
        if callbacks is not None and callbacks.on_retry_scheduled:
            await _maybe_await(
                callbacks.on_retry_scheduled(attempt, max_attempts, delay, last_retry["error_message"])
            )

        try:
            await _sleep(delay, signal)
        except _RetrySleepAbort:
            if callbacks is not None and callbacks.on_retry_finished:
                await _maybe_await(callbacks.on_retry_finished(False, attempt, last_retry["error_message"]))
            aborted = AssistantMessage(**{**vars(response), "stop_reason": "aborted"})
            aborted.error_message = None
            return aborted

        if callbacks is not None and callbacks.on_retry_attempt_start:
            await _maybe_await(callbacks.on_retry_attempt_start())
