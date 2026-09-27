"""OAuth device polling from ``auth/oauth/device-code.ts``."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from ...abort import AbortSignal

CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. This is often caused by "
    "clock drift in WSL or VM environments. Please sync or restart the VM clock and try again."
)
MINIMUM_INTERVAL_MS = 1000
DEFAULT_POLL_INTERVAL_SECONDS = 5
SLOW_DOWN_INTERVAL_INCREMENT_MS = 5000


@dataclass
class OAuthDeviceCodePending:
    status: Literal["pending"] = field(default="pending", init=False)


@dataclass
class OAuthDeviceCodeSlowDown:
    interval_seconds: int | float | None = None
    status: Literal["slow_down"] = field(default="slow_down", init=False)


@dataclass
class OAuthDeviceCodeFailed:
    message: str
    status: Literal["failed"] = field(default="failed", init=False)


@dataclass
class OAuthDeviceCodeComplete[T]:
    value: T
    status: Literal["complete"] = field(default="complete", init=False)


type OAuthDeviceCodePollResult[T] = OAuthDeviceCodePending | OAuthDeviceCodeSlowDown | OAuthDeviceCodeFailed | OAuthDeviceCodeComplete[T]


@dataclass
class OAuthDeviceCodePollOptions[T]:
    poll: Callable[[], Awaitable[OAuthDeviceCodePollResult[T]]]
    signal: AbortSignal
    interval_seconds: int | float | None = None
    expires_in_seconds: int | float | None = None
    wait_before_first_poll: bool = False


async def abortable_sleep(ms: float, signal: AbortSignal, cancel_message: str) -> None:
    if signal.aborted:
        raise RuntimeError(cancel_message)
    loop = asyncio.get_running_loop()
    result: asyncio.Future[None] = loop.create_future()

    def on_abort() -> None:
        timeout.cancel()
        if not result.done():
            result.set_exception(RuntimeError(cancel_message))

    def on_timeout() -> None:
        signal.remove_event_listener("abort", on_abort)
        if not result.done():
            result.set_result(None)

    # Match Node's setTimeout conversion, including NaN/Infinity/overflow.
    delay = 1 if not math.isfinite(ms) or ms < 1 or ms > 2147483647 else math.trunc(ms)
    timeout = loop.call_later(delay / 1000, on_timeout)
    signal.add_event_listener("abort", on_abort, once=True)
    try:
        await result
    finally:
        timeout.cancel()
        signal.remove_event_listener("abort", on_abort)


def _minimum_interval(value: int | float) -> int | float:
    if math.isnan(value):
        return value
    return max(MINIMUM_INTERVAL_MS, math.floor(value) if math.isfinite(value) else value)


async def poll_oauth_device_code_flow[T](options: OAuthDeviceCodePollOptions[T]) -> T:
    expires = options.expires_in_seconds
    deadline = math.floor(time.time() * 1000) + expires * 1000 if type(expires) in (int, float) else math.inf
    interval_ms = _minimum_interval((DEFAULT_POLL_INTERVAL_SECONDS if options.interval_seconds is None else options.interval_seconds) * 1000)
    slow_down_responses = 0
    if options.wait_before_first_poll:
        remaining_ms = deadline - math.floor(time.time() * 1000)
        if remaining_ms > 0:
            await abortable_sleep(min(interval_ms, remaining_ms), options.signal, CANCEL_MESSAGE)
    while math.floor(time.time() * 1000) < deadline:
        if options.signal.aborted:
            raise RuntimeError(CANCEL_MESSAGE)
        result = await options.poll()
        if isinstance(result, OAuthDeviceCodeComplete):
            return result.value
        if isinstance(result, OAuthDeviceCodeFailed):
            raise RuntimeError(result.message)
        if isinstance(result, OAuthDeviceCodeSlowDown):
            slow_down_responses += 1
            interval = result.interval_seconds
            interval_ms = _minimum_interval(
                interval * 1000 if type(interval) in (int, float) and math.isfinite(interval) and interval > 0
                else interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS,
            )
        remaining_ms = deadline - math.floor(time.time() * 1000)
        if remaining_ms <= 0:
            break
        await abortable_sleep(min(interval_ms, remaining_ms), options.signal, CANCEL_MESSAGE)
    raise RuntimeError(SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE)


__all__ = [
    "OAuthDeviceCodePending", "OAuthDeviceCodeSlowDown", "OAuthDeviceCodeFailed", "OAuthDeviceCodeComplete",
    "OAuthDeviceCodePollResult", "OAuthDeviceCodePollOptions", "abortable_sleep", "poll_oauth_device_code_flow",
]
