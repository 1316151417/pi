"""Abortable millisecond sleep ported from ``src/utils/sleep.ts``."""

from __future__ import annotations

import asyncio

from ..abort import AbortError, AbortSignal

__all__ = ["sleep"]


async def sleep(ms: float, signal: AbortSignal) -> None:
    signal.throw_if_aborted()
    loop = asyncio.get_running_loop()
    result: asyncio.Future[None] = loop.create_future()

    def on_abort() -> None:
        timeout.cancel()
        if not result.done():
            reason = signal.reason
            result.set_exception(reason if isinstance(reason, BaseException) else AbortError(str(reason)))

    def on_timeout() -> None:
        signal.remove_event_listener("abort", on_abort)
        if not result.done():
            result.set_result(None)

    timeout = loop.call_later(max(0, ms) / 1000, on_timeout)
    signal.add_event_listener("abort", on_abort, once=True)
    try:
        await result
    finally:
        timeout.cancel()
        signal.remove_event_listener("abort", on_abort)
