"""Retry timing helpers ported from ``harness/runtime/drive/retry.ts``."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from ...._pi_ai.abort import AbortSignal
from ...utils.retry import RetryPolicy, retry_delay_ms

__all__ = ["retry_not_before", "wait_until", "MAX_TIMER_MS"]

#: Node timers cap out here; longer waits are re-armed in slices.
MAX_TIMER_MS = 2_147_483_647

#: Python ints are unbounded; this mirrors JS ``Number.MAX_SAFE_INTEGER``.
MAX_SAFE_INTEGER = 2**53 - 1


def retry_not_before(policy: RetryPolicy, attempt: int, now: Optional[int] = None) -> int:
    """Absolute timestamp for the next retry attempt."""
    base = int(time.time() * 1000) if now is None else now
    total = base + retry_delay_ms(policy, attempt)
    return total if abs(total) <= MAX_SAFE_INTEGER else MAX_SAFE_INTEGER


async def wait_until(not_before: int, signal: AbortSignal) -> None:
    """Sleep until ``not_before`` (epoch ms), rejecting when ``signal`` aborts."""
    while True:
        if signal.aborted:
            raise signal.reason if isinstance(signal.reason, BaseException) else asyncio.CancelledError()
        remaining = not_before - int(time.time() * 1000)
        if remaining <= 0:
            return
        delay = min(remaining, MAX_TIMER_MS) / 1000
        sleep_task = asyncio.ensure_future(asyncio.sleep(delay))
        abort_task = asyncio.ensure_future(signal.wait())
        done, pending = await asyncio.wait(
            {sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if abort_task in done:
            raise signal.reason if isinstance(signal.reason, BaseException) else asyncio.CancelledError()
