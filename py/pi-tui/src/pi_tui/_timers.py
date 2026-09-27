"""Node-style millisecond timers for the synchronous component API."""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable

type TimerHandle = asyncio.TimerHandle | threading.Timer


def set_timeout(callback: Callable[[], None], milliseconds: float, *, unref: bool = False) -> TimerHandle:
    delay = 1 if not math.isfinite(milliseconds) or milliseconds < 1 or milliseconds > 2147483647 else math.trunc(milliseconds)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        timer = threading.Timer(delay / 1000, callback)
        timer.daemon = unref
        timer.start()
        return timer
    return loop.call_later(delay / 1000, callback)


class IntervalHandle:
    def __init__(self, callback: Callable[[], None], milliseconds: float) -> None:
        self._callback = callback
        self._milliseconds = milliseconds
        self._cancelled = False
        self._lock = threading.RLock()
        with self._lock:
            self._handle = set_timeout(self._tick, milliseconds)

    def _tick(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._handle = set_timeout(self._tick, self._milliseconds)
            self._callback()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            self._handle.cancel()
