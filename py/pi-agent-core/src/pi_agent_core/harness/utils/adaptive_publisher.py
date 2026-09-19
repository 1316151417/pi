"""Adaptive publisher ported from ``harness/utils/adaptive-publisher.ts``.

Publishes the latest state without queuing intermediate mutations. The first
dirty state after idle is immediate; each publication buys a delay proportional
to its encoded size, with a minimum interval that also bounds event count. A
single trailing timer guarantees eventual publication.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Generic, Optional, TypeVar

__all__ = ["AdaptivePublisher", "AdaptivePublisherOptions", "DEFAULT_MIN_INTERVAL_MS", "DEFAULT_TARGET_BYTES_PER_SECOND"]

TValue = TypeVar("TValue")
TUpdate = TypeVar("TUpdate")

DEFAULT_MIN_INTERVAL_MS = 100
DEFAULT_TARGET_BYTES_PER_SECOND = 100 * 1024


class AdaptivePublisherOptions(Generic[TValue, TUpdate]):
    """Callbacks and tuning knobs for :class:`AdaptivePublisher`."""

    def __init__(
        self,
        snapshot: Callable[[], TValue],
        update: Callable[[Optional[TValue], TValue], Optional[TUpdate]],
        measure: Callable[[TUpdate], int],
        publish: Callable[[TUpdate], None],
        on_error: Callable[[BaseException], None],
        min_interval_ms: int = DEFAULT_MIN_INTERVAL_MS,
        target_bytes_per_second: int = DEFAULT_TARGET_BYTES_PER_SECOND,
    ) -> None:
        self.snapshot = snapshot
        self.update = update
        self.measure = measure
        self.publish = publish
        self.on_error = on_error
        self.min_interval_ms = min_interval_ms
        self.target_bytes_per_second = target_bytes_per_second


class AdaptivePublisher(Generic[TValue, TUpdate]):
    def __init__(self, options: AdaptivePublisherOptions[TValue, TUpdate]) -> None:
        self._options = options
        self._min_interval_ms = options.min_interval_ms
        self._target_bytes_per_second = options.target_bytes_per_second
        self._published: Optional[TValue] = None
        self._dirty = False
        self._next_emit_at = 0.0
        self._timer: Optional[asyncio.TimerHandle] = None
        self._disposed = False

    def mark_dirty(self) -> None:
        if self._disposed:
            return
        self._dirty = True
        wait = self._next_emit_at - _now_ms()
        if wait <= 0:
            self.flush()
            return
        self._arm_timer(wait)

    def flush(self, force: bool = False) -> None:
        if self._disposed or not self._dirty:
            return
        now = _now_ms()
        if not force and now < self._next_emit_at:
            self._arm_timer(self._next_emit_at - now)
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None

        current = self._options.snapshot()
        update = self._options.update(self._published, current)
        if update is None:
            self._published = current
            self._dirty = False
            return
        encoded_bytes = self._options.measure(update)
        self._published = current
        self._dirty = False
        self._next_emit_at = now + max(
            self._min_interval_ms, (encoded_bytes * 1000) / self._target_bytes_per_second
        )
        # Commit before delivery: a consumer may apply the update and then throw
        # or reenter the producer; retaining the old baseline would duplicate.
        self._options.publish(update)

    def dispose(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._disposed = True

    def _arm_timer(self, wait_ms: float) -> None:
        if self._timer is not None:
            return
        loop = asyncio.get_event_loop()

        def _fire() -> None:
            self._timer = None
            try:
                self.flush()
            except BaseException as error:  # noqa: BLE001 - publisher must not throw
                self._options.on_error(error)

        self._timer = loop.call_later(max(0.0, wait_ms / 1000), _fire)


def _now_ms() -> float:
    return time.monotonic() * 1000
