"""Transient messages composited by the alternate-screen renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from .._timers import TimerHandle, set_timeout
from ..utils import truncate_to_width


@dataclass
class _FlashEntry:
    id: int
    message: str
    timer: TimerHandle


class AltScreenFlashContainer:
    def __init__(self, request_render: Callable[[], None]) -> None:
        self._entries: list[_FlashEntry] = []
        self._next_id = 0
        self._request_render = request_render
        self._lock = RLock()

    def flash(self, message: str, duration_ms: float = 1000) -> None:
        def expire() -> None:
            with self._lock:
                for index, entry in enumerate(self._entries):
                    if entry.id == id:
                        del self._entries[index]
                        self._request_render()
                        return

        # The synchronous timer backend must not expire before the entry exists.
        with self._lock:
            id = self._next_id
            self._next_id += 1
            timer = set_timeout(expire, max(0, duration_ms), unref=True)
            self._entries.append(_FlashEntry(id, message, timer))
            self._request_render()

    def dispose(self) -> None:
        with self._lock:
            for entry in self._entries:
                entry.timer.cancel()
            self._entries.clear()

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        with self._lock:
            lines: list[str] = []
            for entry in self._entries:
                message = truncate_to_width(f" {entry.message} ", width, "")
                lines.append(f"\x1b[7m{message}\x1b[27m")
            return lines
