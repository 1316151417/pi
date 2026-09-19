"""Bounded tracking ported from ``harness/pico3/bounded.ts``.

A bounded byte collector. It retains a prefix or suffix constrained by both a
byte budget and a newline budget, and accounts for every discarded byte and
newline.
"""

from __future__ import annotations

__all__ = ["Bounded"]

_NEWLINE = 0x0A


def _count_newlines(data: bytes) -> int:
    return data.count(_NEWLINE)


def _tail_start(data: bytes, max_bytes: int, max_lines: int) -> int:
    start = max(0, len(data) - max_bytes)
    excess_lines = _count_newlines(data[start:]) - max_lines
    while start < len(data) and excess_lines > 0:
        if data[start] == _NEWLINE:
            excess_lines -= 1
        start += 1
    return start


class Bounded:
    """Retains a head or tail slice of a byte stream within byte and newline budgets."""

    def __init__(self, max_bytes: int, max_lines: int, retain: str) -> None:
        self._bytes = b""
        self._max_bytes = max(0, max_bytes)
        self._max_lines = max(0, max_lines)
        self._retain = retain
        self.dropped_bytes = 0
        self.dropped_lines = 0
        self.total = 0

    def push(self, chunk: bytes) -> None:
        """Append one chunk, discarding whatever no longer fits the budget."""
        self.total += len(chunk)
        if len(chunk) == 0:
            return
        if self._max_bytes == 0 or self._max_lines == 0:
            self._drop(chunk)
            return
        if self._retain == "head":
            self._push_head(chunk)
            return
        self._push_tail(chunk)

    def _push_head(self, chunk: bytes) -> None:
        remaining_bytes = self._max_bytes - len(self._bytes)
        remaining_lines = self._max_lines - _count_newlines(self._bytes)
        if remaining_bytes <= 0 or remaining_lines <= 0:
            self._drop(chunk)
            return
        take = min(len(chunk), remaining_bytes)
        lines = 0
        for i in range(take):
            if chunk[i] != _NEWLINE:
                continue
            lines += 1
            if lines == remaining_lines:
                take = i + 1
                break
        self._bytes = b"".join((self._bytes, chunk[:take]))
        self._drop(chunk[take:])

    def _push_tail(self, chunk: bytes) -> None:
        incoming_start = _tail_start(chunk, self._max_bytes, self._max_lines)
        self._drop(chunk[:incoming_start])
        combined = self._bytes + chunk[incoming_start:]
        start = _tail_start(combined, self._max_bytes, self._max_lines)
        self._drop(combined[:start])
        self._bytes = combined[start:]

    def _drop(self, data: bytes) -> None:
        self.dropped_bytes += len(data)
        self.dropped_lines += _count_newlines(data)

    @property
    def dropped(self) -> int:
        return self.dropped_bytes

    def text(self) -> str:
        return self._bytes.decode("utf-8", errors="replace")
