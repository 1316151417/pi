"""Emacs kill/yank ring, retaining the source's unbounded history."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KillRingPushOptions:
    prepend: bool
    accumulate: bool = False


class KillRing:
    def __init__(self) -> None:
        self._ring: list[str] = []

    def push(self, text: str, opts: KillRingPushOptions) -> None:
        if not text:
            return
        if opts.accumulate and self._ring:
            last = self._ring.pop()
            self._ring.append(text + last if opts.prepend else last + text)
        else:
            self._ring.append(text)

    def peek(self) -> str | None:
        return self._ring[-1] if self._ring else None

    def rotate(self) -> None:
        if len(self._ring) > 1:
            self._ring.insert(0, self._ring.pop())

    @property
    def length(self) -> int:
        return len(self._ring)
