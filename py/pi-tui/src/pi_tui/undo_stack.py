"""Detached undo snapshots from ``packages/tui/src/undo-stack.ts``."""

from __future__ import annotations

from copy import deepcopy


class UndoStack[S]:
    def __init__(self) -> None:
        self._stack: list[S] = []

    def push(self, state: S) -> None:
        self._stack.append(deepcopy(state))

    def pop(self) -> S | None:
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        self._stack.clear()

    @property
    def length(self) -> int:
        return len(self._stack)
