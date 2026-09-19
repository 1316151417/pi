"""Mutation line ported from ``harness/session/mutation-line.ts``.

Serializes complete read-modify-write jobs for one Session.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

__all__ = ["MutationLine"]


class MutationLine:
    def __init__(self) -> None:
        self._tail: Optional[asyncio.Task] = None
        self._sealed_error: Optional[BaseException] = None

    async def run(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        if self._sealed_error is not None:
            raise self._sealed_error

        async def _run_after_tail(previous: Optional[asyncio.Task]) -> Any:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001 - tail errors are absorbed
                    pass
            if self._sealed_error is not None:
                raise self._sealed_error
            return await operation()

        previous = self._tail
        task = asyncio.get_running_loop().create_task(_run_after_tail(previous))
        self._tail = task
        return await task

    async def seal(self, error: BaseException) -> None:
        if self._sealed_error is None:
            self._sealed_error = error
        if self._tail is not None:
            try:
                await asyncio.shield(self._tail)
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
