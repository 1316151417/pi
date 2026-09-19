"""Deterministic commit gating ported from ``session/testing/gating-storage.ts``."""

from __future__ import annotations

import asyncio
from typing import List

from ...context import Context
from ..types import CommitResult, Storage, Write
from .storage_decorator import StorageDecorator

__all__ = ["CommitDiscarded", "GatingStorage"]


class CommitDiscarded(RuntimeError):
    """Thrown for every commit rejected after simulated storage loss."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "CommitDiscarded"


class _ParkedCommit:
    """One admitted commit that waits for an explicit release or drop."""

    __slots__ = ("_released", "_landing")

    def __init__(self, released: asyncio.Future, landing: asyncio.Future) -> None:
        self._released = released
        self._landing = landing

    @property
    def landing(self) -> asyncio.Future:
        return self._landing

    def release(self) -> None:
        if not self._released.done():
            self._released.set_result(None)

    def drop(self, error: Exception) -> None:
        if not self._released.done():
            self._released.set_exception(error)


class _PendingWaiter:
    """One ``wait_pending`` caller waiting for the queue to reach ``count``."""

    __slots__ = ("count", "future")

    def __init__(self, count: int, future: asyncio.Future) -> None:
        self.count = count
        self.future = future


def _is_positive_safe_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


class GatingStorage(StorageDecorator):
    """Test-only storage decorator that deterministically parks admitted commits."""

    def __init__(self, delegate: Storage) -> None:
        super().__init__(delegate)
        self._armed = False
        self._discarded = False
        self._queue: List[_ParkedCommit] = []
        self._waiters: List[_PendingWaiter] = []

    def arm(self) -> None:
        """Fixture setup bypasses gating until explicitly armed."""
        self._armed = True

    def pending(self) -> int:
        return len(self._queue)

    async def wait_pending(self, count: int = 1) -> None:
        """Wait until at least ``count`` commits are parked.

        TypeScript rejects with ``RangeError``; Python raises ``ValueError``.
        """
        if not _is_positive_safe_integer(count):
            raise ValueError("Pending commit count must be a positive safe integer")
        if self._discarded:
            raise CommitDiscarded("storage discarded")
        if len(self._queue) >= count:
            return
        waiter = _PendingWaiter(count, asyncio.get_running_loop().create_future())
        self._waiters.append(waiter)
        await waiter.future

    async def commit(self, writes: List[Write], context: Context) -> CommitResult:
        if self._discarded:
            raise CommitDiscarded("commit rejected: storage discarded")
        if not self._armed:
            return await self._delegate.commit(writes, context)

        loop = asyncio.get_running_loop()
        released: asyncio.Future = loop.create_future()
        landing: asyncio.Future = loop.create_future()
        # Mirror ``void landing.catch(() => {})``: an unobserved loss must not surface
        # as an unhandled future exception.
        landing.add_done_callback(lambda future: future.cancelled() or future.exception())

        self._queue.append(_ParkedCommit(released, landing))
        self._notify_waiters()

        try:
            await released
            if self._discarded:
                raise CommitDiscarded("commit rejected: storage discarded")
            result = await self._delegate.commit(writes, context)
            if not landing.done():
                landing.set_result(None)
            return result
        except BaseException as error:
            # ``error`` is the same value the TypeScript implementation rejects with; a
            # cancellation is surfaced as a failure so ``landing`` stays awaitable.
            normalized = (
                error if isinstance(error, Exception) else RuntimeError(f"commit interrupted: {error}")
            )
            if not landing.done():
                landing.set_exception(normalized)
            raise

    async def next(self, count: int = 1) -> None:
        """Release ``count`` commits in FIFO order and wait until each write lands."""
        if not _is_positive_safe_integer(count):
            raise ValueError("Released commit count must be a positive safe integer")
        for _ in range(count):
            await self.wait_pending()
            parked = self._queue.pop(0) if self._queue else None
            if parked is None:
                raise RuntimeError("No parked commit")
            parked.release()
            await parked.landing

    def discard(self) -> None:
        """Drop parked commits and permanently reject every later commit."""
        if self._discarded:
            return
        self._discarded = True
        error = CommitDiscarded("commit discarded")
        for parked in list(self._queue):
            parked.drop(error)
        self._queue.clear()
        for waiter in list(self._waiters):
            if not waiter.future.done():
                waiter.future.set_exception(error)
        self._waiters.clear()

    def _notify_waiters(self) -> None:
        for index in range(len(self._waiters) - 1, -1, -1):
            waiter = self._waiters[index]
            if len(self._queue) < waiter.count:
                continue
            del self._waiters[index]
            if not waiter.future.done():
                waiter.future.set_result(None)
