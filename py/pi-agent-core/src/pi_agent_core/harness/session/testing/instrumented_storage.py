"""Transparent commit recorder ported from ``session/testing/instrumented-storage.ts``."""

from __future__ import annotations

from typing import List

from ...context import Context
from ..types import CommitResult, Storage, Write
from .storage_decorator import StorageDecorator

__all__ = ["InstrumentedStorage"]


class InstrumentedStorage(StorageDecorator):
    """Test-only transparent Storage decorator that records commit admission."""

    def __init__(self, delegate: Storage) -> None:
        super().__init__(delegate)
        self._commit_attempts: List[List[Write]] = []

    def get_commit_attempts(self) -> List[List[Write]]:
        """Return the admitted write batches in admission order."""
        return list(self._commit_attempts)

    def clear_commit_attempts(self) -> None:
        self._commit_attempts.clear()

    async def commit(self, writes: List[Write], context: Context) -> CommitResult:
        self._commit_attempts.append(writes)
        return await self._delegate.commit(writes, context)
