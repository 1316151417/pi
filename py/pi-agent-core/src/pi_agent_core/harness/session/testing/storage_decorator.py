"""Test-only forwarding base ported from ``session/testing/storage-decorator.ts``."""

from __future__ import annotations

from typing import Dict, List, Optional

from ...context import Context
from ..types import (
    CommitResult,
    Entry,
    EntryScan,
    EntryStructure,
    SessionStats,
    Storage,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from ..values import ListElement, ListReadOptions, StoredValue, Value, ValueList

__all__ = ["StorageDecorator"]


class StorageDecorator:
    """Test-only forwarding base for decorators that alter one part of Storage behavior."""

    def __init__(self, delegate: Storage) -> None:
        self._delegate = delegate

    async def commit(self, writes: List[Write], context: Context) -> CommitResult:
        return await self._delegate.commit(writes, context)

    async def get_entries(self, ids: List[str], context: Context) -> Dict[str, Entry]:
        return await self._delegate.get_entries(ids, context)

    async def get_value(self, address: Value, context: Context) -> Optional[StoredValue]:
        return await self._delegate.get_value(address, context)

    async def scan_values(self, prefix: Value, context: Context) -> List[StoredValue]:
        return await self._delegate.scan_values(prefix, context)

    async def read_list(
        self, address: ValueList, options: Optional[ListReadOptions], context: Context
    ) -> List[ListElement]:
        return await self._delegate.read_list(address, options, context)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> List[Entry]:
        return await self._delegate.scan_branch(query, context)

    async def scan_branch_structure(
        self, query: StorageBranchScan, context: Context
    ) -> List[EntryStructure]:
        return await self._delegate.scan_branch_structure(query, context)

    async def scan_entries(self, query: EntryScan, context: Context) -> List[Entry]:
        return await self._delegate.scan_entries(query, context)

    async def scan_usage(self, query: UsageScan, context: Context) -> List[UsageRow]:
        return await self._delegate.scan_usage(query, context)

    async def get_stats(self, context: Context) -> SessionStats:
        return await self._delegate.get_stats(context)

    async def close(self, context: Context) -> None:
        await self._delegate.close(context)
