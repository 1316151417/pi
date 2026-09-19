"""Memory storage and repo ported from ``harness/session/memory.ts``."""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any, Dict, List, Optional

from ..._chord.context import Context
from pi_ai.uuid_utils import uuidv7
from .in_memory_storage_state import InMemoryStorageState
from .session import StorageBackedSession
from .types import (
    CommitResult,
    Entry,
    EntryScan,
    EntryStructure,
    ForkOptions,
    Session,
    SessionCreateOptions,
    SessionMetadata,
    SessionStats,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from .values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
)

__all__ = ["MemoryStorage", "MemorySessionRepo"]


def _default_now() -> int:
    return int(_time.time() * 1000)


class MemoryStorage:
    """In-memory Storage with a serialized commit queue."""

    def __init__(self, now: Optional[Any] = None) -> None:
        self._now = now or _default_now
        self._state_snapshot = InMemoryStorageState()
        self._commit_lock = asyncio.Lock()
        self._state = "open"
        self._close_task: Optional[asyncio.Task] = None

    async def commit(self, writes: List[Write], _context: Context) -> CommitResult:
        if self._state != "open":
            raise RuntimeError("MemoryStorage is closed")
        async with self._commit_lock:
            if self._state != "open":
                raise RuntimeError("MemoryStorage is closed")
            prepared = self._state_snapshot.prepare_commit(writes, self._now())
            stats = self._state_snapshot.apply_validated(prepared.writes)
            result = CommitResult(
                first_seq=prepared.result.first_seq,
                seqs=prepared.result.seqs,
                timestamp=prepared.result.timestamp,
                stats=stats,
            )
            return result

    async def get_entries(self, ids: List[str], _context: Context) -> Dict[str, Entry]:
        self._assert_open()
        return self._state_snapshot.get_entries(ids)

    async def get_value(self, address: Value, _context: Context) -> Optional[StoredValue]:
        self._assert_open()
        return self._state_snapshot.get_value(address)

    async def scan_values(self, prefix: Value, _context: Context) -> List[StoredValue]:
        self._assert_open()
        return self._state_snapshot.scan_values(prefix)

    async def read_list(
        self, address: ValueList, options: Optional[ListReadOptions], _context: Context
    ) -> List[ListElement]:
        self._assert_open()
        return self._state_snapshot.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan, _context: Context) -> List[Entry]:
        self._assert_open()
        return self._state_snapshot.scan_branch(query)

    async def scan_branch_structure(self, query: StorageBranchScan, _context: Context) -> List[EntryStructure]:
        self._assert_open()
        return self._state_snapshot.scan_branch_structure(query)

    async def scan_entries(self, query: EntryScan, _context: Context) -> List[Entry]:
        self._assert_open()
        return self._state_snapshot.scan_entries(query)

    async def scan_usage(self, query: UsageScan, _context: Context) -> List[UsageRow]:
        self._assert_open()
        return self._state_snapshot.scan_usage(query)

    async def get_stats(self, _context: Context) -> SessionStats:
        self._assert_open()
        return self._state_snapshot.get_stats()

    async def fork(self, options: ForkOptions) -> "MemoryStorage":
        self._assert_open()
        async with self._commit_lock:
            destination = MemoryStorage(now=self._now)
            destination._state_snapshot = self._state_snapshot.create_fork(options)
            return destination

    async def close(self, _context: Context) -> None:
        if self._close_task is not None:
            await self._close_task
            return
        self._state = "closed"

        async def _close() -> None:
            async with self._commit_lock:
                self._state = "closed"

        self._close_task = asyncio.get_running_loop().create_task(_close())
        await self._close_task

    def _assert_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("MemoryStorage is closed")


MEMORY_STORAGE_VERSION = 1


class _MemorySessionRecord:
    def __init__(self, metadata: SessionMetadata, storage: MemoryStorage, session: StorageBackedSession) -> None:
        self.metadata = metadata
        self.storage = storage
        self.session = session
        self.open = True


class MemorySessionRepo:
    """Session repository keeping everything in process memory."""

    def __init__(self, now: Optional[Any] = None) -> None:
        self._now = now or _default_now
        self._sessions: Dict[str, _MemorySessionRecord] = {}
        self._pending_ids: set = set()
        self._closed = False
        self._close_task: Optional[asyncio.Task] = None

    async def create(self, options: Optional[SessionCreateOptions], context: Context) -> Session:
        options = options or SessionCreateOptions()
        self._assert_open()
        created_at = self._now()
        session_id = options.id or uuidv7(created_at)
        self._reserve_id(session_id)
        metadata = SessionMetadata(
            id=session_id,
            created_at=created_at,
            storage_version=MEMORY_STORAGE_VERSION,
            parent_session_id=options.parent_session_id,
        )
        storage = MemoryStorage(now=self._now)
        record_holder: dict = {}

        def _mark_closed() -> None:
            record = record_holder.get("record")
            if record is not None:
                record.open = False

        session = StorageBackedSession(metadata, storage, {"on_close": _mark_closed})
        try:
            record = _MemorySessionRecord(metadata, storage, session)
            record_holder["record"] = record
            self._sessions[session_id] = record
            return self._open_record(record)
        except BaseException:
            await session.close(context)
            raise
        finally:
            self._pending_ids.discard(session_id)

    async def open(self, metadata: SessionMetadata, _context: Context) -> Session:
        self._assert_open()
        record = self._sessions.get(metadata.id)
        if record is None:
            raise RuntimeError(f"Unknown session: {metadata.id}")
        if record.open:
            raise RuntimeError(f"Session is already open: {metadata.id}")
        record.open = True
        return self._open_record(record)
    async def list(self, _options: None, _context: Context) -> List[SessionMetadata]:
        self._assert_open()
        return [record.metadata for record in self._sessions.values()]

    async def delete(self, metadata: SessionMetadata, context: Context) -> None:
        self._assert_open()
        record = self._sessions.get(metadata.id)
        if record is None:
            raise RuntimeError(f"Unknown session: {metadata.id}")
        if record.open:
            raise RuntimeError(f"Session is open: {metadata.id}")
        await record.session.close(context)
        del self._sessions[metadata.id]

    async def fork(self, source: SessionMetadata, options: ForkOptions, _context: Context) -> Session:
        self._assert_open()
        source_record = self._sessions.get(source.id)
        if source_record is None:
            raise RuntimeError(f"Unknown session: {source.id}")
        created_at = self._now()
        session_id = options.id or uuidv7(created_at)
        self._reserve_id(session_id)

        try:
            storage = await source_record.storage.fork(options)
            metadata = SessionMetadata(
                id=session_id,
                created_at=created_at,
                storage_version=MEMORY_STORAGE_VERSION,
                parent_session_id=source_record.metadata.id,
            )
            record_holder: dict = {}

            def _mark_closed() -> None:
                record = record_holder.get("record")
                if record is not None:
                    record.open = False

            session = StorageBackedSession(metadata, storage, {"on_close": _mark_closed})
            record = _MemorySessionRecord(metadata, storage, session)
            record_holder["record"] = record
            self._sessions[session_id] = record
            return self._open_record(record)
        finally:
            self._pending_ids.discard(session_id)

    async def close(self, context: Context) -> None:
        if self._close_task is not None:
            await self._close_task
            return
        self._closed = True

        async def _close() -> None:
            # The repo owns each record's storage; facades never close it.
            await asyncio.gather(
                *(record.session.close(context) for record in self._sessions.values()),
                return_exceptions=True,
            )
            await asyncio.gather(
                *(record.storage.close(context) for record in self._sessions.values()),
                return_exceptions=True,
            )

        self._close_task = asyncio.get_running_loop().create_task(_close())
        await self._close_task

    def _open_record(self, record: _MemorySessionRecord) -> Session:
        """Return a fresh session wrapper sharing the record's storage.

        Mirrors the TS facade-per-open semantics: each open returns an
        independently closeable session whose close marks the record closed.
        """
        record_holder: dict = {"record": record}

        def _mark_closed() -> None:
            held = record_holder.get("record")
            if held is not None:
                held.open = False

        session = StorageBackedSession(
            record.metadata,
            record.storage,
            {"on_close": _mark_closed, "owns_storage": False},
        )
        record.session = session
        return session

    def _reserve_id(self, session_id: str) -> None:
        if session_id in self._sessions or session_id in self._pending_ids:
            raise RuntimeError(f"Session already exists: {session_id}")
        self._pending_ids.add(session_id)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemorySessionRepo is closed")
