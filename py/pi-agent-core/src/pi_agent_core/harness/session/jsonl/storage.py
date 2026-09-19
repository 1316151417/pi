"""JSONL storage ported from ``session/jsonl/storage.ts``.

Durable session storage backed by an injected filesystem capability: a header
line followed by one JSON transaction per commit. Reads replay transactions
into an :class:`InMemoryStorageState`, so all queries share the in-memory
semantics.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, List, Optional

from ...._chord.context import Context
from ...._pi_ai.uuid_utils import uuidv7
from ...types import FileSystem
from ..commit import CommittedWrite, insert_usage
from ..in_memory_storage_state import InMemoryStorageState
from ..types import (
    CommitResult,
    Entry,
    EntryScan,
    EntryStructure,
    SessionStats,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from ..values import ListElement, ListReadOptions, StoredValue, Value, ValueList
from .codec import parse_jsonl_session_header
from .io import (
    file_value,
    parse_jsonl_transaction,
    publish_file_atomically,
    publish_jsonl,
    serialize_jsonl_transaction,
)
from .types import JSONL_STORAGE_VERSION, JsonlStorageHeader, JsonlStorageOptions

__all__ = ["JsonlStorage"]


def _split_complete_lines(content: str) -> tuple:
    if content.endswith("\n"):
        return content[:-1].split("\n"), False
    last_newline = content.rfind("\n")
    if last_newline == -1:
        return [], True
    return content[:last_newline].split("\n"), True


class JsonlStorage:
    """JSONL storage backed by an injected filesystem capability."""

    def __init__(
        self,
        options: JsonlStorageOptions,
        header: JsonlStorageHeader,
        backing_kind: str = "v4",
    ) -> None:
        self._file_system = options.file_system
        self._path = options.path
        self._now = options.now or (lambda: int(__import__("time").time() * 1000))
        self.header = header
        self._backing_kind = backing_kind
        self._legacy_source: Any = None
        self._storage_state = InMemoryStorageState()
        self._commit_queue: Optional[asyncio.Task] = None
        self._state = "open"
        self._close_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        options: JsonlStorageOptions,
        header: JsonlStorageHeader,
        initial_writes: List[Write],
        context: Context,
    ) -> "JsonlStorage":
        storage = cls(options, header, "v4")
        prepared = storage._storage_state.prepare_commit(initial_writes, storage._now())

        async def _write_transactions(append) -> None:
            if prepared.writes:
                await append(prepared.writes)

        await publish_jsonl(options.file_system, options.path, header, context, _write_transactions)
        storage._storage_state.apply_validated(prepared.writes)
        return storage

    @classmethod
    async def open(cls, options: JsonlStorageOptions, context: Context) -> "JsonlStorage":
        content = file_value(
            await options.file_system.read_text_file(options.path, context),
            f"Failed to read JSONL storage {options.path}",
        )
        first_line = content.split("\n", 1)[0]
        parsed = parse_jsonl_session_header(first_line)
        if not parsed.ok:
            raise RuntimeError(f"Invalid JSONL storage {options.path}: invalid header") from parsed.error
        if parsed.value.format == "v3-legacy":
            raise RuntimeError(
                f"JSONL storage {options.path} uses the legacy v3 format, which is not supported yet"
            )
        return await cls._open_v4(options, parsed.value.header, context)

    @classmethod
    async def _open_v4(
        cls, options: JsonlStorageOptions, header: JsonlStorageHeader, context: Context
    ) -> "JsonlStorage":
        content = file_value(
            await options.file_system.read_text_file(options.path, context),
            f"Failed to read JSONL storage {options.path}",
        )
        lines, torn = _split_complete_lines(content)
        if header.storage_version != JSONL_STORAGE_VERSION:
            raise RuntimeError(
                f"Session {header.id} uses unsupported storage version {header.storage_version}"
            )
        storage = cls(options, header, "v4")
        for index in range(1, len(lines)):
            line = lines[index]
            try:
                storage._replay_committed(parse_jsonl_transaction(line))
            except BaseException as error:
                raise RuntimeError(
                    f"Invalid JSONL storage {options.path}: line {index + 1}"
                ) from error
        if header.next_seq is not None:
            storage._storage_state.advance_next_seq(header.next_seq)
        if torn:

            async def _rewrite(append) -> None:
                await append("\n".join(lines) + "\n")

            await publish_file_atomically(options.file_system, options.path, context, _rewrite)
        return storage

    def _replay_committed(self, writes: List[CommittedWrite]) -> None:
        self._storage_state.validate_committed(writes)
        self._storage_state.apply_validated(writes)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    async def commit(self, writes: List[Write], context: Context) -> CommitResult:
        if self._state != "open":
            raise RuntimeError("JsonlStorage is closed")
        previous = self._commit_queue

        async def _run() -> CommitResult:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001 - queue absorbs failures
                    pass
            return await self._apply_commit(writes, context)

        task = asyncio.get_running_loop().create_task(_run())
        self._commit_queue = task
        return await task

    async def _apply_commit(self, writes: List[Write], context: Context) -> CommitResult:
        prepared = self._storage_state.prepare_commit(writes, self._now())
        if prepared.writes:
            file_value(
                await self._file_system.append_file(
                    self._path, f"{serialize_jsonl_transaction(prepared.writes)}\n", context
                ),
                f"Failed to append JSONL storage {self._path}",
            )
        stats = self._storage_state.apply_validated(prepared.writes)
        return CommitResult(
            first_seq=prepared.result.first_seq,
            seqs=prepared.result.seqs,
            timestamp=prepared.result.timestamp,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _assert_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("JsonlStorage is closed")

    async def get_entries(self, ids: List[str], _context: Context) -> dict:
        self._assert_open()
        return self._storage_state.get_entries(ids)

    async def get_value(self, address: Value, _context: Context) -> Optional[StoredValue]:
        self._assert_open()
        return self._storage_state.get_value(address)

    async def scan_values(self, prefix: Value, _context: Context) -> List[StoredValue]:
        self._assert_open()
        return self._storage_state.scan_values(prefix)

    async def read_list(
        self, address: ValueList, options: Optional[ListReadOptions], _context: Context
    ) -> List[ListElement]:
        self._assert_open()
        return self._storage_state.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan, _context: Context) -> List[Entry]:
        self._assert_open()
        return self._storage_state.scan_branch(query)

    async def scan_branch_structure(
        self, query: StorageBranchScan, _context: Context
    ) -> List[EntryStructure]:
        self._assert_open()
        return self._storage_state.scan_branch_structure(query)

    async def scan_entries(self, query: EntryScan, _context: Context) -> List[Entry]:
        self._assert_open()
        return self._storage_state.scan_entries(query)

    async def scan_usage(self, query: UsageScan, _context: Context) -> List[UsageRow]:
        self._assert_open()
        return self._storage_state.scan_usage(query)

    async def get_stats(self, _context: Context) -> SessionStats:
        self._assert_open()
        return self._storage_state.get_stats()

    async def capture_fork_next_seq(self, _context: Context) -> int:
        self._assert_open()
        if self._commit_queue is not None:
            try:
                await asyncio.shield(self._commit_queue)
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        return self._storage_state.get_next_seq()

    @property
    def storage_state(self) -> InMemoryStorageState:
        """The replayed storage state backing this file."""
        return self._storage_state

    def is_legacy_v3(self) -> bool:
        return self._backing_kind == "v3"

    async def close(self, _context: Context) -> None:
        if self._close_task is not None:
            await self._close_task
            return
        self._state = "closing"

        async def _close() -> None:
            if self._commit_queue is not None:
                try:
                    await asyncio.shield(self._commit_queue)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            self._state = "closed"

        self._close_task = asyncio.get_running_loop().create_task(_close())
        await self._close_task
