"""SQLite Storage implementation with serialized commits and fork snapshots."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from pi_agent_core.harness.session.commit import prepare_storage_commit
from pi_agent_core.harness.session.types import CommitResult, EntryScan, EntryStructure, ForkOptions, SessionStats, StorageBranchScan, UsageScan, Write
from pi_agent_core.harness.session.values import ListElement, ListReadOptions, StoredValue, Value, ValueList, branch_tip
from pi_chord.context import Context

from ._async import Promise, capture, rejected, resolved, spawn
from ._branches import append_entry_to_branch_index, scan_branch_entries, scan_branch_entry_structures
from ._json import Record, field
from ._rows import (EntryRowWriter, UsageLedgerRowWriter, add_usage_to_session_stats, advance_next_seq, decode_entry_row,
                    decode_usage_ledger_row, increment_message_count, read_all_entry_rows, read_entry_rows, read_next_seq,
                    read_session_stats, scan_entry_rows, scan_usage_ledger_rows)
from ._values import (append_list_value_row, delete_list_value_rows, delete_scalar_value_row, read_all_scalar_value_rows,
                      read_list_value_rows, read_scalar_value_row, scan_scalar_value_rows, set_scalar_value_row)
from .types import SqliteDatabase


@dataclass
class SqliteStorageOptions:
    session_id: str
    now: Callable[[], int] | None = None


@dataclass
class SqliteStorageSnapshot:
    entries: list[Record]
    scalar_values: list[StoredValue[object]]
    entries_complete: bool


def date_now() -> int:
    return time.time_ns() // 1_000_000


def read_snapshot_entries(db: SqliteDatabase, session_id: str, scalar_values: list[StoredValue[object]], options: ForkOptions) -> list[Record]:
    if options.scope == "tree":
        return [decode_entry_row(row) for row in read_all_entry_rows(db, session_id)]
    address = branch_tip(options.branch)
    tip = next((stored for stored in scalar_values if stored.address.namespace == address.namespace and stored.address.key == address.key), None)
    if tip is None:
        raise RuntimeError(f"Unknown source branch: {options.branch}")
    return [] if tip.value is None else scan_branch_entries(db, session_id, StorageBranchScan(start=tip.value, order="oldestFirst"))


class SqliteStorage:
    def __init__(self, db: SqliteDatabase, options: SqliteStorageOptions) -> None:
        self._db = db
        self._session_id = options.session_id
        self._now = options.now if options.now is not None else date_now
        self._entry_writer = EntryRowWriter(db, self._session_id)
        self._usage_writer = UsageLedgerRowWriter(db, self._session_id)
        self._commit_queue: Promise[None] | None = None
        self._state = "open"
        self._close_promise: Promise[None] | None = None

    def _enqueue[T](self, callback: Callable[[], T]) -> Promise[T]:
        previous = self._commit_queue
        async def apply() -> T:
            if previous is not None:
                await previous
            return callback()
        result = spawn(apply())
        async def settle() -> None:
            try:
                await result
            except BaseException:
                pass
        self._commit_queue = spawn(settle())
        return result

    def commit(self, writes: list[Write], context: Context) -> Promise[CommitResult]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        return self._enqueue(lambda: self._apply_commit(writes))

    def get_entries(self, ids: list[str], context: Context) -> Promise[dict[str, Record]]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        by_id = {cast(str, row["id"]): row for row in read_entry_rows(self._db, self._session_id, ids)}
        entries: dict[str, Record] = {}
        for entry_id in ids:
            if entry_id in by_id:
                entries[entry_id] = decode_entry_row(by_id[entry_id])
        return resolved(entries)

    def get_value[T](self, address: Value[T], context: Context) -> Promise[StoredValue[T] | None]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        return resolved(read_scalar_value_row(self._db, self._session_id, address))

    def scan_values[T](self, prefix: Value[T], context: Context) -> Promise[list[StoredValue[T]]]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        return resolved(scan_scalar_value_rows(self._db, self._session_id, prefix))

    def read_list[T](self, address: ValueList[T], options: ListReadOptions | None, context: Context) -> Promise[list[ListElement[T]]]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        return capture(lambda: read_list_value_rows(self._db, self._session_id, address, options))

    def scan_branch(self, query: StorageBranchScan, context: Context) -> Promise[list[Record]]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        async def read() -> list[Record]:
            return scan_branch_entries(self._db, self._session_id, query)
        return spawn(read())

    def scan_branch_structure(self, query: StorageBranchScan, context: Context) -> Promise[list[EntryStructure]]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        async def read() -> list[EntryStructure]:
            return scan_branch_entry_structures(self._db, self._session_id, query)
        return spawn(read())

    def scan_entries(self, query: EntryScan, context: Context) -> Promise[list[Record]]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        return resolved([decode_entry_row(row) for row in scan_entry_rows(self._db, self._session_id, query)])

    def scan_usage(self, query: UsageScan, context: Context) -> Promise[list[Record]]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        return resolved([decode_usage_ledger_row(row) for row in scan_usage_ledger_rows(self._db, self._session_id, query)])

    def get_stats(self, context: Context) -> Promise[SessionStats]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        return resolved(read_session_stats(self._db, self._session_id))

    def snapshot(self, options: ForkOptions, context: Context) -> Promise[SqliteStorageSnapshot]:
        if self._state != "open":
            return rejected(RuntimeError("SqliteStorage is closed"))
        def read() -> SqliteStorageSnapshot:
            values = read_all_scalar_value_rows(self._db, self._session_id)
            return SqliteStorageSnapshot(read_snapshot_entries(self._db, self._session_id, values, options), values, options.scope == "tree")
        return self._enqueue(read)

    def _apply_commit(self, writes: list[Write]) -> CommitResult:
        def apply() -> CommitResult:
            first_seq = read_next_seq(self._db, self._session_id)
            prepared = prepare_storage_commit(writes, first_seq, self._now())
            for write in prepared.writes:
                if write.kind == "entry":
                    self._entry_writer.insert(write.entry)
                    append_entry_to_branch_index(self._db, self._session_id, write.entry)
                    if field(write.entry, "type") == "message":
                        increment_message_count(self._db, self._session_id)
                elif write.kind == "usage":
                    self._usage_writer.insert(write.entry)
                    add_usage_to_session_stats(self._db, self._session_id, field(write.entry, "usage"))
                elif write.kind == "value":
                    if write.op == "delete":
                        delete_scalar_value_row(self._db, self._session_id, write.namespace, write.key)
                    else:
                        set_scalar_value_row(self._db, self._session_id, write.namespace, write.key, write.seq, write.value)
                elif write.kind == "list":
                    if write.op == "delete":
                        delete_list_value_rows(self._db, self._session_id, write.namespace, write.key)
                    else:
                        append_list_value_row(self._db, self._session_id, write.namespace, write.key, write.seq, write.value)
            advance_next_seq(self._db, self._session_id, first_seq + len(prepared.writes))
            return CommitResult(first_seq=prepared.result.first_seq, seqs=prepared.result.seqs,
                                timestamp=prepared.result.timestamp, stats=read_session_stats(self._db, self._session_id))
        return self._db.transaction(apply)

    def close(self, context: Context) -> Promise[None]:
        if self._close_promise is not None:
            return self._close_promise
        self._state = "closing"
        previous = self._commit_queue
        async def close() -> None:
            if previous is not None:
                await previous
            self._state = "closed"
        self._close_promise = spawn(close())
        return self._close_promise
