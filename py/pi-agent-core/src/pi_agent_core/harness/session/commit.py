"""Commit preparation ported from ``harness/session/commit.ts``."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, make_dataclass
from typing import Any, List

from .types import CommitResult, UsageRow, Write

__all__ = [
    "CommittedWrite",
    "PreparedCommit",
    "insert_entry",
    "insert_usage",
    "commit_write",
    "materialize_committed_entry",
    "prepare_storage_commit",
    "validate_committed_writes",
]


@dataclass
class CommittedWrite:
    kind: str
    seq: int
    op: Optional[str] = None
    id: Optional[str] = None
    parent_id: Optional[str] = None
    timestamp: int = 0
    entry: Any = None
    usage: Any = None
    namespace: Optional[str] = None
    key: Optional[str] = None
    value: Any = None


@dataclass
class PreparedCommit:
    writes: List[CommittedWrite]
    result: CommitResult  # stats filled by the storage


def insert_entry(entry: Any) -> Any:
    from .types import EntryWrite

    return EntryWrite(entry=entry)


def insert_usage(row: UsageRow) -> Any:
    from .types import UsageWrite

    return UsageWrite(row=row)


def commit_write(write: Write, seq: int, timestamp: int) -> CommittedWrite:
    kind = getattr(write, "kind", None)
    if kind == "entry":
        entry = write.entry
        committed = copy.copy(entry)
        committed.seq = seq
        committed.timestamp = timestamp
        return CommittedWrite(kind="entry", seq=seq, timestamp=timestamp, entry=committed,
                              id=entry.id, parent_id=entry.parent_id)
    if kind == "usage":
        row = write.row
        committed_row = copy.copy(row)
        committed_row.seq = seq
        return CommittedWrite(kind="usage", seq=seq, entry=committed_row,
                              id=row.id, usage=row.usage)
    if kind == "value":
        return CommittedWrite(
            kind="value",
            op=write.op,
            seq=seq,
            namespace=write.namespace,
            key=write.key,
            value=write.value if write.op == "set" else None,
        )
    if kind == "list":
        return CommittedWrite(
            kind="list",
            op=write.op,
            seq=seq,
            namespace=write.namespace,
            key=write.key,
            value=write.value if write.op == "append" else None,
        )
    raise ValueError(f"Unknown write kind: {kind!r}")


def materialize_committed_entry(entry: Any, seq: int, timestamp: int) -> Any:
    committed = copy.copy(entry)
    committed.seq = seq
    committed.timestamp = timestamp
    return committed


def prepare_storage_commit(writes: List[Write], first_seq: int, timestamp: int) -> PreparedCommit:
    committed_writes = [commit_write(write, first_seq + index, timestamp) for index, write in enumerate(writes)]
    result = CommitResult(
        first_seq=first_seq,
        seqs=[write.seq for write in committed_writes],
        timestamp=timestamp,
    )
    return PreparedCommit(writes=committed_writes, result=result)


class _ValidationState:
    def __init__(self, entries: dict, usage: dict) -> None:
        self._entries = entries
        self._usage = usage

    def has_entry_or_usage_id(self, entry_id: str) -> bool:
        return entry_id in self._entries or entry_id in self._usage

    def has_entry_id(self, entry_id: str) -> bool:
        return entry_id in self._entries


def validate_committed_writes(
    writes: List[CommittedWrite],
    first_seq: int,
    state: _ValidationState,
) -> None:
    previous_seq = first_seq - 1
    transaction_ids: set = set()
    transaction_entry_ids: set = set()
    for write in writes:
        if write.seq <= previous_seq:
            raise RuntimeError(f"Non-monotonic storage sequence: {write.seq}")
        previous_seq = write.seq
        if write.kind not in ("entry", "usage"):
            continue
        if state.has_entry_or_usage_id(write.id) or write.id in transaction_ids:
            raise RuntimeError(f"Duplicate entry or usage id: {write.id}")
        if (
            write.kind == "entry"
            and write.parent_id is not None
            and not state.has_entry_id(write.parent_id)
            and write.parent_id not in transaction_entry_ids
        ):
            raise RuntimeError(f"Missing parent entry: {write.parent_id}")
        transaction_ids.add(write.id)
        if write.kind == "entry":
            transaction_entry_ids.add(write.id)
