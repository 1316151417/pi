"""In-memory storage state ported from ``harness/session/in-memory-storage-state.ts``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .commit import CommittedWrite, PreparedCommit, prepare_storage_commit, validate_committed_writes
from .fork_policy import ForkCurrentStatePlan, project_fork_current_state_write, select_branch_fork
from .types import (
    Entry,
    EntryScan,
    EntryStructure,
    ForkOptions,
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
    branch_tip,
    lane_config,
    lane_state,
    list as list_address,
    resolve_list_read_options,
    value,
)
from ..utils.usage import add_usage, empty_usage

__all__ = ["InMemoryStorageState"]


def _physical_key(namespace: str, key: str) -> str:
    return f"{namespace}\x00{key}"


class _StoredListSnapshot:
    def __init__(self, address: ValueList, elements: List[ListElement]) -> None:
        self.address = address
        self.elements = elements


class InMemoryStorageState:
    """Complete materialized session state for MemoryStorage and JsonlStorage."""

    def __init__(self) -> None:
        self._entries: Dict[str, Entry] = {}
        self._entries_by_seq: List[Entry] = []
        self._scalar_values: Dict[str, StoredValue] = {}
        self._list_values: Dict[str, _StoredListSnapshot] = {}
        self._usage: Dict[str, UsageRow] = {}
        self._stats = SessionStats(message_count=0, usage=empty_usage())
        self._next_seq = 1

    # ------------------------------------------------------------------
    # Commit pipeline
    # ------------------------------------------------------------------

    def prepare_commit(self, writes: List[Write], timestamp: int) -> PreparedCommit:
        prepared = prepare_storage_commit(writes, self._next_seq, timestamp)
        self.validate_committed(prepared.writes)
        return prepared

    def validate_committed(self, writes: List[CommittedWrite]) -> None:
        state = _CommitValidationState(self._entries, self._usage)
        validate_committed_writes(writes, self._next_seq, state)

    def apply_validated(self, writes: List[CommittedWrite]) -> SessionStats:
        for write in writes:
            if write.kind == "entry":
                entry = write.entry
                self._entries[entry.id] = entry
                self._entries_by_seq.append(entry)
                if entry.type == "message":
                    self._stats = SessionStats(
                        message_count=self._stats.message_count + 1, usage=self._stats.usage
                    )
            elif write.kind == "usage":
                row = write.entry
                self._usage[row.id] = row
                self._stats = SessionStats(
                    message_count=self._stats.message_count,
                    usage=add_usage(self._stats.usage, row.usage),
                )
            elif write.kind == "value":
                if write.op == "delete":
                    self._scalar_values.pop(_physical_key(write.namespace, write.key), None)
                else:
                    self._apply_value_set(write)
            elif write.kind == "list":
                if write.op == "delete":
                    self._list_values.pop(_physical_key(write.namespace, write.key), None)
                else:
                    self._apply_list_append(write)
            self._next_seq = write.seq + 1
        return self._stats

    def _apply_value_set(self, write: CommittedWrite) -> None:
        key = _physical_key(write.namespace, write.key)
        self._scalar_values[key] = StoredValue(
            address=value(write.namespace, write.key),
            value=write.value,
            seq=write.seq,
        )

    def _apply_list_append(self, write: CommittedWrite) -> None:
        key = _physical_key(write.namespace, write.key)
        element = ListElement(seq=write.seq, value=write.value)
        stored = self._list_values.get(key)
        if stored is None:
            self._list_values[key] = _StoredListSnapshot(
                address=list_address(write.namespace, write.key), elements=[element]
            )
        else:
            stored.elements.append(element)

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------

    def create_fork(self, options: ForkOptions) -> "InMemoryStorageState":
        plan_entry_ids: Set[str] = set()
        plan = self._select_fork_plan(options, plan_entry_ids)

        def is_entry_copied(entry_id: str) -> bool:
            if plan.scope == "tree":
                return True
            return entry_id in plan_entry_ids

        destination = InMemoryStorageState()
        message_count = 0
        for entry in self._entries_by_seq:
            if not is_entry_copied(entry.id):
                continue
            destination._entries[entry.id] = entry
            destination._entries_by_seq.append(entry)
            if entry.type == "message":
                message_count += 1
        destination._stats = SessionStats(message_count=message_count, usage=empty_usage())

        for stored in self._scalar_values.values():
            from .commit import CommittedWrite as _W

            projected = project_fork_current_state_write(
                _W(
                    kind="value",
                    op="set",
                    seq=stored.seq,
                    namespace=stored.address.namespace,
                    key=stored.address.key,
                    value=stored.value,
                ),
                plan,
                is_entry_copied,
            )
            if projected is not None:
                destination._apply_value_set(projected)

        for stored in self._list_values.values():
            from .commit import CommittedWrite as _W

            for element in stored.elements:
                projected = project_fork_current_state_write(
                    _W(
                        kind="list",
                        op="append",
                        seq=element.seq,
                        namespace=stored.address.namespace,
                        key=stored.address.key,
                        value=element.value,
                    ),
                    plan,
                    is_entry_copied,
                )
                if projected is not None:
                    destination._apply_list_append(projected)

        destination._next_seq = self._next_seq
        return destination

    def _select_fork_plan(self, options: ForkOptions, entry_ids: Set[str]) -> Any:
        if options.scope == "tree":
            return ForkCurrentStatePlan(scope="tree")

        tip_stored = self.get_value(branch_tip(options.branch))
        plan = select_branch_fork(
            options,
            {
                "tip": tip_stored.value if tip_stored is not None else None,
                "tip_known": tip_stored is not None,
                "get_parent": self._get_entry_parent,
                "select_entry": entry_ids.add,
            },
        )
        if (
            self.get_value(lane_config(options.branch)) is None
            or self.get_value(lane_state(options.branch)) is None
        ):
            raise RuntimeError(f"Source branch {options.branch!r} is not a configured AgentLane")
        return plan

    def _get_entry_parent(self, entry_id: str) -> Optional[str]:
        entry = self._entries.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        return entry.parent_id

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def advance_next_seq(self, next_seq: int) -> None:
        if not isinstance(next_seq, int) or next_seq < 1:
            raise RuntimeError(f"Invalid storage sequence high-water mark: {next_seq}")
        self._next_seq = max(self._next_seq, next_seq)

    def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        return {entry_id: self._entries[entry_id] for entry_id in ids if entry_id in self._entries}

    def get_value(self, address: Value) -> Optional[StoredValue]:
        return self._scalar_values.get(_physical_key(address.namespace, address.key))

    def scan_values(self, prefix: Value) -> List[StoredValue]:
        matches = [
            stored
            for stored in self._scalar_values.values()
            if stored.address.namespace == prefix.namespace and stored.address.key.startswith(prefix.key)
        ]
        return sorted(matches, key=lambda stored: stored.address.key)

    def read_list(self, address: ValueList, options: Optional[ListReadOptions] = None) -> List[ListElement]:
        resolved = resolve_list_read_options(options)
        stored = self._list_values.get(_physical_key(address.namespace, address.key))
        elements = stored.elements if stored is not None else []
        if resolved.cursor is not None:
            if resolved.order == "asc":
                filtered = [e for e in elements if e.seq > resolved.cursor.seq]
            else:
                filtered = [e for e in elements if e.seq < resolved.cursor.seq]
        else:
            filtered = list(elements)
        ordered = filtered if resolved.order == "asc" else list(reversed(filtered))
        return ordered[: resolved.limit]

    def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        start = self._entries.get(query.start)
        if start is None:
            raise RuntimeError(f"Unknown branch start: {query.start}")

        path: List[Entry] = []
        entry: Optional[Entry] = start
        while entry is not None:
            path.append(entry)
            if entry.parent_id is None:
                break
            entry = self._entries.get(entry.parent_id)
            if entry is None:
                raise RuntimeError("Corrupt branch: missing parent")
        if query.order == "oldestFirst":
            path.reverse()

        stopped: List[Entry] = []
        for candidate in path:
            stopped.append(candidate)
            if candidate.id == query.stop_at_id or candidate.type == query.stop_at_type:
                break
        filtered = [
            candidate
            for candidate in stopped
            if (query.type is None or candidate.type == query.type)
            and (query.custom_type is None or candidate.custom_type == query.custom_type)
            and (
                query.cursor is None
                or (
                    candidate.seq > query.cursor.seq
                    if query.order == "oldestFirst"
                    else candidate.seq < query.cursor.seq
                )
            )
        ]
        return filtered if query.limit is None else filtered[: max(0, query.limit)]

    def scan_branch_structure(self, query: StorageBranchScan) -> List[EntryStructure]:
        return [
            EntryStructure(
                id=entry.id,
                parent_id=entry.parent_id,
                seq=entry.seq,
                timestamp=entry.timestamp,
                type=entry.type,
                custom_type=entry.custom_type,
            )
            for entry in self.scan_branch(query)
        ]

    def scan_entries(self, query: EntryScan) -> List[Entry]:
        limit = float("inf") if query.limit is None else max(0, int(query.limit))
        entries: List[Entry] = []
        descending = query.order == "desc"
        index = len(self._entries_by_seq) - 1 if descending else 0
        while 0 <= index < len(self._entries_by_seq) and len(entries) < limit:
            entry = self._entries_by_seq[index]
            if (
                (query.type is None or entry.type == query.type)
                and (query.custom_type is None or entry.custom_type == query.custom_type)
                and (query.from_seq is None or entry.seq >= query.from_seq)
                and (query.to_seq is None or entry.seq <= query.to_seq)
            ):
                entries.append(entry)
            index += -1 if descending else 1
        return entries

    def scan_usage(self, query: UsageScan) -> List[UsageRow]:
        rows = [
            row
            for row in self._usage.values()
            if (query.from_seq is None or row.seq >= query.from_seq)
            and (query.to_seq is None or row.seq <= query.to_seq)
        ]
        rows.sort(key=lambda row: row.seq, reverse=query.order == "desc")
        return rows if query.limit is None else rows[: max(0, query.limit)]

    def get_stats(self) -> SessionStats:
        return self._stats

    def get_next_seq(self) -> int:
        return self._next_seq


class _CommitValidationState:
    def __init__(self, entries: dict, usage: dict) -> None:
        self._entries = entries
        self._usage = usage

    def has_entry_or_usage_id(self, entry_id: str) -> bool:
        return entry_id in self._entries or entry_id in self._usage

    def has_entry_id(self, entry_id: str) -> bool:
        return entry_id in self._entries
