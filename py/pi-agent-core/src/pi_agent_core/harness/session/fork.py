"""Fork snapshot building ported from ``harness/session/fork.ts``.

Builds the complete logical state for a forked destination session: the copied
entry set, the projected scalar values, and the next storage sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .fork_policy import ForkCurrentStatePlan, project_fork_current_state_write, select_branch_fork
from .types import Entry, ForkOptions
from .values import branch_tip, lane_config, lane_state, value

__all__ = ["ForkSourceSnapshot", "ForkDestinationSnapshot", "create_fork_snapshot"]


@dataclass
class ForkSourceSnapshot:
    """Logical state read out of a source session."""

    entries: List[Entry] = field(default_factory=list)
    scalar_values: List[Any] = field(default_factory=list)
    #: False when a backend supplied only the requested branch rather than the full tree.
    entries_complete: Optional[bool] = None


@dataclass
class ForkDestinationSnapshot:
    """Logical state to materialize for a forked destination session."""

    entries: Dict[str, Entry] = field(default_factory=dict)
    scalar_values: List[Any] = field(default_factory=list)
    next_seq: int = 1


def _stored_values_in_namespace(values: List[Any], address: Any) -> List[Any]:
    return [stored for stored in values if stored.address.namespace == address.namespace]


def _find_stored_value(values: List[Any], address: Any) -> Optional[Any]:
    return next(
        (
            stored
            for stored in values
            if stored.address.namespace == address.namespace and stored.address.key == address.key
        ),
        None,
    )


def create_fork_snapshot(source: ForkSourceSnapshot, options: ForkOptions) -> ForkDestinationSnapshot:
    """Build the complete logical state for a forked destination session."""
    source_entries = {entry.id: entry for entry in source.entries}
    source_tips = _stored_values_in_namespace(source.scalar_values, branch_tip(""))
    _validate_fork_source_snapshot(source, source_entries, source_tips, options)

    entry_ids, plan = _select_fork_contents(source_entries, source_tips, options)
    entries = {entry_id: source_entries[entry_id] for entry_id in entry_ids}

    scalar_values: List[Any] = []
    next_seq = max([0, *[entry.seq for entry in entries.values()]]) + 1
    for stored in source.scalar_values:
        projected = project_fork_current_state_write(
            _ValueWrite(
                kind="value",
                op="set",
                seq=stored.seq,
                namespace=stored.address.namespace,
                key=stored.address.key,
                value=stored.value,
            ),
            plan,
            lambda entry_id: entry_id in entry_ids,
        )
        if projected is not None:
            scalar_values.append(
                _StoredValue(
                    address=value(projected.namespace, projected.key),
                    value=projected.value,
                    seq=next_seq,
                )
            )
            next_seq += 1

    return ForkDestinationSnapshot(entries=entries, scalar_values=scalar_values, next_seq=next_seq)


class _ValueWrite:
    """Minimal committed-write shape used to project current state into a fork."""

    def __init__(self, kind: str, op: str, seq: int, namespace: str, key: str, value: Any) -> None:
        self.kind = kind
        self.op = op
        self.seq = seq
        self.namespace = namespace
        self.key = key
        self.value = value


class _StoredValue:
    """Stored value shape returned on the destination snapshot."""

    def __init__(self, address: Any, value: Any, seq: int) -> None:
        self.address = address
        self.value = value
        self.seq = seq


def _select_fork_contents(
    source_entries: Dict[str, Entry], source_tips: List[Any], options: ForkOptions
) -> tuple:
    entry_ids: Set[str] = set()
    if options.scope == "tree":
        for entry_id in source_entries:
            entry_ids.add(entry_id)
        return entry_ids, ForkCurrentStatePlan(scope="tree")

    source_tip = next(
        (stored for stored in source_tips if stored.address.key == options.branch), None
    )
    plan = select_branch_fork(
        options,
        {
            "tip": source_tip.value if source_tip is not None else None,
            "tip_known": source_tip is not None,
            "get_parent": lambda entry_id: (
                source_entries[entry_id].parent_id
                if entry_id in source_entries
                else (_ for _ in ()).throw(KeyError(entry_id))
            ),
            "select_entry": entry_ids.add,
        },
    )
    return entry_ids, plan


def _validate_fork_source_snapshot(
    source: ForkSourceSnapshot,
    source_entries: Dict[str, Entry],
    source_tips: List[Any],
    options: ForkOptions,
) -> None:
    source_tip_keys = {stored.address.key for stored in source_tips}

    for stored in source.scalar_values:
        if (
            stored.address.namespace
            in (lane_config("").namespace, lane_state("").namespace)
            and stored.address.key not in source_tip_keys
        ):
            raise RuntimeError(
                f"Source session branch {stored.address.key!r} is missing branch.tip"
            )

    for tip in source_tips:
        configuration = _find_stored_value(source.scalar_values, lane_config(tip.address.key))
        state = _find_stored_value(source.scalar_values, lane_state(tip.address.key))
        if (configuration is None) != (state is None):
            raise RuntimeError(
                f"Source session branch {tip.address.key!r} has incomplete lane state"
            )
        if options.scope == "branch" and tip.address.key == options.branch and configuration is None:
            raise RuntimeError(
                f"Source branch {options.branch!r} is not a configured AgentLane"
            )
        if (
            (source.entries_complete is not False or options.scope == "tree")
            and tip.value is not None
            and tip.value not in source_entries
        ):
            raise RuntimeError(
                f"Source session branch {tip.address.key!r} has an unknown tip"
            )
