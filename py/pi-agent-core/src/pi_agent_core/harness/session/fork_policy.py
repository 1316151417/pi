"""Fork planning ported from ``harness/session/fork-policy.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Set

from .types import ForkOptions

__all__ = ["ForkCurrentStatePlan", "select_branch_fork", "project_fork_current_state_write"]


@dataclass
class ForkCurrentStatePlan:
    scope: str  # "branch" | "tree"
    branch: Optional[str] = None
    destination_tip: Optional[str] = None


def select_branch_fork(
    options: ForkOptions,
    source: dict,
) -> ForkCurrentStatePlan:
    """Walk the source branch selecting copied entries and the destination tip."""
    tip = source.get("tip")
    if tip is None and not source.get("tip_known", False):
        raise RuntimeError(f"Unknown source branch: {options.branch}")
    requested = options.entry_id if options.entry_id is not None else tip
    found = requested is None
    destination_tip: Optional[str] = None
    entry_id = tip
    while entry_id is not None:
        parent_id = source["get_parent"](entry_id)
        if parent_id is None and not source.get("parent_known", True):
            raise RuntimeError(f"Corrupt source branch: missing parent {entry_id}")
        if entry_id == requested:
            found = True
            destination_tip = parent_id if options.position == "before" else entry_id
            if options.position != "before":
                source["select_entry"](entry_id)
        elif found:
            source["select_entry"](entry_id)
        entry_id = parent_id
    if not found:
        raise RuntimeError(
            f"Fork entry {requested} is not on source branch {options.branch!r}"
        )
    return ForkCurrentStatePlan(scope="branch", branch=options.branch, destination_tip=destination_tip)


def project_fork_current_state_write(
    write: Any,
    plan: ForkCurrentStatePlan,
    is_entry_copied: Callable[[str], bool],
) -> Any:
    """Project one current scalar row or surviving list element into destination state."""
    namespace = write.namespace
    if namespace == "pi.session.name":
        return write
    if namespace == "pi.entry.label":
        return write if is_entry_copied(write.key) else None
    if namespace == "pi.branch.tip":
        if plan.scope == "tree":
            return write
        if write.key == plan.branch:
            projected = type(write)(**{**vars(write), "value": plan.destination_tip})
            return projected
        return None
    if namespace == "pi.lane.config":
        return write if plan.scope == "tree" or write.key == plan.branch else None
    if namespace == "pi.lane.state":
        if plan.scope == "tree" or write.key == plan.branch:
            projected = type(write)(**{**vars(write), "value": {
                "currentOperationId": None,
                "lastOperationId": None,
                "inbox": [],
            }})
            return projected
        return None
    if namespace == "pi.result":
        return None
    if namespace.startswith("pi.op.") or namespace.startswith("pi.pending."):
        return None
    if namespace == "pi" or namespace.startswith("pi."):
        raise RuntimeError(f"Unknown reserved fork namespace: {namespace}")
    return write if plan.scope == "tree" else None
