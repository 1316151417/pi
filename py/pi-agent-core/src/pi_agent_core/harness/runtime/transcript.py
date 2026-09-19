"""Transcript projection helpers ported from ``harness/runtime/transcript.ts``.

Derives harness events from committed entries, reads the bounded run context,
and materializes lane queues and pending steer/follow-up messages.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ..._chord.context import Context
from ..session.commit import materialize_committed_entry
from ..session.context import build_session_context
from ..session.session import SessionInvariantError
from ..session.values import pending_entry
from ..events import HarnessEvent, LaneQueuedItem

__all__ = [
    "chain_entries",
    "entry_lifecycle_events",
    "committed_entry_events",
    "read_bounded_entries",
    "read_bounded_context",
    "read_lane_queues",
    "read_pending_messages",
    "PendingMessage",
]


def chain_entries(parent_id: Optional[str], items: Sequence[Any]) -> List[Any]:
    """Chain entries into a parent-linked sequence starting at ``parent_id``.

    Each element is a shallow copy with an overridden ``parentId``, so the
    caller keeps the original record while the committed write carries the
    chained parent.
    """
    chained: List[Any] = []
    for item in items:
        entry = copy.copy(item)
        entry.parent_id = parent_id
        chained.append(entry)
        parent_id = item.id
    return chained


def entry_lifecycle_events(entry: Any, lane: str, run_id: Optional[str] = None) -> List[HarnessEvent]:
    """Events emitted for one committed entry."""
    if getattr(entry, "type", None) == "message":
        return [
            HarnessEvent(type="message_start", lane=lane, run_id=run_id, message=entry.message),
            HarnessEvent(
                type="message_end", lane=lane, run_id=run_id, message=entry.message, entry_id=entry.id
            ),
            HarnessEvent(type="entry_added", lane=lane, entry=entry),
        ]
    return [HarnessEvent(type="entry_added", lane=lane, entry=entry)]


def committed_entry_events(
    entries: Sequence[Any],
    commit: Any,
    lane: str,
    run_id: Optional[str] = None,
    first_write_index: int = 0,
) -> List[HarnessEvent]:
    """Events for entries materialized from one commit result."""
    events: List[HarnessEvent] = []
    for index, entry in enumerate(entries):
        materialized = materialize_committed_entry(
            entry, commit.seqs[first_write_index + index], commit.timestamp
        )
        events.extend(entry_lifecycle_events(materialized, lane, run_id))
    return events


async def read_bounded_entries(lane: Any, drive: Any, capability: Any) -> Any:
    """Read the branch entries bounded by the last compaction, oldest first."""
    from ..session.types import BranchScan
    from .types import ContinueOperationResult, LaneReturn

    async def _operation(state: Any, _current: Any, _meta: Any, reader: Any) -> Any:
        if state.tip_id is None:
            raise SessionInvariantError("Run operation has no Branch tip")
        entries = await reader.scan_branch(
            BranchScan(start=state.tip_id, stop_at_type="compaction", order="newestFirst"),
            drive.context,
        )
        entries.reverse()
        return LaneReturn(result=entries)

    read = await lane.continue_operation(capability, _operation, drive.context)
    if read.kind == "cancel_requested":
        return read
    return ContinueOperationResult(kind="result", value=read.value)


async def read_bounded_context(lane: Any, drive: Any, capability: Any) -> Any:
    """Read model context messages for the bounded run transcript."""
    from .types import ContinueOperationResult

    entries = await read_bounded_entries(lane, drive, capability)
    if entries.kind == "cancel_requested":
        return entries
    return ContinueOperationResult(
        kind="result",
        value=await build_session_context(
            entries.value,
            {"entry_projectors": lane.read_config().entry_projectors},
            drive.context,
        ),
    )


async def read_lane_queues(reader: Any, inbox: Sequence[Any], context: Context) -> List[LaneQueuedItem]:
    """Materialize the durable lane inbox into queue items."""
    queues: List[LaneQueuedItem] = []
    for item in inbox:
        stored = await reader.get_value(pending_entry(item.entry_id), context)
        if stored is None:
            raise SessionInvariantError(
                f"Pending {item.kind} entry {item.entry_id} is missing its payload"
            )
        value = stored.value
        payload_type = getattr(value, "type", None)
        if isinstance(value, dict):
            payload_type = value.get("type")
        if payload_type == "message":
            payload = getattr(value, "payload", None) if not isinstance(value, dict) else value.get("payload")
            queues.append(
                LaneQueuedItem(
                    entry_id=item.entry_id, kind=item.kind, type="message", message=payload
                )
            )
            continue
        if item.kind != "write":
            raise SessionInvariantError(f"Pending {item.kind} entry {item.entry_id} is not a message")
        if isinstance(value, dict):
            custom_type = value.get("customType") or value.get("custom_type")
            data = value.get("payload")
        else:
            custom_type = getattr(value, "custom_type", None)
            data = getattr(value, "payload", None)
        queues.append(
            LaneQueuedItem(
                entry_id=item.entry_id,
                kind="write",
                type="custom",
                custom_type=custom_type,
                data=data,
            )
        )
    return queues


@dataclass
class PendingMessage:
    """One pending steer/follow-up message resolved from durable storage."""

    entry_id: str
    message: Any


async def read_pending_messages(
    reader: Any, ids: Sequence[str], description: str, context: Context
) -> List[PendingMessage]:
    """Resolve pending message payloads, failing loudly when one is missing."""
    resolved: List[PendingMessage] = []
    for entry_id in ids:
        stored = await reader.get_value(pending_entry(entry_id), context)
        value = stored.value if stored is not None else None
        payload_type = getattr(value, "type", None)
        if isinstance(value, dict):
            payload_type = value.get("type")
        if payload_type != "message":
            raise SessionInvariantError(f"{description} {entry_id} is missing its message payload")
        payload = value.get("payload") if isinstance(value, dict) else getattr(value, "payload", None)
        resolved.append(PendingMessage(entry_id=entry_id, message=payload))
    return resolved
