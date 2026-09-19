"""Run boundary helpers ported from ``harness/runtime/drive/boundary.ts``.

Selects and materializes one boundary's lane-owned input without committing it,
and replans after ``before_run_end`` to commit either renewed work or the
terminal run result.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from pi_ai.types import UserMessage  # noqa: F401  (re-exported for callers)
from ...result import Result  # noqa: F401  (keeps the error import graph stable)
from ...session.commit import insert_entry
from ...session.session import SessionInvariantError
from ...session.types import (
    AssistantReadyOperation,
    CustomEntry,
    MessageEntry,
    CheckpointOperation,
    GenerationContext,
    InboxItem,
    NormalizedRetryPolicy,
    OperationError,  # noqa: F401  (re-exported by callers through this module)
    OperationScope,
    PendingEntry,
    RunSettings,
    SummaryDecidingOperation,
    SummaryEffectPendingOperation,
    SummaryReadyOperation,
    Write,
)
from ...session.values import branch_tip, delete_value, pending_entry, set_value
from ..transcript import (
    committed_entry_events,
    entry_lifecycle_events,
    read_bounded_context,
    read_lane_queues,
)
from ..types import CommitDecision, FinishDecision, ProcedureResult
from .terminal import operation_cleanup_writes, operation_result_record

__all__ = [
    "BoundaryFinishPending",
    "BoundaryPlacement",
    "assistant_ready_at_boundary",
    "boundary_placement_events",
    "finish_run_boundary",
    "normalized_retry_policy",
    "plan_boundary_inbox",
]

#: Node timers cap out here; the TS default mirrors the pi-ai constant.
DEFAULT_MAX_AGENT_RETRY_DELAY_MS = 60_000


@dataclass
class BoundaryFinishPending:
    """A boundary whose terminal decision still waits on pending lane input."""

    kind: str = "finish_pending"
    entry_ids: List[str] = field(default_factory=list)


@dataclass
class BoundaryPlacement:
    """One boundary's selected input, materialized but not yet committed."""

    entries: List[Any] = field(default_factory=list)
    writes: List[Write] = field(default_factory=list)
    tip_id: Optional[str] = None
    inbox: List[InboxItem] = field(default_factory=list)
    trigger_entry_id: Optional[str] = None
    queues: Optional[List[Any]] = None


def normalized_retry_policy(lane: Any) -> NormalizedRetryPolicy:
    """The retry policy one operation captures for its durable generation context."""
    retry = lane.read_config().retry_policy
    return NormalizedRetryPolicy(
        max_attempts=(retry.max_retries + 1) if retry.enabled else 1,
        base_delay_ms=retry.base_delay_ms,
        max_agent_delay_ms=retry.max_agent_delay_ms or DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    )


def assistant_ready_at_boundary(
    lane: Any,
    state: Any,
    scope: OperationScope,
    trigger_entry_id: str,
    overflow_recovery_used: bool,
) -> AssistantReadyOperation:
    """The successor assistant scope created when a boundary admits new input."""
    config = lane.read_config()
    return AssistantReadyOperation(
        control=scope.control,
        settings=scope.settings,
        latest_assistant_entry_id=scope.latest_assistant_entry_id,
        generation_context=GenerationContext(
            step_id=lane.session.id_generator.next(),
            trigger_entry_id=trigger_entry_id,
            configuration=state.configuration,
            stream_options=config.stream_options,
            retry_policy=normalized_retry_policy(lane),
            overflow_recovery_used=overflow_recovery_used,
        ),
        next_attempt=1,
    )


def _pending(value: Any) -> PendingEntry:
    """Normalize a stored pending payload, which may be a replayed mapping."""
    return PendingEntry.from_value(value)


async def plan_boundary_inbox(
    lane: Any,
    drive: Any,
    state: Any,
    scope: OperationScope,
    reader: Any,
    tip_id: Optional[str],
    follow_up_when_no_trigger: bool,
) -> BoundaryPlacement:
    """Select and materialize one boundary's lane-owned input without committing it."""
    steer = [item for item in state.inbox if item.kind == "steer"]
    selected_steer = steer if scope.settings.steering_mode == "all" else steer[:1]
    selected_steer_ids = {item.entry_id for item in selected_steer}
    selected = [
        item
        for item in state.inbox
        if item.kind == "write" or item.entry_id in selected_steer_ids
    ]

    async def _load(items: Sequence[InboxItem]) -> List[Any]:
        loaded = []
        for item in items:
            stored = await reader.get_value(pending_entry(item.entry_id), drive.context)
            if stored is None:
                raise SessionInvariantError(
                    f"Pending {item.kind} entry {item.entry_id} is missing its payload"
                )
            if item.kind != "write" and _pending(stored.value).type != "message":
                raise SessionInvariantError(
                    f"Queued {item.kind} entry {item.entry_id} is not a message"
                )
            loaded.append((item, stored.value))
        return loaded

    pending = await _load(selected)

    def _projects(value: Any) -> bool:
        pending = _pending(value)
        if pending.type == "message":
            return True
        return lane.read_config().entry_projectors.get(pending.custom_type) is not None

    if follow_up_when_no_trigger and not any(_projects(value) for _item, value in pending):
        follow_up = [item for item in state.inbox if item.kind == "followUp"]
        selected_follow_up = (
            follow_up if scope.settings.follow_up_mode == "all" else follow_up[:1]
        )
        merged = [*selected, *selected_follow_up]
        order = {id(item): index for index, item in enumerate(state.inbox)}
        selected = sorted(merged, key=lambda item: order[id(item)])
        pending = await _load(selected)

    parent_id = tip_id
    trigger_entry_id: Optional[str] = None
    entries: List[Any] = []
    for item, value in pending:
        pending_entry_value = _pending(value)
        entry: Any
        if pending_entry_value.type == "message":
            entry = MessageEntry(
                id=item.entry_id, parent_id=parent_id, message=pending_entry_value.payload
            )
        else:
            entry = CustomEntry(
                id=item.entry_id,
                parent_id=parent_id,
                custom_type=pending_entry_value.custom_type or "",
                data=pending_entry_value.payload,
            )
        entries.append(entry)
        parent_id = item.entry_id
        if _projects(value):
            trigger_entry_id = item.entry_id

    selected_ids = {item.entry_id for item in selected}
    inbox = [item for item in state.inbox if item.entry_id not in selected_ids]
    queues = (
        None if not selected else await read_lane_queues(reader, inbox, drive.context)
    )
    writes: List[Write] = [
        *[insert_entry(entry) for entry in entries],
        *[delete_value(pending_entry(item.entry_id)) for item in selected],
    ]
    if entries:
        writes.append(set_value(branch_tip(lane.name), parent_id))
    return BoundaryPlacement(
        entries=entries,
        writes=writes,
        tip_id=parent_id,
        inbox=inbox,
        trigger_entry_id=trigger_entry_id,
        queues=queues,
    )


def boundary_placement_events(
    placement: BoundaryPlacement,
    commit: Any,
    first_write_index: int,
    lane: str,
    run_id: str,
) -> List[Any]:
    """Events for one committed boundary placement."""
    events = list(
        committed_entry_events(placement.entries, commit, lane, run_id, first_write_index)
    )
    if placement.queues is not None:
        events.append({"type": "queue_update", "lane": lane, "queues": placement.queues})
    return events


async def finish_run_boundary(
    lane: Any,
    drive: Any,
    capability: Any,
    continuation: Any,
    planned_entry_ids: Sequence[str],
    pending_events: Optional[List[Any]] = None,
) -> ProcedureResult:
    """Replan after before_run_end and commit either renewed work or the terminal run result."""
    events_carried: List[Any] = list(pending_events or [])
    context = await read_bounded_context(lane, drive, capability)
    if context.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    hook = await lane.hooks.run_with_gate(
        "before_run_end",
        {"lane": lane.name, "runId": drive.operation_id, "messages": context.value},
        drive.gate,
        drive.context,
    )
    hook_follow_up = None if hook is None else _opt(hook, "follow_up", "followUp")
    follow_up = (
        None
        if hook_follow_up is None
        else {
            "id": lane.session.id_generator.next(),
            "message": UserMessage(content=hook_follow_up, timestamp=int(time.time() * 1000)),
        }
    )

    async def _plan(state: Any, current: Any, meta: Any, reader: Any) -> Any:
        placement = await plan_boundary_inbox(
            lane, drive, state, current, reader, state.tip_id, True
        )
        if placement.trigger_entry_id is not None:
            return CommitDecision(
                writes=placement.writes,
                operation_state=assistant_ready_at_boundary(
                    lane, state, current, placement.trigger_entry_id, False
                ),
                lane={"tipId": placement.tip_id, "inbox": placement.inbox},
                materialize=lambda _commit: ProcedureResult(kind="continue"),
                events=lambda commit: [
                    *events_carried,
                    *boundary_placement_events(
                        placement, commit, 0, lane.name, drive.operation_id
                    ),
                ],
            )
        hook_plan_is_current = len(placement.entries) == len(planned_entry_ids) and all(
            placement.entries[index].id == planned_entry_ids[index]
            for index in range(len(placement.entries))
        )
        if hook_plan_is_current and follow_up is not None:
            entry = MessageEntry(
                id=follow_up["id"], parent_id=placement.tip_id, message=follow_up["message"]
            )
            entry_write_index = len(placement.writes)

            def _hook_events(commit: Any) -> List[Any]:
                materialized = copy.copy(entry)
                materialized.seq = commit.seqs[entry_write_index]
                materialized.timestamp = commit.timestamp
                return [
                    *events_carried,
                    *boundary_placement_events(
                        placement, commit, 0, lane.name, drive.operation_id
                    ),
                    *entry_lifecycle_events(materialized, lane.name, drive.operation_id),
                ]

            return CommitDecision(
                writes=[
                    *placement.writes,
                    insert_entry(entry),
                    set_value(branch_tip(lane.name), follow_up["id"]),
                ],
                operation_state=assistant_ready_at_boundary(
                    lane, state, current, follow_up["id"], False
                ),
                lane={"tipId": follow_up["id"], "inbox": placement.inbox},
                materialize=lambda _commit: ProcedureResult(kind="continue"),
                events=_hook_events,
            )
        if placement.tip_id is None:
            raise SessionInvariantError("Completed run has no tip")
        if continuation.include_final_assistant and current.latest_assistant_entry_id is None:
            raise SessionInvariantError("Completed run is missing its final assistant")
        record = operation_result_record(meta, "completed", placement.tip_id)
        cleanup = await operation_cleanup_writes(
            reader, drive.operation_id, current, drive.context
        )

        def _finish_events(commit: Any) -> List[Any]:
            return [
                *events_carried,
                *boundary_placement_events(placement, commit, 0, lane.name, drive.operation_id),
                {
                    "type": "run_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "status": "completed",
                    "fromTipId": meta.source_tip_id,
                    "tipId": placement.tip_id,
                    "endedAt": record.ended_at,
                },
            ]

        return FinishDecision(
            writes=[*placement.writes, *cleanup],
            record=record,
            lane={"tipId": placement.tip_id, "inbox": placement.inbox},
            materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
            events=_finish_events,
        )

    result = await lane.continue_operation(capability, _plan, drive.context)
    if result.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    return result.value


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping or an attribute."""
    if value is None:
        return None
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
