"""Run checkpoint procedures ported from ``harness/runtime/drive/checkpoint.ts``.

``start_run`` consumes ``before_run`` and commits the initial checkpoint;
``run_checkpoint`` advances one durable run boundary with at most one commit.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ...session.commit import insert_entry
from ...session.session import SessionInvariantError
from ...session.types import (
    CheckpointOperation,
    Continuation,
    MessageEntry,
    StartingOperation,
    SummaryDecidingOperation,
    SummaryTask,
    ResultBoundary,
    CheckpointData,
)
from ...session.values import branch_tip, operation_preparation, set_value
from ..transcript import chain_entries, committed_entry_events
from ..types import CommitDecision, LaneReturn, ProcedureResult
from .boundary import (
    BoundaryFinishPending,
    assistant_ready_at_boundary,
    boundary_placement_events,
    finish_run_boundary,
    plan_boundary_inbox,
)
from .structural import prepare_compaction_threshold

__all__ = ["start_run", "run_checkpoint"]


async def start_run(lane: Any, drive: Any, run: StartingOperation) -> ProcedureResult:
    """Consume ``before_run`` and commit the initial checkpoint."""

    async def _read_prompt(_state: Any, _current: Any, meta: Any, reader: Any) -> Any:
        if meta.intent.kind != "run":
            raise SessionInvariantError("Run operation has non-run intent")
        entries = await reader.get_entries(meta.intent.prompt_entry_ids, drive.context)
        messages = []
        for entry_id in meta.intent.prompt_entry_ids:
            entry = entries.get(entry_id)
            if getattr(entry, "type", None) != "message":
                raise SessionInvariantError(f"Run prompt entry {entry_id} is missing its message")
            messages.append(entry.message)
        return LaneReturn(result=messages)

    prompt = await lane.continue_operation(run, _read_prompt, drive.context)
    if prompt.kind == "cancel_requested":
        return ProcedureResult(kind="continue")

    hook = await lane.hooks.run_with_gate(
        "before_run",
        {
            "lane": lane.name,
            "runId": drive.operation_id,
            "prompt": prompt.value,
            "resources": lane.read_config().resources,
        },
        drive.gate,
        drive.context,
    )
    injected = [] if hook is None else (_opt(hook, "messages", "messages") or [])
    for message in injected:
        if (
            getattr(message, "role", None) == "assistant"
            and getattr(message, "stop_reason", None) == "pending"
        ):
            raise SessionInvariantError("before_run returned a pending assistant message")
    reserved = [(lane.session.id_generator.next(), message) for message in injected]

    def _plan(state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        entries = chain_entries(
            state.tip_id,
            [MessageEntry(id=entry_id, parent_id=None, message=message) for entry_id, message in reserved],
        )
        trigger_entry_id = entries[-1].id if entries else state.tip_id
        if trigger_entry_id is None:
            raise SessionInvariantError("Run start has no trigger entry")
        next_state = CheckpointOperation(
            control=current.control,
            settings=current.settings,
            latest_assistant_entry_id=current.latest_assistant_entry_id,
            continuation=Continuation(kind="need_assistant", overflow_recovery_used=False),
            trigger_entry_id=trigger_entry_id,
        )
        writes: List[Any] = [*[insert_entry(entry) for entry in entries]]
        if entries:
            writes.append(set_value(branch_tip(lane.name), trigger_entry_id))
        return CommitDecision(
            writes=writes,
            operation_state=next_state,
            lane={"tipId": trigger_entry_id},
            materialize=lambda _commit: ProcedureResult(kind="continue"),
            events=lambda commit: committed_entry_events(
                entries, commit, lane.name, drive.operation_id
            ),
        )

    result = await lane.continue_operation(run, _plan, drive.context)
    return ProcedureResult(kind="continue") if result.kind == "cancel_requested" else result.value


async def run_checkpoint(lane: Any, drive: Any, run: CheckpointOperation) -> ProcedureResult:
    """Advance one durable run boundary with at most one commit."""
    threshold = await prepare_compaction_threshold(lane, drive, run)
    if threshold.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    threshold_value = threshold.value

    async def _plan(state: Any, current: Any, _meta: Any, reader: Any) -> Any:
        placement = await plan_boundary_inbox(
            lane,
            drive,
            state,
            current,
            reader,
            state.tip_id,
            threshold_value is None and current.continuation.kind == "may_finish",
        )
        if placement.trigger_entry_id is not None:
            return CommitDecision(
                writes=placement.writes,
                operation_state=assistant_ready_at_boundary(
                    lane, state, current, placement.trigger_entry_id, False
                ),
                lane={"tipId": placement.tip_id, "inbox": placement.inbox},
                materialize=lambda _commit: ProcedureResult(kind="continue"),
                events=lambda commit: boundary_placement_events(
                    placement, commit, 0, lane.name, drive.operation_id
                ),
            )
        if threshold_value is not None:
            structural = SummaryDecidingOperation(
                control=current.control,
                settings=current.settings,
                latest_assistant_entry_id=current.latest_assistant_entry_id,
                task=SummaryTask(
                    task_id=threshold_value["taskId"],
                    reason="threshold",
                    boundary=ResultBoundary(
                        kind="resume_checkpoint",
                        resume_after=CheckpointData(
                            continuation=current.continuation,
                            trigger_entry_id=current.trigger_entry_id,
                        ),
                    ),
                ),
            )

            def _threshold_events(commit: Any) -> List[Any]:
                return [
                    *boundary_placement_events(placement, commit, 0, lane.name, drive.operation_id),
                    {
                        "type": "compaction_start",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "reason": "threshold",
                        "startedAt": commit.timestamp,
                    },
                ]

            return CommitDecision(
                writes=[
                    *placement.writes,
                    set_value(
                        operation_preparation(drive.operation_id, threshold_value["taskId"]),
                        threshold_value["preparation"],
                    ),
                ],
                operation_state=structural,
                lane={"tipId": placement.tip_id, "inbox": placement.inbox},
                materialize=lambda _commit: ProcedureResult(kind="continue"),
                events=_threshold_events,
            )
        if current.continuation.kind == "need_assistant":
            return CommitDecision(
                writes=placement.writes,
                operation_state=assistant_ready_at_boundary(
                    lane,
                    state,
                    current,
                    current.trigger_entry_id,
                    current.continuation.overflow_recovery_used,
                ),
                lane={"tipId": placement.tip_id, "inbox": placement.inbox},
                materialize=lambda _commit: ProcedureResult(kind="continue"),
                events=lambda commit: boundary_placement_events(
                    placement, commit, 0, lane.name, drive.operation_id
                ),
            )
        return LaneReturn(
            result=BoundaryFinishPending(
                entry_ids=[entry.id for entry in placement.entries]
            )
        )

    planned = await lane.continue_operation(run, _plan, drive.context)
    if planned.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    if planned.value.kind != "finish_pending":
        return planned.value
    if run.continuation.kind != "may_finish":
        raise SessionInvariantError("Checkpoint finish mediation requires a finish continuation")
    return await finish_run_boundary(
        lane, drive, run, run.continuation, planned.value.entry_ids
    )


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
