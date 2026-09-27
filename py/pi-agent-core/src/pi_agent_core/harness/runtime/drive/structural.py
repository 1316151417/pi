"""Structural procedures ported from ``harness/runtime/drive/structural.ts``.

Owns the durable compaction/branch-summary handoff: the durable preparation
records, the decision and generation leaves, retry waiting, cancellation
recovery and navigation commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import time
from typing import Awaitable, Callable

from pi_ai.types import AssistantMessage
from ...compaction.branch_summarization import (
    BranchPreparation,
    generate_branch_summary_with_request,
)
from ...compaction.utils import create_file_ops
from ...compaction.compaction import (
    CompactGenerationOptions,
    CompactionPreparation,
    compact_with_request,
)
from ...context import get_telemetry_context, with_abort_signal
from ...execution.effect_gate import AbortRequested
from ...hooks import apply_stream_options_patch
from ...session.commit import insert_entry, insert_usage
from ...session.session import SessionInvariantError
from ...session.types import (
    AssistantReadyOperation,
    BranchSummaryEntry,
    CheckpointOperation,
    CompactionEntry,
    OperationError,
    SummaryContext,
    SummaryEffectPendingOperation,
    SummaryReadyOperation,
    SummaryRetryWaitOperation,
    UsageRow,
    operation_scope_of,
)
from ...session.values import (
    branch_tip,
    entry_label,
    operation_preparation,
    set_value,
)
from pi_ai.utils.retry import is_retryable_assistant_error, retry_delay_ms
from ..transcript import committed_entry_events, read_bounded_entries
from ..types import CommitDecision, ContinueOperationResult, FinishDecision, LaneReturn, ProcedureResult
from .boundary import (
    assistant_ready_at_boundary,
    boundary_placement_events,
    finish_run_boundary,
    normalized_retry_policy,
    plan_boundary_inbox,
)
from .retry import retry_not_before, wait_until
from .terminal import operation_cleanup_writes, operation_result_record

__all__ = [
    "DurableFileOperations",
    "DurableCompactionPreparation",
    "DurableBranchPreparation",
    "durable_file_operations",
    "file_operations",
    "durable_compaction_preparation",
    "durable_branch_preparation",
    "compaction_preparation",
    "branch_preparation",
    "summary_kind",
    "compaction_reason",
    "navigation_boundary",
    "run_structural_decision",
    "run_structural_generation",
    "run_structural_retry_wait",
    "recover_structural_generation",
    "prepare_compaction_threshold",
    "prepare_overflow_compaction",
    "commit_navigation",
    "OverflowPreparation",
    "StructuralCancelled",
    "report_operations",
    "summary_context",
    "publish_structural_outcome",
]


@dataclass
class DurableFileOperations:
    """File paths touched by a structural preparation, as durable ordered lists."""

    read: List[str] = field(default_factory=list)
    written: List[str] = field(default_factory=list)
    edited: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"read": list(self.read), "written": list(self.written), "edited": list(self.edited)}


@dataclass
class DurableCompactionPreparation:
    """A compaction preparation in its durable wire shape."""

    kind: str = "compaction"
    messages_to_summarize: List[Any] = field(default_factory=list)
    turn_prefix_messages: List[Any] = field(default_factory=list)
    retained_tail: List[Any] = field(default_factory=list)
    is_split_turn: bool = False
    tokens_before: int = 0
    previous_summary: Optional[str] = None
    file_ops: DurableFileOperations = field(default_factory=DurableFileOperations)
    settings: Any = None

    def to_json(self) -> dict:
        data: dict = {
            "kind": self.kind,
            "messagesToSummarize": list(self.messages_to_summarize),
            "turnPrefixMessages": list(self.turn_prefix_messages),
            "retainedTail": list(self.retained_tail),
            "isSplitTurn": self.is_split_turn,
            "tokensBefore": self.tokens_before,
            "fileOps": self.file_ops,
            "settings": self.settings,
        }
        if self.previous_summary is not None:
            data["previousSummary"] = self.previous_summary
        return data


@dataclass
class DurableBranchPreparation:
    """A branch preparation in its durable wire shape."""

    kind: str = "branch_summary"
    messages: List[Any] = field(default_factory=list)
    file_ops: DurableFileOperations = field(default_factory=DurableFileOperations)
    total_tokens: int = 0

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "messages": list(self.messages),
            "fileOps": self.file_ops,
            "totalTokens": self.total_tokens,
        }


def durable_file_operations(file_ops: Any) -> DurableFileOperations:
    """Freeze one preparation's file operations into durable ordered lists."""
    return DurableFileOperations(
        read=[*file_ops.read],
        written=[*file_ops.written],
        edited=[*file_ops.edited],
    )


def file_operations(file_ops: Any) -> Any:
    """Thaw durable file operations back into the in-process sets."""
    read = _field(file_ops, "read", "read", [])
    written = _field(file_ops, "written", "written", [])
    edited = _field(file_ops, "edited", "edited", [])
    return create_file_ops().__class__(read=set(read), written=set(written), edited=set(edited))


def durable_compaction_preparation(preparation: Any) -> DurableCompactionPreparation:
    """The durable compaction preparation for one admission."""
    return DurableCompactionPreparation(
        messages_to_summarize=list(preparation.messages_to_summarize),
        turn_prefix_messages=list(preparation.turn_prefix_messages),
        retained_tail=list(preparation.retained_tail),
        is_split_turn=preparation.is_split_turn,
        tokens_before=preparation.tokens_before,
        previous_summary=preparation.previous_summary,
        file_ops=durable_file_operations(preparation.file_ops),
        settings=preparation.settings,
    )


def durable_branch_preparation(preparation: Any) -> DurableBranchPreparation:
    """The durable branch preparation for one admission."""
    return DurableBranchPreparation(
        messages=list(preparation.messages),
        file_ops=durable_file_operations(preparation.file_ops),
        total_tokens=preparation.total_tokens,
    )


def compaction_preparation(preparation: Any) -> CompactionPreparation:
    """Thaw one durable compaction preparation for generation."""
    return CompactionPreparation(
        messages_to_summarize=list(_field(preparation, "messages_to_summarize", "messagesToSummarize", [])),
        turn_prefix_messages=list(_field(preparation, "turn_prefix_messages", "turnPrefixMessages", [])),
        retained_tail=list(_field(preparation, "retained_tail", "retainedTail", [])),
        is_split_turn=bool(_field(preparation, "is_split_turn", "isSplitTurn", False)),
        tokens_before=_field(preparation, "tokens_before", "tokensBefore", 0),
        previous_summary=_field(preparation, "previous_summary", "previousSummary"),
        file_ops=file_operations(_field(preparation, "file_ops", "fileOps", {})),
        settings=_field(preparation, "settings", "settings"),
    )


def branch_preparation(preparation: Any) -> BranchPreparation:
    """Thaw one durable branch preparation for generation."""
    return BranchPreparation(
        messages=list(_field(preparation, "messages", "messages", [])),
        file_ops=file_operations(_field(preparation, "file_ops", "fileOps", {})),
        total_tokens=_field(preparation, "total_tokens", "totalTokens", 0),
    )


def summary_kind(task: Any) -> str:
    """Which structural preparation a task expects."""
    return "branch_summary" if task.boundary.kind == "commit_navigation" else "compaction"


def compaction_reason(task: Any) -> str:
    """The recorded trigger reason for a compaction task."""
    if task.reason is not None:
        return task.reason
    if task.boundary.kind == "finish":
        return "manual"
    from ...session.session import SessionInvariantError

    raise SessionInvariantError(
        f"In-run compaction task {task.task_id} is missing its reason"
    )


def navigation_boundary(task: Any) -> Any:
    """The commit_navigation boundary of a summary task, or a loud failure."""
    if task.boundary.kind != "commit_navigation":
        from ...session.session import SessionInvariantError

        raise SessionInvariantError(f"Summary task {task.task_id} is not a navigation")
    return task.boundary


def _field(value: Any, snake: str, camel: str, default: Any = None) -> Any:
    """Read a durable field from a dataclass instance or a JSON-replayed mapping."""
    if value is None:
        return default
    if isinstance(value, dict):
        if camel in value:
            return value[camel]
        if snake in value:
            return value[snake]
        return default
    return getattr(value, snake, default)



# ---------------------------------------------------------------------------
# Structural procedures
# ---------------------------------------------------------------------------


class StructuralCancelled(Exception):
    """Raised when a structural generation was cancelled mid-attempt."""

    def __init__(self) -> None:
        super().__init__("Structural generation was cancelled")
        self.name = "StructuralCancelled"


def summary_context(lane: Any, result_entry_id: str, configuration: Any) -> SummaryContext:
    """The generation scope one structural result commits into."""
    return SummaryContext(
        result_entry_id=result_entry_id,
        configuration=configuration,
        stream_options=_replace_field(
            lane.read_config().stream_options, "deferred", False
        ),
        retry_policy=normalized_retry_policy(lane),
    )


def _usage_event(row: dict, write_index: int, commit: Any, lane: str) -> dict:
    return {
        "type": "usage",
        "lane": lane,
        "row": {**row, "seq": commit.seqs[write_index]},
        "totals": commit.stats.usage,
    }


def _operation_error(code: str, message: str, details: Any = None) -> OperationError:
    return OperationError(code=code, message=message, details=details)


async def _read_structural_preparation(
    lane: Any, drive: Any, deciding: Any
) -> ContinueOperationResult:
    async def _plan(_state: Any, current: Any, _meta: Any, reader: Any) -> Any:
        expected = summary_kind(current.task)
        stored = await reader.get_value(
            operation_preparation(drive.operation_id, current.task.task_id), drive.context
        )
        if stored is None or _field(stored.value, "kind", "kind") != expected:
            raise SessionInvariantError(
                f"Structural task {current.task.task_id} is missing its {expected} preparation"
            )
        if current.task.boundary.kind == "commit_navigation":
            target_id = current.task.boundary.target_id
            if target_id not in (await reader.get_entries([target_id], drive.context)):
                raise SessionInvariantError(f"Navigation target {target_id} is missing")
        return LaneReturn(
            result=(
                compaction_preparation(stored.value)
                if _field(stored.value, "kind", "kind") == "compaction"
                else branch_preparation(stored.value)
            )
        )

    return await lane.continue_operation(deciding, _plan, drive.context)


async def publish_structural_outcome(
    lane: Any, drive: Any, capability: Any, outcome: dict
) -> ProcedureResult:
    """Commit one structural outcome under its task boundary."""
    hook_usage_id = None
    if (
        outcome["kind"] in ("compaction", "branch_summary")
        and outcome["fromHook"]
        and getattr(outcome["result"], "usage", None) is not None
    ):
        hook_usage_id = lane.session.id_generator.next()

    async def _plan(state: Any, current: Any, meta: Any, reader: Any) -> Any:
        expected = summary_kind(current.task)
        terminal_compaction_ended_at: Optional[int] = None
        if outcome["kind"] in ("compaction", "branch_summary") and outcome["kind"] != expected:
            raise SessionInvariantError(
                f"Structural {outcome['kind']} result does not match {expected} task "
                f"{current.task.task_id}"
            )

        writes: List[Any] = []
        base_events: List[Callable[[Any], List[Any]]] = []
        terminal_tip_id = state.tip_id
        if hook_usage_id is not None and outcome["kind"] in ("compaction", "branch_summary"):
            usage = getattr(outcome["result"], "usage", None)
            if usage is None:
                raise SessionInvariantError("Hook usage id exists without structural usage")
            row = {"id": hook_usage_id, "usage": usage, "adjustment": False}
            write_index = len(writes)
            writes.append(insert_usage(_usage_row(row)))
            base_events.append(
                lambda commit, w=write_index, r=row: [
                    _usage_event(r, w, commit, lane.name)
                ]
            )

        if outcome["kind"] == "compaction":
            entry = CompactionEntry(
                id=outcome["resultEntryId"],
                parent_id=state.tip_id,
                summary=outcome["result"].summary,
                retained_tail=list(outcome["result"].retained_tail),
                tokens_before=outcome["result"].tokens_before,
                details=outcome["result"].details,
                **({} if getattr(outcome["result"], "usage", None) is None else {"usage": outcome["result"].usage}),
                from_hook=outcome["fromHook"],
            )
            entry_write_index = len(writes)
            writes.append(insert_entry(entry))
            writes.append(set_value(branch_tip(lane.name), outcome["resultEntryId"]))
            terminal_tip_id = outcome["resultEntryId"]
            base_events.append(
                lambda commit, e=entry, w=entry_write_index: committed_entry_events(
                    [e], commit, lane.name, drive.operation_id, w
                )
            )
        elif outcome["kind"] == "branch_summary":
            boundary = navigation_boundary(current.task)
            entry = BranchSummaryEntry(
                id=outcome["resultEntryId"],
                parent_id=boundary.target_id,
                from_id=meta.source_tip_id,
                summary=outcome["result"].summary,
                details={
                    "readFiles": list(outcome["result"].read_files),
                    "modifiedFiles": list(outcome["result"].modified_files),
                },
                **({} if getattr(outcome["result"], "usage", None) is None else {"usage": outcome["result"].usage}),
                from_hook=outcome["fromHook"],
            )
            writes.append(set_value(branch_tip(lane.name), boundary.target_id))
            entry_write_index = len(writes)
            writes.append(insert_entry(entry))
            writes.append(set_value(branch_tip(lane.name), outcome["resultEntryId"]))
            if boundary.label is not None:
                writes.append(set_value(entry_label(boundary.target_id), boundary.label))
            terminal_tip_id = outcome["resultEntryId"]
            base_events.append(
                lambda commit, e=entry, w=entry_write_index: committed_entry_events(
                    [e], commit, lane.name, drive.operation_id, w
                )
            )

        attempt = None
        if current.at == "summary.ready":
            attempt = current.next_attempt
        elif current.at == "summary.effect_pending":
            attempt = current.attempt
        if attempt is not None and attempt > 1:
            base_events.append(
                lambda _commit, a=attempt: [
                    {
                        "type": "retry_end",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "step": current.task.task_id,
                        "attempt": a,
                        "success": outcome["kind"] in ("compaction", "branch_summary"),
                        **(
                            {}
                            if outcome["kind"] != "failed"
                            else {"finalError": outcome["error"].message}
                        ),
                    }
                ]
            )
        if outcome["kind"] == "compaction":
            base_events.append(
                lambda commit: [
                    {
                        "type": "compaction_end",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "reason": compaction_reason(current.task),
                        "status": "completed",
                        "entryId": outcome["resultEntryId"],
                        "endedAt": terminal_compaction_ended_at
                        if terminal_compaction_ended_at is not None
                        else commit.timestamp,
                    }
                ]
            )

        def _events(commit: Any) -> List[Any]:
            collected: List[Any] = []
            for materialize in base_events:
                collected.extend(materialize(commit))
            return collected

        boundary_kind = current.task.boundary.kind
        if boundary_kind == "resume_checkpoint":
            if outcome["kind"] == "branch_summary":
                raise SessionInvariantError(
                    "Run compaction boundary received a branch summary"
                )
            if outcome["kind"] == "compaction" or (
                outcome["kind"] == "declined" and current.task.reason == "threshold"
            ):
                if terminal_tip_id is None:
                    raise SessionInvariantError("Run compaction has no Branch tip")
                continuation = _field(
                    current.task.boundary.resume_after, "continuation", "continuation"
                )
                placement = await plan_boundary_inbox(
                    lane,
                    drive,
                    state,
                    current,
                    reader,
                    terminal_tip_id,
                    outcome["kind"] == "declined" and continuation.kind == "may_finish",
                )
                if (
                    outcome["kind"] == "declined"
                    and placement.trigger_entry_id is None
                    and continuation.kind == "may_finish"
                ):
                    return LaneReturn(
                        result=_Bag(
                            {
                                "kind": "finish_pending",
                                "entry_ids": [entry.id for entry in placement.entries],
                            }
                        )
                    )
                placement_write_index = len(writes)
                writes.extend(placement.writes)
                operation_state: Any
                if placement.trigger_entry_id is not None or continuation.kind == "need_assistant":
                    overflow_recovery_used = (
                        continuation.overflow_recovery_used
                        if placement.trigger_entry_id is None
                        and continuation.kind == "need_assistant"
                        else False
                    )
                    operation_state = assistant_ready_at_boundary(
                        lane,
                        state,
                        current,
                        placement.trigger_entry_id
                        if placement.trigger_entry_id is not None
                        else current.task.boundary.resume_after.trigger_entry_id,
                        overflow_recovery_used,
                    )
                else:
                    resume_after = current.task.boundary.resume_after
                    operation_state = CheckpointOperation(
                        **operation_scope_of(current),
                        continuation=resume_after.continuation,
                        trigger_entry_id=resume_after.trigger_entry_id,
                    )

                def _resume_events(commit: Any) -> List[Any]:
                    return [
                        *(
                            _events(commit)
                            if outcome["kind"] == "compaction"
                            else [
                                {
                                    "type": "compaction_end",
                                    "lane": lane.name,
                                    "runId": drive.operation_id,
                                    "reason": "threshold",
                                    "status": "declined",
                                    "endedAt": commit.timestamp,
                                }
                            ]
                        ),
                        *boundary_placement_events(
                            placement,
                            commit,
                            placement_write_index,
                            lane.name,
                            drive.operation_id,
                        ),
                    ]

                return CommitDecision(
                    writes=writes,
                    operation_state=operation_state,
                    lane={"tipId": placement.tip_id, "inbox": placement.inbox},
                    materialize=lambda _commit: ProcedureResult(kind="continue"),
                    events=_resume_events,
                )
            if state.tip_id is None:
                raise SessionInvariantError("Failed run has no Branch tip")
            error = (
                _operation_error(
                    "compaction_declined", "Overflow compaction was declined"
                )
                if outcome["kind"] == "declined"
                else outcome["error"]
            )
            cleanup = await operation_cleanup_writes(
                reader, drive.operation_id, current, drive.context
            )
            record = operation_result_record(meta, "failed", state.tip_id, error)
            compaction_end = (
                {
                    "type": "compaction_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "reason": compaction_reason(current.task),
                    "status": "declined",
                    "endedAt": record.ended_at,
                }
                if outcome["kind"] == "declined"
                else {
                    "type": "compaction_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "reason": compaction_reason(current.task),
                    "status": "failed",
                    "error": outcome["error"],
                    "endedAt": record.ended_at,
                }
            )
            return FinishDecision(
                writes=[*writes, *cleanup],
                record=record,
                materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
                events=lambda commit: [
                    *_events(commit),
                    compaction_end,
                    {
                        "type": "run_end",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "status": "failed",
                        "error": error,
                        "fromTipId": meta.source_tip_id,
                        "tipId": state.tip_id,
                        "endedAt": record.ended_at,
                    },
                ],
            )
        if boundary_kind == "finish":
            if outcome["kind"] == "branch_summary":
                raise SessionInvariantError(
                    "Compaction finish boundary received a branch summary"
                )
            if outcome["kind"] != "compaction" and state.tip_id is None:
                raise SessionInvariantError("Standalone compaction has no Branch tip")
            error = outcome["error"] if outcome["kind"] == "failed" else None
            status = (
                "declined"
                if outcome["kind"] == "declined"
                else "failed"
                if outcome["kind"] == "failed"
                else "completed"
            )
            cleanup = await operation_cleanup_writes(
                reader, drive.operation_id, current, drive.context
            )
            record = operation_result_record(meta, status, terminal_tip_id, error)
            compaction_end = None
            if outcome["kind"] == "declined":
                compaction_end = {
                    "type": "compaction_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "reason": "manual",
                    "status": "declined",
                    "endedAt": record.ended_at,
                }
            elif outcome["kind"] == "failed":
                compaction_end = {
                    "type": "compaction_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "reason": "manual",
                    "status": "failed",
                    "error": outcome["error"],
                    "endedAt": record.ended_at,
                }

            def _finish_events(commit: Any, end: Any = compaction_end) -> List[Any]:
                return [*_events(commit), *([] if end is None else [end])]

            return FinishDecision(
                writes=[*writes, *cleanup],
                record=record,
                lane={"tipId": terminal_tip_id} if outcome["kind"] == "compaction" else None,
                materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
                events=_finish_events,
            )
        if outcome["kind"] == "compaction":
            raise SessionInvariantError("Navigation boundary received a compaction result")
        error = outcome["error"] if outcome["kind"] == "failed" else None
        status = (
            "declined"
            if outcome["kind"] == "declined"
            else "failed"
            if outcome["kind"] == "failed"
            else "completed"
        )
        cleanup = await operation_cleanup_writes(
            reader, drive.operation_id, current, drive.context
        )
        record = operation_result_record(meta, status, terminal_tip_id, error)
        if outcome["kind"] == "branch_summary":
            navigation_end = {
                "type": "navigation_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "status": "completed",
                "fromTipId": meta.source_tip_id,
                "tipId": terminal_tip_id,
                "endedAt": record.ended_at,
            }
        elif outcome["kind"] == "declined":
            navigation_end = {
                "type": "navigation_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "status": "declined",
                "fromTipId": meta.source_tip_id,
                "tipId": terminal_tip_id,
                "endedAt": record.ended_at,
            }
        else:
            navigation_end = {
                "type": "navigation_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "status": "failed",
                "error": outcome["error"],
                "fromTipId": meta.source_tip_id,
                "tipId": terminal_tip_id,
                "endedAt": record.ended_at,
            }
        return FinishDecision(
            writes=[*writes, *cleanup],
            record=record,
            lane={"tipId": terminal_tip_id} if outcome["kind"] == "branch_summary" else None,
            materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
            events=lambda commit: [*_events(commit), navigation_end],
        )

    published = await lane.continue_operation(capability, _plan, drive.context)
    if published.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    if published.value.kind != "finish_pending":
        return published.value
    boundary = capability.task.boundary
    if (
        boundary.kind != "resume_checkpoint"
        or boundary.resume_after.continuation.kind != "may_finish"
    ):
        raise SessionInvariantError(
            "Structural finish mediation requires a resumable finish boundary"
        )
    return await finish_run_boundary(
        lane,
        drive,
        capability,
        boundary.resume_after.continuation,
        published.value.entry_ids,
        [
            {
                "type": "compaction_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "reason": "threshold",
                "status": "declined",
                "endedAt": _now(),
            }
        ],
    )


async def _publish_structural_ready(lane: Any, drive: Any, deciding: Any) -> ProcedureResult:
    result_entry_id = lane.session.id_generator.next()

    def _plan(state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        operation_state = SummaryReadyOperation(
            **operation_scope_of(current),
            task=current.task,
            summary_context=summary_context(lane, result_entry_id, state.configuration),
            next_attempt=1,
        )
        return CommitDecision(
            writes=[],
            operation_state=operation_state,
            materialize=lambda _commit: ProcedureResult(kind="continue"),
        )

    published = await lane.continue_operation(deciding, _plan, drive.context)
    return ProcedureResult(kind="continue") if published.kind == "cancel_requested" else published.value


async def run_structural_decision(lane: Any, drive: Any, deciding: Any) -> ProcedureResult:
    """Consume one durable structural preparation and decision hook."""
    preparation = await _read_structural_preparation(lane, drive, deciding)
    if preparation.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    if deciding.task.boundary.kind == "commit_navigation":
        if not hasattr(preparation.value, "messages"):
            raise SessionInvariantError(
                "Navigation task has invalid durable preparation"
            )
        event: dict = {
            "lane": lane.name,
            "runId": drive.operation_id,
            "targetId": deciding.task.boundary.target_id,
            "preparation": preparation.value,
        }
        if deciding.task.custom_instructions is not None:
            event["customInstructions"] = deciding.task.custom_instructions
        hook = await lane.hooks.run_with_gate(
            "before_navigation", event, drive.gate, drive.context
        )
        if _opt(hook, "decline") is True:
            return await publish_structural_outcome(lane, drive, deciding, {"kind": "declined"})
        summary = _opt(hook, "summary")
        if summary is not None:
            return await publish_structural_outcome(
                lane,
                drive,
                deciding,
                {
                    "kind": "branch_summary",
                    "resultEntryId": lane.session.id_generator.next(),
                    "result": summary,
                    "fromHook": True,
                },
            )
        return await _publish_structural_ready(lane, drive, deciding)

    if not hasattr(preparation.value, "messages_to_summarize"):
        raise SessionInvariantError("Compaction task has invalid durable preparation")
    event = {
        "lane": lane.name,
        "runId": drive.operation_id,
        "reason": compaction_reason(deciding.task),
        "preparation": preparation.value,
    }
    if deciding.task.custom_instructions is not None:
        event["customInstructions"] = deciding.task.custom_instructions
    hook = await lane.hooks.run_with_gate(
        "before_compaction", event, drive.gate, drive.context
    )
    if _opt(hook, "decline") is True:
        return await publish_structural_outcome(lane, drive, deciding, {"kind": "declined"})
    compaction = _opt(hook, "compaction")
    if compaction is not None:
        return await publish_structural_outcome(
            lane,
            drive,
            deciding,
            {
                "kind": "compaction",
                "resultEntryId": lane.session.id_generator.next(),
                "result": compaction,
                "fromHook": True,
            },
        )
    return await _publish_structural_ready(lane, drive, deciding)


def _effect_pending_from_ready(ready: Any) -> SummaryEffectPendingOperation:
    return SummaryEffectPendingOperation(
        **operation_scope_of(ready),
        task=ready.task,
        summary_context=ready.summary_context,
        attempt=ready.next_attempt,
        usage_ids=[],
    )


def _retry_wait_from_effect(effect: Any, error_message: str) -> SummaryRetryWaitOperation:
    return SummaryRetryWaitOperation(
        **operation_scope_of(effect),
        task=effect.task,
        summary_context=effect.summary_context,
        next_attempt=effect.attempt + 1,
        not_before=retry_not_before(effect.summary_context.retry_policy, effect.attempt),
        error_message=error_message,
    )


def _ready_from_retry_wait(retry: Any) -> SummaryReadyOperation:
    return SummaryReadyOperation(
        **operation_scope_of(retry),
        task=retry.task,
        summary_context=retry.summary_context,
        next_attempt=retry.next_attempt,
    )


async def _publish_attempt_intent(lane: Any, drive: Any, ready: Any) -> ContinueOperationResult:
    def _plan(_state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        effect_pending = _effect_pending_from_ready(current)
        return CommitDecision(
            writes=[],
            operation_state=effect_pending,
            materialize=lambda _commit: effect_pending,
        )

    return await lane.continue_operation(ready, _plan, drive.context)


async def _publish_nested_request_intent(
    lane: Any, drive: Any, effect: Any, index: int, usage_id: str
) -> ContinueOperationResult:
    def _plan(_state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        import copy as _copy

        next_state = _copy.copy(current)
        next_state.request = {"index": index, "usageId": usage_id}
        return CommitDecision(
            writes=[],
            operation_state=next_state,
            materialize=lambda _commit: next_state,
        )

    return await lane.continue_operation(effect, _plan, drive.context)


async def _publish_nested_request_outcome(
    lane: Any, drive: Any, effect: Any, usage_id: str, response: AssistantMessage
) -> None:
    def _plan(_state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        import copy as _copy

        next_state = _copy.copy(current)
        next_state.usage_ids = [*current.usage_ids, usage_id]
        next_state.request = None
        row = {"id": usage_id, "usage": response.usage, "adjustment": False}
        return CommitDecision(
            writes=[insert_usage(_usage_row(row))],
            operation_state=next_state,
            materialize=lambda _commit: None,
            events=lambda commit, r=row: [_usage_event(r, 0, commit, lane.name)],
        )

    await lane.settle_operation(effect, _plan, drive.context)


def _request_stream_options(
    options: Any, stream_options: Any, context: Any, on_payload: Any
) -> dict:
    return {
        **options,
        "transport": _opt(stream_options, "transport"),
        "timeoutMs": _opt(stream_options, "timeout_ms", "timeoutMs"),
        "maxRetries": _opt(stream_options, "max_retries", "maxRetries"),
        "maxRetryDelayMs": _opt(stream_options, "max_retry_delay_ms", "maxRetryDelayMs"),
        "headers": _opt(stream_options, "headers"),
        "metadata": _opt(stream_options, "metadata"),
        "cacheRetention": "none",
        "deferred": False,
        "signal": context.abort_signal,
        "telemetryContext": get_telemetry_context(context),
        "onPayload": on_payload,
    }


async def _perform_structural_attempt(
    lane: Any, drive: Any, effect: Any, model: Any, preparation: Any
) -> dict:
    request_index = 0
    last_response: Optional[AssistantMessage] = None

    async def _request(ai_context: Any, options: Any, request_context: Any) -> AssistantMessage:
        nonlocal request_index, last_response
        base_options = _replace_field(effect.summary_context.stream_options, "deferred", False)
        try:
            before_request = await lane.hooks.run_with_gate(
                "before_request",
                {
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "model": model,
                    "step": summary_kind(effect.task),
                    "attempt": effect.attempt,
                    "streamOptions": base_options,
                },
                drive.gate,
                request_context,
            )
        except AbortRequested as error:
            await error.cancellation
            raise StructuralCancelled() from error
        patch = None if before_request is None else _opt(before_request, "stream_options", "streamOptions")
        resolved = (
            base_options
            if patch is None
            else apply_stream_options_patch(base_options, patch)
        )
        resolved = _replace_field(resolved, "deferred", False)
        usage_id = lane.session.id_generator.next()
        intent = await _publish_nested_request_intent(
            lane, drive, effect, request_index, usage_id
        )
        request_index += 1
        if intent.kind == "cancel_requested":
            raise StructuralCancelled()
        admitted_context = with_abort_signal(drive.gate.signal, request_context)

        async def _payload_hook(payload: Any, request_model: Any) -> Any:
            hook = await lane.hooks.run_with_gate(
                "before_payload",
                {
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "model": request_model,
                    "payload": payload,
                },
                drive.gate,
                admitted_context,
            )
            return None if hook is None else _opt(hook, "payload")

        async def _complete() -> Any:
            return await lane.models.complete_simple(
                model,
                ai_context,
                _request_stream_options(options, resolved, admitted_context, _payload_hook),
            )

        try:
            response = await drive.gate.admit(_complete)
        except AbortRequested as error:
            await error.cancellation
            raise StructuralCancelled() from error
        last_response = response
        await _publish_nested_request_outcome(lane, drive, intent.value, usage_id, response)
        return response

    try:
        if summary_kind(effect.task) == "compaction":
            if not hasattr(preparation, "messages_to_summarize"):
                raise SessionInvariantError(
                    "Compaction summary has invalid durable preparation"
                )
            result = await compact_with_request(
                preparation,
                CompactGenerationOptions(
                    model=model,
                    custom_instructions=effect.task.custom_instructions,
                    thinking_level=effect.summary_context.configuration.thinking_level,
                ),
                _request,
                drive.context,
            )
            if not result.ok:
                return {
                    "kind": "error",
                    "error": _operation_error(result.error.code, result.error.message),
                    "retryable": last_response is not None
                    and is_retryable_assistant_error(last_response),
                }
            return {
                "kind": "compaction",
                "result": result.value,
                "retryable": last_response is not None
                and is_retryable_assistant_error(last_response),
            }

        if not hasattr(preparation, "messages"):
            raise SessionInvariantError("Branch summary has invalid durable preparation")
        from ...compaction.branch_summarization import PreparedBranchSummaryOptions

        result = await generate_branch_summary_with_request(
            preparation,
            PreparedBranchSummaryOptions(
                custom_instructions=effect.task.custom_instructions
            ),
            _request,
            drive.context,
        )
        if not result.ok:
            return {
                "kind": "error",
                "error": _operation_error(result.error.code, result.error.message),
                "retryable": last_response is not None
                and is_retryable_assistant_error(last_response),
            }
        return {
            "kind": "branch_summary",
            "result": result.value,
            "retryable": last_response is not None
            and is_retryable_assistant_error(last_response),
        }
    except StructuralCancelled:
        return {"kind": "cancel_requested"}


async def _read_attempt_preparation(lane: Any, drive: Any, ready: Any) -> ContinueOperationResult:
    async def _plan(_state: Any, current: Any, _meta: Any, reader: Any) -> Any:
        expected = summary_kind(current.task)
        stored = await reader.get_value(
            operation_preparation(drive.operation_id, current.task.task_id), drive.context
        )
        if stored is None or _field(stored.value, "kind", "kind") != expected:
            raise SessionInvariantError(
                f"Structural task {current.task.task_id} has invalid durable preparation"
            )
        return LaneReturn(
            result=(
                compaction_preparation(stored.value)
                if _field(stored.value, "kind", "kind") == "compaction"
                else branch_preparation(stored.value)
            )
        )

    return await lane.continue_operation(ready, _plan, drive.context)


async def _publish_attempt_result(
    lane: Any, drive: Any, effect: Any, result: dict
) -> ProcedureResult:
    if result["kind"] == "cancel_requested":
        return ProcedureResult(kind="continue")
    if result["kind"] == "compaction":
        return await publish_structural_outcome(
            lane,
            drive,
            effect,
            {
                "kind": "compaction",
                "resultEntryId": effect.summary_context.result_entry_id,
                "result": result["result"],
                "fromHook": False,
            },
        )
    if result["kind"] == "branch_summary":
        return await publish_structural_outcome(
            lane,
            drive,
            effect,
            {
                "kind": "branch_summary",
                "resultEntryId": effect.summary_context.result_entry_id,
                "result": result["result"],
                "fromHook": False,
            },
        )
    if result["error"].code == "aborted" and lane.state.operation.state.control.status == "running":
        raise SessionInvariantError(
            "Structural provider response is aborted while durable control is running"
        )
    if (
        result["retryable"]
        and effect.attempt < effect.summary_context.retry_policy.max_attempts
    ):
        retry_wait = _retry_wait_from_effect(effect, result["error"].message)
        published = await lane.continue_operation(
            effect,
            lambda _state, _current, rw=retry_wait: CommitDecision(
                writes=[],
                operation_state=rw,
                materialize=lambda _commit: ProcedureResult(kind="continue"),
                events=lambda _commit: [
                    {
                        "type": "retry_scheduled",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "step": effect.task.task_id,
                        "attempt": rw.next_attempt,
                        "maxAttempts": effect.summary_context.retry_policy.max_attempts,
                        "delayMs": retry_delay_ms(
                            effect.summary_context.retry_policy, effect.attempt
                        ),
                        "notBefore": rw.not_before,
                        "errorMessage": result["error"].message,
                    }
                ],
            ),
            drive.context,
        )
        return (
            ProcedureResult(kind="continue")
            if published.kind == "cancel_requested"
            else published.value
        )
    return await publish_structural_outcome(
        lane, drive, effect, {"kind": "failed", "error": result["error"]}
    )


async def run_structural_generation(lane: Any, drive: Any, ready: Any) -> ProcedureResult:
    """Execute one ready structural generation attempt."""
    preparation = await _read_attempt_preparation(lane, drive, ready)
    if preparation.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    identity = ready.summary_context.configuration.model
    model = lane.models.get_model(
        _opt(identity, "provider"), _opt(identity, "model_id", "modelId")
    )
    if model is None:
        return await publish_structural_outcome(
            lane,
            drive,
            ready,
            {
                "kind": "failed",
                "error": _operation_error(
                    "model_unavailable",
                    "The configured model is unavailable in this process",
                    identity,
                ),
            },
        )
    intent = await _publish_attempt_intent(lane, drive, ready)
    if intent.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    result = await _perform_structural_attempt(
        lane, drive, intent.value, model, preparation.value
    )
    return await _publish_attempt_result(lane, drive, intent.value, result)


async def run_structural_retry_wait(lane: Any, drive: Any, retry: Any) -> ProcedureResult:
    """Consume one structural retry wait without starting a provider effect."""
    if _now() < retry.not_before:
        if not drive.wait_for_retry:
            return ProcedureResult(
                kind="waiting",
                outcome={
                    "kind": "waiting",
                    "operationId": drive.operation_id,
                    "reason": "retry",
                    "notBefore": retry.not_before,
                },
            )
        await drive.gate.admit(lambda: wait_until(retry.not_before, drive.gate.signal))

    def _plan(_state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        ready = _ready_from_retry_wait(current)
        return CommitDecision(
            writes=[],
            operation_state=ready,
            materialize=lambda _commit: ProcedureResult(kind="continue"),
            events=lambda _commit: [
                {
                    "type": "retry_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "step": current.task.task_id,
                    "attempt": current.next_attempt,
                }
            ],
        )

    published = await lane.continue_operation(retry, _plan, drive.context)
    return (
        ProcedureResult(kind="continue")
        if published.kind == "cancel_requested"
        else published.value
    )


async def recover_structural_generation(lane: Any, drive: Any, effect: Any) -> ProcedureResult:
    """Convert an orphaned structural attempt into a fresh attempt or terminal failure."""
    error = _operation_error(
        "structural_interrupted",
        "Structural summary attempt was interrupted and its external outcome is unknown",
    )
    if effect.attempt >= effect.summary_context.retry_policy.max_attempts:
        return await publish_structural_outcome(
            lane, drive, effect, {"kind": "failed", "error": error}
        )
    retry_wait = _retry_wait_from_effect(effect, error.message)
    published = await lane.continue_operation(
        effect,
        lambda _state, _current, rw=retry_wait: CommitDecision(
            writes=[],
            operation_state=rw,
            materialize=lambda _commit: ProcedureResult(kind="continue"),
            events=lambda _commit: [
                {
                    "type": "retry_scheduled",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "step": effect.task.task_id,
                    "attempt": rw.next_attempt,
                    "maxAttempts": effect.summary_context.retry_policy.max_attempts,
                    "delayMs": retry_delay_ms(
                        effect.summary_context.retry_policy, effect.attempt
                    ),
                    "notBefore": rw.not_before,
                    "errorMessage": error.message,
                    "recovery": True,
                }
            ],
        ),
        drive.context,
    )
    return (
        ProcedureResult(kind="continue")
        if published.kind == "cancel_requested"
        else published.value
    )


def report_operations() -> List[str]:
    """Every structural procedure this module owns, for coverage assertions."""
    return [
        "run_structural_decision",
        "run_structural_generation",
        "run_structural_retry_wait",
        "recover_structural_generation",
        "prepare_compaction_threshold",
        "prepare_overflow_compaction",
        "commit_navigation",
        "publish_structural_outcome",
    ]


def _usage_row(row: dict) -> UsageRow:
    return UsageRow(
        id=row["id"],
        usage=row["usage"],
        adjustment=row["adjustment"],
    )


def _replace_field(options: Any, name: str, value: Any) -> Any:
    """Return a copy of an options object with one field overridden."""
    if options is None:
        return None
    if isinstance(options, dict):
        return {**options, name: value}
    return options.__class__(**{**vars(options), name: value})


def _now() -> int:
    return int(time.time() * 1000)


class _Bag:
    """Attribute view over a plain mapping."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping, a bag, or an attribute."""
    if value is None:
        return None
    if isinstance(value, _Bag):
        value = value._values
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


# ---------------------------------------------------------------------------
# Preparation and navigation commit
# ---------------------------------------------------------------------------


async def prepare_compaction_threshold(lane: Any, drive: Any, checkpoint: Any) -> Any:
    """Prepare threshold compaction only when no newer compaction already guards this trigger."""
    from ...compaction.compaction import prepare_compaction, should_compact

    settings = checkpoint.settings.compaction
    identity = lane.state.configuration.model
    model = lane.models.get_model(
        _opt(identity, "provider"), _opt(identity, "model_id", "modelId")
    )
    if settings is None or not settings.enabled or model is None:
        return ContinueOperationResult(kind="result", value=None)
    path = await read_bounded_entries(lane, drive, checkpoint)
    if path.kind == "cancel_requested":
        return path
    entries = path.value
    trigger_index = next(
        (
            index
            for index, entry in enumerate(entries)
            if entry.id == checkpoint.trigger_entry_id
        ),
        -1,
    )
    newest_compaction_index = -1
    for index in range(len(entries) - 1, -1, -1):
        if getattr(entries[index], "type", None) == "compaction":
            newest_compaction_index = index
            break
    if newest_compaction_index != -1 and newest_compaction_index >= trigger_index:
        return ContinueOperationResult(kind="result", value=None)
    if trigger_index == -1:
        raise SessionInvariantError(
            f"Checkpoint trigger {checkpoint.trigger_entry_id} is missing from its Branch"
        )
    prepared = prepare_compaction(entries, settings)
    if not prepared.ok:
        raise prepared.error
    if prepared.value is None or not should_compact(
        prepared.value.tokens_before, model.context_window, settings
    ):
        return ContinueOperationResult(kind="result", value=None)
    return ContinueOperationResult(
        kind="result",
        value={
            "taskId": lane.session.id_generator.next(),
            "preparation": durable_compaction_preparation(prepared.value),
        },
    )


async def prepare_overflow_compaction(lane: Any, drive: Any, generation: Any) -> Any:
    """Prepare one overflow compaction before the response settlement transaction."""
    from ...compaction.compaction import prepare_compaction

    if generation.generation_context.overflow_recovery_used:
        return None
    path = await read_bounded_entries(lane, drive, generation)
    if path.kind == "cancel_requested":
        return None
    prepared = prepare_compaction(path.value, generation.settings.compaction)
    if not prepared.ok:
        raise prepared.error
    if prepared.value is None:
        return None
    return {
        "taskId": lane.session.id_generator.next(),
        "preparation": durable_compaction_preparation(prepared.value),
    }


async def commit_navigation(lane: Any, drive: Any, navigation: Any) -> ProcedureResult:
    """Atomically move an unsummarized navigation and finish its operation."""

    async def _plan(_state: Any, current: Any, meta: Any, reader: Any) -> Any:
        if current.target_id is not None and current.target_id not in (
            await reader.get_entries([current.target_id], drive.context)
        ):
            raise SessionInvariantError(f"Navigation target {current.target_id} is missing")
        if current.target_id == meta.source_tip_id:
            raise SessionInvariantError(
                "Navigation target must differ from its source tip"
            )
        if current.target_id is None and current.label is not None:
            raise SessionInvariantError("Root navigation cannot set a label")
        writes: List[Any] = [set_value(branch_tip(lane.name), current.target_id)]
        if current.label is not None and current.target_id is not None:
            writes.append(set_value(entry_label(current.target_id), current.label))
        cleanup = await operation_cleanup_writes(
            reader, drive.operation_id, current, drive.context
        )
        record = operation_result_record(meta, "completed", current.target_id)
        return FinishDecision(
            writes=[*writes, *cleanup],
            record=record,
            lane={"tipId": current.target_id},
            materialize=lambda _commit: ProcedureResult(kind="settled", outcome=record),
            events=lambda _commit: [
                {
                    "type": "navigation_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "status": "completed",
                    "fromTipId": meta.source_tip_id,
                    "tipId": current.target_id,
                    "endedAt": record.ended_at,
                }
            ],
        )

    result = await lane.continue_operation(navigation, _plan, drive.context)
    if result.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    return result.value
