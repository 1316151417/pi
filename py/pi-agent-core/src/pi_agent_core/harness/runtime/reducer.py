"""Lane snapshot reduction ported from ``harness/runtime/reducer.ts``."""

from __future__ import annotations

from typing import Any, List, Optional

from ..events import HarnessEvent, LaneOperationState, LaneSnapshot, LaneSnapshotTool
from ..session.types import OperationResultRecord

__all__ = ["LaneSnapshotReduction", "reduce_lane_snapshot"]

#: ``"rebase"`` means the caller must take a fresh snapshot before continuing.
LaneSnapshotReduction = Optional[str]


def _upsert_tool(operation: LaneOperationState, tool: LaneSnapshotTool) -> None:
    index = next(
        (
            i
            for i, candidate in enumerate(operation.running_tools)
            if candidate.tool_call_id == tool.tool_call_id
        ),
        -1,
    )
    if index == -1:
        operation.running_tools.append(tool)
    else:
        operation.running_tools[index] = tool


def _matching_operation(snapshot: LaneSnapshot, operation_id: Optional[str]) -> Optional[LaneOperationState]:
    operation = snapshot.operation
    if operation is None:
        return None
    return operation if operation.id == operation_id else None


def reduce_lane_snapshot(snapshot: LaneSnapshot, event: Any) -> LaneSnapshotReduction:
    """Apply one harness event to a mutable lane snapshot.

    Navigation completion requires a fresh snapshot, so it returns ``"rebase"``.
    """
    # Events carrying a different lane are ignored, except global usage rows.
    event_lane = getattr(event, "lane", None)
    if event_lane is not None and event_lane != snapshot.lane and getattr(event, "type", None) != "usage":
        return None

    event_type = getattr(event, "type", None)

    if event_type == "run_start":
        snapshot.operation = LaneOperationState(
            id=event.run_id,
            kind="run",
            started_at=event.started_at,
            from_tip_id=snapshot.tip_id,
            status="open",
            running_tools=[],
        )
        return None

    if event_type == "compaction_start":
        if snapshot.operation is not None:
            return None
        snapshot.operation = LaneOperationState(
            id=event.run_id,
            kind="compaction",
            started_at=event.started_at,
            from_tip_id=snapshot.tip_id,
            status="open",
            running_tools=[],
        )
        return None

    if event_type == "navigation_start":
        snapshot.operation = LaneOperationState(
            id=event.run_id,
            kind="navigation",
            started_at=event.started_at,
            from_tip_id=snapshot.tip_id,
            status="open",
            running_tools=[],
        )
        return None

    if event_type == "operation_abort":
        operation = _matching_operation(snapshot, event.operation_id)
        if operation is not None:
            operation.status = "aborting"
        return None

    if event_type == "run_resume":
        operation = _matching_operation(snapshot, event.run_id)
        if operation is not None:
            operation.deferred = None
        return None

    if event_type == "run_suspend":
        operation = _matching_operation(snapshot, event.run_id)
        if operation is None:
            return None
        operation.streaming_message = None
        operation.deferred = {"handle": event.deferred, "poll": event.poll}
        return None

    if event_type == "retry_scheduled":
        operation = _matching_operation(snapshot, event.run_id)
        if operation is not None:
            operation.retry = {
                "attempt": event.attempt,
                "maxAttempts": event.max_attempts,
                "nextAttemptAt": event.not_before,
            }
        return None

    if event_type in ("retry_start", "retry_end"):
        operation = _matching_operation(snapshot, event.run_id)
        if operation is not None:
            operation.retry = None
        return None

    if event_type == "message_start":
        message = event.message
        if (
            event.run_id is None
            or getattr(message, "role", None) != "assistant"
            or getattr(message, "stop_reason", None) != "pending"
        ):
            return None
        operation = _matching_operation(snapshot, event.run_id)
        if operation is not None:
            operation.streaming_message = message
        return None

    if event_type == "message_update":
        if getattr(event.message, "role", None) != "assistant":
            return None
        operation = _matching_operation(snapshot, event.run_id)
        if operation is not None:
            operation.streaming_message = event.message
        return None

    if event_type == "message_end":
        if event.run_id is None:
            return None
        operation = _matching_operation(snapshot, event.run_id)
        if operation is not None:
            operation.streaming_message = None
        return None

    if event_type == "tool_start":
        operation = _matching_operation(snapshot, event.run_id)
        if operation is None:
            return None
        _upsert_tool(
            operation,
            LaneSnapshotTool(
                status="running",
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                args=event.args,
            ),
        )
        return None

    if event_type == "tool_update":
        operation = _matching_operation(snapshot, event.run_id)
        tool = next(
            (
                candidate
                for candidate in (operation.running_tools if operation else [])
                if candidate.tool_call_id == event.tool_call_id
            ),
            None,
        )
        if tool is not None and tool.status == "running":
            tool.result = event.partial_result
        return None

    if event_type == "tool_end":
        operation = _matching_operation(snapshot, event.run_id)
        if operation is None:
            return None
        index = next(
            (
                i
                for i, candidate in enumerate(operation.running_tools)
                if candidate.tool_call_id == event.tool_call_id
            ),
            -1,
        )
        if index == -1:
            return None
        current = operation.running_tools[index]
        operation.running_tools[index] = LaneSnapshotTool(
            status="settled",
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            args=current.args,
            result=event.result,
            is_error=event.is_error,
        )
        return None

    if event_type == "entry_added":
        entry = event.entry
        if getattr(entry, "type", None) == "message" and getattr(entry.message, "role", None) == "toolResult":
            operation = snapshot.operation
            if operation is not None:
                tool_call_id = entry.message.tool_call_id
                index = next(
                    (
                        i
                        for i, candidate in enumerate(operation.running_tools)
                        if candidate.tool_call_id == tool_call_id
                    ),
                    -1,
                )
                if index != -1:
                    operation.running_tools.pop(index)
        if getattr(entry, "type", None) == "compaction":
            snapshot.transcript.clear()
            snapshot.transcript.append(entry)
        else:
            snapshot.transcript.append(entry)
        snapshot.tip_id = entry.id
        if getattr(entry, "type", None) == "message" and snapshot.stats is not None:
            snapshot.stats.message_count += 1
        return None

    if event_type == "queue_update":
        snapshot.queues = list(event.queues or [])
        return None

    if event_type == "usage":
        if snapshot.stats is not None:
            snapshot.stats.usage = event.totals
        return None

    if event_type == "config_update":
        if getattr(event, "lane", None) != snapshot.lane:
            return None
        configuration = snapshot.configuration
        if configuration is None:
            return None
        if event.property == "model":
            configuration.model = event.value
        elif event.property == "thinkingLevel":
            configuration.thinking_level = event.value
        elif event.property == "activeTools":
            configuration.active_tool_names = list(event.value or [])
        return None

    if event_type == "run_end":
        operation = _matching_operation(snapshot, event.run_id)
        if operation is None or operation.kind != "run":
            return None
        record = OperationResultRecord(
            operation_id=event.run_id,
            kind="run",
            status=event.status,
            error=event.error if event.status == "failed" else None,
            from_tip_id=event.from_tip_id,
            tip_id=event.tip_id,
            started_at=operation.started_at,
            ended_at=event.ended_at,
        )
        snapshot.last_result = record
        snapshot.operation = None
        snapshot.tip_id = event.tip_id
        return None

    if event_type == "compaction_end":
        operation = _matching_operation(snapshot, event.run_id)
        if operation is None or operation.kind != "compaction":
            return None
        record = OperationResultRecord(
            operation_id=event.run_id,
            kind="compaction",
            status=event.status,
            error=event.error if event.status == "failed" else None,
            from_tip_id=operation.from_tip_id,
            tip_id=snapshot.tip_id,
            started_at=operation.started_at,
            ended_at=event.ended_at,
        )
        snapshot.last_result = record
        snapshot.operation = None
        return None

    if event_type == "navigation_end":
        return "rebase"

    if event_type == "fault":
        snapshot.faulted = True
        return None

    # handler_error, turn_start, turn_end, value_update, lane_created: no snapshot effect.
    return None
