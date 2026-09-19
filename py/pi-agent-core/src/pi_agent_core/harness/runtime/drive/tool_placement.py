"""Tool result placement ported from ``harness/runtime/drive/tool-placement.ts``.

Reads the tool batch's assistant source, materializes outcome-ready tool
results into the Branch in source order, and reports when the batch completes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...result import Result  # noqa: F401  (keeps the error import graph stable)
from ...session.commit import insert_entry, insert_usage
from ...session.session import SessionInvariantError
from ...session.types import (
    CheckpointOperation,
    Continuation,
    MessageEntry,
    PendingEntry,
    ToolBatch,
    UsageRow,
    operation_scope_of,
)
from ...session.values import (
    branch_tip,
    delete_value,
    operation_tool_args_prefix,
    pending_entry,
    set_value,
)
from ..types import CommitDecision, LaneReturn
from .retry import retry_not_before  # noqa: F401  (keeps the drive import graph stable)

__all__ = [
    "ToolBatchSource",
    "PlacementItem",
    "PlacementRead",
    "is_tool_result_message",
    "read_tool_batch_source",
    "tool_call_for",
    "with_tool_batch",
    "materialize_ready",
]


@dataclass
class ToolBatchSource:
    """The assistant message a tool batch belongs to, plus its tool-call blocks."""

    assistant: Any
    calls: Dict[int, Any] = field(default_factory=dict)


@dataclass
class PlacementItem:
    """One outcome-ready tool call paired with its staged result message."""

    call: Any
    message: Any


@dataclass
class PlacementRead:
    """The materializable prefix of one tool batch."""

    items: List[PlacementItem] = field(default_factory=list)
    turn_results: Optional[List[Any]] = None


def is_tool_result_message(value: Any) -> bool:
    """Whether a value is a tool-result message rather than an entry payload."""
    return getattr(value, "role", None) == "toolResult"


async def read_tool_batch_source(lane: Any, drive: Any, batch: ToolBatch) -> ToolBatchSource:
    """The assistant message and tool-call blocks a batch indexes into."""

    async def _read(_state: Any, reader: Any) -> Any:
        entry = (await reader.get_entries([batch.assistant_entry_id], drive.context)).get(
            batch.assistant_entry_id
        )
        message = getattr(entry, "message", None)
        if (
            getattr(entry, "type", None) != "message"
            or getattr(message, "role", None) != "assistant"
        ):
            raise SessionInvariantError("Tool batch assistant entry is invalid")
        calls: Dict[int, Any] = {}
        for call in batch.calls:
            block = message.content[call.source_index]
            if getattr(block, "type", None) != "toolCall":
                raise SessionInvariantError(
                    f"Tool call source index {call.source_index} does not name a tool-call block"
                )
            calls[call.source_index] = block
        return LaneReturn(result=ToolBatchSource(assistant=message, calls=calls))

    return await lane.command(_read, drive.context)


def tool_call_for(sources: ToolBatchSource, call: Any) -> Any:
    """The source tool-call block one durable call indexes."""
    source = sources.calls.get(call.source_index)
    if source is None:
        raise SessionInvariantError(
            f"Tool call source index {call.source_index} is invalid"
        )
    return source


def with_tool_batch(run: Any, batch: ToolBatch) -> Any:
    """The successor tools leaf carrying an updated batch."""
    from ...session.types import ToolsOperation

    return ToolsOperation(**operation_scope_of(run), batch=batch)


async def _read_placement(
    lane: Any, drive: Any, sources: ToolBatchSource
) -> Optional[PlacementRead]:
    async def _read(state: Any, reader: Any) -> Any:
        operation = state.operation
        if operation is None or operation.state.at != "tools":
            return LaneReturn(result=None)
        current = operation.state.batch
        first = next(
            (
                index
                for index, call in enumerate(current.calls)
                if call.status != "completed"
            ),
            -1,
        )
        if first == -1:
            return LaneReturn(result=None)
        ready: List[Any] = []
        while first < len(current.calls):
            call = current.calls[first]
            if call.status != "outcome_ready":
                break
            ready.append(call)
            first += 1
        if not ready:
            return LaneReturn(result=None)

        items: List[PlacementItem] = []
        for call in ready:
            stored = await reader.get_value(
                pending_entry(call.result_entry_id), drive.context
            )
            pending = None if stored is None else PendingEntry.from_value(stored.value)
            payload = None if pending is None else pending.payload
            if (
                pending is None
                or pending.type != "message"
                or not is_tool_result_message(payload)
            ):
                raise SessionInvariantError(
                    f"Tool call {call.result_entry_id} is missing its staged result"
                )
            source = tool_call_for(sources, call)
            if (
                getattr(payload, "tool_call_id", None) != source.id
                or getattr(payload, "tool_name", None) != source.name
            ):
                raise SessionInvariantError(
                    f"Tool call {call.result_entry_id} has a mismatched staged result"
                )
            items.append(PlacementItem(call=call, message=payload))

        turn_results: Optional[List[Any]] = None
        if first == len(current.calls):
            placed_ids = [
                call.result_entry_id
                for call in current.calls
                if call.status == "completed"
            ]
            placed = await reader.get_entries(placed_ids, drive.context)
            staged = {item.call.result_entry_id: item.message for item in items}
            turn_results = []
            for call in current.calls:
                message = staged.get(call.result_entry_id)
                if message is None:
                    entry = placed.get(call.result_entry_id)
                    entry_message = getattr(entry, "message", None)
                    if (
                        getattr(entry, "type", None) == "message"
                        and is_tool_result_message(entry_message)
                    ):
                        message = entry_message
                if message is None:
                    raise SessionInvariantError(
                        f"Completed tool call {call.result_entry_id} is missing its result entry"
                    )
                turn_results.append(message)
        return LaneReturn(result=PlacementRead(items=items, turn_results=turn_results))

    return await lane.command(_read, drive.context)


async def _commit_placement(
    lane: Any, drive: Any, capability: Any, read: PlacementRead
) -> bool:
    usage_ids = [
        None
        if getattr(item.message, "usage", None) is None
        else lane.session.id_generator.next()
        for item in read.items
    ]

    async def _plan(state: Any, run: Any, _meta: Any, reader: Any) -> Any:
        current = run.batch
        writes: List[Any] = []
        event_entries: List[dict] = []
        event_usage: List[dict] = []
        parent_id = state.tip_id
        for index, item in enumerate(read.items):
            entry = MessageEntry(
                id=item.call.result_entry_id,
                parent_id=parent_id,
                message=item.message,
                terminate=True if item.call.terminate else None,
            )
            event_entries.append({"entry": entry, "seq_index": len(writes)})
            writes.append(insert_entry(entry))
            writes.append(delete_value(pending_entry(item.call.result_entry_id)))
            usage_id = usage_ids[index]
            if usage_id is not None and getattr(item.message, "usage", None) is not None:
                row = UsageRow(
                    id=usage_id,
                    usage=item.message.usage,
                    entry_id=item.call.result_entry_id,
                    adjustment=False,
                )
                event_usage.append({"row": row, "seq_index": len(writes)})
                writes.append(insert_usage(row))
            parent_id = item.call.result_entry_id

        completed_calls = []
        for call in current.calls:
            item = next(
                (
                    candidate
                    for candidate in read.items
                    if candidate.call.source_index == call.source_index
                    and candidate.call.result_entry_id == call.result_entry_id
                ),
                None,
            )
            if item is None:
                completed_calls.append(call)
            else:
                completed_calls.append(
                    _tool_call(
                        status="completed",
                        source_index=call.source_index,
                        result_entry_id=call.result_entry_id,
                        terminate=item.call.terminate,
                    )
                )
        complete = all(call.status == "completed" for call in completed_calls)
        writes.append(set_value(branch_tip(lane.name), parent_id))

        next_run: Any
        if complete:
            all_terminate = all(
                call.status == "completed" and bool(call.terminate)
                for call in completed_calls
            )
            next_run = CheckpointOperation(
                **operation_scope_of(run),
                continuation=(
                    Continuation(kind="may_finish", include_final_assistant=False)
                    if all_terminate
                    else Continuation(kind="need_assistant", overflow_recovery_used=False)
                ),
                trigger_entry_id=parent_id,
            )
            args = await reader.scan_values(
                operation_tool_args_prefix(drive.operation_id, capability.batch.turn_id),
                drive.context,
            )
            writes.extend(delete_value(stored.address) for stored in args)
        else:
            next_run = with_tool_batch(
                run,
                ToolBatch(
                    assistant_entry_id=current.assistant_entry_id,
                    configuration=current.configuration,
                    turn_id=current.turn_id,
                    calls=completed_calls,
                ),
            )

        def _events(commit: Any) -> List[Any]:
            events: List[Any] = []
            for item in event_entries:
                entry = item["entry"]
                materialized = copy.copy(entry)
                materialized.seq = commit.seqs[item["seq_index"]]
                materialized.timestamp = commit.timestamp
                events.append(
                    {"type": "entry_added", "lane": lane.name, "entry": materialized}
                )
                usage = next(
                    (
                        candidate
                        for candidate in event_usage
                        if candidate["row"].entry_id == entry.id
                    ),
                    None,
                )
                if usage is not None:
                    events.append(
                        {
                            "type": "usage",
                            "lane": lane.name,
                            "row": {
                                **usage["row"].to_json(),
                                "seq": commit.seqs[usage["seq_index"]],
                            },
                            "totals": commit.stats.usage,
                        }
                    )
            return events

        return CommitDecision(
            writes=writes,
            operation_state=next_run,
            lane={"tipId": parent_id, "configuration": state.configuration},
            materialize=lambda _commit: complete,
            events=_events,
        )

    return await lane.settle_operation(capability, _plan, drive.context)


async def materialize_ready(
    lane: Any,
    drive: Any,
    capability: Any,
    sources: ToolBatchSource,
    recovery: bool,
) -> None:
    """Place every outcome-ready tool result, then report batch completion once."""
    read = await _read_placement(lane, drive, sources)
    if read is None:
        return
    events: List[Any] = []
    for item in read.items:
        suffix = {"recovery": True} if recovery else {}
        events.append(
            {
                "type": "message_start",
                "lane": lane.name,
                "runId": drive.operation_id,
                "message": item.message,
                **suffix,
            }
        )
        events.append(
            {
                "type": "message_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "message": item.message,
                "entryId": item.call.result_entry_id,
                **suffix,
            }
        )
    await lane.emit_batch(events, drive.context)
    complete = await _commit_placement(lane, drive, capability, read)
    if complete and read.turn_results is not None:
        await lane.emit_batch(
            [
                {
                    "type": "turn_end",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "turnId": capability.batch.turn_id,
                    "message": sources.assistant,
                    "toolResults": read.turn_results,
                    **({"recovery": True} if recovery else {}),
                }
            ],
            drive.context,
        )


def _tool_call(**_fields: Any) -> Any:
    from ...session.types import ToolCall

    return ToolCall(**_fields)
