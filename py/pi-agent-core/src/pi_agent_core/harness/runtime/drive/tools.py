"""Tool batch procedure ported from ``harness/runtime/drive/tools.ts``.

``run_tools`` executes one assistant turn's tool batch and settles it durably:
every planned call first publishes its intent, then crosses the external effect
boundary behind the drive gate, then stages its outcome before the batch is
materialized in source order.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from ...._pi_ai.types import TextContent, ToolResultMessage
from ....types import AgentToolCall, AgentToolResult
from ...execution.effect_gate import AbortRequested
from ...execution.tools import (
    AfterToolPatch,
    ClearedToolCall,
    FinalizedToolCall,
    apply_before_tool_decision,
    create_tool_result_message,
    execute_tool_call,
    finalize_tool_call,
    prepare_tool_call,
    tool_result_from_message,
)
from ...session.session import SessionInvariantError
from ...session.types import PendingEntry, ToolBatch, ToolCall, ToolsOperation
from ...session.values import (
    delete_value,
    operation_tool_args,
    operation_tool_memo,
    operation_tool_memo_prefix,
    pending_entry,
    pending_tool_output,
    set_value,
)
from ...types import AgentHarnessTool, AgentHarnessToolInvocation
from ..progress import open_tool_progress
from ..types import (
    CommitDecision,
    ContinueOperationResult,
    LaneReject,
    LaneReturn,
    ProcedureResult,
)
from .tool_placement import (
    ToolBatchSource,
    materialize_ready,
    read_tool_batch_source,
    tool_call_for,
    with_tool_batch,
)

__all__ = ["run_tools"]


#: Appended to a durable checkpoint when an interrupted call cannot be replayed.
INTERRUPTION_MARKER = (
    "[Tool execution was interrupted. The preceding output is the latest durable progress "
    "snapshot; newer live output may be missing, and the external outcome is unknown.]"
)


class ToolInvocationEnded(Exception):
    """Raised when a tool invocation no longer owns its durable effect."""

    def __init__(self) -> None:
        super().__init__("Tool invocation no longer owns its durable effect")
        self.name = "ToolInvocationEnded"


@dataclass
class CurrentBatch:
    """The currently owned tools operation together with its batch."""

    run: ToolsOperation
    batch: ToolBatch


@dataclass
class ToolCallTask:
    """One in-flight tool call whose durable settlement is still pending."""

    completion: Awaitable[None]


@dataclass
class ToolOutcome:
    """One finalized tool call ready to be staged as a transcript message."""

    tool_call: AgentToolCall
    message: ToolResultMessage
    terminate: bool


@dataclass
class PreparedToolInvocation:
    """Either a cleared call to execute or an outcome produced without an effect."""

    kind: str  # "ready" | "outcome"
    cleared: Optional[ClearedToolCall] = None
    outcome: Optional[ToolOutcome] = None


@dataclass
class ToolExecution:
    """Tool catalog plus resolved tool context for one running batch."""

    tools: List[AgentHarnessTool]
    tools_by_name: dict
    tool_context: Any


@dataclass
class InvocationCapability:
    """Live invocation identity plus its ownership expiry control."""

    invocation: AgentHarnessToolInvocation
    expire: Callable[[], None]


def _swallow(task: "asyncio.Task[Any]") -> None:
    """Retrieve a background task result so failures stay silent and reported nowhere."""
    if task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - best-effort callback
        pass


def current_batch(lane: Any) -> Optional[CurrentBatch]:
    """The currently owned tools operation, or ``None`` once it moved on."""
    operation = lane.state.operation
    if getattr(getattr(operation, "state", None), "at", None) != "tools":
        return None
    return CurrentBatch(run=operation.state, batch=operation.state.batch)


def find_call(batch: ToolBatch, source_index: int, result_entry_id: str) -> Optional[ToolCall]:
    """The call identified by both its source index and its reserved result entry."""
    for call in batch.calls:
        if call.source_index == source_index and call.result_entry_id == result_entry_id:
            return call
    return None


def replace_call(batch: ToolBatch, replacement: ToolCall) -> ToolBatch:
    """Copy a batch with one call replaced by its successor leaf."""
    return ToolBatch(
        assistant_entry_id=batch.assistant_entry_id,
        configuration=batch.configuration,
        turn_id=batch.turn_id,
        calls=[
            replacement
            if call.source_index == replacement.source_index
            and call.result_entry_id == replacement.result_entry_id
            else call
            for call in batch.calls
        ],
    )


def validate_memo_name(name: str) -> None:
    """Reject memo names that cannot address durable storage."""
    if len(name) == 0:
        raise TypeError("Tool invocation memo name must not be empty")
    if ":" in name:
        raise TypeError("Tool invocation memo name must not contain ':'")


def invocation_capability(
    lane: Any,
    drive: Any,
    batch: ToolBatch,
    call: ToolCall,
) -> InvocationCapability:
    """Build the invocation handle bound to one durable effect-pending call."""
    liveness = {"active": True}

    def _owns_effect(projection: Any) -> bool:
        operation = projection.operation
        if getattr(getattr(operation, "state", None), "at", None) != "tools":
            return False
        owned = find_call(operation.state.batch, call.source_index, call.result_entry_id)
        return owned is not None and owned.status == "effect_pending"

    def _ended() -> ToolInvocationEnded:
        return ToolInvocationEnded()

    async def _command_value(name: str) -> Any:
        if not liveness["active"]:
            raise _ended()

        async def _plan(projection: Any, reader: Any) -> Any:
            if not _owns_effect(projection):
                return LaneReject(error=_ended())
            stored = await reader.get_value(
                operation_tool_memo(drive.operation_id, call.result_entry_id, name), drive.context
            )
            return LaneReturn(result=None if stored is None else stored.value)

        return await lane.command(_plan, drive.context)

    async def _command_set(name: str, next_value: Any) -> None:
        if not liveness["active"]:
            raise _ended()

        def _plan(projection: Any, _reader: Any) -> Any:
            if not _owns_effect(projection):
                return LaneReject(error=_ended())
            address = operation_tool_memo(drive.operation_id, call.result_entry_id, name)
            return CommitDecision(
                writes=[
                    delete_value(address) if next_value is None else set_value(address, next_value)
                ],
                next=projection,
                materialize=lambda commit: None,
            )

        await lane.command(_plan, drive.context)

    def _get_memo(name: str) -> Awaitable[Any]:
        validate_memo_name(name)
        return _command_value(name)

    def _set_memo(name: str, next_value: Any) -> Awaitable[None]:
        validate_memo_name(name)
        return _command_set(name, next_value)

    def _expire() -> None:
        liveness["active"] = False

    return InvocationCapability(
        invocation=AgentHarnessToolInvocation(
            invocation_id=call.result_entry_id,
            operation_id=drive.operation_id,
            turn_id=batch.turn_id,
            get_memo=_get_memo,
            set_memo=_set_memo,
        ),
        expire=_expire,
    )


def synthetic_message(
    tool_call: AgentToolCall,
    content: List[Any],
    options: Optional[dict] = None,
) -> ToolResultMessage:
    """Build an error tool-result message that never crosses the effect boundary."""
    options = options or {}
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=content,
        details=options.get("details"),
        usage=options.get("usage"),
        is_error=True,
        timestamp=int(time.time() * 1000),
    )


def aborted_outcome(tool_call: AgentToolCall) -> ToolOutcome:
    """Outcome for a call cancelled before it ever reached the effect boundary."""
    return ToolOutcome(
        tool_call=tool_call,
        message=synthetic_message(
            tool_call, [TextContent(text="Tool execution was cancelled before completion.")]
        ),
        terminate=False,
    )


def interrupted_outcome(tool_call: AgentToolCall, checkpoint: Optional[AgentToolResult]) -> ToolOutcome:
    """Outcome for a call whose external effect may have happened without a durable result."""
    content = [
        *(checkpoint.content if checkpoint is not None else []),
        TextContent(text=INTERRUPTION_MARKER),
    ]
    options = (
        {}
        if checkpoint is None
        else {"details": checkpoint.details, "usage": checkpoint.usage}
    )
    return ToolOutcome(
        tool_call=tool_call,
        message=synthetic_message(tool_call, content, options),
        terminate=False,
    )


def truncated_outcome(tool_call: AgentToolCall) -> ToolOutcome:
    """Outcome for a call whose assistant response hit the output token limit."""
    return ToolOutcome(
        tool_call=tool_call,
        message=synthetic_message(
            tool_call,
            [
                TextContent(
                    text=(
                        f"Tool call {json.dumps(tool_call.name)} was not executed because the "
                        "assistant response hit the output token limit, so its arguments may be "
                        "truncated. Re-issue the tool call with complete arguments."
                    )
                )
            ],
        ),
        terminate=False,
    )


def outcome_from_finalized_call(finalized: FinalizedToolCall) -> ToolOutcome:
    """Outcome for a call settled without an effect, e.g. blocked or invalid."""
    return ToolOutcome(
        tool_call=finalized.tool_call,
        message=create_tool_result_message(finalized),
        terminate=finalized.terminate,
    )


async def publish_tool_intent(
    lane: Any,
    drive: Any,
    run: ToolsOperation,
    planned: ToolCall,
    tool_call: AgentToolCall,
    args: Any,
    replay: str,
    recovery: bool,
) -> ContinueOperationResult:
    """Durably publish the intent to execute one tool call before its effect."""

    async def _plan(_state: Any, current: ToolsOperation, _meta: Any, _reader: Any) -> Any:
        effect_pending = ToolCall(
            source_index=planned.source_index,
            result_entry_id=planned.result_entry_id,
            status="effect_pending",
            replay=replay,
        )
        return CommitDecision(
            writes=[
                set_value(
                    operation_tool_args(
                        drive.operation_id, current.batch.turn_id, planned.source_index
                    ),
                    args,
                )
            ],
            operation_state=with_tool_batch(current, replace_call(current.batch, effect_pending)),
            materialize=lambda commit: effect_pending,
            events=lambda commit: [
                {
                    "type": "tool_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "turnId": current.batch.turn_id,
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "args": args,
                    **({"recovery": True} if recovery else {}),
                }
            ],
        )

    return await lane.continue_operation(run, _plan, drive.context)


async def publish_tool_outcome(
    lane: Any,
    drive: Any,
    capability: ToolsOperation,
    call: ToolCall,
    finalized: ToolOutcome,
    recovery: bool,
) -> None:
    """Stage one finalized tool result and delete every invocation-scoped memo."""

    async def _plan(_state: Any, run: ToolsOperation, _meta: Any, reader: Any) -> Any:
        tool_call = finalized.tool_call
        memos = await reader.scan_values(
            operation_tool_memo_prefix(drive.operation_id, call.result_entry_id),
            drive.context,
        )
        durable_terminate = run.control.status == "running" and finalized.terminate
        outcome = ToolCall(
            source_index=call.source_index,
            result_entry_id=call.result_entry_id,
            status="outcome_ready",
            terminate=durable_terminate,
        )
        events: List[Any] = []
        if call.status == "planned":
            events.append(
                {
                    "type": "tool_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "turnId": run.batch.turn_id,
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "args": tool_call.arguments,
                    **({"recovery": True} if recovery else {}),
                }
            )
        events.append(
            {
                "type": "tool_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "turnId": run.batch.turn_id,
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "result": tool_result_from_message(finalized.message, durable_terminate),
                "isError": finalized.message.is_error,
                "terminate": durable_terminate,
                **({"recovery": True} if recovery else {}),
            }
        )
        return CommitDecision(
            writes=[
                set_value(
                    pending_entry(call.result_entry_id),
                    PendingEntry(type="message", payload=finalized.message),
                ),
                delete_value(pending_tool_output(drive.operation_id, call.result_entry_id)),
                *[delete_value(stored.address) for stored in memos],
            ],
            operation_state=with_tool_batch(run, replace_call(run.batch, outcome)),
            materialize=lambda commit: None,
            events=lambda commit: events,
        )

    await lane.settle_operation(capability, _plan, drive.context)


async def clear_replay_checkpoint(
    lane: Any,
    drive: Any,
    batch: ToolBatch,
    call: ToolCall,
    tool_call: AgentToolCall,
) -> Any:
    """Drop the stale output checkpoint and re-read persisted arguments for a safe replay."""

    async def _plan(state: Any, reader: Any) -> Any:
        stored = await reader.get_value(
            operation_tool_args(drive.operation_id, batch.turn_id, call.source_index),
            drive.context,
        )
        if stored is None:
            raise SessionInvariantError(
                f"Tool call {call.result_entry_id} is missing persisted arguments"
            )
        return CommitDecision(
            writes=[delete_value(pending_tool_output(drive.operation_id, call.result_entry_id))],
            next=state,
            materialize=lambda commit: stored.value,
            events=lambda commit: [
                {
                    "type": "tool_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "turnId": batch.turn_id,
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "args": stored.value,
                    "recovery": True,
                }
            ],
        )

    return await lane.command(_plan, drive.context)


async def read_checkpoint(lane: Any, drive: Any, call: ToolCall) -> Optional[AgentToolResult]:
    """Read the latest durable partial output for one interrupted invocation."""

    async def _plan(_state: Any, reader: Any) -> Any:
        stored = await reader.get_value(
            pending_tool_output(drive.operation_id, call.result_entry_id), drive.context
        )
        return LaneReturn(result=None if stored is None else stored.value)

    return await lane.command(_plan, drive.context)


async def resolve_tool_context(lane: Any, drive: Any) -> Any:
    """Resolve the process-local tool context captured by the active configuration."""
    source = lane.read_config().tool_context
    return await source(drive.context) if callable(source) else source


async def perform_tool_invocation(
    lane: Any,
    drive: Any,
    batch: ToolBatch,
    call: ToolCall,
    cleared: ClearedToolCall,
    tool_context: Any,
    recovery: bool,
) -> ToolOutcome:
    """Execute one cleared tool call behind the drive gate and finalize its result."""
    capability = invocation_capability(lane, drive, batch, call)
    progress = open_tool_progress(
        lane, drive, batch.turn_id, call.source_index, call.result_entry_id
    )
    latest_update_delivery: Optional["asyncio.Task[Any]"] = None

    def _publish_update(partial: Any) -> None:
        nonlocal latest_update_delivery

        async def _deliver() -> None:
            await lane.emit_batch(
                [
                    {
                        "type": "tool_update",
                        "lane": lane.name,
                        "runId": drive.operation_id,
                        "turnId": batch.turn_id,
                        "toolCallId": cleared.tool_call.id,
                        "toolName": cleared.tool_call.name,
                        "partialResult": partial,
                        **({"recovery": True} if recovery else {}),
                    }
                ],
                drive.context,
            )

        task = asyncio.get_running_loop().create_task(_deliver())
        latest_update_delivery = task
        task.add_done_callback(_swallow)

    def _on_update(partial: Any, options: Any = None) -> None:
        _publish_update(partial)
        if options is not None and getattr(options, "checkpoint", False) is True:
            progress.write(partial)

    try:
        executed = await execute_tool_call(
            cleared,
            drive.gate,
            _on_update,
            tool_context,
            capability.invocation,
            drive.context,
        )
    except BaseException as error:  # noqa: BLE001 - mirrors the TS catch around the admission
        capability.expire()
        progress.seal()
        await progress.drain()
        if not isinstance(error, AbortRequested):
            raise
        await error.cancellation
        if recovery:
            return interrupted_outcome(cleared.tool_call, None)
        return aborted_outcome(cleared.tool_call)

    capability.expire()
    progress.seal()
    if latest_update_delivery is not None:
        await latest_update_delivery
    await progress.drain()

    patch: Optional[AfterToolPatch]
    try:
        patch = await lane.hooks.run_tool_with_gate(
            "after_tool",
            {
                "lane": lane.name,
                "runId": drive.operation_id,
                "toolCallId": cleared.tool_call.id,
                "toolName": cleared.tool_call.name,
                "args": cleared.args,
                "content": executed.result.content,
                **(
                    {}
                    if executed.result.details is None
                    else {"details": executed.result.details}
                ),
                "isError": executed.is_error,
                **({} if executed.result.usage is None else {"usage": executed.result.usage}),
            },
            drive.gate,
            drive.context,
        )
    except AbortRequested:
        patch = None
    finalized = finalize_tool_call(cleared, executed, patch)
    return ToolOutcome(
        tool_call=finalized.tool_call,
        message=create_tool_result_message(finalized),
        terminate=finalized.terminate,
    )


async def prepare_tool_invocation(
    lane: Any,
    drive: Any,
    sources: ToolBatchSource,
    call: ToolCall,
    tools: List[AgentHarnessTool],
) -> PreparedToolInvocation:
    """Resolve one planned call into either a cleared invocation or a synthetic outcome."""
    tool_call = tool_call_for(sources, call)
    if sources.assistant.stop_reason == "length":
        return PreparedToolInvocation(kind="outcome", outcome=truncated_outcome(tool_call))
    prepared = prepare_tool_call(tool_call, tools)
    if getattr(prepared, "kind", None) is not None:
        return PreparedToolInvocation(
            kind="outcome", outcome=outcome_from_finalized_call(prepared)
        )

    decision: Any
    try:
        decision = await lane.hooks.run_tool_with_gate(
            "before_tool",
            {
                "lane": lane.name,
                "runId": drive.operation_id,
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": prepared.args,
            },
            drive.gate,
            drive.context,
        )
    except AbortRequested as error:
        await error.cancellation
        return PreparedToolInvocation(kind="outcome", outcome=aborted_outcome(tool_call))
    cleared = apply_before_tool_decision(prepared, decision)
    if getattr(cleared, "kind", None) is not None:
        return PreparedToolInvocation(kind="outcome", outcome=outcome_from_finalized_call(cleared))
    return PreparedToolInvocation(kind="ready", cleared=cleared)


async def start_tool_invocation(
    lane: Any,
    drive: Any,
    run: ToolsOperation,
    sources: ToolBatchSource,
    call: ToolCall,
    tools: List[AgentHarnessTool],
    tool_context: Any,
    recovery: bool,
) -> ToolCallTask:
    """Prepare, publish, and start one planned tool call."""
    prepared = await prepare_tool_invocation(lane, drive, sources, call, tools)
    if prepared.kind == "outcome":
        return ToolCallTask(
            completion=publish_tool_outcome(lane, drive, run, call, prepared.outcome, recovery)
        )
    cleared = prepared.cleared
    effect_pending = await publish_tool_intent(
        lane,
        drive,
        run,
        call,
        cleared.tool_call,
        cleared.args,
        getattr(cleared.tool, "replay", None) or "never",
        recovery,
    )
    if effect_pending.kind == "cancel_requested":
        return ToolCallTask(
            completion=publish_tool_outcome(
                lane, drive, run, call, aborted_outcome(cleared.tool_call), recovery
            )
        )

    async def _execute_then_publish() -> None:
        outcome = await perform_tool_invocation(
            lane, drive, run.batch, effect_pending.value, cleared, tool_context, recovery
        )
        await publish_tool_outcome(lane, drive, run, effect_pending.value, outcome, recovery)

    return ToolCallTask(completion=_execute_then_publish())


async def recover_tool_invocation(
    lane: Any,
    drive: Any,
    run: ToolsOperation,
    sources: ToolBatchSource,
    call: ToolCall,
    tools_by_name: dict,
    tool_context: Any,
    cancelled: bool,
) -> ToolCallTask:
    """Replay a safe effect-pending call or settle it from its durable checkpoint."""
    tool_call = tool_call_for(sources, call)
    tool = tools_by_name.get(tool_call.name)
    if (
        not cancelled
        and call.replay == "safe"
        and tool is not None
        and getattr(tool, "replay", None) == "safe"
    ):
        args = await clear_replay_checkpoint(lane, drive, run.batch, call, tool_call)
        cleared = ClearedToolCall(tool_call=tool_call, tool=tool, args=args)

        async def _replay_then_publish() -> None:
            outcome = await perform_tool_invocation(
                lane, drive, run.batch, call, cleared, tool_context, True
            )
            await publish_tool_outcome(lane, drive, run, call, outcome, True)

        return ToolCallTask(completion=_replay_then_publish())
    checkpoint = await read_checkpoint(lane, drive, call)
    return ToolCallTask(
        completion=publish_tool_outcome(
            lane, drive, run, call, interrupted_outcome(tool_call, checkpoint), True
        )
    )


async def run_sequential(
    lane: Any,
    drive: Any,
    run: ToolsOperation,
    sources: ToolBatchSource,
    execution: Optional[ToolExecution],
    recovery: bool,
) -> ProcedureResult:
    """Drive one batch one call at a time, materializing after every settled call."""
    batch = run.batch
    for _transition in range(len(batch.calls) * 2 + 2):
        await materialize_ready(lane, drive, run, sources, recovery)
        current = current_batch(lane)
        if current is None:
            return ProcedureResult(kind="continue")
        call = next(
            (candidate for candidate in current.batch.calls if candidate.status != "completed"),
            None,
        )
        if call is None:
            raise SessionInvariantError("Tool batch remained open after every call completed")
        if call.status == "outcome_ready":
            raise SessionInvariantError("Ready tool outcome was not materialized")

        if current.run.control.status == "cancel_requested":
            tool_call = tool_call_for(sources, call)
            if call.status == "planned":
                await publish_tool_outcome(
                    lane, drive, current.run, call, aborted_outcome(tool_call), recovery
                )
            else:
                checkpoint = await read_checkpoint(lane, drive, call)
                await publish_tool_outcome(
                    lane,
                    drive,
                    current.run,
                    call,
                    interrupted_outcome(tool_call, checkpoint),
                    recovery,
                )
            continue

        if execution is None:
            raise SessionInvariantError("Running tool batch is missing execution context")
        if call.status == "planned":
            started = await start_tool_invocation(
                lane,
                drive,
                current.run,
                sources,
                call,
                execution.tools,
                execution.tool_context,
                recovery,
            )
        else:
            started = await recover_tool_invocation(
                lane,
                drive,
                current.run,
                sources,
                call,
                execution.tools_by_name,
                execution.tool_context,
                False,
            )
        await started.completion
    raise SessionInvariantError("Sequential tool batch exceeded its bounded transition count")


async def run_parallel(
    lane: Any,
    drive: Any,
    run: ToolsOperation,
    sources: ToolBatchSource,
    tools: List[AgentHarnessTool],
    tools_by_name: dict,
    tool_context: Any,
    recovery: bool,
) -> ProcedureResult:
    """Start every remaining call, then materialize settled prefixes in source order."""
    batch = run.batch
    materialization: Optional["asyncio.Task[Any]"] = None

    def _schedule_materialization() -> "asyncio.Task[Any]":
        nonlocal materialization
        # Capture the predecessor synchronously: reading it inside the coroutine
        # would observe this task itself and self-await.
        previous = materialization

        async def _materialize() -> None:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001 - chained delivery
                    pass
            await materialize_ready(lane, drive, run, sources, recovery)

        task = asyncio.get_running_loop().create_task(_materialize())
        materialization = task
        task.add_done_callback(_swallow)
        return task

    def _schedule_job(completion: Awaitable[None]) -> "asyncio.Task[Any]":
        async def _job() -> None:
            await completion
            await _schedule_materialization()

        task = asyncio.get_running_loop().create_task(_job())
        task.add_done_callback(_swallow)
        return task

    jobs: List["asyncio.Task[Any]"] = []
    for call in batch.calls:
        if call.status in ("completed", "outcome_ready"):
            continue
        if call.status == "planned":
            started = await start_tool_invocation(
                lane, drive, run, sources, call, tools, tool_context, recovery
            )
        else:
            started = await recover_tool_invocation(
                lane,
                drive,
                run,
                sources,
                call,
                tools_by_name,
                tool_context,
                lane.state.operation.state.control.status == "cancel_requested",
            )
        jobs.append(_schedule_job(started.completion))
    await asyncio.gather(*jobs)
    await _schedule_materialization()
    return ProcedureResult(kind="continue")


async def run_tools(lane: Any, drive: Any, run: ToolsOperation) -> ProcedureResult:
    """Execute, recover, stage, and source-order one complete durable tool batch."""
    batch = run.batch
    recovery = any(
        call.status in ("effect_pending", "outcome_ready") for call in batch.calls
    )
    if recovery:
        await lane.emit_batch(
            [
                {
                    "type": "turn_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "turnId": batch.turn_id,
                    "recovery": True,
                }
            ],
            drive.context,
        )
    sources = await read_tool_batch_source(lane, drive, batch)
    await materialize_ready(lane, drive, run, sources, recovery)
    current = current_batch(lane)
    if current is None:
        return ProcedureResult(kind="continue")
    if current.run.control.status == "cancel_requested":
        return await run_sequential(lane, drive, current.run, sources, None, recovery)
    config = lane.read_config()
    active = set(batch.configuration.active_tool_names)
    tools = [tool for tool in config.tools if tool.name in active]
    tools_by_name = {tool.name: tool for tool in tools}
    tool_context = await resolve_tool_context(lane, drive)
    if run.settings.tool_execution == "sequential":
        return await run_sequential(
            lane,
            drive,
            current.run,
            sources,
            ToolExecution(tools=tools, tools_by_name=tools_by_name, tool_context=tool_context),
            recovery,
        )
    return await run_parallel(
        lane, drive, current.run, sources, tools, tools_by_name, tool_context, recovery
    )
