"""Agent loop ported from ``packages/agent/src/agent-loop.ts``.

Works with AgentMessage throughout; transforms to Message[] only at the LLM
call boundary.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import time
from typing import Any, List, Optional, Sequence, Union

from ._pi_ai.abort import AbortSignal
from ._pi_ai.event_stream import EventStream
from ._pi_ai.transcript import (
    ToolStateChanges,
    get_current_tools,
    get_tool_state_changes,
    to_tool_declaration,
)
from ._pi_ai.transcript import normalize_context
from ._pi_ai.types import (
    AgentMessage,
    AssistantMessage,
    Context,
    Message,
    SystemMessage,
    ToolResultMessage,
    TranscriptContext,
)
from ._pi_ai.validation import validate_tool_arguments
from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentStartEvent,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = ["agent_loop", "agent_loop_continue", "run_agent_loop", "run_agent_loop_continue"]


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Stream-returning entry points
# ---------------------------------------------------------------------------


def agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    stream_fn: Optional[StreamFn],
) -> EventStream:
    """Start an agent loop with a new prompt message; returns an event stream."""
    stream = _create_agent_stream()

    async def _run() -> None:
        try:
            messages = await run_agent_loop(
                prompts,
                context,
                config,
                lambda event: _emit_to_stream(stream, event),
                signal,
                stream_fn,
            )
            stream.end(messages)
        except BaseException as error:  # noqa: BLE001 - propagate to stream consumers
            future = stream.result
            if not future.done():
                future.set_exception(error)
            stream.end()

    asyncio.get_running_loop().create_task(_run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    stream_fn: Optional[StreamFn],
) -> EventStream:
    """Continue an agent loop from the current context without adding a new message."""
    if len(context.messages) == 0:
        raise RuntimeError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise RuntimeError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def _run() -> None:
        try:
            messages = await run_agent_loop_continue(
                context,
                config,
                lambda event: _emit_to_stream(stream, event),
                signal,
                stream_fn,
            )
            stream.end(messages)
        except BaseException as error:  # noqa: BLE001 - propagate to stream consumers
            future = stream.result
            if not future.done():
                future.set_exception(error)
            stream.end()

    asyncio.get_running_loop().create_task(_run())
    return stream


async def _emit_to_stream(stream: EventStream, event: AgentEvent) -> None:
    stream.push(event)
    await asyncio.sleep(0)


def _create_agent_stream() -> EventStream:
    return EventStream(
        lambda event: isinstance(event, AgentEndEvent),
        lambda event: event.messages if isinstance(event, AgentEndEvent) else [],
    )


# ---------------------------------------------------------------------------
# Awaitable entry points
# ---------------------------------------------------------------------------


async def run_agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Optional[AbortSignal],
    stream_fn: Optional[StreamFn],
) -> List[AgentMessage]:
    """Start an agent loop with new prompt messages; returns all new messages."""
    initial_messages = _declare_tool_changes(context, prompts)
    new_messages: List[AgentMessage] = list(initial_messages)
    current_context = AgentContext(
        messages=[*context.messages, *initial_messages],
        tools=context.tools,
    )

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))
    for message in initial_messages:
        await _maybe_await(emit(MessageStartEvent(message=message)))
        await _maybe_await(emit(MessageEndEvent(message=message)))

    await _run_loop(current_context, new_messages, config, signal, emit, stream_fn or get_default_stream_fn())
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Optional[AbortSignal],
    stream_fn: Optional[StreamFn],
) -> List[AgentMessage]:
    """Continue an agent loop from the current context; returns all new messages."""
    if len(context.messages) == 0:
        raise RuntimeError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise RuntimeError("Cannot continue from message role: assistant")

    new_messages: List[AgentMessage] = []
    current_context = AgentContext(messages=list(context.messages), tools=context.tools)

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))

    await _run_loop(current_context, new_messages, config, signal, emit, stream_fn or get_default_stream_fn())
    return new_messages


# ---------------------------------------------------------------------------
# Main loop logic shared by agentLoop and agentLoopContinue
# ---------------------------------------------------------------------------


async def _run_loop(
    initial_context: AgentContext,
    new_messages: List[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> None:
    current_context = initial_context
    config = initial_config
    last_completed_turn: Optional[PrepareNextTurnContext] = None
    # Check for steering messages at start (user may have typed while waiting)
    pending_messages: List[AgentMessage] = (
        list(await config.get_steering_messages()) if config.get_steering_messages else []
    )

    # Outer loop: continues when queued follow-up messages arrive after agent would stop
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or len(pending_messages) > 0:
            prepared_messages: List[AgentMessage] = []
            if last_completed_turn is not None:
                if config.prepare_next_turn is not None:
                    next_turn_snapshot = await _maybe_await(config.prepare_next_turn(last_completed_turn))
                else:
                    next_turn_snapshot = None
                if next_turn_snapshot is not None:
                    current_context = next_turn_snapshot.context or current_context
                    prepared_messages = list(next_turn_snapshot.messages or [])
                    config = dataclasses.replace(
                        config,
                        model=next_turn_snapshot.model or config.model,
                        reasoning=(
                            config.reasoning
                            if next_turn_snapshot.thinking_level is None
                            else (
                                None
                                if next_turn_snapshot.thinking_level == "off"
                                else next_turn_snapshot.thinking_level
                            )
                        ),
                    )
                # Preparation can be long-running (for example, compaction). Pick up
                # steering queued while it ran. Only poll again if the earlier poll
                # returned nothing; otherwise one-at-a-time mode would deliver two
                # messages in this turn.
                if len(pending_messages) == 0 and config.get_steering_messages:
                    pending_messages = list(await config.get_steering_messages())
                await _maybe_await(emit(TurnStartEvent()))

            # Process prepared and queued messages before the next assistant response.
            for message in _declare_tool_changes(current_context, [*prepared_messages, *pending_messages]):
                await _maybe_await(emit(MessageStartEvent(message=message)))
                await _maybe_await(emit(MessageEndEvent(message=message)))
                current_context.messages.append(message)
                new_messages.append(message)
            pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(current_context, config, signal, emit, stream_function)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _maybe_await(emit(TurnEndEvent(message=message, tool_results=[])))
                await _maybe_await(emit(AgentEndEvent(messages=new_messages)))
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if getattr(c, "type", None) == "toolCall"]

            tool_results: List[ToolResultMessage] = []
            has_more_tool_calls = False
            if len(tool_calls) > 0:
                # A "length" stop means the output was cut off by the token limit, so
                # every tool call in the message may carry truncated arguments. Fail
                # them all instead of executing potentially borked calls.
                if message.stop_reason == "length":
                    executed_tool_batch = await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    executed_tool_batch = await _execute_tool_calls(
                        current_context, message, config, signal, emit
                    )
                tool_results.extend(executed_tool_batch.messages)
                has_more_tool_calls = not executed_tool_batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _maybe_await(emit(TurnEndEvent(message=message, tool_results=tool_results)))

            last_completed_turn = PrepareNextTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )

            if config.should_stop_after_turn is not None and await _maybe_await(
                config.should_stop_after_turn(last_completed_turn)
            ):
                await _maybe_await(emit(AgentEndEvent(messages=new_messages)))
                return

            if config.get_steering_messages:
                pending_messages = list(await config.get_steering_messages())

        # Agent would stop here. Check for follow-up messages.
        follow_up_messages = (
            list(await config.get_follow_up_messages()) if config.get_follow_up_messages else []
        )
        if len(follow_up_messages) > 0:
            # Set as pending so inner loop processes them
            pending_messages = follow_up_messages
            continue

        # No more messages, exit
        break

    await _maybe_await(emit(AgentEndEvent(messages=new_messages)))


# ---------------------------------------------------------------------------
# Tool loadout change declarations
# ---------------------------------------------------------------------------


def _declare_tool_changes(context: AgentContext, pending_messages: List[AgentMessage]) -> List[AgentMessage]:
    """Declare tool loadout changes to the model."""
    system_index = -1
    for i in range(len(pending_messages) - 1, -1, -1):
        if getattr(pending_messages[i], "role", None) == "system":
            system_index = i
            break
    pending = pending_messages[system_index] if system_index >= 0 else None
    if pending is not None:
        baseline = [
            _with_tool_changes(pending, _NO_CHANGES) if index == system_index else message
            for index, message in enumerate(pending_messages)
        ]
    else:
        baseline = list(pending_messages)
    changes = get_tool_state_changes(
        get_current_tools([*context.messages, *baseline]),
        [to_tool_declaration(t) for t in (context.tools or [])],
    )
    unchanged = len(changes.tools_added) == 0 and len(changes.tools_removed) == 0

    if pending is not None:
        # Keep the caller's message object when it already declares no tool changes.
        if unchanged and not (pending.tools_added or []) and not (pending.tools_removed or []):
            return pending_messages
        return [
            _with_tool_changes(pending, changes) if index == system_index else message
            for index, message in enumerate(baseline)
        ]
    if unchanged:
        return pending_messages
    update = _with_tool_changes(SystemMessage(content="", timestamp=int(time.time() * 1000)), changes)
    insert_index = next(
        (i for i, message in enumerate(pending_messages) if getattr(message, "role", None) != "system"),
        len(pending_messages),
    )
    return [*pending_messages[:insert_index], update, *pending_messages[insert_index:]]


_NO_CHANGES = ToolStateChanges(tools_added=[], tools_removed=[])


def _with_tool_changes(message: SystemMessage, changes: ToolStateChanges) -> SystemMessage:
    """Copy a system message with its tool fields replaced by ``changes``; empty lists omit the field."""
    updated = SystemMessage(
        content=copy.deepcopy(message.content),
        sections=None if message.sections is None else dict(message.sections),
        timestamp=message.timestamp,
    )
    if len(changes.tools_added) > 0:
        updated.tools_added = list(changes.tools_added)
    if len(changes.tools_removed) > 0:
        updated.tools_removed = list(changes.tools_removed)
    return updated


# ---------------------------------------------------------------------------
# Assistant response streaming
# ---------------------------------------------------------------------------


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> AssistantMessage:
    """Stream an assistant response from the LLM."""
    # Apply context transform if configured (AgentMessage[] -> AgentMessage[])
    messages = context.messages
    if config.transform_context is not None:
        messages = await config.transform_context(messages, signal)

    # Convert to LLM-compatible messages (AgentMessage[] -> Message[])
    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    llm_context = normalize_context(Context(system_prompt=None, messages=llm_messages, tools=None))

    # Resolve API key (important for expiring tokens)
    resolved_api_key = config.api_key
    if config.get_api_key is not None:
        resolved_api_key = (await _maybe_await(config.get_api_key(config.model.provider))) or config.api_key

    request_options = dataclasses.replace(config, api_key=resolved_api_key, signal=signal)
    response = stream_function(config.model, llm_context, request_options)
    if asyncio.iscoroutine(response) or asyncio.isfuture(response):
        response = await response

    partial_message: Optional[AssistantMessage] = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _maybe_await(emit(MessageStartEvent(message=copy.deepcopy(partial_message))))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _maybe_await(
                    emit(
                        MessageUpdateEvent(
                            message=copy.deepcopy(partial_message),
                            assistant_message_event=event,
                        )
                    )
                )
        elif event.type in ("done", "error"):
            final_message = await response.result
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await _maybe_await(emit(MessageStartEvent(message=copy.deepcopy(final_message))))
            await _maybe_await(emit(MessageEndEvent(message=final_message)))
            return final_message

    final_message = await response.result
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _maybe_await(emit(MessageStartEvent(message=copy.deepcopy(final_message))))
    await _maybe_await(emit(MessageEndEvent(message=final_message)))
    return final_message


# ---------------------------------------------------------------------------
# Tool call execution
# ---------------------------------------------------------------------------


class _ExecutedToolCallBatch:
    def __init__(self, messages: List[ToolResultMessage], terminate: bool) -> None:
        self.messages = messages
        self.terminate = terminate


async def _fail_tool_calls_from_truncated_message(
    tool_calls: List[AgentToolCall], emit: AgentEventSink
) -> _ExecutedToolCallBatch:
    """Fail all tool calls from an assistant message truncated by the output token limit."""
    messages: List[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _maybe_await(
            emit(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                )
            )
        )
        finalized = _FinalizedToolCallOutcome(
            tool_call=tool_call,
            result=_create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output '
                "token limit, so its arguments may be truncated. Re-issue the tool call with "
                "complete arguments."
            ),
            is_error=True,
        )
        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)
    return _ExecutedToolCallBatch(messages=messages, terminate=False)


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    tool_calls = [c for c in assistant_message.content if getattr(c, "type", None) == "toolCall"]
    tools = current_context.tools or []
    has_sequential_tool_call = any(
        next((t.execution_mode for t in tools if t.name == tc.name), None) == "sequential"
        for tc in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential_tool_call:
        return await _execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit
        )
    return await _execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, signal, emit)


class _PreparedToolCall:
    def __init__(self, tool_call: AgentToolCall, tool: AgentTool, args: Any) -> None:
        self.kind = "prepared"
        self.tool_call = tool_call
        self.tool = tool
        self.args = args


class _ImmediateToolCallOutcome:
    def __init__(self, result: AgentToolResult, is_error: bool) -> None:
        self.kind = "immediate"
        self.result = result
        self.is_error = is_error


class _ExecutedToolCallOutcome:
    def __init__(self, result: AgentToolResult, is_error: bool) -> None:
        self.result = result
        self.is_error = is_error


class _FinalizedToolCallOutcome:
    def __init__(self, tool_call: AgentToolCall, result: AgentToolResult, is_error: bool) -> None:
        self.tool_call = tool_call
        self.result = result
        self.is_error = is_error


def _should_terminate_tool_batch(finalized_calls: List[_FinalizedToolCallOutcome]) -> bool:
    return len(finalized_calls) > 0 and all(f.result.terminate is True for f in finalized_calls)


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: AgentToolCall) -> AgentToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared_arguments = tool.prepare_arguments(tool_call.arguments)
    if prepared_arguments is tool_call.arguments:
        return tool_call
    updated = copy.deepcopy(tool_call)
    updated.arguments = prepared_arguments
    return updated


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
) -> Union[_PreparedToolCall, _ImmediateToolCallOutcome]:
    tool = next((t for t in (current_context.tools or []) if t.name == tool_call.name), None)
    if tool is None:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)
        if config.before_tool_call is not None:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=tool_call,
                        args=validated_args,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if signal is not None and signal.aborted:
                return _ImmediateToolCallOutcome(
                    result=_create_error_tool_result("Operation aborted"),
                    is_error=True,
                )
            if before_result is not None and before_result.block:
                result = _create_error_tool_result(before_result.reason or "Tool execution was blocked")
                if before_result.terminate is True:
                    result.terminate = True
                return _ImmediateToolCallOutcome(result=result, is_error=True)
        if signal is not None and signal.aborted:
            return _ImmediateToolCallOutcome(
                result=_create_error_tool_result("Operation aborted"),
                is_error=True,
            )
        return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as error:  # noqa: BLE001 - mirrors TS catch
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(str(error)),
            is_error=True,
        )


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    update_events: List[Any] = []
    accepting_updates = True

    def _on_update(partial_result: Any) -> None:
        if not accepting_updates:
            return
        update_events.append(
            _maybe_await(
                emit(
                    ToolExecutionUpdateEvent(
                        tool_call_id=prepared.tool_call.id,
                        tool_name=prepared.tool_call.name,
                        args=prepared.tool_call.arguments,
                        partial_result=partial_result,
                    )
                )
            )
        )

    try:
        assert prepared.tool.execute is not None
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            signal,
            _on_update,
        )
        accepting_updates = False
        await asyncio.gather(*[f for f in update_events if asyncio.isfuture(f) or asyncio.iscoroutine(f)])
        return _ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as error:  # noqa: BLE001 - mirrors TS catch
        accepting_updates = False
        await asyncio.gather(*[f for f in update_events if asyncio.isfuture(f) or asyncio.iscoroutine(f)])
        return _ExecutedToolCallOutcome(
            result=_create_error_tool_result(str(error)),
            is_error=True,
        )
    finally:
        accepting_updates = False


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
) -> _FinalizedToolCallOutcome:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call is not None:
        try:
            after_result = await _maybe_await(
                config.after_tool_call(
                    AfterToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared.tool_call,
                        args=prepared.args,
                        result=result,
                        is_error=is_error,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if after_result is not None:
                if after_result.content is not None:
                    result.content = after_result.content
                if after_result.details is not None:
                    result.details = after_result.details
                if after_result.usage is not None:
                    result.usage = after_result.usage
                if after_result.terminate is not None:
                    result.terminate = after_result.terminate
                if after_result.is_error is not None:
                    is_error = after_result.is_error
        except Exception as error:  # noqa: BLE001 - mirrors TS catch
            result = _create_error_tool_result(str(error))
            is_error = True

    return _FinalizedToolCallOutcome(tool_call=prepared.tool_call, result=result, is_error=is_error)


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    finalized_calls: List[_FinalizedToolCallOutcome] = []
    messages: List[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _maybe_await(
            emit(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                )
            )
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if preparation.kind == "immediate":
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation.result,
                is_error=preparation.is_error,
            )
        else:
            executed = await _execute_prepared_tool_call(preparation, signal, emit)
            finalized = await _finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, signal
            )

        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if signal is not None and signal.aborted:
            break

    return _ExecutedToolCallBatch(
        messages=messages,
        terminate=_should_terminate_tool_batch(finalized_calls),
    )


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    finalized_entries: List[Union[_FinalizedToolCallOutcome, Any]] = []

    for tool_call in tool_calls:
        await _maybe_await(
            emit(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                )
            )
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if preparation.kind == "immediate":
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation.result,
                is_error=preparation.is_error,
            )
            await _emit_tool_execution_end(finalized, emit)
            finalized_entries.append(finalized)
            if signal is not None and signal.aborted:
                break
            continue

        async def _run_entry(prepared: _PreparedToolCall = preparation) -> _FinalizedToolCallOutcome:
            if signal is not None and signal.aborted:
                finalized = _FinalizedToolCallOutcome(
                    tool_call=prepared.tool_call,
                    result=_create_error_tool_result("Operation aborted"),
                    is_error=True,
                )
                await _emit_tool_execution_end(finalized, emit)
                return finalized
            executed = await _execute_prepared_tool_call(prepared, signal, emit)
            finalized = await _finalize_executed_tool_call(
                current_context, assistant_message, prepared, executed, config, signal
            )
            await _emit_tool_execution_end(finalized, emit)
            return finalized

        finalized_entries.append(_run_entry())
        if signal is not None and signal.aborted:
            break

    ordered_finalized_calls: List[_FinalizedToolCallOutcome] = list(
        await asyncio.gather(
            *[entry if asyncio.iscoroutine(entry) else _wrap_value(entry) for entry in finalized_entries]
        )
    )

    messages: List[ToolResultMessage] = []
    for finalized in ordered_finalized_calls:
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)

    return _ExecutedToolCallBatch(
        messages=messages,
        terminate=_should_terminate_tool_batch(ordered_finalized_calls),
    )


async def _wrap_value(value: Any) -> Any:
    return value


def _create_error_tool_result(message: str) -> AgentToolResult:
    from ._pi_ai.types import TextContent

    return AgentToolResult(content=[TextContent(text=message)], details={})


async def _emit_tool_execution_end(finalized: _FinalizedToolCallOutcome, emit: AgentEventSink) -> None:
    await _maybe_await(
        emit(
            ToolExecutionEndEvent(
                tool_call_id=finalized.tool_call.id,
                tool_name=finalized.tool_call.name,
                result=finalized.result,
                is_error=finalized.is_error,
            )
        )
    )


def _create_tool_result_message(finalized: _FinalizedToolCallOutcome) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        # Untyped tools can return results without content; normalize so the
        # null never enters session history or provider payloads.
        content=finalized.result.content or [],
        details=finalized.result.details,
        usage=finalized.result.usage,
        is_error=finalized.is_error,
        timestamp=int(time.time() * 1000),
    )


async def _emit_tool_result_message(tool_result_message: ToolResultMessage, emit: AgentEventSink) -> None:
    await _maybe_await(emit(MessageStartEvent(message=tool_result_message)))
    await _maybe_await(emit(MessageEndEvent(message=tool_result_message)))
