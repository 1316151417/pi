"""Tests ported from ``packages/agent/test/agent-loop.test.ts``."""

from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional

import pytest

from pi_ai.event_stream import EventStream
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Cost,
    Model,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_agent_core.agent_loop import agent_loop, agent_loop_continue
from pi_agent_core import set_default_stream_fn
from pi_agent_core.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    TurnEndEvent,
)


class MockAssistantStream(EventStream):
    def __init__(self) -> None:
        super().__init__(
            lambda event: event.type in ("done", "error"),
            lambda event: event.message if event.type == "done" else event.error,
        )


def create_usage() -> Usage:
    return Usage(cost=Cost())


def create_model() -> Model:
    return Model(
        id="mock",
        name="mock",
        api="openai-responses",
        provider="openai",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=Model.__dataclass_fields__["cost"].default_factory(),
        context_window=8192,
        max_tokens=2048,
    )


def create_assistant_message(
    content: List[Any], stop_reason: str = "stop"
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=int(time.time() * 1000))


def identity_converter(messages: List[Any]) -> List[Any]:
    return [
        m
        for m in messages
        if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")
    ]


def make_stream_fn(messages_by_call: List[AssistantMessage], on_call: Optional[Any] = None):
    """Scripted stream fn: returns the Nth scripted response on the Nth call."""
    state = {"call_index": 0}

    def stream_fn(_model: Any, context: Any = None, options: Any = None) -> MockAssistantStream:
        stream = MockAssistantStream()
        call_index = state["call_index"]

        async def _push() -> None:
            if on_call is not None:
                result = on_call(call_index, context)
                if asyncio.iscoroutine(result):
                    await result
            message = messages_by_call[call_index]
            stream.push(AssistantMessageEvent(type="done", reason=message.stop_reason, message=message))
            state["call_index"] += 1

        asyncio.get_running_loop().create_task(_push())
        return stream

    return stream_fn, state


# ---------------------------------------------------------------------------
# Default stream function compatibility
# ---------------------------------------------------------------------------


async def test_uses_configured_default_when_stream_fn_omitted() -> None:
    calls = 0

    def default_stream_fn(_model: Any = None, _context: Any = None, _options: Any = None) -> MockAssistantStream:
        nonlocal calls
        calls += 1
        stream = MockAssistantStream()

        async def _push() -> None:
            stream.push(
                AssistantMessageEvent(
                    type="done",
                    reason="stop",
                    message=create_assistant_message([TextContent(text="fallback")]),
                )
            )

        asyncio.get_running_loop().create_task(_push())
        return stream

    set_default_stream_fn(default_stream_fn)
    try:
        context = AgentContext(messages=[], tools=[])
        config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
        stream = agent_loop([create_user_message("Hello")], context, config, None, None)
        await stream.result
        assert calls == 1
    finally:
        set_default_stream_fn(None)


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


async def test_emits_events_with_agent_message_types() -> None:
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    stream_fn, _state = make_stream_fn([create_assistant_message([TextContent(text="Hi there!")])])

    events: List[AgentEvent] = []
    stream = agent_loop([create_user_message("Hello")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    messages = await stream.result

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    event_types = [event.type for event in events]
    assert "agent_start" in event_types
    assert "turn_start" in event_types
    assert "message_start" in event_types
    assert "message_end" in event_types
    assert "turn_end" in event_types
    assert "agent_end" in event_types


async def test_builds_provider_context_exclusively_from_transcript_messages() -> None:
    from pi_ai.types import SystemMessage

    initial_system = SystemMessage(content="Transcript prompt", timestamp=1)
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    observed: List[Any] = []

    def on_call(_index: int, provider_context: Any) -> None:
        observed.append(provider_context)

    stream_fn, _state = make_stream_fn(
        [create_assistant_message([TextContent(text="done")])], on_call=on_call
    )
    stream = agent_loop([initial_system, create_user_message("Hello")], context, config, None, stream_fn)
    await stream.result

    provider_context = observed[0]
    assert list(vars(provider_context).keys()) == ["messages"]
    assert provider_context.messages[0] is initial_system


async def test_handles_custom_message_types_via_convert_to_llm() -> None:
    class CustomNotification:
        def __init__(self, text: str) -> None:
            self.role = "notification"
            self.text = text
            self.timestamp = int(time.time() * 1000)

    notification = CustomNotification("This is a notification")
    context = AgentContext(messages=[notification], tools=[])
    converted_messages: List[Any] = []

    def convert_to_llm(messages: List[Any]) -> List[Any]:
        converted = [m for m in messages if getattr(m, "role", None) != "notification"]
        converted = [m for m in converted if getattr(m, "role", None) in ("user", "assistant", "toolResult")]
        converted_messages.extend(converted)
        return converted

    config = AgentLoopConfig(model=create_model(), convert_to_llm=convert_to_llm)
    stream_fn, _state = make_stream_fn([create_assistant_message([TextContent(text="Response")])])

    stream = agent_loop([create_user_message("Hello")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert len(converted_messages) == 1
    assert converted_messages[0].role == "user"


async def test_applies_transform_context_before_convert_to_llm() -> None:
    context = AgentContext(
        messages=[
            create_user_message("old message 1"),
            create_assistant_message([TextContent(text="old response 1")]),
            create_user_message("old message 2"),
            create_assistant_message([TextContent(text="old response 2")]),
        ],
        tools=[],
    )

    transformed_messages: List[Any] = []
    converted_messages: List[Any] = []

    async def transform_context(messages: List[Any], _signal: Any = None) -> List[Any]:
        transformed = messages[-2:]
        transformed_messages.extend(transformed)
        return transformed

    def convert_to_llm(messages: List[Any]) -> List[Any]:
        converted = [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "toolResult")]
        converted_messages.extend(converted)
        return converted

    config = AgentLoopConfig(
        model=create_model(),
        transform_context=transform_context,
        convert_to_llm=convert_to_llm,
    )
    stream_fn, _state = make_stream_fn([create_assistant_message([TextContent(text="Response")])])

    stream = agent_loop([create_user_message("new message")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert len(transformed_messages) == 2
    assert len(converted_messages) == 2


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def echo_tool(executed: List[str], usage: Optional[Usage] = None) -> AgentTool:
    async def execute(_tool_call_id: str, params: Any, _signal: Any, _on_update: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            usage=usage,
        )

    return AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=execute,
    )


async def test_handles_tool_calls_and_results() -> None:
    executed: List[str] = []
    tool_usage = Usage(input=1, output=2, cache_read=3, cache_write=4, total_tokens=10,
                       cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1))
    patched_tool_usage = Usage(input=5, output=6, cache_read=7, cache_write=8, total_tokens=26,
                               cost=Cost(input=0.5, output=0.6, cache_read=0.7, cache_write=0.8, total=2.6))
    observed_tool_usage: List[Optional[Usage]] = []
    tool = echo_tool(executed, usage=tool_usage)

    context = AgentContext(messages=[], tools=[tool])

    async def after_tool_call(context: Any, _signal: Any = None) -> Any:
        from pi_agent_core.types import AfterToolCallResult

        observed_tool_usage.append(context.result.usage)
        return AfterToolCallResult(usage=patched_tool_usage)

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        after_tool_call=after_tool_call,
    )

    responses = [
        create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
        create_assistant_message([TextContent(text="done")]),
    ]
    stream_fn, _state = make_stream_fn(responses)

    events: List[AgentEvent] = []
    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    assert executed == ["hello"]

    tool_start = next((e for e in events if e.type == "tool_execution_start"), None)
    tool_end = next((e for e in events if e.type == "tool_execution_end"), None)
    assert tool_start is not None
    assert tool_end is not None
    assert tool_end.is_error is False

    assert observed_tool_usage == [tool_usage]
    messages = await stream.result
    tool_result = next((m for m in messages if getattr(m, "role", None) == "toolResult"), None)
    assert tool_result is not None
    assert tool_result.usage == patched_tool_usage


async def test_does_not_execute_tool_calls_from_length_truncated_message() -> None:
    executed: List[str] = []
    tool = echo_tool(executed)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    responses = [
        create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hel"})], "length"),
        create_assistant_message([TextContent(text="done")]),
    ]
    stream_fn, state = make_stream_fn(responses)

    events: List[AgentEvent] = []
    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    assert executed == []

    tool_end = next((e for e in events if e.type == "tool_execution_end"), None)
    assert tool_end is not None
    assert tool_end.is_error is True
    text = next((c for c in tool_end.result.content if c.type == "text"), None)
    assert text is not None
    assert "output token limit" in text.text

    # The loop continues so the model can re-issue the tool call.
    assert state["call_index"] == 2
    messages = await stream.result
    assert messages[-1].role == "assistant"


async def test_executes_mutated_before_tool_call_args_without_revalidation() -> None:
    executed: List[Any] = []
    tool = echo_tool(executed)
    context = AgentContext(messages=[], tools=[tool])

    async def before_tool_call(context: Any, _signal: Any = None) -> None:
        context.args["value"] = 123

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        before_tool_call=before_tool_call,
    )
    responses = [
        create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
        create_assistant_message([TextContent(text="done")]),
    ]
    stream_fn, _state = make_stream_fn(responses)

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert executed == [123]


async def test_prepares_tool_arguments_for_validation() -> None:
    executed: List[Any] = []

    def prepare_arguments(args: Any) -> Any:
        if not isinstance(args, dict):
            return args
        if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
            return args
        return {
            "edits": [
                *(args.get("edits") or []),
                {"oldText": args["oldText"], "newText": args["newText"]},
            ]
        }

    async def execute(_tool_call_id: str, params: Any, _signal: Any, _on_update: Any) -> AgentToolResult:
        executed.append(params["edits"])
        return AgentToolResult(
            content=[TextContent(text=f"edited {len(params['edits'])}")],
            details={"count": len(params["edits"])},
        )

    edit_tool = AgentTool(
        name="edit",
        label="Edit",
        description="Edit tool",
        parameters={
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                        "required": ["oldText", "newText"],
                    },
                }
            },
            "required": ["edits"],
        },
        prepare_arguments=prepare_arguments,
        execute=execute,
    )

    context = AgentContext(messages=[], tools=[edit_tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    responses = [
        create_assistant_message(
            [ToolCall(id="tool-1", name="edit", arguments={"oldText": "before", "newText": "after"})],
            "toolUse",
        ),
        create_assistant_message([TextContent(text="done")]),
    ]
    stream_fn, _state = make_stream_fn(responses)

    stream = agent_loop([create_user_message("edit something")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert executed == [[{"oldText": "before", "newText": "after"}]]


async def test_tool_execution_end_completion_order_and_source_order_persistence() -> None:
    from pi_agent_core.types import ToolExecutionStartEvent

    executed: List[str] = []
    tool = echo_tool(executed)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        tool_execution="parallel",
    )

    release_first = asyncio.Event()
    state_flags = {"first_resolved": False, "parallel_observed": False}

    original_execute = tool.execute

    async def gated_execute(tool_call_id: str, params: Any, signal: Any, on_update: Any) -> AgentToolResult:
        if params["value"] == "first":
            await release_first.wait()
            state_flags["first_resolved"] = True
        if params["value"] == "second" and not state_flags["first_resolved"]:
            state_flags["parallel_observed"] = True
        return await original_execute(tool_call_id, params, signal, on_update)

    tool.execute = gated_execute

    def on_call(call_index: int, _context: Any) -> None:
        if call_index == 0:
            asyncio.get_running_loop().call_later(0.02, release_first.set)

    responses = [
        create_assistant_message(
            [
                ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
            ],
            "toolUse",
        ),
        create_assistant_message([TextContent(text="done")]),
    ]
    stream_fn, _state = make_stream_fn(responses, on_call=on_call)

    events: List[AgentEvent] = []
    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    tool_execution_end_ids = [e.tool_call_id for e in events if e.type == "tool_execution_end"]
    tool_result_ids = [
        e.message.tool_call_id
        for e in events
        if isinstance(e, MessageEndEvent) and getattr(e.message, "role", None) == "toolResult"
    ]
    turn_tool_result_ids = [
        result.tool_call_id for e in events if isinstance(e, TurnEndEvent) for result in e.tool_results
    ]

    assert state_flags["parallel_observed"] is True
    assert tool_execution_end_ids == ["tool-2", "tool-1"]
    assert tool_result_ids == ["tool-1", "tool-2"]
    assert turn_tool_result_ids == ["tool-1", "tool-2"]


async def test_injects_queued_messages_after_all_tool_calls_complete() -> None:
    executed: List[str] = []
    tool = echo_tool(executed)
    context = AgentContext(messages=[], tools=[tool])

    queued_user_message = create_user_message("interrupt")
    state_flags = {"queued_delivered": False, "saw_interrupt_in_context": False}

    async def get_steering_messages() -> List[Any]:
        if len(executed) >= 1 and not state_flags["queued_delivered"]:
            state_flags["queued_delivered"] = True
            return [queued_user_message]
        return []

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        tool_execution="sequential",
        get_steering_messages=get_steering_messages,
    )

    def on_call(call_index: int, provider_context: Any) -> None:
        if call_index == 1:
            state_flags["saw_interrupt_in_context"] = any(
                getattr(m, "role", None) == "user" and m.content == "interrupt"
                for m in provider_context.messages
            )

    responses = [
        create_assistant_message(
            [
                ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
            ],
            "toolUse",
        ),
        create_assistant_message([TextContent(text="done")]),
    ]
    stream_fn, _state = make_stream_fn(responses, on_call=on_call)

    events: List[AgentEvent] = []
    stream = agent_loop([create_user_message("start")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    assert executed == ["first", "second"]

    tool_ends = [e for e in events if e.type == "tool_execution_end"]
    assert len(tool_ends) == 2
    assert tool_ends[0].is_error is False
    assert tool_ends[1].is_error is False

    event_sequence: List[str] = []
    for event in events:
        if isinstance(event, MessageStartEvent):
            if getattr(event.message, "role", None) == "toolResult":
                event_sequence.append(f"tool:{event.message.tool_call_id}")
            elif getattr(event.message, "role", None) == "user" and isinstance(event.message.content, str):
                event_sequence.append(event.message.content)
    assert "interrupt" in event_sequence
    assert event_sequence.index("tool:tool-1") < event_sequence.index("interrupt")
    assert event_sequence.index("tool:tool-2") < event_sequence.index("interrupt")

    assert state_flags["saw_interrupt_in_context"] is True


async def test_forces_sequential_when_tool_has_sequential_execution_mode() -> None:
    executed: List[str] = []
    tool = echo_tool(executed)
    tool.execution_mode = "sequential"
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    release_first = asyncio.Event()
    state_flags = {"first_resolved": False, "parallel_observed": False}
    original_execute = tool.execute

    async def gated_execute(tool_call_id: str, params: Any, signal: Any, on_update: Any) -> AgentToolResult:
        if params["value"] == "first":
            await release_first.wait()
            state_flags["first_resolved"] = True
        if params["value"] == "second" and not state_flags["first_resolved"]:
            state_flags["parallel_observed"] = True
        return await original_execute(tool_call_id, params, signal, on_update)

    tool.execute = gated_execute

    def on_call(call_index: int, _context: Any) -> None:
        if call_index == 0:
            asyncio.get_running_loop().call_later(0.02, release_first.set)

    responses = [
        create_assistant_message(
            [
                ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
            ],
            "toolUse",
        ),
        create_assistant_message([TextContent(text="done")]),
    ]
    stream_fn, _state = make_stream_fn(responses, on_call=on_call)

    events: List[AgentEvent] = []
    stream = agent_loop([create_user_message("run both")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    assert state_flags["parallel_observed"] is False
    tool_result_ids = [
        e.message.tool_call_id
        for e in events
        if isinstance(e, MessageEndEvent) and getattr(e.message, "role", None) == "toolResult"
    ]
    assert tool_result_ids == ["tool-1", "tool-2"]


async def test_agent_loop_continue_requires_non_assistant_tail() -> None:
    context = AgentContext(messages=[create_assistant_message([TextContent(text="hi")])], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        agent_loop_continue(context, config, None, lambda *_: MockAssistantStream())

    empty_context = AgentContext(messages=[], tools=[])
    with pytest.raises(RuntimeError, match="Cannot continue: no messages in context"):
        agent_loop_continue(empty_context, config, None, lambda *_: MockAssistantStream())
