"""Tests ported from ``packages/agent/test/agent.test.ts``."""

from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional

import pytest

from pi_ai.event_stream import EventStream
from pi_ai.transcript import get_current_system_message, to_tool_declaration
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Cost,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolReference,
    Usage,
    UserMessage,
)
from pi_agent_core.agent import Agent, AgentInitialState, AgentOptions
from pi_agent_core import set_default_stream_fn
from pi_agent_core.types import (
    AgentEvent,
    AgentTool,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
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


def create_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def create_assistant_tool_use_message(content: List[ToolCall]) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=create_usage(),
        stop_reason="toolUse",
        timestamp=int(time.time() * 1000),
    )


EMPTY_PARAMETERS: dict = {"type": "object", "properties": {}, "required": []}


def create_tool(name: str) -> AgentTool:
    async def execute(_tool_call_id: str, _params: Any, _signal: Any, _on_update: Any) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=name)], details={})

    return AgentTool(
        name=name,
        label=name,
        description=f"{name} tool",
        parameters=EMPTY_PARAMETERS,
        execute=execute,
    )


def unused_stream_function(*_args: Any) -> Any:
    raise AssertionError("Unexpected stream call")


def make_stream_fn(messages_by_call: List[AssistantMessage], observer: Optional[Any] = None):
    state = {"call_index": 0}

    def stream_fn(_model: Any, context: Any = None, _options: Any = None) -> MockAssistantStream:
        stream = MockAssistantStream()
        call_index = state["call_index"]

        async def _push() -> None:
            if observer is not None:
                observer(call_index, context)
            message = messages_by_call[call_index]
            stream.push(AssistantMessageEvent(type="done", reason=message.stop_reason, message=message))
            state["call_index"] += 1

        asyncio.get_running_loop().create_task(_push())
        return stream

    return stream_fn, state


# ---------------------------------------------------------------------------
# Construction and state
# ---------------------------------------------------------------------------


def test_creates_agent_with_default_state() -> None:
    agent = Agent(AgentOptions(stream_fn=unused_stream_function))
    assert agent.state is not None
    assert agent.state.model is not None
    assert agent.state.thinking_level == "off"
    assert agent.state.tools == []
    assert agent.state.messages == []
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == set()
    assert agent.state.error_message is None


def test_creates_agent_with_custom_initial_state() -> None:
    from pi_ai.types import Model, ModelCost

    custom_model = Model(
        id="gpt-4o-mini",
        name="GPT-4o mini",
        api="openai-completions",
        provider="openai",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=16384,
    )
    agent = Agent(
        AgentOptions(
            stream_fn=unused_stream_function,
            initial_state=AgentInitialState(
                system_prompt="You are a helpful assistant.",
                model=custom_model,
                thinking_level="low",
            ),
        )
    )

    assert len(agent.state.messages) == 1
    initial = agent.state.messages[0]
    assert isinstance(initial, SystemMessage)
    assert initial.content == "You are a helpful assistant."
    assert initial.timestamp == 0
    assert agent.state.model is custom_model
    assert agent.state.thinking_level == "low"


def test_converts_initial_prompt_and_tools_into_transcript_state() -> None:
    tool = create_tool("echo")
    agent = Agent(
        AgentOptions(
            stream_fn=unused_stream_function,
            initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[tool]),
        )
    )

    initial = agent.state.messages[0]
    assert isinstance(initial, SystemMessage)
    assert initial.content == "You are helpful."
    assert [t.name for t in (initial.tools_added or [])] == ["echo"]


async def test_declares_tool_loadout_changes_before_next_request() -> None:
    first = create_tool("first")
    second = create_tool("second")
    requests: List[List[str]] = []

    def observer(_index: int, context: Any) -> None:
        entries: List[str] = []
        for message in context.messages:
            if getattr(message, "role", None) == "system":
                entries.append(f"+{','.join(t.name for t in (message.tools_added or []))}")
                entries.append(f"-{','.join(t.name for t in (message.tools_removed or []))}")
        requests.append(entries)

    stream_fn, _state = make_stream_fn(
        [create_assistant_message("done"), create_assistant_message("done"), create_assistant_message("done")],
        observer=observer,
    )
    agent = Agent(
        AgentOptions(
            stream_fn=stream_fn,
            initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[first]),
        )
    )

    await agent.prompt("one")
    agent.state.tools = [second]
    await agent.prompt("two")
    await agent.prompt("three")

    assert requests == [
        ["+first", "-"],
        ["+first", "-", "+second", "-first"],
        ["+first", "-", "+second", "-first"],
    ]
    update = next(
        (
            m
            for m in agent.state.messages
            if isinstance(m, SystemMessage) and m.tools_removed
        ),
        None,
    )
    assert update is not None
    assert update.content == ""
    assert [t.name for t in (update.tools_added or [])] == ["second"]
    assert [t.name for t in (update.tools_removed or [])] == ["first"]
    initial = agent.state.messages[0]
    assert isinstance(initial, SystemMessage)
    added = (initial.tools_added or [])[0]
    assert not hasattr(added, "execute")


async def test_merges_tool_changes_into_pending_system_message() -> None:
    tool = create_tool("echo")
    observed_counts: List[int] = []

    def observer(_index: int, context: Any) -> None:
        observed_counts.append(
            len([m for m in context.messages if getattr(m, "role", None) == "system"])
        )

    stream_fn, _state = make_stream_fn([create_assistant_message("done")], observer=observer)
    agent = Agent(
        AgentOptions(
            stream_fn=stream_fn,
            initial_state=AgentInitialState(system_prompt="You are helpful."),
        )
    )

    agent.state.tools = [tool]
    await agent.prompt(
        [
            SystemMessage(content="", sections={"skills": "<skills>x</skills>"}, timestamp=1),
            UserMessage(content="hi", timestamp=2),
        ]
    )

    assert observed_counts == [2]
    merged = agent.state.messages[1]
    assert isinstance(merged, SystemMessage)
    assert merged.content == ""
    assert merged.sections == {"skills": "<skills>x</skills>"}
    assert [t.name for t in (merged.tools_added or [])] == ["echo"]
    assert merged.timestamp == 1


async def test_rewrites_pending_tool_declarations_to_match_executable_set() -> None:
    stream_fn, _state = make_stream_fn([create_assistant_message("done")])
    agent = Agent(
        AgentOptions(
            stream_fn=stream_fn,
            initial_state=AgentInitialState(system_prompt="You are helpful.", tools=[create_tool("first")]),
        )
    )

    pending = SystemMessage(content="", sections={"note": "<note>x</note>"}, timestamp=1)
    pending.tools_added = [to_tool_declaration(create_tool("second"))]
    pending.tools_removed = [ToolReference(name="first")]
    await agent.prompt([pending, UserMessage(content="hi", timestamp=2)])

    rewritten = agent.state.messages[1]
    assert isinstance(rewritten, SystemMessage)
    assert rewritten.content == ""
    assert rewritten.sections == {"note": "<note>x</note>"}
    assert rewritten.tools_added is None
    assert rewritten.tools_removed is None
    assert rewritten.timestamp == 1
    current = get_current_system_message(agent.state.messages)
    assert current is not None
    assert [t.name for t in (current.tools_added or [])] == ["first"]


def test_restores_transcript_baseline_when_reset() -> None:
    tool = create_tool("echo")
    agent = Agent(
        AgentOptions(
            stream_fn=unused_stream_function,
            initial_state=AgentInitialState(
                system_prompt="You are helpful.",
                tools=[tool],
                messages=[UserMessage(content="old", timestamp=1)],
            ),
        )
    )

    agent.reset()

    assert len(agent.state.messages) == 1
    initial = agent.state.messages[0]
    assert isinstance(initial, SystemMessage)
    assert initial.content == "You are helpful."
    assert [t.name for t in (initial.tools_added or [])] == ["echo"]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


async def test_prompt_emits_documented_event_sequence() -> None:
    stream_fn, _state = make_stream_fn([create_assistant_message("Hi")])
    agent = Agent(AgentOptions(stream_fn=stream_fn))

    events: List[AgentEvent] = []
    agent.subscribe(lambda event, _signal: events.append(event))

    await agent.prompt("Hello")

    event_types = [event.type for event in events]
    assert event_types == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert isinstance(events[-2], TurnEndEvent)
    assert events[-1].messages[-1].role == "assistant"

    final_state_messages = list(agent.state.messages)
    roles = [getattr(m, "role", None) for m in final_state_messages]
    assert roles == ["user", "assistant"]


async def test_message_update_events_stream_deltas() -> None:
    def stream_fn(_model: Any, _context: Any = None, _options: Any = None) -> MockAssistantStream:
        stream = MockAssistantStream()

        async def _push() -> None:
            message = create_assistant_message("Hi there!")
            partial = AssistantMessage(
                content=[],
                api=message.api,
                provider=message.provider,
                model=message.model,
                usage=create_usage(),
                stop_reason="pending",
                timestamp=message.timestamp,
            )
            stream.push(AssistantMessageEvent(type="start", partial=partial))
            partial.content.append(TextContent(text="Hi"))
            stream.push(
                AssistantMessageEvent(
                    type="text_delta", content_index=0, delta="Hi", partial=partial
                )
            )
            partial.content[0].text = "Hi there!"
            stream.push(
                AssistantMessageEvent(
                    type="text_delta", content_index=0, delta=" there!", partial=partial
                )
            )
            stream.push(AssistantMessageEvent(type="done", reason="stop", message=message))

        asyncio.get_running_loop().create_task(_push())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn))
    events: List[AgentEvent] = []
    agent.subscribe(lambda event, _signal: events.append(event))

    await agent.prompt("Hello")

    updates = [e for e in events if isinstance(e, MessageUpdateEvent)]
    assert len(updates) == 2
    assert updates[0].assistant_message_event.delta == "Hi"
    assert updates[1].assistant_message_event.delta == " there!"
    assert agent.state.streaming_message is None


# ---------------------------------------------------------------------------
# Tool execution through Agent
# ---------------------------------------------------------------------------


async def test_executes_tools_and_continues_until_stop() -> None:
    executed: List[str] = []

    async def execute(_tool_call_id: str, params: Any, _signal: Any, _on_update: Any) -> AgentToolResult:
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={})

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        execute=execute,
    )

    stream_fn, _state = make_stream_fn(
        [
            create_assistant_tool_use_message([ToolCall(id="t1", name="echo", arguments={"value": "hello"})]),
            create_assistant_message("done"),
        ]
    )
    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(tools=[tool])))

    events: List[AgentEvent] = []
    agent.subscribe(lambda event, _signal: events.append(event))
    await agent.prompt("echo something")

    assert executed == ["hello"]
    tool_start = next((e for e in events if isinstance(e, ToolExecutionStartEvent)), None)
    tool_end = next((e for e in events if isinstance(e, ToolExecutionEndEvent)), None)
    assert tool_start is not None and tool_start.tool_call_id == "t1"
    assert tool_end is not None and tool_end.is_error is False

    roles = [getattr(m, "role", None) for m in agent.state.messages]
    assert roles == ["system", "user", "assistant", "toolResult", "assistant"]


# ---------------------------------------------------------------------------
# Steering / follow-up queues
# ---------------------------------------------------------------------------


async def test_steer_injects_message_after_tool_batch() -> None:
    executed: List[str] = []

    async def execute(_tool_call_id: str, params: Any, _signal: Any, _on_update: Any) -> AgentToolResult:
        executed.append(params["value"])
        await asyncio.sleep(0.01)
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        execute=execute,
    )

    stream_fn, _state = make_stream_fn(
        [
            create_assistant_tool_use_message(
                [ToolCall(id="t1", name="echo", arguments={"value": "first"})]
            ),
            create_assistant_message("done"),
        ]
    )
    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(tools=[tool])))

    def on_event(event: AgentEvent, _signal: Any) -> None:
        if isinstance(event, ToolExecutionStartEvent) and event.tool_call_id == "t1":
            agent.steer(UserMessage(content="interrupt", timestamp=int(time.time() * 1000)))

    agent.subscribe(on_event)
    await agent.prompt("start")

    assert executed == ["first"]

    def _user_text(message: Any) -> Optional[str]:
        if not isinstance(message, UserMessage):
            return None
        if isinstance(message.content, str):
            return message.content
        return "".join(b.text for b in message.content if b.type == "text") or None

    contents = [t for m in agent.state.messages if (t := _user_text(m)) is not None]
    assert contents == ["start", "interrupt"]


async def test_follow_up_runs_after_agent_would_stop() -> None:
    stream_fn, _state = make_stream_fn(
        [create_assistant_message("first answer"), create_assistant_message("second answer")]
    )
    agent = Agent(AgentOptions(stream_fn=stream_fn))

    async def prompt_task() -> None:
        await agent.prompt("first")

    task = asyncio.get_running_loop().create_task(prompt_task())
    agent.follow_up(UserMessage(content="second", timestamp=int(time.time() * 1000)))
    await task

    texts = [
        m.content[0].text
        for m in agent.state.messages
        if isinstance(m, AssistantMessage) and m.content
    ]
    assert texts == ["first answer", "second answer"]


# ---------------------------------------------------------------------------
# Error handling and run control
# ---------------------------------------------------------------------------


async def test_stream_error_produces_error_assistant_message() -> None:
    def stream_fn(_model: Any, _context: Any = None, _options: Any = None) -> MockAssistantStream:
        stream = MockAssistantStream()

        async def _push() -> None:
            error = create_assistant_message("")
            error.stop_reason = "error"
            error.error_message = "boom"
            stream.push(AssistantMessageEvent(type="error", reason="error", error=error))

        asyncio.get_running_loop().create_task(_push())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn))
    events: List[AgentEvent] = []
    agent.subscribe(lambda event, _signal: events.append(event))
    await agent.prompt("Hello")

    assert agent.state.error_message == "boom"
    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.message.stop_reason == "error"


async def test_rejects_concurrent_prompts() -> None:
    release = asyncio.Event()

    def stream_fn(_model: Any, _context: Any = None, _options: Any = None) -> MockAssistantStream:
        stream = MockAssistantStream()

        async def _push() -> None:
            await release.wait()
            stream.push(
                AssistantMessageEvent(type="done", reason="stop", message=create_assistant_message("done"))
            )

        asyncio.get_running_loop().create_task(_push())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn))
    task = asyncio.get_running_loop().create_task(agent.prompt("first"))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("second")
    release.set()
    await task
    await agent.wait_for_idle()


async def test_reset_rejects_while_processing() -> None:
    release = asyncio.Event()

    def stream_fn(_model: Any, _context: Any = None, _options: Any = None) -> MockAssistantStream:
        stream = MockAssistantStream()

        async def _push() -> None:
            await release.wait()
            stream.push(
                AssistantMessageEvent(type="done", reason="stop", message=create_assistant_message("done"))
            )

        asyncio.get_running_loop().create_task(_push())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn))
    task = asyncio.get_running_loop().create_task(agent.prompt("first"))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already processing"):
        agent.reset()
    release.set()
    await task


async def test_continue_requires_tail_message() -> None:
    stream_fn, _state = make_stream_fn([create_assistant_message("done")])
    agent = Agent(AgentOptions(stream_fn=stream_fn))
    with pytest.raises(RuntimeError, match="No messages to continue from"):
        await agent.continue_()

    await agent.prompt("Hello")
    # Last message is assistant -> rejected
    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        await agent.continue_()


async def test_default_stream_fn_used_when_omitted() -> None:
    calls = 0

    def default_stream_fn(_model: Any = None, _context: Any = None, _options: Any = None) -> MockAssistantStream:
        nonlocal calls
        calls += 1
        stream = MockAssistantStream()

        async def _push() -> None:
            stream.push(
                AssistantMessageEvent(
                    type="done", reason="stop", message=create_assistant_message("fallback")
                )
            )

        asyncio.get_running_loop().create_task(_push())
        return stream

    set_default_stream_fn(default_stream_fn)
    try:
        agent = Agent(AgentOptions())
        await agent.prompt("Hello")
        assert calls == 1
    finally:
        set_default_stream_fn(None)
