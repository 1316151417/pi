"""Agent ported from ``packages/agent/src/agent.ts``."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Callable, List, Optional, Sequence, Set, Union

from pi_ai.abort import AbortController, AbortSignal
from pi_ai.transcript import (
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    to_tool_declaration,
)
from pi_ai.types import (
    AgentMessage,
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    ThinkingBudgets,
    Transport,
    Usage,
    UserMessage,
)
from .agent_loop import run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    PrepareNextTurnContext,
    QueueMode,
    ShouldStopAfterTurnContext,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
    ToolResultMessage,
    TurnEndEvent,
)

__all__ = ["Agent", "AgentOptions", "AgentInitialState", "QueueMode"]


def _default_convert_to_llm(messages: List[AgentMessage]) -> List[Message]:
    return [
        message
        for message in messages
        if getattr(message, "role", None) in ("system", "user", "assistant", "toolResult")
    ]


def _empty_usage() -> Usage:
    return Usage(cost=Cost())


def _default_model() -> Model:
    return Model(
        id="unknown",
        name="unknown",
        api="unknown",
        provider="unknown",
        base_url="",
        reasoning=False,
        input=[],
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )


class _MutableAgentState:
    def __init__(self, initial_state: Optional["AgentInitialState"] = None) -> None:
        self._tools: List[AgentTool] = list(initial_state.tools) if initial_state and initial_state.tools else []
        self._messages: List[AgentMessage] = (
            list(initial_state.messages) if initial_state and initial_state.messages else []
        )
        initial_message = create_initial_system_message(
            initial_state.system_prompt if initial_state else None,
            [to_tool_declaration(t) for t in self._tools],
        )
        if initial_message is not None and (
            len(self._messages) == 0 or getattr(self._messages[0], "role", None) != "system"
        ):
            self._messages.insert(0, initial_message)

        self.model: Model = (initial_state.model if initial_state and initial_state.model else None) or _default_model()
        self.thinking_level: ThinkingLevel = (
            initial_state.thinking_level if initial_state and initial_state.thinking_level else "off"
        )
        self.is_streaming: bool = False
        self.streaming_message: Optional[AgentMessage] = None
        self.pending_tool_calls: Set[str] = set()
        self.error_message: Optional[str] = None

    @property
    def system_prompt(self) -> str:
        return get_current_system_prompt(self._messages)

    @property
    def tools(self) -> List[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, next_tools: List[AgentTool]) -> None:
        self._tools = list(next_tools)

    @property
    def messages(self) -> List[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, next_messages: List[AgentMessage]) -> None:
        self._messages = list(next_messages)


class AgentInitialState:
    """Initial state for :class:`Agent`."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[Model] = None,
        thinking_level: ThinkingLevel = "off",
        tools: Optional[List[AgentTool]] = None,
        messages: Optional[List[AgentMessage]] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self.thinking_level = thinking_level
        self.tools = tools
        self.messages = messages


class AgentOptions:
    """Options for constructing an :class:`Agent`."""

    def __init__(
        self,
        initial_state: Optional[AgentInitialState] = None,
        convert_to_llm: Optional[Callable[[List[AgentMessage]], Union[List[Message], Any]]] = None,
        transform_context: Optional[
            Callable[[List[AgentMessage], Optional[AbortSignal]], Any]
        ] = None,
        stream_fn: Optional[StreamFn] = None,
        get_api_key: Optional[Callable[[str], Any]] = None,
        on_payload: Optional[Callable[[Any, Model], Any]] = None,
        on_response: Optional[Callable[[Any, Model], Any]] = None,
        before_tool_call: Optional[Callable[[BeforeToolCallContext, Optional[AbortSignal]], Any]] = None,
        after_tool_call: Optional[Callable[[AfterToolCallContext, Optional[AbortSignal]], Any]] = None,
        should_stop_after_turn: Optional[
            Callable[[ShouldStopAfterTurnContext, Optional[AbortSignal]], Any]
        ] = None,
        prepare_next_turn: Optional[Callable[[Optional[AbortSignal]], Any]] = None,
        prepare_next_turn_with_context: Optional[
            Callable[[PrepareNextTurnContext, Optional[AbortSignal]], Any]
        ] = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        session_id: Optional[str] = None,
        thinking_budgets: Optional[ThinkingBudgets] = None,
        transport: Transport = "auto",
        max_retry_delay_ms: Optional[int] = None,
        tool_execution: ToolExecutionMode = "parallel",
    ) -> None:
        self.initial_state = initial_state
        self.convert_to_llm = convert_to_llm
        self.transform_context = transform_context
        self.stream_fn = stream_fn
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.should_stop_after_turn = should_stop_after_turn
        self.prepare_next_turn = prepare_next_turn
        self.prepare_next_turn_with_context = prepare_next_turn_with_context
        self.steering_mode = steering_mode
        self.follow_up_mode = follow_up_mode
        self.session_id = session_id
        self.thinking_budgets = thinking_budgets
        self.transport = transport
        self.max_retry_delay_ms = max_retry_delay_ms
        self.tool_execution = tool_execution


class _PendingMessageQueue:
    def __init__(self, mode: QueueMode) -> None:
        self.mode = mode
        self._messages: List[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> List[AgentMessage]:
        if self.mode == "all":
            drained = list(self._messages)
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


class _ActiveRun:
    def __init__(self) -> None:
        self.promise: asyncio.Future = asyncio.get_running_loop().create_future()
        self.abort_controller = AbortController()

    def resolve(self) -> None:
        if not self.promise.done():
            self.promise.set_result(None)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


class Agent:
    """Stateful wrapper around the low-level agent loop.

    ``Agent`` owns the current transcript, emits lifecycle events, executes tools,
    and exposes queueing APIs for steering and follow-up messages.
    """

    def __init__(self, options: Optional[AgentOptions] = None) -> None:
        runtime_options = options or AgentOptions()
        self._state = _MutableAgentState(runtime_options.initial_state)
        self.convert_to_llm: Callable[[List[AgentMessage]], Union[List[Message], Any]] = (
            runtime_options.convert_to_llm or _default_convert_to_llm
        )
        self.transform_context = runtime_options.transform_context
        self.stream_function: StreamFn = runtime_options.stream_fn or get_default_stream_fn()
        self.get_api_key = runtime_options.get_api_key
        self.on_payload = runtime_options.on_payload
        self.on_response = runtime_options.on_response
        self.before_tool_call = runtime_options.before_tool_call
        self.after_tool_call = runtime_options.after_tool_call
        self.should_stop_after_turn = runtime_options.should_stop_after_turn
        self.prepare_next_turn = runtime_options.prepare_next_turn
        self.prepare_next_turn_with_context = runtime_options.prepare_next_turn_with_context
        self._steering_queue = _PendingMessageQueue(runtime_options.steering_mode or "one-at-a-time")
        self._follow_up_queue = _PendingMessageQueue(runtime_options.follow_up_mode or "one-at-a-time")
        self._listeners: List[Callable[[AgentEvent, AbortSignal], Any]] = []
        self._active_run: Optional[_ActiveRun] = None
        self.session_id: Optional[str] = runtime_options.session_id
        self.thinking_budgets: Optional[ThinkingBudgets] = runtime_options.thinking_budgets
        self.transport: Transport = runtime_options.transport or "auto"
        self.max_retry_delay_ms: Optional[int] = runtime_options.max_retry_delay_ms
        self.tool_execution: ToolExecutionMode = runtime_options.tool_execution or "parallel"

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe(self, listener: Callable[[AgentEvent, AbortSignal], Any]) -> Callable[[], None]:
        """Subscribe to agent lifecycle events. Returns an unsubscribe callable."""
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> _MutableAgentState:
        """Current agent state."""
        return self._state

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    # ------------------------------------------------------------------
    # Queues
    # ------------------------------------------------------------------

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to be injected after the current assistant turn finishes."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    @property
    def signal(self) -> Optional[AbortSignal]:
        """Active abort signal for the current run, if any."""
        return self._active_run.abort_controller.signal if self._active_run else None

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        if self._active_run is not None:
            self._active_run.abort_controller.abort()

    async def wait_for_idle(self) -> None:
        """Resolve when the current run and all awaited event listeners have finished."""
        if self._active_run is not None:
            await self._active_run.promise

    def reset(self) -> None:
        """Clear conversation state and queues while retaining the replayed prompt/tool baseline."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")

        baseline = get_current_system_message(self._state.messages)
        self._state.messages = [baseline] if baseline is not None else []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    async def prompt(
        self,
        input: Union[str, AgentMessage, Sequence[AgentMessage]],
        images: Optional[List[ImageContent]] = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue "
                "messages, or wait for completion."
            )
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_(self) -> None:
        """Continue from the current transcript. The last message must be a user or tool-result message."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        last_message = self._state.messages[-1] if self._state.messages else None
        if last_message is None or all(
            getattr(message, "role", None) == "system" for message in self._state.messages
        ):
            raise RuntimeError("No messages to continue from")

        if getattr(last_message, "role", None) == "assistant":
            queued_steering = self._steering_queue.drain()
            if len(queued_steering) > 0:
                await self._run_prompt_messages(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow_ups = self._follow_up_queue.drain()
            if len(queued_follow_ups) > 0:
                await self._run_prompt_messages(queued_follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # Python alias: `continue` is a reserved keyword.
    continue_run = continue_

    def _normalize_prompt_input(
        self,
        input: Union[str, AgentMessage, Sequence[AgentMessage]],
        images: Optional[List[ImageContent]],
    ) -> List[AgentMessage]:
        if isinstance(input, list):
            return list(input)
        if isinstance(input, str):
            content: List[Union[TextContent, ImageContent]] = [TextContent(text=input)]
            if images:
                content.extend(images)
            return [UserMessage(content=content, timestamp=int(time.time() * 1000))]
        return [input]

    async def _run_prompt_messages(
        self, messages: List[AgentMessage], options: Optional[dict] = None
    ) -> None:
        options = options or {}

        async def _executor(signal: AbortSignal) -> None:
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(options),
                self._process_events,
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(_executor)

    async def _run_continuation(self) -> None:
        async def _executor(signal: AbortSignal) -> None:
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                self._process_events,
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(_executor)

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(messages=list(self._state.messages), tools=list(self._state.tools))

    def _create_loop_config(self, options: Optional[dict] = None) -> AgentLoopConfig:
        options = options or {}
        skip_initial_steering_poll = options.get("skip_initial_steering_poll") is True
        should_stop_after_turn = self.should_stop_after_turn
        loop_config = AgentLoopConfig(
            model=self._state.model,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            transport=self.transport,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
        )
        if should_stop_after_turn is not None:

            async def _should_stop_after_turn(context: ShouldStopAfterTurnContext) -> bool:
                return bool(await _maybe_await(should_stop_after_turn(context, self.signal)))

            loop_config.should_stop_after_turn = _should_stop_after_turn

        if self.prepare_next_turn_with_context is not None or self.prepare_next_turn is not None:

            async def _prepare_next_turn(context: PrepareNextTurnContext) -> Optional[AgentLoopTurnUpdate]:
                if self.prepare_next_turn_with_context is not None:
                    return await _maybe_await(self.prepare_next_turn_with_context(context, self.signal))
                if self.prepare_next_turn is not None:
                    return await _maybe_await(self.prepare_next_turn(self.signal))
                return None

            loop_config.prepare_next_turn = _prepare_next_turn

        async def _get_steering_messages() -> List[AgentMessage]:
            nonlocal skip_initial_steering_poll
            if skip_initial_steering_poll:
                skip_initial_steering_poll = False
                return []
            return self._steering_queue.drain()

        async def _get_follow_up_messages() -> List[AgentMessage]:
            return self._follow_up_queue.drain()

        loop_config.get_steering_messages = _get_steering_messages
        loop_config.get_follow_up_messages = _get_follow_up_messages
        loop_config.before_tool_call = self.before_tool_call
        loop_config.after_tool_call = self.after_tool_call
        return loop_config

    async def _run_with_lifecycle(self, executor: Callable[[AbortSignal], Any]) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")

        active_run = _ActiveRun()
        self._active_run = active_run

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(active_run.abort_controller.signal)
        except Exception as error:  # noqa: BLE001 - mirrors TS catch
            await self._handle_run_failure(error, active_run.abort_controller.signal.aborted)
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: BaseException, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=_empty_usage(),
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=int(time.time() * 1000),
        )
        await self._process_events(MessageStartEvent(message=failure_message))
        await self._process_events(MessageEndEvent(message=failure_message))
        await self._process_events(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        if self._active_run is not None:
            self._active_run.resolve()
            self._active_run = None

    async def _process_events(self, event: AgentEvent) -> None:
        """Reduce internal state for a loop event, then await listeners."""
        from .types import (
            MessageUpdateEvent,
            ToolExecutionEndEvent,
            ToolExecutionStartEvent,
        )

        if isinstance(event, MessageStartEvent):
            self._state.streaming_message = event.message
        elif isinstance(event, MessageUpdateEvent):
            self._state.streaming_message = event.message
        elif isinstance(event, MessageEndEvent):
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif isinstance(event, ToolExecutionStartEvent):
            pending = set(self._state.pending_tool_calls)
            pending.add(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif isinstance(event, ToolExecutionEndEvent):
            pending = set(self._state.pending_tool_calls)
            pending.discard(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif isinstance(event, TurnEndEvent):
            if (
                isinstance(event.message, AssistantMessage)
                and event.message.error_message is not None
            ):
                self._state.error_message = event.message.error_message
        elif isinstance(event, AgentEndEvent):
            self._state.streaming_message = None

        signal = self._active_run.abort_controller.signal if self._active_run else None
        if signal is None:
            raise RuntimeError("Agent listener invoked outside active run")
        for listener in list(self._listeners):
            await _maybe_await(listener(event, signal))
