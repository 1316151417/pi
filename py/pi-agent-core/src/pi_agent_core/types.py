"""Agent-core types ported from ``packages/agent/src/types.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, Set as AbstractSet, TypeVar, Union

from pi_ai.abort import AbortSignal
from pi_ai.types import (
    AgentMessage,
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    JsonObject,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)

__all__ = [
    "StreamFn",
    "ToolExecutionMode",
    "QueueMode",
    "AgentToolCall",
    "BeforeToolCallResult",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "AfterToolCallContext",
    "ShouldStopAfterTurnContext",
    "AgentLoopTurnUpdate",
    "PrepareNextTurnContext",
    "AgentLoopConfig",
    "ThinkingLevel",
    "AgentState",
    "AgentStateProtocol",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "AgentTool",
    "AgentContext",
    "AgentEvent",
    "AgentEventSink",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
]

T = TypeVar("T")

#: Stream function used by the agent loop. ``Models.stream_simple`` satisfies this shape.
StreamFn = Callable[
    [Model, TranscriptContext, Optional[SimpleStreamOptions]],
    Union[Any, Awaitable[Any]],
]

ToolExecutionMode = str  # "sequential" | "parallel"
QueueMode = str  # "all" | "one-at-a-time"
ThinkingLevel = str  # "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"

AgentToolCall = Any  # ToolCall content block from an assistant message


@dataclass
class BeforeToolCallResult:
    """Result returned from ``before_tool_call``."""

    block: Optional[bool] = None
    reason: Optional[str] = None
    terminate: Optional[bool] = None


@dataclass
class AfterToolCallResult:
    """Partial override returned from ``after_tool_call``."""

    content: Optional[List[Union[TextContent, ImageContent]]] = None
    details: Any = None
    is_error: Optional[bool] = None
    usage: Optional[Usage] = None
    terminate: Optional[bool] = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: "AgentContext"


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: "AgentToolResult"
    is_error: bool
    context: "AgentContext"


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: List[ToolResultMessage]
    context: "AgentContext"
    new_messages: List[AgentMessage]


@dataclass
class AgentLoopTurnUpdate:
    """Replacement runtime state used by the agent loop before starting another request."""

    context: Optional["AgentContext"] = None
    messages: Optional[List[AgentMessage]] = None
    model: Optional[Model] = None
    thinking_level: Optional[ThinkingLevel] = None


PrepareNextTurnContext = ShouldStopAfterTurnContext


@dataclass
class AgentLoopConfig(SimpleStreamOptions):
    """Loop configuration; carries stream options plus loop callbacks."""

    model: Model = None  # type: ignore[assignment]
    convert_to_llm: Callable[[List[AgentMessage]], Union[List[Message], Awaitable[List[Message]]]] = None  # type: ignore[assignment]
    transform_context: Optional[
        Callable[[List[AgentMessage], Optional[AbortSignal]], Awaitable[List[AgentMessage]]]
    ] = None
    get_api_key: Optional[Callable[[str], Union[Optional[str], Awaitable[Optional[str]]]]] = None
    should_stop_after_turn: Optional[
        Callable[[ShouldStopAfterTurnContext], Union[bool, Awaitable[bool]]]
    ] = None
    prepare_next_turn: Optional[
        Callable[[PrepareNextTurnContext], Union[Optional[AgentLoopTurnUpdate], Awaitable[Optional[AgentLoopTurnUpdate]]]]
    ] = None
    get_steering_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None
    get_follow_up_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: Optional[
        Callable[
            [BeforeToolCallContext, Optional[AbortSignal]],
            Union[Optional[BeforeToolCallResult], Awaitable[Optional[BeforeToolCallResult]]],
        ]
    ] = None
    after_tool_call: Optional[
        Callable[
            [AfterToolCallContext, Optional[AbortSignal]],
            Union[Optional[AfterToolCallResult], Awaitable[Optional[AfterToolCallResult]]],
        ]
    ] = None


@dataclass
class AgentToolResult:
    """Final or partial result produced by a tool."""

    content: List[Union[TextContent, ImageContent]] = field(default_factory=list)
    details: Any = None
    usage: Optional[Usage] = None
    terminate: Optional[bool] = None


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class AgentStateProtocol(Protocol):
    """Public agent state interface (TS ``AgentState``); implemented by ``Agent.state``."""

    @property
    def system_prompt(self) -> str: ...

    model: Any
    thinking_level: ThinkingLevel

    @property
    def tools(self) -> List["AgentTool"]: ...

    @tools.setter
    def tools(self, tools: List["AgentTool"]) -> None: ...

    @property
    def messages(self) -> List[AgentMessage]: ...

    @messages.setter
    def messages(self, messages: List[AgentMessage]) -> None: ...

    @property
    def is_streaming(self) -> bool: ...

    @property
    def streaming_message(self) -> Optional[AgentMessage]: ...

    @property
    def pending_tool_calls(self) -> AbstractSet[str]: ...

    @property
    def error_message(self) -> Optional[str]: ...


#: Public agent state type. TS declares an interface; Python exposes the runtime
#: class from :mod:`pi_agent_core.agent` and this protocol for typing.
AgentState = AgentStateProtocol


@dataclass
class AgentTool(Tool):
    """Tool definition used by the agent runtime."""

    label: str = ""
    prepare_arguments: Optional[Callable[[Any], Any]] = None
    execute: Optional[
        Callable[
            [str, Any, Optional[AbortSignal], Optional[AgentToolUpdateCallback]],
            Awaitable[AgentToolResult],
        ]
    ] = None
    replay: Optional[str] = None  # "never" | "safe"
    execution_mode: Optional[ToolExecutionMode] = None


@dataclass
class AgentContext:
    """Context snapshot passed into the low-level agent loop."""

    messages: List[AgentMessage] = field(default_factory=list)
    tools: Optional[List[AgentTool]] = None


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class AgentStartEvent:
    type: str = field(default="agent_start", init=False)


@dataclass
class AgentEndEvent:
    type: str = field(default="agent_end", init=False)
    messages: List[AgentMessage] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    type: str = field(default="turn_start", init=False)


@dataclass
class TurnEndEvent:
    type: str = field(default="turn_end", init=False)
    message: AgentMessage = None  # type: ignore[assignment]
    tool_results: List[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStartEvent:
    type: str = field(default="message_start", init=False)
    message: AgentMessage = None  # type: ignore[assignment]


@dataclass
class MessageUpdateEvent:
    type: str = field(default="message_update", init=False)
    message: AgentMessage = None  # type: ignore[assignment]
    assistant_message_event: AssistantMessageEvent = None  # type: ignore[assignment]


@dataclass
class MessageEndEvent:
    type: str = field(default="message_end", init=False)
    message: AgentMessage = None  # type: ignore[assignment]


@dataclass
class ToolExecutionStartEvent:
    type: str = field(default="tool_execution_start", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None


@dataclass
class ToolExecutionUpdateEvent:
    type: str = field(default="tool_execution_update", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    partial_result: Any = None


@dataclass
class ToolExecutionEndEvent:
    type: str = field(default="tool_execution_end", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    result: Any = None
    is_error: bool = False


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]

AgentEventSink = Callable[[AgentEvent], Union[None, Awaitable[None]]]
