"""Transcript replay helpers ported from pi-ai ``src/utils/transcript.ts``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Protocol, TypeGuard, cast

from ._javascript import javascript_json_stringify, javascript_object_keys
from ._json_runtime import parse_json
from .text import content_text, get_system_message_text
from .types import (
    Context,
    JsonObject,
    Message,
    SystemMessage,
    Tool,
    ToolReference,
    TranscriptContext,
)

__all__ = [
    "TranscriptMessages",
    "TranscriptContext",
    "create_initial_system_message",
    "normalize_context",
    "get_initial_system_message",
    "without_initial_system_message",
    "get_current_tools",
    "get_current_system_message",
    "get_current_system_prompt",
    "collapse_system_messages",
    "resolve_transcript",
    "to_tool_declaration",
    "declarations_equal",
    "get_tool_state_changes",
    "get_declared_tools",
    "has_tool_redefinitions",
    "has_non_additive_tool_changes",
    "resolve_transcript_tools",
    "ToolStateChanges",
    "TranscriptTools",
]


class _TranscriptMessage(Protocol):
    @property
    def role(self) -> str: ...


TranscriptMessages = Sequence[_TranscriptMessage]


def _is_system_message(message: _TranscriptMessage) -> TypeGuard[SystemMessage]:
    return getattr(message, "role", None) == "system"


def create_initial_system_message(
    system_prompt: Optional[str], tools: Optional[Sequence[Tool]]
) -> Optional[SystemMessage]:
    """Build the leading system message for a prompt and tool set."""
    has_system_prompt = system_prompt is not None and len(system_prompt) > 0
    has_tools = tools is not None and len(tools) > 0
    if not has_system_prompt and not has_tools:
        return None
    message = SystemMessage(content=system_prompt if system_prompt is not None else "", timestamp=0)
    if has_tools:
        message.tools_added = tools if isinstance(tools, list) else list(cast(Sequence[Tool], tools))
    return message


def normalize_context(context: Context) -> TranscriptContext:
    """Fold ``Context.system_prompt`` and ``Context.tools`` into a leading system message."""
    initial_message = create_initial_system_message(context.system_prompt, context.tools)
    messages: List[Message] = (
        [initial_message, *context.messages] if initial_message else context.messages
    )
    return TranscriptContext(messages=messages)


def get_initial_system_message(messages: TranscriptMessages) -> Optional[SystemMessage]:
    first = messages[0] if len(messages) > 0 else None
    return first if first is not None and _is_system_message(first) else None


def without_initial_system_message(messages: List[Message]) -> List[Message]:
    return messages[1:] if get_initial_system_message(messages) is not None else messages


def get_current_tools(messages: TranscriptMessages) -> List[Tool]:
    """Resolve the tools available after applying every transcript delta in order."""
    tools: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for ref in message.tools_removed or []:
            tools.pop(ref.name, None)
        for tool in message.tools_added or []:
            tools[tool.name] = tool
    return list(tools.values())


def get_current_system_message(messages: TranscriptMessages) -> Optional[SystemMessage]:
    """Replay every system message into one leading system message."""
    content: List[str] = []
    sections: dict[str, str] = {}
    timestamp: Optional[int] = None
    for message in messages:
        if not _is_system_message(message):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        text = content_text(message.content)
        if len(text) > 0:
            content.append(text)
        message_sections = message.sections if message.sections is not None else {}
        for name in javascript_object_keys(message_sections):
            value = message_sections[name]
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
    tools = get_current_tools(messages)
    if timestamp is None and len(tools) == 0:
        return None
    result = SystemMessage(
        content="\n\n".join(content),
        sections=dict(sections) if sections else None,
        tools_added=tools if tools else None,
        timestamp=timestamp if timestamp is not None else 0,
    )
    return result


def get_current_system_prompt(messages: TranscriptMessages) -> str:
    message = get_current_system_message(messages)
    return get_system_message_text(message) if message else ""


def collapse_system_messages(context: TranscriptContext) -> TranscriptContext:
    head = get_current_system_message(context.messages)
    messages = [m for m in context.messages if not _is_system_message(m)]
    return TranscriptContext(messages=[head, *messages] if head else messages)


def resolve_transcript(
    context: TranscriptContext, supports_mid_convo_system_messages: Optional[bool]
) -> TranscriptContext:
    if supports_mid_convo_system_messages:
        return context
    return collapse_system_messages(context)


def to_tool_declaration(tool: Tool) -> Tool:
    """Strip executable and display-only fields from a tool before comparison or persistence."""
    parameters = javascript_json_stringify(tool.parameters)
    if parameters is None:
        raise ValueError("Tool parameters are not JSON-serializable")
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=cast(JsonObject, parse_json(parameters)),
        constrained_sampling=tool.constrained_sampling,
    )


def declarations_equal(left: Tool, right: Tool) -> bool:
    return javascript_json_stringify(to_tool_declaration(left)) == javascript_json_stringify(to_tool_declaration(right))


class ToolStateChanges:
    def __init__(self, tools_added: List[Tool], tools_removed: List[ToolReference]) -> None:
        self.tools_added = tools_added
        self.tools_removed = tools_removed


def get_tool_state_changes(previous: Sequence[Tool], current: Sequence[Tool]) -> ToolStateChanges:
    """Compare two complete tool states. A changed definition is a removal followed by an addition."""
    previous_tools = {tool.name: tool for tool in previous}
    current_tools = {tool.name: tool for tool in current}
    tools_added = [
        to_tool_declaration(tool)
        for tool in current
        if tool.name not in previous_tools or not declarations_equal(previous_tools[tool.name], tool)
    ]
    tools_removed = [
        ToolReference(name=tool.name)
        for tool in previous
        if tool.name not in current_tools or not declarations_equal(tool, current_tools[tool.name])
    ]
    return ToolStateChanges(tools_added=tools_added, tools_removed=tools_removed)


def get_declared_tools(messages: TranscriptMessages) -> List[Tool]:
    definitions: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            definitions[tool.name] = tool
    return list(definitions.values())


def has_tool_redefinitions(messages: TranscriptMessages) -> bool:
    declared: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            previous = declared.get(tool.name)
            if previous is not None and not declarations_equal(previous, tool):
                return True
            declared[tool.name] = tool
    return False


def has_non_additive_tool_changes(messages: TranscriptMessages) -> bool:
    declared: set[str] = set()
    for message in messages:
        if not _is_system_message(message):
            continue
        if len(message.tools_removed or []) > 0:
            return True
        for tool in message.tools_added or []:
            if tool.name in declared:
                return True
            declared.add(tool.name)
    return False


class TranscriptTools:
    def __init__(self, request_tools: List[Tool], anchors_additions: bool) -> None:
        self.request_tools = request_tools
        self.anchors_additions = anchors_additions


def resolve_transcript_tools(messages: TranscriptMessages, supports_tool_additions: bool) -> TranscriptTools:
    anchors_additions = supports_tool_additions and not has_non_additive_tool_changes(messages)
    if anchors_additions:
        initial = get_initial_system_message(messages)
        request_tools = initial.tools_added if initial is not None and initial.tools_added is not None else []
    else:
        request_tools = get_current_tools(messages)
    return TranscriptTools(request_tools=request_tools, anchors_additions=anchors_additions)
