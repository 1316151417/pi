"""Minimal in-package port of the ``@earendil-works/pi-ai`` surface used by agent-core."""

from .abort import AbortController, AbortError, AbortSignal
from .event_stream import (
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
)
from .text import content_text, get_system_message_text, render_system_message_update
from .transcript import (
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_tool_state_changes,
    normalize_context,
    to_tool_declaration,
)
from .types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Cost,
    DeferredHandle,
    ImageContent,
    JsonObject,
    JsonValue,
    Message,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    SystemMessage,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UserMessage,
    message_from_json,
    message_to_json,
)
from .uuid_utils import uuidv7
from .validation import validate_tool_arguments, validate_tool_call
