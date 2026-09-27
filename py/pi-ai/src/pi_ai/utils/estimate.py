"""Context-token estimation ported from ``src/utils/estimate.ts``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..text import get_system_message_text
from ..types import (
    AssistantMessage,
    ImageContent,
    Message,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UserMessage,
)
from ._javascript import javascript_json_stringify, utf16_length

__all__ = [
    "ContextUsageEstimate",
    "calculate_context_tokens",
    "estimate_text_tokens",
    "estimate_text_and_image_content_tokens",
    "estimate_message_tokens",
    "estimate_context_tokens",
]

CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800


@dataclass
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def _safe_json_stringify(value: object) -> str:
    try:
        serialized = javascript_json_stringify(value)
        return "undefined" if serialized is None else serialized
    except Exception:
        return "[unserializable]"


def estimate_text_tokens(text: str) -> int:
    return math.ceil(utf16_length(text) / CHARS_PER_TOKEN)


def estimate_text_and_image_content_tokens(content: str | Sequence[TextContent | ImageContent]) -> int:
    if isinstance(content, str):
        chars = utf16_length(content)
    else:
        chars = sum(utf16_length(block.text) if isinstance(block, TextContent) else ESTIMATED_IMAGE_CHARS for block in content)
    return math.ceil(chars / CHARS_PER_TOKEN)


def _estimate_tools_tokens(tools: Sequence[object] | None) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json_stringify(tools))


def estimate_message_tokens(message: Message) -> int:
    if isinstance(message, SystemMessage):
        return (
            estimate_text_tokens(get_system_message_text(message))
            + _estimate_tools_tokens(message.tools_added)
            + _estimate_tools_tokens(message.tools_removed)
        )
    if isinstance(message, (UserMessage, ToolResultMessage)):
        return estimate_text_and_image_content_tokens(message.content)
    chars = 0
    for block in message.content:
        if isinstance(block, TextContent):
            chars += utf16_length(block.text)
        elif isinstance(block, ThinkingContent):
            chars += utf16_length(block.thinking)
        else:
            chars += utf16_length(block.name) + utf16_length(_safe_json_stringify(block.arguments))
    return math.ceil(chars / CHARS_PER_TOKEN)


def estimate_context_tokens(context: TranscriptContext | Sequence[Message]) -> ContextUsageEstimate:
    messages = context.messages if isinstance(context, TranscriptContext) else context
    latest_prefix_timestamp = -math.inf
    usage_info: tuple[Usage, int] | None = None
    for index, message in enumerate(messages):
        if (
            isinstance(message, AssistantMessage)
            and message.timestamp >= latest_prefix_timestamp
            and message.stop_reason not in ("aborted", "error")
            and calculate_context_tokens(message.usage) > 0
        ):
            usage_info = (message.usage, index)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp)

    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing_tokens = sum(estimate_message_tokens(message) for message in messages[index + 1 :])
        return ContextUsageEstimate(usage_tokens + trailing_tokens, usage_tokens, trailing_tokens, index)
    tokens = sum(estimate_message_tokens(message) for message in messages)
    return ContextUsageEstimate(tokens, 0, tokens, None)
