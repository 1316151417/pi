"""Text helpers ported from pi-ai ``src/utils/text.ts``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from ._javascript import javascript_object_keys
from .types import ImageContent, SystemMessage, TextContent, ThinkingContent, ToolCall

__all__ = ["content_text", "get_system_message_text", "render_system_message_update"]


def content_text(
    content: str | Sequence[TextContent | ImageContent | ThinkingContent | ToolCall],
    separator: str = "\n",
) -> str:
    """Extract and join text from message content."""
    if isinstance(content, str):
        return content
    parts = [cast(TextContent, block).text for block in content if getattr(block, "type", None) == "text"]
    return separator.join(parts)


def get_system_message_text(message: SystemMessage) -> str:
    """Render a system message as a complete prompt: content followed by sections."""
    parts = [content_text(message.content)]
    sections = message.sections if message.sections is not None else {}
    for name in javascript_object_keys(sections):
        text = sections[name]
        if text is not None:
            parts.append(text)
    return "\n\n".join(part for part in parts if len(part) > 0)


def render_system_message_update(message: SystemMessage) -> str:
    """Render a later system message for APIs that accept mid-conversation system messages."""
    parts: list[str] = []
    text = content_text(message.content)
    if len(text) > 0:
        parts.append(text)
    sections = message.sections if message.sections is not None else {}
    for name in javascript_object_keys(sections):
        value = sections[name]
        if value is None:
            parts.append(f'Removed system prompt section "{name}".')
        else:
            parts.append(f'Updated system prompt section "{name}":\n\n{value}')
    return "\n\n".join(parts)
