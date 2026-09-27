"""Cross-provider transcript conversion from ``api/transform-messages.ts``."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from typing import cast

from ..types import (
    AssistantContentBlock, AssistantMessage, ImageContent, Message, Model, TextContent, ThinkingContent,
    ToolCall, ToolResultMessage, UserMessage,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"
_JS_WHITESPACE = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def _replace_images_with_placeholder(content: Sequence[TextContent | ImageContent], placeholder: str) -> list[TextContent]:
    result: list[TextContent] = []
    previous_was_placeholder = False
    for block in content:
        if block.type == "image":
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue
        text = cast(TextContent, block)
        result.append(text)
        previous_was_placeholder = text.text == placeholder
    return result


def transform_messages(
    messages: Sequence[Message], model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    tool_call_id_map: dict[str, str] = {}
    image_aware_messages: list[Message] = []
    for original in messages:
        message = original
        if getattr(message, "content", None) is None:
            message = copy.copy(message)
            message.content = []
        if "image" not in model.input:
            if message.role == "user" and isinstance(message.content, list):
                user = copy.copy(cast(UserMessage, message))
                user.content = _replace_images_with_placeholder(message.content, NON_VISION_USER_IMAGE_PLACEHOLDER)
                message = user
            elif message.role == "toolResult":
                tool_result = copy.copy(cast(ToolResultMessage, message))
                tool_result.content = _replace_images_with_placeholder(tool_result.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER)
                message = tool_result
        image_aware_messages.append(message)

    transformed: list[Message] = []
    for message in image_aware_messages:
        if message.role in ("system", "user"):
            transformed.append(message)
            continue
        if message.role == "toolResult":
            tool_result = cast(ToolResultMessage, message)
            normalized_id = tool_call_id_map.get(tool_result.tool_call_id)
            if normalized_id and normalized_id != tool_result.tool_call_id:
                tool_result = copy.copy(tool_result)
                tool_result.tool_call_id = normalized_id
            transformed.append(tool_result)
            continue
        if message.role != "assistant":
            transformed.append(message)
            continue
        assistant = cast(AssistantMessage, message)
        same_model = assistant.provider == model.provider and assistant.api == model.api and assistant.model == model.id
        content: list[AssistantContentBlock] = []
        for block in assistant.content:
            if block.type == "thinking":
                thinking = cast(ThinkingContent, block)
                if thinking.redacted:
                    if same_model:
                        content.append(thinking)
                    continue
                if same_model and thinking.thinking_signature:
                    content.append(thinking)
                    continue
                if not thinking.thinking or not thinking.thinking.strip(_JS_WHITESPACE):
                    continue
                content.append(thinking if same_model else TextContent(text=thinking.thinking))
                continue
            if block.type == "text":
                text = cast(TextContent, block)
                content.append(text if same_model else TextContent(text=text.text))
                continue
            if block.type == "toolCall":
                tool_call = cast(ToolCall, block)
                normalized_tool_call = tool_call
                if not same_model and tool_call.thought_signature:
                    normalized_tool_call = copy.copy(tool_call)
                    normalized_tool_call.thought_signature = None
                if not same_model and normalize_tool_call_id is not None:
                    normalized_id = normalize_tool_call_id(tool_call.id, model, assistant)
                    if normalized_id != tool_call.id:
                        tool_call_id_map[tool_call.id] = normalized_id
                        normalized_tool_call = copy.copy(normalized_tool_call)
                        normalized_tool_call.id = normalized_id
                content.append(normalized_tool_call)
                continue
            content.append(block)
        updated_assistant = copy.copy(assistant)
        updated_assistant.content = content
        transformed.append(updated_assistant)

    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()
    held_system_messages: list[Message] = []

    def close_pending_tool_calls() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        if pending_tool_calls:
            for tool_call in pending_tool_calls:
                if tool_call.id not in existing_tool_result_ids:
                    result.append(ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text="No result provided")],
                        is_error=True,
                        timestamp=time.time_ns() // 1_000_000,
                    ))
            pending_tool_calls = []
            existing_tool_result_ids = set()
        result.extend(held_system_messages)
        held_system_messages.clear()

    for message in transformed:
        if message.role == "assistant":
            close_pending_tool_calls()
            assistant = cast(AssistantMessage, message)
            if assistant.stop_reason in ("error", "aborted"):
                continue
            tool_calls = [cast(ToolCall, block) for block in assistant.content if block.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()
            result.append(message)
        elif message.role == "toolResult":
            existing_tool_result_ids.add(cast(ToolResultMessage, message).tool_call_id)
            result.append(message)
        elif message.role == "system":
            if pending_tool_calls:
                held_system_messages.append(message)
            else:
                result.append(message)
        elif message.role == "user":
            close_pending_tool_calls()
            result.append(message)
        else:
            result.append(message)
    close_pending_tool_calls()
    return result
