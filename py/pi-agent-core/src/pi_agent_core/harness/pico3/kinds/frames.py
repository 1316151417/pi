"""Assistant-message frame application ported from ``harness/pico3/kinds/frames.ts``.

Applies one encoded frame to the tracked output. Same switch as pi-ai's
``reduceAssistantMessageFrames`` (the test oracle); no buffering, no second copy.

The tracked output is the document-shaped turn state — either a ``dict`` with a
``message`` key or any object exposing ``message`` — and its message content
blocks may be dataclasses (:mod:`pi_agent_core._pi_ai.types`) or the plain JSON
mappings a durable value store yields after replay. Both are accepted.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ...._pi_ai.assistant_message_frame import AssistantMessageFrame
from ...._pi_ai.types import JsonObject

__all__ = ["apply_frame"]

_MISSING = object()

#: camelCase field name -> dataclass attribute for the block/message records we mutate.
_SNAKE = {
    "contentIndex": "content_index",
    "textSignature": "text_signature",
    "thinkingSignature": "thinking_signature",
    "toolCallId": "tool_call_id",
    "thoughtSignature": "thought_signature",
    "isError": "is_error",
    "toolName": "tool_name",
    "stopReason": "stop_reason",
}


def apply_frame(o: Any, frame: AssistantMessageFrame) -> None:
    """Apply ``frame`` to the tracked output ``o`` in place."""
    if frame.type == "start":
        _set_field(o, "message", _clone(frame.partial))  # the encoder's clone
        return
    message = _get_field(o, "message")
    if message is None:
        raise ValueError(f"{frame.type} before start")
    content: List[Any] = _get_field(message, "content")
    index = frame.content_index
    if frame.type == "text_start":
        _set_index(content, index, _clone(frame.content))
        return
    if frame.type == "text_delta":
        block = content[index]
        _set_field(block, "text", (_get_field(block, "text") or "") + (frame.delta or ""))
        return
    if frame.type == "text_end":
        block = content[index]
        _set_field(block, "text", frame.content)
        _delete_field(block, "textSignature")
        if frame.text_signature is not None:
            _set_field(block, "textSignature", frame.text_signature)
        return
    if frame.type == "thinking_start":
        _set_index(content, index, _clone(frame.content))
        return
    if frame.type == "thinking_delta":
        block = content[index]
        _set_field(
            block, "thinking", (_get_field(block, "thinking") or "") + (frame.delta or "")
        )
        return
    if frame.type == "thinking_end":
        block = content[index]
        _set_field(block, "thinking", frame.content)
        _delete_field(block, "thinkingSignature")
        _delete_field(block, "redacted")
        if frame.thinking_signature is not None:
            _set_field(block, "thinkingSignature", frame.thinking_signature)
        if frame.redacted is not None:
            _set_field(block, "redacted", frame.redacted)
        return
    if frame.type == "toolcall_start":
        _set_index(content, index, _clone(frame.tool_call))
        return
    if frame.type == "toolcall_checkpoint":
        _set_field(content[index], "arguments", safe_parse(frame.json))
        return
    if frame.type == "toolcall_delta":
        return  # arguments materialise at checkpoint/end
    if frame.type == "toolcall_end":
        block = content[index]
        _set_field(block, "id", frame.id)
        _set_field(block, "name", frame.name)
        _set_field(block, "arguments", frame.arguments)
        if frame.thought_signature is not None:
            _set_field(block, "thoughtSignature", frame.thought_signature)
        if frame.namespace is not None:
            _set_field(block, "namespace", frame.namespace)
        return


def _set_index(content: List[Any], index: int, value: Any) -> None:
    """``content[i] = value`` with JavaScript's grow-the-array behaviour."""
    while len(content) <= index:
        content.append(None)
    content[index] = value


def safe_parse(text: Optional[str]) -> JsonObject:
    """Parse a streaming tool-call JSON checkpoint; malformed input yields ``{}``."""
    try:
        parsed = json.loads(text or "")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    value = getattr(obj, _SNAKE.get(key, key), _MISSING)
    return None if value is _MISSING else value


def _set_field(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
        return
    setattr(obj, _SNAKE.get(key, key), value)


def _delete_field(obj: Any, key: str) -> None:
    if isinstance(obj, dict):
        obj.pop(key, None)
        return
    attribute = _SNAKE.get(key, key)
    if hasattr(obj, attribute):
        setattr(obj, attribute, None)


def _clone(value: Any) -> Any:
    """Clone a frame payload into the stored representation of the tracked output."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_json"):
        return value.to_json()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


_ = Dict
