"""Assistant-message frame codec ported from pi-ai ``src/utils/assistant-message-frame.ts``.

The TypeScript module models :data:`AssistantMessageFrame` as a plain-object
discriminated union; this port uses one flat dataclass with a leading ``type``
discriminator, matching the ``AssistantMessageEvent`` style in
:mod:`pi_ai.types`. Frames persist through :meth:`AssistantMessageFrame.to_json`
and replay through :func:`frame_from_json`, so the reducer accepts either the
dataclass or the plain wire mapping the durable value store yields after replay.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

from .json_parse import parse_streaming_json
from .types import (
    AssistantContentBlock,
    AssistantMessage,
    AssistantMessageEvent,
    JsonObject,
    TextContent,
    ThinkingContent,
    ToolCall,
    content_block_from_json,
    message_from_json,
)

__all__ = [
    "AssistantMessageFrame",
    "AssistantMessageFrameEncoder",
    "reduce_assistant_message_frames",
    "frame_from_json",
]

#: Largest integer JavaScript still represents exactly (``Number.isSafeInteger``).
MAX_SAFE_INTEGER = 9007199254740991


@dataclass
class AssistantMessageFrame:
    """Compact, replayable assistant-message progress.

    Terminal settlement is intentionally excluded and must be persisted separately.
    ``content`` holds a content block for ``text_start``/``thinking_start`` frames and
    the authoritative block text for ``text_end``/``thinking_end`` frames.
    """

    type: str
    partial: Optional[AssistantMessage] = None
    content_index: Optional[int] = None
    content: Optional[Union[TextContent, ThinkingContent, str]] = None
    delta: Optional[str] = None
    text_signature: Optional[str] = None
    thinking_signature: Optional[str] = None
    redacted: Optional[bool] = None
    tool_call: Optional[ToolCall] = None
    json: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[JsonObject] = None
    thought_signature: Optional[str] = None
    namespace: Optional[str] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"type": self.type}
        if self.partial is not None:
            data["partial"] = self.partial.to_json()
        if self.content_index is not None:
            data["contentIndex"] = self.content_index
        if self.delta is not None:
            data["delta"] = self.delta
        if self.content is not None:
            data["content"] = self.content if isinstance(self.content, str) else self.content.to_json()
        if self.text_signature is not None:
            data["textSignature"] = self.text_signature
        if self.thinking_signature is not None:
            data["thinkingSignature"] = self.thinking_signature
        if self.redacted is not None:
            data["redacted"] = self.redacted
        if self.tool_call is not None:
            data["toolCall"] = self.tool_call.to_json()
        if self.json is not None:
            data["json"] = self.json
        if self.id is not None:
            data["id"] = self.id
        if self.name is not None:
            data["name"] = self.name
        if self.arguments is not None:
            data["arguments"] = copy.deepcopy(self.arguments)
        if self.thought_signature is not None:
            data["thoughtSignature"] = self.thought_signature
        if self.namespace is not None:
            data["namespace"] = self.namespace
        return data


def frame_from_json(data: Mapping[str, Any]) -> AssistantMessageFrame:
    """Decode one stored assistant-message frame from its wire shape."""
    partial = data.get("partial")
    tool_call = data.get("toolCall")
    content = data.get("content")
    if isinstance(content, Mapping):
        content = content_block_from_json(content)
    return AssistantMessageFrame(
        type=data["type"],
        partial=message_from_json(partial) if partial else None,
        content_index=data.get("contentIndex"),
        content=content,
        delta=data.get("delta"),
        text_signature=data.get("textSignature"),
        thinking_signature=data.get("thinkingSignature"),
        redacted=data.get("redacted"),
        tool_call=content_block_from_json(tool_call) if tool_call else None,
        json=data.get("json"),
        id=data.get("id"),
        name=data.get("name"),
        arguments=copy.deepcopy(data.get("arguments")),
        thought_signature=data.get("thoughtSignature"),
        namespace=data.get("namespace"),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assert_content_index(content_index: Optional[int]) -> int:
    if (
        not isinstance(content_index, int)
        or isinstance(content_index, bool)
        or content_index < 0
        or content_index > MAX_SAFE_INTEGER
    ):
        raise ValueError(f"Invalid assistant message frame contentIndex: {content_index}")
    return content_index


def _clone_text_content(content: TextContent) -> TextContent:
    return TextContent(text=content.text, text_signature=content.text_signature)


def _clone_thinking_content(content: ThinkingContent) -> ThinkingContent:
    return ThinkingContent(
        thinking=content.thinking,
        thinking_signature=content.thinking_signature,
        redacted=content.redacted,
    )


def _clone_tool_call(tool_call: ToolCall) -> ToolCall:
    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=copy.deepcopy(tool_call.arguments),
        thought_signature=tool_call.thought_signature,
        namespace=tool_call.namespace,
    )


def _clone_start_message(message: AssistantMessage) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=message.api,
        provider=message.provider,
        model=message.model,
        response_model=message.response_model,
        response_id=message.response_id,
        provider_thinking_level=message.provider_thinking_level,
        diagnostics=copy.deepcopy(message.diagnostics),
        usage=copy.deepcopy(message.usage),
        stop_reason="pending",
        timestamp=message.timestamp,
    )


def _serialize_arguments(arguments: Any) -> str:
    try:
        return json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Tool-call arguments are not JSON-serializable") from error


def _is_same_scalar(snapshot: Any, current: Any) -> bool:
    """``Object.is`` for JSON scalars: booleans are not numbers and ``None`` only matches ``None``."""
    if snapshot is None or current is None:
        return snapshot is None and current is None
    if isinstance(snapshot, bool) or isinstance(current, bool):
        return isinstance(snapshot, bool) and isinstance(current, bool) and snapshot == current
    if isinstance(snapshot, (int, float)) and isinstance(current, (int, float)):
        return snapshot == current
    return snapshot is current


def _is_json_prefix(snapshot: Any, current: Any) -> bool:
    """Report whether ``current`` is ``snapshot`` extended by later streamed JSON."""
    if isinstance(snapshot, str):
        return isinstance(current, str) and current.startswith(snapshot)
    if isinstance(snapshot, list):
        return (
            isinstance(current, list)
            and len(snapshot) <= len(current)
            and all(_is_json_prefix(value, current[index]) for index, value in enumerate(snapshot))
        )
    if not isinstance(snapshot, dict):
        return _is_same_scalar(snapshot, current)
    if not isinstance(current, dict):
        return False
    return all(key in current and _is_json_prefix(value, current[key]) for key, value in snapshot.items())


def _require(value: Any, subject: str, field: str) -> Any:
    if value is None:
        raise ValueError(f"{subject} is missing {field}")
    return value


def _event_block(event: AssistantMessageEvent) -> AssistantContentBlock:
    content_index = _assert_content_index(event.content_index)
    partial = event.partial
    blocks: List[AssistantContentBlock] = partial.content if partial is not None else []
    if content_index >= len(blocks):
        raise ValueError(f"{event.type} event has no content block at index {content_index}")
    return blocks[content_index]


#: Serialized form of the parsed empty tool-call arguments (``parseStreamingJson("")``).
_EMPTY_PARSED_TOOL_ARGUMENTS = _serialize_arguments(parse_streaming_json(""))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


@dataclass
class _EncoderTextState:
    """Block state for an open text or thinking block (``kind`` is ``"text"`` or ``"thinking"``)."""

    kind: str
    covered_chars: int
    delta_chars: int = 0


@dataclass
class _EncoderToolCallState:
    """Block state for an open tool call while its JSON stream catches up."""

    kind: str = "toolCall"
    caught_up: bool = False
    catchup_json: str = ""
    snapshot_arguments: str = ""


EncoderBlockState = Union[_EncoderTextState, _EncoderToolCallState]


class AssistantMessageFrameEncoder:
    """Encodes one assistant stream. ``partial`` remains a shared live accumulator;
    the encoder uses per-block offsets to avoid replaying deltas already visible
    when an older queued event is consumed.
    """

    def __init__(self) -> None:
        self._started = False
        self._terminal = False
        self._blocks: dict[int, EncoderBlockState] = {}

    def encode(self, event: AssistantMessageEvent) -> Optional[AssistantMessageFrame]:
        if self._terminal:
            raise ValueError(f"Assistant message event {event.type} follows a terminal event")

        if event.type == "start":
            if self._started:
                raise ValueError("Assistant message stream contains more than one start event")
            self._started = True
            return AssistantMessageFrame(
                type="start",
                partial=_clone_start_message(_require(event.partial, "start event", "partial")),
            )
        if event.type == "done":
            if not self._started:
                raise ValueError("Assistant message done event appears before start")
            self._terminal = True
            return None
        if event.type == "error":
            self._terminal = True
            return None

        if not self._started:
            raise ValueError(f"Assistant message {event.type} event appears before start")

        if event.type == "text_start":
            content_index = _assert_content_index(event.content_index)
            content = _event_block(event)
            if not isinstance(content, TextContent):
                raise ValueError(f"text_start event points to {content.type} block at index {content_index}")
            self._start_block(content_index, _EncoderTextState(kind="text", covered_chars=len(content.text)))
            return AssistantMessageFrame(
                type="text_start",
                content_index=content_index,
                content=_clone_text_content(content),
            )
        if event.type == "text_delta":
            return self._encode_text_delta(
                _assert_content_index(event.content_index), _require(event.delta, "text_delta event", "delta"), "text"
            )
        if event.type == "text_end":
            content_index = _assert_content_index(event.content_index)
            content = _event_block(event)
            if not isinstance(content, TextContent):
                raise ValueError(f"text_end event points to {content.type} block at index {content_index}")
            self._end_block(content_index, "text")
            return AssistantMessageFrame(
                type="text_end",
                content_index=content_index,
                content=_require(event.content, "text_end event", "content"),
                text_signature=content.text_signature,
            )
        if event.type == "thinking_start":
            content_index = _assert_content_index(event.content_index)
            content = _event_block(event)
            if not isinstance(content, ThinkingContent):
                raise ValueError(f"thinking_start event points to {content.type} block at index {content_index}")
            self._start_block(content_index, _EncoderTextState(kind="thinking", covered_chars=len(content.thinking)))
            return AssistantMessageFrame(
                type="thinking_start",
                content_index=content_index,
                content=_clone_thinking_content(content),
            )
        if event.type == "thinking_delta":
            return self._encode_text_delta(
                _assert_content_index(event.content_index),
                _require(event.delta, "thinking_delta event", "delta"),
                "thinking",
            )
        if event.type == "thinking_end":
            content_index = _assert_content_index(event.content_index)
            content = _event_block(event)
            if not isinstance(content, ThinkingContent):
                raise ValueError(f"thinking_end event points to {content.type} block at index {content_index}")
            self._end_block(content_index, "thinking")
            return AssistantMessageFrame(
                type="thinking_end",
                content_index=content_index,
                content=_require(event.content, "thinking_end event", "content"),
                thinking_signature=content.thinking_signature,
                redacted=content.redacted,
            )
        if event.type == "toolcall_start":
            content_index = _assert_content_index(event.content_index)
            content = _event_block(event)
            if not isinstance(content, ToolCall):
                raise ValueError(f"toolcall_start event points to {content.type} block at index {content_index}")
            snapshot_arguments = _serialize_arguments(content.arguments)
            caught_up = snapshot_arguments == _EMPTY_PARSED_TOOL_ARGUMENTS
            self._start_block(
                content_index,
                _EncoderToolCallState(
                    caught_up=caught_up,
                    catchup_json="",
                    snapshot_arguments="" if caught_up else snapshot_arguments,
                ),
            )
            return AssistantMessageFrame(
                type="toolcall_start",
                content_index=content_index,
                tool_call=_clone_tool_call(content),
            )
        if event.type == "toolcall_delta":
            content_index = _assert_content_index(event.content_index)
            delta = _require(event.delta, "toolcall_delta event", "delta")
            state = self._block(content_index, "toolCall")
            if not isinstance(state, _EncoderToolCallState):
                raise ValueError("Unreachable tool-call encoder state")
            if state.caught_up:
                if len(delta) == 0:
                    return None
                return AssistantMessageFrame(type="toolcall_delta", content_index=content_index, delta=delta)
            state.catchup_json += delta
            arguments_value = parse_streaming_json(state.catchup_json)
            if _serialize_arguments(arguments_value) != state.snapshot_arguments:
                # Legacy grammar calls include the initial input in toolcall_start, but their
                # JSON delta stream still begins at an empty input. Its parsed arguments can
                # therefore extend, rather than exactly reproduce, the start snapshot.
                snapshot_arguments = parse_streaming_json(state.snapshot_arguments)
                if not _is_json_prefix(snapshot_arguments, arguments_value):
                    return None
            state.caught_up = True
            state.snapshot_arguments = ""
            catchup_json = state.catchup_json
            state.catchup_json = ""
            if len(catchup_json) == 0:
                return None
            return AssistantMessageFrame(type="toolcall_checkpoint", content_index=content_index, json=catchup_json)
        if event.type == "toolcall_end":
            content_index = _assert_content_index(event.content_index)
            content = _event_block(event)
            if not isinstance(content, ToolCall):
                raise ValueError(f"toolcall_end event points to {content.type} block at index {content_index}")
            tool_call = event.tool_call
            if not isinstance(tool_call, ToolCall):
                raise ValueError(f"toolcall_end event has invalid tool call at index {content_index}")
            self._end_block(content_index, "toolCall")
            return AssistantMessageFrame(
                type="toolcall_end",
                content_index=content_index,
                id=tool_call.id,
                name=tool_call.name,
                arguments=copy.deepcopy(tool_call.arguments),
                thought_signature=tool_call.thought_signature,
                namespace=tool_call.namespace,
            )

        raise ValueError(f"Unknown assistant message event type: {event.type}")

    # -- block bookkeeping --------------------------------------------------

    def _start_block(self, content_index: int, state: EncoderBlockState) -> None:
        _assert_content_index(content_index)
        if content_index in self._blocks:
            raise ValueError(f"Assistant message block {content_index} starts more than once")
        self._blocks[content_index] = state

    def _block(self, content_index: int, kind: str) -> EncoderBlockState:
        _assert_content_index(content_index)
        state = self._blocks.get(content_index)
        if state is None:
            raise ValueError(f"Assistant message {kind} block {content_index} has not started")
        if state.kind != kind:
            raise ValueError(f"Assistant message block {content_index} is {state.kind}, not {kind}")
        return state

    def _end_block(self, content_index: int, kind: str) -> None:
        self._block(content_index, kind)
        del self._blocks[content_index]

    def _encode_text_delta(
        self, content_index: int, delta: str, kind: str
    ) -> Optional[AssistantMessageFrame]:
        state = self._block(content_index, kind)
        if not isinstance(state, _EncoderTextState):
            raise ValueError("Unreachable text encoder state")
        delta_start = state.delta_chars
        state.delta_chars += len(delta)
        covered = max(0, state.covered_chars - delta_start)
        if covered >= len(delta):
            return None
        uncovered = delta if covered == 0 else delta[covered:]
        return AssistantMessageFrame(type=f"{kind}_delta", content_index=content_index, delta=uncovered)


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


@dataclass
class _ReducerBlockState:
    """Open or ended block state while replaying frames (``kind`` mirrors the block type)."""

    kind: str
    ended: bool = False
    json: str = ""


def _append_block(
    message: AssistantMessage,
    states: dict[int, _ReducerBlockState],
    content_index: int,
    block: AssistantContentBlock,
    state: _ReducerBlockState,
) -> None:
    _assert_content_index(content_index)
    if content_index != len(message.content):
        reason = "already exists" if content_index < len(message.content) else "would leave a gap"
        raise ValueError(f"Cannot start assistant message block at index {content_index}: {reason}")
    message.content.append(copy.deepcopy(block))
    states[content_index] = state


def _active_block(
    message: AssistantMessage,
    states: dict[int, _ReducerBlockState],
    content_index: int,
    expected_kind: str,
    frame_type: str,
) -> Tuple[AssistantContentBlock, _ReducerBlockState]:
    _assert_content_index(content_index)
    state = states.get(content_index)
    block = message.content[content_index] if content_index < len(message.content) else None
    if state is None or block is None:
        raise ValueError(f"{frame_type} frame has no started block at index {content_index}")
    block_kind = getattr(block, "type", None)
    if state.kind != expected_kind or block_kind != expected_kind:
        raise ValueError(
            f"{frame_type} frame expected {expected_kind} block at index {content_index}, found {block_kind}"
        )
    if state.ended:
        raise ValueError(f"{frame_type} frame follows the end of block at index {content_index}")
    return block, state


def _as_frame(frame: Union[AssistantMessageFrame, Mapping[str, Any]]) -> AssistantMessageFrame:
    if isinstance(frame, AssistantMessageFrame):
        return frame
    if isinstance(frame, Mapping):
        return frame_from_json(frame)
    raise TypeError(f"Unsupported assistant message frame: {type(frame).__name__}")


def reduce_assistant_message_frames(
    frames: Iterable[Union[AssistantMessageFrame, Mapping[str, Any]]],
) -> Optional[AssistantMessage]:
    """Replay compact frames without mutating them. Returns ``None`` when the
    iterable contains no start frame.

    Frames may be :class:`AssistantMessageFrame` values or the plain mappings a
    durable value store yields after replay.
    """
    message: Optional[AssistantMessage] = None
    frame_before_start: Optional[str] = None
    states: dict[int, _ReducerBlockState] = {}

    for raw_frame in frames:
        frame = _as_frame(raw_frame)
        if frame.type == "start":
            if message is not None:
                raise ValueError("Assistant message frame sequence contains more than one start frame")
            if frame_before_start is not None:
                raise ValueError(f"{frame_before_start} frame appears before the start frame")
            message = copy.deepcopy(_require(frame.partial, "start frame", "partial"))
            continue
        if message is None:
            if frame_before_start is None:
                frame_before_start = frame.type
            continue

        content_index = frame.content_index
        if frame.type == "text_start":
            content = frame.content
            if not isinstance(content, TextContent):
                raise ValueError(f"text_start frame contains {_frame_content_kind(content)} content")
            _append_block(
                message, states, _assert_content_index(content_index), content, _ReducerBlockState(kind="text")
            )
            continue
        if frame.type == "text_delta":
            block, _state = _active_block(
                message, states, _assert_content_index(content_index), "text", frame.type
            )
            if not isinstance(block, TextContent):
                raise ValueError("Unreachable text frame state")
            block.text += _require(frame.delta, f"{frame.type} frame", "delta")
            continue
        if frame.type == "text_end":
            block, state = _active_block(
                message, states, _assert_content_index(content_index), "text", frame.type
            )
            if not isinstance(block, TextContent):
                raise ValueError("Unreachable text frame state")
            block.text = _require(frame.content, f"{frame.type} frame", "content")
            block.text_signature = frame.text_signature
            state.ended = True
            continue
        if frame.type == "thinking_start":
            content = frame.content
            if not isinstance(content, ThinkingContent):
                raise ValueError(f"thinking_start frame contains {_frame_content_kind(content)} content")
            _append_block(
                message, states, _assert_content_index(content_index), content, _ReducerBlockState(kind="thinking")
            )
            continue
        if frame.type == "thinking_delta":
            block, _state = _active_block(
                message, states, _assert_content_index(content_index), "thinking", frame.type
            )
            if not isinstance(block, ThinkingContent):
                raise ValueError("Unreachable thinking frame state")
            block.thinking += _require(frame.delta, f"{frame.type} frame", "delta")
            continue
        if frame.type == "thinking_end":
            block, state = _active_block(
                message, states, _assert_content_index(content_index), "thinking", frame.type
            )
            if not isinstance(block, ThinkingContent):
                raise ValueError("Unreachable thinking frame state")
            block.thinking = _require(frame.content, f"{frame.type} frame", "content")
            block.thinking_signature = frame.thinking_signature
            block.redacted = frame.redacted
            state.ended = True
            continue
        if frame.type == "toolcall_start":
            tool_call = frame.tool_call
            if not isinstance(tool_call, ToolCall):
                raise ValueError(f"toolcall_start frame contains {_frame_content_kind(tool_call)} content")
            _append_block(
                message,
                states,
                _assert_content_index(content_index),
                tool_call,
                _ReducerBlockState(kind="toolCall", json=""),
            )
            continue
        if frame.type == "toolcall_checkpoint":
            block, state = _active_block(
                message, states, _assert_content_index(content_index), "toolCall", frame.type
            )
            if not isinstance(block, ToolCall):
                raise ValueError("Unreachable tool-call checkpoint state")
            json_payload = _require(frame.json, f"{frame.type} frame", "json")
            state.json = json_payload
            block.arguments = parse_streaming_json(json_payload)
            continue
        if frame.type == "toolcall_delta":
            _block, state = _active_block(
                message, states, _assert_content_index(content_index), "toolCall", frame.type
            )
            state.json += _require(frame.delta, f"{frame.type} frame", "delta")
            continue
        if frame.type == "toolcall_end":
            block, state = _active_block(
                message, states, _assert_content_index(content_index), "toolCall", frame.type
            )
            if not isinstance(block, ToolCall):
                raise ValueError("Unreachable tool-call frame state")
            block.id = _require(frame.id, f"{frame.type} frame", "id")
            block.name = _require(frame.name, f"{frame.type} frame", "name")
            block.arguments = copy.deepcopy(_require(frame.arguments, f"{frame.type} frame", "arguments"))
            block.thought_signature = frame.thought_signature
            block.namespace = frame.namespace
            state.ended = True
            continue

        raise ValueError(f"Unknown assistant message frame type: {frame.type}")

    if message is None:
        return None
    for content_index, state in states.items():
        if state.kind != "toolCall" or state.ended or len(state.json) == 0:
            continue
        block = message.content[content_index] if content_index < len(message.content) else None
        if not isinstance(block, ToolCall):
            raise ValueError("Unreachable tool-call frame state")
        block.arguments = parse_streaming_json(state.json)

    return message


def _frame_content_kind(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, str):
        return "string"
    return str(getattr(value, "type", type(value).__name__))
