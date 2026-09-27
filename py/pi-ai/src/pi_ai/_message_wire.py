"""Mutable message views for the pi-messages and agent proxy wire protocols.

These views retain unknown wire fields and nested JSON values. They do not
coerce token counts, copy tool arguments, or replace sparse content with text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import ClassVar, cast, overload

from ._model_wire import ModelFields, ModelList
from ._values import UNDEFINED
from .types import (
    AssistantContentBlock, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    Cost, DiagnosticErrorInfo, JsonObject, TextContent, ThinkingContent, ToolCall, Usage,
)


class _MessageFields(ModelFields):
    def __delattr__(self, name: str) -> None:
        state = object.__getattribute__(self, "__dict__")
        if "_wire" in state and not name.startswith("_"):
            state["_wire"].pop(type(self)._wire_names.get(name, name), None)
        else:
            object.__delattr__(self, name)


class _WireCost(_MessageFields, Cost):
    _wire_names: ClassVar[dict[str, str]] = {
        "input": "input", "output": "output", "cache_read": "cacheRead", "cache_write": "cacheWrite", "total": "total",
    }


class _WireUsage(_MessageFields, Usage):
    _wire_names: ClassVar[dict[str, str]] = {
        "input": "input", "output": "output", "cache_read": "cacheRead", "cache_write": "cacheWrite",
        "cache_write_1h": "cacheWrite1h", "reasoning": "reasoning", "total_tokens": "totalTokens", "cost": "cost",
    }

    def _view_wire_field(self, name: str, value: object) -> object:
        if name == "cost" and isinstance(value, Mapping):
            cached = self.__dict__.get("_cost_view")
            if cached is None or cached[0] is not value:
                cached = (value, cost_from_wire(cast(Mapping[str, object], value)))
                object.__setattr__(self, "_cost_view", cached)
            return cached[1]
        return value

    def to_json(self) -> JsonObject:
        data = super().to_json()
        cost = data.get("cost")
        if isinstance(cost, Cost):
            return cast(JsonObject, {**data, "cost": cost.to_json()})
        return cast(JsonObject, data)


class _WireText(_MessageFields, TextContent):
    _wire_names: ClassVar[dict[str, str]] = {"type": "type", "text": "text", "text_signature": "textSignature"}


class _WireThinking(_MessageFields, ThinkingContent):
    _wire_names: ClassVar[dict[str, str]] = {
        "type": "type", "thinking": "thinking", "thinking_signature": "thinkingSignature", "redacted": "redacted",
    }


class _WireToolCall(_MessageFields, ToolCall):
    _wire_names: ClassVar[dict[str, str]] = {
        "type": "type", "id": "id", "name": "name", "arguments": "arguments",
        "thought_signature": "thoughtSignature", "namespace": "namespace", "partial_json": "partialJson",
    }


class _WireDiagnosticError(_MessageFields, DiagnosticErrorInfo):
    _wire_names: ClassVar[dict[str, str]] = {"name": "name", "message": "message", "stack": "stack", "code": "code"}


class _WireDiagnostic(_MessageFields, AssistantMessageDiagnostic):
    _wire_names: ClassVar[dict[str, str]] = {"type": "type", "timestamp": "timestamp", "error": "error", "details": "details"}

    def _view_wire_field(self, name: str, value: object) -> object:
        if name == "error" and isinstance(value, Mapping):
            cached = self.__dict__.get("_error_view")
            if cached is None or cached[0] is not value:
                cached = (value, _WireDiagnosticError._from_wire(cast(Mapping[str, object], value)))
                object.__setattr__(self, "_error_view", cached)
            return cached[1]
        return value


class _WireContents(ModelList[ModelFields]):
    _native_types: ClassVar[tuple[type[object], ...]] = (TextContent, ThinkingContent, ToolCall)

    @overload
    def __getitem__(self, index: int) -> ModelFields: ...
    @overload
    def __getitem__(self, index: slice) -> list[ModelFields]: ...

    def __getitem__(self, index: int | slice) -> ModelFields | list[ModelFields]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        raw = self._items[index]
        if raw is None or raw is UNDEFINED or isinstance(raw, self._native_types):
            # A sparse JS array returns undefined at a hole. The runtime value
            # remains UNDEFINED even though the valid protocol type is a block.
            return cast(ModelFields, raw)
        return super().__getitem__(index)

    @overload
    def __setitem__(self, index: int, value: ModelFields) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[ModelFields]) -> None: ...

    def __setitem__(self, index: int | slice, value: ModelFields | Iterable[ModelFields]) -> None:
        # Cached views are keyed by the raw object's identity, not position.
        # Appending/replacing another block must not change existing views.
        if isinstance(index, slice):
            self._items[index] = [self._store_view(item) for item in cast(Iterable[object], value)]
        else:
            self._items[index] = self._store_view(value)

    def __delitem__(self, index: int | slice) -> None:
        del self._items[index]

    def insert(self, index: int, value: ModelFields) -> None:
        self._items.insert(index, self._store_view(value))

    def _store_view(self, value: object) -> object:
        if isinstance(value, ModelFields):
            raw = value.to_json()
            self._views[id(raw)] = value
            return raw
        return value


class _WireDiagnostics(_WireContents):
    _native_types: ClassVar[tuple[type[object], ...]] = (AssistantMessageDiagnostic,)


class _WireAssistantMessage(_MessageFields, AssistantMessage):
    _wire_names: ClassVar[dict[str, str]] = {
        "role": "role", "content": "content", "api": "api", "provider": "provider", "model": "model",
        "response_model": "responseModel", "response_id": "responseId", "provider_thinking_level": "providerThinkingLevel",
        "diagnostics": "diagnostics", "usage": "usage", "stop_reason": "stopReason", "deferred": "deferred",
        "error_message": "errorMessage", "raw_stop_reason": "rawStopReason", "end_turn": "endTurn", "timestamp": "timestamp",
    }

    def _view_wire_field(self, name: str, value: object) -> object:
        if name == "content" and isinstance(value, list):
            cached = self.__dict__.get("_content_view")
            if cached is None or cached[0] is not value:
                cached = (value, _WireContents(value, lambda raw: cast(ModelFields, content_block_from_wire(raw))))
                object.__setattr__(self, "_content_view", cached)
            return cached[1]
        if name == "usage" and isinstance(value, Mapping):
            cached = self.__dict__.get("_usage_view")
            if cached is None or cached[0] is not value:
                cached = (value, usage_from_wire(cast(Mapping[str, object], value)))
                object.__setattr__(self, "_usage_view", cached)
            return cached[1]
        if name == "diagnostics" and isinstance(value, list):
            cached = self.__dict__.get("_diagnostics_view")
            if cached is None or cached[0] is not value:
                cached = (value, _WireDiagnostics(value, _WireDiagnostic._from_wire))
                object.__setattr__(self, "_diagnostics_view", cached)
            return cached[1]
        return value

    def to_json(self) -> JsonObject:
        data = dict(super().to_json())
        raw_content = data.get("content")
        if isinstance(raw_content, list):
            data["content"] = [
                None if item is UNDEFINED else item.to_json() if isinstance(item, (TextContent, ThinkingContent, ToolCall)) else item
                for item in raw_content
            ]
        usage = data.get("usage")
        if isinstance(usage, Usage):
            data["usage"] = usage.to_json()
        diagnostics = data.get("diagnostics")
        if isinstance(diagnostics, list):
            data["diagnostics"] = [item.to_json() if isinstance(item, AssistantMessageDiagnostic) else item for item in diagnostics]
        deferred = data.get("deferred")
        serialize = getattr(deferred, "to_json", None)
        if callable(serialize):
            data["deferred"] = serialize()
        return cast(JsonObject, data)


class _WireAssistantMessageEvent(_MessageFields, AssistantMessageEvent):
    _wire_names: ClassVar[dict[str, str]] = {
        "type": "type", "partial": "partial", "content_index": "contentIndex", "delta": "delta", "content": "content",
        "tool_call": "toolCall", "reason": "reason", "message": "message", "error": "error",
    }

    def _view_wire_field(self, name: str, value: object) -> object:
        if isinstance(value, Mapping) and name in ("partial", "message", "error", "tool_call"):
            cache_name = f"_{name}_view"
            cached = self.__dict__.get(cache_name)
            if cached is None or cached[0] is not value:
                view = tool_call_from_wire(value) if name == "tool_call" else assistant_message_from_wire(value)
                cached = (value, view)
                object.__setattr__(self, cache_name, cached)
            return cached[1]
        return value

    def to_json(self) -> JsonObject:
        data = dict(super().to_json())
        for key in ("partial", "message", "error"):
            value = data.get(key)
            if isinstance(value, AssistantMessage):
                data[key] = value.to_json()
            elif isinstance(value, Mapping):
                data[key] = assistant_message_from_wire(value).to_json()
        tool_call = data.get("toolCall")
        if isinstance(tool_call, ToolCall):
            data["toolCall"] = tool_call.to_json()
        return cast(JsonObject, data)


def cost_from_wire(raw: Mapping[str, object]) -> Cost:
    return _WireCost._from_wire(raw)


def usage_from_wire(raw: Mapping[str, object]) -> Usage:
    return _WireUsage._from_wire(raw)


def text_content_from_wire(raw: Mapping[str, object]) -> TextContent:
    return _WireText._from_wire(raw)


def thinking_content_from_wire(raw: Mapping[str, object]) -> ThinkingContent:
    return _WireThinking._from_wire(raw)


def tool_call_from_wire(raw: Mapping[str, object]) -> ToolCall:
    return _WireToolCall._from_wire(raw)


def content_block_from_wire(raw: Mapping[str, object]) -> AssistantContentBlock:
    block_type = raw.get("type")
    if block_type == "text":
        return text_content_from_wire(raw)
    if block_type == "thinking":
        return thinking_content_from_wire(raw)
    return tool_call_from_wire(raw)


def assistant_message_from_wire(raw: Mapping[str, object]) -> AssistantMessage:
    return _WireAssistantMessage._from_wire(raw)


def assistant_message_event_from_wire(raw: Mapping[str, object]) -> AssistantMessageEvent:
    return _WireAssistantMessageEvent._from_wire(raw)


__all__ = [
    "cost_from_wire", "usage_from_wire", "text_content_from_wire", "thinking_content_from_wire",
    "tool_call_from_wire", "content_block_from_wire", "assistant_message_from_wire", "assistant_message_event_from_wire",
]
