"""Anthropic Messages API provider ported from pi-ai ``src/api/anthropic-messages.ts``.

Implements the ``anthropic-messages`` API surface against the public Anthropic
HTTP endpoint using SSE streaming, mapping provider events onto the
``AssistantMessageEvent`` protocol.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from ..event_stream import AssistantMessageEventStream, create_assistant_message_event_stream
from ..models import Provider, create_provider
from ..text import content_text
from ..transcript import get_current_tools, get_current_system_prompt, collapse_system_messages
from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Cost,
    ImageContent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)
from ..models import calculate_cost

__all__ = ["anthropic_provider", "ANTHROPIC_MODELS", "ANTHROPIC_DEFAULT_BASE_URL"]

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

THINKING_BUDGET_TOKENS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}


def _model(
    model_id: str,
    name: str,
    context_window: int,
    max_tokens: int,
    cost: tuple[float, float, float, float],
    reasoning: bool = False,
) -> Model:
    return Model(
        id=model_id,
        name=name,
        api="anthropic-messages",
        provider="anthropic",
        base_url=ANTHROPIC_DEFAULT_BASE_URL,
        reasoning=reasoning,
        input=["text", "image"],
        cost=ModelCost(input=cost[0], output=cost[1], cache_read=cost[2], cache_write=cost[3]),
        context_window=context_window,
        max_tokens=max_tokens,
    )


ANTHROPIC_MODELS: List[Model] = [
    _model("claude-sonnet-4-5", "Claude Sonnet 4.5", 200000, 64000, (3, 15, 0.3, 3.75), reasoning=True),
    _model("claude-opus-4-6", "Claude Opus 4.6", 200000, 64000, (5, 25, 0.5, 6.25), reasoning=True),
    _model("claude-haiku-4-5", "Claude Haiku 4.5", 200000, 64000, (1, 5, 0.1, 1.25), reasoning=True),
]


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------


def _user_content_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
                }
            )
    return blocks


def _assistant_content_blocks(content: Sequence[Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            entry: Dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
            if block.thinking_signature:
                entry["signature"] = block.thinking_signature
            blocks.append(entry)
        elif isinstance(block, ToolCall):
            blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments})
    return blocks


def _build_messages(context: TranscriptContext) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for message in context.messages:
        role = getattr(message, "role", None)
        if role == "user":
            payload.append({"role": "user", "content": _user_content_blocks(message.content)})
        elif role == "assistant":
            payload.append({"role": "assistant", "content": _assistant_content_blocks(message.content)})
        elif role == "toolResult":
            tool_result: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": _user_content_blocks(message.content),
            }
            if message.is_error:
                tool_result["is_error"] = True
            payload.append({"role": "user", "content": [tool_result]})
    return payload


def _tool_declaration(tool: Tool) -> Dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def build_anthropic_payload(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions],
) -> Dict[str, Any]:
    collapsed = collapse_system_messages(context)
    payload: Dict[str, Any] = {
        "model": model.id,
        "max_tokens": options.max_tokens if options and options.max_tokens else model.max_tokens,
        "stream": True,
        "messages": _build_messages(collapsed),
    }

    system_prompt = get_current_system_prompt(collapsed.messages)
    if system_prompt:
        payload["system"] = system_prompt

    tools = get_current_tools(collapsed.messages)
    if tools:
        payload["tools"] = [_tool_declaration(t) for t in tools]

    reasoning = options.reasoning if options else None
    if reasoning and model.reasoning:
        budget = THINKING_BUDGET_TOKENS.get(reasoning, 4096)
        max_tokens = int(payload["max_tokens"])
        budget = max(1024, min(budget, max_tokens - 1))
        if budget >= 1024 and max_tokens > budget:
            payload["max_tokens"] = max_tokens
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

    if options:
        if options.temperature is not None and "thinking" not in payload:
            payload["temperature"] = options.temperature
        if options.metadata:
            user_id = options.metadata.get("user_id")
            if user_id is not None:
                payload["metadata"] = {"user_id": str(user_id)}
    return payload


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "toolUse",
    "max_tokens": "length",
    "pause_turn": "stop",
    "refusal": "stop",
}


def _parse_sse_lines(chunk: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return events


def stream_anthropic(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    outer = create_assistant_message_event_stream()

    async def _run() -> None:
        partial = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(cost=Cost()),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )
        # Per-index block builders: ("text", "") / ("thinking", "") / ("toolUse", id, name)
        block_states: Dict[int, Dict[str, Any]] = {}
        tool_json_buffers: Dict[int, str] = {}
        started = False
        api_key = options.api_key if options else None
        if not api_key:
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            error = AssistantMessage(
                content=[],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(cost=Cost()),
                stop_reason="error",
                error_message="Missing Anthropic API key (set ANTHROPIC_API_KEY or options.api_key)",
                timestamp=int(time.time() * 1000),
            )
            outer.push(AssistantMessageEvent(type="error", reason="error", error=error))
            outer.end(error)
            return

        payload = build_anthropic_payload(model, context, options)
        if options and options.on_payload:
            replaced = options.on_payload(payload, model)
            if asyncio.iscoroutine(replaced):
                replaced = await replaced
            if replaced is not None:
                payload = replaced

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        def _finish(stop_reason: str, error_message: Optional[str] = None) -> None:
            partial.stop_reason = stop_reason
            if error_message is not None:
                partial.error_message = error_message
            calculate_cost(model, partial.usage)
            if stop_reason in ("error", "aborted"):
                outer.push(AssistantMessageEvent(type="error", reason=stop_reason, error=partial))
            else:
                outer.push(AssistantMessageEvent(type="done", reason=stop_reason, message=partial))
            outer.end(partial)

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{model.base_url.rstrip('/')}/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    if options and options.on_response:
                        awaitable = options.on_response(
                            {"status": response.status_code, "headers": dict(response.headers)}, model
                        )
                        if asyncio.iscoroutine(awaitable):
                            await awaitable
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        _finish("error", f"Anthropic API error {response.status_code}: {body[:2000]}")
                        return

                    async for chunk in response.aiter_text():
                        if options and options.signal and options.signal.aborted:
                            _finish("aborted", "Request was aborted")
                            return
                        for event in _parse_sse_lines(chunk):
                            event_type = event.get("type")
                            if event_type == "message_start":
                                message_data = event.get("message") or {}
                                usage_data = message_data.get("usage") or {}
                                partial.response_id = message_data.get("id")
                                partial.response_model = message_data.get("model")
                                partial.usage.input = usage_data.get("input_tokens", 0)
                                partial.usage.cache_read = usage_data.get("cache_read_input_tokens", 0)
                                partial.usage.cache_write = usage_data.get("cache_creation_input_tokens", 0)
                                started = True
                                outer.push(
                                    AssistantMessageEvent(type="start", partial=_clone_partial(partial))
                                )
                            elif event_type == "content_block_start":
                                index = event.get("index", 0)
                                block = event.get("content_block") or {}
                                block_type = block.get("type")
                                if block_type == "text":
                                    partial.content.append(TextContent(text=""))
                                    block_states[index] = {"kind": "text", "content_index": len(partial.content) - 1}
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="text_start",
                                            content_index=len(partial.content) - 1,
                                            partial=_clone_partial(partial),
                                        )
                                    )
                                elif block_type == "thinking":
                                    partial.content.append(ThinkingContent(thinking=""))
                                    block_states[index] = {"kind": "thinking", "content_index": len(partial.content) - 1}
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="thinking_start",
                                            content_index=len(partial.content) - 1,
                                            partial=_clone_partial(partial),
                                        )
                                    )
                                elif block_type == "tool_use":
                                    tool_call = ToolCall(
                                        id=block.get("id", ""),
                                        name=block.get("name", ""),
                                        arguments={},
                                    )
                                    partial.content.append(tool_call)
                                    tool_json_buffers[index] = ""
                                    block_states[index] = {"kind": "toolUse", "content_index": len(partial.content) - 1}
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="toolcall_start",
                                            content_index=len(partial.content) - 1,
                                            partial=_clone_partial(partial),
                                        )
                                    )
                            elif event_type == "content_block_delta":
                                index = event.get("index", 0)
                                delta = event.get("delta") or {}
                                state = block_states.get(index)
                                if state is None:
                                    continue
                                content_index = state["content_index"]
                                if delta.get("type") == "text_delta":
                                    if state["kind"] == "text":
                                        partial.content[content_index].text += delta.get("text", "")
                                        outer.push(
                                            AssistantMessageEvent(
                                                type="text_delta",
                                                content_index=content_index,
                                                delta=delta.get("text", ""),
                                                partial=_clone_partial(partial),
                                            )
                                        )
                                elif delta.get("type") == "thinking_delta":
                                    if state["kind"] == "thinking":
                                        partial.content[content_index].thinking += delta.get("thinking", "")
                                        outer.push(
                                            AssistantMessageEvent(
                                                type="thinking_delta",
                                                content_index=content_index,
                                                delta=delta.get("thinking", ""),
                                                partial=_clone_partial(partial),
                                            )
                                        )
                                elif delta.get("type") == "input_json_delta":
                                    if state["kind"] == "toolUse":
                                        tool_json_buffers[index] = tool_json_buffers.get(index, "") + delta.get(
                                            "partial_json", ""
                                        )
                                        outer.push(
                                            AssistantMessageEvent(
                                                type="toolcall_delta",
                                                content_index=content_index,
                                                delta=delta.get("partial_json", ""),
                                                partial=_clone_partial(partial),
                                            )
                                        )
                            elif event_type == "content_block_stop":
                                index = event.get("index", 0)
                                state = block_states.get(index)
                                if state is None:
                                    continue
                                content_index = state["content_index"]
                                if state["kind"] == "text":
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="text_end",
                                            content_index=content_index,
                                            content=partial.content[content_index].text,
                                            partial=_clone_partial(partial),
                                        )
                                    )
                                elif state["kind"] == "thinking":
                                    signature = (event.get("content_block") or {}).get("signature")
                                    if signature:
                                        partial.content[content_index].thinking_signature = signature
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="thinking_end",
                                            content_index=content_index,
                                            partial=_clone_partial(partial),
                                        )
                                    )
                                elif state["kind"] == "toolUse":
                                    raw_json = tool_json_buffers.get(index, "")
                                    try:
                                        args = json.loads(raw_json) if raw_json.strip() else {}
                                    except json.JSONDecodeError:
                                        args = {}
                                    partial.content[content_index].arguments = args
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="toolcall_end",
                                            content_index=content_index,
                                            tool_call=partial.content[content_index],
                                            partial=_clone_partial(partial),
                                        )
                                    )
                            elif event_type == "message_delta":
                                delta = event.get("delta") or {}
                                usage_data = event.get("usage") or {}
                                partial.usage.output = usage_data.get("output_tokens", partial.usage.output)
                                raw_stop = delta.get("stop_reason")
                                if raw_stop:
                                    partial.raw_stop_reason = raw_stop
                                    partial.stop_reason = _STOP_REASON_MAP.get(raw_stop, "stop")
                            elif event_type == "message_stop":
                                if not started:
                                    outer.push(
                                        AssistantMessageEvent(type="start", partial=_clone_partial(partial))
                                    )
                                partial.usage.total_tokens = (
                                    partial.usage.input
                                    + partial.usage.output
                                    + partial.usage.cache_read
                                    + partial.usage.cache_write
                                )
                                _finish(partial.stop_reason if partial.stop_reason != "pending" else "stop")
                                return
                            elif event_type == "error":
                                error_data = event.get("error") or {}
                                _finish("error", str(error_data.get("message", "Anthropic stream error")))
                                return
                    if started:
                        partial.usage.total_tokens = (
                            partial.usage.input
                            + partial.usage.output
                            + partial.usage.cache_read
                            + partial.usage.cache_write
                        )
                        _finish(partial.stop_reason if partial.stop_reason != "pending" else "stop")
                    else:
                        _finish("error", "Anthropic stream ended without a message")
        except Exception as error:  # noqa: BLE001 - encode in stream like TS
            _finish("error", str(error))

    def _clone_partial(message: AssistantMessage) -> AssistantMessage:
        import copy as _copy

        return _copy.deepcopy(message)

    asyncio.get_running_loop().create_task(_run())
    return outer


def anthropic_provider(base_url: Optional[str] = None) -> Provider:
    models = [
        Model(
            id=m.id,
            name=m.name,
            api=m.api,
            provider=m.provider,
            base_url=base_url or m.base_url,
            reasoning=m.reasoning,
            input=m.input,
            cost=m.cost,
            context_window=m.context_window,
            max_tokens=m.max_tokens,
        )
        for m in ANTHROPIC_MODELS
    ]
    return create_provider(
        id="anthropic",
        name="Anthropic",
        models=models,
        stream=stream_anthropic,
        base_url=base_url or ANTHROPIC_DEFAULT_BASE_URL,
        api_key_env_var="ANTHROPIC_API_KEY",
    )
