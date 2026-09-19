"""OpenAI chat completions provider ported from pi-ai ``src/api/openai-completions.ts``.

Implements the ``openai-completions`` API surface against OpenAI-compatible
chat completion endpoints using SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..event_stream import AssistantMessageEventStream, create_assistant_message_event_stream
from ..models import Provider, calculate_cost, create_provider
from ..transcript import collapse_system_messages, get_current_system_prompt, get_current_tools
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

__all__ = ["openai_provider", "OPENAI_DEFAULT_BASE_URL", "OPENAI_MODELS"]

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


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
        api="openai-completions",
        provider="openai",
        base_url=OPENAI_DEFAULT_BASE_URL,
        reasoning=reasoning,
        input=["text", "image"],
        cost=ModelCost(input=cost[0], output=cost[1], cache_read=cost[2], cache_write=cost[3]),
        context_window=context_window,
        max_tokens=max_tokens,
    )


OPENAI_MODELS: List[Model] = [
    _model("gpt-4.1", "GPT-4.1", 1024000, 32768, (2, 8, 0.5, 2)),
    _model("gpt-4.1-mini", "GPT-4.1 mini", 1024000, 32768, (0.4, 1.6, 0.1, 0.4)),
    _model("gpt-4o", "GPT-4o", 128000, 16384, (2.5, 10, 1.25, 2.5)),
    _model("gpt-4o-mini", "GPT-4o mini", 128000, 16384, (0.15, 0.6, 0.075, 0.15)),
    _model("o3", "o3", 200000, 100000, (2, 8, 0.5, 2), reasoning=True),
]


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------


def _content_parts(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    return parts


def _build_messages(context: TranscriptContext, supports_developer_role: bool = True) -> List[Dict[str, Any]]:
    system_role = "developer" if supports_developer_role else "system"
    payload: List[Dict[str, Any]] = []
    for message in context.messages:
        role = getattr(message, "role", None)
        if role == "system":
            payload.append({"role": system_role, "content": _content_text(message.content)})
        elif role == "user":
            payload.append({"role": "user", "content": _content_parts(message.content)})
        elif role == "assistant":
            entry: Dict[str, Any] = {"role": "assistant"}
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in message.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingContent):
                    continue
                elif isinstance(block, ToolCall):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": json.dumps(block.arguments)},
                        }
                    )
            if text_parts or not tool_calls:
                entry["content"] = "\n".join(text_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            payload.append(entry)
        elif role == "toolResult":
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": _content_text(message.content),
                }
            )
    return payload


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.text for b in content if isinstance(b, TextContent))


def _tool_declaration(tool: Tool) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def build_openai_payload(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions],
) -> Dict[str, Any]:
    collapsed = collapse_system_messages(context)
    payload: Dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": _build_messages(collapsed),
    }

    max_tokens = options.max_tokens if options and options.max_tokens else model.max_tokens
    if max_tokens:
        payload["max_tokens"] = max_tokens

    tools = get_current_tools(collapsed.messages)
    if tools:
        payload["tools"] = [_tool_declaration(t) for t in tools]
        if options and options.tool_choice:
            payload["tool_choice"] = options.tool_choice

    if options:
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        reasoning = options.reasoning
        if reasoning and model.reasoning:
            payload["reasoning_effort"] = reasoning if reasoning != "xhigh" else "high"
        if options.sampling_params:
            payload.update(options.sampling_params)
    if model.sampling_params:
        for key, value in model.sampling_params.items():
            payload.setdefault(key, value)
    return payload


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


_FINISH_MAP = {"stop": "stop", "length": "length", "tool_calls": "toolUse", "function_call": "toolUse"}


def stream_openai(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    outer = create_assistant_message_event_stream()

    async def _run() -> None:
        import copy as _copy

        def _clone(message: AssistantMessage) -> AssistantMessage:
            return _copy.deepcopy(message)

        partial = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(cost=Cost()),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )
        text_index: Optional[int] = None
        thinking_index: Optional[int] = None
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        started = False
        stop_reason = "stop"

        def _finish(kind: str, error_message: Optional[str] = None) -> None:
            if text_index is not None:
                outer.push(
                    AssistantMessageEvent(
                        type="text_end",
                        content_index=text_index,
                        content=partial.content[text_index].text,
                        partial=_clone(partial),
                    )
                )
            if thinking_index is not None:
                outer.push(
                    AssistantMessageEvent(
                        type="thinking_end", content_index=thinking_index, partial=_clone(partial)
                    )
                )
            for entry in sorted(tool_calls_by_index.values(), key=lambda e: e["content_index"]):
                block = partial.content[entry["content_index"]]
                try:
                    block.arguments = json.loads(entry["json"]) if entry["json"].strip() else {}
                except json.JSONDecodeError:
                    block.arguments = {}
                outer.push(
                    AssistantMessageEvent(
                        type="toolcall_end",
                        content_index=entry["content_index"],
                        tool_call=block,
                        partial=_clone(partial),
                    )
                )
            partial.stop_reason = kind
            if error_message is not None:
                partial.error_message = error_message
            partial.usage.total_tokens = (
                partial.usage.input
                + partial.usage.output
                + partial.usage.cache_read
                + partial.usage.cache_write
            )
            calculate_cost(model, partial.usage)
            if kind in ("error", "aborted"):
                outer.push(AssistantMessageEvent(type="error", reason=kind, error=partial))
            else:
                outer.push(AssistantMessageEvent(type="done", reason=kind, message=partial))
            outer.end(partial)

        api_key = options.api_key if options else None
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            _finish("error", "Missing OpenAI API key (set OPENAI_API_KEY or options.api_key)")
            return

        payload = build_openai_payload(model, context, options)
        if options and options.on_payload:
            replaced = options.on_payload(payload, model)
            if asyncio.iscoroutine(replaced):
                replaced = await replaced
            if replaced is not None:
                payload = replaced

        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{model.base_url.rstrip('/')}/chat/completions",
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
                        _finish("error", f"OpenAI API error {response.status_code}: {body[:2000]}")
                        return

                    async for chunk in response.aiter_text():
                        if options and options.signal and options.signal.aborted:
                            _finish("aborted", "Request was aborted")
                            return
                        for line in chunk.splitlines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if not started:
                                started = True
                                outer.push(AssistantMessageEvent(type="start", partial=_clone(partial)))
                            partial.response_id = event.get("id") or partial.response_id
                            usage_data = event.get("usage")
                            if isinstance(usage_data, dict):
                                partial.usage.input = usage_data.get("prompt_tokens", partial.usage.input)
                                partial.usage.output = usage_data.get(
                                    "completion_tokens", partial.usage.output
                                )
                                details = usage_data.get("prompt_tokens_details") or {}
                                partial.usage.cache_read = details.get(
                                    "cached_tokens", partial.usage.cache_read
                                )
                            choices = event.get("choices") or []
                            for choice in choices:
                                delta = choice.get("delta") or {}
                                finish = choice.get("finish_reason")
                                if finish:
                                    stop_reason = _FINISH_MAP.get(finish, "stop")
                                    partial.raw_stop_reason = finish
                                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                                if reasoning:
                                    if thinking_index is None:
                                        partial.content.append(ThinkingContent(thinking=""))
                                        thinking_index = len(partial.content) - 1
                                        outer.push(
                                            AssistantMessageEvent(
                                                type="thinking_start",
                                                content_index=thinking_index,
                                                partial=_clone(partial),
                                            )
                                        )
                                    partial.content[thinking_index].thinking += reasoning
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="thinking_delta",
                                            content_index=thinking_index,
                                            delta=reasoning,
                                            partial=_clone(partial),
                                        )
                                    )
                                text_delta = delta.get("content")
                                if text_delta:
                                    if text_index is None:
                                        partial.content.append(TextContent(text=""))
                                        text_index = len(partial.content) - 1
                                        outer.push(
                                            AssistantMessageEvent(
                                                type="text_start",
                                                content_index=text_index,
                                                partial=_clone(partial),
                                            )
                                        )
                                    partial.content[text_index].text += text_delta
                                    outer.push(
                                        AssistantMessageEvent(
                                            type="text_delta",
                                            content_index=text_index,
                                            delta=text_delta,
                                            partial=_clone(partial),
                                        )
                                    )
                                for tool_delta in delta.get("tool_calls") or []:
                                    tc_index = tool_delta.get("index", 0)
                                    entry = tool_calls_by_index.get(tc_index)
                                    if entry is None:
                                        partial.content.append(
                                            ToolCall(
                                                id=tool_delta.get("id", ""),
                                                name=(tool_delta.get("function") or {}).get("name", ""),
                                                arguments={},
                                            )
                                        )
                                        entry = {
                                            "content_index": len(partial.content) - 1,
                                            "json": "",
                                            "id": tool_delta.get("id", ""),
                                        }
                                        tool_calls_by_index[tc_index] = entry
                                        outer.push(
                                            AssistantMessageEvent(
                                                type="toolcall_start",
                                                content_index=entry["content_index"],
                                                partial=_clone(partial),
                                            )
                                        )
                                    arguments_fragment = (tool_delta.get("function") or {}).get("arguments")
                                    if arguments_fragment:
                                        entry["json"] += arguments_fragment
                                        outer.push(
                                            AssistantMessageEvent(
                                                type="toolcall_delta",
                                                content_index=entry["content_index"],
                                                delta=arguments_fragment,
                                                partial=_clone(partial),
                                            )
                                        )
                    if started:
                        _finish(stop_reason)
                    else:
                        _finish("error", "OpenAI stream ended without a message")
        except Exception as error:  # noqa: BLE001 - encode in stream like TS
            _finish("error", str(error))

    asyncio.get_running_loop().create_task(_run())
    return outer


def openai_provider(base_url: Optional[str] = None) -> Provider:
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
        for m in OPENAI_MODELS
    ]
    return create_provider(
        id="openai",
        name="OpenAI",
        models=models,
        stream=stream_openai,
        base_url=base_url or OPENAI_DEFAULT_BASE_URL,
        api_key_env_var="OPENAI_API_KEY",
    )
