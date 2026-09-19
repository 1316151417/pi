"""Proxy stream function ported from ``src/proxy.ts``.

For apps that route LLM calls through a server. The server manages auth and
proxies requests to LLM providers; delta events omit the partial message to
save bandwidth, so the client reconstructs it here.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import httpx

from ._pi_ai.abort import AbortSignal
from ._pi_ai.event_stream import EventStream
from ._pi_ai.json_parse import parse_streaming_json
from ._pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Cost,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    TranscriptContext,
    Usage,
)

__all__ = ["ProxyStreamOptions", "stream_proxy", "process_proxy_event", "ProxyMessageEventStream"]


class ProxyMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    def __init__(self) -> None:
        super().__init__(
            lambda event: event.type in ("done", "error"),
            lambda event: (
                event.message
                if event.type == "done"
                else event.error if event.type == "error" else _raise_unexpected()
            ),
        )


def _raise_unexpected() -> AssistantMessage:
    raise RuntimeError("Unexpected event type")


@dataclass
class ProxyStreamOptions:
    """Proxy request options: serializable stream options plus local transport settings."""

    auth_token: str = ""
    proxy_url: str = ""
    signal: Optional[AbortSignal] = None
    temperature: Optional[float] = None
    sampling_params: Optional[dict] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[str] = None
    cache_retention: Optional[str] = None
    session_id: Optional[str] = None
    headers: Optional[dict] = None
    metadata: Optional[dict] = None
    transport: Optional[str] = None
    thinking_budgets: Optional[dict] = None
    max_retry_delay_ms: Optional[int] = None


def build_proxy_request_options(options: ProxyStreamOptions) -> dict:
    return {
        "temperature": options.temperature,
        "samplingParams": options.sampling_params,
        "maxTokens": options.max_tokens,
        "reasoning": options.reasoning,
        "cacheRetention": options.cache_retention,
        "sessionId": options.session_id,
        "headers": options.headers,
        "metadata": options.metadata,
        "transport": options.transport,
        "thinkingBudgets": options.thinking_budgets,
        "maxRetryDelayMs": options.max_retry_delay_ms,
    }


def _model_to_json(model: Model) -> dict:
    return {
        "id": model.id,
        "name": model.name,
        "api": model.api,
        "provider": model.provider,
        "baseUrl": model.base_url,
        "reasoning": model.reasoning,
        "input": model.input,
        "cost": {
            "input": model.cost.input,
            "output": model.cost.output,
            "cacheRead": model.cost.cache_read,
            "cacheWrite": model.cost.cache_write,
        },
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
    }


def _context_to_json(context: TranscriptContext) -> dict:
    return {"messages": [_message_to_json(message) for message in context.messages]}


def _message_to_json(message: Any) -> dict:
    if hasattr(message, "to_json"):
        return message.to_json()
    return {"role": getattr(message, "role", "unknown")}


def stream_proxy(
    model: Model,
    context: TranscriptContext,
    options: ProxyStreamOptions,
) -> ProxyMessageEventStream:
    """Stream through a proxy server instead of calling LLM providers directly."""
    stream = ProxyMessageEventStream()

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

        try:
            payload = {
                "model": _model_to_json(model),
                "context": _context_to_json(context),
                "options": build_proxy_request_options(options),
            }
            headers = {
                "Authorization": f"Bearer {options.auth_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{options.proxy_url.rstrip('/')}/api/stream",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        error_message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                        try:
                            body = (await response.aread()).decode("utf-8", "replace")
                            error_data = json.loads(body)
                            if isinstance(error_data, dict) and error_data.get("error"):
                                error_message = f"Proxy error: {error_data['error']}"
                        except (json.JSONDecodeError, ValueError):
                            pass
                        raise RuntimeError(error_message)

                    saw_terminal_event = False
                    buffer = ""

                    def _process_line(line: str) -> None:
                        nonlocal saw_terminal_event
                        if not line.startswith("data: "):
                            return
                        data = line[6:].strip()
                        if not data:
                            return
                        proxy_event = json.loads(data)
                        event = process_proxy_event(proxy_event, partial)
                        if event is not None:
                            if event.type in ("done", "error"):
                                saw_terminal_event = True
                            stream.push(event)

                    async for chunk in response.aiter_text():
                        if options.signal is not None and options.signal.aborted:
                            raise RuntimeError("Request aborted by user")
                        buffer += chunk
                        lines = buffer.split("\n")
                        buffer = lines.pop() if lines else ""
                        for line in lines:
                            _process_line(line)

                    if options.signal is not None and options.signal.aborted:
                        raise RuntimeError("Request aborted by user")

                    if buffer:
                        _process_line(buffer)

                    if not saw_terminal_event:
                        # A clean EOF without a done/error event means the server
                        # dropped the response mid-stream.
                        partial.stop_reason = "error"
                        partial.error_message = (
                            "Connection closed by proxy server before the response completed"
                        )
                        stream.push(
                            AssistantMessageEvent(type="error", reason="error", error=partial)
                        )
                    stream.end()
        except Exception as error:  # noqa: BLE001 - encode in stream like TS
            aborted = options.signal is not None and options.signal.aborted
            reason = "aborted" if aborted else "error"
            partial.stop_reason = reason
            partial.error_message = str(error)
            stream.push(AssistantMessageEvent(type="error", reason=reason, error=partial))
            stream.end()

    asyncio.get_running_loop().create_task(_run())
    return stream


def process_proxy_event(proxy_event: dict, partial: AssistantMessage) -> Optional[AssistantMessageEvent]:
    """Process a proxy event and update the partial message."""
    event_type = proxy_event.get("type")

    def _content_at(index: int) -> Any:
        return partial.content[index] if 0 <= index < len(partial.content) else None

    if event_type == "start":
        return AssistantMessageEvent(type="start", partial=partial)

    if event_type == "text_start":
        index = proxy_event["contentIndex"]
        _assign(partial, index, TextContent(text=""))
        return AssistantMessageEvent(type="text_start", content_index=index, partial=partial)

    if event_type == "text_delta":
        index = proxy_event["contentIndex"]
        content = _content_at(index)
        if isinstance(content, TextContent):
            content.text += proxy_event["delta"]
            return AssistantMessageEvent(
                type="text_delta", content_index=index, delta=proxy_event["delta"], partial=partial
            )
        raise RuntimeError("Received text_delta for non-text content")

    if event_type == "text_end":
        index = proxy_event["contentIndex"]
        content = _content_at(index)
        if isinstance(content, TextContent):
            content.text_signature = proxy_event.get("contentSignature")
            return AssistantMessageEvent(
                type="text_end", content_index=index, content=content.text, partial=partial
            )
        raise RuntimeError("Received text_end for non-text content")

    if event_type == "thinking_start":
        index = proxy_event["contentIndex"]
        _assign(partial, index, ThinkingContent(thinking=""))
        return AssistantMessageEvent(type="thinking_start", content_index=index, partial=partial)

    if event_type == "thinking_delta":
        index = proxy_event["contentIndex"]
        content = _content_at(index)
        if isinstance(content, ThinkingContent):
            content.thinking += proxy_event["delta"]
            return AssistantMessageEvent(
                type="thinking_delta", content_index=index, delta=proxy_event["delta"], partial=partial
            )
        raise RuntimeError("Received thinking_delta for non-thinking content")

    if event_type == "thinking_end":
        index = proxy_event["contentIndex"]
        content = _content_at(index)
        if isinstance(content, ThinkingContent):
            content.thinking_signature = proxy_event.get("contentSignature")
            return AssistantMessageEvent(
                type="thinking_end", content_index=index, content=content.thinking, partial=partial
            )
        raise RuntimeError("Received thinking_end for non-thinking content")

    if event_type == "toolcall_start":
        index = proxy_event["contentIndex"]
        tool_call = ToolCall(id=proxy_event["id"], name=proxy_event["toolName"], arguments={})
        setattr(tool_call, "partial_json", "")
        _assign(partial, index, tool_call)
        return AssistantMessageEvent(type="toolcall_start", content_index=index, partial=partial)

    if event_type == "toolcall_delta":
        index = proxy_event["contentIndex"]
        content = _content_at(index)
        if isinstance(content, ToolCall):
            buffer = getattr(content, "partial_json", "") + proxy_event["delta"]
            setattr(content, "partial_json", buffer)
            parsed = parse_streaming_json(buffer)
            content.arguments = parsed if isinstance(parsed, dict) else {}
            partial.content[index] = content
            return AssistantMessageEvent(
                type="toolcall_delta", content_index=index, delta=proxy_event["delta"], partial=partial
            )
        raise RuntimeError("Received toolcall_delta for non-toolCall content")

    if event_type == "toolcall_end":
        index = proxy_event["contentIndex"]
        content = _content_at(index)
        if isinstance(content, ToolCall):
            tool_call = proxy_event["toolCall"]
            content.id = tool_call.get("id", content.id)
            content.name = tool_call.get("name", content.name)
            content.arguments = tool_call.get("arguments") or {}
            if hasattr(content, "partial_json"):
                delattr(content, "partial_json")
            return AssistantMessageEvent(
                type="toolcall_end", content_index=index, tool_call=content, partial=partial
            )
        return None

    if event_type == "done":
        partial.stop_reason = proxy_event["reason"]
        partial.usage = _usage_from_json(proxy_event.get("usage"))
        if proxy_event.get("providerThinkingLevel") is not None:
            partial.provider_thinking_level = proxy_event["providerThinkingLevel"]
        return AssistantMessageEvent(type="done", reason=proxy_event["reason"], message=partial)

    if event_type == "error":
        partial.stop_reason = proxy_event["reason"]
        partial.error_message = proxy_event.get("errorMessage")
        partial.usage = _usage_from_json(proxy_event.get("usage"))
        if proxy_event.get("providerThinkingLevel") is not None:
            partial.provider_thinking_level = proxy_event["providerThinkingLevel"]
        return AssistantMessageEvent(type="error", reason=proxy_event["reason"], error=partial)

    import sys

    sys.stderr.write(f"Unhandled proxy event type: {event_type}\n")
    return None


def _assign(partial: AssistantMessage, index: int, block: Any) -> None:
    while len(partial.content) <= index:
        partial.content.append(TextContent(text=""))
    partial.content[index] = block


def _usage_from_json(data: Optional[dict]) -> Usage:
    from ._pi_ai.types import usage_from_json

    return usage_from_json(data) or Usage(cost=Cost())
