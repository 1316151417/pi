"""Proxy stream function ported from ``src/proxy.ts``.

For apps that route LLM calls through a server. The server manages auth and
proxies requests to LLM providers; delta events omit the partial message to
save bandwidth, so the client reconstructs it here.
"""

from __future__ import annotations

import asyncio
import codecs
import copy
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, Optional, TypedDict, cast

from pi_ai._javascript import javascript_json_parse, javascript_json_stringify, javascript_string, javascript_truthy
from pi_ai._json_runtime import JS_WHITESPACE
from pi_ai._message_wire import (
    assistant_message_from_wire,
    text_content_from_wire,
    thinking_content_from_wire,
    tool_call_from_wire,
    usage_from_wire,
)
from pi_ai._values import UNDEFINED
from pi_ai.abort import AbortSignal
from pi_ai.auth.oauth._http import OAuthHttpResponse, fetch
from pi_ai.event_stream import EventStream
from pi_ai.json_parse import parse_streaming_json
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantContentBlock,
    CacheRetention,
    Cost,
    Model,
    ProviderHeaders,
    JsonObject,
    StopReason,
    TextContent,
    ThinkingContent,
    ThinkingBudgets,
    ThinkingLevel,
    Transport,
    ToolCall,
    TranscriptContext,
    Usage,
)

__all__ = ["ProxyAssistantMessageEvent", "ProxyStreamOptions", "stream_proxy", "process_proxy_event", "ProxyMessageEventStream"]


class _ProxyStart(TypedDict):
    type: Literal["start"]


class _ProxyContentStart(TypedDict):
    type: Literal["text_start", "thinking_start"]
    contentIndex: int


class _ProxyContentDelta(TypedDict):
    type: Literal["text_delta", "thinking_delta", "toolcall_delta"]
    contentIndex: int
    delta: str


class _ProxyContentEnd(TypedDict):
    type: Literal["text_end", "thinking_end"]
    contentIndex: int
    contentSignature: NotRequired[str]


class _ProxyToolCallStart(TypedDict):
    type: Literal["toolcall_start"]
    contentIndex: int
    id: str
    toolName: str


class _ProxyToolCallEnd(TypedDict):
    type: Literal["toolcall_end"]
    contentIndex: int
    toolCall: ToolCall


class _ProxyDone(TypedDict):
    type: Literal["done"]
    reason: Literal["stop", "length", "toolUse"]
    usage: Usage
    providerThinkingLevel: NotRequired[str]


class _ProxyError(TypedDict):
    type: Literal["error"]
    reason: Literal["aborted", "error"]
    errorMessage: NotRequired[str]
    usage: Usage
    providerThinkingLevel: NotRequired[str]


type ProxyAssistantMessageEvent = (
    _ProxyStart | _ProxyContentStart | _ProxyContentDelta | _ProxyContentEnd
    | _ProxyToolCallStart | _ProxyToolCallEnd | _ProxyDone | _ProxyError
)


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
    sampling_params: dict[str, object] | None = None
    max_tokens: Optional[int] = None
    reasoning: ThinkingLevel | None = None
    cache_retention: CacheRetention | None = None
    session_id: Optional[str] = None
    headers: ProviderHeaders | None = None
    metadata: dict[str, object] | None = None
    transport: Transport | None = None
    thinking_budgets: ThinkingBudgets | None = None
    max_retry_delay_ms: Optional[int] = None


def build_proxy_request_options(options: ProxyStreamOptions) -> dict[str, object]:
    values: dict[str, object] = {
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
    return {name: UNDEFINED if value is None else value for name, value in values.items()}


def stream_proxy(
    model: Model,
    context: TranscriptContext,
    options: ProxyStreamOptions,
) -> ProxyMessageEventStream:
    """Stream through a proxy server instead of calling LLM providers directly."""
    stream = ProxyMessageEventStream()

    async def _run() -> None:
        partial = assistant_message_from_wire({
            "role": "assistant",
            "stopReason": "pending",
            "content": [],
            "api": model.api,
            "provider": model.provider,
            "model": model.id,
            "usage": Usage(cost=Cost()).to_json(),
            "timestamp": int(time.time() * 1000),
        })

        response: OAuthHttpResponse | None = None
        reader_started = False
        try:
            payload = {
                "model": model,
                "context": context,
                "options": build_proxy_request_options(options),
            }
            headers = {
                "Authorization": f"Bearer {options.auth_token}",
                "Content-Type": "application/json",
            }

            response = await fetch(
                f"{options.proxy_url}/api/stream", method="POST", headers=headers,
                body=javascript_json_stringify(payload), signal=options.signal,
            )
            if not response.ok:
                error_message = f"Proxy error: {response.status} {response.status_text}"
                try:
                    error_data = await response.json()
                    if isinstance(error_data, Mapping) and javascript_truthy(error_data.get("error")):
                        error_message = f"Proxy error: {javascript_string(error_data['error'])}"
                except Exception:
                    pass
                raise RuntimeError(error_message)
            if response.body is None:
                raise TypeError("Cannot read properties of null (reading 'getReader')")
            reader_started = True
            decoder = codecs.getincrementaldecoder("utf-8-sig")("replace")
            saw_terminal_event = False
            buffer = ""

            def process_line(line: str) -> None:
                nonlocal saw_terminal_event
                if not line.startswith("data: "):
                    return
                data = line[6:].strip(JS_WHITESPACE)
                if not data:
                    return
                event = process_proxy_event(cast(dict[str, object], javascript_json_parse(data)), partial)
                if event is not None:
                    if event.type in ("done", "error"):
                        saw_terminal_event = True
                    stream.push(event)

            async for chunk in response.body:
                if options.signal is not None and options.signal.aborted:
                    raise RuntimeError("Request aborted by user")
                buffer += decoder.decode(chunk, final=False)
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    process_line(line)
            if options.signal is not None and options.signal.aborted:
                raise RuntimeError("Request aborted by user")
            buffer += decoder.decode(b"", final=True)
            if buffer:
                process_line(buffer)
            if not saw_terminal_event:
                partial.stop_reason = "error"
                partial.error_message = "Connection closed by proxy server before the response completed"
                stream.push(AssistantMessageEvent(type="error", reason="error", error=partial))
            stream.end()
        except (Exception, asyncio.CancelledError) as error:
            aborted = options.signal is not None and options.signal.aborted
            reason = "aborted" if aborted else "error"
            partial.stop_reason = reason
            partial.error_message = "Request aborted by user" if aborted and reader_started else str(error)
            stream.push(AssistantMessageEvent(type="error", reason=reason, error=partial))
            stream.end()
        finally:
            if response is not None:
                await response.cancel_body()

    asyncio.get_running_loop().create_task(_run())
    return stream


def process_proxy_event(proxy_event: Mapping[str, object], partial: AssistantMessage) -> Optional[AssistantMessageEvent]:
    """Process a proxy event and update the partial message."""
    event_type = proxy_event.get("type")

    def _content_at(index: int) -> AssistantContentBlock | None:
        return partial.content[index] if 0 <= index < len(partial.content) else None

    if event_type == "start":
        return AssistantMessageEvent(type="start", partial=partial)

    if event_type == "text_start":
        index = int(cast(int, proxy_event["contentIndex"]))
        _assign(partial, index, text_content_from_wire({"type": "text", "text": ""}))
        return AssistantMessageEvent(type="text_start", content_index=index, partial=partial)

    if event_type == "text_delta":
        index = int(cast(int, proxy_event["contentIndex"]))
        content = _content_at(index)
        if isinstance(content, TextContent):
            content.text += javascript_string(proxy_event["delta"])
            return AssistantMessageEvent(
                type="text_delta", content_index=index, delta=cast(str, proxy_event["delta"]), partial=partial
            )
        raise RuntimeError("Received text_delta for non-text content")

    if event_type == "text_end":
        index = int(cast(int, proxy_event["contentIndex"]))
        content = _content_at(index)
        if isinstance(content, TextContent):
            content.text_signature = cast(str, proxy_event.get("contentSignature", UNDEFINED))
            return AssistantMessageEvent(
                type="text_end", content_index=index, content=content.text, partial=partial
            )
        raise RuntimeError("Received text_end for non-text content")

    if event_type == "thinking_start":
        index = int(cast(int, proxy_event["contentIndex"]))
        _assign(partial, index, thinking_content_from_wire({"type": "thinking", "thinking": ""}))
        return AssistantMessageEvent(type="thinking_start", content_index=index, partial=partial)

    if event_type == "thinking_delta":
        index = int(cast(int, proxy_event["contentIndex"]))
        content = _content_at(index)
        if isinstance(content, ThinkingContent):
            content.thinking += javascript_string(proxy_event["delta"])
            return AssistantMessageEvent(
                type="thinking_delta", content_index=index, delta=cast(str, proxy_event["delta"]), partial=partial
            )
        raise RuntimeError("Received thinking_delta for non-thinking content")

    if event_type == "thinking_end":
        index = int(cast(int, proxy_event["contentIndex"]))
        content = _content_at(index)
        if isinstance(content, ThinkingContent):
            content.thinking_signature = cast(str, proxy_event.get("contentSignature", UNDEFINED))
            return AssistantMessageEvent(
                type="thinking_end", content_index=index, content=content.thinking, partial=partial
            )
        raise RuntimeError("Received thinking_end for non-thinking content")

    if event_type == "toolcall_start":
        index = int(cast(int, proxy_event["contentIndex"]))
        tool_call = tool_call_from_wire({
            "type": "toolCall", "id": proxy_event["id"], "name": proxy_event["toolName"],
            "arguments": {}, "partialJson": "",
        })
        _assign(partial, index, tool_call)
        return AssistantMessageEvent(type="toolcall_start", content_index=index, partial=partial)

    if event_type == "toolcall_delta":
        index = int(cast(int, proxy_event["contentIndex"]))
        content = _content_at(index)
        if isinstance(content, ToolCall):
            buffer = getattr(content, "partial_json", "") + javascript_string(proxy_event["delta"])
            setattr(content, "partial_json", buffer)
            parsed = parse_streaming_json(buffer)
            content.arguments = cast(JsonObject, parsed if javascript_truthy(parsed) else {})
            partial.content[index] = copy.copy(content)
            return AssistantMessageEvent(
                type="toolcall_delta", content_index=index, delta=cast(str, proxy_event["delta"]), partial=partial
            )
        raise RuntimeError("Received toolcall_delta for non-toolCall content")

    if event_type == "toolcall_end":
        index = int(cast(int, proxy_event["contentIndex"]))
        content = _content_at(index)
        if isinstance(content, ToolCall):
            incoming = proxy_event["toolCall"]
            tool_call_fields = incoming.to_json() if isinstance(incoming, ToolCall) else cast(Mapping[str, object], incoming)
            for key, value in tool_call_fields.items():
                setattr(content, "thought_signature" if key == "thoughtSignature" else "partial_json" if key == "partialJson" else key, value)
            if hasattr(content, "partial_json"):
                delattr(content, "partial_json")
            return AssistantMessageEvent(
                type="toolcall_end", content_index=index, tool_call=content, partial=partial
            )
        return None

    if event_type == "done":
        partial.stop_reason = cast(StopReason, proxy_event["reason"])
        usage = proxy_event["usage"]
        partial.usage = usage if isinstance(usage, Usage) else usage_from_wire(cast(Mapping[str, object], usage))
        if proxy_event.get("providerThinkingLevel", UNDEFINED) is not UNDEFINED:
            partial.provider_thinking_level = cast(str, proxy_event["providerThinkingLevel"])
        return AssistantMessageEvent(type="done", reason=cast(StopReason, proxy_event["reason"]), message=partial)

    if event_type == "error":
        partial.stop_reason = cast(StopReason, proxy_event["reason"])
        partial.error_message = cast(str, proxy_event.get("errorMessage", UNDEFINED))
        usage = proxy_event["usage"]
        partial.usage = usage if isinstance(usage, Usage) else usage_from_wire(cast(Mapping[str, object], usage))
        if proxy_event.get("providerThinkingLevel", UNDEFINED) is not UNDEFINED:
            partial.provider_thinking_level = cast(str, proxy_event["providerThinkingLevel"])
        return AssistantMessageEvent(type="error", reason=cast(StopReason, proxy_event["reason"]), error=partial)

    logging.getLogger(__name__).warning("Unhandled proxy event type: %s", javascript_string(event_type if event_type is not None else UNDEFINED))
    return None


def _assign(partial: AssistantMessage, index: int, block: AssistantContentBlock) -> None:
    while len(partial.content) <= index:
        partial.content.append(cast(AssistantContentBlock, UNDEFINED))
    partial.content[index] = block
