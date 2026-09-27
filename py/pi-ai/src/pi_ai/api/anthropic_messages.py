"""Anthropic Messages request, replay and event handling from the TypeScript API."""

from __future__ import annotations

import asyncio
import codecs
import copy
import inspect
import math
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Literal, NotRequired, TypedDict, cast

from .._javascript import javascript_string
from .._json_runtime import JS_WHITESPACE, utf16_units
from .._values import JSON_NULL, UNDEFINED, Undefined
from ..abort import AbortSignal
from ..event_stream import AssistantMessageEventStream
from ..json_parse import parse_json_with_repair, parse_streaming_json
from ..models import calculate_cost
from ..text import get_system_message_text, render_system_message_update
from ..transcript import get_current_tools, get_declared_tools, get_initial_system_message, has_tool_redefinitions, resolve_transcript
from ..types import (
    AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent, CacheRetention, FetchFunction,
    ImageContent, JsonObject, Message, Model, ModelCost, ProviderEnv, ProviderHeaders, ProviderResponse,
    SimpleStreamOptions, StopReason, StreamOptions, SystemMessage, TextContent, ThinkingContent,
    Tool, ToolCall, ToolResultMessage, TranscriptContext, Usage, UserMessage,
)
from ..utils.diagnostics import append_assistant_message_diagnostic
from ..utils.headers import headers_to_record
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from ._anthropic_http import AnthropicClient, AnthropicHttpClient, AnthropicRequestOptions, AnthropicResponse
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .github_copilot_headers import CopilotDynamicHeaderParams, build_copilot_dynamic_headers, has_copilot_vision_input
from .simple_options import adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context
from .transform_messages import transform_messages

type AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
type AnthropicThinkingDisplay = Literal["summarized", "omitted"]


class AnthropicNamedToolChoice(TypedDict):
    type: Literal["tool"]
    name: str


type AnthropicToolChoice = Literal["auto", "any", "none"] | AnthropicNamedToolChoice


@dataclass
class AnthropicOptions(StreamOptions):
    thinking_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    effort: AnthropicEffort | None = None
    thinking_display: AnthropicThinkingDisplay | None = None
    interleaved_thinking: bool | None = None
    tool_choice: AnthropicToolChoice | None = None
    client: AnthropicClient | None = None


class CacheControlEphemeral(TypedDict):
    type: Literal["ephemeral"]
    ttl: NotRequired[Literal["1h"]]


@dataclass
class _Compat:
    supports_eager_tool_input_streaming: bool = True
    supports_long_cache_retention: bool = True
    send_session_affinity_headers: bool = False
    session_affinity_format: str | None = None
    supports_cache_control_on_tools: bool = True
    supports_temperature: bool = True
    allow_empty_signature: bool = False
    supports_strict_tools: bool = False
    supports_mid_convo_system_messages: bool = False
    supports_mid_convo_tool_changes: bool = False


_COMPAT_FIELDS = {
    "supportsEagerToolInputStreaming": "supports_eager_tool_input_streaming",
    "supportsLongCacheRetention": "supports_long_cache_retention",
    "sendSessionAffinityHeaders": "send_session_affinity_headers",
    "sessionAffinityFormat": "session_affinity_format",
    "supportsCacheControlOnTools": "supports_cache_control_on_tools",
    "supportsTemperature": "supports_temperature", "allowEmptySignature": "allow_empty_signature",
    "supportsStrictTools": "supports_strict_tools",
    "supportsMidConvoSystemMessages": "supports_mid_convo_system_messages",
    "supportsMidConvoToolChanges": "supports_mid_convo_tool_changes",
}
_CLAUDE_CODE_VERSION = "2.1.251"
_CLAUDE_CODE_NAMES = {name.lower(): name for name in (
    "Read", "Write", "Edit", "Bash", "Grep", "Glob", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode",
    "KillShell", "NotebookEdit", "Skill", "Task", "TaskOutput", "TodoWrite", "WebFetch", "WebSearch",
)}
_DEFERRED_TOOL_PLACEHOLDER: dict[str, object] = {
    "name": "__pi_deferred_placeholder__", "description": "Reserved placeholder. Never available. Never call this.",
    "input_schema": {"type": "object", "properties": {}, "required": []}, "defer_loading": True,
}
_ANTHROPIC_MESSAGE_EVENTS = frozenset((
    "message_start", "message_delta", "message_stop", "content_block_start", "content_block_delta", "content_block_stop",
))


def _truthy(value: object) -> bool:
    if value is None or value is UNDEFINED or value is JSON_NULL or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return value != "" if isinstance(value, str) else True


def _property(value: object, name: str, default: object = UNDEFINED) -> object:
    return value.get(name, default) if isinstance(value, Mapping) else default


def _required_property(value: object, name: str) -> object:
    if value is None or value is UNDEFINED or value is JSON_NULL:
        raise TypeError(f"Cannot read properties of {javascript_string(value)} (reading '{name}')")
    return _property(value, name)


def _nullish(value: object, fallback: object) -> object:
    return fallback if value is None or value is UNDEFINED or value is JSON_NULL else value


def _resolve_cache_retention(cache_retention: CacheRetention | None = None, env: ProviderEnv | None = None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else "short"


def _get_compat(model: Model) -> _Compat:
    is_open_router = model.provider == "openrouter" or "openrouter.ai" in model.base_url
    compat = _Compat(send_session_affinity_headers=is_open_router, session_affinity_format="openrouter" if is_open_router else None)
    for wire, name in _COMPAT_FIELDS.items():
        value = (model.compat or {}).get(wire)
        if value is not None:
            setattr(compat, name, value)
    return compat


def _get_cache_control(model: Model, retention: CacheRetention | None, env: ProviderEnv | None) -> CacheControlEphemeral | None:
    resolved = _resolve_cache_retention(retention, env)
    if resolved == "none":
        return None
    result: CacheControlEphemeral = {"type": "ephemeral"}
    if resolved == "long" and _get_compat(model).supports_long_cache_retention:
        result["ttl"] = "1h"
    return result


def _to_claude_code_name(name: str) -> str:
    return _CLAUDE_CODE_NAMES.get(name.lower(), name)


def _convert_content_blocks(content: Sequence[TextContent | ImageContent]) -> str | list[dict[str, object]]:
    if not any(block.type == "image" for block in content):
        return sanitize_surrogates("\n".join(cast(TextContent, block).text for block in content))
    blocks: list[dict[str, object]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": block.mime_type, "data": block.data}})
    if all(block["type"] == "image" for block in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


def _assert_request_auth(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> None:
    if api_key:
        return
    if headers is not None:
        for name, value in headers.items():
            if name.lower() in ("authorization", "x-api-key", "cf-aig-authorization") and value is not None and value.strip(JS_WHITESPACE):
                return
    raise ValueError(f"No API key for provider: {provider}")


def _create_client(
    model: Model, api_key: str | None, options_headers: ProviderHeaders | None, custom_fetch: FetchFunction | None,
    dynamic_headers: Mapping[str, str] | None, session_id: str | None,
) -> tuple[AnthropicClient, bool]:
    base_headers: dict[str, str | None] = {
        "User-Agent": get_pi_user_agent(), "accept": "application/json", "anthropic-dangerous-direct-browser-access": "true",
    }
    if model.provider == "github-copilot":
        base_headers.update(model.headers or {})
        base_headers.update(dynamic_headers or {})
        base_headers.update(options_headers or {})
        return AnthropicHttpClient(api_key=None, auth_token=api_key, base_url=model.base_url, custom_fetch=custom_fetch, default_headers=base_headers), False
    is_oauth = bool(api_key and "sk-ant-oat" in api_key)
    if is_oauth:
        base_headers.update({"user-agent": f"claude-cli/{_CLAUDE_CODE_VERSION}", "x-app": "cli"})
    else:
        compat = _get_compat(model)
        if session_id and compat.send_session_affinity_headers:
            base_headers["x-session-id" if compat.session_affinity_format == "openrouter" else "x-session-affinity"] = session_id
    base_headers.update(model.headers or {})
    base_headers.update(options_headers or {})
    return AnthropicHttpClient(
        api_key=None if is_oauth else api_key, auth_token=api_key if is_oauth else None,
        base_url=model.base_url, custom_fetch=custom_fetch, default_headers=base_headers,
    ), is_oauth


@dataclass
class _ServerSentEvent:
    event: str | None
    data: str
    raw: list[str]


@dataclass
class _SseDecoderState:
    event: str | None = None
    data: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)


def _flush_sse_event(state: _SseDecoderState) -> _ServerSentEvent | None:
    if not state.event and not state.data:
        return None
    event = _ServerSentEvent(state.event, "\n".join(state.data), state.raw[:])
    state.event = None
    state.data.clear()
    state.raw.clear()
    return event


def _decode_sse_line(line: str, state: _SseDecoderState) -> _ServerSentEvent | None:
    if line == "":
        return _flush_sse_event(state)
    state.raw.append(line)
    if line.startswith(":"):
        return None
    field_name, delimiter, value = line.partition(":")
    if not delimiter:
        value = ""
    if value.startswith(" "):
        value = value[1:]
    if field_name == "event":
        state.event = value
    elif field_name == "data":
        state.data.append(value)
    return None


def _consume_line(text: str) -> tuple[str, str] | None:
    carriage = text.find("\r")
    newline = text.find("\n")
    index = newline if carriage == -1 else carriage if newline == -1 else min(carriage, newline)
    if index == -1:
        return None
    next_index = index + (2 if text[index:index + 2] == "\r\n" else 1)
    return text[:index], text[next_index:]


async def _iterate_sse_messages(body: AsyncIterable[bytes], signal: AbortSignal | None) -> AsyncIterator[_ServerSentEvent]:
    # Match this API's parser, including immediate consumption of a trailing CR.
    # Releasing a JS reader does not cancel its body; neither do we explicitly
    # close a caller-owned body iterator when decoding stops early.
    reader = aiter(body)
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    state = _SseDecoderState()
    buffer = ""
    while True:
        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        try:
            value = await anext(reader)
        except StopAsyncIteration:
            break
        buffer += decoder.decode(value, final=False)
        while (consumed := _consume_line(buffer)) is not None:
            line, buffer = consumed
            event = _decode_sse_line(line, state)
            if event is not None:
                yield event
    buffer += decoder.decode(b"", final=True)
    while (consumed := _consume_line(buffer)) is not None:
        line, buffer = consumed
        event = _decode_sse_line(line, state)
        if event is not None:
            yield event
    if buffer:
        event = _decode_sse_line(buffer, state)
        if event is not None:
            yield event
    event = _flush_sse_event(state)
    if event is not None:
        yield event


async def _iterate_anthropic_events(response: AnthropicResponse, signal: AbortSignal | None) -> AsyncIterator[object]:
    body = response.body
    if body is None:
        raise RuntimeError("Attempted to iterate over an Anthropic response with no body")
    saw_start = False
    saw_end = False
    async for sse in _iterate_sse_messages(body, signal):
        if sse.event == "error":
            raise RuntimeError(sse.data)
        if (sse.event or "") not in _ANTHROPIC_MESSAGE_EVENTS:
            continue
        try:
            event = parse_json_with_repair(sse.data)
            event_type = _required_property(event, "type")
            if event_type == "message_start":
                saw_start = True
            elif event_type == "message_stop":
                saw_end = True
        except Exception as error:
            raw = "\\n".join(sse.raw)
            raise RuntimeError(f"Could not parse Anthropic SSE event {sse.event}: {error}; data={sse.data}; raw={raw}") from error
        yield event
    if saw_start and not saw_end:
        raise RuntimeError("Anthropic stream ended before message_stop")


@dataclass
class _TextBlock(TextContent):
    index: object = UNDEFINED

    def to_json(self) -> JsonObject:
        value = super().to_json()
        if self.index is not UNDEFINED:
            value["index"] = cast(int, self.index)
        return value


@dataclass
class _ThinkingBlock(ThinkingContent):
    index: object = UNDEFINED

    def to_json(self) -> JsonObject:
        value = super().to_json()
        if self.index is not UNDEFINED:
            value["index"] = cast(int, self.index)
        return value


@dataclass
class _ToolBlock(ToolCall):
    partial_json: str | Undefined = ""
    index: object = UNDEFINED

    def to_json(self) -> JsonObject:
        value = super().to_json()
        if self.partial_json is not UNDEFINED:
            value["partialJson"] = self.partial_json
        if self.index is not UNDEFINED:
            value["index"] = cast(int, self.index)
        return value


def stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    result = AssistantMessageEventStream()
    normalized = resolve_transcript(context, _get_compat(model).supports_mid_convo_system_messages)
    current_tools = get_current_tools(normalized.messages)
    options = options or AnthropicOptions()

    async def run() -> None:
        output = AssistantMessage(
            content=[], api=model.api, provider=model.provider, model=model.id,
            provider_thinking_level=cast(str, _nullish(getattr(options, "effort", None), "high")) if _truthy((model.compat or {}).get("supportsMidConvoEffort")) else None,
            usage=Usage(), stop_reason="pending", timestamp=time.time_ns() // 1_000_000,
        )
        try:
            usage_model = model
            transformations: list[object] | None = None
            client = cast(AnthropicClient | None, getattr(options, "client", None))
            if client is not None:
                is_oauth = False
            else:
                _assert_request_auth(model.provider, options.api_key, options.headers)
                dynamic_headers = build_copilot_dynamic_headers(CopilotDynamicHeaderParams(
                    messages=normalized.messages, has_images=has_copilot_vision_input(normalized.messages),
                )) if model.provider == "github-copilot" else None
                session = None if _resolve_cache_retention(options.cache_retention, options.env) == "none" else options.session_id
                client, is_oauth = _create_client(model, options.api_key, options.headers, options.fetch, dynamic_headers, session)
            params = _build_params(model, normalized, is_oauth, options)
            if options.on_payload is not None:
                replacement = options.on_payload(params, model)
                if inspect.isawaitable(replacement):
                    replacement = await replacement
                if replacement is not None and replacement is not UNDEFINED:
                    if isinstance(replacement, Mapping):
                        params = dict(replacement)
                    elif isinstance(replacement, str):
                        params = {str(key): value for key, value in enumerate(utf16_units(replacement))}
                    elif isinstance(replacement, (list, tuple)):
                        params = {str(index): value for index, value in enumerate(replacement)}
                    else:
                        params = {}
                    params["stream"] = True
            request_options = AnthropicRequestOptions(signal=options.signal, timeout_ms=options.timeout_ms, max_retries=0)

            async def request() -> AnthropicResponse:
                return await client.beta.messages.create(params, request_options).as_response()

            response = await retry_provider_request(
                request, max_retries=options.max_retries or 0,
                max_retry_delay_ms=options.max_retry_delay_ms, signal=options.signal,
            )
            if options.on_response is not None:
                callback = options.on_response(ProviderResponse(status=response.status, headers=headers_to_record(response.headers)), model)
                if inspect.isawaitable(callback):
                    await callback
            result.push(AssistantMessageEvent(type="start", partial=output))
            blocks = cast(list[_TextBlock | _ThinkingBlock | _ToolBlock], output.content)
            async for event in _iterate_anthropic_events(response, options.signal):
                event_type = _required_property(event, "type")
                if event_type == "message_start":
                    message = _required_property(event, "message")
                    output.response_id = cast(str, _required_property(message, "id"))
                    incoming = _property(message, "input_transformations")
                    if isinstance(incoming, list):
                        transformations = incoming
                    response_model = _required_property(message, "model")
                    if response_model != model.id:
                        output.response_model = cast(str, response_model)
                    fallback_cost = UNDEFINED
                    if response_model != model.id:
                        for fallback in (model.compat or {}).get("allowedFallbackModels") or []:
                            if fallback["provider"] == model.provider and fallback["model"] == response_model:
                                fallback_cost = fallback.get("cost", UNDEFINED)
                                break
                    if _truthy(fallback_cost):
                        usage_model = copy.copy(model)
                        usage_model.id = cast(str, response_model)
                        usage_model.cost = fallback_cost if isinstance(fallback_cost, ModelCost) else ModelCost._from_wire(cast(Mapping[str, object], fallback_cost))
                    else:
                        usage_model = model
                    usage = _required_property(message, "usage")
                    for wire, attribute in (("input_tokens", "input"), ("output_tokens", "output"), ("cache_read_input_tokens", "cache_read"), ("cache_creation_input_tokens", "cache_write")):
                        value = _required_property(usage, wire)
                        setattr(output.usage, attribute, value if _truthy(value) else 0)
                    value = _property(_property(usage, "cache_creation"), "ephemeral_1h_input_tokens")
                    output.usage.cache_write_1h = cast(int, value if _truthy(value) else 0)
                    output.usage.total_tokens = output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
                    calculate_cost(usage_model, output.usage)
                elif event_type == "content_block_start":
                    content = _required_property(event, "content_block")
                    kind = _required_property(content, "type")
                    if kind == "fallback":
                        if output.content:
                            raise RuntimeError("Anthropic performed an unsupported mid-output model fallback")
                        continue
                    index = _property(event, "index")
                    if kind == "text":
                        blocks.append(_TextBlock(text=cast(str, _nullish(_property(content, "text"), "")), index=index))
                        start_type = "text_start"
                    elif kind in ("thinking", "redacted_thinking"):
                        blocks.append(_ThinkingBlock(
                            thinking="[Reasoning redacted]" if kind == "redacted_thinking" else cast(str, _nullish(_property(content, "thinking"), "")),
                            thinking_signature=cast(str, _property(content, "data") if kind == "redacted_thinking" else _nullish(_property(content, "signature"), "")),
                            redacted=True if kind == "redacted_thinking" else None, index=index,
                        ))
                        start_type = "thinking_start"
                    elif kind == "tool_use":
                        name = cast(str, _property(content, "name"))
                        if is_oauth:
                            name = next((tool.name for tool in current_tools if tool.name.lower() == name.lower()), name)
                        blocks.append(_ToolBlock(id=cast(str, _property(content, "id")), name=name,
                            arguments=cast(JsonObject, _nullish(_property(content, "input"), {})), partial_json="", index=index))
                        start_type = "toolcall_start"
                    else:
                        continue
                    result.push(AssistantMessageEvent(type=start_type, content_index=len(blocks) - 1, partial=output))
                elif event_type in ("content_block_delta", "content_block_stop"):
                    event_index = _property(event, "index")
                    index = next((i for i, block in enumerate(blocks) if block.index == event_index and (isinstance(block.index, bool) == isinstance(event_index, bool))), -1)
                    block = blocks[index] if index >= 0 else None
                    if event_type == "content_block_stop":
                        if block is None:
                            continue
                        block.index = UNDEFINED
                        if isinstance(block, _TextBlock):
                            result.push(AssistantMessageEvent(type="text_end", content_index=index, content=block.text, partial=output))
                        elif isinstance(block, _ThinkingBlock):
                            result.push(AssistantMessageEvent(type="thinking_end", content_index=index, content=block.thinking, partial=output))
                        else:
                            block.arguments = cast(JsonObject, parse_streaming_json(None if block.partial_json is UNDEFINED else block.partial_json))
                            block.partial_json = UNDEFINED
                            result.push(AssistantMessageEvent(type="toolcall_end", content_index=index, tool_call=block, partial=output))
                        continue
                    delta = _required_property(event, "delta")
                    delta_type = _required_property(delta, "type")
                    if delta_type == "text_delta" and isinstance(block, _TextBlock):
                        value = cast(str, _property(delta, "text"))
                        block.text += javascript_string(value)
                        result.push(AssistantMessageEvent(type="text_delta", content_index=index, delta=value, partial=output))
                    elif delta_type == "thinking_delta" and isinstance(block, _ThinkingBlock):
                        value = cast(str, _property(delta, "thinking"))
                        block.thinking += javascript_string(value)
                        result.push(AssistantMessageEvent(type="thinking_delta", content_index=index, delta=value, partial=output))
                    elif delta_type == "input_json_delta" and isinstance(block, _ToolBlock):
                        value = cast(str, _property(delta, "partial_json"))
                        block.partial_json = javascript_string(block.partial_json) + javascript_string(value)
                        block.arguments = cast(JsonObject, parse_streaming_json(block.partial_json))
                        result.push(AssistantMessageEvent(type="toolcall_delta", content_index=index, delta=value, partial=output))
                    elif delta_type == "signature_delta" and isinstance(block, _ThinkingBlock):
                        block.thinking_signature = (block.thinking_signature or "") + javascript_string(_property(delta, "signature"))
                elif event_type == "message_delta":
                    incoming = _property(event, "input_transformations")
                    if isinstance(incoming, list):
                        transformations = incoming
                    delta = _required_property(event, "delta")
                    reason = _required_property(delta, "stop_reason")
                    if _truthy(reason):
                        output.raw_stop_reason = cast(str, reason)
                        output.stop_reason, error_message = _map_stop_reason(cast(str, reason), _property(delta, "stop_details"))
                        if error_message:
                            output.error_message = error_message
                    usage = _property(event, "usage")
                    if _truthy(usage):
                        for wire, attribute in (("input_tokens", "input"), ("output_tokens", "output"), ("cache_read_input_tokens", "cache_read"), ("cache_creation_input_tokens", "cache_write")):
                            value = _property(usage, wire)
                            if value is not None and value is not UNDEFINED:
                                setattr(output.usage, attribute, value)
                        value = _property(_property(usage, "output_tokens_details"), "thinking_tokens")
                        if value is not None and value is not UNDEFINED:
                            output.usage.reasoning = cast(int, value)
                    output.usage.total_tokens = output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
                    calculate_cost(usage_model, output.usage)
            if options.signal is not None and options.signal.aborted:
                raise RuntimeError("Request was aborted")
            if output.stop_reason == "pending":
                raise RuntimeError("Anthropic stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")
            if transformations:
                append_assistant_message_diagnostic(output, AssistantMessageDiagnostic(
                    type="anthropic_input_transformations", timestamp=time.time_ns() // 1_000_000,
                    details=cast(JsonObject, {"transformations": [
                        {key: _nullish(_property(item, key), UNDEFINED) for key in ("type", "path", "reason")}
                        for item in transformations
                    ]}),
                ))
            result.push(AssistantMessageEvent(type="done", reason=output.stop_reason, message=output))
            result.end()
        except (Exception, asyncio.CancelledError) as error:
            for block in output.content:
                if hasattr(block, "index"):
                    block.index = UNDEFINED
                if isinstance(block, _ToolBlock):
                    block.partial_json = UNDEFINED
            output.stop_reason = "aborted" if options.signal is not None and options.signal.aborted else "error"
            output.error_message = str(error)
            result.push(AssistantMessageEvent(type="error", reason=output.stop_reason, error=output))
            result.end()

    asyncio.get_running_loop().create_task(run())
    return result


def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
    options = options or SimpleStreamOptions()
    _assert_request_auth(model.provider, options.api_key, options.headers)
    base = build_base_options(model, context, options, options.api_key)
    prepared = AnthropicOptions(**{item.name: getattr(base, item.name) for item in fields(StreamOptions)}, tool_choice=options.tool_choice)
    if not options.reasoning:
        prepared.thinking_enabled = False
        return stream(model, context, prepared)
    prepared.thinking_enabled = True
    if (model.compat or {}).get("forceAdaptiveThinking") is True:
        mapped = (model.thinking_level_map or {}).get(options.reasoning)
        prepared.effort = cast(AnthropicEffort, mapped) if isinstance(mapped, str) else (
            "low" if options.reasoning in ("minimal", "low") else "medium" if options.reasoning == "medium" else "high"
        )
    else:
        adjusted = adjust_max_tokens_for_thinking(base.max_tokens, model.max_tokens, options.reasoning, options.thinking_budgets)
        prepared.max_tokens = clamp_max_tokens_to_context(model, context, adjusted.max_tokens)
        prepared.thinking_budget_tokens = min(adjusted.thinking_budget, max(0, prepared.max_tokens - 1024))
    return stream(model, context, prepared)


def _get_beta_features(
    model: Model, context: TranscriptContext, is_oauth: bool, native_tool_changes: bool, options: StreamOptions,
) -> list[str]:
    configured: str | None | object = UNDEFINED
    for headers in (model.headers, options.headers):
        for name, value in (headers or {}).items():
            if name.lower() == "anthropic-beta":
                configured = value
    if configured is None:
        return []
    if configured is not UNDEFINED:
        return list(dict.fromkeys(feature.strip(JS_WHITESPACE) for feature in cast(str, configured).split(",") if feature.strip(JS_WHITESPACE)))
    features: list[str] = []
    if is_oauth:
        features.extend(("claude-code-20250219", "oauth-2025-04-20"))
    if get_current_tools(context.messages) and not _get_compat(model).supports_eager_tool_input_streaming:
        features.append("fine-grained-tool-streaming-2025-05-14")
    if (model.reasoning and getattr(options, "thinking_enabled", None) is True
            and _nullish(getattr(options, "interleaved_thinking", None), True)
            and (model.compat or {}).get("forceAdaptiveThinking") is not True):
        features.append("interleaved-thinking-2025-05-14")
    if (model.compat or {}).get("allowedFallbackModels"):
        features.append("server-side-fallback-2026-07-01")
    if (model.compat or {}).get("supportsMidConvoEffort") is True:
        features.extend(("mid-conversation-output-config-2026-07-01", "thinking-binding-controls-2026-08-01"))
    if native_tool_changes:
        features.append("mid-conversation-tool-changes-2026-07-01")
    return list(dict.fromkeys(features))


def _normalize_tool_call_id(identifier: str, _model: Model, _message: AssistantMessage) -> str:
    # JS replaces each UTF-16 code unit, so a non-BMP character becomes two '_'.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", utf16_units(identifier))[:64]


@dataclass
class _ConvertedMessages:
    messages: list[dict[str, object]]
    assistant_levels: dict[int, AnthropicEffort]


def _convert_messages(
    transformed_messages: Sequence[Message], is_oauth: bool, cache_control: CacheControlEphemeral | None = None,
    allow_empty_signature: bool = False, managed_provider: str | None = None, native_tool_changes: bool = False,
) -> _ConvertedMessages:
    params: list[dict[str, object]] = []
    assistant_levels: dict[int, AnthropicEffort] = {}
    pending_system_messages: list[dict[str, object]] = []
    index = 0
    while index < len(transformed_messages):
        message = transformed_messages[index]
        index += 1
        if isinstance(message, SystemMessage):
            text = render_system_message_update(message)
            blocks: list[dict[str, object]] = []
            if text:
                blocks.append({"type": "text", "text": sanitize_surrogates(text)})
            if native_tool_changes:
                for tool in message.tools_removed or []:
                    blocks.append({"type": "tool_removal", "tool": {"type": "tool_reference", "name": _to_claude_code_name(tool.name) if is_oauth else tool.name}})
                for tool in message.tools_added or []:
                    blocks.append({"type": "tool_addition", "tool": {"type": "tool_reference", "name": _to_claude_code_name(tool.name) if is_oauth else tool.name}})
            if blocks:
                pending_system_messages.append({"role": "system", "content": blocks})
        elif isinstance(message, UserMessage):
            if isinstance(message.content, str):
                if message.content.strip(JS_WHITESPACE):
                    params.append({"role": "user", "content": sanitize_surrogates(message.content)})
            else:
                blocks = []
                for block in message.content:
                    if isinstance(block, TextContent):
                        text = sanitize_surrogates(block.text)
                        if text.strip(JS_WHITESPACE):
                            blocks.append({"type": "text", "text": text})
                    else:
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": block.mime_type, "data": block.data}})
                if blocks:
                    params.append({"role": "user", "content": blocks})
        elif isinstance(message, AssistantMessage):
            params.extend(pending_system_messages)
            pending_system_messages.clear()
            blocks = []
            for block in message.content:
                if isinstance(block, TextContent):
                    if block.text.strip(JS_WHITESPACE):
                        blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif isinstance(block, ThinkingContent):
                    if block.redacted:
                        blocks.append({"type": "redacted_thinking", "data": block.thinking_signature if block.thinking_signature is not None else UNDEFINED})
                        continue
                    signature = block.thinking_signature
                    has_signature = bool(signature and signature.strip(JS_WHITESPACE))
                    if not block.thinking.strip(JS_WHITESPACE) and not has_signature:
                        continue
                    if not has_signature:
                        blocks.append(
                            {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": ""}
                            if allow_empty_signature else {"type": "text", "text": sanitize_surrogates(block.thinking)}
                        )
                    else:
                        blocks.append({"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": signature})
                elif isinstance(block, ToolCall):
                    blocks.append({"type": "tool_use", "id": block.id,
                        "name": _to_claude_code_name(block.name) if is_oauth else block.name,
                        "input": block.arguments if block.arguments is not None else {}})
            if not blocks:
                continue
            message_index = len(params)
            params.append({"role": "assistant", "content": blocks})
            if (managed_provider is not None and message.api == "anthropic-messages" and message.provider == managed_provider
                    and message.provider_thinking_level in ("low", "medium", "high", "xhigh", "max")):
                assistant_levels[message_index] = cast(AnthropicEffort, message.provider_thinking_level)
        elif isinstance(message, ToolResultMessage):
            tool_results: list[dict[str, object]] = []
            first = index - 1
            while first < len(transformed_messages) and isinstance(transformed_messages[first], ToolResultMessage):
                tool_message = cast(ToolResultMessage, transformed_messages[first])
                tool_results.append({"type": "tool_result", "tool_use_id": tool_message.tool_call_id,
                    "content": _convert_content_blocks(tool_message.content), "is_error": tool_message.is_error})
                first += 1
            index = first
            params.append({"role": "user", "content": tool_results})
    params.extend(pending_system_messages)
    if cache_control is not None and params:
        last_message = params[-1]
        if last_message["role"] in ("user", "system"):
            content = last_message["content"]
            if isinstance(content, list):
                if content and content[-1]["type"] in ("text", "image", "tool_result", "tool_addition", "tool_removal"):
                    content[-1]["cache_control"] = cache_control
            elif isinstance(content, str):
                last_message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
    return _ConvertedMessages(params, assistant_levels)


def _convert_tools(
    tools: Sequence[Tool], is_oauth: bool, supports_eager_tool_input_streaming: bool, supports_strict_tools: bool,
    cache_control: CacheControlEphemeral | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, tool in enumerate(tools):
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)
        parameters = get_json_schema_tool_parameters(tool, strict)
        legacy: dict[str, object] = {"type": "object",
            "properties": _nullish(_property(parameters, "properties"), {}),
            "required": _nullish(_property(parameters, "required"), [])}
        input_schema = {**parameters, **legacy} if strict is True else legacy
        converted: dict[str, object] = {
            "name": _to_claude_code_name(tool.name) if is_oauth else tool.name,
            "description": tool.description, "input_schema": input_schema,
        }
        if supports_eager_tool_input_streaming:
            converted["eager_input_streaming"] = True
        if strict is True:
            converted["strict"] = True
        if cache_control is not None and index == len(tools) - 1:
            converted["cache_control"] = cache_control
        result.append(converted)
    return result


def _build_params(model: Model, context: TranscriptContext, is_oauth: bool, options: StreamOptions) -> dict[str, object]:
    cache_control = _get_cache_control(model, options.cache_retention, options.env)
    compat = _get_compat(model)
    initial = get_initial_system_message(context.messages)
    initial_text = get_system_message_text(initial) if initial is not None else ""
    transformed = transform_messages(context.messages, model, _normalize_tool_call_id)
    conversation = transformed[1:] if initial is not None else transformed
    initial_tools = (initial.tools_added or []) if initial is not None else []
    native_tool_changes = bool(compat.supports_mid_convo_system_messages and compat.supports_mid_convo_tool_changes
        and initial_tools and not has_tool_redefinitions(context.messages))
    managed_effort = (model.compat or {}).get("supportsMidConvoEffort") is True
    converted = _convert_messages(conversation, is_oauth, cache_control, compat.allow_empty_signature,
        model.provider if managed_effort else None, native_tool_changes)
    active_effort = _nullish(getattr(options, "effort", None), "high")
    features = _get_beta_features(model, context, is_oauth, native_tool_changes, options)
    messages = converted.messages
    if managed_effort:
        messages = []
        for index, message in enumerate(converted.messages):
            historical = converted.assistant_levels.get(index)
            if historical is not None:
                messages.append({"role": "system", "content": [], "output_config": {"effort": historical}})
            messages.append(message)
        messages.append({"role": "system", "content": [], "output_config": {"effort": active_effort}})
    params: dict[str, object] = {"model": model.id, "messages": messages,
        "max_tokens": model.max_tokens if options.max_tokens is None else options.max_tokens, "stream": True}
    if features:
        params["betas"] = features
    system: list[dict[str, object]] = []
    if is_oauth:
        identity: dict[str, object] = {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}
        if cache_control is not None:
            identity["cache_control"] = cache_control
        system.append(identity)
    if initial_text:
        prompt: dict[str, object] = {"type": "text", "text": sanitize_surrogates(initial_text)}
        if cache_control is not None:
            prompt["cache_control"] = cache_control
        system.append(prompt)
    if system:
        params["system"] = system
    thinking_enabled = getattr(options, "thinking_enabled", None)
    if options.temperature is not None and not thinking_enabled and not managed_effort and compat.supports_temperature:
        params["temperature"] = options.temperature
    tool_cache = cache_control if compat.supports_cache_control_on_tools else None
    if native_tool_changes:
        initial_names = {tool.name for tool in initial_tools}
        later_tools = [tool for tool in get_declared_tools(context.messages) if tool.name not in initial_names]
        params["tools"] = [
            *_convert_tools(initial_tools, is_oauth, compat.supports_eager_tool_input_streaming, compat.supports_strict_tools, tool_cache),
            _DEFERRED_TOOL_PLACEHOLDER,
            *({**tool, "defer_loading": True} for tool in _convert_tools(later_tools, is_oauth,
                compat.supports_eager_tool_input_streaming, compat.supports_strict_tools)),
        ]
    else:
        tools = get_current_tools(context.messages)
        if tools:
            params["tools"] = _convert_tools(tools, is_oauth, compat.supports_eager_tool_input_streaming, compat.supports_strict_tools, tool_cache)
    display = _nullish(getattr(options, "thinking_display", None), "summarized")
    if managed_effort:
        params["thinking"] = {"type": "adaptive", "display": display, "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
        params["output_config"] = {"effort": "high"}
    elif model.reasoning:
        if thinking_enabled:
            if (model.compat or {}).get("forceAdaptiveThinking") is True:
                params["thinking"] = {"type": "adaptive", "display": display}
                effort = getattr(options, "effort", None)
                if effort:
                    params["output_config"] = {"effort": effort}
            else:
                params["thinking"] = {"type": "enabled", "budget_tokens": getattr(options, "thinking_budget_tokens", None) or 1024, "display": display}
        elif thinking_enabled is False and (model.thinking_level_map or {}).get("off", UNDEFINED) is not None:
            params["thinking"] = {"type": "disabled"}
    if options.metadata is not None:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}
    tool_choice = getattr(options, "tool_choice", None)
    if _truthy(tool_choice):
        params["tool_choice"] = {"type": tool_choice} if isinstance(tool_choice, str) else tool_choice
    fallbacks = (model.compat or {}).get("allowedFallbackModels")
    if fallbacks:
        params["fallbacks"] = [{"model": fallback["model"]} for fallback in fallbacks]
    return params


def _map_stop_reason(reason: str, stop_details: object = None) -> tuple[StopReason, str | None]:
    if reason in ("end_turn", "pause_turn", "stop_sequence"):
        return "stop", None
    if reason == "max_tokens":
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason == "refusal":
        explanation = _property(stop_details, "explanation")
        return "error", cast(str, explanation) if _truthy(explanation) else "The model refused to complete the request"
    if reason == "sensitive":
        return "error", "Provider stopped with: sensitive"
    raise RuntimeError(f"Unhandled stop reason: {reason}")


__all__ = ["AnthropicEffort", "AnthropicThinkingDisplay", "AnthropicNamedToolChoice", "AnthropicToolChoice", "AnthropicOptions", "stream", "stream_simple"]
