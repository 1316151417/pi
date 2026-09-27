"""Full Chat Completions transport and replay from ``api/openai-completions.ts``."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Literal, NotRequired, TypedDict, cast

from .._javascript import javascript_json_parse, javascript_json_stringify, javascript_object_keys, javascript_string, utf16_length, utf16_slice
from .._json_runtime import JS_WHITESPACE, utf16_units
from .._values import JSON_NULL, UNDEFINED
from ..event_stream import AssistantMessageEventStream
from ..json_parse import parse_streaming_json
from ..models import calculate_cost, clamp_thinking_level
from ..text import get_system_message_text, render_system_message_update
from ..transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from ..types import (
    AssistantMessage, AssistantMessageEvent, CacheRetention, ChatTemplateKwargValue, FetchFunction,
    ImageContent, JsonObject, Model, OpenRouterRouting, ProviderEnv, ProviderHeaders,
    ProviderResponse, SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamOptions, SystemMessage,
    TextContent, ThinkingBudgets, ThinkingContent, ThinkingLevel, ThinkingTokenBudgetField, Tool, ToolCall,
    ToolResultMessage, TranscriptContext, Usage, UserMessage, VercelGatewayRouting,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.hash import short_hash
from ..utils.headers import headers_to_record
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from ._openai_http import OpenAIHttpClient, OpenAIResponse
from .constrained_sampling import (
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta, create_grammar_tool_input_properties,
    get_grammar_tool_input, get_json_schema_tool_parameters, resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .github_copilot_headers import CopilotDynamicHeaderParams, build_copilot_dynamic_headers, has_copilot_vision_input
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .simple_options import build_base_options, clamp_thinking_budget_to_answer_room, thinking_budget_for_level
from .transform_messages import transform_messages


class NamedTool(TypedDict):
    name: str


class ChatCompletionNamedToolChoice(TypedDict):
    type: Literal["function"]
    function: NamedTool


class ChatCompletionNamedToolChoiceCustom(TypedDict):
    type: Literal["custom"]
    custom: NamedTool


class ChatCompletionAllowedTools(TypedDict):
    mode: Literal["auto", "required"]
    tools: list[dict[str, object]]


class ChatCompletionAllowedToolChoice(TypedDict):
    type: Literal["allowed_tools"]
    allowed_tools: ChatCompletionAllowedTools


type ChatCompletionToolChoiceOption = (
    Literal["none", "auto", "required"] | ChatCompletionAllowedToolChoice
    | ChatCompletionNamedToolChoice | ChatCompletionNamedToolChoiceCustom
)


@dataclass
class OpenAICompletionsOptions(StreamOptions):
    tool_choice: ChatCompletionToolChoiceOption | None = None
    reasoning_effort: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = None


@dataclass
class ConvertCompletionsMessagesOptions:
    grammar_tool_input_properties: Mapping[str, str] | None = None


class OpenAICompatCacheControl(TypedDict):
    type: Literal["ephemeral"]
    ttl: NotRequired[str]


@dataclass
class ResolvedOpenAICompletionsCompat:
    supports_store: bool = True
    supports_developer_role: bool = True
    supports_reasoning_effort: bool = True
    supports_usage_in_streaming: bool = True
    supports_finish_reason: bool = True
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_completion_tokens"
    requires_tool_result_name: bool = False
    requires_assistant_after_tool_result: bool = False
    requires_thinking_as_text: bool = False
    requires_reasoning_content_on_assistant_messages: bool = False
    thinking_format: Literal[
        "openai", "openrouter", "deepseek", "together", "baseten", "zai", "qwen", "chat-template",
        "qwen-chat-template", "string-thinking", "ant-ling",
    ] = "openai"
    open_router_routing: OpenRouterRouting = field(default_factory=dict)
    vercel_gateway_routing: VercelGatewayRouting = field(default_factory=dict)
    chat_template_kwargs: dict[str, ChatTemplateKwargValue] = field(default_factory=dict)
    chat_template_args: dict[str, ChatTemplateKwargValue] = field(default_factory=dict)
    zai_tool_stream: bool = False
    supports_thinking_token_budget: bool | None = False
    thinking_token_budget_field: ThinkingTokenBudgetField | None = None
    supports_strict_mode: bool = True
    supports_openai_grammar_tools: bool = False
    supports_mid_convo_system_messages: bool | None = False
    supports_mid_convo_tool_additions: bool | None = False
    cache_control_format: Literal["anthropic"] | None = None
    send_session_affinity_headers: bool = False
    session_affinity_format: SessionAffinityFormat = "openai"
    supports_long_cache_retention: bool = True
    vllm_priority: int | float | None = None


_COMPAT_FIELDS = {
    "supportsStore": "supports_store", "supportsDeveloperRole": "supports_developer_role",
    "supportsReasoningEffort": "supports_reasoning_effort", "supportsUsageInStreaming": "supports_usage_in_streaming",
    "supportsFinishReason": "supports_finish_reason", "maxTokensField": "max_tokens_field",
    "requiresToolResultName": "requires_tool_result_name", "requiresAssistantAfterToolResult": "requires_assistant_after_tool_result",
    "requiresThinkingAsText": "requires_thinking_as_text",
    "requiresReasoningContentOnAssistantMessages": "requires_reasoning_content_on_assistant_messages",
    "thinkingFormat": "thinking_format", "openRouterRouting": "open_router_routing",
    "vercelGatewayRouting": "vercel_gateway_routing", "chatTemplateKwargs": "chat_template_kwargs",
    "chatTemplateArgs": "chat_template_args", "zaiToolStream": "zai_tool_stream",
    "supportsThinkingTokenBudget": "supports_thinking_token_budget", "thinkingTokenBudgetField": "thinking_token_budget_field",
    "supportsStrictMode": "supports_strict_mode", "supportsOpenAIGrammarTools": "supports_openai_grammar_tools",
    "supportsMidConvoSystemMessages": "supports_mid_convo_system_messages",
    "supportsMidConvoToolAdditions": "supports_mid_convo_tool_additions", "cacheControlFormat": "cache_control_format",
    "sendSessionAffinityHeaders": "send_session_affinity_headers", "sessionAffinityFormat": "session_affinity_format",
    "supportsLongCacheRetention": "supports_long_cache_retention", "vllmPriority": "vllm_priority",
}


def _truthy(value: object) -> bool:
    if value is None or value is UNDEFINED or value is JSON_NULL or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return value != "" if isinstance(value, str) else True


def _property(value: object, name: str, default: object = UNDEFINED) -> object:
    return value.get(name, default) if isinstance(value, Mapping) else default


def _nullish(value: object, fallback: object) -> object:
    return fallback if value is None or value is UNDEFINED or value is JSON_NULL else value


def _client_api_key(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> str:
    if api_key:
        return api_key
    if headers is not None:
        for key, value in headers.items():
            if key.lower() in ("authorization", "cf-aig-authorization") and value is not None and value.strip(JS_WHITESPACE):
                return "unused"
    raise ValueError(f"No API key for provider: {provider}")


def _is_reasoning_detail(detail: object) -> bool:
    if not isinstance(detail, dict):
        return False
    if detail.get("id") is not None and not isinstance(detail["id"], str):
        return False
    if "format" in detail and not isinstance(detail["format"], str):
        return False
    if "index" in detail and type(detail["index"]) not in (int, float):
        return False
    match detail.get("type"):
        case "reasoning.summary":
            return isinstance(detail.get("summary"), str)
        case "reasoning.encrypted":
            return isinstance(detail.get("data"), str)
        case "reasoning.text":
            return isinstance(detail.get("text"), str) and (
                detail.get("signature") is None or isinstance(detail["signature"], str)
            )
        case _:
            return False


def _parse_reasoning_details(signature: str | None) -> list[dict[str, object]] | None:
    if not signature:
        return None
    try:
        parsed = javascript_json_parse(signature)
        return cast(list[dict[str, object]], parsed) if isinstance(parsed, list) and parsed and all(_is_reasoning_detail(item) for item in parsed) else None
    except (ValueError, TypeError):
        return None


def _parse_legacy_reasoning_detail(signature: str | None) -> dict[str, object] | None:
    if not signature:
        return None
    try:
        parsed = javascript_json_parse(signature)
        if (
            _is_reasoning_detail(parsed) and _property(parsed, "type") == "reasoning.encrypted"
            and isinstance(_property(parsed, "id"), str) and _truthy(_property(parsed, "id"))
            and _truthy(_property(parsed, "data"))
        ):
            return cast(dict[str, object], parsed)
        return None
    except (ValueError, TypeError):
        return None


def _append_reasoning_detail(details: list[dict[str, object]], detail: dict[str, object]) -> None:
    last = details[-1] if details else None
    kind = detail["type"]
    if last is not None and kind in ("reasoning.text", "reasoning.summary") and last["type"] == kind:
        key = "text" if kind == "reasoning.text" else "summary"
        last[key] = cast(str, last[key]) + cast(str, detail[key])
        if kind == "reasoning.text" and not _truthy(last.get("signature")):
            last["signature"] = detail.get("signature", UNDEFINED)
        last["id"] = _nullish(last.get("id"), detail.get("id", UNDEFINED))
        if not _truthy(last.get("format")):
            last["format"] = detail.get("format", UNDEFINED)
        last["index"] = _nullish(last.get("index"), detail.get("index", UNDEFINED))
    else:
        details.append(dict(detail))


_REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_text")


def _resolve_cache_retention(cache_retention: CacheRetention | None, env: ProviderEnv | None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else "short"


def _detect_compat(model: Model) -> ResolvedOpenAICompletionsCompat:
    provider, url = model.provider, model.base_url
    zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in url or "open.bigmodel.cn" in url
    together = provider == "together" or "api.together.ai" in url or "api.together.xyz" in url
    moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in url
    openrouter = provider == "openrouter" or "openrouter.ai" in url
    workers = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in url
    gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in url
    nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in url
    ant_ling = provider == "ant-ling" or "api.ant-ling.com" in url
    deepseek = provider == "deepseek" or "deepseek.com" in url.lower()
    grok = provider == "xai" or "api.x.ai" in url
    nonstandard = (
        nvidia or provider == "cerebras" or "cerebras.ai" in url or grok or together or "chutes.ai" in url
        or deepseek or zai or moonshot or provider == "opencode" or "opencode.ai" in url or workers or gateway or ant_ling
    )
    max_tokens = "chutes.ai" in url or deepseek or moonshot or gateway or together or nvidia or ant_ling or zai
    return ResolvedOpenAICompletionsCompat(
        supports_store=not nonstandard,
        supports_developer_role=(openrouter and model.id.startswith(("anthropic/", "openai/"))) or (not nonstandard and not openrouter),
        supports_reasoning_effort=not (grok or zai or moonshot or together or gateway or nvidia or ant_ling),
        max_tokens_field="max_tokens" if max_tokens else "max_completion_tokens",
        requires_reasoning_content_on_assistant_messages=deepseek,
        thinking_format="deepseek" if deepseek else "zai" if zai else "together" if together else "ant-ling" if ant_ling else "openrouter" if openrouter else "openai",
        supports_strict_mode=not (moonshot or together or gateway or nvidia),
        cache_control_format="anthropic" if provider == "openrouter" and model.id.startswith("anthropic/") else None,
        send_session_affinity_headers=openrouter, session_affinity_format="openrouter" if openrouter else "openai",
        supports_long_cache_retention=not (together or workers or gateway or nvidia or ant_ling),
    )


def _get_compat(model: Model) -> ResolvedOpenAICompletionsCompat:
    detected = _detect_compat(model)
    if model.compat is not None:
        for wire_name, python_name in _COMPAT_FIELDS.items():
            value = model.compat.get(wire_name)
            if value is not None:
                setattr(detected, python_name, value)
    return detected


def _convert_tools(tools: Sequence[Tool], compat: ResolvedOpenAICompletionsCompat) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)
        if grammar is not None:
            result.append({"type": "custom", "custom": {
                "name": tool.name, "description": tool.description,
                "format": {"type": "grammar", "grammar": {"syntax": grammar.format, "definition": grammar.definition}},
            }})
            continue
        strict = resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode is not False)
        function: dict[str, object] = {
            "name": tool.name, "description": tool.description, "parameters": get_json_schema_tool_parameters(tool, strict),
        }
        if compat.supports_strict_mode is not False:
            function["strict"] = strict if strict is not None else False
        result.append({"type": "function", "function": function})
    return result


def convert_messages(
    model: Model, context: TranscriptContext, compat: ResolvedOpenAICompletionsCompat,
    options: ConvertCompletionsMessagesOptions | None = None,
) -> list[dict[str, object]]:
    normalized = resolve_transcript(context, compat.supports_mid_convo_system_messages)

    def normalize_id(id: str, _model: Model, _message: AssistantMessage) -> str:
        if "|" in id:
            call, item = id.split("|", 1)
            call = re.sub(r"[^a-zA-Z0-9_-]", "_", utf16_units(call))
            item = re.sub(r"[^a-zA-Z0-9_-]", "_", utf16_units(item))
            combined = f"{call}_{item}" if item else call
            if len(combined) <= 40:
                return combined
            hash = short_hash(id)[:8]
            return f"{call[:max(1, 40 - len(hash) - 1)]}_{hash}"
        return utf16_slice(id, 40) if model.provider == "openai" and utf16_length(id) > 40 else id

    messages = transform_messages(normalized.messages, model, normalize_id)
    transcript_tools = resolve_transcript_tools(
        normalized.messages, compat.supports_mid_convo_system_messages is True and compat.supports_mid_convo_tool_additions is True,
    )
    role = "developer" if model.reasoning and compat.supports_developer_role else "system"
    params: list[dict[str, object]] = []
    last_role: str | None = None
    index = 0
    while index < len(messages):
        message = messages[index]
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and message.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})
        if message.role == "system":
            system = cast(SystemMessage, message)
            added_tools = system.tools_added or [] if index > 0 and transcript_tools.anchors_additions else []
            if added_tools:
                params.append({"role": "system", "tools": _convert_tools(added_tools, compat)})
            text = get_system_message_text(system) if index == 0 else render_system_message_update(system)
            if text:
                params.append({"role": role, "content": sanitize_surrogates(text)})
        elif message.role == "user":
            user = cast(UserMessage, message)
            if isinstance(user.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(user.content)})
            else:
                content: list[dict[str, object]] = []
                for part in user.content:
                    if isinstance(part, TextContent):
                        content.append({"type": "text", "text": sanitize_surrogates(part.text)})
                    else:
                        content.append({"type": "image_url", "image_url": {"url": f"data:{part.mime_type};base64,{part.data}"}})
                if not content:
                    index += 1
                    continue
                params.append({"role": "user", "content": content})
        elif message.role == "assistant":
            assistant = cast(AssistantMessage, message)
            entry: dict[str, object] = {"role": "assistant", "content": "" if compat.requires_assistant_after_tool_result else None}
            text_parts: list[dict[str, object]] = [
                {"type": "text", "text": sanitize_surrogates(block.text)}
                for block in assistant.content if isinstance(block, TextContent) and block.text.strip(JS_WHITESPACE)
            ]
            text = "".join(cast(str, part["text"]) for part in text_parts)
            thinking = [block for block in assistant.content if isinstance(block, ThinkingContent)]
            calls = [block for block in assistant.content if isinstance(block, ToolCall)]
            signed = next((detail for block in thinking if (detail := _parse_reasoning_details(block.thinking_signature)) is not None), None)
            legacy = [detail for call in calls if (detail := _parse_legacy_reasoning_detail(call.thought_signature)) is not None]
            preserved = signed if signed is not None else legacy if legacy else None
            nonempty = [block for block in thinking if block.thinking.strip(JS_WHITESPACE)]
            if nonempty:
                if compat.requires_thinking_as_text:
                    entry["content"] = [{"type": "text", "text": "\n\n".join(sanitize_surrogates(block.thinking) for block in nonempty)}, *text_parts]
                else:
                    if text:
                        entry["content"] = text
                    if preserved is None:
                        signature = nonempty[0].thinking_signature
                        if model.provider == "opencode-go" and signature == "reasoning":
                            signature = "reasoning_content"
                        if signature and signature in _REASONING_FIELDS:
                            entry[signature] = "\n".join(block.thinking for block in nonempty)
            elif text:
                entry["content"] = text
            if calls:
                tool_calls: list[dict[str, object]] = []
                for call in calls:
                    property = options.grammar_tool_input_properties.get(call.name) if options is not None and options.grammar_tool_input_properties is not None else None
                    if property is not None:
                        tool_calls.append({"id": call.id, "type": "custom", "custom": {
                            "name": call.name, "input": sanitize_surrogates(get_grammar_tool_input(call.name, call.arguments, property)),
                        }})
                    else:
                        tool_calls.append({"id": call.id, "type": "function", "function": {
                            "name": call.name, "arguments": javascript_json_stringify(call.arguments),
                        }})
                entry["tool_calls"] = tool_calls
            if preserved is not None:
                entry["reasoning_details"] = preserved
            if compat.requires_reasoning_content_on_assistant_messages and model.reasoning and "reasoning_content" not in entry:
                entry["reasoning_content"] = ""
            if not entry["content"] and "tool_calls" not in entry:
                index += 1
                continue
            params.append(entry)
        elif message.role == "toolResult":
            images: list[dict[str, object]] = []
            while index < len(messages) and messages[index].role == "toolResult":
                tool_message = cast(ToolResultMessage, messages[index])
                text = "\n".join(block.text for block in tool_message.content if isinstance(block, TextContent))
                has_images = any(isinstance(block, ImageContent) for block in tool_message.content)
                text = text if text else "(see attached image)" if has_images else "(no tool output)"
                result: dict[str, object] = {
                    "role": "tool", "content": sanitize_surrogates(text), "tool_call_id": tool_message.tool_call_id,
                }
                if compat.requires_tool_result_name and tool_message.tool_name:
                    result["name"] = tool_message.tool_name
                params.append(result)
                if has_images and "image" in model.input:
                    for block in tool_message.content:
                        if isinstance(block, ImageContent):
                            images.append({"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}})
                index += 1
            if images:
                if compat.requires_assistant_after_tool_result:
                    params.append({"role": "assistant", "content": "I have processed the tool results."})
                params.append({"role": "user", "content": [{"type": "text", "text": "Attached image(s) from tool result:"}, *images]})
                last_role = "user"
            else:
                last_role = "toolResult"
            continue
        last_role = message.role
        index += 1
    return params


def _add_cache_control(message: dict[str, object], control: OpenAICompatCacheControl) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": control}]
        return True
    if not isinstance(content, list):
        return False
    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = control
            return True
    return False


def _template_values(
    model: Model, options: OpenAICompletionsOptions | None,
    values: Mapping[str, ChatTemplateKwargValue], budget: int | float | None,
) -> dict[str, object] | None:
    resolved: dict[str, object] = {}
    effort = getattr(options, "reasoning_effort", None)
    for key in javascript_object_keys(values):
        value = values[key]
        if not isinstance(value, dict):
            result: object = value
        elif not effort and value.get("omitWhenOff"):
            result = UNDEFINED
        elif value.get("$var") == "thinking.enabled":
            result = bool(effort)
        elif value.get("$var") == "thinking.budget":
            result = budget if budget is not None else UNDEFINED
        else:
            mapped = model.thinking_level_map.get(effort or "off", UNDEFINED) if model.thinking_level_map is not None else UNDEFINED
            result = (effort if effort is not None else UNDEFINED) if mapped is UNDEFINED else mapped if isinstance(mapped, str) else UNDEFINED
        if result is not UNDEFINED:
            resolved[key] = result
    return resolved if resolved else None


def _build_params(
    model: Model, context: TranscriptContext, options: OpenAICompletionsOptions | None = None,
    compat: ResolvedOpenAICompletionsCompat | None = None, cache_retention: CacheRetention | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, object]:
    compat = compat if compat is not None else _get_compat(model)
    if cache_retention is None:
        cache_retention = _resolve_cache_retention(options.cache_retention if options else None, options.env if options else None)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(get_declared_tools(context.messages), compat.supports_openai_grammar_tools)
    transcript_tools = resolve_transcript_tools(
        context.messages, compat.supports_mid_convo_system_messages is True and compat.supports_mid_convo_tool_additions is True,
    )
    messages = convert_messages(model, context, compat, ConvertCompletionsMessagesOptions(grammar_tool_input_properties))
    cache_control: OpenAICompatCacheControl | None = None
    if compat.cache_control_format == "anthropic" and cache_retention != "none":
        cache_control = {"type": "ephemeral"}
        if cache_retention == "long" and compat.supports_long_cache_retention:
            cache_control["ttl"] = "1h"
    cache_key = (
        clamp_openai_prompt_cache_key(options.session_id if options else None)
        if ("api.openai.com" in model.base_url and cache_retention != "none")
        or (cache_retention == "long" and compat.supports_long_cache_retention) else None
    )
    params: dict[str, object] = {
        "model": model.id, "messages": messages, "stream": True,
        "prompt_cache_key": cache_key if cache_key is not None else UNDEFINED,
        "prompt_cache_retention": "24h" if cache_retention == "long" and compat.supports_long_cache_retention else UNDEFINED,
    }
    if compat.supports_usage_in_streaming is not False:
        params["stream_options"] = {"include_usage": True}
    if compat.supports_store:
        params["store"] = False
    if options is not None and options.max_tokens:
        params[compat.max_tokens_field] = options.max_tokens
    if options is not None and options.temperature is not None:
        params["temperature"] = options.temperature
    tools: list[dict[str, object]] | None = None
    if transcript_tools.request_tools:
        tools = _convert_tools(transcript_tools.request_tools, compat)
        params["tools"] = tools
        if compat.zai_tool_stream:
            params["tool_stream"] = True
    elif any(
        message.role == "toolResult" or isinstance(message, AssistantMessage)
        and any(isinstance(block, ToolCall) for block in message.content)
        for message in context.messages
    ):
        tools = []
        params["tools"] = tools
    if cache_control is not None:
        for message in messages:
            if message["role"] in ("system", "developer"):
                _add_cache_control(message, cache_control)
                break
        if tools:
            tools[-1]["cache_control"] = cache_control
        for message in reversed(messages):
            if message["role"] in ("user", "assistant", "tool") and _add_cache_control(message, cache_control):
                break
    if _truthy(getattr(options, "tool_choice", None)):
        params["tool_choice"] = getattr(options, "tool_choice")
    if compat.vllm_priority is not None:
        params["priority"] = compat.vllm_priority

    budget_field = compat.thinking_token_budget_field
    if not budget_field and compat.supports_thinking_token_budget:
        budget_field = "thinking_token_budget"
    effort = cast(ThinkingLevel | None, getattr(options, "reasoning_effort", None))
    budget: int | float | None = None
    if effort and model.reasoning:
        ceiling = params.get("max_tokens", params.get("max_completion_tokens", model.max_tokens))
        computed = clamp_thinking_budget_to_answer_room(
            thinking_budget_for_level(effort, getattr(options, "thinking_budgets", None)), cast(int, ceiling),
        )
        if computed > 0:
            budget = computed
    mapping = model.thinking_level_map if model.thinking_level_map is not None else {}
    mapped = mapping.get(effort, UNDEFINED) if effort else UNDEFINED
    off = mapping.get("off", UNDEFINED)
    format = compat.thinking_format
    if format == "zai" and model.reasoning:
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            chosen = effort if mapped is UNDEFINED else mapped
            if isinstance(chosen, str):
                params["reasoning_effort"] = chosen
    elif format == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(effort)
        if effort and compat.supports_reasoning_effort:
            chosen = _nullish(mapped, effort)
            if isinstance(chosen, str):
                params["reasoning_effort"] = chosen
    elif format == "qwen-chat-template" and model.reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(effort), "preserve_thinking": True}
    elif format == "chat-template" and model.reasoning:
        values = _template_values(model, options, compat.chat_template_kwargs, budget)
        if values is not None:
            params["chat_template_kwargs"] = values
    elif format == "baseten" and model.reasoning:
        values = _template_values(model, options, compat.chat_template_args, budget)
        if values is not None:
            params["chat_template_args"] = values
        if compat.supports_reasoning_effort:
            mapped_effort = mapped if effort else off
            chosen = effort if mapped_effort is UNDEFINED else mapped_effort
            if isinstance(chosen, str):
                params["reasoning_effort"] = chosen
    elif format == "deepseek" and model.reasoning:
        if effort:
            params["thinking"] = {"type": "enabled"}
        elif off is not None:
            params["thinking"] = {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _nullish(mapped, effort)
    elif format == "openrouter" and model.reasoning:
        if effort:
            params["reasoning"] = {"effort": _nullish(mapped, effort)}
        elif off is not None:
            params["reasoning"] = {"effort": _nullish(off, "none")}
    elif format == "ant-ling" and model.reasoning and effort:
        if isinstance(mapped, str):
            params["reasoning"] = {"effort": mapped}
    elif format == "together" and model.reasoning:
        params["reasoning"] = {"enabled": bool(effort)}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _nullish(mapped, effort)
    elif format == "string-thinking" and model.reasoning:
        if effort:
            params["thinking"] = _nullish(mapped, effort)
        elif off is not None:
            params["thinking"] = _nullish(off, "none")
    elif effort and model.reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = _nullish(mapped, effort)
    elif not effort and model.reasoning and compat.supports_reasoning_effort and isinstance(off, str):
        params["reasoning_effort"] = off
    if budget_field and budget is not None:
        params[budget_field] = budget
    if model.compat is not None:
        routing = model.compat.get("openRouterRouting")
        if _truthy(routing):
            params["provider"] = routing
        routing = model.compat.get("vercelGatewayRouting")
        if _truthy(routing) and (_truthy(_property(routing, "only")) or _truthy(_property(routing, "order"))):
            gateway: dict[str, object] = {}
            if _truthy(_property(routing, "only")):
                gateway["only"] = _property(routing, "only")
            if _truthy(_property(routing, "order")):
                gateway["order"] = _property(routing, "order")
            params["providerOptions"] = {"gateway": gateway}
    if options is not None and options.sampling_params is not None:
        params.update(options.sampling_params)
    return params


def _create_client(
    model: Model, context: TranscriptContext, api_key: str,
    options_headers: ProviderHeaders | None, fetch: FetchFunction | None, session_id: str | None,
    compat: ResolvedOpenAICompletionsCompat,
) -> OpenAIHttpClient:
    headers: ProviderHeaders = {"User-Agent": get_pi_user_agent(), **(model.headers if model.headers is not None else {})}
    if model.provider == "github-copilot":
        headers.update(build_copilot_dynamic_headers(CopilotDynamicHeaderParams(
            messages=context.messages, has_images=has_copilot_vision_input(context.messages),
        )))
    if session_id and compat.send_session_affinity_headers:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id
    if options_headers is not None:
        headers.update(options_headers)
    return OpenAIHttpClient(api_key=api_key, base_url=model.base_url, default_headers=headers, custom_fetch=fetch)


def _parse_chunk_usage(raw: object, model: Model) -> Usage:
    prompt = _property(raw, "prompt_tokens")
    prompt = cast(int, prompt if _truthy(prompt) else 0)
    details = _property(raw, "prompt_tokens_details")
    cached = cast(int, _nullish(_property(details, "cached_tokens"), _nullish(
        _property(raw, "prompt_cache_hit_tokens"), _nullish(_property(raw, "cached_tokens"), 0),
    )))
    write = _property(details, "cache_write_tokens")
    write = cast(int, write if _truthy(write) else 0)
    input = prompt - cached - write
    input = input if isinstance(input, float) and math.isnan(input) else max(0, input)
    output = _property(raw, "completion_tokens")
    output = cast(int, output if _truthy(output) else 0)
    reasoning = _property(_property(raw, "completion_tokens_details"), "reasoning_tokens")
    usage = Usage(
        input=input, output=output, cache_read=cached, cache_write=write,
        reasoning=cast(int, reasoning if _truthy(reasoning) else 0), total_tokens=input + output + cached + write,
    )
    calculate_cost(model, usage)
    return usage


def _map_stop_reason(reason: object) -> tuple[StopReason, str | None]:
    if reason is None or reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    return "error", f"Provider finish_reason: {javascript_string(reason)}"


@dataclass
class _CustomToolInput:
    property: str
    json_buffer: GrammarToolInputJsonBuffer = field(default_factory=GrammarToolInputJsonBuffer)


@dataclass
class _StreamingToolCall(ToolCall):
    partial_args: str | None = None
    custom_input: _CustomToolInput | None = None
    stream_index: int | float | None = None

    def to_json(self) -> JsonObject:
        data = super().to_json()
        if self.partial_args is not None:
            data["partialArgs"] = self.partial_args
        if self.custom_input is not None:
            custom = self.custom_input
            data["customInput"] = {
                "property": custom.property, "jsonBuffer": {
                    "input": custom.json_buffer.input, "started": custom.json_buffer.started,
                    "closed": custom.json_buffer.closed,
                },
            }
        if self.stream_index is not None:
            data["streamIndex"] = self.stream_index
        return data


def stream(
    model: Model, context: TranscriptContext, options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    output_stream = AssistantMessageEventStream()
    normalized = resolve_transcript(context, _get_compat(model).supports_mid_convo_system_messages)

    async def run() -> None:
        output = AssistantMessage(
            content=[], api=model.api, provider=model.provider, model=model.id,
            usage=Usage(), stop_reason="pending", timestamp=time.time_ns() // 1_000_000,
        )
        streamed_details: list[dict[str, object]] | None = None
        source_iterator: AsyncIterator[object] | None = None

        def apply_details(block: ThinkingContent) -> None:
            if streamed_details is not None:
                block.thinking_signature = javascript_json_stringify(streamed_details)

        try:
            api_key = _client_api_key(model.provider, options.api_key if options else None, options.headers if options else None)
            compat = _get_compat(model)
            grammar_properties = create_grammar_tool_input_properties(get_declared_tools(normalized.messages), compat.supports_openai_grammar_tools)
            retention = _resolve_cache_retention(options.cache_retention if options else None, options.env if options else None)
            session_id = None if retention == "none" else options.session_id if options else None
            client = _create_client(
                model, normalized, api_key, options.headers if options else None,
                options.fetch if options else None, session_id, compat,
            )
            params: object = _build_params(
                model, normalized, cast(OpenAICompletionsOptions | None, options), compat, retention, grammar_properties,
            )
            if options is not None and options.on_payload is not None:
                next_params = options.on_payload(params, model)
                if inspect.isawaitable(next_params):
                    next_params = await next_params
                # Python's implicit callback None means no replacement. JSON_NULL
                # is the explicit null return that JS distinguishes from undefined.
                if next_params is not None and next_params is not UNDEFINED:
                    params = next_params

            async def request() -> OpenAIResponse:
                if params is None or params is JSON_NULL:
                    raise TypeError("Cannot read properties of null (reading 'stream')")
                return await client.post(
                    "/chat/completions", params, stream=_truthy(_property(params, "stream")),
                    signal=options.signal if options else None, timeout_ms=options.timeout_ms if options else None,
                )

            response = await retry_provider_request(
                request, max_retries=options.max_retries if options and options.max_retries is not None else 0,
                max_retry_delay_ms=options.max_retry_delay_ms if options else None, signal=options.signal if options else None,
            )
            if options is not None and options.on_response is not None:
                notified = options.on_response(ProviderResponse(
                    status=response.response.status, headers=headers_to_record(response.response.headers),
                ), model)
                if inspect.isawaitable(notified):
                    await notified
            output_stream.push(AssistantMessageEvent(type="start", partial=output))

            text_block: TextContent | None = None
            thinking_block: ThinkingContent | None = None
            has_finish_reason = False
            by_index: dict[int | float, _StreamingToolCall] = {}
            by_id: dict[str, _StreamingToolCall] = {}

            def content_index(block: TextContent | ThinkingContent | ToolCall) -> int:
                return next((index for index, candidate in enumerate(output.content) if candidate is block), -1)

            def custom_text(block: _StreamingToolCall) -> str:
                if block.custom_input is None:
                    return ""
                value = _property(block.arguments, block.custom_input.property)
                return value if isinstance(value, str) else ""

            def append_custom(block: _StreamingToolCall, next_input: str, close: bool) -> str | None:
                custom = block.custom_input
                if custom is None:
                    return None
                delta = append_grammar_tool_input_json_delta(custom.json_buffer, custom.property, next_input, close)
                block.arguments = {custom.property: next_input}
                return delta

            def finish_block(block: TextContent | ThinkingContent | ToolCall) -> None:
                index = content_index(block)
                if index == -1:
                    return
                if isinstance(block, TextContent):
                    output_stream.push(AssistantMessageEvent(type="text_end", content_index=index, content=block.text, partial=output))
                elif isinstance(block, ThinkingContent):
                    apply_details(block)
                    output_stream.push(AssistantMessageEvent(type="thinking_end", content_index=index, content=block.thinking, partial=output))
                elif isinstance(block, ToolCall):
                    tool = cast(_StreamingToolCall, block)
                    if tool.custom_input is not None:
                        delta = append_custom(tool, custom_text(tool), True)
                        if delta is not None:
                            output_stream.push(AssistantMessageEvent(type="toolcall_delta", content_index=index, delta=delta, partial=output))
                    else:
                        tool.arguments = cast(JsonObject, parse_streaming_json(tool.partial_args))
                    for name in ("partial_args", "custom_input", "stream_index"):
                        if name in vars(tool):
                            delattr(tool, name)
                    output_stream.push(AssistantMessageEvent(type="toolcall_end", content_index=index, tool_call=tool, partial=output))

            def ensure_text() -> TextContent:
                nonlocal text_block
                if text_block is None:
                    text_block = TextContent(text="")
                    output.content.append(text_block)
                    output_stream.push(AssistantMessageEvent(type="text_start", content_index=content_index(text_block), partial=output))
                return text_block

            def ensure_thinking(signature: str) -> ThinkingContent:
                nonlocal thinking_block
                if thinking_block is None:
                    thinking_block = ThinkingContent(thinking="", thinking_signature=signature)
                    output.content.append(thinking_block)
                    output_stream.push(AssistantMessageEvent(type="thinking_start", content_index=content_index(thinking_block), partial=output))
                return thinking_block

            def ensure_tool(tool_call: object) -> _StreamingToolCall:
                index = _property(tool_call, "index")
                index = cast(int | float, index) if type(index) in (int, float) else None
                function = _property(tool_call, "function")
                custom = _property(tool_call, "custom")
                name = cast(str, _nullish(_property(function, "name"), _nullish(_property(custom, "name"), "")))
                id = _property(tool_call, "id")
                block = by_index.get(index) if index is not None else None
                if block is None and _truthy(id):
                    block = by_id.get(cast(str, id))
                if block is None:
                    property = grammar_properties.get(name, "input") if _truthy(custom) and not _truthy(function) else None
                    block = _StreamingToolCall(
                        id=cast(str, id) if _truthy(id) else "", name=name,
                        arguments={property: ""} if property is not None else {},
                        partial_args=None if property is not None else "",
                        custom_input=_CustomToolInput(property) if property is not None else None,
                        stream_index=index,
                    )
                    if index is not None:
                        by_index[index] = block
                    if _truthy(id):
                        by_id[cast(str, id)] = block
                    output.content.append(block)
                    output_stream.push(AssistantMessageEvent(type="toolcall_start", content_index=content_index(block), partial=output))
                if index is not None and block.stream_index is None:
                    block.stream_index = index
                    by_index[index] = block
                if _truthy(id):
                    by_id[cast(str, id)] = block
                if not block.name and name:
                    block.name = name
                if _truthy(custom) and not _truthy(function) and block.custom_input is None:
                    property = grammar_properties.get(block.name, "input")
                    block.arguments = {property: ""}
                    block.custom_input = _CustomToolInput(property)
                    if "partial_args" in vars(block):
                        delattr(block, "partial_args")
                return block

            source_iterator = aiter(cast(AsyncIterable[object], response.data))
            async for chunk in source_iterator:
                if not isinstance(chunk, (Mapping, list)):
                    continue
                if not output.response_id:
                    output.response_id = cast(str | None, _property(chunk, "id", None))
                chunk_model = _property(chunk, "model")
                if isinstance(chunk_model, str) and chunk_model and chunk_model != model.id and not output.response_model:
                    output.response_model = chunk_model
                chunk_usage = _property(chunk, "usage")
                if _truthy(chunk_usage):
                    output.usage = _parse_chunk_usage(chunk_usage, model)
                choices = _property(chunk, "choices")
                choice = choices[0] if isinstance(choices, list) and choices else None
                if not _truthy(choice):
                    continue
                if not _truthy(chunk_usage) and _truthy(_property(choice, "usage")):
                    output.usage = _parse_chunk_usage(_property(choice, "usage"), model)
                reason = _property(choice, "finish_reason")
                if _truthy(reason):
                    output.raw_stop_reason = cast(str, reason)
                    output.stop_reason, error_message = _map_stop_reason(reason)
                    if error_message:
                        output.error_message = error_message
                    has_finish_reason = True
                delta = _property(choice, "delta")
                if not _truthy(delta):
                    continue
                text = _property(delta, "content")
                length = utf16_length(text) if isinstance(text, str) else len(text) if isinstance(text, list) else _property(text, "length")
                if isinstance(length, (int, float)) and length > 0:
                    block = ensure_text()
                    block.text += javascript_string(text)
                    output_stream.push(AssistantMessageEvent(type="text_delta", content_index=content_index(block), delta=cast(str, text), partial=output))
                found = next((name for name in ("reasoning_content", "reasoning", "reasoning_text") if isinstance(_property(delta, name), str) and _property(delta, name)), None)
                if found is not None:
                    reasoning_delta = cast(str, _property(delta, found))
                    signature = "reasoning_content" if model.provider == "opencode-go" and found == "reasoning" else found
                    thinking = ensure_thinking(signature)
                    thinking.thinking += reasoning_delta
                    output_stream.push(AssistantMessageEvent(type="thinking_delta", content_index=content_index(thinking), delta=reasoning_delta, partial=output))
                tool_calls = _property(delta, "tool_calls")
                if _truthy(tool_calls):
                    for tool_call in cast(Sequence[object], tool_calls):
                        tool = ensure_tool(tool_call)
                        id = _property(tool_call, "id")
                        if not tool.id and _truthy(id):
                            tool.id = cast(str, id)
                            by_id[tool.id] = tool
                        function = _property(tool_call, "function")
                        custom = _property(tool_call, "custom")
                        name = _nullish(_property(function, "name"), _property(custom, "name"))
                        if not tool.name and _truthy(name):
                            tool.name = cast(str, name)
                        tool_delta = ""
                        arguments = _property(function, "arguments")
                        custom_input = _property(custom, "input")
                        if _truthy(arguments):
                            tool_delta = cast(str, arguments)
                            tool.partial_args = (tool.partial_args if tool.partial_args is not None else "") + javascript_string(arguments)
                            tool.arguments = cast(JsonObject, parse_streaming_json(tool.partial_args))
                        elif _truthy(custom_input):
                            tool_delta = append_custom(tool, custom_text(tool) + javascript_string(custom_input), False) or ""
                        output_stream.push(AssistantMessageEvent(type="toolcall_delta", content_index=content_index(tool), delta=tool_delta, partial=output))
                reasoning_details = _property(delta, "reasoning_details")
                if isinstance(reasoning_details, list):
                    for detail in reasoning_details:
                        if not _is_reasoning_detail(detail):
                            continue
                        ensure_thinking("")
                        if streamed_details is None:
                            streamed_details = []
                        _append_reasoning_detail(streamed_details, cast(dict[str, object], detail))

            source_iterator = None
            for block in output.content:
                finish_block(block)
            if options is not None and options.signal is not None and options.signal.aborted:
                raise RuntimeError("Request was aborted")
            if output.stop_reason == "aborted":
                raise RuntimeError("Request was aborted")
            if not has_finish_reason and not compat.supports_finish_reason:
                output.stop_reason = "toolUse" if any(isinstance(block, ToolCall) for block in output.content) else "stop"
            if output.stop_reason == "error":
                raise RuntimeError(output.error_message or "Provider returned an error stop reason")
            if (compat.supports_finish_reason and not has_finish_reason) or output.stop_reason == "pending":
                raise RuntimeError("Stream ended without finish_reason")
            output_stream.push(AssistantMessageEvent(type="done", reason=output.stop_reason, message=output))
            output_stream.end()
        except (Exception, asyncio.CancelledError) as error:
            # Python async-for does not call return()/aclose() when processing
            # a yielded chunk throws. Match JS AsyncIteratorClose explicitly.
            if source_iterator is not None:
                close = getattr(source_iterator, "aclose", None)
                if callable(close):
                    try:
                        closed = close()
                        if inspect.isawaitable(closed):
                            await closed
                    except (Exception, asyncio.CancelledError):
                        pass
            for block in output.content:
                if isinstance(block, ThinkingContent):
                    apply_details(block)
                for name in ("index", "partial_args", "custom_input", "stream_index"):
                    if name in vars(block):
                        delattr(block, name)
            output.stop_reason = "aborted" if options is not None and options.signal is not None and options.signal.aborted else "error"
            output.error_message = format_provider_error(normalize_provider_error(error))
            raw = _property(_property(getattr(error, "error", None), "metadata"), "raw")
            if _truthy(raw) and javascript_string(raw) not in output.error_message:
                output.error_message += "\n" + javascript_string(raw)
            output_stream.push(AssistantMessageEvent(type="error", reason=output.stop_reason, error=output))
            output_stream.end()

    asyncio.get_running_loop().create_task(run())
    return output_stream


def stream_simple(
    model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    _client_api_key(model.provider, options.api_key if options else None, options.headers if options else None)
    base = build_base_options(model, context, options, options.api_key if options else None)
    prepared = OpenAICompletionsOptions()
    for attribute in fields(base):
        setattr(prepared, attribute.name, getattr(base, attribute.name))
    prepared.tool_choice = options.tool_choice if options is not None else None
    clamped = clamp_thinking_level(model, options.reasoning) if options is not None and options.reasoning else None
    prepared.reasoning_effort = None if clamped == "off" else clamped
    prepared.thinking_budgets = options.thinking_budgets if options is not None else None
    return stream(model, context, prepared)


__all__ = [
    "OpenAICompletionsOptions", "ConvertCompletionsMessagesOptions", "ResolvedOpenAICompletionsCompat",
    "ChatCompletionToolChoiceOption", "ChatCompletionNamedToolChoice", "ChatCompletionNamedToolChoiceCustom",
    "ChatCompletionAllowedTools", "ChatCompletionAllowedToolChoice", "convert_messages", "stream", "stream_simple",
]
