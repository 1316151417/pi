"""OpenAI Responses adapter from ``api/openai-responses.ts``."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass, fields
from typing import Literal, TypedDict, cast

from .._json_runtime import JS_WHITESPACE
from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from ..types import (
    JSON_NULL, UNDEFINED, AssistantMessage, AssistantMessageEvent, CacheRetention, Model,
    ProviderEnv, ProviderHeaders, ProviderResponse, SessionAffinityFormat, SimpleStreamOptions,
    StreamOptions, ThinkingLevel, TranscriptContext, Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.headers import headers_to_record
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import retry_provider_request
from ._openai_http import OpenAIHttpClient, OpenAIResponse
from ._openai_responses_sdk import create_response
from .constrained_sampling import create_grammar_tool_input_properties
from .github_copilot_headers import CopilotDynamicHeaderParams, build_copilot_dynamic_headers, has_copilot_vision_input
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, OpenAIResponsesStreamOptions,
    ServiceTier, convert_responses_messages, convert_responses_tools, process_responses_stream,
)
from .simple_options import build_base_options

_OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}
_OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16
type ResponsesToolChoice = Literal["none", "auto", "required"] | dict[str, object]


@dataclass
class OpenAIResponsesOptions(StreamOptions):
    reasoning_effort: ThinkingLevel | None = None
    reasoning_summary: Literal["auto", "detailed", "concise"] | None = None
    service_tier: ServiceTier | None = None
    tool_choice: ResponsesToolChoice | None = None


class _ResolvedCompat(TypedDict):
    supportsDeveloperRole: bool
    supportsMidConvoSystemMessages: bool
    sessionAffinityFormat: SessionAffinityFormat
    supportsLongCacheRetention: bool
    supportsStrictMode: bool
    supportsOpenAIGrammarTools: bool
    supportsAdditionalTools: bool
    supportsToolSearch: bool
    supportsExplicitPromptCacheMode: bool
    supportsMaxOutputTokens: bool


def _get_client_api_key(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> str:
    if api_key:
        return api_key
    if headers is not None and any(
        key.lower() in ("authorization", "cf-aig-authorization")
        and value is not None and value.strip(JS_WHITESPACE)
        for key, value in headers.items()
    ):
        return "unused"
    raise RuntimeError(f"No API key for provider: {provider}")


def _resolve_cache_retention(cache_retention: CacheRetention | None, env: ProviderEnv | None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else "short"


def _get_compat(model: Model) -> _ResolvedCompat:
    defaults: _ResolvedCompat = {
        "supportsDeveloperRole": True,
        "supportsMidConvoSystemMessages": False,
        "sessionAffinityFormat": "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.base_url else "openai",
        "supportsLongCacheRetention": True,
        "supportsStrictMode": False,
        "supportsOpenAIGrammarTools": False,
        "supportsAdditionalTools": False,
        "supportsToolSearch": False,
        "supportsExplicitPromptCacheMode": False,
        "supportsMaxOutputTokens": True,
    }
    compat = model.compat or {}
    return cast(_ResolvedCompat, {
        key: compat[key] if key in compat and compat[key] is not None else value
        for key, value in defaults.items()
    })


def _create_client(
    model: Model, context: TranscriptContext, api_key: str,
    options: StreamOptions | None, session_id: str | None,
) -> OpenAIHttpClient:
    compat = _get_compat(model)
    headers: ProviderHeaders = {"User-Agent": get_pi_user_agent(), **(model.headers or {})}
    if model.provider == "github-copilot":
        headers.update(build_copilot_dynamic_headers(CopilotDynamicHeaderParams(
            messages=context.messages, has_images=has_copilot_vision_input(context.messages),
        )))
    if session_id:
        if compat["sessionAffinityFormat"] == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat["sessionAffinityFormat"] == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
    if options is not None and options.headers is not None:
        headers.update(options.headers)
    return OpenAIHttpClient(
        api_key=api_key, base_url=model.base_url, default_headers=headers,
        custom_fetch=options.fetch if options is not None else None,
    )


def _build_params(
    model: Model, context: TranscriptContext, options: StreamOptions | None,
    compat: _ResolvedCompat | None = None, grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, object]:
    compat = compat if compat is not None else _get_compat(model)
    grammar_properties = grammar_tool_input_properties if grammar_tool_input_properties is not None else create_grammar_tool_input_properties(
        get_declared_tools(context.messages), compat["supportsOpenAIGrammarTools"],
    )
    transcript_tools = resolve_transcript_tools(
        context.messages, compat["supportsAdditionalTools"] or compat["supportsToolSearch"],
    )
    tool_options = ConvertResponsesToolsOptions(
        supports_strict_mode=compat["supportsStrictMode"], supports_openai_grammar_tools=compat["supportsOpenAIGrammarTools"],
    )
    messages = convert_responses_messages(model, context, _OPENAI_TOOL_CALL_PROVIDERS, ConvertResponsesMessagesOptions(
        grammar_tool_input_properties=grammar_properties,
        supports_mid_convo_system_messages=compat["supportsMidConvoSystemMessages"],
        supports_additional_tools=compat["supportsAdditionalTools"], supports_tool_search=compat["supportsToolSearch"],
        tool_options=tool_options,
    ))
    cache_retention = _resolve_cache_retention(
        options.cache_retention if options is not None else None, options.env if options is not None else None,
    )
    cache_key = clamp_openai_prompt_cache_key(options.session_id if options is not None else None)
    cache_options: object = UNDEFINED
    if compat["supportsExplicitPromptCacheMode"]:
        if cache_retention == "none":
            cache_options = {"mode": "explicit"}
        elif cache_retention == "long" and compat["supportsLongCacheRetention"]:
            cache_options = {"ttl": "30m"}
    params: dict[str, object] = {
        "model": model.id, "input": messages, "stream": True,
        "prompt_cache_key": UNDEFINED if cache_retention == "none" or cache_key is None else cache_key,
        "prompt_cache_retention": "24h" if cache_retention == "long" and compat["supportsLongCacheRetention"] and not compat["supportsExplicitPromptCacheMode"] else UNDEFINED,
        "prompt_cache_options": cache_options, "store": False,
    }
    if options is not None:
        if options.max_tokens and compat["supportsMaxOutputTokens"]:
            params["max_output_tokens"] = max(options.max_tokens, _OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)
        if options.temperature is not None:
            params["temperature"] = options.temperature
        if getattr(options, "service_tier", None) is not None:
            params["service_tier"] = getattr(options, "service_tier")
    if transcript_tools.request_tools:
        params["tools"] = convert_responses_tools(transcript_tools.request_tools, tool_options)
    if getattr(options, "tool_choice", None) is not None:
        params["tool_choice"] = getattr(options, "tool_choice")
    if model.reasoning:
        effort: ThinkingLevel | None = getattr(options, "reasoning_effort", None)
        summary: str | None = getattr(options, "reasoning_summary", None)
        mapping = model.thinking_level_map if model.thinking_level_map is not None else {}
        if effort or summary:
            mapped_effort = mapping.get(effort) if effort else None
            params["reasoning"] = {"effort": mapped_effort if mapped_effort is not None else effort if effort else "medium", "summary": summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot" and mapping.get("off", UNDEFINED) is not None:
            params["reasoning"] = {"effort": mapping.get("off", "none")}
        if model.provider == "xai":
            params["include"] = ["reasoning.encrypted_content"]
    if options is not None and options.sampling_params is not None:
        params.update(options.sampling_params)
    return params


def _apply_service_tier_pricing(usage: Usage, service_tier: ServiceTier | None, model: Model) -> None:
    multiplier = 0.5 if service_tier == "flex" else (2.5 if model.id == "gpt-5.5" else 2) if service_tier == "priority" else 1
    if multiplier == 1:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write


def stream(
    model: Model, context: TranscriptContext, options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    outer = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, _get_compat(model)["supportsMidConvoSystemMessages"])

    async def run() -> None:
        output = AssistantMessage(
            content=[], api=model.api, provider=model.provider, model=model.id,
            usage=Usage(), stop_reason="pending", timestamp=time.time_ns() // 1_000_000,
        )
        try:
            api_key = _get_client_api_key(model.provider, options.api_key if options is not None else None, options.headers if options is not None else None)
            retention = _resolve_cache_retention(options.cache_retention if options is not None else None, options.env if options is not None else None)
            session_id = options.session_id if options is not None and retention != "none" else None
            compat = _get_compat(model)
            grammar_properties = create_grammar_tool_input_properties(get_declared_tools(normalized_context.messages), compat["supportsOpenAIGrammarTools"])
            client = _create_client(model, normalized_context, api_key, options, session_id)
            params: object = _build_params(model, normalized_context, options, compat, grammar_properties)
            replacement = options.on_payload(params, model) if options is not None and options.on_payload is not None else UNDEFINED
            if inspect.isawaitable(replacement):
                replacement = await replacement
            else:
                await asyncio.sleep(0)
            if replacement is not None and replacement is not UNDEFINED:
                params = None if replacement is JSON_NULL else replacement

            async def request() -> OpenAIResponse:
                return await create_response(client, params, signal=options.signal if options is not None else None, timeout_ms=options.timeout_ms if options is not None else None)

            result = await retry_provider_request(
                request, max_retries=options.max_retries if options is not None and options.max_retries is not None else 0,
                max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
                signal=options.signal if options is not None else None,
            )
            observed = options.on_response(ProviderResponse(status=result.response.status, headers=headers_to_record(result.response.headers)), model) if options is not None and options.on_response is not None else None
            if inspect.isawaitable(observed):
                await observed
            else:
                await asyncio.sleep(0)
            outer.push(AssistantMessageEvent(type="start", partial=output))
            await process_responses_stream(cast(AsyncIterable[Mapping[str, object]], result.data), output, outer, model, OpenAIResponsesStreamOptions(
                service_tier=getattr(options, "service_tier", None),
                grammar_tool_input_properties=grammar_properties,
                apply_service_tier_pricing=lambda usage, tier: _apply_service_tier_pricing(usage, tier, model),
            ))
            if options is not None and options.signal is not None and options.signal.aborted:
                raise RuntimeError("Request was aborted")
            if output.stop_reason == "pending":
                raise RuntimeError("OpenAI Responses stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")
            outer.push(AssistantMessageEvent(type="done", reason=output.stop_reason, message=output))
            outer.end()
        except (Exception, asyncio.CancelledError) as error:
            for block in output.content:
                for attribute in ("index", "partial_json", "custom_input"):
                    if attribute in vars(block):
                        delattr(block, attribute)
            output.stop_reason = "aborted" if options is not None and options.signal is not None and options.signal.aborted else "error"
            prefix = "OpenAI" if model.provider == "openai" else model.provider
            output.error_message = format_provider_error(normalize_provider_error(error), f"{prefix} API error")
            outer.push(AssistantMessageEvent(type="error", reason=output.stop_reason, error=output))
            outer.end()

    asyncio.get_running_loop().create_task(run())
    return outer


def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
    _get_client_api_key(model.provider, options.api_key if options is not None else None, options.headers if options is not None else None)
    base = build_base_options(model, context, options, options.api_key if options is not None else None)
    request_options = OpenAIResponsesOptions(**{field.name: getattr(base, field.name) for field in fields(base)})
    request_options.tool_choice = options.tool_choice if options is not None else None
    reasoning = clamp_thinking_level(model, options.reasoning) if options is not None and options.reasoning else None
    request_options.reasoning_effort = None if reasoning == "off" else reasoning
    return stream(model, context, request_options)


__all__ = ["OpenAIResponsesOptions", "ResponsesToolChoice", "stream", "stream_simple"]
