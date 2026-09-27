"""Azure OpenAI Responses adapter from ``api/azure-openai-responses.ts``."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass, fields
from typing import Literal, cast

from .._json_runtime import JS_WHITESPACE
from ..auth.oauth._http import parse_url
from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from ..types import (
    JSON_NULL, UNDEFINED, AssistantMessage, AssistantMessageEvent, Model, ProviderHeaders,
    ProviderResponse, SimpleStreamOptions, StreamOptions, ThinkingLevel, TranscriptContext, Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.headers import headers_to_record
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import retry_provider_request
from ._openai_http import OpenAIHttpClient, OpenAIResponse
from ._openai_responses_sdk import create_response
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses import ResponsesToolChoice
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, OpenAIResponsesStreamOptions,
    convert_responses_messages, convert_responses_tools, process_responses_stream,
)
from .simple_options import build_base_options

_DEFAULT_AZURE_API_VERSION = "v1"
_AZURE_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode", "azure-openai-responses"}
_OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16


@dataclass
class AzureOpenAIResponsesOptions(StreamOptions):
    reasoning_effort: ThinkingLevel | None = None
    tool_choice: ResponsesToolChoice | None = None
    reasoning_summary: Literal["auto", "detailed", "concise"] | None = None
    azure_api_version: str | None = None
    azure_resource_name: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None


def _resolve_deployment_name(model: Model, options: StreamOptions | None) -> str:
    explicit: str | None = getattr(options, "azure_deployment_name", None)
    if explicit:
        return explicit
    value = get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.env if options is not None else None)
    names: dict[str, str] = {}
    for entry in value.split(",") if value else []:
        trimmed = entry.strip(JS_WHITESPACE)
        if not trimmed:
            continue
        parts = trimmed.split("=")[:2]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        names[parts[0].strip(JS_WHITESPACE)] = parts[1].strip(JS_WHITESPACE)
    return names.get(model.id) or model.id


def _normalize_azure_base_url(base_url: str) -> str:
    trimmed = base_url.strip(JS_WHITESPACE).rstrip("/")
    try:
        url = parse_url(trimmed)
    except Exception as error:
        raise RuntimeError(f"Invalid Azure OpenAI base URL: {base_url}") from error
    azure_host = url.hostname.endswith((".openai.azure.com", ".cognitiveservices.azure.com", ".ai.azure.com"))
    if azure_host and url.pathname.rstrip("/") in ("", "/", "/openai", "/openai/v1/responses"):
        url.pathname = "/openai/v1"
        url.search = ""
    return url.href.rstrip("/")


def _create_client(model: Model, api_key: str, options: StreamOptions | None) -> OpenAIHttpClient:
    env = options.env if options is not None else None
    version = getattr(options, "azure_api_version", None) or get_provider_env_value("AZURE_OPENAI_API_VERSION", env) or _DEFAULT_AZURE_API_VERSION
    explicit_url: str | None = getattr(options, "azure_base_url", None)
    env_url = get_provider_env_value("AZURE_OPENAI_BASE_URL", env)
    base_url = (explicit_url.strip(JS_WHITESPACE) if explicit_url is not None else None) or (env_url.strip(JS_WHITESPACE) if env_url is not None else None)
    resource_name: str | None = getattr(options, "azure_resource_name", None) or get_provider_env_value("AZURE_OPENAI_RESOURCE_NAME", env)
    if not base_url and resource_name:
        base_url = f"https://{resource_name}.openai.azure.com/openai/v1"
    if not base_url and model.base_url:
        base_url = model.base_url
    if not base_url:
        raise RuntimeError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.",
        )
    headers: ProviderHeaders = {"User-Agent": get_pi_user_agent(), **(model.headers or {})}
    if options is not None and options.headers is not None:
        headers.update(options.headers)
    # SDK 6.40.0 excludes /responses from its deployment-path rewrite set.
    # The selected deployment therefore stays in the body model field.
    return OpenAIHttpClient(
        api_key=api_key, base_url=_normalize_azure_base_url(base_url), default_headers=headers,
        custom_fetch=options.fetch if options is not None else None,
        default_query={"api-version": version}, auth_header="api-key", user_agent_name="AzureOpenAI",
    )


def _build_params(
    model: Model, context: TranscriptContext, options: StreamOptions | None, deployment_name: str,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, object]:
    compat = model.compat or {}
    supports_grammar = cast(bool, compat.get("supportsOpenAIGrammarTools") or False)
    supports_additions = cast(bool, compat.get("supportsAdditionalTools") or False)
    supports_search = cast(bool, compat.get("supportsToolSearch") or False)
    strict = cast(bool, compat["supportsStrictMode"] if compat.get("supportsStrictMode") is not None else True)
    grammar_properties = grammar_tool_input_properties if grammar_tool_input_properties is not None else create_grammar_tool_input_properties(
        get_declared_tools(context.messages), supports_grammar,
    )
    transcript_tools = resolve_transcript_tools(context.messages, supports_additions or supports_search)
    tool_options = ConvertResponsesToolsOptions(supports_strict_mode=strict, supports_openai_grammar_tools=supports_grammar)
    messages = convert_responses_messages(model, context, _AZURE_TOOL_CALL_PROVIDERS, ConvertResponsesMessagesOptions(
        grammar_tool_input_properties=grammar_properties,
        supports_mid_convo_system_messages=cast(bool, compat.get("supportsMidConvoSystemMessages") or False),
        supports_additional_tools=supports_additions, supports_tool_search=supports_search, tool_options=tool_options,
    ))
    cache_key = clamp_openai_prompt_cache_key(options.session_id if options is not None else None)
    params: dict[str, object] = {
        "model": deployment_name, "input": messages, "stream": True,
        "prompt_cache_key": cache_key if cache_key is not None else UNDEFINED, "store": False,
    }
    if options is not None:
        if options.max_tokens:
            params["max_output_tokens"] = max(options.max_tokens, _OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)
        if options.temperature is not None:
            params["temperature"] = options.temperature
    if transcript_tools.request_tools:
        params["tools"] = convert_responses_tools(transcript_tools.request_tools, tool_options)
    if getattr(options, "tool_choice", None) is not None:
        params["tool_choice"] = getattr(options, "tool_choice")
    if model.reasoning:
        effort: ThinkingLevel | None = getattr(options, "reasoning_effort", None)
        summary: str | None = getattr(options, "reasoning_summary", None)
        mapping = model.thinking_level_map if model.thinking_level_map is not None else {}
        if effort or summary:
            mapped = mapping.get(effort) if effort else None
            params["reasoning"] = {"effort": mapped if mapped is not None else effort if effort else "medium", "summary": summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif mapping.get("off", UNDEFINED) is not None:
            params["reasoning"] = {"effort": mapping.get("off", "none")}
    if options is not None and options.sampling_params is not None:
        params.update(options.sampling_params)
    return params


def stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    outer = AssistantMessageEventStream()
    normalized = resolve_transcript(context, cast(bool | None, (model.compat or {}).get("supportsMidConvoSystemMessages")))

    async def run() -> None:
        deployment = _resolve_deployment_name(model, options)
        output = AssistantMessage(
            content=[], api="azure-openai-responses", provider=model.provider, model=model.id,
            usage=Usage(), stop_reason="pending", timestamp=time.time_ns() // 1_000_000,
        )
        try:
            api_key = options.api_key if options is not None else None
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")
            client = _create_client(model, api_key, options)
            grammar_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized.messages), cast(bool, (model.compat or {}).get("supportsOpenAIGrammarTools") or False),
            )
            params: object = _build_params(model, normalized, options, deployment, grammar_properties)
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
            await process_responses_stream(cast(AsyncIterable[Mapping[str, object]], result.data), output, outer, model, OpenAIResponsesStreamOptions(grammar_tool_input_properties=grammar_properties))
            if options is not None and options.signal is not None and options.signal.aborted:
                raise RuntimeError("Request was aborted")
            if output.stop_reason == "pending":
                raise RuntimeError("Azure OpenAI Responses stream ended without a stop reason")
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
            output.error_message = format_provider_error(normalize_provider_error(error), "Azure OpenAI API error")
            outer.push(AssistantMessageEvent(type="error", reason=output.stop_reason, error=output))
            outer.end()

    asyncio.get_running_loop().create_task(run())
    return outer


def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
    api_key = options.api_key if options is not None else None
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")
    base = build_base_options(model, context, options, api_key)
    request_options = AzureOpenAIResponsesOptions(**{field.name: getattr(base, field.name) for field in fields(base)})
    request_options.tool_choice = options.tool_choice if options is not None else None
    reasoning = clamp_thinking_level(model, options.reasoning) if options is not None and options.reasoning else None
    request_options.reasoning_effort = None if reasoning == "off" else reasoning
    return stream(model, context, request_options)


__all__ = ["AzureOpenAIResponsesOptions", "stream", "stream_simple"]
