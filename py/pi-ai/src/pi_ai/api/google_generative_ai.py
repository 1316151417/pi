"""Google Generative AI adapter from ``api/google-generative-ai.ts``."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal, cast

from .._javascript import javascript_json_stringify, javascript_string
from .._values import JSON_NULL, UNDEFINED
from ..auth.oauth._http import fetch
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, clamp_thinking_level
from ..text import get_system_message_text
from ..transcript import collapse_system_messages, get_current_tools, get_initial_system_message
from ..types import (
    AssistantMessage, AssistantMessageEvent, JsonObject, Model, ProviderHeaders,
    SimpleStreamOptions, StreamOptions, TextContent, ThinkingBudgets, ThinkingContent,
    ToolCall, TranscriptContext, Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.headers import provider_headers_to_record
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.sanitize_unicode import sanitize_surrogates
from ._google_genai_http import GoogleGenAI
from ._google_genai_convert import truthy
from .google_shared import (
    GoogleApiThinkingLevel, ResolvedGoogleThinkingLevel, convert_messages, convert_tools,
    get_disabled_google_thinking_config, is_thinking_part, map_stop_reason,
    resolve_google_function_calling_mode, resolve_google_thinking_level,
    retain_thought_signature, retry_google_request, supports_google_strict_tool_sampling,
    to_google_sdk_thinking_level, to_google_thinking_level, uses_google_thinking_level,
)
from .simple_options import build_base_options


@dataclass
class GoogleThinkingOptions:
    enabled: bool
    budget_tokens: int | None = None
    level: GoogleApiThinkingLevel | None = None


@dataclass
class GoogleOptions(StreamOptions):
    tool_choice: Literal["auto", "none", "any"] | None = None
    thinking: GoogleThinkingOptions | None = None


_tool_call_counter = 0


def _create_client(model: Model, api_key: str, options_headers: ProviderHeaders | None) -> GoogleGenAI:
    headers = provider_headers_to_record({
        "User-Agent": get_pi_user_agent(), **(model.headers or {}), **(options_headers or {}),
    })
    return GoogleGenAI(
        api_key=api_key, base_url=model.base_url or None,
        api_version="" if model.base_url else None, headers=headers,
    )


def _build_params(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> dict[str, object]:
    contents = convert_messages(model, context)
    initial_system = get_initial_system_message(context.messages)
    tools = get_current_tools(context.messages)
    config: dict[str, object] = {}
    if options is not None:
        if options.temperature is not None:
            config["temperature"] = options.temperature
        if options.max_tokens is not None:
            config["maxOutputTokens"] = options.max_tokens
    strict = supports_google_strict_tool_sampling(model.id)
    calling_mode = resolve_google_function_calling_mode(tools, getattr(options, "tool_choice", None), strict) if tools else None
    system_instruction = get_system_message_text(initial_system) if initial_system is not None else ""
    if system_instruction:
        config["systemInstruction"] = sanitize_surrogates(system_instruction)
    if tools:
        config["tools"] = convert_tools(tools, False, strict)
    if calling_mode is not None:
        config["toolConfig"] = {"functionCallingConfig": {"mode": calling_mode}}

    thinking: GoogleThinkingOptions | None = getattr(options, "thinking", None)
    if thinking is not None and thinking.enabled and model.reasoning:
        thinking_config: dict[str, object] = {"includeThoughts": True}
        if thinking.level is not None:
            thinking_config["thinkingLevel"] = to_google_sdk_thinking_level(thinking.level)
        elif thinking.budget_tokens is not None:
            thinking_config["thinkingBudget"] = thinking.budget_tokens
        config["thinkingConfig"] = thinking_config
    elif model.reasoning and thinking is not None and not thinking.enabled:
        config["thinkingConfig"] = get_disabled_google_thinking_config(model)
    if options is not None and options.signal is not None:
        if options.signal.aborted:
            raise RuntimeError("Request aborted")
        config["abortSignal"] = options.signal
    return {"model": model.id, "contents": contents, "config": config}


def stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    outer = AssistantMessageEventStream()
    normalized_context = collapse_system_messages(context)

    async def run() -> None:
        global _tool_call_counter
        output = AssistantMessage(
            content=[], api="google-generative-ai", provider=model.provider, model=model.id,
            usage=Usage(), stop_reason="pending", timestamp=time.time_ns() // 1_000_000,
        )
        try:
            if options is not None and options.fetch is not None and options.fetch is not fetch:
                raise RuntimeError("Custom fetch is not supported by the Google Generative AI adapter")
            api_key = options.api_key if options is not None else None
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")
            client = _create_client(model, api_key, options.headers if options is not None else None)
            params: object = _build_params(model, normalized_context, options)
            replacement = options.on_payload(params, model) if options is not None and options.on_payload is not None else UNDEFINED
            if inspect.isawaitable(replacement):
                replacement = await replacement
            else:
                await asyncio.sleep(0)
            if replacement is not None and replacement is not UNDEFINED:
                params = None if replacement is JSON_NULL else replacement
            google_stream = await retry_google_request(lambda: client.generate_content_stream(params), options)
            outer.push(AssistantMessageEvent(type="start", partial=output))
            current: TextContent | ThinkingContent | None = None
            blocks = output.content

            def end_block() -> None:
                if isinstance(current, TextContent):
                    outer.push(AssistantMessageEvent(type="text_end", content_index=len(blocks) - 1, content=current.text, partial=output))
                elif isinstance(current, ThinkingContent):
                    outer.push(AssistantMessageEvent(type="thinking_end", content_index=len(blocks) - 1, content=current.thinking, partial=output))

            async for chunk in google_stream:
                if not output.response_id:
                    output.response_id = cast(str | None, chunk.get("responseId"))
                candidates = cast(list[Mapping[str, object]] | None, chunk.get("candidates"))
                candidate = candidates[0] if candidates else None
                content = cast(Mapping[str, object] | None, candidate.get("content")) if candidate is not None else None
                parts = cast(list[Mapping[str, object]] | None, content.get("parts")) if content is not None else None
                if truthy(parts):
                    for part in cast(list[Mapping[str, object]], parts):
                        if part.get("text", UNDEFINED) is not UNDEFINED:
                            is_thinking = is_thinking_part(part)
                            if current is None or is_thinking != isinstance(current, ThinkingContent):
                                end_block()
                                if is_thinking:
                                    current = ThinkingContent(thinking="")
                                    blocks.append(current)
                                    outer.push(AssistantMessageEvent(type="thinking_start", content_index=len(blocks) - 1, partial=output))
                                else:
                                    current = TextContent(text="")
                                    blocks.append(current)
                                    outer.push(AssistantMessageEvent(type="text_start", content_index=len(blocks) - 1, partial=output))
                            text = cast(str, part["text"])
                            signature = cast(str | None, part.get("thoughtSignature"))
                            if isinstance(current, ThinkingContent):
                                current.thinking += javascript_string(text)
                                current.thinking_signature = retain_thought_signature(current.thinking_signature, signature)
                                outer.push(AssistantMessageEvent(type="thinking_delta", content_index=len(blocks) - 1, delta=text, partial=output))
                            else:
                                current.text += javascript_string(text)
                                current.text_signature = retain_thought_signature(current.text_signature, signature)
                                outer.push(AssistantMessageEvent(type="text_delta", content_index=len(blocks) - 1, delta=text, partial=output))

                        function_call = cast(Mapping[str, object] | None, part.get("functionCall"))
                        if truthy(function_call):
                            function_call = cast(Mapping[str, object], function_call)
                            if current is not None:
                                end_block()
                                current = None
                            provided_id = cast(str | None, function_call.get("id"))
                            needs_new_id = not provided_id or any(isinstance(block, ToolCall) and block.id == provided_id for block in blocks)
                            if needs_new_id:
                                _tool_call_counter += 1
                                tool_call_id = f"{javascript_string(function_call.get('name', UNDEFINED))}_{time.time_ns() // 1_000_000}_{_tool_call_counter}"
                            else:
                                tool_call_id = cast(str, provided_id)
                            arguments = function_call.get("args")
                            tool_call = ToolCall(
                                id=tool_call_id, name=cast(str, function_call.get("name") or ""),
                                arguments=cast(JsonObject, arguments if arguments is not None else {}),
                                thought_signature=cast(str | None, part.get("thoughtSignature") or None),
                            )
                            blocks.append(tool_call)
                            outer.push(AssistantMessageEvent(type="toolcall_start", content_index=len(blocks) - 1, partial=output))
                            outer.push(AssistantMessageEvent(type="toolcall_delta", content_index=len(blocks) - 1, delta=javascript_json_stringify(tool_call.arguments), partial=output))
                            outer.push(AssistantMessageEvent(type="toolcall_end", content_index=len(blocks) - 1, tool_call=tool_call, partial=output))

                finish_reason = cast(str | None, candidate.get("finishReason")) if candidate is not None else None
                if finish_reason:
                    output.raw_stop_reason = finish_reason
                    output.stop_reason = map_stop_reason(finish_reason)
                    if output.stop_reason == "stop" and any(isinstance(block, ToolCall) for block in blocks):
                        output.stop_reason = "toolUse"
                usage = cast(Mapping[str, object] | None, chunk.get("usageMetadata"))
                if truthy(usage):
                    usage = cast(Mapping[str, object], usage)
                    cached = cast(int, usage.get("cachedContentTokenCount") or 0)
                    thoughts = cast(int, usage.get("thoughtsTokenCount") or 0)
                    output.usage = Usage(
                        input=cast(int, usage.get("promptTokenCount") or 0) - cached,
                        output=cast(int, usage.get("candidatesTokenCount") or 0) + thoughts,
                        cache_read=cached, cache_write=0, reasoning=thoughts,
                        total_tokens=cast(int, usage.get("totalTokenCount") or 0),
                    )
                    calculate_cost(model, output.usage)

            end_block()
            if options is not None and options.signal is not None and options.signal.aborted:
                raise RuntimeError("Request was aborted")
            if output.stop_reason == "pending":
                raise RuntimeError("Google stream ended without a finish reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(f"Provider stopped with: {output.raw_stop_reason}" if output.raw_stop_reason else "An unknown error occurred")
            outer.push(AssistantMessageEvent(type="done", reason=output.stop_reason, message=output))
            outer.end()
        except (Exception, asyncio.CancelledError) as error:
            for block in output.content:
                if "index" in vars(block):
                    delattr(block, "index")
            output.stop_reason = "aborted" if options is not None and options.signal is not None and options.signal.aborted else "error"
            output.error_message = format_provider_error(normalize_provider_error(error))
            outer.push(AssistantMessageEvent(type="error", reason=output.stop_reason, error=output))
            outer.end()

    asyncio.get_running_loop().create_task(run())
    return outer


def _get_google_budget(model: Model, level: ResolvedGoogleThinkingLevel, custom_budgets: ThinkingBudgets | None) -> int:
    custom = getattr(custom_budgets, level, None)
    if custom is not None:
        return cast(int, custom)
    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[level]
    if "2.5-flash-lite" in model.id:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[level]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[level]
    return -1


def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
    api_key = options.api_key if options is not None else None
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")
    base = build_base_options(model, context, options, api_key)
    request_options = GoogleOptions(**{field.name: getattr(base, field.name) for field in fields(base)})
    request_options.tool_choice = options.tool_choice if options is not None else None
    if options is None or not options.reasoning:
        request_options.thinking = GoogleThinkingOptions(enabled=False)
        return stream(model, context, request_options)
    clamped = clamp_thinking_level(model, options.reasoning)
    if clamped == "off":
        request_options.thinking = GoogleThinkingOptions(enabled=False)
        return stream(model, context, request_options)
    resolved = resolve_google_thinking_level(model, clamped)
    if uses_google_thinking_level(model):
        request_options.thinking = GoogleThinkingOptions(enabled=True, level=to_google_thinking_level(resolved))
    else:
        request_options.thinking = GoogleThinkingOptions(enabled=True, budget_tokens=_get_google_budget(model, resolved, options.thinking_budgets))
    return stream(model, context, request_options)


__all__ = ["GoogleOptions", "GoogleThinkingOptions", "stream", "stream_simple"]
