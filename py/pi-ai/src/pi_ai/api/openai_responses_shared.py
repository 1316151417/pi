"""Responses message/tool conversion and stream processing shared by adapters."""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, Callable, Mapping, Sequence, Set
from copy import copy
from dataclasses import dataclass
from typing import Literal, cast

from ..event_stream import AssistantMessageEventStream
from ..json_parse import parse_streaming_json
from ..models import calculate_cost
from ..text import get_system_message_text, render_system_message_update
from ..transcript import resolve_transcript, resolve_transcript_tools
from ..types import (
    AssistantMessage, AssistantMessageEvent, ImageContent, JsonObject, Model, StopReason, SystemMessage,
    TextContent, ThinkingContent, Tool, ToolCall, ToolResultMessage, TranscriptContext, UNDEFINED, UserMessage, Usage,
)
from ..utils._javascript import javascript_json_parse, javascript_json_stringify, javascript_string, utf16_length
from ..utils.hash import short_hash
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import (
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta, get_grammar_tool_input,
    get_json_schema_tool_parameters, resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
)
from .transform_messages import transform_messages

type ResponseItem = dict[str, object]
type ServiceTier = Literal["auto", "default", "flex", "scale", "priority"]


@dataclass
class OpenAIResponsesStreamOptions:
    service_tier: ServiceTier | None = None
    grammar_tool_input_properties: Mapping[str, str] | None = None
    resolve_service_tier: Callable[[ServiceTier | None, ServiceTier | None], ServiceTier | None] | None = None
    apply_service_tier_pricing: Callable[[Usage, ServiceTier | None], None] | None = None


@dataclass
class ConvertResponsesToolsOptions:
    strict: bool | None = False
    supports_strict_mode: bool | None = None
    supports_openai_grammar_tools: bool | None = None
    tool_search_result: bool | None = None


@dataclass
class ConvertResponsesMessagesOptions:
    include_system_prompt: bool | None = None
    grammar_tool_input_properties: Mapping[str, str] | None = None
    supports_mid_convo_system_messages: bool | None = None
    supports_additional_tools: bool | None = None
    supports_tool_search: bool | None = None
    tool_options: ConvertResponsesToolsOptions | None = None


def _parse_text_signature(signature: str | None) -> dict[str, str] | None:
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = cast(dict[str, object], javascript_json_parse(signature))
            if type(parsed.get("v")) in (int, float) and parsed["v"] == 1 and isinstance(parsed.get("id"), str):
                result = {"id": cast(str, parsed["id"])}
                if parsed.get("phase") in ("commentary", "final_answer"):
                    result["phase"] = cast(str, parsed["phase"])
                return result
        except (ValueError, TypeError, AttributeError):
            pass
    return {"id": signature}


def _normalize_id_part(part: str) -> str:
    units = part.encode("utf-16-le", errors="surrogatepass")
    part = "".join(chr(units[index] | units[index + 1] << 8) for index in range(0, len(units), 2))
    return re.sub(r"[^a-zA-Z0-9_-]", "_", part)[:64].rstrip("_")


def convert_responses_messages(
    model: Model, context: TranscriptContext, allowed_tool_call_providers: Set[str],
    options: ConvertResponsesMessagesOptions | None = None,
) -> list[ResponseItem]:
    options = options if options is not None else ConvertResponsesMessagesOptions()
    normalized = resolve_transcript(context, options.supports_mid_convo_system_messages)
    messages: list[ResponseItem] = []

    def normalize_tool_call_id(value: str, target: Model, source: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers or "|" not in value:
            return _normalize_id_part(value)
        call_id, item_id = value.split("|")[:2]
        call_id = _normalize_id_part(call_id)
        item_id = (
            ("fc_" + short_hash(item_id))[:64]
            if source.provider != model.provider or source.api != model.api else _normalize_id_part(item_id)
        )
        if not item_id.startswith("fc_"):
            item_id = _normalize_id_part("fc_" + item_id)
        return call_id + "|" + item_id

    transformed = transform_messages(normalized.messages, model, normalize_tool_call_id)
    transcript_tools = resolve_transcript_tools(
        normalized.messages, bool(options.supports_additional_tools or options.supports_tool_search),
    )
    instruction_role = "developer" if model.reasoning and (model.compat or {}).get("supportsDeveloperRole") is not False else "system"
    grammar_properties = options.grammar_tool_input_properties or {}
    message_index = 0
    for source_index, message in enumerate(transformed):
        leading_system = source_index == 0 and message.role == "system"
        if message.role == "system":
            system = cast(SystemMessage, message)
            if not leading_system:
                tools = (system.tools_added or []) if transcript_tools.anchors_additions else []
                if tools:
                    if options.supports_additional_tools:
                        messages.append({"type": "additional_tools", "role": "developer", "tools": convert_responses_tools(tools, options.tool_options)})
                    elif options.supports_tool_search:
                        names = [tool.name for tool in tools]
                        call_id = "pi_tool_load_" + short_hash(f"system:{message_index}:" + ",".join(names))
                        messages.append({
                            "type": "tool_search_call", "call_id": call_id, "execution": "client", "status": "completed",
                            "arguments": {"query": " ".join(names), "limit": len(names)},
                        })
                        tool_options = copy(options.tool_options) if options.tool_options is not None else ConvertResponsesToolsOptions()
                        tool_options.tool_search_result = True
                        messages.append({
                            "type": "tool_search_output", "call_id": call_id, "execution": "client", "status": "completed",
                            "tools": convert_responses_tools(tools, tool_options),
                        })
            if not leading_system or options.include_system_prompt is not False:
                text = get_system_message_text(system) if leading_system else render_system_message_update(system)
                if text:
                    messages.append({"role": instruction_role, "content": sanitize_surrogates(text)})
        elif message.role == "user":
            user = cast(UserMessage, message)
            if isinstance(user.content, str):
                messages.append({"role": "user", "content": [{"type": "input_text", "text": sanitize_surrogates(user.content)}]})
            else:
                content: list[ResponseItem] = []
                for item in user.content:
                    if item.type == "text":
                        content.append({"type": "input_text", "text": sanitize_surrogates(cast(TextContent, item).text)})
                    else:
                        image = cast(ImageContent, item)
                        content.append({"type": "input_image", "detail": "auto", "image_url": f"data:{image.mime_type};base64,{image.data}"})
                if not content:
                    continue
                messages.append({"role": "user", "content": content})
        elif message.role == "assistant":
            assistant = cast(AssistantMessage, message)
            output: list[ResponseItem] = []
            same_api = assistant.provider == model.provider and assistant.api == model.api
            same_model = same_api and assistant.model == model.id
            different_model = same_api and assistant.model != model.id
            text_index = 0
            for block in assistant.content:
                if block.type == "thinking":
                    thinking = cast(ThinkingContent, block)
                    if thinking.thinking_signature:
                        output.append(cast(ResponseItem, javascript_json_parse(thinking.thinking_signature)))
                elif block.type == "text":
                    text_block = cast(TextContent, block)
                    signature = _parse_text_signature(text_block.text_signature)
                    fallback = f"msg_pi_{message_index}" + (f"_{text_index}" if text_index else "")
                    text_index += 1
                    message_id = (signature or {}).get("id") or fallback
                    if utf16_length(message_id) > 64:
                        message_id = "msg_" + short_hash(message_id)
                    item = {
                        "type": "message", "role": "assistant", "status": "completed", "id": message_id,
                        "content": [{"type": "output_text", "text": sanitize_surrogates(text_block.text), "annotations": []}],
                        "phase": (signature or {}).get("phase", UNDEFINED),
                    }
                    output.append(item)
                elif block.type == "toolCall":
                    tool_call = cast(ToolCall, block)
                    parts = tool_call.id.split("|")
                    call_id = parts[0]
                    item_id = parts[1] if len(parts) > 1 else None
                    custom_property = grammar_properties.get(tool_call.name)
                    if (different_model and item_id is not None and item_id.startswith("fc_")) or (
                        custom_property is None and (item_id is None or not item_id.startswith("fc_"))
                    ):
                        item_id = None
                    item = {
                        "type": "custom_tool_call" if custom_property is not None else "function_call",
                        "id": item_id if item_id is not None else UNDEFINED,
                        "call_id": call_id, "name": tool_call.name,
                    }
                    if custom_property is not None:
                        item["input"] = sanitize_surrogates(get_grammar_tool_input(tool_call.name, tool_call.arguments, custom_property))
                    else:
                        item["arguments"] = javascript_json_stringify(tool_call.arguments)
                    if same_model and tool_call.namespace is not None:
                        item["namespace"] = tool_call.namespace
                    output.append(item)
            if not output:
                continue
            messages.extend(output)
        elif message.role == "toolResult":
            result = cast(ToolResultMessage, message)
            text = "\n".join(cast(TextContent, part).text for part in result.content if part.type == "text")
            images = [cast(ImageContent, part) for part in result.content if part.type == "image"]
            tool_output: str | list[ResponseItem]
            if not images or "image" not in model.input:
                tool_output = sanitize_surrogates(text or ("(see attached image)" if images else "(no tool output)"))
            else:
                tool_output = [{"type": "input_text", "text": sanitize_surrogates(text)}] if text else []
                tool_output.extend({"type": "input_image", "detail": "auto", "image_url": f"data:{image.mime_type};base64,{image.data}"} for image in images)
            messages.append({
                "type": "custom_tool_call_output" if result.tool_name in grammar_properties else "function_call_output",
                "call_id": result.tool_call_id.split("|")[0], "output": tool_output,
            })
        if not leading_system:
            message_index += 1
    return messages


def convert_responses_tools(tools: Sequence[Tool], options: ConvertResponsesToolsOptions | None = None) -> list[ResponseItem]:
    options = options if options is not None else ConvertResponsesToolsOptions()
    supports_strict = options.supports_strict_mode is not False
    result: list[ResponseItem] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, bool(options.supports_openai_grammar_tools))
        if grammar is not None:
            converted: ResponseItem = {
                "type": "custom", "name": tool.name, "description": tool.description,
                "format": {"type": "grammar", "syntax": grammar.format, "definition": grammar.definition},
            }
        else:
            constrained = resolve_json_schema_strict_sampling(tool, supports_strict)
            strict = constrained if constrained is not None else options.strict
            converted = {
                "type": "function", "name": tool.name, "description": tool.description,
                "parameters": get_json_schema_tool_parameters(tool, strict is True),
            }
            if supports_strict:
                converted["strict"] = strict
        if options.tool_search_result:
            converted["defer_loading"] = True
        result.append(converted)
    return result


@dataclass
class _CustomInput:
    property: str
    json_buffer: GrammarToolInputJsonBuffer


class _StreamingToolCall(ToolCall):
    partial_json: str
    custom_input: _CustomInput

    def to_json(self) -> JsonObject:
        result = super().to_json()
        if hasattr(self, "partial_json"):
            result["partialJson"] = self.partial_json
        if hasattr(self, "custom_input"):
            buffer = self.custom_input.json_buffer
            result["customInput"] = {
                "property": self.custom_input.property,
                "jsonBuffer": {"input": buffer.input, "started": buffer.started, "closed": buffer.closed},
            }
        return result


@dataclass
class _OutputSlot:
    type: Literal["thinking", "text", "toolCall"]
    block: ThinkingContent | TextContent | _StreamingToolCall
    content_index: int


def _custom_tool_input(block: _StreamingToolCall) -> str:
    custom = getattr(block, "custom_input", None)
    if custom is None:
        return ""
    value = block.arguments.get(custom.property)
    return value if isinstance(value, str) else ""


def _append_custom_tool_input(block: _StreamingToolCall, next_input: str, close: bool) -> str | None:
    custom = getattr(block, "custom_input", None)
    if custom is None:
        return None
    delta = append_grammar_tool_input_json_delta(custom.json_buffer, custom.property, next_input, close)
    block.arguments = {custom.property: next_input}
    return delta


def _map_stop_reason(status: str | None, incomplete_reason: str | None) -> tuple[StopReason, str | None]:
    if not status or status in ("completed", "in_progress", "queued"):
        return "stop", None
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "length", None
        return "error", f"Response incomplete: {incomplete_reason}" if incomplete_reason else "Response incomplete without a provider reason"
    if status in ("failed", "cancelled"):
        return "error", None
    raise RuntimeError(f"Unhandled stop reason: {status}")


async def process_responses_stream(
    openai_stream: AsyncIterable[Mapping[str, object]], output: AssistantMessage,
    stream: AssistantMessageEventStream, model: Model, options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    options = options if options is not None else OpenAIResponsesStreamOptions()
    saw_terminal_event = False
    slots: dict[int, _OutputSlot] = {}
    reasoning_by_id: dict[str, ThinkingContent] = {}

    def apply_phase(item: Mapping[str, object]) -> None:
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            output.stop_reason = "stop"

    def get_slot(output_index: int, kind: str) -> _OutputSlot | None:
        slot = slots.get(output_index)
        return slot if slot is not None and slot.type == kind else None

    def push_tool_delta(slot: _OutputSlot, delta: str | None) -> None:
        if delta is not None:
            stream.push(AssistantMessageEvent(type="toolcall_delta", content_index=slot.content_index, delta=delta, partial=output))

    def create_slot(output_index: int, item: Mapping[str, object]) -> _OutputSlot | None:
        kind = item.get("type")
        block: ThinkingContent | TextContent | _StreamingToolCall
        if kind == "reasoning":
            block = ThinkingContent(thinking="")
            slot_type: Literal["thinking", "text", "toolCall"] = "thinking"
            event_type = "thinking_start"
        elif kind == "message":
            apply_phase(item)
            block = TextContent(text="")
            slot_type = "text"
            event_type = "text_start"
        elif kind in ("function_call", "custom_tool_call"):
            block = _StreamingToolCall(
                id=javascript_string(item.get("call_id", UNDEFINED)) + "|" + javascript_string(item.get("id", UNDEFINED)),
                name=cast(str, item["name"]), arguments={}, namespace=cast(str | None, item.get("namespace")),
            )
            if kind == "function_call":
                block.partial_json = cast(str, item.get("arguments") or "")
            else:
                input_property = (options.grammar_tool_input_properties or {}).get(block.name, "input")
                block.arguments = {input_property: cast(str, item.get("input") or "")}
                block.custom_input = _CustomInput(property=input_property, json_buffer=GrammarToolInputJsonBuffer())
            slot_type = "toolCall"
            event_type = "toolcall_start"
        else:
            return None
        output.content.append(block)
        slot = _OutputSlot(type=slot_type, block=block, content_index=len(output.content) - 1)
        slots[output_index] = slot
        stream.push(AssistantMessageEvent(type=event_type, content_index=slot.content_index, partial=output))
        return slot

    def finalize_response(response: Mapping[str, object]) -> None:
        nonlocal saw_terminal_event
        saw_terminal_event = True
        # #6409: Azure may provide encrypted reasoning only in the terminal output.
        for item in cast(list[Mapping[str, object]], response.get("output") or []):
            if item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = reasoning_by_id.get(cast(str, item.get("id")))
            if block is None or not block.thinking_signature:
                continue
            stored = cast(dict[str, object], javascript_json_parse(block.thinking_signature))
            if not stored.get("encrypted_content"):
                block.thinking_signature = javascript_json_stringify({**stored, "encrypted_content": item["encrypted_content"]})
        if response.get("id"):
            output.response_id = cast(str, response["id"])
        usage = cast(Mapping[str, object] | None, response.get("usage"))
        if usage is not None:
            input_details = cast(Mapping[str, object], usage.get("input_tokens_details") or {})
            output_details = cast(Mapping[str, object], usage.get("output_tokens_details") or {})
            cached = cast(int, input_details.get("cached_tokens") or 0)
            cache_write = cast(int, input_details.get("cache_write_tokens") or 0)
            output.usage = Usage(
                input=max(0, cast(int, usage.get("input_tokens") or 0) - cached - cache_write),
                output=cast(int, usage.get("output_tokens") or 0), cache_read=cached, cache_write=cache_write,
                reasoning=cast(int, output_details.get("reasoning_tokens") or 0), total_tokens=cast(int, usage.get("total_tokens") or 0),
            )
        calculate_cost(model, output.usage)
        if options.apply_service_tier_pricing is not None:
            response_tier = cast(ServiceTier | None, response.get("service_tier"))
            tier = (
                options.resolve_service_tier(response_tier, options.service_tier)
                if options.resolve_service_tier is not None
                else response_tier if response_tier is not None else options.service_tier
            )
            options.apply_service_tier_pricing(output.usage, tier)
        status = cast(str | None, response.get("status"))
        details = cast(Mapping[str, object], response.get("incomplete_details") or {})
        incomplete = details.get("reason")
        reason = incomplete if isinstance(incomplete, str) else None
        output.raw_stop_reason = javascript_string(response.get("status", UNDEFINED)) + "." + reason if reason else status
        output.stop_reason, output.error_message = _map_stop_reason(status, reason)
        if any(block.type == "toolCall" for block in output.content) and output.stop_reason == "stop":
            output.stop_reason = "toolUse"

    async for event in openai_stream:
        event_type = event.get("type")
        output_index = cast(int, event.get("output_index"))
        if event_type == "response.created":
            output.response_id = cast(str, cast(Mapping[str, object], event["response"])["id"])
        elif event_type == "response.output_item.added":
            create_slot(output_index, cast(Mapping[str, object], event["item"]))
        elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_summary_part.done", "response.reasoning_text.delta"):
            slot = get_slot(output_index, "thinking")
            if slot is None:
                continue
            thinking = cast(ThinkingContent, slot.block)
            delta = "\n\n" if event_type == "response.reasoning_summary_part.done" else cast(str, event["delta"])
            thinking.thinking += delta
            stream.push(AssistantMessageEvent(type="thinking_delta", content_index=slot.content_index, delta=delta, partial=output))
        elif event_type in ("response.output_text.delta", "response.refusal.delta"):
            slot = get_slot(output_index, "text")
            if slot is None:
                continue
            delta = cast(str, event["delta"])
            cast(TextContent, slot.block).text += delta
            stream.push(AssistantMessageEvent(type="text_delta", content_index=slot.content_index, delta=delta, partial=output))
        elif event_type in ("response.function_call_arguments.delta", "response.function_call_arguments.done"):
            slot = get_slot(output_index, "toolCall")
            if slot is None or not hasattr(slot.block, "partial_json"):
                continue
            tool = cast(_StreamingToolCall, slot.block)
            if event_type == "response.function_call_arguments.delta":
                delta = cast(str, event["delta"])
                tool.partial_json += delta
                tool.arguments = cast(JsonObject, parse_streaming_json(tool.partial_json))
                push_tool_delta(slot, delta)
            else:
                previous = tool.partial_json.encode("utf-16-le", errors="surrogatepass")
                tool.partial_json = cast(str, event["arguments"])
                tool.arguments = cast(JsonObject, parse_streaming_json(tool.partial_json))
                complete = tool.partial_json.encode("utf-16-le", errors="surrogatepass")
                if complete.startswith(previous) and len(complete) > len(previous):
                    push_tool_delta(slot, complete[len(previous):].decode("utf-16-le", errors="surrogatepass"))
        elif event_type in ("response.custom_tool_call_input.delta", "response.custom_tool_call_input.done"):
            slot = get_slot(output_index, "toolCall")
            if slot is None or not hasattr(slot.block, "custom_input"):
                continue
            tool = cast(_StreamingToolCall, slot.block)
            close = event_type == "response.custom_tool_call_input.done"
            next_input = cast(str, event["input"]) if close else _custom_tool_input(tool) + cast(str, event["delta"])
            push_tool_delta(slot, _append_custom_tool_input(tool, next_input, close))
        elif event_type == "response.output_item.done":
            item = cast(Mapping[str, object], event["item"])
            apply_phase(item)
            slot = slots.get(output_index)
            if slot is None:
                slot = create_slot(output_index, item)
            if slot is None:
                continue
            kind = item.get("type")
            if kind == "reasoning" and slot.type == "thinking":
                thinking = cast(ThinkingContent, slot.block)
                summary = "\n\n".join(cast(str, part.get("text") or "") for part in cast(list[Mapping[str, object]], item.get("summary") or []))
                content = "\n\n".join(cast(str, part.get("text") or "") for part in cast(list[Mapping[str, object]], item.get("content") or []))
                thinking.thinking = summary or content or thinking.thinking
                thinking.thinking_signature = javascript_json_stringify(item)
                reasoning_by_id[cast(str, item["id"])] = thinking
                stream.push(AssistantMessageEvent(type="thinking_end", content_index=slot.content_index, content=thinking.thinking, partial=output))
                slots.pop(output_index, None)
            elif kind == "message" and slot.type == "text":
                text = cast(TextContent, slot.block)
                text.text = "".join(
                    cast(str, part.get("text" if part.get("type") == "output_text" else "refusal") or "")
                    for part in cast(list[Mapping[str, object]], item.get("content") or [])
                )
                signature: dict[str, object] = {"v": 1, "id": item["id"]}
                if item.get("phase"):
                    signature["phase"] = item["phase"]
                text.text_signature = javascript_json_stringify(signature)
                stream.push(AssistantMessageEvent(type="text_end", content_index=slot.content_index, content=text.text, partial=output))
                slots.pop(output_index, None)
            elif kind == "function_call" and slot.type == "toolCall" and hasattr(slot.block, "partial_json"):
                tool = cast(_StreamingToolCall, slot.block)
                tool.arguments = cast(JsonObject, parse_streaming_json(cast(str, item.get("arguments") or tool.partial_json or "{}")))
                if "namespace" in item:
                    tool.namespace = cast(str, item["namespace"])
                del tool.partial_json
                stream.push(AssistantMessageEvent(type="toolcall_end", content_index=slot.content_index, tool_call=tool, partial=output))
                slots.pop(output_index, None)
            elif kind == "custom_tool_call" and slot.type == "toolCall" and hasattr(slot.block, "custom_input"):
                tool = cast(_StreamingToolCall, slot.block)
                value = item.get("input")
                push_tool_delta(slot, _append_custom_tool_input(tool, cast(str, value) if value is not None else _custom_tool_input(tool), True))
                if "namespace" in item:
                    tool.namespace = cast(str, item["namespace"])
                del tool.custom_input
                stream.push(AssistantMessageEvent(type="toolcall_end", content_index=slot.content_index, tool_call=tool, partial=output))
                slots.pop(output_index, None)
        elif event_type in ("response.completed", "response.incomplete"):
            finalize_response(cast(Mapping[str, object], event["response"]))
        elif event_type == "error":
            raise RuntimeError(f"Error Code {javascript_string(event.get('code', UNDEFINED))}: {javascript_string(event.get('message', UNDEFINED))}")
        elif event_type == "response.failed":
            saw_terminal_event = True
            response = cast(Mapping[str, object], event.get("response") or {})
            output.raw_stop_reason = cast(str | None, response.get("status"))
            error = cast(Mapping[str, object] | None, response.get("error"))
            details = cast(Mapping[str, object], response.get("incomplete_details") or {})
            if error is not None:
                message = javascript_string(error.get("code") or "unknown") + ": " + javascript_string(error.get("message") or "no message")
            elif details.get("reason"):
                message = "incomplete: " + javascript_string(details["reason"])
            else:
                message = "Unknown error (no error details in response)"
            raise RuntimeError(message)
    if not saw_terminal_event:
        raise RuntimeError("OpenAI Responses stream ended before a terminal response event")


__all__ = [
    "OpenAIResponsesStreamOptions", "ConvertResponsesMessagesOptions", "ConvertResponsesToolsOptions",
    "convert_responses_messages", "convert_responses_tools", "process_responses_stream",
]
