"""Google Generative AI and Vertex shared conversion from google-shared.ts.

The wire enums below match the pinned @google/genai 2.21.0 declarations; this
module uses native Python records and does not call a JavaScript SDK.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal, TypedDict, cast

from .._json_runtime import JS_WHITESPACE, utf16_units
from .._values import UNDEFINED
from ..models import clamp_thinking_level
from ..transcript import collapse_system_messages, without_initial_system_message
from ..types import (
    AssistantMessage, ImageContent, Model, StopReason, StreamOptions, TextContent, ThinkingContent,
    ThinkingLevel, Tool, ToolCall, ToolResultMessage, TranscriptContext, UserMessage,
)
from ..utils._javascript import javascript_object_keys, javascript_string
from ..utils.provider_retry import retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .transform_messages import transform_messages

type GoogleApiType = Literal["google-generative-ai", "google-vertex"]
type GoogleApiThinkingLevel = Literal["THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"]
type ResolvedGoogleThinkingLevel = Literal["minimal", "low", "medium", "high"]
type FunctionCallingConfigMode = Literal["MODE_UNSPECIFIED", "AUTO", "ANY", "NONE", "VALIDATED"]
type GooglePart = dict[str, object]


class GoogleContent(TypedDict):
    role: str
    parts: list[GooglePart]


def resolve_google_thinking_level(model: Model, level: ThinkingLevel) -> ResolvedGoogleThinkingLevel:
    mapped = (model.thinking_level_map or {}).get(level, UNDEFINED)
    resolved = mapped.lower() if isinstance(mapped, str) else level
    if resolved in ("minimal", "low", "medium", "high"):
        return cast(ResolvedGoogleThinkingLevel, resolved)
    raise RuntimeError(
        f"Unsupported Google thinking level mapping for {model.provider}/{model.id}: {level} -> {javascript_string(mapped)}",
    )


def uses_google_thinking_level(model: Model) -> bool:
    model_id = model.id.lower()
    return bool(
        re.search(r"gemini-3(?:\.[0-9]+)?-(?:pro|flash)", model_id)
        or model_id in ("gemini-flash-latest", "gemini-flash-lite-latest") or re.search(r"gemma-?4", model_id)
    )


def to_google_thinking_level(level: ResolvedGoogleThinkingLevel) -> GoogleApiThinkingLevel:
    return cast(GoogleApiThinkingLevel, {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(level))


def to_google_sdk_thinking_level(level: GoogleApiThinkingLevel) -> GoogleApiThinkingLevel:
    return cast(GoogleApiThinkingLevel, {
        "THINKING_LEVEL_UNSPECIFIED": "THINKING_LEVEL_UNSPECIFIED", "MINIMAL": "MINIMAL", "LOW": "LOW",
        "MEDIUM": "MEDIUM", "HIGH": "HIGH",
    }.get(level))


def get_disabled_google_thinking_config(model: Model) -> dict[str, object]:
    if not uses_google_thinking_level(model):
        return {"thinkingBudget": 0}
    fallback = clamp_thinking_level(model, "off")
    if fallback == "off":
        return {"thinkingBudget": 0}
    level = resolve_google_thinking_level(model, fallback)
    return {"thinkingLevel": to_google_sdk_thinking_level(to_google_thinking_level(level))}


def is_thinking_part(part: Mapping[str, object]) -> bool:
    return part.get("thought") is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    return incoming if isinstance(incoming, str) and len(incoming) > 0 else existing


def _resolve_thought_signature(same_provider_and_model: bool, signature: str | None) -> str | None:
    if not same_provider_and_model or not signature or len(signature) % 4:
        return None
    return signature if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", signature) else None


def _gemini_major_version(model_id: str) -> float | None:
    match = re.match(r"^gemini(?:-live)?-([0-9]+)", model_id.lower())
    return float(match[1]) if match is not None else None


def requires_tool_call_id(model_id: str) -> bool:
    major = _gemini_major_version(model_id)
    return model_id.startswith(("claude-", "gpt-oss-")) or major is not None and major >= 3


def convert_messages(model: Model, context: TranscriptContext) -> list[GoogleContent]:
    conversation = without_initial_system_message(collapse_system_messages(context).messages)
    contents: list[GoogleContent] = []

    def normalize_tool_call_id(value: str, target: Model, source: AssistantMessage) -> str:
        if not requires_tool_call_id(model.id):
            return value
        return re.sub(r"[^a-zA-Z0-9_-]", "_", utf16_units(value))[:64]

    transformed = transform_messages(conversation, model, normalize_tool_call_id)
    for message in transformed:
        if message.role == "user":
            user = cast(UserMessage, message)
            if isinstance(user.content, str):
                contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(user.content)}]})
            else:
                parts: list[GooglePart] = []
                for item in user.content:
                    if item.type == "text":
                        parts.append({"text": sanitize_surrogates(cast(TextContent, item).text)})
                    else:
                        image = cast(ImageContent, item)
                        parts.append({"inlineData": {"mimeType": image.mime_type, "data": image.data}})
                if parts:
                    contents.append({"role": "user", "parts": parts})
        elif message.role == "assistant":
            assistant = cast(AssistantMessage, message)
            parts = []
            same_model = assistant.provider == model.provider and assistant.model == model.id
            for block in assistant.content:
                part: GooglePart
                if block.type == "text":
                    text = cast(TextContent, block)
                    signature = _resolve_thought_signature(same_model, text.text_signature)
                    if (not text.text or not text.text.strip(JS_WHITESPACE)) and not signature:
                        continue
                    part = {"text": sanitize_surrogates(text.text)}
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
                elif block.type == "thinking":
                    thinking = cast(ThinkingContent, block)
                    signature = _resolve_thought_signature(same_model, thinking.thinking_signature) if same_model else None
                    if (not thinking.thinking or not thinking.thinking.strip(JS_WHITESPACE)) and not signature:
                        continue
                    part = {"text": sanitize_surrogates(thinking.thinking)}
                    if same_model:
                        part = {"thought": True, **part}
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
                elif block.type == "toolCall":
                    tool = cast(ToolCall, block)
                    signature = _resolve_thought_signature(same_model, tool.thought_signature)
                    function: dict[str, object] = {"name": tool.name, "args": tool.arguments if tool.arguments is not None else {}}
                    if requires_tool_call_id(model.id):
                        function["id"] = tool.id
                    part = {"functionCall": function}
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif message.role == "toolResult":
            result = cast(ToolResultMessage, message)
            text_result = "\n".join(cast(TextContent, block).text for block in result.content if block.type == "text")
            images = [cast(ImageContent, block) for block in result.content if block.type == "image"] if "image" in model.input else []
            major = _gemini_major_version(model.id)
            multimodal = major is None or major >= 3
            value = sanitize_surrogates(text_result) if text_result else "(see attached image)" if images else ""
            image_parts: list[GooglePart] = [{"inlineData": {"mimeType": image.mime_type, "data": image.data}} for image in images]
            response: dict[str, object] = {"name": result.tool_name, "response": {"error" if result.is_error else "output": value}}
            if images and multimodal:
                response["parts"] = image_parts
            if requires_tool_call_id(model.id):
                response["id"] = result.tool_call_id
            part = {"functionResponse": response}
            last = contents[-1] if contents else None
            if last is not None and last["role"] == "user" and any("functionResponse" in part and part["functionResponse"] is not None for part in last["parts"]):
                last["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            if images and not multimodal:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})
    return contents


_JSON_SCHEMA_META_DECLARATIONS = frozenset({
    "$schema", "$id", "$anchor", "$dynamicAnchor", "$vocabulary", "$comment", "$defs", "definitions",
})


def _sanitize_for_open_api(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    # The source intentionally does not descend through schema arrays.
    return {
        key: _sanitize_for_open_api(schema[key])
        for key in javascript_object_keys(schema) if key not in _JSON_SCHEMA_META_DECLARATIONS
    }


def convert_tools(
    tools: Sequence[Tool], use_parameters: bool = False, supports_strict_mode: bool = True,
) -> list[dict[str, object]] | None:
    if not tools:
        return None
    declarations: list[dict[str, object]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        parameters = get_json_schema_tool_parameters(tool, strict)
        declarations.append({
            "name": tool.name, "description": tool.description,
            "parameters" if use_parameters else "parametersJsonSchema": _sanitize_for_open_api(parameters) if use_parameters else parameters,
        })
    return [{"functionDeclarations": declarations}]


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    major = _gemini_major_version(model_id)
    return major is not None and major >= 3


def map_tool_choice(choice: str) -> FunctionCallingConfigMode:
    return {"auto": "AUTO", "none": "NONE", "any": "ANY"}.get(choice, "AUTO")


def resolve_google_function_calling_mode(
    tools: Sequence[Tool], tool_choice: str | None, supports_strict_mode: bool,
) -> FunctionCallingConfigMode | None:
    strict = any(resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True for tool in tools)
    if tool_choice in ("none", "any"):
        return map_tool_choice(tool_choice)
    if strict:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


def map_stop_reason(reason: str) -> StopReason:
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in {
        "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "SAFETY", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION", "IMAGE_OTHER", "RECITATION", "FINISH_REASON_UNSPECIFIED", "OTHER", "LANGUAGE",
        "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "NO_IMAGE",
    }:
        return "error"
    raise RuntimeError(f"Unhandled stop reason: {reason}")


def map_stop_reason_string(reason: str) -> StopReason:
    return "stop" if reason == "STOP" else "length" if reason == "MAX_TOKENS" else "error"


async def retry_google_request[T](request: Callable[[], Awaitable[T]], options: StreamOptions | None = None) -> T:
    async def run() -> T:
        try:
            return await request()
        except Exception as error:
            if hasattr(error, "status") and not hasattr(error, "headers"):
                setattr(error, "headers", None)
            raise

    return await retry_provider_request(
        run, max_retries=options.max_retries if options is not None and options.max_retries is not None else 0,
        max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
        signal=options.signal if options is not None else None,
    )


__all__ = [
    "GoogleApiType", "GoogleApiThinkingLevel", "ResolvedGoogleThinkingLevel", "FunctionCallingConfigMode",
    "GooglePart", "GoogleContent", "resolve_google_thinking_level", "uses_google_thinking_level",
    "to_google_thinking_level", "to_google_sdk_thinking_level", "get_disabled_google_thinking_config",
    "is_thinking_part", "retain_thought_signature", "requires_tool_call_id", "convert_messages", "convert_tools",
    "supports_google_strict_tool_sampling", "map_tool_choice", "resolve_google_function_calling_mode",
    "map_stop_reason", "map_stop_reason_string", "retry_google_request",
]
