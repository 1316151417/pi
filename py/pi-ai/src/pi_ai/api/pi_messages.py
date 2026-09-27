"""pi's native message protocol from ``api/pi-messages.ts``."""

from __future__ import annotations

import asyncio
import codecs
import inspect
import math
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping, MutableSequence
from copy import copy
from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypedDict, cast

from ada_url import URLSearchParams

from .._javascript import javascript_json_parse, javascript_json_stringify, javascript_string, utf16_length, utf16_slice
from .._json_runtime import JS_WHITESPACE, utf16_units
from .._message_wire import (
    assistant_message_event_from_wire, assistant_message_from_wire,
    text_content_from_wire, thinking_content_from_wire, tool_call_from_wire,
)
from .._values import JSON_NULL, UNDEFINED
from ..auth.oauth._http import fetch, parse_url
from ..event_stream import AssistantMessageEventStream
from ..json_parse import parse_streaming_json
from ..types import (
    AssistantContentBlock, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    CacheRetention, JsonObject, Model, ProviderEnv, ProviderResponse, SimpleStreamOptions,
    StreamOptions, ThinkingLevel, TranscriptContext, Usage,
)
from ..utils.diagnostics import append_assistant_message_diagnostic, create_assistant_message_diagnostic
from ..utils.headers import headers_to_record, provider_headers_to_record
from ..utils.provider_env import get_provider_env_value


class _FunctionName(TypedDict):
    name: str


class NamedToolChoice(TypedDict):
    type: Literal["function"]
    function: _FunctionName


@dataclass
class PiMessagesOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    debug: bool | None = None


class PiMessagesRewriteImpact(TypedDict):
    policyId: str
    policyVersion: int | float
    changed: bool
    tokenCountChange: int | float
    messageCountChange: int | float
    systemPromptChanged: bool


class _UsageCost(TypedDict):
    input: int | float
    output: int | float
    cacheRead: int | float
    cacheWrite: int | float
    total: int | float


class PiMessagesUsage(TypedDict):
    input: int | float
    output: int | float
    cacheRead: int | float
    cacheWrite: int | float
    cacheWrite1h: NotRequired[int | float]
    reasoning: NotRequired[int | float]
    totalTokens: int | float
    cost: _UsageCost


class _ToolCall(TypedDict):
    type: Literal["toolCall"]
    id: str
    name: str
    arguments: JsonObject
    thoughtSignature: NotRequired[str]
    namespace: NotRequired[str]


class _StartEvent(TypedDict):
    type: Literal["start"]


class _ContentStartEvent(TypedDict):
    type: Literal["text_start", "thinking_start"]
    contentIndex: int | float


class _ContentDeltaEvent(TypedDict):
    type: Literal["text_delta", "thinking_delta", "toolcall_delta"]
    contentIndex: int | float
    delta: str


class _TextEndEvent(TypedDict):
    type: Literal["text_end"]
    contentIndex: int | float
    content: str
    contentSignature: NotRequired[str]


class _ThinkingEndEvent(TypedDict):
    type: Literal["thinking_end"]
    contentIndex: int | float
    content: str
    contentSignature: NotRequired[str]
    redacted: NotRequired[bool]


class _ToolCallStartEvent(TypedDict):
    type: Literal["toolcall_start"]
    contentIndex: int | float
    id: str
    toolName: str


class _ToolCallEndEvent(TypedDict):
    type: Literal["toolcall_end"]
    contentIndex: int | float
    toolCall: _ToolCall


class _DoneEvent(TypedDict):
    type: Literal["done"]
    reason: Literal["stop", "length", "toolUse"]
    usage: PiMessagesUsage
    responseId: NotRequired[str]
    providerThinkingLevel: NotRequired[str]
    rewrite: NotRequired[PiMessagesRewriteImpact]


class _ErrorEvent(TypedDict):
    type: Literal["error"]
    reason: Literal["aborted", "error"]
    usage: PiMessagesUsage
    errorMessage: NotRequired[str]
    responseId: NotRequired[str]
    providerThinkingLevel: NotRequired[str]
    rewrite: NotRequired[PiMessagesRewriteImpact]


type PiMessagesEvent = (
    _StartEvent | _ContentStartEvent | _ContentDeltaEvent | _TextEndEvent | _ThinkingEndEvent
    | _ToolCallStartEvent | _ToolCallEndEvent | _DoneEvent | _ErrorEvent
)


class _Response(Protocol):
    status: int
    status_text: str
    ok: bool
    headers: Mapping[str, str]
    body: AsyncIterable[bytes] | None

    async def text(self) -> str: ...


class PiMessagesResponseError(Exception):
    name = "PiMessagesResponseError"

    def __init__(self, message: str, code: str | None, diagnostic_details: JsonObject) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic_details = diagnostic_details

    @property
    def message(self) -> str:
        return str(self)


def _truthy(value: object) -> bool:
    if value is None or value is UNDEFINED or value is JSON_NULL or value is False:
        return False
    if isinstance(value, (str, int, float)):
        return bool(value) and not (isinstance(value, float) and math.isnan(value))
    return True


def _record(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return {str(index): unit for index, unit in enumerate(utf16_units(value))}
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    return {}


def _response_error(model: Model, url: str, response: _Response, body: str) -> PiMessagesResponseError:
    error: Mapping[str, object] | None = None
    try:
        parsed = javascript_json_parse(body)
        candidate = parsed.get("error") if isinstance(parsed, Mapping) else None
        if isinstance(candidate, Mapping):
            error = candidate
    except Exception:
        pass
    message = error.get("message") if error is not None else None
    code = error.get("code") if error is not None else None
    code = code if isinstance(code, str) else None
    suffix = message if isinstance(message, str) else body
    diagnostic: dict[str, object] = {
        "version": 1, "provider": model.provider, "model": model.id, "url": url,
        "status": response.status, "statusText": response.status_text,
    }
    if error is not None:
        diagnostic["error"] = error
    else:
        diagnostic["body"] = utf16_slice(body, 8192) + "…" if utf16_length(body) > 8192 else body
    diagnostic["timestampMs"] = time.time_ns() // 1_000_000
    return PiMessagesResponseError(
        f"{response.status} {response.status_text}: {suffix}" + (f" ({code})" if code else ""),
        code, cast(JsonObject, diagnostic),
    )


def _append_rewrite(partial: AssistantMessage, rewrite: object) -> None:
    if _truthy(rewrite):
        append_assistant_message_diagnostic(partial, AssistantMessageDiagnostic(
            type="pi_messages_rewrite", timestamp=time.time_ns() // 1_000_000, details=cast(JsonObject, _record(rewrite)),
        ))


def _content_key(value: object) -> int | str:
    key = javascript_string(value)
    if key.isascii() and key.isdigit() and (key == "0" or not key.startswith("0")):
        index = int(key)
        if index < 0xFFFFFFFF:
            return index
    return key


def _create_event_converter(model: Model) -> Callable[[object], AssistantMessageEvent]:
    partial = assistant_message_from_wire({
        "role": "assistant", "content": [], "api": model.api, "provider": model.provider, "model": model.id,
        "usage": Usage().to_json(), "stopReason": "pending", "timestamp": time.time_ns() // 1_000_000,
    })
    tool_json: dict[tuple[str, object], str] = {}
    array_properties: dict[str, AssistantContentBlock] = {}

    def convert(event: object) -> AssistantMessageEvent:
        wire = _record(event)
        kind = wire.get("type", UNDEFINED)
        if kind in ("done", "error"):
            partial.stop_reason = cast(str, wire.get("reason", UNDEFINED))
            partial.usage = cast(Usage, wire.get("usage", UNDEFINED))
            partial.response_id = cast(str | None, wire.get("responseId", UNDEFINED))
            if kind == "error":
                partial.error_message = cast(str | None, wire.get("errorMessage", UNDEFINED))
            if wire.get("providerThinkingLevel", UNDEFINED) is not UNDEFINED:
                partial.provider_thinking_level = cast(str | None, wire["providerThinkingLevel"])
            _append_rewrite(partial, wire.get("rewrite", UNDEFINED))
            return assistant_message_event_from_wire({
                "type": kind, "reason": wire.get("reason", UNDEFINED), "message" if kind == "done" else "error": partial,
            })

        index_value = wire.get("contentIndex", UNDEFINED)
        index = _content_key(index_value)
        map_key = ("number" if type(index_value) in (int, float) else type(index_value).__name__, index_value)
        content = cast(MutableSequence[AssistantContentBlock], partial.content)
        if kind in ("text_start", "thinking_start", "toolcall_start"):
            if kind == "text_start":
                block = cast(AssistantContentBlock, text_content_from_wire({"type": "text", "text": ""}))
            elif kind == "thinking_start":
                block = cast(AssistantContentBlock, thinking_content_from_wire({"type": "thinking", "thinking": ""}))
            else:
                block = tool_call_from_wire({"type": "toolCall", "id": wire.get("id", UNDEFINED), "name": wire.get("toolName", UNDEFINED), "arguments": {}})
                tool_json[map_key] = ""
            if isinstance(index, int):
                while len(content) <= index:
                    content.append(cast(AssistantContentBlock, UNDEFINED))
                content[index] = block
            else:
                array_properties[index] = block
        elif kind in ("text_delta", "thinking_delta", "text_end", "thinking_end", "toolcall_delta", "toolcall_end"):
            block_value = content[index] if isinstance(index, int) and index < len(content) else array_properties.get(cast(str, index), UNDEFINED)
            if block_value is UNDEFINED or block_value is None:
                raise TypeError("Cannot convert undefined or null to object" if kind in ("text_end", "thinking_end", "toolcall_end") else "Cannot read properties of undefined")
            block = cast(AssistantContentBlock, block_value)
            fields = cast(dict[str, object], block.to_json())
            if kind in ("text_delta", "thinking_delta"):
                name = "text" if kind == "text_delta" else "thinking"
                fields[name] = javascript_string(fields.get(name, UNDEFINED)) + javascript_string(wire.get("delta", UNDEFINED))
            elif kind == "text_end":
                fields.update(text=wire.get("content", UNDEFINED), textSignature=wire.get("contentSignature", UNDEFINED))
            elif kind == "thinking_end":
                fields.update(thinking=wire.get("content", UNDEFINED), thinkingSignature=wire.get("contentSignature", UNDEFINED), redacted=wire.get("redacted", UNDEFINED))
            elif kind == "toolcall_delta":
                json = tool_json.get(map_key, "") + javascript_string(wire.get("delta", UNDEFINED))
                tool_json[map_key] = json
                fields["arguments"] = parse_streaming_json(json)
            else:
                fields.update(_record(wire.get("toolCall", UNDEFINED)))
                tool_json.pop(map_key, None)
                return assistant_message_event_from_wire({"type": "toolcall_end", "contentIndex": index_value, "toolCall": block, "partial": partial})
        return assistant_message_event_from_wire({**wire, "partial": partial})

    return convert


def _parse_event(raw: str) -> object:
    data = next((line[5:].strip(JS_WHITESPACE) for line in raw.split("\n") if line.startswith("data:")), None)
    return javascript_json_parse(data) if data and data != "[DONE]" else UNDEFINED


async def _read_events(stream: AsyncIterable[bytes]) -> AsyncIterator[object]:
    decoder = codecs.getincrementaldecoder("utf-8-sig")("replace")
    reader = stream.__aiter__()
    buffer = ""
    try:
        while True:
            try:
                value = await anext(reader)
                done = False
            except StopAsyncIteration:
                value, done = b"", True
            buffer += decoder.decode(value, final=done)
            buffer = buffer.replace("\r\n", "\n")
            split = buffer.find("\n\n")
            while split != -1:
                event = _parse_event(buffer[:split])
                if _truthy(event):
                    yield event
                buffer = buffer[split + 2:]
                split = buffer.find("\n\n")
            if done:
                break
        if buffer.strip(JS_WHITESPACE):
            event = _parse_event(buffer)
            if _truthy(event):
                yield event
    finally:
        # ReadableStream reader.releaseLock() does not cancel the body.
        del reader


def _error_event(model: Model, error: object, aborted: bool) -> AssistantMessageEvent:
    reason = "aborted" if aborted else "error"
    message = assistant_message_from_wire({
        "role": "assistant", "content": [], "api": model.api, "provider": model.provider, "model": model.id,
        "usage": Usage().to_json(), "stopReason": reason,
        "errorMessage": getattr(error, "message", str(error)) if isinstance(error, BaseException) else javascript_string(error),
        "timestamp": time.time_ns() // 1_000_000,
    })
    if not aborted and isinstance(error, PiMessagesResponseError):
        append_assistant_message_diagnostic(message, create_assistant_message_diagnostic("pi_messages_response_failure", error, error.diagnostic_details))
    return assistant_message_event_from_wire({"type": "error", "reason": reason, "error": message})


def _resolve_cache_retention(retention: CacheRetention | None, env: ProviderEnv | None) -> object:
    if _truthy(retention):
        return retention
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else UNDEFINED


def stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    event_stream = AssistantMessageEventStream()
    convert = _create_event_converter(model)

    async def run() -> None:
        try:
            api_key = options.api_key if options is not None else None
            if not api_key:
                raise RuntimeError(f'No API key provided for provider "{model.provider}"')
            url = parse_url(model.base_url.rstrip("/") + "/messages")
            if getattr(options, "debug", None):
                parameters = URLSearchParams(url.search)
                parameters.set("debug", "1")
                url.search = str(parameters)
            payload: object = {
                "model": model.id, "context": context,
                "options": {
                    "temperature": options.temperature if options is not None and options.temperature is not None else UNDEFINED,
                    "maxTokens": options.max_tokens if options is not None and options.max_tokens is not None else UNDEFINED,
                    "reasoning": getattr(options, "reasoning", None) if getattr(options, "reasoning", None) is not None else UNDEFINED,
                    "cacheRetention": _resolve_cache_retention(options.cache_retention if options is not None else None, options.env if options is not None else None),
                    "sessionId": options.session_id if options is not None and options.session_id is not None else UNDEFINED,
                    "toolChoice": getattr(options, "tool_choice", None) if getattr(options, "tool_choice", None) is not None else UNDEFINED,
                },
            }
            replacement = options.on_payload(payload, model) if options is not None and options.on_payload is not None else UNDEFINED
            if inspect.isawaitable(replacement):
                replacement = await replacement
            else:
                await asyncio.sleep(0)
            if replacement is not None and replacement is not UNDEFINED:
                payload = replacement
            response = cast(_Response, await (options.fetch if options is not None and options.fetch is not None else fetch)(
                url.href, method="POST", headers={
                    "authorization": f"Bearer {api_key}", "accept": "text/event-stream", "content-type": "application/json",
                    **(provider_headers_to_record(options.headers if options is not None else None) or {}),
                }, body=javascript_json_stringify(payload), signal=options.signal if options is not None else None,
            ))
            callback = options.on_response(ProviderResponse(status=response.status, headers=headers_to_record(response.headers)), model) if options is not None and options.on_response is not None else None
            if inspect.isawaitable(callback):
                await callback
            else:
                await asyncio.sleep(0)
            if not response.ok:
                raise _response_error(model, url.href, response, await response.text())
            if response.body is None:
                raise RuntimeError(f"{model.provider} response has no body")
            events = _read_events(response.body)
            try:
                async for pi_event in events:
                    event = convert(pi_event)
                    event_stream.push(event)
                    if event.type in ("done", "error"):
                        return
            finally:
                await events.aclose()
            raise RuntimeError(f"{model.provider} stream ended without a terminal event")
        except (Exception, asyncio.CancelledError) as error:
            event_stream.push(_error_event(model, error, bool(options is not None and options.signal is not None and options.signal.aborted)))

    asyncio.get_running_loop().create_task(run())
    return event_stream


def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
    return stream(model, context, copy(options) if options is not None else PiMessagesOptions())


__all__ = [
    "NamedToolChoice", "PiMessagesOptions", "PiMessagesRewriteImpact", "PiMessagesUsage", "PiMessagesEvent",
    "PiMessagesResponseError", "stream", "stream_simple",
]
