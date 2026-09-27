"""Faux provider ported from pi-ai ``src/providers/faux.ts``.

Deterministic, scriptable provider used by the test suites. Responses are
queued as :class:`AssistantMessage` objects or factories and streamed back with
deltas, mirroring the TypeScript implementation.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import random
import time
from collections.abc import Mapping
from typing import Awaitable, Callable, Dict, List, Literal, Optional, Sequence, TypedDict, Union, cast, overload

from .._javascript import javascript_json_stringify, utf16_length
from .._json_runtime import utf16_units
from .._random import random_base36
from ..abort import AbortSignal
from ..auth.types import ApiKeyAuth, ApiKeyAuthInput, AuthResult, ModelAuth, ProviderAuth
from ..models import CreateProviderOptions, Provider, create_provider

from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    DeferredHandle,
    DeferredCancelOptions,
    DeferredFetchOptions,
    ImageContent,
    JsonObject,
    Message,
    Model,
    ProviderResponse,
    ProviderRequestOptions,
    ProviderStreams,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    Cost,
    DeferredCancelFunction,
    DeferredFetchFunction,
    SimpleStreamFunction,
    StreamFunction,
    model_from_json,
)
from ..event_stream import AssistantMessageEventStream, create_assistant_message_event_stream
from ..text import get_system_message_text

__all__ = [
    "FauxContentBlock",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "faux_assistant_message",
    "FauxModelDefinition",
    "FauxProviderState",
    "FauxResponseFactory",
    "FauxResponseStep",
    "RegisterFauxProviderOptions",
    "FauxProviderRegistration",
    "FauxProviderHandle",
    "create_faux_core",
    "faux_provider",
]

DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"
DEFAULT_MIN_TOKEN_SIZE = 3
DEFAULT_MAX_TOKEN_SIZE = 5


DEFAULT_USAGE = Usage(cost=Cost())


FauxContentBlock = Union[TextContent, ThinkingContent, ToolCall]


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


class FauxToolCallOptions(TypedDict, total=False):
    id: str


class FauxAssistantMessageOptions(TypedDict, total=False):
    stopReason: str
    deferred: DeferredHandle
    errorMessage: str
    responseId: str
    timestamp: int


class FauxDeferredOptions(TypedDict, total=False):
    pendingFetches: int
    pollAfterMs: int


class FauxTokenSizeOptions(TypedDict, total=False):
    min: int
    max: int


def faux_tool_call(name: str, arguments: JsonObject, options: FauxToolCallOptions | None = None) -> ToolCall:
    options = options or {}
    return ToolCall(
        id=options["id"] if options.get("id") is not None else _random_id("tool"),
        name=name,
        arguments=arguments,
    )


def _normalize_faux_assistant_content(
    content: Union[str, FauxContentBlock, Sequence[FauxContentBlock]]
) -> List[FauxContentBlock]:
    if isinstance(content, str):
        return [faux_text(content)]
    if isinstance(content, list):
        return content
    if isinstance(content, tuple):
        return list(content)
    return [cast(FauxContentBlock, content)]


def faux_assistant_message(
    content: Union[str, FauxContentBlock, Sequence[FauxContentBlock]],
    options: FauxAssistantMessageOptions | None = None,
) -> AssistantMessage:
    options = options or {}
    return AssistantMessage(
        content=_normalize_faux_assistant_content(content),
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=DEFAULT_USAGE,
        stop_reason=options.get("stopReason") if options.get("stopReason") is not None else "stop",
        deferred=options.get("deferred"),
        error_message=options.get("errorMessage"),
        response_id=options.get("responseId"),
        timestamp=options.get("timestamp") if options.get("timestamp") is not None else time.time_ns() // 1_000_000,
    )


class FauxModelDefinition:
    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        reasoning: bool = False,
        input: list[Literal["text", "image"]] | None = None,
        cost: Optional[Dict[str, float]] = None,
        context_window: int = 128000,
        max_tokens: int = 16384,
    ) -> None:
        self.id = id
        self.name = name
        self.reasoning = reasoning
        self.input = input
        self.cost = cost
        self.context_window = context_window
        self.max_tokens = max_tokens


class FauxProviderState:
    def __init__(self) -> None:
        self.call_count = 0
        self.deferred_fetch_count = 0
        self.cancelled_deferred: List[DeferredHandle] = []


FauxResponseFactory = Callable[
    [TranscriptContext, Optional[SimpleStreamOptions], FauxProviderState, Model],
    Union[AssistantMessage, Awaitable[AssistantMessage]],
]
FauxResponseStep = Union[AssistantMessage, FauxResponseFactory]


class RegisterFauxProviderOptions:
    def __init__(
        self,
        api: Optional[str] = None,
        provider: Optional[str] = None,
        models: Optional[List[FauxModelDefinition]] = None,
        deferred: FauxDeferredOptions | None = None,
        tokens_per_second: Optional[float] = None,
        token_size: FauxTokenSizeOptions | None = None,
    ) -> None:
        self.api = api
        self.provider = provider
        self.models = models
        self.deferred = deferred
        self.tokens_per_second = tokens_per_second
        self.token_size = token_size


class FauxProviderHandle:
    def __init__(self) -> None:
        self.api: str = ""
        self.models: List[Model] = []
        self.state = FauxProviderState()
        self._pending_responses: List[FauxResponseStep] = []
        self.provider: Provider | None = None

    @overload
    def get_model(self, model_id: None = None) -> Model: ...

    @overload
    def get_model(self, model_id: str) -> Model | None: ...

    def get_model(self, model_id: Optional[str] = None) -> Model | None:
        if not model_id:
            return self.models[0]
        return next((m for m in self.models if m.id == model_id), None)

    def set_responses(self, responses: List[FauxResponseStep]) -> None:
        self._pending_responses = list(responses)

    def append_responses(self, responses: List[FauxResponseStep]) -> None:
        self._pending_responses.extend(responses)

    def get_pending_response_count(self) -> int:
        return len(self._pending_responses)

    def _shift_response(self) -> Optional[FauxResponseStep]:
        return self._pending_responses.pop(0) if self._pending_responses else None


class FauxProviderRegistration(FauxProviderHandle):
    unregister: Callable[[], None]


def _estimate_tokens(text: str) -> int:
    return math.ceil(utf16_length(text) / 4)


def _random_id(prefix: str) -> str:
    timestamp = time.time_ns() // 1_000_000
    return f"{prefix}:{timestamp}:{random_base36()}"


def _content_to_text(content: Union[str, Sequence[Union[TextContent, ImageContent]]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        block.text if isinstance(block, TextContent) else f"[image:{block.mime_type}:{utf16_length(block.data)}]"
        for block in content
    )


def _assistant_content_to_text(content: Sequence[FauxContentBlock]) -> str:
    return "\n".join(
        block.text
        if getattr(block, "type", None) == "text"
        else block.thinking
        if getattr(block, "type", None) == "thinking"
        else f"{block.name}:{javascript_json_stringify(block.arguments)}"
        for block in content
    )


def _tool_result_to_text(message: ToolResultMessage) -> str:
    parts = [message.tool_name] + [_content_to_text([block]) for block in message.content]
    return "\n".join(parts)


def tool_to_json(tool: Tool | Mapping[str, object]) -> dict[str, object]:
    """The wire shape of one tool declaration; TS tools are plain objects."""
    if isinstance(tool, Tool):
        return cast(dict[str, object], tool.to_json())
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters"),
        **(
            {}
            if tool.get("constrainedSampling") is None
            and tool.get("constrained_sampling") is None
            else {
                "constrainedSampling": tool.get(
                    "constrainedSampling", tool.get("constrained_sampling")
                )
            }
        ),
    }


def _message_to_text(message: Message) -> str:
    role = getattr(message, "role", None)
    if role == "system":
        parts = [get_system_message_text(message)]
        parts.extend(f"tool-:{javascript_json_stringify(t.to_json())}" for t in message.tools_removed or [])
        parts.extend(
            f"tool+:{javascript_json_stringify(tool_to_json(t))}" for t in message.tools_added or []
        )
        return "\n".join(p for p in parts if p)
    if role == "user":
        return _content_to_text(message.content)
    if role == "assistant":
        return _assistant_content_to_text(message.content)
    return _tool_result_to_text(message)


def _serialize_context(context: TranscriptContext) -> str:
    return "\n\n".join(f"{getattr(m, 'role', '?')}:{_message_to_text(m)}" for m in context.messages)


def _common_prefix_length(a: str, b: str) -> int:
    a_units, b_units = utf16_units(a), utf16_units(b)
    length = min(len(a_units), len(b_units))
    index = 0
    while index < length and a_units[index] == b_units[index]:
        index += 1
    return index


def _slice_utf16(text: str, start: int, end: int | None = None) -> str:
    return text.encode("utf-16-le", "surrogatepass")[start * 2:end * 2 if end is not None else None].decode("utf-16-le", "surrogatepass")


def _with_usage_estimate(
    message: AssistantMessage,
    context: TranscriptContext,
    options: StreamOptions | None,
    prompt_cache: Dict[str, str],
) -> AssistantMessage:
    prompt_text = _serialize_context(context)
    prompt_tokens = _estimate_tokens(prompt_text)
    output_tokens = _estimate_tokens(_assistant_content_to_text(message.content))
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = options.session_id if options else None

    if session_id and (options.cache_retention if options else "short") != "none":
        previous_prompt = prompt_cache.get(session_id)
        if previous_prompt:
            cached_chars = _common_prefix_length(previous_prompt, prompt_text)
            cache_read = _estimate_tokens(_slice_utf16(previous_prompt, 0, cached_chars))
            cache_write = _estimate_tokens(_slice_utf16(prompt_text, cached_chars))
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt_text

    result = copy.copy(message)
    result.usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        cost=Cost(),
    )
    return result


def _split_string_by_token_size(text: str, min_token_size: int, max_token_size: int) -> List[str]:
    chunks: List[str] = []
    index = 0
    while index < utf16_length(text):
        token_size = min_token_size + math.floor(random.random() * (max_token_size - min_token_size + 1))
        char_size = max(1, token_size * 4)
        chunks.append(_slice_utf16(text, index, index + char_size))
        index += char_size
    return chunks if chunks else [""]


def _clone_message(message: AssistantMessage, api: str, provider: str, model_id: str) -> AssistantMessage:
    cloned = copy.deepcopy(message)
    cloned.api = api
    cloned.provider = provider
    cloned.model = model_id
    if cloned.timestamp is None:
        cloned.timestamp = time.time_ns() // 1_000_000
    if cloned.usage is None:
        cloned.usage = DEFAULT_USAGE
    return cloned


def _create_deferred_message(model: Model, handle: DeferredHandle) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=DEFAULT_USAGE,
        stop_reason="deferred",
        deferred=handle,
        timestamp=int(time.time() * 1000),
    )


def _create_error_message(error: BaseException, api: str, provider: str, model_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=DEFAULT_USAGE,
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


def _create_aborted_message(partial: AssistantMessage) -> AssistantMessage:
    aborted = copy.copy(partial)
    aborted.stop_reason = "aborted"
    aborted.error_message = "Request was aborted"
    aborted.timestamp = int(time.time() * 1000)
    return aborted


async def _schedule_chunk(chunk: str, tokens_per_second: Optional[float]) -> None:
    if not tokens_per_second or math.isnan(tokens_per_second) or tokens_per_second <= 0:
        await asyncio.sleep(0)
        return
    delay_ms = (_estimate_tokens(chunk) / tokens_per_second) * 1000
    delay = math.trunc(delay_ms) if math.isfinite(delay_ms) and 1 <= delay_ms <= 2_147_483_647 else 1
    await asyncio.sleep(delay / 1000)


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: Optional[float],
    signal: AbortSignal | None,
) -> None:
    partial = copy.copy(message)
    partial.content = []
    partial.stop_reason = "pending"

    def _aborted() -> bool:
        return signal is not None and signal.aborted

    if _aborted():
        aborted = _create_aborted_message(partial)
        stream.push(AssistantMessageEvent(type="error", reason="aborted", error=aborted))
        stream.end(aborted)
        return

    stream.push(AssistantMessageEvent(type="start", partial=copy.copy(partial)))

    for index, block in enumerate(message.content):
        if _aborted():
            aborted = _create_aborted_message(partial)
            stream.push(AssistantMessageEvent(type="error", reason="aborted", error=aborted))
            stream.end(aborted)
            return

        if isinstance(block, ThinkingContent):
            partial.content = [*partial.content, ThinkingContent(thinking="")]
            stream.push(
                AssistantMessageEvent(type="thinking_start", content_index=index, partial=copy.copy(partial))
            )
            for chunk in _split_string_by_token_size(block.thinking, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if _aborted():
                    aborted = _create_aborted_message(partial)
                    stream.push(AssistantMessageEvent(type="error", reason="aborted", error=aborted))
                    stream.end(aborted)
                    return
                current_thinking = cast(ThinkingContent, partial.content[index])
                current_thinking.thinking = (current_thinking.thinking + chunk).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
                stream.push(
                    AssistantMessageEvent(
                        type="thinking_delta", content_index=index, delta=chunk, partial=copy.copy(partial)
                    )
                )
            stream.push(
                AssistantMessageEvent(
                    type="thinking_end", content_index=index, content=block.thinking, partial=copy.copy(partial)
                )
            )
            continue

        if isinstance(block, TextContent):
            partial.content = [*partial.content, TextContent(text="")]
            stream.push(AssistantMessageEvent(type="text_start", content_index=index, partial=copy.copy(partial)))
            for chunk in _split_string_by_token_size(block.text, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if _aborted():
                    aborted = _create_aborted_message(partial)
                    stream.push(AssistantMessageEvent(type="error", reason="aborted", error=aborted))
                    stream.end(aborted)
                    return
                current_text = cast(TextContent, partial.content[index])
                current_text.text = (current_text.text + chunk).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
                stream.push(
                    AssistantMessageEvent(
                        type="text_delta", content_index=index, delta=chunk, partial=copy.copy(partial)
                    )
                )
            stream.push(
                AssistantMessageEvent(
                    type="text_end", content_index=index, content=block.text, partial=copy.copy(partial)
                )
            )
            continue

        # Tool call block
        partial.content = [*partial.content, ToolCall(id=block.id, name=block.name, arguments={})]
        stream.push(AssistantMessageEvent(type="toolcall_start", content_index=index, partial=copy.copy(partial)))
        for chunk in _split_string_by_token_size(cast(str, javascript_json_stringify(block.arguments)), min_token_size, max_token_size):
            await _schedule_chunk(chunk, tokens_per_second)
            if _aborted():
                aborted = _create_aborted_message(partial)
                stream.push(AssistantMessageEvent(type="error", reason="aborted", error=aborted))
                stream.end(aborted)
                return
            stream.push(
                AssistantMessageEvent(
                    type="toolcall_delta", content_index=index, delta=chunk, partial=copy.copy(partial)
                )
            )
        cast(ToolCall, partial.content[index]).arguments = block.arguments
        stream.push(
            AssistantMessageEvent(
                type="toolcall_end", content_index=index, tool_call=block, partial=copy.copy(partial)
            )
        )

    if message.stop_reason == "pending":
        raise RuntimeError("Faux response ended without a stop reason")
    if message.stop_reason in ("error", "aborted"):
        stream.push(AssistantMessageEvent(type="error", reason=message.stop_reason, error=message))
        stream.end(message)
        return

    stream.push(AssistantMessageEvent(type="done", reason=message.stop_reason, message=message))
    stream.end(message)


class _DeferredEntry:
    def __init__(
        self,
        handle: DeferredHandle,
        step: FauxResponseStep,
        context: TranscriptContext,
        options: StreamOptions | None,
        model: Model,
        pending_fetches: int,
    ) -> None:
        self.handle = handle
        self.step = step
        self.context = context
        self.options = options
        self.model = model
        self.pending_fetches = pending_fetches
        self.cancelled = False
        self.final: Optional[AssistantMessage] = None


class FauxStreams(TypedDict):
    stream: StreamFunction
    stream_simple: SimpleStreamFunction
    fetch_deferred: DeferredFetchFunction
    cancel_deferred: DeferredCancelFunction


async def _notify_response(options: ProviderRequestOptions[Model] | None, model: Model) -> None:
    result = options.on_response(ProviderResponse(status=200, headers={}), model) if options is not None and options.on_response is not None else None
    if inspect.isawaitable(result):
        await result
    else:
        await asyncio.sleep(0)


def create_faux_core(options: RegisterFauxProviderOptions) -> tuple[FauxProviderHandle, FauxStreams]:
    """Create a faux provider core.

    Returns the provider handle and its stream functions as a dict with
    ``stream``, ``stream_simple``, ``fetch_deferred``, and ``cancel_deferred``.
    """
    api = options.api if options.api is not None else _random_id(DEFAULT_API)
    provider = options.provider if options.provider is not None else DEFAULT_PROVIDER
    token_size = options.token_size or {}
    requested_min = token_size.get("min")
    requested_max = token_size.get("max")
    requested_min = DEFAULT_MIN_TOKEN_SIZE if requested_min is None else requested_min
    requested_max = DEFAULT_MAX_TOKEN_SIZE if requested_max is None else requested_max
    min_token_size = max(1, min(requested_min, requested_max))
    max_token_size = max(min_token_size, requested_max)
    tokens_per_second = options.tokens_per_second
    prompt_cache: Dict[str, str] = {}
    deferred_responses: Dict[str, _DeferredEntry] = {}

    handle = FauxProviderHandle()
    handle.api = api
    handle.state = FauxProviderState()

    model_definitions = options.models or [
        FauxModelDefinition(
            id=DEFAULT_MODEL_ID,
            name=DEFAULT_MODEL_NAME,
            reasoning=False,
            input=["text", "image"],
            cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            context_window=128000,
            max_tokens=16384,
        )
    ]
    handle.models = []
    for definition in model_definitions:
        handle.models.append(
            model_from_json({
                "id": definition.id,
                "name": definition.name if definition.name is not None else definition.id,
                "api": api,
                "provider": provider,
                "baseUrl": DEFAULT_BASE_URL,
                "reasoning": definition.reasoning if definition.reasoning is not None else False,
                "input": definition.input if definition.input is not None else ["text", "image"],
                "cost": definition.cost if definition.cost is not None else {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": definition.context_window if definition.context_window is not None else 128000,
                "maxTokens": definition.max_tokens if definition.max_tokens is not None else 16384,
            })
        )

    async def _resolve_response(
        step: FauxResponseStep,
        context: TranscriptContext,
        stream_options: StreamOptions | None,
        request_model: Model,
    ) -> AssistantMessage:
        if callable(step):
            resolved = step(context, cast(SimpleStreamOptions | None, stream_options), handle.state, request_model)
            if inspect.isawaitable(resolved):
                resolved = await resolved
        else:
            resolved = step
        return _with_usage_estimate(
            _clone_message(resolved, api, provider, request_model.id),
            context,
            stream_options,
            prompt_cache,
        )

    def stream(request_model: Model, context: TranscriptContext, stream_options: StreamOptions | None = None) -> AssistantMessageEventStream:
        outer = create_assistant_message_event_stream()
        step = handle._shift_response()
        handle.state.call_count += 1

        async def _run() -> None:
            try:
                await _notify_response(stream_options, request_model)
                if step is None:
                    message = _create_error_message(
                        RuntimeError("No more faux responses queued"), api, provider, request_model.id
                    )
                    message = _with_usage_estimate(message, context, stream_options, prompt_cache)
                    outer.push(AssistantMessageEvent(type="error", reason="error", error=message))
                    outer.end(message)
                    return

                deferred = getattr(stream_options, "deferred", None)
                if deferred is not None and deferred is not False:
                    deferred_options = options.deferred or {}
                    deferred_handle = DeferredHandle(
                        provider=request_model.provider,
                        model_id=request_model.id,
                        api=request_model.api,
                        id=_random_id("deferred"),
                        poll_after_ms=deferred_options.get("pollAfterMs"),
                    )
                    deferred_responses[deferred_handle.id] = _DeferredEntry(
                        handle=deferred_handle,
                        step=step,
                        context=context,
                        options=stream_options,
                        model=request_model,
                        pending_fetches=max(0, math.floor(deferred_options.get("pendingFetches", 0) or 0)),
                    )
                    await _stream_with_deltas(
                        outer,
                        _create_deferred_message(request_model, deferred_handle),
                        min_token_size,
                        max_token_size,
                        tokens_per_second,
                        stream_options.signal if stream_options else None,
                    )
                    return

                message = await _resolve_response(step, context, stream_options, request_model)
                await _stream_with_deltas(
                    outer,
                    message,
                    min_token_size,
                    max_token_size,
                    tokens_per_second,
                    stream_options.signal if stream_options else None,
                )
            except BaseException as error:  # noqa: BLE001 - mirrors TS catch
                message = _create_error_message(error, api, provider, request_model.id)
                outer.push(AssistantMessageEvent(type="error", reason="error", error=message))
                outer.end(message)

        asyncio.get_running_loop().create_task(_run())
        return outer

    def stream_simple(
        stream_model: Model, context: TranscriptContext, stream_options: Optional[SimpleStreamOptions] = None
    ) -> AssistantMessageEventStream:
        return stream(stream_model, context, stream_options)

    def fetch_deferred(
        request_model: Model,
        deferred_handle: DeferredHandle,
        fetch_options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        outer = create_assistant_message_event_stream()
        handle.state.deferred_fetch_count += 1

        async def _run() -> None:
            try:
                await _notify_response(fetch_options, request_model)
                entry = deferred_responses.get(deferred_handle.id)
                if (
                    entry is None
                    or entry.handle.provider != deferred_handle.provider
                    or entry.handle.model_id != deferred_handle.model_id
                    or entry.handle.api != deferred_handle.api
                ):
                    raise RuntimeError(f"Unknown faux deferred response: {deferred_handle.id}")
                if entry.cancelled:
                    raise RuntimeError(f"Faux deferred response was cancelled: {deferred_handle.id}")

                if entry.pending_fetches > 0:
                    entry.pending_fetches -= 1
                    await _stream_with_deltas(
                        outer,
                        _create_deferred_message(request_model, entry.handle),
                        min_token_size,
                        max_token_size,
                        tokens_per_second,
                        getattr(fetch_options, "signal", None),
                    )
                    return

                if entry.final is None:
                    submission_options = copy.copy(entry.options) if entry.options is not None else SimpleStreamOptions()
                    submission_options.deferred = None
                    submission_options.signal = None
                    submission_options.on_response = None
                    try:
                        entry.final = await _resolve_response(entry.step, entry.context, submission_options, entry.model)
                    except BaseException as error:  # noqa: BLE001
                        entry.final = _create_error_message(error, api, provider, entry.model.id)

                await _stream_with_deltas(
                    outer, entry.final, min_token_size, max_token_size, tokens_per_second,
                    fetch_options.signal if fetch_options is not None else None,
                )
            except BaseException as error:  # noqa: BLE001
                message = _create_error_message(error, api, provider, request_model.id)
                outer.push(AssistantMessageEvent(type="error", reason="error", error=message))
                outer.end(message)

        asyncio.get_running_loop().create_task(_run())
        return outer

    async def cancel_deferred(
        request_model: Model,
        deferred_handle: DeferredHandle,
        cancel_options: DeferredCancelOptions | None = None,
    ) -> None:
        handle.state.cancelled_deferred.append(copy.deepcopy(deferred_handle))
        entry = deferred_responses.get(deferred_handle.id)
        if entry is not None:
            entry.cancelled = True
        await _notify_response(cancel_options, request_model)

    return handle, {
        "stream": stream,
        "stream_simple": stream_simple,
        "fetch_deferred": fetch_deferred,
        "cancel_deferred": cancel_deferred,
    }


def faux_provider(options: RegisterFauxProviderOptions | None = None) -> FauxProviderHandle:
    handle, streams = create_faux_core(options if options is not None else RegisterFauxProviderOptions())

    async def resolve_auth(_input: ApiKeyAuthInput) -> AuthResult:
        return AuthResult(auth=ModelAuth())

    handle.provider = create_provider(CreateProviderOptions(
        id=handle.models[0].provider,
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Faux", resolve=resolve_auth)),
        models=handle.models,
        api=ProviderStreams(**streams),
    ))
    return handle
