"""Assistant stream consumption ported from ``harness/execution/assistant.ts``.

Provides the process-local assistant stream observer contract and the two
stream drivers: one that consumes an already-created event stream, and one that
builds the request from a harness stream configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Protocol

from pi_ai.types import (
    AssistantMessage,
    Context as AiContext,
)
from ..types import AgentHarnessStreamOptions
from ..context import Context, get_telemetry_context
from .effect_gate import AbortRequested

__all__ = [
    "AssistantResponseMetadata",
    "AssistantStreamObserver",
    "HarnessAssistantStreamConfig",
    "consume_assistant_stream",
    "stream_harness_assistant",
]


@dataclass
class AssistantResponseMetadata:
    """HTTP response metadata captured before the provider body is consumed."""

    status: Optional[int] = None
    headers: Optional[dict] = None


class AssistantStreamObserver(Protocol):
    """Process-local lifecycle observer for one assistant stream."""

    async def start(self, message: AssistantMessage, event: Any, context: Context) -> None: ...

    async def update(self, message: AssistantMessage, event: Any, context: Context) -> None: ...

    async def end(self, message: AssistantMessage, context: Context) -> None: ...


@dataclass
class HarnessAssistantStreamConfig:
    """Executable inputs for one already-approved assistant provider request.

    The callables mirror the TS call signatures; ``metadata`` is captured into
    ``metadata_box`` because Python cannot hand out a rebindable closure slot.
    """

    model: Any
    system_prompt: str
    thinking_level: str
    stream_options: Any
    to_provider_messages: Callable[..., Any]
    request: Callable[..., Any]
    observer: Any
    tools: Optional[List[Any]] = None
    transform_context: Optional[Callable[..., Any]] = None
    before_payload: Optional[Callable[..., Any]] = None
    after_response: Optional[Callable[..., Any]] = None
    metadata_box: dict = field(default_factory=dict)

    @property
    def metadata(self) -> AssistantResponseMetadata:
        return self.metadata_box.get("metadata") or AssistantResponseMetadata()


def _is_update_event(event: Any) -> bool:
    return event.type not in ("start", "done", "error")


def _create_request_options(
    config: HarnessAssistantStreamConfig,
    capture_metadata: Callable[[AssistantResponseMetadata], None],
    context: Context,
) -> dict:
    options = config.stream_options
    resolved: dict = {
        "transport": _opt(options, "transport"),
        "timeoutMs": _opt(options, "timeout_ms", "timeoutMs"),
        "maxRetries": _opt(options, "max_retries", "maxRetries"),
        "maxRetryDelayMs": _opt(options, "max_retry_delay_ms", "maxRetryDelayMs"),
        "headers": _opt(options, "headers"),
        "metadata": _opt(options, "metadata"),
        "cacheRetention": _opt(options, "cache_retention", "cacheRetention"),
        "deferred": _opt(options, "deferred"),
        "signal": context.signal,
        "telemetryContext": get_telemetry_context(context),
    }
    if config.thinking_level != "off":
        resolved["reasoning"] = config.thinking_level
    if config.before_payload is not None:

        async def _on_payload(payload: Any, model: Any) -> Any:
            return await _maybe_await(config.before_payload(payload, model, context))

        resolved["onPayload"] = _on_payload

    async def _on_response(response: Any, _model: Any = None) -> None:
        capture_metadata(
            AssistantResponseMetadata(
                status=_opt(response, "status"), headers=_opt(response, "headers")
            )
        )

    resolved["onResponse"] = _on_response
    return resolved


async def consume_assistant_stream(
    stream: Any,
    observer: Any,
    after_response: Optional[Callable[..., Any]],
    context: Context,
) -> Any:
    """Consume one assistant event stream, notifying the observer per event."""
    started = False
    async for event in stream:
        if event.type == "start":
            if started:
                raise RuntimeError(
                    "Assistant message stream emitted more than one start event"
                )
            started = True
            await _maybe_await(observer.start(_partial(event), event, context))
        elif _is_update_event(event):
            if not started:
                raise RuntimeError(
                    f"Assistant message stream emitted {event.type} before start"
                )
            await _maybe_await(observer.update(_partial(event), event, context))
        elif event.type == "done" and not started:
            raise RuntimeError("Assistant message stream emitted done before start")

    settled = await stream.result
    final_message = settled
    if after_response is not None:
        try:
            final_message = await _maybe_await(after_response(settled, context))
        except AbortRequested as error:
            await error.cancellation
    await _maybe_await(observer.end(final_message, context))
    return final_message


async def stream_harness_assistant(
    messages: List[Any],
    config: HarnessAssistantStreamConfig,
    context: Context,
) -> Any:
    """Stream one assistant response without mutating the caller's message list."""
    request_context = {"messages": list(messages), "systemPrompt": config.system_prompt}
    if config.transform_context is not None:
        request_context = await _maybe_await(config.transform_context(request_context, context))

    provider_messages = await _maybe_await(
        config.to_provider_messages(request_context["messages"], context)
    )
    ai_context = AiContext(
        system_prompt=request_context["systemPrompt"],
        messages=provider_messages,
        tools=config.tools,
    )

    def _capture(metadata: AssistantResponseMetadata) -> None:
        config.metadata_box["metadata"] = metadata

    stream = await _maybe_await(
        config.request(
            ai_context,
            _create_request_options(config, _capture, context),
            context,
        )
    )
    after_response = config.after_response
    wrapped = None
    if after_response is not None:

        async def _wrapped(message: Any, after_context: Context) -> Any:
            return await _maybe_await(
                after_response(message, config.metadata, after_context)
            )

        wrapped = _wrapped
    return await consume_assistant_stream(stream, config.observer, wrapped, context)


def _partial(event: Any) -> Any:
    return event.partial


async def _maybe_await(value: Any) -> Any:
    import asyncio

    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping or an attribute."""
    if value is None:
        return None
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
