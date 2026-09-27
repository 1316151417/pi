"""Synchronous stream creation around asynchronous setup from ``api/lazy.ts``."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from ..event_stream import AssistantMessageEventStream
from ..types import (
    AssistantMessage, AssistantMessageEvent, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle,
    Model, ProviderStreams, SimpleStreamOptions, StreamOptions, TranscriptContext, Usage,
)


def lazy_stream(
    model: Model, setup: Callable[[], Awaitable[AsyncIterable[AssistantMessageEvent]]],
) -> AssistantMessageEventStream:
    """Return immediately; forward setup, iteration and final-result failures."""
    outer = AssistantMessageEventStream()
    pending = setup()

    async def forward() -> None:
        try:
            source = await pending
            async for event in source:
                outer.push(event)
            if isinstance(source, AssistantMessageEventStream):
                outer.end(await source.result)
            else:
                result: object = getattr(source, "result", None)
                if callable(result):
                    value: object = result()
                    if inspect.isawaitable(value):
                        value = await value
                    outer.end(cast(AssistantMessage, value))
                else:
                    outer.end()
        except (Exception, asyncio.CancelledError) as error:
            message = AssistantMessage(
                content=[], api=model.api, provider=model.provider, model=model.id,
                usage=Usage(), stop_reason="error", error_message=str(error), timestamp=time.time_ns() // 1_000_000,
            )
            outer.push(AssistantMessageEvent(type="error", reason="error", error=message))
            outer.end(message)

    asyncio.get_running_loop().create_task(forward())
    return outer


@dataclass
class LazyApiCapabilities:
    fetch_deferred: bool | None = None
    cancel_deferred: bool | None = None


def lazy_api(
    load: Callable[[], Awaitable[ProviderStreams]], capabilities: LazyApiCapabilities | None = None,
) -> ProviderStreams:
    def stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
        async def setup() -> AsyncIterable[AssistantMessageEvent]:
            return (await load()).stream(model, context, options)

        return lazy_stream(model, setup)

    def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
        async def setup() -> AsyncIterable[AssistantMessageEvent]:
            return (await load()).stream_simple(model, context, options)

        return lazy_stream(model, setup)

    api = ProviderStreams(stream=stream, stream_simple=stream_simple)
    if capabilities is not None and capabilities.fetch_deferred:
        def fetch_deferred(model: Model, handle: DeferredHandle, options: DeferredFetchOptions | None = None) -> AssistantMessageEventStream:
            async def setup() -> AsyncIterable[AssistantMessageEvent]:
                implementation = await load()
                if implementation.fetch_deferred is None:
                    raise RuntimeError("API does not support deferred responses")
                return implementation.fetch_deferred(model, handle, options)

            return lazy_stream(model, setup)

        api.fetch_deferred = fetch_deferred
    if capabilities is not None and capabilities.cancel_deferred:
        async def cancel_deferred(model: Model, handle: DeferredHandle, options: DeferredCancelOptions | None = None) -> None:
            implementation = await load()
            if implementation.cancel_deferred is None:
                raise RuntimeError("API cannot cancel deferred responses")
            await implementation.cancel_deferred(model, handle, options)

        api.cancel_deferred = cancel_deferred
    return api


__all__ = ["LazyApiCapabilities", "lazy_api", "lazy_stream"]
