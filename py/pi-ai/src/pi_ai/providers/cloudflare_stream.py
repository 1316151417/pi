"""Materialize account/gateway placeholders immediately before dispatch."""

from __future__ import annotations

from copy import copy

from ..event_stream import AssistantMessageEventStream
from ..types import Model, ProviderEnv, ProviderStreams, SimpleStreamOptions, StreamOptions, TranscriptContext


def resolve_cloudflare_model(model: Model, env: ProviderEnv | None) -> Model:
    if env is None:
        return model
    base_url = model.base_url
    for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID"):
        value = env.get(name)
        base_url = base_url.replace("{" + name + "}", value if value is not None else "{" + name + "}")
    if base_url == model.base_url:
        return model
    resolved = copy(model)
    resolved.base_url = base_url
    return resolved


def cloudflare_streams(streams: ProviderStreams) -> ProviderStreams:
    def stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
        return streams.stream(resolve_cloudflare_model(model, options.env if options is not None else None), context, options)

    def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
        return streams.stream_simple(resolve_cloudflare_model(model, options.env if options is not None else None), context, options)

    return ProviderStreams(stream=stream, stream_simple=stream_simple)


__all__ = ["resolve_cloudflare_model", "cloudflare_streams"]
