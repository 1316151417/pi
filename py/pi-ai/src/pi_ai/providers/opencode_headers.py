"""OpenCode's optional per-conversation routing header."""

from __future__ import annotations

from copy import copy

from ..event_stream import AssistantMessageEventStream
from ..types import Model, ProviderStreams, SimpleStreamOptions, StreamOptions, TranscriptContext

OPENCODE_SESSION_HEADER = "x-opencode-session"


def _with_session_header[TOptions: StreamOptions](options: TOptions | None) -> TOptions | None:
    if options is None or not options.session_id:
        return options
    if any(key.lower() == OPENCODE_SESSION_HEADER for key in (options.headers or {})):
        return options
    resolved = copy(options)
    resolved.headers = {**(options.headers or {}), OPENCODE_SESSION_HEADER: options.session_id}
    return resolved


def with_open_code_session_header(streams: ProviderStreams) -> ProviderStreams:
    def stream(model: Model, context: TranscriptContext, options: StreamOptions | None = None) -> AssistantMessageEventStream:
        return streams.stream(model, context, _with_session_header(options))

    def stream_simple(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None) -> AssistantMessageEventStream:
        return streams.stream_simple(model, context, _with_session_header(options))

    wrapped = copy(streams)
    wrapped.stream = stream
    wrapped.stream_simple = stream_simple
    return wrapped


__all__ = ["with_open_code_session_header"]
