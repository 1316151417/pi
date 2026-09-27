"""Deferred Anthropic Messages provider streams."""

from ..types import ProviderStreams
from . import anthropic_messages
from .lazy import lazy_api


def anthropic_messages_api() -> ProviderStreams:
    async def load() -> ProviderStreams:
        return ProviderStreams(stream=anthropic_messages.stream, stream_simple=anthropic_messages.stream_simple)

    return lazy_api(load)


__all__ = ["anthropic_messages_api"]
