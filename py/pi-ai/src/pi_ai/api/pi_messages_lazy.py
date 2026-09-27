"""Deferred pi-messages provider streams."""

from ..types import ProviderStreams
from . import pi_messages
from .lazy import lazy_api


def pi_messages_api() -> ProviderStreams:
    async def load() -> ProviderStreams:
        return ProviderStreams(stream=pi_messages.stream, stream_simple=pi_messages.stream_simple)

    return lazy_api(load)


__all__ = ["pi_messages_api"]
