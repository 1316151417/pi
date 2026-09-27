"""Chat Completions factory with asynchronous request setup."""

from ..types import ProviderStreams
from .lazy import lazy_api
from .openai_completions import stream, stream_simple


def openai_completions_api() -> ProviderStreams:
    async def load() -> ProviderStreams:
        return ProviderStreams(stream=stream, stream_simple=stream_simple)

    return lazy_api(load)


__all__ = ["openai_completions_api"]
