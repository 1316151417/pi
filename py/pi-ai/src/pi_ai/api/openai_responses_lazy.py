"""Factory corresponding to ``api/openai-responses.lazy.ts``."""

from ..types import ProviderStreams
from . import openai_responses
from .lazy import lazy_api


def openai_responses_api() -> ProviderStreams:
    async def load() -> ProviderStreams:
        return ProviderStreams(stream=openai_responses.stream, stream_simple=openai_responses.stream_simple)

    return lazy_api(load)


__all__ = ["openai_responses_api"]
