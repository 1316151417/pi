"""Factory corresponding to ``api/azure-openai-responses.lazy.ts``."""

from ..types import ProviderStreams
from . import azure_openai_responses
from .lazy import lazy_api


def azure_openai_responses_api() -> ProviderStreams:
    async def load() -> ProviderStreams:
        return ProviderStreams(stream=azure_openai_responses.stream, stream_simple=azure_openai_responses.stream_simple)

    return lazy_api(load)


__all__ = ["azure_openai_responses_api"]
