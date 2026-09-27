"""Factory corresponding to ``api/google-generative-ai.lazy.ts``."""

from ..types import ProviderStreams
from . import google_generative_ai
from .lazy import lazy_api


def google_generative_ai_api() -> ProviderStreams:
    async def load() -> ProviderStreams:
        return ProviderStreams(stream=google_generative_ai.stream, stream_simple=google_generative_ai.stream_simple)

    return lazy_api(load)


__all__ = ["google_generative_ai_api"]
