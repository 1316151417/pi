"""Legacy stream aliases for the representative native API implementations."""

from .api.anthropic_messages_lazy import anthropic_messages_api
from .api.azure_openai_responses_lazy import azure_openai_responses_api
from .api.google_generative_ai_lazy import google_generative_ai_api
from .api.openai_completions_lazy import openai_completions_api
from .api.openai_responses_lazy import openai_responses_api

_anthropic = anthropic_messages_api()
_azure = azure_openai_responses_api()
_google = google_generative_ai_api()
_completions = openai_completions_api()
_responses = openai_responses_api()

stream_anthropic = _anthropic.stream
stream_simple_anthropic = _anthropic.stream_simple
stream_azure_openai_responses = _azure.stream
stream_simple_azure_openai_responses = _azure.stream_simple
stream_google = _google.stream
stream_simple_google = _google.stream_simple
stream_openai_completions = _completions.stream
stream_simple_openai_completions = _completions.stream_simple
stream_openai_responses = _responses.stream
stream_simple_openai_responses = _responses.stream_simple

__all__ = [
    "stream_anthropic", "stream_simple_anthropic", "stream_azure_openai_responses",
    "stream_simple_azure_openai_responses", "stream_google", "stream_simple_google",
    "stream_openai_completions", "stream_simple_openai_completions", "stream_openai_responses",
    "stream_simple_openai_responses",
]
