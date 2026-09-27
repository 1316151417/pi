"""Context-overflow detection ported from pi-ai ``src/utils/overflow.ts``.

The source patterns use ASCII case folding and digits, JavaScript whitespace,
and JavaScript's four line terminators for dot matching. Pattern searches retain
the source's unanchored ``RegExp.prototype.test`` behavior.
"""

from __future__ import annotations

import re

from ._json_runtime import JS_WHITESPACE
from .types import AssistantMessage

__all__ = ["is_context_overflow", "is_recoverable_length", "get_overflow_patterns"]

_JS_SPACE = "[" + re.escape(JS_WHITESPACE) + "]"


def _pattern(source: str) -> re.Pattern[str]:
    return re.compile(source.replace(r"\s", _JS_SPACE), re.IGNORECASE | re.ASCII)


#: Regex patterns to detect context overflow errors from different providers.
#:
#: These patterns match error messages returned when the input exceeds the
#: model's context window.
#:
#: Provider-specific patterns (with example error messages):
#:
#: - Anthropic: "prompt is too long: 213462 tokens > 200000 maximum"
#: - Anthropic: "413 {\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}"
#: - OpenAI: "Your input exceeds the context window of this model"
#: - OpenAI/LiteLLM: "Requested token count exceeds the model's maximum context length of 131072 tokens"
#: - OpenAI-compatible: "Input length (265330) exceeds model's maximum context length (262144)."
#: - Google: "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"
#: - xAI: "This model's maximum prompt length is 131072 but the request contains 537812 tokens"
#: - Groq: "Please reduce the length of the messages or completion"
#: - OpenRouter: "This endpoint's maximum context length is X tokens. However, you requested about Y tokens"
#: - OpenRouter/Poolside: "Input length X exceeds the maximum allowed input length of Y tokens."
#: - Together AI: "The input (X tokens) is longer than the model's context length (Y tokens)."
#: - llama.cpp: "the request exceeds the available context size, try increasing it"
#: - LM Studio: "tokens to keep from the initial prompt is greater than the context length"
#: - GitHub Copilot: "prompt token count of X exceeds the limit of Y"
#: - MiniMax: "invalid params, context window exceeds limit"
#: - Kimi For Coding: "Your request exceeded model token limit: X (requested: Y)"
#: - DS4: "Prompt has X tokens, but the configured context size is Y tokens"
#: - Cerebras: "400/413 status code (no body)"
#: - Mistral: "Prompt contains X tokens ... too large for model with Y maximum context length"
#: - z.ai: Does NOT error, accepts overflow silently - handled via usage.input > contextWindow
#: - Xiaomi MiMo: Truncates input to fill contextWindow exactly, then returns finish_reason "length"
#:   with output=0 (no room left to generate). Detected via stopReason "length" + zero output +
#:   input filling the context window.
#: - DashScope/Qwen: "Range of input length should be [1, X]" (HTTP 400 invalid_parameter_error)
#: - Ollama: Some deployments truncate silently, others return errors like "prompt too long; exceeded max context length by X tokens"
OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    _pattern(r"prompt is too long"),  # Anthropic token overflow
    _pattern(r"request_too_large"),  # Anthropic request byte-size overflow (HTTP 413)
    _pattern(r"input is too long for requested model"),  # Amazon Bedrock
    _pattern(r"exceeds the context window"),  # OpenAI (Completions & Responses API)
    _pattern(
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
    ),  # OpenAI-compatible proxies (LiteLLM)
    _pattern(r"input token count[^\r\n\u2028\u2029]*exceeds the maximum"),  # Google (Gemini)
    _pattern(r"maximum prompt length is \d+"),  # xAI (Grok)
    _pattern(r"reduce the length of the messages"),  # Groq
    _pattern(r"maximum context length is \d+ tokens"),  # OpenRouter (most backends)
    _pattern(r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?"),  # OpenRouter/Poolside
    _pattern(r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)"),  # Together AI
    _pattern(r"exceeds the limit of \d+"),  # GitHub Copilot
    _pattern(r"exceeds the available context size"),  # llama.cpp server
    _pattern(r"greater than the context length"),  # LM Studio
    _pattern(r"context window exceeds limit"),  # MiniMax
    _pattern(r"exceeded model token limit"),  # Kimi For Coding
    _pattern(r"too large for model with \d+ maximum context length"),  # Mistral
    _pattern(r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?"),  # DS4 server
    _pattern(r"model_context_window_exceeded"),  # z.ai non-standard finish_reason surfaced as error text
    _pattern(r"prompt too long; exceeded (?:max )?context length"),  # Ollama explicit overflow error
    _pattern(r"range of input length should be"),  # DashScope / Qwen Token Plan
    _pattern(r"context[_ ]length[_ ]exceeded"),  # Generic fallback
    _pattern(r"too many tokens"),  # Generic fallback
    _pattern(r"token limit exceeded"),  # Generic fallback
]

CEREBRAS_BODYLESS_OVERFLOW_PATTERN = _pattern(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)")

#: Patterns that indicate non-overflow errors (e.g. rate limiting, server errors).
#: Error messages matching any of these are excluded from overflow detection
#: even if they also match an ``OVERFLOW_PATTERNS`` entry.
#:
#: Example: Bedrock formats throttling errors as "ThrottlingException: Too many tokens,
#: please wait before trying again." which would match the ``too many tokens`` overflow
#: pattern without this exclusion.
NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    _pattern(r"^(Throttling error|Service unavailable):"),  # AWS Bedrock (formatBedrockError prefixes)
    _pattern(r"rate limit"),  # Generic rate limiting
    _pattern(r"too many requests"),  # Generic HTTP 429 style
]


def is_context_overflow(message: AssistantMessage, context_window: int | float | None = None) -> bool:
    """Check if an assistant message represents a context overflow error.

    This handles three cases:

    1. Error-based overflow: most providers return ``stop_reason`` ``"error"``
       with a specific error message pattern.
    2. Silent overflow: some providers accept overflow requests and return
       successfully. For these, we check if ``usage.input`` exceeds the context
       window.
    3. Length-stop overflow: Xiaomi MiMo can return ``"length"`` with zero output
       when the input fills the context window.

    ``context_window`` is optional and enables silent/length detection.
    """
    error_message = message.error_message
    if message.stop_reason == "error" and error_message:
        # Skip messages matching known non-overflow patterns (throttling / rate-limit).
        is_non_overflow = any(pattern.search(error_message) for pattern in NON_OVERFLOW_PATTERNS)
        if not is_non_overflow:
            if any(pattern.search(error_message) for pattern in OVERFLOW_PATTERNS):
                return True
            if message.provider == "cerebras" and CEREBRAS_BODYLESS_OVERFLOW_PATTERN.search(error_message):
                return True

    # Silent overflow (z.ai style): successful, but usage exceeds the context window.
    if context_window and message.stop_reason == "stop":
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    # Length-stop overflow (Xiaomi MiMo style): the server truncates oversized
    # input to fit the context window, leaving no room for output. It returns
    # stop_reason "length" with output=0 and input+cache_read filling the window.
    if context_window and message.stop_reason == "length" and message.usage.output == 0:
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: AssistantMessage, desired_max_output: int | float) -> bool:
    """Check whether a length stop ended below the caller or model's intended output limit.

    Such responses may be caused by context pressure or provider-side truncation, so
    callers can make one bounded compact-and-retry attempt. ``desired_max_output`` must
    be the original limit before any context-based clamping.
    """
    return (
        message.stop_reason == "length"
        and desired_max_output > 0
        and message.usage.output < desired_max_output
    )


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """Get the overflow patterns for testing purposes."""
    return list(OVERFLOW_PATTERNS)
