"""pi-ai utilities ported from ``src/utils/*``.

Mirrors the TypeScript layout: the assistant-call retry policy lives here, as
does the provider transport-retry policy.
"""

from __future__ import annotations

from .abort_signals import CombinedAbortSignal, combine_abort_signals
from .diagnostics import (
    AssistantMessageDiagnostic, DiagnosticErrorInfo, append_assistant_message_diagnostic,
    create_assistant_message_diagnostic, extract_diagnostic_error, format_thrown_value,
)
from .error_body import (
    MAX_PROVIDER_ERROR_BODY_CHARS, NormalizedProviderError, format_provider_error,
    normalize_provider_error, safe_json_stringify, truncate_error_text,
)
from .estimate import (
    ContextUsageEstimate, calculate_context_tokens, estimate_context_tokens,
    estimate_message_tokens, estimate_text_and_image_content_tokens, estimate_text_tokens,
)
from .hash import short_hash
from .headers import headers_to_record, provider_headers_to_record
from .pi_user_agent import get_pi_user_agent
from .provider_env import get_provider_env_value
from .provider_retry import (
    DEFAULT_MAX_RETRY_DELAY_MS as PROVIDER_DEFAULT_MAX_RETRY_DELAY_MS,
    ProviderHttpError,
    ProviderRetryAborted,
    is_retryable_provider_error,
    retry_provider_request,
)
from .retry import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms as assistant_retry_delay_ms,
)
from .sanitize_unicode import sanitize_surrogates
from .sleep import sleep
from .typebox_helpers import StringEnumOptions, StringEnumSchema, string_enum

__all__ = [
    "CombinedAbortSignal",
    "combine_abort_signals",
    "AssistantMessageDiagnostic",
    "DiagnosticErrorInfo",
    "append_assistant_message_diagnostic",
    "create_assistant_message_diagnostic",
    "extract_diagnostic_error",
    "format_thrown_value",
    "MAX_PROVIDER_ERROR_BODY_CHARS",
    "NormalizedProviderError",
    "format_provider_error",
    "normalize_provider_error",
    "safe_json_stringify",
    "truncate_error_text",
    "ContextUsageEstimate",
    "calculate_context_tokens",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "estimate_text_and_image_content_tokens",
    "estimate_text_tokens",
    "short_hash",
    "headers_to_record",
    "provider_headers_to_record",
    "get_pi_user_agent",
    "get_provider_env_value",
    "sanitize_surrogates",
    "sleep",
    "StringEnumOptions",
    "StringEnumSchema",
    "string_enum",
    "PROVIDER_DEFAULT_MAX_RETRY_DELAY_MS",
    "ProviderHttpError",
    "ProviderRetryAborted",
    "is_retryable_provider_error",
    "retry_provider_request",
    "DEFAULT_MAX_AGENT_RETRY_DELAY_MS",
    "RetryCallbacks",
    "RetryPolicy",
    "is_retryable_assistant_error",
    "retry_assistant_call",
    "assistant_retry_delay_ms",
]
