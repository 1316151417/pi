"""Normalize HTTP error status and response bodies across provider SDKs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._javascript import javascript_json_stringify, javascript_string, utf16_length, utf16_slice

__all__ = [
    "MAX_PROVIDER_ERROR_BODY_CHARS",
    "NormalizedProviderError",
    "normalize_provider_error",
    "format_provider_error",
    "truncate_error_text",
    "safe_json_stringify",
]

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


@dataclass
class NormalizedProviderError:
    message: str
    message_carries_body: bool
    status: int | float | None = None
    body: str | None = None


def _field(value: object, name: str, native_name: str | None = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, value.get(native_name) if native_name is not None else None)
    return getattr(value, name, getattr(value, native_name, None) if native_name is not None else None)


def normalize_provider_error(error: object) -> NormalizedProviderError:
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error), message_carries_body=False)

    status = None
    for candidate in (
        _field(error, "statusCode", "status_code"),
        _field(error, "status"),
        _field(_field(error, "$metadata", "metadata"), "httpStatusCode", "http_status_code"),
        _field(_field(error, "$response", "response"), "statusCode", "status_code"),
    ):
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            status = candidate
            break

    body_text: str | None = None
    body_value = _field(error, "body")
    parsed_body = _field(error, "error")
    response_body = _field(_field(error, "$response", "response"), "body")
    if isinstance(body_value, str):
        body_text = body_value
    elif type(parsed_body) is dict and parsed_body:
        body_text = safe_json_stringify(parsed_body)
    elif isinstance(response_body, str):
        body_text = response_body
    elif not callable(_field(response_body, "pipe")) and type(response_body) is dict and response_body:
        body_text = safe_json_stringify(response_body)
    body = None
    if body_text is not None:
        trimmed = body_text.strip("\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff")
        if trimmed:
            body = truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)
    message = getattr(error, "message", str(error))
    return NormalizedProviderError(
        status=status,
        body=body,
        message=message,
        message_carries_body=body is None or body in message,
    )


def format_provider_error(norm: NormalizedProviderError, prefix: str | None = None) -> str:
    status = javascript_string(norm.status)
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({status}): {norm.message}"
        return norm.message
    return f"{prefix} ({status}): {norm.body}" if prefix is not None else f"{status}: {norm.body}"


def truncate_error_text(text: str, max_chars: int) -> str:
    length = utf16_length(text)
    if length <= max_chars:
        return text
    return f"{utf16_slice(text, max_chars)}... [truncated {length - max_chars} chars]"


def safe_json_stringify(value: object) -> str:
    try:
        serialized = javascript_json_stringify(value)
        return javascript_string(value) if serialized is None else serialized
    except Exception:
        return javascript_string(value)
