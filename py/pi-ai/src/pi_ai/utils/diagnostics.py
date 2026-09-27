"""Provider/runtime diagnostics ported from ``src/utils/diagnostics.ts``."""

from __future__ import annotations

import time
import traceback
from typing import Protocol

from ..types import AssistantMessageDiagnostic, DiagnosticErrorInfo, JsonObject
from ._javascript import javascript_string

__all__ = [
    "DiagnosticErrorInfo",
    "AssistantMessageDiagnostic",
    "format_thrown_value",
    "extract_diagnostic_error",
    "create_assistant_message_diagnostic",
    "append_assistant_message_diagnostic",
]


class DiagnosticMessage(Protocol):
    diagnostics: list[AssistantMessageDiagnostic] | None


def format_thrown_value(value: object) -> str:
    if isinstance(value, BaseException):
        return getattr(value, "message", str(value)) or getattr(value, "name", type(value).__name__)
    if isinstance(value, str):
        return value
    return javascript_string(value)


def extract_diagnostic_error(error: object) -> DiagnosticErrorInfo:
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(name="ThrownValue", message=format_thrown_value(error))
    name = getattr(error, "name", type(error).__name__)
    code = getattr(error, "code", None)
    stack = getattr(error, "stack", None)
    if stack is None and error.__traceback__ is not None:
        stack = "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip("\n")
    return DiagnosticErrorInfo(
        name=name or None,
        message=getattr(error, "message", str(error)) or name,
        stack=stack,
        code=code if isinstance(code, (str, int, float)) and not isinstance(code, bool) else None,
    )


def create_assistant_message_diagnostic(
    type: str,
    error: object,
    details: JsonObject | None = None,
) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        type=type,
        timestamp=time.time_ns() // 1_000_000,
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(message: DiagnosticMessage, diagnostic: AssistantMessageDiagnostic) -> None:
    message.diagnostics = [*(message.diagnostics or []), diagnostic]
