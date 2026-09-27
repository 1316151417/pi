"""Errors from the pinned Anthropic JavaScript SDK 0.124.0."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .._javascript import javascript_json_stringify
from .._values import JSON_NULL, UNDEFINED


def _truthy(value: object) -> bool:
    if value is None or value is UNDEFINED or value is JSON_NULL or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return value != "" if isinstance(value, str) else True


class AnthropicError(Exception):
    name = "Error"

    @property
    def message(self) -> str:
        return str(self)


class APIError(AnthropicError):
    def __init__(self, status: int | None, error: object, message: str | None, headers: Mapping[str, str] | None, type: str | None = None) -> None:
        detail = error.get("message") if isinstance(error, Mapping) else None
        if _truthy(detail):
            msg = detail if isinstance(detail, str) else javascript_json_stringify(detail)
        elif _truthy(error):
            msg = javascript_json_stringify(error)
        else:
            msg = message
        super().__init__(f"{status} {msg}" if status and msg else f"{status} status code (no body)" if status else msg or "(no status code or body)")
        self.status = status
        self.error = error
        self.headers = headers
        self.request_id = next((value for key, value in (headers or {}).items() if key.lower() == "request-id"), None)
        self.workspace_id = next((value for key, value in (headers or {}).items() if key.lower() == "anthropic-workspace-id"), None)
        self.type = type

    @staticmethod
    def generate(status: int | None, error: object, message: str | None, headers: Mapping[str, str] | None) -> APIError:
        if not status or headers is None:
            return APIConnectionError(message=message, cause=error if isinstance(error, BaseException) else None)
        inner = error.get("error") if isinstance(error, Mapping) else None
        error_type = inner.get("type") if isinstance(inner, Mapping) else None
        error_class = {400: BadRequestError, 401: AuthenticationError, 403: PermissionDeniedError,
            404: NotFoundError, 409: ConflictError, 422: UnprocessableEntityError, 429: RateLimitError,
        }.get(status, InternalServerError if status >= 500 else APIError)
        return error_class(status, error, message, headers, error_type)


class APIUserAbortError(APIError):
    def __init__(self, *, message: str | None = None) -> None:
        super().__init__(None, None, message or "Request was aborted.", None)


class APIConnectionError(APIError):
    def __init__(self, *, message: str | None = None, cause: BaseException | None = None) -> None:
        super().__init__(None, None, message or "Connection error.", None)
        if cause is not None:
            self.cause = cause


class APIConnectionTimeoutError(APIConnectionError):
    def __init__(self, *, message: str | None = None) -> None:
        super().__init__(message="Request timed out." if message is None else message)


class BadRequestError(APIError):
    pass


class AuthenticationError(APIError):
    pass


class PermissionDeniedError(APIError):
    pass


class NotFoundError(APIError):
    pass


class ConflictError(APIError):
    pass


class UnprocessableEntityError(APIError):
    pass


class RateLimitError(APIError):
    pass


class InternalServerError(APIError):
    pass
