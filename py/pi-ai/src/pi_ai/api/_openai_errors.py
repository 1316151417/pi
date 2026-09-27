"""Error and value contracts shared by the native OpenAI SDK transport."""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..types import JSON_NULL, UNDEFINED
from ..utils._javascript import javascript_json_stringify


def truthy(value: object) -> bool:
    if value is None or value is UNDEFINED or value is JSON_NULL or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return value != "" if isinstance(value, str) else True


class OpenAIError(Exception):
    pass


class APIError(OpenAIError):
    def __init__(
        self, status: int | None = None, error: object = None, message: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        detail = error.get("message") if isinstance(error, Mapping) else None
        if truthy(detail):
            text = detail if isinstance(detail, str) else javascript_json_stringify(detail)
        elif truthy(error):
            text = javascript_json_stringify(error)
        else:
            text = message
        self.message = f"{status} {text}" if status and text else (
            f"{status} status code (no body)" if status else text or "(no status code or body)"
        )
        super().__init__(self.message)
        self.status = status
        self.headers = headers
        self.error = error
        self.request_id = headers.get("x-request-id") if headers is not None else None
        self.code = error.get("code") if isinstance(error, Mapping) else None
        self.param = error.get("param") if isinstance(error, Mapping) else None
        self.type = error.get("type") if isinstance(error, Mapping) else None


class APIUserAbortError(APIError):
    def __init__(self) -> None:
        super().__init__(message="Request was aborted.")


class APIConnectionError(APIError):
    def __init__(self, message: str = "Connection error.", cause: BaseException | None = None) -> None:
        super().__init__(message=message)
        self.cause = cause
        self.__cause__ = cause


class APIConnectionTimeoutError(APIConnectionError):
    def __init__(self) -> None:
        super().__init__("Request timed out.")


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


def is_abort_error(error: BaseException) -> bool:
    return getattr(error, "name", None) == "AbortError" or "FetchRequestCanceledException" in str(error)
