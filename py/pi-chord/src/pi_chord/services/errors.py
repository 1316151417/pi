"""Stable error codes crossing the Chord service boundary."""

from typing import Literal

type RemoteServiceErrorCode = Literal[
    "service_not_allowed", "service_not_found", "service_mode_mismatch",
    "service_member_not_found", "service_member_mismatch", "service_instance_not_found",
    "service_stale_instance", "service_invalid_value",
]

REMOTE_SERVICE_ERROR_CODES: tuple[RemoteServiceErrorCode, ...] = (
    "service_not_allowed", "service_not_found", "service_mode_mismatch",
    "service_member_not_found", "service_member_mismatch", "service_instance_not_found",
    "service_stale_instance", "service_invalid_value",
)


def is_remote_service_error_code(value: object) -> bool:
    return isinstance(value, str) and value in REMOTE_SERVICE_ERROR_CODES


class RemoteServiceError(Exception):
    def __init__(self, code: RemoteServiceErrorCode, message: str) -> None:
        super().__init__(message)
        self.name = "RemoteServiceError"
        self.code = code
