"""Bounded service errors from ``errors.ts``."""

from typing import Literal

from pi_chord.services.errors import RemoteServiceErrorCode

type ServerOperationErrorCode = (
    RemoteServiceErrorCode | Literal[
        "wrong_server", "session_not_found", "session_ambiguous",
        "session_not_attached", "server_draining",
    ]
)

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error"


class ServerError(Exception):
    name = "ServerError"

    def __init__(self, code: ServerOperationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class WrongServerError(ServerError):
    name = "WrongServerError"

    def __init__(self) -> None:
        super().__init__("wrong_server", "Request was addressed to another server")


class SessionNotFoundError(ServerError):
    name = "SessionNotFoundError"

    def __init__(self, message: str = "Session was not found") -> None:
        super().__init__("session_not_found", message)


class SessionAmbiguousError(ServerError):
    name = "SessionAmbiguousError"

    def __init__(self) -> None:
        super().__init__("session_ambiguous", "Session ID matches more than one session")


class SessionNotAttachedError(ServerError):
    name = "SessionNotAttachedError"

    def __init__(self) -> None:
        super().__init__("session_not_attached", "Session is not attached to this client")


class ServerDrainingError(ServerError):
    name = "ServerDrainingError"

    def __init__(self) -> None:
        super().__init__("server_draining", "Server is draining")
