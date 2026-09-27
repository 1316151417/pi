"""Client-facing errors from ``errors.ts``."""

from pi_protocol.protocol import ProtocolError, ProtocolErrorCode


class ServerError(Exception):
    name = "ServerError"

    def __init__(self, error: ProtocolError) -> None:
        super().__init__(error["message"])
        self.code: ProtocolErrorCode = error["code"]


class DisconnectedError(Exception):
    name = "DisconnectedError"

    def __init__(self, message: str = "Client is disconnected", cause: Exception | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class ClientDisposedError(Exception):
    name = "ClientDisposedError"

    def __init__(self) -> None:
        super().__init__("Client is disposed")


def to_error(error: object) -> Exception:
    return error if isinstance(error, Exception) else Exception(str(error))


def to_disconnected_error(error: object) -> DisconnectedError:
    cause = to_error(error)
    return cause if isinstance(cause, DisconnectedError) else DisconnectedError(str(cause), cause)
