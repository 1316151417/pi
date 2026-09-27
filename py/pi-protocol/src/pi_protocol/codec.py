"""Validation and framed CBOR message codecs from ``codec.ts``."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from pi_chord.json import is_json_value

from .cbor import CborOptions, decode_cbor, encode_cbor
from .framing import DEFAULT_MAX_FRAME_LENGTH, ByteInput, FrameDecoder, FrameDecoderOptions, encode_frame
from .protocol import (
    CLIENT_MESSAGE_SCHEMA,
    PROTOCOL_VERSION,
    SERVER_MESSAGE_SCHEMA,
    ClientMessage,
    ServerMessage,
    _matches_schema,
)

__all__ = [
    "ProtocolValidationError", "parse_client_message", "parse_server_message",
    "encode_client_message", "encode_server_message", "ClientMessageDecoder", "ServerMessageDecoder",
    "is_supported_protocol_version",
]


class ProtocolValidationError(Exception):
    name = "ProtocolValidationError"


def parse_client_message(value: object) -> ClientMessage:
    if not _matches_schema(CLIENT_MESSAGE_SCHEMA, value) or not is_json_value(value):
        raise ProtocolValidationError("Invalid client protocol message")
    return cast(ClientMessage, value)


def parse_server_message(value: object) -> ServerMessage:
    if not _matches_schema(SERVER_MESSAGE_SCHEMA, value) or not is_json_value(value):
        raise ProtocolValidationError("Invalid server protocol message")
    return cast(ServerMessage, value)


def _bounded_error_message(error: object) -> str:
    if not isinstance(error, Exception):
        return "Unknown codec error"
    message = str(error)
    units = message.encode("utf-16-le", errors="surrogatepass")
    return message if len(units) <= 1000 else units[:994].decode("utf-16-le", errors="surrogatepass") + "..."


def _encode_protocol_message[T](
    value: T, parse: Callable[[object], T], kind: str, options: FrameDecoderOptions | None,
) -> bytes:
    validated = parse(value)
    try:
        maximum = options.max_frame_length if options is not None and options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        return encode_frame(encode_cbor(validated, CborOptions(max_byte_length=maximum)))
    except ProtocolValidationError:
        raise
    except Exception as error:
        raise ProtocolValidationError(f"Unable to encode {kind} protocol message: {_bounded_error_message(error)}") from error


def encode_client_message(message: ClientMessage, options: FrameDecoderOptions | None = None) -> bytes:
    return _encode_protocol_message(message, parse_client_message, "client", options)


def encode_server_message(message: ServerMessage, options: FrameDecoderOptions | None = None) -> bytes:
    return _encode_protocol_message(message, parse_server_message, "server", options)


class _ValidatedMessageDecoder[T]:
    def __init__(self, kind: str, parse: Callable[[object], T], options: FrameDecoderOptions | None) -> None:
        self._frames = FrameDecoder(options)
        self._kind = kind
        self._max_frame_length = options.max_frame_length if options is not None and options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        self._parse = parse
        self._failed = False

    def push(self, chunk: ByteInput) -> list[T]:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            messages: list[T] = []
            for frame in self._frames.push(chunk):
                messages.append(self._parse(decode_cbor(frame, CborOptions(max_byte_length=self._max_frame_length))))
            return messages
        except ProtocolValidationError:
            self._failed = True
            raise
        except Exception as error:
            self._failed = True
            raise ProtocolValidationError(f"Invalid {self._kind} protocol frame: {_bounded_error_message(error)}") from error

    def end(self) -> None:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            self._frames.end()
        except Exception as error:
            self._failed = True
            raise ProtocolValidationError(f"Invalid {self._kind} protocol framing: {_bounded_error_message(error)}") from error


class ClientMessageDecoder:
    def __init__(self, options: FrameDecoderOptions | None = None) -> None:
        self._decoder = _ValidatedMessageDecoder("client", parse_client_message, options)

    def push(self, chunk: ByteInput) -> list[ClientMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


class ServerMessageDecoder:
    def __init__(self, options: FrameDecoderOptions | None = None) -> None:
        self._decoder = _ValidatedMessageDecoder("server", parse_server_message, options)

    def push(self, chunk: ByteInput) -> list[ServerMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


def is_supported_protocol_version(version: object) -> bool:
    return type(version) in (int, float) and version == PROTOCOL_VERSION
