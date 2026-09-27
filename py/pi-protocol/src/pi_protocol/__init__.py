"""Transport-neutral protocol v8: envelopes, CBOR, and incremental framing."""

from .cbor import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    CborError,
    CborOptions,
    decode_cbor,
    encode_cbor,
)
from .codec import (
    ClientMessageDecoder,
    ProtocolValidationError,
    ServerMessageDecoder,
    encode_client_message,
    encode_server_message,
    is_supported_protocol_version,
    parse_client_message,
    parse_server_message,
)
from .framing import DEFAULT_MAX_FRAME_LENGTH, FrameDecoder, FrameDecoderOptions, FrameError, encode_frame
from .protocol import (
    PROTOCOL_VERSION,
    AttachmentEnvelope,
    CancelEnvelope,
    ClientHello,
    ClientMessage,
    FailedResponseEnvelope,
    ProtocolError,
    ProtocolErrorCode,
    RequestEnvelope,
    ResponseEnvelope,
    RpcTarget,
    ServerHello,
    ServerHelloError,
    ServerId,
    ServerMessage,
    ServerTarget,
    ServiceEventEnvelope,
    SessionTarget,
    SuccessfulResponseEnvelope,
    is_server_id,
)

__all__ = [
    "DEFAULT_MAX_CBOR_BYTE_LENGTH", "DEFAULT_MAX_CBOR_CONTAINER_LENGTH", "DEFAULT_MAX_CBOR_DEPTH",
    "CborError", "CborOptions", "decode_cbor", "encode_cbor", "ClientMessageDecoder",
    "ProtocolValidationError", "ServerMessageDecoder", "encode_client_message", "encode_server_message",
    "is_supported_protocol_version", "parse_client_message", "parse_server_message", "DEFAULT_MAX_FRAME_LENGTH",
    "FrameDecoder", "FrameDecoderOptions", "FrameError", "encode_frame", "PROTOCOL_VERSION",
    "AttachmentEnvelope", "CancelEnvelope", "ClientHello", "ClientMessage", "ProtocolError", "ProtocolErrorCode",
    "RequestEnvelope", "ResponseEnvelope", "RpcTarget", "ServerHello", "ServerHelloError", "ServerId",
    "ServerMessage", "ServiceEventEnvelope", "SessionTarget", "ServerTarget", "SuccessfulResponseEnvelope",
    "FailedResponseEnvelope", "is_server_id",
]
