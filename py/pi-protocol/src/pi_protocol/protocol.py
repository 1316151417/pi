"""Protocol v8 envelope contracts and strict schemas from ``protocol.ts``.

Wire dictionaries keep their camelCase names. Opaque payloads are checked as
strict JSON by the codec; their application-specific grammar belongs to Chord.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, TypedDict, cast

from pi_chord.types import JsonValue

PROTOCOL_VERSION = 8
type ServerId = str
type ProtocolErrorCode = str
type _Schema = dict[str, object]


class ProtocolError(TypedDict):
    code: ProtocolErrorCode
    message: str


class ClientHello(TypedDict):
    type: Literal["hello"]
    version: int


class ServerTarget(TypedDict):
    serverId: ServerId


class SessionTarget(TypedDict):
    serverId: ServerId
    sessionId: str
    attachmentId: str


type RpcTarget = ServerTarget | SessionTarget


class RequestEnvelope(TypedDict):
    type: Literal["request"]
    id: str
    target: RpcTarget
    call: JsonValue


class CancelEnvelope(TypedDict):
    type: Literal["cancel"]
    id: str
    target: RpcTarget


type ClientMessage = ClientHello | RequestEnvelope | CancelEnvelope


class ServerHello(TypedDict):
    type: Literal["hello"]
    version: Literal[8]
    serverId: ServerId


class ServerHelloError(TypedDict):
    type: Literal["hello_error"]
    error: ProtocolError


class SuccessfulResponseEnvelope(TypedDict):
    type: Literal["response"]
    id: str
    ok: Literal[True]
    result: NotRequired[JsonValue]


class FailedResponseEnvelope(TypedDict):
    type: Literal["response"]
    id: str
    ok: Literal[False]
    error: ProtocolError


type ResponseEnvelope = SuccessfulResponseEnvelope | FailedResponseEnvelope


class ServiceEventEnvelope(TypedDict):
    type: Literal["service_update"]
    subscriptionId: str
    update: JsonValue


class AttachmentEnvelope(TypedDict):
    type: Literal["attachment"]
    attachment: SessionTarget | None


type ServerMessage = ServerHello | ServerHelloError | ResponseEnvelope | ServiceEventEnvelope | AttachmentEnvelope


def _strict_object(properties: Mapping[str, _Schema], optional: Sequence[str] = ()) -> _Schema:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": [key for key in properties if key not in optional],
        "additionalProperties": False,
    }


_ID_SCHEMA: _Schema = {"type": "string", "minLength": 1}
_OPAQUE_JSON_VALUE_SCHEMA: _Schema = {}
_SERVER_ID_SCHEMA: _Schema = {
    "type": "string",
    "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
}
_PROTOCOL_ERROR_SCHEMA = _strict_object({"code": _ID_SCHEMA, "message": {"type": "string"}})
_CLIENT_HELLO_SCHEMA = _strict_object({
    "type": {"type": "string", "const": "hello"},
    "version": {"type": "integer", "minimum": 0},
})
_SERVER_TARGET_SCHEMA = _strict_object({"serverId": _SERVER_ID_SCHEMA})
_SESSION_TARGET_SCHEMA = _strict_object({
    "serverId": _SERVER_ID_SCHEMA, "sessionId": _ID_SCHEMA, "attachmentId": _ID_SCHEMA,
})
_RPC_TARGET_SCHEMA: _Schema = {"anyOf": [_SERVER_TARGET_SCHEMA, _SESSION_TARGET_SCHEMA]}
_REQUEST_ENVELOPE_SCHEMA = _strict_object({
    "type": {"type": "string", "const": "request"}, "id": _ID_SCHEMA,
    "target": _RPC_TARGET_SCHEMA, "call": _OPAQUE_JSON_VALUE_SCHEMA,
})
_CANCEL_ENVELOPE_SCHEMA = _strict_object({
    "type": {"type": "string", "const": "cancel"}, "id": _ID_SCHEMA, "target": _RPC_TARGET_SCHEMA,
})
CLIENT_MESSAGE_SCHEMA: _Schema = {
    "anyOf": [_CLIENT_HELLO_SCHEMA, _REQUEST_ENVELOPE_SCHEMA, _CANCEL_ENVELOPE_SCHEMA],
}
_SERVER_HELLO_SCHEMA = _strict_object({
    "type": {"type": "string", "const": "hello"},
    "version": {"type": "number", "const": PROTOCOL_VERSION}, "serverId": _SERVER_ID_SCHEMA,
})
_SERVER_HELLO_ERROR_SCHEMA = _strict_object({
    "type": {"type": "string", "const": "hello_error"}, "error": _PROTOCOL_ERROR_SCHEMA,
})
_RESPONSE_ENVELOPE_SCHEMA: _Schema = {"anyOf": [
    _strict_object({
        "type": {"type": "string", "const": "response"}, "id": _ID_SCHEMA,
        "ok": {"type": "boolean", "const": True}, "result": _OPAQUE_JSON_VALUE_SCHEMA,
    }, optional=("result",)),
    _strict_object({
        "type": {"type": "string", "const": "response"}, "id": _ID_SCHEMA,
        "ok": {"type": "boolean", "const": False}, "error": _PROTOCOL_ERROR_SCHEMA,
    }),
]}
_SERVICE_EVENT_ENVELOPE_SCHEMA = _strict_object({
    "type": {"type": "string", "const": "service_update"},
    "subscriptionId": _ID_SCHEMA, "update": _OPAQUE_JSON_VALUE_SCHEMA,
})
_ATTACHMENT_ENVELOPE_SCHEMA = _strict_object({
    "type": {"type": "string", "const": "attachment"},
    "attachment": {"anyOf": [_SESSION_TARGET_SCHEMA, {"type": "null"}]},
})
SERVER_MESSAGE_SCHEMA: _Schema = {"anyOf": [
    _SERVER_HELLO_SCHEMA, _SERVER_HELLO_ERROR_SCHEMA, _RESPONSE_ENVELOPE_SCHEMA,
    _SERVICE_EVENT_ENVELOPE_SCHEMA, _ATTACHMENT_ENVELOPE_SCHEMA,
]}


def is_server_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(cast(str, _SERVER_ID_SCHEMA["pattern"]), value) is not None


def _matches_schema(schema: _Schema, value: object) -> bool:
    """Evaluate the schema vocabulary used by these fixed envelope definitions."""
    if "anyOf" in schema:
        return any(_matches_schema(candidate, value) for candidate in cast(Sequence[_Schema], schema["anyOf"]))
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = cast(dict[str, _Schema], schema["properties"])
        required = cast(Sequence[str], schema["required"])
        if any(key not in value for key in required) or any(key not in properties for key in value):
            return False
        return all(_matches_schema(properties[key], item) for key, item in value.items())
    if kind == "string":
        if not isinstance(value, str):
            return False
        if "minLength" in schema and len(value) < cast(int, schema["minLength"]):
            return False
        if "pattern" in schema and re.fullmatch(cast(str, schema["pattern"]), value) is None:
            return False
    elif kind in ("integer", "number"):
        if type(value) not in (int, float):
            return False
        number = cast(int | float, value)
        if isinstance(number, float) and (not math.isfinite(number) or (kind == "integer" and not number.is_integer())):
            return False
        if "minimum" in schema and number < cast(int | float, schema["minimum"]):
            return False
    elif kind == "boolean":
        if type(value) is not bool:
            return False
    elif kind == "null":
        return value is None
    if "const" in schema:
        return value == schema["const"]
    return True


__all__ = [
    "PROTOCOL_VERSION", "ServerId", "ProtocolErrorCode", "ProtocolError", "ClientHello",
    "ServerTarget", "SessionTarget", "RpcTarget", "RequestEnvelope", "CancelEnvelope", "ClientMessage",
    "ServerHello", "ServerHelloError", "SuccessfulResponseEnvelope", "FailedResponseEnvelope",
    "ResponseEnvelope", "ServiceEventEnvelope", "AttachmentEnvelope", "ServerMessage",
    "CLIENT_MESSAGE_SCHEMA", "SERVER_MESSAGE_SCHEMA", "is_server_id",
]
