"""Service wire records and strict envelope validation from services/wire.ts."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, NotRequired, TypedDict, cast

from .._undefined import UNDEFINED
from ..delta import WireOp, assert_valid_op, assert_valid_wire_op
from ..types import (
    ServiceCall, ServiceCatalogueEntry, ServiceClosedUpdate, ServiceInstanceAddress,
    ServiceMethodSnapshot, ServiceMode, ServiceProviderUpdate, ServiceSubscriptionSnapshot,
    ServiceUnavailableUpdate,
)


class WireServiceStateSnapshot(TypedDict):
    name: str
    kind: Literal["state"]
    sequence: int
    ops: Sequence[WireOp]


type WireServiceMemberSnapshot = ServiceMethodSnapshot | WireServiceStateSnapshot


class WireServiceInstanceSnapshot(TypedDict):
    instance: NotRequired[ServiceInstanceAddress]
    members: Sequence[WireServiceMemberSnapshot]


class WireServiceSubscriptionSnapshot(TypedDict):
    serviceId: str
    mode: ServiceMode
    instances: Sequence[WireServiceInstanceSnapshot]


class WireServiceStateUpdate(TypedDict):
    type: Literal["state"]
    instance: NotRequired[ServiceInstanceAddress]
    member: str
    sequence: int
    ops: Sequence[WireOp]


class WireServiceReplacedUpdate(TypedDict):
    type: Literal["replaced"]
    snapshot: WireServiceInstanceSnapshot


class WireServiceSpawnedUpdate(TypedDict):
    type: Literal["spawned"]
    instance: WireServiceInstanceSnapshot


type WireServiceProviderUpdate = (
    WireServiceStateUpdate | ServiceUnavailableUpdate | WireServiceReplacedUpdate
    | WireServiceSpawnedUpdate | ServiceClosedUpdate
)


class ServiceCatalogueControlCall(TypedDict):
    type: Literal["catalogue"]


class ServiceSubscribeControlCall(TypedDict):
    type: Literal["subscribe"]
    subscriptionId: str
    serviceId: str
    mode: ServiceMode


class ServiceUnsubscribeControlCall(TypedDict):
    type: Literal["unsubscribe"]
    subscriptionId: str


type ServiceControlCall = (
    ServiceCatalogueControlCall | ServiceSubscribeControlCall | ServiceUnsubscribeControlCall
)

SERVICE_CONTROL_ID = "$chord.service"


def create_service_catalogue_call() -> ServiceCall:
    return {"serviceId": SERVICE_CONTROL_ID, "member": "catalogue", "args": []}


def create_service_subscribe_call(
    subscription_id: str, service_id: str, mode: ServiceMode,
) -> ServiceCall:
    return {"serviceId": SERVICE_CONTROL_ID, "member": "subscribe",
            "args": [subscription_id, service_id, mode]}


def create_service_unsubscribe_call(subscription_id: str) -> ServiceCall:
    return {"serviceId": SERVICE_CONTROL_ID, "member": "unsubscribe", "args": [subscription_id]}


def decode_service_control_call(call: ServiceCall) -> ServiceControlCall | None:
    if call["serviceId"] != SERVICE_CONTROL_ID or call.get("instance", UNDEFINED) is not UNDEFINED:
        return None
    args = call["args"]
    if call["member"] == "catalogue" and len(args) == 0:
        return {"type": "catalogue"}
    if (call["member"] == "subscribe" and len(args) == 3
            and _is_id(args[0]) and _is_id(args[1]) and _is_mode(args[2])):
        return {"type": "subscribe", "subscriptionId": cast(str, args[0]),
                "serviceId": cast(str, args[1]), "mode": cast(ServiceMode, args[2])}
    if call["member"] == "unsubscribe" and len(args) == 1 and _is_id(args[0]):
        return {"type": "unsubscribe", "subscriptionId": cast(str, args[0])}
    return None


def parse_service_call(value: object) -> ServiceCall:
    call = _record(value, "service call")
    _assert_keys(call, ("serviceId", "member", "args"), ("instance",), "service call")
    if (not _is_id(call["serviceId"]) or not _is_id(call["member"])
            or not isinstance(call["args"], (list, tuple))):
        raise TypeError("Invalid service call")
    if call.get("instance", UNDEFINED) is not UNDEFINED:
        _assert_address(call["instance"])
    return cast(ServiceCall, value)


def parse_service_catalogue(value: object) -> Sequence[ServiceCatalogueEntry]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("Invalid service catalogue")
    ids: set[str] = set()
    for candidate in value:
        entry = _record(candidate, "service catalogue entry")
        _assert_keys(entry, ("serviceId", "mode"), (), "service catalogue entry")
        service_id = entry["serviceId"]
        if not _is_id(service_id) or not _is_mode(entry["mode"]) or service_id in ids:
            raise TypeError("Invalid service catalogue")
        ids.add(cast(str, service_id))
    return cast(Sequence[ServiceCatalogueEntry], value)


def parse_service_subscription_snapshot(value: object) -> ServiceSubscriptionSnapshot:
    _assert_subscription_snapshot(value, assert_valid_op)
    return cast(ServiceSubscriptionSnapshot, value)


def parse_wire_service_subscription_snapshot(value: object) -> WireServiceSubscriptionSnapshot:
    _assert_subscription_snapshot(value, assert_valid_wire_op)
    return cast(WireServiceSubscriptionSnapshot, value)


def parse_service_provider_update(value: object) -> ServiceProviderUpdate:
    _assert_provider_update(value, assert_valid_op)
    return cast(ServiceProviderUpdate, value)


def parse_wire_service_provider_update(value: object) -> WireServiceProviderUpdate:
    _assert_provider_update(value, assert_valid_wire_op)
    return cast(WireServiceProviderUpdate, value)


def _assert_subscription_snapshot(value: object, assert_op: Callable[[object], None]) -> None:
    snapshot = _record(value, "service subscription snapshot")
    _assert_keys(snapshot, ("serviceId", "mode", "instances"), (), "service subscription snapshot")
    instances = snapshot["instances"]
    if (not _is_id(snapshot["serviceId"]) or not _is_mode(snapshot["mode"])
            or not isinstance(instances, (list, tuple))):
        raise TypeError("Invalid service subscription snapshot")
    for instance in instances:
        _assert_instance(instance, assert_op)


def _assert_provider_update(value: object, assert_op: Callable[[object], None]) -> None:
    update = _record(value, "service provider update")
    match update.get("type"):
        case "state":
            _assert_keys(update, ("type", "member", "sequence", "ops"), ("instance",), "state update")
            ops = update["ops"]
            if (not _is_id(update["member"]) or not _is_integer(update["sequence"], 1)
                    or not isinstance(ops, (list, tuple))):
                raise TypeError("Invalid service state update")
            if update.get("instance", UNDEFINED) is not UNDEFINED:
                _assert_address(update["instance"])
            for op in ops:
                assert_op(op)
        case "unavailable":
            _assert_keys(update, ("type",), (), "unavailable update")
        case "replaced":
            _assert_keys(update, ("type", "snapshot"), (), "replacement update")
            _assert_instance(update["snapshot"], assert_op)
        case "spawned":
            _assert_keys(update, ("type", "instance"), (), "spawn update")
            _assert_instance(update["instance"], assert_op)
        case "closed":
            _assert_keys(update, ("type", "instance"), (), "close update")
            _assert_address(update["instance"])
        case _:
            raise TypeError("Invalid service provider update")


def _assert_instance(value: object, assert_op: Callable[[object], None]) -> None:
    instance = _record(value, "service instance snapshot")
    _assert_keys(instance, ("members",), ("instance",), "service instance snapshot")
    if instance.get("instance", UNDEFINED) is not UNDEFINED:
        _assert_address(instance["instance"])
    members = instance["members"]
    if not isinstance(members, (list, tuple)):
        raise TypeError("Invalid service instance snapshot")
    for candidate in members:
        member = _record(candidate, "service member snapshot")
        if member.get("kind") == "method":
            _assert_keys(member, ("name", "kind"), (), "service method snapshot")
            if not _is_id(member["name"]):
                raise TypeError("Invalid service method snapshot")
        elif member.get("kind") == "state":
            _assert_keys(member, ("name", "kind", "sequence", "ops"), (), "service state snapshot")
            ops = member["ops"]
            if (not _is_id(member["name"]) or not _is_integer(member["sequence"], 0)
                    or not isinstance(ops, (list, tuple))):
                raise TypeError("Invalid service state snapshot")
            for op in ops:
                assert_op(op)
        else:
            raise TypeError("Invalid service member snapshot")


def _assert_address(value: object) -> None:
    address = _record(value, "service instance address")
    _assert_keys(address, ("key", "generation"), (), "service instance address")
    if not _is_id(address["key"]) or not _is_integer(address["generation"], 1):
        raise TypeError("Invalid service instance address")


def _record(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Invalid {description}")
    return value


def _assert_keys(
    value: dict[str, object], required: Sequence[str], optional: Sequence[str], description: str,
) -> None:
    allowed = set(required) | set(optional)
    if any(key not in value for key in required) or any(key not in allowed for key in value):
        raise TypeError(f"Invalid {description}")


def _is_id(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _is_mode(value: object) -> bool:
    return value in ("singleton", "keyed")


def _is_integer(value: object, minimum: int) -> bool:
    return ((type(value) is int and value >= minimum)
            or (type(value) is float and math.isfinite(value) and value.is_integer() and value >= minimum))
