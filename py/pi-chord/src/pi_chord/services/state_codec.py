"""Per-subscription stateful Delta codecs, ported from services/state-codec.ts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from ..delta import UNDEFINED, Decoder, Encoder, decoder, encoder
from ..types import ServiceInstanceAddress, ServiceInstanceSnapshot, ServiceProviderUpdate, ServiceSubscriptionSnapshot
from .wire import WireServiceInstanceSnapshot, WireServiceProviderUpdate, WireServiceSubscriptionSnapshot


@dataclass(frozen=True)
class _CodecEntry[C]:
    instance: ServiceInstanceAddress | None
    codec: C


class _StateCodecRegistry[C]:
    def __init__(self, create: Callable[[], C]) -> None:
        self._create = create
        self._entries: dict[tuple[str | None, int | None, str], _CodecEntry[C]] = {}

    def reset(self) -> None:
        self._entries.clear()

    def add(self, instance: ServiceInstanceAddress | None, member: str) -> C:
        if instance is UNDEFINED:
            instance = None
        key = (instance["key"], instance["generation"], member) if instance else (None, None, member)
        if key in self._entries:
            raise RuntimeError(f"Duplicate service state {_describe_state(instance, member)}")
        codec = self._create()
        self._entries[key] = _CodecEntry(instance, codec)
        return codec

    def get(self, instance: ServiceInstanceAddress | None, member: str) -> C:
        if instance is UNDEFINED:
            instance = None
        key = (instance["key"], instance["generation"], member) if instance else (None, None, member)
        entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError(f"Unknown service state {_describe_state(instance, member)}")
        return entry.codec

    def remove_instance(self, instance: ServiceInstanceAddress) -> None:
        for key, entry in tuple(self._entries.items()):
            if (entry.instance is not None and entry.instance["key"] == instance["key"]
                    and entry.instance["generation"] == instance["generation"]):
                del self._entries[key]


class ServiceStateEncoder:
    def __init__(self) -> None:
        self._codecs: _StateCodecRegistry[Encoder] = _StateCodecRegistry(encoder)

    def encode_snapshot(self, snapshot: ServiceSubscriptionSnapshot) -> WireServiceSubscriptionSnapshot:
        self._codecs.reset()
        return cast(WireServiceSubscriptionSnapshot, {
            **snapshot,
            "instances": [_encode_instance(instance, self._codecs) for instance in snapshot["instances"]],
        })

    def encode_update(self, update: ServiceProviderUpdate) -> WireServiceProviderUpdate:
        match update["type"]:
            case "state":
                return {**update, "ops": self._codecs.get(update.get("instance"), update["member"]).encode(update["ops"])}
            case "replaced":
                self._codecs.reset()
                return {**update, "snapshot": _encode_instance(update["snapshot"], self._codecs)}
            case "spawned":
                return {**update, "instance": _encode_instance(update["instance"], self._codecs)}
            case "unavailable":
                self._codecs.reset()
            case "closed":
                self._codecs.remove_instance(update["instance"])
        return update


class ServiceStateDecoder:
    def __init__(self) -> None:
        self._codecs: _StateCodecRegistry[Decoder] = _StateCodecRegistry(decoder)

    def decode_snapshot(self, snapshot: WireServiceSubscriptionSnapshot) -> ServiceSubscriptionSnapshot:
        self._codecs.reset()
        return cast(ServiceSubscriptionSnapshot, {
            **snapshot,
            "instances": [_decode_instance(instance, self._codecs) for instance in snapshot["instances"]],
        })

    def decode_update(self, update: WireServiceProviderUpdate) -> ServiceProviderUpdate:
        match update["type"]:
            case "state":
                return {**update, "ops": self._codecs.get(update.get("instance"), update["member"]).decode(update["ops"])}
            case "replaced":
                self._codecs.reset()
                return {**update, "snapshot": _decode_instance(update["snapshot"], self._codecs)}
            case "spawned":
                return {**update, "instance": _decode_instance(update["instance"], self._codecs)}
            case "unavailable":
                self._codecs.reset()
            case "closed":
                self._codecs.remove_instance(update["instance"])
        return update


def _encode_instance(
    instance: ServiceInstanceSnapshot, codecs: _StateCodecRegistry[Encoder],
) -> WireServiceInstanceSnapshot:
    return {
        **instance,
        "members": [
            {**member, "ops": codecs.add(instance.get("instance"), member["name"]).encode(member["ops"])}
            if member["kind"] == "state" else member
            for member in instance["members"]
        ],
    }


def _decode_instance(
    instance: WireServiceInstanceSnapshot, codecs: _StateCodecRegistry[Decoder],
) -> ServiceInstanceSnapshot:
    return {
        **instance,
        "members": [
            {**member, "ops": codecs.add(instance.get("instance"), member["name"]).decode(member["ops"])}
            if member["kind"] == "state" else member
            for member in instance["members"]
        ],
    }


def _describe_state(instance: ServiceInstanceAddress | None, member: str) -> str:
    return member if instance is None else f"{instance['key']}@{instance['generation']}.{member}"


def create_service_state_encoder() -> ServiceStateEncoder:
    return ServiceStateEncoder()


def create_service_state_decoder() -> ServiceStateDecoder:
    return ServiceStateDecoder()
