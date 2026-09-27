"""CBOR limits and errors from ``packages/protocol/src/cbor/options.ts``."""

from __future__ import annotations

import math
from dataclasses import dataclass

UINT32_BASE = 0x1_0000_0000
MAX_UINT32 = 0xFFFF_FFFF
MAX_CONFIGURED_DEPTH = 512
MAX_SAFE_INTEGER = 0x1F_FFFF_FFFF_FFFF

DEFAULT_MAX_CBOR_BYTE_LENGTH = 16 * 1024 * 1024
DEFAULT_MAX_CBOR_CONTAINER_LENGTH = 1_000_000
DEFAULT_MAX_CBOR_DEPTH = 64


@dataclass(frozen=True)
class CborOptions:
    max_byte_length: int | None = None
    max_container_length: int | None = None
    max_depth: int | None = None


@dataclass(frozen=True)
class ResolvedCborOptions:
    max_byte_length: int
    max_container_length: int
    max_depth: int


class CborError(Exception):
    name = "CborError"


def _resolve_limit(name: str, value: int, maximum: int) -> int:
    if type(value) is int:
        valid = 0 <= value <= maximum
    elif type(value) is float:
        valid = math.isfinite(value) and value.is_integer() and 0 <= value <= maximum
    else:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return int(value)


def resolve_options(options: CborOptions | None = None) -> ResolvedCborOptions:
    byte_length = options.max_byte_length if options is not None else None
    container_length = options.max_container_length if options is not None else None
    depth = options.max_depth if options is not None else None
    return ResolvedCborOptions(
        max_byte_length=_resolve_limit(
            "maxByteLength", DEFAULT_MAX_CBOR_BYTE_LENGTH if byte_length is None else byte_length, MAX_UINT32,
        ),
        max_container_length=_resolve_limit(
            "maxContainerLength",
            DEFAULT_MAX_CBOR_CONTAINER_LENGTH if container_length is None else container_length,
            MAX_UINT32,
        ),
        max_depth=_resolve_limit("maxDepth", DEFAULT_MAX_CBOR_DEPTH if depth is None else depth, MAX_CONFIGURED_DEPTH),
    )
