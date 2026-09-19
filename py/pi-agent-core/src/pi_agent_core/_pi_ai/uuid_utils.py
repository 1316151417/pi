"""Time-ordered UUIDv7 ported from pi-ai ``src/utils/uuid.ts``."""

from __future__ import annotations

import os
import time

__all__ = ["uuidv7"]

MAX_UUID_V7_TIMESTAMP = 0xFFFFFFFFFFFF
MAX_SEQUENCE = (1 << 41) - 1

_last_ordinary_timestamp = -1
_sequence: int | None = None


def uuidv7(timestamp_ms: int | None = None) -> str:
    global _last_ordinary_timestamp, _sequence

    requested_timestamp = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    if requested_timestamp < 0 or requested_timestamp > MAX_UUID_V7_TIMESTAMP:
        raise ValueError(f"UUIDv7 timestamp must be an integer between 0 and {MAX_UUID_V7_TIMESTAMP}")

    if timestamp_ms is None:
        effective_timestamp = max(requested_timestamp, _last_ordinary_timestamp)
        _last_ordinary_timestamp = effective_timestamp
    else:
        effective_timestamp = timestamp_ms

    bytes_ = bytearray(os.urandom(16))
    if _sequence is None:
        _sequence = (bytes_[1] << 32) | (bytes_[2] << 24) | (bytes_[3] << 16) | (bytes_[4] << 8) | bytes_[5]
    else:
        if _sequence == MAX_SEQUENCE:
            raise ValueError("UUIDv7 generator sequence exhausted")
        _sequence += 1

    timestamp = effective_timestamp
    for index in range(6):
        bytes_[index] = (timestamp >> ((5 - index) * 8)) & 0xFF
    seq = _sequence
    bytes_[6] = 0x70 | ((seq >> 37) & 0x0F)
    bytes_[7] = (seq >> 29) & 0xFF
    bytes_[8] = 0x80 | ((seq >> 23) & 0x3F)
    bytes_[9] = (seq >> 15) & 0xFF
    bytes_[10] = (seq >> 7) & 0xFF
    bytes_[11] = ((seq & 0x7F) << 1) | (bytes_[11] & 0x01)

    hex_ = bytes_.hex()
    return f"{hex_[0:8]}-{hex_[8:12]}-{hex_[12:16]}-{hex_[16:20]}-{hex_[20:32]}"
