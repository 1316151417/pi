"""Strict definite-length CBOR decoder from ``protocol/src/cbor/decoder.ts``."""

from __future__ import annotations

import math
import struct

from .options import CborError, CborOptions, MAX_SAFE_INTEGER, ResolvedCborOptions, UINT32_BASE, resolve_options


class _CborReader:
    def __init__(self, data: memoryview, options: ResolvedCborOptions) -> None:
        self._bytes = data
        self._offset = 0
        self._options = options

    def decode(self) -> object:
        value = self._read_item(0)
        if self._offset != len(self._bytes):
            raise CborError("CBOR payload contains trailing data")
        return value

    def _read_item(self, depth: int) -> object:
        if depth > self._options.max_depth:
            raise CborError(f"CBOR nesting depth exceeds configured limit of {self._options.max_depth}")
        initial = self._read_byte()
        major_type = initial >> 5
        additional_information = initial & 0x1F
        if major_type == 0:
            return self._read_argument(additional_information)
        if major_type == 1:
            value = -1 - self._read_argument(additional_information)
            if value < -MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if major_type == 2:
            length = self._read_length(additional_information, "byte string", self._options.max_byte_length)
            return bytes(self._read_bytes(length))
        if major_type == 3:
            length = self._read_length(additional_information, "text string", self._options.max_byte_length)
            encoded = self._read_bytes(length)
            try:
                return encoded.tobytes().decode("utf-8")
            except UnicodeDecodeError:
                raise CborError("CBOR text string contains invalid UTF-8") from None
        if major_type == 4:
            length = self._read_length(additional_information, "array", self._options.max_container_length)
            array: list[object] = []
            for _ in range(length):
                array.append(self._read_item(depth + 1))
            return array
        if major_type == 5:
            length = self._read_length(additional_information, "map", self._options.max_container_length)
            record: dict[str, object] = {}
            keys: set[str] = set()
            for _ in range(length):
                key = self._read_item(depth + 1)
                if not isinstance(key, str):
                    raise CborError("CBOR map keys must be strings")
                if key in keys:
                    raise CborError("CBOR map contains a duplicate key")
                keys.add(key)
                record[key] = self._read_item(depth + 1)
            return record
        if major_type == 6:
            raise CborError("CBOR tags are not supported")
        if major_type == 7:
            return self._read_simple(additional_information)
        raise CborError("Malformed CBOR major type")

    def _read_simple(self, additional_information: int) -> object:
        if additional_information == 20:
            return False
        if additional_information == 21:
            return True
        if additional_information == 22:
            return None
        if additional_information == 27:
            value = struct.unpack(">d", self._read_bytes(8))[0]
            if not math.isfinite(value):
                raise CborError("Decoded CBOR number must be finite")
            if value.is_integer() and abs(value) > MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if additional_information == 31:
            raise CborError("CBOR break marker is not supported")
        raise CborError("Unsupported CBOR simple value or floating-point width")

    def _read_length(self, additional_information: int, kind: str, limit: int) -> int:
        if additional_information == 31:
            raise CborError(f"Indefinite-length CBOR {kind}s are not supported")
        length = self._read_argument(additional_information)
        if length > limit:
            raise CborError(f"CBOR {kind} length exceeds configured limit of {limit}")
        return length

    def _read_argument(self, additional_information: int) -> int:
        if additional_information < 24:
            return additional_information
        if additional_information == 24:
            return self._read_byte()
        if additional_information == 25:
            return int.from_bytes(self._read_bytes(2), "big")
        if additional_information == 26:
            return int.from_bytes(self._read_bytes(4), "big")
        if additional_information == 27:
            high = self._read_argument(26)
            low = self._read_argument(26)
            if high > 0x1F_FFFF:
                raise CborError("Decoded CBOR integer or length is outside the safe range")
            return high * UINT32_BASE + low
        if additional_information == 31:
            raise CborError("Indefinite-length CBOR items are not supported")
        raise CborError("Malformed CBOR additional information")

    def _read_byte(self) -> int:
        if self._offset >= len(self._bytes):
            raise CborError("Truncated CBOR payload")
        value = self._bytes[self._offset]
        self._offset += 1
        return value

    def _read_bytes(self, length: int) -> memoryview:
        if length > len(self._bytes) - self._offset:
            raise CborError("Truncated CBOR payload")
        value = self._bytes[self._offset:self._offset + length]
        self._offset += length
        return value


def decode_cbor(data: bytes | bytearray | memoryview, options: CborOptions | None = None) -> object:
    """Decode exactly one item; byte strings are detached Python bytes objects."""
    if isinstance(data, (bytes, bytearray)):
        view = memoryview(data)
    elif (isinstance(data, memoryview) and data.itemsize == 1
            and data.format in ("B", "b") and data.c_contiguous):
        view = data.cast("B")
    else:
        raise TypeError("CBOR input must be a Uint8Array")
    resolved = resolve_options(options)
    if len(view) > resolved.max_byte_length:
        raise CborError(f"CBOR byte length exceeds configured limit of {resolved.max_byte_length}")
    return _CborReader(view, resolved).decode()
