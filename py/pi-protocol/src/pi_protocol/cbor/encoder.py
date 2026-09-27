"""Strict definite-length CBOR encoder from ``protocol/src/cbor/encoder.ts``.

Map order follows JavaScript Object.keys, including its numeric index ordering.
Floats always use binary64; this subset does not impose RFC canonical map order.
"""

from __future__ import annotations

import math
import struct

from pi_chord._undefined import UNDEFINED

from .options import CborError, CborOptions, MAX_SAFE_INTEGER, MAX_UINT32, ResolvedCborOptions, resolve_options


class _CborWriter:
    def __init__(self, max_byte_length: int) -> None:
        self._buffer = bytearray()
        self._max_byte_length = max_byte_length

    def write_byte(self, value: int) -> None:
        self._ensure_capacity(1)
        self._buffer.append(value & 0xFF)

    def write_bytes(self, value: bytes | bytearray | memoryview) -> None:
        self._ensure_capacity(value.nbytes if isinstance(value, memoryview) else len(value))
        self._buffer.extend(value)

    def write_uint16(self, value: int) -> None:
        self._ensure_capacity(2)
        self._buffer.extend(value.to_bytes(2, "big"))

    def write_uint32(self, value: int) -> None:
        self._ensure_capacity(4)
        self._buffer.extend(value.to_bytes(4, "big"))

    def write_uint64(self, value: int) -> None:
        self.write_uint32(value >> 32)
        self.write_uint32(value & MAX_UINT32)

    def write_float64(self, value: float) -> None:
        self._ensure_capacity(9)
        self._buffer.extend(b"\xfb")
        self._buffer.extend(struct.pack(">d", value))

    def finish(self) -> bytes:
        return bytes(self._buffer)

    def _ensure_capacity(self, additional_bytes: int) -> None:
        if len(self._buffer) + additional_bytes > self._max_byte_length:
            raise CborError(f"CBOR byte length exceeds configured limit of {self._max_byte_length}")


def _write_argument(writer: _CborWriter, major_type: int, value: int) -> None:
    prefix = major_type << 5
    if value < 24:
        writer.write_byte(prefix | value)
    elif value <= 0xFF:
        writer.write_byte(prefix | 24)
        writer.write_byte(value)
    elif value <= 0xFFFF:
        writer.write_byte(prefix | 25)
        writer.write_uint16(value)
    elif value <= MAX_UINT32:
        writer.write_byte(prefix | 26)
        writer.write_uint32(value)
    else:
        writer.write_byte(prefix | 27)
        writer.write_uint64(value)


def _encode_text(writer: _CborWriter, value: str, options: ResolvedCborOptions) -> None:
    # TextEncoder replaces lone UTF-16 surrogates before the source checks length
    # and rejects strings that cannot round-trip. Paired surrogates are valid.
    utf16 = value.encode("utf-16-le", errors="surrogatepass")
    normalized = utf16.decode("utf-16-le", errors="replace")
    encoded = normalized.encode("utf-8")
    if len(encoded) > options.max_byte_length:
        raise CborError(f"CBOR text string length exceeds configured limit of {options.max_byte_length}")
    if normalized.encode("utf-16-le") != utf16:
        raise CborError("CBOR text strings must contain valid Unicode scalar values")
    _write_argument(writer, 3, len(encoded))
    writer.write_bytes(encoded)


def _encode_value(
    writer: _CborWriter, value: object, options: ResolvedCborOptions, depth: int, ancestors: set[int],
) -> None:
    if depth > options.max_depth:
        raise CborError(f"CBOR nesting depth exceeds configured limit of {options.max_depth}")
    if value is None:
        writer.write_byte(0xF6)
        return
    if isinstance(value, bool):
        writer.write_byte(0xF5 if value else 0xF4)
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CborError("CBOR numbers must be finite")
        integral = isinstance(value, int) or value.is_integer()
        negative_zero = isinstance(value, float) and value == 0 and math.copysign(1.0, value) < 0
        if integral and not negative_zero:
            if abs(value) > MAX_SAFE_INTEGER:
                raise CborError("CBOR integers must be safe JavaScript integers")
            integer = int(value)
            if integer >= 0:
                _write_argument(writer, 0, integer)
            else:
                _write_argument(writer, 1, -1 - integer)
        else:
            writer.write_float64(float(value))
        return
    if isinstance(value, str):
        _encode_text(writer, value, options)
        return
    if isinstance(value, (bytes, bytearray)) or (
        isinstance(value, memoryview) and value.itemsize == 1
        and value.format in ("B", "b") and value.c_contiguous
    ):
        byte_length = value.nbytes if isinstance(value, memoryview) else len(value)
        if byte_length > options.max_byte_length:
            raise CborError(f"CBOR byte string length exceeds configured limit of {options.max_byte_length}")
        _write_argument(writer, 2, byte_length)
        writer.write_bytes(value.cast("B") if isinstance(value, memoryview) else value)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if len(value) > options.max_container_length:
            raise CborError(f"CBOR array length exceeds configured limit of {options.max_container_length}")
        ancestors.add(identity)
        try:
            _write_argument(writer, 4, len(value))
            for item in value:
                if item is UNDEFINED:
                    raise CborError("CBOR arrays must not contain holes or undefined values")
                _encode_value(writer, item, options, depth + 1, ancestors)
        finally:
            ancestors.remove(identity)
        return
    if type(value) is dict:
        identity = id(value)
        if identity in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if any(not isinstance(key, str) for key in value):
            raise CborError("CBOR map keys must be strings")
        indexed_keys: list[tuple[int, str]] = []
        other_keys: list[str] = []
        for key in value:
            if (key.isascii() and key.isdigit() and len(key) <= 10
                    and (key == "0" or key[0] != "0") and int(key) < MAX_UINT32):
                indexed_keys.append((int(key), key))
            else:
                other_keys.append(key)
        ordered_keys = [key for _, key in sorted(indexed_keys)] + other_keys
        entries = [(key, value[key]) for key in ordered_keys if value[key] is not UNDEFINED]
        if len(entries) > options.max_container_length:
            raise CborError(f"CBOR map length exceeds configured limit of {options.max_container_length}")
        ancestors.add(identity)
        try:
            _write_argument(writer, 5, len(entries))
            for key, item in entries:
                _encode_text(writer, key, options)
                _encode_value(writer, item, options, depth + 1, ancestors)
        finally:
            ancestors.remove(identity)
        return
    value_type = "undefined" if value is UNDEFINED else "function" if callable(value) else "object"
    raise CborError(f"Unsupported CBOR value type: {value_type}")


def encode_cbor(value: object, options: CborOptions | None = None) -> bytes:
    """Encode one item using the source protocol's strict RFC 8949 subset."""
    resolved = resolve_options(options)
    writer = _CborWriter(resolved.max_byte_length)
    _encode_value(writer, value, resolved, 0, set())
    return writer.finish()
