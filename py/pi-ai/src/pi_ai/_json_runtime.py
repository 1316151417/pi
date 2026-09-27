"""JavaScript string and JSON primitives used by the streaming parser."""

from __future__ import annotations

import json
import math
from typing import cast

from .types import JsonValue

JS_WHITESPACE = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
    "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def utf16_units(value: str) -> str:
    encoded = value.encode("utf-16-le", "surrogatepass")
    return "".join(chr(encoded[index] | encoded[index + 1] << 8) for index in range(0, len(encoded), 2))


def _normalize_strings(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    if isinstance(value, list):
        return [_normalize_strings(item) for item in value]
    if isinstance(value, dict):
        return {
            cast(str, _normalize_strings(key)): _normalize_strings(item)
            for key, item in value.items()
        }
    return value


def _parse_integer(value: str) -> int | float:
    # JSON.parse produces binary64 Numbers, including rounded large integers
    # and negative zero; Python's unbounded integer parsing would change them.
    number = float(value)
    if number == 0 and value.startswith("-"):
        return number
    return int(number) if math.isfinite(number) else number


def _reject_constant(value: str) -> JsonValue:
    # CPython accepts these constants by default; JSON.parse does not.
    raise json.JSONDecodeError(f"Unexpected token {value!r}", value, 0)


def parse_json(value: str) -> JsonValue:
    return _normalize_strings(
        cast(JsonValue, json.loads(value, parse_int=_parse_integer, parse_constant=_reject_constant))
    )
