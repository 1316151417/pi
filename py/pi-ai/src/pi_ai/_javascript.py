"""String and JSON semantics used by the TypeScript utility ports."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from decimal import Decimal
from typing import cast

from ._json_runtime import parse_json as javascript_json_parse
from ._values import JSON_NULL, UNDEFINED

_OMITTED = object()


def javascript_truthy(value: object) -> bool:
    if value is None or value is UNDEFINED or value is JSON_NULL or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return value != "" if isinstance(value, str) else True


def javascript_object_keys(record: Mapping[str, object]) -> list[str]:
    numeric: list[tuple[int, str]] = []
    other: list[str] = []
    for key in record:
        if (key.isascii() and key.isdigit() and len(key) <= 10
                and (key == "0" or key[0] != "0") and int(key) < 0xFFFFFFFF):
            numeric.append((int(key), key))
        else:
            other.append(key)
    return [key for _, key in sorted(numeric)] + other


def _number_string(value: int | float) -> str:
    if isinstance(value, int) and abs(value) <= 2**53:
        return str(value)
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    if number == 0:
        return "0"
    decimal = Decimal(repr(number))
    if 1e-6 <= abs(number) < 1e21:
        fixed = format(decimal, "f")
        return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    scientific = format(decimal.normalize(), "e")
    mantissa, exponent = scientific.split("e")
    return f"{mantissa}e{int(exponent):+d}"


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def utf16_slice(text: str, end: int) -> str:
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    return encoded[: end * 2].decode("utf-16-le", errors="surrogatepass")


def javascript_string(value: object) -> str:
    if value is UNDEFINED:
        return "undefined"
    if value is JSON_NULL:
        return "null"
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _number_string(value)
    if isinstance(value, (list, tuple)):
        return ",".join(
            "" if item is None or item is UNDEFINED or item is JSON_NULL else javascript_string(item)
            for item in value
        )
    if isinstance(value, dict):
        return "[object Object]"
    return str(value)


def javascript_json_stringify(value: object, *, space: str | int | float | None = None) -> str | None:
    """Serialize JSON values with JS spacing, nonfinite numbers, and omissions."""
    active: set[int] = set()

    def normalize(item: object) -> object:
        if item is UNDEFINED:
            return _OMITTED
        if item is JSON_NULL:
            return None
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                return None
            return item
        if callable(item):
            return _OMITTED
        identity = id(item)
        if identity in active:
            raise ValueError("Converting circular structure to JSON")
        active.add(identity)
        try:
            to_json = getattr(item, "to_json", None)
            if callable(to_json):
                return normalize(to_json())
            if isinstance(item, (list, tuple)):
                values = [normalize(element) for element in item]
                return [None if element is _OMITTED else element for element in values]
            if isinstance(item, dict):
                mapping = cast(dict[object, object], item)
            elif is_dataclass(item) and not isinstance(item, type):
                mapping = {field.name: getattr(item, field.name) for field in fields(item)}
            else:
                mapping = vars(item) if hasattr(item, "__dict__") else {}
            result: dict[str, object] = {}
            for key, element in mapping.items():
                normalized = normalize(element)
                if normalized is not _OMITTED:
                    result[javascript_string(key)] = normalized
            return {key: result[key] for key in javascript_object_keys(result)}
        finally:
            active.remove(identity)

    normalized = normalize(value)
    if normalized is _OMITTED:
        return None

    gap = utf16_slice(space, 10) if isinstance(space, str) else ""
    if type(space) in (int, float):
        width = 0 if math.isnan(space) or space <= 0 else 10 if space >= 10 else math.trunc(space)
        gap = " " * width

    def render(item: object, depth: int = 0) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False)
        if isinstance(item, (int, float)):
            return _number_string(item)
        if isinstance(item, list):
            values = [render(element, depth + 1) for element in item]
            if values and gap:
                indent = gap * (depth + 1)
                return "[\n" + indent + (",\n" + indent).join(values) + "\n" + gap * depth + "]"
            return "[" + ",".join(values) + "]"
        mapping = cast(dict[str, object], item)
        separator = ": " if gap else ":"
        values = [f"{render(key)}{separator}{render(element, depth + 1)}" for key, element in mapping.items()]
        if values and gap:
            indent = gap * (depth + 1)
            return "{\n" + indent + (",\n" + indent).join(values) + "\n" + gap * depth + "}"
        return "{" + ",".join(values) + "}"

    text = render(normalized)
    text = text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="surrogatepass")
    return re.sub(r"[\ud800-\udfff]", lambda match: f"\\u{ord(match.group()):04x}", text)
