"""TypeBox 1.3.27 guards, JSON-schema keyword guards, and value hashing.

Ported from typebox/build/guard and system/hashing. See licenses/typebox-LICENSE.txt.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping
from typing import TypeGuard, cast

from ._javascript import javascript_object_keys
from ._json_runtime import JS_WHITESPACE, utf16_units
from ._values import UNDEFINED

type Schema = bool | dict[str, object]


class _ObjectMethod:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: object) -> object:
        raise TypeError(f"Object.prototype.{self.name} is an inherited JavaScript method")


_OBJECT_PROTOTYPE: dict[str, object] = {
    name: _ObjectMethod(name) for name in (
        "constructor", "__defineGetter__", "__defineSetter__", "hasOwnProperty",
        "__lookupGetter__", "__lookupSetter__", "isPrototypeOf", "propertyIsEnumerable",
        "toString", "valueOf", "toLocaleString",
    )
}


def has_property(value: dict[str, object], key: str, unsafe_guard: bool = True) -> bool:
    if key in value:
        return True
    if unsafe_guard and key in ("__proto__", "constructor", "prototype"):
        return False
    return key == "__proto__" or key in _OBJECT_PROTOTYPE


def get_property(value: dict[str, object], key: str) -> object:
    if key in value:
        return value[key]
    if key == "__proto__":
        return _OBJECT_PROTOTYPE
    return _OBJECT_PROTOTYPE.get(key, UNDEFINED)


def structured_clone(value: object) -> object:
    """Clone JSON data, dropping custom dictionary prototypes like structuredClone."""
    memo: dict[int, object] = {}

    def clone(item: object) -> object:
        if item is None or item is UNDEFINED or isinstance(item, (str, bool, int, float)):
            return item
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, list):
            array: list[object] = []
            memo[id(item)] = array
            array.extend(clone(child) for child in item)
            return array
        if isinstance(item, dict):
            record: dict[str, object] = {}
            memo[id(item)] = record
            record.update((key, clone(child)) for key, child in entries(item))
            return record
        raise TypeError(f"{type(item).__name__} could not be cloned as JSON tool arguments")

    return clone(value)


def is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_integer(value: object) -> TypeGuard[int | float]:
    return is_number(value) and (isinstance(value, int) or value.is_integer())


def strict_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return utf16_units(left) == utf16_units(right)
    return left is right


def deep_equal(left: object, right: object) -> bool:
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            deep_equal(item, right[index]) for index, item in enumerate(left)
        )
    if isinstance(left, dict):
        if not isinstance(right, (dict, list)):
            return False
        right_keys = list(right) if isinstance(right, dict) else [*(str(index) for index in range(len(right))), "length"]
        if len(left) != len(right_keys):
            return False
        for key, item in left.items():
            if isinstance(right, dict):
                other = get_property(right, key)
            elif key == "length":
                other = len(right)
            elif isinstance(key, str) and len(key) < 11 and key.isascii() and key.isdigit() and str(int(key)) == key:
                index = int(key)
                other = right[index] if index < len(right) else UNDEFINED
            else:
                other = UNDEFINED
            if not deep_equal(item, other):
                return False
        return True
    return strict_equal(left, right)


_DECIMAL_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def number_from_string(value: str) -> float:
    value = value.strip(JS_WHITESPACE)
    if not value:
        return 0.0
    if value in ("Infinity", "+Infinity", "-Infinity"):
        return -math.inf if value.startswith("-") else math.inf
    for prefix, digits, radix in (("0x", "0123456789abcdef", 16), ("0b", "01", 2), ("0o", "01234567", 8)):
        if value.lower().startswith(prefix):
            tail = value[2:].lower()
            if not tail or any(char not in digits for char in tail):
                return math.nan
            try:
                return float(int(tail, radix))
            except OverflowError:
                return math.inf
    return float(value) if _DECIMAL_NUMBER.fullmatch(value) else math.nan


def is_schema(value: object) -> TypeGuard[Schema]:
    return isinstance(value, (bool, dict))


def schema_keyword(schema: object, key: str) -> bool:
    if not isinstance(schema, Mapping) or key not in schema:
        return False
    value = schema[key]
    if key == "const":
        return True
    if key in {"$id", "$anchor", "$ref", "$dynamicAnchor", "$dynamicRef", "$recursiveRef", "format"}:
        return isinstance(value, str)
    if key in {"$recursiveAnchor", "uniqueItems"}:
        return isinstance(value, bool)
    if key in {"maxContains", "maxItems", "maxLength", "maxProperties", "minContains", "minItems", "minLength", "minProperties", "maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum", "multipleOf"}:
        return is_number(value)
    if key in {"additionalItems", "additionalProperties", "contains", "if", "then", "else", "not", "unevaluatedItems", "unevaluatedProperties"}:
        return is_schema(value)
    if key == "propertyNames":
        return is_schema(value) or isinstance(value, list)
    if key in {"allOf", "anyOf", "oneOf", "prefixItems"}:
        return isinstance(value, list) and all(is_schema(item) for item in value)
    if key == "items":
        return is_schema(value) or isinstance(value, list) and all(is_schema(item) for item in value)
    if key in {"enum", "required"}:
        return isinstance(value, list) and (key == "enum" or all(isinstance(item, str) for item in value))
    if key == "type":
        return isinstance(value, str) or isinstance(value, list) and all(isinstance(item, str) for item in value)
    if key in {"properties", "patternProperties", "dependentSchemas", "dependencies", "dependentRequired"}:
        if not isinstance(value, (dict, list)):
            return False
        values = value.values() if isinstance(value, dict) else value
        if key == "dependentRequired":
            return all(isinstance(item, list) and all(isinstance(name, str) for name in item) for item in values)
        if key == "dependencies":
            return all(is_schema(item) or isinstance(item, list) and all(isinstance(name, str) for name in item) for item in values)
        return all(is_schema(item) for item in values)
    if key == "pattern":
        return isinstance(value, str) or callable(getattr(value, "search", None)) or callable(getattr(value, "test", None))
    if key == "~refine":
        return isinstance(value, list) and all(
            isinstance(item, dict) and callable(item.get("check")) and callable(item.get("error"))
            for item in value
        )
    return False


def entries(value: object) -> list[tuple[str, object]]:
    if isinstance(value, dict):
        record = cast(dict[str, object], value)
        return [(key, record[key]) for key in javascript_object_keys(record)]
    if isinstance(value, list):
        return [(str(index), item) for index, item in enumerate(value)]
    return []


def grapheme_count(value: str) -> int:
    # This is TypeBox's own algorithm, deliberately not Intl.Segmenter or \X.
    value = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    count = 0
    index = 0
    while index < len(value):
        first = ord(value[index])
        index += 1
        while index < len(value):
            point = ord(value[index])
            if (0x0300 <= point <= 0x036F or 0x1AB0 <= point <= 0x1AFF or 0x1DC0 <= point <= 0x1DFF
                    or 0xFE20 <= point <= 0xFE2F or 0xFE00 <= point <= 0xFE0F):
                index += 1
            elif point == 0x200D and index + 1 < len(value):
                index += 2
            else:
                break
        if 0x1F1E6 <= first <= 0x1F1FF and index < len(value) and 0x1F1E6 <= ord(value[index]) <= 0x1F1FF:
            index += 1
        count += 1
    return count


def value_hash(value: object) -> int:
    accumulator = 14695981039346656037

    def write(byte: int) -> None:
        nonlocal accumulator
        accumulator = ((accumulator ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    def visit(item: object) -> None:
        if item is UNDEFINED:
            write(13)
        elif item is None:
            write(6)
        elif isinstance(item, bool):
            write(2)
            write(int(item))
        elif isinstance(item, (int, float)):
            write(7)
            for byte in struct.pack("<d", item):
                write(byte)
        elif isinstance(item, str):
            write(10)
            scalar = item.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
            for byte in scalar.encode("utf-8"):
                write(byte)
        elif isinstance(item, list):
            write(0)
            for element in item:
                visit(element)
        elif isinstance(item, dict):
            write(8)
            for key in sorted((key for key in item if key != "constructor"), key=utf16_units):
                visit(key)
                visit(item[key])
        else:
            raise TypeError(f"Cannot hash non-JSON TypeBox value {type(item).__name__}")

    visit(value)
    return accumulator
