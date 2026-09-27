"""Provider-compatible string enum schemas from ``utils/typebox-helpers.ts``.

The returned object is ordinary JSON Schema. Literal types are retained in its
Python type parameter without requiring TypeBox's JavaScript runtime markers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

__all__ = ["StringEnumOptions", "StringEnumSchema", "string_enum"]


class StringEnumOptions[T: str](TypedDict, total=False):
    description: str
    default: T


class StringEnumSchema[T: str](TypedDict):
    type: Literal["string"]
    enum: list[T]
    description: NotRequired[str]
    default: NotRequired[T]


def string_enum[T: str](values: Sequence[T], options: StringEnumOptions[T] | None = None) -> StringEnumSchema[T]:
    """Use ``type`` and ``enum`` rather than ``anyOf``/``const`` alternatives.

    Empty description/default strings are omitted, matching the source's
    truthiness checks. A supplied list remains the schema's enum list; other
    Python sequences are converted to JSON arrays.
    """
    schema: StringEnumSchema[T] = {"type": "string", "enum": values if isinstance(values, list) else list(values)}
    if options is not None:
        description = options.get("description")
        default = options.get("default")
        if description:
            schema["description"] = description
        if default:
            schema["default"] = default
    return schema
