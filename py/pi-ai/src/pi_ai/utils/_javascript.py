"""Shared JavaScript semantics; implementation lives below the utils package."""

from .._javascript import (
    _number_string,
    javascript_json_parse,
    javascript_json_stringify,
    javascript_object_keys,
    javascript_string,
    utf16_length,
    utf16_slice,
)

__all__ = [
    "javascript_json_parse",
    "javascript_json_stringify",
    "javascript_object_keys",
    "javascript_string",
    "utf16_length",
    "utf16_slice",
]
