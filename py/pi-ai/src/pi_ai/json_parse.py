"""Streaming JSON parsing ported from pi-ai ``src/utils/json-parse.ts``."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

__all__ = ["repair_json", "parse_json_with_repair", "parse_streaming_json", "partial_parse"]

VALID_JSON_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
_UNICODE_ESCAPE = re.compile(r"^[0-9a-fA-F]{4}$")


def _is_control_character(char: str) -> bool:
    return 0x00 <= ord(char) <= 0x1F


def _escape_control_character(char: str) -> str:
    mapping = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    if char in mapping:
        return mapping[char]
    return f"\\u{ord(char):04x}"


def repair_json(text: str) -> str:
    """Repair malformed JSON string literals: escape control chars, double invalid escapes."""
    repaired = ""
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            repaired += char
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == '"':
            repaired += char
            in_string = False
            index += 1
            continue
        if char == "\\":
            next_char = text[index + 1] if index + 1 < len(text) else None
            if next_char is None:
                repaired += "\\\\"
                index += 1
                continue
            if next_char == "u":
                unicode_digits = text[index + 2 : index + 6]
                if _UNICODE_ESCAPE.match(unicode_digits):
                    repaired += f"\\u{unicode_digits}"
                    index += 6
                    continue
            if next_char in VALID_JSON_ESCAPES:
                repaired += f"\\{next_char}"
                index += 2
                continue
            repaired += "\\\\"
            index += 1
            continue
        repaired += _escape_control_character(char) if _is_control_character(char) else char
        index += 1
    return repaired


def parse_json_with_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        if repaired != text:
            return json.loads(repaired)
        raise


# ---------------------------------------------------------------------------
# Partial (streaming) JSON
# ---------------------------------------------------------------------------


def _scan_string(text: str, start: int) -> tuple:
    """Return ``(end_index, closed)`` for a string starting at ``start`` (the quote)."""
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1, True
        index += 1
    return len(text), False


def _partial_parse_value(text: str, index: int) -> tuple:
    """Parse one value at ``index``; returns ``(value, next_index)`` for valid prefixes."""
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index >= len(text):
        raise ValueError("Unexpected end of input")

    char = text[index]
    if char == "{":
        return _partial_parse_object(text, index)
    if char == "[":
        return _partial_parse_array(text, index)
    if char == '"':
        end, closed = _scan_string(text, index)
        raw = text[index:end]
        if not closed:
            raw += '"'
        try:
            return json.loads(repair_json(raw)), end
        except json.JSONDecodeError:
            raise ValueError("Invalid string")
    match = re.match(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text[index:])
    if match:
        literal = match.group(0)
        return (float(literal) if ("." in literal or "e" in literal.lower()) else int(literal)), index + len(literal)
    for literal, value in (("true", True), ("false", False), ("null", None)):
        if text.startswith(literal, index):
            return value, index + len(literal)
    raise ValueError(f"Unexpected character: {char!r}")


def _partial_parse_object(text: str, index: int) -> tuple:
    index += 1  # skip {
    result: Dict[str, Any] = {}
    while True:
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index >= len(text):
            return result, index
        if text[index] == "}":
            return result, index + 1
        if text[index] == ",":
            index += 1
            continue
        if text[index] != '"':
            return result, index
        end, closed = _scan_string(text, index)
        if not closed:
            return result, len(text)
        key = json.loads(repair_json(text[index:end]))
        index = end
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index >= len(text) or text[index] != ":":
            return result, index
        index += 1
        try:
            value, index = _partial_parse_value(text, index)
        except ValueError:
            return result, index
        result[key] = value


def _partial_parse_array(text: str, index: int) -> tuple:
    index += 1  # skip [
    result = []
    while True:
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index >= len(text):
            return result, index
        if text[index] == "]":
            return result, index + 1
        if text[index] == ",":
            index += 1
            continue
        try:
            value, index = _partial_parse_value(text, index)
        except ValueError:
            return result, index
        result.append(value)


def partial_parse(text: str) -> Any:
    """Parse a possibly-incomplete JSON document, returning the valid prefix."""
    value, _index = _partial_parse_value(text, 0)
    return value


def parse_streaming_json(partial_json: Optional[str]) -> Any:
    """Parse potentially incomplete JSON during streaming; always returns a valid object."""
    if not partial_json or partial_json.strip() == "":
        return {}
    try:
        return parse_json_with_repair(partial_json)
    except (json.JSONDecodeError, ValueError):
        try:
            result = partial_parse(partial_json)
            return result if result is not None else {}
        except (json.JSONDecodeError, ValueError):
            try:
                result = partial_parse(repair_json(partial_json))
                return result if result is not None else {}
            except (json.JSONDecodeError, ValueError):
                return {}
