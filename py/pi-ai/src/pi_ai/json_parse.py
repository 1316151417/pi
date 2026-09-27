"""Streaming JSON parsing ported from pi-ai ``src/utils/json-parse.ts``."""

from __future__ import annotations

import re

from ._json_runtime import JS_WHITESPACE, parse_json
from ._partial_json import partial_parse
from .types import JsonValue

__all__ = ["repair_json", "parse_json_with_repair", "parse_streaming_json", "partial_parse"]

VALID_JSON_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
_UNICODE_ESCAPE = re.compile(r"[0-9a-fA-F]{4}")


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
                if _UNICODE_ESCAPE.fullmatch(unicode_digits):
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
        repaired += _escape_control_character(char) if 0x00 <= ord(char) <= 0x1F else char
        index += 1
    return repaired


def parse_json_with_repair(text: str) -> JsonValue:
    try:
        return parse_json(text)
    except Exception:
        repaired = repair_json(text)
        if repaired != text:
            return parse_json(repaired)
        raise


def parse_streaming_json(partial_json: str | None) -> JsonValue:
    """Parse a streaming JSON value, or return an empty object if recovery fails."""
    if not partial_json or partial_json.strip(JS_WHITESPACE) == "":
        return {}
    try:
        return parse_json_with_repair(partial_json)
    except Exception:
        try:
            result = partial_parse(partial_json)
            return result if result is not None else {}
        except Exception:
            try:
                result = partial_parse(repair_json(partial_json))
                return result if result is not None else {}
            except Exception:
                return {}
