"""Native port of partial-json 0.1.7's parser, including its recovery rules.

The parser deliberately retains the dependency's permissive object grammar,
lowercase-exponent fallback, and NaN/Infinity handling. Its internal positions
are UTF-16 code units. Parse failures use Python exceptions and error messages.

MIT License

Copyright (c) 2023 Promplate Dev Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from typing import NoReturn, cast

from ._json_runtime import JS_WHITESPACE, parse_json, utf16_units
from .types import JsonValue

STR = 0b000000001
NUM = 0b000000010
ARR = 0b000000100
OBJ = 0b000001000
NULL = 0b000010000
BOOL = 0b000100000
NAN = 0b001000000
INFINITY = 0b010000000
_INFINITY = 0b100000000
INF = INFINITY | _INFINITY
SPECIAL = NULL | BOOL | INF | NAN
ATOM = STR | NUM | SPECIAL
COLLECTION = ARR | OBJ
ALL = ATOM | COLLECTION


class PartialJSON(ValueError):
    pass


class MalformedJSON(ValueError):
    pass


class _ObjectPrototype:
    __slots__ = ()


_OBJECT_PROTOTYPE = _ObjectPrototype()


class _PartialObject(dict[str, JsonValue]):
    """Own JSON fields, with prototype state retained for property assignment.

    In particular, assigning __proto__ after a null prototype creates an own
    field, whereas assigning it on a normal object invokes the inherited setter.
    Python mapping iteration and serialization expose only the own JSON fields.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prototype: dict[str, JsonValue] | list[JsonValue] | _ObjectPrototype | None = _OBJECT_PROTOTYPE

    def assign(self, key: str, value: JsonValue) -> None:
        if key == "__proto__" and key not in self:
            prototype = self._prototype
            while isinstance(prototype, dict) and key not in prototype:
                prototype = prototype._prototype if isinstance(prototype, _PartialObject) else _OBJECT_PROTOTYPE
            if prototype is _OBJECT_PROTOTYPE or isinstance(prototype, list):
                if value is None or isinstance(value, (dict, list)):
                    self._prototype = value
                return
        self[key] = value


class _Parser:
    def __init__(self, source: str, allow: int) -> None:
        self.source = source
        self.length = len(source)
        self.index = 0
        self.allow = allow

    def _partial(self, message: str) -> NoReturn:
        raise PartialJSON(f"{message} at position {self.index}")

    def _malformed(self, message: str) -> NoReturn:
        raise MalformedJSON(f"{message} at position {self.index}")

    def _char(self, index: int | None = None) -> str:
        offset = self.index if index is None else index
        return self.source[offset] if 0 <= offset < self.length else ""

    def _substring(self, start: int, end: int | None = None) -> str:
        start = max(0, min(start, self.length))
        end = self.length if end is None else max(0, min(end, self.length))
        if start > end:
            start, end = end, start
        return self.source[start:end]

    def _skip_blank(self) -> None:
        while self.index < self.length and self.source[self.index] in " \n\r\t":
            self.index += 1

    def parse_any(self) -> JsonValue:
        self._skip_blank()
        if self.index >= self.length:
            self._partial("Unexpected end of input")
        char = self._char()
        if char == '"':
            return self._parse_string()
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        remaining = self.length - self.index
        for literal, flag, value in (
            ("null", NULL, None),
            ("true", BOOL, True),
            ("false", BOOL, False),
            ("Infinity", INFINITY, math.inf),
            ("-Infinity", _INFINITY, -math.inf),
            ("NaN", NAN, math.nan),
        ):
            if self._substring(self.index, self.index + len(literal)) == literal or (
                self.allow & flag
                and remaining < len(literal)
                and (literal != "-Infinity" or remaining > 1)
                and literal.startswith(self._substring(self.index))
            ):
                self.index += len(literal)
                return value
        return self._parse_number()

    def _parse_string(self) -> str:
        start = self.index
        escape = False
        self.index += 1
        while self.index < self.length and (
            self._char() != '"' or (escape and self._char(self.index - 1) == "\\")
        ):
            escape = not escape if self._char() == "\\" else False
            self.index += 1
        if self._char() == '"':
            self.index += 1
            try:
                return cast(str, parse_json(self._substring(start, self.index - int(escape))))
            except Exception as error:
                self._malformed(str(error))
        elif self.allow & STR:
            try:
                return cast(str, parse_json(self._substring(start, self.index - int(escape)) + '"'))
            except Exception:
                return cast(str, parse_json(self._substring(start, self.source.rfind("\\")) + '"'))
        self._partial("Unterminated string literal")

    def _parse_object(self) -> dict[str, JsonValue]:
        self.index += 1
        self._skip_blank()
        result = _PartialObject()
        try:
            while self._char() != "}":
                self._skip_blank()
                if self.index >= self.length and self.allow & OBJ:
                    return result
                key = self._parse_string()
                self._skip_blank()
                self.index += 1  # The dependency skips the colon without validating it.
                try:
                    value = self.parse_any()
                    result.assign(key, value)
                except Exception:
                    if self.allow & OBJ:
                        return result
                    raise
                self._skip_blank()
                if self._char() == ",":
                    self.index += 1
        except Exception:
            if self.allow & OBJ:
                return result
            self._partial("Expected '}' at end of object")
        self.index += 1
        return result

    def _parse_array(self) -> list[JsonValue]:
        self.index += 1
        result: list[JsonValue] = []
        try:
            while self._char() != "]":
                result.append(self.parse_any())
                self._skip_blank()
                if self._char() == ",":
                    self.index += 1
        except Exception:
            if self.allow & ARR:
                return result
            self._partial("Expected ']' at end of array")
        self.index += 1
        return result

    def _parse_number(self) -> JsonValue:
        if self.index == 0:
            if self.source == "-":
                self._malformed("Not sure what '-' is")
            try:
                return parse_json(self.source)
            except Exception as error:
                if self.allow & NUM:
                    try:
                        return parse_json(self._substring(0, self.source.rfind("e")))
                    except Exception:
                        pass
                self._malformed(str(error))
        start = self.index
        if self._char() == "-":
            self.index += 1
        while self._char() and self._char() not in ",]}":
            self.index += 1
        if self.index == self.length and not self.allow & NUM:
            self._partial("Unterminated number literal")
        try:
            return parse_json(self._substring(start, self.index))
        except Exception:
            if self._substring(start, self.index) == "-":
                self._partial("Not sure what '-' is")
            try:
                return parse_json(self._substring(start, self.source.rfind("e")))
            except Exception as error:
                self._malformed(str(error))


def partial_parse(text: str, allow_partial: int = ALL) -> JsonValue:
    if not isinstance(text, str):
        raise TypeError(f"expecting str, got {type(text).__name__}")
    trimmed = text.strip(JS_WHITESPACE)
    if not trimmed:
        raise ValueError(f"{text} is empty")
    return _Parser(utf16_units(trimmed), allow_partial).parse_any()
