"""ECMAScript regexp operations used by the native Marked 18.0.11 port.

Patterns without /u run on UTF-16 units; /u patterns run on Unicode scalars
and return UTF-16 offsets. Unicode property ranges come from ICU, as in V8.
Only the regexp syntax and operations used by Marked's pinned rules are exposed.

Marked copyright (c) 2018+, MarkedJS (https://github.com/markedjs/)
Copyright (c) 2011-2018, Christopher Jeffrey (https://github.com/chjj/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Markdown copyright © 2004, John Gruber, http://daringfireball.net/
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name "Markdown" nor the names of its contributors may be used
  to endorse or promote products derived from this software without specific
  prior written permission.

This software is provided by the copyright holders and contributors "as is"
and any express or implied warranties, including, but not limited to, the
implied warranties of merchantability and fitness for a particular purpose
are disclaimed. In no event shall the copyright owner or contributors be
liable for any direct, indirect, incidental, special, exemplary, or
consequential damages (including, but not limited to, procurement of substitute
goods or services; loss of use, data, or profits; or business interruption)
however caused and on any theory of liability, whether in contract, strict
liability, or tort (including negligence or otherwise) arising in any way out
of the use of this software, even if advised of the possibility of such damage.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from icu import UnicodeSet

from ._javascript import JS_WHITESPACE, from_utf16_units, utf16_length, utf16_units


_SPACE = "".join(f"\\u{ord(char):04x}" for char in JS_WHITESPACE)
_PROPERTY_RANGES: dict[str, str] = {}


def _property_ranges(name: str) -> str:
    if name not in _PROPERTY_RANGES:
        characters = UnicodeSet(f"[:{name}:]")
        parts: list[str] = []
        for index in range(characters.getRangeCount()):
            start = ord(characters.getRangeStart(index))
            end = ord(characters.getRangeEnd(index))
            parts.append(f"\\U{start:08x}" if start == end else f"\\U{start:08x}-\\U{end:08x}")
        _PROPERTY_RANGES[name] = "".join(parts)
    return _PROPERTY_RANGES[name]


def _translate(source: str, flags: str) -> str:
    source = re.sub(r"\(\?<([A-Za-z_]\w*)>", r"(?P<\1>", source)
    source = re.sub(r"\\k<([A-Za-z_]\w*)>", r"(?P=\1)", source)
    result: list[str] = []
    in_class = False
    index = 0
    while index < len(source):
        char = source[index]
        if source.startswith(r"[\s\S]", index):
            result.append(r"[\s\S]")
            index += 6
            continue
        if char == "\\" and index + 1 < len(source):
            escaped = source[index + 1]
            if escaped == "s":
                result.append(_SPACE if in_class else "[" + _SPACE + "]")
            elif escaped == "S":
                # Marked only uses standalone \S or the universal [\s\S].
                result.append("[^" + _SPACE + "]")
            elif escaped == "p" and index + 2 < len(source) and source[index + 2] == "{":
                end = source.index("}", index + 3)
                ranges = _property_ranges(source[index + 3:end])
                result.append(ranges if in_class else "[" + ranges + "]")
                index = end + 1
                continue
            else:
                result.append(source[index:index + 2])
            index += 2
            continue
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        if not in_class and char == ".":
            result.append(r"[^\n\r\u2028\u2029]")
        elif not in_class and char == "$":
            result.append(r"(?=[\n\r\u2028\u2029]|\Z)" if "m" in flags else r"\Z")
        elif not in_class and char == "^":
            result.append(r"(?:\A|(?<=[\n\r\u2028\u2029]))" if "m" in flags else r"\A")
        else:
            result.append(char)
        index += 1
    return "".join(result)


@dataclass
class Match:
    values: list[str | None]
    index: int
    end: int
    input: str

    def __getitem__(self, index: int) -> str:
        return self.values[index] or "" if index < len(self.values) else ""

    def __setitem__(self, index: int, value: str) -> None:
        while index >= len(self.values):
            self.values.append(None)
        self.values[index] = value

    def group(self, index: int) -> str | None:
        return self.values[index] if index < len(self.values) else None


class Regex:
    def __init__(self, source: str, flags: str = "") -> None:
        self.source = source
        self.flags = flags
        self.last_index = 0
        self._pattern: re.Pattern[str] | None = None

    def _compile(self) -> re.Pattern[str]:
        if self._pattern is None:
            self._pattern = re.compile(_translate(self.source, self.flags), re.ASCII | (re.IGNORECASE if "i" in self.flags else 0))
        return self._pattern

    def _match(self, native: re.Match[str], original: str) -> Match:
        return Match(
            [None if value is None else utf16_units(value) for value in (native[0], *native.groups())],
            utf16_length(native.string[:native.start()]),
            utf16_length(native.string[:native.end()]),
            utf16_units(original),
        )

    def exec(self, text: str) -> Match | None:
        prepared = from_utf16_units(text) if "u" in self.flags else utf16_units(text)
        start = self.last_index if "g" in self.flags else 0
        if "u" in self.flags and start:
            start = len(from_utf16_units(utf16_units(prepared)[:start]))
        native = self._compile().search(prepared, start)
        if native is None:
            self.last_index = 0
            return None
        match = self._match(native, text)
        if "g" in self.flags:
            self.last_index = match.end
        return match

    def test(self, text: str) -> bool:
        return self.exec(text) is not None

    def finditer(self, text: str) -> Iterator[Match]:
        prepared = from_utf16_units(text) if "u" in self.flags else utf16_units(text)
        for native in self._compile().finditer(prepared):
            yield self._match(native, text)

    def replace(self, text: str, replacement: str | Callable[[Match], str]) -> str:
        units = utf16_units(text)
        parts: list[str] = []
        position = 0
        for match in self.finditer(units):
            parts.append(units[position:match.index])
            if callable(replacement):
                parts.append(utf16_units(replacement(match)))
            else:
                def substitute(token: re.Match[str]) -> str:
                    key = token[1]
                    if key == "$":
                        return "$"
                    if key == "&":
                        return match[0]
                    if key == "`":
                        return units[:match.index]
                    if key == "'":
                        return units[match.end:]
                    number = int(key)
                    if 0 < number < len(match.values):
                        return match[number]
                    if len(key) == 2 and 0 < int(key[0]) < len(match.values):
                        return match[int(key[0])] + key[1]
                    return token[0]
                parts.append(utf16_units(re.sub(r"\$(\$|&|`|'|[0-9]{1,2})", substitute, replacement)))
            position = match.end
            if "g" not in self.flags:
                break
        parts.append(units[position:])
        if "g" in self.flags:
            self.last_index = 0
        return "".join(parts)


class Edit:
    def __init__(self, regex: str | Regex, flags: str = "") -> None:
        self.source = regex.source if isinstance(regex, Regex) else regex
        self.flags = flags

    def replace(self, name: str | Regex, value: str | Regex) -> Edit:
        replacement = value.source if isinstance(value, Regex) else value
        replacement = re.sub(r"(^|[^\[])\^", r"\1", replacement)
        if isinstance(name, Regex):
            self.source = re.sub(name.source, lambda _match: replacement, self.source, count=0 if "g" in name.flags else 1)
        else:
            self.source = self.source.replace(name, replacement, 1)
        return self

    def get_regex(self) -> Regex:
        return Regex(self.source, self.flags)


NOOP = Regex(r"(?!)")
