"""Terminal cell widths, ANSI-preserving layout, and Unicode segmentation."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from icu import UnicodeSet

from ._east_asian_width import east_asian_width
from ._javascript import JS_WHITESPACE, from_utf16_units, js_trim, utf16_length, utf16_units
from ._segmenter import SegmentData, Segmenter, Segments


_grapheme_segmenter = Segmenter("grapheme")
_word_segmenter = Segmenter("word")
_ZERO_WIDTH = UnicodeSet("[[:Default_Ignorable_Code_Point:][:Cc:][:M:][:Cs:]]")
_NON_PRINTING = UnicodeSet("[[:Default_Ignorable_Code_Point:][:Cc:][:Cf:][:M:][:Cs:]]")
_MARK = UnicodeSet("[:M:]")
_SPACING_MARK = UnicodeSet("[:Mc:]")
_RGI_EMOJI = UnicodeSet("[:RGI_Emoji:]")
for _character_set in (_ZERO_WIDTH, _NON_PRINTING, _MARK, _SPACING_MARK, _RGI_EMOJI):
    _character_set.freeze()
_TERMINAL_SPACING_EXCLUSIONS = frozenset("\u1734\u302e\u302f")
_TERMINAL_SPACING_ADDITIONS = frozenset("\u065f\u0f7f\u102b\u102c\u1031\u1033\u1034\u1035\u1038\u103a\u103b\u103c\u103d\u103e")
_WIDTH_CACHE_SIZE = 512
_width_cache: dict[str, int] = {}


class _CjkBreakPattern:
    """The source's Unicode-property regexp with Python search and JS test APIs."""

    _characters = UnicodeSet(
        "[[:scx=Han:][:scx=Hiragana:][:scx=Katakana:][:scx=Hangul:][:scx=Bopomofo:]]"
    )
    _characters.freeze()
    _character_pattern = re.compile(".", re.DOTALL)

    def search(self, text: str) -> re.Match[str] | None:
        text = from_utf16_units(text)
        for index, char in enumerate(text):
            if self._characters.contains(char):
                return self._character_pattern.search(text, index)
        return None

    def test(self, text: str) -> bool:
        return self.search(text) is not None


cjk_break_regex = _CjkBreakPattern()
PUNCTUATION_REGEX = re.compile(r"[(){}\[\]<>.,;:'\"!?+\-=*/\\|&%^$#@~`]")


@dataclass(frozen=True, slots=True)
class AnsiCode:
    code: str
    length: int


@dataclass(frozen=True, slots=True)
class GraphemeCellRange:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TextWithWidth:
    text: str
    width: int


@dataclass(frozen=True, slots=True)
class ExtractedSegments:
    before: str
    before_width: int
    after: str
    after_width: int


def get_grapheme_segmenter() -> Segmenter:
    return _grapheme_segmenter


def get_word_segmenter() -> Segmenter:
    return _word_segmenter


def _is_printable_ascii(text: str) -> bool:
    return all(" " <= char <= "~" for char in text)


def _is_terminal_spacing_mark(char: str) -> bool:
    if 0xD800 <= ord(char) <= 0xDFFF:
        return False
    return (
        char not in _TERMINAL_SPACING_EXCLUSIONS and _SPACING_MARK.contains(char)
    ) or char in _TERMINAL_SPACING_ADDITIONS


def _grapheme_width(segment: str) -> int:
    if segment == "\t":
        return 3
    if segment and all(_is_terminal_spacing_mark(char) for char in segment):
        return len(segment)
    if segment and all(0xD800 <= ord(char) <= 0xDFFF or _ZERO_WIDTH.contains(char) for char in segment):
        return 0

    cp = ord(segment[0]) if segment else -1
    could_be_emoji = (
        0x1F000 <= cp <= 0x1FBFF
        or 0x2300 <= cp <= 0x23FF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B50 <= cp <= 0x2B55
        or "\ufe0f" in segment
        or utf16_length(segment) > 2
    )
    if could_be_emoji and _RGI_EMOJI.contains(segment):
        return 2

    base_index = 0
    while base_index < len(segment) and (
        0xD800 <= ord(segment[base_index]) <= 0xDFFF or _NON_PRINTING.contains(segment[base_index])
    ):
        base_index += 1
    if base_index == len(segment):
        return 0
    base = segment[base_index:]
    cp = ord(base[0])
    if 0x1F1E6 <= cp <= 0x1F1FF:
        return 2
    width = east_asian_width(cp)
    follows_mark = False
    for char in base[1:]:
        if _is_terminal_spacing_mark(char):
            width += 1
            follows_mark = False
        elif _MARK.contains(char):
            follows_mark = True
        elif not (0xD800 <= ord(char) <= 0xDFFF or _NON_PRINTING.contains(char)):
            cp = ord(char)
            if follows_mark or 0xFF00 <= cp <= 0xFFEF:
                width += east_asian_width(cp)
            elif cp in (0x0E33, 0x0EB3):
                width += 1
            follows_mark = False
    return width


def _extract_ansi_units(units: str, pos: int) -> AnsiCode | None:
    if pos < 0 or pos >= len(units) or units[pos] != "\x1b":
        return None
    next_char = units[pos + 1] if pos + 1 < len(units) else ""
    if next_char == "[":
        end = pos + 2
        while end < len(units) and units[end] not in "mGKHJ":
            end += 1
        if end < len(units):
            return AnsiCode(from_utf16_units(units[pos : end + 1]), end + 1 - pos)
    elif next_char in ("]", "_"):
        end = pos + 2
        while end < len(units):
            if units[end] == "\x07":
                return AnsiCode(from_utf16_units(units[pos : end + 1]), end + 1 - pos)
            if units[end : end + 2] == "\x1b\\":
                return AnsiCode(from_utf16_units(units[pos : end + 2]), end + 2 - pos)
            end += 1
    return None


def extract_ansi_code(text: str, pos: int) -> AnsiCode | None:
    """Extract a supported CSI, OSC, or APC sequence at a UTF-16 offset."""
    return _extract_ansi_units(utf16_units(text), pos)


def _iter_terminal_parts(text: str, *, split_tabs: bool = False) -> Iterator[tuple[bool, str]]:
    """Yield ANSI sequences and graphemes, retaining boundaries around ANSI."""
    units = utf16_units(text)
    index = 0
    while index < len(units):
        ansi = _extract_ansi_units(units, index)
        if ansi is not None:
            yield True, ansi.code
            index += ansi.length
            continue
        if split_tabs and units[index] == "\t":
            yield False, "\t"
            index += 1
            continue
        end = index
        while end < len(units):
            if (split_tabs and units[end] == "\t") or _extract_ansi_units(units, end) is not None:
                break
            end += 1
        for item in _grapheme_segmenter.segment(from_utf16_units(units[index:end])):
            yield False, item.segment
        index = end


def visible_width(text: str) -> int:
    """Calculate visible terminal columns, with tabs occupying three cells."""
    if not text:
        return 0
    if _is_printable_ascii(text):
        return len(text)
    cached = _width_cache.get(text)
    if cached is not None:
        return cached
    clean = strip_terminal_sequences(text.replace("\t", "   "))
    width = sum(_grapheme_width(item.segment) for item in _grapheme_segmenter.segment(clean))
    if len(_width_cache) >= _WIDTH_CACHE_SIZE:
        del _width_cache[next(iter(_width_cache))]
    _width_cache[text] = width
    return width


def strip_terminal_sequences(text: str) -> str:
    """Remove supported terminal sequences while preserving visible text."""
    if "\x1b" not in text:
        return text
    units = utf16_units(text)
    result: list[str] = []
    index = 0
    while index < len(units):
        ansi = _extract_ansi_units(units, index)
        if ansi is not None:
            index += ansi.length
        else:
            result.append(units[index])
            index += 1
    return from_utf16_units("".join(result))


def get_grapheme_cell_range(line: str, column: int) -> GraphemeCellRange | None:
    current_col = 0
    for is_ansi, segment in _iter_terminal_parts(line):
        if is_ansi:
            continue
        width = _grapheme_width(segment)
        if width > 0 and current_col <= column < current_col + width:
            return GraphemeCellRange(current_col, current_col + width)
        current_col += width
    return None


_OSC8_URL = re.compile(r"^\x1b\]8;[^;]*;([^\x07\x1b]*)(?:\x07|\x1b\\)$")


def get_osc8_link_at_column(line: str, column: int) -> str | None:
    active_url: str | None = None
    current_col = 0
    for is_ansi, segment in _iter_terminal_parts(line):
        if is_ansi:
            hyperlink = _OSC8_URL.search(segment)
            if hyperlink is not None:
                active_url = hyperlink[1] or None
            continue
        width = _grapheme_width(segment)
        if current_col <= column < current_col + width:
            return active_url
        current_col += width
    return None


def normalize_terminal_output(text: str) -> str:
    """Decompose Thai/Lao AM vowels and expand tabs outside string sequences."""
    normalized = text.replace("\u0e33", "\u0e4d\u0e32").replace("\u0eb3", "\u0ecd\u0eb2")
    if "\t" not in normalized:
        return normalized
    units = utf16_units(normalized)
    result: list[str] = []
    index = 0
    while index < len(units):
        ansi = _extract_ansi_units(units, index)
        if ansi is not None:
            result.append(ansi.code)
            index += ansi.length
        else:
            result.append("   " if units[index] == "\t" else units[index])
            index += 1
    return from_utf16_units("".join(result))


@dataclass(frozen=True, slots=True)
class _ActiveHyperlink:
    params: str
    url: str
    terminator: str


_NOT_OSC8 = object()


def _parse_osc8_hyperlink(ansi_code: str) -> _ActiveHyperlink | None | object:
    if not ansi_code.startswith("\x1b]8;"):
        return _NOT_OSC8
    terminator = "\x07" if ansi_code.endswith("\x07") else "\x1b\\"
    body = ansi_code[4 : -len(terminator)]
    separator_index = body.find(";")
    if separator_index == -1:
        return _NOT_OSC8
    params, url = body[:separator_index], body[separator_index + 1 :]
    return _ActiveHyperlink(params, url, terminator) if url else None


def _format_osc8_hyperlink(hyperlink: _ActiveHyperlink) -> str:
    return f"\x1b]8;{hyperlink.params};{hyperlink.url}{hyperlink.terminator}"


def _format_osc8_close(terminator: str) -> str:
    return f"\x1b]8;;{terminator}"


def _get_active_osc8_close(prefix: str) -> str:
    if "\x1b]8;" not in prefix:
        return ""
    active_hyperlink: _ActiveHyperlink | None = None
    units = utf16_units(prefix)
    index = 0
    while index < len(units):
        ansi = _extract_ansi_units(units, index)
        if ansi is not None:
            hyperlink = _parse_osc8_hyperlink(ansi.code)
            if hyperlink is None or isinstance(hyperlink, _ActiveHyperlink):
                active_hyperlink = hyperlink
            index += ansi.length
        else:
            index += 1
    return _format_osc8_close(active_hyperlink.terminator) if active_hyperlink else ""


class _AnsiCodeTracker:
    def __init__(self) -> None:
        self._attributes: set[int] = set()
        self._fg_color: str | None = None
        self._bg_color: str | None = None
        self._active_hyperlink: _ActiveHyperlink | None = None

    def process(self, ansi_code: str) -> None:
        hyperlink = _parse_osc8_hyperlink(ansi_code)
        if hyperlink is None or isinstance(hyperlink, _ActiveHyperlink):
            self._active_hyperlink = hyperlink
            return
        if not ansi_code.endswith("m"):
            return
        match = re.search(r"\x1b\[([0-9;]*)m", ansi_code)
        if match is None:
            return
        params = match[1]
        if params in ("", "0"):
            self._reset()
            return
        parts = params.split(";")
        index = 0
        while index < len(parts):
            # Only codes through 107 are recognized; avoid Python's integer
            # conversion limit on arbitrarily long JS parseInt parameters.
            normalized = parts[index].lstrip("0") or "0"
            code = int(normalized) if parts[index] and len(normalized) <= 3 else -1
            if code in (38, 48):
                color_length = 0
                if index + 2 < len(parts) and parts[index + 1] == "5":
                    color_length = 3
                elif index + 4 < len(parts) and parts[index + 1] == "2":
                    color_length = 5
                if color_length:
                    color_code = ";".join(parts[index : index + color_length])
                    if code == 38:
                        self._fg_color = color_code
                    else:
                        self._bg_color = color_code
                    index += color_length
                    continue
            if code == 0:
                self._reset()
            elif code in (1, 2, 3, 4, 5, 7, 8, 9):
                self._attributes.add(code)
            elif code == 21:
                self._attributes.discard(1)
            elif code == 22:
                self._attributes.difference_update((1, 2))
            elif code in (23, 24, 25, 27, 28, 29):
                self._attributes.discard(code - 20)
            elif code == 39:
                self._fg_color = None
            elif code == 49:
                self._bg_color = None
            elif 30 <= code <= 37 or 90 <= code <= 97:
                self._fg_color = str(code)
            elif 40 <= code <= 47 or 100 <= code <= 107:
                self._bg_color = str(code)
            index += 1

    def _reset(self) -> None:
        self._attributes.clear()
        self._fg_color = None
        self._bg_color = None
        # SGR reset does not affect OSC 8 hyperlink state.

    def clear(self) -> None:
        self._reset()
        self._active_hyperlink = None

    def get_active_codes(self) -> str:
        codes = [str(code) for code in (1, 2, 3, 4, 5, 7, 8, 9) if code in self._attributes]
        if self._fg_color:
            codes.append(self._fg_color)
        if self._bg_color:
            codes.append(self._bg_color)
        result = f"\x1b[{';'.join(codes)}m" if codes else ""
        if self._active_hyperlink:
            result += _format_osc8_hyperlink(self._active_hyperlink)
        return result

    def get_active_background_code(self) -> str:
        return f"\x1b[{self._bg_color}m" if self._bg_color else ""

    def has_active_codes(self) -> bool:
        return bool(self._attributes) or self._fg_color is not None or self._bg_color is not None or self._active_hyperlink is not None

    def get_line_end_reset(self) -> str:
        result = "\x1b[24m" if 4 in self._attributes else ""
        if self._active_hyperlink:
            result += _format_osc8_close(self._active_hyperlink.terminator)
        return result


def _update_tracker_from_text(text: str, tracker: _AnsiCodeTracker) -> None:
    units = utf16_units(text)
    index = 0
    while index < len(units):
        ansi = _extract_ansi_units(units, index)
        if ansi is not None:
            tracker.process(ansi.code)
            index += ansi.length
        else:
            index += 1


def get_active_background_ansi(text: str) -> str:
    tracker = _AnsiCodeTracker()
    _update_tracker_from_text(text, tracker)
    return tracker.get_active_background_code()


def _split_into_tokens_with_ansi(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    pending_ansi = ""
    current_kind: str | None = None
    for is_ansi, segment in _iter_terminal_parts(text):
        if is_ansi:
            pending_ansi += segment
            continue
        segment_is_space = segment == " "
        if not segment_is_space and cjk_break_regex.test(segment):
            if current:
                tokens.append(current)
                current = ""
                current_kind = None
            tokens.append(pending_ansi + segment)
            pending_ansi = ""
            continue
        segment_kind = "space" if segment_is_space else "word"
        if current and current_kind != segment_kind:
            tokens.append(current)
            current = ""
        current += pending_ansi + segment
        pending_ansi = ""
        current_kind = segment_kind
    if pending_ansi:
        if current:
            current += pending_ansi
        elif tokens:
            tokens[-1] += pending_ansi
        else:
            current = pending_ansi
    if current:
        tokens.append(current)
    return tokens


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """Word-wrap text, carrying SGR and OSC 8 state across physical lines."""
    if not text:
        return [""]
    result: list[str] = []
    tracker = _AnsiCodeTracker()
    for line in re.split(r"\r\n|\r|\n", text):
        prefix = tracker.get_active_codes() if result else ""
        result.extend(_wrap_single_line(prefix + line, width))
        _update_tracker_from_text(line, tracker)
    return result or [""]


def _wrap_single_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    if visible_width(line) <= width:
        return [line]
    wrapped: list[str] = []
    tracker = _AnsiCodeTracker()
    current_line = ""
    current_visible_length = 0
    for token in _split_into_tokens_with_ansi(line):
        token_visible_length = visible_width(token)
        is_whitespace = js_trim(token) == ""
        if token_visible_length > width and not is_whitespace:
            if current_line:
                current_line += tracker.get_line_end_reset()
                wrapped.append(current_line)
                current_line = ""
                current_visible_length = 0
            broken = _break_long_word(token, width, tracker)
            wrapped.extend(broken[:-1])
            current_line = broken[-1]
            current_visible_length = visible_width(current_line)
            continue
        if current_visible_length + token_visible_length > width and current_visible_length > 0:
            wrapped.append(current_line.rstrip(JS_WHITESPACE) + tracker.get_line_end_reset())
            if is_whitespace:
                current_line = tracker.get_active_codes()
                current_visible_length = 0
            else:
                current_line = tracker.get_active_codes() + token
                current_visible_length = token_visible_length
        else:
            current_line += token
            current_visible_length += token_visible_length
        _update_tracker_from_text(token, tracker)
    if current_line:
        wrapped.append(current_line)
    return [part.rstrip(JS_WHITESPACE) for part in wrapped] if wrapped else [""]


def is_whitespace_char(char: str) -> bool:
    return any(value in JS_WHITESPACE for value in char)


def is_punctuation_char(char: str) -> bool:
    return PUNCTUATION_REGEX.search(char) is not None


def _break_long_word(word: str, width: int, tracker: _AnsiCodeTracker) -> list[str]:
    lines: list[str] = []
    current_line = tracker.get_active_codes()
    current_width = 0
    for is_ansi, segment in _iter_terminal_parts(word):
        if is_ansi:
            current_line += segment
            tracker.process(segment)
            continue
        if not segment:
            continue
        grapheme_width = visible_width(segment)
        if current_width + grapheme_width > width:
            lines.append(current_line + tracker.get_line_end_reset())
            current_line = tracker.get_active_codes()
            current_width = 0
        current_line += segment
        current_width += grapheme_width
    if current_line:
        lines.append(current_line)
    return lines or [""]


def apply_background_to_line(line: str, width: int, bg_fn: Callable[[str], str]) -> str:
    return bg_fn(line + " " * max(0, width - visible_width(line)))


def _truncate_fragment_to_width(text: str, max_width: int) -> TextWithWidth:
    if max_width <= 0 or not text:
        return TextWithWidth("", 0)
    if _is_printable_ascii(text):
        clipped = text[:max_width]
        return TextWithWidth(clipped, len(clipped))
    result = ""
    width = 0
    pending_ansi = ""
    for is_ansi, segment in _iter_terminal_parts(text, split_tabs=True):
        if is_ansi:
            pending_ansi += segment
            continue
        segment_width = _grapheme_width(segment)
        if width + segment_width > max_width:
            break
        result += pending_ansi + segment
        pending_ansi = ""
        width += segment_width
    return TextWithWidth(result, width)


def _finalize_truncated_result(
    prefix: str,
    prefix_width: int,
    ellipsis: str,
    ellipsis_width: int,
    max_width: int,
    pad: bool,
) -> str:
    result = prefix + _get_active_osc8_close(prefix) + "\x1b[0m"
    if ellipsis:
        result += ellipsis + "\x1b[0m"
    return result + " " * max(0, max_width - prefix_width - ellipsis_width) if pad else result


def truncate_to_width(text: str, max_width: int, ellipsis: str = "...", pad: bool = False) -> str:
    """Truncate a contiguous grapheme prefix, close its hyperlink, and reset SGR."""
    if max_width <= 0:
        return ""
    if not text:
        return " " * max_width if pad else ""
    ellipsis_width = visible_width(ellipsis)
    if ellipsis_width >= max_width:
        text_width = visible_width(text)
        if text_width <= max_width:
            return text + " " * (max_width - text_width) if pad else text
        clipped = _truncate_fragment_to_width(ellipsis, max_width)
        if clipped.width == 0:
            return " " * max_width if pad else ""
        return _finalize_truncated_result("", 0, clipped.text, clipped.width, max_width, pad)
    target_width = max_width - ellipsis_width
    if _is_printable_ascii(text):
        if len(text) <= max_width:
            return text + " " * (max_width - len(text)) if pad else text
        return _finalize_truncated_result(text[:target_width], target_width, ellipsis, ellipsis_width, max_width, pad)

    result = ""
    pending_ansi = ""
    visible_so_far = 0
    kept_width = 0
    keep_contiguous_prefix = True
    for is_ansi, segment in _iter_terminal_parts(text, split_tabs=True):
        if is_ansi:
            pending_ansi += segment
            continue
        width = _grapheme_width(segment)
        if keep_contiguous_prefix and kept_width + width <= target_width:
            result += pending_ansi + segment
            pending_ansi = ""
            kept_width += width
        else:
            keep_contiguous_prefix = False
            pending_ansi = ""
        visible_so_far += width
        if visible_so_far > max_width:
            return _finalize_truncated_result(result, kept_width, ellipsis, ellipsis_width, max_width, pad)
    return text + " " * max(0, max_width - visible_so_far) if pad else text


def slice_by_column(line: str, start_col: int, length: int, strict: bool = False) -> str:
    return slice_with_width(line, start_col, length, strict).text


def slice_with_width(line: str, start_col: int, length: int, strict: bool = False) -> TextWithWidth:
    """Extract columns, optionally excluding a wide grapheme crossing the end."""
    if length <= 0:
        return TextWithWidth("", 0)
    end_col = start_col + length
    result = ""
    result_width = 0
    current_col = 0
    pending_ansi = ""
    for is_ansi, segment in _iter_terminal_parts(line):
        if is_ansi:
            if start_col <= current_col < end_col:
                result += segment
            elif current_col < start_col:
                pending_ansi += segment
            continue
        width = _grapheme_width(segment)
        in_range = start_col <= current_col < end_col
        fits = not strict or current_col + width <= end_col
        if in_range and fits:
            result += pending_ansi + segment
            pending_ansi = ""
            result_width += width
        current_col += width
        if current_col >= end_col:
            break
    return TextWithWidth(result, result_width)


_pooled_style_tracker = _AnsiCodeTracker()


def extract_segments(
    line: str, before_end: int, after_start: int, after_len: int, strict_after: bool = False
) -> ExtractedSegments:
    """Extract both sides of an overlay and carry active styles to its right."""
    before = ""
    before_width = 0
    after = ""
    after_width = 0
    current_col = 0
    pending_ansi_before = ""
    after_started = False
    after_end = after_start + after_len
    _pooled_style_tracker.clear()
    for is_ansi, segment in _iter_terminal_parts(line):
        if is_ansi:
            _pooled_style_tracker.process(segment)
            if current_col < before_end:
                pending_ansi_before += segment
            elif after_start <= current_col < after_end and after_started:
                after += segment
            continue
        width = _grapheme_width(segment)
        if current_col < before_end and current_col + width <= before_end:
            before += pending_ansi_before + segment
            pending_ansi_before = ""
            before_width += width
        elif after_start <= current_col < after_end:
            if not strict_after or current_col + width <= after_end:
                if not after_started:
                    after += _pooled_style_tracker.get_active_codes()
                    after_started = True
                after += segment
                after_width += width
        current_col += width
        if current_col >= (before_end if after_len <= 0 else after_end):
            break
    return ExtractedSegments(before, before_width, after, after_width)


__all__ = [
    "AnsiCode", "ExtractedSegments", "GraphemeCellRange", "PUNCTUATION_REGEX",
    "SegmentData", "Segmenter", "Segments", "TextWithWidth", "apply_background_to_line",
    "cjk_break_regex", "extract_ansi_code", "extract_segments", "get_active_background_ansi",
    "get_grapheme_cell_range", "get_grapheme_segmenter", "get_osc8_link_at_column",
    "get_word_segmenter", "is_punctuation_char", "is_whitespace_char",
    "normalize_terminal_output", "slice_by_column", "slice_with_width",
    "strip_terminal_sequences", "truncate_to_width", "visible_width", "wrap_text_with_ansi",
]
