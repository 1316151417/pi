"""Rendered-transcript search with source-cell mapping and an input overlay."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from icu import Char

from ._javascript import JS_WHITESPACE, from_utf16_units, js_repeat, js_trim, utf16_length
from .components.input import Input, InputOptions
from .keybindings import get_keybindings
from .utils import get_grapheme_segmenter, strip_terminal_sequences, truncate_to_width, visible_width

_segmenter = get_grapheme_segmenter()
_WHITESPACE = re.compile("[" + re.escape(JS_WHITESPACE) + "]+")


@dataclass
class _SearchSourceSpan:
    text_start: int
    text_end: int
    row: int
    start_col: int
    end_col: int
    linear_columns: bool


@dataclass
class _SearchCorpus:
    text: str
    spans: list[_SearchSourceSpan]


@dataclass
class AltScreenSearchSegment:
    row: int
    start_col: int
    end_col: int


@dataclass
class AltScreenSearchMatch:
    segments: list[AltScreenSearchSegment]


@dataclass
class AltScreenSearchResult:
    matches: list[AltScreenSearchMatch]
    changed: bool


def _build_search_corpus(lines: Sequence[str]) -> _SearchCorpus:
    chunks: list[str] = []
    spans: list[_SearchSourceSpan] = []
    text_length = 0
    pending_separator = False
    for row, source in enumerate(lines):
        line = strip_terminal_sequences(source)
        column = 0
        if all(" " <= char <= "~" for char in line):
            index = 0
            while index < len(line):
                if line[index] == " ":
                    if text_length > 0:
                        pending_separator = True
                    column += 1
                    index += 1
                    continue
                end = index + 1
                while end < len(line) and line[end] != " ":
                    end += 1
                if pending_separator:
                    chunks.append(" ")
                    text_length += 1
                    pending_separator = False
                text = line[index:end]
                chunks.append(text)
                spans.append(_SearchSourceSpan(text_length, text_length + len(text), row, column, column + len(text), True))
                text_length += len(text)
                column += len(text)
                index = end
        else:
            for grapheme in _segmenter.segment(line):
                text = grapheme.segment
                width = visible_width(text)
                if text and all(char in JS_WHITESPACE for char in text):
                    if text_length > 0:
                        pending_separator = True
                    column += width
                    continue
                if pending_separator:
                    chunks.append(" ")
                    text_length += 1
                    pending_separator = False
                chunks.append(text)
                length = utf16_length(text)
                spans.append(_SearchSourceSpan(text_length, text_length + length, row, column, column + width, False))
                text_length += length
                column += width
        if text_length > 0:
            pending_separator = True
    return _SearchCorpus("".join(chunks), spans)


def _find_search_corpus_matches(corpus: _SearchCorpus, normalized_query: str) -> list[AltScreenSearchMatch]:
    if not normalized_query:
        return []
    # /giu uses Unicode simple case folding, not full folds such as ß -> ss.
    text = from_utf16_units(corpus.text)
    query = from_utf16_units(normalized_query)
    folded_text = "".join(chr(Char.foldCase(ord(char))) for char in text)
    folded_query = "".join(chr(Char.foldCase(ord(char))) for char in query)
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + utf16_length(char))
    matches: list[AltScreenSearchMatch] = []
    span_index = 0
    search_from = 0
    while True:
        found = folded_text.find(folded_query, search_from)
        if found == -1:
            break
        match_end = found + len(folded_query)
        start, end = offsets[found], offsets[match_end]
        search_from = match_end
        while span_index < len(corpus.spans) and corpus.spans[span_index].text_end <= start:
            span_index += 1
        segments: list[AltScreenSearchSegment] = []
        for span in corpus.spans[span_index:]:
            if span.text_start >= end:
                break
            if span.text_end <= start:
                continue
            start_col = span.start_col + max(start, span.text_start) - span.text_start if span.linear_columns else span.start_col
            end_col = span.start_col + min(end, span.text_end) - span.text_start if span.linear_columns else span.end_col
            previous = segments[-1] if segments else None
            if previous is not None and previous.row == span.row and start_col <= previous.end_col:
                previous.end_col = max(previous.end_col, end_col)
            else:
                segments.append(AltScreenSearchSegment(span.row, start_col, end_col))
        while span_index < len(corpus.spans) and corpus.spans[span_index].text_end <= end:
            span_index += 1
        if segments:
            matches.append(AltScreenSearchMatch(segments))
    return matches


class AltScreenSearchIndex:
    def __init__(self) -> None:
        self._source_lines: list[str] | None = None
        self._corpus: _SearchCorpus | None = None
        self._normalized_query: str | None = None
        self._matches: list[AltScreenSearchMatch] = []

    def search(self, lines: Sequence[str], query: str) -> AltScreenSearchResult:
        source_changed = self._source_lines is None or len(self._source_lines) != len(lines)
        if not source_changed and self._source_lines is not None:
            source_changed = any(previous != current for previous, current in zip(self._source_lines, lines))
        if source_changed or self._corpus is None:
            self._source_lines = list(lines)
            self._corpus = _build_search_corpus(lines)
        normalized = js_trim(_WHITESPACE.sub(" ", query))
        changed = source_changed or normalized != self._normalized_query
        if changed:
            self._normalized_query = normalized
            self._matches = _find_search_corpus_matches(self._corpus, normalized)
        return AltScreenSearchResult(self._matches, changed)


def find_alt_screen_search_matches(lines: Sequence[str], query: str) -> list[AltScreenSearchMatch]:
    normalized = js_trim(_WHITESPACE.sub(" ", query))
    return _find_search_corpus_matches(_build_search_corpus(lines), normalized) if normalized else []


def get_alt_screen_search_match_key(match: AltScreenSearchMatch) -> str:
    if not match.segments:
        return ""
    first, last = match.segments[0], match.segments[-1]
    return f"{first.row}:{first.start_col}:{last.row}:{last.end_col}"


class AltScreenSearchComponent:
    def __init__(
        self, on_query_change: Callable[[str], None],
        navigation_button_style: Callable[[str, bool], str] | None = None,
    ) -> None:
        self._input = Input(InputOptions(
            prompt=" ", placeholder="Find in transcript", placeholder_style=lambda text: f"\x1b[2m{text}\x1b[22m",
        ))
        self._on_query_change = on_query_change
        self._navigation_button_style = navigation_button_style or (lambda text, hovered: text)
        self._result_count = 0
        self._result_index = -1
        self._previous_button_start = -1
        self._previous_button_end = -1
        self._next_button_start = -1
        self._next_button_end = -1
        self._hovered_navigation_direction: Literal[-1, 1] | None = None
        self._focused = False

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    def set_result(self, index: int, count: int) -> None:
        self._result_index = index
        self._result_count = count

    def get_navigation_direction_at(self, row: int, column: int) -> Literal[-1, 1] | None:
        if row != 2:
            return None
        if self._previous_button_start <= column < self._previous_button_end:
            return -1
        if self._next_button_start <= column < self._next_button_end:
            return 1
        return None

    def set_hovered_navigation_direction(self, direction: Literal[-1, 1] | None) -> bool:
        if direction == self._hovered_navigation_direction:
            return False
        self._hovered_navigation_direction = direction
        return True

    def handle_input(self, data: str) -> None:
        previous = self._input.get_value()
        self._input.handle_input(data)
        query = self._input.get_value()
        if query != previous:
            self._on_query_change(query)

    def invalidate(self) -> None:
        self._input.invalidate()

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        inner_width = max(0, safe_width - 2)

        def format_key(keys: list[str]) -> str:
            if not keys or not keys[0]:
                return "Unbound"
            return "+".join(
                "Option" if sys.platform == "darwin" and part.lower() == "alt" else part[:1].upper() + part[1:]
                for part in keys[0].split("+")
            )

        kb = get_keybindings()
        previous_key = format_key(kb.get_keys("tui.altScreen.searchPrevious"))
        next_key = format_key(kb.get_keys("tui.altScreen.searchNext"))
        query = self._input.get_value()
        result = "" if not query else "No matches" if self._result_count == 0 else f"{self._result_index + 1}/{self._result_count}"
        visible_result = truncate_to_width(result, max(0, inner_width - 3), "")
        result_text = f"\x1b[2m {visible_result} \x1b[22m" if visible_result else ""
        input_width = max(0, inner_width - visible_width(result_text))
        rendered_input = self._input.render(max(1, input_width))
        input_line = truncate_to_width(rendered_input[0] if rendered_input else "", input_width, "")
        content = input_line + js_repeat(" ", max(0, input_width - visible_width(input_line))) + result_text
        previous_button = f"↑ {previous_key}"
        next_button = f"↓ {next_key}"
        separator = " · "
        outer_gap_width = 1
        available = max(0, inner_width - outer_gap_width * 2 - 1)
        controls_width = visible_width(previous_button) + visible_width(separator) + visible_width(next_button)
        if controls_width > available:
            previous_button, next_button, separator = "↑", "↓", " "
            controls_width = visible_width(previous_button) + visible_width(separator) + visible_width(next_button)
        show_buttons = controls_width <= available
        rendered_buttons = (
            self._navigation_button_style(previous_button, self._hovered_navigation_direction == -1)
            + separator + self._navigation_button_style(next_button, self._hovered_navigation_direction == 1)
        ) if show_buttons else ""
        outer_gaps_width = outer_gap_width * 2 if show_buttons else 0
        right_rule_width = 1 if rendered_buttons and inner_width > controls_width + outer_gaps_width else 0
        left_rule_width = max(0, inner_width - (controls_width if show_buttons else 0) - outer_gaps_width - right_rule_width)
        previous_start = 1 + left_rule_width + outer_gap_width
        self._previous_button_start = previous_start if show_buttons else -1
        self._previous_button_end = previous_start + visible_width(previous_button) if show_buttons else -1
        self._next_button_start = self._previous_button_end + visible_width(separator) if show_buttons else -1
        self._next_button_end = self._next_button_start + visible_width(next_button) if show_buttons else -1
        if safe_width == 1:
            return ["┌", "│", "└"]
        gap = " " if rendered_buttons else ""
        return [
            "┌" + js_repeat("─", inner_width) + "┐",
            "│" + content + "│",
            "└" + js_repeat("─", left_rule_width) + gap + rendered_buttons + gap + js_repeat("─", right_rule_width) + "┘",
        ]
