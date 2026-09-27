"""Word movement with Intl-compatible boundaries and UTF-16 cursor offsets."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ._javascript import utf16_length, utf16_slice
from ._segmenter import SegmentData
from .utils import PUNCTUATION_REGEX, get_word_segmenter, is_whitespace_char


_word_segmenter = get_word_segmenter()


@dataclass(frozen=True, slots=True)
class WordNavigationOptions:
    segment: Callable[[str], Iterable[SegmentData]] | None = None
    is_atomic_segment: Callable[[str], bool] | None = None


def find_word_backward(text: str, cursor: int, options: WordNavigationOptions | None = None) -> int:
    if cursor <= 0:
        return 0
    text_before_cursor = utf16_slice(text, 0, cursor)
    segment_fn = options.segment if options else None
    is_atomic = options.is_atomic_segment if options else None
    segments = list(segment_fn(text_before_cursor) if segment_fn else _word_segmenter.segment(text_before_cursor))
    new_cursor = cursor
    while (
        segments
        and not (is_atomic and is_atomic(segments[-1].segment))
        and is_whitespace_char(segments[-1].segment)
    ):
        new_cursor -= utf16_length(segments.pop().segment)
    if not segments:
        return new_cursor
    last = segments[-1]
    if is_atomic and is_atomic(last.segment):
        new_cursor -= utf16_length(last.segment)
    elif last.is_word_like:
        matches = list(PUNCTUATION_REGEX.finditer(last.segment))
        if not matches:
            new_cursor -= utf16_length(last.segment)
        else:
            new_cursor -= utf16_length(last.segment[matches[-1].end() :])
    else:
        while (
            segments
            and not (is_atomic and is_atomic(segments[-1].segment))
            and not segments[-1].is_word_like
            and not is_whitespace_char(segments[-1].segment)
        ):
            new_cursor -= utf16_length(segments.pop().segment)
    return new_cursor


def find_word_forward(text: str, cursor: int, options: WordNavigationOptions | None = None) -> int:
    length = utf16_length(text)
    if cursor >= length:
        return length
    text_after_cursor = utf16_slice(text, cursor)
    segment_fn = options.segment if options else None
    is_atomic = options.is_atomic_segment if options else None
    segments = segment_fn(text_after_cursor) if segment_fn else _word_segmenter.segment(text_after_cursor)
    iterator = iter(segments)
    current = next(iterator, None)
    new_cursor = cursor
    while (
        current is not None
        and not (is_atomic and is_atomic(current.segment))
        and is_whitespace_char(current.segment)
    ):
        new_cursor += utf16_length(current.segment)
        current = next(iterator, None)
    if current is None:
        return new_cursor
    if is_atomic and is_atomic(current.segment):
        new_cursor += utf16_length(current.segment)
    elif current.is_word_like:
        match = PUNCTUATION_REGEX.search(current.segment)
        new_cursor += utf16_length(current.segment[: match.start()] if match else current.segment)
    else:
        while (
            current is not None
            and not (is_atomic and is_atomic(current.segment))
            and not current.is_word_like
            and not is_whitespace_char(current.segment)
        ):
            new_cursor += utf16_length(current.segment)
            current = next(iterator, None)
    return new_cursor


__all__ = ["WordNavigationOptions", "find_word_backward", "find_word_forward"]
