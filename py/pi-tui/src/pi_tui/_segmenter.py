"""ICU-backed equivalents of the Intl.Segmenter operations used by the TUI.

ICU supplies both extended grapheme boundaries and dictionary word boundaries.
Matching a particular Node runtime requires the same ICU data and default locale.
Indices deliberately remain UTF-16 offsets, as in Intl.Segmenter.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from icu import BreakIterator, Locale

from ._javascript import from_utf16_units, utf16_units


@dataclass(frozen=True, slots=True)
class SegmentData:
    segment: str
    index: int = 0
    input: str = ""
    is_word_like: bool | None = None


class Segments:
    def __init__(self, text: str, segmenter: Segmenter) -> None:
        self._text = text
        self._units = utf16_units(text)
        self._segmenter = segmenter

    def __iter__(self) -> Iterator[SegmentData]:
        # Each iterator owns its ICU cursor, so nested and repeated iteration
        # never changes another iterator's current boundary.
        if self._segmenter.granularity == "word":
            iterator = BreakIterator.createWordInstance(self._segmenter._locale)
        else:
            iterator = BreakIterator.createCharacterInstance(self._segmenter._locale)
        # PyICU converts a Python four-byte string via UTF-32, which replaces
        # lone surrogates. A string of UTF-16 units takes its lossless two-byte
        # conversion path, including when a lone surrogate and emoji coexist.
        iterator.setText(self._units)
        start = iterator.first()
        end = iterator.nextBoundary()
        while end != BreakIterator.DONE:
            word_like = None
            if self._segmenter.granularity == "word":
                # V8 uses ICU's NUMBER, LETTER, KANA, and IDEO status ranges.
                word_like = 100 <= iterator.getRuleStatus() < 500
            yield SegmentData(
                from_utf16_units(self._units[start:end]), start, self._text, word_like
            )
            start = end
            end = iterator.nextBoundary()

    def containing(self, index: int) -> SegmentData | None:
        if index < 0 or index >= len(self._units):
            return None
        previous: SegmentData | None = None
        for segment in self:
            if segment.index > index:
                return previous
            previous = segment
        return previous


class Segmenter:
    def __init__(self, granularity: Literal["grapheme", "word"]) -> None:
        self.granularity = granularity
        self._locale = Locale.getDefault()

    def segment(self, text: str) -> Segments:
        return Segments(text, self)
