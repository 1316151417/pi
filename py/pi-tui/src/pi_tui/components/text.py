"""Wrapped, padded text with the source's render-cache behavior."""

from __future__ import annotations

import math
from collections.abc import Callable

from .._javascript import js_repeat, js_trim
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


class Text:
    def __init__(
        self, text: str = "", padding_x: int = 1, padding_y: int = 1,
        custom_bg_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._custom_bg_fn = custom_bg_fn
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self._text = text
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def set_custom_bg_fn(self, custom_bg_fn: Callable[[str], str] | None = None) -> None:
        self._custom_bg_fn = custom_bg_fn
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def invalidate(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_text == self._text and self._cached_width == width:
            return self._cached_lines
        if not self._text or not js_trim(self._text):
            result: list[str] = []
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = result
            return result
        normalized_text = self._text.replace("\t", "   ")
        padding_x = min(self._padding_x, max(0, math.floor((width - 1) / 2)))
        content_width = max(1, width - padding_x * 2)
        wrapped_lines = wrap_text_with_ansi(normalized_text, content_width)
        left_margin = js_repeat(" ", padding_x)
        right_margin = js_repeat(" ", padding_x)
        content_lines: list[str] = []
        for line in wrapped_lines:
            with_margins = left_margin + line + right_margin
            if self._custom_bg_fn is not None:
                content_lines.append(apply_background_to_line(with_margins, width, self._custom_bg_fn))
            else:
                content_lines.append(with_margins + js_repeat(" ", max(0, width - visible_width(with_margins))))
        empty_line = js_repeat(" ", width)
        empty_lines: list[str] = []
        index = 0
        while index < self._padding_y:
            empty_lines.append(apply_background_to_line(empty_line, width, self._custom_bg_fn) if self._custom_bg_fn else empty_line)
            index += 1
        result = [*empty_lines, *content_lines, *empty_lines]
        self._cached_text = self._text
        self._cached_width = width
        self._cached_lines = result
        return result if result else [""]
