"""Single-line text truncation from ``components/truncated-text.ts``."""

from .._javascript import js_repeat
from ..utils import truncate_to_width, visible_width


class TruncatedText:
    def __init__(self, text: str, padding_x: int = 0, padding_y: int = 0) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        result: list[str] = []
        empty_line = js_repeat(" ", width)
        index = 0
        while index < self._padding_y:
            result.append(empty_line)
            index += 1
        available_width = max(1, width - self._padding_x * 2)
        single_line_text = self._text.split("\n", 1)[0]
        display_text = truncate_to_width(single_line_text, available_width)
        with_padding = js_repeat(" ", self._padding_x) + display_text + js_repeat(" ", self._padding_x)
        result.append(with_padding + js_repeat(" ", max(0, width - visible_width(with_padding))))
        index = 0
        while index < self._padding_y:
            result.append(empty_line)
            index += 1
        return result
