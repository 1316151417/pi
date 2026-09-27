"""Command selection, scrolling and configurable primary-column layout."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from .._component import TuiMouseEvent, TuiMouseEventResult
from .._javascript import js_repeat, js_trim
from ..keybindings import get_keybindings
from ..utils import truncate_to_width, visible_width

DEFAULT_PRIMARY_COLUMN_WIDTH = 32
PRIMARY_COLUMN_GAP = 2
MIN_DESCRIPTION_WIDTH = 10


@dataclass
class SelectItem:
    value: str
    label: str
    description: str | None = None


@dataclass
class SelectListTheme:
    selected_prefix: Callable[[str], str]
    selected_text: Callable[[str], str]
    description: Callable[[str], str]
    scroll_info: Callable[[str], str]
    no_match: Callable[[str], str]


@dataclass
class SelectListTruncatePrimaryContext:
    text: str
    max_width: int
    column_width: int
    item: SelectItem
    is_selected: bool


@dataclass
class SelectListLayoutOptions:
    min_primary_column_width: int | None = None
    max_primary_column_width: int | None = None
    truncate_primary: Callable[[SelectListTruncatePrimaryContext], str] | None = None


class SelectList:
    def __init__(
        self, items: list[SelectItem], max_visible: int, theme: SelectListTheme,
        layout: SelectListLayoutOptions | None = None,
    ) -> None:
        self._items = items
        self._filtered_items = items
        self._selected_index = 0
        self._mouse_pressed_index: int | None = None
        self._max_visible = max_visible
        self._theme = theme
        self._layout = layout or SelectListLayoutOptions()
        self.on_select: Callable[[SelectItem], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.on_selection_change: Callable[[SelectItem], None] | None = None

    def set_filter(self, filter: str) -> None:
        self._filtered_items = [item for item in self._items if item.value.lower().startswith(filter.lower())]
        self._selected_index = 0

    def set_selected_index(self, index: int) -> None:
        self._selected_index = max(0, min(index, len(self._filtered_items) - 1))

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        if not self._filtered_items:
            return [self._theme.no_match("  No matching commands")]
        primary_width = self._get_primary_column_width()
        start, end = self._get_visible_range()
        lines: list[str] = []
        for index in range(start, end):
            item = self._filtered_items[index]
            description = js_trim(re.sub(r"[\r\n]+", " ", item.description)) if item.description else None
            lines.append(self._render_item(item, index == self._selected_index, width, description, primary_width))
        if start > 0 or end < len(self._filtered_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered_items)})"
            lines.append(self._theme.scroll_info(truncate_to_width(scroll_text, width - 2, "")))
        return lines

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult | None:
        if not self._filtered_items:
            return None
        if event.type == "wheel" and event.wheel_delta:
            delta = -1 if event.wheel_delta < 0 else 1
            previous = self._selected_index
            self._selected_index = max(0, min(len(self._filtered_items) - 1, self._selected_index + delta))
            if self._selected_index != previous:
                self._notify_selection_change()
            return TuiMouseEventResult(handled=True, render=self._selected_index != previous)
        if event.button != "left" or event.type not in ("press", "click"):
            return None
        start, end = self._get_visible_range()
        item_index = start + event.y
        if item_index < start or item_index >= end:
            return None
        if event.type == "press":
            self._mouse_pressed_index = item_index
            if self._selected_index != item_index:
                self._selected_index = item_index
                self._notify_selection_change()
            return TuiMouseEventResult(handled=True, focus=True)
        clicked = item_index if self._mouse_pressed_index is None else self._mouse_pressed_index
        self._mouse_pressed_index = None
        changed = self._selected_index != clicked
        self._selected_index = clicked
        if changed:
            self._notify_selection_change()
        selected = self.get_selected_item()
        if selected is not None and self.on_select is not None:
            self.on_select(selected)
        return TuiMouseEventResult(handled=True)

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.up"):
            self._selected_index = len(self._filtered_items) - 1 if self._selected_index == 0 else self._selected_index - 1
            self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.down"):
            self._selected_index = 0 if self._selected_index == len(self._filtered_items) - 1 else self._selected_index + 1
            self._notify_selection_change()
        elif kb.matches(key_data, "tui.select.confirm"):
            selected = self.get_selected_item()
            if selected is not None and self.on_select is not None:
                self.on_select(selected)
        elif kb.matches(key_data, "tui.select.cancel"):
            if self.on_cancel is not None:
                self.on_cancel()

    def _get_visible_range(self) -> tuple[int, int]:
        start = max(0, min(self._selected_index - math.floor(self._max_visible / 2), len(self._filtered_items) - self._max_visible))
        return start, min(start + self._max_visible, len(self._filtered_items))

    def _render_item(self, item: SelectItem, selected: bool, width: int, description: str | None, primary_width: int) -> str:
        prefix = "→ " if selected else "  "
        prefix_width = visible_width(prefix)
        if description and width > 40:
            effective_width = max(1, min(primary_width, width - prefix_width - 4))
            max_primary_width = max(1, effective_width - PRIMARY_COLUMN_GAP)
            truncated = self._truncate_primary(item, selected, max_primary_width, effective_width)
            truncated_width = visible_width(truncated)
            spacing = js_repeat(" ", max(1, effective_width - truncated_width))
            remaining_width = width - (prefix_width + truncated_width + len(spacing)) - 2
            if remaining_width > MIN_DESCRIPTION_WIDTH:
                truncated_description = truncate_to_width(description, remaining_width, "")
                if selected:
                    return self._theme.selected_text(prefix + truncated + spacing + truncated_description)
                return prefix + truncated + self._theme.description(spacing + truncated_description)
        max_width = width - prefix_width - 2
        truncated = self._truncate_primary(item, selected, max_width, max_width)
        return self._theme.selected_text(prefix + truncated) if selected else prefix + truncated

    def _get_primary_column_width(self) -> int:
        minimum = self._layout.min_primary_column_width
        maximum = self._layout.max_primary_column_width
        raw_min = minimum if minimum is not None else maximum if maximum is not None else DEFAULT_PRIMARY_COLUMN_WIDTH
        raw_max = maximum if maximum is not None else minimum if minimum is not None else DEFAULT_PRIMARY_COLUMN_WIDTH
        min_width = max(1, min(raw_min, raw_max))
        max_width = max(1, max(raw_min, raw_max))
        widest = max((visible_width(item.label or item.value) + PRIMARY_COLUMN_GAP for item in self._filtered_items), default=0)
        return max(min_width, min(widest, max_width))

    def _truncate_primary(self, item: SelectItem, selected: bool, max_width: int, column_width: int) -> str:
        display = item.label or item.value
        truncated = self._layout.truncate_primary(SelectListTruncatePrimaryContext(
            display, max_width, column_width, item, selected,
        )) if self._layout.truncate_primary else truncate_to_width(display, max_width, "")
        return truncate_to_width(truncated, max_width, "")

    def _notify_selection_change(self) -> None:
        selected = self.get_selected_item()
        if selected is not None and self.on_selection_change is not None:
            self.on_selection_change(selected)

    def get_selected_item(self) -> SelectItem | None:
        return self._filtered_items[self._selected_index] if 0 <= self._selected_index < len(self._filtered_items) else None
