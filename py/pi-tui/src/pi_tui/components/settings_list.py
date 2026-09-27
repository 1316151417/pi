"""Searchable settings list with value cycling and nested submenus."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from .._component import Component, TuiMouseEvent, TuiMouseEventResult
from .._javascript import js_repeat
from ..fuzzy import fuzzy_filter
from ..keybindings import get_keybindings
from ..utils import truncate_to_width, visible_width, wrap_text_with_ansi
from .input import Input


@dataclass
class SettingSubmenuCloseOptions:
    navigate_to: str | None = None


class SettingSubmenuDone(Protocol):
    def __call__(self, selected_value: str | None = None, options: SettingSubmenuCloseOptions | None = None) -> None: ...


@dataclass
class SettingItem:
    id: str
    label: str
    current_value: str
    description: str | None = None
    values: list[str] | None = None
    submenu: Callable[[str, SettingSubmenuDone], Component] | None = None


@dataclass
class SettingsListTheme:
    label: Callable[[str, bool], str]
    value: Callable[[str, bool], str]
    description: Callable[[str], str]
    cursor: str
    hint: Callable[[str], str]


@dataclass
class SettingsListOptions:
    enable_search: bool = False


class SettingsList:
    def __init__(
        self, items: list[SettingItem], max_visible: int, theme: SettingsListTheme,
        on_change: Callable[[str, str], None], on_cancel: Callable[[], None],
        options: SettingsListOptions | None = None,
    ) -> None:
        options = options or SettingsListOptions()
        self._items = items
        self._filtered_items = items
        self._theme = theme
        self._selected_index = 0
        self._mouse_pressed_index: int | None = None
        self._max_visible = max_visible
        self._on_change = on_change
        self._on_cancel = on_cancel
        self._search_enabled = options.enable_search
        self._search_input = Input() if self._search_enabled else None
        self._submenu_component: Component | None = None
        self._submenu_item_index: int | None = None
        self._navigate_after_close: str | None = None

    def update_value(self, id: str, new_value: str) -> None:
        for item in self._items:
            if item.id == id:
                item.current_value = new_value
                return

    def select_item(self, id: str) -> None:
        for index, item in enumerate(self._get_display_items()):
            if item.id == id:
                self._selected_index = index
                return

    def invalidate(self) -> None:
        callback = getattr(self._submenu_component, "invalidate", None)
        if callback is not None:
            callback()

    def render(self, width: int) -> list[str]:
        if self._submenu_component is not None:
            return self._submenu_component.render(width)
        lines: list[str] = []
        if self._search_enabled and self._search_input is not None:
            lines.extend(self._search_input.render(width))
            lines.append("")
        if not self._items:
            lines.append(self._theme.hint("  No settings available"))
            if self._search_enabled:
                self._add_hint_line(lines, width)
            return lines
        displayed = self._get_display_items()
        if not displayed:
            lines.append(truncate_to_width(self._theme.hint("  No matching settings"), width))
            self._add_hint_line(lines, width)
            return lines
        start, end = self._get_visible_range(displayed)
        max_label_width = min(36, max(visible_width(item.label) for item in self._items))
        for index in range(start, end):
            item = displayed[index]
            selected = index == self._selected_index
            prefix = self._theme.cursor if selected else "  "
            prefix_width = visible_width(prefix)
            label_padded = item.label + js_repeat(" ", max(0, max_label_width - visible_width(item.label)))
            label_text = self._theme.label(label_padded, selected)
            separator = "  "
            value_max_width = width - (prefix_width + max_label_width + visible_width(separator)) - 2
            value_text = self._theme.value(truncate_to_width(item.current_value, value_max_width, ""), selected)
            lines.append(truncate_to_width(prefix + label_text + separator + value_text, width))
        if start > 0 or end < len(displayed):
            lines.append(self._theme.hint(truncate_to_width(f"  ({self._selected_index + 1}/{len(displayed)})", width - 2, "")))
        selected_item = displayed[self._selected_index] if 0 <= self._selected_index < len(displayed) else None
        if selected_item is not None and selected_item.description:
            lines.append("")
            for line in wrap_text_with_ansi(selected_item.description, width - 4):
                lines.append(self._theme.description("  " + line))
        self._add_hint_line(lines, width)
        return lines

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult | None:
        if self._submenu_component is not None:
            handler = getattr(self._submenu_component, "handle_mouse", None)
            result = handler(event) if handler is not None else None
            return replace(result, focus=True) if result is not None else None
        if self._search_enabled and self._search_input is not None:
            if event.y == 0:
                result = self._search_input.handle_mouse(event)
                return replace(result, focus=True) if result is not None else None
            if event.y == 1:
                return None
        displayed = self._get_display_items()
        if not displayed:
            return None
        if event.type == "wheel" and event.wheel_delta:
            delta = -1 if event.wheel_delta < 0 else 1
            previous = self._selected_index
            self._selected_index = max(0, min(len(displayed) - 1, self._selected_index + delta))
            return TuiMouseEventResult(handled=True, render=self._selected_index != previous)
        if event.button != "left" or event.type not in ("press", "click"):
            return None
        row_offset = 2 if self._search_enabled else 0
        start, end = self._get_visible_range(displayed)
        item_index = start + event.y - row_offset
        if item_index < start or item_index >= end:
            return None
        if event.type == "press":
            self._mouse_pressed_index = item_index
            self._selected_index = item_index
            return TuiMouseEventResult(handled=True, focus=True)
        self._selected_index = item_index if self._mouse_pressed_index is None else self._mouse_pressed_index
        self._mouse_pressed_index = None
        self._activate_item()
        return TuiMouseEventResult(handled=True)

    def handle_input(self, data: str) -> None:
        if self._submenu_component is not None:
            handler = getattr(self._submenu_component, "handle_input", None)
            if handler is not None:
                handler(data)
            return
        kb = get_keybindings()
        displayed = self._get_display_items()
        if kb.matches(data, "tui.select.up"):
            if not displayed:
                return
            self._selected_index = len(displayed) - 1 if self._selected_index == 0 else self._selected_index - 1
        elif kb.matches(data, "tui.select.down"):
            if not displayed:
                return
            self._selected_index = 0 if self._selected_index == len(displayed) - 1 else self._selected_index + 1
        elif kb.matches(data, "tui.select.confirm") or (
            data == " " and (not self._search_enabled or (self._search_input is not None and not self._search_input.get_value()))
        ):
            self._activate_item()
        elif kb.matches(data, "tui.select.cancel"):
            self._on_cancel()
        elif self._search_enabled and self._search_input is not None:
            self._search_input.handle_input(data)
            self._filtered_items = fuzzy_filter(self._items, self._search_input.get_value(), lambda item: item.label)
            self._selected_index = 0

    def _get_display_items(self) -> list[SettingItem]:
        return self._filtered_items if self._search_enabled else self._items

    def _get_visible_range(self, displayed: Sequence[SettingItem]) -> tuple[int, int]:
        start = max(0, min(self._selected_index - math.floor(self._max_visible / 2), len(displayed) - self._max_visible))
        return start, min(start + self._max_visible, len(displayed))

    def _activate_item(self) -> None:
        displayed = self._get_display_items()
        if not 0 <= self._selected_index < len(displayed):
            return
        item = displayed[self._selected_index]
        if item.submenu is not None:
            self._submenu_item_index = self._selected_index

            def done(selected_value: str | None = None, options: SettingSubmenuCloseOptions | None = None) -> None:
                if selected_value is not None:
                    item.current_value = selected_value
                    self._on_change(item.id, selected_value)
                if options is not None and options.navigate_to:
                    self._navigate_after_close = options.navigate_to
                self._close_submenu()

            self._submenu_component = item.submenu(item.current_value, done)
        elif item.values:
            try:
                current_index = item.values.index(item.current_value)
            except ValueError:
                current_index = -1
            new_value = item.values[(current_index + 1) % len(item.values)]
            item.current_value = new_value
            self._on_change(item.id, new_value)

    def _close_submenu(self) -> None:
        self._submenu_component = None
        if self._navigate_after_close is not None:
            id = self._navigate_after_close
            self._navigate_after_close = None
            self._submenu_item_index = None
            self.select_item(id)
            self._activate_item()
        elif self._submenu_item_index is not None:
            self._selected_index = self._submenu_item_index
            self._submenu_item_index = None

    def _add_hint_line(self, lines: list[str], width: int) -> None:
        lines.append("")
        hint = "  Type to search · Enter/Space to change · Esc to cancel" if self._search_enabled else "  Enter/Space to change · Esc to cancel"
        lines.append(truncate_to_width(self._theme.hint(hint), width))
