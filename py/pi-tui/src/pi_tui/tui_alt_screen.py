"""Application-owned fullscreen viewport from ``tui-alt-screen.ts``."""

from __future__ import annotations

import asyncio
import base64
import math
import os
import re
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from ._component import (
    CURSOR_MARKER, Component, Container, OverlayHandle, OverlayOptions,
    TuiInputListenerResult, TuiMouseButton, TuiMouseDispatchResult,
    TuiMouseDispatchTarget, TuiMouseEvent, TuiMouseEventType, TuiStopOptions,
    dispatch_mouse_event, retarget_mouse_event,
)
from ._javascript import JS_WHITESPACE, from_utf16_units, js_trim, utf16_units
from ._terminal_types import Terminal
from ._timers import TimerHandle, set_timeout
from .alt_screen_search import (
    AltScreenSearchComponent, AltScreenSearchIndex, AltScreenSearchMatch,
    get_alt_screen_search_match_key,
)
from .components.alt_screen_flash import AltScreenFlashContainer
from .components.scroll_view import ScrollView, ScrollViewOptions, ScrollViewScrollToOptions
from .keybindings import get_keybindings
from .keys import is_key_release
from .layout import (
    LayoutFrame, ScrollbarGeometry, get_layout_boxes_at, get_scrollbar_geometry,
    get_scroll_view_box, get_scroll_views_at, render_layout_frame,
)
from .layout_node import get_layout_node
from .terminal_image import (
    ImageProtocol, TerminalCapabilities, delete_all_kitty_images,
    delete_all_kitty_placements, delete_kitty_image, get_capabilities,
    get_kitty_image_placement, is_image_line, set_capabilities,
)
from .tui import AppKeybindings, TuiBase, _QueryPromise, _floor, _maximum, composite_tui_line
from .utils import (
    extract_ansi_code, get_grapheme_cell_range, get_osc8_link_at_column,
    get_word_segmenter, slice_by_column, strip_terminal_sequences,
    truncate_to_width, visible_width,
)

_ENTER_ALT_SCREEN = "\x1b[?1049h"
_EXIT_ALT_SCREEN = "\x1b[?1049l"
_DISABLE_AUTOWRAP = "\x1b[?7l"
_ENABLE_AUTOWRAP = "\x1b[?7h"
_ENABLE_BUTTON_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h"
_ENABLE_ALL_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1004h\x1b[?1006h"
_DISABLE_MOUSE = "\x1b[?1006l\x1b[?1004l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
_FOCUS_IN = "\x1b[I"
_FOCUS_OUT = "\x1b[O"
_BEGIN_SYNCHRONIZED_OUTPUT = "\x1b[?2026h"
_END_SYNCHRONIZED_OUTPUT = "\x1b[?2026l"
_OSC133_ZONE_PREFIX = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")
_OSC133_PROMPT_START = re.compile(r"^\x1b\]133;A(?:\x07|\x1b\\)")
_SGR_MOUSE = re.compile(r"\x1b\[<([0-9]+);([0-9]+);([0-9]+)([Mm])")
_PAGE_SCROLL_OVERLAP = 4
_ALT_WHEEL_SCROLL_MULTIPLIER = 5
_MAX_CACHED_OFFSCREEN_KITTY_IMAGES = 16
_MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES = 32 * 1024 * 1024
_MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES = 64 * 1024 * 1024
_DOUBLE_CLICK_INTERVAL_MS = 500
_COPY_ERROR_FLASH_DURATION_MS = 5000
_TERMINAL_WORD_SELECTION_JOINERS = {"/", "-"}
_WORD_SEGMENTER = get_word_segmenter()

type _Number = int | float
type _Direction = Literal[-1, 1]
type _SelectionGranularity = Literal["character", "word", "line"]
type _SearchSelectionMode = Literal["query", "retain", "next", "previous"]


@dataclass
class _CachedKittyImage:
    transmission_generation: int
    transmission_bytes: int
    estimated_decoded_bytes: int


@dataclass
class _SelectionPoint:
    row: int
    col: int
    scroll_view: ScrollView | None = None
    boundary: bool | None = None


@dataclass
class _SelectionRange:
    start: _SelectionPoint
    end: _SelectionPoint


@dataclass
class _ClickTarget:
    timestamp: int
    count: int
    row: int
    scroll_view: ScrollView | None
    word_start: int
    word_end: int


@dataclass
class _SgrMouseEvent:
    button: _Number
    x: int
    y: int
    release: bool


@dataclass
class _WheelEvent:
    direction: _Direction
    x: int
    y: int
    button: _Number


@dataclass
class _ScrollbarDrag:
    scroll_view: ScrollView
    grab_offset: int


@dataclass
class _ScrollbarTarget:
    scroll_view: ScrollView
    geometry: ScrollbarGeometry


@dataclass
class _ScrollToEndIndicatorRect:
    row: int
    column: int
    width: int


@dataclass
class _ActiveSearch:
    component: AltScreenSearchComponent
    index: AltScreenSearchIndex
    query: str
    matches: list[AltScreenSearchMatch]
    selected_index: int
    anchor_row: int
    selection_mode: _SearchSelectionMode
    overlay: OverlayHandle | None = None
    selected_key: str | None = None


@dataclass
class _SearchHighlightRange:
    start_col: int
    end_col: int
    current: bool


@dataclass
class _Point:
    x: int
    y: int


@dataclass
class _ComponentClick:
    timestamp: int
    count: int
    component: Component
    x: int
    y: int


@dataclass
class _WordSegment:
    start: int
    end: int
    selectable: bool
    joiner: bool


@dataclass
class TuiAltScreenOptions:
    wheel_scroll_lines: _Number | None = None
    mouse: bool | None = None
    search_match_style: Callable[[str], str] | None = None
    search_current_match_style: Callable[[str], str] | None = None
    search_navigation_button_style: Callable[[str, bool], str] | None = None
    scroll_to_end_indicator: Callable[[], str] | None = None
    open_url: Callable[[str], None] | None = None
    on_right_click_paste: Callable[[], None] | None = None
    copy_on_select: bool | None = None
    copy_selection: Callable[[str], Awaitable[bool | str]] | None = None


def _line_at(lines: Sequence[str], row: int) -> str:
    # The layout may contain holes where JavaScript expanded a sparse image array.
    if isinstance(row, float):
        if not row.is_integer():
            return ""
        row = int(row)
    return (lines[row] or "") if 0 <= row < len(lines) else ""


def _parse_decimal(value: str, *, offset: int = 0) -> _Number:
    # Number.parseInt produces a double, including rounding large integer fields.
    parsed = float(value) + offset
    return int(parsed) if math.isfinite(parsed) else parsed


def _button_bits(button: _Number) -> int:
    return math.trunc(button) % 0x100000000 if math.isfinite(button) else 0


def _terminal_extent(value: _Number) -> int:
    extent = _maximum(1, value)
    # Python's integral float and int must both act as a JS array index.
    return int(extent) if math.isfinite(extent) and extent == math.trunc(extent) else cast(int, extent)


class _ImplicitDocument:
    def __init__(self, owner: TuiAltScreen) -> None:
        self._owner = owner

    def render(self, width: int) -> list[str]:
        return Container.render(self._owner, width)

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
        return Container.handle_mouse(self._owner, event)

    def invalidate(self) -> None:
        for child in self._owner.children:
            child.invalidate()


class TuiAltScreen(TuiBase):
    __pi_tui_viewport__: Literal[True] = True

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
        options: TuiAltScreenOptions | None = None,
        *,
        keybindings: AppKeybindings | None = None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor, log_directory, keybindings=keybindings)
        options = TuiAltScreenOptions() if options is None else options
        self._previous_screen: list[str] = []
        self._last_document: list[str] = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._layout_root: Component | None = None
        self._current_layout: LayoutFrame | None = None
        self._implicit_document = _ImplicitDocument(self)
        self._implicit_scroll_view = ScrollView(self._implicit_document, ScrollViewOptions(follow="end", primary=True))
        self._flashes = AltScreenFlashContainer(self.request_render)
        self._alt_screen_active = False
        self._image_protocol: ImageProtocol = None
        self._saved_capabilities: TerminalCapabilities | None = None
        self._uploaded_kitty_images: dict[int, _CachedKittyImage] = {}
        self._selection_anchor: _SelectionPoint | None = None
        self._selection_focus: _SelectionPoint | None = None
        self._selection_granularity: _SelectionGranularity = "character"
        self._selection_initial_range: _SelectionRange | None = None
        self._last_click: _ClickTarget | None = None
        self._selection_drag_pointer: _Point | None = None
        self._selection_auto_scroll_direction: Literal[-1, 0, 1] = 0
        self._selection_auto_scroll_timer: TimerHandle | None = None
        self._selection_timer_generation = 0
        self._selection_press_active = False
        self._scrollbar_drag: _ScrollbarDrag | None = None
        self._scrollbar_hover: ScrollView | None = None
        self._scroll_to_end_indicator_rect: _ScrollToEndIndicatorRect | None = None
        self._active_search: _ActiveSearch | None = None
        self._pressed_url: str | None = None
        self._selection_dragged = False
        self._mouse_capture: TuiMouseDispatchTarget | None = None
        self._mouse_press_target: TuiMouseDispatchTarget | None = None
        self._mouse_press_point: _Point | None = None
        self._mouse_press_moved = False
        self._last_component_click: _ComponentClick | None = None
        self._wheel_scroll_lines = _maximum(1, _floor(1 if options.wheel_scroll_lines is None else options.wheel_scroll_lines))
        self._mouse_enabled = True if options.mouse is None else options.mouse
        self._search_match_style = options.search_match_style if options.search_match_style is not None else lambda text: f"\x1b[4m{text}\x1b[24m"
        self._search_current_match_style = options.search_current_match_style if options.search_current_match_style is not None else lambda text: f"\x1b[1;7m{text}\x1b[22;27m"
        self._search_navigation_button_style = options.search_navigation_button_style if options.search_navigation_button_style is not None else lambda text, _hovered: text
        self._scroll_to_end_indicator = options.scroll_to_end_indicator
        self._open_url = options.open_url
        self._on_right_click_paste = options.on_right_click_paste
        self._copy_on_select = True if options.copy_on_select is None else options.copy_on_select
        self._copy_selection = options.copy_selection
        self._clipboard_tasks: set[asyncio.Task[None]] = set()
        self.add_input_listener(self._handle_viewport_input)

    @property
    def mode(self) -> Literal["fullscreen"]:
        return "fullscreen"

    @property
    def viewport_top(self) -> int:
        return self._get_primary_scroll_view().scroll_top

    @property
    def is_following_output(self) -> bool:
        return self._get_primary_scroll_view().is_following_end

    def get_copy_on_select(self) -> bool:
        return self._copy_on_select

    def set_copy_on_select(self, enabled: bool) -> None:
        self._copy_on_select = enabled

    def has_active_selection(self) -> bool:
        return self._get_active_selection_text() is not None

    def copy_active_selection_to_clipboard(self) -> Awaitable[bool]:
        return self._copy_selection_to_clipboard()

    def set_layout_root(self, component: Component | None) -> None:
        if self._layout_root is component:
            return
        self._layout_root = component
        self._current_layout = None
        self.request_render()

    def render(self, width: int) -> list[str]:
        return self._layout_root.render(width) if self._layout_root is not None else super().render(width)

    def _get_mounted_roots(self) -> Sequence[Component]:
        return [self._layout_root] if self._layout_root is not None else self.children

    def _get_primary_scroll_view(self) -> ScrollView:
        if self._current_layout is not None and self._current_layout.primary_scroll_view is not None:
            return self._current_layout.primary_scroll_view
        return self._implicit_scroll_view

    def _before_terminal_start(self) -> None:
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        self._alt_screen_active = True
        capabilities = get_capabilities()
        self._image_protocol = capabilities.images
        self._uploaded_kitty_images.clear()
        if capabilities.images == "iterm2":
            self._saved_capabilities = capabilities
            set_capabilities(replace(capabilities, images=None))
            self.invalidate()
        self._last_document = []
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._clear_component_mouse_gesture()
        self._last_component_click = None
        self._reset_render_state()
        term = os.environ.get("TERM", "").lower()
        multiplexer = any(name in os.environ for name in ("TMUX", "ZELLIJ", "STY")) or term.startswith(("tmux", "screen"))
        mouse_sequence = _ENABLE_BUTTON_MOTION_MOUSE if multiplexer else _ENABLE_ALL_MOTION_MOUSE
        self.terminal.write(_ENTER_ALT_SCREEN + _DISABLE_AUTOWRAP + (mouse_sequence if self._mouse_enabled else "") + "\x1b[2J\x1b[H\x1b[?25l")

    def _before_terminal_stop(self, _options: TuiStopOptions) -> None:
        self._close_search()
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._clear_component_mouse_gesture()
        self._flashes.dispose()
        if not self._alt_screen_active:
            return
        self.terminal.write(_BEGIN_SYNCHRONIZED_OUTPUT + self._delete_kitty_images() + (_DISABLE_MOUSE if self._mouse_enabled else "") + _ENABLE_AUTOWRAP + _END_SYNCHRONIZED_OUTPUT)
        self._uploaded_kitty_images.clear()

    def _after_terminal_stop(self, options: TuiStopOptions) -> None:
        if not self._alt_screen_active:
            return
        self._alt_screen_active = False
        if options.preserve_screen:
            self.terminal.write(_BEGIN_SYNCHRONIZED_OUTPUT + _EXIT_ALT_SCREEN + "\x1b[?25h" + _END_SYNCHRONIZED_OUTPUT)
        else:
            width = _terminal_extent(self.terminal.columns)
            document_lines = [_OSC133_ZONE_PREFIX.sub("", line) for line in self.render(width)]
            self._last_document = [
                line if is_image_line(line) or visible_width(line) <= width else slice_by_column(line, 0, width, True)
                for line in self._apply_line_resets([line.replace(CURSOR_MARKER, "") for line in document_lines])
            ]
            parts = [_BEGIN_SYNCHRONIZED_OUTPUT, _EXIT_ALT_SCREEN, _DISABLE_AUTOWRAP]
            for row, line in enumerate(self._last_document):
                if row > 0:
                    parts.append("\r\n")
                parts.extend(("\r\x1b[2K", line or ""))
            parts.extend(("\x1b[0m", _ENABLE_AUTOWRAP, "\r\n\x1b[?25h", _END_SYNCHRONIZED_OUTPUT))
            self.terminal.write("".join(parts))
        if self._saved_capabilities is not None:
            set_capabilities(self._saved_capabilities)
            self._saved_capabilities = None

    def _delete_kitty_images(self) -> str:
        return delete_all_kitty_images() if self._image_protocol == "kitty" else ""

    def _prepare_kitty_screen(self, screen: list[str]) -> tuple[list[str], str]:
        visible_ids: set[int] = set()
        lines: list[str] = []
        for line in screen:
            placement = get_kitty_image_placement(line)
            if placement is None:
                lines.append(line)
                continue
            visible_ids.add(placement.image_id)
            cached = self._uploaded_kitty_images.pop(placement.image_id, None)
            self._uploaded_kitty_images[placement.image_id] = _CachedKittyImage(
                placement.transmission_generation, placement.transmission_bytes, placement.estimated_decoded_bytes,
            )
            lines.append(placement.replacement_line if cached is not None and cached.transmission_generation == placement.transmission_generation else line)
        offscreen_count = transmission_bytes = decoded_bytes = 0
        for image_id, cached in self._uploaded_kitty_images.items():
            if image_id in visible_ids:
                continue
            offscreen_count += 1
            transmission_bytes += cached.transmission_bytes
            decoded_bytes += cached.estimated_decoded_bytes
        deletions: list[str] = []
        for image_id, cached in list(self._uploaded_kitty_images.items()):
            if offscreen_count <= _MAX_CACHED_OFFSCREEN_KITTY_IMAGES and transmission_bytes <= _MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES and decoded_bytes <= _MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES:
                break
            if image_id in visible_ids:
                continue
            deletions.append(delete_kitty_image(image_id))
            del self._uploaded_kitty_images[image_id]
            offscreen_count -= 1
            transmission_bytes -= cached.transmission_bytes
            decoded_bytes -= cached.estimated_decoded_bytes
        return lines, "".join(deletions)

    def _reset_render_state(self) -> None:
        self._previous_screen = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._current_layout = None

    def scroll_by(self, lines: float) -> None:
        self._get_primary_scroll_view().scroll_by(lines)
        self.request_render()

    def scroll_to_top(self) -> None:
        self._get_primary_scroll_view().scroll_to_start()
        self.request_render()

    def scroll_to_bottom(self) -> None:
        self._get_primary_scroll_view().scroll_to_end()
        self.request_render()

    def _scroll_to_prompt(self, direction: _Direction) -> None:
        if self._current_layout is None:
            return
        scroll_view = self._get_primary_scroll_view()
        box = get_scroll_view_box(self._current_layout, scroll_view)
        lines = box.scroll_content_lines if box is not None else None
        if lines is None:
            return
        row = scroll_view.scroll_top + direction
        while 0 <= row < len(lines):
            if _OSC133_PROMPT_START.match(_line_at(lines, row)) is not None:
                scroll_view.scroll_to(row)
                self.request_render()
                return
            row += direction

    def _toggle_search(self) -> None:
        if self._active_search is not None:
            self._close_search()
            return
        component = AltScreenSearchComponent(self._update_search_query, self._search_navigation_button_style)
        search = _ActiveSearch(component, AltScreenSearchIndex(), "", [], -1, self._get_primary_scroll_view().scroll_top, "query")
        self._active_search = search
        search.overlay = self.show_overlay(component, OverlayOptions(anchor="top-right", width="40%", min_width=32, margin=1))

    def _close_search(self) -> None:
        search = self._active_search
        if search is None:
            return
        self._active_search = None
        if search.overlay is not None:
            search.overlay.hide()
        self.request_render()

    def _update_search_query(self, query: str) -> None:
        search = self._active_search
        if search is None or utf16_units(query) == utf16_units(search.query):
            return
        selected = search.matches[search.selected_index] if 0 <= search.selected_index < len(search.matches) else None
        search.anchor_row = selected.segments[0].row if selected is not None and selected.segments else self._get_primary_scroll_view().scroll_top
        search.query = query
        search.selection_mode = "query"
        search.component.set_result(-1, 0)
        self.request_render()

    def _navigate_search(self, direction: _Direction) -> None:
        search = self._active_search
        if search is None or not search.query:
            return
        search.selection_mode = "previous" if direction < 0 else "next"
        self.request_render()

    def _get_search_navigation_direction_at(self, x: int, y: int) -> _Direction | None:
        search = self._active_search
        bounds = search.overlay.get_bounds() if search is not None and search.overlay is not None else None
        if search is None or bounds is None:
            return None
        if x < bounds.col or x >= bounds.col + bounds.width or y < bounds.row or y >= bounds.row + bounds.height:
            return None
        return search.component.get_navigation_direction_at(y - bounds.row, x - bounds.col)

    def _handle_search_mouse_event(self, event: _SgrMouseEvent) -> bool:
        search = self._active_search
        if search is None:
            return False
        direction = self._get_search_navigation_direction_at(event.x, event.y)
        if search.component.set_hovered_navigation_direction(direction):
            self.request_render()
        if direction is None or event.release or _button_bits(event.button) & 32 or _button_bits(event.button) & 3:
            return False
        self._navigate_search(direction)
        return True

    def _refresh_search(self, layout: LayoutFrame) -> bool:
        search = self._active_search
        if search is None:
            return False
        scroll_view = layout.primary_scroll_view if layout.primary_scroll_view is not None else self._implicit_scroll_view
        box = get_scroll_view_box(layout, scroll_view)
        lines = box.scroll_content_lines if box is not None else None
        if lines is None or not js_trim(search.query):
            search.matches = []
            search.selected_index = -1
            search.selected_key = None
            search.selection_mode = "retain"
            search.component.set_result(-1, 0)
            return False
        reveal = search.selection_mode != "retain"
        result = search.index.search(lines, search.query)
        matches = result.matches
        search.matches = matches
        if not result.changed and search.selection_mode == "retain":
            return False
        exact_index = search.selected_index
        if result.changed:
            exact_index = next((index for index, match in enumerate(matches) if get_alt_screen_search_match_key(match) == search.selected_key), -1) if search.selected_key else -1
        selected_index = -1
        if matches:
            if search.selection_mode == "query":
                low, high = 0, len(matches)
                while low < high:
                    middle = low + (high - low) // 2
                    if (matches[middle].segments[0].row if matches[middle].segments else 0) < search.anchor_row:
                        low = middle + 1
                    else:
                        high = middle
                selected_index = low if low < len(matches) else 0
            elif search.selection_mode == "next":
                base = exact_index if exact_index >= 0 else min(search.selected_index, len(matches) - 1)
                selected_index = 0 if base < 0 else (base + 1) % len(matches)
            elif search.selection_mode == "previous":
                base = exact_index if exact_index >= 0 else min(search.selected_index, len(matches) - 1)
                selected_index = len(matches) - 1 if base < 0 else (base - 1 + len(matches)) % len(matches)
            else:
                selected_index = exact_index if exact_index >= 0 else min(max(0, search.selected_index), len(matches) - 1)
        search.selected_index = selected_index
        search.selected_key = get_alt_screen_search_match_key(matches[selected_index]) if selected_index >= 0 else None
        search.selection_mode = "retain"
        search.component.set_result(selected_index, len(matches))
        if not reveal:
            return False
        selected = matches[selected_index] if 0 <= selected_index < len(matches) else None
        if box is None or selected is None or not selected.segments or scroll_view.viewport_height <= 0:
            return False
        first_segment, last_segment = selected.segments[0], selected.segments[-1]
        before = scroll_view.scroll_top
        visible_bottom = before + scroll_view.viewport_height - 1
        target = first_segment.row - scroll_view.viewport_height // 3 if first_segment.row < before or last_segment.row > visible_bottom else before
        scroll_view.scroll_to(target, ScrollViewScrollToOptions(disable_follow=True))
        return scroll_view.scroll_top != before

    def flash(self, message: str, duration_ms: float | None = None) -> None:
        if duration_ms is None:
            self._flashes.flash(message)
        else:
            self._flashes.flash(message, duration_ms)

    def _should_defer_viewport_input_to_overlay(self) -> bool:
        search = self._active_search
        return self._is_overlay_focused() and not (search is not None and search.overlay is not None and search.overlay.is_focused() is True)

    def _clear_component_mouse_gesture(self) -> None:
        self._mouse_capture = None
        self._mouse_press_target = None
        self._mouse_press_point = None
        self._mouse_press_moved = False

    def _handle_viewport_input(self, data: str) -> TuiInputListenerResult | None:
        if data == _FOCUS_OUT:
            had_active_selection = self._selection_press_active
            had_nonempty_selection = had_active_selection and self._get_selection_bounds() is not None
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            self._stop_scrollbar_hover()
            if self._active_search is not None and self._active_search.component.set_hovered_navigation_direction(None):
                self.request_render()
            self._stop_scrollbar_drag()
            self._pressed_url = None
            self._selection_dragged = False
            self._clear_component_mouse_gesture()
            self._last_component_click = None
            if had_active_selection:
                self._selection_anchor = None
                self._selection_focus = None
                self._selection_granularity = "character"
                self._selection_initial_range = None
                if had_nonempty_selection:
                    self.request_render()
            self._last_click = None
            return TuiInputListenerResult(consume=True)
        if data == _FOCUS_IN:
            return TuiInputListenerResult(consume=True)
        wheel = self._parse_wheel_event(data)
        if wheel is not None:
            event = self._create_mouse_event("wheel", wheel.button, wheel.x, wheel.y, wheel_delta=cast(int, wheel.direction * self._get_wheel_scroll_lines(wheel.button)))
            overlay = self._dispatch_mouse_to_overlay(event)
            result = overlay.result if overlay.result is not None else None if overlay.hit else self._dispatch_mouse_to_layout(event)
            if result is not None:
                if self._apply_mouse_dispatch_result(event, result):
                    self.request_render()
                return TuiInputListenerResult(consume=True)
            if self._should_defer_viewport_input_to_overlay():
                return None
            self._route_wheel(wheel)
            return TuiInputListenerResult(consume=True)
        mouse = self._parse_sgr_mouse_event(data)
        if mouse is not None:
            self._handle_mouse_event(mouse)
            return TuiInputListenerResult(consume=True)
        if self._is_mouse_sequence(data):
            return TuiInputListenerResult(consume=True)
        keybindings = get_keybindings()
        release = is_key_release(data)
        if keybindings.matches(data, "tui.altScreen.search"):
            if not release:
                self._toggle_search()
            return TuiInputListenerResult(consume=True)
        search = self._active_search
        if search is not None and search.overlay is not None and search.overlay.is_focused():
            if keybindings.matches(data, "tui.altScreen.searchNext"):
                if not release:
                    self._navigate_search(1)
                return TuiInputListenerResult(consume=True)
            if keybindings.matches(data, "tui.altScreen.searchPrevious"):
                if not release:
                    self._navigate_search(-1)
                return TuiInputListenerResult(consume=True)
            if keybindings.matches(data, "tui.altScreen.searchClose"):
                if not release:
                    self._close_search()
                return TuiInputListenerResult(consume=True)
        if self._should_defer_viewport_input_to_overlay():
            return None
        if keybindings.matches(data, "tui.altScreen.pageUp"):
            if not release:
                self.scroll_by(-max(1, self._get_primary_scroll_view().viewport_height - _PAGE_SCROLL_OVERLAP))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.pageDown"):
            if not release:
                self.scroll_by(max(1, self._get_primary_scroll_view().viewport_height - _PAGE_SCROLL_OVERLAP))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.halfPageUp"):
            if not release:
                self.scroll_by(-max(1, self._get_primary_scroll_view().viewport_height // 2))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.halfPageDown"):
            if not release:
                self.scroll_by(max(1, self._get_primary_scroll_view().viewport_height // 2))
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.lineUp"):
            if not release:
                self.scroll_by(-1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.lineDown"):
            if not release:
                self.scroll_by(1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.previousPrompt"):
            if not release:
                self._scroll_to_prompt(-1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.nextPrompt"):
            if not release:
                self._scroll_to_prompt(1)
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.top"):
            if not release:
                self.scroll_to_top()
            return TuiInputListenerResult(consume=True)
        if keybindings.matches(data, "tui.altScreen.bottom"):
            if not release:
                self.scroll_to_bottom()
            return TuiInputListenerResult(consume=True)
        return None

    @staticmethod
    def _decode_mouse_button(button: _Number) -> TuiMouseButton:
        return ("left", "middle", "right", "none")[_button_bits(button) & 3]

    def _create_mouse_event(
        self,
        event_type: TuiMouseEventType,
        button: _Number,
        x: int,
        y: int,
        *,
        wheel_delta: int | None = None,
        click_count: int | None = None,
    ) -> TuiMouseEvent:
        bits = _button_bits(button)
        return TuiMouseEvent(
            type=event_type, button="none" if event_type == "wheel" else self._decode_mouse_button(button),
            x=x, y=y, screen_x=x, screen_y=y,
            width=_terminal_extent(self.terminal.columns), height=_terminal_extent(self.terminal.rows),
            shift=bool(bits & 4), alt=bool(bits & 8), ctrl=bool(bits & 16),
            wheel_delta=wheel_delta, click_count=click_count,
        )

    def _dispatch_mouse_to_layout(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
        if self._current_layout is None:
            return None
        visited: set[int] = set()
        for box in get_layout_boxes_at(self._current_layout, event.screen_x, event.screen_y):
            if id(box.component) in visited:
                continue
            handler = getattr(box.component, "handle_mouse", None)
            if get_layout_node(box.component) is not None and getattr(handler, "__func__", None) is Container.handle_mouse:
                continue
            visited.add(id(box.component))
            result = dispatch_mouse_event(box.component, replace(
                event, x=event.screen_x - box.rect.x, y=event.screen_y - box.rect.y,
                width=box.rect.width, height=box.rect.height,
            ))
            if result is not None:
                return result
        return None

    def _apply_mouse_dispatch_result(self, event: TuiMouseEvent, result: TuiMouseDispatchResult) -> bool:
        focus_target = self._resolve_mouse_focus_target(result.focus_target if result.focus_target is not None else result.target.component)
        focus_changed = result.focus is True and self.get_focused_component() is not focus_target
        if result.focus:
            self.set_focus(focus_target)
        if result.capture:
            self._mouse_capture = result.target
        return result.render if result.render is not None else focus_changed or event.type in ("press", "click", "drag", "wheel")

    @staticmethod
    def _dispatch_mouse_to_target(event: TuiMouseEvent, target: TuiMouseDispatchTarget) -> TuiMouseDispatchResult | None:
        return dispatch_mouse_event(target.component, retarget_mouse_event(event, target))

    def _get_component_click_count(self, target: TuiMouseDispatchTarget, x: int, y: int) -> int:
        now = time.time_ns() // 1_000_000
        previous = self._last_component_click
        count = previous.count % 3 + 1 if previous is not None and now - previous.timestamp <= _DOUBLE_CLICK_INTERVAL_MS and previous.component is target.component and previous.x == x and previous.y == y else 1
        self._last_component_click = _ComponentClick(now, count, target.component, x, y)
        return count

    def _clear_text_selection(self) -> None:
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._pressed_url = None
        self._selection_dragged = False

    def _handle_mouse_event(self, raw: _SgrMouseEvent) -> None:
        motion = bool(_button_bits(raw.button) & 32)
        event_type: TuiMouseEventType = "release" if raw.release else ("move" if self._decode_mouse_button(raw.button) == "none" else "drag") if motion else "press"
        event = self._create_mouse_event(event_type, raw.button, raw.x, raw.y)
        target = self._mouse_capture if self._mouse_capture is not None else self._mouse_press_target
        if target is not None:
            if self._mouse_press_point is not None and (raw.x != self._mouse_press_point.x or raw.y != self._mouse_press_point.y):
                self._mouse_press_moved = True
                self._last_component_click = None
            render = False
            target_result = self._dispatch_mouse_to_target(event, target)
            if target_result is not None:
                render = self._apply_mouse_dispatch_result(event, target_result)
            if raw.release:
                if not self._mouse_press_moved and self._mouse_press_point is not None and self._mouse_press_point.x == raw.x and self._mouse_press_point.y == raw.y:
                    click = self._create_mouse_event("click", raw.button, raw.x, raw.y, click_count=self._get_component_click_count(target, raw.x, raw.y))
                    click_result = self._dispatch_mouse_to_target(click, target)
                    if click_result is not None:
                        render = self._apply_mouse_dispatch_result(click, click_result) or render
                self._clear_component_mouse_gesture()
            if render:
                self.request_render()
            return
        if self._handle_search_mouse_event(raw):
            return
        overlay = self._dispatch_mouse_to_overlay(event)
        if not overlay.hit:
            if self._handle_scroll_to_end_indicator_mouse_event(raw):
                return
            scrollbar_handled = self._handle_scrollbar_mouse_event(raw)
            if self._scrollbar_drag is None:
                self._update_scrollbar_hover(raw.x, raw.y)
            if scrollbar_handled:
                return
        else:
            self._stop_scrollbar_hover()
        result = overlay.result if overlay.result is not None else None if overlay.hit else self._dispatch_mouse_to_layout(event)
        if result is not None:
            render = self._apply_mouse_dispatch_result(event, result)
            if event_type == "press":
                self._clear_text_selection()
                self._mouse_press_target = result.target
                self._mouse_press_point = _Point(raw.x, raw.y)
                self._mouse_press_moved = False
            if render:
                self.request_render()
            return
        if self._handle_right_click_paste(raw):
            return
        self._handle_selection_mouse_event(raw)

    @staticmethod
    def _parse_wheel_event(data: str) -> _WheelEvent | None:
        sgr = _SGR_MOUSE.fullmatch(data)
        if sgr is not None:
            button = _parse_decimal(sgr[1])
            bits = _button_bits(button)
            if not bits & 64 or bits & 3 not in (0, 1):
                return None
            return _WheelEvent(-1 if bits & 3 == 0 else 1, cast(int, _parse_decimal(sgr[2], offset=-1)), cast(int, _parse_decimal(sgr[3], offset=-1)), button)
        units = utf16_units(data)
        if len(units) == 6 and units.startswith("\x1b[M"):
            button = ord(units[3]) - 32
            bits = _button_bits(button)
            if not bits & 64 or bits & 3 not in (0, 1):
                return None
            return _WheelEvent(-1 if bits & 3 == 0 else 1, ord(units[4]) - 33, ord(units[5]) - 33, button)
        return None

    def _get_wheel_scroll_lines(self, button: _Number) -> _Number:
        return self._wheel_scroll_lines * _ALT_WHEEL_SCROLL_MULTIPLIER if _button_bits(button) & 8 else self._wheel_scroll_lines

    def _route_wheel(self, event: _WheelEvent) -> None:
        remaining = event.direction * self._get_wheel_scroll_lines(event.button)
        seen: set[int] = set()
        for scroll_view in get_scroll_views_at(self._current_layout, event.x, event.y) if self._current_layout is not None else []:
            seen.add(id(scroll_view))
            remaining = scroll_view.scroll_by(remaining)
            if remaining == 0 or scroll_view.overscroll == "contain":
                break
        primary = self._get_primary_scroll_view()
        if remaining != 0 and id(primary) not in seen:
            primary.scroll_by(remaining)
        self._update_scrollbar_hover(event.x, event.y)
        self.request_render()

    @staticmethod
    def _parse_sgr_mouse_event(data: str) -> _SgrMouseEvent | None:
        match = _SGR_MOUSE.fullmatch(data)
        if match is None:
            return None
        return _SgrMouseEvent(_parse_decimal(match[1]), cast(int, _parse_decimal(match[2], offset=-1)), cast(int, _parse_decimal(match[3], offset=-1)), match[4] == "m")

    def _handle_right_click_paste(self, event: _SgrMouseEvent) -> bool:
        if self._on_right_click_paste is None or sys.platform != "win32" or os.environ.get("TERM_PROGRAM", "").lower() == "vscode" or event.release or event.button != 2:
            return False
        try:
            self._on_right_click_paste()
        except Exception:
            # This source callback is explicitly best-effort.
            pass
        return True

    def _handle_scroll_to_end_indicator_mouse_event(self, event: _SgrMouseEvent) -> bool:
        rect = self._scroll_to_end_indicator_rect
        if rect is None or event.release or _button_bits(event.button) & 32 or _button_bits(event.button) & 3:
            return False
        if event.y != rect.row or event.x < rect.column or event.x >= rect.column + rect.width:
            return False
        self.scroll_to_bottom()
        return True

    def _get_scrollbar_target_at(self, x: int, y: int, include_hidden_auto: bool = False) -> _ScrollbarTarget | None:
        if self.has_overlay() or self._current_layout is None:
            return None
        for scroll_view in get_scroll_views_at(self._current_layout, x, y):
            box = get_scroll_view_box(self._current_layout, scroll_view)
            geometry = get_scrollbar_geometry(box, include_hidden_auto) if box is not None else None
            if geometry is not None and x == geometry.column and geometry.track_top <= y < geometry.track_top + geometry.track_height:
                return _ScrollbarTarget(scroll_view, geometry)
        return None

    def _set_scrollbar_hover(self, scroll_view: ScrollView | None) -> None:
        if scroll_view is self._scrollbar_hover:
            return
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(False)
        self._scrollbar_hover = scroll_view
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(True)

    def _update_scrollbar_hover(self, x: int, y: int) -> None:
        target = self._get_scrollbar_target_at(x, y, True)
        self._set_scrollbar_hover(target.scroll_view if target is not None else None)

    def _stop_scrollbar_hover(self) -> None:
        self._set_scrollbar_hover(None)

    @staticmethod
    def _scroll_scrollbar_to_pointer(scroll_view: ScrollView, geometry: ScrollbarGeometry, pointer_y: int, grab_offset: int) -> None:
        maximum = geometry.track_height - geometry.thumb_height
        offset = max(0, min(maximum, pointer_y - geometry.track_top - grab_offset))
        ratio = 0 if maximum == 0 else offset / maximum * geometry.max_scroll_top
        scroll_top = math.floor(ratio) + (1 if ratio - math.floor(ratio) >= 0.5 else 0)
        scroll_view.scroll_to(scroll_top)

    def _handle_scrollbar_mouse_event(self, event: _SgrMouseEvent) -> bool:
        if self._scrollbar_drag is not None:
            if event.release:
                self._stop_scrollbar_drag()
                return True
            box = get_scroll_view_box(self._current_layout, self._scrollbar_drag.scroll_view) if self._current_layout is not None else None
            geometry = get_scrollbar_geometry(box) if box is not None else None
            if geometry is not None:
                self._scroll_scrollbar_to_pointer(self._scrollbar_drag.scroll_view, geometry, event.y, self._scrollbar_drag.grab_offset)
            return True
        if event.release or _button_bits(event.button) & 32 or _button_bits(event.button) & 3:
            return False
        target = self._get_scrollbar_target_at(event.x, event.y)
        if target is None:
            return False
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._set_scrollbar_hover(target.scroll_view)
        on_thumb = target.geometry.thumb_top <= event.y < target.geometry.thumb_top + target.geometry.thumb_height
        grab_offset = event.y - target.geometry.thumb_top if on_thumb else target.geometry.thumb_height // 2
        if not on_thumb:
            self._scroll_scrollbar_to_pointer(target.scroll_view, target.geometry, event.y, grab_offset)
        self._scrollbar_drag = _ScrollbarDrag(target.scroll_view, grab_offset)
        return True

    def _stop_scrollbar_drag(self) -> None:
        self._scrollbar_drag = None

    def _get_scroll_selection_point(self, scroll_view: ScrollView, x: int, y: int) -> _SelectionPoint | None:
        if self._current_layout is None:
            return None
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            return None
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(self.terminal.rows - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1)
        if visible_bottom < visible_top:
            return None
        pointer_row = max(visible_top, min(visible_bottom, y))
        max_content_row = max(0, (len(box.scroll_content_lines) if box.scroll_content_lines is not None else 1) - 1)
        return _SelectionPoint(
            cast(int, max(0, min(max_content_row, scroll_view.scroll_top + pointer_row - box.rect.y))),
            max(0, min(box.rect.width - 1, x - box.rect.x)), scroll_view,
        )

    def _get_selection_point(self, event: _SgrMouseEvent, scroll_view: ScrollView | None = None) -> _SelectionPoint:
        if scroll_view is not None:
            point = self._get_scroll_selection_point(scroll_view, event.x, event.y)
            if point is not None:
                return point
        return _SelectionPoint(cast(int, max(0, min(self.terminal.rows - 1, event.y))), cast(int, max(0, min(self.terminal.columns - 1, event.x))))

    def _get_selection_source_line(self, point: _SelectionPoint) -> str:
        if point.scroll_view is not None and self._current_layout is not None:
            box = get_scroll_view_box(self._current_layout, point.scroll_view)
            if box is not None and box.scroll_content_lines is not None:
                return _line_at(box.scroll_content_lines, point.row)
        return _line_at(self._previous_screen, point.row)

    def _get_word_selection(self, point: _SelectionPoint) -> _SelectionRange | None:
        line = strip_terminal_sequences(self._get_selection_source_line(point))
        segments: list[_WordSegment] = []
        start = 0
        for segment in _WORD_SEGMENTER.segment(line):
            end = start + visible_width(segment.segment)
            joiner = segment.segment in _TERMINAL_WORD_SELECTION_JOINERS
            segments.append(_WordSegment(start, end, segment.is_word_like is True or joiner, joiner))
            start = end
        clicked = next((index for index, segment in enumerate(segments) if segment.start <= point.col < segment.end), -1)
        if clicked < 0:
            return None
        selection_start, selection_end = segments[clicked].start, segments[clicked].end
        index = clicked
        while index > 0:
            left, right = segments[index - 1], segments[index]
            if not (left.selectable and right.selectable and (left.joiner or right.joiner)):
                break
            selection_start = left.start
            index -= 1
        index = clicked
        while index < len(segments) - 1:
            left, right = segments[index], segments[index + 1]
            if not (left.selectable and right.selectable and (left.joiner or right.joiner)):
                break
            selection_end = right.end
            index += 1
        return _SelectionRange(replace(point, col=selection_start), replace(point, col=selection_end, boundary=True))

    def _get_line_selection(self, point: _SelectionPoint) -> _SelectionRange:
        return _SelectionRange(replace(point, col=0), replace(point, col=visible_width(self._get_selection_source_line(point)), boundary=True))

    def _update_selection_focus(self, point: _SelectionPoint) -> None:
        if self._selection_granularity == "character" or self._selection_initial_range is None:
            self._selection_focus = point
            return
        selected_range = self._get_word_selection(point) if self._selection_granularity == "word" else self._get_line_selection(point)
        if selected_range is None:
            return
        initial = self._selection_initial_range
        before = selected_range.start.row < initial.start.row or selected_range.start.row == initial.start.row and selected_range.start.col < initial.start.col
        if before:
            self._selection_anchor = initial.end
            self._selection_focus = selected_range.start
        else:
            self._selection_anchor = initial.start
            self._selection_focus = selected_range.end

    def _get_click_count(self, point: _SelectionPoint, word: _SelectionRange | None) -> int:
        now = time.time_ns() // 1_000_000
        previous = self._last_click
        count = previous.count % 3 + 1 if word is not None and previous is not None and now - previous.timestamp <= _DOUBLE_CLICK_INTERVAL_MS and previous.row == point.row and previous.scroll_view is point.scroll_view and previous.word_start == word.start.col and previous.word_end == word.end.col else 1
        self._last_click = _ClickTarget(now, count, point.row, point.scroll_view, word.start.col, word.end.col) if word is not None else None
        return count

    def _update_selection_auto_scroll(self, event: _SgrMouseEvent) -> None:
        scroll_view = self._selection_anchor.scroll_view if self._selection_anchor is not None else None
        if scroll_view is None or self._current_layout is None:
            self._stop_selection_auto_scroll()
            return
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            self._stop_selection_auto_scroll()
            return
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(self.terminal.rows - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1)
        self._selection_drag_pointer = _Point(event.x, event.y)
        self._selection_auto_scroll_direction = -1 if event.y <= visible_top else 1 if event.y >= visible_bottom else 0
        if self._selection_auto_scroll_direction == 0:
            self._stop_selection_auto_scroll()
            return
        if self._selection_auto_scroll_timer is not None:
            return
        self._selection_timer_generation += 1
        generation = self._selection_timer_generation

        def tick() -> None:
            with self._state_lock:
                if generation != self._selection_timer_generation:
                    return
                # Start the next interval from this callback's admission, as Node does.
                self._selection_auto_scroll_timer = set_timeout(tick, 50, unref=True)
                self._auto_scroll_selection()

        self._selection_auto_scroll_timer = set_timeout(tick, 50, unref=True)

    def _auto_scroll_selection(self) -> None:
        scroll_view = self._selection_anchor.scroll_view if self._selection_anchor is not None else None
        pointer = self._selection_drag_pointer
        direction = self._selection_auto_scroll_direction
        if scroll_view is None or pointer is None or direction == 0:
            self._stop_selection_auto_scroll()
            return
        remaining = scroll_view.scroll_by(direction)
        if remaining == direction:
            self._stop_selection_auto_scroll()
            return
        point = self._get_scroll_selection_point(scroll_view, pointer.x, pointer.y)
        if point is not None:
            self._update_selection_focus(point)
        self.request_render()

    def _stop_selection_auto_scroll(self) -> None:
        self._selection_timer_generation += 1
        if self._selection_auto_scroll_timer is not None:
            self._selection_auto_scroll_timer.cancel()
            self._selection_auto_scroll_timer = None
        self._selection_auto_scroll_direction = 0
        self._selection_drag_pointer = None

    def _handle_selection_mouse_event(self, event: _SgrMouseEvent) -> None:
        button = _button_bits(event.button) & 3
        if button != 0 and not (event.release and button == 3):
            return
        anchor_scroll_view = self._selection_anchor.scroll_view if self._selection_anchor is not None else None
        point = self._get_selection_point(event, anchor_scroll_view)
        if event.release:
            if not self._selection_press_active:
                return
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            if self._selection_anchor is None:
                return
            self._update_selection_focus(point)
            is_click = not self._selection_dragged and self._selection_anchor.scroll_view is point.scroll_view and self._selection_anchor.row == point.row and self._selection_anchor.col == point.col
            clicked_url = self._pressed_url if is_click else None
            self._pressed_url = None
            if clicked_url and self._open_url is not None:
                self._selection_anchor = None
                self._selection_focus = None
                try:
                    self._open_url(clicked_url)
                except Exception:
                    # This source callback is explicitly best-effort.
                    pass
                self.request_render()
                return
            if is_click:
                click = self._create_mouse_event("click", event.button, event.x, event.y, click_count=self._last_click.count if self._last_click is not None else 1)
                overlay = self._dispatch_mouse_to_overlay(click)
                result = overlay.result if overlay.result is not None else None if overlay.hit else self._dispatch_mouse_to_layout(click)
                if result is not None:
                    render = self._apply_mouse_dispatch_result(click, result)
                    self._clear_text_selection()
                    if render:
                        self.request_render()
                    return
            if self._copy_on_select:
                self._copy_selection_to_clipboard(report_unhandled=True)
            self.request_render()
            return
        if _button_bits(event.button) & 32:
            if not self._selection_press_active or self._selection_anchor is None:
                return
            self._selection_dragged = True
            self._last_click = None
            self._pressed_url = None
            self._update_selection_focus(point)
            self._update_selection_auto_scroll(event)
            self.request_render()
            return
        self._stop_selection_auto_scroll()
        self._selection_press_active = True
        scroll_views = get_scroll_views_at(self._current_layout, event.x, event.y) if not self.has_overlay() and self._current_layout is not None else []
        anchor = self._get_selection_point(event, scroll_views[0] if scroll_views else None)
        word = self._get_word_selection(anchor)
        click_count = self._get_click_count(anchor, word)
        selected_range = word if click_count == 2 else self._get_line_selection(anchor) if click_count == 3 else None
        self._selection_granularity = ("word" if click_count == 2 else "line") if selected_range is not None else "character"
        self._selection_initial_range = selected_range
        self._selection_anchor = selected_range.start if selected_range is not None else anchor
        self._selection_focus = selected_range.end if selected_range is not None else anchor
        self._selection_dragged = False
        self._pressed_url = None if selected_range is not None else get_osc8_link_at_column(
            _line_at(self._previous_screen, cast(int, max(0, min(self.terminal.rows - 1, event.y)))),
            cast(int, max(0, min(self.terminal.columns - 1, event.x))),
        )
        self.request_render()

    def _get_selection_bounds(self) -> _SelectionRange | None:
        anchor, focus = self._selection_anchor, self._selection_focus
        if anchor is None or focus is None or anchor.scroll_view is not focus.scroll_view:
            return None
        before = anchor.row < focus.row or anchor.row == focus.row and anchor.col < focus.col
        if anchor.row == focus.row and anchor.col == focus.col:
            return None
        return _SelectionRange(anchor, focus) if before else _SelectionRange(focus, anchor)

    @staticmethod
    def _get_selection_columns(line: str, row: int, selection: _SelectionRange, min_column: int = 0, max_column: int | None = None) -> tuple[int, int]:
        line_width = visible_width(line)
        if max_column is None:
            max_column = line_width
        start, end = max(0, min_column), min(line_width, max_column)
        if row == selection.start.row:
            grapheme = get_grapheme_cell_range(line, selection.start.col)
            start = grapheme.start if grapheme is not None else min(selection.start.col, line_width)
        if row == selection.end.row:
            if selection.end.boundary:
                end = min(selection.end.col, line_width)
            else:
                grapheme = get_grapheme_cell_range(line, selection.end.col)
                end = grapheme.end if grapheme is not None else min(selection.end.col + 1, line_width)
        return max(min_column, start), min(max_column, end)

    def _get_active_selection_text(self) -> str | None:
        selection = self._get_selection_bounds()
        if selection is None:
            return None
        source_lines: Sequence[str] = self._previous_screen
        if selection.start.scroll_view is not None:
            if self._current_layout is None:
                return None
            box = get_scroll_view_box(self._current_layout, selection.start.scroll_view)
            if box is None or box.scroll_content_lines is None:
                return None
            source_lines = box.scroll_content_lines
        lines: list[str] = []
        for row in range(selection.start.row, selection.end.row + 1):
            line = _line_at(source_lines, row)
            start, end = self._get_selection_columns(line, row, selection)
            lines.append(strip_terminal_sequences(slice_by_column(line, start, max(0, end - start), True)).rstrip(JS_WHITESPACE))
        text = "\n".join(lines)
        return text if text else None

    def _copy_selection_to_clipboard(self, *, report_unhandled: bool = False) -> Awaitable[bool]:
        # An async JS function admits work synchronously, even when its promise
        # is discarded. Do the same instead of returning a lazy Python coroutine.
        promise = _QueryPromise[bool]()
        try:
            text = self._get_active_selection_text()
            if not text:
                promise.resolve(False)
            else:
                self._copy_text_to_clipboard(text, promise, report_unhandled)
        except Exception as error:
            promise.reject(error)
            if report_unhandled:
                self._report_copy_error(error)
        return promise

    def _report_copy_error(self, error: BaseException) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.call_exception_handler, {
                "message": "Unhandled fullscreen clipboard exception", "exception": error,
            })
        else:
            self._next_tick(lambda: sys.excepthook(type(error), error, error.__traceback__))

    def _copy_text_to_clipboard(self, text: str, promise: _QueryPromise[bool], report_unhandled: bool) -> None:
        if self._copy_selection is None:
            # Buffer.from replaces unpaired UTF-16 surrogates with U+FFFD.
            encoded = text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace").encode("utf-8")
            self.terminal.write("\x1b]52;c;" + base64.b64encode(encoded).decode("ascii") + "\x07")
            self.flash("Copied!")
            promise.resolve(True)
            return
        pending = self._copy_selection(text)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is not None and isinstance(pending, Coroutine):
            # Match the synchronous prefix of an injected JS async function.
            # Result handling below still resumes in a later loop callback.
            pending = asyncio.Task(pending, loop=current_loop, eager_start=True)

        async def finish() -> None:
            try:
                result = await pending
                ok = result is True
                with self._state_lock:
                    self.flash("Copied!" if ok else result if isinstance(result, str) else "Copy failed", None if ok else _COPY_ERROR_FLASH_DURATION_MS)
                promise.resolve(ok)
            except BaseException as error:
                promise.reject(error)
                if report_unhandled:
                    self._report_copy_error(error)

        loop = current_loop if current_loop is not None else self._event_loop
        if loop is not None and loop.is_running():
            def start() -> None:
                task = loop.create_task(finish())
                self._clipboard_tasks.add(task)
                task.add_done_callback(self._clipboard_tasks.discard)

            # The callback itself was already called; its awaitable resumes on
            # the UI loop. This is also safe when native input runs on a thread.
            loop.call_soon_threadsafe(start)
        else:
            def run() -> None:
                asyncio.run(finish())

            threading.Thread(target=run, name="pi-tui-clipboard", daemon=True).start()

    def _apply_search_text_highlight(self, text: str, current: bool) -> str:
        style = self._search_current_match_style if current else self._search_match_style
        parts: list[str] = []
        plain_start = index = 0
        units = utf16_units(text)
        while index < len(units):
            ansi = extract_ansi_code(text, index) if units[index] == "\x1b" else None
            if ansi is None:
                index += 1
                continue
            if index > plain_start:
                parts.append(style(from_utf16_units(units[plain_start:index])))
            parts.append(ansi.code)
            index += ansi.length
            plain_start = index
        if plain_start < len(units):
            parts.append(style(from_utf16_units(units[plain_start:])))
        return "".join(parts)

    def _apply_search_highlights(self, screen: list[str], layout: LayoutFrame) -> list[str]:
        search = self._active_search
        if search is None or search.selected_index < 0 or not search.matches:
            return screen
        scroll_view = layout.primary_scroll_view if layout.primary_scroll_view is not None else self._implicit_scroll_view
        box = get_scroll_view_box(layout, scroll_view)
        if box is None:
            return screen
        ranges_by_row: dict[int, list[_SearchHighlightRange]] = {}
        geometry = get_scrollbar_geometry(box)
        min_row = max(0, box.rect.y, box.clip.y)
        max_row = min(len(screen), box.rect.y + box.rect.height, box.clip.y + box.clip.height)
        min_column = max(0, box.rect.x, box.clip.x)
        max_column = cast(int, min(self.terminal.columns, box.rect.x + box.rect.width, box.clip.x + box.clip.width, geometry.column if geometry is not None else math.inf))
        min_content_row = scroll_view.scroll_top + min_row - box.rect.y
        max_content_row = scroll_view.scroll_top + max_row - box.rect.y - 1
        low, high = 0, len(search.matches)
        while low < high:
            middle = low + (high - low) // 2
            match = search.matches[middle]
            last_row = match.segments[-1].row if match.segments else -1
            if last_row < min_content_row:
                low = middle + 1
            else:
                high = middle
        for match_index in range(low, len(search.matches)):
            match = search.matches[match_index]
            if (match.segments[0].row if match.segments else 0) > max_content_row:
                break
            for segment in match.segments:
                row = box.rect.y + segment.row - scroll_view.scroll_top
                if row < min_row or row >= max_row:
                    continue
                start_col = max(min_column, box.rect.x + segment.start_col)
                end_col = min(max_column, box.rect.x + segment.end_col)
                if end_col <= start_col:
                    continue
                ranges_by_row.setdefault(row, []).append(_SearchHighlightRange(start_col, end_col, match_index == search.selected_index))
        result = list(screen)
        for row, ranges in ranges_by_row.items():
            line = _line_at(result, row)
            if is_image_line(line):
                continue
            line_width = visible_width(line)
            for selected_range in sorted(ranges, key=lambda item: -item.start_col):
                start_col = min(selected_range.start_col, line_width)
                end_col = min(selected_range.end_col, line_width)
                if end_col <= start_col:
                    continue
                before = slice_by_column(line, 0, start_col, True)
                highlighted = slice_by_column(line, start_col, end_col - start_col, True)
                after = slice_by_column(line, end_col, max(0, line_width - end_col), True)
                line = before + self._apply_search_text_highlight(highlighted, selected_range.current) + after
            result[row] = line
        return result

    @staticmethod
    def _apply_selection_highlight(text: str) -> str:
        parts = ["\x1b[7m"]
        index = 0
        units = utf16_units(text)
        while index < len(units):
            ansi = extract_ansi_code(text, index) if units[index] == "\x1b" else None
            if ansi is None:
                parts.append(units[index])
                index += 1
                continue
            parts.append(ansi.code)
            if ansi.code.endswith("m"):
                parts.append("\x1b[7m")
            index += ansi.length
        parts.append("\x1b[27m")
        return from_utf16_units("".join(parts))

    def _apply_selection(self, screen: list[str], layout: LayoutFrame | None = None) -> list[str]:
        if layout is None:
            layout = self._current_layout
        selection = self._get_selection_bounds()
        if selection is None:
            return screen
        screen_selection = selection
        min_row, max_row = 0, len(screen) - 1
        min_column, max_column = 0, cast(int, self.terminal.columns)
        if selection.start.scroll_view is not None:
            if layout is None:
                return screen
            box = get_scroll_view_box(layout, selection.start.scroll_view)
            if box is None:
                return screen
            min_row = max(0, box.rect.y, box.clip.y)
            max_row = min(len(screen) - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1)
            min_column = max(0, box.rect.x, box.clip.x)
            max_column = cast(int, min(self.terminal.columns, box.rect.x + box.rect.width, box.clip.x + box.clip.width))
            screen_selection = _SelectionRange(
                replace(selection.start, row=box.rect.y + selection.start.row - selection.start.scroll_view.scroll_top, col=box.rect.x + selection.start.col),
                replace(selection.end, row=box.rect.y + selection.end.row - selection.start.scroll_view.scroll_top, col=box.rect.x + selection.end.col),
            )
        result: list[str] = []
        for row, line in enumerate(screen):
            if line is None or row < min_row or row > max_row or row < screen_selection.start.row or row > screen_selection.end.row or is_image_line(line):
                result.append(line)
                continue
            line_width = visible_width(line)
            start, end = self._get_selection_columns(line, row, screen_selection, min_column, max_column)
            if end <= start:
                result.append(line)
                continue
            before = slice_by_column(line, 0, start, True)
            selected = slice_by_column(line, start, end - start, True)
            after = slice_by_column(line, end, max(0, line_width - end), True)
            result.append(before + self._apply_selection_highlight(selected) + after)
        return result

    @staticmethod
    def _is_mouse_sequence(data: str) -> bool:
        return _SGR_MOUSE.fullmatch(data) is not None or len(utf16_units(data)) == 6 and data.startswith("\x1b[M")

    def _composite_scroll_to_end_indicator(self, screen: list[str], layout: LayoutFrame, width: int) -> list[str]:
        self._scroll_to_end_indicator_rect = None
        scroll_view = layout.primary_scroll_view if layout.primary_scroll_view is not None else self._implicit_scroll_view
        if self._scroll_to_end_indicator is None or not scroll_view.follow_end or scroll_view.is_following_end:
            return screen
        box = get_scroll_view_box(layout, scroll_view)
        clip = box.clip if box is not None else None
        if clip is None or clip.width <= 0 or clip.height <= 0:
            return screen
        row = clip.y + clip.height - 1
        if row >= len(screen) or is_image_line(_line_at(screen, row)):
            return screen
        geometry = get_scrollbar_geometry(box) if box is not None else None
        available_width = max(0, (geometry.column if geometry is not None else clip.x + clip.width) - clip.x)
        text = truncate_to_width(self._scroll_to_end_indicator(), available_width, "")
        text_width = visible_width(text)
        if text_width == 0:
            return screen
        column = clip.x + (available_width - text_width) // 2
        result = list(screen)
        result[row] = composite_tui_line(_line_at(result, row), text, column, text_width, width)
        self._scroll_to_end_indicator_rect = _ScrollToEndIndicatorRect(row, column, text_width)
        return result

    def _composite_flashes(self, screen: list[str], width: int, height: int) -> list[str]:
        flash_lines = self._flashes.render(width)[-height:]
        if not flash_lines:
            return screen
        result = list(screen)
        while len(result) < height:
            result.append("")
        for row, line in enumerate(flash_lines):
            flash_width = visible_width(line)
            if flash_width == 0:
                continue
            result[row] = composite_tui_line(_line_at(result, row), line, width - flash_width, flash_width, width)
        return result

    def _do_render(self) -> None:
        if self._stopped or not self._alt_screen_active:
            return
        width = _terminal_extent(self.terminal.columns)
        height = _terminal_extent(self.terminal.rows)
        root = self._layout_root if self._layout_root is not None else self._implicit_scroll_view
        next_layout = render_layout_frame(root, width, height, self.request_render)
        if self._refresh_search(next_layout):
            next_layout = render_layout_frame(root, width, height, self.request_render)
        screen = [line if line is None else _OSC133_ZONE_PREFIX.sub("", line) for line in next_layout.lines]
        screen = self._apply_search_highlights(screen, next_layout)
        screen = self._composite_scroll_to_end_indicator(screen, next_layout, width)
        screen = self._composite_overlays(screen, width, height)
        if len(screen) > height:
            screen = screen[len(screen) - height:]
        screen = self._apply_selection(screen, next_layout)
        screen = self._composite_flashes(screen, width, height)
        cursor = self._extract_cursor_position(screen, height)
        screen = [
            line if is_image_line(line) or visible_width(line) <= width else slice_by_column(line, 0, width, True)
            for line in self._apply_line_resets(screen)
        ]
        full_redraw = not self._previous_screen or self._previous_screen_width != width or self._previous_screen_height != height
        images_need_redraw = any(
            (row >= len(self._previous_screen) or utf16_units(line) != utf16_units(_line_at(self._previous_screen, row)))
            and (is_image_line(line) or is_image_line(_line_at(self._previous_screen, row)))
            for row, line in enumerate(screen)
        )
        redraw_images = full_redraw or images_need_redraw
        had_uploaded_kitty_images = bool(self._uploaded_kitty_images)
        prepared_lines, evicted_deletion = self._prepare_kitty_screen(screen) if redraw_images and self._image_protocol == "kitty" else (screen, "")
        parts = [_BEGIN_SYNCHRONIZED_OUTPUT]
        if full_redraw:
            self._full_redraw_count += 1
            clear_images = delete_all_kitty_placements() if self._image_protocol == "kitty" and had_uploaded_kitty_images else self._delete_kitty_images()
            parts.extend((clear_images, "\x1b[2J"))
        elif images_need_redraw:
            if self._image_protocol == "iterm2":
                parts.append("\x1b[2J")
            elif self._image_protocol == "kitty":
                parts.append(delete_all_kitty_placements())
        parts.append(evicted_deletion)
        for row in range(height):
            if not full_redraw and not images_need_redraw:
                current = screen[row] if row < len(screen) else None
                previous = self._previous_screen[row] if row < len(self._previous_screen) else None
                if current is None and previous is None or current is not None and previous is not None and utf16_units(current) == utf16_units(previous):
                    continue
            parts.extend((f"\x1b[{row + 1};1H\x1b[2K", _line_at(prepared_lines, row)))
        if cursor is not None:
            parts.append(f"\x1b[{cursor.row + 1};{min(width, cursor.col) + 1}H")
            parts.append("\x1b[?25h" if self.get_show_hardware_cursor() else "\x1b[?25l")
        else:
            parts.append("\x1b[?25l")
        parts.append(_END_SYNCHRONIZED_OUTPUT)
        self.terminal.write("".join(parts))
        self._previous_screen = screen
        self._previous_screen_width = width
        self._previous_screen_height = height
        self._current_layout = next_layout


__all__ = ["TuiAltScreen", "TuiAltScreenOptions"]
