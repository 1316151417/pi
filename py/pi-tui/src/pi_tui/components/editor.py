"""Multiline editor ported from ``components/editor.ts``.

Cursor and segment offsets remain UTF-16 offsets. ``word_wrap_line`` retains
the source's recursive boundary for an indivisible grapheme wider than the
available width; runtime behavior has not yet been tested.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import math
import re
import threading
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal, TypedDict, cast

from .._component import CURSOR_MARKER, TUI, TuiMouseEvent, TuiMouseEventResult
from .._javascript import JS_WHITESPACE, js_repeat, js_trim, utf16_length, utf16_slice, utf16_units
from .._segmenter import SegmentData, Segmenter
from .._timers import TimerHandle, set_timeout
from ..abort import AbortController
from ..autocomplete import AutocompleteItem, AutocompleteOptions, AutocompleteProvider, AutocompleteSuggestions
from ..keybindings import KeybindingsManager, get_keybindings
from ..keys import decode_printable_key
from ..kill_ring import KillRing, KillRingPushOptions
from ..undo_stack import UndoStack
from ..utils import cjk_break_regex, get_grapheme_segmenter, get_word_segmenter, is_whitespace_char, slice_by_column, visible_width
from ..word_navigation import WordNavigationOptions, find_word_backward, find_word_forward
from .select_list import SelectItem, SelectList, SelectListLayoutOptions, SelectListTheme

_grapheme_segmenter = get_grapheme_segmenter()
_word_segmenter = get_word_segmenter()
_PASTE_MARKER_REGEX = re.compile(r"\[paste #([0-9]+)( (\+[0-9]+ lines|[0-9]+ chars))?\]")
_SLASH_COMMAND_SELECT_LIST_LAYOUT = SelectListLayoutOptions(min_primary_column_width=12, max_primary_column_width=32)
_ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS = 20
_DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS = ["@", "#"]
_background_loop: asyncio.AbstractEventLoop | None = None
_background_loop_lock = threading.Lock()


def _completion_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    global _background_loop
    with _background_loop_lock:
        if _background_loop is None:
            ready = threading.Event()

            def run() -> None:
                global _background_loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _background_loop = loop
                ready.set()
                loop.run_forever()

            threading.Thread(target=run, daemon=True, name="pi.editor.autocomplete").start()
            ready.wait()
        assert _background_loop is not None
        return _background_loop


def _is_paste_marker(segment: str) -> bool:
    return utf16_length(segment) >= 10 and _PASTE_MARKER_REGEX.fullmatch(segment) is not None


def _segment_with_markers(text: str, base_segmenter: Segmenter, valid_ids: set[int]) -> Iterable[SegmentData]:
    if not valid_ids or "[paste #" not in text:
        return base_segmenter.segment(text)
    markers: list[tuple[int, int]] = []
    for match in _PASTE_MARKER_REGEX.finditer(text):
        if float(match[1]) in valid_ids:
            markers.append((utf16_length(text[:match.start()]), utf16_length(text[:match.end()])))
    if not markers:
        return base_segmenter.segment(text)
    result: list[SegmentData] = []
    marker_index = 0
    for segment in base_segmenter.segment(text):
        while marker_index < len(markers) and markers[marker_index][1] <= segment.index:
            marker_index += 1
        marker = markers[marker_index] if marker_index < len(markers) else None
        if marker is not None and marker[0] <= segment.index < marker[1]:
            if segment.index == marker[0]:
                result.append(SegmentData(utf16_slice(text, *marker), marker[0], text))
        else:
            result.append(segment)
    return result


@dataclass
class TextChunk:
    text: str
    start_index: int
    end_index: int


def word_wrap_line(line: str, max_width: int, pre_segmented: list[SegmentData] | None = None) -> list[TextChunk]:
    if not line or max_width <= 0:
        return [TextChunk("", 0, 0)]
    if visible_width(line) <= max_width:
        return [TextChunk(line, 0, utf16_length(line))]
    chunks: list[TextChunk] = []
    segments = pre_segmented if pre_segmented is not None else list(_grapheme_segmenter.segment(line))
    current_width = 0
    chunk_start = 0
    wrap_opportunity_index = -1
    wrap_opportunity_width = 0
    for index, segment in enumerate(segments):
        grapheme = segment.segment
        grapheme_width = visible_width(grapheme)
        char_index = segment.index
        whitespace = not _is_paste_marker(grapheme) and is_whitespace_char(grapheme)
        if current_width + grapheme_width > max_width:
            if wrap_opportunity_index >= 0 and current_width - wrap_opportunity_width + grapheme_width <= max_width:
                chunks.append(TextChunk(utf16_slice(line, chunk_start, wrap_opportunity_index), chunk_start, wrap_opportunity_index))
                chunk_start = wrap_opportunity_index
                current_width -= wrap_opportunity_width
            elif chunk_start < char_index:
                chunks.append(TextChunk(utf16_slice(line, chunk_start, char_index), chunk_start, char_index))
                chunk_start = char_index
                current_width = 0
            wrap_opportunity_index = -1
        if grapheme_width > max_width:
            sub_chunks = word_wrap_line(grapheme, max_width)
            for chunk in sub_chunks[:-1]:
                chunks.append(TextChunk(chunk.text, char_index + chunk.start_index, char_index + chunk.end_index))
            last = sub_chunks[-1]
            chunk_start = char_index + last.start_index
            current_width = visible_width(last.text)
            wrap_opportunity_index = -1
            continue
        current_width += grapheme_width
        next_segment = segments[index + 1] if index + 1 < len(segments) else None
        if whitespace and next_segment and (_is_paste_marker(next_segment.segment) or not is_whitespace_char(next_segment.segment)):
            wrap_opportunity_index = next_segment.index
            wrap_opportunity_width = current_width
        elif not whitespace and next_segment and not is_whitespace_char(next_segment.segment):
            is_cjk = not _is_paste_marker(grapheme) and cjk_break_regex.test(grapheme)
            next_is_cjk = not _is_paste_marker(next_segment.segment) and cjk_break_regex.test(next_segment.segment)
            if is_cjk or next_is_cjk:
                wrap_opportunity_index = next_segment.index
                wrap_opportunity_width = current_width
    chunks.append(TextChunk(utf16_slice(line, chunk_start), chunk_start, utf16_length(line)))
    return chunks


@dataclass
class _EditorState:
    lines: list[str] = field(default_factory=lambda: [""])
    cursor_line: int = 0
    cursor_col: int = 0


@dataclass
class _EditorSnapshot:
    state: _EditorState
    pastes: dict[int, str]
    paste_counter: int


@dataclass
class _LayoutLine:
    text: str
    has_cursor: bool
    cursor_pos: int | None = None


@dataclass
class _VisualLine:
    logical_line: int
    start_col: int
    length: int


@dataclass
class EditorTheme:
    border_color: Callable[[str], str]
    select_list: SelectListTheme


@dataclass
class EditorOptions:
    padding_x: int | float | None = None
    autocomplete_max_visible: int | float | None = None


class EditorCursor(TypedDict):
    line: int
    col: int


@dataclass
class _AutocompleteRequestOptions:
    force: bool
    explicit_tab: bool


def _build_trigger_pattern(characters: list[str]) -> re.Pattern[str]:
    whitespace = re.escape(JS_WHITESPACE)
    return re.compile(r"(?:^|[" + whitespace + r"])[" + "".join(re.escape(char) for char in characters) + r"][^" + whitespace + r"]*\Z")


def _build_debounce_pattern(characters: list[str]) -> re.Pattern[str]:
    without_at = "".join(re.escape(char) for char in characters if char != "@")
    non_whitespace = "[^" + re.escape(JS_WHITESPACE) + "]*"
    return re.compile(r'(?:^|[ \t])(?:@(?:"[^"]*|' + non_whitespace + r")|[" + without_at + "]" + non_whitespace + r")\Z")


def _create_scroll_border(direction: Literal["↑", "↓"], hidden_line_count: int, width: int) -> str:
    available_width = max(0, width)
    label = f" {direction} {hidden_line_count} more "
    label_width = visible_width(label)
    if label_width + 2 <= available_width:
        left_width = math.floor((available_width - label_width) / 2)
        return js_repeat("─", left_width) + label + js_repeat("─", available_width - left_width - label_width)
    indicator = f"─── {direction} {hidden_line_count} more "
    remaining = available_width - visible_width(indicator)
    if remaining >= 0:
        return indicator + js_repeat("─", remaining)
    ellipsis = "..."[:available_width]
    return slice_by_column(indicator, 0, available_width - visible_width(ellipsis), True) + ellipsis


class Editor:
    def __init__(self, tui: TUI, theme: EditorTheme, options: EditorOptions | None = None) -> None:
        options = options if options is not None else EditorOptions()
        self._state = _EditorState()
        self.focused = False
        self.tui = tui
        self._theme = theme
        self.border_color = theme.border_color
        padding = options.padding_x if options.padding_x is not None else 0
        self._padding_x = max(0, math.floor(padding)) if math.isfinite(padding) else 0
        maximum = options.autocomplete_max_visible if options.autocomplete_max_visible is not None else 5
        self._autocomplete_max_visible = max(3, min(20, math.floor(maximum))) if math.isfinite(maximum) else 5
        self._last_width = 80
        self._rendered_visible_line_count = 1
        self._rendered_autocomplete_height = 0
        self._scroll_offset = 0
        self._autocomplete_provider: AutocompleteProvider | None = None
        self._autocomplete_trigger_characters = list(_DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS)
        self._autocomplete_trigger_pattern = _build_trigger_pattern(self._autocomplete_trigger_characters)
        self._autocomplete_debounce_pattern = _build_debounce_pattern(self._autocomplete_trigger_characters)
        self._autocomplete_list: SelectList | None = None
        self._autocomplete_state: Literal["regular", "force"] | None = None
        self._autocomplete_prefix = ""
        self._autocomplete_abort: AbortController | None = None
        self._autocomplete_debounce_timer: TimerHandle | None = None
        self._autocomplete_request_task: asyncio.Future[None] | concurrent.futures.Future[None] | None = None
        self._autocomplete_start_token = 0
        self._autocomplete_request_id = 0
        self._autocomplete_loop: asyncio.AbstractEventLoop | None = None
        self._autocomplete_start_lock = threading.RLock()
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0
        self._paste_buffer = ""
        self._is_in_paste = False
        self._history: list[str] = []
        self._history_index = -1
        self._history_draft: _EditorState | None = None
        self._kill_ring = KillRing()
        self._last_action: Literal["kill", "yank", "type-word"] | None = None
        self._jump_mode: Literal["forward", "backward"] | None = None
        self._preferred_visual_col: int | None = None
        self._snapped_from_cursor_col: int | None = None
        self._undo_stack = UndoStack[_EditorSnapshot]()
        self.on_submit: Callable[[str], None] | None = None
        self.on_change: Callable[[str], None] | None = None
        self.disable_submit = False

    def _segment(self, text: str, mode: Literal["word", "grapheme"]) -> Iterable[SegmentData]:
        return _segment_with_markers(text, _word_segmenter if mode == "word" else _grapheme_segmenter, set(self._pastes))

    def get_padding_x(self) -> int:
        return self._padding_x

    def set_padding_x(self, padding: int | float) -> None:
        value = max(0, math.floor(padding)) if math.isfinite(padding) else 0
        if self._padding_x != value:
            self._padding_x = value
            self.tui.request_render()

    def get_autocomplete_max_visible(self) -> int:
        return self._autocomplete_max_visible

    def set_autocomplete_max_visible(self, maximum: int | float) -> None:
        value = max(3, min(20, math.floor(maximum))) if math.isfinite(maximum) else 5
        if self._autocomplete_max_visible != value:
            self._autocomplete_max_visible = value
            self.tui.request_render()

    def set_autocomplete_provider(self, provider: AutocompleteProvider) -> None:
        self._cancel_autocomplete()
        self._autocomplete_provider = provider
        self._set_autocomplete_trigger_characters(getattr(provider, "trigger_characters", None) or [])

    def add_to_history(self, text: str) -> None:
        trimmed = js_trim(text)
        if not trimmed or self._history and self._history[0] == trimmed:
            return
        self._history.insert(0, trimmed)
        if len(self._history) > 100:
            self._history.pop()

    def _is_editor_empty(self) -> bool:
        return self._state.lines == [""]

    def _is_on_first_visual_line(self) -> bool:
        return self._find_current_visual_line(self._build_visual_line_map(self._last_width)) == 0

    def _is_on_last_visual_line(self) -> bool:
        visual_lines = self._build_visual_line_map(self._last_width)
        return self._find_current_visual_line(visual_lines) == len(visual_lines) - 1

    def _navigate_history(self, direction: Literal[-1, 1]) -> None:
        self._last_action = None
        if not self._history:
            return
        new_index = self._history_index - direction
        if new_index < -1 or new_index >= len(self._history):
            return
        if self._history_index == -1 and new_index >= 0:
            self._push_undo_snapshot()
            self._history_draft = deepcopy(self._state)
        self._history_index = new_index
        if new_index == -1:
            draft = self._history_draft
            self._history_draft = None
            if draft is not None:
                self._state = draft
                self._preferred_visual_col = None
                self._snapped_from_cursor_col = None
                self._scroll_offset = 0
                if self.on_change is not None:
                    self.on_change(self.get_text())
            else:
                self._set_text_internal("")
        else:
            self._set_text_internal(self._history[new_index] or "", "start" if direction == -1 else "end")

    def _exit_history_browsing(self) -> None:
        self._history_index = -1
        self._history_draft = None

    def _set_text_internal(self, text: str, cursor_placement: Literal["start", "end"] = "end") -> None:
        self._state.lines = text.split("\n") or [""]
        self._state.cursor_line = 0 if cursor_placement == "start" else len(self._state.lines) - 1
        self._set_cursor_col(0 if cursor_placement == "start" else utf16_length(self._state.lines[self._state.cursor_line]))
        self._scroll_offset = 0
        if self.on_change is not None:
            self.on_change(self.get_text())

    def invalidate(self) -> None:
        # The source has no cached state to invalidate.
        pass

    def render_top_border(self, width: int, hidden_line_count: int) -> str:
        return self.border_color(_create_scroll_border("↑", hidden_line_count, width) if hidden_line_count > 0 else js_repeat("─", width))

    def render_bottom_border(self, width: int, hidden_line_count: int) -> str:
        return self.border_color(_create_scroll_border("↓", hidden_line_count, width) if hidden_line_count > 0 else js_repeat("─", width))

    def render(self, width: int) -> list[str]:
        padding_x = min(self._padding_x, max(0, math.floor((width - 1) / 2)))
        content_width = max(1, width - padding_x * 2)
        layout_width = max(1, content_width - (0 if padding_x else 1))
        self._last_width = layout_width
        layout_lines = self._layout_text(layout_width)
        visible_rows = self.tui.terminal.rows * 0.3
        if visible_rows == math.inf:
            maximum = len(layout_lines)
        elif visible_rows == -math.inf:
            maximum = 5
        else:
            maximum = max(5, math.floor(visible_rows))
        cursor_line = next((index for index, line in enumerate(layout_lines) if line.has_cursor), 0)
        if cursor_line < self._scroll_offset:
            self._scroll_offset = cursor_line
        elif cursor_line >= self._scroll_offset + maximum:
            self._scroll_offset = cursor_line - maximum + 1
        self._scroll_offset = max(0, min(self._scroll_offset, max(0, len(layout_lines) - maximum)))
        visible_lines = layout_lines[self._scroll_offset:self._scroll_offset + maximum]
        self._rendered_visible_line_count = len(visible_lines)
        result = [self.render_top_border(width, self._scroll_offset)]
        left_padding = js_repeat(" ", padding_x)
        right_padding = left_padding
        for line in visible_lines:
            display_text = line.text
            line_width = visible_width(display_text)
            cursor_in_padding = False
            if line.has_cursor and line.cursor_pos is not None:
                before = utf16_slice(display_text, 0, line.cursor_pos)
                after = utf16_slice(display_text, line.cursor_pos)
                marker = CURSOR_MARKER if self.focused else ""
                if after:
                    after_graphemes = list(self._segment(after, "grapheme"))
                    first = after_graphemes[0].segment if after_graphemes else ""
                    display_text = before + marker + f"\x1b[7m{first}\x1b[0m" + utf16_slice(after, utf16_length(first))
                else:
                    display_text = before + marker + "\x1b[7m \x1b[0m"
                    line_width += 1
                    if line_width > content_width and padding_x > 0:
                        cursor_in_padding = True
            padding = js_repeat(" ", max(0, content_width - line_width))
            result.append(left_padding + display_text + padding + (right_padding[1:] if cursor_in_padding else right_padding))
        below = len(layout_lines) - (self._scroll_offset + len(visible_lines))
        result.append(self.render_bottom_border(width, below))
        self._rendered_autocomplete_height = 0
        if self._autocomplete_state and self._autocomplete_list is not None:
            autocomplete_result = self._autocomplete_list.render(content_width)
            self._rendered_autocomplete_height = len(autocomplete_result)
            for line in autocomplete_result:
                result.append(left_padding + line + js_repeat(" ", max(0, content_width - visible_width(line))) + right_padding)
        return result

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult | None:
        autocomplete_start_row = self._rendered_visible_line_count + 2
        if self._autocomplete_state and self._autocomplete_list is not None and autocomplete_start_row <= event.y < autocomplete_start_row + self._rendered_autocomplete_height:
            padding = min(self._padding_x, max(0, math.floor((event.width - 1) / 2)))
            result = self._autocomplete_list.handle_mouse(replace(
                event, x=event.x - padding, y=event.y - autocomplete_start_row,
                width=max(1, event.width - padding * 2), height=self._rendered_autocomplete_height,
            ))
            return replace(result, focus=True) if result is not None else None
        if event.type != "click" or event.button != "left":
            return None
        if event.y <= 0 or event.y > self._rendered_visible_line_count:
            return TuiMouseEventResult(handled=True, focus=True)
        visual_lines = self._build_visual_line_map(self._last_width)
        index = self._scroll_offset + event.y - 1
        if not 0 <= index < len(visual_lines):
            return TuiMouseEventResult(handled=True, focus=True)
        visual_line = visual_lines[index]
        logical_line = self._line(visual_line.logical_line)
        chunk = utf16_slice(logical_line, visual_line.start_col, visual_line.start_col + visual_line.length)
        padding = min(self._padding_x, max(0, math.floor((event.width - 1) / 2)))
        target_column = max(0, event.x - padding)
        visible_column = 0
        target_index = utf16_length(chunk)
        last_grapheme_index = 0
        for grapheme in self._segment(chunk, "grapheme"):
            next_column = visible_column + visible_width(grapheme.segment)
            last_grapheme_index = grapheme.index
            if target_column < next_column:
                target_index = grapheme.index
                break
            visible_column = next_column
        is_last = index == len(visual_lines) - 1 or visual_lines[index + 1].logical_line != visual_line.logical_line
        if not is_last and target_index == utf16_length(chunk) and chunk:
            target_index = last_grapheme_index
        self._state.cursor_line = visual_line.logical_line
        self._set_cursor_col(visual_line.start_col + target_index)
        self._last_action = None
        self._exit_history_browsing()
        if self._autocomplete_state:
            self._update_autocomplete()
        return TuiMouseEventResult(handled=True, focus=True)

    def handle_input(self, data: str) -> None:
        kb = get_keybindings()
        if self._jump_mode is not None:
            if kb.matches(data, "tui.editor.jumpForward") or kb.matches(data, "tui.editor.jumpBackward"):
                self._jump_mode = None
                return
            printable = decode_printable_key(data)
            if printable is None and data and ord(data[0]) >= 32:
                printable = data
            if printable is not None:
                direction = self._jump_mode
                self._jump_mode = None
                self._jump_to_char(printable, direction)
                return
            self._jump_mode = None
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "", 1)
        if self._is_in_paste:
            self._paste_buffer += data
            end = self._paste_buffer.find("\x1b[201~")
            if end != -1:
                pasted = self._paste_buffer[:end]
                if pasted:
                    self._handle_paste(pasted)
                self._is_in_paste = False
                remaining = self._paste_buffer[end + 6:]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
            return
        if kb.matches(data, "tui.input.copy"):
            return
        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return
        if self._autocomplete_state and self._autocomplete_list is not None:
            if kb.matches(data, "tui.select.cancel"):
                self._cancel_autocomplete()
                return
            if kb.matches(data, "tui.select.up") or kb.matches(data, "tui.select.down"):
                self._autocomplete_list.handle_input(data)
                return
            if kb.matches(data, "tui.input.tab"):
                selected = self._autocomplete_list.get_selected_item()
                if selected is not None and self._autocomplete_provider is not None:
                    self._apply_completion(cast(AutocompleteItem, selected), self._autocomplete_prefix)
                    self._cancel_autocomplete()
                    if self.on_change is not None:
                        self.on_change(self.get_text())
                return
            if kb.matches(data, "tui.select.confirm"):
                selected = self._autocomplete_list.get_selected_item()
                if selected is not None and self._autocomplete_provider is not None:
                    self._apply_completion(cast(AutocompleteItem, selected), self._autocomplete_prefix)
                    if self._autocomplete_prefix.startswith("/"):
                        self._cancel_autocomplete()
                    else:
                        self._cancel_autocomplete()
                        if self.on_change is not None:
                            self.on_change(self.get_text())
                        return
        if kb.matches(data, "tui.input.tab") and not self._autocomplete_state:
            self._handle_tab_completion()
            return
        for action, handler in (
            ("tui.editor.deleteToLineEnd", self._delete_to_end_of_line),
            ("tui.editor.deleteToLineStart", self._delete_to_start_of_line),
            ("tui.editor.deleteWordBackward", self._delete_word_backwards),
            ("tui.editor.deleteWordForward", self._delete_word_forward),
        ):
            if kb.matches(data, action):
                handler()
                return
        if kb.matches(data, "tui.editor.deleteCharBackward") or kb.matches(data, "tui.editor.shiftedBackward"):
            self._handle_backspace()
            return
        if kb.matches(data, "tui.editor.deleteCharForward") or kb.matches(data, "tui.editor.shiftedForward"):
            self._handle_forward_delete()
            return
        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            return
        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            return
        if kb.matches(data, "tui.editor.historyPrevious"):
            self._cancel_autocomplete()
            self._navigate_history(-1)
            return
        if kb.matches(data, "tui.editor.historyNext"):
            self._cancel_autocomplete()
            self._navigate_history(1)
            return
        for action, handler in (
            ("tui.editor.cursorLineStart", self._move_to_line_start),
            ("tui.editor.cursorLineEnd", self._move_to_line_end),
            ("tui.editor.cursorWordLeft", self._move_word_backwards),
            ("tui.editor.cursorWordRight", self._move_word_forwards),
        ):
            if kb.matches(data, action):
                handler()
                return
        if (
            kb.matches(data, "tui.input.newLine")
            or (data.startswith("\n") and utf16_length(data) > 1)
            or data in ("\x1b\r", "\x1b[13;2~", "\n")
            or (utf16_length(data) > 1 and "\x1b" in data and "\r" in data)
        ):
            if self._should_submit_on_backslash_enter(data, kb):
                self._handle_backspace()
                self._submit_value()
                return
            self._add_new_line()
            return
        if kb.matches(data, "tui.input.submit"):
            if self.disable_submit:
                return
            if self._state.cursor_col > 0 and utf16_slice(self._line(), self._state.cursor_col - 1, self._state.cursor_col) == "\\":
                self._handle_backspace()
                self._add_new_line()
                return
            self._submit_value()
            return
        if kb.matches(data, "tui.editor.cursorUp"):
            if self._is_on_first_visual_line() and (self._is_editor_empty() or self._history_index > -1 or self._state.cursor_col == 0):
                self._navigate_history(-1)
            elif self._is_on_first_visual_line():
                self._move_to_line_start()
            else:
                self._move_cursor(-1, 0)
            return
        if kb.matches(data, "tui.editor.cursorDown"):
            if self._history_index > -1 and self._is_on_last_visual_line():
                self._navigate_history(1)
            elif self._is_on_last_visual_line():
                self._move_to_line_end()
            else:
                self._move_cursor(1, 0)
            return
        if kb.matches(data, "tui.editor.cursorRight"):
            self._move_cursor(0, 1)
            return
        if kb.matches(data, "tui.editor.cursorLeft"):
            self._move_cursor(0, -1)
            return
        if kb.matches(data, "tui.editor.pageUp"):
            self._page_scroll(-1)
            return
        if kb.matches(data, "tui.editor.pageDown"):
            self._page_scroll(1)
            return
        if kb.matches(data, "tui.editor.jumpForward"):
            self._jump_mode = "forward"
            return
        if kb.matches(data, "tui.editor.jumpBackward"):
            self._jump_mode = "backward"
            return
        if kb.matches(data, "tui.editor.shiftedSpace"):
            self._insert_character(" ")
            return
        printable = decode_printable_key(data)
        if printable is not None:
            self._insert_character(printable)
            return
        if data and ord(data[0]) >= 32:
            self._insert_character(data)

    def _line(self, index: int | None = None) -> str:
        index = self._state.cursor_line if index is None else index
        return self._state.lines[index] if 0 <= index < len(self._state.lines) else ""

    def _layout_text(self, content_width: int) -> list[_LayoutLine]:
        if not self._state.lines or self._state.lines == [""]:
            return [_LayoutLine("", True, 0)]
        result: list[_LayoutLine] = []
        for index, line in enumerate(self._state.lines):
            current = index == self._state.cursor_line
            if visible_width(line) <= content_width:
                result.append(_LayoutLine(line, current, self._state.cursor_col if current else None))
                continue
            chunks = word_wrap_line(line, content_width, list(self._segment(line, "grapheme")))
            for chunk_index, chunk in enumerate(chunks):
                cursor = self._state.cursor_col
                last = chunk_index == len(chunks) - 1
                has_cursor = False
                adjusted = 0
                if current:
                    if last:
                        has_cursor = cursor >= chunk.start_index
                        adjusted = cursor - chunk.start_index
                    else:
                        has_cursor = chunk.start_index <= cursor < chunk.end_index
                        if has_cursor:
                            adjusted = min(cursor - chunk.start_index, utf16_length(chunk.text))
                result.append(_LayoutLine(chunk.text, has_cursor, adjusted if has_cursor else None))
        return result

    def get_text(self) -> str:
        return "\n".join(self._state.lines)

    def _expand_paste_markers(self, text: str) -> str:
        result = text
        for paste_id, content in self._pastes.items():
            marker = re.compile(r"\[paste #" + str(paste_id) + r"( (\+[0-9]+ lines|[0-9]+ chars))?\]")
            result = marker.sub(lambda _match: content, result)
        return result

    def get_expanded_text(self) -> str:
        return self._expand_paste_markers(self.get_text())

    def get_lines(self) -> list[str]:
        return list(self._state.lines)

    def get_cursor(self) -> EditorCursor:
        return {"line": self._state.cursor_line, "col": self._state.cursor_col}

    def set_text(self, text: str) -> None:
        self._cancel_autocomplete()
        self._last_action = None
        self._exit_history_browsing()
        normalized = self._normalize_text(text)
        if self.get_text() != normalized:
            self._push_undo_snapshot()
        self._pastes.clear()
        self._paste_counter = 0
        self._set_text_internal(normalized)

    def insert_text_at_cursor(self, text: str) -> None:
        if not text:
            return
        self._cancel_autocomplete()
        self._push_undo_snapshot()
        self._last_action = None
        self._exit_history_browsing()
        self._insert_text_at_cursor_internal(text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")

    def _insert_text_at_cursor_internal(self, text: str) -> None:
        if not text:
            return
        normalized = self._normalize_text(text)
        inserted = normalized.split("\n")
        current = self._line()
        before = utf16_slice(current, 0, self._state.cursor_col)
        after = utf16_slice(current, self._state.cursor_col)
        if len(inserted) == 1:
            self._state.lines[self._state.cursor_line] = before + normalized + after
            self._set_cursor_col(self._state.cursor_col + utf16_length(normalized))
        else:
            self._state.lines = [
                *self._state.lines[:self._state.cursor_line], before + inserted[0],
                *inserted[1:-1], inserted[-1] + after,
                *self._state.lines[self._state.cursor_line + 1:],
            ]
            self._state.cursor_line += len(inserted) - 1
            self._set_cursor_col(utf16_length(inserted[-1]))
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _insert_character(self, char: str, skip_undo_coalescing: bool = False) -> None:
        self._exit_history_browsing()
        if not skip_undo_coalescing:
            if is_whitespace_char(char) or self._last_action != "type-word":
                self._push_undo_snapshot()
            self._last_action = "type-word"
        line = self._line()
        self._state.lines[self._state.cursor_line] = utf16_slice(line, 0, self._state.cursor_col) + char + utf16_slice(line, self._state.cursor_col)
        self._set_cursor_col(self._state.cursor_col + utf16_length(char))
        if self.on_change is not None:
            self.on_change(self.get_text())
        if not self._autocomplete_state:
            if char == "/" and self._is_at_start_of_message():
                self._try_trigger_autocomplete()
            elif char in self._autocomplete_trigger_characters:
                before = utf16_slice(self._line(), 0, self._state.cursor_col)
                before_symbol = utf16_slice(before, utf16_length(before) - 2, utf16_length(before) - 1) if utf16_length(before) >= 2 else ""
                if utf16_length(before) == 1 or before_symbol in (" ", "\t"):
                    self._try_trigger_autocomplete()
            elif re.search(r"[a-zA-Z0-9.\-_]", char):
                before = utf16_slice(self._line(), 0, self._state.cursor_col)
                if self._is_in_slash_command_context(before) or self._autocomplete_trigger_pattern.search(before):
                    self._try_trigger_autocomplete()
        else:
            self._update_autocomplete()

    def _handle_paste(self, pasted_text: str) -> None:
        self._cancel_autocomplete()
        self._exit_history_browsing()
        self._last_action = None
        self._push_undo_snapshot()

        def decode_control(match: re.Match[str]) -> str:
            codepoint = float(match[1])
            if 97 <= codepoint <= 122:
                return chr(int(codepoint) - 96)
            if 65 <= codepoint <= 90:
                return chr(int(codepoint) - 64)
            return match[0]

        decoded = re.sub(r"\x1b\[([0-9]+);5u", decode_control, pasted_text)
        clean = self._normalize_text(decoded)
        filtered = "".join(char for char in clean if char == "\n" or ord(char) >= 32)
        if filtered.startswith(("/", "~", ".")):
            previous = utf16_slice(self._line(), self._state.cursor_col - 1, self._state.cursor_col) if self._state.cursor_col > 0 else ""
            if previous and re.search(r"[a-zA-Z0-9_]", previous):
                filtered = " " + filtered
        pasted_lines = filtered.split("\n")
        total_chars = utf16_length(filtered)
        if len(pasted_lines) > 10 or total_chars > 1000:
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = filtered
            marker = f"[paste #{paste_id} +{len(pasted_lines)} lines]" if len(pasted_lines) > 10 else f"[paste #{paste_id} {total_chars} chars]"
            self._insert_text_at_cursor_internal(marker)
            return
        self._insert_text_at_cursor_internal(filtered)

    def _add_new_line(self) -> None:
        self._cancel_autocomplete()
        self._exit_history_browsing()
        self._last_action = None
        self._push_undo_snapshot()
        current = self._line()
        before = utf16_slice(current, 0, self._state.cursor_col)
        after = utf16_slice(current, self._state.cursor_col)
        self._state.lines[self._state.cursor_line] = before
        self._state.lines.insert(self._state.cursor_line + 1, after)
        self._state.cursor_line += 1
        self._set_cursor_col(0)
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _should_submit_on_backslash_enter(self, data: str, kb: KeybindingsManager) -> bool:
        if self.disable_submit or not kb.matches(data, "tui.editor.enterFallback"):
            return False
        submit_keys = kb.get_keys("tui.input.submit")
        if "shift+enter" not in submit_keys and "shift+return" not in submit_keys:
            return False
        return self._state.cursor_col > 0 and utf16_slice(self._line(), self._state.cursor_col - 1, self._state.cursor_col) == "\\"

    def _submit_value(self) -> None:
        self._cancel_autocomplete()
        result = js_trim(self._expand_paste_markers(self.get_text()))
        self._state = _EditorState()
        self._pastes.clear()
        self._paste_counter = 0
        self._exit_history_browsing()
        self._scroll_offset = 0
        self._undo_stack.clear()
        self._last_action = None
        if self.on_change is not None:
            self.on_change("")
        if self.on_submit is not None:
            self.on_submit(result)

    def _handle_backspace(self) -> None:
        self._exit_history_browsing()
        self._last_action = None
        if self._state.cursor_col > 0:
            self._push_undo_snapshot()
            graphemes = list(self._segment(utf16_slice(self._line(), 0, self._state.cursor_col), "grapheme"))
            last = graphemes[-1]
            length = utf16_length(last.segment)
            marker = _PASTE_MARKER_REGEX.fullmatch(last.segment)
            if marker:
                target_id = float(marker[1])
                self._pastes.pop(target_id, None)
                self._paste_counter -= 1
                for paste_id in sorted(value for value in self._pastes if value > target_id):
                    self._pastes[paste_id - 1] = self._pastes[paste_id]
                    del self._pastes[paste_id]

                def renumber(match: re.Match[str]) -> str:
                    number = float(match[1])
                    if number <= target_id:
                        return match[0]
                    value = number - 1
                    if math.isinf(value):
                        identifier = "Infinity"
                    elif abs(value) < 1e21:
                        identifier = str(int(value))
                    else:
                        mantissa, exponent = repr(value).split("e")
                        identifier = mantissa.removesuffix(".0") + "e+" + str(int(exponent))
                    suffix = match[2] if match[2] is not None else "undefined"
                    return f"[paste #{identifier}{suffix}]"

                self._state.lines = [_PASTE_MARKER_REGEX.sub(renumber, line) for line in self._state.lines]
            line = self._line()
            self._state.lines[self._state.cursor_line] = utf16_slice(line, 0, self._state.cursor_col - length) + utf16_slice(line, self._state.cursor_col)
            self._set_cursor_col(self._state.cursor_col - length)
        elif self._state.cursor_line > 0:
            self._push_undo_snapshot()
            current = self._line()
            previous = self._line(self._state.cursor_line - 1)
            self._state.lines[self._state.cursor_line - 1] = previous + current
            del self._state.lines[self._state.cursor_line]
            self._state.cursor_line -= 1
            self._set_cursor_col(utf16_length(previous))
        if self.on_change is not None:
            self.on_change(self.get_text())
        self._refresh_autocomplete_after_delete()

    def _set_cursor_col(self, col: int) -> None:
        self._state.cursor_col = col
        self._preferred_visual_col = None
        self._snapped_from_cursor_col = None

    def _move_to_line_start(self) -> None:
        self._last_action = None
        self._set_cursor_col(0)

    def _move_to_line_end(self) -> None:
        self._last_action = None
        self._set_cursor_col(utf16_length(self._line()))

    def _delete_to_start_of_line(self) -> None:
        self._exit_history_browsing()
        current = self._line()
        if self._state.cursor_col > 0:
            self._push_undo_snapshot()
            deleted = utf16_slice(current, 0, self._state.cursor_col)
            self._kill_ring.push(deleted, KillRingPushOptions(prepend=True, accumulate=self._last_action == "kill"))
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = utf16_slice(current, self._state.cursor_col)
            self._set_cursor_col(0)
        elif self._state.cursor_line > 0:
            self._push_undo_snapshot()
            self._kill_ring.push("\n", KillRingPushOptions(prepend=True, accumulate=self._last_action == "kill"))
            self._last_action = "kill"
            previous = self._line(self._state.cursor_line - 1)
            self._state.lines[self._state.cursor_line - 1] = previous + current
            del self._state.lines[self._state.cursor_line]
            self._state.cursor_line -= 1
            self._set_cursor_col(utf16_length(previous))
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _delete_to_end_of_line(self) -> None:
        self._exit_history_browsing()
        current = self._line()
        if self._state.cursor_col < utf16_length(current):
            self._push_undo_snapshot()
            deleted = utf16_slice(current, self._state.cursor_col)
            self._kill_ring.push(deleted, KillRingPushOptions(prepend=False, accumulate=self._last_action == "kill"))
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = utf16_slice(current, 0, self._state.cursor_col)
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo_snapshot()
            self._kill_ring.push("\n", KillRingPushOptions(prepend=False, accumulate=self._last_action == "kill"))
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = current + self._line(self._state.cursor_line + 1)
            del self._state.lines[self._state.cursor_line + 1]
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _delete_word_backwards(self) -> None:
        self._exit_history_browsing()
        current = self._line()
        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._push_undo_snapshot()
                self._kill_ring.push("\n", KillRingPushOptions(prepend=True, accumulate=self._last_action == "kill"))
                self._last_action = "kill"
                previous = self._line(self._state.cursor_line - 1)
                self._state.lines[self._state.cursor_line - 1] = previous + current
                del self._state.lines[self._state.cursor_line]
                self._state.cursor_line -= 1
                self._set_cursor_col(utf16_length(previous))
        else:
            self._push_undo_snapshot()
            was_kill = self._last_action == "kill"
            old_cursor = self._state.cursor_col
            self._move_word_backwards()
            delete_from = self._state.cursor_col
            self._set_cursor_col(old_cursor)
            self._kill_ring.push(utf16_slice(current, delete_from, old_cursor), KillRingPushOptions(prepend=True, accumulate=was_kill))
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = utf16_slice(current, 0, delete_from) + utf16_slice(current, old_cursor)
            self._set_cursor_col(delete_from)
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _delete_word_forward(self) -> None:
        self._exit_history_browsing()
        current = self._line()
        if self._state.cursor_col >= utf16_length(current):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._push_undo_snapshot()
                self._kill_ring.push("\n", KillRingPushOptions(prepend=False, accumulate=self._last_action == "kill"))
                self._last_action = "kill"
                self._state.lines[self._state.cursor_line] = current + self._line(self._state.cursor_line + 1)
                del self._state.lines[self._state.cursor_line + 1]
        else:
            self._push_undo_snapshot()
            was_kill = self._last_action == "kill"
            old_cursor = self._state.cursor_col
            self._move_word_forwards()
            delete_to = self._state.cursor_col
            self._set_cursor_col(old_cursor)
            self._kill_ring.push(utf16_slice(current, old_cursor, delete_to), KillRingPushOptions(prepend=False, accumulate=was_kill))
            self._last_action = "kill"
            self._state.lines[self._state.cursor_line] = utf16_slice(current, 0, old_cursor) + utf16_slice(current, delete_to)
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _handle_forward_delete(self) -> None:
        self._exit_history_browsing()
        self._last_action = None
        current = self._line()
        if self._state.cursor_col < utf16_length(current):
            self._push_undo_snapshot()
            graphemes = list(self._segment(utf16_slice(current, self._state.cursor_col), "grapheme"))
            length = utf16_length(graphemes[0].segment) if graphemes else 1
            self._state.lines[self._state.cursor_line] = utf16_slice(current, 0, self._state.cursor_col) + utf16_slice(current, self._state.cursor_col + length)
        elif self._state.cursor_line < len(self._state.lines) - 1:
            self._push_undo_snapshot()
            self._state.lines[self._state.cursor_line] = current + self._line(self._state.cursor_line + 1)
            del self._state.lines[self._state.cursor_line + 1]
        if self.on_change is not None:
            self.on_change(self.get_text())
        self._refresh_autocomplete_after_delete()

    def _refresh_autocomplete_after_delete(self) -> None:
        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            before = utf16_slice(self._line(), 0, self._state.cursor_col)
            if self._is_in_slash_command_context(before) or self._autocomplete_trigger_pattern.search(before):
                self._try_trigger_autocomplete()

    def _build_visual_line_map(self, width: int) -> list[_VisualLine]:
        result: list[_VisualLine] = []
        for index, line in enumerate(self._state.lines):
            if not line:
                result.append(_VisualLine(index, 0, 0))
            elif visible_width(line) <= width:
                result.append(_VisualLine(index, 0, utf16_length(line)))
            else:
                for chunk in word_wrap_line(line, width, list(self._segment(line, "grapheme"))):
                    result.append(_VisualLine(index, chunk.start_index, chunk.end_index - chunk.start_index))
        return result

    @staticmethod
    def _find_visual_line_at(visual_lines: list[_VisualLine], line: int, col: int) -> int:
        for index, visual in enumerate(visual_lines):
            if visual.logical_line != line:
                continue
            offset = col - visual.start_col
            last = index == len(visual_lines) - 1 or visual_lines[index + 1].logical_line != visual.logical_line
            if offset >= 0 and (offset < visual.length or last and offset == visual.length):
                return index
        return len(visual_lines) - 1

    def _find_current_visual_line(self, visual_lines: list[_VisualLine]) -> int:
        return self._find_visual_line_at(visual_lines, self._state.cursor_line, self._state.cursor_col)

    def _move_to_visual_line(self, visual_lines: list[_VisualLine], current_index: int, target_index: int) -> None:
        if not 0 <= current_index < len(visual_lines) or not 0 <= target_index < len(visual_lines):
            return
        current = visual_lines[current_index]
        target = visual_lines[target_index]
        if self._snapped_from_cursor_col is not None:
            index = self._find_visual_line_at(visual_lines, current.logical_line, self._snapped_from_cursor_col)
            current_col = self._snapped_from_cursor_col - visual_lines[index].start_col
        else:
            current_col = self._state.cursor_col - current.start_col
        last_source = current_index == len(visual_lines) - 1 or visual_lines[current_index + 1].logical_line != current.logical_line
        source_max = current.length if last_source else max(0, current.length - 1)
        last_target = target_index == len(visual_lines) - 1 or visual_lines[target_index + 1].logical_line != target.logical_line
        target_max = target.length if last_target else max(0, target.length - 1)
        move_to = self._compute_vertical_move_column(current_col, source_max, target_max)
        self._state.cursor_line = target.logical_line
        logical_line = self._line(target.logical_line)
        self._state.cursor_col = min(target.start_col + move_to, utf16_length(logical_line))
        for segment in self._segment(logical_line, "grapheme"):
            if segment.index > self._state.cursor_col:
                break
            length = utf16_length(segment.segment)
            if length <= 1:
                continue
            if self._state.cursor_col < segment.index + length:
                continuation = segment.index < target.start_col
                moving_down = target_index > current_index
                if continuation and moving_down:
                    end = segment.index + length
                    next_index = target_index + 1
                    while next_index < len(visual_lines) and visual_lines[next_index].logical_line == target.logical_line and visual_lines[next_index].start_col < end:
                        next_index += 1
                    if next_index < len(visual_lines):
                        self._move_to_visual_line(visual_lines, current_index, next_index)
                        return
                self._snapped_from_cursor_col = self._state.cursor_col
                self._state.cursor_col = segment.index
                return
        self._snapped_from_cursor_col = None

    def _compute_vertical_move_column(self, current_col: int, source_max: int, target_max: int) -> int:
        has_preferred = self._preferred_visual_col is not None
        cursor_in_middle = current_col < source_max
        target_too_short = target_max < current_col
        if not has_preferred or cursor_in_middle:
            if target_too_short:
                self._preferred_visual_col = current_col
                return target_max
            self._preferred_visual_col = None
            return current_col
        assert self._preferred_visual_col is not None
        if target_too_short or target_max < self._preferred_visual_col:
            return target_max
        result = self._preferred_visual_col
        self._preferred_visual_col = None
        return result

    def _move_cursor(self, delta_line: int, delta_col: int) -> None:
        self._last_action = None
        visual_lines = self._build_visual_line_map(self._last_width)
        current_index = self._find_current_visual_line(visual_lines)
        if delta_line != 0:
            target_index = current_index + delta_line
            if 0 <= target_index < len(visual_lines):
                self._move_to_visual_line(visual_lines, current_index, target_index)
        if delta_col != 0:
            current_line = self._line()
            if delta_col > 0:
                if self._state.cursor_col < utf16_length(current_line):
                    graphemes = list(self._segment(utf16_slice(current_line, self._state.cursor_col), "grapheme"))
                    self._set_cursor_col(self._state.cursor_col + (utf16_length(graphemes[0].segment) if graphemes else 1))
                elif self._state.cursor_line < len(self._state.lines) - 1:
                    self._state.cursor_line += 1
                    self._set_cursor_col(0)
                elif 0 <= current_index < len(visual_lines):
                    self._preferred_visual_col = self._state.cursor_col - visual_lines[current_index].start_col
            elif self._state.cursor_col > 0:
                graphemes = list(self._segment(utf16_slice(current_line, 0, self._state.cursor_col), "grapheme"))
                self._set_cursor_col(self._state.cursor_col - (utf16_length(graphemes[-1].segment) if graphemes else 1))
            elif self._state.cursor_line > 0:
                self._state.cursor_line -= 1
                self._set_cursor_col(utf16_length(self._line()))
        if self._autocomplete_state:
            self._update_autocomplete()

    def _page_scroll(self, direction: Literal[-1, 1]) -> None:
        self._last_action = None
        visible_rows = self.tui.terminal.rows * 0.3
        visual_lines = self._build_visual_line_map(self._last_width)
        if visible_rows == math.inf:
            page_size = len(visual_lines)
        elif visible_rows == -math.inf:
            page_size = 5
        else:
            page_size = max(5, math.floor(visible_rows))
        current_index = self._find_current_visual_line(visual_lines)
        target_index = max(0, min(len(visual_lines) - 1, current_index + direction * page_size))
        self._move_to_visual_line(visual_lines, current_index, target_index)

    def _move_word_backwards(self) -> None:
        self._last_action = None
        current = self._line()
        if self._state.cursor_col == 0:
            if self._state.cursor_line > 0:
                self._state.cursor_line -= 1
                self._set_cursor_col(utf16_length(self._line()))
            return
        self._set_cursor_col(find_word_backward(current, self._state.cursor_col, WordNavigationOptions(
            segment=lambda text: self._segment(text, "word"), is_atomic_segment=_is_paste_marker,
        )))

    def _move_word_forwards(self) -> None:
        self._last_action = None
        current = self._line()
        if self._state.cursor_col >= utf16_length(current):
            if self._state.cursor_line < len(self._state.lines) - 1:
                self._state.cursor_line += 1
                self._set_cursor_col(0)
            return
        self._set_cursor_col(find_word_forward(current, self._state.cursor_col, WordNavigationOptions(
            segment=lambda text: self._segment(text, "word"), is_atomic_segment=_is_paste_marker,
        )))

    def _yank(self) -> None:
        if self._kill_ring.length == 0:
            return
        self._push_undo_snapshot()
        text = self._kill_ring.peek()
        assert text is not None
        self._insert_yanked_text(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return
        self._push_undo_snapshot()
        self._delete_yanked_text()
        self._kill_ring.rotate()
        text = self._kill_ring.peek()
        assert text is not None
        self._insert_yanked_text(text)
        self._last_action = "yank"

    def _insert_yanked_text(self, text: str) -> None:
        self._exit_history_browsing()
        lines = text.split("\n")
        current = self._line()
        before = utf16_slice(current, 0, self._state.cursor_col)
        after = utf16_slice(current, self._state.cursor_col)
        if len(lines) == 1:
            self._state.lines[self._state.cursor_line] = before + text + after
            self._set_cursor_col(self._state.cursor_col + utf16_length(text))
        else:
            self._state.lines[self._state.cursor_line] = before + lines[0]
            for index in range(1, len(lines) - 1):
                self._state.lines.insert(self._state.cursor_line + index, lines[index])
            last_index = self._state.cursor_line + len(lines) - 1
            self._state.lines.insert(last_index, lines[-1] + after)
            self._state.cursor_line = last_index
            self._set_cursor_col(utf16_length(lines[-1]))
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _delete_yanked_text(self) -> None:
        text = self._kill_ring.peek()
        if not text:
            return
        lines = text.split("\n")
        if len(lines) == 1:
            current = self._line()
            length = utf16_length(text)
            self._state.lines[self._state.cursor_line] = utf16_slice(current, 0, self._state.cursor_col - length) + utf16_slice(current, self._state.cursor_col)
            self._set_cursor_col(self._state.cursor_col - length)
        else:
            start_line = self._state.cursor_line - (len(lines) - 1)
            start_col = utf16_length(self._line(start_line)) - utf16_length(lines[0])
            after = utf16_slice(self._line(), self._state.cursor_col)
            before = utf16_slice(self._line(start_line), 0, start_col)
            self._state.lines[start_line:start_line + len(lines)] = [before + after]
            self._state.cursor_line = start_line
            self._set_cursor_col(start_col)
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.push(_EditorSnapshot(self._state, self._pastes, self._paste_counter))

    def _undo(self) -> None:
        self._exit_history_browsing()
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._state.lines = snapshot.state.lines
        self._state.cursor_line = snapshot.state.cursor_line
        self._state.cursor_col = snapshot.state.cursor_col
        self._pastes = snapshot.pastes
        self._paste_counter = snapshot.paste_counter
        self._last_action = None
        self._preferred_visual_col = None
        if self.on_change is not None:
            self.on_change(self.get_text())

    def _jump_to_char(self, char: str, direction: Literal["forward", "backward"]) -> None:
        self._last_action = None
        forward = direction == "forward"
        end = len(self._state.lines) if forward else -1
        step = 1 if forward else -1
        needle = utf16_units(char)
        for index in range(self._state.cursor_line, end, step):
            line = utf16_units(self._line(index))
            current = index == self._state.cursor_line
            if forward:
                found = line.find(needle, self._state.cursor_col + 1 if current else 0)
            else:
                start = max(0, self._state.cursor_col - 1) if current else len(line)
                found = line.rfind(needle, 0, start + len(needle))
            if found != -1:
                self._state.cursor_line = index
                self._set_cursor_col(found)
                return

    def _is_slash_menu_allowed(self) -> bool:
        return self._state.cursor_line == 0

    def _is_at_start_of_message(self) -> bool:
        if not self._is_slash_menu_allowed():
            return False
        before = utf16_slice(self._line(), 0, self._state.cursor_col)
        return js_trim(before) in ("", "/")

    def _is_in_slash_command_context(self, before: str) -> bool:
        return self._is_slash_menu_allowed() and before.lstrip(JS_WHITESPACE).startswith("/")

    @staticmethod
    def _get_best_autocomplete_match_index(items: list[AutocompleteItem], prefix: str) -> int:
        if not prefix:
            return -1
        first_prefix_index = -1
        for index, item in enumerate(items):
            if item.value == prefix:
                return index
            if first_prefix_index == -1 and item.value.startswith(prefix):
                first_prefix_index = index
        return first_prefix_index

    def _apply_completion(self, item: AutocompleteItem, prefix: str) -> None:
        assert self._autocomplete_provider is not None
        self._push_undo_snapshot()
        self._last_action = None
        result = self._autocomplete_provider.apply_completion(
            self._state.lines, self._state.cursor_line, self._state.cursor_col, item, prefix,
        )
        self._state.lines = result.lines
        self._state.cursor_line = result.cursor_line
        self._set_cursor_col(result.cursor_col)

    def _create_autocomplete_list(self, prefix: str, items: list[AutocompleteItem]) -> SelectList:
        layout = _SLASH_COMMAND_SELECT_LIST_LAYOUT if prefix.startswith("/") else None
        # Both source contracts are structural; retain the provider's original
        # item objects so completion callbacks receive the same identities.
        select_list = SelectList(cast(list[SelectItem], items), self._autocomplete_max_visible, self._theme.select_list, layout)

        def selected(item: SelectItem) -> None:
            if self._autocomplete_provider is None:
                return
            self._apply_completion(cast(AutocompleteItem, item), self._autocomplete_prefix)
            self._cancel_autocomplete()
            if self.on_change is not None:
                self.on_change(self.get_text())

        select_list.on_select = selected
        return select_list

    def _try_trigger_autocomplete(self, explicit_tab: bool = False) -> None:
        self._request_autocomplete(_AutocompleteRequestOptions(False, explicit_tab))

    def _handle_tab_completion(self) -> None:
        if self._autocomplete_provider is None:
            return
        before = utf16_slice(self._line(), 0, self._state.cursor_col)
        if self._is_in_slash_command_context(before) and " " not in before.lstrip(JS_WHITESPACE):
            self._request_autocomplete(_AutocompleteRequestOptions(False, True))
        else:
            self._request_autocomplete(_AutocompleteRequestOptions(True, True))

    def _request_autocomplete(self, options: _AutocompleteRequestOptions) -> None:
        if self._autocomplete_provider is None:
            return
        if options.force:
            trigger = getattr(self._autocomplete_provider, "should_trigger_file_completion", None)
            if trigger is not None and not trigger(self._state.lines, self._state.cursor_line, self._state.cursor_col):
                return
        self._cancel_autocomplete_request()
        self._autocomplete_start_token += 1
        start_token = self._autocomplete_start_token
        debounce = self._get_autocomplete_debounce_ms(options)
        if debounce > 0:
            def expired() -> None:
                self._autocomplete_debounce_timer = None
                self._start_autocomplete_request(start_token, options)

            self._autocomplete_debounce_timer = set_timeout(expired, debounce)
            return
        self._start_autocomplete_request(start_token, options)

    def _start_autocomplete_request(self, start_token: int, options: _AutocompleteRequestOptions) -> None:
        with self._autocomplete_start_lock:
            previous_task = self._autocomplete_request_task
            if self._autocomplete_loop is None:
                self._autocomplete_loop = _completion_loop()
            loop = self._autocomplete_loop

            async def run() -> None:
                if previous_task is None:
                    await asyncio.sleep(0)
                else:
                    already_done = previous_task.done()
                    if isinstance(previous_task, concurrent.futures.Future):
                        await asyncio.wrap_future(previous_task)
                    else:
                        await previous_task
                    if already_done:
                        await asyncio.sleep(0)
                if start_token != self._autocomplete_start_token or self._autocomplete_provider is None:
                    return
                controller = AbortController()
                self._autocomplete_abort = controller
                self._autocomplete_request_id += 1
                await self._run_autocomplete_request(
                    self._autocomplete_request_id, controller, self.get_text(),
                    self._state.cursor_line, self._state.cursor_col, options,
                )

            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is loop:
                self._autocomplete_request_task = loop.create_task(run())
            else:
                self._autocomplete_request_task = asyncio.run_coroutine_threadsafe(run(), loop)

            def report_failure(task: asyncio.Future[None] | concurrent.futures.Future[None]) -> None:
                if task.cancelled():
                    return
                error = task.exception()
                if error is not None:
                    loop.call_soon_threadsafe(loop.call_exception_handler, {
                        "message": "Unhandled editor autocomplete exception",
                        "exception": error,
                        "future": task,
                    })

            self._autocomplete_request_task.add_done_callback(report_failure)

    def _set_autocomplete_trigger_characters(self, characters: list[str]) -> None:
        next_characters = list(_DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS)
        for character in characters:
            if utf16_length(character) != 1 or character == "/" or is_whitespace_char(character) or character in next_characters:
                continue
            next_characters.append(character)
        self._autocomplete_trigger_characters = next_characters
        self._autocomplete_trigger_pattern = _build_trigger_pattern(next_characters)
        self._autocomplete_debounce_pattern = _build_debounce_pattern(next_characters)

    def _get_autocomplete_debounce_ms(self, options: _AutocompleteRequestOptions) -> int:
        if options.explicit_tab or options.force:
            return 0
        before = utf16_slice(self._line(), 0, self._state.cursor_col)
        return _ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS if self._autocomplete_debounce_pattern.search(before) else 0

    async def _run_autocomplete_request(
        self, request_id: int, controller: AbortController, snapshot_text: str,
        snapshot_line: int, snapshot_col: int, options: _AutocompleteRequestOptions,
    ) -> None:
        if self._autocomplete_provider is None:
            return
        pending = self._autocomplete_provider.get_suggestions(
            self._state.lines, self._state.cursor_line, self._state.cursor_col,
            AutocompleteOptions(signal=controller.signal, force=options.force),
        )
        suggestions = await pending if inspect.isawaitable(pending) else pending
        if not self._is_autocomplete_request_current(request_id, controller, snapshot_text, snapshot_line, snapshot_col):
            return
        self._autocomplete_abort = None
        if suggestions is None or not isinstance(suggestions.items, list) or not suggestions.items:
            self._cancel_autocomplete()
            self.tui.request_render()
            return
        if options.force and options.explicit_tab and len(suggestions.items) == 1:
            self._apply_completion(suggestions.items[0], suggestions.prefix)
            if self.on_change is not None:
                self.on_change(self.get_text())
            self.tui.request_render()
            return
        self._apply_autocomplete_suggestions(suggestions, "force" if options.force else "regular")
        self.tui.request_render()

    def _is_autocomplete_request_current(
        self, request_id: int, controller: AbortController, snapshot_text: str,
        snapshot_line: int, snapshot_col: int,
    ) -> bool:
        return (
            not controller.signal.aborted and request_id == self._autocomplete_request_id
            and self.get_text() == snapshot_text and self._state.cursor_line == snapshot_line
            and self._state.cursor_col == snapshot_col
        )

    def _apply_autocomplete_suggestions(self, suggestions: AutocompleteSuggestions, state: Literal["regular", "force"]) -> None:
        self._autocomplete_prefix = suggestions.prefix
        self._autocomplete_list = self._create_autocomplete_list(suggestions.prefix, suggestions.items)
        best = self._get_best_autocomplete_match_index(suggestions.items, suggestions.prefix)
        if best >= 0:
            self._autocomplete_list.set_selected_index(best)
        self._autocomplete_state = state

    def _cancel_autocomplete_request(self) -> None:
        self._autocomplete_start_token += 1
        if self._autocomplete_debounce_timer is not None:
            self._autocomplete_debounce_timer.cancel()
            self._autocomplete_debounce_timer = None
        if self._autocomplete_abort is not None:
            self._autocomplete_abort.abort()
            self._autocomplete_abort = None

    def _clear_autocomplete_ui(self) -> None:
        self._autocomplete_state = None
        self._autocomplete_list = None
        self._autocomplete_prefix = ""

    def _cancel_autocomplete(self) -> None:
        self._cancel_autocomplete_request()
        self._clear_autocomplete_ui()

    def is_showing_autocomplete(self) -> bool:
        return self._autocomplete_state is not None

    def _update_autocomplete(self) -> None:
        if self._autocomplete_state is None or self._autocomplete_provider is None:
            return
        self._request_autocomplete(_AutocompleteRequestOptions(self._autocomplete_state == "force", False))


__all__ = ["Editor", "EditorOptions", "EditorTheme", "EditorCursor", "TextChunk", "word_wrap_line"]
