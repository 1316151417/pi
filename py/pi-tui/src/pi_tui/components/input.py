"""Single-line editor with horizontal scrolling, paste, kill ring, and undo."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .._component import CURSOR_MARKER, TuiMouseEvent, TuiMouseEventResult
from .._javascript import js_repeat, utf16_length, utf16_slice
from ..keybindings import get_keybindings
from ..keys import decode_kitty_printable
from ..kill_ring import KillRing, KillRingPushOptions
from ..undo_stack import UndoStack
from ..utils import get_grapheme_segmenter, is_whitespace_char, slice_by_column, truncate_to_width, visible_width
from ..word_navigation import find_word_backward, find_word_forward

_segmenter = get_grapheme_segmenter()


@dataclass
class InputState:
    value: str
    cursor: int


@dataclass
class InputOptions:
    prompt: str | None = None
    placeholder: str | None = None
    placeholder_style: Callable[[str], str] | None = None


class Input:
    def __init__(self, options: InputOptions | None = None) -> None:
        options = options or InputOptions()
        self._value = ""
        self._cursor = 0
        self._prompt = "> " if options.prompt is None else options.prompt
        self._placeholder = "" if options.placeholder is None else options.placeholder
        self._placeholder_style = options.placeholder_style or (lambda text: text)
        self._rendered_start_column = 0
        self.on_submit: Callable[[str], None] | None = None
        self.on_escape: Callable[[], None] | None = None
        self.focused = False
        self._paste_buffer = ""
        self._is_in_paste = False
        self._kill_ring = KillRing()
        self._last_action: Literal["kill", "yank", "type-word"] | None = None
        self._undo_stack = UndoStack[InputState]()

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value
        self._cursor = min(self._cursor, utf16_length(value))

    def handle_input(self, data: str) -> None:
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "", 1)
        if self._is_in_paste:
            self._paste_buffer += data
            end_index = self._paste_buffer.find("\x1b[201~")
            if end_index != -1:
                self._handle_paste(self._paste_buffer[:end_index])
                self._is_in_paste = False
                remaining = self._paste_buffer[end_index + 6:]
                self._paste_buffer = ""
                if remaining:
                    self.handle_input(remaining)
            return
        kb = get_keybindings()
        if kb.matches(data, "tui.select.cancel"):
            if self.on_escape is not None:
                self.on_escape()
            return
        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return
        if kb.matches(data, "tui.input.submit") or data == "\n":
            if self.on_submit is not None:
                self.on_submit(self._value)
            return
        if kb.matches(data, "tui.editor.deleteCharBackward"):
            self._handle_backspace()
            return
        if kb.matches(data, "tui.editor.deleteCharForward"):
            self._handle_forward_delete()
            return
        if kb.matches(data, "tui.editor.deleteWordBackward"):
            self._delete_word_backwards()
            return
        if kb.matches(data, "tui.editor.deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "tui.editor.deleteToLineStart"):
            self._delete_to_line_start()
            return
        if kb.matches(data, "tui.editor.deleteToLineEnd"):
            self._delete_to_line_end()
            return
        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            return
        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            return
        if kb.matches(data, "tui.editor.cursorLeft"):
            self._last_action = None
            if self._cursor > 0:
                graphemes = list(_segmenter.segment(utf16_slice(self._value, 0, self._cursor)))
                self._cursor -= utf16_length(graphemes[-1].segment) if graphemes else 1
            return
        if kb.matches(data, "tui.editor.cursorRight"):
            self._last_action = None
            if self._cursor < utf16_length(self._value):
                graphemes = list(_segmenter.segment(utf16_slice(self._value, self._cursor)))
                self._cursor += utf16_length(graphemes[0].segment) if graphemes else 1
            return
        if kb.matches(data, "tui.editor.cursorLineStart"):
            self._last_action = None
            self._cursor = 0
            return
        if kb.matches(data, "tui.editor.cursorLineEnd"):
            self._last_action = None
            self._cursor = utf16_length(self._value)
            return
        if kb.matches(data, "tui.editor.cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "tui.editor.cursorWordRight"):
            self._move_word_forwards()
            return
        kitty_printable = decode_kitty_printable(data)
        if kitty_printable is not None:
            self._insert_character(kitty_printable)
            return
        if not any(ord(char) < 32 or ord(char) == 0x7F or 0x80 <= ord(char) <= 0x9F for char in data):
            self._insert_character(data)

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult | None:
        if event.type != "press" or event.button != "left" or event.y != 0:
            return None
        visible_column = max(0, event.x - 2)
        target_column = self._rendered_start_column + visible_column
        current_column = 0
        self._cursor = utf16_length(self._value)
        for grapheme in _segmenter.segment(self._value):
            next_column = current_column + visible_width(grapheme.segment)
            if target_column < next_column:
                self._cursor = grapheme.index
                break
            current_column = next_column
        self._last_action = None
        return TuiMouseEventResult(handled=True, focus=True)

    def _insert_character(self, char: str) -> None:
        if is_whitespace_char(char) or self._last_action != "type-word":
            self._push_undo()
        self._last_action = "type-word"
        self._value = utf16_slice(self._value, 0, self._cursor) + char + utf16_slice(self._value, self._cursor)
        self._cursor += utf16_length(char)

    def _handle_backspace(self) -> None:
        self._last_action = None
        if self._cursor > 0:
            self._push_undo()
            graphemes = list(_segmenter.segment(utf16_slice(self._value, 0, self._cursor)))
            length = utf16_length(graphemes[-1].segment) if graphemes else 1
            self._value = utf16_slice(self._value, 0, self._cursor - length) + utf16_slice(self._value, self._cursor)
            self._cursor -= length

    def _handle_forward_delete(self) -> None:
        self._last_action = None
        if self._cursor < utf16_length(self._value):
            self._push_undo()
            graphemes = list(_segmenter.segment(utf16_slice(self._value, self._cursor)))
            length = utf16_length(graphemes[0].segment) if graphemes else 1
            self._value = utf16_slice(self._value, 0, self._cursor) + utf16_slice(self._value, self._cursor + length)

    def _delete_to_line_start(self) -> None:
        if self._cursor == 0:
            return
        self._push_undo()
        deleted = utf16_slice(self._value, 0, self._cursor)
        self._kill_ring.push(deleted, KillRingPushOptions(prepend=True, accumulate=self._last_action == "kill"))
        self._last_action = "kill"
        self._value = utf16_slice(self._value, self._cursor)
        self._cursor = 0

    def _delete_to_line_end(self) -> None:
        if self._cursor >= utf16_length(self._value):
            return
        self._push_undo()
        deleted = utf16_slice(self._value, self._cursor)
        self._kill_ring.push(deleted, KillRingPushOptions(prepend=False, accumulate=self._last_action == "kill"))
        self._last_action = "kill"
        self._value = utf16_slice(self._value, 0, self._cursor)

    def _delete_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_backwards()
        delete_from = self._cursor
        self._cursor = old_cursor
        deleted = utf16_slice(self._value, delete_from, self._cursor)
        self._kill_ring.push(deleted, KillRingPushOptions(prepend=True, accumulate=was_kill))
        self._last_action = "kill"
        self._value = utf16_slice(self._value, 0, delete_from) + utf16_slice(self._value, self._cursor)
        self._cursor = delete_from

    def _delete_word_forward(self) -> None:
        if self._cursor >= utf16_length(self._value):
            return
        was_kill = self._last_action == "kill"
        self._push_undo()
        old_cursor = self._cursor
        self._move_word_forwards()
        delete_to = self._cursor
        self._cursor = old_cursor
        deleted = utf16_slice(self._value, self._cursor, delete_to)
        self._kill_ring.push(deleted, KillRingPushOptions(prepend=False, accumulate=was_kill))
        self._last_action = "kill"
        self._value = utf16_slice(self._value, 0, self._cursor) + utf16_slice(self._value, delete_to)

    def _yank(self) -> None:
        text = self._kill_ring.peek()
        if not text:
            return
        self._push_undo()
        self._value = utf16_slice(self._value, 0, self._cursor) + text + utf16_slice(self._value, self._cursor)
        self._cursor += utf16_length(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return
        self._push_undo()
        previous_text = self._kill_ring.peek() or ""
        self._value = utf16_slice(self._value, 0, self._cursor - utf16_length(previous_text)) + utf16_slice(self._value, self._cursor)
        self._cursor -= utf16_length(previous_text)
        self._kill_ring.rotate()
        text = self._kill_ring.peek() or ""
        self._value = utf16_slice(self._value, 0, self._cursor) + text + utf16_slice(self._value, self._cursor)
        self._cursor += utf16_length(text)
        self._last_action = "yank"

    def _push_undo(self) -> None:
        self._undo_stack.push(InputState(self._value, self._cursor))

    def _undo(self) -> None:
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._value = snapshot.value
        self._cursor = snapshot.cursor
        self._last_action = None

    def _move_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        self._last_action = None
        self._cursor = find_word_backward(self._value, self._cursor)

    def _move_word_forwards(self) -> None:
        if self._cursor >= utf16_length(self._value):
            return
        self._last_action = None
        self._cursor = find_word_forward(self._value, self._cursor)

    def _handle_paste(self, pasted_text: str) -> None:
        self._last_action = None
        self._push_undo()
        clean = pasted_text.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("\t", "    ")
        self._value = utf16_slice(self._value, 0, self._cursor) + clean + utf16_slice(self._value, self._cursor)
        self._cursor += utf16_length(clean)

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        available_width = width - visible_width(self._prompt)
        if available_width <= 0:
            return [truncate_to_width(self._prompt, width, "")]
        if not self._value and self._placeholder:
            placeholder = truncate_to_width(self._placeholder, available_width, "")
            graphemes = list(_segmenter.segment(placeholder))
            at_cursor = graphemes[0].segment if graphemes else " "
            after_cursor = utf16_slice(placeholder, utf16_length(at_cursor))
            marker = CURSOR_MARKER if self.focused else ""
            cursor_char = f"\x1b[7m{self._placeholder_style(at_cursor)}\x1b[27m"
            with_cursor = marker + cursor_char + self._placeholder_style(after_cursor)
            return [self._prompt + with_cursor + js_repeat(" ", max(0, available_width - visible_width(with_cursor)))]
        visible_text = ""
        cursor_display = self._cursor
        self._rendered_start_column = 0
        total_width = visible_width(self._value)
        if total_width < available_width:
            visible_text = self._value
        else:
            scroll_width = available_width - 1 if self._cursor == utf16_length(self._value) else available_width
            cursor_col = visible_width(utf16_slice(self._value, 0, self._cursor))
            if scroll_width > 0:
                half_width = math.floor(scroll_width / 2)
                if cursor_col < half_width:
                    start_col = 0
                elif cursor_col > total_width - half_width:
                    start_col = max(0, total_width - scroll_width)
                else:
                    start_col = max(0, cursor_col - half_width)
                self._rendered_start_column = start_col
                visible_text = slice_by_column(self._value, start_col, scroll_width, True)
                before_cursor = slice_by_column(self._value, start_col, max(0, cursor_col - start_col), True)
                cursor_display = utf16_length(before_cursor)
            else:
                cursor_display = 0
        graphemes = list(_segmenter.segment(utf16_slice(visible_text, cursor_display)))
        before_cursor = utf16_slice(visible_text, 0, cursor_display)
        at_cursor = graphemes[0].segment if graphemes else " "
        after_cursor = utf16_slice(visible_text, cursor_display + utf16_length(at_cursor))
        marker = CURSOR_MARKER if self.focused else ""
        with_cursor = before_cursor + marker + f"\x1b[7m{at_cursor}\x1b[27m" + after_cursor
        return [self._prompt + with_cursor + js_repeat(" ", max(0, available_width - visible_width(with_cursor)))]
