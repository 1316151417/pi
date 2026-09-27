"""Main-screen differential rendering and scrollback from ``tui-main-screen.ts``."""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ._javascript import JS_WHITESPACE
from ._terminal_types import Terminal
from .terminal_image import delete_kitty_image, is_image_line
from .tui import AppKeybindings, CursorPosition, TuiBase, TuiStopOptions
from .utils import visible_width

_KITTY_SEQUENCE_PREFIX = "\x1b_G"
_MAX_RENDER_WRITE_CHARS = 1024 * 1024


class _BoundedTerminalWriter:
    def __init__(self, write: Callable[[str], None]) -> None:
        self._buffer = bytearray()
        self._written_chars = 0
        self._write = write

    def append(self, value: str) -> None:
        encoded = value.encode("utf-16-le", errors="surrogatepass")
        offset = 0
        while offset < len(encoded):
            capacity = _MAX_RENDER_WRITE_CHARS * 2 - len(self._buffer)
            if capacity == 0:
                self.flush()
                continue
            end = min(len(encoded), offset + capacity)
            if end < len(encoded):
                previous = encoded[end - 2] | encoded[end - 1] << 8
                following = encoded[end] | encoded[end + 1] << 8
                if 0xD800 <= previous <= 0xDBFF and 0xDC00 <= following <= 0xDFFF:
                    end -= 2
            if end == offset:
                self.flush()
                continue
            self._buffer.extend(encoded[offset:end])
            offset = end
            if len(self._buffer) == _MAX_RENDER_WRITE_CHARS * 2:
                self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self._write(self._buffer.decode("utf-16-le", errors="surrogatepass"))
        self._written_chars += len(self._buffer) // 2
        self._buffer.clear()

    @property
    def length(self) -> int:
        return self._written_chars + len(self._buffer) // 2


@dataclass
class _KittyImageHeader:
    ids: list[int]
    rows: int


def _parse_kitty_image_header(line: str) -> _KittyImageHeader | None:
    sequence_start = line.find(_KITTY_SEQUENCE_PREFIX)
    if sequence_start == -1:
        return None
    parameters_start = sequence_start + len(_KITTY_SEQUENCE_PREFIX)
    parameters_end = line.find(";", parameters_start)
    if parameters_end == -1:
        return None
    ids: list[int] = []
    rows = 1
    for parameter in line[parameters_start:parameters_end].split(","):
        parts = parameter.split("=")
        if len(parts) < 2:
            continue
        key, value = parts[0], parts[1].strip(JS_WHITESPACE)
        if not value:
            number = 0.0
        elif re.fullmatch(r"0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+", value):
            try:
                number = float(int(value, 0))
            except OverflowError:
                number = math.inf
        elif re.fullmatch(r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|Infinity)", value):
            number = float(value)
        else:
            continue
        if not math.isfinite(number) or not number.is_integer() or number <= 0 or number > 0xFFFFFFFF:
            continue
        if key == "i":
            ids.append(int(number))
        elif key == "r":
            rows = int(number)
    return _KittyImageHeader(ids, rows)


def _extract_kitty_image_ids(line: str) -> list[int]:
    header = _parse_kitty_image_header(line)
    return header.ids if header is not None else []


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_log(filename: str, text: str, *, append: bool = False) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")
    with path.open("a" if append else "w", encoding="utf-8", newline="") as output:
        output.write(normalized)


def _json_stringify(value: object, *, indent: int | None = None) -> str:
    result = json.dumps(value, ensure_ascii=False, indent=indent, separators=(",", ":") if indent is None else None)
    result = result.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="surrogatepass")
    return result.encode("utf-8", errors="backslashreplace").decode("utf-8")


@dataclass
class TuiMainScreenRenderState:
    previous_lines: list[str]
    previous_width: int
    previous_height: int
    cursor_row: int
    hardware_cursor_row: int
    max_lines_rendered: int
    previous_viewport_top: int


class TuiMainScreen(TuiBase):
    mode: Literal["regular"] = "regular"

    def __init__(
        self, terminal: Terminal, show_hardware_cursor: bool | None = None,
        log_directory: str | None = None, *, keybindings: AppKeybindings | None = None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor, log_directory, keybindings=keybindings)
        self._previous_lines: list[str] = []
        self._previous_kitty_image_ids: dict[int, None] = {}
        self._previous_width = 0
        self._previous_height = 0
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0

    def capture_render_state(self) -> TuiMainScreenRenderState:
        return TuiMainScreenRenderState(
            list(self._previous_lines), self._previous_width, self._previous_height,
            self._cursor_row, self._hardware_cursor_row, self._max_lines_rendered, self._previous_viewport_top,
        )

    def restore_render_state(self, state: TuiMainScreenRenderState) -> None:
        self._previous_lines = ["" if is_image_line(line) else line for line in state.previous_lines]
        self._previous_kitty_image_ids = {}
        self._previous_width = state.previous_width
        self._previous_height = state.previous_height
        self._cursor_row = state.cursor_row
        self._hardware_cursor_row = state.hardware_cursor_row
        self._max_lines_rendered = state.max_lines_rendered
        self._previous_viewport_top = state.previous_viewport_top

    def _reset_render_state(self) -> None:
        self._previous_lines = []
        self._previous_width = -1
        self._previous_height = -1
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0

    def _before_terminal_stop(self, options: TuiStopOptions) -> None:
        if options.preserve_screen or not self._previous_lines:
            return
        self.terminal.write(" ")
        target_row = len(self._previous_lines)
        line_diff = target_row - self._hardware_cursor_row
        if line_diff > 0:
            self.terminal.write(f"\x1b[{line_diff}B")
        elif line_diff < 0:
            self.terminal.write(f"\x1b[{-line_diff}A")
        self.terminal.write("\r\n")

    def _collect_kitty_image_ids(self, lines: list[str]) -> dict[int, None]:
        ids: dict[int, None] = {}
        for line in lines:
            for image_id in _extract_kitty_image_ids(line):
                ids[image_id] = None
        return ids

    def _delete_kitty_images(self, ids: Iterable[int]) -> str:
        return "".join(delete_kitty_image(image_id) for image_id in ids)

    def _get_kitty_image_reserved_rows(self, lines: list[str], index: int, max_index: int | None = None) -> int:
        if max_index is None:
            max_index = len(lines) - 1
        header = _parse_kitty_image_header(lines[index] if 0 <= index < len(lines) else "")
        rows = header.rows if header is not None else 1
        if rows <= 1:
            return 1
        maximum_rows = min(rows, max_index - index + 1, len(lines) - index)
        reserved_rows = 1
        while reserved_rows < maximum_rows:
            line = lines[index + reserved_rows]
            if is_image_line(line) or visible_width(line) > 0:
                break
            reserved_rows += 1
        return reserved_rows

    def _expand_changed_range_for_kitty_images(self, first_changed: int, last_changed: int, new_lines: list[str]) -> tuple[int, int]:
        expanded_first, expanded_last = first_changed, last_changed
        for lines in (self._previous_lines, new_lines):
            for index, line in enumerate(lines):
                if not _extract_kitty_image_ids(line):
                    continue
                block_end = index + self._get_kitty_image_reserved_rows(lines, index) - 1
                if index >= first_changed or (index <= last_changed and block_end >= first_changed):
                    expanded_first = min(expanded_first, index)
                    expanded_last = max(expanded_last, block_end)
        return expanded_first, expanded_last

    def _delete_changed_kitty_images(self, first_changed: int, last_changed: int) -> str:
        if first_changed < 0 or last_changed < first_changed:
            return ""
        ids: dict[int, None] = {}
        maximum_line = min(last_changed, len(self._previous_lines) - 1)
        for index in range(first_changed, maximum_line + 1):
            for image_id in _extract_kitty_image_ids(self._previous_lines[index]):
                ids[image_id] = None
        return self._delete_kitty_images(ids)

    def _do_render(self) -> None:
        if self._stopped:
            return
        width, height = self.terminal.columns, self.terminal.rows
        width_changed = self._previous_width != 0 and self._previous_width != width
        height_changed = self._previous_height != 0 and self._previous_height != height
        previous_buffer_length = self._previous_viewport_top + self._previous_height if self._previous_height > 0 else height
        previous_viewport_top = max(0, previous_buffer_length - height) if height_changed else self._previous_viewport_top
        viewport_top = previous_viewport_top
        hardware_cursor_row = self._hardware_cursor_row

        def compute_line_diff(target_row: int) -> int:
            return (target_row - viewport_top) - (hardware_cursor_row - previous_viewport_top)

        new_lines = self.render(width)
        if self.has_overlay_entries:
            new_lines = self._composite_overlays(new_lines, width, height)
        cursor_position = self._extract_cursor_position(new_lines, height)
        new_lines = self._apply_line_resets(new_lines)

        def full_render(clear: bool) -> None:
            self._full_redraw_count += 1
            output = _BoundedTerminalWriter(self.terminal.write)
            output.append("\x1b[?2026h")
            if clear:
                output.append(self._delete_kitty_images(self._previous_kitty_image_ids))
                output.append("\x1b[2J\x1b[H\x1b[3J")
            index = 0
            while index < len(new_lines):
                if index > 0:
                    output.append("\r\n")
                line = new_lines[index]
                reserved_rows = self._get_kitty_image_reserved_rows(new_lines, index) if is_image_line(line) else 1
                if reserved_rows > 1 and reserved_rows <= height:
                    for _ in range(1, reserved_rows):
                        output.append("\r\n")
                    output.append(f"\x1b[{reserved_rows - 1}A")
                    output.append(line)
                    output.append(f"\x1b[{reserved_rows - 1}B")
                    index += reserved_rows
                    continue
                output.append(line)
                index += 1
            output.append("\x1b[?2026l")
            output.flush()
            self._cursor_row = max(0, len(new_lines) - 1)
            self._hardware_cursor_row = self._cursor_row
            self._max_lines_rendered = len(new_lines) if clear else max(self._max_lines_rendered, len(new_lines))
            buffer_length = max(height, len(new_lines))
            self._previous_viewport_top = max(0, buffer_length - height)
            self._position_hardware_cursor(cursor_position, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width, self._previous_height = width, height

        redraw_log_directory = self._log_directory if os.environ.get("PI_TUI_DEBUG_REDRAW") == "1" else None

        def log_redraw(reason: str) -> None:
            if redraw_log_directory is None:
                return
            filename = os.path.join(redraw_log_directory, "pi-tui-debug.log")
            message = f"[{_iso_now()}] fullRender: {reason} (prev={len(self._previous_lines)}, new={len(new_lines)}, height={height})\n"
            _write_log(filename, message, append=True)

        if not self._previous_lines and not width_changed and not height_changed:
            log_redraw("first render")
            full_render(False)
            return
        if width_changed:
            log_redraw(f"terminal width changed ({self._previous_width} -> {width})")
            full_render(True)
            return
        if height_changed and not os.environ.get("TERMUX_VERSION"):
            log_redraw(f"terminal height changed ({self._previous_height} -> {height})")
            full_render(True)
            return
        if self.get_clear_on_shrink() and len(new_lines) < self._max_lines_rendered and not self.has_overlay_entries:
            log_redraw(f"clearOnShrink (maxLinesRendered={self._max_lines_rendered})")
            full_render(True)
            return
        first_changed = last_changed = -1
        for index in range(max(len(new_lines), len(self._previous_lines))):
            old_line = self._previous_lines[index] if index < len(self._previous_lines) else ""
            new_line = new_lines[index] if index < len(new_lines) else ""
            if old_line != new_line and old_line.encode("utf-16-le", errors="surrogatepass") != new_line.encode("utf-16-le", errors="surrogatepass"):
                if first_changed == -1:
                    first_changed = index
                last_changed = index
        appended_lines = len(new_lines) > len(self._previous_lines)
        if appended_lines:
            if first_changed == -1:
                first_changed = len(self._previous_lines)
            last_changed = len(new_lines) - 1
        if first_changed != -1:
            first_changed, last_changed = self._expand_changed_range_for_kitty_images(first_changed, last_changed, new_lines)
        append_start = appended_lines and first_changed == len(self._previous_lines) and first_changed > 0
        if first_changed == -1:
            self._position_hardware_cursor(cursor_position, len(new_lines))
            self._previous_viewport_top = previous_viewport_top
            self._previous_height = height
            return
        if first_changed >= len(new_lines):
            if len(self._previous_lines) > len(new_lines):
                output = _BoundedTerminalWriter(self.terminal.write)
                output.append("\x1b[?2026h")
                output.append(self._delete_changed_kitty_images(first_changed, last_changed))
                target_row = max(0, len(new_lines) - 1)
                if target_row < previous_viewport_top:
                    log_redraw(f"deleted lines moved viewport up ({target_row} < {previous_viewport_top})")
                    full_render(True)
                    return
                line_diff = compute_line_diff(target_row)
                if line_diff > 0:
                    output.append(f"\x1b[{line_diff}B")
                elif line_diff < 0:
                    output.append(f"\x1b[{-line_diff}A")
                output.append("\r")
                extra_lines = len(self._previous_lines) - len(new_lines)
                if extra_lines > height:
                    log_redraw(f"extraLines > height ({extra_lines} > {height})")
                    full_render(True)
                    return
                clear_start_offset = 0 if not new_lines else 1
                if extra_lines > 0 and clear_start_offset > 0:
                    output.append(f"\x1b[{clear_start_offset}B")
                for index in range(extra_lines):
                    output.append("\r\x1b[2K")
                    if index < extra_lines - 1:
                        output.append("\x1b[1B")
                move_back = max(0, extra_lines - 1 + clear_start_offset)
                if move_back > 0:
                    output.append(f"\x1b[{move_back}A")
                output.append("\x1b[?2026l")
                output.flush()
                self._cursor_row = target_row
                self._hardware_cursor_row = target_row
            self._position_hardware_cursor(cursor_position, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width, self._previous_height = width, height
            self._previous_viewport_top = previous_viewport_top
            return
        if first_changed < previous_viewport_top:
            log_redraw(f"firstChanged < viewportTop ({first_changed} < {previous_viewport_top})")
            full_render(True)
            return
        output = _BoundedTerminalWriter(self.terminal.write)
        output.append("\x1b[?2026h")
        output.append(self._delete_changed_kitty_images(first_changed, last_changed))
        previous_viewport_bottom = previous_viewport_top + height - 1
        move_target_row = first_changed - 1 if append_start else first_changed
        if move_target_row > previous_viewport_bottom:
            current_screen_row = max(0, min(height - 1, hardware_cursor_row - previous_viewport_top))
            move_to_bottom = height - 1 - current_screen_row
            if move_to_bottom > 0:
                output.append(f"\x1b[{move_to_bottom}B")
            scroll = move_target_row - previous_viewport_bottom
            output.append("\r\n" * scroll)
            previous_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row
        line_diff = compute_line_diff(move_target_row)
        if line_diff > 0:
            output.append(f"\x1b[{line_diff}B")
        elif line_diff < 0:
            output.append(f"\x1b[{-line_diff}A")
        output.append("\r\n" if append_start else "\r")
        render_end = min(last_changed, len(new_lines) - 1)
        index = first_changed
        while index <= render_end:
            if index > first_changed:
                output.append("\r\n")
            line = new_lines[index]
            is_image = is_image_line(line)
            reserved_rows = self._get_kitty_image_reserved_rows(new_lines, index, render_end) if is_image else 1
            if reserved_rows > 1:
                image_start_screen_row = index - viewport_top
                if image_start_screen_row < 0 or image_start_screen_row + reserved_rows > height:
                    log_redraw(f"kitty image pre-clear would scroll ({image_start_screen_row} + {reserved_rows} > {height})")
                    full_render(True)
                    return
                output.append("\x1b[2K")
                for _ in range(1, reserved_rows):
                    output.append("\r\n\x1b[2K")
                output.append(f"\x1b[{reserved_rows - 1}A")
                output.append(line)
                output.append(f"\x1b[{reserved_rows - 1}B")
                index += reserved_rows
                continue
            output.append("\x1b[2K")
            if not is_image and visible_width(line) > width:
                crash_log_path = os.path.join(self._log_directory if self._log_directory is not None else tempfile.gettempdir(), "pi-tui-crash.log")
                crash_data = "\n".join([
                    f"Crash at {_iso_now()}", f"Terminal width: {width}",
                    f"Line {index} visible width: {visible_width(line)}", "", "=== All rendered lines ===",
                    *(f"[{line_index}] (w={visible_width(value)}) {value}" for line_index, value in enumerate(new_lines)), "",
                ])
                _write_log(crash_log_path, crash_data)
                self.stop()
                raise RuntimeError("\n".join([
                    f"Rendered line {index} exceeds terminal width ({visible_width(line)} > {width}).", "",
                    "This is likely caused by a custom TUI component not truncating its output.",
                    "Use visibleWidth() to measure and truncateToWidth() to truncate lines.", "",
                    f"Debug log written to: {crash_log_path}",
                ]))
            output.append(line)
            index += 1
        final_cursor_row = render_end
        if len(self._previous_lines) > len(new_lines):
            if render_end < len(new_lines) - 1:
                move_down = len(new_lines) - 1 - render_end
                output.append(f"\x1b[{move_down}B")
                final_cursor_row = len(new_lines) - 1
            extra_lines = len(self._previous_lines) - len(new_lines)
            for _ in range(len(new_lines), len(self._previous_lines)):
                output.append("\r\n\x1b[2K")
            output.append(f"\x1b[{extra_lines}A")
        output.append("\x1b[?2026l")
        if os.environ.get("PI_TUI_DEBUG") == "1":
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"
            timestamp = time.time_ns() // 1_000_000
            fraction = random.random()
            suffix = ""
            for _ in range(12):
                if fraction == 0:
                    break
                fraction *= 36
                digit = math.floor(fraction)
                suffix += digits[digit]
                fraction -= digit
            debug_path = os.path.join("/tmp/tui", f"render-{timestamp}-{suffix}.log")
            cursor_data = None if cursor_position is None else {"row": cursor_position.row, "col": cursor_position.col}
            debug_data = "\n".join([
                f"firstChanged: {first_changed}", f"viewportTop: {viewport_top}", f"cursorRow: {self._cursor_row}",
                f"height: {height}", f"lineDiff: {line_diff}", f"hardwareCursorRow: {hardware_cursor_row}",
                f"renderEnd: {render_end}", f"finalCursorRow: {final_cursor_row}",
                f"cursorPos: {_json_stringify(cursor_data)}", f"newLines.length: {len(new_lines)}",
                f"previousLines.length: {len(self._previous_lines)}", "", "=== newLines ===",
                _json_stringify(new_lines, indent=2), "", "=== previousLines ===",
                _json_stringify(self._previous_lines, indent=2), "", "=== buffer ===",
                f"[{output.length} chars written in bounded chunks]",
            ])
            _write_log(debug_path, debug_data)
        output.flush()
        self._cursor_row = max(0, len(new_lines) - 1)
        self._hardware_cursor_row = final_cursor_row
        self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
        self._previous_viewport_top = max(previous_viewport_top, final_cursor_row - height + 1)
        self._position_hardware_cursor(cursor_position, len(new_lines))
        self._previous_lines = new_lines
        self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
        self._previous_width, self._previous_height = width, height

    def _position_hardware_cursor(self, cursor_position: CursorPosition | None, total_lines: int) -> None:
        if cursor_position is None or total_lines <= 0:
            self.terminal.hide_cursor()
            return
        target_row = max(0, min(cursor_position.row, total_lines - 1))
        target_col = max(0, cursor_position.col)
        row_delta = target_row - self._hardware_cursor_row
        buffer = ""
        if row_delta > 0:
            buffer += f"\x1b[{row_delta}B"
        elif row_delta < 0:
            buffer += f"\x1b[{-row_delta}A"
        buffer += f"\x1b[{target_col + 1}G"
        self.terminal.write(buffer)
        self._hardware_cursor_row = target_row
        if self.get_show_hardware_cursor():
            self.terminal.show_cursor()
        else:
            self.terminal.hide_cursor()


__all__ = ["TuiMainScreen", "TuiMainScreenRenderState"]
