"""Real process terminal, keyboard negotiation, and output control sequences."""

from __future__ import annotations

import asyncio
import math
import os
import re
import signal
import stat
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
from typing import Literal, TypedDict

from ._javascript import js_trim
from ._process_io import ProcessIO, get_process_io, utf8_bytes
from ._terminal_types import Terminal
from ._timers import IntervalHandle, TimerHandle, set_timeout
from .keys import set_kitty_protocol_active
from .native_modifiers import is_native_modifier_pressed
from .native_platform import get_native_platform_helper
from .stdin_buffer import StdinBuffer, StdinBufferOptions

TERMINAL_PROGRESS_KEEPALIVE_MS = 1000
TERMINAL_PROGRESS_ACTIVE_SEQUENCE = "\x1b]9;4;3\x07"
TERMINAL_PROGRESS_CLEAR_SEQUENCE = "\x1b]9;4;0\x07"
NATIVE_SHIFT_ENTER_SEQUENCE = "\x1b[13;2u"
DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS = 7
KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS = 150
KITTY_KEYBOARD_PROTOCOL_QUERY = f"\x1b[>{DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS}u\x1b[?u\x1b[c"
DEFAULT_ESCAPE_TIMEOUT_MS = 10
DEFAULT_SSH_ESCAPE_TIMEOUT_MS = 100


class KittyFlagsNegotiationSequence(TypedDict):
    type: Literal["kitty-flags"]
    flags: int | float


class DeviceAttributesNegotiationSequence(TypedDict):
    type: Literal["device-attributes"]


type KeyboardProtocolNegotiationSequence = KittyFlagsNegotiationSequence | DeviceAttributesNegotiationSequence


def parse_keyboard_protocol_negotiation_sequence(sequence: str) -> KeyboardProtocolNegotiationSequence | None:
    kitty_flags = re.fullmatch(r"\x1b\[\?([0-9]+)u", sequence)
    if kitty_flags:
        number = float(kitty_flags[1])
        return {"type": "kitty-flags", "flags": int(number) if math.isfinite(number) else number}
    if re.fullmatch(r"\x1b\[\?[0-9;]*c", sequence):
        return {"type": "device-attributes"}
    return None


def _is_keyboard_protocol_negotiation_sequence_prefix(sequence: str) -> bool:
    return sequence == "\x1b[" or re.fullmatch(r"\x1b\[\?[0-9;]*", sequence) is not None


def is_apple_terminal_session() -> bool:
    return sys.platform == "darwin" and os.environ.get("TERM_PROGRAM") == "Apple_Terminal"


def refresh_terminal_dimensions() -> None:
    if sys.platform == "win32" or os.getpid() <= 0:
        return
    try:
        os.kill(os.getpid(), signal.SIGWINCH)
    except OSError:
        pass


def normalize_native_shift_enter_input(data: str, should_detect_native_shift_enter: bool, is_shift_pressed: bool) -> str:
    return NATIVE_SHIFT_ENTER_SEQUENCE if should_detect_native_shift_enter and data == "\r" and is_shift_pressed else data


def normalize_apple_terminal_input(data: str, is_apple_terminal: bool, is_shift_pressed: bool) -> str:
    return normalize_native_shift_enter_input(data, is_apple_terminal, is_shift_pressed)


def _number(value: str | None) -> float:
    if value is None:
        return math.nan
    text = js_trim(value)
    if not text:
        return 0.0
    if text in ("Infinity", "+Infinity", "-Infinity"):
        return -math.inf if text.startswith("-") else math.inf
    for pattern, base in ((r"0[xX][0-9a-fA-F]+", 16), (r"0[bB][01]+", 2), (r"0[oO][0-7]+", 8)):
        if re.fullmatch(pattern, text):
            try:
                return float(int(text[2:], base))
            except OverflowError:
                return math.inf
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
        return math.nan
    return float(text)


def resolve_escape_timeout_ms(env: Mapping[str, str] | None = None) -> float:
    environment = os.environ if env is None else env
    configured = _number(environment.get("PI_TUI_ESC_TIMEOUT"))
    if math.isfinite(configured) and configured > 0:
        return configured
    if environment.get("SSH_CONNECTION") or environment.get("SSH_TTY"):
        return DEFAULT_SSH_ESCAPE_TIMEOUT_MS
    return DEFAULT_ESCAPE_TIMEOUT_MS


def _number_string(value: int | float) -> str:
    number = float(value)
    if math.isinf(number):
        return "-Infinity" if number < 0 else "Infinity"
    if math.isnan(number):
        return "NaN"
    if number == 0:
        return "0"
    result = repr(number)
    if 1e-6 <= abs(number) < 1e21:
        result = format(Decimal(result), "f")
        return result.rstrip("0").rstrip(".") if "." in result else result
    mantissa, exponent = result.split("e")
    if mantissa.endswith(".0"):
        mantissa = mantissa[:-2]
    exponent_value = int(exponent)
    return f"{mantissa}e{'+' if exponent_value >= 0 else '-'}{abs(exponent_value)}"


class ProcessTerminal:
    def __init__(self, io: ProcessIO | None = None) -> None:
        self._io = get_process_io() if io is None else io
        self._was_raw = False
        self._input_handler: Callable[[str], None] | None = None
        self._resize_handler: Callable[[], None] | None = None
        self._kitty_protocol_active = False
        self._modify_other_keys_active = False
        self._keyboard_protocol_pushed = False
        self._keyboard_protocol_negotiation_buffer = ""
        self._keyboard_protocol_buffer_flush_timer: TimerHandle | None = None
        self._stdin_buffer: StdinBuffer | None = None
        self._stdin_data_handler: Callable[[str], None] | None = None
        self._progress_interval: IntervalHandle | None = None
        self._write_log_path = self._io.env.get("PI_TUI_WRITE_LOG") or ""
        if self._write_log_path:
            try:
                if stat.S_ISDIR(os.stat(self._write_log_path).st_mode):
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    self._write_log_path = os.path.join(self._write_log_path, f"tui-{timestamp}-{self._io.pid}.log")
            except Exception:
                pass

    @property
    def kitty_protocol_active(self) -> bool:
        return self._kitty_protocol_active

    @property
    def modify_other_keys_active(self) -> bool:
        return self._modify_other_keys_active

    def start(self, on_input: Callable[[str], None], on_resize: Callable[[], None]) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize
        self._was_raw = self._io.stdin.is_raw or False
        if self._io.stdin.set_raw_mode is not None:
            self._io.stdin.set_raw_mode(True)
        self._io.stdin.set_encoding("utf8")
        self._io.stdin.resume()
        self._io.stdout.write("\x1b[?2004h")
        self._io.stdout.on("resize", self._resize_handler)
        self._io.refresh_dimensions()
        self._enable_windows_vt_input()
        self._query_and_enable_kitty_protocol()

    def _setup_stdin_buffer(self) -> None:
        self._stdin_buffer = StdinBuffer(StdinBufferOptions(escape_timeout=resolve_escape_timeout_ms(self._io.env)))

        def on_data(sequence: str) -> None:
            negotiation_sequence = self._read_keyboard_protocol_negotiation_sequence(sequence)
            if negotiation_sequence == "pending":
                self._schedule_keyboard_protocol_negotiation_buffer_flush()
                return
            if self._handle_keyboard_protocol_negotiation_sequence(negotiation_sequence):
                return
            self._forward_input_sequence(sequence)

        def on_paste(content: str) -> None:
            if self._input_handler is not None:
                self._input_handler(f"\x1b[200~{content}\x1b[201~")

        self._stdin_buffer.on("data", on_data)
        self._stdin_buffer.on("paste", on_paste)

        def pipe_input(data: str) -> None:
            buffer = self._stdin_buffer
            if buffer is not None:
                buffer.process(data)

        self._stdin_data_handler = pipe_input

    def _query_and_enable_kitty_protocol(self) -> None:
        self._setup_stdin_buffer()
        if self._stdin_data_handler is not None:
            self._io.stdin.on("data", self._stdin_data_handler)
        self._keyboard_protocol_pushed = True
        self._clear_keyboard_protocol_negotiation_buffer()
        self._io.stdout.write(KITTY_KEYBOARD_PROTOCOL_QUERY)

    def _handle_keyboard_protocol_negotiation_sequence(self, sequence: KeyboardProtocolNegotiationSequence | None) -> bool:
        if sequence is None:
            return False
        self._clear_keyboard_protocol_negotiation_buffer()
        if sequence["type"] == "kitty-flags":
            if sequence["flags"] != 0:
                self._disable_modify_other_keys()
                if not self._kitty_protocol_active:
                    self._kitty_protocol_active = True
                    set_kitty_protocol_active(True)
            else:
                self._enable_modify_other_keys()
            return True
        if not self._kitty_protocol_active:
            self._enable_modify_other_keys()
        return True

    def _read_keyboard_protocol_negotiation_sequence(self, sequence: str) -> KeyboardProtocolNegotiationSequence | Literal["pending"] | None:
        if self._keyboard_protocol_negotiation_buffer:
            buffered_sequence = self._keyboard_protocol_negotiation_buffer + sequence
            negotiation_sequence = parse_keyboard_protocol_negotiation_sequence(buffered_sequence)
            if negotiation_sequence is not None:
                self._clear_keyboard_protocol_negotiation_buffer()
                return negotiation_sequence
            if _is_keyboard_protocol_negotiation_sequence_prefix(buffered_sequence):
                self._set_keyboard_protocol_negotiation_buffer(buffered_sequence)
                return "pending"
            self._flush_keyboard_protocol_negotiation_buffer_as_input()
        negotiation_sequence = parse_keyboard_protocol_negotiation_sequence(sequence)
        if negotiation_sequence is not None:
            return negotiation_sequence
        if _is_keyboard_protocol_negotiation_sequence_prefix(sequence):
            self._set_keyboard_protocol_negotiation_buffer(sequence)
            return "pending"
        return None

    def _set_keyboard_protocol_negotiation_buffer(self, sequence: str) -> None:
        self._clear_keyboard_protocol_negotiation_buffer_flush_timer()
        self._keyboard_protocol_negotiation_buffer = sequence

    def _clear_keyboard_protocol_negotiation_buffer(self) -> None:
        self._clear_keyboard_protocol_negotiation_buffer_flush_timer()
        self._keyboard_protocol_negotiation_buffer = ""

    def _flush_keyboard_protocol_negotiation_buffer_as_input(self) -> None:
        if not self._keyboard_protocol_negotiation_buffer:
            return
        sequence = self._keyboard_protocol_negotiation_buffer
        self._clear_keyboard_protocol_negotiation_buffer()
        self._forward_input_sequence(sequence)

    def _schedule_keyboard_protocol_negotiation_buffer_flush(self) -> None:
        if not self._keyboard_protocol_negotiation_buffer or self._keyboard_protocol_buffer_flush_timer is not None:
            return

        def expired() -> None:
            self._keyboard_protocol_buffer_flush_timer = None
            self._flush_keyboard_protocol_negotiation_buffer_as_input()

        self._keyboard_protocol_buffer_flush_timer = set_timeout(expired, KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS)

    def _clear_keyboard_protocol_negotiation_buffer_flush_timer(self) -> None:
        if self._keyboard_protocol_buffer_flush_timer is not None:
            self._keyboard_protocol_buffer_flush_timer.cancel()
            self._keyboard_protocol_buffer_flush_timer = None

    def _forward_input_sequence(self, sequence: str) -> None:
        if self._input_handler is None:
            return
        apple_terminal = self._io.platform == "darwin" and self._io.env.get("TERM_PROGRAM") == "Apple_Terminal"
        detect = sequence == "\r" and (apple_terminal or self._io.platform == "win32")
        data = normalize_native_shift_enter_input(sequence, detect, detect and is_native_modifier_pressed("shift"))
        self._input_handler(data)

    def _enable_modify_other_keys(self) -> None:
        if self._kitty_protocol_active or self._modify_other_keys_active:
            return
        self._io.stdout.write("\x1b[>4;2m")
        self._modify_other_keys_active = True

    def _disable_modify_other_keys(self) -> None:
        if self._modify_other_keys_active:
            self._io.stdout.write("\x1b[>4;0m")
            self._modify_other_keys_active = False

    def _enable_windows_vt_input(self) -> None:
        if self._io.platform != "win32":
            return
        try:
            enable = getattr(get_native_platform_helper(), "enable_virtual_terminal_input", None)
            if callable(enable):
                enable()
        except Exception:
            pass

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None:
        should_disable = self._keyboard_protocol_pushed or self._kitty_protocol_active
        self._clear_keyboard_protocol_negotiation_buffer()
        if should_disable:
            self._io.stdout.write("\x1b[<u")
            self._keyboard_protocol_pushed = False
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        self._disable_modify_other_keys()
        previous_handler = self._input_handler
        self._input_handler = None
        last_data_time = time.time_ns() // 1_000_000

        def on_data(_data: str) -> None:
            nonlocal last_data_time
            last_data_time = time.time_ns() // 1_000_000

        self._io.stdin.on("data", on_data)
        end_time = time.time_ns() // 1_000_000 + max_ms
        try:
            while True:
                now = time.time_ns() // 1_000_000
                time_left = end_time - now
                if time_left <= 0 or now - last_data_time >= idle_ms:
                    break
                milliseconds = math.nan if math.isnan(idle_ms) or math.isnan(time_left) else min(idle_ms, time_left)
                delay = 1 if not math.isfinite(milliseconds) or milliseconds < 1 or milliseconds > 2147483647 else math.trunc(milliseconds)
                await asyncio.sleep(delay / 1000)
        finally:
            self._io.stdin.remove_listener("data", on_data)
            self._input_handler = previous_handler

    def stop(self) -> None:
        if self._clear_progress_interval():
            self._io.stdout.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE)
        self._io.stdout.write("\x1b[?2004l")
        should_disable = self._keyboard_protocol_pushed or self._kitty_protocol_active
        self._clear_keyboard_protocol_negotiation_buffer()
        if should_disable:
            self._io.stdout.write("\x1b[<u")
            self._keyboard_protocol_pushed = False
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        self._disable_modify_other_keys()
        if self._stdin_buffer is not None:
            self._stdin_buffer.destroy()
            self._stdin_buffer = None
        if self._stdin_data_handler is not None:
            self._io.stdin.remove_listener("data", self._stdin_data_handler)
            self._stdin_data_handler = None
        self._input_handler = None
        if self._resize_handler is not None:
            self._io.stdout.remove_listener("resize", self._resize_handler)
            self._resize_handler = None
        self._io.stdin.pause()
        if self._io.stdin.set_raw_mode is not None:
            self._io.stdin.set_raw_mode(self._was_raw)

    def write(self, data: str) -> None:
        self._io.stdout.write(data)
        if self._write_log_path:
            try:
                with open(self._write_log_path, "ab") as log:
                    log.write(utf8_bytes(data))
            except Exception:
                pass

    @property
    def columns(self) -> int | float:
        columns = self._io.stdout.columns
        if columns:
            return columns
        configured = _number(self._io.env.get("COLUMNS"))
        if configured != 0 and not math.isnan(configured):
            return int(configured) if math.isfinite(configured) and configured.is_integer() else configured
        return 80

    @property
    def rows(self) -> int | float:
        rows = self._io.stdout.rows
        if rows:
            return rows
        configured = _number(self._io.env.get("LINES"))
        if configured != 0 and not math.isnan(configured):
            return int(configured) if math.isfinite(configured) and configured.is_integer() else configured
        return 24

    def move_by(self, lines: int | float) -> None:
        if lines > 0:
            self._io.stdout.write(f"\x1b[{_number_string(lines)}B")
        elif lines < 0:
            self._io.stdout.write(f"\x1b[{_number_string(-lines)}A")

    def hide_cursor(self) -> None:
        self._io.stdout.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self._io.stdout.write("\x1b[?25h")

    def clear_line(self) -> None:
        self._io.stdout.write("\x1b[K")

    def clear_from_cursor(self) -> None:
        self._io.stdout.write("\x1b[J")

    def clear_screen(self) -> None:
        self._io.stdout.write("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        self._io.stdout.write(f"\x1b]0;{title}\x07")

    def set_progress(self, active: bool) -> None:
        if active:
            self._io.stdout.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE)
            if self._progress_interval is None:
                self._progress_interval = IntervalHandle(lambda: self._io.stdout.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE), TERMINAL_PROGRESS_KEEPALIVE_MS)
        else:
            self._clear_progress_interval()
            self._io.stdout.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE)

    def _clear_progress_interval(self) -> bool:
        if self._progress_interval is None:
            return False
        self._progress_interval.cancel()
        self._progress_interval = None
        return True


__all__ = [
    "Terminal", "ProcessIO", "ProcessTerminal", "KeyboardProtocolNegotiationSequence",
    "KittyFlagsNegotiationSequence", "DeviceAttributesNegotiationSequence",
    "parse_keyboard_protocol_negotiation_sequence", "is_apple_terminal_session", "refresh_terminal_dimensions",
    "normalize_native_shift_enter_input", "normalize_apple_terminal_input", "resolve_escape_timeout_ms",
]
