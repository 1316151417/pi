"""Incremental stdin escape/paste buffering ported from stdin-buffer.ts.

Input callbacks run synchronously in registration order. Timeouts run on the
active asyncio loop, or a serialized daemon timer when no loop is running.

Based on OpenTUI (https://github.com/anomalyco/opentui), as credited upstream.
MIT License - Copyright (c) 2025 opentui.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

ESC = "\x1b"
DEFAULT_SEQUENCE_TIMEOUT_MS = 50
DEFAULT_ESCAPE_TIMEOUT_MS = 10
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"
type StdinBufferEvent = Literal["data", "paste"]
type StdinBufferListener = Callable[..., object]


class StdinBufferEventMap(TypedDict):
    data: tuple[str]
    paste: tuple[str]


@dataclass
class StdinBufferOptions:
    timeout: float | None = None
    escape_timeout: float | None = None


@dataclass(eq=False)
class _Listener:
    callback: StdinBufferListener
    once: bool = False
    wrapper: StdinBufferListener | None = None


def _units(text: str) -> str:
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(encoded[index] | encoded[index + 1] << 8) for index in range(0, len(encoded), 2))


def _text(units: str) -> str:
    return units.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="surrogatepass")


def _is_complete_sequence(data: str) -> Literal["complete", "incomplete", "not-escape"]:
    if not data.startswith(ESC):
        return "not-escape"
    if len(data) == 1:
        return "incomplete"
    after_escape = data[1:]
    if after_escape.startswith("["):
        if after_escape.startswith("[M"):
            return "complete" if len(data) >= 6 else "incomplete"
        if len(data) < 3:
            return "incomplete"
        payload = data[2:]
        if 0x40 <= ord(payload[-1]) <= 0x7E:
            if payload.startswith("<"):
                return "complete" if re.fullmatch(r"<[0-9]+;[0-9]+;[0-9]+[Mm]", payload) else "incomplete"
            return "complete"
        return "incomplete"
    if after_escape.startswith("]"):
        return "complete" if data.endswith((ESC + "\\", "\x07")) else "incomplete"
    if after_escape.startswith(("P", "_")):
        return "complete" if data.endswith(ESC + "\\") else "incomplete"
    if after_escape.startswith("O"):
        return "complete" if len(after_escape) >= 2 else "incomplete"
    return "complete"


def _extract_complete_sequences(buffer: str) -> tuple[list[str], str]:
    sequences: list[str] = []
    position = 0
    while position < len(buffer):
        remaining = buffer[position:]
        if not remaining.startswith(ESC):
            sequences.append(remaining[0])
            position += 1
            continue
        end = 1
        while end <= len(remaining):
            candidate = remaining[:end]
            status = _is_complete_sequence(candidate)
            if status == "complete":
                if candidate == ESC + ESC and remaining[end:end + 1] in ("[", "]", "O", "P", "_"):
                    sequences.append(ESC)
                    position += 1
                    break
                sequences.append(candidate)
                position += end
                break
            if status == "incomplete":
                end += 1
            else:
                sequences.append(candidate)
                position += end
                break
        if end > len(remaining):
            return sequences, remaining
    return sequences, ""


class StdinBuffer:
    def __init__(self, options: StdinBufferOptions | None = None) -> None:
        options = options or StdinBufferOptions()
        self._timeout_ms = DEFAULT_SEQUENCE_TIMEOUT_MS if options.timeout is None else options.timeout
        self._escape_timeout_ms = DEFAULT_ESCAPE_TIMEOUT_MS if options.escape_timeout is None else options.escape_timeout
        self._buffer = ""
        self._timeout: asyncio.TimerHandle | threading.Timer | None = None
        self._timeout_generation = 0
        self._paste_mode = False
        self._paste_buffer = ""
        self._pending_kitty_printable_codepoint: float | None = None
        self._listeners: dict[str, list[_Listener]] = {}
        self._max_listeners: int | float = 10
        self._warned_events: set[str] = set()
        self._lock = threading.RLock()

    def _add_listener(self, event: str, callback: StdinBufferListener, *, once: bool, prepend: bool) -> StdinBuffer:
        if not callable(callback):
            raise TypeError("The listener argument must be a function")
        with self._lock:
            self.emit("newListener", event, callback)
            entry = _Listener(callback, once)
            if once:
                def wrapper(*arguments: object) -> object:
                    with self._lock:
                        if not entry.once:
                            return None
                        entry.once = False
                        self._remove_entry(event, entry)
                        return callback(*arguments)

                entry.wrapper = wrapper
            listeners = self._listeners.setdefault(event, [])
            listeners.insert(0, entry) if prepend else listeners.append(entry)
            if self._max_listeners and len(listeners) > self._max_listeners and event not in self._warned_events:
                self._warned_events.add(event)
                warnings.warn(f"Possible EventEmitter memory leak detected: {len(listeners)} {event} listeners", RuntimeWarning)
        return self

    def on(self, event: str, listener: StdinBufferListener) -> StdinBuffer:
        return self._add_listener(event, listener, once=False, prepend=False)

    add_listener = on

    def once(self, event: str, listener: StdinBufferListener) -> StdinBuffer:
        return self._add_listener(event, listener, once=True, prepend=False)

    def prepend_listener(self, event: str, listener: StdinBufferListener) -> StdinBuffer:
        return self._add_listener(event, listener, once=False, prepend=True)

    def prepend_once_listener(self, event: str, listener: StdinBufferListener) -> StdinBuffer:
        return self._add_listener(event, listener, once=True, prepend=True)

    def emit(self, event: str, *arguments: object) -> bool:
        with self._lock:
            entries = tuple(self._listeners.get(event, ()))
            if not entries:
                if event == "error":
                    if arguments and isinstance(arguments[0], BaseException):
                        raise arguments[0]
                    raise RuntimeError(f"Unhandled error: {arguments[0] if arguments else None}")
                return False
            for entry in entries:
                (entry.wrapper or entry.callback)(*arguments)
            return True

    def _remove_entry(self, event: str, entry: _Listener) -> None:
        listeners = self._listeners.get(event)
        if listeners is None or entry not in listeners:
            return
        listeners.remove(entry)
        if not listeners:
            del self._listeners[event]
            self._warned_events.discard(event)
        self.emit("removeListener", event, entry.callback)

    def remove_listener(self, event: str, listener: StdinBufferListener) -> StdinBuffer:
        with self._lock:
            for entry in reversed(self._listeners.get(event, ())):
                if entry.callback is listener or entry.wrapper is listener:
                    self._remove_entry(event, entry)
                    break
        return self

    off = remove_listener
    remove = remove_listener

    def remove_all_listeners(self, event: str | None = None) -> StdinBuffer:
        with self._lock:
            events = [event] if event is not None else [key for key in self._listeners if key != "removeListener"] + ["removeListener"]
            for name in events:
                for entry in reversed(tuple(self._listeners.get(name, ()))):
                    self._remove_entry(name, entry)
        return self

    def listeners(self, event: str) -> list[StdinBufferListener]:
        with self._lock:
            return [entry.callback for entry in self._listeners.get(event, ())]

    def raw_listeners(self, event: str) -> list[StdinBufferListener]:
        with self._lock:
            return [entry.wrapper or entry.callback for entry in self._listeners.get(event, ())]

    def listener_count(self, event: str, listener: StdinBufferListener | None = None) -> int:
        with self._lock:
            entries = self._listeners.get(event, ())
            return len(entries) if listener is None else sum(entry.callback is listener or entry.wrapper is listener for entry in entries)

    def event_names(self) -> list[str]:
        with self._lock:
            return list(self._listeners)

    def set_max_listeners(self, count: int | float) -> StdinBuffer:
        if isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0 or math.isnan(count):
            raise ValueError("Maximum listener count must be nonnegative")
        self._max_listeners = count
        return self

    def get_max_listeners(self) -> int | float:
        return self._max_listeners

    def _cancel_timeout(self) -> None:
        self._timeout_generation += 1
        if self._timeout is not None:
            self._timeout.cancel()
            self._timeout = None

    def process(self, data: str | bytes | bytearray | memoryview) -> None:
        with self._lock:
            self._cancel_timeout()
            if isinstance(data, (bytes, bytearray, memoryview)):
                raw = bytes(data)
                text = ESC + chr(raw[0] - 128) if len(raw) == 1 and raw[0] > 127 else raw.decode("utf-8", errors="replace")
            else:
                text = data
            text = _units(text)
            if not text and not self._buffer:
                self._emit_data_sequence("")
                return
            self._buffer += text
            if self._paste_mode:
                self._paste_buffer += self._buffer
                self._buffer = ""
                self._finish_paste_if_complete()
                return
            start = self._buffer.find(BRACKETED_PASTE_START)
            if start != -1:
                if start > 0:
                    sequences, _ = _extract_complete_sequences(self._buffer[:start])
                    for sequence in sequences:
                        self._emit_data_sequence(sequence)
                self._pending_kitty_printable_codepoint = None
                self._buffer = self._buffer[start + len(BRACKETED_PASTE_START):]
                self._paste_mode = True
                self._paste_buffer = self._buffer
                self._buffer = ""
                self._finish_paste_if_complete()
                return
            sequences, self._buffer = _extract_complete_sequences(self._buffer)
            for sequence in sequences:
                self._emit_data_sequence(sequence)
            if self._buffer:
                milliseconds = self._escape_timeout_ms if self._buffer == ESC else self._timeout_ms
                delay = 1 if not math.isfinite(milliseconds) or milliseconds < 1 or milliseconds > 2**31 - 1 else math.trunc(milliseconds)
                generation = self._timeout_generation

                def expired() -> None:
                    with self._lock:
                        if generation != self._timeout_generation:
                            return
                        for sequence in self.flush():
                            self._emit_data_sequence(_units(sequence))

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    timer = threading.Timer(delay / 1000, expired)
                    timer.daemon = True
                    self._timeout = timer
                    timer.start()
                else:
                    self._timeout = loop.call_later(delay / 1000, expired)

    def _finish_paste_if_complete(self) -> None:
        end = self._paste_buffer.find(BRACKETED_PASTE_END)
        if end == -1:
            return
        pasted_content = self._paste_buffer[:end]
        remaining = self._paste_buffer[end + len(BRACKETED_PASTE_END):]
        self._paste_mode = False
        self._paste_buffer = ""
        self._pending_kitty_printable_codepoint = None
        self.emit("paste", _text(pasted_content))
        if remaining:
            self.process(_text(remaining))

    def _emit_data_sequence(self, sequence: str) -> None:
        codepoint = ord(sequence) if len(sequence) == 1 else None
        if codepoint is not None and codepoint == self._pending_kitty_printable_codepoint:
            self._pending_kitty_printable_codepoint = None
            return
        match = re.fullmatch(r"\x1b\[([0-9]+)(?::[0-9]*)?(?::[0-9]+)?u", sequence)
        parsed = float(match[1]) if match else None
        self._pending_kitty_printable_codepoint = parsed if parsed is not None and parsed >= 32 else None
        self.emit("data", _text(sequence))

    def flush(self) -> list[str]:
        with self._lock:
            self._cancel_timeout()
            if not self._buffer:
                return []
            sequences = [_text(self._buffer)]
            self._buffer = ""
            self._pending_kitty_printable_codepoint = None
            return sequences

    def clear(self) -> None:
        with self._lock:
            self._cancel_timeout()
            self._buffer = ""
            self._paste_mode = False
            self._paste_buffer = ""
            self._pending_kitty_printable_codepoint = None

    def get_buffer(self) -> str:
        with self._lock:
            return _text(self._buffer)

    def destroy(self) -> None:
        self.clear()
