"""XCB clipboard reads with the native worker's bounded-wait ownership model."""

from __future__ import annotations

import ctypes
import errno
import select
import struct
import threading
import time
from dataclasses import dataclass
from typing import cast

from ._native_common import UNDEFINED, ClipboardImage, ClipboardText, Undefined, run_clipboard

MAX_CLIPBOARD_BYTES = 50 * 1024 * 1024
CLIPBOARD_TIMEOUT_MS = 2000
_TEXT_TYPES = (b"text/plain;charset=utf-8", b"text/plain;charset=UTF-8", b"UTF8_STRING", b"text/plain", b"STRING")
_IMAGE_TYPES = (b"image/png", b"image/jpeg", b"image/webp", b"image/gif", b"image/bmp", b"image/tiff")


class _Cookie(ctypes.Structure):
    _fields_ = [("sequence", ctypes.c_uint)]


class _ScreenIterator(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("rem", ctypes.c_int), ("index", ctypes.c_int)]


@dataclass
class _PropertyData:
    data: bytearray | None = None
    items: int = 0
    format: int = 0
    type: int = 0

    @property
    def length(self) -> int:
        return len(self.data) if self.data is not None else 0

    def append_bytes(self, data: bytes | bytearray) -> bool:
        if len(data) > MAX_CLIPBOARD_BYTES - self.length:
            return False
        try:
            if self.data is None:
                self.data = bytearray(data)
            else:
                self.data.extend(data)
        except MemoryError:
            return False
        return True

    def append_property(self, chunk: _PropertyData) -> bool:
        if (
            chunk.format not in (8, 16, 32) or chunk.type == 0
            or chunk.items * (chunk.format // 8) != chunk.length
            or (self.type != 0 and (self.type != chunk.type or self.format != chunk.format))
            or chunk.items > 0xFFFFFFFF - self.items
        ):
            return False
        if not self.append_bytes(chunk.data if chunk.data is not None else b""):
            return False
        self.type = chunk.type
        self.format = chunk.format
        self.items += chunk.items
        return True


class _Xcb:
    def __init__(self) -> None:
        self.library = ctypes.CDLL("libxcb.so.1")
        self._libc = ctypes.CDLL(None)
        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype = None
        pointer, uint, integer = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
        byte, ushort, short = ctypes.c_uint8, ctypes.c_uint16, ctypes.c_int16
        signatures = {
            "xcb_connect": ([ctypes.c_char_p, ctypes.POINTER(integer)], pointer),
            "xcb_connection_has_error": ([pointer], integer),
            "xcb_disconnect": ([pointer], None),
            "xcb_get_setup": ([pointer], pointer),
            "xcb_setup_roots_iterator": ([pointer], _ScreenIterator),
            "xcb_screen_next": ([ctypes.POINTER(_ScreenIterator)], None),
            "xcb_generate_id": ([pointer], uint),
            "xcb_create_window": ([pointer, byte, uint, uint, short, short, ushort, ushort, ushort, ushort, uint, uint, pointer], _Cookie),
            "xcb_intern_atom": ([pointer, byte, ushort, ctypes.c_char_p], _Cookie),
            "xcb_flush": ([pointer], integer),
            "xcb_get_file_descriptor": ([pointer], integer),
            "xcb_poll_for_reply": ([pointer, ctypes.c_uint, ctypes.POINTER(pointer), ctypes.POINTER(pointer)], integer),
            "xcb_discard_reply": ([pointer, ctypes.c_uint], None),
            "xcb_poll_for_event": ([pointer], pointer),
            "xcb_get_property": ([pointer, byte, uint, uint, uint, uint, uint], _Cookie),
            "xcb_get_property_value": ([pointer], pointer),
            "xcb_delete_property": ([pointer, uint, uint], _Cookie),
            "xcb_convert_selection": ([pointer, uint, uint, uint, uint, uint], _Cookie),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result

    def free(self, pointer: int | None) -> None:
        self._libc.free(pointer)


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class _X11Clipboard:
    def __init__(self, xcb: _Xcb) -> None:
        self._xcb = xcb
        self._library = xcb.library
        self.connection: int | None = None
        self.window = 0
        self.clipboard = 0
        self.property = 0
        self.targets = 0
        self.incr = 0
        self.deadline = _monotonic_ms() + CLIPBOARD_TIMEOUT_MS

    def _wait_for_input(self) -> bool:
        while True:
            remaining = self.deadline - _monotonic_ms()
            if remaining <= 0 or self._library.xcb_connection_has_error(self.connection):
                return False
            descriptor = self._library.xcb_get_file_descriptor(self.connection)
            poller = select.poll()
            if descriptor >= 0:
                poller.register(descriptor, select.POLLIN)
            try:
                ready = poller.poll(remaining)
                return bool(ready and ready[0][1] & select.POLLIN)
            except OSError as error:
                if error.errno != errno.EINTR:
                    return False

    def _wait_for_reply(self, sequence: int) -> int | None:
        if self._library.xcb_flush(self.connection) <= 0:
            return None
        while _monotonic_ms() < self.deadline:
            reply = ctypes.c_void_p()
            error = ctypes.c_void_p()
            if self._library.xcb_poll_for_reply(self.connection, sequence, ctypes.byref(reply), ctypes.byref(error)):
                self._xcb.free(error.value)
                return reply.value
            if not self._wait_for_input():
                break
        self._library.xcb_discard_reply(self.connection, sequence)
        return None

    def _wait_for_event(self, type: int) -> bytes | None:
        if self._library.xcb_flush(self.connection) <= 0:
            return None
        while _monotonic_ms() < self.deadline:
            event = self._library.xcb_poll_for_event(self.connection)
            if event:
                try:
                    if ctypes.c_uint8.from_address(event).value & 0x7F == type:
                        return ctypes.string_at(event, 32)
                finally:
                    self._xcb.free(event)
            elif not self._wait_for_input():
                break
        return None

    def _read_property(self, remove: bool) -> _PropertyData | None:
        cookie = self._library.xcb_get_property(self.connection, remove, self.window, self.property, 0, 0, MAX_CLIPBOARD_BYTES // 4)
        reply = self._wait_for_reply(cookie.sequence)
        if not reply:
            return None
        try:
            _, format, _, reply_length, type, bytes_after, items = struct.unpack("=BBHIIII", ctypes.string_at(reply, 20))
            length = items * (format // 8)
            if bytes_after or not type or format not in (8, 16, 32) or length > MAX_CLIPBOARD_BYTES or length > reply_length * 4:
                return None
            result = _PropertyData(items=items, format=format, type=type)
            try:
                data = ctypes.string_at(self._library.xcb_get_property_value(reply), length)
            except MemoryError:
                return None
            return result if result.append_bytes(data) else None
        finally:
            self._xcb.free(reply)

    def _request_selection(self, target: int) -> _PropertyData | None:
        result = _PropertyData()
        self._library.xcb_delete_property(self.connection, self.window, self.property)
        self._library.xcb_convert_selection(self.connection, self.window, self.clipboard, target, self.property, 0)
        while True:
            event = self._wait_for_event(31)
            if event is None:
                return None
            selection, event_target, property = struct.unpack_from("=III", event, 12)
            if selection != self.clipboard or event_target != target:
                continue
            if not property:
                return result
            received = self._read_property(False)
            if received is None:
                return None
            result = received
            break
        if result.type != self.incr:
            return result
        result = _PropertyData()
        self._library.xcb_delete_property(self.connection, self.window, self.property)
        while True:
            event = self._wait_for_event(28)
            if event is None:
                return None
            atom = struct.unpack_from("=I", event, 8)[0]
            if atom != self.property or event[16] != 0:
                continue
            chunk = self._read_property(True)
            if chunk is None or not result.append_property(chunk):
                return None
            if chunk.items == 0:
                return result

    def _intern_atom(self, name: bytes) -> int:
        cookie = self._library.xcb_intern_atom(self.connection, False, len(name), name)
        reply = self._wait_for_reply(cookie.sequence)
        if not reply:
            return 0
        try:
            return ctypes.c_uint32.from_address(reply + 8).value
        finally:
            self._xcb.free(reply)

    def open(self) -> bool:
        screen_number = ctypes.c_int()
        self.connection = self._library.xcb_connect(None, ctypes.byref(screen_number))
        if not self.connection or self._library.xcb_connection_has_error(self.connection):
            return False
        screens = self._library.xcb_setup_roots_iterator(self._library.xcb_get_setup(self.connection))
        for _ in range(max(0, screen_number.value)):
            if not screens.rem:
                break
            self._library.xcb_screen_next(ctypes.byref(screens))
        if not screens.rem:
            return False
        root = ctypes.c_uint32.from_address(screens.data).value
        self.window = self._library.xcb_generate_id(self.connection)
        mask = ctypes.c_uint32(1 << 22)
        self._library.xcb_create_window(self.connection, 0, self.window, root, 0, 0, 1, 1, 0, 1, 0, 1 << 11, ctypes.byref(mask))
        self.clipboard = self._intern_atom(b"CLIPBOARD")
        self.property = self._intern_atom(b"PI_CLIPBOARD")
        self.targets = self._intern_atom(b"TARGETS")
        self.incr = self._intern_atom(b"INCR")
        return bool(self.clipboard and self.property and self.targets and self.incr)

    def close(self) -> None:
        if self.connection:
            self._library.xcb_disconnect(self.connection)

    def _preferred_target(self, image: bool) -> int | None:
        wanted: list[int] = []
        for type in _IMAGE_TYPES if image else _TEXT_TYPES:
            atom = self._intern_atom(type)
            if not atom:
                return None
            wanted.append(atom)
        targets = self._request_selection(self.targets)
        if targets is None:
            return None
        valid = targets.type == 4 and targets.format == 32 and targets.items == targets.length // 4 and targets.length % 4 == 0
        if valid:
            for atom in wanted:
                for offered, in struct.iter_unpack("=I", targets.data if targets.data is not None else b""):
                    if atom == offered:
                        return atom
        if not image and targets.type == 0:
            return self._intern_atom(b"UTF8_STRING") or None
        return 0 if valid or targets.type == 0 else None

    def read(self, image: bool) -> str | bytes | None:
        target = self._preferred_target(image)
        contents = self._request_selection(target) if target else _PropertyData()
        if target is None or contents is None:
            raise RuntimeError("Could not read X11 clipboard")
        if contents.data is None:
            return None
        data = bytes(contents.data)
        if image:
            return data
        return data.decode("latin-1" if contents.type == 31 else "utf-8", "replace")


class LinuxClipboard:
    set_text = None

    def __init__(self) -> None:
        self._xcb = _Xcb()
        self._changed = threading.Condition()
        self._busy = False
        self._waiting = False
        self._finished = False
        self._result: str | bytes | None | Undefined = UNDEFINED
        self._error: Exception | None = None

    async def get_text(self) -> ClipboardText:
        return cast(ClipboardText, await run_clipboard(lambda: self._execute(False)))

    async def get_image(self) -> ClipboardImage:
        return cast(ClipboardImage, await run_clipboard(lambda: self._execute(True)))

    def _read_on_private_thread(self, image: bool) -> None:
        result: str | bytes | None | Undefined = UNDEFINED
        error: Exception | None = None
        clipboard = _X11Clipboard(self._xcb)
        try:
            try:
                if clipboard.open():
                    result = clipboard.read(image)
            finally:
                clipboard.close()
        except Exception as caught:
            error = caught
        with self._changed:
            self._finished = True
            if self._waiting:
                self._result = result
                self._error = error
            else:
                # A timed-out caller no longer owns the late data or error.
                self._busy = False
            self._changed.notify()

    def _execute(self, image: bool) -> str | bytes | None | Undefined:
        with self._changed:
            if self._busy:
                return UNDEFINED
            self._busy = self._waiting = True
            self._finished = False
            self._result = UNDEFINED
            self._error = None
            deadline = time.monotonic() + 3
            try:
                worker = threading.Thread(target=self._read_on_private_thread, args=(image,), daemon=True, name="pi.clipboard.x11")
                try:
                    worker.start()
                except RuntimeError:
                    self._busy = False
                    return UNDEFINED
                while not self._finished:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._changed.wait(remaining):
                        break
                if self._finished:
                    self._busy = False
                    result, error = self._result, self._error
                    self._result = UNDEFINED
                    self._error = None
                    if error is not None:
                        raise error
                    return result
                return UNDEFINED
            finally:
                self._waiting = False
