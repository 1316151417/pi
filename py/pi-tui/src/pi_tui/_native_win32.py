"""Windows clipboard, console mode, and physical modifier-state operations."""

from __future__ import annotations

import ctypes
import struct
import time
from typing import cast

from ._native_common import ClipboardImage, ClipboardText, ModifierKey, modifier_name, run_clipboard


def _dib_pixel_offset(dib: bytes) -> int:
    size = len(dib)
    if size < 12:
        return 0
    header_size = int.from_bytes(dib[:4], "little")
    if header_size == 12:
        bits_per_pixel = int.from_bytes(dib[10:12], "little")
        color_count = 1 << bits_per_pixel if bits_per_pixel <= 8 else 0
        offset = header_size + color_count * 3
        return offset if offset <= size else 0
    if header_size < 40 or header_size > size:
        return 0
    bits_per_pixel = int.from_bytes(dib[14:16], "little")
    compression = int.from_bytes(dib[16:20], "little")
    color_count = int.from_bytes(dib[32:36], "little")
    if not color_count and bits_per_pixel <= 8:
        color_count = 1 << bits_per_pixel
    offset = header_size + color_count * 4
    if header_size == 40 and compression == 3:
        offset += 12
    if header_size == 40 and compression == 6:
        offset += 16
    return offset if offset <= size else 0


class WindowsPlatform:
    def __init__(self) -> None:
        # WinDLL is only exposed by ctypes on Windows; defer access until load.
        library = getattr(ctypes, "WinDLL")
        self._kernel = library("kernel32", use_last_error=True)
        self._user = library("user32", use_last_error=True)
        pointer, uint, integer = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        kernel_signatures = {
            "GetStdHandle": ([uint], pointer),
            "GetConsoleMode": ([pointer, ctypes.POINTER(uint)], integer),
            "SetConsoleMode": ([pointer, uint], integer),
            "GetModuleHandleA": ([ctypes.c_char_p], pointer),
            "GlobalAlloc": ([uint, ctypes.c_size_t], pointer),
            "GlobalLock": ([pointer], pointer),
            "GlobalUnlock": ([pointer], integer),
            "GlobalSize": ([pointer], ctypes.c_size_t),
            "GlobalFree": ([pointer], pointer),
        }
        user_signatures = {
            "GetAsyncKeyState": ([integer], ctypes.c_int16),
            "OpenClipboard": ([pointer], integer),
            "CloseClipboard": ([], integer),
            "EmptyClipboard": ([], integer),
            "GetClipboardData": ([uint], pointer),
            "SetClipboardData": ([uint, pointer], pointer),
            "IsClipboardFormatAvailable": ([uint], integer),
            "RegisterClipboardFormatW": ([ctypes.c_wchar_p], uint),
            "CreateWindowExW": ([uint, ctypes.c_wchar_p, ctypes.c_wchar_p, uint, integer, integer, integer, integer, pointer, pointer, pointer, pointer], pointer),
            "DestroyWindow": ([pointer], integer),
        }
        for module, signatures in ((self._kernel, kernel_signatures), (self._user, user_signatures)):
            for name, (arguments, result) in signatures.items():
                function = getattr(module, name)
                function.argtypes = arguments
                function.restype = result

    def enable_virtual_terminal_input(self) -> bool:
        handle = self._kernel.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        return bool(handle != ctypes.c_void_p(-1).value and self._kernel.GetConsoleMode(handle, ctypes.byref(mode)) and self._kernel.SetConsoleMode(handle, mode.value | 0x200))

    def is_modifier_pressed(self, name: ModifierKey) -> bool:
        key = modifier_name(name)
        if key == "shift":
            codes = (0x10, 0xA0, 0xA1)
        elif key == "control":
            codes = (0x11, 0xA2, 0xA3)
        elif key in ("option", "alt"):
            codes = (0x12, 0xA4, 0xA5)
        elif key in ("command", "super", "win"):
            codes = (0x5B, 0x5C)
        else:
            return False
        return any(self._user.GetAsyncKeyState(code) & 0x8000 for code in codes)

    async def get_text(self) -> ClipboardText:
        return cast(ClipboardText, await run_clipboard(lambda: self._execute("text")))

    async def get_image(self) -> ClipboardImage:
        return cast(ClipboardImage, await run_clipboard(lambda: self._execute("image")))

    async def set_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise RuntimeError("setText requires a string")
        encoded = text.encode("utf-16-le", "surrogatepass") + b"\0\0"
        await run_clipboard(lambda: self._execute("write", encoded))

    def _execute(self, operation: str, encoded: bytes = b"") -> str | bytes | None:
        owner = None
        opened = False
        try:
            if operation == "write":
                owner = self._user.CreateWindowExW(0, "STATIC", None, 0, 0, 0, 0, 0, ctypes.c_void_p(-3), None, self._kernel.GetModuleHandleA(None), None)
                if not owner:
                    raise RuntimeError("Could not create clipboard owner window")
            for _ in range(10):
                if self._user.OpenClipboard(owner):
                    opened = True
                    break
                time.sleep(0.005)
            if not opened:
                raise RuntimeError("Could not open clipboard")
            if operation == "write":
                handle = self._kernel.GlobalAlloc(2, len(encoded))
                pointer = self._kernel.GlobalLock(handle) if handle else None
                success = False
                if pointer:
                    ctypes.memmove(pointer, encoded, len(encoded))
                    self._kernel.GlobalUnlock(handle)
                    success = bool(self._user.EmptyClipboard() and self._user.SetClipboardData(13, handle))
                if not success:
                    if handle:
                        self._kernel.GlobalFree(handle)
                    raise RuntimeError("Could not set clipboard text")
                # SetClipboardData transfers ownership of the allocation.
                return None
            if operation == "text":
                if not self._user.IsClipboardFormatAvailable(13):
                    return None
                handle = self._user.GetClipboardData(13)
                pointer = self._kernel.GlobalLock(handle) if handle else None
                if not pointer:
                    raise RuntimeError("Clipboard does not contain text")
                try:
                    capacity = self._kernel.GlobalSize(handle) // 2
                    units = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint16))
                    length = 0
                    while length < capacity and units[length]:
                        length += 1
                    return ctypes.string_at(pointer, length * 2).decode("utf-16-le", "surrogatepass")
                finally:
                    self._kernel.GlobalUnlock(handle)
            return self._read_image()
        finally:
            if opened:
                self._user.CloseClipboard()
            if owner:
                self._user.DestroyWindow(owner)

    def _read_image(self) -> bytes | None:
        png_format = self._user.RegisterClipboardFormatW("PNG")
        if not (png_format and self._user.IsClipboardFormatAvailable(png_format)) and not self._user.IsClipboardFormatAvailable(17) and not self._user.IsClipboardFormatAvailable(8):
            return None
        handle = self._user.GetClipboardData(png_format) if png_format and self._user.IsClipboardFormatAvailable(png_format) else None
        is_png = bool(handle)
        if not handle and self._user.IsClipboardFormatAvailable(17):
            handle = self._user.GetClipboardData(17)
        if not handle and self._user.IsClipboardFormatAvailable(8):
            handle = self._user.GetClipboardData(8)
        pointer = self._kernel.GlobalLock(handle) if handle else None
        size = self._kernel.GlobalSize(handle) if handle else 0
        try:
            if not pointer or not size:
                raise RuntimeError("Clipboard does not contain an image")
            source = ctypes.string_at(pointer, size)
            if is_png:
                return source
            offset = _dib_pixel_offset(source)
            if not offset or size > 0xFFFFFFFF - 14:
                raise RuntimeError("Could not create clipboard image buffer")
            try:
                return struct.pack("<2sIHHI", b"BM", size + 14, 0, 0, offset + 14) + source
            except MemoryError as error:
                raise RuntimeError("Could not create clipboard image buffer") from error
        finally:
            if pointer:
                self._kernel.GlobalUnlock(handle)
