"""AppKit clipboard and CoreGraphics modifier access through the Objective-C ABI."""

from __future__ import annotations

import ctypes
from typing import cast

from ._native_common import ClipboardImage, ClipboardText, ModifierKey, modifier_name, run_clipboard

type _NativeType = type[ctypes.c_void_p] | type[ctypes.c_char_p] | type[ctypes.c_ulong] | type[ctypes.c_bool]


class DarwinPlatform:
    enable_virtual_terminal_input = None

    def __init__(self) -> None:
        self._foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        self._appkit = ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
        self._graphics = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        self._objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self._objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self._objc.objc_getClass.restype = ctypes.c_void_p
        self._objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self._objc.sel_registerName.restype = ctypes.c_void_p
        self._message_address = ctypes.cast(self._objc.objc_msgSend, ctypes.c_void_p).value
        self._graphics.CGEventSourceFlagsState.argtypes = [ctypes.c_int32]
        self._graphics.CGEventSourceFlagsState.restype = ctypes.c_uint64
        self._string_type = ctypes.c_void_p.in_dll(self._appkit, "NSPasteboardTypeString").value
        self._png_type = ctypes.c_void_p.in_dll(self._appkit, "NSPasteboardTypePNG").value
        self._tiff_type = ctypes.c_void_p.in_dll(self._appkit, "NSPasteboardTypeTIFF").value

    def _class(self, name: bytes) -> int:
        result = self._objc.objc_getClass(name)
        if not result:
            raise RuntimeError(f"Objective-C class unavailable: {name.decode()}")
        return int(result)

    def _send(
        self,
        receiver: int | None,
        selector: bytes,
        result_type: _NativeType | None = ctypes.c_void_p,
        argument_types: tuple[_NativeType, ...] = (),
        arguments: tuple[object, ...] = (),
    ) -> int | None:
        # Each signature has its own function wrapper, so concurrent clipboard
        # workers never mutate the argtypes of the shared objc_msgSend symbol.
        function = ctypes.CFUNCTYPE(result_type, ctypes.c_void_p, ctypes.c_void_p, *argument_types)(self._message_address)
        return cast(int | None, function(receiver, self._objc.sel_registerName(selector), *arguments))

    def is_modifier_pressed(self, name: ModifierKey) -> bool:
        mask = {"shift": 1 << 17, "command": 1 << 20, "control": 1 << 18, "option": 1 << 19}.get(modifier_name(name), 0)
        return bool(mask and self._graphics.CGEventSourceFlagsState(0) & mask)

    async def get_text(self) -> ClipboardText:
        return cast(ClipboardText, await run_clipboard(lambda: self._execute("text")))

    async def get_image(self) -> ClipboardImage:
        return cast(ClipboardImage, await run_clipboard(lambda: self._execute("image")))

    async def set_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise RuntimeError("setText requires a string")
        # napi_get_value_string_utf8 replaces unpaired UTF-16 surrogates.
        encoded = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace").encode("utf-8")
        await run_clipboard(lambda: self._execute("write", encoded))

    def _execute(self, operation: str, encoded: bytes = b"") -> str | bytes | None:
        pool = self._send(self._send(self._class(b"NSAutoreleasePool"), b"alloc"), b"init")
        owned: list[int] = []
        try:
            pasteboard = self._send(self._class(b"NSPasteboard"), b"generalPasteboard")
            if not pasteboard:
                raise RuntimeError("Could not open clipboard")
            if operation == "write":
                text = self._send(
                    self._send(self._class(b"NSString"), b"alloc"), b"initWithBytes:length:encoding:",
                    argument_types=(ctypes.c_char_p, ctypes.c_ulong, ctypes.c_ulong),
                    arguments=(encoded, len(encoded), 4),
                )
                if not text:
                    raise RuntimeError("Clipboard text is not valid UTF-8")
                owned.append(text)
                self._send(pasteboard, b"clearContents", ctypes.c_ulong)
                if not self._send(pasteboard, b"setString:forType:", ctypes.c_bool, (ctypes.c_void_p, ctypes.c_void_p), (text, self._string_type)):
                    raise RuntimeError("Could not set clipboard text")
                return None
            if operation == "text":
                text = self._send(pasteboard, b"stringForType:", argument_types=(ctypes.c_void_p,), arguments=(self._string_type,))
                if not text:
                    return None
                pointer = self._send(text, b"UTF8String")
                if not pointer:
                    raise RuntimeError("Could not encode clipboard text")
                length = self._send(text, b"lengthOfBytesUsingEncoding:", ctypes.c_ulong, (ctypes.c_ulong,), (4,))
                return ctypes.string_at(pointer, length or 0).decode("utf-8", "replace")
            types = (ctypes.c_void_p * 2)(self._png_type, self._tiff_type)
            types_array = self._send(
                self._class(b"NSArray"), b"arrayWithObjects:count:",
                argument_types=(ctypes.c_void_p, ctypes.c_ulong), arguments=(ctypes.cast(types, ctypes.c_void_p), 2),
            )
            if not self._send(pasteboard, b"availableTypeFromArray:", argument_types=(ctypes.c_void_p,), arguments=(types_array,)):
                return None
            png = self._send(pasteboard, b"dataForType:", argument_types=(ctypes.c_void_p,), arguments=(self._png_type,))
            if not png:
                image = self._send(self._send(self._class(b"NSImage"), b"alloc"), b"initWithPasteboard:", argument_types=(ctypes.c_void_p,), arguments=(pasteboard,))
                if image:
                    owned.append(image)
                tiff = self._send(image, b"TIFFRepresentation")
                bitmap = self._send(self._class(b"NSBitmapImageRep"), b"imageRepWithData:", argument_types=(ctypes.c_void_p,), arguments=(tiff,)) if tiff else None
                if bitmap:
                    properties = self._send(self._class(b"NSDictionary"), b"dictionary")
                    png = self._send(bitmap, b"representationUsingType:properties:", argument_types=(ctypes.c_ulong, ctypes.c_void_p), arguments=(4, properties))
            if not png:
                raise RuntimeError("Clipboard does not contain an image")
            pointer = self._send(png, b"bytes")
            length = self._send(png, b"length", ctypes.c_ulong)
            return ctypes.string_at(pointer, length or 0)
        finally:
            for value in reversed(owned):
                self._send(value, b"release", None)
            self._send(pool, b"drain", None)
