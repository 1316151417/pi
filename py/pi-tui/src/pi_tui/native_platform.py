"""Platform-native clipboard and input helpers, with cached library loading.

Python extensions use ``darwin_platform``, ``win32_platform`` or
``linux_platform_x11`` plus a suffix from ``EXTENSION_SUFFIXES`` in the source's
packaging directories. The bundled ctypes implementations need no extension
binary and provide the same OS operations when no packaged extension exists.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import platform
import struct
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from ._native_common import UNDEFINED, ClipboardImage, ClipboardText, ModifierKey, Undefined
from ._native_darwin import DarwinPlatform
from ._native_linux import LinuxClipboard
from ._native_win32 import WindowsPlatform
from .native_module_path import get_native_module_candidates


class NativeClipboard(Protocol):
    # UNDEFINED is unavailable; None is an available clipboard with no value.
    async def get_text(self) -> ClipboardText: ...

    async def get_image(self) -> ClipboardImage: ...

    set_text: Callable[[str], Awaitable[None]] | None


class NativePlatformHelper(NativeClipboard, Protocol):
    enable_virtual_terminal_input: Callable[[], bool] | None
    is_modifier_pressed: Callable[[ModifierKey], bool] | None


_helpers: dict[str, NativeClipboard | Undefined] = {}
_helpers_lock = threading.RLock()


def _load_native_platform_helper(platform_name: str, suffix: str = "") -> NativeClipboard | Undefined:
    machine = platform.machine().lower()
    architecture = "x64" if machine in ("x86_64", "amd64") else "arm64" if machine in ("arm64", "aarch64") else machine
    if architecture not in ("x64", "arm64") or struct.calcsize("P") != 8:
        return UNDEFINED
    name = f"{platform_name}_platform{suffix.replace('-', '_')}"
    directory = os.path.join("native", platform_name, "prebuilds", f"{platform_name}-{architecture}")
    cache_key = os.path.join(directory, name)
    with _helpers_lock:
        if cache_key in _helpers:
            return _helpers[cache_key]
        for extension in importlib.machinery.EXTENSION_SUFFIXES:
            for module_path in get_native_module_candidates(os.path.join(directory, name + extension)):
                try:
                    specification = importlib.util.spec_from_file_location(name, module_path)
                    if specification is None or specification.loader is None:
                        continue
                    module = importlib.util.module_from_spec(specification)
                    specification.loader.exec_module(module)
                    if callable(getattr(module, "get_text", None)) and callable(getattr(module, "get_image", None)):
                        helper = cast(NativeClipboard, module)
                        _helpers[cache_key] = helper
                        return helper
                except Exception:
                    # Try the next Python packaging location.
                    pass
        try:
            if platform_name == "darwin":
                backend = DarwinPlatform()
            elif platform_name == "win32":
                backend = WindowsPlatform()
            elif platform_name == "linux":
                backend = LinuxClipboard()
            else:
                _helpers[cache_key] = UNDEFINED
                return UNDEFINED
            helper = cast(NativeClipboard, backend)
            _helpers[cache_key] = helper
            return helper
        except Exception:
            # Cache a missing native library, never a disconnected display.
            _helpers[cache_key] = UNDEFINED
            return UNDEFINED


def get_native_platform_helper() -> NativePlatformHelper | Undefined:
    if sys.platform not in ("darwin", "win32"):
        return UNDEFINED
    return cast(NativePlatformHelper | Undefined, _load_native_platform_helper(sys.platform))


def get_native_clipboard() -> NativeClipboard | Undefined:
    if sys.platform != "linux":
        return get_native_platform_helper()
    if not os.environ.get("DISPLAY"):
        return UNDEFINED
    return _load_native_platform_helper("linux", "-x11")


__all__ = [
    "UNDEFINED", "Undefined", "ModifierKey", "NativeClipboard", "NativePlatformHelper",
    "get_native_platform_helper", "get_native_clipboard",
]
