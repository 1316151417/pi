"""Injectable process streams backed by real POSIX or Windows terminal I/O."""

from __future__ import annotations

import asyncio
import codecs
import ctypes
import os
import select
import signal
import sys
import threading
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, TextIO

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty


class ProcessStdin(Protocol):
    is_raw: bool
    set_raw_mode: Callable[[bool], None] | None

    def set_encoding(self, encoding: str) -> None: ...
    def resume(self) -> None: ...
    def pause(self) -> None: ...
    def on(self, event: Literal["data"], callback: Callable[[str], None]) -> object: ...
    def remove_listener(self, event: Literal["data"], callback: Callable[[str], None]) -> object: ...


class ProcessStdout(Protocol):
    @property
    def columns(self) -> int | None: ...
    @property
    def rows(self) -> int | None: ...

    def write(self, data: str) -> object: ...
    def on(self, event: Literal["resize"], callback: Callable[[], None]) -> object: ...
    def remove_listener(self, event: Literal["resize"], callback: Callable[[], None]) -> object: ...


class ProcessIO(Protocol):
    stdin: ProcessStdin
    stdout: ProcessStdout
    env: Mapping[str, str]
    platform: str
    pid: int

    def refresh_dimensions(self) -> None: ...


def utf8_bytes(value: str) -> bytes:
    return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace").encode("utf-8")


class _WindowsInput:
    def __init__(self, descriptor: int) -> None:
        self.handle = msvcrt.get_osfhandle(descriptor)
        self.library = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        pointer, uint, integer = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        signatures = {
            "GetConsoleMode": ([pointer, ctypes.POINTER(uint)], integer),
            "SetConsoleMode": ([pointer, uint], integer),
            "ReadConsoleW": ([pointer, pointer, uint, ctypes.POINTER(uint), pointer], integer),
            "OpenThread": ([uint, integer, uint], pointer),
            "CancelSynchronousIo": ([pointer], integer),
            "CloseHandle": ([pointer], integer),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result
        mode = ctypes.c_uint32()
        self.is_console = bool(self.library.GetConsoleMode(self.handle, ctypes.byref(mode)))
        self.original_mode = mode.value
        self._decoder = codecs.getincrementaldecoder("utf-16-le")("replace")

    def set_raw(self, active: bool) -> None:
        if not self.is_console:
            return
        if active:
            mode = ctypes.c_uint32()
            if not self.library.GetConsoleMode(self.handle, ctypes.byref(mode)):
                raise OSError("Could not read console input mode")
            # libuv raw input removes echo, line input and processed Ctrl+C.
            mode.value = (mode.value & ~(0x1 | 0x2 | 0x4 | 0x200)) | 0x8
            target = mode.value
        else:
            target = self.original_mode
        if not self.library.SetConsoleMode(self.handle, target):
            raise OSError("Could not configure console input mode")

    def read(self) -> str:
        buffer = ctypes.create_unicode_buffer(4096)
        count = ctypes.c_uint32()
        if not self.library.ReadConsoleW(self.handle, buffer, 4096, ctypes.byref(count), None):
            raise OSError("Could not read console input")
        return self._decoder.decode(buffer[:count.value].encode("utf-16-le", "surrogatepass"))

    def cancel(self, thread: threading.Thread) -> None:
        if thread.native_id is None:
            return
        handle = self.library.OpenThread(1, False, thread.native_id)
        if handle:
            try:
                self.library.CancelSynchronousIo(handle)
            finally:
                self.library.CloseHandle(handle)


class _RealStdin:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._descriptor = stream.fileno()
        self.is_raw = False
        self._saved_attributes: list[object] | None = None
        self._windows = _WindowsInput(self._descriptor) if os.name == "nt" else None
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._listeners: list[Callable[[str], None]] = []
        self._resumed = False
        self._ended = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader_registered = False
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._wake: tuple[int, int] | None = None
        self._generation = 0
        self._lock = threading.RLock()

    def set_raw_mode(self, active: bool) -> None:
        if not os.isatty(self._descriptor):
            return
        if self._windows is not None:
            self._windows.set_raw(active)
        elif active:
            if not self.is_raw:
                self._saved_attributes = termios.tcgetattr(self._descriptor)
            tty.setraw(self._descriptor, termios.TCSANOW)
        elif self._saved_attributes is not None:
            termios.tcsetattr(self._descriptor, termios.TCSANOW, self._saved_attributes)
        self.is_raw = active

    def set_encoding(self, encoding: str) -> None:
        if encoding.lower().replace("-", "") != "utf8":
            raise ValueError("Terminal input requires UTF-8")

    def on(self, event: Literal["data"], callback: Callable[[str], None]) -> _RealStdin:
        if event != "data":
            raise ValueError(f"Unsupported stdin event: {event}")
        with self._lock:
            self._listeners.append(callback)
            self._ensure_reader()
        return self

    def remove_listener(self, event: Literal["data"], callback: Callable[[str], None]) -> _RealStdin:
        with self._lock:
            for index in range(len(self._listeners) - 1, -1, -1):
                if self._listeners[index] is callback:
                    del self._listeners[index]
                    break
        return self

    def resume(self) -> None:
        with self._lock:
            self._resumed = True
            self._ensure_reader()

    def _ensure_reader(self) -> None:
        if not self._resumed or self._ended or not self._listeners or self._reader_registered or self._thread is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        if self._windows is None and self._loop is not None:
            try:
                self._loop.add_reader(self._descriptor, self._read_ready, self._generation)
                self._reader_registered = True
                return
            except (OSError, NotImplementedError):
                pass
        stopped = threading.Event()
        self._stop = stopped
        if self._windows is None:
            self._wake = os.pipe()
        self._thread = threading.Thread(target=self._read_on_thread, args=(stopped, self._generation, self._wake), daemon=True, name="pi.stdin")
        self._thread.start()

    def _deliver(self, data: str, generation: int) -> None:
        if not data:
            return
        with self._lock:
            if generation != self._generation or not self._resumed:
                return
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(data)

    def _dispatch(self, data: str, generation: int) -> None:
        if not data:
            return
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._deliver, data, generation)
        else:
            self._deliver(data, generation)

    def _read_ready(self, generation: int) -> None:
        if generation != self._generation or not self._resumed:
            return
        try:
            data = os.read(self._descriptor, 65536)
        except (BlockingIOError, InterruptedError):
            return
        if data:
            self._deliver(self._decoder.decode(data), generation)
        else:
            self._deliver(self._decoder.decode(b"", final=True), generation)
            self._ended = True
            if self._loop is not None:
                self._loop.remove_reader(self._descriptor)
            self._reader_registered = False

    def _read_on_thread(self, stopped: threading.Event, generation: int, wake: tuple[int, int] | None) -> None:
        try:
            while not stopped.is_set():
                if self._windows is not None and self._windows.is_console:
                    try:
                        text = self._windows.read()
                    except OSError:
                        if stopped.is_set():
                            return
                        raise
                    if not stopped.is_set():
                        self._dispatch(text, generation)
                    continue
                if wake is not None:
                    try:
                        ready, _, _ = select.select([self._descriptor, wake[0]], [], [])
                    except InterruptedError:
                        continue
                    if stopped.is_set() or wake[0] in ready:
                        return
                try:
                    data = os.read(self._descriptor, 65536)
                except (InterruptedError, BlockingIOError):
                    continue
                except OSError:
                    if stopped.is_set():
                        return
                    raise
                if stopped.is_set():
                    return
                if not data:
                    self._dispatch(self._decoder.decode(b"", final=True), generation)
                    self._ended = True
                    return
                self._dispatch(self._decoder.decode(data), generation)
        finally:
            with self._lock:
                if generation == self._generation:
                    self._thread = None
                    self._wake = None
                if wake is not None:
                    os.close(wake[0])
                    os.close(wake[1])

    def pause(self) -> None:
        with self._lock:
            self._resumed = False
            self._generation += 1
            if self._reader_registered and self._loop is not None:
                self._loop.remove_reader(self._descriptor)
                self._reader_registered = False
            thread, stopped, wake = self._thread, self._stop, self._wake
            self._thread = None
            self._stop = None
            self._wake = None
            if stopped is not None:
                stopped.set()
            if wake is not None:
                try:
                    os.write(wake[1], b"\0")
                except OSError:
                    pass
            if thread is not None and self._windows is not None:
                self._windows.cancel(thread)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.25)


class _RealStdout:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._listeners: list[Callable[[], None]] = []
        self._write_lock = threading.RLock()
        self._last_size = self._size()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._previous_signal: object = None
        self._signal_installed = False
        self._poll_stop: threading.Event | None = None

    def _size(self) -> os.terminal_size | None:
        try:
            return os.get_terminal_size(self._stream.fileno())
        except (OSError, ValueError, AttributeError):
            return None

    @property
    def columns(self) -> int | None:
        size = self._size()
        return size.columns if size is not None else None

    @property
    def rows(self) -> int | None:
        size = self._size()
        return size.lines if size is not None else None

    def write(self, data: str) -> object:
        with self._write_lock:
            binary = getattr(self._stream, "buffer", None)
            if binary is None:
                result = self._stream.write(data)
            else:
                result = binary.write(utf8_bytes(data))
            self._stream.flush()
            return result

    def _resized(self) -> None:
        size = self._size()
        if size == self._last_size:
            return
        self._last_size = size
        for callback in tuple(self._listeners):
            callback()

    def _on_signal(self, signum: int, frame: object) -> None:
        if callable(self._previous_signal):
            self._previous_signal(signum, frame)
        self._resized()

    def _poll_size(self, stopped: threading.Event) -> None:
        while not stopped.wait(0.05):
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._resized)
            else:
                self._resized()

    def on(self, event: Literal["resize"], callback: Callable[[], None]) -> _RealStdout:
        if event != "resize":
            raise ValueError(f"Unsupported stdout event: {event}")
        self._listeners.append(callback)
        if len(self._listeners) != 1:
            return self
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        if self._signal_installed:
            return self
        if os.name != "nt":
            try:
                self._previous_signal = signal.signal(signal.SIGWINCH, self._on_signal)
                self._signal_installed = True
                return self
            except ValueError:
                pass
        stopped = threading.Event()
        self._poll_stop = stopped
        threading.Thread(target=self._poll_size, args=(stopped,), daemon=True, name="pi.resize").start()
        return self

    def remove_listener(self, event: Literal["resize"], callback: Callable[[], None]) -> _RealStdout:
        for index in range(len(self._listeners) - 1, -1, -1):
            if self._listeners[index] is callback:
                del self._listeners[index]
                break
        if not self._listeners:
            if self._signal_installed:
                try:
                    signal.signal(signal.SIGWINCH, self._previous_signal)
                except ValueError:
                    # A synchronous input callback may stop the terminal from
                    # its reader thread. Python permits signal changes only
                    # on the main thread; the installed handler now has no
                    # terminal listeners and still forwards the prior signal.
                    pass
                else:
                    self._signal_installed = False
            if self._poll_stop is not None:
                self._poll_stop.set()
                self._poll_stop = None
        return self


class RealProcessIO:
    def __init__(self) -> None:
        self.stdin = _RealStdin(sys.stdin)
        self.stdout = _RealStdout(sys.stdout)
        self.env = os.environ
        self.platform = sys.platform
        self.pid = os.getpid()

    def refresh_dimensions(self) -> None:
        if self.platform == "win32" or self.pid <= 0:
            return
        try:
            os.kill(self.pid, signal.SIGWINCH)
        except OSError:
            pass


_process_io: RealProcessIO | None = None


def get_process_io() -> RealProcessIO:
    global _process_io
    if _process_io is None:
        _process_io = RealProcessIO()
    return _process_io
