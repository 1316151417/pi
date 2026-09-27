"""Unix-domain socket listener and Server preset from ``transports/unix``."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import math
import os
import stat
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pi_protocol import DEFAULT_MAX_FRAME_LENGTH, is_server_id

from .connection import ByteConnectionHandler
from .listener import ByteConnectionAcceptor
from .server import Server
from .types import ServerHost, ServerOptions, SessionMetadata

DEFAULT_SOCKET_MODE = 0o600
DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS = 5_000
MAX_TIMER_DELAY_MS = 2_147_483_647


def _safe_integer(value: object, maximum: int) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and int(value) == value and 0 < value <= maximum)


@dataclass
class UnixListenerOptions:
    path: str
    mode: int | None = None
    max_pending_bytes: int | None = None
    graceful_close_timeout_ms: int | None = None
    max_frame_length: int | None = None
    on_error: Callable[[Exception], None] | None = None


@dataclass
class UnixServerOptions(UnixListenerOptions):
    server_id: str = ""
    handshake_timeout_ms: int | None = None
    on_connection_count_changed: Callable[[int], None] | None = None


class UnixByteConnection:
    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        graceful_close_timeout_ms: int, max_pending_bytes: int,
        remove: Callable[[UnixByteConnection], None],
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._graceful_close_timeout_ms = graceful_close_timeout_ms
        self._max_pending_bytes = max_pending_bytes
        self._remove = remove
        self._pending_bytes = 0
        self._closed = False
        self._closing = False
        self._write_tail: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._write_tail.set_result(None)
        self._close_promise: asyncio.Task[None] | None = None
        self._handler: ByteConnectionHandler | None = None
        self._read_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self, accept: ByteConnectionAcceptor) -> None:
        self._handler = accept(self)
        self._read_task = asyncio.get_running_loop().create_task(self._read())

    async def _read(self) -> None:
        handler = self._handler
        if handler is None:
            return
        try:
            while not self._closed:
                chunk = await self._reader.read(64 * 1024)
                if not chunk:
                    break
                handler.on_data(chunk)
        except Exception as error:
            if not self._closed:
                handler.on_error(error)
        finally:
            self._writer.close()
            self._mark_closed()
            handler.on_close()

    def send(self, chunk: bytes) -> asyncio.Future[None]:
        loop = asyncio.get_running_loop()
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            return _rejected(TypeError("Unix connection chunks must be Uint8Array"))
        if self._closed or self._closing:
            return _rejected(RuntimeError("Unix connection is closed"))
        if self._pending_bytes + len(chunk) > self._max_pending_bytes:
            return _rejected(RuntimeError("Unix connection exceeded its pending byte limit"))
        copied = bytes(chunk)
        self._pending_bytes += len(copied)
        previous = self._write_tail

        async def write() -> None:
            try:
                await previous
                if self._closed or self._writer.is_closing():
                    raise RuntimeError("Unix connection is closed")
                self._writer.write(copied)
                await self._writer.drain()
            finally:
                self._pending_bytes -= len(copied)

        result = loop.create_task(write())

        async def settle() -> None:
            try:
                await result
            except Exception:
                pass

        self._write_tail = loop.create_task(settle())
        return result

    def close(self, final_chunk: bytes | None = None) -> asyncio.Task[None]:
        if self._close_promise is not None:
            return self._close_promise
        self._closing = True

        async def close_internal() -> None:
            try:
                await self._write_tail
                if not self._writer.is_closing():
                    if final_chunk is not None:
                        self._writer.write(bytes(final_chunk))
                        await self._writer.drain()
                    self._writer.close()
                await asyncio.wait_for(
                    self._writer.wait_closed(), self._graceful_close_timeout_ms / 1000,
                )
            except (Exception, asyncio.TimeoutError):
                self._writer.close()
            finally:
                self._mark_closed()

        self._close_promise = asyncio.get_running_loop().create_task(close_internal())
        return self._close_promise

    def _mark_closed(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closing = True
        self._remove(self)


def _rejected(error: Exception) -> asyncio.Future[None]:
    result: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    result.set_exception(error)
    return result


class UnixListener:
    def __init__(self, options: UnixListenerOptions) -> None:
        if not options.path:
            raise TypeError("Server Unix socket path must not be empty")
        mode = options.mode if options.mode is not None else DEFAULT_SOCKET_MODE
        if type(mode) is not int or mode < 0 or mode > 0o777:
            raise TypeError("Server Unix socket mode must be an integer between 0 and 0o777")
        frame = options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        if not _safe_integer(frame, 0xFFFFFFFF):
            raise TypeError("Server maxFrameLength must be an integer between 1 and 4294967295")
        pending = options.max_pending_bytes if options.max_pending_bytes is not None else int(frame) * 4
        if not _safe_integer(pending, 2**53 - 1) or pending < frame + 4:
            raise TypeError("Server maxPendingBytes must be a safe integer at least maxFrameLength + 4")
        timeout = options.graceful_close_timeout_ms if options.graceful_close_timeout_ms is not None else DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS
        if not _safe_integer(timeout, MAX_TIMER_DELAY_MS):
            raise TypeError(f"Server gracefulCloseTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")
        self._path = options.path
        self._mode = mode
        self._pending = int(pending)
        self._timeout = int(timeout)
        self._on_error = options.on_error
        self._server: asyncio.AbstractServer | None = None
        self._connections: set[UnixByteConnection] = set()
        self._identity: tuple[int, int] | None = None
        self._owned_bind_path: str | None = None
        self._closing = False
        self._close_promise: asyncio.Task[None] | None = None

    async def start(self, accept: ByteConnectionAcceptor) -> None:
        if self._server is not None:
            raise RuntimeError("Unix listener is already started")
        if self._closing:
            raise RuntimeError("Unix listener is closing or closed")
        if sys.platform == "win32":
            raise RuntimeError("Unix listener is not supported on Windows")
        path = Path(self._path)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True, mode=0o700)
        bind_path = str(path.parent / f"bind-{hashlib.sha256(self._path.encode()).hexdigest()[:8]}")
        await _remove_stale_socket(self._path)
        await _remove_stale_socket(bind_path)
        self._owned_bind_path = bind_path

        async def receive(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            if self._closing:
                writer.close()
                return
            connection = UnixByteConnection(reader, writer, self._timeout, self._pending, self._connections.discard)
            self._connections.add(connection)
            connection.start(accept)

        try:
            self._server = await asyncio.start_unix_server(receive, path=bind_path)
            original = await asyncio.to_thread(os.lstat, bind_path)
            if not stat.S_ISSOCK(original.st_mode):
                raise RuntimeError(f"Unix listener path is not a socket after binding: {bind_path}")
            self._identity = (original.st_dev, original.st_ino)
            await asyncio.to_thread(os.link, bind_path, self._path)
            await asyncio.to_thread(os.chmod, self._path, self._mode)
            await _remove_path(bind_path)
            self._owned_bind_path = None
        except Exception:
            await self._close_internal()
            raise

    def close(self) -> asyncio.Task[None]:
        if self._close_promise is None:
            self._closing = True
            self._close_promise = asyncio.get_running_loop().create_task(self._close_internal())
        return self._close_promise

    async def _close_internal(self) -> None:
        server = self._server
        if server is not None:
            server.close()
        await asyncio.gather(*(item.close() for item in list(self._connections)))
        if server is not None:
            await server.wait_closed()
        if self._owned_bind_path is not None:
            await _remove_path(self._owned_bind_path)
            self._owned_bind_path = None
        await self._cleanup_owned_socket()
        self._server = None

    async def _cleanup_owned_socket(self) -> None:
        identity = self._identity
        self._identity = None
        if identity is None:
            return
        try:
            current = await asyncio.to_thread(os.lstat, self._path)
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            return
        preserved = str(Path(self._path).parent / f"cleanup-{uuid.uuid4().hex[:6]}")
        await asyncio.to_thread(os.rename, self._path, preserved)
        moved = await asyncio.to_thread(os.lstat, preserved)
        if stat.S_ISSOCK(moved.st_mode) and (moved.st_dev, moved.st_ino) == identity:
            await _remove_path(preserved)
            return
        if not Path(self._path).exists():
            await asyncio.to_thread(os.rename, preserved, self._path)
        raise RuntimeError(f"Unix listener path changed during cleanup; preserved replacement at {preserved}")


async def _remove_stale_socket(path: str) -> None:
    try:
        original = await asyncio.to_thread(os.lstat, path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(original.st_mode):
        raise RuntimeError(f"Refusing to remove non-socket Unix listener path: {path}")
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), 1)
    except OSError as error:
        if error.errno not in (errno.ECONNREFUSED, errno.ENOENT, errno.EPIPE, errno.ECONNRESET):
            raise
    except asyncio.TimeoutError:
        raise RuntimeError(f"Unix listener is already running: {path}") from None
    else:
        writer.close()
        await writer.wait_closed()
        raise RuntimeError(f"Unix listener is already running: {path}")
    preserved = str(Path(path).parent / f"stale-{uuid.uuid4().hex[:6]}")
    try:
        await asyncio.to_thread(os.rename, path, preserved)
    except FileNotFoundError:
        return
    moved = await asyncio.to_thread(os.lstat, preserved)
    if not stat.S_ISSOCK(moved.st_mode) or (moved.st_dev, moved.st_ino) != (original.st_dev, original.st_ino):
        if not Path(path).exists():
            await asyncio.to_thread(os.rename, preserved, path)
        raise RuntimeError(f"Unix listener path changed while checking for a stale socket: {path}")
    await _remove_path(preserved)


async def _remove_path(path: str) -> None:
    try:
        await asyncio.to_thread(os.unlink, path)
    except FileNotFoundError:
        pass


def get_unix_socket_path(server_id: str, server_directory: str) -> str:
    if not is_server_id(server_id):
        raise TypeError("Unix serverId must be a canonical lowercase UUIDv4")
    return str(Path(server_directory) / f"{server_id}.sock")


def create_unix_listener(options: UnixListenerOptions) -> UnixListener:
    return UnixListener(options)


def create_unix_server[TMetadata: SessionMetadata](
    host: ServerHost[TMetadata], options: UnixServerOptions,
) -> Server[TMetadata]:
    listener = create_unix_listener(UnixListenerOptions(
        path=options.path, mode=options.mode, max_pending_bytes=options.max_pending_bytes,
        graceful_close_timeout_ms=options.graceful_close_timeout_ms,
        max_frame_length=options.max_frame_length, on_error=options.on_error,
    ))
    return Server(host, ServerOptions(
        listeners=[listener], server_id=options.server_id,
        max_frame_length=options.max_frame_length,
        handshake_timeout_ms=options.handshake_timeout_ms,
        on_connection_count_changed=options.on_connection_count_changed,
        on_error=options.on_error,
    ))


__all__ = [
    "UnixListenerOptions", "UnixServerOptions", "UnixByteConnection", "UnixListener",
    "create_unix_listener", "create_unix_server", "get_unix_socket_path",
]
