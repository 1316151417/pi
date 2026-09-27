"""Unix socket byte transport and local server discovery from ``unix.ts``."""

from __future__ import annotations

import asyncio
import errno
import math
import os
import stat
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from pi_protocol import DEFAULT_MAX_FRAME_LENGTH, ProtocolValidationError, is_server_id

from .client import Client
from .errors import DisconnectedError, ServerError
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers
from .types import ClientOptions

DEFAULT_DISCOVERY_TIMEOUT_MS = 1_000
MAX_TIMER_DELAY_MS = 2_147_483_647
UNIX_SOCKET_SUFFIX = ".sock"
MAX_CONCURRENT_DISCOVERY_PROBES = 16


@dataclass
class UnixTransportOptions:
    path: str
    max_pending_bytes: int | float | None = None


@dataclass(frozen=True)
class UnixServerRoute:
    server_id: str
    path: str


@dataclass
class DiscoverUnixServersOptions:
    directory: str
    timeout_ms: int | float | None = None


class UnixDiscoveryTimeoutError(Exception):
    pass


def _safe_integer(value: object, maximum: int | None = None) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and value == int(value)
            and 0 < value <= (maximum if maximum is not None else 2**53 - 1))


def _validate_options(options: UnixTransportOptions) -> int:
    if not options.path:
        raise TypeError("Unix transport path must not be empty")
    maximum = options.max_pending_bytes if options.max_pending_bytes is not None else DEFAULT_MAX_FRAME_LENGTH * 4
    if not _safe_integer(maximum):
        raise TypeError("Unix transport maxPendingBytes must be a positive safe integer")
    if sys.platform == "win32":
        raise RuntimeError("Unix transport is not supported on Windows")
    return int(maximum)


class _UnixByteTransport:
    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        max_pending_bytes: int, handlers: ByteTransportHandlers,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._max_pending_bytes = max_pending_bytes
        self._handlers = handlers
        self._closed = False
        self._terminal = False
        self._pending_bytes = 0
        loop = asyncio.get_running_loop()
        self._write_tail: asyncio.Future[None] = loop.create_future()
        self._write_tail.set_result(None)
        self._reader_task = loop.create_task(self._read())

    async def _read(self) -> None:
        try:
            while not self._terminal:
                chunk = await self._reader.read(64 * 1024)
                if not chunk:
                    break
                self._handlers.on_data(chunk)
        except Exception as error:
            if not self._terminal:
                self._terminal = True
                self._writer.close()
                self._handlers.on_error(error)
            return
        if not self._terminal:
            self._terminal = True
            self._writer.close()
            self._handlers.on_close()

    def send(self, chunk: bytes) -> asyncio.Future[None]:
        loop = asyncio.get_running_loop()
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            return _rejected_future(TypeError("Unix transport chunks must be Uint8Array"))
        if self._closed:
            return _rejected_future(RuntimeError("Unix transport is closed"))
        if self._pending_bytes + len(chunk) > self._max_pending_bytes:
            return _rejected_future(RuntimeError("Unix transport exceeded its pending byte limit"))
        self._pending_bytes += len(chunk)
        copied = bytes(chunk)
        previous = self._write_tail

        async def write() -> None:
            try:
                await previous
                if self._closed or self._writer.is_closing():
                    raise RuntimeError("Unix transport is closed")
                self._writer.write(copied)
                try:
                    await self._writer.drain()
                except (BrokenPipeError, ConnectionResetError) as error:
                    raise RuntimeError("Unix transport closed during write") from error
            finally:
                self._pending_bytes -= len(copied)

        tracked = loop.create_task(write())

        async def settle_tail() -> None:
            try:
                await tracked
            except Exception:
                pass

        self._write_tail = loop.create_task(settle_tail())
        return tracked

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminal = True
        self._writer.close()


def _rejected_future(error: Exception) -> asyncio.Future[None]:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_exception(error)
    return future


async def _connect_unix_socket(
    path: str, max_pending_bytes: int, handlers: ByteTransportHandlers,
    on_socket: Callable[[asyncio.StreamWriter], None] | None = None,
) -> ByteTransport:
    reader, writer = await asyncio.open_unix_connection(path)
    if on_socket is not None:
        on_socket(writer)
    return _UnixByteTransport(reader, writer, max_pending_bytes, handlers)


def create_unix_transport_factory(options: UnixTransportOptions) -> ByteTransportFactory:
    max_pending_bytes = _validate_options(options)
    return lambda handlers: _connect_unix_socket(options.path, max_pending_bytes, handlers)


async def _probe_unix_server(route: UnixServerRoute, timeout_ms: int) -> UnixServerRoute | None:
    max_pending_bytes = _validate_options(UnixTransportOptions(route.path))
    writer: asyncio.StreamWriter | None = None

    def on_socket(created: asyncio.StreamWriter) -> None:
        nonlocal writer
        writer = created

    client = Client(ClientOptions(
        server_id=route.server_id,
        transport_factory=lambda handlers: _connect_unix_socket(
            route.path, max_pending_bytes, handlers, on_socket,
        ),
    ))
    handshake = client.connect()
    try:
        await asyncio.wait_for(asyncio.shield(handshake), timeout_ms / 1000)
        return route
    except Exception as error:
        if isinstance(error, (asyncio.TimeoutError, ProtocolValidationError)):
            return None
        if isinstance(error, DisconnectedError) and error.__cause__ is None:
            return None
        if isinstance(error, ServerError) and error.code == "version":
            return None
        if _is_error_code(error, (errno.ENOENT, errno.ECONNREFUSED, errno.ECONNRESET, errno.EPIPE, errno.ETIMEDOUT)):
            return None
        raise
    finally:
        if not handshake.done():
            handshake.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
        await client.dispose()
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


def _is_error_code(error: BaseException, codes: tuple[int, ...]) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in codes:
            return True
        current = current.__cause__
    return False


async def discover_unix_servers(options: DiscoverUnixServersOptions) -> list[UnixServerRoute]:
    if sys.platform == "win32":
        raise RuntimeError("Unix transport is not supported on Windows")
    timeout_ms = options.timeout_ms if options.timeout_ms is not None else DEFAULT_DISCOVERY_TIMEOUT_MS
    if not _safe_integer(timeout_ms, MAX_TIMER_DELAY_MS):
        raise TypeError(f"Unix discovery timeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")
    try:
        names = await asyncio.to_thread(os.listdir, options.directory)
    except FileNotFoundError:
        return []
    candidates = [
        UnixServerRoute(name[:-len(UNIX_SOCKET_SUFFIX)], str(Path(options.directory) / name))
        for name in names
        if name.endswith(UNIX_SOCKET_SUFFIX) and is_server_id(name[:-len(UNIX_SOCKET_SUFFIX)])
    ]
    routes: list[UnixServerRoute] = []
    next_index = 0
    failure: BaseException | None = None

    async def worker() -> None:
        nonlocal next_index, failure
        while failure is None and next_index < len(candidates):
            candidate = candidates[next_index]
            next_index += 1
            try:
                try:
                    mode = await asyncio.to_thread(os.lstat, candidate.path)
                except FileNotFoundError:
                    continue
                if not stat.S_ISSOCK(mode.st_mode):
                    continue
                reachable = await _probe_unix_server(candidate, int(timeout_ms))
                if reachable is not None:
                    routes.append(reachable)
            except BaseException as error:
                if failure is None:
                    failure = error

    await asyncio.gather(*(worker() for _ in range(min(MAX_CONCURRENT_DISCOVERY_PROBES, len(candidates)))))
    if failure is not None:
        raise failure
    return sorted(routes, key=lambda route: route.server_id)


__all__ = [
    "UnixTransportOptions", "UnixServerRoute", "DiscoverUnixServersOptions",
    "create_unix_transport_factory", "discover_unix_servers",
]
