"""Native asyncio HTTP listener for the local OAuth callback routes."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.utils import formatdate
from http import HTTPStatus


@dataclass
class CallbackRequest:
    method: str
    target: str


class CallbackResponse:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self.sent = False
        self.head_request = False

    def send(self, status: int, body: str, *, content_type: str = "text/html; charset=utf-8", no_store: bool = False) -> None:
        if self.sent:
            raise RuntimeError("Cannot write headers after they are sent")
        self.sent = True
        payload = body.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace").encode("utf-8")
        headers = [
            f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n",
            f"Content-Type: {content_type}\r\n",
            f"Content-Length: {len(payload)}\r\n",
            f"Date: {formatdate(usegmt=True)}\r\n",
            "Connection: close\r\n",
        ]
        if no_store:
            headers.append("Cache-Control: no-store\r\n")
        self._writer.write("".join(headers).encode("ascii") + b"\r\n" + (b"" if self.head_request else payload))


class CallbackServer:
    def __init__(self, handler: Callable[[CallbackRequest, CallbackResponse], Awaitable[None]]) -> None:
        self._handler = handler
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()

    @classmethod
    async def listen(cls, host: str, port: int, handler: Callable[[CallbackRequest, CallbackResponse], Awaitable[None]]) -> CallbackServer:
        instance = cls(handler)
        addresses = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        family, kind, protocol, _, address = addresses[0]
        listener = socket.socket(family, kind, protocol)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(address)
            listener.setblocking(False)
            # One resolved address and one port, matching Node's server.listen;
            # asyncio's host overload can bind separate ephemeral IPv4/IPv6 ports.
            instance._server = await asyncio.start_server(instance._accept, sock=listener, limit=16384)
        except BaseException:
            listener.close()
            raise
        return instance

    @property
    def port(self) -> int | None:
        if self._server is None or not self._server.sockets:
            return None
        return self._server.sockets[0].getsockname()[1]

    def close(self) -> None:
        if self._server is not None:
            self._server.close()

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._serve(reader, writer))
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        response = CallbackResponse(writer)
        try:
            try:
                request_line = await reader.readline()
                if not request_line:
                    return
                parts = request_line.rstrip(b"\r\n").split(b" ")
                if len(parts) != 3 or not parts[2].startswith(b"HTTP/1."):
                    response.send(400, "Bad Request", content_type="text/plain; charset=utf-8")
                    return
                header_size = len(request_line)
                while True:
                    line = await reader.readline()
                    header_size += len(line)
                    if header_size > 16384:
                        response.send(431, "Request Header Fields Too Large", content_type="text/plain; charset=utf-8")
                        return
                    if line in (b"\r\n", b"\n"):
                        break
                    if not line or b":" not in line:
                        response.send(400, "Bad Request", content_type="text/plain; charset=utf-8")
                        return
                request = CallbackRequest(parts[0].decode("ascii"), parts[1].decode("latin-1"))
                response.head_request = request.method == "HEAD"
            except (ValueError, UnicodeError):
                response.send(400, "Bad Request", content_type="text/plain; charset=utf-8")
                return
            await self._handler(request, response)
        except Exception as error:
            asyncio.get_running_loop().call_exception_handler({
                "message": "Exception handling local OAuth callback", "exception": error,
            })
        finally:
            try:
                await writer.drain()
            except (ConnectionError, OSError):
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
