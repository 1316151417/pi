"""Native esbuild 0.28.2 service client for Chord's non-plugin build requests.

The wire protocol follows esbuild lib/shared/stdio_protocol.ts (MIT, Evan Wallace;
see THIRD_PARTY_LICENSES.md). No Node process or JavaScript wrapper is used.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .._undefined import UNDEFINED
from .bundle_loader import _object_keys, _utf8

ESBUILD_VERSION = "0.28.2"
type Value = None | bool | int | str | bytes | list[Value] | dict[str, Value]


@dataclass
class _Packet:
    id: int
    is_request: bool
    value: Value


def _encode_packet(packet: _Packet) -> bytes:
    body = bytearray(struct.pack("<I", (packet.id << 1) | int(not packet.is_request)))

    def visit(value: Value) -> None:
        if value is None:
            body.append(0)
        elif isinstance(value, bool):
            body.extend((1, int(value)))
        elif isinstance(value, int):
            body.append(2)
            body.extend(struct.pack("<I", value & 0xFFFFFFFF))
        elif isinstance(value, (str, bytes)):
            encoded = _utf8(value) if isinstance(value, str) else value
            body.append(3 if isinstance(value, str) else 4)
            body.extend(struct.pack("<I", len(encoded)))
            body.extend(encoded)
        elif isinstance(value, list):
            body.append(5)
            body.extend(struct.pack("<I", len(value)))
            for item in value:
                visit(item)
        else:
            body.append(6)
            body.extend(struct.pack("<I", len(value)))
            for key in _object_keys(value):
                encoded = _utf8(key)
                body.extend(struct.pack("<I", len(encoded)))
                body.extend(encoded)
                visit(value[key])

    visit(packet.value)
    return struct.pack("<I", len(body)) + body


def _decode_packet(data: bytes) -> _Packet:
    offset = 0

    def read(count: int) -> bytes:
        nonlocal offset
        if count < 0 or offset + count > len(data):
            raise ValueError("Invalid packet")
        value = data[offset:offset + count]
        offset += count
        return value

    def read32() -> int:
        return struct.unpack("<I", read(4))[0]

    def visit() -> Value:
        tag = read(1)[0]
        if tag == 0:
            return None
        if tag == 1:
            return bool(read(1)[0])
        if tag == 2:
            return read32()
        if tag == 3:
            return read(read32()).decode("utf-8-sig", errors="replace")
        if tag == 4:
            return read(read32())
        if tag == 5:
            return [visit() for _ in range(read32())]
        if tag == 6:
            result: dict[str, Value] = {}
            for _ in range(read32()):
                key = read(read32()).decode("utf-8-sig", errors="replace")
                result[key] = visit()
            return result
        raise ValueError("Invalid packet")

    encoded_id = read32()
    value = visit()
    if offset != len(data):
        raise ValueError("Invalid packet")
    return _Packet(encoded_id >> 1, not bool(encoded_id & 1), value)


class EsbuildFailure(RuntimeError):
    def __init__(self, errors: list[dict[str, object]], warnings: list[dict[str, object]]) -> None:
        self.errors = errors
        self.warnings = warnings
        message = "Build failed"
        if errors:
            message += f" with {len(errors)} error{'s' if len(errors) > 1 else ''}:"
            for index, error in enumerate(errors[:6]):
                if index == 5:
                    message += "\n..."
                    break
                location = error.get("location")
                if not isinstance(location, dict):
                    message += f"\nerror: {error.get('text', '')}"
                else:
                    plugin = error.get("pluginName")
                    prefix = f"[plugin: {plugin}] " if plugin else ""
                    message += f"\n{location['file']}:{location['line']}:{location['column']}: ERROR: {prefix}{error.get('text', '')}"
        super().__init__(message)


def _find_binary() -> str:
    configured = os.environ.get("ESBUILD_BINARY_PATH")
    if configured:
        return configured
    system = {"Darwin": "darwin", "Windows": "win32", "Linux": "linux", "FreeBSD": "freebsd", "OpenBSD": "openbsd", "NetBSD": "netbsd", "SunOS": "sunos", "AIX": "aix"}.get(platform.system())
    arch = {"aarch64": "arm64", "arm64": "arm64", "AMD64": "x64", "x86_64": "x64", "i386": "ia32", "i686": "ia32", "armv7l": "arm", "ppc64le": "ppc64", "ppc64": "ppc64", "riscv64": "riscv64", "s390x": "s390x", "loongarch64": "loong64", "mips64el": "mips64el"}.get(platform.machine(), platform.machine())
    executable = "esbuild.exe" if os.name == "nt" else "bin/esbuild"
    if system:
        seen: set[Path] = set()
        for start in (Path.cwd(), Path(__file__).resolve().parent):
            for directory in (start, *start.parents):
                if directory in seen:
                    continue
                seen.add(directory)
                candidate = directory / "node_modules" / "@esbuild" / f"{system}-{arch}" / executable
                if candidate.is_file():
                    return str(candidate)
    binary = shutil.which("esbuild")
    if binary:
        return binary
    raise FileNotFoundError(f"esbuild {ESBUILD_VERSION} native executable not found; set ESBUILD_BINARY_PATH")


class EsbuildService:
    """A sequential, scoped native service; only the used build API is exposed."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0

    async def _read_frame(self) -> bytes:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("The service is no longer running")
        try:
            header = await self._process.stdout.readexactly(4)
            return await self._process.stdout.readexactly(struct.unpack("<I", header)[0])
        except asyncio.IncompleteReadError as error:
            raise RuntimeError("The service was stopped") from error

    async def build(self, source: str, flags: list[str], working_directory: str) -> dict[str, object] | None:
        if self._process is None:
            self._process = await asyncio.create_subprocess_exec(
                _find_binary(), f"--service={ESBUILD_VERSION}", "--ping",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            )
            version = (await self._read_frame()).decode("latin-1")
            if version != ESBUILD_VERSION:
                raise RuntimeError(f'Cannot start service: Host version "{ESBUILD_VERSION}" does not match binary version {json.dumps(version)}')
        writer = self._process.stdin
        if writer is None:
            raise RuntimeError("The service is no longer running")
        id = self._next_id
        self._next_id += 1
        writer.write(_encode_packet(_Packet(id, True, {
            "command": "build", "key": id, "entries": [["", source]], "flags": cast(list[Value], flags),
            "write": True, "stdinContents": None, "stdinResolveDir": None,
            "absWorkingDir": working_directory, "nodePaths": [], "context": False,
        })))
        await writer.drain()
        while True:
            packet = _decode_packet(await self._read_frame())
            if not isinstance(packet.value, dict):
                raise RuntimeError("Invalid esbuild response")
            if packet.is_request:
                if packet.value.get("command") == "ping":
                    response: Value = {}
                else:
                    response = {"errors": [{"text": f"Invalid command: {packet.value.get('command')}"}]}
                writer.write(_encode_packet(_Packet(packet.id, False, response)))
                await writer.drain()
                continue
            if packet.id != id:
                raise RuntimeError("Unexpected esbuild response ID")
            if packet.value.get("error"):
                raise RuntimeError(str(packet.value["error"]))
            errors = cast(list[dict[str, object]], packet.value.get("errors", []))
            warnings = cast(list[dict[str, object]], packet.value.get("warnings", []))
            for message in (*errors, *warnings):
                message["detail"] = UNDEFINED
            if errors:
                raise EsbuildFailure(errors, warnings)
            metadata = packet.value.get("metafile")
            if not isinstance(metadata, bytes) or not metadata:
                return None
            return cast(dict[str, object], json.loads(metadata.decode("utf-8-sig", errors="replace")))

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
