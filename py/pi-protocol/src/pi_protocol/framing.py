"""Incremental four-byte big-endian length framing from ``framing.ts``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NoReturn

FRAME_HEADER_LENGTH = 4
MAX_UINT32 = 0xFFFFFFFF
PAYLOAD_BLOCK_SIZE = 64 * 1024
DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024
type ByteInput = bytes | bytearray | memoryview

__all__ = ["DEFAULT_MAX_FRAME_LENGTH", "FrameDecoderOptions", "FrameError", "encode_frame", "FrameDecoder"]


@dataclass(frozen=True)
class FrameDecoderOptions:
    max_frame_length: int | None = None


class FrameError(Exception):
    name = "FrameError"


def _byte_view(value: object, message: str) -> memoryview:
    if isinstance(value, (bytes, bytearray)):
        return memoryview(value)
    if isinstance(value, memoryview) and value.c_contiguous and value.itemsize == 1 and value.format in ("B", "b"):
        return value.cast("B")
    raise TypeError(message)


def encode_frame(payload: ByteInput) -> bytes:
    view = _byte_view(payload, "Frame payload must be a Uint8Array")
    if view.nbytes > MAX_UINT32:
        raise ValueError("Frame payload exceeds the unsigned 32-bit length limit")
    return view.nbytes.to_bytes(FRAME_HEADER_LENGTH, "big") + bytes(view)


class FrameDecoder:
    def __init__(self, options: FrameDecoderOptions | None = None) -> None:
        value = options.max_frame_length if options is not None and options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        if type(value) not in (int, float) or value < 0 or value > MAX_UINT32 or not math.isfinite(value) or int(value) != value:
            raise ValueError(f"maxFrameLength must be an integer between 0 and {MAX_UINT32}")
        self._max_frame_length = int(value)
        self._header = bytearray(FRAME_HEADER_LENGTH)
        self._header_length = 0
        self._payload_blocks: list[bytearray] = []
        self._current_payload_block: bytearray | None = None
        self._current_payload_block_length = 0
        self._expected_payload_length: int | None = None
        self._payload_length = 0
        self._state: Literal["open", "ended", "failed"] = "open"

    def push(self, chunk: ByteInput) -> list[bytes]:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        view = _byte_view(chunk, "Frame chunk must be a Uint8Array")
        frames: list[bytes] = []
        chunk_offset = 0
        while chunk_offset < view.nbytes:
            if self._expected_payload_length is None:
                header_bytes = min(FRAME_HEADER_LENGTH - self._header_length, view.nbytes - chunk_offset)
                self._header[self._header_length:self._header_length + header_bytes] = view[chunk_offset:chunk_offset + header_bytes]
                self._header_length += header_bytes
                chunk_offset += header_bytes
                if self._header_length < FRAME_HEADER_LENGTH:
                    continue
                frame_length = int.from_bytes(self._header, "big")
                self._header_length = 0
                if frame_length > self._max_frame_length:
                    self._fail(f"Frame length {frame_length} exceeds configured limit of {self._max_frame_length}")
                if frame_length == 0:
                    frames.append(b"")
                    continue
                self._expected_payload_length = frame_length
                self._payload_blocks = []
                self._current_payload_block = None
                self._current_payload_block_length = 0
                self._payload_length = 0
            expected_payload_length = self._expected_payload_length
            if expected_payload_length is None:
                continue
            while chunk_offset < view.nbytes and self._payload_length < expected_payload_length:
                block = self._current_payload_block
                if block is None or self._current_payload_block_length == len(block):
                    block = bytearray(min(PAYLOAD_BLOCK_SIZE, expected_payload_length - self._payload_length))
                    self._payload_blocks.append(block)
                    self._current_payload_block = block
                    self._current_payload_block_length = 0
                payload_bytes = min(len(block) - self._current_payload_block_length, view.nbytes - chunk_offset)
                offset = self._current_payload_block_length
                block[offset:offset + payload_bytes] = view[chunk_offset:chunk_offset + payload_bytes]
                self._current_payload_block_length += payload_bytes
                self._payload_length += payload_bytes
                chunk_offset += payload_bytes
            if self._payload_length == expected_payload_length:
                frames.append(b"".join(self._payload_blocks))
                self._payload_blocks = []
                self._current_payload_block = None
                self._current_payload_block_length = 0
                self._expected_payload_length = None
                self._payload_length = 0
        return frames

    def end(self) -> None:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if self._header_length != 0 or self._expected_payload_length is not None:
            self._fail("Truncated frame at end of stream")
        self._state = "ended"

    def _fail(self, message: str) -> NoReturn:
        self._state = "failed"
        self._header_length = 0
        self._payload_blocks = []
        self._current_payload_block = None
        self._current_payload_block_length = 0
        self._expected_payload_length = None
        self._payload_length = 0
        raise FrameError(message)
