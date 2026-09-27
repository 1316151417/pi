"""Bounded streaming shell output from ``core/tools/output-accumulator.ts``."""

from __future__ import annotations

import codecs
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pi_agent_core.harness.utils.truncate import (
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationOptions,
    TruncationResult, truncate_tail,
)


@dataclass
class OutputAccumulatorOptions:
    max_lines: int = DEFAULT_MAX_LINES
    max_bytes: int = DEFAULT_MAX_BYTES
    temp_file_prefix: str = "pi-output"


@dataclass
class OutputSnapshot:
    content: str
    truncation: TruncationResult
    full_output_path: str | None = None


class OutputAccumulator:
    def __init__(self, options: OutputAccumulatorOptions | None = None) -> None:
        chosen = options or OutputAccumulatorOptions()
        self.max_lines = chosen.max_lines
        self.max_bytes = chosen.max_bytes
        self.max_rolling_bytes = max(self.max_bytes * 2, 1)
        self.temp_file_prefix = chosen.temp_file_prefix
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._raw_chunks: list[bytes] = []
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_raw_bytes = 0
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._total_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False
        self._temp_file_path: str | None = None
        self._temp_file_stream: BinaryIO | None = None

    def append(self, data: bytes) -> None:
        if self._finished:
            raise RuntimeError("Cannot append to a finished output accumulator")
        self._total_raw_bytes += len(data)
        self._append_decoded_text(self._decoder.decode(data, final=False))
        if self._temp_file_stream is not None or self._should_use_temp_file():
            self._ensure_temp_file()
            assert self._temp_file_stream is not None
            self._temp_file_stream.write(data)
        elif data:
            self._raw_chunks.append(data)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._append_decoded_text(self._decoder.decode(b"", final=True))
        if self._should_use_temp_file():
            self._ensure_temp_file()

    def snapshot(self, *, persist_if_truncated: bool = False) -> OutputSnapshot:
        tail = truncate_tail(
            self._snapshot_text(),
            TruncationOptions(max_lines=self.max_lines, max_bytes=self.max_bytes),
        )
        truncated = self._total_lines > self.max_lines or self._total_decoded_bytes > self.max_bytes
        truncated_by = (
            tail.truncated_by or ("bytes" if self._total_decoded_bytes > self.max_bytes else "lines")
            if truncated else None
        )
        result = TruncationResult(
            content=tail.content,
            truncated=truncated,
            truncated_by=truncated_by,
            total_lines=self._total_lines,
            total_bytes=self._total_decoded_bytes,
            output_lines=tail.output_lines,
            output_bytes=tail.output_bytes,
            last_line_partial=tail.last_line_partial,
            first_line_exceeds_limit=tail.first_line_exceeds_limit,
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
        )
        if persist_if_truncated and result.truncated:
            self._ensure_temp_file()
        return OutputSnapshot(result.content, result, self._temp_file_path)

    def close_temp_file(self) -> None:
        if self._temp_file_stream is not None:
            stream = self._temp_file_stream
            self._temp_file_stream = None
            stream.close()

    def get_last_line_bytes(self) -> int:
        return self._current_line_bytes

    def _append_decoded_text(self, text: str) -> None:
        if not text:
            return
        byte_count = len(text.encode("utf-8"))
        self._total_decoded_bytes += byte_count
        self._tail_text += text
        self._tail_bytes += byte_count
        if self._tail_bytes > self.max_rolling_bytes * 2:
            self._trim_tail()
        newlines = text.count("\n")
        if not newlines:
            self._current_line_bytes += byte_count
            self._has_open_line = True
        else:
            self._completed_lines += newlines
            remainder = text[text.rfind("\n") + 1:]
            self._current_line_bytes = len(remainder.encode("utf-8"))
            self._has_open_line = bool(remainder)
        self._total_lines = self._completed_lines + int(self._has_open_line)

    def _trim_tail(self) -> None:
        data = self._tail_text.encode("utf-8")
        if len(data) <= self.max_rolling_bytes:
            self._tail_bytes = len(data)
            return
        start = len(data) - self.max_rolling_bytes
        while start < len(data) and data[start] & 0xC0 == 0x80:
            start += 1
        self._tail_starts_at_line_boundary = (
            self._tail_starts_at_line_boundary if start == 0 else data[start - 1] == 0x0A
        )
        self._tail_text = data[start:].decode("utf-8")
        self._tail_bytes = len(data[start:])

    def _snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text
        newline = self._tail_text.find("\n")
        return self._tail_text if newline == -1 else self._tail_text[newline + 1:]

    def _should_use_temp_file(self) -> bool:
        return (
            self._total_raw_bytes > self.max_bytes
            or self._total_decoded_bytes > self.max_bytes
            or self._total_lines > self.max_lines
        )

    def _ensure_temp_file(self) -> None:
        if self._temp_file_path is not None:
            return
        name = f"{self.temp_file_prefix}-{secrets.token_hex(8)}.log"
        path = Path(tempfile.gettempdir()) / name
        stream = path.open("xb")
        for chunk in self._raw_chunks:
            stream.write(chunk)
        self._raw_chunks.clear()
        self._temp_file_path = os.fspath(path)
        self._temp_file_stream = stream


__all__ = ["OutputAccumulatorOptions", "OutputSnapshot", "OutputAccumulator"]
