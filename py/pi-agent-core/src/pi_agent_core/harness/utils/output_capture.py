"""Shell output capture ported from ``harness/utils/output-capture.ts``.

Maintains and publishes one bounded shell-output view. Writes received while
publication is rate-limited collapse into the latest view; small changes stay
responsive while complete window turnovers earn a proportionally longer delay.
The first update after idle and an explicit final flush are immediate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from ..._chord.context import Context
from ..types import (
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputMetadata,
    ShellOutputTruncation,
    ShellOutputUpdate,
    ShellOutputView,
)
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head, truncate_tail, utf8_byte_length
from .adaptive_publisher import AdaptivePublisher, AdaptivePublisherOptions

__all__ = [
    "OUTPUT_MIN_EMIT_INTERVAL_MS",
    "OUTPUT_TARGET_BYTES_PER_SECOND",
    "OutputCapture",
    "apply_shell_output_update",
    "sanitize_shell_output",
    "update_from",
]

OUTPUT_MIN_EMIT_INTERVAL_MS = 100
OUTPUT_TARGET_BYTES_PER_SECOND = 100 * 1024

_INVALID_SHELL_OUTPUT = list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20)) + [
    0xFFF9,
    0xFFFA,
    0xFFFB,
]
_INVALID_SHELL_OUTPUT_TABLE = {code: None for code in _INVALID_SHELL_OUTPUT}


def sanitize_shell_output(text: str) -> str:
    return text.translate(_INVALID_SHELL_OUTPUT_TABLE)


def _count_newlines(text: str) -> int:
    return text.count("\n")


def _trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8", errors="replace")


def _trim_to_first_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    end = max_bytes
    while end > 0 and (encoded[end] & 0xC0) == 0x80:
        end -= 1
    return encoded[:end].decode("utf-8", errors="replace")


def _suffix_prefix_overlap(before: str, after: str, scan: int) -> int:
    if not before or not after or scan == 0:
        return 0
    tail = before[len(before) - scan :] if len(before) > scan else before
    for probe_length in (min(64, len(after)), 1):
        probe = after[:probe_length]
        candidates = 0
        index = tail.find(probe)
        while index != -1:
            candidates += 1
            if candidates > 8:
                break
            overlap_length = len(tail) - index
            if overlap_length <= len(after) and tail[index:] == after[:overlap_length]:
                return overlap_length
            index = tail.find(probe, index + 1)
        if probe_length == 1:
            break
    return 0


def update_from(previous: Optional[ShellOutputView], current: ShellOutputView) -> ShellOutputUpdate:
    """Derive the minimal source-side update transforming ``previous`` into ``current``."""
    if previous is None:
        return ShellOutputUpdate(kind="replace", output=current)
    metadata = ShellOutputMetadata(
        truncation=current.truncation,
        spill_path=current.spill_path,
        last_line_bytes=current.last_line_bytes,
    )
    if current.text == previous.text:
        return ShellOutputUpdate(kind="metadata", metadata=metadata)
    if len(current.text) > len(previous.text) and current.text[: len(previous.text)] == previous.text:
        return ShellOutputUpdate(
            kind="append",
            text=current.text[len(previous.text) :],
            metadata=metadata,
        )
    shared = _suffix_prefix_overlap(
        previous.text,
        current.text,
        min(len(previous.text), len(current.text), current.truncation.max_bytes * 2),
    )
    if shared > 0:
        return ShellOutputUpdate(
            kind="slide",
            drop=len(previous.text) - shared,
            text=current.text[shared:],
            metadata=metadata,
        )
    return ShellOutputUpdate(kind="replace", output=current)


def apply_shell_output_update(
    current: Optional[ShellOutputView], update: ShellOutputUpdate
) -> ShellOutputView:
    """Fold one update into the running view."""
    if update.kind == "replace":
        return update.output
    if update.kind == "append":
        base = current.text if current is not None else ""
        view = ShellOutputView(
            truncation=update.metadata.truncation,
            spill_path=update.metadata.spill_path,
            last_line_bytes=update.metadata.last_line_bytes,
            text=f"{base}{update.text or ''}",
        )
        return view
    if update.kind == "slide":
        base = current.text if current is not None else ""
        view = ShellOutputView(
            truncation=update.metadata.truncation,
            spill_path=update.metadata.spill_path,
            last_line_bytes=update.metadata.last_line_bytes,
            text=f"{base[update.drop:]}{update.text or ''}",
        )
        return view
    # metadata
    base = current.text if current is not None else ""
    view = ShellOutputView(
        truncation=update.metadata.truncation,
        spill_path=update.metadata.spill_path,
        last_line_bytes=update.metadata.last_line_bytes,
        text=base,
    )
    return view


class OutputCapture:
    """Maintains and publishes one bounded shell-output view."""

    def __init__(
        self,
        options: Optional[ShellOutputCaptureOptions],
        context: Context,
        on_update: Optional[Callable[[ShellOutputUpdate, Context], None]],
        on_error: Optional[Callable[[Any], None]] = None,
    ) -> None:
        limits = (options or ShellOutputCaptureOptions()).limits or ShellOutputLimits()
        self._max_bytes = limits.max_bytes if limits.max_bytes is not None else DEFAULT_MAX_BYTES
        self._max_lines = limits.max_lines if limits.max_lines is not None else DEFAULT_MAX_LINES
        self._retain = limits.retain or "tail"
        self._context = context
        self._on_update = on_update
        self._on_error = on_error
        self._buffer = ""
        self._buffer_bytes = 0
        self._total_bytes = 0
        self._newlines = 0
        self._ends_with_newline = True
        self._current_line_bytes = 0
        self._spill_path: Optional[str] = None
        self._disposed = False
        self._publisher = AdaptivePublisher(
            AdaptivePublisherOptions(
                snapshot=self.snapshot,
                update=update_from,
                measure=lambda update: utf8_byte_length(json.dumps(_update_to_json(update))),
                publish=self._emit,
                on_error=self._report_error,
            )
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def truncated(self) -> bool:
        return self._total_bytes > self._max_bytes or self._total_lines() > self._max_lines

    def push(self, chunk: Any) -> None:
        if self._disposed:
            return
        if isinstance(chunk, str):
            self._append_text(chunk)
            return
        self._append_text(bytes(chunk).decode("utf-8", errors="replace"))

    def finish(self) -> None:
        if self._disposed:
            return
        self.flush()

    def set_spill_path(self, path: str) -> None:
        if self._disposed or self._spill_path == path:
            return
        self._spill_path = path
        self._publisher.mark_dirty()

    def snapshot(self) -> ShellOutputView:
        if self._retain == "head":
            retained = truncate_head(self._buffer, options=_limits_options(self._max_bytes, self._max_lines))
        else:
            retained = truncate_tail(self._buffer, options=_limits_options(self._max_bytes, self._max_lines))
        total_lines = self._total_lines()
        truncated = self.truncated
        truncation = ShellOutputTruncation(
            truncated=truncated,
            truncated_by=("lines" if total_lines > self._max_lines else "bytes") if truncated else None,
            total_bytes=self._total_bytes,
            total_lines=total_lines,
            output_lines=retained.output_lines,
            output_bytes=retained.output_bytes,
            last_line_partial=retained.last_line_partial,
            first_line_exceeds_limit=retained.first_line_exceeds_limit,
            max_lines=retained.max_lines,
            max_bytes=retained.max_bytes,
        )
        return ShellOutputView(
            truncation=truncation,
            spill_path=self._spill_path,
            last_line_bytes=self._current_line_bytes if retained.last_line_partial else None,
            text=sanitize_shell_output(retained.content),
        )

    def flush(self) -> None:
        """Publish pending changes immediately (adaptive interval still applies)."""
        self._publisher.flush()

    def dispose(self) -> None:
        self._disposed = True
        self._publisher.dispose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit(self, update: ShellOutputUpdate) -> None:
        if self._on_update is not None:
            self._on_update(update, self._context)

    def _report_error(self, error: BaseException) -> None:
        if self._on_error is not None:
            self._on_error(error)

    def _append_text(self, text: str) -> None:
        if text == "":
            return
        text_bytes = utf8_byte_length(text)
        self._total_bytes += text_bytes
        self._newlines += _count_newlines(text)
        self._ends_with_newline = text.endswith("\n")
        last_newline = text.rfind("\n")
        self._current_line_bytes = (
            self._current_line_bytes + text_bytes
            if last_newline == -1
            else utf8_byte_length(text[last_newline + 1 :])
        )
        self._buffer += text
        self._buffer_bytes += text_bytes

        guard = self._max_bytes * 2
        if self._buffer_bytes > guard * 2:
            self._buffer = (
                _trim_to_last_utf8_bytes(self._buffer, guard)
                if self._retain == "tail"
                else _trim_to_first_utf8_bytes(self._buffer, guard)
            )
            self._buffer_bytes = utf8_byte_length(self._buffer)
        self._publisher.mark_dirty()

    def _total_lines(self) -> int:
        return self._newlines + (0 if (self._ends_with_newline or self._total_bytes == 0) else 1)


def _limits_options(max_bytes: int, max_lines: int):
    from .truncate import TruncationOptions

    return TruncationOptions(max_lines=max_lines, max_bytes=max_bytes)


def _update_to_json(update: ShellOutputUpdate) -> dict:
    data: dict = {"kind": update.kind}
    if update.output is not None:
        data["output"] = update.output.text
    if update.text is not None:
        data["text"] = update.text
    if update.drop:
        data["drop"] = update.drop
    if update.metadata is not None:
        data["metadata"] = update.metadata.truncation.to_json()
    return data
