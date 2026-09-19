"""Truncation utilities ported from ``harness/utils/truncate.ts``.

Truncation is based on two independent limits — whichever is hit first wins:
- Line limit (default: 2000 lines)
- Byte limit (default: 50KB)

Never returns partial lines (except tail truncation edge case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "GREP_MAX_LINE_LENGTH",
    "TruncationResult",
    "TruncationOptions",
    "utf8_byte_length",
    "format_size",
    "truncate_head",
    "truncate_tail",
    "truncate_line",
]

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500  # Max chars per grep match line


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: Optional[str]  # "lines" | "bytes" | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


@dataclass
class TruncationOptions:
    max_lines: Optional[int] = None
    max_bytes: Optional[int] = None


def utf8_byte_length(content: str) -> int:
    return len(content.encode("utf-8", errors="replace"))


def _split_lines_for_counting(content: str) -> List[str]:
    if len(content) == 0:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(content: str, options: Optional[TruncationOptions] = None) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes).

    Suitable for file reads where you want to see the beginning. Never returns
    partial lines. If the first line exceeds the byte limit, returns empty
    content with ``first_line_exceeds_limit=True``.
    """
    options = options or TruncationOptions()
    max_lines = options.max_lines if options.max_lines is not None else DEFAULT_MAX_LINES
    max_bytes = options.max_bytes if options.max_bytes is not None else DEFAULT_MAX_BYTES

    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    if lines and utf8_byte_length(lines[0]) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: List[str] = []
    output_bytes_count = 0
    truncated_by = "lines"

    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = utf8_byte_length(line) + (1 if i > 0 else 0)  # +1 for newline
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines_arr.append(line)
        output_bytes_count += line_bytes

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[-max_bytes:]
    # Drop a leading partial multi-byte sequence.
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[1:]
    return ""


def truncate_tail(content: str, options: Optional[TruncationOptions] = None) -> TruncationResult:
    """Truncate content from the tail (keep last N lines/bytes).

    Suitable for shell output where you want to see the end (errors, final
    results). May return a partial first line when the last line of the
    original content exceeds the byte limit.
    """
    options = options or TruncationOptions()
    max_lines = options.max_lines if options.max_lines is not None else DEFAULT_MAX_LINES
    max_bytes = options.max_bytes if options.max_bytes is not None else DEFAULT_MAX_BYTES

    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: List[str] = []
    output_bytes_count = 0
    truncated_by = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines_arr) >= max_lines:
            break
        line = lines[i]
        line_bytes = utf8_byte_length(line) + (1 if output_lines_arr else 0)  # +1 for newline
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            # Edge case: if we haven't added ANY lines yet and this line exceeds
            # max_bytes, take the end of the line (partial).
            if not output_lines_arr:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes_count = utf8_byte_length(truncated_line)
                last_line_partial = True
            break
        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple:
    """Truncate a single line to max characters, adding ``[truncated]`` suffix."""
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True
