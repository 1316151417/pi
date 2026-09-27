"""Interactive command execution from ``core/bash-executor.ts``."""

from __future__ import annotations

import codecs
import re
import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from pi_agent_core.harness.utils.truncate import DEFAULT_MAX_BYTES, truncate_tail
from pi_ai.abort import AbortSignal

from .tools.bash import BashExecOptions, BashOperations

# Equivalent to the ANSI CSI/OSC matching used by the TypeScript stripAnsi utility.
_ANSI = re.compile(
    r"\x1b\][\s\S]*?(?:\x07|\x1b\\|\x9c)"
    r"|[\x1b\x9b][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]"
)


def _sanitize_binary_output(text: str) -> str:
    return "".join(
        char for char in text
        if char in "\t\n\r"
        or (ord(char) > 0x1F and not 0xFFF9 <= ord(char) <= 0xFFFB)
    )


def _utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


@dataclass
class BashExecutorOptions:
    on_chunk: Callable[[str], None] | None = None
    signal: AbortSignal | None = None


@dataclass
class BashResult:
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    full_output_path: str | None = None


async def execute_bash_with_operations(
    command: str, cwd: str, operations: BashOperations,
    options: BashExecutorOptions | None = None,
) -> BashResult:
    chosen = options or BashExecutorOptions()
    output_chunks: list[str] = []
    output_length = 0
    max_output_length = DEFAULT_MAX_BYTES * 2
    total_raw_bytes = 0
    temp_file_path: str | None = None
    temp_file_stream: TextIO | None = None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def ensure_temp_file() -> None:
        nonlocal temp_file_path, temp_file_stream
        if temp_file_path is not None:
            return
        path = Path(tempfile.gettempdir()) / f"pi-bash-{secrets.token_hex(8)}.log"
        temp_file_stream = path.open("x", encoding="utf-8")
        temp_file_path = str(path)
        for chunk in output_chunks:
            temp_file_stream.write(chunk)

    def on_data(data: bytes) -> None:
        nonlocal total_raw_bytes, output_length
        total_raw_bytes += len(data)
        text = _sanitize_binary_output(_ANSI.sub("", decoder.decode(data, final=False))).replace("\r", "")
        if total_raw_bytes > DEFAULT_MAX_BYTES:
            ensure_temp_file()
        if temp_file_stream is not None:
            temp_file_stream.write(text)
        output_chunks.append(text)
        output_length += _utf16_length(text)
        while output_length > max_output_length and len(output_chunks) > 1:
            removed = output_chunks.pop(0)
            output_length -= _utf16_length(removed)
        if chosen.on_chunk is not None:
            chosen.on_chunk(text)

    def result(exit_code: int | None, cancelled: bool) -> BashResult:
        full_output = "".join(output_chunks)
        truncated = truncate_tail(full_output)
        if truncated.truncated:
            ensure_temp_file()
        if temp_file_stream is not None:
            temp_file_stream.close()
        return BashResult(
            output=truncated.content if truncated.truncated else full_output,
            exit_code=exit_code, cancelled=cancelled, truncated=truncated.truncated,
            full_output_path=temp_file_path,
        )

    try:
        executed = await operations.exec(command, cwd, BashExecOptions(on_data, chosen.signal))
        cancelled = chosen.signal.aborted if chosen.signal is not None else False
        return result(None if cancelled else executed.exit_code, cancelled)
    except Exception:
        if chosen.signal is not None and chosen.signal.aborted:
            return result(None, True)
        if temp_file_stream is not None:
            temp_file_stream.close()
        raise


__all__ = ["BashExecutorOptions", "BashResult", "execute_bash_with_operations"]
