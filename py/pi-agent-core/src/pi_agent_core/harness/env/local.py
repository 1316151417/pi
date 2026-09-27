"""Local execution environment ported from ``harness/env/nodejs.ts``.

Implements the full ExecutionEnv surface for the host machine: filesystem
operations via :mod:`pathlib` (dispatched to worker threads) and shell
execution via :mod:`asyncio.subprocess` with bounded output capture, spill
files, timeouts, and abort support. Filesystem failures are encoded as
:class:`FileError` results; shell failures as :class:`ExecutionError` results.
"""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import tempfile
import time
import unicodedata
from pathlib import Path, PurePath
from typing import Any, List, Optional

from ..._chord._abort import wait_for_abort
from ..._chord.context import Context
from pi_ai.types import JsonObject
from ..result import Result, err, ok
from ..types import (
    ExecutionError,
    FileError,
    FileInfo,
    ShellExecOptions,
    ShellExecResult,
    ShellOutputTruncation,
)
from ..utils.output_capture import OutputCapture

__all__ = ["LocalExecutionEnv", "create_local_execution_env", "errno_to_file_error_code"]

_ERRNO_CODE_MAP = {
    errno.ENOENT: "not_found",
    errno.EACCES: "permission_denied",
    errno.EPERM: "permission_denied",
    errno.ENOTDIR: "not_a_directory",
    errno.EISDIR: "is_a_directory",
    errno.EEXIST: "already_exists",
    errno.ENOTEMPTY: "directory_not_empty",
    errno.ELOOP: "symlink_loop",
}


def errno_to_file_error_code(exc_errno: int) -> str:
    return _ERRNO_CODE_MAP.get(exc_errno, "unknown")


def _kind_from_path(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _sync_read_text(cwd: str, path: str) -> Result:
    try:
        target = Path(path) if os.path.isabs(path) else Path(cwd) / path
        return ok(target.read_text(encoding="utf-8"))
    except OSError as error:
        code = errno_to_file_error_code(error.errno or 0)
        return err(FileError(code=code, message=str(error), path=path, cause=error))


def _sync_write(cwd: str, path: str, content: Any, mode: str) -> Result:
    try:
        target = Path(path) if os.path.isabs(path) else Path(cwd) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            if mode == "a":
                with open(target, mode, encoding="utf-8") as handle:
                    handle.write(content)
            else:
                target.write_text(content, encoding="utf-8")
        else:
            data = bytes(content)
            with open(target, mode + "b" if "b" not in mode else mode) as handle:
                handle.write(data)
        return ok(None)
    except OSError as error:
        return err(FileError(code=errno_to_file_error_code(error.errno or 0), message=str(error), path=path, cause=error))


def _sync_file_info(cwd: str, path: str) -> Result:
    try:
        target = Path(path) if os.path.isabs(path) else Path(cwd) / path
        stat = target.lstat()
        kind = _kind_from_path(target)
        try:
            size = stat.st_size
        except OSError:
            size = 0
        return ok(
            FileInfo(
                name=target.name,
                path=str(target),
                kind=kind,
                size=size,
                mtime_ms=int(stat.st_mtime * 1000),
            )
        )
    except OSError as error:
        return err(FileError(code=errno_to_file_error_code(error.errno or 0), message=str(error), path=path, cause=error))


def _sync_list_dir(cwd: str, path: str) -> Result:
    try:
        target = Path(path) if os.path.isabs(path) else Path(cwd) / path
        entries: List[FileInfo] = []
        for child in sorted(target.iterdir()):
            info = _sync_file_info(str(target), str(child))
            if info.ok:
                entries.append(info.value)
        return ok(entries)
    except OSError as error:
        return err(FileError(code=errno_to_file_error_code(error.errno or 0), message=str(error), path=path, cause=error))


def _sync_list_all_files(cwd: str, path: str) -> Result:
    try:
        target = Path(path) if os.path.isabs(path) else Path(cwd) / path
        entries: List[FileInfo] = []

        def _walk(directory: Path) -> None:
            for child in sorted(directory.iterdir()):
                info = _sync_file_info(str(directory), str(child))
                if not info.ok:
                    continue
                entries.append(info.value)
                if info.value.kind == "directory":
                    _walk(child)

        _walk(target)
        return ok(entries)
    except OSError as error:
        return err(FileError(code=errno_to_file_error_code(error.errno or 0), message=str(error), path=path, cause=error))


class LocalExecutionEnv:
    """ExecutionEnv implementation bound to the host filesystem and shell."""

    def __init__(self, cwd: Optional[str] = None, extra_env: Optional[dict] = None) -> None:
        self._cwd = os.path.abspath(cwd or os.getcwd())
        self._extra_env = dict(extra_env or {})
        self._active_children: set = set()

    # ------------------------------------------------------------------
    # FileSystem
    # ------------------------------------------------------------------

    @property
    def cwd(self) -> str:
        return self._cwd

    def _abs(self, path: str) -> PurePath:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return PurePath(os.path.normpath(expanded))
        return PurePath(os.path.normpath(os.path.join(self._cwd, expanded)))

    async def absolute_path(self, path: str, _context: Context) -> Result:
        return ok(str(self._abs(path)))

    async def join_path(self, parts: List[str], _context: Context) -> Result:
        joined = PurePath(self._cwd)
        for part in parts:
            if os.path.isabs(part):
                joined = PurePath(part)
            else:
                joined = joined / part
        return ok(str(joined))

    async def read_text_file(self, path: str, _context: Context) -> Result:
        return await asyncio.to_thread(_sync_read_text, self._cwd, path)

    async def open_text_line_reader(self, path: str, context: Context) -> Result:
        reader_result = await self.read_text_lines(path, None, context)
        if not reader_result.ok:
            return reader_result
        from .types import TextLineReader as _TextLineReader  # local protocol

        lines = reader_result.value

        class _SyncLineReader:
            def __init__(self, lines: List[str]) -> None:
                self._lines = list(lines)
                self._index = 0

            async def read_lines(self, max_lines: Optional[int] = None) -> Result:
                if max_lines is None:
                    chunk = self._lines[self._index :]
                    self._index = len(self._lines)
                else:
                    chunk = self._lines[self._index : self._index + max_lines]
                    self._index += len(chunk)
                return ok(chunk)

            async def close(self) -> None:
                return None

        return ok(_SyncLineReader(lines))

    async def read_text_lines(self, path: str, options: Optional[dict], _context: Context) -> Result:
        result = await asyncio.to_thread(_sync_read_text, self._cwd, path)
        if not result.ok:
            return result
        lines = result.value.split("\n")
        max_lines = (options or {}).get("maxLines")
        if isinstance(max_lines, int) and max_lines >= 0:
            lines = lines[:max_lines]
        return ok(lines)

    async def read_binary_file(self, path: str, _context: Context) -> Result:
        def _read() -> Result:
            try:
                target = Path(path) if os.path.isabs(path) else Path(self._cwd) / path
                return ok(target.read_bytes())
            except OSError as error:
                return err(
                    FileError(
                        code=errno_to_file_error_code(error.errno or 0),
                        message=str(error),
                        path=path,
                        cause=error,
                    )
                )

        return await asyncio.to_thread(_read)

    async def write_file(self, path: str, content: Any, _context: Context) -> Result:
        return await asyncio.to_thread(_sync_write, self._cwd, path, content, "w")

    async def append_file(self, path: str, content: Any, _context: Context) -> Result:
        return await asyncio.to_thread(_sync_write, self._cwd, path, content, "a")

    async def rename_file(self, source_path: str, destination_path: str, _context: Context) -> Result:
        def _rename() -> Result:
            try:
                source = Path(source_path) if os.path.isabs(source_path) else Path(self._cwd) / source_path
                destination = (
                    Path(destination_path) if os.path.isabs(destination_path) else Path(self._cwd) / destination_path
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                return ok(None)
            except OSError as error:
                return err(
                    FileError(
                        code=errno_to_file_error_code(error.errno or 0),
                        message=str(error),
                        path=source_path,
                        cause=error,
                    )
                )

        return await asyncio.to_thread(_rename)

    async def file_info(self, path: str, _context: Context) -> Result:
        return await asyncio.to_thread(_sync_file_info, self._cwd, path)

    async def list_dir(self, path: str, _context: Context) -> Result:
        return await asyncio.to_thread(_sync_list_dir, self._cwd, path)

    async def list_all_files(self, path: str, _context: Context) -> Result:
        return await asyncio.to_thread(_sync_list_all_files, self._cwd, path)

    async def canonical_path(self, path: str, _context: Context) -> Result:
        def _resolve() -> Result:
            try:
                target = Path(path) if os.path.isabs(path) else Path(self._cwd) / path
                return ok(str(target.resolve(strict=True)))
            except OSError as error:
                code = errno_to_file_error_code(error.errno or 0)
                if code == "unknown":
                    code = "not_found"
                return err(FileError(code=code, message=str(error), path=path, cause=error))

        return await asyncio.to_thread(_resolve)

    async def exists(self, path: str, context: Context) -> Result:
        result = await self.file_info(path, context)
        if result.ok:
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return result

    async def create_dir(self, path: str, options: Optional[dict], _context: Context) -> Result:
        def _mkdir() -> Result:
            try:
                target = Path(path) if os.path.isabs(path) else Path(self._cwd) / path
                recursive = (options or {}).get("recursive", True)
                target.mkdir(parents=recursive, exist_ok=True)
                return ok(None)
            except OSError as error:
                return err(
                    FileError(
                        code=errno_to_file_error_code(error.errno or 0),
                        message=str(error),
                        path=path,
                        cause=error,
                    )
                )

        return await asyncio.to_thread(_mkdir)

    async def remove_dir(self, path: str, options: Optional[dict], _context: Context) -> Result:
        def _rmdir() -> Result:
            try:
                target = Path(path) if os.path.isabs(path) else Path(self._cwd) / path
                recursive = (options or {}).get("recursive", False)
                if recursive:
                    import shutil

                    shutil.rmtree(target)
                else:
                    target.rmdir()
                return ok(None)
            except OSError as error:
                return err(
                    FileError(
                        code=errno_to_file_error_code(error.errno or 0),
                        message=str(error),
                        path=path,
                        cause=error,
                    )
                )

        return await asyncio.to_thread(_rmdir)

    async def remove_file(self, path: str, _context: Context) -> Result:
        def _unlink() -> Result:
            try:
                target = Path(path) if os.path.isabs(path) else Path(self._cwd) / path
                target.unlink()
                return ok(None)
            except OSError as error:
                return err(
                    FileError(
                        code=errno_to_file_error_code(error.errno or 0),
                        message=str(error),
                        path=path,
                        cause=error,
                    )
                )

        return await asyncio.to_thread(_unlink)

    # ------------------------------------------------------------------
    # Temp files
    # ------------------------------------------------------------------

    async def create_temp_file(self, options: Optional[dict], _context: Context) -> Result:
        prefix = (options or {}).get("prefix", "pi-")
        suffix = (options or {}).get("suffix", "")

        def _create() -> Result:
            try:
                handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
                handle.close()
                return ok(handle.name)
            except OSError as error:
                return err(
                    FileError(code="unknown", message=str(error), path="", cause=error)
                )

        return await asyncio.to_thread(_create)

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def _build_env(self, options: Optional[ShellExecOptions]) -> dict:
        environment = dict(os.environ) if (options is None or options.inherit_env) else {}
        environment.update(self._extra_env)
        if options is not None and options.env:
            environment.update(options.env)
        return environment

    async def exec(
        self,
        command: str,
        options: Optional[ShellExecOptions],
        context: Context,
    ) -> Result:
        abort_signal = context.abort_signal
        capture = OutputCapture(
            options.capture if options else None,
            context,
            options.on_update if options else None,
        )
        cwd = (options.cwd if options and options.cwd else self._cwd)
        process: Optional[asyncio.subprocess.Process] = None
        abort_task: asyncio.Task[None] | None = None
        timed_out = False
        spill_path: Optional[str] = None
        spill_file = None
        spill_prefix: list = []
        spill_prefix_bytes = 0

        def _feed(chunk: bytes) -> None:
            nonlocal spill_path, spill_file, spill_prefix, spill_prefix_bytes
            was_truncated = capture.truncated
            capture.push(chunk)
            if not (options and options.capture and options.capture.spill) or len(chunk) == 0:
                return
            if spill_file is not None or was_truncated:
                spill_file.write(chunk)
            elif capture.truncated:
                # First truncation: start spilling with everything buffered so far.
                handle = tempfile.NamedTemporaryFile(
                    prefix="pi-output-", suffix=".log", delete=False
                )
                spill_path = handle.name
                spill_file = handle
                capture.set_spill_path(spill_path)
                for prefix_chunk in spill_prefix:
                    spill_file.write(prefix_chunk)
                spill_prefix = []
                spill_prefix_bytes = 0
                spill_file.write(chunk)
            else:
                spill_prefix.append(chunk)
                spill_prefix_bytes += len(chunk)
                if spill_prefix_bytes > 8 * 1024 * 1024:
                    spill_prefix.pop(0)
                    spill_prefix_bytes = sum(len(c) for c in spill_prefix)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                env=self._build_env(options),
                stdin=asyncio.subprocess.PIPE if (options and options.stdin) else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._active_children.add(process)

            async def _abort_watch() -> None:
                if abort_signal is None:
                    return
                await wait_for_abort(abort_signal)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass

            abort_task = asyncio.get_running_loop().create_task(_abort_watch()) if abort_signal else None

            async def _pump_stdin() -> None:
                if process.stdin is not None and options and options.stdin is not None:
                    process.stdin.write(options.stdin.encode(options.stdin_encoding or "utf-8"))
                    await process.stdin.drain()
                    process.stdin.close()

            async def _pump_output() -> None:
                assert process.stdout is not None
                while True:
                    chunk = await process.stdout.read(65536)
                    if not chunk:
                        break
                    _feed(chunk)
                    # Give the adaptive publisher a chance to emit within the
                    # stream instead of only at completion.
                    await asyncio.sleep(0)
                capture.flush()
                capture.finish()

            stdin_task = asyncio.get_running_loop().create_task(_pump_stdin())
            output_task = asyncio.get_running_loop().create_task(_pump_output())

            timeout_seconds = options.timeout if options else None
            try:
                if timeout_seconds is not None:
                    await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                else:
                    await process.wait()
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()

            await output_task
            await stdin_task
            if abort_signal is not None and abort_signal.aborted:
                return err(ExecutionError(code="aborted", message="aborted"))
            if timed_out:
                return err(
                    ExecutionError(
                        code="timeout",
                        message=f"Command timed out after {timeout_seconds} seconds",
                    )
                )

            output = capture.snapshot()
            exit_code = process.returncode if process.returncode is not None else 1
            return ok(
                ShellExecResult(
                    exit_code=exit_code,
                    truncation=output.truncation,
                    spill_path=output.spill_path,
                    last_line_bytes=output.last_line_bytes,
                )
            )
        except Exception as error:  # noqa: BLE001 - encode in Result like TS
            return err(ExecutionError(code="unknown", message=str(error), cause=error))
        finally:
            if abort_task is not None:
                abort_task.cancel()
                await asyncio.gather(abort_task, return_exceptions=True)
            capture.dispose()
            if spill_file is not None:
                try:
                    spill_file.close()
                except OSError:
                    pass
            if process is not None:
                self._active_children.discard(process)

    async def cleanup(self, _context: Context) -> None:
        for process in list(self._active_children):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        self._active_children.clear()

    # Extra helpers used by tools -----------------------------------------

    async def read_binary_file_compat(self, path: str, context: Context) -> Result:
        return await self.read_binary_file(path, context)


def create_local_execution_env(
    cwd: Optional[str] = None, extra_env: Optional[dict] = None
) -> LocalExecutionEnv:
    """Create a host-bound execution environment rooted at ``cwd``."""
    return LocalExecutionEnv(cwd=cwd, extra_env=extra_env)
