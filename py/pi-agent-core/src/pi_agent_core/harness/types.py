"""Harness types ported from ``harness/types.ts``.

Larger surface than the first pass: FileInfo, the full FileSystem protocol,
shell capture metadata, ExecutionEnv, and the harness-native tool shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .._chord.context import Context
from ..types import AgentTool, AgentToolResult
from .result import Result, err, ok

__all__ = [
    "FileKind",
    "FileError",
    "CompactionError",
    "BranchSummaryError",
    "is_compaction_error",
    "is_branch_summary_error",
    "FileInfo",
    "FileContent",
    "FileStat",
    "FileBuffer",
    "AllFilesListEntry",
    "TextLineReader",
    "FileSystem",
    "ShellOutputLimits",
    "ShellOutputCaptureOptions",
    "ShellOutputTruncation",
    "ShellOutputMetadata",
    "ShellOutputView",
    "ShellOutputUpdate",
    "ShellExecOptions",
    "ShellExecResult",
    "ExecutionError",
    "Shell",
    "ExecutionEnv",
    "ExecutionToolContext",
    "AgentHarnessStreamOptions",
    "AgentHarnessStreamOptionsPatch",
    "Skill",
    "AgentHarnessToolInvocation",
    "AgentHarnessToolUpdateOptions",
    "AgentHarnessTool",
    "is_file_error",
    "is_execution_error",
    "fs_error",
    "fs_ok",
    "fs_err",
    "exec_ok",
    "exec_err",
    "get_or_throw",
    "FileKind",
    "Permission",
]

FileKind = str  # "file" | "directory" | "symlink" | "other"
Permission = str  # "read" | "edit"


@dataclass
class FileError(Exception):
    """Error returned (never raised) by a FileSystem implementation."""

    code: str  # "not_found" | "permission_denied" | "aborted" | "is_a_directory" | ...
    message: str
    path: str = ""
    cause: Any = None

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "path": self.path}


def is_file_error(error: Any) -> bool:
    return isinstance(error, FileError)


def fs_error(code: str, message: str, path: str = "", cause: Any = None) -> FileError:
    return FileError(code=code, message=message, path=path, cause=cause)


def fs_ok(value: Any = None) -> Result:
    return ok(value)


def fs_err(error: FileError) -> Result:
    return err(error)


def get_or_throw(result: Result) -> Any:
    """Return the success value or raise the failure error."""
    if not bool(getattr(result, "ok", False)):
        raise result.error
    return result.value


@dataclass
class FileInfo:
    """Metadata for one filesystem object in a :class:`FileSystem`."""

    name: str
    path: str
    kind: FileKind
    size: int  # bytes
    mtime_ms: int


@dataclass
class FileContent:
    kind: str = field(default="text", init=False)  # "text"
    encoding: str = "utf-8"
    content: str = ""
    bytes: int = 0


@dataclass
class FileStat:
    kind: FileKind
    size: int


@dataclass
class FileBuffer:
    kind: str = field(default="binary", init=False)  # "binary"
    bytes: int
    base64: str


@dataclass
class AllFilesListEntry:
    path: str
    kind: FileKind = "file"


@runtime_checkable
class TextLineReader(Protocol):
    """Pull-based line reader."""

    def read_lines(self, max_lines: Optional[int] = None) -> Awaitable[Result]: ...
    def close(self) -> Awaitable[None]: ...


@runtime_checkable
class FileSystem(Protocol):
    """Filesystem capability used by the harness.

    Paths may be absolute or relative to ``cwd``. Operation methods must never
    raise; all failures are encoded in the returned :class:`Result`.
    """

    @property
    def cwd(self) -> str: ...

    def absolute_path(self, path: str, context: Context) -> Awaitable[Result]: ...
    def join_path(self, parts: List[str], context: Context) -> Awaitable[Result]: ...
    def read_text_file(self, path: str, context: Context) -> Awaitable[Result]: ...
    def open_text_line_reader(self, path: str, context: Context) -> Awaitable[Result]: ...
    def read_text_lines(self, path: str, options: Optional[dict], context: Context) -> Awaitable[Result]: ...
    def read_binary_file(self, path: str, context: Context) -> Awaitable[Result]: ...
    def write_file(self, path: str, content: Any, context: Context) -> Awaitable[Result]: ...
    def append_file(self, path: str, content: Any, context: Context) -> Awaitable[Result]: ...
    def rename_file(self, source_path: str, destination_path: str, context: Context) -> Awaitable[Result]: ...
    def file_info(self, path: str, context: Context) -> Awaitable[Result]: ...
    def list_dir(self, path: str, context: Context) -> Awaitable[Result]: ...
    def canonical_path(self, path: str, context: Context) -> Awaitable[Result]: ...
    def exists(self, path: str, context: Context) -> Awaitable[Result]: ...
    def create_dir(self, path: str, options: Optional[dict], context: Context) -> Awaitable[Result]: ...
    def remove_dir(self, path: str, options: Optional[dict], context: Context) -> Awaitable[Result]: ...
    def remove_file(self, path: str, context: Context) -> Awaitable[Result]: ...
    def list_all_files(self, path: str, context: Context) -> Awaitable[Result]: ...


# ---------------------------------------------------------------------------
# Shell capture
# ---------------------------------------------------------------------------


@dataclass
class ShellOutputLimits:
    max_bytes: Optional[int] = None
    max_lines: Optional[int] = None
    retain: str = "tail"  # "head" | "tail"


@dataclass
class ShellOutputCaptureOptions:
    limits: Optional[ShellOutputLimits] = None
    spill: bool = False


@dataclass
class ShellOutputTruncation:
    content: Optional[str] = None  # present only in full TruncationResult payloads
    truncated: bool = False
    truncated_by: Optional[str] = None  # "lines" | "bytes" | None
    total_lines: int = 0
    total_bytes: int = 0
    output_lines: int = 0
    output_bytes: int = 0
    last_line_partial: bool = False
    first_line_exceeds_limit: bool = False
    max_lines: int = 0
    max_bytes: int = 0

    def to_json(self) -> dict:
        return {
            "truncated": self.truncated,
            "truncatedBy": self.truncated_by,
            "totalLines": self.total_lines,
            "totalBytes": self.total_bytes,
            "outputLines": self.output_lines,
            "outputBytes": self.output_bytes,
            "lastLinePartial": self.last_line_partial,
            "firstLineExceedsLimit": self.first_line_exceeds_limit,
            "maxLines": self.max_lines,
            "maxBytes": self.max_bytes,
        }


@dataclass
class ShellOutputMetadata:
    truncation: ShellOutputTruncation
    spill_path: Optional[str] = None
    last_line_bytes: Optional[int] = None


@dataclass
class ShellOutputView(ShellOutputMetadata):
    text: str = ""


@dataclass
class ShellOutputUpdate:
    """Incremental source-side change to one bounded shell output view."""

    kind: str  # "replace" | "append" | "slide" | "metadata"
    output: Optional[ShellOutputView] = None
    text: Optional[str] = None
    drop: int = 0
    metadata: Optional[ShellOutputMetadata] = None


@dataclass
class ShellExecOptions:
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    inherit_env: bool = True
    timeout: Optional[float] = None  # seconds
    capture: Optional[ShellOutputCaptureOptions] = None
    stdin: Optional[str] = None
    stdin_encoding: str = "utf-8"
    on_update: Optional[Callable[[ShellOutputUpdate, Context], None]] = None


@dataclass
class ShellExecResult(ShellOutputMetadata):
    exit_code: int = 0


@dataclass
class ExecutionError(Exception):
    """Shell execution failure with a stable ``code`` discriminator."""

    code: str  # "unknown" | "timeout" | "aborted"
    message: str
    cause: Any = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message}


def is_execution_error(error: Any) -> bool:
    return isinstance(error, ExecutionError)


@dataclass
class CompactionError(Exception):
    """User-facing compaction failure: ``code`` is "aborted" | "summarization_failed"."""

    code: str
    message: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class BranchSummaryError(Exception):
    """User-facing branch summary failure."""

    code: str  # "aborted" | "summarization_failed"
    message: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message}


def is_compaction_error(error: Any) -> bool:
    return isinstance(error, CompactionError)


def is_branch_summary_error(error: Any) -> bool:
    return isinstance(error, BranchSummaryError)


def exec_ok(value: Any) -> Result:
    return ok(value)


def exec_err(error: ExecutionError) -> Result:
    return err(error)


@runtime_checkable
class Shell(Protocol):
    def exec(
        self, command: str, options: Optional[ShellExecOptions], context: Context
    ) -> Awaitable[Result]: ...
    def cleanup(self, context: Context) -> Awaitable[None]: ...


@runtime_checkable
class ExecutionEnv(Protocol):
    """Filesystem and process execution environment used by the harness."""


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """Skill loaded from a ``SKILL.md`` file or provided by an application.

    ``name``, ``description``, and ``file_path`` are inserted into the system
    prompt in an XML-formatted block as suggested by agentskills.io.
    """

    name: str
    description: str
    content: str
    file_path: str
    disable_model_invocation: bool = False


# ---------------------------------------------------------------------------
# Harness-native tools
# ---------------------------------------------------------------------------



@dataclass
class AgentHarnessStreamOptions:
    """Curated provider request options owned by the harness and snapshotted per turn."""

    transport: Optional[str] = None
    timeout_ms: Optional[int] = None
    max_retries: Optional[int] = None
    max_retry_delay_ms: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    metadata: Optional[dict] = None
    cache_retention: Optional[str] = None
    #: Ask a capable provider to continue generation asynchronously.
    deferred: Any = None

    def to_json(self) -> dict:
        data: dict = {}
        for key, value in (
            ("transport", self.transport),
            ("timeoutMs", self.timeout_ms),
            ("maxRetries", self.max_retries),
            ("maxRetryDelayMs", self.max_retry_delay_ms),
            ("headers", self.headers),
            ("metadata", self.metadata),
            ("cacheRetention", self.cache_retention),
            ("deferred", self.deferred),
        ):
            if value is not None:
                data[key] = value
        return data


class AgentHarnessStreamOptionsPatch:
    """Per-request stream option patch returned by provider hooks.

    Tracks which fields the hook explicitly provided: TS distinguishes an absent
    key from an explicit ``undefined`` (which deletes the key), and a dataclass
    with defaults cannot express that on its own.
    """

    _FIELDS = (
        "transport",
        "timeout_ms",
        "max_retries",
        "max_retry_delay_ms",
        "headers",
        "metadata",
        "cache_retention",
        "deferred",
    )

    def __init__(self, **kwargs: Any) -> None:
        for name in self._FIELDS:
            setattr(self, name, None)
        for name, value in kwargs.items():
            if name not in self._FIELDS:
                raise TypeError(f"Unexpected stream option patch field: {name!r}")
            setattr(self, name, value)
        #: Field names explicitly provided by the hook.
        self.present = set(kwargs)

    def has(self, name: str) -> bool:
        """Whether the hook explicitly provided ``name``."""
        return name in self.present

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        provided = {name: getattr(self, name) for name in self.present}
        return f"AgentHarnessStreamOptionsPatch({provided})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AgentHarnessStreamOptionsPatch):
            return NotImplemented
        return self.present == other.present and all(
            getattr(self, name) == getattr(other, name) for name in self.present
        )


@dataclass
class ExecutionToolContext:
    """Context supplied to harness tools."""

    env: Any  # ExecutionEnv
    cwd: str = ""
    description: str = ""


@dataclass
class AgentHarnessToolUpdateOptions:
    checkpoint: bool = False


AgentHarnessToolUpdateCallback = Callable[[Any, Optional[AgentHarnessToolUpdateOptions]], None]


@dataclass
class AgentHarnessToolInvocation:
    """Stable harness identity for one logical tool call."""

    invocation_id: str
    operation_id: str
    turn_id: str
    get_memo: Optional[Callable[[str], Awaitable[Any]]] = None
    set_memo: Optional[Callable[[str, Any], Awaitable[None]]] = None


HarnessToolExecute = Callable[
    [
        str,  # tool_call_id
        Any,  # params
        Optional[AgentHarnessToolUpdateCallback],
        ExecutionToolContext,
        Optional[AgentHarnessToolInvocation],
        Context,
    ],
    Awaitable[AgentToolResult],
]


@dataclass
class AgentHarnessTool(AgentTool):
    """Tool whose execute receives the harness execution context.

    Overrides the agent-core ``execute`` signature:
    ``(tool_call_id, params, on_update, tool_context, invocation, context)``.
    """

    execute: Optional[HarnessToolExecute] = None
