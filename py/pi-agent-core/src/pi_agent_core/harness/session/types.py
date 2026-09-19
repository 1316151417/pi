"""Session entry and storage types ported from ``harness/session/types.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

from ..._pi_ai.types import AgentMessage, AssistantMessage, Usage
from ..._chord.context import Context
from .values import ListElement, ListReadOptions, StoredValue, Value, ValueList

__all__ = [
    "EntryType",
    "EntryBase",
    "MessageEntry",
    "CompactionEntry",
    "BranchSummaryEntry",
    "CustomEntry",
    "Entry",
    "NewEntry",
    "EntryWrite",
    "UsageRow",
    "UsageWrite",
    "Write",
    "EntryStructure",
    "EntryCursor",
    "BranchScan",
    "StorageBranchScan",
    "EntryScan",
    "UsageScan",
    "EntryQuery",
    "SessionStats",
    "OperationError",
    "OperationResultRecord",
    "SuspendedRunRecord",
    "SettledAssistantMessage",
    "TerminalStatus",
    "SessionMetadata",
    "IdGenerator",
    "CommitResult",
    "Storage",
    "SessionReader",
    "SessionMutation",
    "SessionMutator",
    "Session",
    "Branch",
    "SessionCreateOptions",
    "ForkOptions",
    "SessionRepo",
    "LaneConfiguration",
    "LaneState",
    "PendingEntry",
    "InboxItem",
    "JsonValue",
    "OperationMeta",
    "OperationIntent",
    "Operation",
    "OperationAt",
    "OperationScope",
    "OperationState",
    "CONTROL_RUNNING",
    "CONTROL_CANCEL_REQUESTED",
    "Continuation",
    "CheckpointData",
    "Control",
    "GenerationContext",
    "NormalizedRetryPolicy",
    "ResultBoundary",
    "RetryWait",
    "RunSettings",
    "SummaryContext",
    "SummaryTask",
    "ToolBatch",
    "ToolCall",
    "StartingOperation",
    "CheckpointOperation",
    "AssistantReadyOperation",
    "AssistantEffectPendingOperation",
    "AssistantRetryWaitOperation",
    "ToolsOperation",
    "DeferredSuspendedOperation",
    "DeferredEffectPendingOperation",
    "SummaryDecidingOperation",
    "SummaryReadyOperation",
    "SummaryEffectPendingOperation",
    "SummaryRetryWaitOperation",
    "NavigationReadyToCommitOperation",
    "operation_scope_of",
    "revive_operation_meta",
    "revive_operation_state",
]

JsonValue = Any
EntryType = str  # "message" | "compaction" | "branch_summary" | "custom"


@dataclass
class EntryBase:
    id: str
    parent_id: Optional[str]
    type: EntryType
    custom_type: Optional[str] = None
    seq: int = 0
    timestamp: int = 0


@dataclass
class MessageEntry(EntryBase):
    type: str = field(default="message", init=False)
    message: Any = None
    terminate: Optional[bool] = None

    def to_json(self) -> dict:
        data = {
            "type": "message",
            "id": self.id,
            "parentId": self.parent_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "message": self.message.to_json() if hasattr(self.message, "to_json") else self.message,
        }
        if self.terminate:
            data["terminate"] = True
        return data


@dataclass
class CompactionEntry(EntryBase):
    type: str = field(default="compaction", init=False)
    summary: str = ""
    retained_tail: List[AgentMessage] = field(default_factory=list)
    tokens_before: int = 0
    details: JsonValue = None
    usage: Optional[Usage] = None
    from_hook: bool = False

    def to_json(self) -> dict:
        data: dict = {
            "type": "compaction",
            "id": self.id,
            "parentId": self.parent_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "retainedTail": [
                message.to_json() if hasattr(message, "to_json") else message
                for message in self.retained_tail
            ],
            "tokensBefore": self.tokens_before,
            "fromHook": self.from_hook,
        }
        if self.details is not None:
            data["details"] = self.details
        if self.usage is not None:
            data["usage"] = self.usage.to_json()
        return data


@dataclass
class BranchSummaryEntry(EntryBase):
    type: str = field(default="branch_summary", init=False)
    from_id: Optional[str] = None
    summary: str = ""
    details: JsonValue = None
    usage: Optional[Usage] = None
    from_hook: bool = False

    def to_json(self) -> dict:
        data: dict = {
            "type": "branch_summary",
            "id": self.id,
            "parentId": self.parent_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "fromId": self.from_id,
            "summary": self.summary,
            "fromHook": self.from_hook,
        }
        if self.details is not None:
            data["details"] = self.details
        if self.usage is not None:
            data["usage"] = self.usage.to_json()
        return data


@dataclass
class CustomEntry(EntryBase):
    type: str = field(default="custom", init=False)
    custom_type: str = ""
    data: JsonValue = None

    def to_json(self) -> dict:
        data: dict = {
            "type": "custom",
            "id": self.id,
            "parentId": self.parent_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "customType": self.custom_type,
        }
        if self.data is not None:
            data["data"] = self.data
        return data


Entry = Any  # MessageEntry | CompactionEntry | BranchSummaryEntry | CustomEntry

#: Entry supplied to a transaction before storage assigns sequence and timestamp.
NewEntry = Any


@dataclass
class EntryWrite:
    kind: str = field(default="entry", init=False)
    entry: NewEntry = None


@dataclass
class UsageRow:
    id: str
    usage: Usage = field(default_factory=Usage)
    entry_id: Optional[str] = None
    adjustment: bool = False
    details: JsonValue = None
    seq: int = 0

    def to_json(self) -> dict:
        data: dict = {
            "id": self.id,
            "seq": self.seq,
            "usage": self.usage.to_json() if hasattr(self.usage, "to_json") else self.usage,
            "adjustment": self.adjustment,
        }
        if self.entry_id is not None:
            data["entryId"] = self.entry_id
        if self.details is not None:
            data["details"] = self.details
        return data


@dataclass
class UsageWrite:
    kind: str = field(default="usage", init=False)
    row: UsageRow = None  # type: ignore[assignment]


Write = Any  # EntryWrite | UsageWrite | ValueSetWrite | ValueDeleteWrite | ListAppendWrite | ListDeleteWrite


@dataclass
class LaneConfiguration:
    model: dict = field(default_factory=dict)  # {provider, modelId}
    thinking_level: str = "off"
    active_tool_names: List[str] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: Any) -> "LaneConfiguration":
        """Normalize a stored lane configuration, which may be a replayed mapping."""
        if value is None:
            return cls()
        if isinstance(value, LaneConfiguration):
            return value
        if not isinstance(value, dict):
            return value
        model = value.get("model") or {}
        return cls(
            model=dict(model) if isinstance(model, dict) else model,
            thinking_level=value.get("thinkingLevel", value.get("thinking_level", "off")),
            active_tool_names=list(
                value.get("activeToolNames", value.get("active_tool_names", [])) or []
            ),
        )

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "thinkingLevel": self.thinking_level,
            "activeToolNames": list(self.active_tool_names),
        }


@dataclass
class InboxItem:
    entry_id: str
    kind: str  # "steer" | "followUp" | "nextRun" | "write"

    def to_json(self) -> dict:
        return {"entryId": self.entry_id, "kind": self.kind}


@dataclass
class LaneState:
    current_operation_id: Optional[str] = None
    last_operation_id: Optional[str] = None
    inbox: List[InboxItem] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "currentOperationId": self.current_operation_id,
            "lastOperationId": self.last_operation_id,
            "inbox": [{"entryId": item.entry_id, "kind": item.kind} for item in self.inbox],
        }


@dataclass
class PendingEntry:
    type: str  # "message" | "custom"
    payload: Any = None
    custom_type: Optional[str] = None

    @classmethod
    def from_value(cls, value: Any) -> "PendingEntry":
        """Normalize a stored pending payload, which may be a replayed mapping."""
        if value is None:
            return cls(type="")
        if isinstance(value, PendingEntry):
            return value
        return cls(
            type=value.get("type", ""),
            payload=value.get("payload"),
            custom_type=value.get("customType", value.get("custom_type")),
        )

    def to_json(self) -> dict:
        data: dict = {"type": self.type}
        if self.custom_type is not None:
            data["customType"] = self.custom_type
        if self.payload is not None:
            data["payload"] = self.payload
        return data


@dataclass
class EntryStructure:
    id: str
    parent_id: Optional[str]
    seq: int
    timestamp: int
    type: EntryType
    custom_type: Optional[str] = None


@dataclass
class EntryCursor:
    seq: int


@dataclass
class BranchScan:
    start: Optional[str] = None
    stop_at_type: Optional[EntryType] = None
    stop_at_id: Optional[str] = None
    type: Optional[EntryType] = None
    custom_type: Optional[str] = None
    order: str = "newestFirst"  # "newestFirst" | "oldestFirst"
    limit: Optional[int] = None
    cursor: Optional[EntryCursor] = None


@dataclass
class StorageBranchScan:
    start: str
    stop_at_type: Optional[EntryType] = None
    stop_at_id: Optional[str] = None
    type: Optional[EntryType] = None
    custom_type: Optional[str] = None
    order: str = "newestFirst"
    limit: Optional[int] = None
    cursor: Optional[EntryCursor] = None


@dataclass
class EntryScan:
    type: Optional[EntryType] = None
    custom_type: Optional[str] = None
    from_seq: Optional[int] = None
    to_seq: Optional[int] = None
    order: str = "asc"
    limit: Optional[int] = None


@dataclass
class UsageScan:
    from_seq: Optional[int] = None
    to_seq: Optional[int] = None
    order: str = "asc"
    limit: Optional[int] = None


@dataclass
class EntryQuery:
    type: Optional[EntryType] = None
    custom_type: Optional[str] = None
    order: str = "desc"
    limit: Optional[int] = None
    cursor: Optional[EntryCursor] = None


@dataclass
class SessionStats:
    message_count: int = 0
    usage: Usage = field(default_factory=Usage)


@dataclass
class OperationError:
    """Operation failure payload."""

    code: str = "unknown"
    message: str = ""
    details: Any = None

    def to_json(self) -> dict:
        data = {"code": self.code, "message": self.message}
        if self.details is not None:
            data["details"] = self.details
        return data


#: Terminal operation statuses recorded on result records.
TerminalStatus = str  # "completed" | "declined" | "aborted" | "failed"


@dataclass
class OperationResultRecord:
    """Immutable lane-lived observation record written by one terminal transaction."""

    operation_id: str
    kind: str  # "run" | "compaction" | "navigation"
    status: TerminalStatus
    from_tip_id: Optional[str] = None
    tip_id: Optional[str] = None
    started_at: int = 0
    ended_at: int = 0
    error: Optional[OperationError] = None

    def to_json(self) -> dict:
        data: dict = {
            "operationId": self.operation_id,
            "kind": self.kind,
            "status": self.status,
            "fromTipId": self.from_tip_id,
            "tipId": self.tip_id,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
        }
        if self.error is not None:
            data["error"] = self.error.to_json()
        return data


@dataclass
class SuspendedRunRecord:
    """Persisted observation of a run suspended on a provider deferred handle."""

    operation_id: str
    deferred: Any = None
    reason: str = "deferred"

    def to_json(self) -> dict:
        return {
            "operationId": self.operation_id,
            "reason": self.reason,
            "deferred": self.deferred.to_json() if hasattr(self.deferred, "to_json") else self.deferred,
        }


#: Assistant message guaranteed to carry a settled stop reason.
SettledAssistantMessage = AssistantMessage


@dataclass
class SessionMetadata:
    id: str
    created_at: int
    storage_version: int
    cwd: Optional[str] = None
    parent_session_id: Optional[str] = None
    legacy_parent_session_path: Optional[str] = None


class IdGenerator(Protocol):
    def next(self, timestamp_ms: Optional[int] = None) -> str: ...


@dataclass
class CommitResult:
    first_seq: int
    seqs: List[int] = field(default_factory=list)
    timestamp: int = 0
    stats: SessionStats = None  # type: ignore[assignment]


@runtime_checkable
class Storage(Protocol):
    def commit(self, writes: List[Write], context: Context) -> Awaitable[CommitResult]: ...
    def get_entries(self, ids: List[str], context: Context) -> Awaitable[Dict[str, Entry]]: ...
    def get_value(self, address: Value, context: Context) -> Awaitable[Any]: ...
    def scan_values(self, prefix: Value, context: Context) -> Awaitable[List[StoredValue]]: ...
    def read_list(self, address: ValueList, options: Optional[ListReadOptions], context: Context) -> Awaitable[List[ListElement]]: ...
    def scan_branch(self, query: StorageBranchScan, context: Context) -> Awaitable[List[Entry]]: ...
    def scan_branch_structure(self, query: StorageBranchScan, context: Context) -> Awaitable[List[EntryStructure]]: ...
    def scan_entries(self, query: EntryScan, context: Context) -> Awaitable[List[Entry]]: ...
    def scan_usage(self, query: UsageScan, context: Context) -> Awaitable[List[UsageRow]]: ...
    def get_stats(self, context: Context) -> Awaitable[SessionStats]: ...
    def close(self, context: Context) -> Awaitable[None]: ...


@runtime_checkable
class SessionReader(Protocol):
    def get_entries(self, ids: List[str], context: Context) -> Awaitable[Dict[str, Entry]]: ...
    def get_stats(self, context: Context) -> Awaitable[SessionStats]: ...
    def get_value(self, address: Value, context: Context) -> Awaitable[Any]: ...
    def scan_values(self, prefix: Value, context: Context) -> Awaitable[List[StoredValue]]: ...
    def read_list(self, address: ValueList, options: Optional[ListReadOptions], context: Context) -> Awaitable[List[ListElement]]: ...
    def scan_branch(self, query: StorageBranchScan, context: Context) -> Awaitable[List[Entry]]: ...


class SessionMutation(Protocol):
    """Exclusive keyless mutation barrier for one Session."""

    def commit(self, writes: List[Write], context: Context) -> Awaitable[CommitResult]: ...
    def end(self, context: Context) -> Awaitable[None]: ...


#: Callback-scoped mutation capability without authority to release its Session barrier.
SessionMutator = SessionMutation

SessionMutationCallback = Callable[[SessionMutator, Context], Any]


@runtime_checkable
class Branch(Protocol):
    @property
    def name(self) -> str: ...

    def get_tip_id(self, context: Context) -> Awaitable[Optional[str]]: ...
    def find_entries(self, query: Optional[BranchScan], context: Context) -> Awaitable[List[Entry]]: ...
    def find_entry(self, query: Optional[BranchScan], context: Context) -> Awaitable[Any]: ...
    def append_message(self, message: AgentMessage, context: Context) -> Awaitable[str]: ...
    def append_custom_entry(self, custom_type: str, data: Any, context: Context) -> Awaitable[str]: ...


@runtime_checkable
class Session(Protocol):
    @property
    def metadata(self) -> SessionMetadata: ...

    @property
    def id_generator(self) -> IdGenerator: ...

    def get_entry(self, id: str, context: Context) -> Awaitable[Any]: ...
    def get_stats(self, context: Context) -> Awaitable[SessionStats]: ...
    def get_name(self, context: Context) -> Awaitable[Optional[str]]: ...
    def get_label(self, target_id: str, context: Context) -> Awaitable[Optional[str]]: ...
    def find_entries(self, query: Optional[EntryQuery], context: Context) -> Awaitable[List[Entry]]: ...
    def find_entry(self, query: Optional[EntryQuery], context: Context) -> Awaitable[Any]: ...
    def branch(self, name: str, context: Context) -> Awaitable[Any]: ...
    def create_branch(self, name: str, at: Optional[str], context: Context) -> Awaitable[Branch]: ...
    def begin_mutation(self, context: Context) -> Awaitable[SessionMutation]: ...
    def mutate(self, mutation: SessionMutationCallback, context: Context) -> Awaitable[Any]: ...
    def set_value(self, address: Value, next_value: Any, context: Context) -> Awaitable[None]: ...
    def delete_value(self, address: Value, context: Context) -> Awaitable[None]: ...
    def append_list(self, address: ValueList, element: Any, context: Context) -> Awaitable[None]: ...
    def delete_list(self, address: ValueList, context: Context) -> Awaitable[None]: ...
    def set_name(self, name: Optional[str], context: Context) -> Awaitable[None]: ...
    def set_label(self, target_id: str, label: Optional[str], context: Context) -> Awaitable[None]: ...
    def close(self, context: Context) -> Awaitable[None]: ...


@dataclass
class SessionCreateOptions:
    id: Optional[str] = None
    parent_session_id: Optional[str] = None


@dataclass
class ForkOptions:
    """Fork scope: copy one branch path or the whole conversation tree."""

    scope: str  # "branch" | "tree"
    branch: Optional[str] = None
    entry_id: Optional[str] = None
    position: Optional[str] = None  # "before" | "at"
    id: Optional[str] = None


class SessionRepo(Protocol):
    def create(self, options: SessionCreateOptions, context: Context) -> Awaitable[Session]: ...
    def open(self, metadata: SessionMetadata, context: Context) -> Awaitable[Session]: ...
    def list(self, options: Any, context: Context) -> Awaitable[List[SessionMetadata]]: ...
    def delete(self, metadata: SessionMetadata, context: Context) -> Awaitable[None]: ...
    def fork(self, source: SessionMetadata, options: ForkOptions, context: Context) -> Awaitable[Session]: ...


# ---------------------------------------------------------------------------
# Durable operation state
#
# Durable operation state is one flat union with one family-neutral discriminator
# per dispatcher leaf. ToolBatch/ToolCall remain the nested child collection state
# machine, and cancellation stays orthogonal via Control.
# ---------------------------------------------------------------------------


@dataclass
class Control:
    """Orthogonal cancellation state carried by every operation leaf."""

    status: str = "running"  # "running" | "cancel_requested"
    requested_at: Optional[int] = None

    def to_json(self) -> dict:
        if self.status == "cancel_requested":
            return {"status": "cancel_requested", "requestedAt": self.requested_at or 0}
        return {"status": "running"}


CONTROL_RUNNING = Control()
CONTROL_CANCEL_REQUESTED = "cancel_requested"


@dataclass
class NormalizedRetryPolicy:
    """Retry policy normalized for durable storage."""

    max_attempts: int = 0
    base_delay_ms: int = 0
    max_agent_delay_ms: int = 0

    def to_json(self) -> dict:
        return {
            "maxAttempts": self.max_attempts,
            "baseDelayMs": self.base_delay_ms,
            "maxAgentDelayMs": self.max_agent_delay_ms,
        }


@dataclass
class RunSettings:
    """Run settings captured at acceptance time."""

    compaction: Any = None
    steering_mode: str = "one-at-a-time"
    follow_up_mode: str = "one-at-a-time"
    tool_execution: str = "parallel"  # "sequential" | "parallel"

    def to_json(self) -> dict:
        return {
            "compaction": self.compaction.to_json() if hasattr(self.compaction, "to_json") else self.compaction,
            "steeringMode": self.steering_mode,
            "followUpMode": self.follow_up_mode,
            "toolExecution": self.tool_execution,
        }


@dataclass
class GenerationContext:
    """One assistant generation scope captured at its first attempt."""

    step_id: str = ""
    trigger_entry_id: str = ""
    configuration: Any = None
    stream_options: Any = None
    retry_policy: NormalizedRetryPolicy = field(default_factory=NormalizedRetryPolicy)
    overflow_recovery_used: bool = False

    def to_json(self) -> dict:
        return {
            "stepId": self.step_id,
            "triggerEntryId": self.trigger_entry_id,
            "configuration": self.configuration,
            "streamOptions": self.stream_options,
            "retryPolicy": self.retry_policy,
            "overflowRecoveryUsed": self.overflow_recovery_used,
        }


@dataclass
class RetryWait:
    """Shared backoff data for every retry-wait leaf."""

    next_attempt: int = 0
    not_before: int = 0
    error_message: str = ""

    def to_json(self) -> dict:
        return {
            "nextAttempt": self.next_attempt,
            "notBefore": self.not_before,
            "errorMessage": self.error_message,
        }


@dataclass
class Continuation:
    """What a checkpoint decided the run must do next."""

    kind: str  # "need_assistant" | "may_finish"
    overflow_recovery_used: bool = False
    include_final_assistant: bool = False

    def to_json(self) -> dict:
        if self.kind == "need_assistant":
            return {"kind": "need_assistant", "overflowRecoveryUsed": self.overflow_recovery_used}
        return {"kind": "may_finish", "includeFinalAssistant": self.include_final_assistant}


@dataclass
class CheckpointData:
    """Checkpoint payload; the flat leaf literal replaces the old nested phase tag."""

    continuation: Continuation = field(default_factory=lambda: Continuation(kind="may_finish"))
    trigger_entry_id: str = ""

    def to_json(self) -> dict:
        return {
            "continuation": self.continuation,
            "triggerEntryId": self.trigger_entry_id,
        }


@dataclass
class ResultBoundary:
    """Where a completed summary result is committed."""

    kind: str  # "resume_checkpoint" | "finish" | "commit_navigation"
    resume_after: Optional[CheckpointData] = None
    target_id: Optional[str] = None
    label: Optional[str] = None

    def to_json(self) -> dict:
        if self.kind == "resume_checkpoint":
            return {"kind": "resume_checkpoint", "resumeAfter": self.resume_after}
        if self.kind == "commit_navigation":
            data: dict = {"kind": "commit_navigation", "targetId": self.target_id}
            if self.label is not None:
                data["label"] = self.label
            return data
        return {"kind": "finish"}


@dataclass
class SummaryTask:
    """One structural summary task and where its result belongs."""

    task_id: str = ""
    reason: Optional[str] = None  # "manual" | "threshold" | "overflow"
    custom_instructions: Optional[str] = None
    boundary: ResultBoundary = field(default_factory=lambda: ResultBoundary(kind="finish"))

    def to_json(self) -> dict:
        data: dict = {"taskId": self.task_id, "boundary": self.boundary}
        if self.reason is not None:
            data["reason"] = self.reason
        if self.custom_instructions is not None:
            data["customInstructions"] = self.custom_instructions
        return data


@dataclass
class SummaryContext:
    """One structural summary generation scope."""

    result_entry_id: str = ""
    configuration: Any = None
    stream_options: Any = None
    retry_policy: NormalizedRetryPolicy = field(default_factory=NormalizedRetryPolicy)

    def to_json(self) -> dict:
        return {
            "resultEntryId": self.result_entry_id,
            "configuration": self.configuration,
            "streamOptions": self.stream_options,
            "retryPolicy": self.retry_policy,
        }


@dataclass
class ToolCall:
    """One tool call inside a tool batch, keyed by its assistant content index."""

    source_index: int = 0
    result_entry_id: str = ""
    status: str = "planned"  # "planned" | "effect_pending" | "outcome_ready" | "completed"
    replay: Optional[str] = None  # "never" | "safe"
    terminate: Optional[bool] = None

    def to_json(self) -> dict:
        data: dict = {
            "sourceIndex": self.source_index,
            "resultEntryId": self.result_entry_id,
            "status": self.status,
        }
        if self.status == "effect_pending":
            data["replay"] = self.replay
        elif self.status in ("outcome_ready", "completed"):
            data["terminate"] = bool(self.terminate)
        return data


@dataclass
class ToolBatch:
    """The durable child collection of one assistant turn's tool calls."""

    assistant_entry_id: str = ""
    configuration: Any = None
    turn_id: str = ""
    calls: List[ToolCall] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "assistantEntryId": self.assistant_entry_id,
            "configuration": self.configuration,
            "turnId": self.turn_id,
            "calls": list(self.calls),
        }


@dataclass
class OperationScope:
    """Uniform scope carried by every operation leaf."""

    control: Control = field(default_factory=Control)
    settings: RunSettings = field(default_factory=RunSettings)
    latest_assistant_entry_id: Optional[str] = None


def operation_scope_of(state: Any) -> dict:
    """Copy only the uniform operation scope when constructing a successor leaf."""
    return {
        "control": state.control,
        "settings": state.settings,
        "latest_assistant_entry_id": state.latest_assistant_entry_id,
    }


def _scope_json(state: Any) -> dict:
    return {
        "control": state.control,
        "settings": state.settings,
        "latestAssistantEntryId": state.latest_assistant_entry_id,
    }


@dataclass
class StartingOperation(OperationScope):
    at: str = field(default="starting", init=False)

    def to_json(self) -> dict:
        return {"at": self.at, **_scope_json(self)}


@dataclass
class CheckpointOperation(OperationScope):
    at: str = field(default="checkpoint", init=False)
    continuation: Continuation = field(default_factory=lambda: Continuation(kind="may_finish"))
    trigger_entry_id: str = ""

    def to_json(self) -> dict:
        return {
            "at": self.at,
            **_scope_json(self),
            "continuation": self.continuation,
            "triggerEntryId": self.trigger_entry_id,
        }


@dataclass
class AssistantReadyOperation(OperationScope):
    at: str = field(default="assistant.ready", init=False)
    generation_context: GenerationContext = field(default_factory=GenerationContext)
    next_attempt: int = 0

    def to_json(self) -> dict:
        return {
            "at": self.at,
            **_scope_json(self),
            "generationContext": self.generation_context,
            "nextAttempt": self.next_attempt,
        }


@dataclass
class AssistantEffectPendingOperation(OperationScope):
    at: str = field(default="assistant.effect_pending", init=False)
    generation_context: GenerationContext = field(default_factory=GenerationContext)
    attempt: int = 0
    response_entry_id: str = ""
    usage_id: str = ""
    intended_output_limit: int = 0
    context_window: int = 0

    def to_json(self) -> dict:
        return {
            "at": self.at,
            **_scope_json(self),
            "generationContext": self.generation_context,
            "attempt": self.attempt,
            "responseEntryId": self.response_entry_id,
            "usageId": self.usage_id,
            "intendedOutputLimit": self.intended_output_limit,
            "contextWindow": self.context_window,
        }


@dataclass
class AssistantRetryWaitOperation(OperationScope):
    at: str = field(default="assistant.retry_wait", init=False)
    generation_context: GenerationContext = field(default_factory=GenerationContext)
    next_attempt: int = 0
    not_before: int = 0
    error_message: str = ""

    def to_json(self) -> dict:
        return {
            "at": self.at,
            **_scope_json(self),
            "generationContext": self.generation_context,
            "nextAttempt": self.next_attempt,
            "notBefore": self.not_before,
            "errorMessage": self.error_message,
        }


@dataclass
class ToolsOperation(OperationScope):
    at: str = field(default="tools", init=False)
    batch: ToolBatch = field(default_factory=ToolBatch)

    def to_json(self) -> dict:
        return {"at": self.at, **_scope_json(self), "batch": self.batch}


@dataclass
class DeferredSuspendedOperation(OperationScope):
    at: str = field(default="deferred.suspended", init=False)
    step_id: str = ""
    source_entry_id: str = ""
    poll: int = 0
    configuration: Any = None
    stream_options: Any = None

    def to_json(self) -> dict:
        return {
            "at": self.at,
            **_scope_json(self),
            "stepId": self.step_id,
            "sourceEntryId": self.source_entry_id,
            "poll": self.poll,
            "configuration": self.configuration,
            "streamOptions": self.stream_options,
        }


@dataclass
class DeferredEffectPendingOperation(DeferredSuspendedOperation):
    at: str = field(default="deferred.effect_pending", init=False)
    response_entry_id: str = ""
    usage_id: str = ""

    def to_json(self) -> dict:
        return {
            **super().to_json(),
            "responseEntryId": self.response_entry_id,
            "usageId": self.usage_id,
        }


@dataclass
class SummaryDecidingOperation(OperationScope):
    at: str = field(default="summary.deciding", init=False)
    task: SummaryTask = field(default_factory=SummaryTask)

    def to_json(self) -> dict:
        return {"at": self.at, **_scope_json(self), "task": self.task}


@dataclass
class SummaryReadyOperation(OperationScope):
    at: str = field(default="summary.ready", init=False)
    task: SummaryTask = field(default_factory=SummaryTask)
    summary_context: SummaryContext = field(default_factory=SummaryContext)
    next_attempt: int = 0

    def to_json(self) -> dict:
        return {
            "at": self.at,
            **_scope_json(self),
            "task": self.task,
            "summaryContext": self.summary_context,
            "nextAttempt": self.next_attempt,
        }


@dataclass
class SummaryEffectPendingOperation(OperationScope):
    at: str = field(default="summary.effect_pending", init=False)
    task: SummaryTask = field(default_factory=SummaryTask)
    summary_context: SummaryContext = field(default_factory=SummaryContext)
    attempt: int = 0
    request: Optional[dict] = None  # {index, usageId}
    usage_ids: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        data = {
            "at": self.at,
            **_scope_json(self),
            "task": self.task,
            "summaryContext": self.summary_context,
            "attempt": self.attempt,
            "usageIds": list(self.usage_ids),
        }
        if self.request is not None:
            data["request"] = self.request
        return data


@dataclass
class SummaryRetryWaitOperation(OperationScope):
    at: str = field(default="summary.retry_wait", init=False)
    task: SummaryTask = field(default_factory=SummaryTask)
    summary_context: SummaryContext = field(default_factory=SummaryContext)
    next_attempt: int = 0
    not_before: int = 0
    error_message: str = ""

    def to_json(self) -> dict:
        return {
            "at": self.at,
            **_scope_json(self),
            "task": self.task,
            "summaryContext": self.summary_context,
            "nextAttempt": self.next_attempt,
            "notBefore": self.not_before,
            "errorMessage": self.error_message,
        }


@dataclass
class NavigationReadyToCommitOperation(OperationScope):
    at: str = field(default="navigation.ready_to_commit", init=False)
    #: Unsummarized navigation may target the branch root (null).
    target_id: Optional[str] = None
    label: Optional[str] = None

    def to_json(self) -> dict:
        data = {"at": self.at, **_scope_json(self), "targetId": self.target_id}
        if self.label is not None:
            data["label"] = self.label
        return data


#: Flat durable operation state: exactly 13 family-neutral dispatcher leaves.
OperationState = Any

#: The discriminator of one operation state leaf.
OperationAt = str


@dataclass
class OperationIntent:
    """Why an operation was accepted, discriminated by ``kind``.

    Mirrors the TS intent union; ``to_json`` emits only the keys the matching
    union member carries so durable records stay byte-comparable with the
    TypeScript writer.
    """

    kind: str = "run"  # "run" | "compaction" | "navigation"
    prompt_entry_ids: List[str] = field(default_factory=list)
    custom_instructions: Optional[str] = None
    target_id: Optional[str] = None
    summarize: bool = False
    label: Optional[str] = None

    def to_json(self) -> dict:
        if self.kind == "run":
            return {"kind": "run", "promptEntryIds": list(self.prompt_entry_ids)}
        if self.kind == "compaction":
            data: dict = {"kind": "compaction"}
            if self.custom_instructions is not None:
                data["customInstructions"] = self.custom_instructions
            return data
        data = {
            "kind": "navigation",
            "targetId": self.target_id,
            "summarize": self.summarize,
        }
        if self.label is not None:
            data["label"] = self.label
        if self.custom_instructions is not None:
            data["customInstructions"] = self.custom_instructions
        return data


@dataclass
class OperationMeta:
    """Immutable acceptance record for one operation."""

    operation_id: str = ""
    lane: str = ""
    source_tip_id: Optional[str] = None
    started_at: int = 0
    intent: OperationIntent = field(default_factory=OperationIntent)

    def to_json(self) -> dict:
        return {
            "operationId": self.operation_id,
            "lane": self.lane,
            "sourceTipId": self.source_tip_id,
            "startedAt": self.started_at,
            "intent": self.intent,
        }


@dataclass
class Operation:
    """One durable operation: its immutable metadata plus its current state."""

    meta: OperationMeta = field(default_factory=OperationMeta)
    state: OperationState = None


_OPERATION_STATE_LEAVES = {
    "starting": StartingOperation,
    "checkpoint": CheckpointOperation,
    "assistant.ready": AssistantReadyOperation,
    "assistant.effect_pending": AssistantEffectPendingOperation,
    "assistant.retry_wait": AssistantRetryWaitOperation,
    "tools": ToolsOperation,
    "deferred.suspended": DeferredSuspendedOperation,
    "deferred.effect_pending": DeferredEffectPendingOperation,
    "summary.deciding": SummaryDecidingOperation,
    "summary.ready": SummaryReadyOperation,
    "summary.effect_pending": SummaryEffectPendingOperation,
    "summary.retry_wait": SummaryRetryWaitOperation,
    "navigation.ready_to_commit": NavigationReadyToCommitOperation,
}


def _read(value: Any, snake: str, camel: str, default: Any = None) -> Any:
    """Read one durable field from a dataclass instance or a JSON-replayed mapping."""
    if value is None:
        return default
    if isinstance(value, dict):
        if camel in value:
            return value[camel]
        if snake in value:
            return value[snake]
        return default
    return getattr(value, snake, default)


def _revive_control(value: Any) -> Control:
    if isinstance(value, Control):
        return value
    return Control(
        status=_read(value, "status", "status", "running"),
        requested_at=_read(value, "requested_at", "requestedAt"),
    )


def _revive_run_settings(value: Any) -> RunSettings:
    if isinstance(value, RunSettings):
        return value
    return RunSettings(
        compaction=_read(value, "compaction", "compaction"),
        steering_mode=_read(value, "steering_mode", "steeringMode", "one-at-a-time"),
        follow_up_mode=_read(value, "follow_up_mode", "followUpMode", "one-at-a-time"),
        tool_execution=_read(value, "tool_execution", "toolExecution", "parallel"),
    )


def _revive_retry_policy(value: Any) -> NormalizedRetryPolicy:
    if isinstance(value, NormalizedRetryPolicy):
        return value
    return NormalizedRetryPolicy(
        max_attempts=_read(value, "max_attempts", "maxAttempts", 0),
        base_delay_ms=_read(value, "base_delay_ms", "baseDelayMs", 0),
        max_agent_delay_ms=_read(value, "max_agent_delay_ms", "maxAgentDelayMs", 0),
    )


def _revive_generation_context(value: Any) -> GenerationContext:
    if isinstance(value, GenerationContext):
        return value
    return GenerationContext(
        step_id=_read(value, "step_id", "stepId", ""),
        trigger_entry_id=_read(value, "trigger_entry_id", "triggerEntryId", ""),
        configuration=LaneConfiguration.from_value(
            _read(value, "configuration", "configuration")
        ),
        stream_options=_read(value, "stream_options", "streamOptions"),
        retry_policy=_revive_retry_policy(_read(value, "retry_policy", "retryPolicy")),
        overflow_recovery_used=bool(_read(value, "overflow_recovery_used", "overflowRecoveryUsed", False)),
    )


def _revive_summary_context(value: Any) -> SummaryContext:
    if isinstance(value, SummaryContext):
        return value
    return SummaryContext(
        result_entry_id=_read(value, "result_entry_id", "resultEntryId", ""),
        configuration=LaneConfiguration.from_value(
            _read(value, "configuration", "configuration")
        ),
        stream_options=_read(value, "stream_options", "streamOptions"),
        retry_policy=_revive_retry_policy(_read(value, "retry_policy", "retryPolicy")),
    )


def _revive_continuation(value: Any) -> Continuation:
    if isinstance(value, Continuation):
        return value
    return Continuation(
        kind=_read(value, "kind", "kind", "may_finish"),
        overflow_recovery_used=bool(_read(value, "overflow_recovery_used", "overflowRecoveryUsed", False)),
        include_final_assistant=bool(
            _read(value, "include_final_assistant", "includeFinalAssistant", False)
        ),
    )


def _revive_checkpoint_data(value: Any) -> CheckpointData:
    if isinstance(value, CheckpointData):
        return value
    return CheckpointData(
        continuation=_revive_continuation(_read(value, "continuation", "continuation")),
        trigger_entry_id=_read(value, "trigger_entry_id", "triggerEntryId", ""),
    )


def _revive_boundary(value: Any) -> ResultBoundary:
    if isinstance(value, ResultBoundary):
        return value
    resume_after = _read(value, "resume_after", "resumeAfter")
    return ResultBoundary(
        kind=_read(value, "kind", "kind", "finish"),
        resume_after=None if resume_after is None else _revive_checkpoint_data(resume_after),
        target_id=_read(value, "target_id", "targetId"),
        label=_read(value, "label", "label"),
    )


def _revive_summary_task(value: Any) -> SummaryTask:
    if isinstance(value, SummaryTask):
        return value
    return SummaryTask(
        task_id=_read(value, "task_id", "taskId", ""),
        reason=_read(value, "reason", "reason"),
        custom_instructions=_read(value, "custom_instructions", "customInstructions"),
        boundary=_revive_boundary(_read(value, "boundary", "boundary")),
    )


def _revive_tool_call(value: Any) -> ToolCall:
    if isinstance(value, ToolCall):
        return value
    return ToolCall(
        source_index=_read(value, "source_index", "sourceIndex", 0),
        result_entry_id=_read(value, "result_entry_id", "resultEntryId", ""),
        status=_read(value, "status", "status", "planned"),
        replay=_read(value, "replay", "replay"),
        terminate=_read(value, "terminate", "terminate"),
    )


def _revive_tool_batch(value: Any) -> ToolBatch:
    if isinstance(value, ToolBatch):
        return value
    calls = _read(value, "calls", "calls", []) or []
    return ToolBatch(
        assistant_entry_id=_read(value, "assistant_entry_id", "assistantEntryId", ""),
        configuration=LaneConfiguration.from_value(
            _read(value, "configuration", "configuration")
        ),
        turn_id=_read(value, "turn_id", "turnId", ""),
        calls=[_revive_tool_call(call) for call in calls],
    )


def revive_operation_meta(value: Any) -> Optional[OperationMeta]:
    """Rebuild one durable operation metadata record.

    Returns ``None`` for an absent value so callers can distinguish a missing
    ``op.meta`` from a present-but-empty one, matching the TS read sites.
    """
    if value is None:
        return None
    if isinstance(value, OperationMeta):
        return value
    intent = _read(value, "intent", "intent")
    return OperationMeta(
        operation_id=_read(value, "operation_id", "operationId", ""),
        lane=_read(value, "lane", "lane", ""),
        source_tip_id=_read(value, "source_tip_id", "sourceTipId"),
        started_at=_read(value, "started_at", "startedAt", 0),
        intent=OperationIntent(
            kind=_read(intent, "kind", "kind", "run"),
            prompt_entry_ids=list(_read(intent, "prompt_entry_ids", "promptEntryIds", []) or []),
            custom_instructions=_read(intent, "custom_instructions", "customInstructions"),
            target_id=_read(intent, "target_id", "targetId"),
            summarize=bool(_read(intent, "summarize", "summarize", False)),
            label=_read(intent, "label", "label"),
        ),
    )


def revive_operation_state(value: Any) -> Optional[Any]:
    """Rebuild one durable operation state leaf from its ``at`` discriminator."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return value
    at = value.get("at")
    leaf = _OPERATION_STATE_LEAVES.get(at)
    if leaf is None:
        raise ValueError(f"Unknown operation state discriminator: {at!r}")

    scope = {
        "control": _revive_control(value.get("control")),
        "settings": _revive_run_settings(value.get("settings")),
        "latest_assistant_entry_id": value.get("latestAssistantEntryId"),
    }
    if leaf is StartingOperation:
        return StartingOperation(**scope)
    if leaf is CheckpointOperation:
        return CheckpointOperation(
            **scope,
            continuation=_revive_continuation(value.get("continuation")),
            trigger_entry_id=value.get("triggerEntryId", ""),
        )
    if leaf is AssistantReadyOperation:
        return AssistantReadyOperation(
            **scope,
            generation_context=_revive_generation_context(value.get("generationContext")),
            next_attempt=value.get("nextAttempt", 0),
        )
    if leaf is AssistantEffectPendingOperation:
        return AssistantEffectPendingOperation(
            **scope,
            generation_context=_revive_generation_context(value.get("generationContext")),
            attempt=value.get("attempt", 0),
            response_entry_id=value.get("responseEntryId", ""),
            usage_id=value.get("usageId", ""),
            intended_output_limit=value.get("intendedOutputLimit", 0),
            context_window=value.get("contextWindow", 0),
        )
    if leaf is AssistantRetryWaitOperation:
        return AssistantRetryWaitOperation(
            **scope,
            generation_context=_revive_generation_context(value.get("generationContext")),
            next_attempt=value.get("nextAttempt", 0),
            not_before=value.get("notBefore", 0),
            error_message=value.get("errorMessage", ""),
        )
    if leaf is ToolsOperation:
        return ToolsOperation(**scope, batch=_revive_tool_batch(value.get("batch")))
    if leaf is DeferredSuspendedOperation:
        return DeferredSuspendedOperation(
            **scope,
            step_id=value.get("stepId", ""),
            source_entry_id=value.get("sourceEntryId", ""),
            poll=value.get("poll", 0),
            configuration=LaneConfiguration.from_value(value.get("configuration")),
            stream_options=value.get("streamOptions"),
        )
    if leaf is DeferredEffectPendingOperation:
        return DeferredEffectPendingOperation(
            **scope,
            step_id=value.get("stepId", ""),
            source_entry_id=value.get("sourceEntryId", ""),
            poll=value.get("poll", 0),
            configuration=LaneConfiguration.from_value(value.get("configuration")),
            stream_options=value.get("streamOptions"),
            response_entry_id=value.get("responseEntryId", ""),
            usage_id=value.get("usageId", ""),
        )
    if leaf is SummaryDecidingOperation:
        return SummaryDecidingOperation(**scope, task=_revive_summary_task(value.get("task")))
    if leaf is SummaryReadyOperation:
        return SummaryReadyOperation(
            **scope,
            task=_revive_summary_task(value.get("task")),
            summary_context=_revive_summary_context(value.get("summaryContext")),
            next_attempt=value.get("nextAttempt", 0),
        )
    if leaf is SummaryEffectPendingOperation:
        return SummaryEffectPendingOperation(
            **scope,
            task=_revive_summary_task(value.get("task")),
            summary_context=_revive_summary_context(value.get("summaryContext")),
            attempt=value.get("attempt", 0),
            request=value.get("request"),
            usage_ids=list(value.get("usageIds") or []),
        )
    if leaf is SummaryRetryWaitOperation:
        return SummaryRetryWaitOperation(
            **scope,
            task=_revive_summary_task(value.get("task")),
            summary_context=_revive_summary_context(value.get("summaryContext")),
            next_attempt=value.get("nextAttempt", 0),
            not_before=value.get("notBefore", 0),
            error_message=value.get("errorMessage", ""),
        )
    return NavigationReadyToCommitOperation(
        **scope,
        target_id=value.get("targetId"),
        label=value.get("label"),
    )
