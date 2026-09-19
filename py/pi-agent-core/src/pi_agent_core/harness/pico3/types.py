"""Records, documents, storage, kinds, transactions and runtime types ported from
``harness/pico3/types.ts``.

Durable model: four tables (conversations, entries, tasks, inputs) and three
documents (session, per-conversation rewindable, per-conversation sticky). Every
durable position is strict JSON; nothing durable is arbitrary Python.

Port notes:

* Data records are ``@dataclass``; TypeScript's structural, method-bearing
  interfaces (``Storage``, ``TxReads``, ``Runtime``, ``ToolDeclaration``, ...) are
  ``Protocol`` classes because Python has no interface/type-level shape that a
  dataclass can express for behaviour.
* Discriminated unions are single dataclasses with a leading defaulted
  ``type``/``kind``/``doc`` field, so callers may omit the discriminator and
  still switch on it exactly like the TypeScript union.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, Union

from ..._chord.context import Context
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    DeferredHandle,
    ImageContent,
    JsonObject,
    JsonValue,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from .delta import Op

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "JsonObject",
    "Stored",
    "Id",
    "Seq",
    "to_stored",
    # records
    "Conversation",
    "ConversationParent",
    "SystemMessage",
    "Message",
    "StoredMessage",
    "RequestMessage",
    "Model",
    "ContextEdit",
    "Entry",
    "NewEntry",
    "EntryKind",
    "define_entry",
    "Checkpoint",
    "Completion",
    "Outcome",
    "Task",
    "TaskPatch",
    "Input",
    # documents
    "ModelRef",
    "RetryPolicy",
    "ToolSlot",
    "TurnState",
    "PluginSlices",
    "RewindableState",
    "StickyState",
    "SessionState",
    "QueuedInput",
    "DocRef",
    "NamespaceDefaults",
    "Namespace",
    "NamespaceRegistration",
    "memo_once",
    "new_rewindable_state",
    "new_sticky_state",
    "new_session_state",
    "doc_key",
    # config facade
    "ConfigFacade",
    # storage
    "Write",
    "EntryScan",
    "TaskScan",
    "Storage",
    # kinds
    "HookResult",
    "HookApi",
    "HookInfo",
    "HookBinding",
    "HookRunner",
    "Step",
    "Closure",
    "AbortClosure",
    "PhaseHandler",
    "ConfigShape",
    "KindConfig",
    "KindTypes",
    "Kind",
    "CoreKind",
    "define_task",
    "AnyKind",
    "TaskRef",
    "EntryRef",
    "TaskSpec",
    # transactions
    "ConversationSpec",
    "OwnedConversationSpec",
    "UserInput",
    "SendInput",
    "ContextView",
    "GenerationStatus",
    "TurnView",
    "ConversationView",
    "ViewEvent",
    "Envelope",
    "TxReads",
    "SharedTx",
    "HostTx",
    "TaskTx",
    "CoreTx",
    "ReadAfterWrite",
    # runtime
    "RequestOptions",
    "Models",
    "Runtime",
    # tools
    "ToolControl",
    "ToolDiagnostic",
    "ToolResult",
    "ToolDeclaration",
    "AnyToolDeclaration",
    "OwnedConversation",
    "BeforeToolApi",
    "ToolApi",
    "ProcessSpec",
    "ProcessStatus",
    "ProcessHost",
    "PluginHandler",
    # invocation tokens
    "InvocationToken",
    "Invoker",
    # errors
    "Forbidden",
    "ConversationBusy",
    "GenerationInProgress",
    "CollapseInProgress",
    "Faulted",
    "Closed",
    "TaskContractFault",
    # record <-> JSON
    "UNSET",
    "conversation_to_json",
    "conversation_from_json",
    "context_edit_to_json",
    "context_edit_from_json",
    "entry_to_json",
    "entry_from_json",
    "new_entry_to_json",
    "new_entry_from_json",
    "task_to_json",
    "task_from_json",
    "task_patch_to_json",
    "task_patch_from_json",
    "input_to_json",
    "input_from_json",
    "doc_ref_to_json",
    "doc_ref_from_json",
    "write_to_json",
    "write_from_json",
    "queued_input_to_json",
    "queued_input_from_json",
    "tool_slot_to_json",
    "tool_slot_from_json",
    "view_event_to_json",
    "view_event_from_json",
]

# ---------------------------------------------------------------------------
# Strict durable JSON
# ---------------------------------------------------------------------------

JsonPrimitive = Union[None, bool, int, float, str]


def to_stored(value: Any) -> Any:
    """Coerce an in-memory value (e.g. a pi-ai message) to its stored representation."""
    if hasattr(value, "to_json"):
        return copy.deepcopy(value.to_json())
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return value.to_json()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


#: The JSON representation of a value: what survives ``to_stored``.
Stored = Any
Id = int
Seq = int


class _Unset:
    """Sentinel for an absent key, distinct from a stored ``null``."""

    _instance: Optional["_Unset"] = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"

    def __bool__(self) -> bool:
        return False


#: Marks a patch field the caller did not mention (``None`` clears).
UNSET = _Unset()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ConversationParent:
    """Where a conversation was forked from."""

    conversation_id: Id
    at: Id


@dataclass
class Conversation:
    id: Id
    parent: Optional[ConversationParent] = None
    owner: Optional[Id] = None
    sections: Optional[List[Any]] = None


@dataclass
class SystemMessage:
    """System instructions carried at their historical position inside ``messages``."""

    role: str = field(default="system", init=False)
    content: str = ""
    tools_added: Optional[List[Any]] = None
    tools_removed: Optional[List[Any]] = None
    timestamp: int = 0

    def to_json(self) -> JsonObject:
        data: JsonObject = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.tools_added is not None:
            data["toolsAdded"] = to_stored(self.tools_added)
        if self.tools_removed is not None:
            data["toolsRemoved"] = to_stored(self.tools_removed)
        return data

    @staticmethod
    def from_json(data: JsonObject) -> "SystemMessage":
        return SystemMessage(
            content=str(data.get("content", "")),
            tools_added=data.get("toolsAdded"),
            tools_removed=data.get("toolsRemoved"),
            timestamp=int(data.get("timestamp", 0)),
        )


#: Model messages plus the pico3 system message.
Message = Any
StoredMessage = Any
#: Alias kept for the TypeScript name used at request boundaries.
RequestMessage = StoredMessage
Model = Any


@dataclass
class ContextEdit:
    target: Id
    action: str = "omit"  # "omit" | "replace"
    messages: Optional[List[StoredMessage]] = None


@dataclass
class Entry:
    id: Id
    conversation_id: Id
    kind: str
    model: Optional[List[StoredMessage]] = None
    data: Optional[JsonObject] = None
    head: Optional[Id] = None
    edits: Optional[List[ContextEdit]] = None
    by_task_id: Optional[Id] = None


@dataclass
class NewEntry:
    """An entry to append; ``head="self"`` refers to the entry being written."""

    kind: str
    model: Optional[List[StoredMessage]] = None
    data: Optional[JsonObject] = None
    head: Union[Id, str, None] = None
    edits: Optional[List[ContextEdit]] = None


@dataclass(frozen=True)
class EntryKind:
    """A typed witness for an entry kind."""

    kind: str

    def is_entry(self, entry: Optional[Entry]) -> bool:
        return entry is not None and entry.kind == self.kind


def define_entry(kind: str) -> EntryKind:
    """Author an entry kind. Names may not begin with ``pi.``."""
    if kind.startswith("pi."):
        raise ValueError(f'entry kind names beginning with "pi." are reserved: {kind}')
    return EntryKind(kind=kind)


Checkpoint = Dict[str, Any]


@dataclass
class Completion:
    """A task closure's resolution. ``result``/``failure`` default to ``UNSET``:
    the scheduler faults a closure that leaves out the payload its status promises."""

    status: str = "completed"  # "completed" | "failed"
    result: Any = UNSET
    failure: Any = UNSET


@dataclass
class Outcome:
    status: str = "orphaned"  # "completed" | "failed" | "aborted" | "orphaned" | "faulted"
    result: Any = None
    failure: Any = None
    error: Optional[str] = None


@dataclass
class Task:
    id: Id
    conversation_id: Id
    kind: str
    input: JsonValue = None
    status: str = "pending"  # "pending" | "running" | "terminal"
    checkpoint: Optional[Checkpoint] = None
    abort: bool = False
    outcome: Optional[Outcome] = None
    after: List[Id] = field(default_factory=list)
    owns: List[Id] = field(default_factory=list)
    background: bool = False
    slot: Optional[JsonObject] = None


@dataclass
class TaskPatch:
    id: Id
    status: Optional[str] = None
    checkpoint: Any = UNSET
    abort: Optional[bool] = None
    outcome: Optional[Outcome] = None
    owns: Optional[List[Id]] = None


@dataclass
class Input:
    id: Id
    conversation_id: Id
    request_id: Optional[str] = None
    status: str = "queued"  # "queued" | "placed" | "done" | "unanswered"
    entry: Optional[Id] = None
    answer: Optional[Id] = None
    reason: Optional[str] = None  # "aborted" | "stale" | "terminated" | "failed"
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@dataclass
class ModelRef:
    provider: str = ""
    model_id: str = ""

    def to_json(self) -> JsonObject:
        return {"provider": self.provider, "modelId": self.model_id}


@dataclass
class RetryPolicy:
    enabled: bool = True
    max_retries: int = 0
    base_delay_ms: int = 0
    max_agent_delay_ms: Optional[int] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "enabled": self.enabled,
            "maxRetries": self.max_retries,
            "baseDelayMs": self.base_delay_ms,
        }
        if self.max_agent_delay_ms is not None:
            data["maxAgentDelayMs"] = self.max_agent_delay_ms
        return data


@dataclass
class ToolSlot:
    """One running tool call, at the index of its call in the assistant message."""

    call_id: str = ""
    name: str = ""
    args: JsonValue = None
    status: str = "pending"  # "pending" | "running" | "done" | "error" | "aborted"
    waiting_on: Optional[str] = None
    output: Optional[str] = None
    progress: Optional[str] = None
    details: Optional[JsonValue] = None
    continued_by: Optional[Id] = None
    entry: Optional[Id] = None
    memos: Optional[Dict[str, JsonValue]] = None


@dataclass
class TurnState:
    """The current turn, shaped for rendering."""

    message: Optional[Stored[AssistantMessage]] = None
    tools: List[ToolSlot] = field(default_factory=list)


PluginSlices = Dict[str, JsonObject]
RewindableState = Dict[str, Any]
StickyState = Dict[str, Any]
SessionState = Dict[str, Any]


@dataclass
class QueuedInput:
    id: Id
    mode: str = "steer"  # "steer" | "followUp" | "write"
    input: Optional[Stored[Any]] = None
    entry: Optional[Stored[Any]] = None


@dataclass
class DocRef:
    doc: str = "session"  # "session" | "rewindable" | "sticky"
    conversation_id: Optional[Id] = None


def doc_key(ref: DocRef) -> str:
    """Stable string key for one document reference."""
    if ref.doc == "session":
        return "session"
    return f"{ref.doc}:{ref.conversation_id}"


@dataclass
class NamespaceDefaults:
    rewindable: Optional[JsonObject] = None
    sticky: Optional[JsonObject] = None
    session: Optional[JsonObject] = None


@dataclass
class Namespace:
    """Current process authority for one durable namespace string."""

    id: str
    value_type: Optional[Callable[[Any], Any]] = None
    unregister: Optional[Callable[[], None]] = None


@dataclass
class NamespaceRegistration:
    token: object
    defaults: NamespaceDefaults
    routes: Dict[str, str]
    project: Optional[Callable[[JsonObject], JsonValue]] = None


def memo_once(slot: JsonObject, key: str, candidate: JsonValue) -> JsonValue:
    """First writer wins, including when the stored winner is ``None``."""
    memos = slot.get("memos")
    if not isinstance(memos, dict):
        memos = {}
        slot["memos"] = memos
    if key in memos:
        return memos[key]
    memos[key] = candidate
    return candidate


def new_rewindable_state() -> RewindableState:
    """The built-in rewindable slice, matching the core namespace defaults."""
    return {
        "thinkingLevel": "off",
        "selectedTools": [],
        "profile": "default",
        "threshold": 0.8,
        "keepRecent": 0,
        "plugins": {},
    }


def new_sticky_state() -> StickyState:
    """The built-in sticky slice, matching the core namespace defaults."""
    from .system import DEFAULT_RETRY_POLICY

    return {
        "retry": DEFAULT_RETRY_POLICY,
        "steeringMode": "one-at-a-time",
        "followUpMode": "one-at-a-time",
        "inbox": [],
        "turn": {"tools": []},
        "tasks": {},
        "plugins": {},
    }


def new_session_state() -> SessionState:
    return {"plugins": {}}


class ConfigFacade(Protocol):
    """The conversation-level configuration facade over every registered kind."""

    async def get(self, ctx: Context) -> Dict[str, JsonValue | None]: ...

    async def set(self, patch: Dict[str, JsonValue], ctx: Context) -> None: ...

    async def reset(self, keys: List[str], ctx: Context) -> None: ...


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@dataclass
class Write:
    """One durable write. ``type`` selects which payload field carries it."""

    type: str = "conversation"  # "conversation" | "entry" | "task" | "task.patch" | "input" | "doc"
    conversation: Optional[Conversation] = None
    entry: Optional[Entry] = None
    task: Optional[Task] = None
    patch: Optional[TaskPatch] = None
    input: Optional[Input] = None
    ref: Optional[DocRef] = None
    ops: List[Op] = field(default_factory=list)


@dataclass
class EntryScan:
    conversation_id: Id
    kind: Optional[str] = None
    with_head: bool = False
    before: Optional[Id] = None
    limit: int = 256


@dataclass
class TaskScan:
    conversation_id: Optional[Id] = None
    status: Optional[List[str]] = None
    kind: Optional[str] = None


class Storage(Protocol):
    """Six write types, a handful of reads. One owning session per instance."""

    async def commit(self, writes: List[Write], ctx: Context) -> Seq: ...

    def mint_id(self) -> Id: ...

    async def conversation(self, id: Id, ctx: Context) -> Optional[Conversation]: ...

    async def conversations(self, ctx: Context) -> List[Conversation]: ...

    async def entries(self, ids: List[Id], ctx: Context) -> Dict[Id, Entry]: ...

    async def scan_entries(self, scan: EntryScan, ctx: Context) -> List[Entry]: ...

    async def task(self, id: Id, ctx: Context) -> Optional[Task]: ...

    async def scan_tasks(self, scan: TaskScan, ctx: Context) -> List[Task]: ...

    async def input(self, id: Id, ctx: Context) -> Optional[Input]: ...

    async def input_by_request(
        self, conversation_id: Id, request_id: str, ctx: Context
    ) -> Optional[Input]: ...

    async def doc(self, ref: DocRef, ctx: Context) -> Optional[JsonObject]: ...

    async def doc_as_of(
        self, conversation_id: Id, at: Id, ctx: Context
    ) -> Optional[JsonObject]: ...

    async def truncate(self, ref: DocRef, ctx: Context) -> None: ...

    async def close(self, ctx: Context) -> None: ...


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------

#: A hook may synchronously or asynchronously omit its replacement value.
HookResult = Any


@dataclass
class HookApi:
    kind: str = ""
    task_id: Id = 0
    conversation_id: Id = 0


#: Kept as the concise name used by existing kind hook declarations.
HookInfo = HookApi


@dataclass
class HookBinding:
    handlers: Any
    namespace: Namespace
    api: HookApi


class HookRunner(Protocol):
    def handlers(self) -> List[HookBinding]: ...

    async def each(
        self,
        ctx: Context,
        fn: Callable[[Any, HookApi], HookResult],
        on_value: Optional[Callable[[Any], Any]] = None,
    ) -> None: ...


@dataclass
class Step:
    """What a phase handler returns: advance to a checkpoint, or finish."""

    next: Any = None
    done: Optional[Callable[..., Any]] = None


Closure = Callable[..., Any]
AbortClosure = Callable[..., Any]
PhaseHandler = Callable[..., Any]

ConfigShape = Dict[str, JsonValue]


@dataclass
class KindConfig:
    rewindable: Optional[ConfigShape] = None
    sticky: Optional[ConfigShape] = None


@dataclass
class KindTypes:
    """Phantom carrier so a kind's types can be read back out. Never set at runtime."""

    input: Any = None
    checkpoint: Any = None
    result: Any = None
    failure: Any = None
    aborted: Any = None
    hooks: Any = None
    config: Any = None
    slot: Any = None


@dataclass
class Kind:
    """A task kind: ``initial`` runs when there is no checkpoint, ``phases`` is exhaustive."""

    name: str
    turn: bool = False
    config: Optional[KindConfig] = None
    slot: Optional[Callable[[Any], JsonObject]] = None
    describe: Any = None
    inflight: Optional[List[str]] = None
    initial: Any = None
    phases: Dict[str, Any] = field(default_factory=dict)
    abort: Any = None
    types: Optional[KindTypes] = None


#: Internal kind type for fixed core machinery.
CoreKind = Kind


def define_task(definition: Kind) -> Kind:
    """Author an ordinary kind. Names may not begin with ``pi.``."""
    if definition.name.startswith("pi."):
        raise ValueError(f'task kind names beginning with "pi." are reserved: {definition.name}')
    return definition


#: The erased kind the kernel stores.
AnyKind = Kind


@dataclass
class TaskRef:
    id: Id
    kind: Any


@dataclass
class EntryRef:
    id: Id
    kind: EntryKind


@dataclass
class TaskSpec:
    kind: str
    input: JsonValue = None
    conversation_id: Optional[Id] = None
    after: Optional[List[Id]] = None
    background: bool = False


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


@dataclass
class ConversationSpec:
    parent: Optional[Dict[str, Any]] = None
    rewindable: Optional[JsonObject] = None
    sticky: Optional[JsonObject] = None
    sections: Optional[List[Any]] = None


@dataclass
class OwnedConversationSpec:
    inherit: bool = False
    rewindable: Optional[JsonObject] = None
    sticky: Optional[JsonObject] = None


UserInput = Any


@dataclass
class SendInput:
    content: UserInput = None
    request_id: Optional[str] = None
    when_busy: Optional[str] = None  # "steer" | "followUp" | "reject"


@dataclass
class ContextView:
    head: Optional[Entry] = None
    entries: List[Entry] = field(default_factory=list)
    messages: List[StoredMessage] = field(default_factory=list)


@dataclass
class GenerationStatus:
    """Generation progress. ``stage`` selects which fields are meaningful."""

    stage: str = "preparing"  # waiting | preparing | requesting | streaming | retrying | deferred
    on: Optional[str] = None
    attempt: int = 0
    retry_at: Optional[int] = None
    last_error: Optional[str] = None
    poll_at: Optional[int] = None


@dataclass
class TurnView:
    inputs: List[Id] = field(default_factory=list)
    generation: Optional[GenerationStatus] = None
    message: Optional[Stored[AssistantMessage]] = None
    tools: List[ToolSlot] = field(default_factory=list)


@dataclass
class ConversationView:
    conversation: Optional[Dict[str, Any]] = None
    entries: List[Entry] = field(default_factory=list)
    config: Dict[str, JsonValue | None] = field(default_factory=dict)
    inbox: List[QueuedInput] = field(default_factory=list)
    turn: Optional[TurnView] = None
    compaction: Optional[Dict[str, Any]] = None
    tasks: Dict[str, Any] = field(default_factory=dict)
    plugins: Dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class ViewEvent:
    """Durable, JSON-serializable view notification (the TS discriminated union)."""

    type: str = ""
    entry: Optional[Entry] = None
    inputs: Optional[List[Id]] = None
    status: Optional[str] = None
    answer: Optional[Id] = None
    reason: Optional[str] = None
    detail: Optional[str] = None
    input: Optional[Id] = None
    mode: Optional[str] = None
    entry_id: Optional[Id] = None
    task_id: Optional[Id] = None
    attempt: Optional[int] = None
    retry_at: Optional[int] = None
    error: Optional[str] = None
    poll_at: Optional[int] = None
    tool_calls: Optional[int] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    is_error: Optional[bool] = None
    control: Optional["ToolControl"] = None
    source: Optional[str] = None
    message: Optional[str] = None
    keys: Optional[List[str]] = None
    data: Optional[JsonValue] = None
    through: Optional[Id] = None
    summary: Optional[Id] = None
    kind: Optional[str] = None
    outcome: Optional[str] = None
    background: Optional[bool] = None


@dataclass
class Envelope:
    revision: int = 0
    ops: List[Op] = field(default_factory=list)
    events: List[ViewEvent] = field(default_factory=list)


class TxReads(Protocol):
    """Reads available to every view. Scans reject after a same-batch write."""

    async def conversation(self, id: Id) -> Optional[Conversation]: ...

    async def entry(self, id: Id) -> Optional[Entry]: ...

    async def entries(self, ids: List[Id]) -> Dict[Id, Entry]: ...

    async def newest_entry(
        self, conversation_id: Id, opts: Optional[Dict[str, Any]] = None
    ) -> Optional[Entry]: ...

    async def scan_entries(self, scan: EntryScan) -> List[Entry]: ...

    async def context(self, conversation_id: Id, at: Optional[Id] = None) -> ContextView: ...

    async def task(self, id: Id) -> Optional[Task]: ...

    async def tasks(self, scan: TaskScan) -> List[Task]: ...

    async def input(self, id: Id) -> Optional[Input]: ...

    async def rewindable_as_of(
        self, conversation_id: Id, at: Id
    ) -> Optional[RewindableState]: ...

    def snapshot(self, ref: DocRef) -> Any: ...


class SharedTx(TxReads, Protocol):
    def plugins(self, namespace: Namespace) -> JsonObject: ...

    def emit(self, namespace: Namespace, name: str, data: JsonValue) -> None: ...

    def config(self, conversation_id: Id) -> Any: ...

    async def write(self, conversation_id: Id, entry: NewEntry) -> Id: ...

    def create_task(self, kind: Any, input: Any, opts: Optional[Dict[str, Any]] = None) -> TaskRef: ...

    def create_conversation(self, spec: ConversationSpec) -> Id: ...


class HostTx(SharedTx, Protocol):
    """What the host gets."""


class TaskTx(SharedTx, Protocol):
    """What an ordinary task gets: scoped shared operations plus its own checkpoint and slot."""

    def checkpoint(self, value: Checkpoint) -> None: ...

    def slot(self, ref: TaskRef) -> JsonObject: ...


class CoreTx(TaskTx, Protocol):
    """Internal full authority for the fixed turn machinery."""

    def rewindable(self, conversation_id: Id) -> RewindableState: ...

    def sticky(self, conversation_id: Id) -> StickyState: ...

    def session(self) -> SessionState: ...

    def tool_slot(self, task: Any) -> ToolSlot: ...

    def append_entry(self, conversation_id: Id, entry: Any, kind: Optional[EntryKind] = None) -> Any: ...

    async def send(self, conversation_id: Id, input: SendInput) -> Id: ...

    async def resolve_inputs(self, ids: List[Id], resolution: Any) -> None: ...

    async def boundary(
        self, conversation_id: Id, at: str, head_boundary: Optional[Id]
    ) -> Dict[str, Any]: ...

    async def create_owned_conversation(
        self, owner_task_id: Id, source_conversation_id: Id, spec: OwnedConversationSpec
    ) -> Id: ...

    def create_fork_conversation(self, spec: ConversationSpec) -> Id: ...

    def mark_task(self, id: Id) -> None: ...


class ReadAfterWrite(Exception):
    """Thrown when a scan-shaped read follows a same-batch write to its domain."""

    def __init__(self, read: str, write: str) -> None:
        super().__init__(
            f"{read} after {write} in the same transaction: "
            "the answer would not include the buffered write"
        )
        self.name = "ReadAfterWrite"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class RequestOptions:
    messages: List[StoredMessage] = field(default_factory=list)
    thinking_level: ThinkingLevel = "off"


class Models(Protocol):
    def resolve(self, ref: ModelRef) -> Optional[Model]: ...

    def stream(
        self, model: Model, request: RequestOptions, ctx: Context
    ) -> Any: ...

    async def fetch_deferred(
        self, model: Model, handle: DeferredHandle, ctx: Context
    ) -> Any: ...

    async def cancel_deferred(
        self, model: Model, handle: DeferredHandle, ctx: Context
    ) -> None: ...


class Runtime(Protocol):
    """What a kind's handlers get. Every commit checks the invocation token."""

    task_id: Id
    conversation_id: Id
    kind: AnyKind
    hooks: HookRunner
    models: Models
    tools: Dict[str, "ToolDeclaration"]
    registries: Any
    kinds: Dict[str, AnyKind]
    process_host: Optional["ProcessHost"]
    plugins: Dict[str, "PluginHandler"]

    async def commit(self, fn: Callable[..., Any], ctx: Context) -> Any: ...

    def now(self) -> int: ...

    async def sleep(self, until_ms: int, ctx: Context) -> None: ...

    async def wait_for_input(self, id: Id, ctx: Context) -> Input: ...

    async def wait_for_task(self, id: Id, ctx: Context) -> Task: ...

    async def abort_task(self, id: Id, ctx: Context) -> str: ...

    async def abort_conversation(self, id: Id, ctx: Context) -> None: ...

    async def create_owned_conversation(
        self, spec: OwnedConversationSpec, ctx: Context
    ) -> Id: ...

    async def send_owned(self, conversation_id: Id, input: SendInput, ctx: Context) -> Id: ...

    async def context(
        self, conversation_id: Id, at: Optional[Id], ctx: Context
    ) -> ContextView: ...

    async def newest_entry(
        self,
        conversation_id: Id,
        opts: Optional[Dict[str, Any]],
        ctx: Context,
    ) -> Optional[Entry]: ...

    async def rewindable(self, conversation_id: Id, ctx: Context) -> RewindableState: ...

    async def sticky(self, conversation_id: Id, ctx: Context) -> StickyState: ...

    async def rewindable_as_of(
        self, conversation_id: Id, at: Id, ctx: Context
    ) -> Optional[RewindableState]: ...


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass
class ToolControl:
    terminate: bool = False
    handoff: Optional[str] = None
    add_tools: Optional[List[str]] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"terminate": self.terminate}
        if self.handoff is not None:
            data["handoff"] = self.handoff
        if self.add_tools is not None:
            data["addTools"] = list(self.add_tools)
        return data


@dataclass
class ToolDiagnostic:
    severity: str = "info"  # "info" | "warn" | "error"
    message: str = ""
    code: Optional[str] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"severity": self.severity, "message": self.message}
        if self.code is not None:
            data["code"] = self.code
        return data


@dataclass
class ToolResult:
    content: Optional[List[Any]] = None
    is_error: Optional[bool] = None
    details: Optional[JsonValue] = None
    diagnostics: Optional[List[ToolDiagnostic]] = None
    control: Optional[ToolControl] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {}
        if self.content is not None:
            data["content"] = to_stored(self.content)
        if self.is_error is not None:
            data["isError"] = self.is_error
        if self.details is not None:
            data["details"] = to_stored(self.details)
        if self.diagnostics is not None:
            data["diagnostics"] = to_stored(self.diagnostics)
        if self.control is not None:
            data["control"] = to_stored(self.control)
        return data


class ToolDeclaration(Protocol):
    name: str
    description: str
    parameters: Any
    replay: Optional[str]
    output: Optional[Dict[str, Any]]

    async def execute(self, args: Any, api: "ToolApi", ctx: Context) -> ToolResult: ...


#: Erased declaration held by the registry.
AnyToolDeclaration = Any


class OwnedConversation(Protocol):
    """The narrow handle a task gets to a conversation it owns."""

    id: Id

    async def send(self, input: SendInput, ctx: Context) -> Any: ...

    async def abort(self, ctx: Context) -> None: ...


class BeforeToolApi(Protocol):
    kind: str
    task_id: Id
    conversation_id: Id
    call_id: str

    async def waiting(self, ctx: Context) -> None: ...

    async def memo(self, name: str, *args: Any) -> JsonValue: ...

    async def emit(self, name: str, data: JsonValue, ctx: Context) -> None: ...


class ToolApi(Protocol):
    task_id: Id
    conversation_id: Id
    call_id: str

    def stream(self, chunk: Any) -> None: ...

    async def progress(self, update: Callable[[ToolSlot], None], ctx: Context) -> None: ...

    async def memo(self, name: str, *args: Any) -> JsonValue: ...

    async def conversation(
        self, spec: OwnedConversationSpec, ctx: Context
    ) -> OwnedConversation: ...

    async def task(
        self, kind: Any, input: Any, opts: Dict[str, Any], ctx: Context
    ) -> TaskRef: ...

    async def get_task(self, ref: TaskRef, ctx: Context) -> Optional[Task]: ...

    async def wait_for_task(self, ref: TaskRef, ctx: Context) -> Task: ...

    async def slot(self, ref: TaskRef, ctx: Context) -> Optional[JsonObject]: ...


@dataclass
class ProcessSpec:
    command: str = ""
    args: List[str] = field(default_factory=list)
    cwd: str = ""
    env: Optional[Dict[str, str]] = None


@dataclass
class ProcessStatus:
    status: str = "unknown"  # "running" | "exited" | "unknown"
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    dropped_stdout: int = 0
    dropped_stderr: int = 0


class ProcessHost(Protocol):
    async def start(self, key: str, spec: ProcessSpec, ctx: Context) -> None: ...

    async def status(self, key: str, ctx: Context) -> ProcessStatus: ...

    async def kill(self, key: str, signal: str, ctx: Context) -> None: ...


PluginHandler = Callable[[JsonValue, ToolApi, Context], Awaitable[JsonValue]]


# ---------------------------------------------------------------------------
# Invocation tokens and invokers
# ---------------------------------------------------------------------------


class InvocationToken:
    """An unforgeable capability: only the scheduler creates these, one per invocation."""

    __slots__ = ("task_id", "mode", "_alive")

    def __init__(self, task_id: Id, mode: str) -> None:
        self.task_id = task_id
        self.mode = mode
        self._alive = True

    @property
    def alive(self) -> bool:
        return self._alive

    def revoke(self) -> None:
        """Called by the scheduler when the invocation returns."""
        self._alive = False


@dataclass
class Invoker:
    """Who is invoking a transaction: the host, the kernel, or a task."""

    type: str = "kernel"  # "host" | "kernel" | "task"
    conversation_id: Optional[Id] = None
    token: Optional[InvocationToken] = None
    id: Optional[Id] = None
    kind: Optional[AnyKind] = None
    core: bool = False
    mode: str = "run"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class Forbidden(Exception):
    def __init__(self, what: str) -> None:
        super().__init__(f"forbidden: {what}")
        self.name = "Forbidden"


class ConversationBusy(Exception):
    def __init__(self, id: Id) -> None:
        super().__init__(f"conversation {id} is busy")
        self.name = "ConversationBusy"


class GenerationInProgress(Exception):
    def __init__(self, id: Id) -> None:
        super().__init__(f"conversation {id} already has a live generation")
        self.name = "GenerationInProgress"


class CollapseInProgress(Exception):
    def __init__(self, id: Id) -> None:
        super().__init__(f"conversation {id} already has a live collapse")
        self.name = "CollapseInProgress"


class Faulted(Exception):
    def __init__(self, cause: Any) -> None:
        super().__init__(f"Session faulted: {cause}")
        self.name = "Faulted"
        self.cause = cause


class Closed(Exception):
    def __init__(self) -> None:
        super().__init__("Session is closed")
        self.name = "Closed"


class TaskContractFault(Exception):
    def __init__(self, kind: str, what: str) -> None:
        super().__init__(f"kind {kind} broke its contract: {what}")
        self.name = "TaskContractFault"


# ---------------------------------------------------------------------------
# Record <-> JSON conversion. Durable records keep their camelCase wire shape.
# ---------------------------------------------------------------------------


def conversation_to_json(conversation: Conversation) -> JsonObject:
    data: JsonObject = {"id": conversation.id}
    if conversation.parent is not None:
        data["parent"] = {
            "conversationId": conversation.parent.conversation_id,
            "at": conversation.parent.at,
        }
    if conversation.owner is not None:
        data["owner"] = conversation.owner
    if conversation.sections is not None:
        data["sections"] = [
            item.to_json() if hasattr(item, "to_json") else item for item in conversation.sections
        ]
    return data


def conversation_from_json(data: JsonObject) -> Conversation:
    parent = data.get("parent")
    return Conversation(
        id=int(data.get("id")),
        parent=(
            ConversationParent(
                conversation_id=int(parent.get("conversationId")), at=int(parent.get("at"))
            )
            if isinstance(parent, dict)
            else None
        ),
        owner=data.get("owner"),
        sections=data.get("sections"),
    )


def context_edit_to_json(edit: ContextEdit) -> JsonObject:
    data: JsonObject = {"target": edit.target, "action": edit.action}
    if edit.messages is not None:
        data["messages"] = to_stored(edit.messages)
    return data


def context_edit_from_json(data: JsonObject) -> ContextEdit:
    return ContextEdit(
        target=int(data.get("target")),
        action=str(data.get("action", "omit")),
        messages=data.get("messages"),
    )


def entry_to_json(entry: Entry) -> JsonObject:
    data: JsonObject = {
        "id": entry.id,
        "conversationId": entry.conversation_id,
        "kind": entry.kind,
    }
    if entry.model is not None:
        data["model"] = to_stored(entry.model)
    if entry.data is not None:
        data["data"] = to_stored(entry.data)
    if entry.head is not None:
        data["head"] = entry.head
    if entry.edits is not None:
        data["edits"] = [context_edit_to_json(edit) for edit in entry.edits]
    if entry.by_task_id is not None:
        data["byTaskId"] = entry.by_task_id
    return data


def entry_from_json(data: JsonObject) -> Entry:
    edits = data.get("edits")
    return Entry(
        id=int(data.get("id")),
        conversation_id=int(data.get("conversationId")),
        kind=str(data.get("kind")),
        model=data.get("model"),
        data=data.get("data"),
        head=data.get("head"),
        edits=[context_edit_from_json(edit) for edit in edits] if edits else None,
        by_task_id=data.get("byTaskId"),
    )


def new_entry_to_json(entry: NewEntry) -> JsonObject:
    data: JsonObject = {"kind": entry.kind}
    if entry.model is not None:
        data["model"] = to_stored(entry.model)
    if entry.data is not None:
        data["data"] = to_stored(entry.data)
    if entry.head is not None:
        data["head"] = entry.head
    if entry.edits is not None:
        data["edits"] = [context_edit_to_json(edit) for edit in entry.edits]
    return data


def new_entry_from_json(data: JsonObject) -> NewEntry:
    edits = data.get("edits")
    return NewEntry(
        kind=str(data.get("kind")),
        model=data.get("model"),
        data=data.get("data"),
        head=data.get("head"),
        edits=[context_edit_from_json(edit) for edit in edits] if edits else None,
    )


def outcome_to_json(outcome: Optional[Outcome]) -> Optional[JsonObject]:
    if outcome is None:
        return None
    data: JsonObject = {"status": outcome.status}
    if outcome.result is not None and outcome.result is not UNSET:
        data["result"] = to_stored(outcome.result)
    if outcome.failure is not None and outcome.failure is not UNSET:
        data["failure"] = to_stored(outcome.failure)
    if outcome.error is not None:
        data["error"] = outcome.error
    return data


def outcome_from_json(data: Optional[JsonObject]) -> Optional[Outcome]:
    if data is None:
        return None
    if isinstance(data, Outcome):
        return data
    return Outcome(
        status=str(data.get("status")),
        result=data.get("result"),
        failure=data.get("failure"),
        error=data.get("error"),
    )


def completion_to_outcome(completion: Completion) -> Outcome:
    """A closure's completion IS the durable outcome in TypeScript; Python converts it."""
    return Outcome(
        status=completion.status,
        result=None if completion.result is UNSET else completion.result,
        failure=None if completion.failure is UNSET else completion.failure,
    )


def task_to_json(task: Task) -> JsonObject:
    data: JsonObject = {
        "id": task.id,
        "conversationId": task.conversation_id,
        "kind": task.kind,
        "input": to_stored(task.input),
        "status": task.status,
        "after": list(task.after),
        "owns": list(task.owns),
    }
    if task.checkpoint is not None:
        data["checkpoint"] = to_stored(task.checkpoint)
    if task.abort:
        data["abort"] = True
    outcome = outcome_to_json(task.outcome)
    if outcome is not None:
        data["outcome"] = outcome
    if task.background:
        data["background"] = True
    if task.slot is not None:
        data["slot"] = to_stored(task.slot)
    return data


def task_from_json(data: JsonObject) -> Task:
    return Task(
        id=int(data.get("id")),
        conversation_id=int(data.get("conversationId")),
        kind=str(data.get("kind")),
        input=data.get("input"),
        status=str(data.get("status", "pending")),
        checkpoint=data.get("checkpoint"),
        abort=bool(data.get("abort")),
        outcome=outcome_from_json(data.get("outcome")),
        after=[int(item) for item in data.get("after") or []],
        owns=[int(item) for item in data.get("owns") or []],
        background=bool(data.get("background")),
        slot=data.get("slot"),
    )


def task_patch_to_json(patch: TaskPatch) -> JsonObject:
    data: JsonObject = {"id": patch.id}
    if patch.status is not None:
        data["status"] = patch.status
    if patch.checkpoint is not UNSET:
        data["checkpoint"] = None if patch.checkpoint is None else to_stored(patch.checkpoint)
    if patch.abort is not None:
        data["abort"] = patch.abort
    if patch.outcome is not None:
        data["outcome"] = outcome_to_json(patch.outcome)
    if patch.owns is not None:
        data["owns"] = list(patch.owns)
    return data


def task_patch_from_json(data: JsonObject) -> TaskPatch:
    checkpoint: Any = UNSET
    if "checkpoint" in data:
        checkpoint = data["checkpoint"]
    return TaskPatch(
        id=int(data.get("id")),
        status=data.get("status"),
        checkpoint=checkpoint,
        abort=data.get("abort"),
        outcome=outcome_from_json(data.get("outcome")),
        owns=data.get("owns"),
    )


def input_to_json(input_record: Input) -> JsonObject:
    data: JsonObject = {
        "id": input_record.id,
        "conversationId": input_record.conversation_id,
        "status": input_record.status,
    }
    if input_record.request_id is not None:
        data["requestId"] = input_record.request_id
    if input_record.entry is not None:
        data["entry"] = input_record.entry
    if input_record.answer is not None:
        data["answer"] = input_record.answer
    if input_record.reason is not None:
        data["reason"] = input_record.reason
    if input_record.detail is not None:
        data["detail"] = input_record.detail
    return data


def input_from_json(data: JsonObject) -> Input:
    return Input(
        id=int(data.get("id")),
        conversation_id=int(data.get("conversationId")),
        request_id=data.get("requestId"),
        status=str(data.get("status", "queued")),
        entry=data.get("entry"),
        answer=data.get("answer"),
        reason=data.get("reason"),
        detail=data.get("detail"),
    )


def doc_ref_to_json(ref: DocRef) -> JsonObject:
    if ref.doc == "session":
        return {"doc": "session"}
    return {"doc": ref.doc, "conversationId": ref.conversation_id}


def doc_ref_from_json(data: JsonObject) -> DocRef:
    doc = str(data.get("doc", "session"))
    return DocRef(
        doc=doc,
        conversation_id=None if doc == "session" else data.get("conversationId"),
    )


def write_to_json(write: Write) -> JsonObject:
    data: JsonObject = {"type": write.type}
    if write.type == "conversation" and write.conversation is not None:
        data["conversation"] = conversation_to_json(write.conversation)
    elif write.type == "entry" and write.entry is not None:
        data["entry"] = entry_to_json(write.entry)
    elif write.type == "task" and write.task is not None:
        data["task"] = task_to_json(write.task)
    elif write.type == "task.patch" and write.patch is not None:
        data["patch"] = task_patch_to_json(write.patch)
    elif write.type == "input" and write.input is not None:
        data["input"] = input_to_json(write.input)
    elif write.type == "doc" and write.ref is not None:
        data["ref"] = doc_ref_to_json(write.ref)
        data["ops"] = [list(op) for op in write.ops]
    return data


def write_from_json(data: JsonObject) -> Write:
    write_type = str(data.get("type"))
    if write_type == "conversation":
        return Write(type="conversation", conversation=conversation_from_json(data.get("conversation")))
    if write_type == "entry":
        return Write(type="entry", entry=entry_from_json(data.get("entry")))
    if write_type == "task":
        return Write(type="task", task=task_from_json(data.get("task")))
    if write_type == "task.patch":
        return Write(type="task.patch", patch=task_patch_from_json(data.get("patch")))
    if write_type == "input":
        return Write(type="input", input=input_from_json(data.get("input")))
    if write_type == "doc":
        raw_ops = data.get("ops") or []
        return Write(
            type="doc",
            ref=doc_ref_from_json(data.get("ref")),
            ops=[tuple(op) for op in raw_ops],
        )
    raise ValueError(f"unknown write type {write_type!r}")


def queued_input_to_json(queued: QueuedInput) -> JsonObject:
    data: JsonObject = {"id": queued.id, "mode": queued.mode}
    if queued.input is not None:
        data["input"] = to_stored(queued.input)
    if queued.entry is not None:
        data["entry"] = to_stored(queued.entry)
    return data


def queued_input_from_json(data: JsonObject) -> QueuedInput:
    return QueuedInput(
        id=int(data.get("id")),
        mode=str(data.get("mode", "steer")),
        input=data.get("input"),
        entry=data.get("entry"),
    )


def tool_slot_to_json(slot: ToolSlot) -> JsonObject:
    data: JsonObject = {
        "callId": slot.call_id,
        "name": slot.name,
        "args": to_stored(slot.args),
        "status": slot.status,
    }
    if slot.waiting_on is not None:
        data["waitingOn"] = slot.waiting_on
    if slot.output is not None:
        data["output"] = slot.output
    if slot.progress is not None:
        data["progress"] = slot.progress
    if slot.details is not None:
        data["details"] = to_stored(slot.details)
    if slot.continued_by is not None:
        data["continuedBy"] = slot.continued_by
    if slot.entry is not None:
        data["entry"] = slot.entry
    if slot.memos is not None:
        data["memos"] = to_stored(slot.memos)
    return data


def tool_slot_from_json(data: JsonObject) -> ToolSlot:
    return ToolSlot(
        call_id=str(data.get("callId", "")),
        name=str(data.get("name", "")),
        args=data.get("args"),
        status=str(data.get("status", "pending")),
        waiting_on=data.get("waitingOn"),
        output=data.get("output"),
        progress=data.get("progress"),
        details=data.get("details"),
        continued_by=data.get("continuedBy"),
        entry=data.get("entry"),
        memos=data.get("memos"),
    )


def view_event_to_json(event: ViewEvent) -> JsonObject:
    data: JsonObject = {"type": event.type}
    for key, value in event.__dict__.items():
        if key == "type" or value is None:
            continue
        if hasattr(value, "to_json"):
            data[_CAMEL.get(key, key)] = value.to_json()
        else:
            data[_CAMEL.get(key, key)] = value
    return data


def view_event_from_json(data: JsonObject) -> ViewEvent:
    kwargs: Dict[str, Any] = {"type": str(data.get("type"))}
    for key, value in data.items():
        if key == "type":
            continue
        snake = _SNAKE_FIELDS.get(key)
        if snake is None:
            continue
        if snake == "entry" and isinstance(value, dict):
            kwargs[snake] = entry_from_json(value)
        elif snake == "control" and isinstance(value, dict):
            kwargs[snake] = ToolControl(
                terminate=bool(value.get("terminate")),
                handoff=value.get("handoff"),
                add_tools=value.get("addTools"),
            )
        else:
            kwargs[snake] = value
    return ViewEvent(**kwargs)


#: snake_case ViewEvent field -> durable camelCase key.
_CAMEL = {
    "entry_id": "entryId",
    "task_id": "taskId",
    "retry_at": "retryAt",
    "poll_at": "pollAt",
    "tool_calls": "toolCalls",
    "call_id": "callId",
    "is_error": "isError",
}

#: durable camelCase key -> snake_case ViewEvent field.
_SNAKE_FIELDS = {value: key for key, value in _CAMEL.items()}
_SNAKE_FIELDS.update(
    {
        "entry": "entry",
        "inputs": "inputs",
        "status": "status",
        "answer": "answer",
        "reason": "reason",
        "detail": "detail",
        "input": "input",
        "mode": "mode",
        "attempt": "attempt",
        "error": "error",
        "name": "name",
        "control": "control",
        "source": "source",
        "message": "message",
        "keys": "keys",
        "data": "data",
        "through": "through",
        "summary": "summary",
        "kind": "kind",
        "outcome": "outcome",
        "background": "background",
    }
)


# Unused imports kept explicit for type-checkers reading the port's surface.
_ = (
    AssistantMessage,
    AssistantMessageEvent,
    DeferredHandle,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    Tuple,
)
