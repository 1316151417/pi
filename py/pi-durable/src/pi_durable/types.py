"""Durable record contracts from ``packages/durable/src/types.ts``.

Stored dictionaries retain their TypeScript field names. Optional keys are
absent, rather than filled with null. Readonly is a static ownership contract;
the storage boundary produces detached dictionaries and arrays.
"""

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Never, NotRequired, Protocol, TypedDict

from pi_chord.context import Context
from pi_chord.types import JsonValue
from pi_chord._undefined import UNDEFINED, Undefined

if TYPE_CHECKING:
    from pi_ai.types import Message

type Id = int | float
type Seq = int | float
ROOT_CONVERSATION_ID: Id = 1


class StoredError(TypedDict):
    message: str
    detail: NotRequired[JsonValue]


class ConversationParent(TypedDict):
    conversationId: Id
    at: Id


class ConversationOwner(TypedDict):
    conversationId: Id
    taskId: Id


class ConversationRecord(TypedDict):
    id: Id
    parent: NotRequired[ConversationParent]
    owner: NotRequired[ConversationOwner]


class OmitContextEdit(TypedDict):
    target: Id
    action: Literal["omit"]
    messages: NotRequired[Never]


class ReplaceContextEdit(TypedDict):
    target: Id
    action: Literal["replace"]
    messages: Sequence["Message"]


type ContextEdit = OmitContextEdit | ReplaceContextEdit


class _EntryContent(TypedDict):
    kind: str
    model: NotRequired[Sequence["Message"]]
    data: NotRequired[JsonValue]
    edits: NotRequired[Sequence[ContextEdit]]


class _EntryIdentity(_EntryContent):
    id: Id
    conversationId: Id
    byTaskId: NotRequired[Id]


class EntryRecord(_EntryIdentity):
    head: NotRequired[Id]


class HeadEntryRecord(_EntryIdentity):
    head: Id


class EntryDraft(_EntryContent):
    head: NotRequired[Id | Literal["self"]]


class _InputBase(TypedDict):
    id: Id
    conversationId: Id
    requestId: NotRequired[str]


class QueuedInput(_InputBase):
    status: Literal["queued"]
    entry: NotRequired[Never]
    answer: NotRequired[Never]
    reason: NotRequired[Never]
    detail: NotRequired[Never]


class PlacedInput(_InputBase):
    status: Literal["placed"]
    entry: Id
    answer: NotRequired[Never]
    reason: NotRequired[Never]
    detail: NotRequired[Never]


class DoneInput(_InputBase):
    status: Literal["done"]
    entry: Id
    answer: NotRequired[Id]
    reason: NotRequired[Never]
    detail: NotRequired[Never]


class UnansweredInput(_InputBase):
    status: Literal["unanswered"]
    entry: NotRequired[Id]
    answer: NotRequired[Never]
    reason: str
    detail: NotRequired[JsonValue]


type Input = QueuedInput | PlacedInput | DoneInput | UnansweredInput


class CompletedTaskOutcome[R](TypedDict):
    status: Literal["completed"]
    result: R
    error: NotRequired[Never]
    reason: NotRequired[Never]


class FailedTaskOutcome[R](TypedDict):
    status: Literal["failed"]
    error: StoredError
    result: NotRequired[R]
    reason: NotRequired[Never]


class AbortedTaskOutcome[R](TypedDict):
    status: Literal["aborted"]
    reason: NotRequired[str]
    result: NotRequired[R]
    error: NotRequired[Never]


class OrphanedTaskOutcome(TypedDict):
    status: Literal["orphaned"]
    reason: str
    result: NotRequired[Never]
    error: NotRequired[Never]


class FaultedTaskOutcome(TypedDict):
    status: Literal["faulted"]
    error: StoredError
    result: NotRequired[Never]
    reason: NotRequired[Never]


type TaskOutcome[R] = CompletedTaskOutcome[R] | FailedTaskOutcome[R] | AbortedTaskOutcome[R] | OrphanedTaskOutcome | FaultedTaskOutcome


class PendingTaskState[S](TypedDict):
    status: Literal["pending"]
    checkpoint: S
    outcome: NotRequired[Never]


class RunningTaskState[S](TypedDict):
    status: Literal["running"]
    checkpoint: S
    outcome: NotRequired[Never]


class TerminalTaskState[R](TypedDict):
    status: Literal["terminal"]
    checkpoint: NotRequired[Never]
    outcome: TaskOutcome[R]


type TaskState[S, R] = PendingTaskState[S] | RunningTaskState[S] | TerminalTaskState[R]


class _TaskRecordBase[I](TypedDict):
    id: Id
    conversationId: Id
    kind: str
    version: int | float
    input: I
    after: Sequence[Id]
    background: bool
    abortRequested: bool


class LiveTaskRecord[I, S](_TaskRecordBase[I]):
    state: PendingTaskState[S] | RunningTaskState[S]
    memos: NotRequired[Mapping[str, JsonValue]]


class TerminalTaskRecord[I, R](_TaskRecordBase[I]):
    state: TerminalTaskState[R]
    memos: NotRequired[Never]


type TaskRecord[I, S, R] = LiveTaskRecord[I, S] | TerminalTaskRecord[I, R]


class SessionDocumentScope(TypedDict):
    kind: Literal["session"]


class ConversationDocumentScope(TypedDict):
    kind: Literal["conversation"]
    conversationId: Id


class TaskDocumentScope(TypedDict):
    kind: Literal["task"]
    taskId: Id


class _DocumentIdentity(TypedDict):
    id: Id
    kind: str
    key: NotRequired[str]


class SessionDocumentCreate(_DocumentIdentity):
    scope: SessionDocumentScope
    history: NotRequired[Never]
    fork: NotRequired[Never]


class LatestDocumentCreate(_DocumentIdentity):
    scope: ConversationDocumentScope
    history: Literal["latest"]
    fork: Literal["current", "initial"]


class RewindableDocumentCreate(_DocumentIdentity):
    scope: ConversationDocumentScope
    history: Literal["rewindable"]
    fork: Literal["asOf", "current", "initial"]


class TaskDocumentCreate(_DocumentIdentity):
    scope: TaskDocumentScope
    history: NotRequired[Never]
    fork: NotRequired[Never]


type DocumentCreate = SessionDocumentCreate | LatestDocumentCreate | RewindableDocumentCreate | TaskDocumentCreate


class _DocumentTimestamps(TypedDict):
    createdAt: Seq
    retiredAt: NotRequired[Seq]


class SessionDocumentRecord(SessionDocumentCreate, _DocumentTimestamps):
    pass


class LatestDocumentRecord(LatestDocumentCreate, _DocumentTimestamps):
    pass


class RewindableDocumentRecord(RewindableDocumentCreate, _DocumentTimestamps):
    pass


class TaskDocumentRecord(TaskDocumentCreate, _DocumentTimestamps):
    pass


type DocumentRecord = SessionDocumentRecord | LatestDocumentRecord | RewindableDocumentRecord | TaskDocumentRecord


class Page[T, C](TypedDict):
    items: list[T]
    next: NotRequired[C]


type Cursor = Mapping[str, JsonValue]


class EntryQuery(TypedDict):
    conversationId: Id
    minEntryId: NotRequired[Id]
    maxEntryId: NotRequired[Id]


class TaskQuery(TypedDict):
    conversationId: NotRequired[Id]
    kind: NotRequired[str]
    status: NotRequired[Literal["pending", "running", "terminal"]]
    abortRequested: NotRequired[bool]
    background: NotRequired[bool]


class ConversationWrite(TypedDict):
    type: Literal["conversation"]
    value: ConversationRecord


class EntryWrite(TypedDict):
    type: Literal["entry"]
    value: EntryRecord


class TaskWrite(TypedDict):
    type: Literal["task"]
    value: TaskRecord[JsonValue, JsonValue, JsonValue]


class InputWrite(TypedDict):
    type: Literal["input"]
    value: Input


type StorageWrite = ConversationWrite | EntryWrite | TaskWrite | InputWrite


class EntryAtCommit(TypedDict):
    entry: EntryRecord
    commitSeq: Seq


class Storage(Protocol):
    """Atomic, detached storage; the owning Session validates record semantics."""

    def commit(self, writes: Sequence[StorageWrite], context: Context) -> Awaitable[Seq]: ...
    def mint_id(self) -> Id: ...
    def conversation(self, id: Id, context: Context) -> Awaitable[ConversationRecord | Undefined]: ...
    def scan_conversations(self, cursor: Cursor | Undefined, limit: int | float, context: Context) -> Awaitable[Page[ConversationRecord, Cursor]]: ...
    def entry(self, id: Id, context: Context) -> Awaitable[EntryAtCommit | Undefined]: ...
    def find_latest_head_marker(self, conversation_id: Id, at_or_before_entry_id: Id | Undefined, context: Context) -> Awaitable[HeadEntryRecord | Undefined]: ...
    def scan_entries(self, query: EntryQuery, cursor: Cursor | Undefined, limit: int | float, context: Context) -> Awaitable[Page[EntryRecord, Cursor]]: ...
    def task(self, id: Id, context: Context) -> Awaitable[TaskRecord[JsonValue, JsonValue, JsonValue] | Undefined]: ...
    def scan_tasks(self, query: TaskQuery, cursor: Cursor | Undefined, limit: int | float, context: Context) -> Awaitable[Page[TaskRecord[JsonValue, JsonValue, JsonValue], Cursor]]: ...
    def input(self, id: Id, context: Context) -> Awaitable[Input | Undefined]: ...
    def input_by_request(self, conversation_id: Id, request_id: str, context: Context) -> Awaitable[Input | Undefined]: ...
    def close(self, context: Context) -> Awaitable[None]: ...


__all__ = [
    "ROOT_CONVERSATION_ID", "UNDEFINED", "Undefined", "Id", "Seq", "StoredError",
    "ConversationRecord", "ContextEdit", "EntryRecord", "EntryDraft", "Input",
    "TaskOutcome", "TaskState", "TaskRecord", "DocumentRecord", "DocumentCreate",
    "Page", "Cursor", "EntryQuery", "TaskQuery", "StorageWrite", "Storage",
    "ConversationParent", "ConversationOwner", "OmitContextEdit", "ReplaceContextEdit",
    "HeadEntryRecord", "EntryAtCommit", "QueuedInput", "PlacedInput", "DoneInput",
    "UnansweredInput", "CompletedTaskOutcome", "FailedTaskOutcome", "AbortedTaskOutcome",
    "OrphanedTaskOutcome", "FaultedTaskOutcome", "PendingTaskState", "RunningTaskState",
    "TerminalTaskState", "LiveTaskRecord", "TerminalTaskRecord", "SessionDocumentScope",
    "ConversationDocumentScope", "TaskDocumentScope", "SessionDocumentCreate",
    "LatestDocumentCreate", "RewindableDocumentCreate", "TaskDocumentCreate",
    "SessionDocumentRecord", "LatestDocumentRecord", "RewindableDocumentRecord",
    "TaskDocumentRecord", "ConversationWrite", "EntryWrite", "TaskWrite", "InputWrite",
]
