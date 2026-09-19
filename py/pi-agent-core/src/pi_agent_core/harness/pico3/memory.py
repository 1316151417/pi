"""In-memory storage ported from ``harness/pico3/memory.ts``.

The reference backend, and the read path of :class:`~.jsonl.JsonlStorage`.

A batch is validated and staged in full before any table changes: one failed
batch changes no table, no document, no ID high-water, no sequence. Committed
IDs are never reused; IDs minted but never committed may be reused after reopen.

Port note: records are dataclasses and are cloned through their JSON form, which
is what ``JSON.parse(JSON.stringify(v))`` does in TypeScript.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..._chord.context import Context
from ..._pi_ai.types import JsonObject
from .delta import Op, apply_immutable, is_base
from .types import (
    UNSET,
    Conversation,
    DocRef,
    Entry,
    EntryScan,
    Id,
    Input,
    Seq,
    Storage,
    Task,
    TaskPatch,
    TaskScan,
    Write,
    conversation_from_json,
    conversation_to_json,
    doc_ref_to_json,
    entry_from_json,
    entry_to_json,
    input_from_json,
    input_to_json,
    task_from_json,
    task_patch_from_json,
    task_patch_to_json,
    task_to_json,
)

__all__ = ["MemoryStorage"]


def _clone(value: Any) -> Any:
    """A plain JSON copy of one record."""
    if isinstance(value, Conversation):
        return conversation_from_json(conversation_to_json(value))
    if isinstance(value, Entry):
        return entry_from_json(entry_to_json(value))
    if isinstance(value, Task):
        return task_from_json(task_to_json(value))
    if isinstance(value, Input):
        return input_from_json(input_to_json(value))
    if isinstance(value, TaskPatch):
        return task_patch_from_json(task_patch_to_json(value))
    if isinstance(value, DocRef):
        return doc_ref_to_json(value)
    return value


class MemoryStorage:
    """In-memory :class:`~.types.Storage` implementation."""

    def __init__(self) -> None:
        self.conversations_by_id: Dict[Id, Conversation] = {}
        self.entries_by_id: Dict[Id, Entry] = {}
        self.entries_by_conversation: Dict[Id, List[Entry]] = {}  # ascending
        self.tasks_by_id: Dict[Id, Task] = {}
        self.inputs_by_id: Dict[Id, Input] = {}
        self.inputs_by_request: Dict[str, Input] = {}
        #: Rewindable docs keep their full op history with the seq of each batch, for ``doc_as_of``.
        self.rewindable_log: Dict[Id, List[Tuple[Seq, List[Op]]]] = {}
        self.sticky_log: Dict[Id, List[List[Op]]] = {}
        self.session_doc: Optional[JsonObject] = None
        self.entry_seq: Dict[Id, Seq] = {}  # entry id -> seq of its batch
        self.next_id_value = 1
        self.seq: Seq = 0
        self._closed = False

    # -- ids ---------------------------------------------------------------

    def mint_id(self) -> Id:
        id = self.next_id_value
        self.next_id_value += 1
        return id

    def set_next_id(self, n: Id) -> None:
        self.next_id_value = max(self.next_id_value, n)

    # -- writes ------------------------------------------------------------

    async def commit(self, writes: List[Write], ctx: Optional[Context] = None) -> Seq:
        if self._closed:
            raise RuntimeError("storage closed")
        seq = self.seq + 1
        staged = self.stage(writes, seq)  # throws before any mutation
        staged()
        self.seq = seq
        return seq

    def stage(self, writes: List[Write], seq: Seq) -> Any:
        """Validate the whole batch against the current tables; return the applier."""
        operations: List[Any] = []
        created: set = set()
        max_id = 0

        def claim(id: Id, what: str) -> None:
            nonlocal max_id
            if not isinstance(id, int) or isinstance(id, bool) or id <= 0:
                raise ValueError(f"{what}: invalid id {id}")
            if id in created:
                raise ValueError(f"{what}: id {id} created twice in one batch")
            created.add(id)
            max_id = max(max_id, id)

        for write in writes:
            if write.type == "conversation":
                conversation = write.conversation
                if conversation.id in self.conversations_by_id:
                    raise ValueError(f"conversation {conversation.id} exists")
                claim(conversation.id, "conversation")
                stored = _clone(conversation)

                def apply_conversation(stored: Conversation = stored) -> None:
                    self.conversations_by_id[stored.id] = stored
                    self.entries_by_conversation.setdefault(stored.id, [])

                operations.append(apply_conversation)
            elif write.type == "entry":
                entry = write.entry
                if entry.id in self.entries_by_id:
                    raise ValueError(f"entry {entry.id} exists")
                claim(entry.id, "entry")
                stored_entry = _clone(entry)

                def apply_entry(stored_entry: Entry = stored_entry) -> None:
                    self.entries_by_id[stored_entry.id] = stored_entry
                    self.entries_by_conversation.setdefault(
                        stored_entry.conversation_id, []
                    ).append(stored_entry)
                    self.entry_seq[stored_entry.id] = seq

                operations.append(apply_entry)
            elif write.type == "task":
                task = write.task
                if task.id in self.tasks_by_id:
                    raise ValueError(f"task {task.id} exists")
                claim(task.id, "task")
                stored_task = _clone(task)
                operations.append(lambda stored_task=stored_task: self.tasks_by_id.__setitem__(stored_task.id, stored_task))
            elif write.type == "task.patch":
                patch = write.patch
                previous = self.tasks_by_id.get(patch.id)
                if previous is None:
                    for candidate in writes:
                        if candidate.type == "task" and candidate.task.id == patch.id:
                            previous = candidate.task
                            break
                if previous is None:
                    raise ValueError(f"patch for unknown task {patch.id}")
                stored_patch = _clone(patch)

                def apply_patch(stored_patch: TaskPatch = stored_patch) -> None:
                    current = self.tasks_by_id[stored_patch.id]
                    next_task = task_from_json(task_to_json(current))
                    if stored_patch.status is not None:
                        next_task.status = stored_patch.status
                    if stored_patch.checkpoint is not UNSET:
                        next_task.checkpoint = stored_patch.checkpoint
                    if stored_patch.abort is not None:
                        next_task.abort = stored_patch.abort
                    if stored_patch.outcome is not None:
                        next_task.outcome = stored_patch.outcome
                    if stored_patch.owns is not None:
                        next_task.owns = list(stored_patch.owns)
                    self.tasks_by_id[stored_patch.id] = next_task

                operations.append(apply_patch)
            elif write.type == "input":
                input_record = _clone(write.input)
                if input_record.id not in self.inputs_by_id:
                    claim(input_record.id, "input")

                def apply_input(input_record: Input = input_record) -> None:
                    self.inputs_by_id[input_record.id] = input_record
                    if input_record.request_id is not None:
                        self.inputs_by_request[
                            f"{input_record.conversation_id}:{input_record.request_id}"
                        ] = input_record

                operations.append(apply_input)
            elif write.type == "doc":
                ref = write.ref
                ops = [tuple(op) for op in write.ops]
                if ref.doc == "session":

                    def apply_session(ops: List[Op] = ops) -> None:
                        base = self.session_doc if self.session_doc is not None else {"plugins": {}}
                        self.session_doc = apply_immutable(base, ops)

                    operations.append(apply_session)
                elif ref.doc == "rewindable":
                    operations.append(
                        lambda ref=ref, ops=ops: self.rewindable_log.setdefault(
                            ref.conversation_id, []
                        ).append((seq, ops))
                    )
                else:
                    operations.append(
                        lambda ref=ref, ops=ops: self.sticky_log.setdefault(
                            ref.conversation_id, []
                        ).append(ops)
                    )
            else:
                raise ValueError(f"unknown write type {write.type!r}")

        def apply_all() -> None:
            for operation in operations:
                operation()
            self.set_next_id(max_id + 1)

        return apply_all

    # -- reads -------------------------------------------------------------

    async def conversation(self, id: Id, ctx: Optional[Context] = None) -> Optional[Conversation]:
        return self._optional_clone(self.conversations_by_id.get(id))

    async def conversations(self, ctx: Optional[Context] = None) -> List[Conversation]:
        return [self._clone(item) for item in self.conversations_by_id.values()]

    async def entries(self, ids: List[Id], ctx: Optional[Context] = None) -> Dict[Id, Entry]:
        out: Dict[Id, Entry] = {}
        for id in ids:
            entry = self.entries_by_id.get(id)
            if entry is not None:
                out[id] = self._clone(entry)
        return out

    async def scan_entries(self, scan: EntryScan, ctx: Optional[Context] = None) -> List[Entry]:
        """Newest-first, fork-aware: own entries, then the parent's up to the fork point."""
        out: List[Entry] = []
        conversation_id: Optional[Id] = scan.conversation_id
        cap: Optional[Id] = scan.before
        while conversation_id is not None and len(out) < scan.limit:
            own = self.entries_by_conversation.get(conversation_id) or []
            for index in range(len(own) - 1, -1, -1):
                if len(out) >= scan.limit:
                    break
                entry = own[index]
                if cap is not None and entry.id >= cap:
                    continue
                if scan.kind is not None and entry.kind != scan.kind:
                    continue
                if scan.with_head and entry.head is None:
                    continue
                out.append(self._clone(entry))
            conversation = self.conversations_by_id.get(conversation_id)
            parent = conversation.parent if conversation is not None else None
            conversation_id = parent.conversation_id if parent is not None else None
            cap = None if parent is None else min(cap if cap is not None else parent.at + 1, parent.at + 1)
        return out

    async def task(self, id: Id, ctx: Optional[Context] = None) -> Optional[Task]:
        return self._optional_clone(self.tasks_by_id.get(id))

    async def scan_tasks(self, scan: TaskScan, ctx: Optional[Context] = None) -> List[Task]:
        out: List[Task] = []
        for task in self.tasks_by_id.values():
            if scan.conversation_id is not None and task.conversation_id != scan.conversation_id:
                continue
            if scan.status is not None and task.status not in scan.status:
                continue
            if scan.kind is not None and task.kind != scan.kind:
                continue
            out.append(self._clone(task))
        return out

    async def input(self, id: Id, ctx: Optional[Context] = None) -> Optional[Input]:
        return self._optional_clone(self.inputs_by_id.get(id))

    async def input_by_request(
        self, conversation_id: Id, request_id: str, ctx: Optional[Context] = None
    ) -> Optional[Input]:
        return self._optional_clone(self.inputs_by_request.get(f"{conversation_id}:{request_id}"))

    async def doc(self, ref: DocRef, ctx: Optional[Context] = None) -> Optional[JsonObject]:
        if ref.doc == "session":
            return {"plugins": {}} if self.session_doc is None else _json_clone(self.session_doc)
        if ref.doc == "rewindable":
            history = self.rewindable_log.get(ref.conversation_id)
            if not history:
                return None
            log = [ops for _seq, ops in history]
        else:
            log = self.sticky_log.get(ref.conversation_id)
            if log is None:
                return None
        return _json_clone(self.fold(log))

    async def doc_as_of(
        self, conversation_id: Id, at: Id, ctx: Optional[Context] = None
    ) -> Optional[JsonObject]:
        """Commit-granular history: the rewindable state after the commit containing entry ``at``."""
        entry = self.entries_by_id.get(at)
        if entry is None:
            return None
        owner = entry.conversation_id
        chain: List[Id] = []
        conversation = self.conversations_by_id.get(conversation_id)
        while conversation is not None and conversation.id != owner:
            chain.append(conversation.id)
            conversation = (
                self.conversations_by_id.get(conversation.parent.conversation_id)
                if conversation.parent is not None
                else None
            )
        if conversation is None:
            return None
        seq_at = self.entry_seq.get(at, 0)
        log = [
            ops for seq, ops in self.rewindable_log.get(owner, []) if seq <= seq_at
        ]
        return None if len(log) == 0 else _json_clone(self.fold(log))

    def fold(self, log: List[List[Op]]) -> JsonObject:
        start = 0
        for index in range(len(log) - 1, -1, -1):
            if is_base(log[index]):
                start = index
                break
        state: JsonObject = {}
        for index in range(start, len(log)):
            state = apply_immutable(state, log[index])
        return state

    async def truncate(self, ref: DocRef, ctx: Optional[Context] = None) -> None:
        if ref.doc != "sticky":
            return
        log = self.sticky_log.get(ref.conversation_id)
        if log is None:
            return
        start = 0
        for index in range(len(log) - 1, -1, -1):
            if is_base(log[index]):
                start = index
                break
        del log[:start]

    async def close(self, ctx: Optional[Context] = None) -> None:
        self._closed = True

    # -- helpers -----------------------------------------------------------

    def _clone(self, value: Any) -> Any:
        return _clone(value)

    def _optional_clone(self, value: Any) -> Any:
        return None if value is None else _clone(value)


def _json_clone(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value))


_ = Storage
