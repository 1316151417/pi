"""Conversation views and watches ported from ``harness/pico3/view.ts``.

The manager owns one tracked view per conversation, folds each commit's changes
into it, and publishes an :class:`~.types.Envelope` (ops + events) to every
watcher. Watchers start with a plain copy of the view and replay buffered
envelopes on ``start``; a listener that throws fails its own watch only.

Port note: the published view is the JSON document shape (a ``dict``), because
``apply_envelope`` applies chord ops to it. TypeScript's ``structuredClone`` is
``copy.deepcopy`` here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set

from pi_ai.types import JsonObject, JsonValue
from .delta import Op, apply_immutable, track
from .session import CommitChanges, CommitResult
from .types import (
    AnyKind,
    Entry,
    Conversation,
    ConversationView,
    DocRef,
    Envelope,
    GenerationStatus,
    Id,
    Task,
    ToolSlot,
    TurnView,
    ViewEvent,
    entry_to_json,
    tool_slot_to_json,
)

__all__ = [
    "WATCH_CAPACITY",
    "Watch",
    "WatchImpl",
    "apply_envelope",
    "ViewManager",
    "touches_key",
]

WATCH_CAPACITY = 256

Listener = Callable[[Envelope], None]


@dataclass
class _ViewRecord:
    conversation_id: Id
    tracker: Any
    watchers: Set["WatchImpl"] = field(default_factory=set)
    revision: int = 0


@dataclass
class _Delivery:
    envelope: Envelope
    watchers: List["WatchImpl"]


class Watch(Protocol):
    """A started-or-buffered view subscription."""

    view: Dict[str, Any]
    revision: int
    closed: bool

    def start(self, listener: Listener) -> None: ...

    def stop(self) -> None: ...


def apply_envelope(view: ConversationView, envelope: Envelope) -> Any:
    """Apply one envelope's ops to a published view."""
    return apply_immutable(view, envelope.ops)


class ViewManager:
    """Owns the per-conversation tracked views and their watchers."""

    def __init__(self, session: Any, on_report: Callable[[Any], None]) -> None:
        self.records: Dict[Id, _ViewRecord] = {}
        self.deliveries: List[_Delivery] = []
        self.session = session
        self.on_report = on_report

    # -- watching ----------------------------------------------------------

    def watch(self, conversation: Conversation, entries: List[Dict[str, Any]]) -> "WatchImpl":
        record = self.records.get(conversation.id)
        if record is None:
            tracker = track(self.build(conversation, entries))
            tracker.flush()
            record = _ViewRecord(conversation_id=conversation.id, tracker=tracker)
            self.records[conversation.id] = record

        def on_stop() -> None:
            record.watchers.discard(watcher)
            if not record.watchers and self.records.get(conversation.id) is record:
                del self.records[conversation.id]

        watcher = WatchImpl(copy.deepcopy(record.tracker.target), record.revision, self.on_report, on_stop)
        record.watchers.add(watcher)
        return watcher

    # -- commit fan-in -----------------------------------------------------

    def update(self, result: CommitResult) -> None:
        """Runs on the session line after persistence and in-memory indexes update."""
        for record in list(self.records.values()):
            conversation = self.session.conversation_records.get(record.conversation_id)
            if conversation is None:
                continue
            try:
                changes: CommitChanges = result.changes
                appended = [
                    entry
                    for entry in changes.entries
                    if entry.conversation_id == record.conversation_id
                ]
                document_changes = [
                    change
                    for change in changes.docs
                    if change.ref.doc == "session"
                    or change.ref.conversation_id == record.conversation_id
                ]
                task_changed = any(
                    task.conversation_id == record.conversation_id for task in changes.tasks
                )
                events = [
                    scoped.event
                    for scoped in changes.events
                    if scoped.conversation_id == record.conversation_id
                ]
                if not appended and not document_changes and not task_changed and not events:
                    continue

                entry_ops = apply_entries(record.tracker, appended)
                state = record.tracker.state
                rewindable_changed = any(
                    change.ref.doc == "rewindable"
                    and change.ref.conversation_id == record.conversation_id
                    for change in document_changes
                )
                sticky_changed = any(
                    change.ref.doc == "sticky"
                    and change.ref.conversation_id == record.conversation_id
                    for change in document_changes
                )
                config_changed = any(event.type == "config.changed" for event in events)
                plugin_changed = any(touches_key(change.ops, "plugins") for change in document_changes)
                rewindable = (
                    self.document(DocRef(doc="rewindable", conversation_id=record.conversation_id))
                    if (rewindable_changed or config_changed or plugin_changed)
                    else None
                )
                sticky = (
                    self.document(DocRef(doc="sticky", conversation_id=record.conversation_id))
                    if (sticky_changed or task_changed or config_changed or plugin_changed)
                    else None
                )

                if config_changed:
                    sync_record(state["config"], self.config(rewindable, sticky))
                if sticky_changed:
                    sync_array(state["inbox"], copy.deepcopy(sticky.get("inbox") or []))
                if sticky_changed or task_changed:
                    sync_optional(state, "turn", self.turn(record.conversation_id, sticky).get("turn"))
                    sync_optional(
                        state,
                        "compaction",
                        self.compaction(record.conversation_id).get("compaction"),
                    )
                    sync_record(state["tasks"], self.tasks(record.conversation_id, sticky))
                if plugin_changed:
                    session_doc = self.document(DocRef(doc="session"))
                    sync_record(state["plugins"], self.plugins(rewindable, sticky, session_doc))

                ops = [*entry_ops, *record.tracker.flush()]
                if not ops and not events:
                    continue
                record.revision += 1
                envelope = Envelope(revision=record.revision, ops=ops, events=events)
                self.deliveries.append(_Delivery(envelope=envelope, watchers=list(record.watchers)))
            except Exception as error:  # noqa: BLE001 - a broken view fails its watchers only
                del self.records[record.conversation_id]
                for watcher in list(record.watchers):
                    watcher.fail(error)

    def deliver(self) -> None:
        """Runs after the session line; listeners are synchronous and ordered."""
        pending = self.deliveries
        self.deliveries = []
        for delivery in pending:
            for watcher in delivery.watchers:
                watcher.accept(delivery.envelope)

    def close(self) -> None:
        records = list(self.records.values())
        self.records.clear()
        self.deliveries = []
        for record in records:
            for watcher in list(record.watchers):
                watcher.stop()

    # -- view construction -------------------------------------------------

    def build(self, conversation: Conversation, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        rewindable = self.document(
            DocRef(doc="rewindable", conversation_id=conversation.id)
        )
        sticky = self.document(DocRef(doc="sticky", conversation_id=conversation.id))
        session_doc = self.document(DocRef(doc="session"))
        public_conversation = {
            key: copy.deepcopy(value)
            for key, value in _conversation_json(conversation).items()
            if key != "sections"
        }
        view: Dict[str, Any] = {
            "conversation": public_conversation,
            "entries": copy.deepcopy(entries),
            "config": self.config(rewindable, sticky),
            "inbox": copy.deepcopy(sticky.get("inbox") or []),
            "tasks": self.tasks(conversation.id, sticky),
            "plugins": self.plugins(rewindable, sticky, session_doc),
        }
        view.update(self.turn(conversation.id, sticky))
        view.update(self.compaction(conversation.id))
        return view

    def document(self, ref: DocRef) -> Dict[str, Any]:
        value = self.session.loaded_document(ref)
        if value is None:
            raise ValueError(f"view document {ref.doc} is not loaded")
        return value

    def config(self, rewindable: Optional[Dict[str, Any]], sticky: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        defaults = self.session.defaults
        for key, doc in defaults.route.items():
            source = rewindable if doc == "rewindable" else sticky
            if source is not None and key in source:
                value = source[key]
            else:
                value = getattr(defaults, doc).get(key)
            if value is not None:
                out[key] = copy.deepcopy(value)
        return out

    def turn(self, conversation_id: Id, sticky: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        live = [
            task
            for task in self.session.live_tasks.values()
            if task.conversation_id == conversation_id
        ]
        turn_tasks = [task for task in live if _kind(self.session, task.kind, "turn") is True]
        if not turn_tasks:
            return {}
        generation = next((task for task in turn_tasks if task.kind == "pi.generation"), None)
        post_tools = next((task for task in turn_tasks if task.kind == "pi.post_tools"), None)
        input_task = generation if generation is not None else post_tools
        inputs = [] if input_task is None else input_ids(input_task)
        turn_message = (sticky or {}).get("turn", {}).get("message")
        turn: Dict[str, Any] = {
            "inputs": inputs,
            "tools": [
                strip_private_tool_state(slot) for slot in (sticky or {}).get("turn", {}).get("tools", [])
            ],
        }
        if generation is not None:
            turn["generation"] = generation_status(generation, turn_message is not None)
        if turn_message is not None:
            turn["message"] = copy.deepcopy(turn_message)
        return {"turn": turn}

    def compaction(self, conversation_id: Id) -> Dict[str, Any]:
        task = next(
            (
                candidate
                for candidate in self.session.live_tasks.values()
                if candidate.conversation_id == conversation_id and candidate.kind == "pi.collapse"
            ),
            None,
        )
        if task is None:
            return {}
        input_record = task.input or {}
        checkpoint = task.checkpoint or {}
        compaction: Dict[str, Any] = {
            "taskId": task.id,
            "reason": input_record.get("reason"),
            "stage": "retrying" if checkpoint.get("phase") == "retrying" else "summarizing",
            "attempt": checkpoint.get("attempt", 1),
        }
        if checkpoint.get("phase") == "retrying" and checkpoint.get("untilMs") is not None:
            compaction["retryAt"] = checkpoint.get("untilMs")
        return {"compaction": compaction}

    def tasks(self, conversation_id: Id, sticky: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for task in self.session.live_tasks.values():
            kind = self.session.kinds.get(task.kind)
            if (
                task.conversation_id != conversation_id
                or kind is None
                or kind.turn is True
                or task.kind == "pi.collapse"
            ):
                continue
            slot = (sticky or {}).get("tasks", {}).get(task.id)
            status = describe(kind, task, slot)
            entry: Dict[str, Any] = {"kind": task.kind, "status": status}
            if task.background:
                entry["background"] = True
            if task.abort:
                entry["marked"] = True
            out[task.id] = entry
        return out

    def plugins(
        self,
        rewindable: Optional[Dict[str, Any]],
        sticky: Optional[Dict[str, Any]],
        session_doc: Dict[str, Any],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        session_plugins = session_doc.get("plugins") or {}
        for id, registration in self.session.namespaces.items():
            if registration.project is None:
                continue
            merged: Dict[str, Any] = {
                **(registration.defaults.rewindable or {}),
                **(registration.defaults.sticky or {}),
                **(registration.defaults.session or {}),
                **((rewindable or {}).get("plugins", {}).get(id) or {}),
                **((sticky or {}).get("plugins", {}).get(id) or {}),
                **(session_plugins.get(id) or {}),
            }
            out[id] = strict_json(registration.project(copy.deepcopy(merged)), f"namespace {id} view")
        return out


class WatchImpl:
    """One watcher: identity-preserving view copy plus a bounded pre-start buffer."""

    def __init__(
        self,
        view: Dict[str, Any],
        revision: int,
        on_report: Callable[[Any], None],
        on_stop: Callable[[], None],
    ) -> None:
        self.view = view
        self.initial_revision = revision
        self.on_report = on_report
        self.on_stop = on_stop
        self.listener: Optional[Listener] = None
        self.buffer: List[Envelope] = []
        self.stopped = False

    @property
    def revision(self) -> int:
        return self.initial_revision

    @property
    def closed(self) -> bool:
        return self.stopped

    def start(self, listener: Listener) -> None:
        if self.listener is not None or self.stopped:
            return
        self.listener = listener
        pending = self.buffer
        self.buffer = []
        for envelope in pending:
            if self.stopped:
                break
            self.deliver(envelope)

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.buffer = []
        self.on_stop()

    def accept(self, envelope: Envelope) -> None:
        if self.stopped:
            return
        if self.listener is None:
            if len(self.buffer) >= WATCH_CAPACITY:
                self.stop()
                self.report(RuntimeError(f"watch capacity {WATCH_CAPACITY} exceeded before start()"))
                return
            self.buffer.append(envelope)
            return
        self.deliver(envelope)

    def deliver(self, envelope: Envelope) -> None:
        try:
            assert self.listener is not None
            self.listener(envelope)
        except Exception as error:  # noqa: BLE001 - reported through on_report
            self.fail(error)

    def fail(self, error: Any) -> None:
        self.stop()
        self.report(error)

    def report(self, error: Any) -> None:
        try:
            self.on_report(error)
        except Exception:  # noqa: BLE001 - a reporter must never break the line
            pass


# ---------------------------------------------------------------------------
# Folding helpers
# ---------------------------------------------------------------------------


def apply_entries(tracker: Any, appended: List[Any]) -> List[Op]:
    """Append entries to a tracked view, cutting at a moved head, and return the ops."""
    ops: List[Op] = []
    entries = tracker.state["entries"]
    for entry in appended:
        head = entry.head if hasattr(entry, "head") else entry.get("head")
        if head is not None:
            retained = next(
                (
                    index
                    for index, candidate in enumerate(entries)
                    if _entry_id(candidate) >= head
                ),
                -1,
            )
            remove = len(entries) if retained < 0 else retained
            if remove > 0:
                ops.extend(tracker.flush())
                del entries[0:remove]
                tracker.flush()
                ops.append(("p", ("entries",), 0, remove, []))
        entries.append(
            copy.deepcopy(entry_to_json(entry) if isinstance(entry, Entry) else entry)
        )
    return ops


def _entry_id(entry: Any) -> int:
    if isinstance(entry, dict):
        return entry.get("id", 0)
    return getattr(entry, "id", 0)


def input_ids(task: Task) -> List[Id]:
    inputs = task.input.get("inputs") if isinstance(task.input, dict) else None
    if not isinstance(inputs, list):
        return []
    return [value for value in inputs if isinstance(value, int) and not isinstance(value, bool)]


def generation_status(task: Task, streaming: bool) -> GenerationStatus:
    if task.status == "pending" and task.after:
        return GenerationStatus(stage="waiting", on="compaction")
    checkpoint = task.checkpoint or {}
    phase = checkpoint.get("phase")
    if phase == "requesting":
        return GenerationStatus(
            stage="streaming" if streaming else "requesting", attempt=checkpoint.get("attempt", 1)
        )
    if phase == "retrying":
        return GenerationStatus(
            stage="retrying",
            attempt=checkpoint.get("attempt", 1),
            retry_at=checkpoint.get("untilMs") or 0,
            last_error=checkpoint.get("lastError") or "",
        )
    if phase == "deferred":
        return GenerationStatus(
            stage="deferred",
            attempt=checkpoint.get("attempt", 1),
            poll_at=checkpoint.get("pollAt") or 0,
        )
    return GenerationStatus(stage="preparing")


def strip_private_tool_state(slot: Any) -> Dict[str, Any]:
    if isinstance(slot, ToolSlot):
        view = tool_slot_to_json(slot)
    else:
        view = copy.deepcopy(slot.to_json() if hasattr(slot, "to_json") else slot)
    view.pop("memos", None)
    return view


def describe(kind: AnyKind, task: Task, slot: Optional[Dict[str, Any]]) -> JsonValue:
    if not callable(getattr(kind, "describe", None)):
        checkpoint = task.checkpoint or {}
        return {"phase": checkpoint.get("phase") or ("pending" if task.status == "pending" else "running")}
    public_slot = None if slot is None else copy.deepcopy(slot)
    if public_slot is not None:
        public_slot.pop("memos", None)
    described = dict(task.__dict__)
    described.pop("slot", None)
    if public_slot is not None:
        described["slot"] = public_slot
    return strict_json(kind.describe(described), f"task kind {task.kind} describe")


def strict_json(value: Any, what: str) -> JsonValue:
    """Round-trip a value through JSON, rejecting anything that cannot survive it."""
    try:
        encoded = _dumps(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what} returned a non-JSON value: {error}") from error
    return encoded


def _dumps(value: Any) -> Any:
    import json

    return json.loads(json.dumps(_jsonable_value(value)))


def _jsonable_value(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return value.to_json()
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    return value


def touches_key(ops: List[Op], key: str) -> bool:
    for op in ops:
        if op[0] == "r":
            return True
        path = op[1]
        if path and path[0] == key:
            return True
    return False


def sync_optional(target: Dict[str, Any], key: str, next_value: Any) -> None:
    if next_value is None:
        target.pop(key, None)
        return
    current = target.get(key)
    if current is not None and same(current, next_value):
        return
    if is_record(current) and is_record(next_value):
        sync_record(current, next_value)
    else:
        target[key] = copy.deepcopy(next_value)


def sync_record(target: Dict[str, Any], next_value: Dict[str, Any]) -> None:
    for key in list(target.keys()):
        if key not in next_value:
            del target[key]
    for key, value in next_value.items():
        current = target.get(key)
        if current is not None and same(current, value):
            continue
        if isinstance(current, list) and isinstance(value, list):
            sync_array(current, value)
            continue
        if is_record(current) and is_record(value):
            sync_record(current, value)
            continue
        target[key] = copy.deepcopy(value)


def sync_array(target: List[Any], next_value: List[Any]) -> None:
    prefix = 0
    while prefix < len(target) and prefix < len(next_value):
        if same(target[prefix], next_value[prefix]):
            prefix += 1
            continue
        current = target[prefix]
        replacement = next_value[prefix]
        if not is_record(current) or not is_record(replacement) or not same_array_item(current, replacement):
            break
        sync_record(current, replacement)
        prefix += 1
    if prefix == len(target) and prefix == len(next_value):
        return
    if prefix == len(target):
        target.extend(copy.deepcopy(next_value[prefix:]))
        return
    target[prefix:] = copy.deepcopy(next_value[prefix:])


def same_array_item(current: Dict[str, Any], next_value: Dict[str, Any]) -> bool:
    if isinstance(current.get("id"), int) or isinstance(next_value.get("id"), int):
        return current.get("id") == next_value.get("id")
    if isinstance(current.get("callId"), str) or isinstance(next_value.get("callId"), str):
        return current.get("callId") == next_value.get("callId")
    if isinstance(current.get("type"), str) or isinstance(next_value.get("type"), str):
        return current.get("type") == next_value.get("type")
    return False


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def same(a: Any, b: Any) -> bool:
    import json

    if a is b:
        return True
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except (TypeError, ValueError):
        return a == b


def _kind(session: Any, name: str, attribute: str) -> Any:
    kind = session.kinds.get(name)
    return None if kind is None else getattr(kind, attribute, None)


def _conversation_json(conversation: Any) -> Dict[str, Any]:
    from .types import conversation_to_json

    return conversation_to_json(conversation)


_ = (ToolSlot, TurnView, ViewEvent)
