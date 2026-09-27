"""The session line and its capability-checked transaction, ported from
``harness/pico3/session.ts``.

Everything in this module is the real implementation: the document cache and
defaults, the transaction change record, ``TxImpl`` (the capability-checked
transaction with its callback surface) and ``Session`` (the serialized line).

Port notes:

* TypeScript's ``callbackSurface()`` Proxy over a frozen whitelist of method
  names is modelled by :class:`_CallbackSurface`, which exposes exactly the same
  names and throws once the transaction closes its surface.
* ``enter`` reserves its place on the line synchronously, exactly like the
  TypeScript original, so a returned-but-not-yet-awaited operation keeps its
  ordering slot.
* ``plain`` is a strict JSON copy for JSON payloads; records (``Entry``,
  ``Task``, ``Input``) are copied through their durable JSON form by the
  ``_copy_*`` helpers, because Python keeps them as dataclasses.
"""

from __future__ import annotations

import asyncio
import copy
import json
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from pi_ai.types import JsonObject, JsonValue, ThinkingLevel
from ..._chord.context import Context, create_context_key, with_context_value, without_abort_signal
from .context import derive_context
from .delta import Op, Tracker, is_base, track
from .membrane import Membrane
from .types import (
    UNSET,
    AnyKind,
    Checkpoint,
    Completion,
    Conversation,
    ConversationSpec,
    ContextView,
    DocRef,
    Entry,
    EntryKind,
    EntryScan,
    EntryRef,
    Faulted,
    Forbidden,
    CollapseInProgress,
    GenerationInProgress,
    ConversationBusy,
    Closed,
    Id,
    Input,
    Invoker,
    Namespace,
    NamespaceRegistration,
    NewEntry,
    OwnedConversationSpec,
    QueuedInput,
    ReadAfterWrite,
    RewindableState,
    Runtime,
    Seq,
    SendInput,
    SessionState,
    StickyState,
    Storage,
    Task,
    TaskPatch,
    TaskRef,
    TaskScan,
    TaskSpec,
    ViewEvent,
    Write,
    conversation_to_json,
    conversation_from_json,
    doc_key,
    entry_from_json,
    entry_to_json,
    input_from_json,
    input_to_json,
    new_entry_from_json,
    new_entry_to_json,
    task_from_json,
    task_patch_to_json,
    task_to_json,
    to_stored,
    view_event_from_json,
)

__all__ = [
    "remove_where",
    "is_core_kind",
    "CORE_KINDS",
    "CORE_CONFIG_VALIDATORS",
    "CALLBACK_TX_METHODS",
    "finite_nonnegative",
    "exact_object",
    "plain",
    "validate_entry",
    "Defaults",
    "Docs",
    "CommitChanges",
    "DocChange",
    "ScopedEvent",
    "ConversationIndex",
    "CommitResult",
    "TransactionControl",
    "TransactionControlImpl",
    "TxImpl",
    "Session",
    "NestedLineOperation",
    "LINE_KEY",
    "STICKY_BASE_BUDGET",
]

_UNPORTED: Tuple[str, ...] = ()


def remove_where(items: List[Any], pred: Callable[[Any], bool]) -> None:
    """Remove matching elements from a tracked array IN PLACE."""
    for index in range(len(items) - 1, -1, -1):
        if pred(items[index]):
            del items[index]


CORE_KINDS = {"pi.generation", "pi.tool", "pi.post_tools", "pi.collapse"}


def finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float("inf"), float("-inf"))
        and value >= 0
    )


def exact_object(value: Any, required: Sequence[str], optional: Sequence[str] = ()) -> bool:
    if not isinstance(value, dict):
        return False
    keys = list(value.keys())
    return all(key in value for key in required) and all(
        key in required or key in optional for key in keys
    )


def _is_model_ref(value: Any) -> bool:
    return (
        exact_object(value, ["provider", "modelId"])
        and isinstance(value.get("provider"), str)
        and value.get("provider") != ""
        and isinstance(value.get("modelId"), str)
        and value.get("modelId") != ""
    )


def _is_thinking_level(value: Any) -> bool:
    return isinstance(value, str) and value in (
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )


def _is_selected_tools(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(name, str) for name in value)


def _is_retry(value: Any) -> bool:
    if not exact_object(value, ["enabled", "maxRetries", "baseDelayMs"], ["maxAgentDelayMs"]):
        return False
    return (
        isinstance(value.get("enabled"), bool)
        and isinstance(value.get("maxRetries"), int)
        and not isinstance(value.get("maxRetries"), bool)
        and value.get("maxRetries") >= 0
        and finite_nonnegative(value.get("baseDelayMs"))
        and (
            value.get("maxAgentDelayMs") is None
            or finite_nonnegative(value.get("maxAgentDelayMs"))
        )
    )


def _is_mode(value: Any) -> bool:
    return value in ("all", "one-at-a-time")


#: Validators for the core config keys, keyed by their disjoint names.
CORE_CONFIG_VALIDATORS: Dict[str, Callable[[Any], bool]] = {
    "model": _is_model_ref,
    "thinkingLevel": _is_thinking_level,
    "selectedTools": _is_selected_tools,
    "profile": lambda value: isinstance(value, str),
    "retry": _is_retry,
    "threshold": lambda value: isinstance(value, (int, float))
    and not isinstance(value, bool)
    and value == value,
    "keepRecent": finite_nonnegative,
    "steeringMode": _is_mode,
    "followUpMode": _is_mode,
}


def is_core_kind(name: str) -> bool:
    return name in CORE_KINDS


def plain(value: Any) -> Any:
    """A plain JSON copy: no document proxy, no live reference, strict JSON."""
    if isinstance(value, Task):
        return task_to_json(value)
    if isinstance(value, Entry):
        return entry_to_json(value)
    if isinstance(value, Input):
        return input_to_json(value)
    if isinstance(value, Conversation):
        return conversation_to_json(value)
    if hasattr(value, "to_json"):
        return json.loads(json.dumps(value.to_json()))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def _copy_task(task: Task) -> Task:
    return task_from_json(task_to_json(task))


def _copy_entry(entry: Entry) -> Entry:
    return entry_from_json(entry_to_json(entry))


def _copy_input(input_record: Input) -> Input:
    return input_from_json(input_to_json(input_record))


def _copy_new_entry(entry: NewEntry) -> NewEntry:
    return new_entry_from_json(new_entry_to_json(entry))


def validate_entry(entry: Entry) -> None:
    if entry.head is not None and entry.head > entry.id:
        raise ValueError(f"entry {entry.id}: head {entry.head} is in the future")
    if entry.model is not None:
        for message in entry.model:
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            if role is None or not isinstance(role, str):
                raise ValueError(f"entry {entry.id}: malformed model message")
    json.dumps(entry_to_json(entry))  # strict JSON: throws on cycles


# ---------------------------------------------------------------------------
# Declared config defaults
# ---------------------------------------------------------------------------


class Defaults:
    """Declared config defaults, derived from the registered kinds."""

    def __init__(self, kinds: Optional[Iterable[AnyKind]] = None) -> None:
        self.rewindable: JsonObject = {}
        self.sticky: JsonObject = {}
        self.route: Dict[str, str] = {}
        self._owners: Dict[str, AnyKind] = {}
        for kind in kinds or []:
            self.register(kind)

    def register(self, kind: AnyKind) -> None:
        declarations: List[tuple] = []
        config = kind.config
        for doc in ("rewindable", "sticky"):
            values = getattr(config, doc, None) if config is not None else None
            for key, value in (values or {}).items():
                declarations.append((doc, key, value))
        local: Set[str] = set()
        for _doc, key, _value in declarations:
            if key in local or key in self.route:
                raise ValueError(f'config key "{key}" declared by more than one kind')
            local.add(key)
        for doc, key, value in declarations:
            self.route[key] = doc
            self._owners[key] = kind
            if value is not None:
                getattr(self, doc)[key] = plain(value)

    def unregister(self, kind: AnyKind) -> None:
        for key, owner in list(self._owners.items()):
            if owner is not kind:
                continue
            del self._owners[key]
            doc = self.route.pop(key, None)
            if doc is not None:
                getattr(self, doc).pop(key, None)

    def validate(self, key: str, value: Any) -> bool:
        if not _is_json_value(value):
            return False
        core = CORE_CONFIG_VALIDATORS.get(key)
        if core is not None:
            return core(value)
        return key in self.route

    def validate_seed(self, doc: str, seed: Dict[str, Any]) -> JsonObject:
        out: JsonObject = {}
        for key, value in seed.items():
            if value is None:
                continue
            if self.route.get(key) != doc or not self.validate(key, value):
                raise TypeError(f'invalid {doc} config value for "{key}"')
            out[key] = plain(value)
        return out

    def fill(self, doc: str, target: JsonObject) -> None:
        """Fill declared keys that are absent (never ``??``: a stored null is a value)."""
        for key, value in getattr(self, doc).items():
            if key not in target:
                target[key] = plain(value)

    def fresh_rewindable(
        self, over: Optional[JsonObject] = None, preserve_plugins: bool = False
    ) -> RewindableState:
        over = dict(over or {})
        base: JsonObject = {"plugins": {}}
        self.fill("rewindable", base)
        plugins = over.pop("plugins", None)
        rest = over if preserve_plugins else self.validate_seed("rewindable", over)
        result: Dict[str, Any] = {**base, **rest}
        if preserve_plugins and plugins is not None:
            result["plugins"] = plain(plugins)
        return result

    def fresh_sticky(self, over: Optional[JsonObject] = None) -> StickyState:
        over = dict(over or {})
        base: JsonObject = {"inbox": [], "turn": {"tools": []}, "tasks": {}, "plugins": {}}
        self.fill("sticky", base)
        for key in ("inbox", "turn", "tasks", "plugins"):
            over.pop(key, None)
        return {**base, **self.validate_seed("sticky", over)}


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


# ---------------------------------------------------------------------------
# Document cache
# ---------------------------------------------------------------------------


class Docs:
    """Loaded document trackers, keyed by document reference."""

    def __init__(self, storage: Storage, defaults: Defaults) -> None:
        self.trackers: Dict[str, Tracker] = {}
        self.storage = storage
        self.defaults = defaults
        #: Bytes of ops written since the last base, per document. Drives rebase+truncate.
        self.since_base: Dict[str, int] = {}

    def note_ops(self, ref: DocRef, ops: List[Op]) -> None:
        key = doc_key(ref)
        self.since_base[key] = (
            0 if is_base(ops) else self.since_base.get(key, 0) + len(json.dumps(ops))
        )

    def request_base(self, ref: DocRef) -> None:
        tracker = self.trackers.get(doc_key(ref))
        if tracker is not None:
            tracker.rebase()

    async def get(self, ref: DocRef, ctx: Context) -> Tracker:
        key = doc_key(ref)
        cached = self.trackers.get(key)
        if cached is not None:
            return cached
        stored = await self.storage.doc(ref, ctx)
        if stored is None:
            raise ValueError(f"document {key} does not exist")
        if ref.doc != "session":
            # Declared defaults are visible without being persisted.
            self.defaults.fill(ref.doc, stored)
        tracker = track(stored)
        tracker.flush()  # consume the synthetic first flush; never persisted
        self.trackers[key] = tracker
        return tracker

    def evict(self, ref: DocRef) -> None:
        self.trackers.pop(doc_key(ref), None)

    def peek(self, ref: DocRef) -> Optional[Tracker]:
        return self.trackers.get(doc_key(ref))

    def loaded(self, ref: DocRef) -> Optional[JsonObject]:
        tracker = self.trackers.get(doc_key(ref))
        return None if tracker is None else tracker.target

    def adopt(self, ref: DocRef, tracker: Tracker) -> None:
        self.trackers[doc_key(ref)] = tracker


# ---------------------------------------------------------------------------
# Commit results
# ---------------------------------------------------------------------------


@dataclass
class DocChange:
    ref: DocRef
    ops: List[Op] = field(default_factory=list)


@dataclass
class ScopedEvent:
    conversation_id: Id
    event: Optional[ViewEvent] = None


@dataclass
class CommitChanges:
    """Everything one commit changed, as the view manager and listeners see it."""

    entries: List[Entry] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    inputs: List[Input] = field(default_factory=list)
    conversations: List[Conversation] = field(default_factory=list)
    docs: List[DocChange] = field(default_factory=list)
    events: List[ScopedEvent] = field(default_factory=list)


class ConversationIndex:
    """Session-side conversation index: owner/parent graph for subtree checks and ancestry."""

    def get(self, id: Id) -> Optional[Conversation]: ...

    def subtree(self, root: Id) -> Set[Id]: ...

    def ancestors(self, id: Id) -> List[Id]: ...


@dataclass
class CommitResult:
    value: Any = None
    seq: Optional[Seq] = None
    changes: CommitChanges = field(default_factory=CommitChanges)


class TransactionControl:
    """A handle the transaction hands to its caller to replace the current task."""

    def set_task(self, task: Task) -> None: ...


class TransactionControlImpl(TransactionControl):
    """The concrete control handle: the only way a closure terminalizes a task."""

    def __init__(self, tx: "TxImpl") -> None:
        self._tx = tx

    def set_task(self, task: Task) -> None:
        self._tx.set_task_internal_for_control(task)


class NestedLineOperation(Exception):
    def __init__(self) -> None:
        super().__init__("nested line operation")
        self.name = "NestedLineOperation"


#: Marks the serialized mutation line in a context.
LINE_KEY = create_context_key("pico3.session.line")

#: Sticky document budget before a base is requested (bytes of ops since the last base).
STICKY_BASE_BUDGET = 256 * 1024


# ---------------------------------------------------------------------------
# TxImpl
# ---------------------------------------------------------------------------

HOST_TX_METHODS = (
    "conversation",
    "entry",
    "entries",
    "newest_entry",
    "scan_entries",
    "context",
    "task",
    "tasks",
    "input",
    "rewindable_as_of",
    "snapshot",
    "plugins",
    "emit",
    "config",
    "write",
    "create_task",
    "create_conversation",
)
TASK_TX_METHODS = ("checkpoint", "slot")
CORE_TX_METHODS = (
    "rewindable",
    "sticky",
    "session",
    "tool_slot",
    "append_entry",
    "send",
    "resolve_inputs",
    "boundary",
    "mark_task",
    "set_task",
    "withdraw_input",
    "create_owned_conversation",
    "create_fork_conversation",
)
#: The exact surface a transaction callback may use.
CALLBACK_TX_METHODS = frozenset((*HOST_TX_METHODS, *TASK_TX_METHODS, *CORE_TX_METHODS))


class _CallbackSurface:
    """The frozen whitelist the transaction callback sees, instead of the raw ``TxImpl``."""

    __slots__ = ("_tx",)

    def __init__(self, tx: "TxImpl") -> None:
        object.__setattr__(self, "_tx", tx)

    def __getattr__(self, name: str) -> Any:
        if name not in CALLBACK_TX_METHODS:
            raise AttributeError(name)
        tx = object.__getattribute__(self, "_tx")
        method = getattr(tx, name)
        if not callable(method):
            raise AttributeError(name)

        def bound(*args: Any, **kwargs: Any) -> Any:
            tx.assert_surface_active()
            return method(*args, **kwargs)

        return bound


class PluginSlicesView(dict):
    """A namespace's merged plugin slice: reads and writes the per-document slices in place."""

    def __init__(self, tx: "TxImpl", namespace: Namespace, registration: NamespaceRegistration) -> None:
        super().__init__()
        self._tx = tx
        self._namespace = namespace
        self._registration = registration
        self._keys = list(registration.routes.keys())

    def _slice(self, key: str) -> JsonObject:
        doc = self._registration.routes[key]
        conversation_id = self._tx.invoker.conversation_id
        ref = DocRef(doc="session") if doc == "session" else DocRef(doc=doc, conversation_id=conversation_id)
        document = self._tx.raw(ref)
        plugins = document.setdefault("plugins", {})
        slice_value = plugins.setdefault(self._namespace.id, {})
        default = (getattr(self._registration.defaults, doc, None) or {}).get(key)
        if key not in slice_value and default is not None:
            slice_value[key] = plain(default)
        return slice_value

    def __getitem__(self, key: str) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        return self._slice(key)[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._keys:
            raise KeyError(key)
        self._slice(key)[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __iter__(self) -> Any:
        return iter(list(self._keys))

    def __len__(self) -> int:
        return len(self._keys)

    def keys(self) -> List[str]:
        return list(self._keys)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._keys:
            return default
        return self._slice(key).get(key)

    def items(self) -> List[Any]:
        return [(key, self.get(key)) for key in self._keys]


class TxConfig:
    """The per-conversation configuration facade handed to callbacks."""

    def __init__(self, tx: "TxImpl", conversation_id: Id) -> None:
        self._tx = tx
        self._conversation_id = conversation_id

    def _definition(self, key: str) -> Tuple[str, JsonValue]:
        tx = self._tx
        invoker = tx.invoker
        if invoker.type == "task" and invoker.kind is not None and invoker.kind.config is not None:
            for doc in ("rewindable", "sticky"):
                declared = getattr(invoker.kind.config, doc, None)
                if declared is not None and key in declared:
                    return doc, declared[key]
        doc = tx.defaults.route.get(key)
        if doc is None:
            raise ValueError(f'unknown config key "{key}"')
        return doc, getattr(tx.defaults, doc).get(key)

    def _assert_writable(self, key: str) -> None:
        invoker = self._tx.invoker
        if invoker.type == "task" and not invoker.core:
            raise Forbidden(f"config({key}): ordinary tasks cannot write config")

    def _mark_changed(self, key: str) -> None:
        keys = self._tx.changed_config.setdefault(self._conversation_id, set())
        keys.add(key)

    def get(self, key: str) -> JsonValue:
        self._tx.assert_surface_active()
        doc, fallback = self._definition(key)
        document = self._tx.doc(DocRef(doc=doc, conversation_id=self._conversation_id)).target
        return plain(document[key]) if key in document else plain(fallback)

    def set(self, key: str, value: JsonValue) -> None:
        self._tx.assert_surface_active()
        self._assert_writable(key)
        doc, _fallback = self._definition(key)
        if not self._tx.defaults.validate(key, value):
            raise TypeError(f'invalid config value for "{key}"')
        self._tx.raw(DocRef(doc=doc, conversation_id=self._conversation_id))[key] = plain(value)
        self._mark_changed(key)

    def reset(self, key: str) -> None:
        self._tx.assert_surface_active()
        self._assert_writable(key)
        doc, _fallback = self._definition(key)
        self._tx.raw(DocRef(doc=doc, conversation_id=self._conversation_id)).pop(key, None)
        self._mark_changed(key)


class TxImpl:
    """One transaction: capability checks, overlays for direct reads, and the writes it built."""

    def __init__(
        self,
        invoker: Invoker,
        storage: Storage,
        docs: Docs,
        defaults: Defaults,
        kinds: Dict[str, AnyKind],
        namespaces: Dict[str, NamespaceRegistration],
        live_tasks: Dict[Id, Task],
        conversations: ConversationIndex,
        now: Callable[[], int],
        ctx: Context,
    ) -> None:
        self.writes: List[Write] = []
        self.touched: Dict[str, Tuple[DocRef, Tracker]] = {}
        self.membrane = Membrane("tx")
        # Overlays for direct reads: complete for the tables a closure reads after writing.
        self.created_tasks: Dict[Id, Task] = {}
        self.created_entries: Dict[Id, Entry] = {}
        self.created_conversations: Dict[Id, Conversation] = {}
        self.inputs_by_id: Dict[Id, Input] = {}
        self.inputs_by_request: Dict[str, Input] = {}
        self.namespace_views: Dict[int, Any] = {}
        self.changed_config: Dict[Id, Set[str]] = {}
        # Domains written: scans over them reject.
        self.wrote_entries: Set[Id] = set()
        self.wrote_tasks = False
        self.poisoned: Optional[Exception] = None
        self.surface_active = True
        #: Set while a terminal closure runs: the invoker's task counts as gone for busy().
        self.closing = False
        self.changes = CommitChanges()
        self.invoker = invoker
        self.storage = storage
        self.docs = docs
        self.defaults = defaults
        self.kinds = kinds
        self.namespaces = namespaces
        self.live_tasks = live_tasks
        self.conversations = conversations
        self.now = now
        self.ctx = ctx

    # --- surface ------------------------------------------------------------

    def callback_surface(self) -> _CallbackSurface:
        return _CallbackSurface(self)

    def close_surface(self) -> None:
        self.surface_active = False

    def assert_surface_active(self) -> None:
        if not self.surface_active:
            raise TypeError("transaction used outside its callback")

    # --- capability & scope -------------------------------------------------

    @property
    def core(self) -> bool:
        return self.invoker.type == "kernel" or (
            self.invoker.type == "task" and bool(self.invoker.core)
        )

    def assert_core(self, what: str) -> None:
        if not self.core:
            raise Forbidden(f"{what}: core turn machinery only")

    def assert_not_host(self, what: str) -> None:
        if self.invoker.type != "task":
            raise Forbidden(f"{what} outside a task")

    def in_scope(self, conversation_id: Id) -> bool:
        """A task may touch its own conversation and the subtree it owns; host and core may touch anything."""
        if self.invoker.type != "task" or self.invoker.core:
            return True
        if conversation_id == self.invoker.conversation_id or conversation_id in self.created_conversations:
            return True
        task = self.created_tasks.get(self.invoker.id or 0) or self.live_tasks.get(self.invoker.id or 0)
        for root in (task.owns if task is not None else []) or []:
            if conversation_id in self.conversations.subtree(root):
                return True
        return False

    def assert_scope(self, conversation_id: Id, what: str) -> None:
        if not self.in_scope(conversation_id):
            raise Forbidden(
                f"{what}: conversation {conversation_id} is outside this task's subtree"
            )

    def assert_entry_scope(self, entry: Entry, what: str) -> None:
        if self.in_scope(entry.conversation_id):
            return
        if self.invoker.type != "task" or self.invoker.core:
            raise Forbidden(f"{what}: entry {entry.id} is outside this task's subtree")
        candidates: Set[Id] = {self.invoker.conversation_id, *self.created_conversations.keys()}  # type: ignore[arg-type]
        task = self.created_tasks.get(self.invoker.id or 0) or self.live_tasks.get(self.invoker.id or 0)
        for root in (task.owns if task is not None else []) or []:
            candidates.update(self.conversations.subtree(root))
        for candidate in candidates:
            conversation = self.created_conversations.get(candidate) or self.conversations.get(candidate)
            while conversation is not None and conversation.parent is not None:
                if (
                    conversation.parent.conversation_id == entry.conversation_id
                    and entry.id <= conversation.parent.at
                ):
                    return
                conversation = self.conversations.get(conversation.parent.conversation_id)
        raise Forbidden(f"{what}: entry {entry.id} is not visible from this task's subtree")

    def poison(self, error: Exception) -> Any:
        if self.poisoned is None:
            self.poisoned = error
        raise error

    def assert_no_entry_writes(self, conversation_id: Id, read: str) -> None:
        if conversation_id in self.wrote_entries:
            self.poison(ReadAfterWrite(read, "entry append"))

    def assert_no_task_writes(self, read: str) -> None:
        if self.wrote_tasks:
            self.poison(ReadAfterWrite(read, "task write"))

    # --- reads (direct reads see this transaction's writes) -------------------

    async def conversation(self, id: Id) -> Optional[Conversation]:
        self.assert_scope(id, "conversation")
        created = self.created_conversations.get(id)
        if created is not None:
            return created
        return await self.storage.conversation(id, self.ctx)

    async def entry(self, a: Union[Id, EntryKind], b: Optional[Id] = None) -> Optional[Entry]:
        id = a if isinstance(a, int) else b
        assert id is not None
        entry = self.created_entries.get(id)
        if entry is None:
            entry = (await self.storage.entries([id], self.ctx)).get(id)
        if entry is not None:
            self.assert_entry_scope(entry, "entry")
        if isinstance(a, EntryKind):
            return entry if a.is_entry(entry) else None
        return entry

    async def entries(self, ids: Sequence[Id]) -> Dict[Id, Entry]:
        out = await self.storage.entries(
            [id for id in ids if id not in self.created_entries], self.ctx
        )
        for id in ids:
            created = self.created_entries.get(id)
            if created is not None:
                out[id] = created
        for entry in out.values():
            self.assert_entry_scope(entry, "entries")
        return out

    async def newest_entry(
        self, conversation_id: Id, opts: Optional[Union[Dict[str, Any], EntryKind]] = None
    ) -> Optional[Entry]:
        self.assert_scope(conversation_id, "newestEntry")
        self.assert_no_entry_writes(conversation_id, "newestEntry")
        kind: Optional[str] = None
        with_head = False
        if isinstance(opts, EntryKind):
            kind = opts.kind
        elif opts:
            kind = opts.get("kind")
            with_head = bool(opts.get("withHead"))
        rows = await self.storage.scan_entries(
            EntryScan(conversation_id=conversation_id, kind=kind, with_head=with_head, limit=1),
            self.ctx,
        )
        return rows[0] if rows else None

    async def scan_entries(self, scan: EntryScan) -> List[Entry]:
        self.assert_scope(scan.conversation_id, "scanEntries")
        self.assert_no_entry_writes(scan.conversation_id, "scanEntries")
        return await self.storage.scan_entries(scan, self.ctx)

    async def context(self, conversation_id: Id, at: Optional[Id] = None) -> ContextView:
        self.assert_scope(conversation_id, "context")
        self.assert_no_entry_writes(conversation_id, "context")
        return await derive_context(self.storage, conversation_id, at, self.ctx)

    async def task(self, a: Union[Id, TaskRef]) -> Optional[Task]:
        id = a if isinstance(a, int) else a.id
        task_record = (
            self.created_tasks.get(id)
            or self.live_tasks.get(id)
            or (await self.storage.task(id, self.ctx))
        )
        if task_record is not None:
            self.assert_scope(task_record.conversation_id, "task")
        return task_record

    async def tasks(self, scan: TaskScan) -> List[Task]:
        self.assert_no_task_writes("tasks")
        if scan.conversation_id is not None:
            self.assert_scope(scan.conversation_id, "tasks")
        rows = await self.storage.scan_tasks(scan, self.ctx)
        seen = {task.id for task in rows}
        for task_record in self.live_tasks.values():
            if task_record.id in seen:
                continue
            if scan.conversation_id is not None and task_record.conversation_id != scan.conversation_id:
                continue
            if scan.kind is not None and task_record.kind != scan.kind:
                continue
            rows.append(task_record)
        return [
            task_record
            for task_record in rows
            if scan.status is None or task_record.status in scan.status
        ]

    async def input(self, id: Id) -> Optional[Input]:
        input_record = self.inputs_by_id.get(id)
        if input_record is None:
            input_record = await self.storage.input(id, self.ctx)
        if input_record is not None:
            self.assert_scope(input_record.conversation_id, "input")
        return input_record

    async def input_by_request(self, conversation_id: Id, request_id: str) -> Optional[Input]:
        self.assert_scope(conversation_id, "inputByRequest")
        hit = self.inputs_by_request.get(f"{conversation_id}:{request_id}")
        if hit is not None:
            return hit
        return await self.storage.input_by_request(conversation_id, request_id, self.ctx)

    async def rewindable_as_of(self, conversation_id: Id, at: Id) -> Optional[RewindableState]:
        self.assert_scope(conversation_id, "rewindableAsOf")
        return await self.storage.doc_as_of(conversation_id, at, self.ctx)

    def snapshot(self, ref: DocRef) -> JsonObject:
        if ref.doc != "session":
            assert ref.conversation_id is not None
            self.assert_scope(ref.conversation_id, "snapshot")
        value = copy.deepcopy(self.doc(ref).target)
        if ref.doc != "session":
            self.defaults.fill(ref.doc, value)
            if self.invoker.type == "task" and self.invoker.kind is not None and self.invoker.kind.config:
                declared = getattr(self.invoker.kind.config, ref.doc, None) or {}
                for key, fallback in declared.items():
                    if key not in value and fallback is not None:
                        value[key] = copy.deepcopy(fallback)
        return value

    # --- documents ----------------------------------------------------------

    def doc(self, ref: DocRef) -> Tracker:
        key = doc_key(ref)
        hit = self.touched.get(key)
        if hit is not None:
            return hit[1]
        cached = self.docs.peek(ref)
        if cached is not None:
            self.touched[key] = (ref, cached)
            return cached
        raise ValueError(f"document {key} not loaded; pass it in commit(docs=...)")

    def view(self, ref: DocRef) -> Any:
        return self.membrane.wrap(self.doc(ref).state)

    async def preload(self, refs: Sequence[DocRef]) -> None:
        for ref in refs:
            key = doc_key(ref)
            if key in self.touched:
                continue
            self.touched[key] = (ref, await self.docs.get(ref, self.ctx))

    def rewindable(self, conversation_id: Id) -> RewindableState:
        self.assert_core("rewindable document")
        return self.view(DocRef(doc="rewindable", conversation_id=conversation_id))

    def sticky(self, conversation_id: Id) -> StickyState:
        self.assert_core("sticky document")
        return self.view(DocRef(doc="sticky", conversation_id=conversation_id))

    def session_state(self) -> SessionState:
        self.assert_core("session document")
        return self.view(DocRef(doc="session"))

    def raw(self, ref: DocRef) -> Any:
        """Internal, unchecked."""
        return self.view(ref)

    def plugins(self, namespace: Namespace) -> Any:
        registration = self.namespaces.get(namespace.id)
        if registration is None or registration.token is not namespace:
            raise Forbidden(f'namespace "{namespace.id}" is stale')
        cached = self.namespace_views.get(id(namespace))
        if cached is not None:
            return cached
        self.invocation_conversation_id(f"plugins({namespace.id})")
        target = PluginSlicesView(self, namespace, registration)
        view = self.membrane.wrap(target)
        self.namespace_views[id(namespace)] = view
        return view

    def emit(self, namespace_or_event: Any, name: Optional[str] = None, data: JsonValue = None) -> None:
        conversation_id = self.invocation_conversation_id("emit")
        if isinstance(namespace_or_event, Namespace):
            namespace = namespace_or_event
            registration = self.namespaces.get(namespace.id)
            if registration is None or registration.token is not namespace:
                raise Forbidden(f'namespace "{namespace.id}" is stale')
            if name is None or data is None or not _is_event_name(name):
                raise ValueError("invalid plugin event")
            self.changes.events.append(
                ScopedEvent(
                    conversation_id=conversation_id,
                    event=ViewEvent(type=f"plugin.{namespace.id}.{name}", data=plain(data)),
                )
            )
            return
        self.assert_core("core event")
        event = (
            namespace_or_event
            if isinstance(namespace_or_event, ViewEvent)
            else view_event_from_json(plain(namespace_or_event))
        )
        self.changes.events.append(ScopedEvent(conversation_id=conversation_id, event=event))

    def invocation_conversation_id(self, what: str) -> Id:
        conversation_id = self.invoker.conversation_id
        if conversation_id is None:
            raise Forbidden(f"{what}: no conversation is bound to this transaction")
        self.assert_scope(conversation_id, what)
        return conversation_id

    def config(self, conversation_id: Id) -> TxConfig:
        self.assert_scope(conversation_id, "config")
        return TxConfig(self, conversation_id)

    def slot(self, ref: TaskRef) -> JsonObject:
        self.assert_not_host("slot")
        invoker = self.invoker
        task_record = self.live_tasks.get(ref.id) or self.created_tasks.get(ref.id)
        if task_record is None:
            raise ValueError(f"task {ref.id} is not live")
        if not invoker.core and ref.id != invoker.id:
            raise Forbidden("slot: another task's slot")
        sticky = self.raw(DocRef(doc="sticky", conversation_id=task_record.conversation_id))
        tasks_slice = sticky.setdefault("tasks", {})
        if str(ref.id) not in tasks_slice:
            slot_factory = ref.kind.slot if ref.kind is not None and ref.kind.slot else None
            tasks_slice[str(ref.id)] = slot_factory(task_record.input) if slot_factory else {}
        return tasks_slice[str(ref.id)]

    def tool_slot(self, task_record: Any) -> JsonObject:
        self.assert_core("toolSlot")
        index = task_record.input.get("index")
        slots = self.raw(
            DocRef(doc="sticky", conversation_id=task_record.conversation_id)
        ).setdefault("turn", {}).setdefault("tools", [])
        slot = slots[index] if index is not None and index < len(slots) else None
        if slot is None:
            raise ValueError(f"no tool slot at index {index}")
        return slot

    # --- writes -------------------------------------------------------------

    def append_entry(self, conversation_id: Id, a: Any, b: Optional[Any] = None) -> Any:
        self.assert_core("appendEntry")
        if isinstance(a, EntryKind):
            entry = b if isinstance(b, NewEntry) else NewEntry(kind=a.kind, **dict(b or {}))
            id = self.append_entry_internal(conversation_id, entry)
            return EntryRef(id=id, kind=a)
        return self.append_entry_internal(conversation_id, a)

    def append_entry_internal(self, conversation_id: Id, entry: NewEntry) -> Id:
        if (
            self.invoker.type == "task"
            and (self.invoker.id or 0) not in self.live_tasks
            and not self.closing
        ):
            raise Forbidden("appendEntry from a task that is not live")
        id = self.storage.mint_id()
        head = id if entry.head == "self" else entry.head
        record = entry_from_json(
            {
                **new_entry_to_json(entry),
                "id": id,
                "conversationId": conversation_id,
                **({} if head is None else {"head": head}),
                **({} if self.invoker.type != "task" else {"byTaskId": self.invoker.id}),
            }
        )
        validate_entry(record)
        self.writes.append(Write(type="entry", entry=record))
        self.changes.entries.append(record)
        if record.head is not None:
            self.changes.events.append(
                ScopedEvent(conversation_id=conversation_id, event=ViewEvent(type="head.moved", entry=record))
            )
        self.changes.events.append(
            ScopedEvent(conversation_id=conversation_id, event=ViewEvent(type="entry.added", entry=record))
        )
        self.created_entries[id] = record
        self.wrote_entries.add(conversation_id)
        return id

    async def write(self, conversation_id: Id, a: Any, b: Optional[Any] = None) -> Id:
        self.assert_scope(conversation_id, "write")
        entry = (
            NewEntry(kind=a.kind, **dict(b or {})) if isinstance(a, EntryKind) else a
        )
        if not self.core:
            if entry.head is not None:
                raise Forbidden("write: head entries are core only")
            if entry.edits is not None:
                raise Forbidden("write: edits are core only")
            if entry.kind.startswith("pi.") and entry.kind != "pi.notice":
                raise Forbidden(f"write: kind {entry.kind} is reserved")
            if entry.kind == "pi.notice" and (
                entry.model is None or len(entry.model) != 1 or entry.model[0].get("role") != "user"
            ):
                raise Forbidden("write: pi.notice requires exactly one user model message")
        if self.busy(conversation_id):
            id = self.storage.mint_id()
            self.raw(DocRef(doc="sticky", conversation_id=conversation_id)).setdefault("inbox", []).append(
                queued_input_to_json_plain(QueuedInput(id=id, mode="write", entry=new_entry_to_json(entry)))
            )
            self.put_input(Input(id=id, conversation_id=conversation_id, status="queued"))
            self.changes.events.append(
                ScopedEvent(
                    conversation_id=conversation_id,
                    event=ViewEvent(type="input.queued", input=id, mode="write"),
                )
            )
            return id
        id = self.storage.mint_id()
        entry_id = self.append_entry_internal(conversation_id, entry)
        self.put_input(
            Input(id=id, conversation_id=conversation_id, status="done", entry=entry_id)
        )
        return id

    def checkpoint(self, value: Checkpoint) -> None:
        self.assert_not_host("checkpoint")
        invoker = self.invoker
        if invoker.mode == "abort":
            raise Forbidden("checkpoint from an abort invocation")
        current = self.created_tasks.get(invoker.id or 0) or self.live_tasks.get(invoker.id or 0)
        if current is None:
            raise ValueError("task not live")
        self.set_task_internal(_with_checkpoint(current, plain(value)))

    def set_task(self, task: Task) -> None:
        self.assert_core("setTask")
        self.set_task_internal(task)

    def set_task_internal_for_control(self, task: Task) -> None:
        self.set_task_internal(task)

    def set_task_internal(self, task: Task) -> None:
        """Kernel-internal: replace a task's mutable fields. Persists only what changed."""
        task = _copy_task(task)
        prev = self.created_tasks.get(task.id) or self.live_tasks.get(task.id)
        patch: Dict[str, Any] = {"id": task.id}
        for key in ("status", "checkpoint", "abort", "outcome", "owns"):
            before = _json_of(getattr(prev, key) if prev is not None else None)
            after = _json_of(getattr(task, key))
            if before == after:
                continue
            value = getattr(task, key)
            if key == "checkpoint" and value is None:
                patch[key] = None
            else:
                patch[key] = value
        if task.status == "terminal":
            patch["status"] = "terminal"
            patch["checkpoint"] = None
            patch["owns"] = list(task.owns)
            patch["outcome"] = task.outcome
            if task.abort is True:
                patch["abort"] = True
            task = _copy_task(task)
            task.checkpoint = None
        if len(patch) == 1:
            return
        if task.status == "terminal":
            sticky = self.raw(DocRef(doc="sticky", conversation_id=task.conversation_id))
            (sticky.get("tasks") or {}).pop(str(task.id), None)
        self.writes.append(Write(type="task.patch", patch=_patch_from_dict(patch)))
        self.changes.tasks.append(task)
        if task.status == "terminal" and (prev is None or prev.status != "terminal"):
            kind = self.kinds.get(task.kind)
            if task.kind == "pi.collapse":
                if task.outcome is not None and task.outcome.status == "completed":
                    result = task.outcome.result or {}
                    self.changes.events.append(
                        ScopedEvent(
                            conversation_id=task.conversation_id,
                            event=ViewEvent(
                                type="compaction.finished",
                                task_id=task.id,
                                summary=result.get("summary"),
                            ),
                        )
                    )
                elif task.outcome is not None and task.outcome.status == "failed":
                    failure = task.outcome.failure or {}
                    self.changes.events.append(
                        ScopedEvent(
                            conversation_id=task.conversation_id,
                            event=ViewEvent(
                                type="compaction.failed",
                                task_id=task.id,
                                reason=failure.get("reason"),
                                detail=failure.get("detail"),
                            ),
                        )
                    )
            elif (kind is None or kind.turn is not True) and task.outcome is not None:
                self.changes.events.append(
                    ScopedEvent(
                        conversation_id=task.conversation_id,
                        event=ViewEvent(
                            type="task.ended",
                            task_id=task.id,
                            kind=task.kind,
                            outcome=task.outcome.status,
                        ),
                    )
                )
        self.created_tasks[task.id] = task  # overlay: later direct reads see the patch
        self.wrote_tasks = True

    def create_task(self, a: Any, input: JsonValue = None, opts: Optional[Dict[str, Any]] = None) -> Any:
        opts = dict(opts or {})
        if isinstance(a, TaskSpec):
            self.assert_core("createTask by name")
            return self.create_task_internal(a)
        registered = self.kinds.get(a.name)
        if registered is not a:
            raise Forbidden(f'createTask: kind "{a.name}" is not the registered token')
        return TaskRef(
            id=self.create_task_internal(
                TaskSpec(
                    kind=a.name,
                    input=input,
                    conversation_id=opts.get("conversationId"),
                    after=opts.get("after"),
                    background=bool(opts.get("background")),
                )
            ),
            kind=a,
        )

    def create_task_internal(self, spec: TaskSpec) -> Id:
        kind = self.kinds.get(spec.kind)
        if kind is None:
            raise ValueError(f"unknown task kind {spec.kind}")
        if is_core_kind(spec.kind) and not self.core:
            raise Forbidden(f"create core task {spec.kind}")
        conversation_id = spec.conversation_id
        if conversation_id is None:
            conversation_id = self.invoker.conversation_id if self.invoker.type == "task" else None
        if conversation_id is None:
            raise ValueError("createTask: conversationId required")
        self.assert_scope(conversation_id, "createTask")
        if spec.kind == "pi.generation" and self.has_live_kind(conversation_id, spec.kind):
            raise GenerationInProgress(conversation_id)
        if spec.kind == "pi.collapse" and self.has_live_kind(conversation_id, spec.kind):
            raise CollapseInProgress(conversation_id)
        id = self.storage.mint_id()
        task_record = Task(
            id=id,
            conversation_id=conversation_id,
            kind=spec.kind,
            input=plain(spec.input),
            status="pending",
            after=list(spec.after or []),
            owns=[],
            background=bool(spec.background),
        )
        self.writes.append(Write(type="task", task=task_record))
        self.changes.tasks.append(task_record)
        if spec.kind == "pi.collapse":
            payload = spec.input if isinstance(spec.input, dict) else {}
            self.changes.events.append(
                ScopedEvent(
                    conversation_id=conversation_id,
                    event=ViewEvent(
                        type="compaction.started",
                        task_id=id,
                        reason=payload.get("reason"),
                        through=payload.get("through"),
                    ),
                )
            )
        elif kind.turn is not True:
            self.changes.events.append(
                ScopedEvent(
                    conversation_id=conversation_id,
                    event=ViewEvent(
                        type="task.started",
                        task_id=id,
                        kind=task_record.kind,
                        background=True if task_record.background else None,
                    ),
                )
            )
        self.created_tasks[id] = task_record
        self.wrote_tasks = True
        return id

    def has_live_kind(self, conversation_id: Id, kind: str) -> bool:
        for task_record in self.live_tasks.values():
            if task_record.conversation_id != conversation_id or task_record.kind != kind:
                continue
            current = self.created_tasks.get(task_record.id) or task_record
            if current.status == "terminal":
                continue
            if self.closing and self.invoker.type == "task" and self.invoker.id == task_record.id:
                continue
            return True
        for task_record in self.created_tasks.values():
            if (
                task_record.conversation_id == conversation_id
                and task_record.kind == kind
                and task_record.status != "terminal"
                and task_record.id not in self.live_tasks
            ):
                return True
        return False

    def create_conversation(self, spec: ConversationSpec) -> Id:
        owner = self.invoker.id if self.invoker.type == "task" else None
        if spec.parent is not None:
            self.assert_scope(spec.parent["conversationId"], "createConversation: parent")
        return self.insert_conversation(spec, owner, False)

    def create_fork_conversation(self, spec: ConversationSpec) -> Id:
        self.assert_core("createForkConversation")
        return self.insert_conversation(spec, None, True)

    async def create_owned_conversation(
        self, owner_task_id: Id, source_conversation_id: Id, spec: OwnedConversationSpec
    ) -> Id:
        self.assert_core("createOwnedConversation")
        owner = self.live_tasks.get(owner_task_id) or self.created_tasks.get(owner_task_id)
        if (
            owner is None
            or owner.status == "terminal"
            or owner.conversation_id != source_conversation_id
        ):
            raise Forbidden(f"task {owner_task_id} cannot create an owned conversation")
        tip = await self.newest_entry(source_conversation_id) if spec.inherit else None
        inherited = (
            None
            if tip is None
            else await self.rewindable_as_of(source_conversation_id, tip.id)
        )
        raw_overrides = copy.deepcopy(spec.rewindable or {})
        raw_overrides.pop("plugins", None)
        overrides = self.defaults.validate_seed("rewindable", raw_overrides)
        if inherited is None:
            rewindable = overrides
        else:
            rewindable = {
                **inherited,
                **overrides,
                "plugins": copy.deepcopy(inherited.get("plugins") or {}),
            }
        return self.insert_conversation(
            ConversationSpec(
                parent=None if tip is None else {"conversationId": source_conversation_id, "at": tip.id},
                rewindable=rewindable,
                sticky=spec.sticky,
            ),
            owner_task_id,
            inherited is not None,
        )

    def insert_conversation(
        self, spec: ConversationSpec, owner: Optional[Id], preserve_plugins: bool
    ) -> Id:
        id = self.storage.mint_id()
        parent = spec.parent
        if parent is None or parent.get("at") == "start":
            parent_value = None
        else:
            parent_value = {"conversationId": parent["conversationId"], "at": parent["at"]}
        conversation = Conversation(
            id=id,
            parent=(
                None
                if parent_value is None
                else _conversation_parent(parent_value["conversationId"], parent_value["at"])
            ),
            owner=owner,
            sections=spec.sections if spec.sections else None,
        )
        self.writes.append(Write(type="conversation", conversation=conversation))
        self.changes.conversations.append(conversation)
        self.created_conversations[id] = conversation
        self.seed_doc(
            DocRef(doc="rewindable", conversation_id=id),
            self.defaults.fresh_rewindable(spec.rewindable, preserve_plugins),
        )
        self.seed_doc(
            DocRef(doc="sticky", conversation_id=id),
            self.defaults.fresh_sticky(spec.sticky),
        )
        if owner is not None:
            task_record = self.live_tasks.get(owner) or self.created_tasks.get(owner)
            if task_record is not None:
                updated = _copy_task(task_record)
                updated.owns = [*task_record.owns, id]
                self.set_task_internal(updated)
        return id

    def seed_doc(self, ref: DocRef, value: Any) -> None:
        tracker = track(copy.deepcopy(value))
        tracker.rebase()
        base = tracker.flush()
        self.writes.append(Write(type="doc", ref=ref, ops=base))
        self.changes.docs.append(DocChange(ref=ref, ops=base))
        self.docs.adopt(ref, tracker)
        self.touched[doc_key(ref)] = (ref, tracker)

    def mark_task(self, id: Id) -> None:
        self.assert_core("markTask")
        task_record = self.created_tasks.get(id) or self.live_tasks.get(id)
        if task_record is None:
            raise ValueError(f"task {id} not live")
        if task_record.abort is not True:
            marked = _copy_task(task_record)
            marked.abort = True
            self.set_task_internal(marked)

    # --- admission ----------------------------------------------------------

    def busy(self, conversation_id: Id) -> bool:
        """Prospective: live turn tasks, minus this transaction's terminals and the closing task."""

        def is_turn(task_record: Task) -> bool:
            kind = self.kinds.get(task_record.kind)
            return kind is not None and kind.turn is True and task_record.background is not True

        for task_record in self.live_tasks.values():
            if task_record.conversation_id != conversation_id or not is_turn(task_record):
                continue
            overlay = self.created_tasks.get(task_record.id)
            if overlay is not None and overlay.status == "terminal":
                continue
            if self.closing and self.invoker.type == "task" and task_record.id == self.invoker.id:
                continue
            return True
        for task_record in self.created_tasks.values():
            if (
                task_record.conversation_id == conversation_id
                and is_turn(task_record)
                and task_record.status != "terminal"
                and task_record.id not in self.live_tasks
            ):
                return True
        return False

    def put_input(self, input_record: Input) -> None:
        input_record = _copy_input(input_record)
        self.writes.append(Write(type="input", input=input_record))
        self.changes.inputs.append(input_record)
        self.inputs_by_id[input_record.id] = input_record
        if input_record.request_id is not None:
            self.inputs_by_request[
                f"{input_record.conversation_id}:{input_record.request_id}"
            ] = input_record

    async def send(self, conversation_id: Id, input: SendInput) -> Id:
        self.assert_core("send")
        if input.request_id is not None:
            existing = await self.input_by_request(conversation_id, input.request_id)
            if existing is not None:
                return existing.id
        if self.busy(conversation_id):
            mode = input.when_busy or "followUp"
            if mode == "reject":
                raise ConversationBusy(conversation_id)
            id = self.storage.mint_id()
            self.raw(DocRef(doc="sticky", conversation_id=conversation_id)).setdefault("inbox", []).append(
                {"id": id, "mode": mode, "input": copy.deepcopy(input.content)}
            )
            self.put_input(
                Input(
                    id=id,
                    conversation_id=conversation_id,
                    status="queued",
                    request_id=input.request_id,
                )
            )
            self.changes.events.append(
                ScopedEvent(
                    conversation_id=conversation_id,
                    event=ViewEvent(type="input.queued", input=id, mode=mode),
                )
            )
            return id
        # Idle: read the head before the first append, place older queued items, then this one.
        head = await self.newest_entry(conversation_id, {"withHead": True})
        boundary = await self.boundary(conversation_id, "final", head.id if head is not None else None)
        id = self.storage.mint_id()
        event_start = len(self.changes.events)
        entry = self.append_entry_internal(
            conversation_id,
            NewEntry(
                kind="pi.user",
                model=[{"role": "user", "content": input.content, "timestamp": self.now()}],
            ),
        )
        entry_events = self.changes.events[event_start:]
        del self.changes.events[event_start:]
        self.put_input(
            Input(
                id=id,
                conversation_id=conversation_id,
                status="placed",
                entry=entry,
                request_id=input.request_id,
            )
        )
        self.changes.events.append(
            ScopedEvent(
                conversation_id=conversation_id,
                event=ViewEvent(type="input.placed", input=id, entry=entry),
            )
        )
        self.changes.events.extend(entry_events)
        inputs = [*boundary["triggers"], id]
        self.create_task_internal(
            TaskSpec(kind="pi.generation", conversation_id=conversation_id, input={"inputs": inputs})
        )
        self.changes.events.append(
            ScopedEvent(
                conversation_id=conversation_id,
                event=ViewEvent(type="turn.started", inputs=inputs),
            )
        )
        return id

    async def set_input(self, id: Id, patch: Dict[str, Any]) -> None:
        current = await self.input(id)
        if current is None:
            raise ValueError(f"input {id} not found")
        merged = _copy_input(current)
        for key, value in patch.items():
            setattr(merged, key, value)
        self.put_input(merged)

    async def withdraw_input(self, id: Id) -> str:
        """Kernel: withdraw a queued input."""
        self.assert_core("withdrawInput")
        input_record = await self.input(id)
        if input_record is None:
            return "not_found"
        if input_record.status != "queued":
            return "already_placed"
        await self.preload([DocRef(doc="sticky", conversation_id=input_record.conversation_id)])
        remove_where(
            self.raw(
                DocRef(doc="sticky", conversation_id=input_record.conversation_id)
            ).setdefault("inbox", []),
            lambda q: q.get("id") == id,
        )
        await self.set_input(id, {"status": "unanswered", "reason": "aborted"})
        self.changes.events.append(
            ScopedEvent(
                conversation_id=input_record.conversation_id,
                event=ViewEvent(type="input.aborted", input=id),
            )
        )
        return "aborted"

    async def resolve_inputs(self, ids: Sequence[Id], resolution: Dict[str, Any]) -> None:
        self.assert_core("resolveInputs")
        for id in ids:
            await self.set_input(id, resolution)

    async def boundary(
        self, conversation_id: Id, at: str, head_boundary: Optional[Id]
    ) -> Dict[str, Any]:
        """Boundary placement (pico §9.5): no storage scan, only the local head."""
        self.assert_core("boundary")
        sticky = self.raw(DocRef(doc="sticky", conversation_id=conversation_id))
        inbox = sorted(list(sticky.setdefault("inbox", [])), key=lambda q: q.get("id") or 0)
        cut: Optional[Id] = None
        for queued in inbox:
            if queued.get("mode") == "write" and (queued.get("entry") or {}).get("head") == "self":
                cut = queued["id"]
        stale = (
            []
            if cut is None
            else [q["id"] for q in inbox if q.get("mode") != "write" and q["id"] < cut]
        )
        for id in stale:
            await self.set_input(id, {"status": "unanswered", "reason": "stale"})
            self.changes.events.append(
                ScopedEvent(
                    conversation_id=conversation_id,
                    event=ViewEvent(type="input.aborted", input=id),
                )
            )
        survivors = [q for q in inbox if q["id"] not in stale]

        def pick(mode: str, policy: Optional[str]) -> List[Dict[str, Any]]:
            items = [q for q in survivors if q.get("mode") == mode]
            return items if policy == "all" else items[:1]

        selected: Set[Id] = set(stale)
        for queued in survivors:
            if queued.get("mode") == "write":
                selected.add(queued["id"])
        for queued in pick("steer", sticky.get("steeringMode")):
            selected.add(queued["id"])
        if at == "final":
            for queued in pick("followUp", sticky.get("followUpMode")):
                selected.add(queued["id"])
        head = head_boundary
        triggers: List[Id] = []
        for queued in survivors:
            if queued["id"] not in selected:
                continue
            entry_spec = queued.get("entry") or {}
            if queued.get("mode") == "write":
                queued_head = entry_spec.get("head")
                if (
                    queued_head is not None
                    and queued_head != "self"
                    and head is not None
                    and queued_head < head
                ):
                    await self.set_input(queued["id"], {"status": "unanswered", "reason": "stale"})
                    self.changes.events.append(
                        ScopedEvent(
                            conversation_id=conversation_id,
                            event=ViewEvent(type="input.aborted", input=queued["id"]),
                        )
                    )
                    continue
                event_start = len(self.changes.events)
                entry = self.append_entry_internal(
                    conversation_id, new_entry_from_json(entry_spec)
                )
                entry_events = self.changes.events[event_start:]
                del self.changes.events[event_start:]
                if queued_head is not None:
                    head = entry if queued_head == "self" else queued_head
                await self.set_input(queued["id"], {"status": "done", "entry": entry})
                self.changes.events.append(
                    ScopedEvent(
                        conversation_id=conversation_id,
                        event=ViewEvent(type="input.placed", input=queued["id"], entry=entry),
                    )
                )
                self.changes.events.extend(entry_events)
            else:
                event_start = len(self.changes.events)
                entry = self.append_entry_internal(
                    conversation_id,
                    NewEntry(
                        kind="pi.user",
                        model=[{"role": "user", "content": queued.get("input"), "timestamp": self.now()}],
                    ),
                )
                entry_events = self.changes.events[event_start:]
                del self.changes.events[event_start:]
                await self.set_input(queued["id"], {"status": "placed", "entry": entry})
                self.changes.events.append(
                    ScopedEvent(
                        conversation_id=conversation_id,
                        event=ViewEvent(type="input.placed", input=queued["id"], entry=entry),
                    )
                )
                self.changes.events.extend(entry_events)
                triggers.append(queued["id"])
        remove_where(sticky["inbox"], lambda q: q.get("id") in selected)
        return {"triggers": triggers, "terminated": cut is not None}

    # --- finish -------------------------------------------------------------

    def finish(self) -> List[Write]:
        if self.poisoned is not None:
            raise self.poisoned
        for conversation_id, keys in self.changed_config.items():
            self.changes.events.append(
                ScopedEvent(
                    conversation_id=conversation_id,
                    event=ViewEvent(type="config.changed", keys=sorted(keys)),
                )
            )
        for ref, tracker in self.touched.values():
            ops = tracker.flush()
            if not ops:
                continue
            self.writes.append(Write(type="doc", ref=ref, ops=ops))
            self.changes.docs.append(DocChange(ref=ref, ops=ops))
            self.docs.note_ops(ref, ops)
        return self.writes

    def revoke(self) -> None:
        """Every wrapper handed out by this transaction throws from now on."""
        self.membrane.revoke()
        self.namespace_views.clear()

    def evict_touched(self) -> None:
        for ref, _tracker in self.touched.values():
            self.docs.evict(ref)


def _with_checkpoint(task: Task, checkpoint: Checkpoint) -> Task:
    updated = _copy_task(task)
    updated.checkpoint = checkpoint
    return updated


def _json_of(value: Any) -> Any:
    try:
        return json.dumps(plain(value), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _patch_from_dict(patch: Dict[str, Any]) -> TaskPatch:
    from .types import task_patch_from_json

    return task_patch_from_json(patch)


def _conversation_parent(conversation_id: Id, at: Id) -> Any:
    from .types import ConversationParent

    return ConversationParent(conversation_id=conversation_id, at=at)


def queued_input_to_json_plain(queued: QueuedInput) -> JsonObject:
    from .types import queued_input_to_json

    return queued_input_to_json(queued)


def _is_event_name(name: str) -> bool:
    if not name:
        return False
    first = name[0]
    if not (first.isalpha() and first.isascii()):
        return False
    return all(char.isascii() and (char.isalnum() or char in "_.-") for char in name)


# ---------------------------------------------------------------------------
# Session: the line
# ---------------------------------------------------------------------------

try:  # pragma: no cover - depends on the storage type supporting weak references
    _owners: Any = weakref.WeakKeyDictionary()
except TypeError:  # pragma: no cover - fallback for non-weakref-able storages
    _owners = {}


def _owners_has(storage: Storage) -> bool:
    try:
        return storage in _owners
    except TypeError:  # pragma: no cover - unhashable storage
        return False


def _owners_add(storage: Storage, session: "Session") -> None:
    _owners[storage] = session


def _owners_delete(storage: Storage) -> None:
    try:
        del _owners[storage]
    except (KeyError, TypeError):  # pragma: no cover - already gone
        pass


def _as_doc_ref(value: Any) -> DocRef:
    """Accept a :class:`~.types.DocRef` or the TS-shaped literal ``{doc, conversationId}``."""
    if isinstance(value, DocRef):
        return value
    if isinstance(value, dict):
        doc = str(value.get("doc", "session"))
        conversation_id = value.get("conversationId", value.get("conversation_id"))
        return DocRef(doc=doc, conversation_id=None if doc == "session" else conversation_id)
    raise TypeError(f"unsupported document reference {value!r}")


def _normalize_invoker(invoker: Any) -> Invoker:
    """Accept an :class:`~.types.Invoker` or the TS-shaped literal ``{type, conversationId}``."""
    if isinstance(invoker, Invoker):
        return invoker
    if isinstance(invoker, dict):
        return Invoker(
            type=str(invoker.get("type", "kernel")),
            conversation_id=invoker.get("conversationId"),
            id=invoker.get("id"),
            mode=str(invoker.get("mode", "run")),
        )
    raise TypeError(f"unsupported invoker {invoker!r}")


async def _resolve(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


class Session:
    """The serialized line: one mutation at a time, each with its own transaction."""

    def __init__(
        self,
        storage: Storage,
        kinds: Union[Dict[str, AnyKind], Iterable[AnyKind]],
        namespaces: Union[Dict[str, NamespaceRegistration], Iterable[NamespaceRegistration]],
        now: Optional[Callable[[], int]] = None,
    ) -> None:
        self.storage = storage
        self.now = now or (lambda: 0)
        if isinstance(kinds, dict):
            self.kinds = dict(kinds)
        else:
            self.kinds = {kind.name: kind for kind in kinds}
        if isinstance(namespaces, dict):
            self.namespaces = dict(namespaces)
        else:
            self.namespaces = {registration.token.id: registration for registration in namespaces}
        if _owners_has(storage):
            raise ValueError("this Storage already has an owning Session")
        self.defaults = Defaults(self.kinds.values())
        self.docs = Docs(storage, self.defaults)
        _owners_add(storage, self)
        self._tail: asyncio.Future = _completed_future()
        self.closed = False
        self.fault: Optional[Faulted] = None
        self.live_tasks: Dict[Id, Task] = {}
        #: Owner/parent graph, loaded at open and maintained on every commit.
        self.conversation_records: Dict[Id, Conversation] = {}
        self.line_listeners: Set[Callable[[CommitResult], None]] = set()
        self.listeners: Set[Callable[[CommitResult], None]] = set()
        #: Errors from listeners and other post-commit work; never surface to the writer.
        self.on_report: Callable[[Any], None] = lambda _error: None
        #: Owner tasks that are terminal but whose conversations still exist (ancestry after reopen).
        self.owner_task_cache: Dict[Id, Task] = {}
        self.index = _SessionIndex(self)

    # -- line --------------------------------------------------------------

    async def commit(
        self,
        invoker: Any,
        fn: Callable[[Any, Context, TransactionControl], Any],
        ctx: Context,
        opts: Optional[Dict[str, Any]] = None,
    ) -> CommitResult:
        opts = opts or {}
        if ctx.abort_signal is not None:
            ctx.abort_signal.throw_if_aborted()
        authority = _normalize_invoker(invoker)

        async def operation(line_ctx: Context) -> CommitResult:
            self.assert_usable()
            if line_ctx.abort_signal is not None:
                line_ctx.abort_signal.throw_if_aborted()
            if authority.type == "task":
                # A captured runtime cannot write after its invocation returned,
                # after terminalization, or after a mark (run mode).
                if authority.token is None or not authority.token.alive:
                    raise Forbidden("commit from a finished invocation")
                live = self.live_tasks.get(authority.id or 0)
                if live is None:
                    raise Forbidden("commit from a task that is not live")
                if authority.mode == "run" and live.abort is True:
                    raise Forbidden("commit from a marked run invocation")
            tx = TxImpl(
                authority,
                self.storage,
                self.docs,
                self.defaults,
                self.kinds,
                self.namespaces,
                self.live_tasks,
                self.index,
                self.now,
                line_ctx,
            )
            tx.closing = opts.get("closing") is True
            # ``docs`` accepts DocRef records and the TS-shaped literals callers write inline.
            refs: List[DocRef] = [DocRef(doc="session")]
            for ref in opts.get("docs") or []:
                refs.append(_as_doc_ref(ref))
            if authority.type == "task":
                refs.append(DocRef(doc="rewindable", conversation_id=authority.conversation_id))
                refs.append(DocRef(doc="sticky", conversation_id=authority.conversation_id))
            persisted = False
            try:
                await tx.preload(refs)
                try:
                    value = await _resolve(fn(tx.callback_surface(), line_ctx, TransactionControlImpl(tx)))
                finally:
                    tx.close_surface()
                await tx.preload(
                    [
                        DocRef(doc="sticky", conversation_id=conversation_id)
                        for conversation_id in {
                            task.conversation_id for task in tx.changes.tasks
                        }
                    ]
                )
                writes = tx.finish()
                seq: Optional[Seq] = None
                if writes or tx.changes.events:
                    persisted = True
                    try:
                        seq = await self.storage.commit(writes, without_abort_signal(line_ctx))
                    except Exception as error:  # noqa: BLE001 - the line faults
                        self.fault = Faulted(error)
                        try:
                            await self.storage.close(without_abort_signal(line_ctx))
                            _owners_delete(self.storage)
                        except Exception:  # noqa: BLE001 - best effort, as in TS
                            pass
                        raise self.fault
                    self.apply_changes(tx.changes)
                result = CommitResult(value=value, seq=seq, changes=tx.changes)
                if seq is not None:
                    for listener in list(self.line_listeners):
                        try:
                            listener(result)
                        except Exception as error:  # noqa: BLE001 - reported, never surfaces
                            try:
                                self.on_report(error)
                            except Exception:  # noqa: BLE001 - a reporter must not break the line
                                pass
                return result
            except BaseException:
                if not persisted:
                    tx.evict_touched()
                raise
            finally:
                tx.revoke()

        result = await self.enter(ctx, operation)
        if result.seq is not None:
            for listener in list(self.listeners):
                try:
                    listener(result)
                except Exception as error:  # noqa: BLE001 - reported, never surfaces
                    self.on_report(error)
        return result

    async def read(self, fn: Callable[[Storage, Context], Any], ctx: Context) -> Any:
        """Read-only line operation without a transaction."""
        self.assert_usable()
        return await self.enter(ctx, lambda line_ctx: _resolve(fn(self.storage, line_ctx)))

    async def on_line(self, fn: Callable[[Context], Any], ctx: Context) -> Any:
        """Generic line operation (waiter registration and asynchronous lifecycle work)."""
        self.assert_usable()
        return await self.enter(ctx, lambda line_ctx: _resolve(fn(line_ctx)))

    # -- lifecycle ---------------------------------------------------------

    async def retire(self, task: Task, ctx: Context) -> None:
        """After a task terminalizes: retire its slot; base + truncate when idle or over budget."""
        ref = DocRef(doc="sticky", conversation_id=task.conversation_id)
        idle = not any(
            other.conversation_id == task.conversation_id for other in self.live_tasks.values()
        )
        over = self.docs.since_base.get(doc_key(ref), 0) > STICKY_BASE_BUDGET

        def clear(tx: Any, _ctx: Any = None, _control: Any = None) -> None:
            (tx.sticky(task.conversation_id).setdefault("tasks", {})).pop(str(task.id), None)
            if idle or over:
                self.docs.request_base(ref)

        await self.commit({"type": "kernel"}, clear, ctx, {"docs": [ref]})
        if idle or over:
            await self.enter(
                ctx,
                lambda line_ctx: self.storage.truncate(ref, without_abort_signal(line_ctx)),
            )

    async def fork(
        self,
        parent_id: Id,
        at: Union[Id, str],
        spec: Dict[str, Any],
        ctx: Context,
    ) -> Id:
        inherited: Optional[RewindableState] = None
        if at != "start":
            async def load(storage: Storage, line_ctx: Context) -> Optional[RewindableState]:
                rows = await storage.scan_entries(
                    EntryScan(conversation_id=parent_id, before=at + 1, limit=1), line_ctx
                )
                if not rows or rows[0].id != at:
                    raise ValueError(f"entry {at} is not visible from conversation {parent_id}")
                return await storage.doc_as_of(parent_id, at, line_ctx)

            inherited = await self.read(load, ctx)
        raw_overrides = copy.deepcopy(spec.get("rewindable") or {})
        raw_overrides.pop("plugins", None)
        overrides = self.defaults.validate_seed("rewindable", raw_overrides)
        if inherited is None:
            rewindable = overrides
        else:
            rewindable = {
                **inherited,
                **overrides,
                "plugins": copy.deepcopy(inherited.get("plugins") or {}),
            }

        def create(tx: Any, _ctx: Any = None, _control: Any = None) -> Id:
            return tx.create_fork_conversation(
                ConversationSpec(
                    parent={"conversationId": parent_id, "at": at},
                    rewindable=rewindable,
                    sticky=spec.get("sticky"),
                    sections=spec.get("sections"),
                )
            )

        result = await self.commit({"type": "kernel"}, create, ctx)
        return result.value

    async def close(self, ctx: Context) -> None:
        async def shutdown(line_ctx: Context) -> None:
            if self.closed:
                return
            await self.storage.close(without_abort_signal(line_ctx))
            self.closed = True
            _owners_delete(self.storage)

        await self.enter(ctx, shutdown)

    def loaded_document(self, ref: DocRef) -> Optional[JsonObject]:
        return self.docs.loaded(ref)

    # -- internals ---------------------------------------------------------

    def apply_changes(self, changes: CommitChanges) -> None:
        for task in changes.tasks:
            if task.status == "terminal":
                self.live_tasks.pop(task.id, None)
                if task.owns:
                    self.owner_task_cache[task.id] = task
            else:
                self.live_tasks[task.id] = task
        for conversation in changes.conversations:
            self.conversation_records[conversation.id] = conversation

    def enter(self, ctx: Context, op: Callable[[Context], Any]) -> Any:
        if ctx.value(LINE_KEY) is True:
            return _failed_future(NestedLineOperation())
        previous = self._tail
        # Reserve this operation's own ticket: releasing a future that a later
        # operation has since replaced would deadlock the line.
        mine = _pending_future()
        self._tail = mine

        async def run() -> Any:
            await previous
            try:
                return await _resolve(op(with_context_value(LINE_KEY, True, ctx)))
            finally:
                if not mine.done():
                    mine.set_result(None)

        return run()

    def assert_usable(self) -> None:
        if self.fault is not None:
            raise self.fault
        if self.closed:
            raise Closed()


class _SessionIndex(ConversationIndex):
    """The session-side conversation index: owner/parent graph for subtree checks and ancestry."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, id: Id) -> Optional[Conversation]:
        return self._session.conversation_records.get(id)

    def subtree(self, root: Id) -> Set[Id]:
        out: Set[Id] = {root}
        grew = True
        while grew:
            grew = False
            for conversation in self._session.conversation_records.values():
                if conversation.id in out or conversation.owner is None:
                    continue
                owner_task = self._session.live_tasks.get(
                    conversation.owner
                ) or self._session.owner_task_cache.get(conversation.owner)
                if owner_task is not None and owner_task.conversation_id in out:
                    out.add(conversation.id)
                    grew = True
        return out

    def ancestors(self, id: Id) -> List[Id]:
        chain: List[Id] = []
        conversation = self._session.conversation_records.get(id)
        while conversation is not None and conversation.owner is not None:
            task = self._session.live_tasks.get(
                conversation.owner
            ) or self._session.owner_task_cache.get(conversation.owner)
            if task is None:
                break
            chain.insert(0, task.conversation_id)
            conversation = self._session.conversation_records.get(task.conversation_id)
        return chain


def _event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:  # pragma: no cover - no loop has ever been created
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def _completed_future() -> asyncio.Future:
    future: asyncio.Future = _event_loop().create_future()
    future.set_result(None)
    return future


def _pending_future() -> asyncio.Future:
    return _event_loop().create_future()


def _failed_future(error: Exception) -> Any:
    async def fail() -> Any:
        raise error

    return fail()


_ = (Any, EntryScan, Runtime, ThinkingLevel, UNSET, to_stored)
