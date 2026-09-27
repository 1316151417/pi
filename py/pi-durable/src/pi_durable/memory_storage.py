"""Detached reference storage from ``packages/durable/src/memory-storage.ts``."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from dataclasses import fields, is_dataclass
from decimal import Decimal
from functools import wraps
from typing import Literal, cast

from pi_chord._undefined import UNDEFINED, Undefined
from pi_chord.context import Context
from pi_chord.types import JsonValue

from .types import (
    ConversationParent, ConversationRecord, Cursor, EntryAtCommit, EntryQuery, EntryRecord, HeadEntryRecord,
    Id, Input, Page, Seq, StorageWrite, TaskQuery, TaskRecord,
)

type _StoredTask = TaskRecord[JsonValue, JsonValue, JsonValue]
type _TaskStatus = Literal["pending", "running", "terminal"]
type _TableName = Literal["conversation", "entry", "task", "input"]
_NAN_KEY = float("nan")


class _Settled[T](Awaitable[T]):
    """An eagerly settled, reusable Promise represented as a Python awaitable."""

    def __init__(self, value: T | Undefined = UNDEFINED, error: Exception | None = None) -> None:
        self._value, self._error = value, error

    def __await__(self) -> Generator[None, None, T]:
        # Awaiting a settled JS Promise still resumes in a later microtask.
        yield
        if self._error is not None:
            raise self._error
        return cast(T, self._value)


def _promise[**P, T](operation: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    @wraps(operation)
    def invoke(*args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
        try:
            return _Settled(operation(*args, **kwargs))
        except Exception as error:
            return _Settled(error=error)

    return invoke


def _clone[T](value: T) -> T:
    if isinstance(value, (list, tuple)):
        return cast(T, [_clone(item) for item in value])
    if isinstance(value, Mapping):
        # Object.keys places canonical array-index strings before other keys.
        numeric: list[tuple[int, str]] = []
        other: list[str] = []
        for key in value:
            if isinstance(key, str) and key.isascii() and key.isdigit() and len(key) <= 10 and (key == "0" or key[0] != "0") and int(key) < 0xFFFFFFFF:
                numeric.append((int(key), key))
            else:
                other.append(cast(str, key))
        return cast(T, {key: _clone(value[key]) for key in [key for _, key in sorted(numeric)] + other})
    if is_dataclass(value) and not isinstance(value, type):
        # pi-ai represents its plain TS message records as dataclasses. Keep
        # that Python representation, without invoking constructors/hooks.
        result = object.__new__(type(value))
        names = dict.fromkeys([item.name for item in fields(value)])
        if hasattr(value, "__dict__"):
            names.update(dict.fromkeys(vars(value)))
        for name in names:
            object.__setattr__(result, name, _clone(getattr(value, name)))
        return cast(T, result)
    if value is None or isinstance(value, (str, bool, int, float, Undefined)) or callable(value):
        return value
    return cast(T, _clone(vars(value)) if hasattr(value, "__dict__") else {})


def _number(value: Id) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _key(value: Id) -> float:
    number = _number(value)
    return _NAN_KEY if math.isnan(number) else number


def _id_string(value: Id) -> str:
    number = _number(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    if number == 0:
        return "0"
    decimal = Decimal(repr(number))
    if 1e-6 <= abs(number) < 1e21:
        fixed = format(decimal, "f")
        return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    mantissa, exponent = format(decimal.normalize(), "e").split("e")
    return f"{mantissa}e{int(exponent):+d}"


def _lower_bound(ids: Sequence[Id], target: Id) -> int:
    low, high = 0, len(ids)
    while low < high:
        middle = ((low + high) & 0xFFFFFFFF) >> 1
        if _number(ids[middle]) < _number(target):
            low = middle + 1
        else:
            high = middle
    return low


def _upper_bound(ids: Sequence[Id], target: Id) -> int:
    low, high = 0, len(ids)
    while low < high:
        middle = ((low + high) & 0xFFFFFFFF) >> 1
        if _number(ids[middle]) <= _number(target):
            low = middle + 1
        else:
            high = middle
    return low


def _insert_sorted(ids: list[Id], id: Id) -> None:
    if not ids or _number(ids[-1]) < _number(id):
        ids.append(id)
    else:
        ids.insert(_lower_bound(ids, id), id)


def _remove_sorted(ids: list[Id], id: Id) -> None:
    index = _lower_bound(ids, id)
    if index < len(ids) and _number(ids[index]) == _number(id):
        del ids[index]


def _minimum(left: Id, right: Id) -> float:
    first, second = _number(left), _number(right)
    return math.nan if math.isnan(first) or math.isnan(second) else min(first, second)


def _slice[T](items: Sequence[T], start: Id, end: Id) -> list[T]:
    bounds: list[int] = []
    for raw in (start, end):
        number = _number(raw)
        if math.isnan(number) or number == -math.inf:
            bounds.append(0)
        elif number == math.inf:
            bounds.append(len(items))
        else:
            integer = math.trunc(number)
            bounds.append(max(len(items) + integer, 0) if integer < 0 else min(integer, len(items)))
    return list(items[bounds[0]:bounds[1]])


def _page[T](values: Sequence[T], limit: Id) -> Page[T, Cursor]:
    items = _slice(values, 0, limit)
    if len(values) <= _number(limit):
        return {"items": _clone(items)}
    if not items:
        raise TypeError("Cannot read properties of undefined (reading 'id')")
    last = cast(Mapping[str, JsonValue], items[-1])
    return {"items": _clone(items), "next": {"after": last["id"]}}


class MemoryStorage:
    """Atomic batches with global IDs, detached records, and fork-aware scans.

    Like the source, semantic validity of records and transitions belongs to
    the owning Session; this backend checks ID ownership and immutability.
    Methods execute immediately and return an awaitable result or rejection.
    """

    def __init__(self) -> None:
        self._conversations: dict[float, ConversationRecord] = {}
        self._conversation_ids: list[Id] = []
        self._entries: dict[float, EntryRecord] = {}
        self._entry_ids: dict[float, list[Id]] = {}
        self._head_entry_ids: dict[float, list[Id]] = {}
        self._entry_commit_seqs: dict[float, Seq] = {}
        self._tasks: dict[float, _StoredTask] = {}
        self._task_ids: list[Id] = []
        self._task_ids_by_status: dict[_TaskStatus, list[Id]] = {"pending": [], "running": [], "terminal": []}
        self._inputs: dict[float, Input] = {}
        self._input_ids_by_request: dict[float, dict[str, Id]] = {}
        self._next_id: Id = 2
        self._next_seq: Seq = 1
        self._closed = False

    @_promise
    def commit(self, writes: Sequence[StorageWrite], context: Context) -> Seq:
        self._assert_open()
        prepared = [_clone(write) for write in writes]
        self._check_immutable_ids(prepared)
        seq = self._next_seq
        for write in prepared:
            match write["type"]:
                case "conversation":
                    conversation = cast(ConversationRecord, write["value"])
                    self._conversations[_key(conversation["id"])] = conversation
                    _insert_sorted(self._conversation_ids, conversation["id"])
                case "entry":
                    entry = cast(EntryRecord, write["value"])
                    id, conversation_id = entry["id"], _key(entry["conversationId"])
                    self._entries[_key(id)] = entry
                    self._entry_commit_seqs[_key(id)] = seq
                    _insert_sorted(self._entry_ids.setdefault(conversation_id, []), id)
                    if entry.get("head", UNDEFINED) is not UNDEFINED:
                        _insert_sorted(self._head_entry_ids.setdefault(conversation_id, []), id)
                case "task":
                    task = cast(_StoredTask, write["value"])
                    id, status = task["id"], task["state"]["status"]
                    previous_task = self._tasks.get(_key(id))
                    if previous_task is None:
                        _insert_sorted(self._task_ids, id)
                        _insert_sorted(self._task_ids_by_status[status], id)
                    elif previous_task["state"]["status"] != status:
                        _remove_sorted(self._task_ids_by_status[previous_task["state"]["status"]], id)
                        _insert_sorted(self._task_ids_by_status[status], id)
                    self._tasks[_key(id)] = task
                case "input":
                    record = cast(Input, write["value"])
                    id = record["id"]
                    previous_input = self._inputs.get(_key(id))
                    if previous_input is not None and previous_input.get("requestId", UNDEFINED) is not UNDEFINED:
                        previous_id = _key(previous_input["conversationId"])
                        previous_requests = self._input_ids_by_request.get(previous_id)
                        if previous_requests is not None and previous_requests.get(previous_input["requestId"], UNDEFINED) == id:
                            del previous_requests[previous_input["requestId"]]
                            if not previous_requests:
                                del self._input_ids_by_request[previous_id]
                    self._inputs[_key(id)] = record
                    if record.get("requestId", UNDEFINED) is not UNDEFINED:
                        requests = self._input_ids_by_request.setdefault(_key(record["conversationId"]), {})
                        requests[record["requestId"]] = id
            candidate = _number(write["value"]["id"]) + 1.0
            self._next_id = math.nan if math.isnan(candidate) or math.isnan(_number(self._next_id)) else max(_number(self._next_id), candidate)
        self._next_seq = _number(self._next_seq) + 1.0
        return seq

    def mint_id(self) -> Id:
        self._assert_open()
        current = _number(self._next_id)
        if not math.isfinite(current) or not current.is_integer() or abs(current) > 2**53 - 1:
            raise RuntimeError("ID space is exhausted")
        self._next_id = current + 1.0
        return int(current)

    @_promise
    def conversation(self, id: Id, context: Context) -> ConversationRecord | Undefined:
        self._assert_open()
        return _clone(self._conversations.get(_key(id), UNDEFINED))

    @_promise
    def scan_conversations(self, cursor: Cursor | Undefined, limit: Id, context: Context) -> Page[ConversationRecord, Cursor]:
        self._assert_open()
        after = UNDEFINED if cursor is UNDEFINED or cursor is None else cursor.get("after", UNDEFINED)
        start = 0 if after is UNDEFINED else _upper_bound(self._conversation_ids, cast(Id, after))
        ids = _slice(self._conversation_ids, start, start + _number(limit) + 1.0)
        return _page([self._conversations[_key(id)] for id in ids], limit)

    @_promise
    def entry(self, id: Id, context: Context) -> EntryAtCommit | Undefined:
        self._assert_open()
        entry = self._entries.get(_key(id))
        if entry is None:
            return UNDEFINED
        return {"entry": _clone(entry), "commitSeq": self._entry_commit_seqs[_key(id)]}

    @_promise
    def find_latest_head_marker(self, conversation_id: Id, at_or_before_entry_id: Id | Undefined, context: Context) -> HeadEntryRecord | Undefined:
        self._assert_open()
        if _key(conversation_id) not in self._conversations:
            raise RuntimeError(f"Unknown conversation: {_id_string(conversation_id)}")
        current_id = conversation_id
        upper_entry_id = math.inf if at_or_before_entry_id is UNDEFINED or at_or_before_entry_id is None else at_or_before_entry_id
        while True:
            ids = self._head_entry_ids.get(_key(current_id), [])
            index = _upper_bound(ids, upper_entry_id) - 1
            if index >= 0:
                entry = self._entries[_key(ids[index])]
                return cast(HeadEntryRecord, _clone({**entry, "head": entry["head"]}))
            conversation = self._conversations[_key(current_id)]
            parent = conversation.get("parent", UNDEFINED)
            if parent is UNDEFINED:
                return UNDEFINED
            parent = cast(ConversationParent, parent)
            upper_entry_id = _minimum(upper_entry_id, parent["at"])
            current_id = parent["conversationId"]

    @_promise
    def scan_entries(self, query: EntryQuery, cursor: Cursor | Undefined, limit: Id, context: Context) -> Page[EntryRecord, Cursor]:
        self._assert_open()
        after = UNDEFINED if cursor is UNDEFINED or cursor is None else cursor.get("after", UNDEFINED)
        maximum = query.get("maxEntryId", UNDEFINED)
        if after is not UNDEFINED:
            maximum = _minimum(math.inf if maximum is UNDEFINED or maximum is None else maximum, _number(cast(Id, after)) - 1.0)
        visible: list[EntryRecord] = []
        for entry in self._visible_entries(query["conversationId"], query.get("minEntryId", UNDEFINED), maximum):
            visible.append(entry)
            if len(visible) > _number(limit):
                break
        return _page(visible, limit)

    @_promise
    def task(self, id: Id, context: Context) -> _StoredTask | Undefined:
        self._assert_open()
        return _clone(self._tasks.get(_key(id), UNDEFINED))

    @_promise
    def scan_tasks(self, query: TaskQuery, cursor: Cursor | Undefined, limit: Id, context: Context) -> Page[_StoredTask, Cursor]:
        self._assert_open()
        after = UNDEFINED if cursor is UNDEFINED or cursor is None else cursor.get("after", UNDEFINED)
        status = query.get("status", UNDEFINED)
        ids = self._task_ids if status is UNDEFINED else self._task_ids_by_status[cast(_TaskStatus, status)]
        start = 0 if after is UNDEFINED else _upper_bound(ids, cast(Id, after))
        values: list[_StoredTask] = []
        index = start
        while index < len(ids) and len(values) <= _number(limit):
            value = self._tasks[_key(ids[index])]
            index += 1
            if query.get("conversationId", UNDEFINED) is not UNDEFINED and value["conversationId"] != query["conversationId"]:
                continue
            if query.get("kind", UNDEFINED) is not UNDEFINED and value["kind"] != query["kind"]:
                continue
            if query.get("abortRequested", UNDEFINED) is not UNDEFINED and value["abortRequested"] != query["abortRequested"]:
                continue
            if query.get("background", UNDEFINED) is not UNDEFINED and value["background"] != query["background"]:
                continue
            values.append(value)
        return _page(values, limit)

    @_promise
    def input(self, id: Id, context: Context) -> Input | Undefined:
        self._assert_open()
        return _clone(self._inputs.get(_key(id), UNDEFINED))

    @_promise
    def input_by_request(self, conversation_id: Id, request_id: str, context: Context) -> Input | Undefined:
        self._assert_open()
        requests = self._input_ids_by_request.get(_key(conversation_id), {})
        id = requests.get(request_id, UNDEFINED)
        return UNDEFINED if id is UNDEFINED else _clone(self._inputs[_key(cast(Id, id))])

    @_promise
    def close(self, context: Context) -> None:
        self._closed = True

    def _visible_entries(self, conversation_id: Id, min_entry_id: Id | Undefined = UNDEFINED, max_entry_id: Id | Undefined = UNDEFINED) -> Generator[EntryRecord, None, None]:
        if _key(conversation_id) not in self._conversations:
            raise RuntimeError(f"Unknown conversation: {_id_string(conversation_id)}")
        lower = -math.inf if min_entry_id is UNDEFINED else cast(Id, min_entry_id)
        upper = math.inf if max_entry_id is UNDEFINED else cast(Id, max_entry_id)
        current_id = conversation_id
        while True:
            ids = self._entry_ids.get(_key(current_id), [])
            for index in range(_upper_bound(ids, upper) - 1, -1, -1):
                id = ids[index]
                if _number(id) < _number(lower):
                    break
                yield self._entries[_key(id)]
            conversation = self._conversations[_key(current_id)]
            parent = conversation.get("parent", UNDEFINED)
            if parent is UNDEFINED:
                break
            parent = cast(ConversationParent, parent)
            upper = _minimum(upper, parent["at"])
            if upper < _number(lower):
                break
            current_id = parent["conversationId"]

    def _check_immutable_ids(self, writes: Sequence[StorageWrite]) -> None:
        claimed: dict[float, _TableName] = {}
        for write in writes:
            table, id = write["type"], write["value"]["id"]
            key = _key(id)
            existing: _TableName | None = None
            if key in self._conversations:
                existing = "conversation"
            elif key in self._entries:
                existing = "entry"
            elif key in self._tasks:
                existing = "task"
            elif key in self._inputs:
                existing = "input"
            earlier = claimed.get(key)
            if table in ("conversation", "entry"):
                if existing is not None:
                    raise RuntimeError(f"ID {_id_string(id)} already belongs to {existing}")
                if earlier is not None:
                    raise RuntimeError(f"ID {_id_string(id)} is written more than once")
            else:
                if existing is not None and existing != table:
                    raise RuntimeError(f"ID {_id_string(id)} already belongs to {existing}")
                if earlier is not None and earlier != table:
                    raise RuntimeError(f"ID {_id_string(id)} is written as two record types")
            claimed[key] = table

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryStorage is closed")


__all__ = ["MemoryStorage"]
