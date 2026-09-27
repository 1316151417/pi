"""Operation-log JSON tracking, application, and streaming path compression.

Decoded operations use Python tuples; encoded operations and paths use JSON
arrays (Python lists). JSON-decoded lists are accepted at every boundary.
Tracked dictionaries and lists expose mutable mapping/sequence views. Strings
use UTF-16 code units, including unpaired surrogates, as in JavaScript.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from functools import cmp_to_key
import json
import math
from typing import Literal, cast, overload
from weakref import WeakValueDictionary

from .types import JsonValue
from ._undefined import UNDEFINED, Undefined

type Seg = str | int
type Path = tuple[Seg, ...]
type NonEmptyPath = tuple[Seg, ...]
type PathRef = Path | int
type Op = (
    tuple[Literal["r"], JsonValue]
    | tuple[Literal["s"], NonEmptyPath, JsonValue]
    | tuple[Literal["d"], NonEmptyPath]
    | tuple[Literal["a"], NonEmptyPath, str]
    | tuple[Literal["t"], NonEmptyPath, int]
    | tuple[Literal["p"], Path, int, int, list[JsonValue]]
)
# Python has no fixed-arity list type. Shape validation below retains the exact
# wire grammar while this representation remains strict JSON without encoding.
type WireOp = list[JsonValue]
type _Container = dict[str, JsonValue] | list[JsonValue]
_MISSING = object()
RESERVED_SEGMENTS = frozenset(("__proto__", "constructor", "prototype"))

__all__ = [
    "JsonValue", "Seg", "Path", "NonEmptyPath", "PathRef", "Op", "WireOp",
    "is_replace", "is_base", "overlap", "TrackerOptions", "Tracker", "track",
    "RESERVED_SEGMENTS", "UnsafePathError", "PathError", "assert_safe_path",
    "assert_valid_op", "assert_valid_wire_op", "apply", "apply_immutable",
    "Encoder", "Decoder", "encoder", "decoder", "TrackedDict", "TrackedList",
    "UNDEFINED", "Undefined",
]


def is_replace(op: Sequence[object]) -> bool:
    return op[0] == "r"


def is_base(ops: Sequence[Sequence[object]]) -> bool:
    return bool(ops) and ops[0][0] == "r"


def _units(value: str) -> str:
    raw = value.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(raw[index] | raw[index + 1] << 8) for index in range(0, len(raw), 2))


def _text(value: str) -> str:
    raw = bytearray()
    for character in value:
        code = ord(character)
        raw.extend((code & 255, code >> 8))
    return raw.decode("utf-16-le", errors="surrogatepass")


def _cut(value: str, count: int) -> str:
    return _text(_units(value)[count:])


def overlap(a: str, b: str, scan: int, probe: int = 64, max_candidates: int = 8) -> int:
    a, b = _units(a), _units(b)
    if not a or not b or scan == 0:
        return 0
    tail = a[len(a) - scan:] if len(a) > scan else a
    for head_length in (min(probe, len(b)), 1):
        head = b[:head_length]
        tried = 0
        offset = tail.find(head)
        while offset != -1:
            tried += 1
            if tried > max_candidates:
                break
            length = len(tail) - offset
            if length <= len(b) and tail[offset:] == b[:length]:
                return length
            offset = tail.find(head, offset + 1)
        if head_length == 1:
            break
    return 0


def _clone(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return value


def _is_container(value: object) -> bool:
    return isinstance(value, (dict, list))


def _same(left: object, right: object) -> bool:
    if _is_container(left) or _is_container(right):
        return left is right
    if isinstance(left, str) and isinstance(right, str):
        return _units(left) == _units(right)
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _equal(left: JsonValue, right: JsonValue) -> bool:
    if _same(left, right):
        return True
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return len(left) == len(right) and all(key in right and _equal(value, right[key]) for key, value in left.items())
    return False


def _set_op(path: Path, value: JsonValue) -> Op:
    return ("s", path, _clone(value)) if path else ("r", _clone(value))


def _diff_string(before: str, after: str, path: Path, scan: int, out: list[Op]) -> None:
    if before == after:
        return
    if not path:
        out.append(_set_op(path, after))
        return
    old, new = _units(before), _units(after)
    if len(new) > len(old) and new[:len(old)] == old:
        out.append(("a", path, _text(new[len(old):])))
        return
    shared = overlap(before, after, scan)
    if shared == 0:
        out.append(("s", path, after))
        return
    out.append(("t", path, len(old) - shared))
    if len(new) > shared:
        out.append(("a", path, _text(new[shared:])))


def _diff(before: object, after: object, path: Path, scan: int, out: list[Op]) -> None:
    if before is _MISSING:
        if after is not _MISSING:
            out.append(_set_op(path, cast(JsonValue, after)))
        return
    if after is _MISSING:
        if not path:
            raise TypeError("the tracked root cannot be deleted")
        out.append(("d", path))
        return
    if _same(before, after):
        return
    if isinstance(before, str) and isinstance(after, str):
        _diff_string(before, after, path, scan, out)
    elif isinstance(before, list) and isinstance(after, list):
        _diff_array(before, after, path, scan, out)
    elif isinstance(before, dict) and isinstance(after, dict):
        if any(key in RESERVED_SEGMENTS for key in (*before, *after)):
            out.append(_set_op(path, after))
            return
        for key, value in after.items():
            _diff(before.get(key, _MISSING), value, (*path, key), scan, out)
        for key in before:
            if key not in after:
                out.append(("d", (*path, key)))
    else:
        out.append(_set_op(path, cast(JsonValue, after)))


def _diff_array(before: list[JsonValue], after: list[JsonValue], path: Path, scan: int, out: list[Op]) -> None:
    if len(before) == len(after):
        for index, value in enumerate(after):
            _diff(before[index], value, (*path, index), scan, out)
        return
    prefix = 0
    while prefix < min(len(before), len(after)) and _equal(before[prefix], after[prefix]):
        prefix += 1
    suffix = 0
    while suffix < min(len(before), len(after)) - prefix and _equal(before[-1 - suffix], after[-1 - suffix]):
        suffix += 1
    shorter = min(len(before), len(after))
    if prefix + suffix == shorter:
        remove = len(before) - prefix - suffix
        items = after[prefix:len(after) - suffix]
        if prefix == 0 and remove == len(before):
            out.append(_set_op(path, after))
        else:
            out.append(("p", path, prefix, remove, cast(list[JsonValue], _clone(items))))
        return
    for index in range(shorter):
        _diff(before[index], after[index], (*path, index), scan, out)
    if len(after) > len(before):
        out.append(("p", path, len(before), 0, cast(list[JsonValue], _clone(after[len(before):]))))
    elif not after:
        out.append(_set_op(path, after))
    else:
        out.append(("p", path, len(after), len(before) - len(after), []))


class UnsafePathError(Exception):
    name = "UnsafePathError"

    def __init__(self, segment: object) -> None:
        self.segment = segment
        super().__init__(f"unsafe path segment: {_js_string(segment)}")


class PathError(Exception):
    name = "PathError"

    def __init__(self, path: Sequence[Seg] | int) -> None:
        self.path = path
        super().__init__(f"unresolvable path: {json.dumps(path, ensure_ascii=False, separators=(',', ':'))}")


def _nonnegative_integer(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return False
    try:
        return math.isfinite(value) and int(value) == value
    except OverflowError:
        return False


def assert_safe_path(path: Sequence[object]) -> None:
    for segment in path:
        if isinstance(segment, str):
            if segment in RESERVED_SEGMENTS:
                raise UnsafePathError(segment)
        elif not _nonnegative_integer(segment):
            raise UnsafePathError(segment)


def _assert_path(path: object, nonempty: bool = False) -> None:
    if not isinstance(path, (tuple, list)):
        raise TypeError("path is not an array")
    if nonempty and not path:
        raise TypeError("path is empty")
    assert_safe_path(path)


def assert_valid_op(op: object) -> None:
    if not isinstance(op, (tuple, list)) or not op:
        raise TypeError("op is not a tuple")
    verb = op[0]
    if verb == "r":
        if len(op) != 2:
            raise TypeError("r arity")
    elif verb in ("s", "d"):
        if len(op) != (3 if verb == "s" else 2):
            raise TypeError(f"{verb} arity")
        _assert_path(op[1], True)
    elif verb == "a":
        if len(op) != 3 or not isinstance(op[2], str):
            raise TypeError("a shape")
        _assert_path(op[1], True)
    elif verb == "t":
        if len(op) != 3 or not _nonnegative_integer(op[2]):
            raise TypeError("t shape")
        _assert_path(op[1], True)
    elif verb == "p":
        if len(op) != 5:
            raise TypeError("p arity")
        _assert_path(op[1])
        if not _nonnegative_integer(op[2]):
            raise TypeError("p index")
        if not _nonnegative_integer(op[3]):
            raise TypeError("p remove")
        if not isinstance(op[4], list):
            raise TypeError("p items")
    else:
        raise TypeError(f"unknown op verb: {_js_string(verb)}")


def _assert_ref(reference: object) -> None:
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if not _nonnegative_integer(reference):
            raise TypeError("bad path id")
        return
    _assert_path(reference)


def assert_valid_wire_op(op: object) -> None:
    if not isinstance(op, (tuple, list)) or not op:
        raise TypeError("op is not a tuple")
    verb = op[0]
    if verb == "r":
        if len(op) != 2:
            raise TypeError("r arity")
    elif verb in ("s", "d"):
        full = 3 if verb == "s" else 2
        if len(op) == full:
            _assert_ref(op[1])
        elif len(op) != full - 1:
            raise TypeError(f"{verb} arity")
    elif verb in ("a", "t"):
        if len(op) not in (2, 3):
            raise TypeError(f"{verb} arity")
        if len(op) == 3:
            _assert_ref(op[1])
        value = op[-1]
        if verb == "a" and not isinstance(value, str):
            raise TypeError("a value")
        if verb == "t" and not _nonnegative_integer(value):
            raise TypeError("t count")
    elif verb == "p":
        if len(op) not in (4, 5):
            raise TypeError("p arity")
        if len(op) == 5:
            _assert_ref(op[1])
        index, remove, items = op[-3:]
        if not _nonnegative_integer(index):
            raise TypeError("p index")
        if not _nonnegative_integer(remove):
            raise TypeError("p remove")
        if not isinstance(items, list):
            raise TypeError("p items")
    elif verb == "#":
        if len(op) != 3 or not _nonnegative_integer(op[1]) or not isinstance(op[2], (list, tuple)):
            raise TypeError("# shape")
        assert_safe_path(op[2])
    else:
        raise TypeError(f"unknown op verb: {_js_string(verb)}")


def _property_key(segment: Seg) -> str:
    return segment if isinstance(segment, str) else str(int(segment))


def _read(container: _Container, segment: Seg) -> object:
    if isinstance(container, list):
        if not isinstance(segment, (int, float)) or isinstance(segment, bool):
            raise UnsafePathError(segment)
        index = int(segment)
        return container[index] if 0 <= index < len(container) else _MISSING
    return container.get(_property_key(segment), _MISSING)


def _write(container: _Container, segment: Seg, value: JsonValue) -> None:
    if isinstance(container, list):
        index = int(segment)
        if index == len(container):
            container.append(value)
        else:
            container[index] = value
    else:
        container[_property_key(segment)] = value


def _resolve(root: JsonValue | Undefined, path: Sequence[Seg]) -> _Container:
    node: object = root
    for segment in path:
        if not _is_container(node):
            raise PathError(path)
        node = _read(cast(_Container, node), segment)
        if node is _MISSING:
            raise PathError(path)
    if not _is_container(node):
        raise PathError(path)
    return cast(_Container, node)


def apply(target: JsonValue | Undefined, ops: Sequence[Sequence[object]]) -> JsonValue | Undefined:
    root = target
    for op in ops:
        assert_valid_op(op)
        verb = op[0]
        if verb == "r":
            root = cast(JsonValue, op[1])
            continue
        path = cast(Path, op[1])
        if verb == "p":
            destination = _resolve(root, path)
            if not isinstance(destination, list):
                raise PathError(path)
            index, remove = int(cast(int, op[2])), int(cast(int, op[3]))
            destination[index:index + remove] = cast(list[JsonValue], op[4])
            continue
        parent = _resolve(root, path[:-1])
        segment = path[-1]
        if isinstance(parent, list):
            if not isinstance(segment, (int, float)) or isinstance(segment, bool):
                raise UnsafePathError(segment)
            if segment > len(parent):
                raise UnsafePathError(segment)
        if verb == "s":
            _write(parent, segment, cast(JsonValue, op[2]))
        elif verb == "d":
            if isinstance(parent, list):
                index = int(segment)
                if index >= len(parent):
                    raise PathError(path)
                del parent[index]
            else:
                parent.pop(_property_key(segment), None)
        else:
            current = _read(parent, segment)
            if not isinstance(current, str):
                raise PathError(path)
            _write(parent, segment, current + cast(str, op[2]) if verb == "a" else _cut(current, int(cast(int, op[2]))))
    return root


def _copy_containers(root: JsonValue | Undefined, path: Sequence[Seg]) -> _Container:
    if not _is_container(root):
        raise PathError(path)
    source = cast(_Container, root)
    copied = source.copy()
    destination = copied
    for segment in path:
        child = _read(source, segment)
        if not _is_container(child):
            raise PathError(path)
        source_child = cast(_Container, child)
        copied_child = source_child.copy()
        _write(destination, segment, copied_child)
        source, destination = source_child, copied_child
    return copied


def apply_immutable(target: JsonValue | Undefined, ops: Sequence[Sequence[object]]) -> JsonValue | Undefined:
    root = target
    for op in ops:
        assert_valid_op(op)
        if op[0] == "r":
            root = cast(JsonValue, op[1])
            continue
        path = cast(Path, op[1])
        root = _copy_containers(root, path if op[0] == "p" else path[:-1])
        root = apply(root, [op])
    return root


class Encoder:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._ids: dict[str, int] = {}
        self._next_id = 0

    def encode(self, ops: Sequence[Sequence[object]]) -> list[WireOp]:
        previous: str | None = None
        out: list[WireOp] = []
        for op in ops:
            if op[0] == "r":
                out.append(["r", cast(JsonValue, op[1])])
                self._seen.clear()
                self._ids.clear()
                self._next_id = 0
                previous = None
                continue
            path = cast(Path, op[1])
            key = json.dumps(path, ensure_ascii=True, separators=(",", ":"))
            if key == previous:
                out.append(cast(WireOp, [op[0], *op[2:]]))
                continue
            reference: JsonValue = list(path)
            if key in self._ids:
                reference = self._ids[key]
            elif key in self._seen:
                reference = self._next_id
                self._next_id += 1
                self._ids[key] = reference
                out.append(["#", reference, list(path)])
            else:
                self._seen.add(key)
            out.append(cast(WireOp, [op[0], reference, *op[2:]]))
            previous = key
        return out


class Decoder:
    def __init__(self) -> None:
        self._paths: dict[int, Path] = {}

    def decode(self, wire: Sequence[Sequence[object]]) -> list[Op]:
        previous: Path | None = None
        out: list[Op] = []
        for op in wire:
            assert_valid_wire_op(op)
            verb = op[0]
            if verb == "#":
                self._paths[int(cast(int, op[1]))] = cast(Path, op[2])
                continue
            if verb == "r":
                out.append(cast(Op, op))
                self._paths.clear()
                previous = None
                continue
            short = (verb == "d" and len(op) == 1) or (verb not in ("d", "p") and len(op) == 2) or (verb == "p" and len(op) == 4)
            if short:
                if previous is None:
                    raise PathError(())
                path = previous
            else:
                reference = op[1]
                if isinstance(reference, (int, float)):
                    resolved = self._paths.get(int(reference))
                    if resolved is None:
                        raise PathError(int(reference))
                    path = resolved
                else:
                    path = cast(Path, reference)
                previous = path
            if verb != "p" and not path:
                raise PathError(path)
            out.append(cast(Op, (verb, path, *op[1 if short else 2:])))
        return out


def encoder() -> Encoder:
    return Encoder()


def decoder() -> Decoder:
    return Decoder()


@dataclass(frozen=True)
class TrackerOptions:
    max_overlap_scan: int = 65_536


@dataclass
class _StringSlot:
    anchor: str
    value: str


@dataclass(eq=False)
class _Slot:
    op: Op
    dead: bool = False
    order: int = 0
    index: int = 0
    string: _StringSlot | None = None


@dataclass
class _LogNode:
    slots: list[_Slot] = field(default_factory=list)
    kids: dict[Seg, _LogNode] = field(default_factory=dict)
    retired_kids: list[dict[Seg, _LogNode]] = field(default_factory=list)
    last_order: int | None = None


@dataclass
class _Fold:
    slot: _Slot
    rest: Path
    item: int | None = None


class Tracker:
    """Own a JSON tree and record mutations through its mapping/sequence views.

    Public views are weakly cached. Python's built-in containers cannot be weak
    keys, so live placements are resolved by identity in the current root when a
    mutation occurs. This preserves aliases, moves, and detached held views
    without keeping every previously read subtree alive. Placement lookup is
    linear in the tree size; operation recording and coalescing mirror Chord.
    """

    def __init__(self, root: _Container, options: TrackerOptions | Mapping[str, int] | None = None) -> None:
        if not _is_container(root):
            raise TypeError("the tracked root must be an object or array")
        self._root = root
        self._scan = options.max_overlap_scan if isinstance(options, TrackerOptions) else (options or {}).get("max_overlap_scan", 65_536)
        self._proxies: WeakValueDictionary[tuple[int, str | None], _Tracked] = WeakValueDictionary()
        self._force_base = True
        self._clear_pending()
        self._state = self._wrap(root)

    @property
    def target(self) -> _Container:
        return self._root

    @property
    def state(self) -> TrackedDict | TrackedList:
        return cast(TrackedDict | TrackedList, self._state)

    @state.setter
    def state(self, value: _Container | TrackedDict | TrackedList) -> None:
        raw = _unwrap(value)
        if not _is_container(raw):
            raise TypeError("the tracked root must be an object or array")
        self._clear_pending()
        if raw is not self._root:
            self._root = cast(_Container, raw)
            self._state = self._wrap(self._root)
        self._force_base = True

    @property
    def dirty(self) -> bool:
        return self._force_base or self._has_pending

    def rebase(self) -> None:
        self._clear_pending()
        self._force_base = True

    def discard(self) -> None:
        self._clear_pending()

    def flush(self) -> list[Op]:
        if self._force_base:
            value = _clone(self._root)
            self._force_base = False
            self._clear_pending()
            return [("r", value)]
        if not self._has_pending:
            return []
        out: list[Op] = []
        for slot in self._log:
            if slot is None or slot.dead:
                continue
            if slot.string is not None:
                _diff(slot.string.anchor, slot.string.value, cast(Path, slot.op[1]), self._scan, out)
            else:
                out.append(slot.op)
        self._clear_pending()
        return out

    def _clear_pending(self) -> None:
        self._log: list[_Slot | None] = []
        self._trie = _LogNode()
        self._next_order = 0
        self._tombstones = 0
        self._live_slots = 0
        self._node_count = 1
        self._last_added_slot: _Slot | None = None
        self._has_pending = False

    def _log_node(self, path: Path) -> _LogNode:
        node = self._trie
        for segment in path:
            child = node.kids.get(segment)
            if child is None:
                child = _LogNode()
                node.kids[segment] = child
                self._node_count += 1
            node = child
        return node

    def _find_log_node(self, path: Path) -> _LogNode | None:
        node = self._trie
        for segment in path:
            child = node.kids.get(segment)
            if child is None:
                return None
            node = child
        return node

    def _compact_log(self) -> None:
        if self._tombstones < 1024 or self._tombstones * 2 < len(self._log):
            return
        compacted: list[_Slot | None] = []
        for slot in self._log:
            if slot is not None:
                slot.index = len(compacted)
                compacted.append(slot)
        self._log = compacted
        self._tombstones = 0

    def _kill_slot(self, slot: _Slot) -> None:
        if slot.dead:
            return
        slot.dead = True
        self._live_slots -= 1
        if self._log[slot.index] is slot:
            self._log[slot.index] = None
            self._tombstones += 1

    @staticmethod
    def _live_slot(node: _LogNode) -> _Slot | None:
        while node.slots and node.slots[-1].dead:
            node.slots.pop()
        return node.slots[-1] if node.slots else None

    def _kill_here(self, node: _LogNode) -> None:
        for slot in node.slots:
            self._kill_slot(slot)
        node.slots.clear()

    def _add_slot(self, node: _LogNode, slot: _Slot) -> None:
        self._compact_log()
        slot.order = self._next_order
        self._next_order += 1
        slot.index = len(self._log)
        node.last_order = slot.order
        node.slots.append(slot)
        self._log.append(slot)
        self._live_slots += 1
        self._last_added_slot = slot

    def _collapse_pending(self) -> None:
        if self._force_base or (self._live_slots <= 4096 and self._node_count <= 4096):
            return
        value = _clone(self._root)
        self._clear_pending()
        self._has_pending = True
        self._add_slot(self._trie, _Slot(("r", value)))

    def _kill_subtree(self, node: _LogNode) -> None:
        self._kill_here(node)
        for child in node.kids.values():
            self._kill_subtree(child)
        node.kids.clear()
        for generation in node.retired_kids:
            for child in generation.values():
                self._kill_subtree(child)
        node.retired_kids.clear()

    def _fold_target(self, path: Path) -> _Fold | None:
        node = self._trie
        found: tuple[_Slot, int, int | None] | None = None
        ancestor_max = -1
        for depth, segment in enumerate(path):
            if found is not None and node.last_order is not None and node.last_order > found[0].order:
                found = None
            slot = self._live_slot(node)
            if slot is not None and slot.order >= ancestor_max and node.last_order == slot.order:
                if slot.op[0] in ("s", "r"):
                    found = (slot, depth, None)
                elif slot.op[0] == "p":
                    index, items = slot.op[2], slot.op[4]
                    if isinstance(segment, int) and index <= segment < index + len(items):
                        found = (slot, depth + 1, segment - index)
            if node.last_order is not None and node.last_order > ancestor_max:
                ancestor_max = node.last_order
            child = node.kids.get(segment)
            if child is None:
                break
            node = child
        return None if found is None else _Fold(found[0], path[found[1]:], found[2])

    @staticmethod
    def _fold_into(container: JsonValue, rest: Path, op: Op) -> bool:
        if not rest:
            return False
        target: object = container
        for segment in rest[:-1]:
            if not _is_container(target):
                return False
            target = _read(cast(_Container, target), segment)
        if not _is_container(target):
            return False
        holder = cast(_Container, target)
        key = rest[-1]
        if op[0] == "s":
            if key == "__proto__":
                return False
            _write(holder, key, _clone(op[2]))
            return True
        if op[0] == "d":
            if isinstance(holder, dict):
                holder.pop(_property_key(key), None)
            else:
                # Array deletions are never generated by the tracked proxy.
                return False
            return True
        value = _read(holder, key)
        if op[0] in ("a", "t"):
            if not isinstance(value, str):
                return False
            _write(holder, key, value + op[2] if op[0] == "a" else _cut(value, op[2]))
            return True
        if op[0] == "p" and isinstance(value, list):
            value[op[2]:op[2] + op[3]] = cast(list[JsonValue], _clone(op[4]))
            return True
        return False

    def _try_fold(self, fold: _Fold, op: Op) -> bool:
        if fold.item is not None:
            items = cast(list[JsonValue], fold.slot.op[4])
            if not fold.rest:
                if op[0] == "s":
                    items[fold.item] = _clone(op[2])
                    return True
                return False
            return self._fold_into(items[fold.item], fold.rest, op)
        payload = fold.slot.op[1] if fold.slot.op[0] == "r" else fold.slot.op[2]
        return self._fold_into(cast(JsonValue, payload), fold.rest, op)

    def _record_string(self, path: Path, previous: str, value: str) -> None:
        if self._force_base:
            return
        self._has_pending = True
        fold = self._fold_target(path)
        if fold is not None and self._try_fold(fold, ("s", path, value)):
            return
        node = self._log_node(path)
        live = self._live_slot(node)
        if live is not None and live.string is not None:
            live.string.value = value
            return
        if live is not None:
            if live.op[0] in ("s", "r"):
                live.op = ("r", value) if live.op[0] == "r" else ("s", live.op[1], value)
                return
            if live.op[0] == "d":
                self._kill_here(node)
                self._add_slot(node, _Slot(("s", path, value)))
                return
            self._kill_here(node)
        self._kill_subtree(node)
        self._add_slot(node, _Slot(("s", path, value), string=_StringSlot(previous, value)))

    def _record(self, op: Op) -> None:
        if self._force_base:
            return
        self._has_pending = True
        path = () if op[0] == "r" else cast(Path, op[1])
        existing = self._find_log_node(path)
        anchored = None if existing is None else self._live_slot(existing)
        if anchored is not None and anchored.string is not None:
            if op[0] == "a":
                anchored.string.value += op[2]
                return
            if op[0] == "t":
                anchored.string.value = _cut(anchored.string.value, op[2])
                return
            if op[0] == "s" and isinstance(op[2], str):
                anchored.string.value = op[2]
                return
            if existing is not None:
                self._kill_subtree(existing)
        elif existing is not None and op[0] in ("s", "d", "r"):
            self._kill_subtree(existing)
        if path:
            fold = self._fold_target(path)
            if fold is not None and self._try_fold(fold, op):
                return
        node = self._log_node(path)
        live = self._live_slot(node)
        if live is not None:
            previous = live.op
            if op[0] == "a" and previous[0] == "a":
                live.op = ("a", previous[1], previous[2] + op[2])
                return
            if op[0] in ("a", "t") and previous[0] in ("s", "r"):
                value = previous[1] if previous[0] == "r" else previous[2]
                if isinstance(value, str):
                    changed = value + op[2] if op[0] == "a" else _cut(value, op[2])
                    live.op = ("r", changed) if previous[0] == "r" else ("s", previous[1], changed)
                    return
            if op[0] == "p" and previous[0] == "p":
                old_items = previous[4]
                if previous[3] == 0 and op[3] == 0 and previous[2] + len(old_items) == op[2] and self._last_added_slot is live:
                    old_items.extend(op[4])
                    return
                if previous[3] == 0 and self._last_added_slot is live and op[2] >= previous[2] and op[2] + op[3] <= previous[2] + len(old_items):
                    offset = op[2] - previous[2]
                    old_items[offset:offset + op[3]] = op[4]
                    if not old_items:
                        self._kill_slot(live)
                    return
                if op[3] > 0 and not op[4] and old_items and self._last_added_slot is live:
                    offset = op[2] - previous[2]
                    if offset >= 0 and offset + op[3] == len(old_items):
                        del old_items[offset:]
                        if not old_items and previous[3] == 0:
                            self._kill_slot(live)
                        return
            if op[0] in ("s", "d", "r"):
                self._kill_here(node)
        if op[0] in ("s", "r", "d"):
            self._kill_subtree(node)
        elif op[0] == "p" and node.kids:
            node.retired_kids.append(node.kids)
            node.kids = {}
        self._add_slot(node, _Slot(op))

    def _diff_into(self, before: JsonValue, after: JsonValue, path: Path) -> None:
        if self._force_base:
            return
        self._has_pending = True
        if _same(before, after):
            return
        if isinstance(before, str) and isinstance(after, str):
            self._record_string(path, before, after)
        elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
            for index, value in enumerate(after):
                self._diff_into(before[index], value, (*path, index))
        elif isinstance(before, dict) and isinstance(after, dict):
            if any(key in RESERVED_SEGMENTS for key in (*before, *after)):
                self._record(_set_op(path, after))
                return
            for key, value in after.items():
                if key in before:
                    self._diff_into(before[key], value, (*path, key))
                else:
                    self._record(("s", (*path, key), _clone(value)))
            for key in before:
                if key not in after:
                    self._record(("d", (*path, key)))
        else:
            out: list[Op] = []
            _diff(before, after, path, self._scan, out)
            for op in out:
                self._record(op)

    def _paths(self, target: _Container) -> list[Path]:
        found: list[Path] = []
        stack: list[tuple[JsonValue, Path, frozenset[int]]] = [(self._root, (), frozenset())]
        while stack:
            value, path, ancestors = stack.pop()
            if value is target:
                found.append(path)
            if not _is_container(value) or id(value) in ancestors:
                continue
            ancestors = ancestors | {id(value)}
            children = tuple(enumerate(value)) if isinstance(value, list) else tuple(cast(dict[str, JsonValue], value).items())
            for segment, child in reversed(children):
                if segment not in RESERVED_SEGMENTS:
                    stack.append((child, (*path, segment), ancestors))
        return found

    def _wrap(self, target: _Container, blocked: str | None = None) -> _Tracked:
        key = (id(target), blocked)
        proxy = self._proxies.get(key)
        if proxy is not None and proxy._target is target:
            return proxy
        proxy = TrackedList(self, target, blocked) if isinstance(target, list) else TrackedDict(self, target, blocked)
        self._proxies[key] = proxy
        return proxy


def _unwrap(value: object) -> object:
    return value._target if isinstance(value, _Tracked) else value


class _Tracked:
    def __init__(self, tracker: Tracker, target: _Container, blocked: str | None = None) -> None:
        object.__setattr__(self, "_tracker", tracker)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_blocked", blocked)

    def _guard(self) -> None:
        if self._blocked is not None:
            raise UnsafePathError(self._blocked)

    def _get(self, key: Seg) -> object:
        value = _read(self._target, key)
        if value is _MISSING:
            return UNDEFINED
        if _is_container(value):
            blocked = self._blocked or (key if isinstance(key, str) and key in RESERVED_SEGMENTS else None)
            return self._tracker._wrap(cast(_Container, value), blocked)
        return value

    def _set(self, key: Seg, value: object) -> None:
        self._guard()
        raw = _unwrap(value)
        paths = self._tracker._paths(self._target)
        if not paths:
            if raw is UNDEFINED and isinstance(self._target, dict):
                self._target.pop(_property_key(key), None)
            else:
                _write(self._target, key, cast(JsonValue, raw))
            return
        assert_safe_path((key,))
        if isinstance(self._target, list):
            if not isinstance(key, int) or isinstance(key, bool) or key > len(self._target):
                raise UnsafePathError(key)
        if raw is UNDEFINED:
            if isinstance(self._target, list):
                raise TypeError("undefined would create a sparse array; use splice instead")
            if _property_key(key) in self._target:
                for path in paths:
                    self._tracker._record(("d", (*path, key)))
                self._target.pop(_property_key(key), None)
            self._tracker._collapse_pending()
            return
        previous = _read(self._target, key)
        if _same(previous, raw):
            return
        for path in paths:
            at = (*path, key)
            if isinstance(self._target, list) and key == len(self._target):
                self._tracker._record(("p", path, len(self._target), 0, [_clone(cast(JsonValue, raw))]))
            elif _is_container(previous) and _is_container(raw):
                self._tracker._diff_into(cast(JsonValue, previous), cast(JsonValue, raw), at)
            elif isinstance(previous, str) and isinstance(raw, str):
                self._tracker._record_string(at, previous, raw)
            else:
                self._tracker._record(("s", at, _clone(cast(JsonValue, raw))))
        _write(self._target, key, cast(JsonValue, raw))
        self._tracker._collapse_pending()

    def _delete(self, key: Seg) -> None:
        self._guard()
        paths = self._tracker._paths(self._target)
        if isinstance(self._target, list):
            raise TypeError("delete would create a sparse array; use splice instead")
        if paths:
            assert_safe_path((key,))
            if _property_key(key) in self._target:
                for path in paths:
                    self._tracker._record(("d", (*path, key)))
        self._target.pop(_property_key(key), None)
        self._tracker._collapse_pending()

    def __repr__(self) -> str:
        return repr(self._target)


class TrackedDict(_Tracked, MutableMapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key not in self._target:
            raise KeyError(key)
        return self._get(key)

    def __setitem__(self, key: str, value: object) -> None:
        self._set(key, value)

    def __delitem__(self, key: str) -> None:
        self._delete(key)

    def __iter__(self) -> Iterator[str]:
        return iter(cast(dict[str, JsonValue], self._target))

    def __len__(self) -> int:
        return len(self._target)

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._get(name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._set(name, value)

    def __delattr__(self, name: str) -> None:
        self._delete(name)


def _integer(value: object) -> int | float:
    if value is UNDEFINED:
        return 0
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        number = value
    else:
        try:
            rendered = str(value).strip()
            number = float(rendered) if rendered else 0
        except (TypeError, ValueError):
            return 0
    if math.isnan(number) or number == 0:
        return 0
    return math.trunc(number) if math.isfinite(number) else number


def _relative_index(value: object, length: int) -> int:
    integer = _integer(value)
    return int(max(0, length + integer) if integer < 0 else min(integer, length))


def _js_string(value: object) -> str:
    value = _unwrap(value)
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, dict):
        return "[object Object]"
    if isinstance(value, list):
        return ",".join("" if item is None or item is UNDEFINED else _js_string(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class TrackedList(_Tracked, MutableSequence[object]):
    @property
    def _array(self) -> list[JsonValue]:
        return cast(list[JsonValue], self._target)

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> list[object]: ...

    def __getitem__(self, index: int | slice) -> object:
        if isinstance(index, slice):
            return [self._get(at) for at in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("list index out of range")
        return self._get(index)

    def __setitem__(self, index: int | slice, value: object) -> None:
        if isinstance(index, slice):
            items = list(cast(Iterable[object], value))
            start, stop, step = index.indices(len(self))
            if step != 1:
                indices = list(range(start, stop, step))
                if len(indices) != len(items):
                    raise ValueError(f"attempt to assign sequence of size {len(items)} to extended slice of size {len(indices)}")
                for at, item in zip(indices, items):
                    self._set(at, item)
            else:
                self.splice(start, max(0, stop - start), *items)
            return
        self._set(index, value)

    def __delitem__(self, index: int | slice) -> None:
        self._guard()
        raise TypeError("delete would create a sparse array; use splice instead")

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> Iterator[object]:
        index = 0
        while index < len(self):
            yield self._get(index)
            index += 1

    def __eq__(self, other: object) -> bool:
        return self._array == _unwrap(other)

    @property
    def length(self) -> int:
        return len(self)

    @length.setter
    def length(self, value: object) -> None:
        self._guard()
        if isinstance(value, str):
            try:
                size = float(value.strip()) if value.strip() else 0.0
            except ValueError as error:
                raise ValueError("Invalid array length") from error
        elif value is None:
            size = 0
        elif isinstance(value, (int, float)):
            size = value
        else:
            raise ValueError("Invalid array length")
        if not math.isfinite(size) or int(size) != size or not 0 <= size <= 4_294_967_295:
            raise ValueError("Invalid array length")
        next_size = int(size)
        previous_size = len(self)
        if next_size < previous_size:
            self._splice(next_size, previous_size - next_size, [], replace_all=next_size == 0)
        elif next_size > previous_size:
            self._splice(previous_size, 0, [None] * (next_size - previous_size))

    def _splice(self, index: int, remove: int, items: list[object], *, replace_all: bool = False) -> list[JsonValue]:
        self._guard()
        adopted = cast(list[JsonValue], [_unwrap(item) for item in items])
        paths = self._tracker._paths(self._array)
        if paths and (remove > 0 or adopted):
            for path in paths:
                if replace_all:
                    self._tracker._record(_set_op(path, adopted))
                else:
                    self._tracker._record(("p", path, index, remove, cast(list[JsonValue], _clone(adopted))))
        removed = self._array[index:index + remove]
        self._array[index:index + remove] = adopted
        self._tracker._collapse_pending()
        return removed

    def push(self, *items: object) -> int:
        self._splice(len(self), 0, list(items))
        return len(self)

    def append(self, value: object) -> None:
        self.push(value)

    def extend(self, values: Iterable[object]) -> None:
        self.push(*values)

    def insert(self, index: int, value: object) -> None:
        self.splice(index, 0, value)

    def remove(self, value: object) -> None:
        raw = _unwrap(value)
        for index, item in enumerate(self._array):
            if _same(item, raw):
                self.splice(index, 1)
                return
        raise ValueError("list.remove(x): x not in list")

    def unshift(self, *items: object) -> int:
        self._splice(0, 0, list(items))
        return len(self)

    def pop(self, index: int = -1) -> object:
        self._guard()
        if not self:
            return UNDEFINED
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("pop index out of range")
        return self._splice(index, 1, [])[0]

    def shift(self) -> object:
        self._guard()
        return self._splice(0, 1, [])[0] if self else UNDEFINED

    def splice(self, *args: object) -> list[JsonValue]:
        length = len(self)
        index = _relative_index(args[0], length) if args else 0
        if not args:
            remove = 0
        elif len(args) == 1:
            remove = length - index
        else:
            remove = int(max(0, min(_integer(args[1]), length - index)))
        items = list(args[2:])
        return self._splice(index, remove, items, replace_all=index == 0 and remove == length)

    def clear(self) -> None:
        self.length = 0

    def _snapshot(self, paths: list[Path]) -> None:
        for path in paths:
            self._tracker._record(_set_op(path, self._array))
        self._tracker._collapse_pending()

    def reverse(self) -> TrackedList:
        self._guard()
        paths = self._tracker._paths(self._array)
        self._array.reverse()
        self._snapshot(paths)
        return self

    def sort(self, compare: Callable[[JsonValue, JsonValue], float] | None = None, *, key: Callable[[JsonValue], object] | None = None, reverse: bool = False) -> TrackedList:
        self._guard()
        paths = self._tracker._paths(self._array)
        if compare is not None:
            def comparator(left: JsonValue, right: JsonValue) -> int:
                compared = compare(left, right)
                return 0 if math.isnan(compared) or compared == 0 else (-1 if compared < 0 else 1)

            self._array.sort(key=cmp_to_key(comparator), reverse=reverse)
        elif key is not None:
            self._array.sort(key=key, reverse=reverse)
        else:
            self._array.sort(key=lambda item: _units(_js_string(item)), reverse=reverse)
        self._snapshot(paths)
        return self

    def fill(self, value: object, start: object = 0, end: object = UNDEFINED) -> TrackedList:
        self._guard()
        paths = self._tracker._paths(self._array)
        first = _relative_index(start, len(self))
        last = len(self) if end is UNDEFINED else _relative_index(end, len(self))
        raw = cast(JsonValue, _unwrap(value))
        for index in range(first, last):
            self._array[index] = raw
        self._snapshot(paths)
        return self

    def copy_within(self, target: object, start: object = 0, end: object = UNDEFINED) -> TrackedList:
        self._guard()
        paths = self._tracker._paths(self._array)
        destination = _relative_index(target, len(self))
        first = _relative_index(start, len(self))
        last = len(self) if end is UNDEFINED else _relative_index(end, len(self))
        count = min(max(0, last - first), len(self) - destination)
        self._array[destination:destination + count] = self._array[first:first + count]
        self._snapshot(paths)
        return self

    def slice(self, start: object = 0, end: object = UNDEFINED) -> list[object]:
        first = _relative_index(start, len(self))
        last = len(self) if end is UNDEFINED else _relative_index(end, len(self))
        return [self._get(index) for index in range(first, last)]


def track(root: _Container, options: TrackerOptions | Mapping[str, int] | None = None) -> Tracker:
    return Tracker(root, options)
