"""Operation-log change tracking ported from ``chord/src/delta/index.ts``.

The Python port ships this file inside ``pico3`` because ``pi_agent_core._chord``
has no ``delta`` module yet. It implements exactly the slice the pico3 kernel
uses: the ``Op`` tuple vocabulary, ``is_base``, ``apply_immutable`` and a
``Tracker`` that produces ops for a mutated JSON document.

Port note (behavioural difference, deliberate): TypeScript tracks mutations
through a ``Proxy``, so ``tracker.state.foo.push(...)`` is observed. Python has
no transparent proxy for plain ``dict``/``list``, so ``Tracker`` computes its
ops by structural diff at ``flush()`` time instead of recording them on the
faulting mutator. The emitted batches are equivalent under ``apply_immutable``
and ``is_base`` — only op *granularity* differs (adjacent string edits are
``s`` sets rather than ``a`` appends), which no pico3 consumer inspects.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pi_ai.types import JsonValue

__all__ = [
    "Seg",
    "Path",
    "Op",
    "op_replace",
    "op_set",
    "op_delete",
    "op_append",
    "op_truncate",
    "op_splice",
    "is_replace",
    "is_base",
    "apply",
    "apply_immutable",
    "Tracker",
    "clone_json",
    "PathError",
]

Seg = Union[str, int]
Path = Tuple[Seg, ...]

#: Tuples are the form: ``("r", value)``, ``("s", path, value)``, ``("d", path)``,
#: ``("a", path, text)``, ``("t", path, prefixLength)``, ``("p", path, index, remove, items)``.
Op = Tuple[Any, ...]


def op_replace(value: JsonValue) -> Op:
    return ("r", value)


def op_set(path: Path, value: JsonValue) -> Op:
    return ("s", path, value)


def op_delete(path: Path) -> Op:
    return ("d", path)


def op_append(path: Path, text: str) -> Op:
    return ("a", path, text)


def op_truncate(path: Path, length: int) -> Op:
    return ("t", path, length)


def op_splice(path: Path, index: int, remove: int, items: List[JsonValue]) -> Op:
    return ("p", path, index, remove, items)


def is_replace(op: Op) -> bool:
    return op[0] == "r"


def is_base(ops: Sequence[Op]) -> bool:
    """A batch begins with a replacement."""
    return len(ops) > 0 and ops[0][0] == "r"


def clone_json(value: JsonValue) -> JsonValue:
    """Deep copy of a JSON value. Scalars are returned unchanged."""
    if isinstance(value, dict):
        return {key: clone_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_json(item) for item in value]
    return value


class PathError(Exception):
    """An op could not be applied to the value it targets."""

    def __init__(self, path: Any) -> None:
        try:
            rendered = json.dumps(path)
        except (TypeError, ValueError):
            rendered = str(path)
        super().__init__(f"unresolvable path: {rendered}")
        self.path = path
        self.name = "PathError"


def _resolve(root: JsonValue, path: Path) -> JsonValue:
    node = root
    for segment in path:
        if isinstance(node, list):
            if not isinstance(segment, int) or isinstance(segment, bool):
                raise PathError(path)
            if segment < 0 or segment >= len(node):
                raise PathError(path)
            node = node[segment]
            continue
        if isinstance(node, dict):
            if not isinstance(segment, str) or segment not in node:
                raise PathError(path)
            node = node[segment]
            continue
        raise PathError(path)
    return node


def apply(target: JsonValue, ops: Sequence[Op]) -> JsonValue:
    """Apply ops to a mutable value in place. Returns the value because ``r`` replaces it."""
    root = target
    for op in ops:
        tag = op[0]
        if tag == "r":
            root = op[1]
            continue
        path: Path = op[1]
        if tag == "p":
            node = root if len(path) == 0 else _resolve(root, path)
            if not isinstance(node, list):
                raise PathError(path)
            index, remove, items = op[2], op[3], op[4]
            del node[index : index + remove]
            for offset, item in enumerate(items):
                node.insert(index + offset, item)
            continue
        parent = _resolve(root, path[:-1])
        key = path[-1]
        if isinstance(parent, list):
            if not isinstance(key, int) or isinstance(key, bool):
                raise PathError(path)
            if key < 0 or key >= len(parent):
                raise PathError(path)
        elif not isinstance(parent, dict):
            raise PathError(path)
        if tag == "s":
            parent[key] = op[2]
        elif tag == "d":
            if isinstance(parent, list):
                del parent[key]
            else:
                parent.pop(key, None)
        elif tag == "a":
            current = parent.get(key)
            if not isinstance(current, str):
                raise PathError(path)
            parent[key] = current + op[2]
        elif tag == "t":
            current = parent.get(key)
            if not isinstance(current, str):
                raise PathError(path)
            parent[key] = current[op[2] :]
        else:
            raise PathError(op)
    return root


def apply_immutable(target: JsonValue, ops: Sequence[Op]) -> JsonValue:
    """Apply ops without mutating the previous immutable value."""
    root: JsonValue = target
    for op in ops:
        if op[0] == "r":
            root = clone_json(op[1])
            continue
        working = clone_json(root)
        root = apply(working, [op])
    return root


def _diff(previous: JsonValue, current: JsonValue, path: Path, ops: List[Op]) -> None:
    if isinstance(current, dict) and isinstance(previous, dict):
        for key in previous:
            if key not in current:
                ops.append(op_delete(path + (key,)))
        for key, value in current.items():
            if key not in previous:
                ops.append(op_set(path + (key,), clone_json(value)))
            else:
                _diff(previous[key], value, path + (key,), ops)
        return
    if isinstance(current, list) and isinstance(previous, list):
        _diff_list(previous, current, path, ops)
        return
    if previous != current or type(previous) is not type(current):
        if len(path) == 0:
            ops.append(op_replace(clone_json(current)))
        else:
            ops.append(op_set(path, clone_json(current)))


def _diff_list(previous: List[JsonValue], current: List[JsonValue], path: Path, ops: List[Op]) -> None:
    prefix = 0
    limit = min(len(previous), len(current))
    while prefix < limit and previous[prefix] == current[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and previous[len(previous) - suffix - 1] == current[len(current) - suffix - 1]
    ):
        suffix += 1
    remove = len(previous) - prefix - suffix
    items = [clone_json(item) for item in current[prefix : len(current) - suffix]]
    if remove == 0 and len(items) == 0:
        return
    if len(previous) == len(current) and remove == len(items) and path[-1:] != ():
        # Equal-length rewrite: recurse so nested containers keep their identity.
        for index in range(prefix, len(current) - suffix):
            _diff(previous[index], current[index], path + (index,), ops)
        return
    ops.append(op_splice(path, prefix, remove, items))


@dataclass
class Tracker:
    """A tracked JSON document: mutate ``state``, then ``flush()`` the ops.

    ``state`` and ``target`` are the same object. Mutating it directly is the
    supported mutation path (see the module docstring for why the TS proxy is
    modelled as a structural diff).
    """

    target: JsonValue = None
    options: Dict[str, Any] = field(default_factory=dict)
    _snapshot: JsonValue = None
    _rebase: bool = False

    def __post_init__(self) -> None:
        self._snapshot = clone_json(self.target)

    @property
    def state(self) -> JsonValue:
        return self.target

    def flush(self) -> List[Op]:
        """Emit the ops for every mutation since the last flush."""
        if self._rebase:
            self._rebase = False
            self._snapshot = clone_json(self.target)
            return [op_replace(clone_json(self.target))]
        ops: List[Op] = []
        _diff(self._snapshot, self.target, (), ops)
        self._snapshot = clone_json(self.target)
        return ops

    def rebase(self) -> None:
        """Make the next flush a complete base batch without changing the value."""
        self._rebase = True

    def discard(self) -> None:
        """Accept pending mutations locally without emitting them."""
        self._rebase = False
        self._snapshot = clone_json(self.target)

    @property
    def dirty(self) -> bool:
        if self._rebase:
            return True
        ops: List[Op] = []
        _diff(self._snapshot, self.target, (), ops)
        return len(ops) > 0


def track(root: JsonValue, options: Optional[Dict[str, Any]] = None) -> Tracker:
    """Track ``root``; the returned tracker owns the mutations made through ``state``."""
    return Tracker(target=root, options=options or {})


_ = Dict
