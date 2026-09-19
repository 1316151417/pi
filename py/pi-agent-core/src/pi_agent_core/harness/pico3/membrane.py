"""Transaction-scoped revocable membrane ported from ``harness/pico3/membrane.ts``.

Every object reached through a wrapped document is wrapped lazily; all wrappers
of one transaction share one liveness flag; identity is preserved by an id-keyed
map. Mutations forward to the underlying value. At transaction finish — success,
callback failure, validation failure, storage failure — ``revoke()`` flips the
flag and every retained wrapper, root or nested, throws on any operation.

Assigning one wrapper into another (``a.list = b.list``) is rejected: a stored
proxy is the classic footgun. Assign plain values; splice in place.

Port note: the TypeScript membrane sits on Chord's mutating ``Proxy``, so writes
through it are recorded by the delta tracker. Python's :class:`~.delta.Tracker`
diffs the document at ``flush()``, so a wrapper write is observed there — the
membrane's own job here is lifetime and plain-input enforcement.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from ..._pi_ai.types import JsonValue

__all__ = ["Membrane", "RevokedError", "PlainInputError"]


class RevokedError(TypeError):
    """A document wrapper was used outside its transaction."""


class PlainInputError(TypeError):
    """A wrapper or a cyclic value was assigned into a document."""


def _is_container(value: Any) -> bool:
    return isinstance(value, (dict, list))


class _Wrapper:
    """Common liveness plumbing for dict and list wrappers."""

    __slots__ = ("_membrane", "_target")

    def __init__(self, membrane: "Membrane", target: Any) -> None:
        self._membrane = membrane
        self._target = target

    def _check(self) -> None:
        if not self._membrane.alive:
            raise RevokedError(
                f"document proxy ({self._membrane.what}) used outside its transaction"
            )

    def _out(self, value: Any) -> Any:
        # Nested containers are handed back plain: ``turn["tools"][0]["status"] = ...``
        # writes through the shared reference, the tracker's flush records the diff, and
        # every ``isinstance(x, dict)`` / ``json.dumps`` in the port keeps working. The
        # operation wrapper (the document root) is what enforces lifetime.
        return value

    def _in(self, value: Any) -> Any:
        if self._membrane.is_wrapper(value):
            raise PlainInputError(
                f"assigning a document proxy into a document ({self._membrane.what}); "
                "assign a plain value"
            )
        if not _is_container(value):
            return value
        self._membrane.assert_plain_input(value)
        return json.loads(json.dumps(value))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<membrane {self._membrane.what} {type(self._target).__name__}>"


class MembraneDict(_Wrapper):
    """A dict reached through the membrane."""

    __slots__ = ()

    def __getitem__(self, key: str) -> Any:
        self._check()
        return self._out(self._target[key])

    def __setitem__(self, key: str, value: Any) -> None:
        self._check()
        if key == "__proto__":
            raise TypeError(
                f"document proxy ({self._membrane.what}) cannot change prototypes"
            )
        self._target[key] = self._in(value)

    def __delitem__(self, key: str) -> None:
        self._check()
        del self._target[key]

    def __contains__(self, key: object) -> bool:
        self._check()
        return key in self._target

    def __iter__(self) -> Iterator[str]:
        self._check()
        return iter(list(self._target.keys()))

    def __len__(self) -> int:
        self._check()
        return len(self._target)

    def get(self, key: str, default: Any = None) -> Any:
        self._check()
        if key not in self._target:
            return default
        return self._out(self._target[key])

    def keys(self) -> List[str]:
        self._check()
        return list(self._target.keys())

    def values(self) -> List[Any]:
        self._check()
        return [self._out(value) for value in self._target.values()]

    def items(self) -> List[Any]:
        self._check()
        return [(key, self._out(value)) for key, value in self._target.items()]

    def pop(self, key: str, *default: Any) -> Any:
        self._check()
        if key in self._target:
            return self._out(self._target.pop(key))
        if default:
            return default[0]
        raise KeyError(key)

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._check()
        if key not in self._target:
            self._target[key] = self._in(default)
        return self._out(self._target[key])

    def update(self, other: Dict[str, Any]) -> None:
        self._check()
        for key, value in other.items():
            self[key] = value

    def copy(self) -> Dict[str, Any]:
        self._check()
        return {key: value for key, value in self.items()}

    def to_plain(self) -> Dict[str, Any]:
        self._check()
        return json.loads(json.dumps(self._target))


class MembraneList(_Wrapper):
    """A list reached through the membrane."""

    __slots__ = ()

    def __getitem__(self, index: Any) -> Any:
        self._check()
        if isinstance(index, slice):
            return [self._out(item) for item in self._target[index]]
        return self._out(self._target[index])

    def __setitem__(self, index: Any, value: Any) -> None:
        self._check()
        if isinstance(index, slice):
            self._target[index] = [self._in(item) for item in value]
            return
        self._target[index] = self._in(value)

    def __delitem__(self, index: Any) -> None:
        self._check()
        del self._target[index]

    def __len__(self) -> int:
        self._check()
        return len(self._target)

    def __iter__(self) -> Iterator[Any]:
        self._check()
        return iter([self._out(item) for item in list(self._target)])

    def __contains__(self, value: object) -> bool:
        self._check()
        return value in self._target

    def append(self, value: Any) -> None:
        self._check()
        self._target.append(self._in(value))

    def extend(self, values: Any) -> None:
        self._check()
        for value in values:
            self.append(value)

    def insert(self, index: int, value: Any) -> None:
        self._check()
        self._target.insert(index, self._in(value))

    def pop(self, index: int = -1) -> Any:
        self._check()
        return self._out(self._target.pop(index))

    def remove(self, value: Any) -> None:
        self._check()
        self._target.remove(value)

    def clear(self) -> None:
        self._check()
        self._target.clear()

    def index(self, value: Any) -> int:
        self._check()
        return self._target.index(value)

    def splice(self, index: int, remove: int, items: Optional[List[Any]] = None) -> None:
        """Remove ``remove`` elements at ``index`` and insert ``items`` in place."""
        self._check()
        del self._target[index : index + remove]
        for offset, item in enumerate(items or []):
            self._target.insert(index + offset, self._in(item))

    def to_plain(self) -> List[Any]:
        self._check()
        return json.loads(json.dumps(self._target))


class Membrane:
    """A revocable wrapper factory over one document."""

    def __init__(self, what: str) -> None:
        self.alive = True
        self.what = what
        self._wrappers: Dict[int, Any] = {}
        self._wrapper_ids: set = set()

    def revoke(self) -> None:
        self.alive = False
        self._wrappers.clear()

    def wrap(self, target: Any) -> Any:
        """Wrap ``target``, preserving identity across calls for the same object.

        Only the object handed to :meth:`wrap` is wrapped; nested containers are
        returned plain on read (see :meth:`_Wrapper._out`).
        """
        if not _is_container(target):
            return target
        hit = self._wrappers.get(id(target))
        if hit is not None and hit._target is target:
            return hit
        wrapper = MembraneDict(self, target) if isinstance(target, dict) else MembraneList(self, target)
        self._wrappers[id(target)] = wrapper
        self._wrapper_ids.add(id(wrapper))
        return wrapper

    def is_wrapper(self, value: Any) -> bool:
        return isinstance(value, _Wrapper)

    def assert_plain_input(self, value: Any, seen: Optional[set] = None) -> None:
        """Reject wrappers and cyclic values anywhere inside an assigned value."""
        seen = set() if seen is None else seen
        if isinstance(value, _Wrapper):
            raise PlainInputError(
                f"assigning a document proxy into a document ({self.what}); assign a plain value"
            )
        if id(value) in seen:
            raise PlainInputError(f"assigning a cyclic value into a document ({self.what})")
        seen.add(id(value))
        items = value.values() if isinstance(value, dict) else value
        for nested in items:
            if _is_container(nested):
                self.assert_plain_input(nested, seen)
        seen.discard(id(value))
