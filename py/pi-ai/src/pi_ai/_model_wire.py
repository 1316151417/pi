"""Mutable views over the plain model objects used by the TypeScript API.

Radius deliberately copies only the outer model object. Keeping nested mappings
behind views preserves that behavior without changing Python attribute names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableSequence
from typing import ClassVar, Self, cast, overload
from weakref import WeakValueDictionary


class ModelFields:
    _wire_names: ClassVar[dict[str, str]] = {}
    _optional_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _from_wire(cls, data: Mapping[str, object]) -> Self:
        value = cls.__new__(cls)
        object.__setattr__(value, "_wire", data if isinstance(data, dict) else dict(data))
        return value

    def __getattribute__(self, name: str) -> object:
        names = object.__getattribute__(self, "_wire_names")
        if name in names:
            state = object.__getattribute__(self, "__dict__")
            if "_wire" in state:
                raw = state["_wire"].get(names[name])
                return self._view_wire_field(name, raw)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> object:
        wire = object.__getattribute__(self, "__dict__").get("_wire", {})
        if name in wire:
            return wire[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: object) -> None:
        state = object.__getattribute__(self, "__dict__")
        names = type(self)._wire_names
        if "_wire" in state and not name.startswith("_"):
            state["_wire"][names.get(name, name)] = _wire_value(value)
        else:
            object.__setattr__(self, name, value)

    def _view_wire_field(self, name: str, value: object) -> object:
        return value

    def __copy__(self) -> Self:
        value = type(self).__new__(type(self))
        state = object.__getattribute__(self, "__dict__").copy()
        if "_wire" in state:
            state["_wire"] = dict(state["_wire"])
        object.__setattr__(value, "__dict__", state)
        return value

    def to_json(self) -> dict[str, object]:
        state = object.__getattribute__(self, "__dict__")
        if "_wire" in state:
            return cast(dict[str, object], state["_wire"])
        return {
            wire_name: _wire_value(value)
            for name, wire_name in self._wire_names.items()
            if (value := getattr(self, name)) is not None or name not in self._optional_fields
        }


def _wire_value(value: object) -> object:
    if isinstance(value, ModelFields):
        return value.to_json()
    if isinstance(value, ModelList):
        return value._items
    if isinstance(value, list) and any(isinstance(item, ModelFields) for item in value):
        return [_wire_value(item) for item in value]
    return value


class ModelList[T: ModelFields](MutableSequence[T]):
    """A live list view: appends, deletions, and field edits update the raw list."""

    def __init__(self, items: list[object], wrap: Callable[[Mapping[str, object]], T]) -> None:
        self._items = items
        self._wrap = wrap
        self._views: WeakValueDictionary[int, T] = WeakValueDictionary()

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        raw = self._items[index]
        if isinstance(raw, ModelFields):
            return cast(T, raw)
        cached = self._views.get(id(raw))
        if cached is not None and cached.to_json() is raw:
            return cached
        value = self._wrap(cast(Mapping[str, object], raw))
        self._views[id(raw)] = value
        return value

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        if isinstance(index, slice):
            self._items[index] = [_wire_value(item) for item in cast(Iterable[T], value)]
        else:
            self._items[index] = _wire_value(value)
        self._views.clear()

    def __delitem__(self, index: int | slice) -> None:
        del self._items[index]
        self._views.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        index = 0
        while index < len(self):
            yield self[index]
            index += 1

    def insert(self, index: int, value: T) -> None:
        self._items.insert(index, _wire_value(value))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, ModelList)):
            return list(self) == list(other)
        return NotImplemented
