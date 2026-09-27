"""Private weak identity registry for publishable replicated state sources."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import Protocol

from ..context import Context
from ..delta import Op


class ReplicatedStateInternals(Protocol):
    @property
    def sequence(self) -> int: ...

    @property
    def value(self) -> object: ...

    def publish(self, context: Context) -> None: ...

    def subscribe(
        self, listener: Callable[[Sequence[Op], int, Context], None],
    ) -> Callable[[], None]: ...


_sources: dict[int, tuple[weakref.ReferenceType[object], ReplicatedStateInternals]] = {}


def register_replicated_state_internals(value: object, internals: ReplicatedStateInternals) -> None:
    identity = id(value)

    def collected(reference: weakref.ReferenceType[object]) -> None:
        current = _sources.get(identity)
        if current is not None and current[0] is reference:
            del _sources[identity]

    _sources[identity] = (weakref.ref(value, collected), internals)


def get_replicated_state_internals(value: object) -> ReplicatedStateInternals | None:
    entry = _sources.get(id(value))
    return entry[1] if entry is not None and entry[0]() is value else None
