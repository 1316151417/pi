"""Publishing state and cold read-only replicas from services/state.ts."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Sequence
from typing import cast

from ..context import BACKGROUND_CONTEXT, Context
from ..delta import UNDEFINED, Op, Undefined, apply_immutable, is_base, track
from ..types import JsonValue, ReplicatedStateDelivery
from ._listeners import ListenerSet
from .state_internals import register_replicated_state_internals


class MutableReplicatedStateImpl[T]:
    def __init__(self, initial: T) -> None:
        self._listeners: ListenerSet[Callable[[T, Context, ReplicatedStateDelivery], None]] = ListenerSet()
        self._source_listeners: ListenerSet[Callable[[Sequence[Op], int, Context], None]] = ListenerSet()
        self._tracker = track(initial)
        self._published_value = cast(T, apply_immutable(UNDEFINED, self._tracker.flush()))
        self._sequence = 0
        register_replicated_state_internals(self, _StateInternals(self))

    @property
    def value(self) -> T:
        return self._published_value

    @property
    def state(self) -> T:
        return cast(T, self._tracker.state)

    def publish(self, context: Context) -> None:
        ops = self._tracker.flush()
        if not ops:
            return
        self._sequence += 1
        self._published_value = cast(T, apply_immutable(cast(JsonValue, self._published_value), ops))
        for listener in tuple(self._source_listeners):
            listener(ops, self._sequence, context)
        delivery: ReplicatedStateDelivery = {"kind": "update", "sequence": self._sequence}
        for listener in tuple(self._listeners):
            listener(self._published_value, context, delivery)

    def subscribe(self, listener: Callable[[T, Context, ReplicatedStateDelivery], None]) -> Callable[[], None]:
        context = service_delivery_context()
        self.publish(context)
        self._listeners.add(listener)
        listener(self._published_value, context, {"kind": "hydrate", "sequence": self._sequence})
        return lambda: self._listeners.discard(listener)


class _StateInternals:
    def __init__(self, source: MutableReplicatedStateImpl[object]) -> None:
        # The registry must not keep its weak key alive through the stored value.
        self._source = weakref.ref(source)

    def _get_source(self) -> MutableReplicatedStateImpl[object]:
        source = self._source()
        if source is None:
            raise ReferenceError("Replicated state source is no longer alive")
        return source

    @property
    def sequence(self) -> int:
        return self._get_source()._sequence

    @property
    def value(self) -> object:
        return self._get_source()._published_value

    def publish(self, context: Context) -> None:
        self._get_source().publish(context)

    def subscribe(self, listener: Callable[[Sequence[Op], int, Context], None]) -> Callable[[], None]:
        source = self._get_source()
        source._source_listeners.add(listener)
        return lambda: source._source_listeners.discard(listener)


class ReplicatedStateReplica[T]:
    def __init__(self, report_error: Callable[[Exception], None]) -> None:
        self._listeners: ListenerSet[Callable[[T, Context, ReplicatedStateDelivery], None]] = ListenerSet()
        self._report_error = report_error
        self._value: T | Undefined = UNDEFINED
        self._sequence: int | None = None

    @property
    def value(self) -> T | Undefined:
        return self._value

    def subscribe(self, listener: Callable[[T, Context, ReplicatedStateDelivery], None]) -> Callable[[], None]:
        self._listeners.add(listener)
        if self._value is not UNDEFINED:
            self._deliver(listener, cast(T, self._value), service_delivery_context(),
                          {"kind": "hydrate", "sequence": cast(int, self._sequence)})
        return lambda: self._listeners.discard(listener)

    def hydrate(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        if not is_base(ops):
            raise RuntimeError("Replicated state snapshot is not a base operation batch")
        value = apply_immutable(UNDEFINED, ops)
        self._sequence = sequence
        self._value = cast(T, value)
        self._deliver_all(context, {"kind": "hydrate", "sequence": sequence})

    def update(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        if self._sequence is None or self._value is UNDEFINED:
            raise RuntimeError("Replicated state received an update before hydration")
        if sequence != self._sequence + 1:
            self.clear()
            raise RuntimeError("Replicated state update sequence has a gap")
        value = apply_immutable(cast(JsonValue, self._value), ops)
        self._sequence = sequence
        self._value = cast(T, value)
        self._deliver_all(context, {"kind": "update", "sequence": sequence})

    def clear(self) -> None:
        self._value = UNDEFINED
        self._sequence = None

    def _deliver_all(self, context: Context, delivery: ReplicatedStateDelivery) -> None:
        if self._value is UNDEFINED:
            return
        for listener in self._listeners:
            self._deliver(listener, cast(T, self._value), context, delivery)

    def _deliver(
        self, listener: Callable[[T, Context, ReplicatedStateDelivery], None], value: T,
        context: Context, delivery: ReplicatedStateDelivery,
    ) -> None:
        try:
            listener(value, context, delivery)
        except (Exception, asyncio.CancelledError) as error:
            self._report_error(error if isinstance(error, Exception) else RuntimeError(str(error)))


def service_delivery_context() -> Context:
    return BACKGROUND_CONTEXT
