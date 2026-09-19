"""Chord integration ported from ``harness/pico3/chord.ts``.

Two things live here: the keyed conversation service a Chord facet exposes, and
the bridge that adapts Pico's commit-granular envelopes into one Chord
publication per commit (the raw listener only enqueues; applying ops and
publishing happen together, off the Session line, with no await between them).

Port notes:

* ``pi_agent_core._chord`` has no service/replicated-state module yet, so the
  three Chord primitives this file uses — ``defineService``, ``ReplicatedState``
  and ``MutableReplicatedState`` — are ported locally at the bottom of this
  module with exactly the surface ``chord.ts`` touches. They are named
  ``define_service``/``ServiceDefinition``, ``ReplicatedState`` and
  ``MutableReplicatedState``. Nothing else is invented.
* The TypeScript module exports one name for both the service type and the
  service token; Python keeps them apart: ``PicoConversationService`` is the
  shape, ``pico_conversation_service`` is the token.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..._chord.context import Context
from ..._pi_ai.types import JsonObject
from .harness import ConversationHandle as PicoConversationHandle, Harness as PicoHarness
from .session import CommitResult
from .types import (
    ConversationSpec,
    DocRef,
    Entry,
    EntryScan,
    Envelope,
    Id,
    JsonValue,
    NewEntry,
    SendInput,
    ViewEvent,
    view_event_to_json,
)
from .view import WATCH_CAPACITY, Watch

__all__ = [
    "PublishedConversationView",
    "PicoConversationService",
    "PicoHarnessServiceShape",
    "create_pico_conversation_service",
    "ChordViewBridge",
    "ChordViewBridgeOptions",
    "attach_chord_view",
    "apply_tracked",
    "define_service",
    "ServiceDefinition",
    "ReplicatedState",
    "MutableReplicatedState",
    "pico_harness_service",
    "pico_conversation_service",
]


#: A published conversation view: the Pico view plus this commit's events.
PublishedConversationView = Dict[str, Any]


class PicoConversationService(Protocol):
    """The keyed conversation contract used by Chord facets."""

    view: "MutableReplicatedState"

    async def send(self, input: SendInput, ctx: Context) -> Id: ...

    async def write(self, entry: NewEntry, ctx: Context) -> Id: ...

    async def input_abort(self, id: Id, ctx: Context) -> str: ...

    async def config_set(self, patch: JsonObject, ctx: Context) -> None: ...

    async def abort(self, ctx: Context) -> None: ...

    async def reset(self, handoff: Optional[str], ctx: Context) -> None: ...

    async def collapse(self, instructions: Optional[str], ctx: Context) -> Id: ...

    async def fork(self, at: Any, spec: Dict[str, Any], ctx: Context) -> Id: ...

    async def entries(self, scan: EntryScan, ctx: Context) -> List[Entry]: ...


class PicoHarnessServiceShape(Protocol):
    """The local registration/control plane contract."""


#: Local registration/control plane and remote keyed conversation contract used by Chord
#: facets. The tokens are created at the bottom of the module, once ``define_service`` is defined.
class ConversationService:
    """The object ``create_pico_conversation_service`` builds."""

    def __init__(self, harness: Any, conversation: PicoConversationHandle, view: "MutableReplicatedState") -> None:
        self.view = view
        self._harness = harness
        self._conversation = conversation

    async def send(self, input: SendInput, ctx: Context) -> Id:
        handle = await self._conversation.send(input, ctx)
        return handle.id

    async def write(self, entry: NewEntry, ctx: Context) -> Id:
        return await self._conversation.write(entry, ctx)

    async def input_abort(self, id: Id, ctx: Context) -> str:
        return await self._harness.abort_input(id, ctx, self._conversation.id)

    async def config_set(self, patch: JsonObject, ctx: Context) -> None:
        await self._conversation.config.set(patch, ctx)

    async def abort(self, ctx: Context) -> None:
        await self._conversation.abort(ctx)

    async def reset(self, handoff: Optional[str], ctx: Context) -> None:
        await self._conversation.reset(handoff, ctx)

    async def collapse(self, instructions: Optional[str], ctx: Context) -> Id:
        return await self._conversation.collapse(instructions, ctx)

    async def fork(self, at: Any, spec: Dict[str, Any], ctx: Context) -> Id:
        child = await self._conversation.fork(at, spec, ctx)
        return child.id

    async def entries(self, scan: EntryScan, ctx: Context) -> List[Entry]:
        adjusted = EntryScan(
            conversation_id=self._conversation.id,
            kind=scan.kind,
            with_head=scan.with_head,
            before=scan.before,
            limit=scan.limit,
        )
        return await self._harness.entries(adjusted, ctx)


def create_pico_conversation_service(
    harness: Any,
    conversation: PicoConversationHandle,
    view: "MutableReplicatedState",
) -> ConversationService:
    """Bind one conversation handle and its published view into a keyed service object."""
    return ConversationService(harness, conversation, view)


@dataclass
class ChordViewBridgeOptions:
    capacity: Optional[int] = None
    #: Called after the raw watch is closed; the owner should respawn its keyed service.
    on_failure: Optional[Callable[[Exception], None]] = None


class ChordViewBridge:
    """A publication bridge: ops applied and published together, off the Session line."""

    def __init__(
        self,
        view: "MutableReplicatedState",
        watch: Watch,
        ctx: Context,
        opts: ChordViewBridgeOptions,
    ) -> None:
        capacity = WATCH_CAPACITY if opts.capacity is None else opts.capacity
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("Chord view queue capacity must be positive")
        self.view = view
        self._watch = watch
        self._ctx = ctx
        self._on_failure = opts.on_failure
        self._capacity = capacity
        self._queue: List[Envelope] = []
        self._scheduled = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.clear()
        self._watch.stop()

    def _fail(self, cause: Any) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.clear()
        self._watch.stop()
        if self._on_failure is not None:
            try:
                self._on_failure(cause if isinstance(cause, Exception) else RuntimeError(str(cause)))
            except Exception:  # noqa: BLE001 - a failing reporter must not break the bridge
                pass

    def _drain(self) -> None:
        self._scheduled = False
        while not self._closed and self._queue:
            envelope = self._queue.pop(0)
            try:
                apply_tracked(self.view.state, envelope.ops)
                self.view.state.setdefault("commit", {})["events"] = [
                    view_event_to_json(event) for event in envelope.events
                ]
                self.view.publish(self._ctx)
            except Exception as error:  # noqa: BLE001 - the bridge fails closed
                self._fail(error)

    def accept(self, envelope: Envelope) -> None:
        if self._closed:
            return
        if len(self._queue) >= self._capacity:
            self._fail(RuntimeError(f"Pico-to-Chord view queue exceeded {self._capacity} envelopes"))
            return
        self._queue.append(envelope)
        if self._scheduled:
            return
        self._scheduled = True
        asyncio.get_running_loop().call_soon(self._drain)


async def attach_chord_view(
    conversation: Any,
    create_state: Callable[[PublishedConversationView], "MutableReplicatedState"],
    ctx: Context,
    opts: Optional[ChordViewBridgeOptions] = None,
) -> ChordViewBridge:
    """Adapt Pico's commit-granular envelopes to one Chord publication per commit."""
    watch = await conversation.watch(ctx)
    initial: PublishedConversationView = {**_deep_clone(watch.view), "commit": {"events": []}}
    view = create_state(initial)
    return _bridge_watch(watch, view, ctx, opts or ChordViewBridgeOptions())


def _bridge_watch(
    watch: Watch,
    view: "MutableReplicatedState",
    ctx: Context,
    opts: ChordViewBridgeOptions,
) -> ChordViewBridge:
    bridge = ChordViewBridge(view, watch, ctx, opts)
    watch.start(bridge.accept)
    return bridge


def apply_tracked(root: PublishedConversationView, ops: List[Any]) -> None:
    """Apply one Pico envelope to the published root IN PLACE (never a root replace)."""
    for op in ops:
        if op[0] == "r":
            raise ValueError("live Pico envelope unexpectedly replaced the view root")
        path = op[1]
        if op[0] == "p":
            target = _resolve(root, path)
            if not isinstance(target, list):
                raise ValueError(f"Pico splice path is not an array: {_join(path)}")
            del target[op[2] : op[2] + op[3]]
            for offset, item in enumerate(_deep_clone(op[4])):
                target.insert(op[2] + offset, item)
            continue
        parent = _resolve(root, path[:-1])
        if not isinstance(parent, (dict, list)):
            raise ValueError(f"Pico operation parent is not an object: {_join(path)}")
        key = path[-1] if path else None
        if op[0] == "d":
            if isinstance(parent, list):
                del parent[key]
            else:
                parent.pop(key, None)
            continue
        if op[0] == "s":
            parent[key] = _deep_clone(op[2])
        elif op[0] == "a":
            parent[key] = f"{parent.get(key)}" + op[2]
        else:
            parent[key] = str(parent.get(key))[op[2] :]


def _resolve(root: Any, path: Any) -> Any:
    value = root
    for segment in path:
        if not isinstance(value, (dict, list)):
            return None
        value = value[segment]
    return value


def _join(path: Any) -> str:
    return ".".join(str(segment) for segment in path)


def _deep_clone(value: Any) -> Any:
    import copy

    return copy.deepcopy(value)


# ---------------------------------------------------------------------------
# The three Chord primitives this module uses (ported locally; see the module note)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceDefinition:
    """A service token: a stable name plus whether the service is registered locally."""

    name: str
    local: bool = False


def define_service(name: str, opts: Optional[Dict[str, Any]] = None) -> ServiceDefinition:
    """Declare a service token, mirroring ``defineService`` from ``@earendil-works/chord``."""
    return ServiceDefinition(name=name, local=bool((opts or {}).get("local")))


class ReplicatedState:
    """A replicated state holder: ``state`` is the published value."""

    def __init__(self, initial: Any) -> None:
        self.state = initial


class MutableReplicatedState(ReplicatedState):
    """A state holder the local side may publish into."""

    def __init__(self, initial: Any) -> None:
        super().__init__(initial)
        self.published: List[Any] = []

    def publish(self, ctx: Context) -> None:
        self.published.append(_deep_clone(self.state))


#: Local registration/control plane used by Chord facets.
pico_harness_service = define_service("pi.harness", {"local": True})
#: Remote keyed conversation contract used by Chord facets.
pico_conversation_service = define_service("pi.conversation")


_ = (CommitResult, DocRef, JsonValue, ViewEvent, PicoHarness, field)
