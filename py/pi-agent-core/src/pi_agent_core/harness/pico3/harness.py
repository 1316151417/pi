"""The harness ported from ``harness/pico3/harness.ts``.

Composition root: it owns the storage-backed :class:`~.session.Session`, the
:class:`~.view.ViewManager`, the :class:`~.scheduler.Scheduler`, the registries
(kinds, tools, sections, entry kinds, namespaces, plugins, hooks) and the
per-conversation handles the host talks to.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

from ..._chord.context import Context, with_abort_signal
from ..._pi_ai.abort import AbortSignal
from ..._pi_ai.types import JsonObject, JsonValue
from .hooks import HookRegistration, create_hook_runners
from .kinds.collapse import collapse, collapse_kind, choose_through
from .kinds.entries import entries as builtin_entries
from .kinds.generation import generation, generation_kind
from .kinds.job import job, job_kind
from .kinds.plugin import plugin, plugin_kind
from .kinds.post_tools import post_tools, post_tools_kind
from .kinds.tool import tool, tool_kind
from .scheduler import Scheduler, SchedulerDeps
from .session import Session, is_core_kind
from .system import SectionRegistry, SystemSection, ToolRegistry, system_sections
from .types import (
    AnyKind,
    AnyToolDeclaration,
    ConfigFacade,
    ContextView,
    Conversation,
    ConversationSpec,
    DocRef,
    Entry,
    EntryKind,
    EntryScan,
    Forbidden,
    HookInfo,
    Id,
    Input,
    Invoker,
    Namespace,
    NamespaceDefaults,
    NamespaceRegistration,
    NewEntry,
    PluginHandler,
    ProcessHost,
    RewindableState,
    Runtime,
    SendInput,
    StickyState,
    Storage,
    Task,
    TaskScan,
    ToolDeclaration,
    UserInput,
)
from .view import WATCH_CAPACITY, ViewManager, Watch, apply_envelope  # noqa: F401 - re-exported

__all__ = [
    "HarnessOptions",
    "Harness",
    "ConversationHandle",
    "ConversationConfig",
    "InputHandle",
    "CaptureScan",
    "capture_active_transcript",
    "kinds",
    "entries",
    "BUILTIN_KINDS",
]


#: The built-in kinds, as typed witnesses for ``api.task``, ``waitForTask``, and ``hooks(kind, ...)``.
kinds = {
    "pi.generation": generation,
    "pi.tool": tool,
    "pi.post_tools": post_tools,
    "pi.collapse": collapse,
    "pi.job": job,
    "pi.plugin": plugin,
}

#: The built-in entry kind witnesses.
entries = builtin_entries

BUILTIN_KINDS = [generation, tool, post_tools, collapse, job, plugin]


@dataclass
class HarnessOptions:
    models: Any
    tools: Optional[List[AnyToolDeclaration]] = None
    #: Ordinary kinds authored with ``define_task``; their config must be disjoint.
    task_kinds: Optional[List[AnyKind]] = None
    sections: Optional[List[SystemSection]] = None
    plugins: Optional[Dict[str, PluginHandler]] = None
    process_host: Optional[ProcessHost] = None
    #: Clock for durable kernel timestamps and retry scheduling.
    now: Optional[Callable[[], int]] = None
    root: Optional[Dict[str, Any]] = None
    #: Errors from listeners, hooks, watches, and the scheduler. Never commit failures.
    on_report: Optional[Callable[[Any], None]] = None


@dataclass
class _Registries:
    """``rt.registries``: the section and tool registries with their revisions."""

    sections: "SectionRegistry"
    tools: "ToolRegistry"


@dataclass
class InputHandle:
    id: Id
    result: Callable[[Context], Any]
    wait: Callable[[Context], Any]
    abort: Callable[[Context], Any]


CaptureScan = Callable[[EntryScan], Any]


class ConversationConfig:
    """The conversation-level configuration facade."""

    def __init__(self, harness: "Harness", conversation_id: Id) -> None:
        self._harness = harness
        self._conversation_id = conversation_id

    async def get(self, ctx: Context) -> Dict[str, JsonValue | None]:
        async def read(tx: Any, _ctx: Any = None) -> Dict[str, JsonValue | None]:
            facade = tx.config(self._conversation_id)
            out: Dict[str, JsonValue | None] = {}
            for key in self._harness.session.defaults.route:
                out[key] = facade.get(key)
            return out

        return await self._harness._host(self._conversation_id, read, ctx)

    async def set(self, patch: Dict[str, JsonValue], ctx: Context) -> None:
        async def write(tx: Any, _ctx: Any = None) -> None:
            facade = tx.config(self._conversation_id)
            for key, value in patch.items():
                if value is None:
                    raise ValueError(f"config.set({key}): use config.reset()")
                facade.set(key, value)

        await self._harness._host(self._conversation_id, write, ctx)

    async def reset(self, keys: List[str], ctx: Context) -> None:
        async def write(tx: Any, _ctx: Any = None) -> None:
            facade = tx.config(self._conversation_id)
            for key in keys:
                facade.reset(str(key))

        await self._harness._host(self._conversation_id, write, ctx)


class ConversationHandle:
    """What the host gets for one conversation. Capability-scoped, never a raw transaction."""

    def __init__(self, harness: "Harness", conversation: Conversation) -> None:
        self._harness = harness
        self.id = conversation.id
        self.config = ConversationConfig(harness, conversation.id)

    async def send(self, input: SendInput, ctx: Context) -> InputHandle:
        id = await self._harness._kernel(self.id, lambda tx, _ctx=None: tx.send(self.id, input), ctx)
        return self._harness._input_handle(id)

    async def write(self, entry: NewEntry, ctx: Context) -> Id:
        return await self._harness._host(self.id, lambda tx, _ctx=None: tx.write(self.id, entry), ctx)

    async def commit(self, fn: Callable[[Any, Context], Any], ctx: Context) -> Any:
        return await self._harness._host(self.id, lambda tx, line_ctx=None: fn(tx, line_ctx), ctx)

    async def rewindable(self, ctx: Context) -> RewindableState:
        return await self._harness._host(
            self.id,
            lambda tx, _ctx=None: tx.snapshot(DocRef(doc="rewindable", conversation_id=self.id)),
            ctx,
        )

    async def sticky(self, ctx: Context) -> StickyState:
        return await self._harness._host(
            self.id,
            lambda tx, _ctx=None: tx.snapshot(DocRef(doc="sticky", conversation_id=self.id)),
            ctx,
        )

    async def context(self, ctx: Context) -> ContextView:
        return await self._harness._host(self.id, lambda tx, _ctx=None: tx.context(self.id), ctx)

    async def fork(self, at: Union[Id, str], spec: Dict[str, Any], ctx: Context) -> "ConversationHandle":
        id = await self._harness.session.fork(self.id, at, spec, ctx)
        handle = await self._harness.conversation(id, ctx)
        assert handle is not None
        return handle

    async def collapse(self, instructions: Optional[str], ctx: Context) -> Id:
        async def create(tx: Any, _ctx: Any = None) -> Id:
            view = await tx.context(self.id)
            state = tx.rewindable(self.id)
            through = choose_through(view.entries, state.get("keepRecent") or 0)
            if through is None:
                raise ValueError("nothing to collapse")
            input_value: Dict[str, Any] = {"reason": "manual", "through": through}
            if instructions is not None:
                input_value["instructions"] = instructions
            return tx.create_task(
                tx.kinds["pi.collapse"], input_value, {"conversationId": self.id, "background": True}
            ).id

        return await self._harness._kernel(self.id, create, ctx)

    async def reset(self, handoff: Optional[str], ctx: Context) -> None:
        async def write(tx: Any, _ctx: Any = None) -> Id:
            if handoff is None:
                entry = NewEntry(kind="pi.reset", head="self")
            else:
                entry = NewEntry(
                    kind="pi.handoff",
                    head="self",
                    model=[{"role": "user", "content": handoff, "timestamp": self._harness.now()}],
                )
            return tx.write(self.id, entry)

        await self._harness._kernel(self.id, write, ctx)

    async def abort(self, ctx: Context) -> None:
        owned: List[Id] = []

        async def mark(tx: Any, _ctx: Any = None) -> List[Id]:
            sticky = tx.sticky(self.id)
            inbox = sticky.setdefault("inbox", [])
            withdrawn = [q["id"] for q in inbox if q.get("mode") != "write"]
            for index in range(len(inbox) - 1, -1, -1):
                if inbox[index].get("mode") != "write":
                    del inbox[index]
            await tx.resolve_inputs(withdrawn, {"status": "unanswered", "reason": "aborted"})
            for input_id in withdrawn:
                tx.emit({"type": "input.aborted", "input": input_id})
            ids: List[Id] = []
            for task in list(self._harness.session.live_tasks.values()):
                if task.conversation_id != self.id or task.background:
                    continue
                owned.extend(task.owns)
                if task.abort is not True:
                    tx.mark_task(task.id)
                ids.append(task.id)
            return ids

        marked = await self._harness._kernel(self.id, mark, ctx)
        for id in marked:
            await self._harness.scheduler.abort_task(id, ctx)
        await self._harness.scheduler.wait_for_idle(self.id, ctx)
        await asyncio.gather(
            *[self._harness.scheduler.wait_for_idle(o, ctx) for o in owned]
        )

    async def wait_for_idle(self, ctx: Context) -> None:
        await self._harness.scheduler.wait_for_idle(self.id, ctx)

    def hooks(
        self,
        namespace: Namespace,
        kind: AnyKind,
        handlers: Any,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Callable[[], None]:
        return self._harness.add_hooks(
            HookRegistration(
                namespace=self._harness.check_namespace(namespace),
                kind=self._harness.check_kind(kind),
                handlers=handlers,
                conversation_id=self.id,
                subtree=bool((opts or {}).get("subtree")),
            )
        )

    async def watch(self, ctx: Context) -> Watch:
        return await self._harness._watch(self.id, ctx)


class Harness:
    """The composition root: session, scheduler, views, registries."""

    def __init__(self, storage: Storage, options: HarnessOptions, ctx: Context) -> None:
        self.ctx = ctx
        self.options = options
        self.now = options.now or (lambda: 0)
        self._reported: Callable[[Any], None] = options.on_report or (lambda _error: None)

        def on_report(error: Any) -> None:
            try:
                self._reported(error)
            except Exception:  # noqa: BLE001 - a reporter must never break the caller
                pass

        self.on_report = on_report
        # Fixed core, installed internally. User kinds may not replace or shadow a built-in.
        self.kinds: Dict[str, AnyKind] = {kind.name: kind for kind in BUILTIN_KINDS}
        for kind in options.task_kinds or []:
            if kind.name.startswith("pi."):
                raise ValueError(f'task kind "{kind.name}": names beginning with "pi." are reserved')
            if kind.name in self.kinds:
                raise ValueError(f'task kind "{kind.name}" registered twice')
            self.kinds[kind.name] = kind
        self.tools: Dict[str, AnyToolDeclaration] = {}
        self.section_map: Dict[str, SystemSection] = {}
        self.entry_kinds: Dict[str, EntryKind] = {}
        self.revisions = {"sections": 0, "tools": 0}
        self.namespaces: Dict[str, NamespaceRegistration] = {}
        for tool_declaration in options.tools or []:
            self.register_tool(tool_declaration, "open")
        for section in system_sections:
            self.section_map[section.key] = section
        for section in options.sections or []:
            self.register_section(section, "open")
        for kind in builtin_entries:
            self.entry_kinds[kind.kind] = kind
        self.plugins: Dict[str, PluginHandler] = dict(options.plugins or {})
        self.hook_registrations: List[HookRegistration] = []
        self.conversation_listeners: Set[Callable[[ConversationHandle], None]] = set()
        self.resumed = False
        self.suspended = False

        self.session = Session(storage, self.kinds, self.namespaces, self.now)
        self.session.on_report = self.on_report
        self.views = ViewManager(self.session, self.on_report)
        self.session.line_listeners.add(lambda result: self.views.update(result))

        def on_commit(result: Any) -> None:
            self.views.deliver()
            for conversation in result.changes.conversations:
                self.notify_conversation(conversation)

        self.session.listeners.add(on_commit)
        hooks_for = create_hook_runners(
            lambda: self.hook_registrations,
            lambda c: self.session.index.ancestors(c),
            self.on_report,
        )
        self.scheduler = Scheduler(
            SchedulerDeps(
                session=self.session,
                kinds=self.kinds,
                runtime=lambda task, invoker, ictx: self._runtime(task, invoker, ictx, hooks_for),
                on_report=self.on_report,
                ctx=ctx,
            )
        )

    @staticmethod
    async def open(storage: Storage, options: HarnessOptions, ctx: Context) -> "Harness":
        harness = Harness(storage, options, ctx)
        await harness.init()
        return harness

    # -- runtime factory ---------------------------------------------------

    def _registries(self) -> Any:
        """The section and tool registries with their current revisions, as ``rt.registries``."""
        return _Registries(
            sections=SectionRegistry(map=self.section_map, revision=self.revisions["sections"]),
            tools=ToolRegistry(map=self.tools, revision=self.revisions["tools"]),
        )

    def _runtime(
        self,
        task: Task,
        invoker: Invoker,
        _ictx: Context,
        hooks_for: Callable[[AnyKind, HookInfo], Any],
    ) -> Any:
        harness = self

        class _Runtime:
            task_id = task.id
            conversation_id = task.conversation_id
            kind = invoker.kind
            models = self.options.models
            tools = self.tools
            kinds = self.kinds
            process_host = self.options.process_host
            plugins = self.plugins
            hooks = hooks_for(invoker.kind, HookInfo(kind=invoker.kind.name, task_id=task.id, conversation_id=task.conversation_id))

            registries = harness._registries()

            def now(self_inner) -> int:  # pragma: no cover - bound below
                return harness.now()

            def commit(self_inner, fn: Callable[..., Any], c: Context) -> Any:
                # Every conversation this task owns may be read/written by its callbacks.
                live = harness.session.live_tasks.get(task.id)
                docs: List[DocRef] = []
                for id in live.owns if live is not None else []:
                    docs.append(DocRef(doc="rewindable", conversation_id=id))
                    docs.append(DocRef(doc="sticky", conversation_id=id))

                async def call(tx: Any, line_ctx: Context, control: Any = None) -> Any:
                    current = harness.session.live_tasks.get(task.id)
                    return await _resolve(fn(tx, current, line_ctx))

                async def run() -> Any:
                    result = await harness.session.commit(invoker, call, c, {"docs": docs})
                    return result.value

                return asyncio.ensure_future(run())

            async def sleep(self_inner, until_ms: int, c: Context) -> None:
                milliseconds = max(0, until_ms - harness.now())
                await _sleep_with_abort(milliseconds, c)

            async def wait_for_input(self_inner, id: Id, c: Context) -> Input:
                return await harness.scheduler.wait_for_input(id, c)

            async def wait_for_task(self_inner, id: Id, c: Context) -> Task:
                return await harness.scheduler.wait_for_task(id, c)

            async def abort_task(self_inner, id: Id, c: Context) -> str:
                async def check(line_ctx: Context) -> None:
                    harness.assert_invocation(invoker)
                    target = harness.session.live_tasks.get(id) or await harness.session.storage.task(id, line_ctx)
                    if target is None:
                        raise ValueError(f"task {id} not found")
                    harness.assert_task_conversation_scope(task.id, target.conversation_id)

                await harness.session.on_line(check, c)
                return await harness.scheduler.abort_task(id, c)

            async def abort_conversation(self_inner, id: Id, c: Context) -> None:
                def check(_line_ctx: Context) -> None:
                    harness.assert_invocation(invoker)
                    harness.assert_owned_conversation(task.id, id)

                await harness.session.on_line(check, c)
                handle = await harness.conversation(id, c)
                if handle is not None:
                    await handle.abort(c)

            async def create_owned_conversation(self_inner, spec: Any, c: Context) -> Id:
                async def create(tx: Any, _line_ctx: Context = None, _control: Any = None) -> Id:
                    harness.assert_invocation(invoker)
                    return await tx.create_owned_conversation(task.id, task.conversation_id, spec)

                result = await harness.session.commit({"type": "kernel"}, create, c)
                return result.value

            async def send_owned(self_inner, id: Id, input: SendInput, c: Context) -> Id:
                async def send(tx: Any, _line_ctx: Context = None, _control: Any = None) -> Id:
                    harness.assert_invocation(invoker)
                    harness.assert_owned_conversation(task.id, id)
                    return await tx.send(id, input)

                result = await harness.session.commit(
                    {"type": "kernel"},
                    send,
                    c,
                    {
                        "docs": [
                            DocRef(doc="rewindable", conversation_id=id),
                            DocRef(doc="sticky", conversation_id=id),
                        ]
                    },
                )
                return result.value

            async def context(self_inner, conversation_id: Id, at: Optional[Id], c: Context) -> ContextView:
                async def read(tx: Any, _line_ctx: Context = None, _control: Any = None) -> ContextView:
                    return await tx.context(conversation_id, at)

                result = await harness.session.commit(invoker, read, c)
                return result.value

            async def newest_entry(self_inner, conversation_id: Id, opts: Any, c: Context) -> Any:
                async def read(tx: Any, _line_ctx: Context = None, _control: Any = None) -> Any:
                    return await tx.newest_entry(conversation_id, opts)

                result = await harness.session.commit(invoker, read, c)
                return result.value

            async def rewindable(self_inner, conversation_id: Id, c: Context) -> Any:
                async def read(tx: Any, _line_ctx: Context = None, _control: Any = None) -> Any:
                    return tx.snapshot(DocRef(doc="rewindable", conversation_id=conversation_id))

                result = await harness.session.commit(
                    invoker,
                    read,
                    c,
                    {"docs": [DocRef(doc="rewindable", conversation_id=conversation_id)]},
                )
                return result.value

            async def sticky(self_inner, conversation_id: Id, c: Context) -> Any:
                async def read(tx: Any, _line_ctx: Context = None, _control: Any = None) -> Any:
                    return tx.snapshot(DocRef(doc="sticky", conversation_id=conversation_id))

                result = await harness.session.commit(
                    invoker,
                    read,
                    c,
                    {"docs": [DocRef(doc="sticky", conversation_id=conversation_id)]},
                )
                return result.value

            async def rewindable_as_of(self_inner, conversation_id: Id, at: Id, c: Context) -> Any:
                async def read(tx: Any, _line_ctx: Context = None, _control: Any = None) -> Any:
                    return await tx.rewindable_as_of(conversation_id, at)

                result = await harness.session.commit(invoker, read, c)
                return result.value

            commit = commit
            sleep = sleep
            wait_for_input = wait_for_input
            wait_for_task = wait_for_task
            abort_task = abort_task
            abort_conversation = abort_conversation
            create_owned_conversation = create_owned_conversation
            send_owned = send_owned
            context = context
            newest_entry = newest_entry
            rewindable = rewindable
            sticky = sticky
            rewindable_as_of = rewindable_as_of

        return _Runtime()

    # -- lifecycle ---------------------------------------------------------

    async def init(self) -> None:
        session = self.session
        conversations = await session.read(lambda s, ctx: s.conversations(ctx), self.ctx)
        for conversation in conversations:
            session.conversation_records[conversation.id] = conversation
        live = await session.read(lambda s, ctx: s.scan_tasks(TaskScan(status=["pending", "running"]), ctx), self.ctx)
        for task in live:
            session.live_tasks[task.id] = task
        # Owner tasks that are terminal but still own existing conversations: needed for ancestry.
        owners = {conversation.owner for conversation in conversations if conversation.owner is not None}
        for id in owners:
            if id not in session.live_tasks:
                task = await session.read(lambda s, ctx, id=id: s.task(id, ctx), self.ctx)
                if task is not None:
                    session.owner_task_cache[id] = task
        if not conversations:
            await session.commit(
                {"type": "kernel"},
                lambda tx, _ctx=None, _control=None: tx.create_conversation(
                    ConversationSpec(
                        rewindable=(self.options.root or {}).get("rewindable"),
                        sticky=(self.options.root or {}).get("sticky"),
                    )
                ),
                self.ctx,
            )

    def resume(self) -> None:
        if self.suspended:
            raise ValueError("cannot resume a suspended harness; reopen storage with a new harness")
        if self.resumed:
            return
        self.resumed = True

        async def start() -> None:
            await self.reconcile_orphans()
            if not self.suspended:
                self.scheduler.resume()

        task = asyncio.ensure_future(start())
        task.add_done_callback(
            lambda future: None if future.cancelled() else self.on_report(future.exception())
        )

    async def reconcile_orphans(self) -> None:
        orphaned = [
            task for task in self.session.live_tasks.values() if task.kind not in self.kinds
        ]
        if not orphaned:
            return
        docs = [
            DocRef(doc="sticky", conversation_id=conversation_id)
            for conversation_id in {task.conversation_id for task in orphaned}
        ]

        async def terminalize(tx: Any, _ctx: Any = None, _control: Any = None) -> None:
            for task in orphaned:
                from .session import _copy_task

                updated = _copy_task(task)
                updated.status = "terminal"
                updated.outcome = _outcome("orphaned")
                tx.set_task(updated)

        await self.session.commit({"type": "kernel"}, terminalize, self.ctx, {"docs": docs})

    def quiescent(self) -> bool:
        return self.scheduler.quiescent()

    def hold(self) -> Callable[[], None]:
        if not self.scheduler.quiescent():
            raise ValueError("cannot hold a non-quiescent harness; suspend it instead")
        return self.scheduler.hold()

    async def suspend(self, ctx: Context) -> None:
        """Cancel and join in-process invocations, clear transient waits, then close."""
        if self.suspended:
            return
        self.suspended = True
        await self.scheduler.join_all()
        tools = [task for task in self.session.live_tasks.values() if task.kind == "pi.tool"]
        if tools:
            docs = [
                DocRef(doc="sticky", conversation_id=conversation_id)
                for conversation_id in {task.conversation_id for task in tools}
            ]

            async def clear_waiting(tx: Any, _ctx: Any = None, _control: Any = None) -> None:
                for task in tools:
                    index = _input_index(task)
                    slots = tx.sticky(task.conversation_id).setdefault("turn", {}).setdefault("tools", [])
                    slot = slots[index] if index is not None and index < len(slots) else None
                    if slot is not None and slot.get("waitingOn") is not None:
                        slot.pop("waitingOn", None)

            await self.session.commit({"type": "kernel"}, clear_waiting, ctx, {"docs": docs})
        self.views.close()
        await self.session.close(ctx)

    # -- registries --------------------------------------------------------

    def register_task_kind(self, kind: AnyKind) -> Callable[[], None]:
        if kind.name.startswith("pi."):
            raise ValueError(f'task kind "{kind.name}": names beginning with "pi." are reserved')
        if kind.name in self.kinds:
            raise ValueError(f'task kind "{kind.name}" already registered')
        self.session.defaults.register(kind)
        self.kinds[kind.name] = kind
        self.scheduler.kick()

        def unregister() -> None:
            if self.kinds.get(kind.name) is not kind:
                return
            del self.kinds[kind.name]
            self.session.defaults.unregister(kind)

        return unregister

    def namespace(
        self,
        id: str,
        defaults: NamespaceDefaults,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Namespace:
        if not _is_namespace_id(id) or id.startswith("pi."):
            raise ValueError(f'invalid namespace "{id}"')
        if id in self.namespaces:
            raise ValueError(f'namespace "{id}" already registered')
        stored = {
            "rewindable": json.loads(json.dumps(defaults.rewindable or {})),
            "sticky": json.loads(json.dumps(defaults.sticky or {})),
            "session": json.loads(json.dumps(defaults.session or {})),
        }
        routes: Dict[str, str] = {}
        for doc in ("rewindable", "sticky", "session"):
            for key in stored[doc]:
                if key in routes:
                    raise ValueError(f'namespace "{id}" key "{key}" is declared in more than one document')
                routes[key] = doc
        token = Namespace(id=id)

        def unregister() -> None:
            current = self.namespaces.get(id)
            if current is None or current.token is not token:
                return
            del self.namespaces[id]
            for index in range(len(self.hook_registrations) - 1, -1, -1):
                if self.hook_registrations[index].namespace is token:
                    del self.hook_registrations[index]

        token.unregister = unregister
        view = (opts or {}).get("view")
        self.namespaces[id] = NamespaceRegistration(
            token=token,
            defaults=NamespaceDefaults(
                rewindable=stored["rewindable"],
                sticky=stored["sticky"],
                session=stored["session"],
            ),
            routes=routes,
            project=(None if view is None else (lambda slice_value: view(slice_value))),
        )
        return token

    def register_tool(self, tool_declaration: AnyToolDeclaration, at: str = "runtime") -> Callable[[], None]:
        if tool_declaration.name in self.tools:
            raise ValueError(f'tool "{tool_declaration.name}" already registered')
        self.tools[tool_declaration.name] = tool_declaration
        self.revisions["tools"] += 1

        def unregister() -> None:
            if self.tools.get(tool_declaration.name) is tool_declaration:
                del self.tools[tool_declaration.name]
                self.revisions["tools"] += 1

        return unregister

    def register_section(self, section: SystemSection, at: str = "runtime") -> Callable[[], None]:
        if section.key in self.section_map:
            raise ValueError(f'section "{section.key}" already registered')
        self.section_map[section.key] = section
        self.revisions["sections"] += 1

        def unregister() -> None:
            if self.section_map.get(section.key) is section:
                del self.section_map[section.key]
                self.revisions["sections"] += 1

        return unregister

    def register_entry_kind(self, kind: EntryKind) -> Callable[[], None]:
        if kind.kind.startswith("pi."):
            raise ValueError(f'entry kind "{kind.kind}": names beginning with "pi." are reserved')
        if kind.kind in self.entry_kinds:
            raise ValueError(f'entry kind "{kind.kind}" already registered')
        self.entry_kinds[kind.kind] = kind

        def unregister() -> None:
            if self.entry_kinds.get(kind.kind) is kind:
                del self.entry_kinds[kind.kind]

        return unregister

    def hooks(
        self,
        namespace: Namespace,
        kind: AnyKind,
        handlers: Any,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Callable[[], None]:
        """Register namespace-bound handlers for one kind's hook points, harness-wide."""
        return self.add_hooks(
            HookRegistration(
                namespace=self.check_namespace(namespace),
                kind=self.check_kind(kind),
                handlers=handlers,
                subtree=bool((opts or {}).get("subtree")),
            )
        )

    def check_kind(self, kind: AnyKind) -> AnyKind:
        if self.kinds.get(kind.name) is not kind:
            raise ValueError(f'kind "{kind.name}" is not the registered token')
        return kind

    def check_namespace(self, namespace: Namespace) -> Namespace:
        current = self.namespaces.get(namespace.id)
        if current is None or current.token is not namespace:
            raise Forbidden(f'namespace "{namespace.id}" is stale')
        return namespace

    def add_hooks(self, registration: HookRegistration) -> Callable[[], None]:
        self.hook_registrations.append(registration)
        done = False

        def remove() -> None:
            nonlocal done
            if done:
                return
            done = True
            if registration in self.hook_registrations:
                self.hook_registrations.remove(registration)

        return remove

    # -- conversations -----------------------------------------------------

    async def root(self, ctx: Context) -> ConversationHandle:
        handle = await self.conversation(1, ctx)
        assert handle is not None
        return handle

    def on_conversation(self, listener: Callable[[ConversationHandle], None]) -> Callable[[], None]:
        self.conversation_listeners.add(listener)
        for conversation in list(self.session.conversation_records.values()):
            try:
                listener(self.handle(conversation))
            except Exception as error:  # noqa: BLE001 - reported, never raised
                self.on_report(error)

        def remove() -> None:
            self.conversation_listeners.discard(listener)

        return remove

    async def conversation(self, id: Id, ctx: Context) -> Optional[ConversationHandle]:
        conversation = await self.session.read(lambda s, line_ctx: s.conversation(id, line_ctx), ctx)
        return None if conversation is None else self.handle(conversation)

    async def create_conversation(self, spec: Dict[str, Any], ctx: Context) -> ConversationHandle:
        input_value = spec.get("input")
        rest = {key: value for key, value in spec.items() if key != "input"}

        async def create(tx: Any, _ctx: Any = None, _control: Any = None) -> Id:
            id = tx.create_conversation(ConversationSpec(**rest))
            if input_value is not None:
                await tx.send(id, SendInput(content=input_value))
            return id

        result = await self.session.commit({"type": "kernel"}, create, ctx)
        handle = await self.conversation(result.value, ctx)
        assert handle is not None
        return handle

    def entries(self, scan: EntryScan, ctx: Context) -> Any:
        return self.session.read(lambda s, line_ctx: s.scan_entries(scan, line_ctx), ctx)

    def get_task(self, id: Id, ctx: Context) -> Any:
        return self.session.read(lambda s, line_ctx: s.task(id, line_ctx), ctx)

    async def abort_input(self, id: Id, ctx: Context, conversation_id: Optional[Id] = None) -> str:
        if conversation_id is not None:
            input_record = await self.session.read(lambda s, line_ctx: s.input(id, line_ctx), ctx)
            if input_record is not None and input_record.conversation_id != conversation_id:
                raise Forbidden(f"input {id} is outside conversation {conversation_id}")
        return await self.input_handle(id).abort(ctx)

    def abort_task(self, id: Id, ctx: Context) -> Any:
        return self.scheduler.abort_task(id, ctx)

    async def mark_task(self, id: Id, ctx: Context) -> str:
        """Durably mark a task for abort without signalling its invocation."""

        async def mark(tx: Any, _ctx: Any = None, _control: Any = None) -> str:
            task = await tx.task(id)
            if task is None:
                raise ValueError(f"task {id} not found")
            if task.status == "terminal":
                return "terminal"
            tx.mark_task(id)
            return "marked"

        result = await self.session.commit({"type": "kernel"}, mark, ctx)
        return result.value

    def wait_for_idle(self, ctx: Context) -> Any:
        return self.scheduler.wait_for_idle(None, ctx)

    def wait_for_task(self, id: Id, ctx: Context) -> Any:
        return self.scheduler.wait_for_task(id, ctx)

    async def close(self, ctx: Context) -> None:
        """Signals every invocation, waits for them, closes storage. Writes nothing."""
        await self.suspend(ctx)

    # -- internals ---------------------------------------------------------

    def notify_conversation(self, conversation: Conversation) -> None:
        if not self.conversation_listeners:
            return
        handle = self.handle(conversation)
        for listener in list(self.conversation_listeners):
            try:
                listener(handle)
            except Exception as error:  # noqa: BLE001 - reported, never raised
                self.on_report(error)

    def assert_invocation(self, invoker: Invoker) -> None:
        if invoker.token is None or not invoker.token.alive:
            raise Forbidden("operation from a finished invocation")
        live = self.session.live_tasks.get(invoker.id or 0)
        if live is None:
            raise Forbidden("operation from a task that is not live")
        if invoker.mode == "run" and live.abort is True:
            raise Forbidden("operation from a marked run invocation")

    def assert_task_conversation_scope(self, task_id: Id, conversation_id: Id) -> None:
        task = self.session.live_tasks.get(task_id)
        if task is not None and task.conversation_id == conversation_id:
            return
        if task is not None and any(
            conversation_id in self.session.index.subtree(root) for root in task.owns
        ):
            return
        raise Forbidden(f"conversation {conversation_id} is outside task {task_id}'s subtree")

    def assert_owned_conversation(self, task_id: Id, conversation_id: Id) -> None:
        task = self.session.live_tasks.get(task_id)
        if task is None or not any(
            conversation_id in self.session.index.subtree(root) for root in task.owns
        ):
            raise Forbidden(f"conversation {conversation_id} is not owned by task {task_id}")

    def handle(self, conversation: Conversation) -> ConversationHandle:
        return ConversationHandle(self, conversation)

    async def _host(self, conversation_id: Id, fn: Callable[..., Any], ctx: Context) -> Any:
        invoker = Invoker(type="host", conversation_id=conversation_id)

        async def call(tx: Any, line_ctx: Context, _control: Any = None) -> Any:
            return await _resolve(fn(tx, line_ctx))

        result = await self.session.commit(invoker, call, ctx, {"docs": _conversation_docs(conversation_id)})
        return result.value

    async def _kernel(self, conversation_id: Id, fn: Callable[..., Any], ctx: Context) -> Any:
        invoker = Invoker(type="kernel", conversation_id=conversation_id)

        async def call(tx: Any, line_ctx: Context, _control: Any = None) -> Any:
            return await _resolve(fn(tx, line_ctx))

        result = await self.session.commit(invoker, call, ctx, {"docs": _conversation_docs(conversation_id)})
        return result.value

    async def _watch(self, conversation_id: Id, ctx: Context) -> Watch:
        """Capture and subscribe in one line operation."""
        conversation = await self.session.read(
            lambda s, line_ctx: s.conversation(conversation_id, line_ctx), ctx
        )

        async def capture(tx: Any, _line_ctx: Context = None, _control: Any = None) -> Watch:
            captured = await capture_active_transcript(lambda scan: tx.scan_entries(scan), conversation_id)
            return self.views.watch(conversation, captured)

        result = await self.session.commit(
            {"type": "host", "conversationId": conversation_id},
            capture,
            ctx,
            {"docs": _conversation_docs(conversation_id)},
        )
        return result.value

    def _input_handle(self, id: Id) -> InputHandle:
        return InputHandle(
            id=id,
            result=lambda ctx: self.session.read(lambda s, line_ctx: s.input(id, line_ctx), ctx),
            wait=lambda ctx: self.scheduler.wait_for_input(id, ctx),
            abort=lambda ctx: self._kernel(id, lambda tx, _ctx=None: tx.withdraw_input(id), ctx),
        )


async def capture_active_transcript(scan: CaptureScan, conversation_id: Id) -> List[Entry]:
    """H = newest fork-visible entry with a head; the transcript from H.head forward."""
    head_page = await scan(EntryScan(conversation_id=conversation_id, with_head=True, limit=1))
    head = head_page[0] if head_page else None
    start = head.head if head is not None else None
    out: List[Entry] = []
    before: Optional[Id] = None
    while True:
        page = await scan(
            EntryScan(conversation_id=conversation_id, limit=256, before=before)
        )
        done = len(page) < 256
        for entry in page:
            if start is not None and entry.id < start:
                done = True
                break
            out.append(entry)
        if done:
            break
        before = page[-1].id
    out.reverse()
    return out


def _conversation_docs(conversation_id: Id) -> List[DocRef]:
    return [
        DocRef(doc="rewindable", conversation_id=conversation_id),
        DocRef(doc="sticky", conversation_id=conversation_id),
    ]


def _outcome(status: str, **fields: Any) -> Any:
    from .types import Outcome

    return Outcome(status=status, **fields)


def _input_index(task: Task) -> Optional[int]:
    if isinstance(task.input, dict):
        return task.input.get("index")
    value = getattr(task.input, "index", None)
    return value if isinstance(value, int) else None


def _is_namespace_id(id: str) -> bool:
    if not id:
        return False
    first = id[0]
    if not (first.isalpha() and first.isascii()):
        return False
    return all(char.isascii() and (char.isalnum() or char in "_.-") for char in id)


async def _resolve(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _sleep_with_abort(milliseconds: int, ctx: Context) -> None:
    """Sleep, rejecting if the context aborts first."""
    sleep_task = asyncio.ensure_future(asyncio.sleep(milliseconds / 1000))
    if ctx.signal is None:
        await sleep_task
        return
    if ctx.signal.aborted:
        sleep_task.cancel()
        ctx.signal.throw_if_aborted()
        return
    watch = asyncio.ensure_future(ctx.signal.wait())
    try:
        done, _pending = await asyncio.wait(
            {sleep_task, watch}, return_when=asyncio.FIRST_COMPLETED
        )
        if watch in done:
            raise ctx.signal.reason
    finally:
        if not watch.done():
            watch.cancel()
        if not sleep_task.done():
            sleep_task.cancel()


_ = (AbortSignal, Iterable, ToolDeclaration, UserInput, with_abort_signal, re, WATCH_CAPACITY, apply_envelope, collapse, collapse_kind, generation_kind, job, job_kind, plugin, plugin_kind, post_tools, post_tools_kind, tool, tool_kind)
