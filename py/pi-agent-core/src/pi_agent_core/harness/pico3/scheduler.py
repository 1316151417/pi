"""The task scheduler ported from ``harness/pico3/scheduler.ts``.

"Reserve on the line, run off the line, re-check before persisting." Every
invocation runs on a lease: an :class:`~.types.InvocationToken` minted by the
scheduler, one per handler call, revoked the moment that handler returns.
A reopen makes no scheduling decision: nothing is dispatched until ``resume()``.
A resume reconciles state by dispatching dirty tasks into whatever phase their
checkpoint names.

Port notes:

* ``AbortSignal`` has no event listeners in this port, so waiting on an abort is
  await ``signal.wait()`` in a task that is cancelled when the wait finishes.
* The session is duck-typed through :class:`SessionLike`: the scheduler needs
  ``commit``, ``live_tasks``, ``listeners``, ``on_line``, ``storage`` and
  ``retire``, and nothing else.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Set

from pi_ai.abort import AbortController
from ..._chord.context import Context, with_abort_signal
from .types import (
    UNSET,
    AnyKind,
    completion_to_outcome,
    Checkpoint,
    Completion,
    Faulted,
    Id,
    Input,
    InvocationToken,
    Invoker,
    JsonValue,
    Outcome,
    Runtime,
    Step,
    Task,
    TaskContractFault,
)

__all__ = ["Scheduler", "SchedulerDeps", "SessionLike", "handler_for", "validate_step"]


class SessionLike(Protocol):
    """What the scheduler needs from the session line."""

    live_tasks: Dict[Id, Task]
    listeners: Set[Callable[[Any], None]]
    storage: Any

    async def commit(
        self,
        invoker: Any,
        fn: Callable[[Any, Context, Any], Any],
        ctx: Context,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any: ...

    async def on_line(self, fn: Callable[[Context], Any], ctx: Context) -> Any: ...

    async def retire(self, task: Task, ctx: Context) -> None: ...


@dataclass
class SchedulerDeps:
    session: SessionLike
    kinds: Dict[str, AnyKind]
    runtime: Callable[[Task, Invoker, Context], Runtime]
    on_report: Callable[[Any], None]
    ctx: Context


@dataclass
class _Invocation:
    task: Task
    mode: str  # "run" | "abort"
    controller: AbortController
    done: Any = None


@dataclass(eq=False)
class _IdleWaiter:
    conversation_id: Optional[Id]
    resolve: Callable[[], None]


@dataclass(eq=False)
class _Reserved:
    task: Task
    mode: str


# ---------------------------------------------------------------------------
# The erased-kind adapter: the one place a concrete kind is called through AnyKind.
# ---------------------------------------------------------------------------


def handler_for(kind: AnyKind, phase: Optional[str]) -> Callable[..., Any]:
    handler = kind.initial if phase is None else kind.phases.get(phase)
    if not callable(handler):
        raise TaskContractFault(kind.name, f"no handler for phase {phase if phase is not None else 'initial'}")
    return handler


def validate_step(kind: AnyKind, step: Any) -> Step:
    if not isinstance(step, Step):
        raise TaskContractFault(kind.name, "handler returned a non-object step")
    if callable(step.done):
        return step
    if step.next is not None:
        if callable(step.next):
            return step
        phase = step.next.get("phase") if isinstance(step.next, dict) else None
        if isinstance(phase, str):
            if phase not in kind.phases:
                raise TaskContractFault(kind.name, f"transition to unknown phase {phase}")
            return step
    raise TaskContractFault(kind.name, "handler returned an invalid step")


def validate_completion(kind: AnyKind, completion: Any) -> Completion:
    """A completion must carry the payload its status promises."""
    if isinstance(completion, Completion):
        if completion.status == "completed" and completion.result is not UNSET:
            return completion
        if completion.status == "failed" and completion.failure is not UNSET:
            return completion
    raise TaskContractFault(kind.name, "closure returned an invalid completion")


def validate_checkpoint(kind: AnyKind, checkpoint: Any) -> Checkpoint:
    phase = checkpoint.get("phase") if isinstance(checkpoint, dict) else None
    if not isinstance(phase, str):
        raise TaskContractFault(kind.name, "invalid checkpoint")
    if phase not in kind.phases:
        raise TaskContractFault(kind.name, f"checkpoint names unknown phase {phase}")
    if kind.inflight is not None and phase in kind.inflight:
        raise TaskContractFault(
            kind.name,
            f"transition into in-flight phase {phase}; "
            "write it with rt.commit before the effect instead",
        )
    json.dumps(checkpoint)
    return checkpoint


class Scheduler:
    """Reserves on the line, dispatches off the line, joins on abort."""

    def __init__(self, deps: SchedulerDeps) -> None:
        self.deps = deps
        self.enabled = False
        self.holds = 0
        self.dirty = False
        self.draining = False
        self.invocations: Dict[Id, _Invocation] = {}
        self.task_waiters: Dict[Id, Set[Callable[[Task], None]]] = {}
        self.input_waiters: Dict[Id, Set[Callable[[Input], None]]] = {}
        self.idle_waiters: Set[_IdleWaiter] = set()
        deps.session.listeners.add(self._on_commit)

    # -- commit fan-in -----------------------------------------------------

    def _on_commit(self, result: Any) -> None:
        changes = result.changes
        for task in changes.tasks:
            invocation = self.invocations.get(task.id)
            if task.abort is True and invocation is not None and invocation.mode == "run":
                invocation.controller.abort()
        for task in changes.tasks:
            if task.status == "terminal":
                for waiter in self.task_waiters.get(task.id, set()):
                    waiter(task)
        for input_record in changes.inputs:
            if input_record.status in ("done", "unanswered"):
                for waiter in self.input_waiters.get(input_record.id, set()):
                    waiter(input_record)
        if changes.tasks:
            self.kick()
        self.check_idle()

    # -- lifecycle ---------------------------------------------------------

    def resume(self) -> None:
        self.enabled = True
        self.kick()

    def stop(self) -> None:
        self.enabled = False

    def hold(self) -> Callable[[], None]:
        self.holds += 1
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            self.holds -= 1
            if self.holds == 0:
                self.kick()

        return release

    def kick(self) -> None:
        if not self.enabled:
            return
        self.dirty = True
        if self.holds == 0 and not self.draining:
            drain = asyncio.ensure_future(self._drain())
            drain.add_done_callback(self._report_drain)

    def _report_drain(self, future: asyncio.Future) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            self.deps.on_report(error)

    async def _drain(self) -> None:
        self.draining = True
        try:
            while self.dirty and self.enabled and self.holds == 0:
                self.dirty = False
                reserved = await self._reserve_eligible()
                if not self.enabled or self.holds > 0:
                    self.dirty = True
                    break
                for item in reserved:
                    self.dispatch(item.task, item.mode)
        finally:
            self.draining = False
            self.check_idle()

    async def _reserve_eligible(self) -> List[_Reserved]:
        """On the line: reserve every eligible task."""
        session = self.deps.session
        out: List[_Reserved] = []

        async def reserve(transaction: Any, line_ctx: Context, control: Any = None) -> None:
            for task in list(session.live_tasks.values()):
                if task.kind not in self.deps.kinds:
                    continue
                invocation = self.invocations.get(task.id)
                if task.abort is True:
                    if invocation is not None and invocation.mode in ("abort", "run"):
                        continue  # winding down; dispatched after it returns
                    running = task
                    if task.status == "pending":
                        running = _copy_task(task, status="running")
                        transaction.set_task(running)
                    out.append(_Reserved(task=running, mode="abort"))
                    continue
                if invocation is not None:
                    continue
                if task.status == "pending":
                    ready = True
                    for dependency in task.after:
                        stored = await transaction.task(dependency)
                        if stored is None or stored.status != "terminal":
                            ready = False
                            break
                    if not ready:
                        continue
                    running = _copy_task(task, status="running")
                    transaction.set_task(running)
                    out.append(_Reserved(task=running, mode="run"))
                elif task.status == "running":
                    out.append(_Reserved(task=task, mode="run"))

        await session.commit({"type": "kernel"}, reserve, self.deps.ctx)
        return out

    def dispatch(self, task: Task, mode: str) -> None:
        session = self.deps.session
        kind = self.deps.kinds.get(task.kind)
        if kind is None:
            self.deps.on_report(RuntimeError(f"unknown kind {task.kind} for task {task.id}"))
            return
        controller = AbortController()
        ctx = with_abort_signal(controller.signal, self.deps.ctx)
        done = asyncio.ensure_future(self._invoke(task, kind, mode, controller, ctx))
        self.invocations[task.id] = _Invocation(task=task, mode=mode, controller=controller, done=done)

    async def _invoke(
        self, task: Task, kind: AnyKind, mode: str, controller: AbortController, ctx: Context
    ) -> None:
        session = self.deps.session
        try:
            if mode == "run":
                closure = await self._run_phases(task, kind, ctx)
                if closure is None or controller.signal.aborted:
                    return
                lease = self._lease(task, kind, mode, ctx)
                try:

                    async def finish(transaction: Any, _line_ctx: Context, control: Any) -> None:
                        current = session.live_tasks.get(task.id)
                        if current is None or current.abort is True:
                            return
                        completion = validate_completion(
                            kind, await _resolve(closure(transaction, current, _line_ctx))
                        )
                        patched = await transaction.task(task.id)
                        control.set_task(
                            _copy_task(
                                patched or current,
                                status="terminal",
                                outcome=completion_to_outcome(completion),
                            )
                        )

                    await session.commit(lease["invoker"], finish, ctx, {"closing": True})
                finally:
                    lease["token"].revoke()
            else:
                handler_lease = self._lease(task, kind, mode, ctx)
                try:
                    closure = await _resolve(kind.abort(task, handler_lease["runtime"], ctx))
                finally:
                    handler_lease["token"].revoke()
                closure_lease = self._lease(task, kind, mode, ctx)
                try:

                    async def finish_abort(transaction: Any, _line_ctx: Context, control: Any) -> None:
                        current = session.live_tasks.get(task.id)
                        if current is None:
                            return
                        result = await _resolve(closure(transaction, current, _line_ctx))
                        json.dumps(result)
                        patched = await transaction.task(task.id)
                        control.set_task(
                            _copy_task(
                                patched or current,
                                status="terminal",
                                outcome=Outcome(status="aborted", result=result),
                            )
                        )

                    await session.commit(closure_lease["invoker"], finish_abort, ctx, {"closing": True})
                finally:
                    closure_lease["token"].revoke()
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001 - contract fault or unexpected throw
            if controller.signal.aborted:
                return
            self.deps.on_report(error)

            async def fault(transaction: Any, _line_ctx: Context, control: Any = None) -> None:
                current = session.live_tasks.get(task.id)
                if current is None:
                    return
                outcome = Outcome(
                    status="faulted",
                    error=(
                        str(error)
                        if isinstance(error, TaskContractFault)
                        else f"{kind.name}: {error}"
                    ),
                )
                transaction.set_task(_copy_task(current, status="terminal", outcome=outcome))
                if kind.turn:
                    inputs = current.input.get("inputs") if isinstance(current.input, dict) else None
                    if isinstance(inputs, list):
                        await transaction.resolve_inputs(
                            [value for value in inputs if isinstance(value, int) and not isinstance(value, bool)],
                            {"status": "unanswered", "reason": "failed", "detail": outcome.error},
                        )
                    transaction.sticky(current.conversation_id)["turn"] = {"tools": []}

            try:
                await session.commit(
                    {"type": "kernel"},
                    fault,
                    self.deps.ctx,
                    {"docs": [{"doc": "sticky", "conversationId": task.conversation_id}]},
                )
            except Exception as report_error:  # noqa: BLE001 - reported through on_report
                self.deps.on_report(report_error)
        finally:
            self.invocations.pop(task.id, None)
            now = session.live_tasks.get(task.id)
            if now is None:
                try:
                    await session.retire(task, self.deps.ctx)
                except Exception as error:  # noqa: BLE001 - reported through on_report
                    self.deps.on_report(error)
            self.kick()

    def _lease(self, task: Task, kind: AnyKind, mode: str, ctx: Context) -> Dict[str, Any]:
        token = InvocationToken(task.id, mode)
        invoker = Invoker(
            type="task",
            token=token,
            id=task.id,
            conversation_id=task.conversation_id,
            kind=kind,
            core=task.kind.startswith("pi.")
            and task.kind in ("pi.generation", "pi.tool", "pi.post_tools", "pi.collapse"),
            mode=mode,
        )
        return {
            "token": token,
            "invoker": invoker,
            "runtime": self.deps.runtime(task, invoker, ctx),
        }

    async def _run_phases(self, task: Task, kind: AnyKind, ctx: Context):
        """The phase loop. Each handler gets a lease revoked when that handler returns."""
        session = self.deps.session
        current = task
        while True:
            checkpoint = current.checkpoint
            phase = checkpoint.get("phase") if isinstance(checkpoint, dict) else None
            handler = handler_for(kind, phase)
            handler_lease = self._lease(task, kind, "run", ctx)
            try:
                step = validate_step(
                    kind, await _resolve(handler(current, handler_lease["runtime"], ctx))
                )
            finally:
                handler_lease["token"].revoke()
            if step.done is not None:
                return step.done
            if ctx.signal is not None and ctx.signal.aborted:
                return None
            transition_lease = self._lease(task, kind, "run", ctx)
            try:

                async def transition(transaction: Any, line_ctx: Context, control: Any) -> str:
                    nonlocal current
                    live = session.live_tasks.get(task.id)
                    if live is None or live.abort is True:
                        return "discard"
                    step_next = step.next
                    if callable(step_next):
                        nxt = await _resolve(step_next(transaction, live, line_ctx))
                    else:
                        nxt = step_next
                    if nxt == "retry":
                        return "retry"
                    if isinstance(nxt, dict) and "status" in nxt:
                        patched = await transaction.task(task.id)
                        control.set_task(
                            _copy_task(
                                patched or live,
                                status="terminal",
                                outcome=completion_to_outcome(validate_completion(kind, nxt)),
                            )
                        )
                        return "terminal"
                    checkpoint_value = validate_checkpoint(kind, nxt)
                    transaction.checkpoint(checkpoint_value)
                    current = _copy_task(live, checkpoint=checkpoint_value)
                    return "advanced"

                outcome = await session.commit(transition_lease["invoker"], transition, ctx)
            finally:
                transition_lease["token"].revoke()
            value = outcome.value if hasattr(outcome, "value") else outcome
            if value == "retry":
                current = session.live_tasks.get(task.id) or current
                continue
            if value != "advanced":
                return None

    # -- abort -------------------------------------------------------------

    async def abort_task(self, id: Id, ctx: Context) -> str:
        """Mark, revoke, signal, join. The next drain reserves the fresh abort."""
        session = self.deps.session

        async def mark(transaction: Any, _line_ctx: Context, control: Any = None) -> str:
            task = await transaction.task(id)
            if task is None:
                raise ValueError(f"task {id} not found")
            if task.status == "terminal":
                return "terminal"
            if task.abort is not True:
                transaction.set_task(_copy_task(task, abort=True))
            return "marked"

        result = await session.commit({"type": "kernel"}, mark, ctx)
        if (result.value if hasattr(result, "value") else result) == "terminal":
            return "terminal"
        invocation = self.invocations.get(id)
        if invocation is not None and invocation.mode == "run":
            invocation.controller.abort()
            try:
                await invocation.done
            except asyncio.CancelledError:  # pragma: no cover - join only
                pass
        self.kick()
        return "marked"

    # -- waiters: registered on the line, atomically with the state read ---

    async def wait_for_task(self, id: Id, ctx: Context) -> Task:
        async def terminal(line_ctx: Context) -> Optional[Task]:
            live = self.deps.session.live_tasks.get(id)
            if live is not None:
                return None
            return await self.deps.session.storage.task(id, line_ctx)

        return await self._waiter(ctx, self.task_waiters, id, terminal)

    async def wait_for_input(self, id: Id, ctx: Context) -> Input:
        async def terminal(line_ctx: Context) -> Optional[Input]:
            input_record = await self.deps.session.storage.input(id, line_ctx)
            if input_record is not None and input_record.status in ("done", "unanswered"):
                return input_record
            return None

        return await self._waiter(ctx, self.input_waiters, id, terminal)

    async def _waiter(
        self,
        ctx: Context,
        table: Dict[Id, Set[Callable[[Any], None]]],
        id: Id,
        terminal: Callable[[Context], Awaitable[Any]],
    ) -> Any:
        """Read state and install the waiter in one line operation; the wait itself is off-line."""
        if ctx.signal is not None:
            ctx.signal.throw_if_aborted()

        async def register(line_ctx: Context) -> Dict[str, Any]:
            now = await terminal(line_ctx)
            if now is not None:
                return {"now": now}
            return {"pending": self._pending(table, id, ctx)}

        result = await self.deps.session.on_line(register, ctx)
        return result["now"] if "now" in result else await result["pending"]

    def _pending(self, table: Dict[Id, Set[Callable[[Any], None]]], id: Id, ctx: Context) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        waiters = table.setdefault(id, set())

        def waiter(value: Any) -> None:
            waiters.discard(waiter)
            if not future.done():
                future.set_result(value)

        waiters.add(waiter)
        cancel_abort = _on_abort(ctx, lambda: _reject(future, ctx))
        future.add_done_callback(lambda _f: cancel_abort())
        return future

    async def wait_for_idle(self, conversation_id: Optional[Id], ctx: Context) -> None:
        if ctx.signal is not None:
            ctx.signal.throw_if_aborted()

        def register(_line_ctx: Context) -> Dict[str, Any]:
            if self.is_idle(conversation_id):
                return {}
            return {"pending": self._idle_pending(conversation_id, ctx)}

        result = await self.deps.session.on_line(register, ctx)
        if "pending" in result:
            await result["pending"]

    def _idle_pending(self, conversation_id: Optional[Id], ctx: Context) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def resolve() -> None:
            self.idle_waiters.discard(waiter)
            if not future.done():
                future.set_result(None)

        waiter = _IdleWaiter(conversation_id=conversation_id, resolve=resolve)
        self.idle_waiters.add(waiter)
        cancel_abort = _on_abort(ctx, lambda: _reject(future, ctx))
        future.add_done_callback(lambda _f: cancel_abort())
        return future

    def is_idle(self, conversation_id: Optional[Id]) -> bool:
        for task in self.deps.session.live_tasks.values():
            if conversation_id is not None and task.conversation_id != conversation_id:
                continue
            if not task.background:
                return False
        return True

    def check_idle(self) -> None:
        for waiter in list(self.idle_waiters):
            if self.is_idle(waiter.conversation_id) and not self.draining:
                waiter.resolve()

    # -- shutdown ----------------------------------------------------------

    async def join_all(self) -> None:
        """Signal every invocation and wait for them; nothing is written."""
        self.enabled = False
        for invocation in self.invocations.values():
            invocation.controller.abort()
        await asyncio.gather(
            *[invocation.done for invocation in self.invocations.values()], return_exceptions=True
        )

    def quiescent(self) -> bool:
        return len(self.invocations) == 0

    @property
    def live_invocations(self) -> int:
        return len(self.invocations)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _resolve(value: Any) -> Any:
    """Await a value that may be a plain result or an awaitable, like ``T | Promise<T>``."""
    if hasattr(value, "__await__"):
        return await value
    return value


def _copy_task(task: Task, **changes: Any) -> Task:
    """``{...task, ...changes}``: a fresh record, as the TS spread builds."""
    from .types import task_from_json, task_to_json

    copied = task_from_json(task_to_json(task))
    for key, value in changes.items():
        setattr(copied, key, value)
    return copied


def _reject(future: Any, ctx: Context) -> None:
    if future.done():
        return
    reason = ctx.signal.reason if ctx.signal is not None else None
    future.set_exception(reason if isinstance(reason, BaseException) else RuntimeError("aborted"))


def _on_abort(ctx: Context, callback: Callable[[], None]) -> Callable[[], None]:
    """Run ``callback`` when the context aborts; the returned call cancels the watch."""
    if ctx.signal is None:
        return lambda: None
    if ctx.signal.aborted:
        callback()
        return lambda: None
    watch = asyncio.ensure_future(ctx.signal.wait())

    def fired(future: asyncio.Future) -> None:
        if future.cancelled():
            return
        callback()

    watch.add_done_callback(fired)

    def cancel() -> None:
        watch.cancel()

    return cancel


_ = (Checkpoint, Faulted, JsonValue, field)
