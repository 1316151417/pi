"""Runtime lane ported from ``harness/runtime/lane.ts``.

One configured lane: the serialized mutation line everything durable about a
conversation goes through, the operation admission and cancellation protocol,
the drive claim loop, and the configuration/queue accessors the harness API
exposes.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..._chord.context import Context, await_with_context
from pi_ai.types import AgentMessage, ImageContent, Usage
from ..compaction.branch_summarization import BranchPreparation, prepare_branch_entries
from ..compaction.compaction import prepare_compaction
from ..execution.tools import tool_result_from_message
from ..hooks import HookRegistry
from ..prompt_templates import format_prompt_template_invocation
from ..result import (
    Closed,
    HarnessClosed,
    HarnessFault,
    InvalidMessage,
    InvalidNavigation,
    LaneBusy,
    NoActiveOperation,
    NothingToCompact,
    NothingToResume,
    OperationMismatch,
    UnknownSkill,
    UnknownTarget,
    UnknownTemplate,
    err,
    ok,
)
from ..session.commit import insert_entry, insert_usage
from ..session.session import SessionInvariantError, SessionPendingAssistantMessageError
from ..session.types import (
    InboxItem,
    MessageEntry,
    NavigationReadyToCommitOperation,
    PendingEntry,
    Operation,
    OperationIntent,
    OperationMeta,
    RunSettings,
    StartingOperation,
    SummaryDecidingOperation,
)
from ..session.values import (
    branch_tip,
    delete_value,
    lane_config,
    operation_meta,
    operation_preparation,
    operation_result,
    operation_state,
    operation_tool_args,
    pending_entry,
    pending_tool_output,
    set_value,
)
from ..session.values import lane_state as lane_state_value
from ..skills import format_skill_invocation
from .drive.structural import durable_branch_preparation, durable_compaction_preparation
from .progress import read_assistant_frames
from .transcript import chain_entries, committed_entry_events, read_lane_queues
from .types import (
    CommitDecision,
    Config,
    ContinueOperationResult,
    Drive,
    FinishDecision,
    LaneReject,
    LaneReturn,
    LaneState,
    OperationCommand,
)

__all__ = [
    "Lane",
    "LaneCommandOutcome",
    "IdleObservation",
    "IdleClaimObservation",
    "DriveClaim",
    "captured_settings",
    "durable_lane_state",
    "pending_entry_write",
    "captured_model",
    "inbox_items",
    "without_inbox_items",
    "select_accepted_inbox",
    "is_promise_like",
]


EmitBatch = Callable[[List[Any], Context], Awaitable[None]]
FaultHandler = Callable[[Any, Context], BaseException]


@dataclass
class LaneCommandOutcome:
    """Result of one command callback inside the session line."""

    kind: str = "return"  # "return" | "reject" | "idle_blocked"
    result: Any = None
    error: Optional[BaseException] = None
    delivery: Optional[Awaitable[None]] = None
    owner: Optional["asyncio.Future"] = None
    change: Optional["asyncio.Future"] = None


@dataclass
class IdleObservation:
    kind: str = "idle"  # "idle" | "wait"
    drive: Optional[Drive] = None
    change: Optional["asyncio.Future"] = None


@dataclass
class IdleClaimObservation:
    kind: str = "claimed"  # "claimed" | "wait"
    drive: Optional[Drive] = None
    change: Optional["asyncio.Future"] = None


@dataclass
class DriveClaim:
    kind: str = "observe"  # "observe" | "occupied" | "settled" | "mismatch"
    drive: Optional[Drive] = None
    installed: bool = False
    outcome: Any = None
    error: Optional[BaseException] = None


def is_promise_like(value: Any) -> bool:
    """Whether a materializer returned something awaitable, which is forbidden."""
    return asyncio.iscoroutine(value) or asyncio.isfuture(value)


def inbox_items(inbox: List[InboxItem], kind: str) -> List[InboxItem]:
    return [item for item in inbox if item.kind == kind]


def without_inbox_items(inbox: List[InboxItem], removed: List[InboxItem]) -> List[InboxItem]:
    removed_ids = {item.entry_id for item in removed}
    return [item for item in inbox if item.entry_id not in removed_ids]


def select_accepted_inbox(
    inbox: List[InboxItem],
    steering_mode: str,
    follow_up_mode: str,
) -> Dict[str, List[InboxItem]]:
    """Split the inbox into the items this acceptance consumes and the remainder."""
    steer_taken = False
    follow_up_taken = False
    selected: List[InboxItem] = []
    remainder: List[InboxItem] = []
    for item in inbox:
        eligible = (
            item.kind in ("write", "nextRun")
            or (item.kind == "steer" and (steering_mode == "all" or not steer_taken))
            or (item.kind == "followUp" and (follow_up_mode == "all" or not follow_up_taken))
        )
        if eligible:
            selected.append(item)
            if item.kind == "steer":
                steer_taken = True
            if item.kind == "followUp":
                follow_up_taken = True
        else:
            remainder.append(item)
    return {"selected": selected, "remainder": remainder}


def captured_settings(config: Config) -> RunSettings:
    """Snapshot the run settings one acceptance captures."""
    return RunSettings(
        compaction=config.compaction,
        steering_mode=config.steering_mode,
        follow_up_mode=config.follow_up_mode,
        tool_execution=config.tool_execution,
    )


def durable_lane_state(
    state: LaneState,
    current_operation_id: Optional[str],
    inbox: Optional[List[InboxItem]] = None,
    last_operation_id: Optional[str] = None,
) -> dict:
    """The durable ``lane.state`` value, as a plain mapping for the value store."""
    return {
        "currentOperationId": current_operation_id,
        "lastOperationId": state.last_operation_id if last_operation_id is None else last_operation_id,
        "inbox": list(state.inbox if inbox is None else inbox),
    }


def pending_entry_write(entry_id: str, pending: Any) -> Any:
    """Materialize one pending inbox payload into the entry it will commit."""
    from ..session.types import CustomEntry, MessageEntry

    payload_type = getattr(pending, "type", None)
    payload = getattr(pending, "payload", None)
    custom_type = getattr(pending, "custom_type", None)
    if isinstance(pending, dict):
        payload_type = pending.get("type")
        payload = pending.get("payload")
        custom_type = pending.get("customType", pending.get("custom_type"))
    if payload_type == "message":
        return MessageEntry(id=entry_id, parent_id=None, message=payload)
    return CustomEntry(id=entry_id, parent_id=None, custom_type=custom_type or "", data=payload)


def captured_model(operation: Operation) -> Optional[dict]:
    """The model identity one in-flight operation captured, when it has one."""
    state = operation.state
    at = getattr(state, "at", None)
    if at in ("assistant.ready", "assistant.effect_pending", "assistant.retry_wait"):
        return state.generation_context.configuration.model
    if at == "tools":
        return state.batch.configuration.model
    if at in ("deferred.suspended", "deferred.effect_pending"):
        return state.configuration.model
    if at in ("summary.ready", "summary.effect_pending", "summary.retry_wait"):
        return state.summary_context.configuration.model
    return None


class Lane:
    """Runtime implementation of one configured lane."""

    def __init__(
        self,
        name: str,
        session: Any,
        models: Any,
        hooks: HookRegistry,
        state: LaneState,
        on_fault: FaultHandler,
        emit_batch: EmitBatch,
        install_watch: Callable[..., Any],
        read_config: Callable[[], Config],
    ) -> None:
        self.session = session
        self.models = models
        self.hooks = hooks
        self.name = name
        self.state = state
        self._on_fault = on_fault
        self.emit_batch = emit_batch
        self._install_watch = install_watch
        self._config = read_config
        self._state_change: Optional["asyncio.Future"] = None
        self.idle_owner: Optional["asyncio.Future"] = None
        #: Package-internal drive owner. Public only because deterministic procedure
        #: tests install exact owners directly.
        self.active_drive: Optional[Drive] = None
        #: Authoritative live control projection while this harness owns the Session.
        self.closed_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # State change signalling
    # ------------------------------------------------------------------

    @property
    def state_change(self) -> "asyncio.Future":
        """A future that resolves on the next published state change."""
        if self._state_change is None:
            self._state_change = asyncio.get_running_loop().create_future()
        return self._state_change

    def signal_state_change(self) -> None:
        future = self._state_change
        self._state_change = None
        if future is not None and not future.done():
            future.set_result(None)

    def assert_open(self) -> None:
        if self.closed_error is not None:
            raise self.closed_error

    # ------------------------------------------------------------------
    # Read-only and serialized commands
    # ------------------------------------------------------------------

    async def get_tip_id(self, _context: Context) -> Optional[str]:
        self.assert_open()
        return self.state.tip_id

    async def get_result(self, operation_id: str, context: Context) -> Any:
        self.assert_open()
        stored = await self.session.get_value(operation_result(operation_id), context)
        return None if stored is None else stored.value

    def read_config(self) -> Config:
        return self._config()

    def mismatch(
        self,
        expected: str,
        current_operation_id: Optional[str],
        last_operation_id: Optional[str],
    ) -> OperationMismatch:
        return OperationMismatch(
            lane=self.name,
            expected_operation_id=expected,
            message=f"Operation {expected} does not own lane {self.name!r}",
            current_operation_id=current_operation_id,
            last_operation_id=last_operation_id,
        )

    async def read_lane(
        self,
        read: Callable[[LaneState, Any], Any],
        context: Context,
    ) -> Any:
        """Run one effect-free read on this lane's serialized mutation line."""
        self.assert_open()

        async def _mutation(reader: Any, _context: Any) -> Any:
            self.assert_open()
            try:
                value = read(self.state, reader)
                if asyncio.iscoroutine(value):
                    value = await value
                return value
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                if self.closed_error is not None:
                    raise self.closed_error
                raise self._on_fault(error, context)

        try:
            return await self.session.mutate(_mutation, context)
        except asyncio.CancelledError:
            raise
        except BaseException:
            if self.closed_error is not None:
                raise self.closed_error
            raise

    async def command(
        self,
        plan: Callable[[LaneState, Any], Any],
        context: Context,
    ) -> Any:
        """Run one effect-free command on this lane's serialized mutation line."""
        self.assert_open()
        while self.idle_owner is not None:
            await _race(self.idle_owner, self.state_change)
            self.assert_open()

        outcome: LaneCommandOutcome

        async def _mutation(mutator: Any, _context: Any) -> LaneCommandOutcome:
            self.assert_open()
            if self.idle_owner is not None:
                return LaneCommandOutcome(
                    kind="idle_blocked", owner=self.idle_owner, change=self.state_change
                )
            try:
                decision = plan(self.state, mutator)
                if asyncio.iscoroutine(decision):
                    decision = await decision
                kind = getattr(decision, "kind", None)
                if kind == "return":
                    return LaneCommandOutcome(kind="return", result=decision.result)
                if kind == "reject":
                    return LaneCommandOutcome(kind="reject", error=decision.error)
                if kind != "commit":
                    raise TypeError(f"Unknown lane command kind: {kind!r}")
                commit = await mutator.commit(decision.writes, context)
                self.state = decision.next
                self.signal_state_change()
                if decision.materialize is None:
                    raise TypeError("Lane command commit requires a materialize() callback")
                result = decision.materialize(commit)
                if is_promise_like(result):
                    raise TypeError("Lane command materialize() must be synchronous")
                events = decision.events(commit) if decision.events is not None else []
                delivery = None if not events else self.emit_batch(events, context)
                return LaneCommandOutcome(kind="return", result=result, delivery=delivery)
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                if self.closed_error is not None:
                    raise self.closed_error
                raise self._on_fault(error, context)

        try:
            outcome = await self.session.mutate(_mutation, context)
        except asyncio.CancelledError:
            raise
        except BaseException:
            if self.closed_error is not None:
                raise self.closed_error
            raise
        if outcome.kind == "idle_blocked":
            await _race(outcome.owner, outcome.change)
            self.assert_open()
            return await self.command(plan, context)
        if outcome.kind == "reject":
            raise outcome.error
        if outcome.delivery is not None:
            await await_with_context(context, outcome.delivery)
        return outcome.result

    async def settle_operation(
        self,
        _capability: Any,
        plan: Callable[[LaneState, Any, OperationMeta, Any], Any],
        context: Context,
    ) -> Any:
        """Run a command against the current operation even after cancellation."""

        async def _plan(state: LaneState, reader: Any) -> Any:
            operation = state.operation
            decision = plan(state, operation.state, operation.meta, reader)
            if asyncio.iscoroutine(decision):
                decision = await decision
            if getattr(decision, "kind", None) == "commit":
                writes = list(decision.writes)
                writes.append(set_value(operation_state(operation.meta.operation_id), decision.operation_state))
                lane_patch = decision.lane or {}
                if lane_patch.get("inbox") is not None:
                    writes.append(
                        set_value(
                            lane_state_value(self.name),
                            durable_lane_state(state, operation.meta.operation_id, lane_patch["inbox"]),
                        )
                    )
                return CommitDecision(
                    writes=writes,
                    next=LaneState(
                        tip_id=lane_patch.get("tipId", state.tip_id),
                        configuration=lane_patch.get("configuration", state.configuration),
                        inbox=lane_patch.get("inbox", state.inbox),
                        last_operation_id=state.last_operation_id,
                        operation=Operation(meta=operation.meta, state=decision.operation_state),
                    ),
                    materialize=decision.materialize,
                    events=decision.events,
                )
            if getattr(decision, "kind", None) != "finish":
                return decision
            lane_patch = decision.lane or {}
            inbox = lane_patch.get("inbox", state.inbox)
            return CommitDecision(
                writes=[
                    *decision.writes,
                    set_value(operation_result(operation.meta.operation_id), decision.record),
                    set_value(
                        lane_state_value(self.name),
                        durable_lane_state(state, None, inbox, operation.meta.operation_id),
                    ),
                ],
                next=LaneState(
                    tip_id=lane_patch.get("tipId", state.tip_id),
                    configuration=lane_patch.get("configuration", state.configuration),
                    inbox=inbox,
                    last_operation_id=operation.meta.operation_id,
                    operation=None,
                ),
                materialize=decision.materialize,
                events=decision.events,
            )

        return await self.command(_plan, context)

    async def continue_operation(
        self,
        capability: Any,
        plan: Callable[[LaneState, Any, OperationMeta, Any], Any],
        context: Context,
    ) -> ContinueOperationResult:
        """Run an ordinary operation command only while durable control is running."""

        async def _plan(state: LaneState, current: Any, meta: OperationMeta, reader: Any) -> Any:
            if current.control.status == "cancel_requested":
                return LaneReturn(result=ContinueOperationResult(kind="cancel_requested"))
            decision = plan(state, current, meta, reader)
            if asyncio.iscoroutine(decision):
                decision = await decision
            if getattr(decision, "kind", None) == "return":
                return LaneReturn(result=ContinueOperationResult(kind="result", value=decision.result))
            original = decision.materialize

            def _materialize(commit: Any) -> Any:
                return ContinueOperationResult(kind="result", value=original(commit))

            decision.materialize = _materialize
            return decision

        return await self.settle_operation(capability, _plan, context)

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    async def accept(self, request: Any, context: Context) -> Any:
        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()
        started_at = _now()
        operation_id = _opt(request, "operation_id", "operationId") or self.session.id_generator.next(started_at)
        acceptance_config = self.read_config()
        if _opt(request, "kind") == "compaction":
            return await self._accept_compaction(request, operation_id, started_at, acceptance_config, context)
        if _opt(request, "kind") == "navigation":
            return await self._accept_navigation(
                request, operation_id, started_at, acceptance_config, context
            )
        return await self._accept_run(request, operation_id, started_at, acceptance_config, context)

    async def _accept_run(
        self,
        request: Any,
        operation_id: str,
        started_at: int,
        acceptance_config: Config,
        context: Context,
    ) -> Any:
        messages: List[Any]
        if _opt(request, "kind") == "prompt":
            prompt = _opt(request, "prompt")
            if isinstance(prompt, str):
                images = _opt(request, "images") or []
                messages = (
                    []
                    if len(prompt) == 0 and not images
                    else [
                        _user_message(
                            _text_blocks(prompt),
                            images,
                            started_at,
                        )
                    ]
                )
            else:
                messages = list(prompt) if isinstance(prompt, (list, tuple)) else [prompt]
        elif _opt(request, "kind") == "skill":
            resources = acceptance_config.resources
            skills = getattr(resources, "skills", None) if resources is not None else None
            request_name = _opt(request, "name")
            skill = _find_named(skills, request_name)
            if skill is None:
                return err(
                    UnknownSkill(name=request_name, message=f"Unknown skill: {request_name}")
                )
            messages = [
                _user_message(
                    [
                        {
                            "type": "text",
                            "text": format_skill_invocation(
                                skill, _opt(request, "additional_instructions", "additionalInstructions")
                            ),
                        }
                    ],
                    [],
                    started_at,
                )
            ]
        else:
            resources = acceptance_config.resources
            templates = getattr(resources, "prompt_templates", None) if resources is not None else None
            template = _find_named(templates, request_name)
            if template is None:
                return err(
                    UnknownTemplate(
                        name=request_name, message=f"Unknown prompt template: {request_name}"
                    )
                )
            content = format_prompt_template_invocation(template, _opt(request, "args"))
            messages = (
                [] if len(content) == 0 else [_user_message([content], [], started_at)]
            )

        for message in messages:
            if getattr(message, "role", None) == "assistant" and getattr(message, "stop_reason", None) == "pending":
                return err(
                    InvalidMessage(
                        lane=self.name,
                        reason="pending_assistant",
                        message="Cannot accept a pending assistant message",
                    )
                )
        prompt = [
            (self.session.id_generator.next(started_at), message) for message in messages
        ]

        async def _plan(state: LaneState, reader: Any) -> Any:
            if state.operation is not None:
                return LaneReturn(
                    result=err(
                        LaneBusy(
                            lane=self.name,
                            operation_id=state.operation.meta.operation_id,
                            operation_kind=state.operation.meta.intent.kind,
                            message=f"Lane {self.name!r} already has an active operation",
                        )
                    )
                )
            split = select_accepted_inbox(
                state.inbox, acceptance_config.steering_mode, acceptance_config.follow_up_mode
            )
            selected_items = split["selected"]
            inbox = split["remainder"]
            captured = []
            for item in selected_items:
                stored = await reader.get_value(pending_entry(item.entry_id), context)
                if stored is None:
                    raise SessionInvariantError(
                        f"Pending {item.kind} entry {item.entry_id} is missing its payload"
                    )
                pending = PendingEntry.from_value(stored.value)
                if item.kind != "write" and pending.type != "message":
                    raise SessionInvariantError(
                        f"Pending {item.kind} entry {item.entry_id} is not a message"
                    )
                payload = pending.payload
                if (
                    pending.type == "message"
                    and getattr(payload, "role", None) == "assistant"
                    and getattr(payload, "stop_reason", None) == "pending"
                ):
                    raise SessionInvariantError(
                        f"Pending {item.kind} entry {item.entry_id} contains a pending assistant"
                    )
                captured.append((item, stored.value))
            has_captured_conversation = any(item.kind != "write" for item in selected_items)
            if not prompt and not has_captured_conversation:
                return LaneReturn(
                    result=err(
                        InvalidMessage(
                            lane=self.name,
                            reason="empty",
                            message="Acceptance must append at least one message",
                        )
                    )
                )

            entries = chain_entries(
                state.tip_id,
                [
                    *[pending_entry_write(item.entry_id, pending) for item, pending in captured],
                    *[
                        MessageEntry(id=entry_id, parent_id=None, message=message)
                        for entry_id, message in prompt
                    ],
                ],
            )
            parent_id = entries[-1].id
            meta = OperationMeta(
                operation_id=operation_id,
                lane=self.name,
                source_tip_id=state.tip_id,
                started_at=started_at,
                intent=OperationIntent(
                    kind="run", prompt_entry_ids=[entry_id for entry_id, _ in prompt]
                ),
            )
            operation_state_value_obj = StartingOperation(
                settings=captured_settings(acceptance_config),
                latest_assistant_entry_id=None,
            )
            remaining_queues = await read_lane_queues(reader, inbox, context)

            def _events(commit: Any) -> List[Any]:
                events = [
                    {"type": "run_start", "runId": operation_id, "startedAt": started_at, "lane": self.name},
                    *committed_entry_events(entries, commit, self.name, operation_id),
                ]
                if selected_items:
                    events.append({"type": "queue_update", "queues": remaining_queues, "lane": self.name})
                return events

            return CommitDecision(
                writes=[
                    *[insert_entry(entry) for entry in entries],
                    *[delete_value(pending_entry(item.entry_id)) for item in selected_items],
                    set_value(branch_tip(self.name), parent_id),
                    set_value(operation_meta(operation_id), meta),
                    set_value(operation_state(operation_id), operation_state_value_obj),
                    set_value(
                        lane_state_value(self.name), durable_lane_state(state, operation_id, inbox)
                    ),
                ],
                next=LaneState(
                    tip_id=parent_id,
                    configuration=state.configuration,
                    inbox=inbox,
                    last_operation_id=state.last_operation_id,
                    operation=Operation(meta=meta, state=operation_state_value_obj),
                ),
                materialize=lambda _commit: ok(
                    {"operationId": operation_id, "kind": "run", "startedAt": started_at}
                ),
                events=_events,
            )

        return await self.command(_plan, context)

    async def _accept_compaction(
        self,
        request: Any,
        operation_id: str,
        started_at: int,
        acceptance_config: Config,
        context: Context,
    ) -> Any:
        task_id = self.session.id_generator.next(started_at)

        async def _plan(state: LaneState, reader: Any) -> Any:
            if state.operation is not None:
                return LaneReturn(
                    result=err(
                        LaneBusy(
                            lane=self.name,
                            operation_id=state.operation.meta.operation_id,
                            operation_kind=state.operation.meta.intent.kind,
                            message=f"Lane {self.name!r} already has an active operation",
                        )
                    )
                )
            path: List[Any] = []
            if state.tip_id is not None:
                path = await reader.scan_branch(
                    _branch_scan(state.tip_id, stop_at_type="compaction"), context
                )
                path.reverse()
            prepared = prepare_compaction(path, acceptance_config.compaction)
            if not prepared.ok:
                raise prepared.error
            if prepared.value is None:
                return LaneReturn(
                    result=err(
                        NothingToCompact(
                            lane=self.name,
                            message=f"Lane {self.name!r} has nothing to compact",
                        )
                    )
                )
            custom_instructions = _opt(request, "custom_instructions", "customInstructions")
            meta = OperationMeta(
                operation_id=operation_id,
                lane=self.name,
                source_tip_id=state.tip_id,
                started_at=started_at,
                intent=OperationIntent(
                    kind="compaction", custom_instructions=custom_instructions
                ),
            )
            operation_state_value_obj = SummaryDecidingOperation(
                settings=captured_settings(acceptance_config),
                latest_assistant_entry_id=None,
                task=_summary_task(task_id, "manual", custom_instructions, {"kind": "finish"}),
            )
            return CommitDecision(
                writes=[
                    set_value(
                        operation_preparation(operation_id, task_id),
                        durable_compaction_preparation(prepared.value),
                    ),
                    set_value(operation_meta(operation_id), meta),
                    set_value(operation_state(operation_id), operation_state_value_obj),
                    set_value(lane_state_value(self.name), durable_lane_state(state, operation_id)),
                ],
                next=LaneState(
                    tip_id=state.tip_id,
                    configuration=state.configuration,
                    inbox=state.inbox,
                    last_operation_id=state.last_operation_id,
                    operation=Operation(meta=meta, state=operation_state_value_obj),
                ),
                materialize=lambda _commit: ok(
                    {"operationId": operation_id, "kind": "compaction", "startedAt": started_at}
                ),
                events=lambda _commit: [
                    {
                        "type": "compaction_start",
                        "lane": self.name,
                        "runId": operation_id,
                        "reason": "manual",
                        "startedAt": started_at,
                    }
                ],
            )

        return await self.command(_plan, context)

    async def _accept_navigation(
        self,
        request: Any,
        operation_id: str,
        started_at: int,
        acceptance_config: Config,
        context: Context,
    ) -> Any:
        task_id = self.session.id_generator.next(started_at)
        target_id = _opt(request, "target_id", "targetId")
        target_options = _opt(request, "options")
        options = target_options if target_options is not None else {}
        summarize = _opt(options, "summarize") or False
        while True:
            observed_tip_id = self.state.tip_id
            preparation: Optional[BranchPreparation] = None
            if summarize and observed_tip_id is not None and target_id is not None:
                target = await self.session.get_entries([target_id], context)
                if target_id in target:
                    old_path, target_path = await asyncio.gather(
                        self.session.scan_branch(_branch_scan(observed_tip_id), context),
                        self.session.scan_branch(_branch_scan(target_id), context),
                    )
                    old_ids = {entry.id for entry in old_path}
                    common_ancestor_id = next(
                        (entry.id for entry in target_path if entry.id in old_ids), None
                    )
                    cut = (
                        len(old_path)
                        if common_ancestor_id is None
                        else _index_of(old_path, common_ancestor_id)
                    )
                    preparation = prepare_branch_entries(list(reversed(old_path[:cut])))
            accepted = await self.command(
                _navigation_plan(
                    self,
                    context,
                    operation_id=operation_id,
                    target_id=target_id,
                    options=options,
                    summarize=summarize,
                    preparation=preparation,
                    observed_tip_id=observed_tip_id,
                    started_at=started_at,
                    acceptance_config=acceptance_config,
                    task_id=task_id,
                ),
                context,
            )
            if accepted is not None:
                return accepted

    # ------------------------------------------------------------------
    # Driving
    # ------------------------------------------------------------------

    async def drive(self, options: Any, context: Context) -> Any:
        from .drive import drive_operation

        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()

        while True:
            claim = await self.command(_drive_claim_plan(self, options, context), context)
            if claim.kind == "settled":
                return ok({"kind": "settled", "outcome": claim.outcome})
            if claim.kind == "mismatch":
                return err(claim.error)
            if claim.kind == "occupied":
                await await_with_context(context, claim.drive.completion)
                continue
            drive = claim.drive
            if claim.installed:
                _install_drive(self, drive, context, drive_operation)
            return ok(await await_with_context(context, drive.completion))

    async def request_operation_abort(self, operation_id: str, context: Context) -> Any:
        """Package-private durable cancellation primitive."""
        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()

        drive = self.active_drive if self.active_drive is not None and self.active_drive.operation_id == operation_id else None
        cancellation: "asyncio.Future" = asyncio.get_running_loop().create_future()
        cancellation.add_done_callback(_swallow_future)
        if drive is not None:
            drive.begin_abort(cancellation)
        gate_settled = False

        def settle_gate(signal: bool) -> None:
            nonlocal gate_settled
            if gate_settled:
                return
            gate_settled = True
            if not cancellation.done():
                cancellation.set_result(None)
            if signal and drive is not None:
                drive.signal_abort()

        try:
            result = await self.command(
                _abort_plan(self, context, operation_id, settle_gate), context
            )
            # A fresh Drive may observe an already-durable marker; pull its gate on the
            # repeat path too. A mismatch can leave only the stale Drive's admission gate
            # in aborting state; its cancellation wait is still released.
            settle_gate(bool(getattr(result, "ok", False)) and result.value["newlyRequested"] is False)
            return result
        except BaseException as error:
            if not gate_settled and not cancellation.done():
                cancellation.set_exception(error)
                _swallow_future(cancellation)
            raise

    async def request_abort(self, operation_id: str, context: Context) -> Any:
        return await self.request_operation_abort(operation_id, context)

    async def inspect_execution(self, context: Context) -> Any:
        def _read(state: LaneState, _reader: Any) -> Any:
            operation = state.operation
            captured = None if operation is None else captured_model(operation)
            current = None
            if operation is not None:
                current = {
                    "id": operation.meta.operation_id,
                    "kind": operation.meta.intent.kind,
                    "status": "aborting"
                    if operation.state.control.status == "cancel_requested"
                    else "open",
                    "startedAt": operation.meta.started_at,
                }
                if captured is not None:
                    current["capturedModel"] = captured
            return {
                "lane": self.name,
                "tipId": state.tip_id,
                "configuredModel": state.configuration.model,
                "current": current,
                "lastOperationId": state.last_operation_id,
            }

        return await self.read_lane(_read, context)

    async def prompt(self, *args: Any) -> Any:
        if len(args) == 3:
            message, images, context = args
            request = {"kind": "prompt", "prompt": message}
            if images is not None:
                request["images"] = images
            return await self.drive_run_request(_dict(request), context)
        message, context = args
        return await self.drive_run_request(_dict({"kind": "prompt", "prompt": message}), context)

    async def skill(self, name: str, additional_instructions: Optional[str], context: Context) -> Any:
        request = {"kind": "skill", "name": name}
        if additional_instructions is not None:
            request["additionalInstructions"] = additional_instructions
        return await self.drive_run_request(_dict(request), context)

    async def prompt_from_template(self, name: str, args: Optional[List[str]], context: Context) -> Any:
        request = {"kind": "prompt_template", "name": name}
        if args is not None:
            request["args"] = args
        return await self.drive_run_request(_dict(request), context)

    async def drive_run_request(self, request: Any, context: Context) -> Any:
        admission = await self.accept(request, context)
        if not admission.ok:
            tag = getattr(admission.error, "tag", None)
            if tag in ("LaneBusy", "InvalidMessage", "UnknownSkill", "UnknownTemplate", "Closed"):
                return err(admission.error)
            raise self._on_fault(
                SessionInvariantError(f"Run acceptance returned {tag}"), context
            )
        driven = await self.drive(
            _dict({"operationId": admission.value["operationId"], "waitForRetry": True}), context
        )
        if not driven.ok:
            if getattr(driven.error, "tag", None) == "Closed":
                return err(driven.error)
            raise self._on_fault(
                SessionInvariantError(
                    f"Accepted run {admission.value['operationId']} no longer matches its lane"
                ),
                context,
            )
        if driven.value["kind"] == "settled":
            return ok(driven.value["outcome"])
        if driven.value["reason"] == "deferred":
            return ok(
                {
                    "operationId": admission.value["operationId"],
                    "status": "suspended",
                    "deferred": driven.value["deferred"],
                }
            )
        raise self._on_fault(
            SessionInvariantError(
                f"Run {admission.value['operationId']} returned an unwaited retry"
            ),
            context,
        )

    async def compact(self, options: Any, context: Context) -> Any:
        custom = _opt(options, "custom_instructions") if options is not None else None
        request = {"kind": "compaction"}
        if custom is not None:
            request["customInstructions"] = custom
        admission = await self.accept(_dict(request), context)
        if not admission.ok:
            tag = getattr(admission.error, "tag", None)
            if tag in ("LaneBusy", "NothingToCompact", "Closed"):
                return err(admission.error)
            raise self._on_fault(
                SessionInvariantError(f"Compaction acceptance returned {tag}"), context
            )
        compacted = await self.drive_structural_admission(admission.value, context)
        if not compacted.ok:
            return compacted
        continuation = await self.continue_after_structural(compacted.value, context)
        if not continuation.ok:
            return continuation
        result = {"compaction": compacted.value}
        if continuation.value is not None:
            result["run"] = continuation.value
        return ok(result)

    async def navigate_tree(self, target_id: Optional[str], options: Any, context: Context) -> Any:
        request = {"kind": "navigation", "targetId": target_id}
        if options is not None:
            request["options"] = options
        admission = await self.accept(_dict(request), context)
        if not admission.ok:
            tag = getattr(admission.error, "tag", None)
            if tag in ("LaneBusy", "InvalidNavigation", "UnknownTarget", "Closed"):
                return err(admission.error)
            raise self._on_fault(
                SessionInvariantError(f"Navigation acceptance returned {tag}"), context
            )
        navigated = await self.drive_structural_admission(admission.value, context)
        if not navigated.ok:
            return navigated
        continuation = await self.continue_after_structural(navigated.value, context)
        if not continuation.ok:
            return continuation
        result = {"navigation": navigated.value}
        if continuation.value is not None:
            result["run"] = continuation.value
        return ok(result)

    async def drive_structural_admission(self, admission: Any, context: Context) -> Any:
        operation_id = admission["operationId"]
        driven = await self.drive(
            _dict({"operationId": operation_id, "waitForRetry": True}), context
        )
        if not driven.ok:
            if getattr(driven.error, "tag", None) == "Closed":
                return err(driven.error)
            raise self._on_fault(
                SessionInvariantError(
                    f"Accepted {admission['kind']} {operation_id} no longer matches its lane"
                ),
                context,
            )
        if driven.value["kind"] == "settled":
            return ok(driven.value["outcome"])
        raise self._on_fault(
            SessionInvariantError(
                f"{admission['kind']} {operation_id} returned {driven.value['reason']}"
            ),
            context,
        )

    async def continue_after_structural(self, record: Any, context: Context) -> Any:
        if record.status == "aborted":
            return ok(None)
        admission = await self.accept(_dict({"kind": "prompt", "prompt": ""}), context)
        if not admission.ok:
            tag = getattr(admission.error, "tag", None)
            if tag in ("InvalidMessage", "LaneBusy"):
                return ok(None)
            if tag == "Closed":
                return err(admission.error)
            raise self._on_fault(
                SessionInvariantError(f"Structural continuation acceptance returned {tag}"), context
            )
        operation_id = admission.value["operationId"]
        driven = await self.drive(
            _dict({"operationId": operation_id, "waitForRetry": True}), context
        )
        if not driven.ok:
            if getattr(driven.error, "tag", None) == "Closed":
                return err(driven.error)
            raise self._on_fault(
                SessionInvariantError(
                    f"Continuation run {operation_id} no longer matches its lane"
                ),
                context,
            )
        if driven.value["kind"] == "settled":
            return ok(driven.value["outcome"])
        if driven.value["reason"] == "deferred":
            return ok(
                {
                    "operationId": operation_id,
                    "status": "suspended",
                    "deferred": driven.value["deferred"],
                }
            )
        raise self._on_fault(
            SessionInvariantError(f"Continuation run {operation_id} returned an unwaited retry"),
            context,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def resume(self, context: Context) -> Any:
        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()

        def _inspect(state: LaneState, _reader: Any) -> Any:
            operation = state.operation
            if operation is None:
                return LaneReturn(
                    result=err(
                        NothingToResume(
                            lane=self.name,
                            message=f"Lane {self.name!r} has no active operation to resume",
                        )
                    )
                )
            return LaneReturn(result=ok({"operationId": operation.meta.operation_id}))

        inspected = await self.command(_inspect, context)
        if not inspected.ok:
            return inspected
        operation_id = inspected.value["operationId"]
        driven = await self.drive(
            _dict({"operationId": operation_id, "pollDeferred": True, "waitForRetry": True}), context
        )
        if not driven.ok:
            if getattr(driven.error, "tag", None) == "Closed":
                return err(driven.error)
            raise self._on_fault(
                SessionInvariantError(f"Operation {operation_id} no longer matches its lane"),
                context,
            )
        if driven.value["kind"] == "settled":
            return ok(driven.value["outcome"])
        if driven.value["reason"] == "deferred":
            return ok(
                {
                    "operationId": operation_id,
                    "status": "suspended",
                    "deferred": driven.value["deferred"],
                }
            )
        raise self._on_fault(
            SessionInvariantError(f"Operation {operation_id} returned an unwaited retry"), context
        )

    async def abort(self, context: Context) -> Any:
        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()
        operation_id = await self.command(
            lambda state, _reader: LaneReturn(
                result=None if state.operation is None else state.operation.meta.operation_id
            ),
            context,
        )
        if operation_id is None:
            return err(
                NoActiveOperation(
                    lane=self.name, message=f"Lane {self.name!r} has no active operation"
                )
            )
        requested = await self.request_abort(operation_id, context)
        if not requested.ok:
            if getattr(requested.error, "tag", None) == "Closed":
                return err(requested.error)
            return err(
                NoActiveOperation(
                    lane=self.name,
                    message=f"Lane {self.name!r} no longer has the inspected operation",
                )
            )
        driven = await self.drive(_dict({"operationId": operation_id}), context)
        if not driven.ok and getattr(driven.error, "tag", None) == "Closed":
            return err(driven.error)
        if not driven.ok:
            raise self._on_fault(
                SessionInvariantError(
                    f"Cancelled operation {operation_id} no longer matches its lane"
                ),
                context,
            )
        return ok(
            {
                "operationId": operation_id,
                "steer": requested.value["steer"],
                "followUp": requested.value["followUp"],
            }
        )

    async def seal(self, error: BaseException) -> None:
        if self.closed_error is None:
            self.closed_error = error
        if self.active_drive is not None:
            self.active_drive.close_gate(error)
        self.signal_state_change()
        if self.idle_owner is not None:
            await await_with_context(None, _swallow(self.idle_owner))

    # ------------------------------------------------------------------
    # Queues
    # ------------------------------------------------------------------

    async def steer(self, message: Any, images: Optional[List[ImageContent]], context: Context) -> Any:
        return await self.enqueue("steer", message, images, context)

    async def follow_up(self, message: Any, images: Optional[List[ImageContent]], context: Context) -> Any:
        return await self.enqueue("followUp", message, images, context)

    async def next_run(self, message: Any, images: Optional[List[ImageContent]], context: Context) -> Any:
        return await self.enqueue("nextRun", message, images, context)

    async def enqueue(
        self,
        kind: str,
        message_input: Any,
        images: Optional[List[ImageContent]],
        context: Context,
    ) -> Any:
        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()
        at = _now()
        if isinstance(message_input, str):
            if len(message_input) == 0 and not images:
                return err(
                    InvalidMessage(
                        lane=self.name,
                        reason="empty",
                        message="Queued input must contain text or an image",
                    )
                )
            message = _user_message(
                _text_blocks(message_input),
                images or [],
                at,
            )
        else:
            if (
                getattr(message_input, "role", None) == "assistant"
                and getattr(message_input, "stop_reason", None) == "pending"
            ):
                return err(
                    InvalidMessage(
                        lane=self.name,
                        reason="pending_assistant",
                        message="Cannot queue a pending assistant message",
                    )
                )
            if images and getattr(message_input, "role", None) != "user":
                return err(
                    InvalidMessage(
                        lane=self.name,
                        reason="images_with_non_user",
                        message="Images can be added only to queued user messages",
                    )
                )
            if not images or getattr(message_input, "role", None) != "user":
                message = copy.deepcopy(message_input)
            else:
                cloned = copy.deepcopy(message_input)
                content = getattr(cloned, "content", None)
                if isinstance(content, str):
                    leading = [_text_block(content)] if content else []
                else:
                    leading = list(content or [])
                cloned.content = [*leading, *images]
                message = cloned
        entry_id = self.session.id_generator.next(at)

        async def _plan(state: LaneState, reader: Any) -> Any:
            inbox = [*state.inbox, InboxItem(entry_id=entry_id, kind=kind)]
            queues = [
                *await read_lane_queues(reader, state.inbox, context),
                {"entryId": entry_id, "kind": kind, "type": "message", "message": message},
            ]
            current_operation_id = (
                None if state.operation is None else state.operation.meta.operation_id
            )
            return CommitDecision(
                writes=[
                    set_value(
                        pending_entry(entry_id),
                        PendingEntry(type="message", payload=message),
                    ),
                    set_value(
                        lane_state_value(self.name),
                        durable_lane_state(state, current_operation_id, inbox),
                    ),
                ],
                next=LaneState(
                    tip_id=state.tip_id,
                    configuration=state.configuration,
                    inbox=inbox,
                    last_operation_id=state.last_operation_id,
                    operation=state.operation,
                ),
                materialize=lambda _commit: ok({"entryId": entry_id}),
                events=lambda _commit: [
                    {"type": "queue_update", "queues": queues, "lane": self.name}
                ],
            )

        return await self.command(_plan, context)

    async def cancel_queued(self, entry_id: str, context: Context) -> Any:
        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()

        async def _plan(state: LaneState, reader: Any) -> Any:
            queued = next((item for item in state.inbox if item.entry_id == entry_id), None)
            if queued is None:
                consumed = entry_id in (await reader.get_entries([entry_id], context))
                return LaneReturn(
                    result=ok({"kind": "already_consumed" if consumed else "not_found"})
                )
            if await reader.get_value(pending_entry(entry_id), context) is None:
                raise SessionInvariantError(
                    f"Queued {queued.kind} entry {entry_id} is missing its payload"
                )
            inbox = [item for item in state.inbox if item.entry_id != entry_id]
            queues = await read_lane_queues(reader, inbox, context)
            current_operation_id = (
                None if state.operation is None else state.operation.meta.operation_id
            )
            return CommitDecision(
                writes=[
                    delete_value(pending_entry(entry_id)),
                    set_value(
                        lane_state_value(self.name),
                        durable_lane_state(state, current_operation_id, inbox),
                    ),
                ],
                next=LaneState(
                    tip_id=state.tip_id,
                    configuration=state.configuration,
                    inbox=inbox,
                    last_operation_id=state.last_operation_id,
                    operation=state.operation,
                ),
                materialize=lambda _commit: ok({"kind": "cancelled"}),
                events=lambda _commit: [
                    {"type": "queue_update", "queues": queues, "lane": self.name}
                ],
            )

        return await self.command(_plan, context)

    async def record_usage(self, usage: Usage, options: Any, context: Context) -> Any:
        if isinstance(self.closed_error, HarnessClosed):
            return err(Closed(message=str(self.closed_error)))
        self.assert_open()

        def _plan(state: LaneState, _reader: Any) -> Any:
            usage_id = self.session.id_generator.next()
            row = {"id": usage_id, "usage": usage, "adjustment": True}
            entry_id = _opt(options, "entry_id")
            details = _opt(options, "details")
            if entry_id is not None:
                row["entryId"] = entry_id
            if details is not None:
                row["details"] = details

            def _events(commit: Any) -> List[Any]:
                return [
                    {
                        "type": "usage",
                        "lane": self.name,
                        "row": {**row, "seq": commit.seqs[0]},
                        "totals": commit.stats.usage,
                    }
                ]

            return CommitDecision(
                writes=[insert_usage(_usage_row(row))],
                next=state,
                materialize=lambda _commit: ok({"usageId": usage_id}),
                events=_events,
            )

        return await self.command(_plan, context)

    # ------------------------------------------------------------------
    # Idle coordination
    # ------------------------------------------------------------------

    async def wait_for_idle(self, context: Context) -> None:
        while True:
            observation = await self.command(
                lambda state, _reader: LaneReturn(
                    result=IdleObservation(kind="idle")
                    if state.operation is None and self.active_drive is None
                    else IdleObservation(kind="wait", drive=self.active_drive, change=self.state_change)
                ),
                context,
            )
            if observation.kind == "idle":
                return
            await await_with_context(context, _idle_wait(observation.drive, observation.change))

    async def run_when_idle(self, callback: Callable[[Context], Any], context: Context) -> None:
        owner: Optional["asyncio.Future"] = None
        while True:
            observation = await self._claim_idle(context)
            if observation.kind == "claimed":
                owner = observation.drive
                break
            await await_with_context(context, _idle_wait(observation.drive, observation.change))
        if owner is None:
            raise self._on_fault(
                SessionInvariantError("Idle callback claim was not published"), context
            )
        try:
            self.assert_open()
            result = callback(context)
            if asyncio.iscoroutine(result):
                await result
        finally:
            if self.idle_owner is owner:
                self.idle_owner = None
            if not owner.done():
                owner.set_result(None)
            self.signal_state_change()

    async def _claim_idle(self, context: Context) -> IdleClaimObservation:
        def _plan(state: LaneState, _reader: Any) -> Any:
            if state.operation is not None or self.active_drive is not None:
                return LaneReturn(
                    result=IdleClaimObservation(
                        kind="wait", drive=self.active_drive, change=self.state_change
                    )
                )
            claimed: "asyncio.Future" = asyncio.get_running_loop().create_future()
            self.idle_owner = claimed
            self.signal_state_change()
            return LaneReturn(result=IdleClaimObservation(kind="claimed", drive=claimed))

        return await self.command(_plan, context)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    async def get_model(self, _context: Context) -> Any:
        self.assert_open()
        model = self.state.configuration.model
        return self.models.get_model(_opt(model, "provider"), _opt(model, "model_id", "modelId"))

    async def set_model(self, model: Any, context: Context) -> None:
        model_identity = {
            "provider": _opt(model, "provider"),
            "modelId": _opt(model, "model_id", "modelId"),
        }

        def _update(configuration: Any) -> Any:
            return _replace_configuration(configuration, model=model_identity)

        def _event(previous: Any, value: Any) -> dict:
            return {
                "type": "config_update",
                "property": "model",
                "previous": _opt(previous.model, "provider") or previous.model,
                "value": value.model,
            }

        await self.set_configuration(_update, _event, context)

    async def get_thinking_level(self, _context: Context) -> Any:
        self.assert_open()
        return self.state.configuration.thinking_level

    async def set_thinking_level(self, thinking_level: str, context: Context) -> None:
        await self.set_configuration(
            lambda configuration: _replace_configuration(
                configuration, thinking_level=thinking_level
            ),
            lambda previous, value: {
                "type": "config_update",
                "property": "thinkingLevel",
                "previous": previous.thinking_level,
                "value": value.thinking_level,
            },
            context,
        )

    async def get_active_tools(self, _context: Context) -> List[str]:
        self.assert_open()
        return self.state.configuration.active_tool_names

    async def set_active_tools(self, active_tool_names: List[str], context: Context) -> None:
        await self.set_configuration(
            lambda configuration: _replace_configuration(
                configuration, active_tool_names=active_tool_names
            ),
            lambda previous, value: {
                "type": "config_update",
                "property": "activeTools",
                "previous": previous.active_tool_names,
                "value": value.active_tool_names,
            },
            context,
        )

    async def set_configuration(
        self,
        update: Callable[[Any], Any],
        event: Callable[[Any, Any], dict],
        context: Context,
    ) -> None:
        def _plan(state: LaneState, _reader: Any) -> Any:
            configuration = update(state.configuration)
            return CommitDecision(
                writes=[set_value(lane_config(self.name), configuration)],
                next=LaneState(
                    tip_id=state.tip_id,
                    configuration=configuration,
                    inbox=state.inbox,
                    last_operation_id=state.last_operation_id,
                    operation=state.operation,
                ),
                materialize=lambda _commit: None,
                events=lambda _commit: [
                    {**event(state.configuration, configuration), "lane": self.name}
                ],
            )

        await self.command(_plan, context)

    # ------------------------------------------------------------------
    # Watch and reads
    # ------------------------------------------------------------------

    async def watch(self, context: Context) -> Any:
        async def _read(state: LaneState, reader: Any) -> Any:
            async def _resnapshot(resnapshot_context: Context, mark_boundary: Callable[[], None]) -> Any:
                async def _inner(latest: LaneState, latest_reader: Any) -> Any:
                    snapshot = await self.capture_lane_snapshot(latest, latest_reader, resnapshot_context)
                    mark_boundary()
                    return snapshot

                return await self.read_lane(_inner, resnapshot_context)

            watcher = self._install_watch(
                None,
                lambda event: event.type == "usage"
                or not hasattr(event, "lane")
                or event.lane == self.name,
                context,
                _resnapshot,
            )
            try:
                watcher.snapshot = await self.capture_lane_snapshot(state, reader, context)
                return watcher
            except BaseException:
                watcher.unsubscribe()
                raise

        return await self.read_lane(_read, context)

    async def capture_lane_snapshot(self, state: LaneState, reader: Any, context: Context) -> Any:
        captured = copy.deepcopy(state)
        transcript: List[Any] = []
        if captured.tip_id is not None:
            transcript = await reader.scan_branch(
                _branch_scan(captured.tip_id, stop_at_type="compaction"), context
            )
            transcript.reverse()
        queues = await read_lane_queues(reader, captured.inbox, context)
        last_result = None
        if captured.last_operation_id is not None:
            stored = await reader.get_value(operation_result(captured.last_operation_id), context)
            if stored is None:
                raise SessionInvariantError(
                    f"Lane {self.name!r} is missing result {captured.last_operation_id}"
                )
            last_result = stored.value
        stats = await reader.get_stats(context)
        operation = captured.operation
        operation_snapshot = None
        if operation is not None:
            operation_snapshot = await self._capture_operation_snapshot(
                operation, reader, context
            )
        return {
            "lane": self.name,
            "transcript": transcript,
            "tipId": captured.tip_id,
            **({} if last_result is None else {"lastResult": last_result}),
            "configuration": captured.configuration,
            "stats": stats,
            "operation": operation_snapshot,
            "queues": queues,
            "faulted": isinstance(self.closed_error, HarnessFault),
        }

    async def _capture_operation_snapshot(self, operation: Operation, reader: Any, context: Context) -> dict:
        running_tools: List[Any] = []
        streaming_message = None
        retry = None
        deferred = None

        async def _read_streaming(response_entry_id: str) -> Any:
            frames = await read_assistant_frames(
                reader, operation.meta.operation_id, response_entry_id, context
            )
            return _reduce_frames(frames)

        state = operation.state
        at = state.at
        if at == "assistant.retry_wait":
            retry = {
                "attempt": state.next_attempt,
                "maxAttempts": state.generation_context.retry_policy.max_attempts,
                "nextAttemptAt": state.not_before,
            }
        elif at == "assistant.effect_pending":
            streaming_message = await _read_streaming(state.response_entry_id)
        elif at in ("deferred.suspended", "deferred.effect_pending"):
            source = (await reader.get_entries([state.source_entry_id], context)).get(
                state.source_entry_id
            )
            message = getattr(source, "message", None)
            if (
                getattr(source, "type", None) != "message"
                or getattr(message, "role", None) != "assistant"
                or getattr(message, "deferred", None) is None
            ):
                raise SessionInvariantError("Deferred source is missing its assistant handle")
            deferred = {"handle": message.deferred, "poll": state.poll}
            if at == "deferred.effect_pending":
                streaming_message = await _read_streaming(state.response_entry_id)
        elif at == "tools":
            batch = state.batch
            assistant = (await reader.get_entries([batch.assistant_entry_id], context)).get(
                batch.assistant_entry_id
            )
            message = getattr(assistant, "message", None)
            if getattr(assistant, "type", None) != "message" or getattr(message, "role", None) != "assistant":
                raise SessionInvariantError("Tool batch assistant entry is invalid")
            for call in batch.calls:
                if call.status in ("planned", "completed"):
                    continue
                block = message.content[call.source_index]
                if _opt(block, "type") != "toolCall":
                    raise SessionInvariantError(
                        f"Tool call source index {call.source_index} does not name a tool-call block"
                    )
                stored_args = await reader.get_value(
                    operation_tool_args(
                        operation.meta.operation_id, batch.turn_id, call.source_index
                    ),
                    context,
                )
                if call.status == "effect_pending":
                    if stored_args is None:
                        raise SessionInvariantError(
                            f"Tool call {block.id} is missing persisted arguments"
                        )
                    checkpoint = await reader.get_value(
                        pending_tool_output(operation.meta.operation_id, call.result_entry_id),
                        context,
                    )
                    entry = {
                        "status": "running",
                        "toolCallId": block.id,
                        "toolName": block.name,
                        "args": stored_args.value,
                    }
                    if checkpoint is not None:
                        entry["result"] = checkpoint.value
                    running_tools.append(entry)
                    continue
                staged = await reader.get_value(pending_entry(call.result_entry_id), context)
                staged_type = getattr(staged.value, "type", None) if staged is not None else None
                staged_payload = getattr(staged.value, "payload", None) if staged is not None else None
                if staged_type != "message" or getattr(staged_payload, "role", None) != "toolResult":
                    raise SessionInvariantError(
                        f"Tool call {call.result_entry_id} is missing its staged result"
                    )
                if (
                    getattr(staged_payload, "tool_call_id", None) != block.id
                    or getattr(staged_payload, "tool_name", None) != block.name
                ):
                    raise SessionInvariantError(
                        f"Tool call {call.result_entry_id} has a mismatched staged result"
                    )
                running_tools.append(
                    {
                        "status": "settled",
                        "toolCallId": block.id,
                        "toolName": block.name,
                        "args": stored_args.value if stored_args is not None else block.arguments,
                        "result": tool_result_from_message(staged_payload, call.terminate),
                        "isError": staged_payload.is_error,
                    }
                )
        elif at == "summary.retry_wait":
            retry = {
                "attempt": state.next_attempt,
                "maxAttempts": state.summary_context.retry_policy.max_attempts,
                "nextAttemptAt": state.not_before,
            }

        snapshot: dict = {
            "id": operation.meta.operation_id,
            "kind": operation.meta.intent.kind,
            "startedAt": operation.meta.started_at,
            "fromTipId": operation.meta.source_tip_id,
            "status": "aborting"
            if operation.state.control.status == "cancel_requested"
            else "open",
            "runningTools": running_tools,
        }
        if retry is not None:
            snapshot["retry"] = retry
        if deferred is not None:
            snapshot["deferred"] = deferred
        if streaming_message is not None:
            snapshot["streamingMessage"] = streaming_message
        return snapshot

    # ------------------------------------------------------------------
    # Branch writes
    # ------------------------------------------------------------------

    async def find_entries(self, query: Any, context: Context) -> List[Any]:
        if query is None:
            query = {}
        self.assert_open()
        start = _opt(query, "start")
        if start is None:
            start = self.state.tip_id
        if start is None:
            return []
        order = _opt(query, "order")
        return await self.session.scan_branch(
            _branch_scan(
                start,
                stop_at_type=_opt(query, "stop_at_type", "stopAtType"),
                stop_at_id=_opt(query, "stop_at_id", "stopAtId"),
                type=_opt(query, "type"),
                custom_type=_opt(query, "custom_type", "customType"),
                order="newestFirst" if order is None else order,
                limit=_opt(query, "limit"),
                cursor=_opt(query, "cursor"),
            ),
            context,
        )

    async def find_entry(self, query: Any, context: Context) -> Any:
        if query is None:
            query = {}
        limit = _opt(query, "limit")
        narrowed = dict(query) if isinstance(query, dict) else {}
        narrowed["limit"] = 1 if limit is None else min(limit, 1)
        entries = await self.find_entries(narrowed, context)
        return entries[0] if entries else None

    async def append_message(self, message: AgentMessage, context: Context) -> str:
        return await self.append({"type": "message", "payload": message}, context)

    async def append_custom_entry(self, custom_type: str, data: Any, context: Context) -> str:
        pending: dict = {"type": "custom", "customType": custom_type}
        if data is not None:
            pending["payload"] = data
        return await self.append(pending, context)

    async def append(self, pending: Any, context: Context) -> str:
        self.assert_open()
        payload = _opt(pending, "payload")
        if (
            _opt(pending, "type") == "message"
            and getattr(payload, "role", None) == "assistant"
            and getattr(payload, "stop_reason", None) == "pending"
        ):
            raise SessionPendingAssistantMessageError()
        entry_id = self.session.id_generator.next()

        async def _plan(state: LaneState, reader: Any) -> Any:
            if state.operation is None:
                queued = inbox_items(state.inbox, "write")
                captured = []
                for item in queued:
                    stored = await reader.get_value(pending_entry(item.entry_id), context)
                    if stored is None:
                        raise SessionInvariantError(
                            f"Pending write {item.entry_id} is missing its payload"
                        )
                    captured.append(pending_entry_write(item.entry_id, stored.value))
                inbox = without_inbox_items(state.inbox, queued)
                queues = (
                    None if not queued else await read_lane_queues(reader, inbox, context)
                )
                entries = chain_entries(
                    state.tip_id,
                    [*captured, pending_entry_write(entry_id, pending)],
                )

                def _events(commit: Any) -> List[Any]:
                    events = list(committed_entry_events(entries, commit, self.name))
                    if queues is not None:
                        events.append(
                            {"type": "queue_update", "queues": queues, "lane": self.name}
                        )
                    return events

                return CommitDecision(
                    writes=[
                        *[insert_entry(entry) for entry in entries],
                        *[delete_value(pending_entry(item.entry_id)) for item in queued],
                        set_value(branch_tip(self.name), entry_id),
                        set_value(lane_state_value(self.name), durable_lane_state(state, None, inbox)),
                    ],
                    next=LaneState(
                        tip_id=entry_id,
                        configuration=state.configuration,
                        inbox=inbox,
                        last_operation_id=state.last_operation_id,
                        operation=None,
                    ),
                    materialize=lambda _commit: entry_id,
                    events=_events,
                )

            operation = state.operation
            inbox = [*state.inbox, InboxItem(entry_id=entry_id, kind="write")]
            queue_item: dict = {"entryId": entry_id, "kind": "write"}
            if _opt(pending, "type") == "message":
                queue_item["type"] = "message"
                queue_item["message"] = payload
            else:
                queue_item["type"] = "custom"
                queue_item["customType"] = _opt(pending, "customType", "custom_type")
                if payload is not None:
                    queue_item["data"] = payload
            queues = [
                *await read_lane_queues(reader, state.inbox, context),
                queue_item,
            ]
            return CommitDecision(
                writes=[
                    set_value(
                        pending_entry(entry_id),
                        PendingEntry(
                            type=payload_type,
                            payload=payload,
                            custom_type=custom_type,
                        ),
                    ),
                    set_value(
                        lane_state_value(self.name),
                        durable_lane_state(state, operation.meta.operation_id, inbox),
                    ),
                ],
                next=LaneState(
                    tip_id=state.tip_id,
                    configuration=state.configuration,
                    inbox=inbox,
                    last_operation_id=state.last_operation_id,
                    operation=operation,
                ),
                materialize=lambda _commit: entry_id,
                events=lambda _commit: [
                    {"type": "queue_update", "queues": queues, "lane": self.name}
                ],
            )

        return await self.command(_plan, context)


# ----------------------------------------------------------------------
# Module-level plan builders (kept out of the class so each stays readable)
# ----------------------------------------------------------------------


def _navigation_plan(lane: Lane, context: Context, **kwargs: Any) -> Callable[[LaneState, Any], Any]:
    operation_id: str = kwargs["operation_id"]
    target_id: Optional[str] = kwargs["target_id"]
    options: Any = kwargs["options"]
    summarize: bool = kwargs["summarize"]
    preparation: Optional[BranchPreparation] = kwargs["preparation"]
    observed_tip_id: Optional[str] = kwargs["observed_tip_id"]
    started_at: int = kwargs["started_at"]
    acceptance_config: Config = kwargs["acceptance_config"]
    task_id: str = kwargs["task_id"]
    label = _opt(options, "label")
    custom_instructions = _opt(options, "custom_instructions", "customInstructions")

    async def _plan(state: LaneState, reader: Any) -> Any:
        if state.operation is not None:
            return LaneReturn(
                result=err(
                    LaneBusy(
                        lane=lane.name,
                        operation_id=state.operation.meta.operation_id,
                        operation_kind=state.operation.meta.intent.kind,
                        message=f"Lane {lane.name!r} already has an active operation",
                    )
                )
            )
        if state.tip_id != observed_tip_id:
            return LaneReturn(result=None)
        if target_id == state.tip_id:
            return LaneReturn(
                result=err(
                    InvalidNavigation(
                        lane=lane.name,
                        reason="current_tip",
                        message="Navigation target must differ from the current tip",
                    )
                )
            )
        if target_id is None and label is not None:
            return LaneReturn(
                result=err(
                    InvalidNavigation(
                        lane=lane.name,
                        reason="root_label",
                        message="Root navigation cannot set a label",
                    )
                )
            )
        if summarize and (state.tip_id is None or target_id is None):
            return LaneReturn(
                result=err(
                    InvalidNavigation(
                        lane=lane.name,
                        reason="source_root" if state.tip_id is None else "target_root",
                        message="Summarized navigation requires non-root source and target entries",
                    )
                )
            )
        if target_id is not None and target_id not in (
            await reader.get_entries([target_id], context)
        ):
            return LaneReturn(
                result=err(
                    UnknownTarget(target_id=target_id, message=f"Unknown target: {target_id}")
                )
            )

        meta = OperationMeta(
            operation_id=operation_id,
            lane=lane.name,
            source_tip_id=state.tip_id,
            started_at=started_at,
            intent=OperationIntent(
                kind="navigation",
                target_id=target_id,
                summarize=summarize,
                label=label,
                custom_instructions=custom_instructions,
            ),
        )
        settings = captured_settings(acceptance_config)
        writes: List[Any] = []
        if summarize:
            if state.tip_id is None or target_id is None or preparation is None:
                raise SessionInvariantError(
                    "Validated summarized navigation is missing its preparation"
                )
            writes.append(
                set_value(
                    operation_preparation(operation_id, task_id),
                    durable_branch_preparation(preparation),
                )
            )
            operation_state_value_obj: Any = SummaryDecidingOperation(
                settings=settings,
                latest_assistant_entry_id=None,
                task=_summary_task(
                    task_id,
                    None,
                    custom_instructions,
                    {"kind": "commit_navigation", "targetId": target_id, "label": label},
                ),
            )
        else:
            operation_state_value_obj = NavigationReadyToCommitOperation(
                settings=settings,
                latest_assistant_entry_id=None,
                target_id=target_id,
                label=label,
            )
        writes.extend(
            [
                set_value(operation_meta(operation_id), meta),
                set_value(operation_state(operation_id), operation_state_value_obj),
                set_value(lane_state_value(lane.name), durable_lane_state(state, operation_id)),
            ]
        )
        return CommitDecision(
            writes=writes,
            next=LaneState(
                tip_id=state.tip_id,
                configuration=state.configuration,
                inbox=state.inbox,
                last_operation_id=state.last_operation_id,
                operation=Operation(meta=meta, state=operation_state_value_obj),
            ),
            materialize=lambda _commit: ok(
                {"operationId": operation_id, "kind": "navigation", "startedAt": started_at}
            ),
            events=lambda _commit: [
                {
                    "type": "navigation_start",
                    "lane": lane.name,
                    "runId": operation_id,
                    "targetId": target_id,
                    "startedAt": started_at,
                }
            ],
        )

    return _plan


def _drive_claim_plan(lane: Lane, options: Any, context: Context) -> Callable[[LaneState, Any], Any]:
    operation_id = _opt(options, "operation_id", "operationId")

    async def _plan(state: LaneState, reader: Any) -> Any:
        signal = context.signal
        if signal is not None and signal.aborted:
            reason = signal.reason
            if isinstance(reason, BaseException):
                return LaneReject(error=reason)
            return LaneReject(error=asyncio.CancelledError("The operation was aborted"))
        if state.operation is not None and state.operation.meta.operation_id == operation_id:
            if lane.active_drive is None:
                drive = Drive(options, context)
                lane.active_drive = drive
                lane.signal_state_change()
                return LaneReturn(result=DriveClaim(drive=drive, installed=True))
            if lane.active_drive.operation_id == operation_id:
                return LaneReturn(result=DriveClaim(drive=lane.active_drive, installed=False))
            return LaneReturn(result=DriveClaim(kind="occupied", drive=lane.active_drive))

        stored = await reader.get_value(operation_result(operation_id), context)
        if stored is None:
            current = None if state.operation is None else state.operation.meta.operation_id
            return LaneReturn(
                result=DriveClaim(
                    kind="mismatch",
                    error=lane.mismatch(operation_id, current, state.last_operation_id),
                )
            )
        return LaneReturn(result=DriveClaim(kind="settled", outcome=stored.value))

    return _plan


def _abort_plan(
    lane: Lane,
    context: Context,
    operation_id: str,
    settle_gate: Callable[[bool], None],
) -> Callable[[LaneState, Any], Any]:
    async def _plan(state: LaneState, reader: Any) -> Any:
        operation = state.operation
        if operation is None or operation.meta.operation_id != operation_id:
            current = None if operation is None else operation.meta.operation_id
            return LaneReturn(
                result=err(lane.mismatch(operation_id, current, state.last_operation_id))
            )
        if operation.state.control.status == "cancel_requested":
            return LaneReturn(
                result=ok(
                    {"operationId": operation_id, "newlyRequested": False, "steer": [], "followUp": []}
                )
            )

        removed = [item for item in state.inbox if item.kind in ("steer", "followUp")]
        payloads = []
        for item in removed:
            stored = await reader.get_value(pending_entry(item.entry_id), context)
            if stored is None:
                raise SessionInvariantError(
                    f"Pending {item.kind} entry {item.entry_id} is missing its message"
                )
            pending = PendingEntry.from_value(stored.value)
            if pending.type != "message":
                raise SessionInvariantError(
                    f"Pending {item.kind} entry {item.entry_id} is missing its message"
                )
            payloads.append((item, pending.payload))
        steer = [message for item, message in payloads if item.kind == "steer"]
        follow_up = [message for item, message in payloads if item.kind == "followUp"]
        removed_ids = {item.entry_id for item in removed}
        inbox = [item for item in state.inbox if item.entry_id not in removed_ids]
        queues = await read_lane_queues(reader, inbox, context)
        successor = copy.copy(operation.state)
        successor.control = _cancel_requested_control(_now())

        def _events(_commit: Any) -> List[Any]:
            events = [
                {
                    "type": "operation_abort",
                    "operationId": operation_id,
                    "steer": steer,
                    "followUp": follow_up,
                    "lane": lane.name,
                }
            ]
            if removed:
                events.append({"type": "queue_update", "queues": queues, "lane": lane.name})
            return events

        def _materialize(_commit: Any) -> Any:
            settle_gate(True)
            return ok(
                {"operationId": operation_id, "newlyRequested": True, "steer": steer, "followUp": follow_up}
            )

        return CommitDecision(
            writes=[
                *[delete_value(pending_entry(item.entry_id)) for item in removed],
                set_value(operation_state(operation_id), successor),
                set_value(lane_state_value(lane.name), durable_lane_state(state, operation_id, inbox)),
            ],
            next=LaneState(
                tip_id=state.tip_id,
                configuration=state.configuration,
                inbox=inbox,
                last_operation_id=state.last_operation_id,
                operation=Operation(meta=operation.meta, state=successor),
            ),
            materialize=_materialize,
            events=_events,
        )

    return _plan


def _install_drive(lane: Lane, drive: Drive, context: Context, drive_operation: Callable[..., Any]) -> None:
    """Start one installed drive pass and settle its completion future from it."""
    task = asyncio.ensure_future(drive_operation(lane, drive))

    def _done(completed: "asyncio.Future") -> None:
        if lane.active_drive is drive:
            lane.active_drive = None
            lane.signal_state_change()
        if completed.cancelled():
            drive.fail(asyncio.CancelledError())
            return
        error = completed.exception()
        if error is not None:
            failure: Any = lane.closed_error
            if failure is None:
                try:
                    failure = lane._on_fault(error, drive.context)
                except BaseException as fault_error:  # noqa: BLE001
                    failure = fault_error
            drive.fail(failure)
            return
        drive.settle(completed.result())

    task.add_done_callback(_done)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _now() -> int:
    import time

    return int(time.time() * 1000)


def _dict(values: dict) -> Any:
    """A plain attribute bag so plans can read request fields uniformly."""
    return _Bag(values)


class _Bag:
    """Attribute view over a request mapping, keeping camelCase keys reachable."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        for key, value in self._values.items():
            if key == name or _snake(key) == name:
                return value
        raise AttributeError(name)


def _snake(value: str) -> str:
    out = []
    for char in value:
        if char.isupper():
            out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out)


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping, an object, or an attribute bag."""
    if value is None:
        return None
    if isinstance(value, _Bag):
        value = value._values
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
            snake = _snake(name)
            if snake in value:
                return value[snake]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
        snake = _snake(name)
        if snake != name and hasattr(value, snake):
            return getattr(value, snake)
    return None


def _text_blocks(text: str) -> List[Any]:
    """The leading text blocks for one user message, or none when empty."""
    return [_text_block(text)] if text else []


def _text_block(text: str) -> Any:
    from pi_ai.types import TextContent

    return TextContent(text=text)


def _user_message(leading: List[Any], images: List[Any], timestamp: int) -> Any:
    """Build one user message from already-built leading blocks plus image blocks."""
    from pi_ai.types import UserMessage

    return UserMessage(content=[*leading, *images], timestamp=timestamp)


def _find_named(values: Any, name: str) -> Any:
    if not values:
        return None
    return next((candidate for candidate in values if getattr(candidate, "name", None) == name), None)


def _summary_task(
    task_id: str,
    reason: Optional[str],
    custom_instructions: Optional[str],
    boundary: dict,
) -> Any:
    from ..session.types import ResultBoundary, SummaryTask

    if boundary["kind"] == "commit_navigation":
        resolved = ResultBoundary(
            kind="commit_navigation", target_id=boundary["targetId"], label=boundary.get("label")
        )
    elif boundary["kind"] == "resume_checkpoint":
        resolved = ResultBoundary(kind="resume_checkpoint", resume_after=boundary["resumeAfter"])
    else:
        resolved = ResultBoundary(kind="finish")
    return SummaryTask(
        task_id=task_id,
        reason=reason,
        custom_instructions=custom_instructions,
        boundary=resolved,
    )


def _branch_scan(start: str, **overrides: Any) -> Any:
    from ..session.types import BranchScan

    values = {"start": start}
    values.update(overrides)
    return BranchScan(**values)


def _replace_configuration(configuration: Any, **changes: Any) -> Any:
    from ..session.types import LaneConfiguration

    return LaneConfiguration(
        model=changes.get("model", configuration.model),
        thinking_level=changes.get("thinking_level", configuration.thinking_level),
        active_tool_names=changes.get("active_tool_names", configuration.active_tool_names),
    )


def _cancel_requested_control(requested_at: int) -> Any:
    from ..session.types import Control

    return Control(status="cancel_requested", requested_at=requested_at)


def _usage_row(row: dict) -> Any:
    from ..session.types import UsageRow

    return UsageRow(
        id=row["id"],
        usage=row["usage"],
        entry_id=row.get("entryId"),
        adjustment=row["adjustment"],
        details=row.get("details"),
    )


def _index_of(entries: List[Any], entry_id: str) -> int:
    for index, entry in enumerate(entries):
        if entry.id == entry_id:
            return index
    raise SessionInvariantError(f"Entry {entry_id} is not on the scanned branch")


def _reduce_frames(frames: List[Any]) -> Any:
    from pi_ai.assistant_message_frame import reduce_assistant_message_frames

    return reduce_assistant_message_frames(frames)


def _swallow_future(future: "asyncio.Future") -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


async def _swallow(future: "asyncio.Future") -> None:
    try:
        await future
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


async def _race(*futures: Any) -> None:
    """Resolve when the first of the supplied futures settles, ignoring the rest."""
    pending = [asyncio.ensure_future(asyncio.shield(item)) if not isinstance(item, asyncio.Future) else item for item in futures if item is not None]
    if not pending:
        return
    try:
        await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for item in pending:
            if not item.done():
                item.cancel()


async def _idle_wait(drive: Optional[Drive], change: Optional["asyncio.Future"]) -> None:
    if drive is None:
        await _race(change)
        return
    await _race(drive.completion, change)
