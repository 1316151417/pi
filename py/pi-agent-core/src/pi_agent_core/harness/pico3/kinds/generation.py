"""Generation task kind ported from ``harness/pico3/kinds/generation.ts``.

The turn's provider call: one managed-system-entry preparation, then a durable
``requesting`` checkpoint before the network effect, streaming into the sticky
turn view, and a classified completion that starts the tool tasks.

Every step that writes before acting (``prepared`` -> ``requesting``,
``retrying``) is preserved, so a crash is recovered by dispatching the same
checkpoint again.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...._chord.context import Context
from pi_ai.assistant_message_frame import AssistantMessageFrameEncoder
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    DeferredHandle,
    JsonObject,
    JsonValue,
    Usage,
    message_from_json,
)
from ...compaction.compaction import estimate_context_tokens
from pi_ai.utils.retry import is_retryable_assistant_error
from ..system import (
    plan_managed_entry,
    prepare_draft,
    same_snapshot,
    take_snapshot,
)
from ..types import (
    Completion,
    CoreTx,
    HookInfo,
    HookResult,
    Id,
    Kind,
    KindConfig,
    ModelRef,
    NewEntry,
    RequestMessage,
    RequestOptions,
    RetryPolicy,
    Runtime,
    Step,
    Task,
    TaskScan,
    ThinkingLevel,
    to_stored,
)
from .collapse import choose_through
from .frames import apply_frame

__all__ = [
    "GenerationInput",
    "GenerationCheckpoint",
    "GenerationResult",
    "GenerationFailure",
    "DisplayAssistantData",
    "GenerationHooks",
    "generation_config",
    "generation",
    "generation_kind",
    "retry_decision",
    "empty_usage",
    "estimate",
    "normalize_request_messages",
]


@dataclass
class GenerationInput:
    inputs: List[Id] = field(default_factory=list)

    def to_json(self) -> JsonObject:
        return {"inputs": list(self.inputs)}


@dataclass
class GenerationCheckpoint:
    """``phase`` selects the meaningful fields: prepared, requesting, retrying, deferred."""

    phase: str = "prepared"
    cutoff: Id = 0
    system: Optional[Id] = None
    model: Optional[ModelRef] = None
    thinking_level: str = "off"
    tools: List[str] = field(default_factory=list)
    retry: Optional[RetryPolicy] = None
    attempt: int = 0
    until_ms: Optional[int] = None
    last_error: Optional[str] = None
    handle: Any = None
    poll_at: Optional[int] = None


@dataclass
class GenerationResult:
    assistant: Id = 0
    tools: List[Id] = field(default_factory=list)
    postTools: Optional[Id] = None
    successor: Optional[Id] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"assistant": self.assistant, "tools": list(self.tools)}
        if self.postTools is not None:
            data["postTools"] = self.postTools
        if self.successor is not None:
            data["successor"] = self.successor
        return data


@dataclass
class GenerationFailure:
    reason: str = "provider"  # "provider" | "overflow" | "retries_exhausted" | "no_model"
    detail: str = ""
    assistant: Optional[Id] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"reason": self.reason, "detail": self.detail}
        if self.assistant is not None:
            data["assistant"] = self.assistant
        return data


@dataclass
class DisplayAssistantData:
    """Display-only assistant entries carry the message here, never in ``model``."""

    attempt: int = 0
    display: Any = None
    reason: str = "error"  # "error" | "aborted"


class GenerationHooks:
    """Hook points this kind calls."""

    async def system_instructions(
        self, input: Dict[str, Any], info: HookInfo, ctx: Context
    ) -> HookResult: ...

    async def before_request(
        self, request: Dict[str, Any], info: HookInfo, ctx: Context
    ) -> HookResult: ...

    async def after_response(
        self, message: AssistantMessage, info: HookInfo, ctx: Context
    ) -> None: ...

    async def on_yield(self, answer: AssistantMessage, info: HookInfo, ctx: Context) -> HookResult: ...


#: Configuration this kind reads, with defaults. ``model`` has none: absent -> failed/no_model.
generation_config = {
    "rewindable": {
        "model": None,
        "thinkingLevel": "off",
        "selectedTools": [],
        "profile": "default",
    },
    "sticky": {"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 2000, "maxAgentDelayMs": 60_000}},
}


def empty_usage() -> Usage:
    return Usage()


def normalize_request_messages(messages: List[Any]) -> List[Any]:
    """Stored messages -> pi-ai message records.

    The token estimator reads attributes, not durable JSON, so every caller that
    holds stored messages must normalize before estimating.
    """
    out: List[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        data = message
        if isinstance(data.get("content"), str):
            data = {**data, "content": [{"type": "text", "text": data["content"]}]}
        out.append(message_from_json(data))
    return out


def estimate(messages: List[RequestMessage]) -> int:
    """What pi-ai actually sends: no system messages, no aborted/error assistants."""
    return estimate_context_tokens(
        normalize_request_messages(
            [
                message
                for message in messages
                if _role(message) != "system"
                and not (
                    _role(message) == "assistant"
                    and _field(message, "stopReason") in ("aborted", "error")
                )
            ]
        )
    ).tokens


def fail(reason: str, detail: str, assistant: Optional[Id] = None) -> Completion:
    """Build a declared generation failure."""
    failure = GenerationFailure(reason=reason, detail=detail, assistant=assistant)
    return Completion(status="failed", failure=failure)


def retry_decision(
    checkpoint: Dict[str, Any], message: Any, now: int
) -> Dict[str, Any]:
    """Retry policy over a durable attempt counter."""
    retry = checkpoint.get("retry") or {}
    if isinstance(retry, RetryPolicy):
        retry = {
            "enabled": retry.enabled,
            "maxRetries": retry.max_retries,
            "baseDelayMs": retry.base_delay_ms,
            "maxAgentDelayMs": retry.max_agent_delay_ms,
        }
    if message is not None and not is_retryable_assistant_error(_as_message(message)):
        return {"kind": "fail", "reason": "provider"}
    if not retry.get("enabled"):
        return {"kind": "fail", "reason": "provider"}
    attempt = checkpoint.get("attempt", 0)
    if attempt > retry.get("maxRetries", 0):
        return {"kind": "fail", "reason": "retries_exhausted"}
    delay = retry.get("baseDelayMs", 0) * 2 ** max(0, attempt - 1)
    safe_delay = delay if delay <= 2**53 - 1 else 2**53 - 1
    return {"kind": "retry", "untilMs": now + min(safe_delay, retry.get("maxAgentDelayMs") or 60_000)}


# ---------------------------------------------------------------------------
# The kind
# ---------------------------------------------------------------------------


async def _initial(task: Task, rt: Any, ctx: Context) -> Step:
    snapshot_result = await rt.commit(
        lambda tx, current=None, _ctx=None: take_snapshot(
            tx, current.conversation_id, rt.registries.sections, rt.registries.tools
        ),
        ctx,
    )
    snapshot = snapshot_result.snapshot
    canonical = snapshot_result.canonical
    seed = snapshot_result.seed
    if snapshot.settings.get("model") is None:
        return _fail_step(task, "no_model", "no model configured")
    retry = (await rt.sticky(task.conversation_id, ctx)).get("retry") or RetryPolicy()

    warnings: List[str] = []
    prepared = await prepare_draft(
        rt,
        rt.registries.sections,
        canonical,
        seed,
        snapshot.settings,
        warnings.append,
        ctx,
    )
    model = snapshot.settings.get("model")
    thinking_level = snapshot.settings.get("thinkingLevel", "off")

    async def next(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Any:
        conversation_id = current.conversation_id
        again = await take_snapshot(
            tx, conversation_id, rt.registries.sections, rt.registries.tools
        )
        if not same_snapshot(again.snapshot, snapshot):
            return "retry"
        plan = await plan_managed_entry(
            tx, conversation_id, snapshot, canonical, prepared.desired, prepared.tools, rt.now()
        )
        for message in warnings:
            tx.emit({"type": "warning", "source": "generation", "message": message})
        system = (
            None
            if plan is None
            else tx.append_entry(
                conversation_id,
                NewEntry(kind="pi.system", model=plan.model, data=plan.data.to_json(), edits=plan.edits),
            )
        )
        if system is None:
            newest = await tx.newest_entry(conversation_id)
            cutoff = newest.id
        else:
            cutoff = system
        tx.sticky(conversation_id)["turn"] = {"tools": []}  # a new generation starts a fresh turn view
        return {
            "phase": "prepared",
            "cutoff": cutoff,
            "system": system,
            "model": model,
            "thinkingLevel": thinking_level,
            "tools": [_tool_name(tool) for tool in prepared.tools],
            "retry": retry,
            "attempt": 0,
        }

    return Step(next=next)


async def _prepared(task: Task, rt: Any, ctx: Context) -> Step:
    checkpoint = _checkpoint(task)
    model = rt.models.resolve(_model_ref(checkpoint["model"]))
    if model is None:
        ref = _model_ref(checkpoint["model"])
        return _fail_step(task, "no_model", f"model {ref.provider}/{ref.model_id} unavailable")
    derived = await _derive(task, checkpoint, rt, ctx)
    if estimate(derived["messages"]) > model.context_window - model.max_tokens:
        return await _overflow(task, rt)
    attempt = checkpoint.get("attempt", 0) + 1

    async def mark_requesting(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> None:
        tx.checkpoint({**checkpoint, "phase": "requesting", "attempt": attempt})
        tx.emit({"type": "generation.started", "taskId": task.id, "attempt": attempt})

    await rt.commit(mark_requesting, ctx)  # before the effect
    terminal, deferred = await _stream(checkpoint, model, derived["messages"], rt, ctx)
    if deferred is not None:
        poll_at = rt.now() + (getattr(deferred, "poll_after_ms", None) or 5000)

        def next_deferred(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
            tx.emit({"type": "generation.deferred", "taskId": task.id, "pollAt": poll_at})
            return {
                **checkpoint,
                "phase": "deferred",
                "attempt": attempt,
                "handle": to_stored(deferred),
                "pollAt": poll_at,
            }

        return Step(next=next_deferred)
    return await _classify(task, {**checkpoint, "attempt": attempt}, terminal, rt, ctx)


async def _requesting(task: Task, rt: Any, ctx: Context) -> Step:
    checkpoint = _checkpoint(task)
    decision = retry_decision(checkpoint, None, rt.now())
    if decision["kind"] == "fail":
        return _fail_step(task, decision["reason"], "interrupted")

    def next_retry(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
        tx.append_entry(
            current.conversation_id,
            NewEntry(kind="pi.usage", data={"attempt": checkpoint.get("attempt", 0), "error": "interrupted"}),
        )
        tx.emit(
            {
                "type": "generation.retrying",
                "taskId": current.id,
                "attempt": checkpoint.get("attempt", 0),
                "retryAt": decision["untilMs"],
                "error": "interrupted",
            }
        )
        return {
            **checkpoint,
            "phase": "retrying",
            "untilMs": decision["untilMs"],
            "lastError": "interrupted",
        }

    return Step(next=next_retry)


async def _retrying(task: Task, rt: Any, ctx: Context) -> Step:
    checkpoint = _checkpoint(task)
    await rt.sleep(checkpoint["untilMs"], ctx)
    prep = {
        key: value
        for key, value in checkpoint.items()
        if key not in ("untilMs", "lastError")
    }

    def next_prepared(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
        turn = tx.sticky(current.conversation_id).setdefault("turn", {})
        turn["message"] = None
        return {**prep, "phase": "prepared"}

    return Step(next=next_prepared)


async def _deferred(task: Task, rt: Any, ctx: Context) -> Step:
    checkpoint = _checkpoint(task)
    model = rt.models.resolve(_model_ref(checkpoint["model"]))
    if model is None:
        return _fail_step(task, "no_model", "model disappeared")
    await rt.sleep(checkpoint["pollAt"], ctx)
    result = await rt.models.fetch_deferred(model, _handle(checkpoint["handle"]), ctx)
    if isinstance(result, dict) and result.get("deferred") is not None:
        poll_at = rt.now() + (result["deferred"].get("pollAfterMs") or 5000)

        def next_deferred(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
            tx.emit({"type": "generation.deferred", "taskId": task.id, "pollAt": poll_at})
            return {**checkpoint, "handle": to_stored(result["deferred"]), "pollAt": poll_at}

        return Step(next=next_deferred)
    message = result

    async def store_partial(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> None:
        tx.sticky(current.conversation_id).setdefault("turn", {})["message"] = to_stored(message)

    await rt.commit(store_partial, ctx)
    prep = {key: value for key, value in checkpoint.items() if key not in ("handle", "pollAt")}
    return await _classify(task, prep, message, rt, ctx)


async def _abort(task: Task, rt: Any, ctx: Context) -> Any:
    checkpoint = _checkpoint(task)
    if checkpoint.get("phase") == "deferred":
        model = rt.models.resolve(_model_ref(checkpoint["model"]))
        if model is not None:
            try:
                await rt.models.cancel_deferred(model, _handle(checkpoint["handle"]), ctx)
            except Exception:  # noqa: BLE001 - cancellation is best effort, as in TS
                pass

    async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
        turn = tx.sticky(current.conversation_id).setdefault("turn", {})
        partial = turn.get("message")
        assistant: Optional[Id] = None
        # Display-only: the partial goes in `data`, never in `model`, so it cannot enter a request.
        if partial and (partial.get("content") or []):
            assistant = tx.append_entry(
                current.conversation_id,
                NewEntry(
                    kind="pi.assistant",
                    data={
                        "attempt": checkpoint.get("attempt", 0),
                        "display": {**to_stored(partial), "stopReason": "aborted"},
                        "reason": "aborted",
                    },
                ),
            )
        tx.sticky(current.conversation_id)["turn"] = {"tools": []}
        inputs = _input(task).get("inputs") or []
        await tx.resolve_inputs(inputs, {"status": "unanswered", "reason": "aborted"})
        tx.emit(
            {
                "type": "turn.ended",
                "inputs": inputs,
                "status": "unanswered",
                "reason": "aborted",
            }
        )
        return {} if assistant is None else {"assistant": assistant}

    return done


generation_kind = Kind(
    name="pi.generation",
    turn=True,
    config=KindConfig(
        rewindable=dict(generation_config["rewindable"]),
        sticky={"retry": dict(generation_config["sticky"]["retry"])},
    ),
    inflight=["requesting"],
    initial=_initial,
    phases={"prepared": _prepared, "requesting": _requesting, "retrying": _retrying, "deferred": _deferred},
    abort=_abort,
    types=None,
)

#: The built-in kind token.
generation = generation_kind


# ---------------------------------------------------------------------------
# Request derivation and streaming
# ---------------------------------------------------------------------------


async def _derive(task: Task, checkpoint: Dict[str, Any], rt: Any, ctx: Context) -> Dict[str, Any]:
    view = await rt.context(task.conversation_id, checkpoint["cutoff"], ctx)
    request: Dict[str, Any] = {"messages": view.messages}

    def on_value(value: Any) -> None:
        nonlocal request
        if value is not None:
            request = value

    await rt.hooks.each(
        ctx,
        lambda hook, api: _call_before_request(hook, request, api, checkpoint["cutoff"], ctx),
        on_value,
    )
    return request


async def _stream(
    checkpoint: Dict[str, Any],
    model: Any,
    messages: List[Any],
    rt: Any,
    ctx: Context,
) -> Any:
    """Stream, coalescing frames into the turn view. Flush on size or time."""
    encoder = AssistantMessageFrameEncoder()
    pending: List[Any] = []
    pending_bytes = 0
    last_flush = rt.now()
    terminal: Any = None
    deferred: Any = None
    flushed_content = False

    async def flush() -> None:
        nonlocal pending, pending_bytes, last_flush, flushed_content
        batch = pending
        if not batch:
            return
        pending = []
        pending_bytes = 0
        last_flush = rt.now()
        if any(getattr(frame, "delta", None) is not None for frame in batch):
            flushed_content = True

        async def write(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> None:
            turn = tx.sticky(current.conversation_id).setdefault("turn", {})
            for frame in batch:
                apply_frame(turn, frame)

        await rt.commit(write, ctx)

    stream = rt.models.stream(model, RequestOptions(messages=messages, thinking_level=checkpoint["thinkingLevel"]), ctx)
    try:
        iterator = stream.__aiter__()
        pull = asyncio.ensure_future(iterator.__anext__())
        while True:
            if not pending:
                selected = await _await_task(pull)
            else:
                selected = await _race_pull(pull, rt, last_flush, ctx)
            if selected[0] == "flush":
                await flush()
                continue
            kind, result = selected
            if kind == "done":
                break
            event = result
            if event.type == "done":
                encoder.encode(event)
                if event.reason == "deferred" and getattr(event.message, "deferred", None) is not None:
                    deferred = event.message.deferred
                else:
                    terminal = event.message
                break
            if event.type == "error":
                encoder.encode(event)
                terminal = event.error
                break
            frame = encoder.encode(event)
            pull = asyncio.ensure_future(iterator.__anext__())
            if frame is None:
                continue
            pending.append(frame)
            pending_bytes += len(frame.delta) if getattr(frame, "delta", None) is not None else 64
            if pending_bytes >= 256 or (not flushed_content and getattr(frame, "delta", None) is not None):
                await flush()
    except StopAsyncIteration:
        pass
    except Exception as error:  # noqa: BLE001 - classified as a provider error
        if ctx.signal is not None and ctx.signal.aborted:
            raise
        terminal = AssistantMessage(
            content=[],
            api=getattr(model, "api", ""),
            provider=getattr(model, "provider", ""),
            model=getattr(model, "id", ""),
            usage=empty_usage(),
            stop_reason="error",
            error_message=str(error),
            timestamp=rt.now(),
        )
    if pending:
        await flush()
    if terminal is None and deferred is None:
        raise ValueError("stream ended without a terminal message")
    return terminal, deferred


async def _await_task(task: asyncio.Future) -> Any:
    try:
        return ("event", await task)
    except StopAsyncIteration:
        return ("done", None)


async def _race_pull(pull: asyncio.Future, rt: Any, last_flush: int, ctx: Context) -> Any:
    timer = asyncio.ensure_future(rt.sleep(last_flush + 100, ctx))
    try:
        done, _pending = await asyncio.wait({pull, timer}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        if not timer.done():
            timer.cancel()
    if pull in done:
        try:
            return ("event", pull.result())
        except StopAsyncIteration:
            return ("done", None)
    return ("flush", None)


# ---------------------------------------------------------------------------
# Steps 3-5: classify the terminal message.
# ---------------------------------------------------------------------------


async def _classify(
    task: Task, checkpoint: Dict[str, Any], message: Any, rt: Any, ctx: Context
) -> Step:
    attempt = checkpoint.get("attempt", 0)

    async def notify(hook: Any, api: HookInfo) -> None:
        handler = getattr(hook, "after_response", None)
        if handler is None:
            return
        await _resolve(handler(_as_message(message), _api_extra(api, attempt=attempt), ctx))

    await rt.hooks.each(ctx, notify)

    stop_reason = _field(message, "stopReason")
    if stop_reason == "error":
        decision = retry_decision(checkpoint, message, rt.now())
        error_message = _field(message, "errorMessage") or "provider error"
        if decision["kind"] == "retry":

            def next_retry(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
                tx.append_entry(
                    current.conversation_id,
                    NewEntry(
                        kind="pi.usage",
                        data={
                            "attempt": attempt,
                            "usage": to_stored(_field(message, "usage") or {}),
                            "error": error_message,
                        },
                    ),
                )
                tx.emit(
                    {
                        "type": "generation.retrying",
                        "taskId": current.id,
                        "attempt": attempt,
                        "retryAt": decision["untilMs"],
                        "error": error_message,
                    }
                )
                return {
                    **checkpoint,
                    "phase": "retrying",
                    "untilMs": decision["untilMs"],
                    "lastError": error_message,
                }

            return Step(next=next_retry)
        return Step(done=_terminal_error(task, checkpoint, message, decision["reason"]))
    if stop_reason == "aborted":
        return Step(done=_terminal_error(task, checkpoint, message, "provider"))

    calls = [block for block in _content(message) if _block_type(block) == "toolCall"]
    stored = to_stored(message)
    if calls:
        async def done_tools(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
            conversation_id = current.conversation_id
            collapse_through = await _threshold_collapse_through(tx, conversation_id, message)
            assistant = tx.append_entry(
                conversation_id,
                NewEntry(kind="pi.assistant", model=[stored], data={"attempt": attempt}),
            )
            turn = tx.sticky(conversation_id).setdefault("turn", {})
            turn["message"] = None
            turn["tools"] = [
                {
                    "callId": _field(call, "id"),
                    "name": _field(call, "name"),
                    "args": to_stored(_field(call, "arguments")),
                    "status": "pending",
                }
                for call in calls
            ]
            tools = [
                tx.create_task(
                    _tool_kind_ref(rt),
                    {"assistant": assistant, "call": to_stored(call), "offered": checkpoint.get("tools") or [], "index": index},
                    {},
                ).id
                for index, call in enumerate(calls)
            ]
            post_tools = tx.create_task(
                _post_tools_kind_ref(rt),
                {"inputs": _input(task).get("inputs") or [], "assistant": assistant, "tools": tools},
                {"after": tools},
            ).id
            if collapse_through is not None:
                tx.create_task(
                    _collapse_kind_ref(rt),
                    {"reason": "threshold", "through": collapse_through},
                    {},
                )
            tx.emit(
                {
                    "type": "generation.completed",
                    "taskId": current.id,
                    "entry": assistant,
                    "toolCalls": len(calls),
                }
            )
            return Completion(
                status="completed",
                result=GenerationResult(assistant=assistant, tools=tools, postTools=post_tools),
            )

        return Step(done=done_tools)

    continue_text: Optional[str] = None

    def on_yield(value: Any) -> Any:
        nonlocal continue_text
        if value is None:
            return None
        continue_text = value.get("continue")
        return True

    await rt.hooks.each(
        ctx,
        lambda hook, api: _call_on_yield(hook, message, api, ctx),
        on_yield,
    )

    async def done_text(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
        conversation_id = current.conversation_id
        collapse_through = await _threshold_collapse_through(tx, conversation_id, message)
        head = await tx.newest_entry(conversation_id, {"withHead": True})
        assistant = tx.append_entry(
            conversation_id,
            NewEntry(kind="pi.assistant", model=[stored], data={"attempt": attempt}),
        )
        tx.emit(
            {
                "type": "generation.completed",
                "taskId": current.id,
                "entry": assistant,
                "toolCalls": 0,
            }
        )
        tx.sticky(conversation_id)["turn"] = {"tools": []}
        if collapse_through is not None:
            tx.create_task(
                _collapse_kind_ref(rt), {"reason": "threshold", "through": collapse_through}, {}
            )
        boundary = await tx.boundary(conversation_id, "final", head.id if head is not None else None)
        triggers = boundary["triggers"]
        inputs = _input(task).get("inputs") or []
        if continue_text is not None and not triggers and not boundary["terminated"]:
            tx.append_entry(
                conversation_id,
                NewEntry(
                    kind="pi.user",
                    model=[{"role": "user", "content": continue_text, "timestamp": rt.now()}],
                    data={"continuation": True, "from": assistant},
                ),
            )
            successor = tx.create_task(
                _generation_kind_ref(rt), {"inputs": inputs}, {}
            ).id
            return Completion(
                status="completed",
                result=GenerationResult(assistant=assistant, tools=[], successor=successor),
            )
        await tx.resolve_inputs(inputs, {"status": "done", "answer": assistant})
        tx.emit(
            {"type": "turn.ended", "inputs": inputs, "status": "done", "answer": assistant}
        )
        if triggers:
            successor = tx.create_task(_generation_kind_ref(rt), {"inputs": triggers}, {}).id
            tx.emit({"type": "turn.started", "inputs": triggers})
            return Completion(
                status="completed",
                result=GenerationResult(assistant=assistant, tools=[], successor=successor),
            )
        return Completion(status="completed", result=GenerationResult(assistant=assistant, tools=[]))

    return Step(done=done_text)


def _terminal_error(task: Task, checkpoint: Dict[str, Any], message: Any, reason: str) -> Any:
    """Provider error / aborted stop: a display-only assistant entry, inputs unanswered."""
    attempt = checkpoint.get("attempt", 0)

    async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
        head = await tx.newest_entry(current.conversation_id, {"withHead": True})
        assistant = tx.append_entry(
            current.conversation_id,
            NewEntry(
                kind="pi.assistant",
                data={
                    "attempt": attempt,
                    "display": to_stored(message),
                    "reason": "aborted" if _field(message, "stopReason") == "aborted" else "error",
                },
            ),
        )
        detail = _field(message, "errorMessage") or "provider error"
        tx.emit(
            {
                "type": "generation.failed",
                "taskId": current.id,
                "reason": reason,
                "detail": detail,
                "entry": assistant,
            }
        )
        await _settle_failed_turn(
            tx, current, _input(task).get("inputs") or [], detail, head.id if head is not None else None
        )
        return fail(reason, detail, assistant)

    return done


async def _settle_failed_turn(
    tx: Any, current: Task, inputs: List[Id], detail: str, head_boundary: Optional[Id]
) -> None:
    tx.sticky(current.conversation_id)["turn"] = {"tools": []}
    await tx.resolve_inputs(inputs, {"status": "unanswered", "reason": "failed", "detail": detail})
    tx.emit(
        {
            "type": "turn.ended",
            "inputs": list(inputs),
            "status": "unanswered",
            "reason": "failed",
            "detail": detail,
        }
    )
    boundary = await tx.boundary(current.conversation_id, "final", head_boundary)
    if boundary["triggers"]:
        tx.create_task(
            tx.kinds["pi.generation"],
            {"inputs": boundary["triggers"]},
            {"conversationId": current.conversation_id},
        )
        tx.emit({"type": "turn.started", "inputs": boundary["triggers"]})


def _fail_step(task: Task, reason: str, detail: str) -> Step:
    """Every terminal generation failure settles its group and admits queued triggers."""

    async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
        head = await tx.newest_entry(current.conversation_id, {"withHead": True})
        tx.emit({"type": "generation.failed", "taskId": current.id, "reason": reason, "detail": detail})
        await _settle_failed_turn(
            tx, current, _input(task).get("inputs") or [], detail, head.id if head is not None else None
        )
        return fail(reason, detail)

    return Step(done=done)


async def _threshold_collapse_through(tx: Any, conversation_id: Id, message: Any) -> Optional[Id]:
    """Read-side of threshold collapse: computed before the assistant is appended, applied after."""
    live = await tx.tasks(
        _task_scan(conversation_id, kind="pi.collapse", status=["pending", "running"])
    )
    if live:
        return None
    state = tx.rewindable(conversation_id)
    if (state.get("threshold") or 0) <= 0:
        return None
    view = await tx.context(conversation_id)
    usage = _field(message, "usage") or {}
    used = max(
        (usage.get("input") or 0) + (usage.get("output") or 0),
        estimate([*view.messages, to_stored(message)]),
    )
    if used <= (state.get("threshold") or 0):
        return None
    return choose_through(view.entries, state.get("keepRecent") or 0)


async def _overflow(task: Task, rt: Any) -> Step:
    async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
        conversation_id = current.conversation_id
        state = tx.rewindable(conversation_id)
        view = await tx.context(conversation_id)
        through = choose_through(view.entries, state.get("keepRecent") or 0)
        tx.sticky(conversation_id)["turn"] = {"tools": []}
        if through is None:
            await tx.write(
                conversation_id,
                NewEntry(
                    kind="pi.notice",
                    model=[
                        {
                            "role": "user",
                            "content": "Context too large; nothing to compact.",
                            "timestamp": rt.now(),
                        }
                    ],
                ),
            )
            tx.emit(
                {
                    "type": "generation.failed",
                    "taskId": current.id,
                    "reason": "overflow",
                    "detail": "request exceeds context window and nothing is collapsible",
                }
            )
            await _settle_failed_turn(
                tx,
                current,
                _input(task).get("inputs") or [],
                "overflow",
                view.head.id if view.head is not None else None,
            )
            return fail("overflow", "request exceeds context window and nothing is collapsible")
        collapse_id = tx.create_task(
            _collapse_kind_ref(rt), {"reason": "overflow", "through": through}, {}
        ).id
        tx.create_task(
            _generation_kind_ref(rt),
            {"inputs": _input(task).get("inputs") or []},
            {"after": [collapse_id]},
        )
        tx.emit(
            {
                "type": "generation.failed",
                "taskId": current.id,
                "reason": "overflow",
                "detail": f"collapsing through {through}",
            }
        )
        return fail("overflow", f"collapsing through {through}")

    return Step(done=done)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tool_name(tool: Any) -> Any:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


def _checkpoint(task: Task) -> Dict[str, Any]:
    return dict(task.checkpoint or {})


def _input(task: Task) -> Dict[str, Any]:
    if isinstance(task.input, dict):
        return task.input
    return dict(task.input.__dict__) if task.input is not None else {}


def _model_ref(value: Any) -> ModelRef:
    if isinstance(value, ModelRef):
        return value
    if isinstance(value, dict):
        return ModelRef(provider=value.get("provider", ""), model_id=value.get("modelId", ""))
    return ModelRef()


def _handle(value: Any) -> Any:
    if isinstance(value, DeferredHandle):
        return value
    if isinstance(value, dict):
        return DeferredHandle(
            id=value.get("id", ""),
            provider=value.get("provider", ""),
            expires_at=value.get("expiresAt"),
        )
    return value


def _as_message(message: Any) -> Any:
    if isinstance(message, dict):
        return message_from_json(message)
    return message


@dataclass
class _InfoExtra:
    """``{...api, cutoff}`` / ``{...api, attempt}``: the hook info plus its extra field."""

    kind: str = ""
    task_id: Id = 0
    conversation_id: Id = 0
    cutoff: Optional[Id] = None
    attempt: Optional[int] = None


def _api_extra(api: Any, **extra: Any) -> _InfoExtra:
    return _InfoExtra(
        kind=getattr(api, "kind", ""),
        task_id=getattr(api, "task_id", 0),
        conversation_id=getattr(api, "conversation_id", 0),
        **extra,
    )


def _role(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    snake = _SNAKE.get(key, key)
    return getattr(value, snake, None)


_SNAKE = {
    "stopReason": "stop_reason",
    "errorMessage": "error_message",
    "thoughtSignature": "thought_signature",
    "toolCallId": "tool_call_id",
}


def _content(message: Any) -> List[Any]:
    if isinstance(message, dict):
        return message.get("content") or []
    return getattr(message, "content", None) or []


def _block_type(block: Any) -> Any:
    return _field(block, "type")


def _task_scan(conversation_id: Id, kind: Optional[str] = None, status: Optional[List[str]] = None) -> Any:
    return TaskScan(conversation_id=conversation_id, kind=kind, status=status)


def _tool_kind_ref(rt: Any) -> Any:
    return rt.kinds["pi.tool"]


def _post_tools_kind_ref(rt: Any) -> Any:
    return rt.kinds["pi.post_tools"]


def _collapse_kind_ref(rt: Any) -> Any:
    return rt.kinds["pi.collapse"]


def _generation_kind_ref(rt: Any) -> Any:
    return rt.kinds["pi.generation"]


async def _resolve(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _call_before_request(
    hook: Any, request: Dict[str, Any], api: HookInfo, cutoff: Id, ctx: Context
) -> HookResult:
    handler = getattr(hook, "before_request", None)
    if handler is None:
        return None
    return await _resolve(handler(request, _api_extra(api, cutoff=cutoff), ctx))


async def _call_on_yield(hook: Any, message: Any, api: HookInfo, ctx: Context) -> HookResult:
    handler = getattr(hook, "on_yield", None)
    if handler is None:
        return None
    return await _resolve(handler(_as_message(message), api, ctx))


_ = (AssistantMessageEvent, JsonObject, JsonValue, ThinkingLevel, field)
