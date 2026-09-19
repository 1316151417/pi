"""Collapse task kind ported from ``harness/pico3/kinds/collapse.ts``.

Summarizes a prefix of the conversation into one ``pi.summary`` entry, with the
same staleness rules, hook, retry policy and failure vocabulary as TypeScript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...._chord.context import Context
from ...._pi_ai.types import JsonObject, JsonValue
from ..bounded import Bounded  # noqa: F401 - imported for parity with the TS module graph
from ..system import effective_tools
from ..types import (
    Completion,
    CoreTx,
    Entry,
    HookInfo,
    HookResult,
    Id,
    Kind,
    KindConfig,
    ModelRef,
    RequestMessage,
    RetryPolicy,
    Runtime,
    Step,
    SystemMessage,
    Task,
    ThinkingLevel,
    to_stored,
)
__all__ = [
    "CollapseInput",
    "CollapseCheckpoint",
    "CollapseFailure",
    "CollapseHooks",
    "collapse_config",
    "collapse",
    "collapse_kind",
    "choose_through",
    "estimate",
]


@dataclass
class CollapseInput:
    reason: str = "manual"  # "threshold" | "manual" | "overflow"
    through: Id = 0
    instructions: Optional[str] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"reason": self.reason, "through": self.through}
        if self.instructions is not None:
            data["instructions"] = self.instructions
        return data


@dataclass
class CollapseCheckpoint:
    """``phase`` selects the meaningful fields: summarizing, retrying, prepared."""

    phase: str = "summarizing"
    expected_head: Optional[Id] = None
    model: Optional[ModelRef] = None
    thinking_level: str = "off"
    retry: Optional[RetryPolicy] = None
    attempt: int = 1
    instructions: Optional[str] = None
    until_ms: Optional[int] = None
    last_error: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class CollapseFailure:
    reason: str = "provider"  # "stale" | "declined" | "provider" | "retries_exhausted" | "no_model"
    detail: str = ""

    def to_json(self) -> JsonObject:
        return {"reason": self.reason, "detail": self.detail}


class CollapseHooks:
    """Hook points this kind calls: ``before_collapse``."""

    async def before_collapse(
        self,
        reason: str,
        through: Id,
        entries: List[Entry],
        info: HookInfo,
        ctx: Context,
    ) -> HookResult: ...


#: Declared configuration this kind reads.
collapse_config = {"rewindable": {"threshold": 0, "keepRecent": 20_000}}


def estimate(messages: List[RequestMessage]) -> int:
    from ...compaction.compaction import estimate_context_tokens
    from .generation import normalize_request_messages

    return estimate_context_tokens(
        normalize_request_messages(
            [message for message in messages if _role(message) != "system"]
        )
    ).tokens


def _role(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _failed(reason: str, detail: str) -> Completion:
    return Completion(status="failed", failure=CollapseFailure(reason=reason, detail=detail))


def _base_of(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """The checkpoint minus its phase and phase-specific fields."""
    return {
        key: value
        for key, value in checkpoint.items()
        if key not in ("phase", "untilMs", "lastError", "summary")
    }


def _checkpoint(task: Task) -> Dict[str, Any]:
    return dict(task.checkpoint or {})


def _input(task: Task) -> Dict[str, Any]:
    if isinstance(task.input, dict):
        return task.input
    return dict(task.input.__dict__) if task.input is not None else {}


async def _head_moved(tx: Any, conversation_id: Id, base: Dict[str, Any]) -> bool:
    head = await tx.newest_entry(conversation_id, {"withHead": True})
    return (head.id if head is not None else None) != base.get("expectedHead")


async def _initial(task: Task, runtime: Runtime, ctx: Context) -> Step:
    input_record = _input(task)
    state = await runtime.rewindable(task.conversation_id, ctx)
    if state.get("model") is None:
        return Step(done=lambda _tx=None, _task=None: _failed("no_model", "no model configured"))
    sticky = await runtime.sticky(task.conversation_id, ctx)
    head = await runtime.newest_entry(task.conversation_id, {"withHead": True}, ctx)
    view = await runtime.context(task.conversation_id, input_record.get("through"), ctx)
    decision: Dict[str, Any] = {}

    def on_value(value: Any) -> Any:
        nonlocal decision
        if value is None:
            return None
        decision = value
        return True

    await runtime.hooks.each(
        ctx,
        lambda hook, api: _call_before_collapse(
            hook, input_record.get("reason"), input_record.get("through"), view.entries, api, ctx
        ),
        on_value,
    )
    if isinstance(decision, dict) and decision.get("decline"):
        return Step(
            done=lambda _tx=None, _task=None: _failed("declined", "declined by beforeCollapse")
        )
    base: Dict[str, Any] = {
        "expectedHead": head.id if head is not None else None,
        "model": state.get("model"),
        "thinkingLevel": state.get("thinkingLevel", "off"),
        "retry": sticky.get("retry") or RetryPolicy(),
        "attempt": 1,
    }
    instructions = decision.get("instructions") or input_record.get("instructions")
    if instructions:
        base["instructions"] = instructions
    summary = decision.get("summary")
    if summary is not None:
        async def next_summary(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Any:
            if await _head_moved(tx, current.conversation_id, base):
                return _failed("stale", "head moved during beforeCollapse")
            return {"phase": "prepared", "summary": summary, **base}

        return Step(next=next_summary)
    return await _summarize_now(task, base, runtime, ctx, check_head=True)


async def _summarizing(task: Task, runtime: Runtime, ctx: Context) -> Step:
    return _after_failure(_base_of(_checkpoint(task)), None, "interrupted", runtime)


async def _retrying(task: Task, runtime: Runtime, ctx: Context) -> Step:
    checkpoint = _checkpoint(task)
    await runtime.sleep(checkpoint["untilMs"], ctx)
    base = _base_of(checkpoint)
    base["attempt"] = checkpoint.get("attempt", 0) + 1
    return await _summarize_now(task, base, runtime, ctx)


async def _prepared(task: Task, runtime: Runtime, ctx: Context) -> Step:
    checkpoint = _checkpoint(task)
    summary = checkpoint.get("summary")
    base = _base_of(checkpoint)
    input_record = _input(task)

    async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
        view = await tx.context(current.conversation_id)
        head_id = view.head.id if view.head is not None else None
        if head_id != base.get("expectedHead"):
            return _failed("stale", "head moved during collapse")
        retained = next(
            (entry for entry in view.entries if entry.id > input_record.get("through")), None
        )
        id = tx.append_entry(
            current.conversation_id,
            _new_entry(
                kind="pi.summary",
                data={"through": input_record.get("through")},
                model=[{"role": "user", "content": summary, "timestamp": runtime.now()}],
                head=retained.id if retained is not None else "self",
            ),
        )
        return Completion(status="completed", result={"summary": id})

    return Step(done=done)


async def _abort(_task: Task, _runtime: Runtime, _ctx: Context) -> Any:
    return lambda *_args: None


async def _summarize_now(
    task: Task,
    base: Dict[str, Any],
    runtime: Runtime,
    ctx: Context,
    check_head: bool = False,
) -> Step:
    input_record = _input(task)

    async def prepare(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> bool:
        if check_head and await _head_moved(tx, current.conversation_id, base):
            return True
        tx.checkpoint({"phase": "summarizing", **base})
        return False

    stale = await runtime.commit(prepare, ctx)
    if stale:
        return Step(done=lambda _tx=None, _task=None: _failed("stale", "head moved during beforeCollapse"))
    model = runtime.models.resolve(base["model"])
    if model is None:
        return Step(done=lambda _tx=None, _task=None: _failed("no_model", "model unavailable"))
    view = await runtime.context(task.conversation_id, input_record.get("through"), ctx)
    tools = effective_tools(view.messages)
    request: List[Any] = list(view.messages)
    if tools:
        request.append(
            SystemMessage(
                content="",
                tools_removed=tools,
                timestamp=runtime.now(),
            ).to_json()
        )
    instructions = base.get("instructions") or "Summarize the conversation so far for continuation."
    request.append(
        {
            "role": "user",
            "content": f"{instructions}\n\nRespond with the summary only.",
            "timestamp": runtime.now(),
        }
    )
    message: Any = None
    try:
        stream = runtime.models.stream(
            model, _request_options(request, base["thinkingLevel"]), ctx
        )
        async for event in stream:
            if event.type == "done":
                message = event.message
                break
            if event.type == "error":
                message = event.error
                break
    except Exception as error:  # noqa: BLE001 - classified as a failed attempt
        if ctx.signal is not None and ctx.signal.aborted:
            raise
        return _after_failure(base, None, str(error), runtime)
    if message is None:
        return _after_failure(base, None, "no message", runtime)
    if any(_block_type(block) == "toolCall" for block in _content(message)):
        message = _with_stop_reason(message, "error")
        return _after_failure(base, message, "summarizer returned tool calls", runtime)
    if _field(message, "stopReason") == "error":
        return _after_failure(
            base, message, _field(message, "errorMessage") or "provider error", runtime
        )
    summary = "".join(
        _field(block, "text") or "" for block in _content(message) if _block_type(block) == "text"
    )
    return Step(next={"phase": "prepared", "summary": summary, **base})


def _after_failure(base: Dict[str, Any], message: Any, detail: str, runtime: Runtime) -> Step:
    # generation and collapse share the retry policy and import each other, exactly like the
    # TypeScript pair; the binding is resolved at call time to break the module cycle.
    from .generation import retry_decision

    decision = retry_decision(base, message, runtime.now())
    if decision["kind"] == "fail":
        return Step(done=lambda _tx=None, _task=None: _failed(decision["reason"], detail))

    def next_checkpoint(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
        tx.emit(
            {
                "type": "compaction.retrying",
                "taskId": current.id,
                "attempt": base["attempt"],
                "retryAt": decision["untilMs"],
                "error": detail,
            }
        )
        return {
            "phase": "retrying",
            **base,
            "untilMs": decision["untilMs"],
            "lastError": detail,
        }

    return Step(next=next_checkpoint)


collapse_kind = Kind(
    name="pi.collapse",
    config=KindConfig(rewindable=dict(collapse_config["rewindable"])),
    inflight=["summarizing"],
    initial=_initial,
    phases={"summarizing": _summarizing, "retrying": _retrying, "prepared": _prepared},
    abort=_abort,
)

#: The built-in kind token.
collapse = collapse_kind


def choose_through(entries: List[Entry], keepRecent: int) -> Optional[Id]:
    """The newest entry that may be dropped, keeping ``keepRecent`` estimated tokens."""
    exchanges: List[Dict[str, Any]] = []
    open_exchange: Optional[Dict[str, Any]] = None
    for entry in entries:
        model = entry.model or []
        role = _role(model[0]) if model else None
        tokens = estimate(list(model))
        if role == "assistant":
            open_exchange = {"last": entry.id, "tokens": tokens}
            exchanges.append(open_exchange)
        elif role == "toolResult" and open_exchange is not None:
            open_exchange["last"] = entry.id
            open_exchange["tokens"] += tokens
        else:
            open_exchange = None
            exchanges.append({"last": entry.id, "tokens": tokens})
    retained = 0
    index = len(exchanges) - 1
    while index >= 0 and retained + exchanges[index]["tokens"] <= keepRecent:
        retained += exchanges[index]["tokens"]
        index -= 1
    if index < 0:
        return None
    if index == len(exchanges) - 1:
        return exchanges[index - 1]["last"] if index - 1 >= 0 else None
    return exchanges[index]["last"]


def _new_entry(**fields: Any) -> Any:
    from ..types import NewEntry

    return NewEntry(**fields)


def _request_options(messages: List[Any], thinking_level: Any) -> Any:
    from ..types import RequestOptions

    return RequestOptions(messages=messages, thinking_level=thinking_level)


def _content(message: Any) -> List[Any]:
    if isinstance(message, dict):
        return message.get("content") or []
    return getattr(message, "content", None) or []


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _block_type(block: Any) -> Any:
    return _field(block, "type")


def _with_stop_reason(message: Any, reason: str) -> Any:
    if isinstance(message, dict):
        return {**message, "stopReason": reason}
    return message


async def _call_before_collapse(
    hook: Any, reason: Any, through: Any, entries: List[Entry], api: HookInfo, ctx: Context
) -> HookResult:
    handler = getattr(hook, "before_collapse", None)
    if handler is None:
        return None
    return await _resolve(handler(reason, through, entries, api, ctx))


async def _resolve(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


_ = (CoreTx, JsonObject, JsonValue, ThinkingLevel, to_stored, field)
