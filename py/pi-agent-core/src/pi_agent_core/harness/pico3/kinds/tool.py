"""Tool task kind ported from ``harness/pico3/kinds/tool.ts``.

One tool call: offered-set check, lookup, argument validation, ``beforeTool``,
validation again, the durable ``started`` checkpoint, the invocation itself, and
an ``afterTool`` chain, then the bounded tool-result entry.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...._chord.context import Context
from pi_ai.types import JsonObject, JsonValue
from ...utils.truncate import truncate_head  # noqa: F401 - imported for parity with the TS module graph
from ..bounded import Bounded
from ..types import (
    BeforeToolApi,
    Completion,
    HookInfo,
    HookResult,
    Id,
    Kind,
    NewEntry,
    Step,
    Task,
    ToolApi,
    ToolControl,
    ToolDiagnostic,
    ToolResult,
    ToolSlot,
    to_stored,
)
from .task_api import task_api

__all__ = [
    "StoredToolCall",
    "ToolInput",
    "ToolCheckpoint",
    "ToolTaskResult",
    "ToolHooks",
    "tool",
    "tool_kind",
    "invalid",
    "same_identity",
    "DEFAULT_BOUNDS",
]


@dataclass
class ToolInput:
    assistant: Id = 0
    call: JsonObject = field(default_factory=dict)
    offered: List[str] = field(default_factory=list)
    #: Position of the call in the assistant message; also this tool's slot in sticky.turn.tools.
    index: int = 0


@dataclass
class ToolCheckpoint:
    phase: str = "started"
    replay: str = "unsafe"  # "safe" | "unsafe"
    call: JsonObject = field(default_factory=dict)


@dataclass
class ToolTaskResult:
    entry: Id = 0
    control: Optional[ToolControl] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"entry": self.entry}
        if self.control is not None:
            data["control"] = to_stored(self.control)
        return data


class ToolHooks:
    """Hook points this kind calls: ``before_tool`` and ``after_tool``."""

    async def before_tool(
        self, call: JsonObject, api: BeforeToolApi, ctx: Context
    ) -> HookResult: ...

    async def after_tool(
        self, call: JsonObject, result: ToolResult, api: HookInfo, ctx: Context
    ) -> HookResult: ...


#: Stored tool-call shape.
StoredToolCall = JsonObject

DEFAULT_BOUNDS = {"maxBytes": 64 * 1024, "maxLines": 200, "retain": "head"}


def synthetic(text: str, code: str) -> ToolResult:
    return ToolResult(
        content=[{"type": "text", "text": text}],
        is_error=True,
        diagnostics=[ToolDiagnostic(severity="error", message=text, code=code)],
    )


def invalid(declaration: Any, call: StoredToolCall) -> Optional[str]:
    """Validate arguments with the tool's real schema."""
    from pi_ai.validation import collect_errors

    parameters = getattr(declaration, "parameters", None)
    if isinstance(declaration, dict):
        parameters = declaration.get("parameters")
    errors = collect_errors(call.get("arguments"), parameters or {})
    if not errors:
        return None
    return "; ".join(f"{path or '/'}: {message}" for path, message in errors)


def same_identity(a: StoredToolCall, b: StoredToolCall) -> bool:
    return a.get("id") == b.get("id") and a.get("name") == b.get("name") and a.get("namespace") == b.get("namespace")


async def _initial(task: Task, rt: Any, ctx: Context) -> Any:
    input_record = _input(task)
    call = input_record.get("call") or {}
    offered = input_record.get("offered") or []
    if call.get("name") not in offered:
        return Step(
            done=_close(
                task,
                synthetic(f"tool {call.get('name')} was not offered", "not_offered"),
                None,
                rt.now(),
            )
        )
    declaration = rt.tools.get(call.get("name"))
    if declaration is None:
        return Step(
            done=_close(
                task,
                synthetic(f"tool {call.get('name')} is not registered", "missing_tool"),
                None,
                rt.now(),
            )
        )
    bad = invalid(declaration, call)
    if bad is not None:
        return Step(
            done=_close(
                task, synthetic(f"invalid arguments: {bad}", "invalid_arguments"), declaration, rt.now()
            )
        )
    hooked = await _before_tool(task, rt, call, ctx)
    if "block" in hooked:
        return Step(
            done=_close(
                task, synthetic(f"blocked: {hooked['block']}", "blocked"), declaration, rt.now()
            )
        )
    if not same_identity(call, hooked["call"]):
        return Step(
            done=_close(
                task, synthetic("blocked: call identity changed", "blocked"), declaration, rt.now()
            )
        )
    bad_final = invalid(declaration, hooked["call"])
    if bad_final is not None:
        return Step(
            done=_close(
                task,
                synthetic(f"invalid arguments after hook: {bad_final}", "invalid_arguments"),
                declaration,
                rt.now(),
            )
        )
    final = dict(hooked["call"])

    async def start(tx: Any, _current: Optional[Task] = None, _ctx: Any = None) -> None:
        tx.checkpoint({"phase": "started", "replay": getattr(declaration, "replay", None) or "unsafe", "call": final})
        slot = tx.tool_slot(task)
        slot["status"] = "running"
        slot.pop("waitingOn", None)
        tx.emit({"type": "tool.started", "taskId": task.id, "callId": final.get("id"), "name": final.get("name")})

    await rt.commit(start, ctx)  # durable before the effect
    return Step(done=await _invoke(task, final, declaration, rt, ctx))


async def _started(task: Task, rt: Any, ctx: Context) -> Any:
    """Only entered by the scheduler after reopen. The stored final call is the evidence."""
    checkpoint = dict(task.checkpoint or {})
    call = checkpoint.get("call") or {}
    declaration = rt.tools.get(call.get("name"))
    if declaration is None:
        return Step(
            done=_close(
                task,
                synthetic(f"tool {call.get('name')} unavailable after restart", "unavailable"),
                None,
                rt.now(),
                call,
            )
        )
    if checkpoint.get("replay") != "safe" or (getattr(declaration, "replay", None) or "unsafe") != "safe":
        return Step(
            done=_close(
                task,
                synthetic(f"tool {call.get('name')} was interrupted", "interrupted"),
                declaration,
                rt.now(),
                call,
            )
        )
    if invalid(declaration, call) is not None:
        return Step(
            done=_close(
                task,
                synthetic(
                    f"tool {call.get('name')} was interrupted; arguments no longer validate",
                    "interrupted",
                ),
                declaration,
                rt.now(),
                call,
            )
        )
    return Step(done=await _invoke(task, call, declaration, rt, ctx))


async def _abort(task: Task, rt: Any, ctx: Context) -> Any:
    for conversation_id in task.owns:
        async def read(
            tx: Any,
            _current: Optional[Task] = None,
            _ctx: Any = None,
            conversation_id: Id = conversation_id,
        ) -> List[Task]:
            return await tx.tasks(_task_scan(conversation_id))

        children = await rt.commit(read, ctx)
        for child in children:
            if not child.background:
                await rt.abort_task(child.id, ctx)

    def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Dict[str, Any]:
        checkpoint = dict(task.checkpoint or {})
        call = checkpoint.get("call") or _input(task).get("call") or {}
        entry = tx.append_entry(
            current.conversation_id,
            NewEntry(
                kind="pi.tool_result",
                model=[_to_message(call, synthetic("tool aborted", "aborted"), rt.now())],
                data={"diagnostics": [{"severity": "error", "message": "aborted", "code": "aborted"}]},
            ),
        )
        turn = tx.sticky(current.conversation_id).setdefault("turn", {})
        tools = turn.setdefault("tools", [])
        index = _input(task).get("index")
        slot = tools[index] if index is not None and index < len(tools) else None
        if slot is not None:
            slot["status"] = "aborted"
            slot["entry"] = entry
            slot.pop("waitingOn", None)
        tx.emit({"type": "tool.aborted", "taskId": current.id, "callId": call.get("id"), "entry": entry})
        return {"entry": entry}

    return done


tool_kind = Kind(
    name="pi.tool",
    turn=True,
    inflight=["started"],
    initial=_initial,
    phases={"started": _started},
    abort=_abort,
)

#: The built-in kind token.
tool = tool_kind


# ---------------------------------------------------------------------------
# beforeTool / invoke / close
# ---------------------------------------------------------------------------


async def _before_tool(
    task: Task, rt: Any, call: StoredToolCall, ctx: Context
) -> Dict[str, Any]:
    """The one point where a throwing handler blocks rather than being skipped."""
    current = call
    for binding in rt.hooks.handlers():
        namespace = binding.namespace

        def memo(*args: Any, binding: Any = binding, task: Task = task) -> Any:
            # memo(name, candidate, context) or memo(name, context)
            name = args[0]
            candidate = args[1] if len(args) > 2 else None
            context = args[2] if len(args) > 2 else args[1]
            has_candidate = len(args) > 2

            async def store(tx: Any, _current: Optional[Task] = None, _ctx: Any = None) -> Any:
                tx.plugins(binding.namespace)
                slot = tx.tool_slot(task)
                memos = slot.setdefault("memos", {})
                key = f"hook:{namespace.id}:{name}"
                if key in memos:
                    return memos[key]
                if not has_candidate:
                    return None
                memos[key] = candidate
                return candidate

            return rt.commit(store, context)

        def waiting(context: Context) -> Any:
            async def mark(tx: Any, current_task: Optional[Task] = None, _ctx: Any = None) -> None:
                tx.plugins(namespace)
                slot = tx.tool_slot(current_task)
                if slot.get("waitingOn") == namespace.id:
                    return
                slot["waitingOn"] = namespace.id
                tx.emit(
                    {
                        "type": "tool.waiting",
                        "taskId": current_task.id,
                        "callId": call.get("id"),
                        "on": namespace.id,
                    }
                )

            return rt.commit(mark, context)

        def emit(name: str, data: JsonValue, context: Context) -> Any:
            return rt.commit(
                lambda tx, _current=None, _ctx=None: tx.emit(namespace, name, data), context
            )

        api = _BeforeTool(binding.api, call.get("id"), waiting, memo, emit)
        try:
            handler = getattr(binding.handlers, "before_tool", None)
            value = None if handler is None else await _resolve(handler(current, api, ctx))
            if not value:
                continue
            if value.get("block") is not None:
                return {"block": value["block"]}
            if value.get("call") is not None:
                current = value["call"]
        except Exception as error:  # noqa: BLE001 - a throw blocks
            if ctx.signal is not None and ctx.signal.aborted:
                raise
            return {"block": f"hook threw: {error}"}
    return {"call": current}


class _BeforeTool:
    """``{...binding.api, callId, waiting, memo, emit}``."""

    def __init__(
        self,
        api: HookInfo,
        call_id: Any,
        waiting: Any,
        memo: Any,
        emit: Any,
    ) -> None:
        self.kind = getattr(api, "kind", "")
        self.task_id = getattr(api, "task_id", 0)
        self.conversation_id = getattr(api, "conversation_id", 0)
        self.call_id = call_id
        self.waiting = waiting
        self.memo = memo
        self.emit = emit


async def _invoke(
    task: Task,
    call: StoredToolCall,
    declaration: Any,
    rt: Any,
    ctx: Context,
) -> Any:
    """Invoke the tool: the kernel owns the stream, one bounded buffer, one flush path."""
    bounds = {**DEFAULT_BOUNDS, **(getattr(declaration, "output", None) or {})}
    buffer = Bounded(int(bounds["maxBytes"]), int(bounds["maxLines"]), str(bounds["retain"]))
    streamed = False
    last_flush = 0
    flushing = asyncio.ensure_future(_noop())
    flush_error: Optional[BaseException] = None

    def flush() -> None:
        nonlocal last_flush, flushing
        last_flush = rt.now()
        text = buffer.text()
        previous = flushing

        async def write() -> None:
            await previous

            async def store(tx: Any, _current: Optional[Task] = None, _ctx: Any = None) -> None:
                tx.tool_slot(task)["output"] = text

            await rt.commit(store, ctx)

        async def guarded() -> None:
            nonlocal flush_error
            try:
                await write()
            except Exception as error:  # noqa: BLE001 - a persistence failure is never hidden
                if flush_error is None:
                    flush_error = error

        flushing = asyncio.ensure_future(guarded())

    base_api = task_api(task, rt)

    def progress(update: Any, context: Context) -> Any:
        async def apply(tx: Any, _current: Optional[Task] = None, _ctx: Any = None) -> None:
            slot = tx.tool_slot(task)
            free = {
                "progress": slot.get("progress"),
                "details": slot.get("details"),
                "continuedBy": slot.get("continuedBy"),
            }
            update(free)
            if free["progress"] is not None:
                slot["progress"] = free["progress"]
            else:
                slot.pop("progress", None)
            if free["details"] is not None:
                slot["details"] = to_stored(free["details"])
            else:
                slot.pop("details", None)
            if free["continuedBy"] is not None:
                slot["continuedBy"] = free["continuedBy"]
            else:
                slot.pop("continuedBy", None)

        return rt.commit(apply, context)

    def stream(chunk: Any) -> None:
        nonlocal streamed
        streamed = True
        data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        buffer.push(data)
        if rt.now() - last_flush >= 100:
            flush()

    def memo(*args: Any) -> Any:
        # memo(name, candidate, context) or memo(name, context)
        name = args[0]
        has_candidate = len(args) > 2
        candidate = args[1] if has_candidate else None
        context = args[2] if has_candidate else args[1]

        async def store(tx: Any, _current: Optional[Task] = None, _ctx: Any = None) -> Any:
            slot = tx.tool_slot(task)
            memos = slot.setdefault("memos", {})
            key = f"tool:{name}"
            if key in memos:
                return memos[key]
            if not has_candidate:
                return None
            memos[key] = candidate
            return candidate

        return rt.commit(store, context)

    tool_api = _ToolApiFacade(base_api, call.get("id"), memo, stream, progress)
    try:
        result = await _resolve(declaration.execute(call.get("arguments") or {}, tool_api, ctx))
    except Exception as error:  # noqa: BLE001 - a throw becomes a synthetic error result
        if ctx.signal is not None and ctx.signal.aborted:
            raise
        result = synthetic(f"tool threw: {error}", "threw")
    if streamed:
        flush()
        await flushing
        if flush_error is not None:
            raise flush_error  # a persistence failure is never hidden
        if result.content is None:
            diagnostics = list(result.diagnostics or [])
            if buffer.dropped:
                diagnostics.append(
                    ToolDiagnostic(
                        severity="warn",
                        code="truncated",
                        message=(
                            f"{buffer.dropped_bytes} bytes / {buffer.dropped_lines} lines dropped "
                            f"({bounds['retain']} {bounds['maxBytes']} retained)"
                        ),
                    )
                )
            result = ToolResult(
                content=[{"type": "text", "text": buffer.text()}],
                is_error=result.is_error,
                details=result.details,
                diagnostics=diagnostics,
                control=result.control,
            )
    final = result

    def on_value(value: Any) -> None:
        nonlocal final
        if value is not None:
            final = value

    async def after(hook: Any, api: HookInfo) -> None:
        handler = getattr(hook, "after_tool", None)
        if handler is None:
            return None
        return await _resolve(handler(call, final, _with_call_id(api, call.get("id")), ctx))

    await rt.hooks.each(ctx, after, on_value)
    truncated = {"bytes": buffer.dropped_bytes, "lines": buffer.dropped_lines} if streamed else None
    return _close(task, final, declaration, rt.now(), call, truncated)


class _ToolApiFacade:
    """The task-scoped API plus the tool-only surface."""

    def __init__(self, base: Any, call_id: Any, memo: Any, stream: Any, progress: Any) -> None:
        self._base = base
        self.task_id = base.task_id
        self.conversation_id = base.conversation_id
        self.call_id = call_id
        self._memo = memo
        self._stream = stream
        self._progress = progress

    def stream(self, chunk: Any) -> None:
        self._stream(chunk)

    def progress(self, update: Any, ctx: Context) -> Any:
        return self._progress(update, ctx)

    def memo(self, *args: Any) -> Any:
        return self._memo(*args)

    def conversation(self, spec: Any, ctx: Context) -> Any:
        return self._base.conversation(spec, ctx)

    def task(self, kind: Any, input: Any, opts: Dict[str, Any], ctx: Context) -> Any:
        return self._base.task(kind, input, opts, ctx)

    def get_task(self, ref: Any, ctx: Context) -> Any:
        return self._base.get_task(ref, ctx)

    def wait_for_task(self, ref: Any, ctx: Context) -> Any:
        return self._base.wait_for_task(ref, ctx)

    def slot(self, ref: Any, ctx: Context) -> Any:
        return self._base.slot(ref, ctx)


def _close(
    task: Task,
    raw: ToolResult,
    declaration: Any,
    now: int,
    call: Optional[StoredToolCall] = None,
    stream_truncated: Optional[Dict[str, int]] = None,
) -> Any:
    """Closure: bound the output, store the model message, rest as strict-JSON data."""
    call = call if call is not None else (_input(task).get("call") or {})

    def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
        bounded = _bound(
            ToolResult(
                content=raw.content if raw.content is not None else [],
                is_error=raw.is_error,
                details=raw.details,
                diagnostics=raw.diagnostics,
                control=raw.control,
            ),
            getattr(declaration, "output", None) if declaration is not None else None,
        )
        result = bounded["result"]
        bytes_dropped = (bounded.get("truncated") or {}).get("bytes", 0) + (
            stream_truncated or {}
        ).get("bytes", 0)
        lines_dropped = (bounded.get("truncated") or {}).get("lines", 0) + (
            stream_truncated or {}
        ).get("lines", 0)
        truncated = {"bytes": bytes_dropped, "lines": lines_dropped} if (
            bytes_dropped > 0 or lines_dropped > 0
        ) else None
        data: JsonObject = {}
        if result.details is not None:
            data["details"] = to_stored(result.details)
        if result.diagnostics is not None:
            data["diagnostics"] = to_stored(result.diagnostics)
        if result.control is not None:
            data["control"] = to_stored(result.control)
        if truncated is not None:
            data["truncated"] = truncated
        entry = tx.append_entry(
            current.conversation_id,
            NewEntry(
                kind="pi.tool_result",
                model=[_to_message(call, result, now)],
                data=data,
            ),
        )
        slot = tx.tool_slot(task)
        slot["status"] = "error" if result.is_error is True else "done"
        slot["entry"] = entry
        slot.pop("waitingOn", None)
        tx.emit(
            {
                "type": "tool.finished",
                "taskId": current.id,
                "callId": call.get("id"),
                "entry": entry,
                "isError": result.is_error is True,
                **({} if result.control is None else {"control": result.control}),
            }
        )
        if truncated is not None:
            tx.emit(
                {
                    "type": "warning",
                    "source": "tool",
                    "message": (
                        f"tool output truncated: {truncated['bytes']} bytes / "
                        f"{truncated['lines']} lines dropped"
                    ),
                }
            )
        return Completion(
            status="completed",
            result=ToolTaskResult(
                entry=entry,
                control=None if result.control is None else result.control,
            ),
        )

    return done


def _bound(result: ToolResult, declared: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    bounds = {**DEFAULT_BOUNDS, **(declared or {})}
    blocks = list(result.content or [])
    texts = [_field(block, "text") or "" for block in blocks if _field(block, "type") == "text"]
    text = "".join(texts)
    dropped_lines = 0
    dropped_bytes = 0
    lines = text.split("\n")
    if len(lines) > bounds["maxLines"]:
        dropped_lines = len(lines) - bounds["maxLines"]
        kept = lines[: bounds["maxLines"]] if bounds["retain"] == "head" else lines[-bounds["maxLines"] :]
        text = "\n".join(kept)
    encoded = text.encode("utf-8")
    if len(encoded) > bounds["maxBytes"]:
        dropped_bytes = len(encoded) - bounds["maxBytes"]
        kept_bytes = (
            encoded[: bounds["maxBytes"]] if bounds["retain"] == "head" else encoded[-bounds["maxBytes"] :]
        )
        text = kept_bytes.decode("utf-8", errors="replace")
    if dropped_bytes == 0 and dropped_lines == 0:
        return {"result": result}
    text_index = -1
    for index, block in enumerate(blocks):
        if _field(block, "type") != "text":
            continue
        if bounds["retain"] == "head" and text_index >= 0:
            continue
        text_index = index
    content: List[Any] = []
    for index, block in enumerate(blocks):
        if _field(block, "type") != "text":
            content.append(block)
        elif index == text_index:
            if isinstance(block, dict):
                content.append({**block, "text": text})
            else:
                content.append({"type": "text", "text": text})
    note = ToolDiagnostic(
        severity="warn",
        code="truncated",
        message=f"output truncated: {dropped_lines} lines, {dropped_bytes} bytes dropped",
    )
    return {
        "result": ToolResult(
            content=content,
            is_error=result.is_error,
            details=result.details,
            diagnostics=[*(result.diagnostics or []), note],
            control=result.control,
        ),
        "truncated": {"bytes": dropped_bytes, "lines": dropped_lines},
    }


def _to_message(call: StoredToolCall, result: ToolResult, now: int) -> JsonObject:
    return to_stored(
        {
            "role": "toolResult",
            "toolCallId": call.get("id"),
            "toolName": call.get("name"),
            "content": result.content or [],
            "isError": result.is_error or False,
            "timestamp": now,
        }
    )


def _input(task: Task) -> Dict[str, Any]:
    if isinstance(task.input, dict):
        return task.input
    return dict(task.input.__dict__) if task.input is not None else {}


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    if hasattr(value, "to_json"):
        return value.to_json().get(key)
    return getattr(value, key, None)


def _with_call_id(api: HookInfo, call_id: Any) -> Any:
    class _Api:
        kind = getattr(api, "kind", "")
        task_id = getattr(api, "task_id", 0)
        conversation_id = getattr(api, "conversation_id", 0)

    _Api.call_id = call_id
    return _Api()


def _task_scan(conversation_id: Id) -> Any:
    from ..types import TaskScan

    return TaskScan(conversation_id=conversation_id, status=["pending", "running"])


async def _noop() -> None:
    return None


async def _resolve(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


_ = (ToolApi, ToolSlot, field)
