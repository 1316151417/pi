"""Post-tools task kind ported from ``harness/pico3/kinds/post-tools.ts``.

Runs after a generation's tool tasks settle: folds their control into the
rewindable state, synthesizes results for orphaned or aborted calls, applies the
final boundary, and hands the turn to its successor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...._chord.context import Context
from ..types import (
    Completion,
    HookInfo,
    Id,
    Kind,
    KindConfig,
    NewEntry,
    Step,
    Task,
    TaskSpec,
    to_stored,
)

__all__ = [
    "PostToolsInput",
    "PostToolsResult",
    "PostToolsHooks",
    "post_tools_config",
    "post_tools",
    "post_tools_kind",
]


@dataclass
class PostToolsInput:
    inputs: List[Id] = field(default_factory=list)
    assistant: Id = 0
    tools: List[Id] = field(default_factory=list)

    def to_json(self) -> JsonObject:
        return {"inputs": list(self.inputs), "assistant": self.assistant, "tools": list(self.tools)}


@dataclass
class PostToolsResult:
    successor: Optional[Id] = None
    ended: Optional[str] = None  # "terminate" | "handoff"

    def to_json(self) -> JsonObject:
        data: JsonObject = {}
        if self.successor is not None:
            data["successor"] = self.successor
        if self.ended is not None:
            data["ended"] = self.ended
        return data


class PostToolsHooks:
    """Hook points this kind calls: ``after_tools``."""

    async def after_tools(
        self, assistant: Id, results: List[Id], info: HookInfo, ctx: Context
    ) -> None: ...


#: Declared configuration this kind reads.
post_tools_config = {
    "sticky": {"steeringMode": "one-at-a-time", "followUpMode": "one-at-a-time"}
}


async def _initial(task: Task, rt: Any, ctx: Context) -> Any:
    input_record = _input(task)

    async def read(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> List[Dict[str, Any]]:
        assistant = await tx.entry(input_record.get("assistant"))
        calls = [
            block
            for block in _content(_first_message(assistant))
            if _field(block, "type") == "toolCall"
        ]
        rows: List[Dict[str, Any]] = []
        for index, id in enumerate(input_record.get("tools") or []):
            tool_task = await tx.task(id)
            call = calls[index] if index < len(calls) else None
            outcome = tool_task.outcome if tool_task is not None else None
            if outcome is not None and outcome.status == "completed":
                result = outcome.result or {}
                rows.append(
                    {
                        "call": call,
                        "entry": result.get("entry"),
                        "control": result.get("control"),
                        "missing": None,
                    }
                )
            elif outcome is not None and outcome.status == "aborted":
                aborted = outcome.result or {}
                rows.append(
                    {
                        "call": call,
                        "entry": aborted.get("entry"),
                        "control": None,
                        "missing": None if aborted.get("entry") is not None else "aborted",
                    }
                )
            else:
                rows.append({"call": call, "entry": None, "control": None, "missing": "orphaned"})
        return rows

    rows = await rt.commit(read, ctx)
    results = [row["entry"] for row in rows if row["entry"] is not None]

    async def notify(hook: Any, api: HookInfo) -> None:
        handler = getattr(hook, "after_tools", None)
        if handler is None:
            return
        value = handler(input_record.get("assistant"), results, api, ctx)
        if hasattr(value, "__await__"):
            await value

    await rt.hooks.each(ctx, notify)

    async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> Completion:
        conversation_id = current.conversation_id
        head = await tx.newest_entry(conversation_id, {"withHead": True})
        state = tx.rewindable(conversation_id)
        selected_tools = list(state.get("selectedTools") or [])
        terminate = False
        handoff: Optional[str] = None
        for row in rows:
            control = row["control"] or {}
            for name in control.get("addTools") or []:
                if name not in selected_tools:
                    selected_tools.append(name)
            if control.get("terminate"):
                terminate = True
            if control.get("handoff") is not None:
                handoff = control["handoff"]
        if len(selected_tools) != len(state.get("selectedTools") or []):
            state["selectedTools"] = selected_tools
            tx.emit({"type": "config.changed", "keys": ["selectedTools"]})
        for row in rows:
            if row["missing"] is None:
                continue
            call = row["call"] or {}
            tx.append_entry(
                conversation_id,
                NewEntry(
                    kind="pi.tool_result",
                    model=[
                        to_stored(
                            {
                                "role": "toolResult",
                                "toolCallId": call.get("id"),
                                "toolName": call.get("name"),
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Tool result unavailable: task {row['missing']}.",
                                    }
                                ],
                                "isError": True,
                                "timestamp": rt.now(),
                            }
                        )
                    ],
                    data={
                        "diagnostics": [
                            {
                                "severity": "error",
                                "message": row["missing"],
                                "code": row["missing"],
                            }
                        ]
                    },
                ),
            )
            tx.emit(
                {
                    "type": "warning",
                    "source": "post_tools",
                    "message": f"synthesized {row['missing']} tool result for {call.get('id')}",
                }
            )
        head_boundary = head.id if head is not None else None
        if handoff is not None:
            head_boundary = tx.append_entry(
                conversation_id,
                NewEntry(
                    kind="pi.handoff",
                    head="self",
                    model=[{"role": "user", "content": handoff, "timestamp": rt.now()}],
                ),
            )
        inputs = input_record.get("inputs") or []
        assistant = input_record.get("assistant")
        if handoff is not None or terminate:
            tx.sticky(conversation_id)["turn"] = {"tools": []}
            await tx.resolve_inputs(inputs, {"status": "done", "answer": assistant})
            tx.emit(
                {"type": "turn.ended", "inputs": inputs, "status": "done", "answer": assistant}
            )
            boundary = await tx.boundary(conversation_id, "final", head_boundary)
            triggers = boundary["triggers"]
            if triggers:
                tx.emit({"type": "turn.started", "inputs": triggers})
            ended = "handoff" if handoff is not None else "terminate"
            if triggers:
                return Completion(
                    status="completed",
                    result=PostToolsResult(
                        ended=ended,
                        successor=tx.create_task(TaskSpec(kind="pi.generation", input={"inputs": triggers})),
                    ),
                )
            return Completion(status="completed", result=PostToolsResult(ended=ended))
        boundary = await tx.boundary(conversation_id, "postTools", head_boundary)
        triggers = boundary["triggers"]
        if boundary["terminated"]:
            tx.sticky(conversation_id)["turn"] = {"tools": []}
            await tx.resolve_inputs(inputs, {"status": "unanswered", "reason": "terminated"})
            tx.emit(
                {
                    "type": "turn.ended",
                    "inputs": inputs,
                    "status": "unanswered",
                    "reason": "terminated",
                }
            )
            if triggers:
                tx.emit({"type": "turn.started", "inputs": triggers})
                return Completion(
                    status="completed",
                    result=PostToolsResult(
                        successor=tx.create_task(TaskSpec(kind="pi.generation", input={"inputs": triggers}))
                    ),
                )
            return Completion(status="completed", result=PostToolsResult())
        return Completion(
            status="completed",
            result=PostToolsResult(
                successor=tx.create_task(
                    TaskSpec(kind="pi.generation", input={"inputs": [*inputs, *triggers]})
                )
            ),
        )

    return Step(done=done)


async def _abort(task: Task, _rt: Any, _ctx: Context) -> Any:
    input_record = _input(task)

    async def done(tx: Any, current: Optional[Task] = None, _ctx: Any = None) -> None:
        tx.sticky(current.conversation_id)["turn"] = {"tools": []}
        await tx.resolve_inputs(
            input_record.get("inputs") or [], {"status": "unanswered", "reason": "aborted"}
        )
        tx.emit(
            {
                "type": "turn.ended",
                "inputs": input_record.get("inputs") or [],
                "status": "unanswered",
                "reason": "aborted",
            }
        )
        return None

    return done


post_tools_kind = Kind(
    name="pi.post_tools",
    turn=True,
    config=KindConfig(sticky=dict(post_tools_config["sticky"])),
    phases={},
    initial=_initial,
    abort=_abort,
)

#: The built-in kind token.
post_tools = post_tools_kind


def _input(task: Task) -> Dict[str, Any]:
    if isinstance(task.input, dict):
        return task.input
    return dict(task.input.__dict__) if task.input is not None else {}


def _first_message(entry: Any) -> Any:
    model = getattr(entry, "model", None) if entry is not None else None
    if not model:
        return {}
    return model[0]


def _content(message: Any) -> List[Any]:
    if isinstance(message, dict):
        return message.get("content") or []
    return getattr(message, "content", None) or []


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
