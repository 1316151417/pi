"""Task-facing tool API ported from ``harness/pico3/kinds/task-api.ts``.

Ordinary tasks get the same handle a tool gets, except that the tool-only
surfaces (``stream``/``progress``/``memo``) fail loudly instead of silently
doing nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...._chord.context import Context
from pi_ai.types import JsonValue
from ..types import AnyKind, DocRef, Id, OwnedConversationSpec, SendInput, Task, TaskRef

__all__ = ["task_api"]


def _is_kind(value: Any) -> bool:
    """A kind token is an object carrying ``name`` and ``initial``."""
    return (
        value is not None
        and hasattr(value, "name")
        and hasattr(value, "initial")
    )


class _OwnedConversation:
    """The narrow handle a task gets to a conversation it owns."""

    def __init__(self, id: Id, runtime: Any) -> None:
        self.id = id
        self._runtime = runtime

    async def send(self, input: SendInput, ctx: Context) -> Dict[str, Any]:
        input_id = await self._runtime.send_owned(self.id, input, ctx)

        async def wait(wait_ctx: Context) -> Any:
            return await self._runtime.wait_for_input(input_id, wait_ctx)

        async def result(result_ctx: Context) -> Any:
            return await self._runtime.commit(
                lambda tx, current=None, _ctx=None: tx.input(input_id), result_ctx
            )

        return {"id": input_id, "wait": wait, "result": result}

    async def abort(self, ctx: Context) -> None:
        await self._runtime.abort_conversation(self.id, ctx)


class TaskApi:
    """``ToolApi`` bound to a task rather than to one tool invocation."""

    def __init__(self, task: Task, runtime: Any) -> None:
        self.task_id = task.id
        self.conversation_id = task.conversation_id
        self.call_id = ""
        self._runtime = runtime

    def stream(self, chunk: Any) -> None:
        raise RuntimeError("stream is only available to tools")

    def progress(self, update: Any, ctx: Optional[Context] = None) -> Any:
        raise RuntimeError("progress is only available to tools")

    def memo(self, name: str, *args: Any) -> Any:
        raise RuntimeError("memo is only available to tools")

    async def conversation(self, spec: OwnedConversationSpec, ctx: Context) -> _OwnedConversation:
        id = await self._runtime.create_owned_conversation(spec, ctx)
        return _OwnedConversation(id, self._runtime)

    async def task(
        self,
        kind: Any,
        input: JsonValue,
        opts: Dict[str, Any],
        ctx: Context,
    ) -> TaskRef:
        if not _is_kind(kind):
            raise ValueError("task(): pass a kind token")
        return await self._runtime.commit(
            lambda tx, current=None, _ctx=None: tx.create_task(kind, input, opts), ctx
        )

    async def get_task(self, ref: TaskRef, ctx: Context) -> Optional[Task]:
        return await self._runtime.commit(
            lambda tx, current=None, _ctx=None: tx.task(ref.id), ctx
        )

    async def wait_for_task(self, ref: TaskRef, ctx: Context) -> Task:
        return await self._runtime.wait_for_task(ref.id, ctx)

    async def slot(self, ref: TaskRef, ctx: Context) -> Optional[Dict[str, Any]]:
        async def read(tx: Any, current: Any = None, _ctx: Any = None) -> Optional[Dict[str, Any]]:
            stored_task = await tx.task(ref.id)
            if stored_task is None:
                return None
            snapshot = tx.snapshot(
                DocRef(doc="sticky", conversation_id=stored_task.conversation_id)
            )
            return (snapshot.get("tasks") or {}).get(ref.id)

        return await self._runtime.commit(read, ctx)


def task_api(task: Task, runtime: Any) -> TaskApi:
    """Build the task-scoped API exposed to kind handlers."""
    return TaskApi(task, runtime)


_ = (AnyKind, List)
