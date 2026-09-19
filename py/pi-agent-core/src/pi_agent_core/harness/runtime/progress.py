"""Progress channels ported from ``harness/runtime/progress.ts``.

Frames and partial tool outputs are appended to durable lists while an
operation owns them; each write passes through the lane's serialized command
line and is skipped once the operation no longer owns the address.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Generic, List, Optional, TypeVar

from ..._chord.context import Context
from .types import CommitDecision, LaneReturn
from ..session.values import (
    ListCursor,
    ListReadOptions,
    append_list,
    pending_assistant_frames,
    pending_tool_output,
    set_value,
)
from ..events import HarnessEvent  # noqa: F401  (keeps module import graph aligned with TS)

__all__ = [
    "ProgressChannel",
    "read_assistant_frames",
    "open_frame_progress",
    "open_tool_progress",
    "FRAME_PAGE_SIZE",
]

T = TypeVar("T")

#: Page size used when draining a durable frame list.
FRAME_PAGE_SIZE = 1000


class ProgressChannel(Generic[T]):
    """Write/seal/drain channel for one streaming progress stream."""

    def __init__(
        self,
        write_impl: Callable[[T], None],
        seal_impl: Callable[[], None],
        drain_impl: Callable[[], Awaitable[None]],
    ) -> None:
        self._write = write_impl
        self._seal = seal_impl
        self._drain = drain_impl

    def write(self, item: T) -> None:
        self._write(item)

    def seal(self) -> None:
        self._seal()

    async def drain(self) -> None:
        await self._drain()


async def read_assistant_frames(
    reader: Any, operation_id: str, response_entry_id: str, context: Context
) -> List[Any]:
    """Drain every durable assistant frame for one response entry, oldest first."""
    frames: List[Any] = []
    cursor: Optional[ListCursor] = None
    while True:
        page = await reader.read_list(
            pending_assistant_frames(operation_id, response_entry_id),
            ListReadOptions(cursor=cursor, order="asc", limit=FRAME_PAGE_SIZE),
            context,
        )
        frames.extend(element.value for element in page)
        if len(page) < FRAME_PAGE_SIZE:
            return frames
        cursor = ListCursor(seq=page[-1].seq)


def _open_progress(lane: Any, drive: Any, commit_write: Callable[[Any], Any], still_owns: Callable[[Any], bool]) -> ProgressChannel:
    sealed = False
    latest: Optional[asyncio.Task] = None

    def _write(item: Any) -> None:
        nonlocal latest
        if sealed:
            return

        async def _command() -> None:
            def _decision(projection: Any, _reader: Any) -> Any:
                if not still_owns(projection):
                    return LaneReturn(result=None)
                return CommitDecision(
                    writes=[commit_write(item)],
                    next=projection,
                    materialize=lambda _commit: None,
                )

            await lane.command(_decision, drive.context)

        task = asyncio.get_running_loop().create_task(_command())
        latest = task
        task.add_done_callback(lambda finished: _swallow(finished))

    def _seal() -> None:
        nonlocal sealed
        sealed = True

    async def _drain() -> None:
        if latest is not None:
            try:
                await asyncio.shield(latest)
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - drain is best effort
                pass

    return ProgressChannel(write_impl=_write, seal_impl=_seal, drain_impl=_drain)


def _swallow(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


def open_frame_progress(lane: Any, drive: Any, response_entry_id: str) -> ProgressChannel:
    """Open the assistant-frame channel for one response entry."""
    address = pending_assistant_frames(drive.operation_id, response_entry_id)

    def _still_owns(state: Any) -> bool:
        operation = state.operation
        if operation is None:
            return False
        run = operation.state
        return (
            getattr(run, "at", None) in ("assistant.effect_pending", "deferred.effect_pending")
            and getattr(run, "response_entry_id", None) == response_entry_id
        )

    return _open_progress(
        lane, drive, lambda frame: append_list(address, frame), _still_owns
    )


def open_tool_progress(
    lane: Any, drive: Any, turn_id: str, source_index: int, invocation_id: str
) -> ProgressChannel:
    """Open the partial tool-output channel for one invocation."""
    address = pending_tool_output(drive.operation_id, invocation_id)

    def _still_owns(state: Any) -> bool:
        operation = state.operation
        if operation is None or getattr(operation.state, "at", None) != "tools":
            return False
        batch = operation.state.batch
        return getattr(batch, "turn_id", None) == turn_id and any(
            call.source_index == source_index
            and call.result_entry_id == invocation_id
            and call.status == "effect_pending"
            for call in batch.calls
        )

    return _open_progress(lane, drive, lambda snapshot: set_value(address, snapshot), _still_owns)
