"""Storage-backed session ported from ``harness/session/session.ts``."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pi_ai.uuid_utils import uuidv7
from ..._chord.context import Context
from .commit import insert_entry
from .mutation_line import MutationLine
from .types import (
    Branch,
    BranchScan,
    CommitResult,
    Entry,
    EntryQuery,
    EntryScan,
    IdGenerator,
    SessionMetadata,
    SessionStats,
    Storage,
    StorageBranchScan,
    Write,
)
from .values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
    append_list as append_list_write,
    branch_tip,
    delete_list as delete_list_write,
    delete_value as delete_value_write,
    entry_label,
    set_value as set_value_write,
    session_name,
)

__all__ = [
    "StorageBackedSession",
    "SessionInvariantError",
    "SessionInvalidBranchError",
    "SessionBranchExistsError",
    "SessionPendingAssistantMessageError",
    "SessionUnknownTargetError",
]


class SessionInvariantError(RuntimeError):
    """Durable session state is internally inconsistent and cannot be safely advanced."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "SessionInvariantError"


class SessionInvalidBranchError(RuntimeError):
    def __init__(self, branch: str, reason: str) -> None:
        super().__init__(f"Invalid branch {branch!r}: {reason}")
        self.name = "SessionInvalidBranchError"
        self.branch = branch
        self.reason = reason


class SessionBranchExistsError(RuntimeError):
    def __init__(self, branch: str) -> None:
        super().__init__(f"Branch already exists: {branch}")
        self.name = "SessionBranchExistsError"
        self.branch = branch


class SessionPendingAssistantMessageError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Cannot persist a pending assistant message")
        self.name = "SessionPendingAssistantMessageError"


class SessionUnknownTargetError(RuntimeError):
    def __init__(self, target_id: str) -> None:
        super().__init__(f"Unknown target: {target_id}")
        self.name = "SessionUnknownTargetError"
        self.target_id = target_id


class _Uuid7IdGenerator:
    def next(self, timestamp_ms: Optional[int] = None) -> str:
        return uuidv7(timestamp_ms)


class _StorageBackedSessionMutation:
    """One exclusive mutation barrier grant; commit allowed at most once."""

    def __init__(self, storage: Storage, release: Callable[[], None]) -> None:
        self._storage = storage
        self._release = release
        self._active = True
        self._commit_task: Optional[asyncio.Task] = None
        self._end_task: Optional[asyncio.Task] = None

    def commit(self, writes: List[Write], context: Context) -> Awaitable[CommitResult]:
        self._assert_active()
        if self._commit_task is not None:
            raise RuntimeError("SessionMutator commit already attempted")

        async def _commit() -> CommitResult:
            for write in writes:
                if (
                    getattr(write, "kind", None) == "entry"
                    and write.entry.type == "message"
                    and getattr(write.entry.message, "role", None) == "assistant"
                    and getattr(write.entry.message, "stop_reason", None) == "pending"
                ):
                    raise SessionPendingAssistantMessageError()
            return await self._storage.commit(writes, context)

        self._commit_task = asyncio.get_running_loop().create_task(_commit())
        return self._commit_task

    def end(self, _context: Context) -> Awaitable[None]:
        if self._end_task is not None:
            return self._end_task
        self._active = False

        async def _end() -> None:
            if self._commit_task is not None:
                try:
                    await asyncio.shield(self._commit_task)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            self._release()

        self._end_task = asyncio.get_running_loop().create_task(_end())
        return self._end_task

    async def get_entries(self, ids: List[str], context: Context) -> Dict[str, Entry]:
        self._assert_active()
        return await self._storage.get_entries(ids, context)

    async def get_stats(self, context: Context) -> SessionStats:
        self._assert_active()
        return await self._storage.get_stats(context)

    async def get_value(self, address: Value, context: Context) -> Optional[StoredValue]:
        self._assert_active()
        return await self._storage.get_value(address, context)

    async def scan_values(self, prefix: Value, context: Context) -> List[StoredValue]:
        self._assert_active()
        return await self._storage.scan_values(prefix, context)

    async def read_list(
        self, address: ValueList, options: Optional[ListReadOptions], context: Context
    ) -> List[ListElement]:
        self._assert_active()
        return await self._storage.read_list(address, options, context)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> List[Entry]:
        self._assert_active()
        return await self._storage.scan_branch(query, context)

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError("SessionMutator cannot be used outside its mutation callback")


class _StorageBackedBranch:
    def __init__(self, name: str, session: "StorageBackedSession") -> None:
        self._name = name
        self._session = session

    @property
    def name(self) -> str:
        return self._name

    async def get_tip_id(self, context: Context) -> Optional[str]:
        return await self._session.get_branch_tip(self._name, context)

    async def find_entries(self, query: Optional[BranchScan], context: Context) -> List[Entry]:
        query = query or BranchScan()
        start = query.start if query.start is not None else await self.get_tip_id(context)
        if start is None:
            return []
        scan = StorageBranchScan(
            start=start,
            stop_at_type=query.stop_at_type,
            stop_at_id=query.stop_at_id,
            type=query.type,
            custom_type=query.custom_type,
            order=query.order or "newestFirst",
            limit=query.limit,
            cursor=query.cursor,
        )
        return await self._session.scan_branch(scan, context)

    async def find_entry(self, query: Optional[BranchScan], context: Context) -> Optional[Entry]:
        query = query or BranchScan()
        limited = BranchScan(
            start=query.start,
            stop_at_type=query.stop_at_type,
            stop_at_id=query.stop_at_id,
            type=query.type,
            custom_type=query.custom_type,
            order=query.order,
            limit=1 if query.limit is None else min(query.limit, 1),
            cursor=query.cursor,
        )
        entries = await self.find_entries(limited, context)
        return entries[0] if entries else None

    async def append_message(self, message: Any, context: Context) -> str:
        return await self._session.append_to_branch(
            self._name, {"type": "message", "message": message}, context
        )

    async def append_custom_entry(self, custom_type: str, data: Any, context: Context) -> str:
        return await self._session.append_to_branch(
            self._name, {"type": "custom", "customType": custom_type, "data": data}, context
        )


class StorageBackedSession:
    """Package-internal typed boundary shared by concrete session repositories."""

    def __init__(
        self,
        metadata: SessionMetadata,
        storage: Storage,
        options: Optional[dict] = None,
    ) -> None:
        options = options or {}
        self.metadata = metadata
        self.id_generator: IdGenerator = options.get("id_generator") or _Uuid7IdGenerator()
        self._storage = storage
        self._mutation_line: MutationLine = options.get("mutation_line") or MutationLine()
        self._on_close = options.get("on_close")
        # A facade shares its storage with the owning repo, which closes it.
        self._owns_storage = options.get("owns_storage", True)
        self._branches: Dict[str, _StorageBackedBranch] = {}
        self._state = "open"
        self._close_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def begin_mutation(self, _context: Context):
        self._assert_open()
        granted: asyncio.Future = asyncio.get_running_loop().create_future()
        released = asyncio.Event()

        def _release() -> None:
            released.set()

        async def _line_job() -> None:
            granted.set_result(_StorageBackedSessionMutation(self._storage, _release))
            await released.wait()

        async def _run_line() -> None:
            await self._mutation_line.run(_line_job)

        line_task = asyncio.get_running_loop().create_task(_run_line())

        def _on_line_failed(task: asyncio.Task) -> None:
            if task.cancelled():
                if not granted.done():
                    granted.cancel()
                return
            error = task.exception()
            if error is not None and not granted.done():
                granted.set_exception(error)

        line_task.add_done_callback(_on_line_failed)
        return await granted

    async def mutate(self, mutation: Callable, context: Context) -> Any:
        mutator = await self.begin_mutation(context)
        try:
            result = mutation(mutator, context)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            end = mutator.end(context)
            if asyncio.iscoroutine(end) or asyncio.isfuture(end):
                await end

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_entries(self, ids: List[str], context: Context) -> Dict[str, Entry]:
        self._assert_open()
        return await self._storage.get_entries(ids, context)

    async def get_entry(self, entry_id: str, context: Context) -> Optional[Entry]:
        return (await self.get_entries([entry_id], context)).get(entry_id)

    async def get_value(self, address: Value, context: Context) -> Optional[StoredValue]:
        self._assert_open()
        return await self._storage.get_value(address, context)

    async def scan_values(self, prefix: Value, context: Context) -> List[StoredValue]:
        self._assert_open()
        return await self._storage.scan_values(prefix, context)

    async def read_list(
        self, address: ValueList, options: Optional[ListReadOptions], context: Context
    ) -> List[ListElement]:
        self._assert_open()
        return await self._storage.read_list(address, options, context)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> List[Entry]:
        self._assert_open()
        return await self._storage.scan_branch(query, context)

    async def get_stats(self, context: Context) -> SessionStats:
        self._assert_open()
        return await self._storage.get_stats(context)

    async def get_name(self, context: Context) -> Optional[str]:
        stored = await self.get_value(session_name(), context)
        return stored.value if stored is not None else None

    async def get_label(self, target_id: str, context: Context) -> Optional[str]:
        stored = await self.get_value(entry_label(target_id), context)
        return stored.value if stored is not None else None

    async def find_entries(self, query: Optional[EntryQuery], context: Context) -> List[Entry]:
        query = query or EntryQuery()
        self._assert_open()
        order = query.order or "desc"
        if query.cursor is not None:
            if order == "asc" and query.cursor.seq >= (2**53 - 1):
                return []
            if order == "desc" and query.cursor.seq <= 1:
                return []
        scan = EntryScan(
            type=query.type,
            custom_type=query.custom_type,
            order=order,
            limit=query.limit,
            from_seq=(query.cursor.seq + 1) if (query.cursor is not None and order == "asc") else None,
            to_seq=(query.cursor.seq - 1) if (query.cursor is not None and order == "desc") else None,
        )
        return await self._storage.scan_entries(scan, context)

    async def find_entry(self, query: Optional[EntryQuery], context: Context) -> Optional[Entry]:
        query = query or EntryQuery()
        limited = EntryQuery(
            type=query.type,
            custom_type=query.custom_type,
            order=query.order,
            limit=1 if query.limit is None else min(query.limit, 1),
            cursor=query.cursor,
        )
        entries = await self.find_entries(limited, context)
        return entries[0] if entries else None

    # ------------------------------------------------------------------
    # Branches
    # ------------------------------------------------------------------

    async def branch(self, name: str, context: Context) -> Optional[Branch]:
        self._assert_valid_branch_name(name)
        if await self.get_value(branch_tip(name), context) is None:
            return None
        return self._get_or_create_branch_object(name)

    async def create_branch(self, name: str, at: Optional[str], context: Context) -> Branch:
        self._assert_open()
        self._assert_valid_branch_name(name)

        async def _mutation(mutator, mutation_context):
            if await mutator.get_value(branch_tip(name), mutation_context) is not None:
                raise SessionBranchExistsError(name)
            if at is not None and at not in await mutator.get_entries([at], mutation_context):
                raise SessionUnknownTargetError(at)
            await mutator.commit([set_value_write(branch_tip(name), at)], mutation_context)

        await self.mutate(_mutation, context)
        return self._get_or_create_branch_object(name)

    # ------------------------------------------------------------------
    # Public writers (queue behind mutation line)
    # ------------------------------------------------------------------

    async def set_value(self, address: Value, next_value: Any, context: Context) -> None:
        async def _mutation(mutator, mutation_context):
            await mutator.commit([set_value_write(address, next_value)], mutation_context)

        await self.mutate(_mutation, context)

    async def delete_value(self, address: Value, context: Context) -> None:
        async def _mutation(mutator, mutation_context):
            await mutator.commit([delete_value_write(address)], mutation_context)

        await self.mutate(_mutation, context)

    async def append_list(self, address: ValueList, element: Any, context: Context) -> None:
        async def _mutation(mutator, mutation_context):
            await mutator.commit([append_list_write(address, element)], mutation_context)

        await self.mutate(_mutation, context)

    async def delete_list(self, address: ValueList, context: Context) -> None:
        async def _mutation(mutator, mutation_context):
            await mutator.commit([delete_list_write(address)], mutation_context)

        await self.mutate(_mutation, context)

    async def set_name(self, name: Optional[str], context: Context) -> None:
        if name is None:
            await self.delete_value(session_name(), context)
        else:
            await self.set_value(session_name(), name, context)

    async def set_label(self, target_id: str, label: Optional[str], context: Context) -> None:
        address = entry_label(target_id)
        if label is None:
            await self.delete_value(address, context)
        else:
            await self.set_value(address, label, context)

    async def close(self, context: Context) -> None:
        if self._close_task is not None:
            await self._close_task
            return
        self._state = "closing"

        async def _close() -> None:
            await self._mutation_line.seal(RuntimeError("Session is closed"))
            if self._owns_storage:
                await self._storage.close(context)
            self._state = "closed"
            if self._on_close is not None:
                self._on_close()

        self._close_task = asyncio.get_running_loop().create_task(_close())
        await self._close_task

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def get_branch_tip(self, name: str, context: Context) -> Optional[str]:
        stored = await self.get_value(branch_tip(name), context)
        if stored is None:
            raise SessionInvariantError(f"Unknown branch: {name}")
        return stored.value

    async def append_to_branch(self, name: str, entry: dict, context: Context) -> str:
        self._assert_open()
        if (
            entry.get("type") == "message"
            and getattr(entry.get("message"), "role", None) == "assistant"
            and getattr(entry.get("message"), "stop_reason", None) == "pending"
        ):
            raise SessionPendingAssistantMessageError()
        entry_id = self.id_generator.next()

        async def _mutation(mutator, mutation_context):
            tip = await mutator.get_value(branch_tip(name), mutation_context)
            if tip is None:
                raise SessionInvariantError(f"Unknown branch: {name}")
            if entry.get("type") == "message":
                from .types import MessageEntry

                new_entry = MessageEntry(id=entry_id, parent_id=tip.value, message=entry["message"])
            else:
                from .types import CustomEntry

                new_entry = CustomEntry(
                    id=entry_id, parent_id=tip.value, custom_type=entry.get("customType", ""),
                    data=entry.get("data"),
                )
            await mutator.commit(
                [insert_entry(new_entry), set_value_write(branch_tip(name), entry_id)],
                mutation_context,
            )

        await self.mutate(_mutation, context)
        return entry_id

    def _get_or_create_branch_object(self, name: str) -> Branch:
        branch = self._branches.get(name)
        if branch is None:
            branch = _StorageBackedBranch(name, self)
            self._branches[name] = branch
        return branch

    def _assert_valid_branch_name(self, name: str) -> None:
        if len(name) == 0:
            raise SessionInvalidBranchError(name, "branch name must not be empty")
        if "\x00" in name:
            raise SessionInvalidBranchError(name, "branch name must not contain \\u0000")

    def _assert_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("Session is closed")
