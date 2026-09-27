"""Admission and close lifecycle around the shared StorageBackedSession."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from pi_agent_core.harness.session.session import StorageBackedSession
from pi_agent_core.harness.session.types import Branch, CommitResult, EntryQuery, SessionMutation, SessionStats, StorageBranchScan, Write
from pi_agent_core.harness.session.values import ListElement, ListReadOptions, StoredValue, Value, ValueList
from pi_chord.context import Context

from ._async import Promise, promise, rejected, settled_all, spawn
from ._json import Record
from ._rows import SqliteSessionMetadata


@dataclass
class SqliteOpenSessionOptions:
    on_close: Callable[[], None]


class _Mutation:
    def __init__(self, source: SessionMutation, finished: Promise[None], owner: SqliteOpenSession) -> None:
        self._source, self._finished, self._owner = source, finished, owner
        self._ended = False

    def commit(self, writes: list[Write], context: Context) -> Awaitable[CommitResult]:
        return self._source.commit(writes, context)

    def end(self, context: Context) -> Promise[None]:
        async def end() -> None:
            try:
                await self._source.end(context)
            finally:
                if not self._ended:
                    self._ended = True
                    self._owner._admitted.pop(self._finished, None)
                    self._finished.future.set_result(None)
        return spawn(end())

    def get_entries(self, ids: list[str], context: Context) -> Awaitable[dict[str, Record]]:
        return self._source.get_entries(ids, context)

    def get_stats(self, context: Context) -> Awaitable[SessionStats]:
        return self._source.get_stats(context)

    def get_value[T](self, address: Value[T], context: Context) -> Awaitable[StoredValue[T] | None]:
        return self._source.get_value(address, context)

    def scan_values[T](self, prefix: Value[T], context: Context) -> Awaitable[list[StoredValue[T]]]:
        return self._source.scan_values(prefix, context)

    def read_list[T](self, address: ValueList[T], options: ListReadOptions | None, context: Context) -> Awaitable[list[ListElement[T]]]:
        return self._source.read_list(address, options, context)

    def scan_branch(self, query: StorageBranchScan, context: Context) -> Awaitable[list[Record]]:
        return self._source.scan_branch(query, context)


class _Branch:
    def __init__(self, source: Branch, owner: SqliteOpenSession) -> None:
        self.name = source.name
        self._source, self._owner = source, owner

    def get_tip_id(self, context: Context) -> Promise[str | None]:
        return self._owner._admit(lambda: self._source.get_tip_id(context))

    def find_entries(self, query: EntryQuery | None, context: Context) -> Promise[list[Record]]:
        return self._owner._admit(lambda: self._source.find_entries(query, context))

    def find_entry(self, query: EntryQuery | None, context: Context) -> Promise[Record | None]:
        return self._owner._admit(lambda: self._source.find_entry(query, context))

    def append_message(self, message: object, context: Context) -> Promise[Record]:
        return self._owner._admit(lambda: self._source.append_message(message, context))

    def append_custom_entry(self, custom_type: str, data: object, context: Context) -> Promise[Record]:
        return self._owner._admit(lambda: self._source.append_custom_entry(custom_type, data, context))


class SqliteOpenSession:
    def __init__(self, session: StorageBackedSession, options: SqliteOpenSessionOptions) -> None:
        self._session = session
        self.metadata = cast(SqliteSessionMetadata, session.metadata)
        self.id_generator = session.id_generator
        self._on_close = options.on_close
        self._admitted: dict[Promise, None] = {}
        self._closed_error = RuntimeError("Session is closed")
        self._state = "open"
        self._close_promise: Promise[None] | None = None

    def _admit[T](self, operation: Callable[[], Awaitable[T]]) -> Promise[T]:
        if self._state != "open":
            return rejected(self._closed_error)
        try:
            result = promise(operation())
        except BaseException as error:
            result = rejected(error)
        self._admitted[result] = None
        result.then_settled(lambda: self._admitted.pop(result, None))
        return result

    def begin_mutation(self, context: Context) -> Promise[SessionMutation]:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        finished = Promise(future)
        self._admitted[finished] = None
        admitted = self._admit(lambda: self._session.begin_mutation(context))
        async def begin() -> SessionMutation:
            try:
                source = await admitted
            except BaseException:
                self._admitted.pop(finished, None)
                future.set_result(None)
                raise
            if self._state != "open":
                await source.end(context)
                self._admitted.pop(finished, None)
                future.set_result(None)
                raise self._closed_error
            return cast(SessionMutation, _Mutation(source, finished, self))
        return spawn(begin())

    def mutate[T](self, mutation: Callable[[SessionMutation, Context], T | Awaitable[T]], context: Context) -> Promise[T]:
        def callback(mutator: SessionMutation, mutation_context: Context) -> T | Awaitable[T]:
            if self._state != "open":
                raise self._closed_error
            return mutation(mutator, mutation_context)
        return self._admit(lambda: self._session.mutate(callback, context))

    def get_entries(self, ids: list[str], context: Context) -> Promise[dict[str, Record]]:
        return self._admit(lambda: self._session.get_entries(ids, context))

    def get_entry(self, entry_id: str, context: Context) -> Promise[Record | None]:
        return self._admit(lambda: self._session.get_entry(entry_id, context))

    def get_value[T](self, address: Value[T], context: Context) -> Promise[StoredValue[T] | None]:
        return self._admit(lambda: self._session.get_value(address, context))

    def scan_values[T](self, prefix: Value[T], context: Context) -> Promise[list[StoredValue[T]]]:
        return self._admit(lambda: self._session.scan_values(prefix, context))

    def read_list[T](self, address: ValueList[T], options: ListReadOptions | None, context: Context) -> Promise[list[ListElement[T]]]:
        return self._admit(lambda: self._session.read_list(address, options, context))

    def scan_branch(self, query: StorageBranchScan, context: Context) -> Promise[list[Record]]:
        return self._admit(lambda: self._session.scan_branch(query, context))

    def get_stats(self, context: Context) -> Promise[SessionStats]:
        return self._admit(lambda: self._session.get_stats(context))

    def get_name(self, context: Context) -> Promise[str | None]:
        return self._admit(lambda: self._session.get_name(context))

    def get_label(self, target_id: str, context: Context) -> Promise[str | None]:
        return self._admit(lambda: self._session.get_label(target_id, context))

    def find_entries(self, query: EntryQuery | None, context: Context) -> Promise[list[Record]]:
        return self._admit(lambda: self._session.find_entries(query, context))

    def find_entry(self, query: EntryQuery | None, context: Context) -> Promise[Record | None]:
        return self._admit(lambda: self._session.find_entry(query, context))

    def branch(self, name: str, context: Context) -> Promise[Branch | None]:
        admitted = self._admit(lambda: self._session.branch(name, context))
        async def branch() -> Branch | None:
            source = await admitted
            return None if source is None else cast(Branch, _Branch(source, self))
        return spawn(branch())

    def create_branch(self, name: str, at: str | None, context: Context) -> Promise[Branch]:
        admitted = self._admit(lambda: self._session.create_branch(name, at, context))
        async def create() -> Branch:
            return cast(Branch, _Branch(await admitted, self))
        return spawn(create())

    def set_value[T](self, address: Value[T], next_value: T, context: Context) -> Promise[None]:
        return self._admit(lambda: self._session.set_value(address, next_value, context))

    def delete_value[T](self, address: Value[T], context: Context) -> Promise[None]:
        return self._admit(lambda: self._session.delete_value(address, context))

    def append_list[T](self, address: ValueList[T], element: T, context: Context) -> Promise[None]:
        return self._admit(lambda: self._session.append_list(address, element, context))

    def delete_list[T](self, address: ValueList[T], context: Context) -> Promise[None]:
        return self._admit(lambda: self._session.delete_list(address, context))

    def set_name(self, name: str | None, context: Context) -> Promise[None]:
        return self._admit(lambda: self._session.set_name(name, context))

    def set_label(self, target_id: str, label: str | None, context: Context) -> Promise[None]:
        return self._admit(lambda: self._session.set_label(target_id, label, context))

    def close(self, context: Context) -> Promise[None]:
        if self._close_promise is not None:
            return self._close_promise
        self._state = "closing"
        admitted = list(self._admitted)
        async def close() -> None:
            try:
                await settled_all(admitted)
                await self._session.close(context)
            finally:
                self._state = "closed"
                self._on_close()
        self._close_promise = spawn(close())
        return self._close_promise
