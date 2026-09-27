"""SQLite session repository and consistent local/external fork snapshots."""

from __future__ import annotations

import asyncio
import base64
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from pi_agent_core.harness.session.fork import ForkSourceSnapshot, create_fork_snapshot
from pi_agent_core.harness.session.session import StorageBackedSession
from pi_agent_core.harness.session.types import ForkOptions, SessionCreateOptions
from pi_agent_core.harness.session.values import StoredValue
from pi_ai.uuid_utils import uuidv7
from pi_chord.context import Context

from ._async import Promise, rejected, settled_all, spawn
from ._branches import append_entry_to_branch_index
from ._json import Record, field
from ._rows import (EntryRowWriter, SqliteSessionMetadata, delete_session_rows, has_session_row, insert_session_row,
                    metadata_from_session_row, read_all_session_rows, read_session_row)
from ._values import read_all_scalar_value_rows, set_scalar_value_row
from .migrations import apply_initial_schema
from .session import SqliteOpenSession, SqliteOpenSessionOptions
from .sql import SqlQuery
from .storage import SqliteStorage, SqliteStorageOptions, SqliteStorageSnapshot, date_now, read_snapshot_entries
from .types import SqliteDatabase, SqliteDatabaseFactory

SQLITE_STORAGE_VERSION = 1
SQLITE_SESSION_EXTENSION = ".sqlite"
SqliteSessionCreateOptions = SessionCreateOptions
_SAFE_SESSION_FILE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


@dataclass
class SqliteSessionRepoOptions:
    directory: str
    database_factory: SqliteDatabaseFactory
    database_path: str | None = None
    now: Callable[[], int] | None = None


@dataclass
class _ForkSnapshot:
    entries: list[Record]
    scalar_values: list[StoredValue[object]]
    message_count: int
    next_seq: int


def _session_file_name(session_id: str) -> str:
    if _SAFE_SESSION_FILE_ID.fullmatch(session_id):
        return session_id + SQLITE_SESSION_EXTENSION
    encoded = base64.urlsafe_b64encode(session_id.encode("utf-16-le", "surrogatepass")).decode("ascii").rstrip("=")
    return "~" + encoded + SQLITE_SESSION_EXTENSION


async def _remove_session_files(path: str, *, force: bool) -> None:
    for suffix, ignore_missing in (("", force), ("-wal", True), ("-shm", True)):
        try:
            await asyncio.to_thread(os.unlink, path + suffix)
        except FileNotFoundError:
            if not ignore_missing:
                raise


def _build_fork_snapshot(source: SqliteStorageSnapshot, options: ForkOptions) -> _ForkSnapshot:
    snapshot = create_fork_snapshot(ForkSourceSnapshot(entries=source.entries, scalar_values=source.scalar_values,
                                                       entries_complete=source.entries_complete), options)
    entries = sorted(snapshot.entries.values(), key=lambda entry: entry.seq)
    return _ForkSnapshot(entries, snapshot.scalar_values, sum(field(entry, "type") == "message" for entry in entries), snapshot.next_seq)


def _read_external_snapshot(db: SqliteDatabase, source: SqliteSessionMetadata, options: ForkOptions) -> _ForkSnapshot:
    db.exec("BEGIN")
    committed = False
    try:
        metadata_from_session_row(source.path, read_session_row(db, source.id), SQLITE_STORAGE_VERSION)
        values = read_all_scalar_value_rows(db, source.id)
        snapshot = _build_fork_snapshot(SqliteStorageSnapshot(
            read_snapshot_entries(db, source.id, values, options), values, options.scope == "tree"), options)
        db.exec("COMMIT")
        committed = True
        return snapshot
    except BaseException:
        if not committed:
            db.exec("ROLLBACK")
        raise


class SqliteSessionRepo:
    def __init__(self, options: SqliteSessionRepoOptions) -> None:
        self._directory = options.directory
        self._database_path = options.database_path
        self._database_factory = options.database_factory
        self._now = options.now if options.now is not None else date_now
        self._pending_ids: set[str] = set()
        self._open_storages: dict[tuple[str, str], SqliteStorage] = {}
        self._open_sessions: dict[SqliteOpenSession, None] = {}
        self._closed = False
        self._close_promise: Promise[None] | None = None

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("SqliteSessionRepo is closed")

    def _reserve_id(self, session_id: str) -> None:
        if session_id in self._pending_ids:
            raise RuntimeError(f"Session is already open: {session_id}")
        self._pending_ids.add(session_id)

    def _path_for_session(self, session_id: str) -> str:
        if self._database_path is not None:
            return self._database_path
        return os.path.normpath(os.path.join(self._directory, _session_file_name(session_id)))

    async def _repository_path_for_metadata(self, metadata: SqliteSessionMetadata) -> str:
        expected, actual = await asyncio.gather(
            asyncio.to_thread(os.path.realpath, self._path_for_session(metadata.id), strict=True),
            asyncio.to_thread(os.path.realpath, metadata.path, strict=True),
        )
        if expected != actual:
            raise RuntimeError(f"SQLite session metadata path is outside this repository: {metadata.path}")
        return actual

    def create(self, options: SqliteSessionCreateOptions | None, context: Context) -> Promise[SqliteOpenSession]:
        try:
            self._assert_open()
            options = options if options is not None else SqliteSessionCreateOptions()
            created_at = self._now()
            session_id = options.id if options.id is not None else uuidv7(created_at)
            self._reserve_id(session_id)
            path = self._path_for_session(session_id)
        except BaseException as error:
            return rejected(error)

        async def create() -> SqliteOpenSession:
            db: SqliteDatabase | None = None
            reserved_file = False
            initialized = False
            session: SqliteOpenSession | None = None
            try:
                await asyncio.to_thread(os.makedirs, os.path.dirname(path) or ".", exist_ok=True)
                if self._database_path is None:
                    descriptor = await asyncio.to_thread(os.open, path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                    await asyncio.to_thread(os.close, descriptor)
                    reserved_file = True
                db = await self._database_factory.open(path)
                db.exec("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;")
                await apply_initial_schema(db)
                canonical_path = await asyncio.to_thread(os.path.realpath, path, strict=True)
                metadata = SqliteSessionMetadata(id=session_id, created_at=created_at, storage_version=SQLITE_STORAGE_VERSION,
                                                 parent_session_id=options.parent_session_id, path=canonical_path)
                def initialize() -> None:
                    if has_session_row(db, session_id):
                        raise RuntimeError(f"SQLite session already exists: {session_id}")
                    insert_session_row(db, metadata, SQLITE_STORAGE_VERSION, 1)
                db.transaction(initialize)
                initialized = True
                session = self._open_storage_backed_session(metadata, db)
                return session
            except BaseException:
                if reserved_file and not initialized:
                    await _remove_session_files(path, force=True)
                raise
            finally:
                if session is None:
                    try:
                        if db is not None:
                            db.close()
                    finally:
                        self._pending_ids.discard(session_id)
        return spawn(create())

    def open(self, metadata: SqliteSessionMetadata, context: Context) -> Promise[SqliteOpenSession]:
        try:
            self._assert_open()
            self._reserve_id(metadata.id)
        except BaseException as error:
            return rejected(error)
        async def open() -> SqliteOpenSession:
            db: SqliteDatabase | None = None
            session: SqliteOpenSession | None = None
            try:
                path = await self._repository_path_for_metadata(metadata)
                db = await self._database_factory.open_existing(path)
                db.exec("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;")
                stored = metadata_from_session_row(path, read_session_row(db, metadata.id), SQLITE_STORAGE_VERSION)
                session = self._open_storage_backed_session(stored, db)
                return session
            finally:
                if session is None:
                    try:
                        if db is not None:
                            db.close()
                    finally:
                        self._pending_ids.discard(metadata.id)
        return spawn(open())

    def list(self, options: None, context: Context) -> Promise[list[SqliteSessionMetadata]]:
        try:
            self._assert_open()
        except BaseException as error:
            return rejected(error)
        async def discover() -> list[SqliteSessionMetadata]:
            if self._database_path is not None:
                paths = [self._database_path]
            else:
                try:
                    names = await asyncio.to_thread(os.listdir, self._directory)
                except FileNotFoundError:
                    return []
                paths = [os.path.join(self._directory, name) for name in names if name.endswith(SQLITE_SESSION_EXTENSION)]
            sessions: list[SqliteSessionMetadata] = []
            for path in paths:
                db: SqliteDatabase | None = None
                try:
                    canonical_path = await asyncio.to_thread(os.path.realpath, path, strict=True)
                    db = await self._database_factory.open_read_only(canonical_path)
                    db.exec("PRAGMA busy_timeout = 5000;")
                    for row in read_all_session_rows(db):
                        sessions.append(metadata_from_session_row(canonical_path, row, SQLITE_STORAGE_VERSION))
                except BaseException:
                    pass
                finally:
                    if db is not None:
                        db.close()
            sessions.sort(key=lambda metadata: metadata.created_at, reverse=True)
            return sessions
        return spawn(discover())

    def delete(self, metadata: SqliteSessionMetadata, context: Context) -> Promise[None]:
        try:
            self._assert_open()
            self._reserve_id(metadata.id)
        except BaseException as error:
            return rejected(error)
        async def delete() -> None:
            try:
                path = await self._repository_path_for_metadata(metadata)
                db = await self._database_factory.open_existing(path)
                try:
                    db.exec("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;")
                    if self._database_path is not None:
                        def remove() -> None:
                            metadata_from_session_row(path, read_session_row(db, metadata.id), SQLITE_STORAGE_VERSION)
                            delete_session_rows(db, metadata.id)
                        db.transaction(remove)
                    else:
                        metadata_from_session_row(path, read_session_row(db, metadata.id), SQLITE_STORAGE_VERSION)
                finally:
                    db.close()
                if self._database_path is None:
                    await _remove_session_files(path, force=False)
            finally:
                self._pending_ids.discard(metadata.id)
        return spawn(delete())

    def fork(self, source: SqliteSessionMetadata, options: ForkOptions, context: Context) -> Promise[SqliteOpenSession]:
        try:
            self._assert_open()
            created_at = self._now()
            session_id = options.id if options.id is not None else uuidv7(created_at)
            self._reserve_id(session_id)
            source_storage = self._open_storages.get((source.path, source.id))
            active_snapshot = None if source_storage is None else source_storage.snapshot(options, context)
            if active_snapshot is not None:
                active_snapshot.then_settled(lambda: None)
            path = self._path_for_session(session_id)
        except BaseException as error:
            return rejected(error)
        async def fork() -> SqliteOpenSession:
            db: SqliteDatabase | None = None
            reserved_file = False
            initialized = False
            session: SqliteOpenSession | None = None
            try:
                await asyncio.to_thread(os.makedirs, os.path.dirname(path) or ".", exist_ok=True)
                if self._database_path is None:
                    descriptor = await asyncio.to_thread(os.open, path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                    await asyncio.to_thread(os.close, descriptor)
                    reserved_file = True
                snapshot = (await self._create_external_fork_snapshot(source, options) if active_snapshot is None
                            else _build_fork_snapshot(await active_snapshot, options))
                db = await self._database_factory.open(path)
                db.exec("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;")
                await apply_initial_schema(db)
                canonical_path = await asyncio.to_thread(os.path.realpath, path, strict=True)
                metadata = SqliteSessionMetadata(id=session_id, created_at=created_at, storage_version=SQLITE_STORAGE_VERSION,
                                                 parent_session_id=source.id, path=canonical_path)
                def initialize() -> None:
                    if has_session_row(db, session_id):
                        raise RuntimeError(f"SQLite session already exists: {session_id}")
                    insert_session_row(db, metadata, SQLITE_STORAGE_VERSION, snapshot.next_seq)
                    writer = EntryRowWriter(db, session_id)
                    for entry in snapshot.entries:
                        writer.insert(entry)
                        append_entry_to_branch_index(db, session_id, entry)
                    for stored in snapshot.scalar_values:
                        set_scalar_value_row(db, session_id, stored.address.namespace, stored.address.key, stored.seq, stored.value)
                    SqlQuery("UPDATE sessions SET message_count = ? WHERE id = ?", [snapshot.message_count, session_id]).run(db)
                db.transaction(initialize)
                initialized = True
                session = self._open_storage_backed_session(metadata, db)
                return session
            except BaseException:
                if reserved_file and not initialized:
                    await _remove_session_files(path, force=True)
                raise
            finally:
                if session is None:
                    try:
                        if db is not None:
                            db.close()
                    finally:
                        self._pending_ids.discard(session_id)
        return spawn(fork())

    async def _create_external_fork_snapshot(self, source: SqliteSessionMetadata, options: ForkOptions) -> _ForkSnapshot:
        path = await asyncio.to_thread(os.path.realpath, source.path, strict=True)
        db = await self._database_factory.open_read_only(path)
        try:
            db.exec("PRAGMA busy_timeout = 5000;")
            canonical = SqliteSessionMetadata(id=source.id, created_at=source.created_at, storage_version=source.storage_version,
                                              parent_session_id=source.parent_session_id, path=path)
            return _read_external_snapshot(db, canonical, options)
        finally:
            db.close()

    def _open_storage_backed_session(self, metadata: SqliteSessionMetadata, db: SqliteDatabase) -> SqliteOpenSession:
        key = (metadata.path, metadata.id)
        storage = SqliteStorage(db, SqliteStorageOptions(session_id=metadata.id, now=self._now))
        self._open_storages[key] = storage
        session = StorageBackedSession(metadata, storage)
        def on_close() -> None:
            try:
                db.close()
            finally:
                if self._open_storages.get(key) is storage:
                    del self._open_storages[key]
                self._open_sessions.pop(open_session, None)
                self._pending_ids.discard(metadata.id)
        open_session = SqliteOpenSession(session, SqliteOpenSessionOptions(on_close=on_close))
        self._open_sessions[open_session] = None
        return open_session

    def close(self, context: Context) -> Promise[None]:
        if self._close_promise is not None:
            return self._close_promise
        self._closed = True
        closing = [session.close(context) for session in self._open_sessions]
        async def close() -> None:
            results = await settled_all(closing)
            errors = [result for result in results if isinstance(result, BaseException)]
            if len(errors) == 1:
                raise errors[0]
            if len(errors) > 1:
                raise BaseExceptionGroup("Failed to close SQLite Sessions", errors)
        self._close_promise = spawn(close())
        return self._close_promise
