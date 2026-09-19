"""JSONL session repository ported from ``session/jsonl/repo.ts``.

Sessions live under ``<sessionsRoot>/<encoded cwd>/<timestamp>_<id>.jsonl``.
Directory encoding is lossy, so discovered files are filtered by their header
``cwd`` when listing.
"""

from __future__ import annotations

import asyncio
import time
import urllib.parse
from typing import Any, Callable, List, Optional

from ...._chord.context import Context
from pi_ai.uuid_utils import uuidv7
from ...types import FileSystem
from ..session import StorageBackedSession
from ..types import (
    ForkOptions,
    Session,
    SessionCreateOptions,
)
from .codec import parse_jsonl_session_header
from .io import file_value
from .storage import JsonlStorage
from .types import (
    JSONL_FORMAT_VERSION,
    JSONL_STORAGE_VERSION,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepoOptions,
    JsonlStorageHeader,
)

__all__ = ["JsonlSessionRepo", "metadata_from_header", "session_directory_name", "session_file_name"]


def session_directory_name(cwd: str) -> str:
    """Encode an absolute working directory into a single path segment."""
    return f"--{cwd.replace('/', '-').strip('-')}--"


def session_file_name(created_at: int, session_id: str) -> str:
    """Session file name: creation timestamp plus the percent-encoded id."""
    return f"{created_at}_{urllib.parse.quote(session_id, safe='')}.jsonl"


def metadata_from_header(header: JsonlStorageHeader, path: str, modified_at: int) -> JsonlSessionMetadata:
    return JsonlSessionMetadata(
        id=header.id,
        created_at=header.created_at,
        storage_version=header.storage_version,
        cwd=header.cwd,
        parent_session_id=header.parent_session_id,
        legacy_parent_session_path=header.legacy_parent_session_path,
        path=path,
        modified_at=modified_at,
    )


class JsonlSessionRepo:
    """File-backed session repository."""

    def __init__(self, options: JsonlSessionRepoOptions) -> None:
        self._file_system: FileSystem = options.file_system
        self._sessions_root_input: str = options.sessions_root
        self._now: Callable[[], int] = options.now or (lambda: int(time.time() * 1000))
        self._open_sessions: dict = {}
        self._pending_creates: set = set()
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("JsonlSessionRepo is closed")

    async def close(self, _context: Context) -> None:
        self._closed = True

    # ------------------------------------------------------------------
    # Create / open / list / delete
    # ------------------------------------------------------------------

    async def create(self, options: JsonlSessionCreateOptions, context: Context) -> Session:
        self._assert_open()
        created_at = self._now()
        cwd = file_value(
            await self._file_system.absolute_path(options.cwd, context),
            f"Failed to resolve session cwd {options.cwd}",
        )
        session_id = options.id or uuidv7(created_at)
        key = self._session_key(cwd, session_id)
        if key in self._open_sessions or key in self._pending_creates:
            raise RuntimeError(f"Session already exists: {session_id}")
        self._pending_creates.add(key)
        path: Optional[str] = None
        storage: Optional[JsonlStorage] = None
        try:
            path = await self._resolve_new_session_path(cwd, created_at, session_id, context)
            header = JsonlStorageHeader(
                v=JSONL_FORMAT_VERSION,
                kind="header",
                id=session_id,
                storage_version=JSONL_STORAGE_VERSION,
                created_at=created_at,
                cwd=cwd,
                parent_session_id=options.parent_session_id,
            )
            storage = await JsonlStorage.create(
                self._storage_options(path), header, [], context
            )
            info = file_value(
                await self._file_system.file_info(path, context), f"Failed to read session {path}"
            )
            return self._publish_open_session(
                metadata_from_header(header, path, info.mtime_ms), storage, key
            )
        except BaseException:
            if storage is not None:
                try:
                    await storage.close(context)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if path is not None:
                await self._file_system.remove_file(path, context)
            raise
        finally:
            self._pending_creates.discard(key)

    async def open(self, metadata: JsonlSessionMetadata, context: Context) -> Session:
        self._assert_open()
        key = self._session_key(metadata.cwd, metadata.id)
        if key in self._open_sessions:
            raise RuntimeError(f"Session is already open: {metadata.id}")
        storage: Optional[JsonlStorage] = None
        try:
            storage = await self._load_storage(metadata, context)
            return self._publish_open_session(metadata, storage, key)
        except BaseException:
            if storage is not None:
                await storage.close(context)
            raise

    async def list(
        self, options: Optional[JsonlSessionListOptions], context: Context
    ) -> List[JsonlSessionMetadata]:
        options = options or JsonlSessionListOptions()
        self._assert_open()
        cwd: Optional[str] = None
        if options.cwd is not None:
            cwd = file_value(
                await self._file_system.absolute_path(options.cwd, context),
                f"Failed to resolve session cwd {options.cwd}",
            )
        root = await self._root(context)
        if not file_value(await self._file_system.exists(root, context), f"Failed to check sessions root {root}"):
            return []
        directories = (
            await self._session_directories(root, context)
            if cwd is None
            else [await self._session_directory(cwd, context)]
        )
        metadata: List[JsonlSessionMetadata] = []
        for directory in directories:
            metadata.extend(await self._list_directory(directory, cwd, context))
        metadata.sort(key=lambda item: (-item.created_at, item.id, item.cwd))
        return metadata

    async def delete(self, metadata: JsonlSessionMetadata, context: Context) -> None:
        self._assert_open()
        key = self._session_key(metadata.cwd, metadata.id)
        if key in self._open_sessions:
            raise RuntimeError(f"Session is open: {metadata.id}")
        if not file_value(
            await self._file_system.exists(metadata.path, context),
            f"Failed to check session {metadata.path}",
        ):
            raise RuntimeError(f"Session file does not exist: {metadata.path}")
        file_value(
            await self._file_system.remove_file(metadata.path, context),
            f"Failed to delete session {metadata.path}",
        )

    async def fork(self, source: JsonlSessionMetadata, options: ForkOptions, context: Context) -> Session:
        self._assert_open()
        created_at = self._now()
        cwd = source.cwd
        session_id = options.id or uuidv7(created_at)
        destination_key = self._session_key(cwd, session_id)
        if destination_key in self._open_sessions or destination_key in self._pending_creates:
            raise RuntimeError(f"Session already exists: {session_id}")
        self._pending_creates.add(destination_key)

        source_storage = self._open_sessions.get(self._session_key(source.cwd, source.id))
        path: Optional[str] = None
        storage: Optional[JsonlStorage] = None
        try:
            path = await self._resolve_new_session_path(cwd, created_at, session_id, context)
            header = JsonlStorageHeader(
                v=JSONL_FORMAT_VERSION,
                kind="header",
                id=session_id,
                storage_version=JSONL_STORAGE_VERSION,
                created_at=created_at,
                cwd=cwd,
                parent_session_id=source.id,
            )
            await self._run_fork(source, source_storage, path, header, options, context)
            storage = await JsonlStorage.open(self._storage_options(path), context)
            info = file_value(
                await self._file_system.file_info(path, context), f"Failed to read session {path}"
            )
            return self._publish_open_session(
                metadata_from_header(header, path, info.mtime_ms), storage, destination_key
            )
        except BaseException:
            if storage is not None:
                try:
                    await storage.close(context)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if path is not None:
                await self._file_system.remove_file(path, context)
            raise
        finally:
            self._pending_creates.discard(destination_key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_fork(
        self,
        source: JsonlSessionMetadata,
        source_storage: Optional[JsonlStorage],
        destination_path: str,
        destination_header: JsonlStorageHeader,
        options: ForkOptions,
        context: Context,
    ) -> None:
        """Copy the source session into a fresh file honoring the fork scope."""
        from .fork import JsonlForkInput, run_jsonl_fork

        async def _open_source(path: str) -> JsonlStorage:
            return await JsonlStorage.open(self._storage_options(path), context)

        await run_jsonl_fork(
            JsonlForkInput(
                kind="storage" if source_storage is not None else "path",
                metadata=source,
                storage=source_storage,
            ),
            self._file_system,
            destination_path,
            destination_header,
            options,
            _open_source,
            context,
        )

    async def _list_directory(
        self, directory: str, cwd: Optional[str], context: Context
    ) -> List[JsonlSessionMetadata]:
        if not file_value(
            await self._file_system.exists(directory, context),
            f"Failed to check sessions directory {directory}",
        ):
            return []
        files = [
            entry
            for entry in file_value(
                await self._file_system.list_dir(directory, context),
                f"Failed to list sessions directory {directory}",
            )
            if entry.kind != "directory" and entry.name.endswith(".jsonl")
        ]
        metadata: List[JsonlSessionMetadata] = []
        for file in files:
            discovered = await self._read_session_metadata(file, context)
            if discovered is None:
                continue
            # Directory encoding is lossy: /a/b and /a-b both map to --a-b--.
            if cwd is None or discovered.cwd == cwd:
                metadata.append(discovered)
        return metadata

    async def _read_session_metadata(self, file: Any, context: Context) -> Optional[JsonlSessionMetadata]:
        lines = file_value(
            await self._file_system.read_text_lines(file.path, {"maxLines": 1}, context),
            f"Failed to read session header {file.path}",
        )
        if not lines:
            return None
        parsed = parse_jsonl_session_header(lines[0])
        if not parsed.ok:
            return None
        if parsed.value.format == "v3-legacy":
            header = parsed.value.header
            created_at = _timestamp_to_ms(header.timestamp)
            return JsonlSessionMetadata(
                id=header.id,
                created_at=created_at,
                storage_version=0,
                cwd=header.cwd,
                parent_session_id=header.parent_session,
                path=file.path,
                modified_at=file.mtime_ms,
            )
        return metadata_from_header(parsed.value.header, file.path, file.mtime_ms)

    async def _session_directories(self, root: str, context: Context) -> List[str]:
        return [
            entry.path
            for entry in file_value(
                await self._file_system.list_dir(root, context), f"Failed to list sessions root {root}"
            )
            if entry.kind == "directory"
        ]

    async def _session_directory(self, cwd: str, context: Context) -> str:
        return file_value(
            await self._file_system.join_path(
                [await self._root(context), session_directory_name(cwd)], context
            ),
            f"Failed to resolve sessions directory for {cwd}",
        )

    async def _resolve_new_session_path(
        self, cwd: str, created_at: int, session_id: str, context: Context
    ) -> str:
        directory = await self._session_directory(cwd, context)
        await self._assert_session_id_available(directory, session_id, context)
        file_value(
            await self._file_system.create_dir(directory, None, context),
            f"Failed to create sessions directory {directory}",
        )
        return file_value(
            await self._file_system.join_path(
                [directory, session_file_name(created_at, session_id)], context
            ),
            f"Failed to resolve path for session {session_id}",
        )

    async def _assert_session_id_available(self, directory: str, session_id: str, context: Context) -> None:
        if not file_value(
            await self._file_system.exists(directory, context),
            f"Failed to check sessions directory {directory}",
        ):
            return
        suffix = f"_{urllib.parse.quote(session_id, safe='')}.jsonl"
        id_exists = any(
            entry.kind != "directory" and entry.name.endswith(suffix)
            for entry in file_value(
                await self._file_system.list_dir(directory, context),
                f"Failed to list sessions directory {directory}",
            )
        )
        if id_exists:
            raise RuntimeError(f"Session already exists: {session_id}")

    def _publish_open_session(
        self, metadata: JsonlSessionMetadata, storage: JsonlStorage, key: str
    ) -> Session:
        if key in self._open_sessions:
            raise RuntimeError(f"Session is already open: {metadata.id}")

        def _on_close() -> None:
            if self._open_sessions.get(key) is storage:
                del self._open_sessions[key]

        session = StorageBackedSession(metadata, storage, {"on_close": _on_close})
        self._open_sessions[key] = storage
        return session

    def _session_key(self, cwd: str, session_id: str) -> str:
        return f"{cwd}\x00{session_id}"

    async def _load_storage(self, metadata: JsonlSessionMetadata, context: Context) -> JsonlStorage:
        if not file_value(
            await self._file_system.exists(metadata.path, context),
            f"Failed to check session {metadata.path}",
        ):
            raise RuntimeError(f"Session file does not exist: {metadata.path}")
        storage = await JsonlStorage.open(self._storage_options(metadata.path), context)
        try:
            if storage.header.id != metadata.id or storage.header.cwd != metadata.cwd:
                raise RuntimeError(f"Session identity does not match header: {metadata.id}")
            if storage.header.storage_version != JSONL_STORAGE_VERSION:
                raise RuntimeError(
                    f"Session {metadata.id} uses unsupported storage version "
                    f"{storage.header.storage_version}"
                )
            return storage
        except BaseException:
            await storage.close(context)
            raise

    async def _root(self, context: Context) -> str:
        return file_value(
            await self._file_system.absolute_path(self._sessions_root_input, context),
            f"Failed to resolve sessions root {self._sessions_root_input}",
        )

    def _storage_options(self, path: str) -> Any:
        from .types import JsonlStorageOptions

        return JsonlStorageOptions(file_system=self._file_system, path=path, now=self._now)


def _timestamp_to_ms(timestamp: str) -> int:
    from datetime import datetime

    try:
        return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0
