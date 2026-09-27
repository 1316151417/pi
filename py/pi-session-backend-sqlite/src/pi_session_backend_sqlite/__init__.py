"""Native Python SQLite session backend for pi-agent-core."""

from .adapter import create_node_sqlite_factory, create_sqlite_factory, wrap_node_sqlite_database, wrap_sqlite_database
from .repo import (SQLITE_SESSION_EXTENSION, SQLITE_STORAGE_VERSION, SqliteSessionCreateOptions,
                   SqliteSessionMetadata, SqliteSessionRepo, SqliteSessionRepoOptions)
from .session import SqliteOpenSession, SqliteOpenSessionOptions
from .sql import SqlQuery, join_sql_fragments, sql
from .storage import SqliteStorage, SqliteStorageOptions, SqliteStorageSnapshot
from .types import SqliteDatabase, SqliteDatabaseFactory, SqliteRow, SqliteRunResult, SqliteStatement

__all__ = [
    "SQLITE_SESSION_EXTENSION", "SQLITE_STORAGE_VERSION", "SqliteSessionCreateOptions", "SqliteSessionMetadata",
    "SqliteSessionRepo", "SqliteSessionRepoOptions", "SqliteOpenSession", "SqliteOpenSessionOptions",
    "SqliteStorage", "SqliteStorageOptions", "SqliteStorageSnapshot", "SqliteDatabase", "SqliteDatabaseFactory",
    "SqliteStatement", "SqliteRunResult", "SqliteRow", "SqlQuery", "sql", "join_sql_fragments",
    "create_sqlite_factory", "wrap_sqlite_database", "create_node_sqlite_factory", "wrap_node_sqlite_database",
]
