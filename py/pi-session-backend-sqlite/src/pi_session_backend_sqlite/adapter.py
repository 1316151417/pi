"""Native sqlite3 adapter for the source package's Node DatabaseSync boundary."""

from __future__ import annotations

import inspect
import math
import re
import sqlite3
import weakref
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from ._async import Promise, capture
from .types import SqliteDatabase, SqliteRow, SqliteRunResult

_TOKEN = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^]]*\]|--[^\r\n]*|/\*[\s\S]*?\*/|\?[0-9]*|[:@$][\w\x80-\U0010ffff]+(?:::[\w\x80-\U0010ffff]+)*(?:\([^)]*\))?", re.UNICODE)


def _utf8_text(text: str) -> str:
    # V8 encodes lone UTF-16 surrogates as U+FFFD and joins valid pairs.
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _first_statement(text: str) -> str:
    text = _utf8_text(text).split("\x00", 1)[0]
    for match in re.finditer(";", text):
        if sqlite3.complete_statement(text[:match.end()]):
            return text[:match.end()]
    return text


class _Statement:
    def __init__(self, db: _Database, query: str) -> None:
        self._db = db
        self._query = _first_statement(query)
        self._names: dict[str, int] = {}
        self._slots: dict[int, str | None] = {}
        largest = 0
        for match in _TOKEN.finditer(self._query):
            token = match.group()
            if token[0] not in "?:@$":
                continue
            if token == "?":
                largest += 1
                self._slots[largest] = None
            elif token[0] == "?":
                index = int(token[1:])
                largest = max(largest, index)
                if self._slots.get(index) is None:
                    self._slots[index] = token
                self._names.setdefault(token, index)
            elif token not in self._names:
                largest += 1
                self._slots[largest] = token
                self._names[token] = largest
        self._count = largest
        self._bare_names: dict[str, str] | None = None
        self._cursor: sqlite3.Cursor | None = None
        db._assert_open()
        # sqlite3 exposes no prepare-only operation. EXPLAIN compiles the exact
        # statement with NULL parameters without executing the statement itself.
        explain = self._query
        if not re.match(r"\s*EXPLAIN\b", explain, re.IGNORECASE):
            explain = "EXPLAIN " + explain
        cursor = db._connection.cursor()
        try:
            cursor.execute(explain, [None] * self._count)
        finally:
            cursor.close()
        db._statements.add(self)

    def _bind(self, params: tuple[object, ...]) -> list[object]:
        values: list[object] = [None] * self._count
        positional = params
        if params and isinstance(params[0], Mapping):
            if self._bare_names is None:
                bare_names: dict[str, str] = {}
                for index in range(1, self._count + 1):
                    full = self._slots.get(index)
                    if full is None:
                        continue
                    bare = full[1:]
                    existing = bare_names.get(bare)
                    if existing is not None and existing != full:
                        raise RuntimeError(f"Cannot create bare named parameter '{bare}' because of conflicting names '{existing}' and '{full}'.")
                    bare_names[bare] = full
                self._bare_names = bare_names
            for name, value in params[0].items():
                name = str(name)
                index = self._names.get(name)
                if index is None:
                    index = self._names.get(self._bare_names.get(name, ""))
                if index is None:
                    raise RuntimeError(f"Unknown named parameter '{name}'")
                values[index - 1] = self._bind_value(value, index)
            positional = params[1:]
        index = 1
        for value in positional:
            while self._slots.get(index) is not None:
                index += 1
            bound = self._bind_value(value, index)
            if index > self._count:
                raise sqlite3.ProgrammingError("column index out of range")
            values[index - 1] = bound
            index += 1
        return values

    @staticmethod
    def _bind_value(value: object, index: int) -> object:
        if isinstance(value, bool):
            raise TypeError(f"Provided value cannot be bound to SQLite parameter {index}.")
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except OverflowError:
                return math.inf if value > 0 else -math.inf
        if isinstance(value, str):
            return _utf8_text(value)
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        raise TypeError(f"Provided value cannot be bound to SQLite parameter {index}.")

    def _reset(self) -> None:
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None

    def _execute(self, values: list[object]) -> sqlite3.Cursor:
        if self._db._closed:
            raise RuntimeError("statement has been finalized")
        self._reset()
        self._cursor = self._db._connection.cursor()
        self._cursor.execute(self._query, values)
        return self._cursor

    @staticmethod
    def _row(cursor: sqlite3.Cursor, row: tuple[object, ...]) -> SqliteRow:
        result: SqliteRow = {}
        for column, value in zip(cursor.description or (), row):
            if isinstance(value, int) and abs(value) > 9007199254740991:
                raise OverflowError(f"Value is too large to be represented as a JavaScript number: {value}")
            result[column[0]] = value
        return result

    def run(self, *params: object) -> SqliteRunResult:
        values = self._bind(params)
        try:
            self._execute(values)
        finally:
            self._reset()
        cursor = self._db._connection.execute("SELECT changes(), last_insert_rowid()")
        try:
            row = cursor.fetchone()
            return SqliteRunResult(float(row[0]), float(row[1]))
        finally:
            cursor.close()

    def get(self, *params: object) -> SqliteRow | None:
        values = self._bind(params)
        try:
            cursor = self._execute(values)
            row = cursor.fetchone()
            return None if row is None else self._row(cursor, row)
        finally:
            self._reset()

    def all(self, *params: object) -> list[SqliteRow]:
        values = self._bind(params)
        try:
            cursor = self._execute(values)
            return [self._row(cursor, row) for row in cursor]
        finally:
            self._reset()

    def iterate(self, *params: object) -> Iterator[SqliteRow]:
        if self._db._closed:
            raise RuntimeError("statement has been finalized")
        self._reset()
        values = self._bind(params)

        def rows() -> Iterator[SqliteRow]:
            try:
                cursor = self._execute(values)
                for row in cursor:
                    yield self._row(cursor, row)
            finally:
                self._reset()
        return rows()


class _Database:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._closed = False
        self._statements: weakref.WeakSet[_Statement] = weakref.WeakSet()

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("database is not open")

    def exec(self, sql: str) -> None:
        self._assert_open()
        if not isinstance(sql, str):
            raise TypeError('The "sql" argument must be a string.')
        sql = _utf8_text(sql).split("\x00", 1)[0]
        start = 0
        for match in re.finditer(";", sql):
            statement = sql[start:match.end()]
            if sqlite3.complete_statement(statement):
                cursor = self._connection.execute(statement)
                try:
                    for _ in cursor:
                        pass
                finally:
                    cursor.close()
                start = match.end()
        if sql[start:].strip():
            cursor = self._connection.execute(sql[start:])
            try:
                for _ in cursor:
                    pass
            finally:
                cursor.close()

    def prepare(self, sql: str) -> _Statement:
        self._assert_open()
        if not isinstance(sql, str):
            raise TypeError('The "sql" argument must be a string.')
        return _Statement(self, sql)

    def transaction[T](self, callback: Callable[[], T]) -> T:
        self.exec("BEGIN IMMEDIATE")
        try:
            result = callback()
            if inspect.isawaitable(result) or (isinstance(result, Mapping) and "then" in result) or hasattr(result, "then"):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("SQLite transaction callbacks must be synchronous")
            self.exec("COMMIT")
            return result
        except BaseException:
            try:
                self.exec("ROLLBACK")
            except BaseException:
                pass
            raise

    def close(self) -> None:
        self._assert_open()
        for statement in self._statements:
            statement._reset()
        self._connection.close()
        self._closed = True


def wrap_sqlite_database(db: sqlite3.Connection) -> SqliteDatabase:
    """Wrap an autocommit connection; transaction boundaries belong to this adapter."""
    return _Database(db)


class _Factory:
    @staticmethod
    def _open(path: str, mode: str) -> SqliteDatabase:
        memory = mode == "rwc" and path in (":memory:", "")
        uri = path if memory else Path(path).absolute().as_uri() + "?mode=" + mode
        connection = sqlite3.connect(uri, uri=not memory, isolation_level=None, timeout=0)
        try:
            connection.text_factory = lambda data: data.split(b"\0", 1)[0].decode("utf-8", "replace")
            connection.execute("PRAGMA foreign_keys = ON").close()
            if hasattr(connection, "setconfig"):
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
            return _Database(connection)
        except BaseException:
            connection.close()
            raise

    def open(self, path: str) -> Promise[SqliteDatabase]:
        return capture(lambda: self._open(path, "rwc"))

    def open_existing(self, path: str) -> Promise[SqliteDatabase]:
        return capture(lambda: self._open(path, "rw"))

    def open_read_only(self, path: str) -> Promise[SqliteDatabase]:
        return capture(lambda: self._open(path, "ro"))


def create_sqlite_factory() -> _Factory:
    return _Factory()


# The Node-qualified names are source exports; both use native sqlite3 in Python.
wrap_node_sqlite_database = wrap_sqlite_database
create_node_sqlite_factory = create_sqlite_factory
