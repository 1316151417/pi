"""Scalar and list value SQL, including Unicode prefix range pagination."""

from typing import cast

from pi_agent_core.harness.session.values import (
    ListElement, ListReadOptions, StoredValue, Value, ValueList, resolve_list_read_options, value,
)

from ._json import parse, stringify
from .sql import SqlQuery
from .types import SqliteDatabase, SqliteRow


def set_scalar_value_row(db: SqliteDatabase, session_id: str, namespace: str, key: str, seq: int, stored_value: object) -> None:
    SqlQuery("INSERT INTO scalar_values (session_id, namespace, key, seq, value) VALUES (?, ?, ?, ?, ?) "
             "ON CONFLICT(session_id, namespace, key) DO UPDATE SET seq = excluded.seq, value = excluded.value",
             [session_id, namespace, key, seq, stringify(stored_value)]).run(db)


def delete_scalar_value_row(db: SqliteDatabase, session_id: str, namespace: str, key: str) -> None:
    SqlQuery("DELETE FROM scalar_values WHERE session_id = ? AND namespace = ? AND key = ?", [session_id, namespace, key]).run(db)


def append_list_value_row(db: SqliteDatabase, session_id: str, namespace: str, key: str, seq: int, element: object) -> None:
    SqlQuery("INSERT INTO list_values (session_id, namespace, key, seq, value) VALUES (?, ?, ?, ?, ?)",
             [session_id, namespace, key, seq, stringify(element)]).run(db)


def delete_list_value_rows(db: SqliteDatabase, session_id: str, namespace: str, key: str) -> None:
    SqlQuery("DELETE FROM list_values WHERE session_id = ? AND namespace = ? AND key = ?", [session_id, namespace, key]).run(db)


def _decode[T](address: Value[T], row: SqliteRow) -> StoredValue[T]:
    if row["namespace"] != address.namespace or row["key"] != address.key:
        raise RuntimeError(f"Expected value {address.namespace}:{address.key}, found {row['namespace']}:{row['key']}")
    return StoredValue(address=address, seq=cast(int, row["seq"]), value=parse(cast(str, row["value"])))


def read_scalar_value_row[T](db: SqliteDatabase, session_id: str, address: Value[T]) -> StoredValue[T] | None:
    row = SqlQuery("SELECT namespace, key, seq, value FROM scalar_values WHERE session_id = ? AND namespace = ? AND key = ?",
                   [session_id, address.namespace, address.key]).get(db)
    return None if row is None else _decode(address, row)


def read_all_scalar_value_rows(db: SqliteDatabase, session_id: str) -> list[StoredValue[object]]:
    rows = SqlQuery("SELECT namespace, key, seq, value FROM scalar_values WHERE session_id = ? ORDER BY seq ASC", [session_id]).all(db)
    return [StoredValue(address=value(cast(str, row["namespace"]), cast(str, row["key"])),
                        seq=cast(int, row["seq"]), value=parse(cast(str, row["value"]))) for row in rows]


def _next_prefix_boundary(prefix: str) -> str | None:
    points = prefix.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    for index in range(len(points) - 1, -1, -1):
        point = ord(points[index])
        if point < 0x10ffff:
            next_point = 0xe000 if 0xd7ff <= point < 0xe000 else point + 1
            return points[:index] + chr(next_point)
    return None


def scan_scalar_value_rows[T](db: SqliteDatabase, session_id: str, prefix: Value[T]) -> list[StoredValue[T]]:
    boundary = _next_prefix_boundary(prefix.key)
    text = "SELECT namespace, key, seq, value FROM scalar_values WHERE session_id = ? AND namespace = ? AND key >= ?"
    params: list[object] = [session_id, prefix.namespace, prefix.key]
    if boundary is not None:
        text += " AND key < ?"
        params.append(boundary)
    rows = SqlQuery(text + " ORDER BY key ASC", params).all(db)
    return [_decode(value(cast(str, row["namespace"]), cast(str, row["key"])), row) for row in rows]


def list_value_read_query[T](session_id: str, address: ValueList[T], options: ListReadOptions | None = None) -> SqlQuery:
    resolved = resolve_list_read_options(options)
    text = "SELECT seq, value FROM list_values WHERE session_id = ? AND namespace = ? AND key = ?"
    params: list[object] = [session_id, address.namespace, address.key]
    if resolved.cursor is not None:
        text += " AND seq > ?" if resolved.order == "asc" else " AND seq < ?"
        params.append(resolved.cursor.seq)
    text += " ORDER BY seq ASC LIMIT ?" if resolved.order == "asc" else " ORDER BY seq DESC LIMIT ?"
    params.append(resolved.limit)
    return SqlQuery(text, params)


def read_list_value_rows[T](db: SqliteDatabase, session_id: str, address: ValueList[T], options: ListReadOptions | None = None) -> list[ListElement[T]]:
    return [ListElement(seq=cast(int, row["seq"]), value=parse(cast(str, row["value"])))
            for row in list_value_read_query(session_id, address, options).all(db)]
