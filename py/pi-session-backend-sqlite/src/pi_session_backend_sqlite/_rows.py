"""Session, sequence, entry, usage-ledger, and stats SQL row codecs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from pi_agent_core.harness.session.types import EntryScan, EntryStructure, SessionMetadata, SessionStats, UsageScan
from pi_ai._javascript import javascript_string
from pi_ai.types import UNDEFINED, Usage

from ._json import Record, field, object_spread, parse, parse_record, stringify
from .sql import SqlQuery
from .types import SqliteDatabase, SqliteRow


@dataclass(kw_only=True)
class SqliteSessionMetadata(SessionMetadata):
    path: str


_SESSION_COLUMNS = "id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq"
_ENTRY_COLUMNS = "id, parent_id, seq, type, custom_type, timestamp, payload"
_ENTRY_INSERT = "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
_USAGE_INSERT = "INSERT INTO usage_ledger (session_id, id, seq, entry_id, adjustment, usage, details) VALUES (?, ?, ?, ?, ?, ?, ?)"


def read_session_row(db: SqliteDatabase, session_id: str) -> SqliteRow:
    row = SqlQuery(f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = ?", [session_id]).get(db)
    if row is None:
        raise RuntimeError(f"Unknown SQLite session: {session_id}")
    return row


def read_all_session_rows(db: SqliteDatabase) -> list[SqliteRow]:
    return SqlQuery(f"SELECT {_SESSION_COLUMNS} FROM sessions").all(db)


def has_session_row(db: SqliteDatabase, session_id: str) -> bool:
    return SqlQuery("SELECT id FROM sessions WHERE id = ?", [session_id]).get(db) is not None


def metadata_from_session_row(path: str, row: SqliteRow, current_storage_version: int) -> SqliteSessionMetadata:
    version = cast(int, row["storage_version"])
    if version > current_storage_version:
        raise RuntimeError(f"SQLite session storage version {javascript_string(version)} is newer than {current_storage_version}")
    if version < current_storage_version:
        raise RuntimeError(f"SQLite session storage version {javascript_string(version)} requires migrations")
    return SqliteSessionMetadata(id=cast(str, row["id"]), created_at=cast(int, row["created_at"]),
                                 storage_version=version, parent_session_id=cast(str | None, row["parent_session_id"]), path=path)


def insert_session_row(db: SqliteDatabase, metadata: SqliteSessionMetadata, storage_version: int, next_seq: int) -> None:
    zero = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}
    SqlQuery(f"INSERT INTO sessions ({_SESSION_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
             [metadata.id, metadata.created_at, metadata.parent_session_id, storage_version, None, 0, stringify(zero), next_seq]).run(db)


def delete_session_rows(db: SqliteDatabase, session_id: str) -> None:
    for table in ("entries", "scalar_values", "list_values", "usage_ledger", "branch_entries", "branch_meta"):
        SqlQuery(f"DELETE FROM {table} WHERE session_id = ?", [session_id]).run(db)
    result = SqlQuery("DELETE FROM sessions WHERE id = ?", [session_id]).run(db)
    if result.changes != 1:
        raise RuntimeError(f"Expected to delete one SQLite session {session_id}, deleted {javascript_string(result.changes)}")


def read_next_seq(db: SqliteDatabase, session_id: str) -> int:
    row = SqlQuery("SELECT next_seq FROM sessions WHERE id = ?", [session_id]).get(db)
    if row is None:
        raise RuntimeError(f"Unknown SQLite session: {session_id}")
    return cast(int, row["next_seq"])


def advance_next_seq(db: SqliteDatabase, session_id: str, next_seq: int) -> None:
    result = SqlQuery("UPDATE sessions SET next_seq = ? WHERE id = ?", [next_seq, session_id]).run(db)
    if result.changes != 1:
        raise RuntimeError(f"Expected to update one SQLite session {session_id}, updated {javascript_string(result.changes)}")


def _entry_params(session_id: str, entry: object) -> list[object]:
    entry_type = field(entry, "type")
    payload: dict[str, object] = {}
    if entry_type == "message":
        payload["message"] = field(entry, "message")
        terminate = field(entry, "terminate")
        if terminate is not UNDEFINED and (isinstance(entry, dict) or terminate is not None):
            payload["terminate"] = terminate
    elif entry_type == "compaction":
        payload = {"summary": field(entry, "summary"), "retainedTail": field(entry, "retained_tail", "retainedTail"),
                   "tokensBefore": field(entry, "tokens_before", "tokensBefore")}
    elif entry_type == "branch_summary":
        payload = {"fromId": field(entry, "from_id", "fromId"), "summary": field(entry, "summary")}
    elif entry_type == "custom":
        data = field(entry, "data")
        if data is not UNDEFINED:
            payload["data"] = data
    if entry_type in ("compaction", "branch_summary"):
        details = field(entry, "details")
        if details is not UNDEFINED:
            payload["details"] = details
        usage = field(entry, "usage")
        if usage is not UNDEFINED and (isinstance(entry, dict) or usage is not None):
            payload["usage"] = usage
        payload["fromHook"] = field(entry, "from_hook", "fromHook")
    return [session_id, field(entry, "id"), field(entry, "parent_id", "parentId"), field(entry, "seq"), entry_type,
            field(entry, "custom_type", "customType") if entry_type == "custom" else None,
            field(entry, "timestamp"), stringify(payload)]


class EntryRowWriter:
    def __init__(self, db: SqliteDatabase, session_id: str) -> None:
        self._statement = db.prepare(_ENTRY_INSERT)
        self._session_id = session_id

    def insert(self, entry: object) -> None:
        self._statement.run(*_entry_params(self._session_id, entry))


def insert_entry_row(db: SqliteDatabase, session_id: str, entry: object) -> None:
    db.prepare(_ENTRY_INSERT).run(*_entry_params(session_id, entry))


def decode_entry_row(row: SqliteRow) -> Record:
    result = Record(id=row["id"], parentId=row["parent_id"], seq=row["seq"], timestamp=row["timestamp"], type=row["type"])
    if row["type"] == "custom":
        if row["custom_type"] is None:
            raise RuntimeError(f"Custom entry {row['id']} is missing custom_type")
        result["customType"] = row["custom_type"]
    if row["type"] not in ("message", "compaction", "branch_summary", "custom"):
        return cast(Record, UNDEFINED)
    result.update(object_spread(parse(cast(str, row["payload"]))))
    return result


def entry_structure_from_row(row: SqliteRow) -> EntryStructure:
    return EntryStructure(id=cast(str, row["id"]), parent_id=cast(str | None, row["parent_id"]),
                          seq=cast(int, row["seq"]), timestamp=cast(int, row["timestamp"]), type=cast(str, row["type"]),
                          custom_type=cast(str | None, row["custom_type"]))


def read_entry_rows(db: SqliteDatabase, session_id: str, ids: Sequence[str]) -> list[SqliteRow]:
    if not ids:
        return []
    return SqlQuery(f"SELECT {_ENTRY_COLUMNS} FROM entries WHERE session_id = ? AND id IN ({', '.join('?' for _ in ids)})",
                    [session_id, *ids]).all(db)


def read_all_entry_rows(db: SqliteDatabase, session_id: str) -> list[SqliteRow]:
    return SqlQuery(f"SELECT {_ENTRY_COLUMNS} FROM entries WHERE session_id = ? ORDER BY seq ASC", [session_id]).all(db)


def scan_entry_rows(db: SqliteDatabase, session_id: str, query: EntryScan) -> list[SqliteRow]:
    predicates = ["session_id = ?"]
    params: list[object] = [session_id]
    for column, value, operation in (("type", query.type, "="), ("custom_type", query.custom_type, "="),
                                      ("seq", query.from_seq, ">="), ("seq", query.to_seq, "<=")):
        if value is not None:
            predicates.append(f"{column} {operation} ?")
            params.append(value)
    text = f"SELECT {_ENTRY_COLUMNS} FROM entries WHERE {' AND '.join(predicates)} ORDER BY seq {'DESC' if query.order == 'desc' else 'ASC'}"
    if query.limit is not None:
        text += " LIMIT ?"
        params.append(max(0, query.limit))
    return SqlQuery(text, params).all(db)


def _usage_params(session_id: str, row: object) -> list[object]:
    entry_id = field(row, "entry_id", "entryId")
    details = field(row, "details")
    return [session_id, field(row, "id"), field(row, "seq"), None if entry_id is UNDEFINED else entry_id,
            1 if field(row, "adjustment") else 0, stringify(field(row, "usage")),
            None if details is UNDEFINED else stringify(details)]


class UsageLedgerRowWriter:
    def __init__(self, db: SqliteDatabase, session_id: str) -> None:
        self._statement = db.prepare(_USAGE_INSERT)
        self._session_id = session_id

    def insert(self, row: object) -> None:
        self._statement.run(*_usage_params(self._session_id, row))


def insert_usage_ledger_row(db: SqliteDatabase, session_id: str, row: object) -> None:
    db.prepare(_USAGE_INSERT).run(*_usage_params(session_id, row))


def decode_usage_ledger_row(row: SqliteRow) -> Record:
    result = Record(id=row["id"], seq=row["seq"], usage=parse(cast(str, row["usage"])), adjustment=row["adjustment"] != 0)
    if row["entry_id"] is not None:
        result["entryId"] = row["entry_id"]
    if row["details"] is not None:
        result["details"] = parse(cast(str, row["details"]))
    return result


def scan_usage_ledger_rows(db: SqliteDatabase, session_id: str, query: UsageScan) -> list[SqliteRow]:
    filters = ["session_id = ?"]
    params: list[object] = [session_id]
    if query.from_seq is not None:
        filters.append("seq >= ?")
        params.append(query.from_seq)
    if query.to_seq is not None:
        filters.append("seq <= ?")
        params.append(query.to_seq)
    text = f"SELECT id, seq, entry_id, adjustment, usage, details FROM usage_ledger WHERE {' AND '.join(filters)} ORDER BY seq {'DESC' if query.order == 'desc' else 'ASC'}"
    if query.limit is not None:
        text += " LIMIT ?"
        params.append(max(0, query.limit))
    return SqlQuery(text, params).all(db)


def read_session_stats(db: SqliteDatabase, session_id: str) -> SessionStats:
    row = read_session_row(db, session_id)
    return SessionStats(message_count=cast(int, row["message_count"]), usage=cast(Usage, parse(row["usage_payload"])))


def increment_message_count(db: SqliteDatabase, session_id: str) -> None:
    SqlQuery("UPDATE sessions SET message_count = message_count + 1 WHERE id = ?", [session_id]).run(db)


def add_usage_to_session_stats(db: SqliteDatabase, session_id: str, usage: object) -> None:
    current = read_session_stats(db, session_id).usage
    result: dict[str, object] = {}
    for snake, camel in (("input", "input"), ("output", "output"), ("cache_read", "cacheRead"),
                         ("cache_write", "cacheWrite"), ("total_tokens", "totalTokens")):
        result[camel] = cast(float, field(current, snake, camel)) + cast(float, field(usage, snake, camel))
    for snake, camel in (("cache_write_1h", "cacheWrite1h"), ("reasoning", "reasoning")):
        left, right = field(current, snake, camel), field(usage, snake, camel)
        # Python's optional Usage fields use None for absent; JSON views keep null.
        left_missing = left is UNDEFINED or (not isinstance(current, dict) and left is None)
        right_missing = right is UNDEFINED or (not isinstance(usage, dict) and right is None)
        if not (left_missing and right_missing):
            result[camel] = (0 if left_missing or left is None else cast(float, left)) + (0 if right_missing or right is None else cast(float, right))
    left_cost, right_cost = field(current, "cost"), field(usage, "cost")
    result["cost"] = {camel: cast(float, field(left_cost, snake, camel)) + cast(float, field(right_cost, snake, camel))
                      for snake, camel in (("input", "input"), ("output", "output"), ("cache_read", "cacheRead"),
                                           ("cache_write", "cacheWrite"), ("total", "total"))}
    SqlQuery("UPDATE sessions SET usage_payload = ? WHERE id = ?", [stringify(result), session_id]).run(db)
