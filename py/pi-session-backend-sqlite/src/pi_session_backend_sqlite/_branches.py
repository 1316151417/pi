"""Private branch cache, compaction sharing, and segment scans."""

from dataclasses import dataclass
from typing import cast

from pi_agent_core.harness.session.types import EntryStructure, StorageBranchScan
from pi_ai._javascript import javascript_string

from ._json import Record, field
from ._rows import decode_entry_row, entry_structure_from_row
from .sql import SqlQuery
from .types import SqliteDatabase, SqliteRow


@dataclass
class _Segment:
    branch_id: str
    lower_seq: int
    upper_seq: int


def _segments(db: SqliteDatabase, session_id: str, start: str) -> list[_Segment]:
    row = SqlQuery("SELECT b.branch_id, b.entry_seq FROM branch_entries b "
                   "JOIN branch_meta m ON m.session_id = b.session_id AND m.branch_id = b.branch_id "
                   "WHERE b.session_id = ? AND b.entry_id = ? "
                   "AND ((m.base_seq IS NULL AND b.entry_seq > 0) OR (m.base_seq IS NOT NULL AND b.entry_seq > m.base_seq)) "
                   "AND b.entry_seq <= m.tip_seq ORDER BY m.tip_seq DESC, b.branch_id LIMIT 1",
                   [session_id, start]).get(db)
    if row is None:
        raise RuntimeError(f"Branch cache missing entry {start}")
    branch_id, upper_seq = cast(str, row["branch_id"]), cast(int, row["entry_seq"])
    segments: list[_Segment] = []
    while True:
        meta = SqlQuery("SELECT branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq FROM branch_meta "
                        "WHERE session_id = ? AND branch_id = ?", [session_id, branch_id]).get(db)
        if meta is None:
            raise RuntimeError(f"Branch metadata missing for branch {branch_id}")
        lower_seq = cast(int | None, meta["base_seq"])
        segments.append(_Segment(branch_id, 0 if lower_seq is None else lower_seq, upper_seq))
        if meta["base_branch_id"] is None:
            break
        if lower_seq is None:
            raise RuntimeError(f"Branch {branch_id} has base branch without base_seq")
        branch_id, upper_seq = cast(str, meta["base_branch_id"]), lower_seq
    return segments


def _insert(db: SqliteDatabase, session_id: str, branch_id: str, entry: object) -> None:
    SqlQuery("INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type) VALUES (?, ?, ?, ?, ?)",
             [session_id, branch_id, field(entry, "id"), field(entry, "seq"), field(entry, "type")]).run(db)


def append_entry_to_branch_index(db: SqliteDatabase, session_id: str, entry: object) -> None:
    entry_id = cast(str, field(entry, "id"))
    seq = field(entry, "seq")
    parent_id = field(entry, "parent_id", "parentId")
    if parent_id is None:
        SqlQuery("INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq) "
                 "VALUES (?, ?, ?, ?, ?, ?)", [session_id, entry_id, entry_id, seq, None, None]).run(db)
        _insert(db, session_id, entry_id, entry)
        return
    branch = SqlQuery("SELECT branch_id FROM branch_meta WHERE session_id = ? AND tip_entry_id = ?", [session_id, parent_id]).get(db)
    if branch is not None:
        branch_id = cast(str, branch["branch_id"])
        _insert(db, session_id, branch_id, entry)
        result = SqlQuery("UPDATE branch_meta SET tip_entry_id = ?, tip_seq = ? WHERE session_id = ? AND branch_id = ?",
                          [entry_id, seq, session_id, branch_id]).run(db)
        if result.changes != 1:
            raise RuntimeError(f"Expected to update branch {branch_id}, updated {javascript_string(result.changes)}")
        return
    segments = _segments(db, session_id, cast(str, parent_id))
    compaction_branch: str | None = None
    compaction_seq: int | None = None
    for segment in segments:
        row = SqlQuery("SELECT MAX(entry_seq) AS entry_seq FROM branch_entries WHERE session_id = ? AND branch_id = ? "
                       "AND entry_seq > ? AND entry_seq <= ? AND entry_type = ?",
                       [session_id, segment.branch_id, segment.lower_seq, segment.upper_seq, "compaction"]).get(db)
        if row is not None and row["entry_seq"] is not None:
            compaction_branch, compaction_seq = segment.branch_id, cast(int, row["entry_seq"])
            break
    SqlQuery("INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq) "
             "VALUES (?, ?, ?, ?, ?, ?)", [session_id, entry_id, entry_id, seq, compaction_branch, compaction_seq]).run(db)
    for segment in reversed(segments):
        lower_seq = max(segment.lower_seq, compaction_seq if compaction_seq is not None else 0)
        if segment.upper_seq <= lower_seq:
            continue
        SqlQuery("INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type) "
                 "SELECT ?, ?, entry_id, entry_seq, entry_type FROM branch_entries "
                 "WHERE session_id = ? AND branch_id = ? AND entry_seq > ? AND entry_seq <= ?",
                 [session_id, entry_id, session_id, segment.branch_id, lower_seq, segment.upper_seq]).run(db)
    _insert(db, session_id, entry_id, entry)


def _stop_seq(db: SqliteDatabase, session_id: str, segment: _Segment, query: StorageBranchScan, oldest: bool) -> int | None:
    predicates: list[str] = []
    params: list[object] = [session_id, segment.branch_id, segment.lower_seq, segment.upper_seq]
    if query.stop_at_type is not None:
        predicates.append("b.entry_type = ?")
        params.append(query.stop_at_type)
    if query.stop_at_id is not None:
        predicates.append("b.entry_id = ?")
        params.append(query.stop_at_id)
    if not predicates:
        return None
    aggregate = "MIN" if oldest else "MAX"
    row = SqlQuery(f"SELECT {aggregate}(b.entry_seq) AS stop_seq FROM branch_entries b WHERE b.session_id = ? "
                   f"AND b.branch_id = ? AND b.entry_seq > ? AND b.entry_seq <= ? AND ({' OR '.join(predicates)})", params).get(db)
    return None if row is None else cast(int | None, row["stop_seq"])


def _scan(db: SqliteDatabase, session_id: str, query: StorageBranchScan, *, structure: bool) -> list[SqliteRow]:
    oldest = query.order == "oldestFirst"
    segments = _segments(db, session_id, query.start)
    if oldest:
        segments.reverse()
    limit = None if query.limit is None else max(0, query.limit)
    if limit == 0:
        return []
    rows: list[SqliteRow] = []
    for segment in segments:
        remaining = None if limit is None else limit - len(rows)
        if remaining is not None and remaining <= 0:
            break
        stop = _stop_seq(db, session_id, segment, query, oldest)
        predicates = ["b.session_id = ?", "b.branch_id = ?", "b.entry_seq > ?", "b.entry_seq <= ?", "e.session_id = b.session_id"]
        params: list[object] = [session_id, segment.branch_id, segment.lower_seq, segment.upper_seq]
        if stop is not None:
            predicates.append("b.entry_seq <= ?" if oldest else "b.entry_seq >= ?")
            params.append(stop)
        if query.type is not None:
            predicates.append("b.entry_type = ?")
            params.append(query.type)
        if query.custom_type is not None:
            predicates.append("e.custom_type = ?")
            params.append(query.custom_type)
        if query.cursor is not None:
            predicates.append("b.entry_seq > ?" if oldest else "b.entry_seq < ?")
            params.append(query.cursor.seq)
        columns = "e.id, e.parent_id, e.seq, e.type, e.custom_type, e.timestamp"
        if not structure:
            columns += ", e.payload"
        text = (f"SELECT {columns} FROM branch_entries b CROSS JOIN entries e ON e.session_id = b.session_id AND e.id = b.entry_id "
                f"WHERE {' AND '.join(predicates)} ORDER BY b.entry_seq {'ASC' if oldest else 'DESC'}")
        if remaining is not None:
            text += " LIMIT ?"
            params.append(max(0, remaining))
        rows.extend(SqlQuery(text, params).all(db))
        if stop is not None:
            break
    return rows


def scan_branch_entries(db: SqliteDatabase, session_id: str, query: StorageBranchScan) -> list[Record]:
    return [decode_entry_row(row) for row in _scan(db, session_id, query, structure=False)]


def scan_branch_entry_structures(db: SqliteDatabase, session_id: str, query: StorageBranchScan) -> list[EntryStructure]:
    return [entry_structure_from_row(row) for row in _scan(db, session_id, query, structure=True)]
