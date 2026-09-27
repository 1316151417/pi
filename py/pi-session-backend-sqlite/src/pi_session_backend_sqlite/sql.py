"""Parameterized SQL fragments; ``sql((prefix, suffix), value)`` is a template."""

from collections.abc import Iterable, Sequence

from .types import SqliteDatabase, SqliteRow, SqliteRunResult


class SqlQuery:
    def __init__(self, query_text: str, params: Sequence[object] = ()) -> None:
        self.query_text = query_text
        self.params = params

    def exec(self, db: SqliteDatabase) -> None:
        if self.params:
            raise TypeError("SQLite exec queries cannot have parameters")
        db.exec(self.query_text)

    def run(self, db: SqliteDatabase) -> SqliteRunResult:
        return db.prepare(self.query_text).run(*self.params)

    def get(self, db: SqliteDatabase) -> SqliteRow | None:
        return db.prepare(self.query_text).get(*self.params)

    def all(self, db: SqliteDatabase) -> list[SqliteRow]:
        return db.prepare(self.query_text).all(*self.params)

    def iterate(self, db: SqliteDatabase) -> Iterable[SqliteRow]:
        return db.prepare(self.query_text).iterate(*self.params)


def sql(strings: Sequence[str], *values: object) -> SqlQuery:
    query_text = strings[0] if strings else ""
    params: list[object] = []
    for index, value in enumerate(values):
        if isinstance(value, SqlQuery):
            query_text += value.query_text
            params.extend(value.params)
        else:
            query_text += "?"
            params.append(value)
        query_text += strings[index + 1] if index + 1 < len(strings) else ""
    return SqlQuery(query_text, params)


def join_sql_fragments(fragments: Sequence[SqlQuery], separator: str) -> SqlQuery:
    return SqlQuery(separator.join(fragment.query_text for fragment in fragments),
                    [param for fragment in fragments for param in fragment.params])
