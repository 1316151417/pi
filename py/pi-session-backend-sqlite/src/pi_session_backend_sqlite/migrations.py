"""The unmodified source schema, loaded as a package resource."""

import asyncio
from importlib.resources import files

from .types import SqliteDatabase


async def apply_initial_schema(db: SqliteDatabase) -> None:
    migration = await asyncio.to_thread(
        files("pi_session_backend_sqlite").joinpath("migrations/001_initial.sql").read_text,
        encoding="utf-8",
    )
    db.exec(migration)
