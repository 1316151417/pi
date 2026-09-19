"""JSONL fork materialization helpers.

Materializes a forked storage state into a fresh JSONL file: header plus one
transaction per surviving entry, usage row, scalar value, and list element,
preserving sequence order so replay reconstructs the same state.
"""

from __future__ import annotations

import json
from typing import Any, List

from ...._chord.context import Context
from ...types import FileSystem
from ..commit import CommittedWrite
from .io import publish_jsonl
from .types import JsonlStorageHeader

__all__ = ["state_to_committed_writes", "materialize_fork_file"]


def state_to_committed_writes(state: Any) -> List[CommittedWrite]:
    """Enumerate the committed writes that reproduce ``state`` in sequence order."""
    writes: List[CommittedWrite] = []

    for entry in state._entries_by_seq:
        writes.append(
            CommittedWrite(
                kind="entry",
                seq=entry.seq,
                timestamp=getattr(entry, "timestamp", 0),
                id=entry.id,
                parent_id=entry.parent_id,
                entry=entry,
            )
        )

    for row in state._usage.values():
        writes.append(CommittedWrite(kind="usage", seq=row.seq, id=row.id, entry=row))

    for stored in state._scalar_values.values():
        writes.append(
            CommittedWrite(
                kind="value",
                op="set",
                seq=stored.seq,
                namespace=stored.address.namespace,
                key=stored.address.key,
                value=stored.value,
            )
        )

    for stored_list in state._list_values.values():
        for element in stored_list.elements:
            writes.append(
                CommittedWrite(
                    kind="list",
                    op="append",
                    seq=element.seq,
                    namespace=stored_list.address.namespace,
                    key=stored_list.address.key,
                    value=element.value,
                )
            )

    writes.sort(key=lambda write: write.seq)
    return writes


async def materialize_fork_file(
    file_system: FileSystem,
    destination_path: str,
    destination_header: JsonlStorageHeader,
    state: Any,
    context: Context,
) -> None:
    """Write a forked state to ``destination_path`` through the atomic publisher."""
    writes = state_to_committed_writes(state)
    next_seq = state.get_next_seq()
    header = JsonlStorageHeader(
        v=destination_header.v,
        kind=destination_header.kind,
        id=destination_header.id,
        storage_version=destination_header.storage_version,
        created_at=destination_header.created_at,
        cwd=destination_header.cwd,
        parent_session_id=destination_header.parent_session_id,
        legacy_parent_session_path=destination_header.legacy_parent_session_path,
        next_seq=next_seq,
    )

    async def _write_transactions(append) -> None:
        for write in writes:
            await append([write])

    await publish_jsonl(file_system, destination_path, header, context, _write_transactions)
