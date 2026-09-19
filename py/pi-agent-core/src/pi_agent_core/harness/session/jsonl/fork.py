"""JSONL fork ported from ``session/jsonl/fork.ts``.

``run_jsonl_fork`` copies one source session into a fresh format-4 file under a
fork scope, excluding usage rows and open-operation state.

Port strategy note: ``fork.ts`` indexes the source by folding complete
transactions and then streams selected writes into the destination, which keeps
peak memory bounded for very large files. This port instead replays the source
into the in-memory storage state and asks it for the fork projection, which is
the same observable result through the storage state's own fork rules. Both
approaches preserve copied sequences and the source's ``nextSeq`` high-water
mark, and neither modifies the source.
"""

from __future__ import annotations

from typing import Any, Optional

from ...._chord.context import Context
from .io_helpers import materialize_fork_file
from .storage import JsonlStorage
from .types import JsonlSessionMetadata, JsonlStorageHeader

__all__ = ["JsonlForkInput", "run_jsonl_fork"]


class JsonlForkInput:
    """Where a fork reads its source from.

    ``kind`` is ``"path"`` for a closed format-4 file, or ``"storage"`` for a
    source session this process currently holds open. ``storage`` is set for
    the open case; otherwise ``metadata.path`` names the file to replay.
    """

    def __init__(
        self,
        kind: str,
        metadata: Optional[JsonlSessionMetadata] = None,
        storage: Optional[JsonlStorage] = None,
    ) -> None:
        self.kind = kind
        self.metadata = metadata
        self.storage = storage


async def run_jsonl_fork(
    fork_input: JsonlForkInput,
    file_system: Any,
    destination_path: str,
    destination_header: JsonlStorageHeader,
    options: Any,
    open_source: Any,
    context: Context,
) -> None:
    """Index the source, project the fork, and publish the destination atomically."""
    storage = fork_input.storage
    if storage is not None and storage.is_legacy_v3():
        raise RuntimeError(
            "Cannot fork an open legacy v3 JSONL session; commit a non-empty transaction "
            "to upgrade it to format 4 first"
        )
    metadata = fork_input.metadata
    if metadata is not None:
        # Replay the file: every committed write is durable, so the file is the
        # single source of truth for both the open and closed cases.
        opened = await open_source(metadata.path)
        try:
            source_state = opened.storage_state
        finally:
            await opened.close(context)
    elif storage is not None:
        source_state = storage.storage_state
    else:
        raise RuntimeError("JSONL fork source is missing both its metadata and storage")
    forked = source_state.create_fork(options)
    await materialize_fork_file(
        file_system, destination_path, destination_header, forked, context
    )
