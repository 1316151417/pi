"""Session context building ported from ``harness/session/context.ts``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..._chord.context import Context
from ..messages import create_branch_summary_message, create_compaction_summary_message

__all__ = ["build_context_entries", "session_entry_to_context_messages", "build_session_context"]


def build_context_entries(path_entries: List[Any]) -> List[Any]:
    """Trim history to the last compaction entry plus everything after it."""
    compaction = None
    compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        entry = path_entries[index]
        if getattr(entry, "type", None) == "compaction":
            compaction = entry
            compaction_index = index
            break
    if compaction is None:
        return list(path_entries)
    return [compaction, *path_entries[compaction_index + 1 :]]


def _is_context_message(message: Any) -> bool:
    if getattr(message, "role", None) != "assistant":
        return True
    return getattr(message, "stop_reason", None) not in ("error", "aborted", "deferred")


def session_entry_to_context_messages(entry: Any) -> List[Any]:
    entry_type = getattr(entry, "type", None)
    if entry_type == "message":
        return [entry.message] if _is_context_message(entry.message) else []
    if entry_type == "compaction":
        retained = [m for m in entry.retained_tail if _is_context_message(m)]
        return [
            create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp),
            *retained,
        ]
    if entry_type == "branch_summary":
        if not entry.summary:
            return []
        return [create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)]
    return []


async def build_session_context(
    path_entries: List[Any],
    options: Optional[dict],
    context: Context,
) -> List[Any]:
    """Project session entries into model context messages."""
    options = options or {}
    entry_projectors: Dict[str, Any] = options.get("entry_projectors") or {}
    entries = build_context_entries(path_entries)
    messages: List[Any] = []
    for entry in entries:
        if getattr(entry, "type", None) != "custom":
            messages.extend(session_entry_to_context_messages(entry))
            continue
        projector = entry_projectors.get(entry.custom_type)
        if projector is not None:
            import asyncio

            projected = projector(entry, context)
            if asyncio.iscoroutine(projected):
                projected = await projected
            messages.extend(projected or [])
    return messages
