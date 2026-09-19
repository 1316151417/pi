"""Context derivation ported from ``harness/pico3/context.ts``.

pico §1.3, verbatim::

    H       = newest fork-visible entry at or before T with a head
    from    = H ? H.head : transcript start
    range   = fork-visible entries from `from` through T
    edits   = per target, newest edit in range wins
    entries = H ? [H, ...range without any head entries] : range
    model   = concat(entries.map(e => edits[e.id] ? apply : e.model)), then reorder tool results

Display-only entries (aborted/error assistants, pi.usage, model-less plugin
entries) have no ``model`` and contribute nothing. Nothing here inspects an
error string.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..._chord.context import Context
from pi_ai.types import JsonObject
from .types import (
    ContextEdit,
    ContextView,
    Entry,
    EntryScan,
    Id,
    RequestMessage,
    Storage,
    StoredMessage,
)

__all__ = ["derive_context", "reorder_tool_results", "PAGE_SIZE"]

PAGE_SIZE = 256

#: Text substituted for a tool call whose result was cut off by a fork.
MISSING_TOOL_RESULT_TEXT = "Tool result unavailable: history ends before this call completed."


async def derive_context(
    storage: Storage,
    conversation_id: Id,
    at: Optional[Id],
    ctx: Context,
) -> ContextView:
    """Derive the model-visible context at ``at`` (default: the newest entry)."""
    head_scan = EntryScan(conversation_id=conversation_id, with_head=True, limit=1)
    if at is not None:
        head_scan.before = at + 1
    head_page = await storage.scan_entries(head_scan, ctx)
    head: Optional[Entry] = head_page[0] if head_page else None
    start = head.head if head is not None else None

    # Walk newest-first until we pass `from`.
    collected: List[Entry] = []
    before = None if at is None else at + 1
    while True:
        scan = EntryScan(conversation_id=conversation_id, limit=PAGE_SIZE, before=before)
        page = await storage.scan_entries(scan, ctx)
        done = len(page) < PAGE_SIZE
        for entry in page:
            if start is not None and entry.id < start:
                done = True
                break
            collected.append(entry)
        if done:
            break
        before = page[-1].id
    collected.reverse()

    edits: Dict[Id, ContextEdit] = {}
    for entry in collected:
        for edit in entry.edits or []:
            edits[edit.target] = edit  # later wins by iteration order

    entries = collected if head is None else [head, *[e for e in collected if e.head is None]]

    messages: List[RequestMessage] = []
    for entry in entries:
        edit = edits.get(entry.id)
        if edit is not None and edit.action == "omit":
            continue
        if edit is not None and edit.action == "replace":
            messages.extend(edit.messages or [])
            continue
        if entry.model:
            messages.extend(entry.model)
    return ContextView(head=head, entries=entries, messages=reorder_tool_results(messages))


def _role(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _blocks(message: Any) -> List[Any]:
    if isinstance(message, dict):
        return message.get("content") or []
    return getattr(message, "content", None) or []


def _block_type(block: Any) -> Any:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_field(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _set_field(message: Any, key: str, value: Any) -> None:
    if isinstance(message, dict):
        message[key] = value
    else:
        setattr(message, key, value)


def _get_field(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def reorder_tool_results(messages: List[RequestMessage]) -> List[RequestMessage]:
    """Put tool results back in call order, synthesising one after a fork cut."""
    out: List[RequestMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        out.append(message)
        index += 1
        if _role(message) != "assistant":
            continue
        calls = [block for block in _blocks(message) if _block_type(block) == "toolCall"]
        if len(calls) == 0:
            continue
        results: Dict[str, Any] = {}
        cursor = index
        while cursor < len(messages) and _role(messages[cursor]) == "toolResult":
            result = messages[cursor]
            results[str(_get_field(result, "toolCallId"))] = result
            cursor += 1
        for call in calls:
            call_id = str(_block_field(call, "id"))
            existing = results.get(call_id)
            if existing is not None:
                out.append(existing)
                continue
            out.append(
                _synthesize_tool_result(
                    call_id, _block_field(call, "name"), _get_field(message, "timestamp")
                )
            )
        index = cursor
    return out


def _synthesize_tool_result(call_id: str, name: Any, timestamp: Any) -> JsonObject:
    return {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": name,
        "content": [{"type": "text", "text": MISSING_TOOL_RESULT_TEXT}],
        "isError": True,
        "details": {"reason": "missing_after_fork"},
        "timestamp": timestamp,
    }


_ = StoredMessage
