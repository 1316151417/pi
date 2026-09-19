"""Legacy v3 JSONL decoding ported from ``session/jsonl/legacy-v3.ts``.

A v3-legacy file is a raw append log: a session header plus one record per line,
with ``parentId`` references between records. This module exposes such a file as
repeatable logical v4 writes: one scan builds the structural index, mints v4 IDs for
retained records and derives the current values (session name, entry labels, branch
tip, lane configuration); every ``writes()`` pass reopens the path, replays the
captured records and yields the committed entry writes followed by those values.

Records are the external wire format, so they stay plain JSON mappings with their
``camelCase`` keys; where the TS original hands its JSON-shaped messages straight on,
the port materializes them through the v4 codec (``message_from_json`` and the entry
dataclasses).

The TS line reader reports ``terminated`` per line, which lets the scan ignore an
unterminated final line (a torn append, or a record still being written). The Python
``TextLineReader`` yields plain strings, and the in-tree local reader splits file
content on ``"\\n"`` while keeping the trailing remainder, so the last element of a
chunk is never a complete line: it is either the empty artifact of a final newline or
the torn tail the TS reader ignores. Both are dropped, which reproduces that rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Set, Tuple

from ...._chord.context import Context
from ...._pi_ai.types import Usage, message_from_json, usage_from_json
from ...._pi_ai.uuid_utils import uuidv7
from ...messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from ...types import FileSystem, TextLineReader
from ...utils.usage import add_usage, empty_usage
from ..commit import CommittedWrite
from ..types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    LaneConfiguration,
    LaneState,
    MessageEntry,
)
from ..values import Value, branch_tip, entry_label, lane_config, lane_state, session_name, set_value
from .codec import LegacyV3SessionHeader, parse_jsonl_session_header
from .io import file_value, read_jsonl_header
from .types import JSONL_FORMAT_VERSION, JSONL_STORAGE_VERSION, JsonlSessionMetadata, JsonlStorageHeader

__all__ = [
    "LegacyV3EntryStructure",
    "metadata_from_legacy_v3_header",
    "normalize_legacy_v3_header",
    "LegacyV3Source",
]

#: One legacy v3 record. Records are the external wire format, so keys stay ``camelCase``.
_LegacyV3Entry = Dict[str, Any]

#: Record types that keep conversation payloads; everything else only carries metadata.
_DISCARDED_ENTRY_TYPES = frozenset(
    {"model_change", "thinking_level_change", "active_tools_change", "session_info", "label"}
)

_SUPPORTED_RECORD_TYPES = frozenset(
    {
        "message",
        "custom",
        "custom_message",
        "branch_summary",
        "compaction",
        "model_change",
        "thinking_level_change",
        "active_tools_change",
        "session_info",
        "label",
    }
)

#: Stands in for a JSON key that is absent, so error text can mirror TS ``undefined`` reads.
_MISSING = object()

#: Resolve a legacy ID (``None`` is the root) to its imported ID.
_ResolveLegacyId = Callable[[Optional[str]], Optional[str]]


@dataclass
class LegacyV3EntryStructure:
    """``Pick<CommittedEntryWrite, "id" | "parentId" | "seq">`` for one retained entry."""

    id: str
    parent_id: Optional[str]
    seq: int


@dataclass
class _LegacyV3IndexEntry:
    """Structural index of one v3 record.

    Retained records carry their minted ID and sequence; discarded records keep the
    metadata needed to project labels and lane configuration. TS splits this into a
    retained/discarded union; the port keeps one dataclass whose variant fields stay
    ``None`` when they do not apply.
    """

    type: str
    id: str
    parent_id: Optional[str] = None
    #: This node's new ID, or its parent's mapped ID when the node is discarded.
    mapped_id: Optional[str] = None
    seq: Optional[int] = None
    #: branch_summary
    from_id: Optional[str] = None
    #: compaction
    first_kept_entry_id: Optional[str] = None
    #: label
    target_id: Optional[str] = None
    label: Optional[str] = None
    #: model_change
    provider: Optional[str] = None
    model_id: Optional[str] = None
    #: thinking_level_change
    thinking_level: Optional[str] = None
    #: active_tools_change
    active_tool_names: Optional[List[str]] = None


@dataclass
class _LegacyV3Inventory:
    """One full scan: the structural index plus the derived identity and usage."""

    entries: Dict[str, _LegacyV3IndexEntry] = field(default_factory=dict)
    imported_usage: Usage = field(default_factory=Usage)
    name: Optional[str] = None
    final_id: Optional[str] = None
    next_seq: int = 1


def _timestamp_ms(timestamp: Any) -> int:
    """Milliseconds since the epoch for an ISO timestamp (TS ``Date.parse``).

    Values that are not ISO timestamps yield ``0``, matching the house helper used by
    the JSONL repo for legacy headers.
    """
    if not isinstance(timestamp, str):
        return 0
    try:
        return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0


def _stringify(value: Any) -> str:
    """Approximate the TS ``String(value)`` interpolation used in the TS error messages."""
    if value is _MISSING:
        return "undefined"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ",".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return "[object Object]"
    return str(value)


def _complete_lines(chunk: List[str]) -> List[str]:
    """Drop the trailing newline artifact / unterminated tail from a reader chunk.

    Readers in this port split file content on ``"\\n"``, so the last element is never
    a terminated line; the TS reader reports that element as ``terminated: false`` and
    both variants are therefore ignored.
    """
    lines = list(chunk)
    if lines:
        lines.pop()
    return lines


def _record_usage(value: Any) -> Optional[Usage]:
    """Wire usages arrive as JSON mappings; anything else carries no usage to import."""
    return usage_from_json(value) if isinstance(value, dict) else None


async def _resolve_legacy_v3_parent_session_id(
    file_system: FileSystem,
    parent_session_path: str,
    context: Context,
) -> Optional[str]:
    lines = await file_system.read_text_lines(parent_session_path, {"maxLines": 1}, context)
    if not bool(getattr(lines, "ok", False)) or not lines.value:
        return None
    parsed = parse_jsonl_session_header(lines.value[0])
    return parsed.value.header.id if parsed.ok else None


async def metadata_from_legacy_v3_header(
    file_system: FileSystem,
    header: LegacyV3SessionHeader,
    context: Context,
) -> JsonlSessionMetadata:
    """Metadata for a v3-legacy session, without ``path``/``modified_at``.

    TS returns ``Omit<JsonlSessionMetadata, "path" | "modifiedAt">``; the port returns
    that dataclass with the two carried fields left at their defaults for the caller.
    """
    metadata = JsonlSessionMetadata(
        id=header.id,
        created_at=_timestamp_ms(header.timestamp),
        storage_version=JSONL_STORAGE_VERSION,
        cwd=header.cwd,
    )
    if header.parent_session is not None:
        parent_session_id = await _resolve_legacy_v3_parent_session_id(
            file_system, header.parent_session, context
        )
        if parent_session_id is not None:
            metadata.parent_session_id = parent_session_id
        else:
            metadata.legacy_parent_session_path = header.parent_session
    return metadata


async def normalize_legacy_v3_header(
    file_system: FileSystem,
    header: LegacyV3SessionHeader,
    context: Context,
) -> JsonlStorageHeader:
    """Project a v3-legacy header onto the current v4 storage header."""
    metadata = await metadata_from_legacy_v3_header(file_system, header, context)
    return JsonlStorageHeader(
        v=JSONL_FORMAT_VERSION,
        kind="header",
        id=metadata.id,
        storage_version=metadata.storage_version,
        created_at=metadata.created_at,
        cwd=metadata.cwd,
        parent_session_id=metadata.parent_session_id,
        legacy_parent_session_path=metadata.legacy_parent_session_path,
    )


def _parse_legacy_v3_entry(line: str) -> _LegacyV3Entry:
    try:
        entry = json.loads(line)
    except json.JSONDecodeError as error:
        wrapped = RuntimeError("Invalid legacy v3 JSONL record: not valid JSON")
        wrapped.__cause__ = error
        raise wrapped from error
    record_type = entry["type"] if isinstance(entry, dict) and "type" in entry else _MISSING
    if not isinstance(record_type, str) or record_type not in _SUPPORTED_RECORD_TYPES:
        raise RuntimeError(f"Unsupported legacy v3 record type: {_stringify(record_type)}")
    return entry


def _imported_custom_message(entry: _LegacyV3Entry) -> Any:
    """Rebuild a ``custom_message`` record as the coding-agent custom message."""
    return create_custom_message(
        entry["customType"],
        entry.get("content"),
        entry.get("display", False),
        entry.get("details"),
        _timestamp_ms(entry["timestamp"]),
    )


def _is_retained_entry_type(entry_type: Optional[str]) -> bool:
    """Keep structure, labels, and configuration changes, but not conversation payloads."""
    return entry_type not in _DISCARDED_ENTRY_TYPES


def _create_legacy_id_resolver(entries: Dict[str, _LegacyV3IndexEntry]) -> _ResolveLegacyId:
    """Resolve a legacy ID to its imported ID.

    Discarded records resolve to the minted ID of their nearest retained ancestor.
    Parent mappings are already folded during the scan; later references need only a
    lookup.
    """

    def resolve(legacy_id: Optional[str]) -> Optional[str]:
        if legacy_id is None:
            return None
        entry = entries.get(legacy_id)
        if entry is None:
            raise RuntimeError(f"Missing legacy v3 entry reference: {_stringify(legacy_id)}")
        return entry.mapped_id

    return resolve


def _resolve_branch_summary_from_id(resolve_legacy_id: _ResolveLegacyId, legacy_from_id: str) -> Optional[str]:
    # Legacy branchWithSummary() encoded a root source as the "root" sentinel instead of null.
    return None if legacy_from_id == "root" else resolve_legacy_id(legacy_from_id)


def _project_context_message(entry: _LegacyV3Entry, resolve_legacy_id: _ResolveLegacyId) -> Optional[Any]:
    """Project a record onto the context message it contributes, if any."""
    entry_type = entry.get("type")
    if entry_type == "message":
        return message_from_json(entry["message"])
    if entry_type == "custom_message":
        return _imported_custom_message(entry)
    if entry_type == "branch_summary":
        if not entry.get("summary"):
            return None
        return create_branch_summary_message(
            entry["summary"],
            _resolve_branch_summary_from_id(resolve_legacy_id, entry["fromId"]),
            _timestamp_ms(entry["timestamp"]),
        )
    if entry_type == "compaction":
        return create_compaction_summary_message(
            entry["summary"], entry["tokensBefore"], _timestamp_ms(entry["timestamp"])
        )
    return None


def _retained_tail_structure(
    compaction: _LegacyV3IndexEntry,
    entries_by_id: Dict[str, _LegacyV3IndexEntry],
) -> Iterator[_LegacyV3IndexEntry]:
    """Walk physical ancestry, including discarded nodes and the exact kept boundary."""
    current_id = compaction.parent_id
    while current_id is not None:
        # The scan guarantees that every parent exists earlier in the file, so cycles are impossible.
        entry = entries_by_id[current_id]
        yield entry
        if current_id == compaction.first_kept_entry_id:
            return
        current_id = entry.parent_id
    raise RuntimeError(
        f"Legacy v3 compaction {compaction.id} firstKeptEntryId is not on its parent branch: "
        f"{_stringify(compaction.first_kept_entry_id)}"
    )


def _normalize_retained_entry(
    entry: _LegacyV3Entry,
    indexed: _LegacyV3IndexEntry,
    retained_tail: List[Any],
    resolve_legacy_id: _ResolveLegacyId,
) -> CommittedWrite:
    """Materialize one retained record as a committed entry write."""
    entry_id = indexed.mapped_id
    parent_id = resolve_legacy_id(indexed.parent_id)
    seq = indexed.seq
    timestamp = _timestamp_ms(entry["timestamp"])
    entry_type = entry.get("type")
    materialized: Any
    if entry_type == "message":
        materialized = MessageEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            message=message_from_json(entry["message"]),
        )
    elif entry_type == "custom_message":
        materialized = MessageEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            message=_imported_custom_message(entry),
        )
    elif entry_type == "branch_summary":
        materialized = BranchSummaryEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            from_id=_resolve_branch_summary_from_id(resolve_legacy_id, entry["fromId"]),
            summary=entry["summary"],
            details=entry.get("details"),
            usage=_record_usage(entry.get("usage")),
            from_hook=bool(entry.get("fromHook")),
        )
    elif entry_type == "compaction":
        materialized = CompactionEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            summary=entry["summary"],
            retained_tail=retained_tail,
            tokens_before=entry["tokensBefore"],
            details=entry.get("details"),
            usage=_record_usage(entry.get("usage")),
            from_hook=bool(entry.get("fromHook")),
        )
    else:
        materialized = CustomEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            custom_type=entry.get("customType", ""),
            data=entry.get("data"),
        )
    return CommittedWrite(
        kind="entry",
        seq=seq,
        timestamp=timestamp,
        id=entry_id,
        parent_id=parent_id,
        entry=materialized,
    )


def _selected_configuration(
    entries_by_id: Dict[str, _LegacyV3IndexEntry],
    selected_id: Optional[str],
) -> Optional[LaneConfiguration]:
    """Nearest configuration changes on the selected ancestry, when all of them exist."""
    remaining = {"model_change", "thinking_level_change", "active_tools_change"}
    model: Optional[dict] = None
    thinking_level: Optional[str] = None
    active_tool_names: Optional[List[str]] = None
    current_id = selected_id
    while current_id is not None and remaining:
        entry = entries_by_id[current_id]
        # Consume the nearest change even when invalid: older values must not become fallbacks.
        if entry.type in remaining:
            remaining.discard(entry.type)
            if entry.type == "model_change":
                model = {"provider": entry.provider, "modelId": entry.model_id}
            elif entry.type == "thinking_level_change":
                thinking_level = entry.thinking_level
            elif entry.type == "active_tools_change":
                active_tool_names = list(entry.active_tool_names or [])
        current_id = entry.parent_id
    if model is None or thinking_level is None:
        return None
    return LaneConfiguration(
        model=model,
        thinking_level=thinking_level,
        active_tool_names=active_tool_names or [],
    )


def _lane_configuration_json(configuration: LaneConfiguration) -> dict:
    """Wire shape of a lane configuration value (TS stores the interface as JSON)."""
    return {
        "model": {
            "provider": configuration.model.get("provider"),
            "modelId": configuration.model.get("modelId"),
        },
        "thinkingLevel": configuration.thinking_level,
        "activeToolNames": list(configuration.active_tool_names),
    }


def _index_legacy_v3_entry(
    entry: _LegacyV3Entry,
    line_number: int,
    seq: int,
    entries: Dict[str, _LegacyV3IndexEntry],
) -> _LegacyV3IndexEntry:
    entry_id = entry["id"]
    # SessionManager appends after an existing leaf and writes extracted branches in parent order.
    parent_id = entry.get("parentId", _MISSING)
    if parent_id is None:
        mapped_parent_id: Any = None
    else:
        parent = entries.get(parent_id)
        mapped_parent_id = parent.mapped_id if parent is not None else _MISSING
    if mapped_parent_id is _MISSING:
        raise RuntimeError(
            f"Legacy v3 entry {_stringify(entry_id)} has a missing or forward parent at line "
            f"{line_number}: {_stringify(parent_id)}"
        )
    entry_type = entry["type"]
    if not _is_retained_entry_type(entry_type):
        if entry_type == "label":
            return _LegacyV3IndexEntry(
                type=entry_type,
                id=entry_id,
                parent_id=parent_id,
                mapped_id=mapped_parent_id,
                target_id=entry["targetId"],
                label=entry.get("label"),
            )
        if entry_type == "model_change":
            return _LegacyV3IndexEntry(
                type=entry_type,
                id=entry_id,
                parent_id=parent_id,
                mapped_id=mapped_parent_id,
                provider=entry["provider"],
                model_id=entry["modelId"],
            )
        if entry_type == "thinking_level_change":
            return _LegacyV3IndexEntry(
                type=entry_type,
                id=entry_id,
                parent_id=parent_id,
                mapped_id=mapped_parent_id,
                thinking_level=entry["thinkingLevel"],
            )
        if entry_type == "active_tools_change":
            return _LegacyV3IndexEntry(
                type=entry_type,
                id=entry_id,
                parent_id=parent_id,
                mapped_id=mapped_parent_id,
                active_tool_names=list(entry["activeToolNames"]),
            )
        return _LegacyV3IndexEntry(
            # session_info: the only discarded type left, and the only one with no payload.
            type=entry_type,
            id=entry_id,
            parent_id=parent_id,
            mapped_id=mapped_parent_id,
        )
    # Null denotes the root, not a retained node identity that can be reminted.
    if "id" in entry and entry["id"] is None:
        raise RuntimeError("Legacy v3 entry reference has no retained ancestor: null")
    retained = _LegacyV3IndexEntry(
        type=entry_type,
        id=entry_id,
        parent_id=parent_id,
        mapped_id=uuidv7(_timestamp_ms(entry["timestamp"])),
        seq=seq,
    )
    if entry_type == "branch_summary":
        retained.from_id = entry["fromId"]
    elif entry_type == "compaction":
        retained.first_kept_entry_id = entry["firstKeptEntryId"]
    return retained


def _legacy_entry_usage(entry: _LegacyV3Entry) -> Optional[Usage]:
    entry_type = entry.get("type")
    if entry_type == "message":
        message = entry.get("message")
        if isinstance(message, dict) and message.get("role") in ("assistant", "toolResult"):
            return _record_usage(message.get("usage"))
        return None
    if entry_type in ("compaction", "branch_summary"):
        return _record_usage(entry.get("usage"))
    return None


async def _read_legacy_v3_inventory(reader: TextLineReader, context: Context) -> _LegacyV3Inventory:
    """Scan complete v3 records, ignoring the unterminated final line of the reader chunk."""
    lines = _complete_lines(file_value(await reader.read_lines(None), "Failed to read legacy v3 source"))
    entries: Dict[str, _LegacyV3IndexEntry] = {}
    next_seq = 1
    imported_usage = empty_usage()
    name: Optional[str] = None
    final_id: Optional[str] = None
    for text in lines:
        line_number = len(entries) + 2
        entry = _parse_legacy_v3_entry(text)
        entry_id = entry["id"]
        if entry_id in entries:
            raise RuntimeError(f"Duplicate legacy v3 entry id: {_stringify(entry_id)}")
        indexed = _index_legacy_v3_entry(entry, line_number, next_seq, entries)
        entries[entry_id] = indexed
        if _is_retained_entry_type(indexed.type):
            next_seq += 1
        final_id = entry_id
        if entry.get("type") == "session_info":
            name = entry.get("name")
        usage = _legacy_entry_usage(entry)
        if usage is not None:
            imported_usage = add_usage(imported_usage, usage)
    return _LegacyV3Inventory(
        entries=entries,
        imported_usage=imported_usage,
        name=name,
        final_id=final_id,
        next_seq=next_seq,
    )


def _committed_value_set(address: Value, value: Any, seq: int) -> CommittedWrite:
    """``{...setValue(address, value), seq}``: a derived value write with its sequence."""
    write = set_value(address, value)
    return CommittedWrite(
        kind=write.kind,
        op=write.op,
        seq=seq,
        namespace=write.namespace,
        key=write.key,
        value=write.value,
    )


def _normalize_legacy_v3_values(inventory: _LegacyV3Inventory) -> List[CommittedWrite]:
    """Derive the current values a v3 file implies for the main branch."""
    entries = inventory.entries
    resolve_legacy_id = _create_legacy_id_resolver(entries)
    next_seq = inventory.next_seq
    values: List[CommittedWrite] = []

    def take_seq() -> int:
        nonlocal next_seq
        seq = next_seq
        next_seq += 1
        return seq

    # session name
    if inventory.name:
        values.append(_committed_value_set(session_name(), inventory.name, take_seq()))

    # labels
    labels: Dict[str, str] = {}
    for entry in entries.values():
        if entry.type != "label":
            continue
        target_id = resolve_legacy_id(entry.target_id)
        if target_id is None:
            continue
        if entry.label:
            labels[target_id] = entry.label
        else:
            labels.pop(target_id, None)
    for target_id, label in labels.items():
        values.append(_committed_value_set(entry_label(target_id), label, take_seq()))

    # branch tip
    values.append(
        _committed_value_set(branch_tip("main"), resolve_legacy_id(inventory.final_id), take_seq())
    )

    # configuration
    configuration = _selected_configuration(entries, inventory.final_id)
    if configuration is not None:
        values.append(
            _committed_value_set(
                lane_config("main"), _lane_configuration_json(configuration), take_seq()
            )
        )
        values.append(_committed_value_set(lane_state("main"), LaneState().to_json(), take_seq()))
    return values


class LegacyV3Source:
    """A captured legacy file exposed as repeatable logical v4 writes.

    Each pass reopens the path; callers must not replace or edit the source between
    passes. Structural indexes, label/configuration metadata, and derived current
    values survive between scans.
    """

    def __init__(
        self,
        file_system: FileSystem,
        path: str,
        header: JsonlStorageHeader,
        entries: Dict[str, _LegacyV3IndexEntry],
        imported_usage: Usage,
        values: List[CommittedWrite],
        next_seq: int,
    ) -> None:
        """Internal constructor; capture a source with :meth:`read`."""
        self._file_system = file_system
        self._path = path
        self.header = header
        self._entries = entries
        self._resolve_legacy_id = _create_legacy_id_resolver(entries)
        self.imported_usage = imported_usage
        self.values = values
        self.next_seq = next_seq

    @classmethod
    async def read(cls, file_system: FileSystem, path: str, context: Context) -> "LegacyV3Source":
        """Scan complete v3 records without modifying the file, ignoring an unterminated final line.

        Build parent mappings, assign IDs stable for this source instance, and derive
        current values and imported usage. Retain metadata, not conversation payloads or
        an open reader. ``writes()`` reopens the path to materialize captured records and
        resolve their payload-specific references.
        """
        reader = file_value(
            await file_system.open_text_line_reader(path, context),
            f"Failed to open legacy v3 source {path}",
        )
        try:
            parsed = await read_jsonl_header(reader, path, context)
            if parsed.format != "v3-legacy":
                raise RuntimeError(f"Invalid legacy v3 JSONL storage {path}: expected format 3 header")
            inventory = await _read_legacy_v3_inventory(reader, context)
            values = _normalize_legacy_v3_values(inventory)
            return cls(
                file_system,
                path,
                await normalize_legacy_v3_header(file_system, parsed.header, context),
                inventory.entries,
                inventory.imported_usage,
                values,
                inventory.next_seq + len(values),
            )
        finally:
            await reader.close()

    def entry_structures(self) -> Iterator[LegacyV3EntryStructure]:
        """Captured retained entries: imported ID, imported parent ID, and sequence."""
        for entry in self._entries.values():
            if not _is_retained_entry_type(entry.type):
                continue
            yield LegacyV3EntryStructure(
                id=entry.mapped_id,
                parent_id=self._resolve_legacy_id(entry.parent_id),
                seq=entry.seq,
            )

    def translate_fork_entry_id(self, legacy_id: str) -> str:
        """Map one legacy fork target onto its imported ID."""
        entry = self._entries.get(legacy_id)
        if entry is None:
            raise RuntimeError(f"Legacy v3 fork entry does not exist: {legacy_id}")
        if not _is_retained_entry_type(entry.type):
            raise RuntimeError(f"Legacy v3 fork entry is not a retained entry: {legacy_id}")
        return entry.mapped_id

    def _collect_required_tail_message_ids(
        self, is_entry_selected: Optional[Callable[[str], bool]] = None
    ) -> Set[str]:
        required_ids: Set[str] = set()
        for entry in self._entries.values():
            if entry.type != "compaction":
                continue
            compaction_is_selected = is_entry_selected is None or is_entry_selected(entry.mapped_id)
            if not compaction_is_selected:
                continue

            # Walk from the compaction's parent through firstKeptEntryId, inclusive.
            for tail_entry in _retained_tail_structure(entry, self._entries):
                can_produce_context_message = (
                    _is_retained_entry_type(tail_entry.type) and tail_entry.type != "custom"
                )
                if can_produce_context_message:
                    required_ids.add(tail_entry.id)
        return required_ids

    async def writes(
        self,
        context: Context,
        is_entry_selected: Optional[Callable[[str], bool]] = None,
    ) -> AsyncIterator[CommittedWrite]:
        """Stream normalized v4 entries, optionally filtered by imported ID, then derived values.

        Each pass owns its reader and message cache, sharing only the captured IDs and
        metadata.
        """
        required_tail_message_ids = self._collect_required_tail_message_ids(is_entry_selected)
        # Keep needed context messages for this entire pass; tails may revisit old or shared branches.
        tail_messages_by_legacy_id: Dict[str, Any] = {}
        async for entry, indexed in self._read_captured_entries(context):
            if entry["id"] in required_tail_message_ids:
                message = _project_context_message(entry, self._resolve_legacy_id)
                if message is not None:
                    tail_messages_by_legacy_id[entry["id"]] = message
            if not _is_retained_entry_type(indexed.type) or not _is_retained_entry_type(entry.get("type")):
                continue
            if is_entry_selected is not None and not is_entry_selected(indexed.mapped_id):
                continue
            retained_tail: List[Any] = []
            if indexed.type == "compaction":
                for ancestor in _retained_tail_structure(indexed, self._entries):
                    message = tail_messages_by_legacy_id.get(ancestor.id)
                    if message is not None:
                        retained_tail.append(message)
                retained_tail.reverse()
            yield _normalize_retained_entry(entry, indexed, retained_tail, self._resolve_legacy_id)
        for value_write in self.values:
            yield value_write

    async def _read_captured_entries(
        self, context: Context
    ) -> AsyncIterator[Tuple[_LegacyV3Entry, _LegacyV3IndexEntry]]:
        """Replay only the captured prefix and verify its physical identities before materialization."""
        reader = file_value(
            await self._file_system.open_text_line_reader(self._path, context),
            f"Failed to reopen legacy v3 source {self._path}",
        )
        try:
            parsed = await read_jsonl_header(reader, self._path, context)
            if (
                parsed.format != "v3-legacy"
                or parsed.header.id != self.header.id
                or parsed.header.cwd != self.header.cwd
            ):
                raise RuntimeError("Legacy v3 source header changed")
            captured = list(self._entries.values())
            lines = _complete_lines(
                file_value(await reader.read_lines(None), "Failed to reread legacy v3 source")
            )
            if len(lines) < len(captured):
                raise RuntimeError("Legacy v3 source ended before captured entries")
            for text, indexed in zip(lines, captured):
                entry = _parse_legacy_v3_entry(text)
                if entry["id"] != indexed.id or entry["type"] != indexed.type:
                    raise RuntimeError("Legacy v3 source changed")
                yield entry, indexed
        finally:
            await reader.close()
