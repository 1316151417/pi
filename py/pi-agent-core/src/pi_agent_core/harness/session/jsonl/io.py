"""JSONL transaction IO ported from ``session/jsonl/io.ts``."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, List, Optional

from ...._chord.context import Context
from ...result import Result
from ...types import FileSystem, get_or_throw
from ..commit import CommittedWrite
from .codec import parse_jsonl_session_header
from .types import JsonlStorageHeader

__all__ = [
    "file_value",
    "read_jsonl_header",
    "parse_jsonl_transaction",
    "serialize_jsonl_transaction",
    "to_json_value",
    "publish_file_atomically",
    "publish_jsonl",
]


def to_json_value(value: Any) -> Any:
    """Normalize one durable value into a JSON-serializable tree.

    Durable values are opaque to the value store: a same-process write keeps the
    dataclass while a replay yields plain mappings. Serialization is the only
    place that has to bridge the two, so it walks the tree and lets every node
    with an explicit ``to_json()`` describe its own wire shape.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        # A wire shape may itself embed further durable nodes, so keep walking.
        return to_json_value(to_json())
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def file_value(result: Result, action: str) -> Any:
    """Unwrap a filesystem ``Result`` or raise a contextual error."""
    if not bool(getattr(result, "ok", False)):
        error = RuntimeError(f"{action}: {result.error.message}")
        error.__cause__ = result.error
        raise error
    return result.value


async def read_jsonl_header(reader: Any, path: str, context: Context) -> Any:
    """Read and parse the header line through a text line reader."""
    line_result = file_value(await reader.read_lines(1), f"Failed to read JSONL storage {path}")
    line = line_result[0] if line_result else None
    if line is None or line == "":
        raise RuntimeError(f"Invalid JSONL storage {path}: missing header")
    parsed = parse_jsonl_session_header(line)
    if not parsed.ok:
        error = RuntimeError(f"Invalid JSONL storage {path}: invalid header")
        error.__cause__ = parsed.error
        raise error
    return parsed.value


def _require_safe_integer(value: Any, field_name: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RuntimeError(f"Invalid JSONL {field_name}")


def _parse_committed_write(value: Any) -> CommittedWrite:
    if not isinstance(value, dict):
        raise RuntimeError("Invalid JSONL transaction write")
    _require_safe_integer(value.get("seq"), "write seq", 1)
    kind = value.get("kind")
    if kind == "entry":
        _require_safe_integer(value.get("timestamp"), "entry timestamp", 0)
        return CommittedWrite(
            kind="entry",
            seq=value["seq"],
            timestamp=value.get("timestamp", 0),
            id=value.get("id"),
            parent_id=value.get("parentId"),
            entry=_entry_from_json(value),
        )
    if kind == "usage":
        return CommittedWrite(
            kind="usage",
            seq=value["seq"],
            id=value.get("id"),
            entry=_usage_row_from_json(value),
        )
    if kind == "value":
        if value.get("op") == "set":
            return CommittedWrite(
                kind="value",
                op="set",
                seq=value["seq"],
                namespace=value.get("namespace"),
                key=value.get("key"),
                value=value.get("value"),
            )
        if value.get("op") == "delete":
            return CommittedWrite(
                kind="value",
                op="delete",
                seq=value["seq"],
                namespace=value.get("namespace"),
                key=value.get("key"),
            )
        raise RuntimeError(f"Invalid JSONL value operation: {value.get('op')}")
    if kind == "list":
        if value.get("op") == "append":
            return CommittedWrite(
                kind="list",
                op="append",
                seq=value["seq"],
                namespace=value.get("namespace"),
                key=value.get("key"),
                value=value.get("value"),
            )
        if value.get("op") == "delete":
            return CommittedWrite(
                kind="list",
                op="delete",
                seq=value["seq"],
                namespace=value.get("namespace"),
                key=value.get("key"),
            )
        raise RuntimeError(f"Invalid JSONL list operation: {value.get('op')}")
    raise RuntimeError(f"Invalid JSONL write kind: {kind}")


def _entry_from_json(value: dict) -> Any:
    """Rebuild the entry object carried on a committed entry write.

    A committed entry write is the entry record itself plus a ``kind`` tag, so
    the entry fields sit at the top level of the write.
    """
    from ..types import (
        BranchSummaryEntry,
        CompactionEntry,
        CustomEntry,
        MessageEntry,
    )
    from pi_ai.types import message_from_json

    entry_data = value
    if entry_data.get("type") == "message" or "message" in entry_data:
        return MessageEntry(
            id=entry_data["id"],
            parent_id=entry_data.get("parentId"),
            seq=entry_data.get("seq", 0),
            timestamp=entry_data.get("timestamp", 0),
            message=message_from_json(entry_data["message"]),
        )
    if entry_data.get("type") == "compaction":
        return CompactionEntry(
            id=entry_data["id"],
            parent_id=entry_data.get("parentId"),
            seq=entry_data.get("seq", 0),
            timestamp=entry_data.get("timestamp", 0),
            summary=entry_data.get("summary", ""),
            retained_tail=[message_from_json(m) for m in entry_data.get("retainedTail") or []],
            tokens_before=entry_data.get("tokensBefore", 0),
            details=entry_data.get("details"),
            from_hook=bool(entry_data.get("fromHook", False)),
        )
    if entry_data.get("type") == "branch_summary":
        return BranchSummaryEntry(
            id=entry_data["id"],
            parent_id=entry_data.get("parentId"),
            seq=entry_data.get("seq", 0),
            timestamp=entry_data.get("timestamp", 0),
            from_id=entry_data.get("fromId"),
            summary=entry_data.get("summary", ""),
            details=entry_data.get("details"),
            from_hook=bool(entry_data.get("fromHook", False)),
        )
    if entry_data.get("type") == "custom":
        return CustomEntry(
            id=entry_data["id"],
            parent_id=entry_data.get("parentId"),
            seq=entry_data.get("seq", 0),
            timestamp=entry_data.get("timestamp", 0),
            custom_type=entry_data.get("customType", ""),
            data=entry_data.get("data"),
        )
    raise RuntimeError(f"Invalid JSONL entry type: {entry_data.get('type')}")


def _usage_row_from_json(value: dict) -> Any:
    from pi_ai.types import usage_from_json
    from ..types import UsageRow

    row = value.get("row") if isinstance(value.get("row"), dict) else value
    return UsageRow(
        id=row.get("id", value.get("id", "")),
        usage=usage_from_json(row.get("usage")) or _empty_usage(),
        entry_id=row.get("entryId"),
        adjustment=bool(row.get("adjustment", False)),
        details=row.get("details"),
        seq=value.get("seq", 0),
    )


def _empty_usage():
    from pi_ai.types import Usage

    return Usage()


def parse_jsonl_transaction(line: str) -> List[CommittedWrite]:
    """Parse one JSONL transaction line into committed writes."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        wrapped = RuntimeError("Invalid JSONL transaction: not valid JSON")
        wrapped.__cause__ = error
        raise wrapped
    items = value if isinstance(value, list) else [value]
    return [_parse_committed_write(item) for item in items]


def serialize_jsonl_transaction(writes: List[CommittedWrite]) -> str:
    """Serialize committed writes as one JSONL transaction line payload."""
    payloads = [_committed_write_to_json(write) for write in writes]
    return json.dumps(payloads[0] if len(payloads) == 1 else payloads)


def _committed_write_to_json(write: CommittedWrite) -> dict:
    # A committed entry/usage write is the record itself plus a ``kind`` tag, so
    # the payload stays flat on the wire exactly like the TS write object.
    data: dict = {"kind": write.kind, "seq": write.seq}
    if write.kind == "entry":
        data.update(to_json_value(write.entry))
    elif write.kind == "usage":
        data.update(to_json_value(write.entry))
    else:
        data["op"] = write.op
        data["namespace"] = write.namespace
        data["key"] = write.key
        # A set/append carries its payload even when that payload is JSON null;
        # a delete has no payload key at all, matching the TS write object.
        if write.op in ("set", "append"):
            data["value"] = to_json_value(write.value)
    return data


async def publish_file_atomically(
    file_system: FileSystem,
    destination_path: str,
    context: Context,
    write_content: Callable[[Callable[[str], Awaitable[None]]], Awaitable[None]],
) -> None:
    """Publish only after the callback succeeds; it must await each append before returning."""
    temp_path = f"{destination_path}.tmp"
    try:
        file_value(
            await file_system.write_file(temp_path, "", context),
            f"Failed to stage JSONL storage {destination_path}",
        )

        async def _append(content: str) -> None:
            file_value(
                await file_system.append_file(temp_path, content, context),
                f"Failed to append JSONL storage {destination_path}",
            )

        await write_content(_append)
        file_value(
            await file_system.rename_file(temp_path, destination_path, context),
            f"Failed to publish JSONL storage {destination_path}",
        )
    except BaseException:
        await file_system.remove_file(temp_path, context)
        raise


async def publish_jsonl(
    file_system: FileSystem,
    destination_path: str,
    header: JsonlStorageHeader,
    context: Context,
    write_transactions: Callable[[Callable[[List[CommittedWrite]], Awaitable[None]]], Awaitable[None]],
) -> None:
    """Stream a header and complete transactions through the shared atomic publisher."""

    async def _write(append: Callable[[str], Awaitable[None]]) -> None:
        await append(f"{json.dumps(header.to_json())}\n")

        async def _append_transaction(writes: List[CommittedWrite]) -> None:
            await append(f"{serialize_jsonl_transaction(writes)}\n")

        await write_transactions(_append_transaction)

    await publish_file_atomically(file_system, destination_path, context, _write)
