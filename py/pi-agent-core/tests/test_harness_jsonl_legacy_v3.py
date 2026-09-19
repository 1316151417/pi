"""Tests for the legacy v3 JSONL decoding port (``jsonl/legacy_v3.py``).

Fixtures are small v3-legacy JSONL strings served by an in-memory filesystem whose
line reader mirrors the in-tree local env: file content is split on ``"\\n"`` and the
trailing remainder is kept.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core._pi_ai.types import Usage
from pi_agent_core.harness.messages import CustomMessage
from pi_agent_core.harness.result import Result, err, ok
from pi_agent_core.harness.session.jsonl.legacy_v3 import (
    LegacyV3EntryStructure,
    LegacyV3Source,
    metadata_from_legacy_v3_header,
    normalize_legacy_v3_header,
)
from pi_agent_core.harness.session.values import (
    branch_tip,
    entry_label,
    lane_config,
    lane_state,
    session_name,
)
from pi_agent_core.harness.types import FileError
from pi_agent_core.harness.session.jsonl.codec import LegacyV3SessionHeader

NOW = 1_700_000_000_000
STAMP = datetime.fromtimestamp(NOW / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
PATH = "legacy.jsonl"
HEADER = {"type": "session", "version": 3, "id": "legacy", "timestamp": STAMP, "cwd": "/workspace"}


def _ms(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)


def _uuid_timestamp_prefix(mapped_id: str) -> str:
    return mapped_id.replace("-", "")[:12]


def record(entry_type: str, entry_id: str, parent_id: Optional[str], **fields: Any) -> dict:
    return {"type": entry_type, "id": entry_id, "parentId": parent_id, "timestamp": STAMP, **fields}


def message(entry_id: str, parent_id: Optional[str], content: Optional[str] = None) -> dict:
    return record(
        "message",
        entry_id,
        parent_id,
        message={"role": "user", "content": entry_id if content is None else content, "timestamp": NOW},
    )


class FakeReader:
    def __init__(self, content: str) -> None:
        self._lines = content.split("\n")
        self._index = 0

    async def read_lines(self, max_lines: Optional[int] = None) -> Result:
        if max_lines is None:
            chunk = self._lines[self._index :]
            self._index = len(self._lines)
        else:
            chunk = self._lines[self._index : self._index + max_lines]
            self._index += len(chunk)
        return ok(list(chunk))

    async def close(self) -> None:
        return None


class FakeFileSystem:
    def __init__(self, files: Dict[str, str]) -> None:
        self.files = dict(files)
        self.cwd = "/workspace"

    async def open_text_line_reader(self, path: str, context: Any) -> Result:
        content = self.files.get(path)
        if content is None:
            return err(FileError(code="not_found", message=f"File not found: {path}", path=path))
        return ok(FakeReader(content))

    async def read_text_lines(self, path: str, options: Optional[dict], context: Any) -> Result:
        content = self.files.get(path)
        if content is None:
            return err(FileError(code="not_found", message=f"File not found: {path}", path=path))
        lines = content.split("\n")
        max_lines = (options or {}).get("maxLines")
        if isinstance(max_lines, int) and max_lines >= 0:
            lines = lines[:max_lines]
        return ok(lines)


def fixture(records: List[dict], suffix: str = "", header: Optional[dict] = None) -> FakeFileSystem:
    content = "\n".join(json.dumps(item) for item in [header or HEADER, *records]) + "\n" + suffix
    return FakeFileSystem({PATH: content})


async def collect(source: LegacyV3Source, selected: Optional[Any] = None) -> List[Any]:
    writes: List[Any] = []
    if selected is None:
        async for write in source.writes(BACKGROUND_CONTEXT):
            writes.append(write)
    else:
        async for write in source.writes(BACKGROUND_CONTEXT, selected):
            writes.append(write)
    return writes


def structures(source: LegacyV3Source) -> List[LegacyV3EntryStructure]:
    return list(source.entry_structures())


async def test_imports_records_mints_ids_and_derives_values() -> None:
    file_system = fixture(
        [
            message("m1", None, "hi"),
            record("model_change", "mc1", "m1", provider="faux", modelId="faux-1"),
            record("thinking_level_change", "tl1", "mc1", thinkingLevel="high"),
            record("session_info", "si1", "tl1", name="My session"),
            record("label", "l1", "si1", targetId="m1", label="first"),
        ]
    )
    source = await LegacyV3Source.read(file_system, PATH, BACKGROUND_CONTEXT)

    # Discarded types keep no identity of their own; only the message is a retained entry.
    captured = structures(source)
    assert len(captured) == 1
    mapped = captured[0].id
    assert captured[0].parent_id is None
    assert captured[0].seq == 1
    assert _uuid_timestamp_prefix(mapped) == f"{NOW:012x}"

    assert [write.seq for write in source.values] == [2, 3, 4, 5, 6]
    assert (source.values[0].namespace, source.values[0].key) == (session_name().namespace, "")
    assert source.values[0].value == "My session"
    assert source.values[1].key == entry_label(mapped).key
    assert source.values[1].value == "first"
    # The final record is a discarded label, so the branch tip resolves to its retained ancestor.
    assert source.values[2].key == branch_tip("main").key
    assert source.values[2].value == mapped
    assert source.values[3].key == lane_config("main").key
    assert source.values[3].value == {
        "model": {"provider": "faux", "modelId": "faux-1"},
        "thinkingLevel": "high",
        "activeToolNames": [],
    }
    assert source.values[4].key == lane_state("main").key
    assert source.values[4].value == {"currentOperationId": None, "lastOperationId": None, "inbox": []}
    assert source.next_seq == 7

    writes = await collect(source)
    assert len(writes) == 6
    entry_write = writes[0]
    assert entry_write.kind == "entry"
    assert entry_write.id == mapped
    assert entry_write.parent_id is None
    assert entry_write.seq == 1
    assert entry_write.timestamp == NOW
    assert entry_write.entry.id == mapped
    assert entry_write.entry.type == "message"
    assert entry_write.entry.message.content == "hi"
    assert writes[1:] == source.values


async def test_custom_message_and_compaction_entries() -> None:
    file_system = fixture(
        [
            message("before", None),
            record("session_info", "boundary", "before"),
            message("kept", "boundary", "Unicode é漢字"),
            record(
                "custom_message",
                "custom",
                "kept",
                customType="note",
                content="body",
                details={"k": 1},
                display=True,
            ),
            record(
                "compaction",
                "selected",
                "custom",
                summary="summary",
                firstKeptEntryId="boundary",
                tokensBefore=20,
                fromHook=True,
                details={"a": 2},
                usage={"input": 7, "output": 2, "totalTokens": 9},
            ),
        ]
    )
    source = await LegacyV3Source.read(file_system, PATH, BACKGROUND_CONTEXT)
    captured = structures(source)
    assert [structure.seq for structure in captured] == [1, 2, 3, 4]

    selected_id = captured[-1].id
    writes = await collect(source, lambda entry_id: entry_id == selected_id)
    assert len(writes) == 2  # The selected compaction plus the single derived branch tip value.
    compaction = writes[0].entry
    assert compaction.type == "compaction"
    assert compaction.seq == 4
    assert compaction.tokens_before == 20
    assert compaction.from_hook is True
    assert compaction.details == {"a": 2}
    assert compaction.usage is not None and compaction.usage.input == 7
    assert compaction.parent_id == captured[2].id  # The discarded session_info keeps no sequence slot.
    # The tail walks physical ancestry down to firstKeptEntryId, oldest message first.
    assert [message.content for message in compaction.retained_tail] == ["Unicode é漢字", "body"]
    assert compaction.retained_tail[0].role == "user"
    assert compaction.retained_tail[0].timestamp == NOW
    assert compaction.retained_tail[1].role == "custom"

    custom = (await collect(source))[2].entry
    assert isinstance(custom.message, CustomMessage)
    assert custom.message.role == "custom"
    assert custom.message.custom_type == "note"
    assert custom.message.display is True
    assert custom.message.details == {"k": 1}
    assert custom.message.timestamp == NOW


async def test_imported_usage_sums_assistant_and_tool_result_branches() -> None:
    file_system = fixture(
        [
            message("u1", None),
            record(
                "message",
                "a1",
                "u1",
                message={
                    "role": "assistant",
                    "content": [{"type": "text", "text": "yo"}],
                    "usage": {"input": 3, "output": 4, "cacheRead": 5, "cacheWrite": 6, "totalTokens": 18},
                    "stopReason": "stop",
                    "timestamp": NOW,
                },
            ),
            record(
                "compaction",
                "c1",
                "a1",
                summary="kept",
                firstKeptEntryId="u1",
                tokensBefore=1,
                usage={"input": 1, "output": 1, "totalTokens": 2},
            ),
        ]
    )
    source = await LegacyV3Source.read(file_system, PATH, BACKGROUND_CONTEXT)
    assert isinstance(source.imported_usage, Usage)
    assert source.imported_usage.input == 4
    assert source.imported_usage.output == 5
    assert source.imported_usage.cache_read == 5
    assert source.imported_usage.cache_write == 6
    assert source.imported_usage.total_tokens == 20


async def test_selection_filter_still_yields_values_and_reuses_tail_cache() -> None:
    file_system = fixture(
        [
            message("root", None),
            message("left", "root"),
            record(
                "compaction",
                "left-compaction",
                "left",
                summary="left summary",
                firstKeptEntryId="root",
                tokensBefore=10,
            ),
            message("right", "root"),
            record(
                "compaction",
                "right-compaction",
                "right",
                summary="right summary",
                firstKeptEntryId="root",
                tokensBefore=20,
            ),
        ]
    )
    source = await LegacyV3Source.read(file_system, PATH, BACKGROUND_CONTEXT)
    captured = structures(source)
    assert [structure.seq for structure in captured] == [1, 2, 3, 4, 5]

    assert await collect(source, lambda _entry_id: False) == source.values

    selected = {captured[2].id, captured[4].id}
    writes = await collect(source, lambda entry_id: entry_id in selected)
    assert [write.entry.summary for write in writes[:2]] == ["left summary", "right summary"]
    assert [message.content for message in writes[0].entry.retained_tail] == ["root", "left"]
    assert [message.content for message in writes[1].entry.retained_tail] == ["root", "right"]
    assert writes[2:] == source.values
    assert await collect(source, lambda entry_id: entry_id in selected) == writes


async def test_ignores_torn_tail_and_captured_prefix_survives_later_appends() -> None:
    torn = json.dumps(message("torn", "a"))
    file_system = fixture([message("a", None)], suffix=torn)
    source = await LegacyV3Source.read(file_system, PATH, BACKGROUND_CONTEXT)
    assert source.next_seq == 3

    # The torn tail is not captured, so the later record is invisible to the captured prefix.
    file_system.files[PATH] += "\n" + json.dumps(message("later", "torn")) + "\n"
    writes = await collect(source)
    assert len(writes) == 2
    assert writes[0].entry.message.content == "a"


async def test_errors_mirror_the_typescript_messages() -> None:
    duplicate = fixture([message("m1", None), message("m1", "m1")])
    with pytest.raises(RuntimeError, match="Duplicate legacy v3 entry id: m1"):
        await LegacyV3Source.read(duplicate, PATH, BACKGROUND_CONTEXT)

    unsupported = fixture([record("wat", "x", None)])
    with pytest.raises(RuntimeError, match="Unsupported legacy v3 record type: wat"):
        await LegacyV3Source.read(unsupported, PATH, BACKGROUND_CONTEXT)

    missing_type = fixture([{"id": "x", "parentId": None, "timestamp": STAMP}])
    with pytest.raises(RuntimeError, match="Unsupported legacy v3 record type: undefined"):
        await LegacyV3Source.read(missing_type, PATH, BACKGROUND_CONTEXT)

    forward_parent = fixture([message("child", "parent"), message("parent", None)])
    with pytest.raises(
        RuntimeError,
        match="Legacy v3 entry child has a missing or forward parent at line 2: parent",
    ):
        await LegacyV3Source.read(forward_parent, PATH, BACKGROUND_CONTEXT)

    broken_json = FakeFileSystem({PATH: json.dumps(HEADER) + "\nnot json\n"})
    with pytest.raises(RuntimeError, match="Invalid legacy v3 JSONL record: not valid JSON"):
        await LegacyV3Source.read(broken_json, PATH, BACKGROUND_CONTEXT)


async def test_rejects_wrong_format_and_missing_source() -> None:
    v4 = FakeFileSystem(
        {
            PATH: json.dumps(
                {
                    "v": 4,
                    "kind": "header",
                    "id": "x",
                    "storageVersion": 1,
                    "createdAt": 0,
                    "cwd": "/workspace",
                }
            )
            + "\n"
        }
    )
    with pytest.raises(
        RuntimeError, match=f"Invalid legacy v3 JSONL storage {PATH}: expected format 3 header"
    ):
        await LegacyV3Source.read(v4, PATH, BACKGROUND_CONTEXT)

    with pytest.raises(RuntimeError, match=f"Failed to open legacy v3 source {PATH}: File not found"):
        await LegacyV3Source.read(FakeFileSystem({}), PATH, BACKGROUND_CONTEXT)


async def test_header_metadata_resolves_or_retains_the_parent_session() -> None:
    parent_header = {"type": "session", "version": 3, "id": "parent", "timestamp": STAMP, "cwd": "/workspace"}
    file_system = fixture([message("m1", None)], header={**HEADER, "parentSession": "parent.jsonl"})
    file_system.files["parent.jsonl"] = json.dumps(parent_header) + "\n"

    header = LegacyV3SessionHeader(
        id="legacy", timestamp=STAMP, cwd="/workspace", parent_session="parent.jsonl"
    )
    metadata = await metadata_from_legacy_v3_header(file_system, header, BACKGROUND_CONTEXT)
    assert metadata.id == "legacy"
    assert metadata.created_at == NOW
    assert metadata.storage_version == 1
    assert metadata.cwd == "/workspace"
    assert metadata.parent_session_id == "parent"
    assert metadata.legacy_parent_session_path is None

    normalized = await normalize_legacy_v3_header(file_system, header, BACKGROUND_CONTEXT)
    assert normalized.v == 4
    assert normalized.kind == "header"
    assert normalized.id == "legacy"
    assert normalized.created_at == NOW
    assert normalized.parent_session_id == "parent"
    assert normalized.legacy_parent_session_path is None

    # An unreadable parent keeps the path instead of a resolved identity.
    missing_parent_header = LegacyV3SessionHeader(
        id="legacy", timestamp=STAMP, cwd="/workspace", parent_session="gone.jsonl"
    )
    missing = await metadata_from_legacy_v3_header(file_system, missing_parent_header, BACKGROUND_CONTEXT)
    assert missing.parent_session_id is None
    assert missing.legacy_parent_session_path == "gone.jsonl"


async def test_fork_translation_and_source_integrity_checks() -> None:
    file_system = fixture([message("m1", None), record("label", "l1", "m1", targetId="m1", label="x")])
    source = await LegacyV3Source.read(file_system, PATH, BACKGROUND_CONTEXT)
    mapped = structures(source)[0].id

    # Retained records translate directly, discarded ones are rejected as fork targets.
    assert source.translate_fork_entry_id("m1") == mapped
    with pytest.raises(RuntimeError, match="Legacy v3 fork entry is not a retained entry: l1"):
        source.translate_fork_entry_id("l1")
    with pytest.raises(RuntimeError, match="Legacy v3 fork entry does not exist: nope"):
        source.translate_fork_entry_id("nope")

    truncated = fixture([message("m1", None)])
    captured = await LegacyV3Source.read(truncated, PATH, BACKGROUND_CONTEXT)
    truncated.files[PATH] = json.dumps(HEADER) + "\n"
    with pytest.raises(RuntimeError, match="Legacy v3 source ended before captured entries"):
        await collect(captured)

    replaced = fixture([message("m1", None)])
    other = await LegacyV3Source.read(replaced, PATH, BACKGROUND_CONTEXT)
    replaced.files[PATH] = json.dumps({**HEADER, "id": "other"}) + "\n" + json.dumps(message("m1", None)) + "\n"
    with pytest.raises(RuntimeError, match="Legacy v3 source header changed"):
        await collect(other)

    rewritten = fixture([message("m1", None)])
    stale = await LegacyV3Source.read(rewritten, PATH, BACKGROUND_CONTEXT)
    rewritten.files[PATH] = json.dumps(HEADER) + "\n" + json.dumps(message("m2", None)) + "\n"
    with pytest.raises(RuntimeError, match="Legacy v3 source changed"):
        await collect(stale)


async def test_invalid_compaction_boundary_is_reported() -> None:
    file_system = fixture(
        [
            message("root", None),
            record(
                "compaction",
                "c1",
                "root",
                summary="s",
                firstKeptEntryId="absent",
                tokensBefore=1,
            ),
        ]
    )
    source = await LegacyV3Source.read(file_system, PATH, BACKGROUND_CONTEXT)
    with pytest.raises(
        RuntimeError,
        match="Legacy v3 compaction c1 firstKeptEntryId is not on its parent branch: absent",
    ):
        await collect(source)
