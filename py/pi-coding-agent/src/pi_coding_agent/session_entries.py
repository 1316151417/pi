"""Session entry migrations and context projection from ``core/session-manager.ts``."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pi_ai.types import (
    ImageContent, Message, SystemMessage, TextContent,
    content_block_from_json, message_from_json,
)

from .messages import (
    BashExecutionMessage, CodingAgentMessage, CustomMessage, create_branch_summary_message,
    create_compaction_summary_message, create_custom_message,
)

CURRENT_SESSION_VERSION = 3
type FileEntry = dict[str, object]
type SessionEntry = dict[str, object]
_UNSET = object()


@dataclass
class SessionContext:
    messages: list[CodingAgentMessage]
    thinking_level: str
    model: dict[str, str] | None


def _short_id(used: set[str]) -> str:
    for _ in range(100):
        candidate = str(uuid.uuid4())[:8]
        if candidate not in used:
            return candidate
    return str(uuid.uuid4())


def migrate_session_entries(entries: list[FileEntry]) -> bool:
    header = next((entry for entry in entries if entry.get("type") == "session"), None)
    version = header.get("version", 1) if header is not None else 1
    if isinstance(version, (int, float)) and version >= CURRENT_SESSION_VERSION:
        return False
    if not isinstance(version, (int, float)) or version < 2:
        used: set[str] = set()
        previous_id: str | None = None
        for entry in entries:
            if entry.get("type") == "session":
                entry["version"] = 2
                continue
            entry_id = _short_id(used)
            used.add(entry_id)
            entry["id"] = entry_id
            entry["parentId"] = previous_id
            previous_id = entry_id
            if entry.get("type") == "compaction":
                index = entry.pop("firstKeptEntryIndex", None)
                if isinstance(index, int) and 0 <= index < len(entries):
                    target = entries[index]
                    if target.get("type") != "session":
                        entry["firstKeptEntryId"] = target.get("id")
    if not isinstance(version, (int, float)) or version < 3:
        for entry in entries:
            if entry.get("type") == "session":
                entry["version"] = 3
            elif entry.get("type") == "message":
                message = entry.get("message")
                if isinstance(message, dict) and message.get("role") == "hookMessage":
                    message["role"] = "custom"
    return True


def parse_session_entries(content: str) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            parsed: object = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            entries.append(cast(FileEntry, parsed))
    return entries


def get_latest_compaction_entry(entries: Sequence[SessionEntry]) -> SessionEntry | None:
    return next((entry for entry in reversed(entries) if entry.get("type") == "compaction"), None)


def build_session_path(
    entries: Sequence[SessionEntry], leaf_id: str | None | object = _UNSET,
    by_id: Mapping[str, SessionEntry] | None = None,
) -> list[SessionEntry]:
    index = by_id if by_id is not None else {
        cast(str, entry["id"]): entry for entry in entries if isinstance(entry.get("id"), str)
    }
    if leaf_id is None:
        return []
    leaf = index.get(leaf_id) if isinstance(leaf_id, str) else None
    if leaf is None:
        leaf = entries[-1] if entries else None
    path: list[SessionEntry] = []
    current = leaf
    while current is not None:
        path.append(current)
        parent_id = current.get("parentId")
        current = index.get(parent_id) if isinstance(parent_id, str) else None
    path.reverse()
    return path


def _session_context_settings(path: Sequence[SessionEntry]) -> tuple[str, dict[str, str] | None]:
    thinking_level = "off"
    model: dict[str, str] | None = None
    for entry in path:
        kind = entry.get("type")
        if kind == "thinking_level_change":
            thinking_level = cast(str, entry["thinkingLevel"])
        elif kind == "model_change":
            model = {"provider": cast(str, entry["provider"]), "modelId": cast(str, entry["modelId"])}
        elif kind == "message":
            message = entry.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                model = {
                    "provider": cast(str, message.get("provider", "")),
                    "modelId": cast(str, message.get("model", "")),
                }
    return thinking_level, model


def _decode_message(raw: Mapping[str, object]) -> CodingAgentMessage:
    role = raw.get("role")
    if role == "bashExecution":
        return BashExecutionMessage(
            command=cast(str, raw.get("command", "")),
            output=cast(str, raw.get("output", "")),
            exit_code=cast(int | None, raw.get("exitCode")),
            cancelled=bool(raw.get("cancelled", False)),
            truncated=bool(raw.get("truncated", False)),
            timestamp=cast(int, raw.get("timestamp", 0)),
            full_output_path=cast(str | None, raw.get("fullOutputPath")),
            exclude_from_context=bool(raw.get("excludeFromContext", False)),
        )
    if role == "custom":
        content = raw.get("content", [])
        blocks = (
            [cast(TextContent | ImageContent, content_block_from_json(block)) for block in content if isinstance(block, dict)]
            if isinstance(content, list) else content
        )
        return CustomMessage(
            custom_type=cast(str, raw.get("customType", "")),
            content=cast(str | list[TextContent | ImageContent], blocks),
            display=bool(raw.get("display", False)),
            timestamp=cast(int, raw.get("timestamp", 0)),
            details=raw.get("details"),
        )
    return cast(Message, message_from_json(cast(Mapping[str, object], raw)))


def session_entry_to_context_messages(entry: SessionEntry) -> list[CodingAgentMessage]:
    kind = entry.get("type")
    timestamp = cast(str, entry.get("timestamp", "1970-01-01T00:00:00Z"))
    if kind == "message":
        message = entry.get("message")
        if not isinstance(message, dict):
            return []
        adjusted = dict(message)
        if adjusted.get("content") is None:
            if adjusted.get("role") == "system":
                adjusted["content"] = ""
            elif adjusted.get("role") in ("user", "assistant", "toolResult"):
                adjusted["content"] = []
        return [_decode_message(adjusted)]
    if kind == "custom_message":
        content = entry.get("content")
        blocks = (
            [cast(TextContent | ImageContent, content_block_from_json(block)) for block in content if isinstance(block, dict)]
            if isinstance(content, list) else content
        )
        return [create_custom_message(
            cast(str, entry.get("customType", "")),
            cast(str | list[TextContent | ImageContent], blocks if blocks is not None else []),
            bool(entry.get("display", False)), entry.get("details"), timestamp,
        )]
    if kind == "branch_summary" and entry.get("summary"):
        return [create_branch_summary_message(
            cast(str, entry["summary"]), cast(str, entry.get("fromId", "")), timestamp,
        )]
    if kind == "compaction":
        summary = create_compaction_summary_message(
            cast(str, entry.get("summary", "")), cast(int, entry.get("tokensBefore", 0)), timestamp,
        )
        system_message = entry.get("systemMessage")
        if isinstance(system_message, dict):
            return [cast(SystemMessage, message_from_json(system_message)), summary]
        return [summary]
    return []


def build_context_entries(
    entries: Sequence[SessionEntry], leaf_id: str | None | object = _UNSET,
    by_id: Mapping[str, SessionEntry] | None = None,
) -> list[SessionEntry]:
    path = build_session_path(entries, leaf_id, by_id)
    compaction = get_latest_compaction_entry(path)
    if compaction is None:
        return path
    compaction_index = next((i for i, entry in enumerate(path) if entry.get("id") == compaction.get("id")), -1)
    if compaction_index < 0:
        return path
    result = [compaction]
    found_first_kept = False
    for entry in path[:compaction_index]:
        if entry.get("id") == compaction.get("firstKeptEntryId"):
            found_first_kept = True
        message = entry.get("message")
        is_system = (
            entry.get("type") == "message" and isinstance(message, dict)
            and message.get("role") == "system"
        )
        if found_first_kept and not is_system:
            result.append(entry)
    result.extend(path[compaction_index + 1:])
    return result


def build_session_context(
    entries: Sequence[SessionEntry], leaf_id: str | None | object = _UNSET,
    by_id: Mapping[str, SessionEntry] | None = None,
) -> SessionContext:
    path = build_session_path(entries, leaf_id, by_id)
    thinking_level, model = _session_context_settings(path)
    messages = [
        message for entry in build_context_entries(entries, leaf_id, by_id)
        for message in session_entry_to_context_messages(entry)
    ]
    return SessionContext(messages, thinking_level, model)


__all__ = [
    "CURRENT_SESSION_VERSION", "FileEntry", "SessionEntry", "SessionContext",
    "migrate_session_entries", "parse_session_entries", "get_latest_compaction_entry",
    "build_session_path", "session_entry_to_context_messages", "build_context_entries",
    "build_session_context",
]
