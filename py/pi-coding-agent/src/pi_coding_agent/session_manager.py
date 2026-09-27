"""Append-only JSONL session tree from ``core/session-manager.ts``."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from pi_ai.transcript import get_current_system_message
from pi_ai.types import (
    ImageContent, Message, TextContent, Usage,
    content_block_to_json, message_to_json,
)
from pi_ai.uuid_utils import uuidv7

from .config import get_agent_dir
from .messages import (
    BashExecutionMessage, BranchSummaryMessage, CodingAgentMessage,
    CompactionSummaryMessage, CustomMessage,
)
from .paths import normalize_path, resolve_path
from .session_entries import (
    CURRENT_SESSION_VERSION, FileEntry, SessionContext, SessionEntry,
    build_context_entries, build_session_context, migrate_session_entries,
    parse_session_entries,
)

_SESSION_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_MAX_HEADER_SCAN_BYTES = 1024 * 1024


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def assert_valid_session_id(value: str) -> None:
    if _SESSION_ID.fullmatch(value) is None:
        raise ValueError(
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', "
            "and start and end with an alphanumeric character"
        )


def _generate_id(by_id: Mapping[str, SessionEntry]) -> str:
    for _ in range(100):
        value = str(uuid.uuid4())[:8]
        if value not in by_id:
            return value
    return str(uuid.uuid4())


def _agent_dir() -> str:
    return get_agent_dir()


def _default_session_dir_path(cwd: str, agent_dir: str | None = None) -> str:
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir or _agent_dir())
    safe_path = "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", resolved_cwd)) + "--"
    return str(Path(resolved_agent_dir) / "sessions" / safe_path)


def get_default_session_dir(cwd: str, agent_dir: str | None = None) -> str:
    directory = _default_session_dir_path(cwd, agent_dir)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def load_entries_from_file(file_path: str) -> list[FileEntry]:
    path = Path(normalize_path(file_path))
    if not path.exists():
        return []
    entries: list[FileEntry] = []
    pending = ""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for line in stream:
            pending = line if not line.endswith("\n") else ""
            parsed = parse_session_entries(line)
            entries.extend(parsed)
    if not entries:
        return []
    header = entries[0]
    if header.get("type") != "session" or not isinstance(header.get("id"), str):
        return []
    if pending:
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write("\n")
    return entries


def _read_session_header(file_path: str) -> FileEntry | None:
    with open(file_path, "rb") as stream:
        data = stream.read(_MAX_HEADER_SCAN_BYTES + 1)
    if len(data) > _MAX_HEADER_SCAN_BYTES and b"\n" not in data[:_MAX_HEADER_SCAN_BYTES]:
        raise RuntimeError(f"Session header exceeds {_MAX_HEADER_SCAN_BYTES}-byte scan limit: {file_path}")
    for line in data[:_MAX_HEADER_SCAN_BYTES].decode("utf-8", errors="replace").splitlines():
        entries = parse_session_entries(line)
        if not entries:
            continue
        entry = entries[0]
        return entry if entry.get("type") == "session" and isinstance(entry.get("id"), str) else None
    return None


def _header_for_discovery(file_path: str) -> FileEntry | None:
    try:
        return _read_session_header(file_path)
    except (OSError, RuntimeError):
        return None


def find_most_recent_session(session_dir: str, cwd: str | None = None) -> str | None:
    resolved_cwd = resolve_path(cwd) if cwd else None
    try:
        candidates: list[tuple[float, str]] = []
        for path in Path(normalize_path(session_dir)).glob("*.jsonl"):
            header = _header_for_discovery(str(path))
            if header is None:
                continue
            header_cwd = header.get("cwd")
            if resolved_cwd and (not isinstance(header_cwd, str) or not header_cwd or resolve_path(header_cwd) != resolved_cwd):
                continue
            candidates.append((path.stat().st_mtime, str(path)))
        return max(candidates)[1] if candidates else None
    except OSError:
        return None


def _encode_message(message: CodingAgentMessage) -> dict[str, object]:
    if isinstance(message, BashExecutionMessage):
        result: dict[str, object] = {
            "role": "bashExecution", "command": message.command, "output": message.output,
            "exitCode": message.exit_code, "cancelled": message.cancelled,
            "truncated": message.truncated, "timestamp": message.timestamp,
        }
        if message.full_output_path is not None:
            result["fullOutputPath"] = message.full_output_path
        if message.exclude_from_context:
            result["excludeFromContext"] = True
        return result
    if isinstance(message, CustomMessage):
        content: object = (
            message.content if isinstance(message.content, str)
            else [content_block_to_json(block) for block in message.content]
        )
        result = {
            "role": "custom", "customType": message.custom_type,
            "content": content, "display": message.display, "timestamp": message.timestamp,
        }
        if message.details is not None:
            result["details"] = message.details
        return result
    return cast(dict[str, object], message_to_json(cast(Message, message)))


@dataclass
class SessionTreeNode:
    entry: SessionEntry
    children: list[SessionTreeNode] = field(default_factory=list)
    label: str | None = None
    label_timestamp: str | None = None


class SessionManager:
    def __init__(
        self, cwd: str, session_dir: str, session_file: str | None,
        persist: bool, *, session_id: str | None = None,
        parent_session: str | None = None,
        preloaded_entries: list[FileEntry] | None = None,
    ) -> None:
        self.cwd = resolve_path(cwd)
        self.session_dir = normalize_path(session_dir)
        self.persist = persist
        self.session_id = ""
        self.session_file: str | None = None
        self.flushed = False
        self.file_entries: list[FileEntry] = []
        self.by_id: dict[str, SessionEntry] = {}
        self.labels_by_id: dict[str, str] = {}
        self.label_timestamps_by_id: dict[str, str] = {}
        self.leaf_id: str | None = None
        if persist and self.session_dir:
            Path(self.session_dir).mkdir(parents=True, exist_ok=True)
        if session_file:
            self._set_session_file(session_file, preloaded_entries)
        elif preloaded_entries:
            self._load_entries(preloaded_entries, session_id, parent_session)
        else:
            self.new_session(session_id=session_id, parent_session=parent_session)

    def set_session_file(self, session_file: str) -> None:
        self._set_session_file(session_file)

    def _set_session_file(self, session_file: str, preloaded: list[FileEntry] | None = None) -> None:
        explicit = resolve_path(session_file)
        self.session_file = explicit
        if Path(explicit).exists():
            entries = preloaded if preloaded is not None else load_entries_from_file(explicit)
            if not entries:
                if Path(explicit).stat().st_size > 0:
                    raise RuntimeError(f"Session file is not a valid pi session: {explicit}")
                self.new_session()
                self.session_file = explicit
                self._rewrite_file()
                self.flushed = True
                return
            self._load_entries(entries)
            self.flushed = True
        else:
            self.new_session()
            self.session_file = explicit

    def new_session(self, *, session_id: str | None = None, parent_session: str | None = None) -> str | None:
        if session_id is not None:
            assert_valid_session_id(session_id)
        self.session_id = session_id or uuidv7()
        timestamp = _timestamp()
        header: FileEntry = {
            "type": "session", "version": CURRENT_SESSION_VERSION, "id": self.session_id,
            "timestamp": timestamp, "cwd": self.cwd,
        }
        if parent_session is not None:
            header["parentSession"] = parent_session
        self.file_entries = [header]
        self.by_id.clear()
        self.labels_by_id.clear()
        self.label_timestamps_by_id.clear()
        self.leaf_id = None
        self.flushed = False
        if self.persist:
            name = f"{timestamp.replace(':', '-').replace('.', '-')}_{self.session_id}.jsonl"
            self.session_file = str(Path(self.session_dir) / name)
        return self.session_file

    def _load_entries(
        self, entries: list[FileEntry], session_id: str | None = None,
        parent_session: str | None = None,
    ) -> None:
        header = next((entry for entry in entries if entry.get("type") == "session"), None)
        if header is not None:
            self.file_entries = entries
            self.session_id = cast(str, header["id"])
            if migrate_session_entries(self.file_entries):
                self._rewrite_file()
        else:
            self.new_session(session_id=session_id, parent_session=parent_session)
            self.file_entries.extend(entries)
        self._build_index()

    def _build_index(self) -> None:
        self.by_id.clear()
        self.labels_by_id.clear()
        self.label_timestamps_by_id.clear()
        self.leaf_id = None
        for entry in self.file_entries:
            if entry.get("type") == "session":
                continue
            entry_id = cast(str, entry["id"])
            self.by_id[entry_id] = entry
            self.leaf_id = entry_id
            if entry.get("type") == "label":
                target_id = cast(str, entry["targetId"])
                label = entry.get("label")
                if isinstance(label, str) and label:
                    self.labels_by_id[target_id] = label
                    self.label_timestamps_by_id[target_id] = cast(str, entry["timestamp"])
                else:
                    self.labels_by_id.pop(target_id, None)
                    self.label_timestamps_by_id.pop(target_id, None)

    def _rewrite_file(self) -> None:
        if not self.persist or self.session_file is None:
            return
        with open(self.session_file, "w", encoding="utf-8", newline="") as stream:
            for entry in self.file_entries:
                stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _persist(self, entry: SessionEntry) -> None:
        if not self.persist or self.session_file is None:
            return
        has_assistant = any(
            item.get("type") == "message" and isinstance(item.get("message"), dict)
            and cast(dict, item["message"]).get("role") == "assistant"
            for item in self.file_entries
        )
        if not has_assistant:
            if self.flushed:
                with open(self.session_file, "a", encoding="utf-8", newline="") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            else:
                self.flushed = False
            return
        if not self.flushed:
            with open(self.session_file, "x", encoding="utf-8", newline="") as stream:
                for item in self.file_entries:
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.flushed = True
        else:
            with open(self.session_file, "a", encoding="utf-8", newline="") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _append_entry(self, entry: SessionEntry) -> str:
        self.file_entries.append(entry)
        entry_id = cast(str, entry["id"])
        self.by_id[entry_id] = entry
        self.leaf_id = entry_id
        self._persist(entry)
        return entry_id

    def _base_entry(self, kind: str) -> SessionEntry:
        return {
            "type": kind, "id": _generate_id(self.by_id),
            "parentId": self.leaf_id, "timestamp": _timestamp(),
        }

    def append_message(self, message: CodingAgentMessage) -> str:
        if isinstance(message, (BranchSummaryMessage, CompactionSummaryMessage)):
            raise ValueError("Summary messages must be appended as session entries")
        entry = self._base_entry("message")
        entry["message"] = _encode_message(message)
        return self._append_entry(entry)

    def append_thinking_level_change(self, thinking_level: str) -> str:
        entry = self._base_entry("thinking_level_change")
        entry["thinkingLevel"] = thinking_level
        return self._append_entry(entry)

    def append_model_change(self, provider: str, model_id: str) -> str:
        entry = self._base_entry("model_change")
        entry.update(provider=provider, modelId=model_id)
        return self._append_entry(entry)

    def append_compaction(
        self, summary: str, first_kept_entry_id: str, tokens_before: int,
        details: object = None, from_hook: bool | None = None, usage: Usage | None = None,
    ) -> str:
        entry = self._base_entry("compaction")
        entry.update(summary=summary, firstKeptEntryId=first_kept_entry_id, tokensBefore=tokens_before)
        if details is not None:
            entry["details"] = details
        if usage is not None:
            entry["usage"] = usage.to_json()
        if from_hook is not None:
            entry["fromHook"] = from_hook
        current_system = get_current_system_message(self.build_session_context().messages)
        if current_system is not None:
            system = current_system.to_json()
            system["timestamp"] = _timestamp_ms(cast(str, entry["timestamp"]))
            entry["systemMessage"] = system
        return self._append_entry(entry)

    def append_custom_entry(self, custom_type: str, data: object = None) -> str:
        entry = self._base_entry("custom")
        entry["customType"] = custom_type
        if data is not None:
            entry["data"] = data
        return self._append_entry(entry)

    def append_session_info(self, name: str) -> str:
        entry = self._base_entry("session_info")
        entry["name"] = re.sub(r"[\r\n]+", " ", name).strip()
        return self._append_entry(entry)

    def get_session_name(self) -> str | None:
        for entry in reversed(self.get_entries()):
            if entry.get("type") == "session_info":
                name = entry.get("name")
                return name.strip() or None if isinstance(name, str) else None
        return None

    def append_custom_message_entry(
        self, custom_type: str, content: str | list[TextContent | ImageContent],
        display: bool, details: object = None,
    ) -> str:
        entry = self._base_entry("custom_message")
        entry.update(
            customType=custom_type,
            content=content if isinstance(content, str) else [content_block_to_json(block) for block in content],
            display=display,
        )
        if details is not None:
            entry["details"] = details
        return self._append_entry(entry)

    def append_label_change(self, target_id: str, label: str | None) -> str:
        if target_id not in self.by_id:
            raise ValueError(f"Entry {target_id} not found")
        entry = self._base_entry("label")
        entry.update(targetId=target_id, label=label)
        entry_id = self._append_entry(entry)
        if label:
            self.labels_by_id[target_id] = label
            self.label_timestamps_by_id[target_id] = cast(str, entry["timestamp"])
        else:
            self.labels_by_id.pop(target_id, None)
            self.label_timestamps_by_id.pop(target_id, None)
        return entry_id

    def get_leaf_id(self) -> str | None:
        return self.leaf_id

    def get_leaf_entry(self) -> SessionEntry | None:
        return self.by_id.get(self.leaf_id) if self.leaf_id else None

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self.by_id.get(entry_id)

    def get_children(self, parent_id: str) -> list[SessionEntry]:
        return [entry for entry in self.by_id.values() if entry.get("parentId") == parent_id]

    def get_label(self, entry_id: str) -> str | None:
        return self.labels_by_id.get(entry_id)

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        current = self.by_id.get(from_id or self.leaf_id or "")
        path: list[SessionEntry] = []
        while current is not None:
            path.append(current)
            parent = current.get("parentId")
            current = self.by_id.get(parent) if isinstance(parent, str) else None
        path.reverse()
        return path

    def build_context_entries(self) -> list[SessionEntry]:
        return build_context_entries(self.get_entries(), self.leaf_id, self.by_id)

    def build_session_context(self) -> SessionContext:
        return build_session_context(self.get_entries(), self.leaf_id, self.by_id)

    def get_header(self) -> FileEntry | None:
        return next((entry for entry in self.file_entries if entry.get("type") == "session"), None)

    def get_entries(self) -> list[SessionEntry]:
        return [entry for entry in self.file_entries if entry.get("type") != "session"]

    def get_tree(self) -> list[SessionTreeNode]:
        entries = self.get_entries()
        nodes = {
            cast(str, entry["id"]): SessionTreeNode(
                entry=entry,
                label=self.labels_by_id.get(cast(str, entry["id"])),
                label_timestamp=self.label_timestamps_by_id.get(cast(str, entry["id"])),
            )
            for entry in entries
        }
        roots: list[SessionTreeNode] = []
        for entry in entries:
            entry_id = cast(str, entry["id"])
            parent_id = entry.get("parentId")
            parent = nodes.get(parent_id) if isinstance(parent_id, str) else None
            if parent is None or parent_id == entry_id:
                roots.append(nodes[entry_id])
            else:
                parent.children.append(nodes[entry_id])
        stack = list(roots)
        while stack:
            node = stack.pop()
            node.children.sort(key=lambda child: cast(str, child.entry.get("timestamp", "")))
            stack.extend(node.children)
        return roots

    def branch(self, branch_from_id: str) -> None:
        if branch_from_id not in self.by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self.leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        self.leaf_id = None

    def branch_with_summary(
        self, branch_from_id: str | None, summary: str,
        details: object = None, from_hook: bool | None = None,
        usage: Usage | None = None,
    ) -> str:
        if branch_from_id is not None and branch_from_id not in self.by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        from_id = self.leaf_id or "root"
        self.leaf_id = branch_from_id
        entry = self._base_entry("branch_summary")
        entry.update(fromId=from_id, summary=summary)
        if details is not None:
            entry["details"] = details
        if from_hook is not None:
            entry["fromHook"] = from_hook
        if usage is not None:
            entry["usage"] = usage.to_json()
        return self._append_entry(entry)

    def create_branched_session(self, leaf_id: str) -> str | None:
        previous_file = self.session_file
        path = self.get_branch(leaf_id)
        if not path:
            raise ValueError(f"Entry {leaf_id} not found")
        path_without_labels: list[SessionEntry] = []
        replacement_by_label_id: dict[str, str] = {}
        pending_label_ids: list[str] = []
        parent_id: str | None = None
        for entry in path:
            if entry.get("type") == "label":
                pending_label_ids.append(cast(str, entry["id"]))
                continue
            entry_id = cast(str, entry["id"])
            for label_id in pending_label_ids:
                replacement_by_label_id[label_id] = entry_id
            pending_label_ids.clear()
            copied = {**entry, "parentId": parent_id}
            if copied.get("type") == "compaction":
                first_kept = copied.get("firstKeptEntryId")
                if isinstance(first_kept, str):
                    copied["firstKeptEntryId"] = replacement_by_label_id.get(first_kept, first_kept)
            path_without_labels.append(copied)
            parent_id = entry_id

        new_id = uuidv7()
        timestamp = _timestamp()
        new_file = str(Path(self.session_dir) / f"{timestamp.replace(':', '-').replace('.', '-')}_{new_id}.jsonl")
        header: FileEntry = {
            "type": "session", "version": CURRENT_SESSION_VERSION,
            "id": new_id, "timestamp": timestamp, "cwd": self.cwd,
        }
        if self.persist and previous_file is not None:
            header["parentSession"] = previous_file

        path_ids = {cast(str, entry["id"]) for entry in path_without_labels}
        labels = [
            (target_id, label, self.label_timestamps_by_id[target_id])
            for target_id, label in self.labels_by_id.items() if target_id in path_ids
        ]
        label_entries: list[SessionEntry] = []
        for target_id, label, label_timestamp in labels:
            entry_id = _generate_id({name: {} for name in path_ids})
            path_ids.add(entry_id)
            label_entry: SessionEntry = {
                "type": "label", "id": entry_id, "parentId": parent_id,
                "timestamp": label_timestamp, "targetId": target_id, "label": label,
            }
            label_entries.append(label_entry)
            parent_id = entry_id

        self.file_entries = [header, *path_without_labels, *label_entries]
        self.session_id = new_id
        self._build_index()
        if not self.persist:
            return None
        self.session_file = new_file
        has_assistant = any(
            entry.get("type") == "message" and isinstance(entry.get("message"), dict)
            and cast(dict, entry["message"]).get("role") == "assistant"
            for entry in self.file_entries
        )
        if has_assistant:
            self._rewrite_file()
            self.flushed = True
        else:
            self.flushed = False
        return new_file

    def get_cwd(self) -> str:
        return self.cwd

    def get_session_dir(self) -> str:
        return self.session_dir

    def uses_default_session_dir(self) -> bool:
        return self.session_dir == _default_session_dir_path(self.cwd)

    def get_session_id(self) -> str:
        return self.session_id

    def get_session_file(self) -> str | None:
        return self.session_file

    def is_persisted(self) -> bool:
        return self.persist

    @classmethod
    def create(
        cls, cwd: str, session_dir: str | None = None,
        *, session_id: str | None = None, parent_session: str | None = None,
    ) -> SessionManager:
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        return cls(cwd, directory, None, True, session_id=session_id, parent_session=parent_session)

    @classmethod
    def open(
        cls, path: str, session_dir: str | None = None,
        cwd_override: str | None = None,
    ) -> SessionManager:
        resolved = resolve_path(path)
        header: FileEntry | None = None
        preloaded: list[FileEntry] | None = None
        if cwd_override is None and Path(resolved).exists():
            try:
                header = _read_session_header(resolved)
            except RuntimeError:
                preloaded = load_entries_from_file(resolved)
                header = preloaded[0] if preloaded and preloaded[0].get("type") == "session" else None
        cwd = cwd_override or (cast(str, header.get("cwd")) if header and isinstance(header.get("cwd"), str) else os.getcwd())
        directory = normalize_path(session_dir) if session_dir else str(Path(resolved).parent)
        return cls(cwd, directory, resolved, True, preloaded_entries=preloaded)

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: str | None = None) -> SessionManager:
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        filter_cwd = session_dir is not None and directory != _default_session_dir_path(cwd)
        most_recent = find_most_recent_session(directory, cwd if filter_cwd else None)
        return cls(cwd, directory, most_recent, True)

    @classmethod
    def in_memory(
        cls, cwd: str | None = None, *, session_id: str | None = None,
        parent_session: str | None = None, entries: list[FileEntry] | None = None,
    ) -> SessionManager:
        return cls(
            cwd or os.getcwd(), "", None, False, session_id=session_id,
            parent_session=parent_session, preloaded_entries=entries,
        )

    @classmethod
    def fork_from(
        cls, source_path: str, target_cwd: str, session_dir: str | None = None,
        *, session_id: str | None = None,
    ) -> SessionManager:
        source = resolve_path(source_path)
        target = resolve_path(target_cwd)
        entries = load_entries_from_file(source)
        if not entries:
            raise ValueError(f"Cannot fork: source session file is empty or invalid: {source}")
        if not any(entry.get("type") == "session" for entry in entries):
            raise ValueError(f"Cannot fork: source session has no header: {source}")
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(target)
        Path(directory).mkdir(parents=True, exist_ok=True)
        if session_id is not None:
            assert_valid_session_id(session_id)
        new_id = session_id or uuidv7()
        timestamp = _timestamp()
        new_file = str(Path(directory) / f"{timestamp.replace(':', '-').replace('.', '-')}_{new_id}.jsonl")
        header: FileEntry = {
            "type": "session", "version": CURRENT_SESSION_VERSION, "id": new_id,
            "timestamp": timestamp, "cwd": target, "parentSession": source,
        }
        with open(new_file, "x", encoding="utf-8", newline="") as stream:
            for entry in (header, *(item for item in entries if item.get("type") != "session")):
                stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        return cls(target, directory, new_file, True)

    @staticmethod
    def find_by_id(cwd: str, session_id: str, session_dir: str | None = None) -> str | None:
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        filter_cwd = session_dir is not None and directory != _default_session_dir_path(cwd)
        resolved_cwd = resolve_path(cwd)
        try:
            for path in Path(directory).glob("*.jsonl"):
                header = _header_for_discovery(str(path))
                if header is None or header.get("id") != session_id:
                    continue
                header_cwd = header.get("cwd")
                if filter_cwd and (
                    not isinstance(header_cwd, str) or not header_cwd
                    or resolve_path(header_cwd) != resolved_cwd
                ):
                    continue
                return str(path)
        except OSError:
            pass
        return None


__all__ = [
    "SessionManager", "SessionTreeNode", "assert_valid_session_id",
    "get_default_session_dir", "load_entries_from_file", "find_most_recent_session",
]
