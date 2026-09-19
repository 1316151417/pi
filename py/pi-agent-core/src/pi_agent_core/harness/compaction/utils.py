"""Compaction helpers ported from ``harness/compaction/utils.ts``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Set

from pi_ai.text import content_text

__all__ = [
    "FileOperations",
    "create_file_ops",
    "extract_file_ops_from_message",
    "compute_file_lists",
    "format_file_operations",
    "serialize_conversation",
]

TOOL_RESULT_MAX_CHARS = 2000


@dataclass
class FileOperations:
    """File paths touched by a session branch or compaction range."""

    read: Set[str] = field(default_factory=set)
    written: Set[str] = field(default_factory=set)
    edited: Set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    return FileOperations()


def _safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return "[unserializable]"


def extract_file_ops_from_message(message: Any, file_ops: FileOperations) -> None:
    """Accumulate read/write/edit paths from assistant tool calls."""
    if getattr(message, "role", None) != "assistant":
        return
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return
    for block in content:
        if getattr(block, "type", None) != "toolCall":
            continue
        args = getattr(block, "arguments", None)
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str):
            continue
        name = getattr(block, "name", None)
        if name == "read":
            file_ops.read.add(path)
        elif name == "write":
            file_ops.written.add(path)
        elif name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> tuple:
    """Return ``(read_files, modified_files)``, both sorted; read excludes modified."""
    modified = {*file_ops.edited, *file_ops.written}
    read_only = sorted(path for path in file_ops.read if path not in modified)
    return read_only, sorted(modified)


def format_file_operations(read_files: List[str], modified_files: List[str]) -> str:
    sections: List[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: List[Any]) -> str:
    """Serialize LLM messages to plain text for summarization prompts."""
    parts: List[str] = []

    for message in messages:
        role = getattr(message, "role", None)
        if role == "user":
            content = content_text(message.content, "")
            if content:
                parts.append(f"[User]: {content}")
        elif role == "assistant":
            thinking_parts: List[str] = []
            tool_calls: List[str] = []
            for block in message.content:
                block_type = getattr(block, "type", None)
                if block_type == "thinking":
                    thinking_parts.append(block.thinking)
                elif block_type == "toolCall":
                    args = block.arguments or {}
                    args_str = ", ".join(f"{key}={_safe_json_stringify(value)}" for key, value in args.items())
                    tool_calls.append(f"{block.name}({args_str})")
            if thinking_parts:
                parts.append("[Assistant thinking]: " + "\n".join(thinking_parts))
            if any(getattr(block, "type", None) == "text" for block in message.content):
                parts.append(f"[Assistant]: {content_text(message.content)}")
            if tool_calls:
                parts.append("[Assistant tool calls]: " + "; ".join(tool_calls))
        elif role == "toolResult":
            content = content_text(message.content, "")
            if content:
                parts.append(f"[Tool result]: {_truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)}")

    return "\n\n".join(parts)
