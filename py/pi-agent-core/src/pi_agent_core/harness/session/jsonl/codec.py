"""JSONL header codec ported from ``session/jsonl/codec.ts``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union

from ...result import Result, err, ok
from .types import JSONL_FORMAT_VERSION, JsonlStorageHeader

__all__ = [
    "LegacyV3SessionHeader",
    "JsonlParsedSessionHeader",
    "is_legacy_v3_session_header",
    "is_jsonl_storage_header",
    "parse_jsonl_session_header",
]


@dataclass
class LegacyV3SessionHeader:
    """Header of the pre-v4 session file format."""

    type: str = "session"
    version: int = 3
    id: str = ""
    timestamp: str = ""
    cwd: str = ""
    parent_session: Optional[str] = None


@dataclass
class JsonlParsedSessionHeader:
    format: str  # "v4" | "v3-legacy"
    header: Any = None


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_safe_integer_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_finite_date(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def is_legacy_v3_session_header(value: Any) -> bool:
    if not _is_record(value):
        return False
    return (
        value.get("type") == "session"
        and value.get("version") == 3
        and isinstance(value.get("id"), str)
        and isinstance(value.get("cwd"), str)
        and isinstance(value.get("timestamp"), str)
        and _is_finite_date(value["timestamp"])
        and (value.get("parentSession") is None or isinstance(value.get("parentSession"), str))
    )


def is_jsonl_storage_header(value: Any) -> bool:
    if not _is_record(value):
        return False
    return (
        value.get("kind") == "header"
        and value.get("v") == JSONL_FORMAT_VERSION
        and isinstance(value.get("id"), str)
        and isinstance(value.get("cwd"), str)
        and _is_safe_integer_at_least(value.get("storageVersion"), 1)
        and _is_safe_integer_at_least(value.get("createdAt"), 0)
        and (value.get("nextSeq") is None or _is_safe_integer_at_least(value.get("nextSeq"), 1))
        and (value.get("parentSessionId") is None or isinstance(value.get("parentSessionId"), str))
        and (
            value.get("legacyParentSessionPath") is None
            or isinstance(value.get("legacyParentSessionPath"), str)
        )
    )


def parse_jsonl_session_header(line: str) -> Result:
    """Parse the first line of a JSONL session file into a v4 or v3-legacy header."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        wrapped = RuntimeError("Invalid JSONL session header: not valid JSON")
        wrapped.__cause__ = error
        return err(wrapped)
    if is_jsonl_storage_header(value):
        return ok(JsonlParsedSessionHeader(format="v4", header=JsonlStorageHeader.from_json(value)))
    if is_legacy_v3_session_header(value):
        header = LegacyV3SessionHeader(
            id=value["id"],
            timestamp=value["timestamp"],
            cwd=value["cwd"],
            parent_session=value.get("parentSession"),
        )
        return ok(JsonlParsedSessionHeader(format="v3-legacy", header=header))
    return err(RuntimeError("Unsupported JSONL session header"))
