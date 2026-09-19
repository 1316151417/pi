"""JSONL storage header types ported from ``session/jsonl/types.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ...types import FileSystem
from ..types import SessionCreateOptions, SessionMetadata

__all__ = [
    "JSONL_FORMAT_VERSION",
    "JSONL_STORAGE_VERSION",
    "JsonlStorageHeader",
    "JsonlStorageOptions",
    "JsonlSessionMetadata",
    "JsonlSessionCreateOptions",
    "JsonlSessionListOptions",
    "JsonlSessionRepoOptions",
]

JSONL_FORMAT_VERSION = 4
JSONL_STORAGE_VERSION = 1


@dataclass
class JsonlStorageHeader:
    v: int = JSONL_FORMAT_VERSION
    kind: str = "header"
    id: str = ""
    storage_version: int = JSONL_STORAGE_VERSION
    created_at: int = 0
    cwd: str = ""
    parent_session_id: Optional[str] = None
    legacy_parent_session_path: Optional[str] = None
    #: Sequence high-water mark written by snapshot rewrites.
    next_seq: Optional[int] = None

    def to_json(self) -> dict:
        data: dict = {
            "v": self.v,
            "kind": self.kind,
            "id": self.id,
            "storageVersion": self.storage_version,
            "createdAt": self.created_at,
            "cwd": self.cwd,
        }
        if self.parent_session_id is not None:
            data["parentSessionId"] = self.parent_session_id
        if self.legacy_parent_session_path is not None:
            data["legacyParentSessionPath"] = self.legacy_parent_session_path
        if self.next_seq is not None:
            data["nextSeq"] = self.next_seq
        return data

    @classmethod
    def from_json(cls, data: dict) -> "JsonlStorageHeader":
        return cls(
            v=data.get("v", JSONL_FORMAT_VERSION),
            kind=data.get("kind", "header"),
            id=data["id"],
            storage_version=data["storageVersion"],
            created_at=data["createdAt"],
            cwd=data["cwd"],
            parent_session_id=data.get("parentSessionId"),
            legacy_parent_session_path=data.get("legacyParentSessionPath"),
            next_seq=data.get("nextSeq"),
        )


@dataclass
class JsonlStorageOptions:
    file_system: FileSystem = None  # type: ignore[assignment]
    path: str = ""
    now: Optional[Callable[[], int]] = None


@dataclass
class JsonlSessionMetadata(SessionMetadata):
    """Session metadata for a JSONL-backed session."""

    cwd: str = ""
    path: str = ""
    #: Filesystem modification time in milliseconds since the Unix epoch.
    modified_at: int = 0


@dataclass
class JsonlSessionCreateOptions(SessionCreateOptions):
    cwd: str = ""


@dataclass
class JsonlSessionListOptions:
    cwd: Optional[str] = None


@dataclass
class JsonlSessionRepoOptions:
    file_system: FileSystem = None  # type: ignore[assignment]
    sessions_root: str = ""
    now: Optional[Callable[[], int]] = None
