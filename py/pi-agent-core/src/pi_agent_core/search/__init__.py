"""Search service interfaces ported from ``search/index.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, List, Optional, Protocol, runtime_checkable

__all__ = ["SearchQuery", "SessionSearchHit", "EntrySearchHit", "SessionSearchService"]


@dataclass
class SearchQuery:
    text: str
    limit: Optional[int] = None


@dataclass
class SessionSearchHit:
    session_id: str
    score: Optional[float] = None
    top: Optional[dict] = None  # {"entryId": str, "snippet": str, "timestamp": int}


@dataclass
class EntrySearchHit:
    session_id: str
    entry_id: str
    timestamp: int
    snippet: Optional[str] = None
    score: Optional[float] = None


@runtime_checkable
class SessionSearchService(Protocol):
    def search_sessions(self, query: SearchQuery) -> Awaitable[List[SessionSearchHit]]: ...
    def search_entries(self, query: SearchQuery) -> Awaitable[List[EntrySearchHit]]: ...
    def sync(self) -> Awaitable[None]: ...
    def notify(self, session_id: str) -> None: ...
    def remove(self, session_id: str) -> Awaitable[None]: ...
    def close(self) -> Awaitable[None]: ...
