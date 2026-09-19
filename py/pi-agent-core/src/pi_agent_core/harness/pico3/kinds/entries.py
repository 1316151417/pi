"""Core entry kinds ported from ``harness/pico3/kinds/entries.ts``.

The TypeScript module declares type aliases for the shape of each built-in entry
kind; Python keeps the durable entry shape open (``Entry`` carries ``data`` as a
JSON object), so this module is the registry of witnesses plus the documented
``data`` payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pi_ai.types import JsonValue
from ..types import EntryKind, Id, ToolControl

__all__ = [
    "entries",
    "EntryKinds",
    "UserEntryData",
    "AssistantEntryData",
    "ToolResultEntryData",
    "SystemEntryData",
    "UsageEntryData",
    "SummaryEntryData",
    "core_entry",
]

#: Reserved kind names for the built-in entries.
USER_KIND = "pi.user"
ASSISTANT_KIND = "pi.assistant"
TOOL_RESULT_KIND = "pi.tool_result"
SYSTEM_KIND = "pi.system"
NOTICE_KIND = "pi.notice"
USAGE_KIND = "pi.usage"
SUMMARY_KIND = "pi.summary"
HANDOFF_KIND = "pi.handoff"
RESET_KIND = "pi.reset"


@dataclass
class UserEntryData:
    """``pi.user`` data: a continuation marker naming the entry it continues from."""

    continuation: bool = True
    from_id: Optional[Id] = None


@dataclass
class AssistantEntryData:
    """``pi.assistant`` data."""

    attempt: int = 0


@dataclass
class ToolResultEntryData:
    """``pi.tool_result`` data."""

    details: Optional[JsonValue] = None
    diagnostics: Optional[JsonValue] = None
    control: Optional[ToolControl] = None
    truncated: Optional[Dict[str, int]] = None


@dataclass
class SystemEntryData:
    """``pi.system`` data: the managed entry's section records and baseline flag."""

    sections: Optional[list] = None
    baseline: bool = False


@dataclass
class UsageEntryData:
    """``pi.usage`` data: display-only usage bookkeeping for one attempt."""

    attempt: int = 0
    usage: Optional[JsonValue] = None
    error: str = ""


@dataclass
class SummaryEntryData:
    """``pi.summary`` data: the entry the summary folds through."""

    through: Id = 0


def core_entry(kind: str) -> EntryKind:
    """A frozen witness for a core entry kind, mirroring the TS ``coreEntry`` helper."""
    return EntryKind(kind=kind)


class EntryKinds:
    """The built-in entry witnesses (``entries`` in TypeScript)."""

    def __init__(self) -> None:
        self.user = core_entry(USER_KIND)
        self.assistant = core_entry(ASSISTANT_KIND)
        self.tool_result = core_entry(TOOL_RESULT_KIND)
        self.system = core_entry(SYSTEM_KIND)
        self.notice = core_entry(NOTICE_KIND)
        self.usage = core_entry(USAGE_KIND)
        self.summary = core_entry(SUMMARY_KIND)
        self.handoff = core_entry(HANDOFF_KIND)
        self.reset = core_entry(RESET_KIND)

    def __iter__(self):
        return iter(
            (
                self.user,
                self.assistant,
                self.tool_result,
                self.system,
                self.notice,
                self.usage,
                self.summary,
                self.handoff,
                self.reset,
            )
        )


entries = EntryKinds()


_ = Any
