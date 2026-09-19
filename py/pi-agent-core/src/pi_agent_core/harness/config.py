"""Harness configuration defaults and validation ported from ``harness/config.ts``."""

from __future__ import annotations

from typing import Any, List, Sequence

from .compaction.compaction import CompactionSettings
from .utils.retry import DEFAULT_MAX_AGENT_RETRY_DELAY_MS, RetryPolicy

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "validate_tool_names",
    "validate_retry_policy",
    "validate_compaction_settings",
]

#: Default retry policy used when a harness option omits one.
DEFAULT_RETRY_POLICY = RetryPolicy(
    enabled=True,
    max_retries=3,
    base_delay_ms=1000,
    max_agent_delay_ms=DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
)

#: Python ints are unbounded; the TS bound is preserved so validation matches.
MAX_SAFE_INTEGER = 2**53 - 1


def _is_safe_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= MAX_SAFE_INTEGER


def validate_tool_names(tools: Sequence[Any]) -> None:
    """Reject duplicate tool names before they reach the model."""
    names: set = set()
    for tool in tools:
        name = tool.name
        if name in names:
            raise TypeError(f"Duplicate tool name: {name!r}")
        names.add(name)


def validate_retry_policy(policy: RetryPolicy) -> None:
    """Retry policy numbers must be finite non-negative safe integers."""
    if (
        not _is_safe_integer(policy.max_retries)
        or policy.max_retries < 0
        or policy.max_retries == MAX_SAFE_INTEGER
        or not _is_safe_integer(policy.base_delay_ms)
        or policy.base_delay_ms < 0
        or (
            policy.max_agent_delay_ms is not None
            and (not _is_safe_integer(policy.max_agent_delay_ms) or policy.max_agent_delay_ms < 0)
        )
    ):
        raise ValueError("Retry policy values must be finite non-negative safe integers")


def validate_compaction_settings(settings: CompactionSettings) -> None:
    """Compaction token counts must be finite non-negative safe integers."""
    if (
        not _is_safe_integer(settings.reserve_tokens)
        or settings.reserve_tokens < 0
        or not _is_safe_integer(settings.keep_recent_tokens)
        or settings.keep_recent_tokens < 0
    ):
        raise ValueError("Compaction token counts must be finite non-negative safe integers")
