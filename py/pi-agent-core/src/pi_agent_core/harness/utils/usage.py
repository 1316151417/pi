"""Usage helpers ported from ``harness/utils/usage.ts``."""

from __future__ import annotations

from ..._pi_ai.types import Usage

__all__ = ["empty_usage", "add_usage"]


def empty_usage() -> Usage:
    return Usage()


def add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_write=left.cache_write + right.cache_write,
        cache_write_1h=(
            None
            if left.cache_write_1h is None and right.cache_write_1h is None
            else (left.cache_write_1h or 0) + (right.cache_write_1h or 0)
        ),
        reasoning=(
            None
            if left.reasoning is None and right.reasoning is None
            else (left.reasoning or 0) + (right.reasoning or 0)
        ),
        total_tokens=left.total_tokens + right.total_tokens,
        cost=type(left.cost)(
            input=left.cost.input + right.cost.input,
            output=left.cost.output + right.cost.output,
            cache_read=left.cost.cache_read + right.cost.cache_read,
            cache_write=left.cost.cache_write + right.cost.cache_write,
            total=left.cost.total + right.cost.total,
        ),
    )
