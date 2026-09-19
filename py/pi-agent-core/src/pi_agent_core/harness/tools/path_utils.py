"""Path helpers ported from ``harness/tools/path-utils.ts``."""

from __future__ import annotations

import re
import unicodedata
from typing import List

from ..._chord.context import Context
from ..types import ExecutionEnv, get_or_throw

__all__ = ["resolve_tool_path", "resolve_read_tool_path"]

UNICODE_SPACES = re.compile("[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
NARROW_NO_BREAK_SPACE = "\u202F"


def normalize_tool_path(path: str) -> str:
    normalized = UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(env: ExecutionEnv, path: str, context: Context) -> str:
    return get_or_throw(await env.absolute_path(normalize_tool_path(path), context))


async def resolve_read_tool_path(env: ExecutionEnv, path: str, context: Context) -> str:
    resolved = await resolve_tool_path(env, path, context)
    variants: List[str] = []
    for variant in (
        resolved,
        re.sub(r" (AM|PM)\.", lambda m: f"{NARROW_NO_BREAK_SPACE}{m.group(1)}.", resolved, flags=re.IGNORECASE),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "\u2019"),
        unicodedata.normalize("NFD", resolved).replace("'", "\u2019"),
    ):
        if variant not in variants:
            variants.append(variant)

    for variant in variants:
        if get_or_throw(await env.exists(variant, context)):
            return variant
    return resolved
