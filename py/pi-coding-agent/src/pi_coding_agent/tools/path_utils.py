"""Read-path fallbacks from ``core/tools/path-utils.ts``."""

from __future__ import annotations

import asyncio
import os
import re
import unicodedata
from collections.abc import Iterator

from ..paths import PathInputOptions, normalize_path, resolve_path

_OPTIONS = PathInputOptions(normalize_unicode_spaces=True, strip_at_prefix=True)
_SCREENSHOT_AM_PM = re.compile(r" (AM|PM)\.", re.I)


def _variants(resolved: str) -> Iterator[str]:
    am_pm = _SCREENSHOT_AM_PM.sub(lambda match: "\u202f" + match.group(1) + ".", resolved)
    nfd = unicodedata.normalize("NFD", resolved)
    curly = resolved.replace("'", "\u2019")
    nfd_curly = nfd.replace("'", "\u2019")
    yield from (am_pm, nfd, curly, nfd_curly)


def expand_path(file_path: str) -> str:
    return normalize_path(file_path, _OPTIONS)


def resolve_to_cwd(file_path: str, cwd: str) -> str:
    return resolve_path(file_path, cwd, _OPTIONS)


def resolve_read_path(file_path: str, cwd: str) -> str:
    resolved = resolve_to_cwd(file_path, cwd)
    if os.access(resolved, os.F_OK):
        return resolved
    for candidate in _variants(resolved):
        if candidate != resolved and os.access(candidate, os.F_OK):
            return candidate
    return resolved


async def path_exists(file_path: str) -> bool:
    return await asyncio.to_thread(os.access, file_path, os.F_OK)


async def resolve_read_path_async(file_path: str, cwd: str) -> str:
    resolved = resolve_to_cwd(file_path, cwd)
    if await path_exists(resolved):
        return resolved
    for candidate in _variants(resolved):
        if candidate != resolved and await path_exists(candidate):
            return candidate
    return resolved


__all__ = [
    "expand_path", "resolve_to_cwd", "resolve_read_path", "path_exists",
    "resolve_read_path_async",
]
