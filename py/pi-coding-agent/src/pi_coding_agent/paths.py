"""Path normalization used by coding-agent tools from ``utils/paths.ts``."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, unquote

_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_REMOTE_PREFIXES = ("npm:", "git:", "github:", "http:", "https:", "ssh:")


@dataclass(frozen=True)
class PathInputOptions:
    trim: bool = False
    expand_tilde: bool = True
    home_dir: str | None = None
    strip_at_prefix: bool = False
    normalize_unicode_spaces: bool = False


def canonicalize_path(path: str) -> str:
    try:
        return str(Path(path).resolve(strict=True))
    except OSError:
        return path


def get_file_revision(path: str) -> str | None:
    try:
        status = os.stat(path)
    except OSError:
        return None
    return f"{status.st_dev}:{status.st_ino}:{status.st_size}:{status.st_mtime_ns}:{status.st_ctime_ns}"


def is_local_path(value: str) -> bool:
    return not value.strip().startswith(_REMOTE_PREFIXES)


def normalize_windows_shell_path(file_path: str) -> str:
    if not file_path.startswith("/") or file_path.startswith("//") or "\\" in file_path:
        return file_path
    match = re.fullmatch(r"/(?:mnt/|cygdrive/)?([a-z])(?:/(.*))?", file_path, re.I)
    if match is None:
        return file_path
    suffix = (match.group(2) or "").replace("/", "\\")
    return f"{match.group(1).upper()}:\\{suffix}"


def normalize_path(value: str, options: PathInputOptions | None = None) -> str:
    options = options or PathInputOptions()
    normalized = value.strip() if options.trim else value
    if options.normalize_unicode_spaces:
        normalized = _UNICODE_SPACES.sub(" ", normalized)
    if options.strip_at_prefix and normalized.startswith("@"):
        normalized = normalized[1:]
    if sys.platform == "win32":
        normalized = normalize_windows_shell_path(normalized)
    if options.expand_tilde:
        home = options.home_dir if options.home_dir is not None else str(Path.home())
        if normalized == "~":
            return home
        if normalized.startswith("~/") or sys.platform == "win32" and normalized.startswith("~\\"):
            return os.path.join(home, normalized[2:])
    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        if parsed.scheme == "file":
            if sys.platform != "win32" and parsed.netloc not in ("", "localhost"):
                raise ValueError("File URL host must be empty or localhost")
            return unquote(parsed.path)
    return normalized


def resolve_path(value: str, base_dir: str | None = None, options: PathInputOptions | None = None) -> str:
    normalized = normalize_path(value, options)
    base = normalize_path(base_dir if base_dir is not None else os.getcwd())
    return os.path.abspath(normalized if os.path.isabs(normalized) else os.path.join(base, normalized))


def get_cwd_relative_path(file_path: str, cwd: str) -> str | None:
    resolved_cwd = resolve_path(cwd)
    resolved_path = resolve_path(file_path, resolved_cwd)
    relative = os.path.relpath(resolved_path, resolved_cwd)
    if relative == "." or relative != ".." and not relative.startswith(".." + os.sep) and not os.path.isabs(relative):
        return relative
    return None


def format_path_relative_to_cwd_or_absolute(file_path: str, cwd: str) -> str:
    absolute = resolve_path(file_path, cwd)
    return (get_cwd_relative_path(absolute, cwd) or absolute).replace(os.sep, "/")


__all__ = [
    "PathInputOptions", "canonicalize_path", "get_file_revision", "is_local_path",
    "normalize_windows_shell_path", "normalize_path", "resolve_path",
    "get_cwd_relative_path", "format_path_relative_to_cwd_or_absolute",
]
