"""Packaging candidates equivalent to ``native-module-path.ts``."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

TUI_PACKAGE_NAME = "@earendil-works/pi-tui"


@dataclass
class NativeModuleCandidateOptions:
    module_url: str | None = None
    exec_path: str | None = None
    resolve_package: Callable[[str], str] | None = None


def _resolve_package(specifier: str) -> str:
    name = "pi_tui" if specifier == TUI_PACKAGE_NAME else specifier
    specification = importlib.util.find_spec(name)
    if specification is None or specification.origin is None:
        raise ModuleNotFoundError(specifier)
    return specification.origin


def _join(*parts: str) -> str:
    # Unlike os.path.join, Node's join does not discard earlier components
    # when a later component starts with a separator.
    joined = os.path.normpath(os.sep.join(parts))
    return "/" + joined.lstrip("/") if os.name != "nt" and joined.startswith("/") else joined


def get_native_module_candidates(
    native_path: str,
    options: NativeModuleCandidateOptions | None = None,
) -> list[str]:
    options = options if options is not None else NativeModuleCandidateOptions()
    module_url = options.module_url if options.module_url is not None else Path(__file__).as_uri()
    parsed = urlsplit(module_url)
    if parsed.scheme != "file":
        raise ValueError("The URL must be of scheme file")
    if re.search(r"%2f", parsed.path, re.IGNORECASE) or (os.name == "nt" and re.search(r"%5c", parsed.path, re.IGNORECASE)):
        raise ValueError("File URL path must not include encoded path separators")
    module_path = unquote(parsed.path, errors="strict")
    if os.name == "nt":
        if parsed.hostname and parsed.hostname != "localhost":
            module_path = "\\\\" + parsed.hostname + module_path.replace("/", "\\")
        else:
            if len(module_path) < 3 or module_path[0] != "/" or module_path[2] != ":":
                raise ValueError("File URL path must be absolute")
            module_path = module_path[1:].replace("/", "\\")
    elif parsed.hostname and parsed.hostname != "localhost":
        raise ValueError("File URL host must be localhost or empty")
    module_dir = os.path.dirname(module_path)
    candidates: list[str] = []
    try:
        resolver = options.resolve_package if options.resolve_package is not None else _resolve_package
        package_entry = resolver(TUI_PACKAGE_NAME)
        candidates.append(_join(os.path.dirname(package_entry) or ".", "..", native_path))
    except Exception:
        # Standalone applications need not have an installed package.
        pass
    candidates.extend((
        _join(module_dir, "..", native_path),
        _join(module_dir, native_path),
        _join(os.path.dirname(options.exec_path if options.exec_path is not None else sys.executable) or ".", native_path),
    ))
    return list(dict.fromkeys(candidates))


__all__ = ["NativeModuleCandidateOptions", "get_native_module_candidates"]
