"""Ambient environment and file access for provider auth resolution."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from .types import AuthContext

__all__ = ["default_provider_auth_context"]

_JS_WHITESPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


class _DefaultAuthContext:
    async def env(self, name: str) -> str | None:
        value = os.environ.get(name)
        return value if isinstance(value, str) and value.strip(_JS_WHITESPACE) else None

    async def file_exists(self, path: str) -> bool:
        if sys.platform in ("emscripten", "wasi"):
            return False
        try:
            resolved = str(Path.home()) + path[1:] if path.startswith("~") else path
            return await asyncio.to_thread(os.access, resolved, os.F_OK)
        except Exception:
            return False


def default_provider_auth_context() -> AuthContext:
    return _DefaultAuthContext()
