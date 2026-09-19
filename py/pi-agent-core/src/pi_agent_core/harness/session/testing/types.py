"""Shared testing types ported from ``session/testing/types.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ...context import BACKGROUND_CONTEXT
from ..types import Storage

__all__ = ["StorageFixture", "ConformanceCase"]


@dataclass
class StorageFixture:
    """A fresh backend storage instance owned by one test or benchmark case.

    ``dispose`` is the Python counterpart of the TypeScript fixture's
    ``[Symbol.asyncDispose]``. When it is omitted the fixture closes
    ``storage`` itself, which matches every in-repo fixture factory.
    """

    storage: Storage
    dispose: Optional[Callable[[], Awaitable[None]]] = None

    async def aclose(self) -> None:
        """Dispose the fixture exactly once per call, mirroring ``await using``."""
        if self.dispose is not None:
            await self.dispose()
            return
        await self.storage.close(BACKGROUND_CONTEXT)

    async def __aenter__(self) -> "StorageFixture":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


@dataclass
class ConformanceCase:
    """A runner-independent conformance case that can be registered with any test framework."""

    group: str
    name: str
    run: Callable[[], Awaitable[None]]
