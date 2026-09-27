"""Persistent catalogue contracts and memory implementation from models-store.ts."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from .abort import AbortSignal
from .types import Model


@dataclass
class ModelsStoreEntry:
    models: Sequence[Model]
    last_modified: int | float | None = None
    checked_at: int | float | None = None
    etag: str | None = None


@dataclass
class ModelsStoreOperationOptions:
    signal: AbortSignal | None = None


class ModelsStore(Protocol):
    async def read(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> ModelsStoreEntry | None: ...

    async def write(self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None) -> None: ...

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None: ...


class InMemoryModelsStore:
    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> ModelsStoreEntry | None:
        if options is not None and options.signal is not None:
            options.signal.throw_if_aborted()
        entry = self._entries.get(provider_id)
        return deepcopy(entry) if entry is not None else None

    async def write(self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None) -> None:
        if options is not None and options.signal is not None:
            options.signal.throw_if_aborted()
        self._entries[provider_id] = deepcopy(entry)

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        if options is not None and options.signal is not None:
            options.signal.throw_if_aborted()
        self._entries.pop(provider_id, None)


__all__ = ["ModelsStoreEntry", "ModelsStoreOperationOptions", "ModelsStore", "InMemoryModelsStore"]
