"""Keyed instance ownership and cancellable observers from services/instances.ts."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..context import BACKGROUND_CONTEXT, CancelContext, Context, with_cancel
from ._listeners import LiveMap


class InstanceDirectoryEntry(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def generation(self) -> int: ...

    @property
    def service(self) -> object: ...

    def deactivate(self) -> None: ...


@dataclass(eq=False)
class _Observer:
    handler: Callable[[object, Context], None | Awaitable[None]]
    tasks: LiveMap[int, tuple[InstanceDirectoryEntry, CancelContext]] = field(default_factory=LiveMap)
    closed: bool = False


class InstanceDirectory[TEntry: InstanceDirectoryEntry]:
    def __init__(self, *, ready: bool, on_error: Callable[[Exception], None]) -> None:
        self._entries: LiveMap[str, TEntry] = LiveMap()
        self._observers: LiveMap[_Observer, None] = LiveMap()
        self._report_error = on_error
        self._ready = ready
        self._disposed = False
        self._pending: set[asyncio.Task[None]] = set()

    @property
    def observer_count(self) -> int:
        return len(self._observers)

    def get(self, key: str) -> TEntry | None:
        return self._entries.get(key)

    def insert(self, entry: TEntry) -> None:
        self._assert_active()
        if entry.key in self._entries:
            raise RuntimeError(f"Keyed service already has a live instance with key {entry.key}")
        self._entries[entry.key] = entry
        if self._ready:
            self._start_all(entry)

    def replace(self, entry: TEntry) -> None:
        self._assert_active()
        previous = self._entries.get(entry.key)
        if previous is not None:
            if previous.generation == entry.generation:
                raise RuntimeError("Keyed service repeated a live generation")
            self._remove(previous)
        self._entries[entry.key] = entry
        if self._ready:
            self._start_all(entry)

    def remove(self, entry: TEntry) -> None:
        if self._entries.get(entry.key) is entry:
            self._remove(entry)

    def ready(self) -> None:
        self._assert_active()
        if self._ready:
            return
        self._ready = True
        for entry in self._entries.values():
            self._start_all(entry)

    def reset(self) -> None:
        if self._disposed:
            return
        self._ready = False
        for entry in tuple(self._entries.values()):
            self._remove(entry)

    def observe(
        self, handler: Callable[[object, Context], None | Awaitable[None]],
    ) -> Callable[[], None]:
        self._assert_active()
        observer = _Observer(handler)
        self._observers[observer] = None
        if self._ready:
            for entry in self._entries.values():
                self._start(observer, entry)

        def close() -> None:
            if observer.closed:
                return
            observer.closed = True
            for _, task in observer.tasks.values():
                task.cancel(None)
            observer.tasks.clear()
            self._observers.pop(observer, None)

        return close

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for observer in self._observers:
            observer.closed = True
            for _, task in observer.tasks.values():
                task.cancel(None)
            observer.tasks.clear()
        self._observers.clear()
        for entry in tuple(self._entries.values()):
            entry.deactivate()
        self._entries.clear()

    def _remove(self, entry: TEntry) -> None:
        if self._entries.get(entry.key) is not entry:
            return
        del self._entries[entry.key]
        entry.deactivate()
        for observer in self._observers:
            task = observer.tasks.get(id(entry))
            if task is not None:
                task[1].cancel(None)
            observer.tasks.pop(id(entry), None)

    def _start_all(self, entry: TEntry) -> None:
        for observer in self._observers:
            self._start(observer, entry)

    def _start(self, observer: _Observer, entry: TEntry) -> None:
        if observer.closed or id(entry) in observer.tasks:
            return
        task_context = with_cancel(BACKGROUND_CONTEXT)
        observer.tasks[id(entry)] = (entry, task_context)
        context = task_context.context
        try:
            result = observer.handler(entry.service, context)
        except (Exception, asyncio.CancelledError) as error:
            if context.abort_signal is None or not context.abort_signal.aborted:
                self._report_error(error if isinstance(error, Exception) else RuntimeError(str(error)))
            return
        if inspect.isawaitable(result):
            async def observe_completion() -> None:
                try:
                    await result
                except (Exception, asyncio.CancelledError) as error:
                    if context.abort_signal is None or not context.abort_signal.aborted:
                        self._report_error(error if isinstance(error, Exception) else RuntimeError(str(error)))

            pending = asyncio.get_running_loop().create_task(observe_completion())
            self._pending.add(pending)
            pending.add_done_callback(self._pending.discard)

    def _assert_active(self) -> None:
        if self._disposed:
            raise RuntimeError("Keyed service directory is disposed")
