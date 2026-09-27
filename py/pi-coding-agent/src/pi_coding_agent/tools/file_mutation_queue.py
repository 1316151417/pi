"""Serialize mutations to each canonical path from ``file-mutation-queue.ts``."""

from __future__ import annotations

import asyncio
import errno
import os
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _QueueState:
    registration: asyncio.Lock = field(default_factory=asyncio.Lock)
    tails: dict[str, asyncio.Future[None]] = field(default_factory=dict)


_STATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _QueueState] = weakref.WeakKeyDictionary()


async def _mutation_queue_key(file_path: str) -> str:
    resolved = os.path.abspath(file_path)
    try:
        return str(await asyncio.to_thread(Path(resolved).resolve, strict=True))
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ENOTDIR):
            return resolved
        raise


async def with_file_mutation_queue[T](file_path: str, operation: Callable[[], Awaitable[T]]) -> T:
    loop = asyncio.get_running_loop()
    state = _STATES.setdefault(loop, _QueueState())
    async with state.registration:
        key = await _mutation_queue_key(file_path)
        previous = state.tails.get(key)
        tail: asyncio.Future[None] = loop.create_future()
        state.tails[key] = tail
    try:
        if previous is not None:
            await previous
        return await operation()
    finally:
        if not tail.done():
            tail.set_result(None)
        if state.tails.get(key) is tail:
            del state.tails[key]


__all__ = ["with_file_mutation_queue"]
