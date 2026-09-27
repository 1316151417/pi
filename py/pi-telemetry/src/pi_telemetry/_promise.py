"""Translate eager JavaScript promises into asyncio awaitables."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Coroutine
from typing import Never


def schedule[Result](coroutine: Coroutine[object, object, Result]) -> Awaitable[Result]:
    """Start work immediately when a loop exists, otherwise defer until awaited."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return coroutine
    return loop.create_task(coroutine)


async def resolve[Result](value: Result | Awaitable[Result]) -> Result:
    if inspect.isawaitable(value):
        return await value
    return value


async def reject(error: BaseException) -> Never:
    raise error
