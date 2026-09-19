"""File mutation serialization ported from ``harness/tools/file-mutation-queue.ts``."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from ..._chord.context import Context
from ..types import ExecutionEnv, get_or_throw

__all__ = ["with_file_mutation_queue"]


class _MutationQueueState:
    def __init__(self) -> None:
        self.queues: Dict[str, asyncio.Future] = {}
        self.lock = asyncio.Lock()


_states: Dict[int, _MutationQueueState] = {}


def _get_state(env: ExecutionEnv) -> _MutationQueueState:
    key = id(env)
    state = _states.get(key)
    if state is None:
        state = _MutationQueueState()
        _states[key] = state
    return state


async def _get_mutation_queue_key(env: ExecutionEnv, path: str, context: Context) -> str:
    absolute_path = get_or_throw(await env.absolute_path(path, context))
    canonical_path = await env.canonical_path(absolute_path, context)
    if canonical_path.ok:
        return canonical_path.value
    error_code = getattr(canonical_path.error, "code", "")
    if error_code in ("not_found", "not_supported"):
        return absolute_path
    raise canonical_path.error


async def with_file_mutation_queue(env: ExecutionEnv, path: str, fn, context: Context) -> Any:
    """Serialize file mutations targeting the same environment and canonical path."""
    state = _get_state(env)
    async with state.lock:
        key = await _get_mutation_queue_key(env, path, context)
        current_queue = state.queues.get(key)
        next_queue: asyncio.Future = asyncio.get_running_loop().create_future()
        if current_queue is not None:
            chained = current_queue
        else:
            chained = asyncio.get_running_loop().create_future()
            chained.set_result(None)
        state.queues[key] = next_queue

    try:
        if current_queue is not None:
            await current_queue
        return await fn()
    finally:
        next_queue.set_result(None)
        if state.queues.get(key) is next_queue:
            del state.queues[key]
