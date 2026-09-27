"""Wait for either Chord or AI abort signals through their shared event API."""

from __future__ import annotations

import asyncio

from pi_chord.context import AbortSignalLike


async def wait_for_abort(signal: AbortSignalLike) -> None:
    if signal.aborted:
        return
    waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def on_abort() -> None:
        if not waiter.done():
            waiter.set_result(None)

    signal.add_event_listener("abort", on_abort, once=True)
    try:
        # Cover aborts occurring between the initial check and registration.
        if signal.aborted:
            on_abort()
        await waiter
    finally:
        signal.remove_event_listener("abort", on_abort)
