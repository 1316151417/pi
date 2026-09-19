"""AbortController / AbortSignal mirroring the WHATWG DOM API used by pi."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

__all__ = ["AbortError", "AbortController", "AbortSignal"]


class AbortError(Exception):
    pass


class AbortSignal:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: Any = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Any:
        return self._reason

    def _abort(self, reason: Any = None) -> None:
        if not self._event.is_set():
            if reason is None:
                reason = AbortError("This operation was aborted")
            self._reason = reason
            self._event.set()

    async def wait(self) -> None:
        """Await until the signal is aborted."""
        await self._event.wait()

    def throw_if_aborted(self) -> None:
        if self._event.is_set():
            raise self._reason if isinstance(self._reason, BaseException) else AbortError(str(self._reason))


class AbortController:
    def __init__(self) -> None:
        self._signal = AbortSignal()

    @property
    def signal(self) -> AbortSignal:
        return self._signal

    def abort(self, reason: Any = None) -> None:
        self._signal._abort(reason)


def any_signal(*signals: Optional[AbortSignal]) -> Optional[AbortSignal]:
    """Return a signal that aborts when any of the given signals aborts."""
    live = [s for s in signals if s is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    combined = AbortSignal()

    def _on_abort(signal: AbortSignal) -> None:
        combined._abort(signal.reason)

    for live_signal in live:
        if live_signal.aborted:
            combined._abort(live_signal.reason)
            return combined
        live_signal._event.add_done_callback(lambda f, s=live_signal: _on_abort(s) if f.done() else None)
    return combined
