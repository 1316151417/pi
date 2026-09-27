"""Combine abort signals with explicit listener cleanup."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from ..abort import AbortController, AbortSignal

__all__ = ["CombinedAbortSignal", "combine_abort_signals"]


@dataclass
class CombinedAbortSignal:
    cleanup: Callable[[], None]
    signal: AbortSignal | None = None


def combine_abort_signals(signals: Sequence[AbortSignal | None]) -> CombinedAbortSignal:
    active_signals = [signal for signal in signals if signal is not None]
    if not active_signals:
        return CombinedAbortSignal(cleanup=lambda: None)
    if len(active_signals) == 1:
        return CombinedAbortSignal(signal=active_signals[0], cleanup=lambda: None)

    controller = AbortController()
    listeners: list[tuple[AbortSignal, Callable[[], None]]] = []

    def abort(signal: AbortSignal) -> None:
        if not controller.signal.aborted:
            controller.abort(signal.reason)

    for signal in active_signals:
        if signal.aborted:
            abort(signal)
            break
        listener = partial(abort, signal)
        signal.add_event_listener("abort", listener, once=True)
        listeners.append((signal, listener))

    def cleanup() -> None:
        for signal, listener in listeners:
            signal.remove_event_listener("abort", listener)

    return CombinedAbortSignal(signal=controller.signal, cleanup=cleanup)
