"""Session resource cleanup registration from ``src/session-resources.ts``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "SessionResourceCleanup", "register_session_resource_cleanup", "cleanup_session_resources",
]

type SessionResourceCleanup = Callable[[str | None], None]


@dataclass(frozen=True)
class _CleanupEntry:
    order: int
    cleanup: SessionResourceCleanup


_cleanups: dict[int, _CleanupEntry] = {}
_next_order = 0


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    """Register one callback by identity and return its idempotent unregisterer."""
    global _next_order
    identity = id(cleanup)
    if identity not in _cleanups:
        _cleanups[identity] = _CleanupEntry(_next_order, cleanup)
        _next_order += 1

    def unregister() -> None:
        # Retain the callback so Python cannot reuse its identity while this
        # unregisterer is alive. Like Set.delete, this also removes a callback
        # that was removed and registered again since this handle was created.
        _cleanups.pop(id(cleanup), None)

    return unregister


def cleanup_session_resources(session_id: str | None = None) -> None:
    """Run live callbacks in registration order, then aggregate every failure.

    Callbacks added during cleanup are visited, callbacks removed before their
    turn are skipped, and removing then registering a callback gives it another
    turn. Registrations persist after cleanup, exactly as a JavaScript Set.
    """
    errors: list[BaseException] = []
    previous_order = -1
    while True:
        entry = next((candidate for candidate in _cleanups.values() if candidate.order > previous_order), None)
        if entry is None:
            break
        previous_order = entry.order
        try:
            entry.cleanup(session_id)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup("Failed to cleanup session resources", errors)
