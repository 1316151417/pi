"""Effect gate ported from ``harness/execution/effect-gate.ts``.

A gate separates the procedure-facing synchronous admission capability from the
owner-facing lifecycle controls for one drive pass.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

from ..._pi_ai.abort import AbortController, AbortSignal

__all__ = ["AbortRequested", "Gate", "GateControl", "create_gate"]

T = TypeVar("T")


class AbortRequested(Exception):
    """Expected internal control flow when cancellation wins effect admission."""

    def __init__(self, cancellation: Awaitable[None]) -> None:
        super().__init__("Abort requested")
        self.name = "AbortRequested"
        self.cancellation = cancellation


class Gate:
    """Procedure-facing synchronous admission capability for one drive pass."""

    def __init__(self, controller: AbortController, check: Callable[[], None]) -> None:
        self._controller = controller
        self._check = check

    @property
    def signal(self) -> AbortSignal:
        return self._controller.signal

    def admit(self, invoke: Callable[[], T]) -> T:
        self._check()
        return invoke()


class GateControl:
    """Owner-facing lifecycle controls for one drive pass."""

    def __init__(
        self,
        get_state: Callable[[], dict],
        set_state: Callable[[dict], None],
        controller: AbortController,
    ) -> None:
        self._get_state = get_state
        self._set_state = set_state
        self._controller = controller

    def begin_abort(self, cancellation: Awaitable[None]) -> None:
        if self._get_state()["status"] != "open":
            return
        self._set_state({"status": "aborting", "cancellation": cancellation})

    def signal_abort(self) -> None:
        state = self._get_state()
        if state["status"] != "aborting" or self._controller.signal.aborted:
            return
        self._controller.abort(AbortRequested(state["cancellation"]))

    def close(self, error: BaseException) -> None:
        if self._get_state()["status"] == "closed":
            return
        self._set_state({"status": "closed", "error": error})
        if not self._controller.signal.aborted:
            self._controller.abort(error)


def create_gate() -> tuple:
    """Create separate procedure-facing and owner-facing views of one effect gate."""
    state: dict = {"status": "open"}
    controller = AbortController()

    def _check() -> None:
        if state["status"] == "aborting":
            raise AbortRequested(state["cancellation"])
        if state["status"] == "closed":
            raise state["error"]

    gate = Gate(controller, _check)
    control = GateControl(lambda: state, lambda next_state: state.update(next_state), controller)
    return gate, control
