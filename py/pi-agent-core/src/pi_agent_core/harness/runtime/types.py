"""Runtime types ported from ``harness/runtime/types.ts``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from ..._pi_ai.abort import AbortController, AbortSignal
from ..._chord.context import Context, without_abort_signal
from ..compaction.compaction import CompactionSettings
from ..execution.effect_gate import Gate, create_gate
from ..session.types import (
    CommitResult,
    InboxItem,
    LaneConfiguration,
    OperationResultRecord,
    Write,
)

__all__ = [
    "SliceNotImplemented",
    "Config",
    "LaneState",
    "CommitDecision",
    "LaneCommand",
    "ContinueOperationResult",
    "FinishDecision",
    "LaneReturn",
    "LaneReject",
    "OperationCommand",
    "Drive",
    "ProcedureResult",
]


class SliceNotImplemented(Exception):
    """Raised for operations deferred to a later AgentHarness slice."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} is not implemented until its later AgentHarness slice")
        self.name = "SliceNotImplemented"


@dataclass
class Config:
    """Current process-local harness configuration."""

    tools: List[Any] = field(default_factory=list)
    resources: Any = None
    stream_options: Any = None
    retry_policy: Any = None
    compaction: Optional[CompactionSettings] = None
    steering_mode: str = "one-at-a-time"
    follow_up_mode: str = "one-at-a-time"
    tool_execution: str = "parallel"
    tool_context: Any = None
    system_prompt: Any = None
    to_provider_messages: Optional[Callable] = None
    entry_projectors: dict = field(default_factory=dict)


@dataclass
class LaneState:
    """The current durable state owned by one lane."""

    tip_id: Optional[str] = None
    configuration: Optional[LaneConfiguration] = None
    inbox: List[InboxItem] = field(default_factory=list)
    last_operation_id: Optional[str] = None
    operation: Any = None


@dataclass
class CommitDecision:
    """One effect-free commit decision made on a lane's serialized mutation line."""

    kind: str = "commit"
    writes: List[Write] = field(default_factory=list)
    materialize: Optional[Callable[[CommitResult], Any]] = None
    events: Optional[Callable[[CommitResult], List[Any]]] = None
    next: Optional[LaneState] = None
    operation_state: Any = None
    lane: Optional[dict] = None


#: One effect-free decision made on a lane's serialized mutation line.
LaneCommand = Any  # CommitDecision | LaneReturn | LaneReject


@dataclass
class LaneReturn:
    """A plan that returns without a commit."""

    kind: str = "return"
    result: Any = None


@dataclass
class LaneReject:
    """A plan that rejects as an expected caller error without a commit."""

    kind: str = "reject"
    error: Any = None


@dataclass
class ContinueOperationResult:
    kind: str  # "cancel_requested" | "result"
    value: Any = None


@dataclass
class FinishDecision:
    """A durable operation transition; the lane pairs the state write with projection publication."""

    kind: str = "finish"
    writes: List[Write] = field(default_factory=list)
    record: Optional[OperationResultRecord] = None
    lane: Optional[dict] = None
    materialize: Optional[Callable[[CommitResult], Any]] = None
    events: Optional[Callable[[CommitResult], List[Any]]] = None


#: A durable operation transition command.
OperationCommand = Any  # CommitDecision(operationState) | FinishDecision | LaneReturn


class Drive:
    """One installed process-local drive pass."""

    def __init__(self, options: Any, context: Context) -> None:
        self.operation_id: str = options.operation_id
        self.context: Context = without_abort_signal(context)
        self.wait_for_retry: bool = bool(getattr(options, "wait_for_retry", False))
        self.deferred_permits: int = 1 if getattr(options, "poll_deferred", False) else 0
        self._loop = asyncio.get_running_loop()
        self.completion: "asyncio.Future" = self._loop.create_future()
        # The completion future is consumed by callers; keep the loop quiet about
        # unretrieved exceptions the way the TS `.catch(() => {})` does.
        self.completion.add_done_callback(lambda future: _swallow(future))
        self.gate, self._control = create_gate()
        self._close_controller = AbortController()
        self.close_signal: AbortSignal = self._close_controller.signal

    def settle(self, outcome: Any) -> None:
        if not self.completion.done():
            self.completion.set_result(outcome)

    def fail(self, error: BaseException) -> None:
        if not self.completion.done():
            self.completion.set_exception(error)

    def begin_abort(self, cancellation: Awaitable[None]) -> None:
        self._control.begin_abort(cancellation)

    def signal_abort(self) -> None:
        self._control.signal_abort()

    def close_gate(self, error: BaseException) -> None:
        self._control.close(error)
        if not self.close_signal.aborted:
            self._close_controller.abort(error)
        if not self.completion.done():
            self.completion.set_exception(error)


def _swallow(future: "asyncio.Future") -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


@dataclass
class ProcedureResult:
    kind: str  # "continue" | "waiting" | "settled"
    outcome: Any = None
