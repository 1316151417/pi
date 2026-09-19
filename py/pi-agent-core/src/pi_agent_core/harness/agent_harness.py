"""AgentHarness public surface ported from ``harness/agent-harness.ts``.

Defines the durable, lane-based orchestrator API: options, the lane and harness
protocols, hook names and payloads, operation requests and results, and the
``AgentHarness.create`` constructor.

Runtime note: the concrete ``Harness``/``Lane`` runtime lives under
``harness/runtime``; until that runtime is installed, ``create_agent_harness``
raises :class:`SliceNotImplemented` so callers fail loudly rather than silently
receiving a partially wired harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Union

from pi_ai.models import Models
from pi_ai.types import AssistantMessage, ImageContent, Model, Tool as AiTool, Usage
from .._chord.context import Context
from .compaction.branch_summarization import BranchPreparation, BranchSummaryResult
from .compaction.compaction import CompactResult, CompactionPreparation, CompactionSettings
from .events import (
    CurrentOperationInfo,
    HarnessEvent,
    LaneInfo,
    LaneSnapshot,
    ModelIdentity,
    OperationError,
    SessionSnapshot,
    WatchHandle,
)
from .prompt_templates import PromptTemplate
from .result import Result, err, ok
from .session.types import Entry, OperationResultRecord
from .skills import Skill
from .types import AgentHarnessStreamOptions, AgentHarnessTool

__all__ = [
    "Resources",
    "AgentHarnessResources",
    "SuspendedRun",
    "AgentHarnessOptions",
    "AgentLane",
    "AgentHarness",
    "AgentHarnessConstructor",
    "AcquireLaneOptions",
    "NavigateOptions",
    "OperationRequest",
    "OperationAdmission",
    "OperationAdmissionError",
    "DriveOptions",
    "DriveOutcome",
    "AbortRequestResult",
    "LaneExecutionInfo",
    "HookMap",
    "HookName",
    "HookInvocation",
    "HookHandler",
    "Hooks",
    "create_agent_harness",
    "AgentHarnessNamespace",
]

# ---------------------------------------------------------------------------
# Result aliases (TS type aliases over Result<...>)
# ---------------------------------------------------------------------------

#: ``Result[OperationResultRecord | SuspendedRun, LaneBusy | InvalidMessage | UnknownSkill | UnknownTemplate | Closed]``
RunResult = Result
#: ``Result[{compaction, run?}, LaneBusy | NothingToCompact | Closed]``
CompactionResult = Result
#: ``Result[{navigation, run?}, LaneBusy | InvalidNavigation | UnknownTarget | Closed]``
NavigationResult = Result
#: ``Result[OperationResultRecord | SuspendedRun, NothingToResume | Closed]``
ResumeResult = Result
#: ``Result[{entryId}, InvalidMessage | Closed]``
QueueResult = Result
#: ``Result[{kind}, Closed]``
CancelQueuedResult = Result
#: ``Result[{operationId, steer, followUp}, NoActiveOperation | Closed]``
AbortResult = Result
#: ``Result[{usageId}, Closed]``
RecordUsageResult = Result
#: ``Result[DriveOutcome, OperationMismatch | Closed]``
DriveResult = Result
#: ``Result[OperationAdmission, OperationAdmissionError]``
OperationAdmissionResult = Result


@dataclass
class SuspendedRun:
    """Convenience-only suspended run observation."""

    operation_id: str
    status: str = "suspended"
    deferred: Any = None


@dataclass
class NavigateOptions:
    summarize: Optional[bool] = None
    label: Optional[str] = None
    custom_instructions: Optional[str] = None


@dataclass
class OperationRequest:
    """One admitted unit of work for a lane."""

    kind: str  # "prompt" | "skill" | "prompt_template" | "compaction" | "navigation"
    operation_id: Optional[str] = None
    prompt: Any = None  # str | AgentMessage | list[AgentMessage]
    images: Optional[List[ImageContent]] = None
    name: Optional[str] = None
    additional_instructions: Optional[str] = None
    args: Optional[List[str]] = None
    custom_instructions: Optional[str] = None
    target_id: Optional[str] = None
    options: Optional[NavigateOptions] = None


@dataclass
class OperationAdmission:
    operation_id: str
    kind: str = "run"


@dataclass
class DriveOptions:
    operation_id: str = ""
    wait_for_retry: bool = False
    poll_deferred: bool = False


@dataclass
class DriveOutcome:
    kind: str  # "settled" | "waiting"
    outcome: Optional[OperationResultRecord] = None
    operation_id: Optional[str] = None
    reason: Optional[str] = None  # "retry" | "deferred"
    not_before: Optional[int] = None
    deferred: Any = None


@dataclass
class AbortRequestResult:
    operation_id: str
    newly_requested: bool
    steer: List[Any] = field(default_factory=list)
    follow_up: List[Any] = field(default_factory=list)


@dataclass
class LaneExecutionInfo:
    lane: str
    tip_id: Optional[str]
    configured_model: Optional[ModelIdentity]
    current: Optional[CurrentOperationInfo]
    last_operation_id: Optional[str]


@dataclass
class AcquireLaneOptions:
    create_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Resources and hooks
# ---------------------------------------------------------------------------


@dataclass
class AgentHarnessResources:
    """User-visible durable resources attached to a harness session."""

    skills: List[Skill] = field(default_factory=list)
    prompt_templates: List[PromptTemplate] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


#: Concrete resource bundle used by the harness.
Resources = AgentHarnessResources


@dataclass
class HookMap:
    """Hook names and their payload/result shapes.

    Mirrors the TS ``HookMap`` interface as data so hook dispatch can validate
    names and build payloads without a separate type per hook.
    """

    BEFORE_RUN = "before_run"
    BEFORE_DRIVE = "before_drive"
    BEFORE_RUN_END = "before_run_end"
    TRANSFORM_CONTEXT = "transform_context"
    BEFORE_REQUEST = "before_request"
    BEFORE_PAYLOAD = "before_payload"
    AFTER_RESPONSE = "after_response"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_COMPACTION = "before_compaction"
    BEFORE_NAVIGATION = "before_navigation"

    NAMES = (
        "before_run",
        "before_drive",
        "before_run_end",
        "transform_context",
        "before_request",
        "before_payload",
        "after_response",
        "before_tool",
        "after_tool",
        "before_compaction",
        "before_navigation",
    )


#: Hook name literal.
HookName = str

#: Hook invocation: hook event payload plus lane and run identity.
HookInvocation = dict

#: Hook handler returning the hook's result payload.
HookHandler = Callable[[dict, Context], Union[Any, Awaitable[Any]]]


class Hooks(Protocol):
    """Hook registration surface."""

    def on(self, name: HookName, handler: HookHandler, options: Optional[dict] = None) -> Callable[[], None]: ...


class Events(Protocol):
    """Event subscription surface."""

    def on(self, event_type: str, listener: Callable[[HarnessEvent, Context], Any]) -> Callable[[], None]: ...


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class AgentHarnessOptions:
    """Construction options for an :class:`AgentHarness`."""

    session: Any = None
    models: Optional[Models] = None
    model: Optional[Model] = None
    thinking_level: Optional[str] = None
    active_tool_names: Optional[List[str]] = None
    tools: List[AgentHarnessTool] = field(default_factory=list)
    tool_context: Any = None
    system_prompt: Any = None  # str | Callable[[TContext, Context], str]
    resources: Optional[Resources] = None
    stream_options: Optional[AgentHarnessStreamOptions] = None
    retry: Any = None
    compaction: Optional[CompactionSettings] = None
    steering_mode: str = "all"
    follow_up_mode: str = "all"
    tool_execution: str = "parallel"
    to_provider_messages: Optional[Callable] = None
    entry_projectors: Dict[str, Any] = field(default_factory=dict)


class AgentLane(Protocol):
    """Durable lane: the unit of work that owns a branch of the session tree."""

    @property
    def name(self) -> str: ...

    def get_tip_id(self, context: Context) -> Awaitable[Optional[str]]: ...
    def find_entries(self, query: Any, context: Context) -> Awaitable[List[Entry]]: ...
    def find_entry(self, query: Any, context: Context) -> Awaitable[Optional[Entry]]: ...
    def append_message(self, message: Any, context: Context) -> Awaitable[str]: ...
    def append_custom_entry(self, custom_type: str, data: Any, context: Context) -> Awaitable[str]: ...
    def get_result(self, operation_id: str, context: Context) -> Awaitable[Optional[OperationResultRecord]]: ...
    def accept(self, request: OperationRequest, context: Context) -> Awaitable[OperationAdmissionResult]: ...
    def drive(self, options: DriveOptions, context: Context) -> Awaitable[DriveResult]: ...
    def request_abort(self, operation_id: str, context: Context) -> Awaitable[AbortRequestResult]: ...
    def inspect_execution(self, context: Context) -> Awaitable[LaneExecutionInfo]: ...
    def prompt(self, prompt: Any, images: Optional[List[ImageContent]], context: Context) -> Awaitable[RunResult]: ...
    def skill(self, name: str, additional_instructions: Optional[str], context: Context) -> Awaitable[RunResult]: ...
    def prompt_from_template(self, name: str, args: Optional[List[str]], context: Context) -> Awaitable[RunResult]: ...
    def compact(self, options: Optional[dict], context: Context) -> Awaitable[CompactionResult]: ...
    def navigate_tree(
        self, target_id: Optional[str], options: Optional[NavigateOptions], context: Context
    ) -> Awaitable[NavigationResult]: ...
    def resume(self, context: Context) -> Awaitable[ResumeResult]: ...
    def abort(self, context: Context) -> Awaitable[AbortResult]: ...
    def steer(self, message: Any, images: Optional[List[ImageContent]], context: Context) -> Awaitable[QueueResult]: ...
    def follow_up(
        self, message: Any, images: Optional[List[ImageContent]], context: Context
    ) -> Awaitable[QueueResult]: ...
    def next_run(
        self, message: Any, images: Optional[List[ImageContent]], context: Context
    ) -> Awaitable[QueueResult]: ...
    def cancel_queued(self, entry_id: str, context: Context) -> Awaitable[CancelQueuedResult]: ...
    def record_usage(
        self, usage: Usage, options: Optional[dict], context: Context
    ) -> Awaitable[RecordUsageResult]: ...
    def wait_for_idle(self, context: Context) -> Awaitable[None]: ...
    def run_when_idle(self, callback: Callable[[Context], Any], context: Context) -> Awaitable[None]: ...
    def get_model(self, context: Context) -> Awaitable[Optional[Model]]: ...
    def set_model(self, model: ModelIdentity, context: Context) -> Awaitable[None]: ...
    def get_thinking_level(self, context: Context) -> Awaitable[str]: ...
    def set_thinking_level(self, level: str, context: Context) -> Awaitable[None]: ...
    def get_active_tools(self, context: Context) -> Awaitable[List[str]]: ...
    def set_active_tools(self, names: List[str], context: Context) -> Awaitable[None]: ...
    def watch(self, context: Context) -> Awaitable[WatchHandle]: ...


class AgentHarness(Protocol):
    """Runtime harness: manages lanes but is not itself a lane."""

    @property
    def hooks(self) -> Hooks: ...

    @property
    def events(self) -> Events: ...

    def lane(self, name: str, *args: Any) -> Awaitable[AgentLane]: ...
    def lanes(self, context: Context) -> Awaitable[List[LaneInfo]]: ...
    def get_name(self, context: Context) -> Awaitable[Optional[str]]: ...
    def set_name(self, name: Optional[str], context: Context) -> Awaitable[None]: ...
    def get_label(self, target_id: str, context: Context) -> Awaitable[Optional[str]]: ...
    def set_label(self, target_id: str, label: Optional[str], context: Context) -> Awaitable[None]: ...
    def get_tools(self, context: Context) -> Awaitable[List[AgentHarnessTool]]: ...
    def set_tools(self, tools: List[AgentHarnessTool], context: Context) -> Awaitable[None]: ...
    def get_resources(self, context: Context) -> Awaitable[Resources]: ...
    def set_resources(self, resources: Resources, context: Context) -> Awaitable[None]: ...
    def get_stream_options(self, context: Context) -> Awaitable[AgentHarnessStreamOptions]: ...
    def set_stream_options(self, options: AgentHarnessStreamOptions, context: Context) -> Awaitable[None]: ...
    def get_retry_policy(self, context: Context) -> Awaitable[Any]: ...
    def set_retry_policy(self, policy: Any, context: Context) -> Awaitable[None]: ...
    def get_compaction_settings(self, context: Context) -> Awaitable[CompactionSettings]: ...
    def set_compaction_settings(self, settings: CompactionSettings, context: Context) -> Awaitable[None]: ...
    def get_steering_mode(self, context: Context) -> Awaitable[str]: ...
    def set_steering_mode(self, mode: str, context: Context) -> Awaitable[None]: ...
    def get_follow_up_mode(self, context: Context) -> Awaitable[str]: ...
    def set_follow_up_mode(self, mode: str, context: Context) -> Awaitable[None]: ...
    def watch_session(self, context: Context) -> Awaitable[WatchHandle]: ...
    def close(self, context: Context) -> Awaitable[None]: ...


class AgentHarnessConstructor(Protocol):
    """Attaches the durable harness to one open session."""

    def create(self, options: AgentHarnessOptions, context: Context) -> Awaitable[Any]: ...


async def create_agent_harness(options: AgentHarnessOptions, context: Context) -> dict:
    """Attach runtime without starting provider, tool, hook, or timer effects.

    Returns ``{"harness": AgentHarness, "open": list[OpenOperation]}``. The
    concrete :class:`Harness` lives in :mod:`.runtime.harness`; this is the
    public entry point and the single implementation both share.
    """
    from .runtime.harness import create_agent_harness as _create

    harness, open_operations = await _create(options, context)
    return {"harness": harness, "open": open_operations}


class AgentHarnessNamespace:
    """Runtime constructor namespace mirroring the TS ``AgentHarness`` const object."""

    @staticmethod
    async def create(options: AgentHarnessOptions, context: Context) -> dict:
        return await create_agent_harness(options, context)
