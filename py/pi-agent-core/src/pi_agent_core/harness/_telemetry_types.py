"""Literal agent telemetry contracts derived from ``harness/telemetry.ts``.

Python cannot index a type by the value of a schema dictionary. Concrete
contracts and the starter overloads retain each declared name and attribute
vocabulary. These declarations add no runtime validation or span wrappers.
Name-parameterized aggregate aliases expose unions; starter overloads provide
the per-name relationship. TypedDict follows Python's structural typing, so
it cannot reproduce TypeScript's ExactTelemetryAttributes excess-key checks
for every independently declared value. None uses pi-telemetry's existing
representation of an omitted optional attribute. Neither schema declares
span events, and their event-name and event-attribute types are therefore Never.
"""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Literal, Never, NotRequired, Protocol, TypedDict

from pi_telemetry import SpanCallback, SpanOptions, SpanStatus

type HookName = Literal[
    "before_run", "before_drive", "before_run_end", "transform_context",
    "before_request", "before_payload", "after_response", "before_tool",
    "after_tool", "before_compaction", "before_navigation",
]
type HarnessEventType = Literal[
    "run_start", "run_resume", "run_suspend", "operation_abort", "run_end",
    "fault", "handler_error", "turn_start", "turn_end", "retry_scheduled",
    "retry_start", "retry_end", "message_start", "message_update", "message_end",
    "tool_start", "tool_update", "tool_end", "entry_added", "queue_update",
    "value_update", "config_update", "compaction_start", "compaction_end",
    "navigation_start", "navigation_end", "lane_created", "usage",
]
type AiSpanName = Literal["pi.ai.request"]
type HarnessSpanName = Literal[
    "pi.harness.run", "pi.harness.compaction", "pi.harness.navigation",
    "pi.harness.checkpoint", "pi.harness.turn", "pi.harness.step",
    "pi.harness.tool", "pi.harness.hook", "pi.harness.sleep",
    "pi.harness.event_handler", "pi.session.write",
]

AiRequestStartAttributes = TypedDict("AiRequestStartAttributes", {
    "pi.ai.operation": Literal["stream", "fetch_deferred", "cancel_deferred", "generate_images"],
    "pi.ai.provider": str,
    "pi.ai.model": str,
    "pi.ai.api": str,
    "pi.ai.streaming": bool,
    "pi.ai.deferred": NotRequired[bool | None],
})
AiRequestEndAttributes = TypedDict("AiRequestEndAttributes", {
    "pi.ai.response.model": str | None,
    "pi.ai.response.id": str | None,
    "pi.ai.response.stop_reason": Literal["stop", "length", "tool_use", "error", "aborted", "deferred"] | None,
    "pi.ai.http.status_code": int | float | None,
    "pi.ai.usage.input_tokens": int | float | None,
    "pi.ai.usage.output_tokens": int | float | None,
    "pi.ai.usage.cache_read_tokens": int | float | None,
    "pi.ai.usage.cache_write_tokens": int | float | None,
    "pi.ai.usage.reasoning_tokens": int | float | None,
    "pi.ai.usage.total_tokens": int | float | None,
    "pi.ai.usage.cost": int | float | None,
    "pi.ai.stream.chunk_count": int | float | None,
    "pi.ai.stream.time_to_first_chunk_ms": int | float | None,
    "pi.ai.error.type": str | None,
}, total=False)


class AiRequestAttributes(AiRequestStartAttributes, AiRequestEndAttributes):
    pass


_OperationStart = TypedDict("_OperationStart", {
    "pi.session.id": str, "pi.lane.name": str,
    "pi.operation.id": str, "pi.operation.recovery": bool,
})
_RunKind = TypedDict("_RunKind", {"pi.operation.kind": Literal["run"]})
_CompactionKind = TypedDict("_CompactionKind", {"pi.operation.kind": Literal["compaction"]})
_NavigationKind = TypedDict("_NavigationKind", {"pi.operation.kind": Literal["navigation"]})


class RunSpanStartAttributes(_OperationStart, _RunKind):
    pass


class CompactionSpanStartAttributes(_OperationStart, _CompactionKind):
    pass


class NavigationSpanStartAttributes(_OperationStart, _NavigationKind):
    pass


RunSpanEndAttributes = TypedDict("RunSpanEndAttributes", {
    "pi.operation.outcome": Literal["completed", "aborted", "failed", "suspended"] | None,
    "pi.error.code": str | None, "pi.error.type": str | None,
}, total=False)
CompactionSpanEndAttributes = TypedDict("CompactionSpanEndAttributes", {
    "pi.operation.outcome": Literal["completed", "declined", "aborted", "failed"] | None,
    "pi.error.code": str | None, "pi.error.type": str | None,
}, total=False)
NavigationSpanEndAttributes = CompactionSpanEndAttributes
CheckpointSpanStartAttributes = TypedDict("CheckpointSpanStartAttributes", {
    "pi.lane.name": str, "pi.operation.id": str,
    "pi.checkpoint.kind": Literal["normal", "abort_reconcile"],
})
TurnSpanStartAttributes = TypedDict("TurnSpanStartAttributes", {
    "pi.lane.name": str, "pi.operation.id": str, "pi.turn.id": str,
})
EmptySpanEndAttributes = TypedDict("EmptySpanEndAttributes", {})
StepSpanStartAttributes = TypedDict("StepSpanStartAttributes", {
    "pi.lane.name": str, "pi.operation.id": str,
    "pi.step.kind": Literal["assistant", "compaction", "branch_summary"],
    "pi.step.attempt": int | float,
    "pi.compaction.reason": NotRequired[Literal["manual", "threshold", "overflow"] | None],
})
StepSpanEndAttributes = TypedDict("StepSpanEndAttributes", {
    "pi.step.outcome": Literal["succeeded", "retry", "failed", "aborted", "deferred", "overflow"] | None,
}, total=False)
ToolSpanStartAttributes = TypedDict("ToolSpanStartAttributes", {
    "pi.lane.name": str, "pi.operation.id": str,
    "pi.turn.id": NotRequired[str | None], "pi.tool.name": str, "pi.tool.call_id": str,
    "pi.tool.replay": Literal["never", "safe"], "pi.tool.recovery": bool,
})
ToolSpanEndAttributes = TypedDict("ToolSpanEndAttributes", {"pi.tool.is_error": bool | None}, total=False)
HookSpanStartAttributes = TypedDict("HookSpanStartAttributes", {
    "pi.lane.name": str, "pi.operation.id": NotRequired[str | None],
    "pi.hook.name": HookName, "pi.hook.registration_id": NotRequired[str | None],
})
HookSpanEndAttributes = TypedDict("HookSpanEndAttributes", {
    "pi.hook.outcome": Literal["completed", "skipped", "blocked", "failed"] | None,
}, total=False)
SleepSpanStartAttributes = TypedDict("SleepSpanStartAttributes", {
    "pi.operation.id": str, "pi.sleep.delay_ms": int | float,
})
SleepSpanEndAttributes = TypedDict("SleepSpanEndAttributes", {
    "pi.sleep.outcome": Literal["elapsed", "aborted"] | None,
}, total=False)
EventHandlerSpanStartAttributes = TypedDict("EventHandlerSpanStartAttributes", {
    "pi.event.type": HarnessEventType, "pi.lane.name": NotRequired[str | None],
})
SessionWriteSpanStartAttributes = TypedDict("SessionWriteSpanStartAttributes", {
    "pi.session.id": str, "pi.lane.name": NotRequired[str | None],
    "pi.operation.id": NotRequired[str | None], "pi.session.item_count": int | float,
    "pi.session.item_kinds": Sequence[Literal["entry", "usage", "value", "list"]],
})
SessionWriteSpanEndAttributes = TypedDict("SessionWriteSpanEndAttributes", {
    "pi.session.first_seq": int | float | None, "pi.session.last_seq": int | float | None,
}, total=False)


class RunSpanAttributes(RunSpanStartAttributes, RunSpanEndAttributes):
    pass


class CompactionSpanAttributes(CompactionSpanStartAttributes, CompactionSpanEndAttributes):
    pass


class NavigationSpanAttributes(NavigationSpanStartAttributes, NavigationSpanEndAttributes):
    pass


class StepSpanAttributes(StepSpanStartAttributes, StepSpanEndAttributes):
    pass


class ToolSpanAttributes(ToolSpanStartAttributes, ToolSpanEndAttributes):
    pass


class HookSpanAttributes(HookSpanStartAttributes, HookSpanEndAttributes):
    pass


class SleepSpanAttributes(SleepSpanStartAttributes, SleepSpanEndAttributes):
    pass


class SessionWriteSpanAttributes(SessionWriteSpanStartAttributes, SessionWriteSpanEndAttributes):
    pass


type AiSpanStartAttributes[Name] = AiRequestStartAttributes
type AiSpanEndAttributes[Name] = AiRequestEndAttributes
type AiSpanAttributes[Name] = AiRequestAttributes
type AiSpanEventName[Name] = Never
type AiSpanEventAttributes[Name, EventName] = Never
type HarnessSpanStartAttributes[Name] = (
    RunSpanStartAttributes | CompactionSpanStartAttributes | NavigationSpanStartAttributes
    | CheckpointSpanStartAttributes | TurnSpanStartAttributes | StepSpanStartAttributes
    | ToolSpanStartAttributes | HookSpanStartAttributes | SleepSpanStartAttributes
    | EventHandlerSpanStartAttributes | SessionWriteSpanStartAttributes
)
type HarnessSpanEndAttributes[Name] = (
    RunSpanEndAttributes | CompactionSpanEndAttributes | NavigationSpanEndAttributes
    | EmptySpanEndAttributes | StepSpanEndAttributes | ToolSpanEndAttributes
    | HookSpanEndAttributes | SleepSpanEndAttributes | SessionWriteSpanEndAttributes
)
type HarnessSpanAttributes[Name] = (
    RunSpanAttributes | CompactionSpanAttributes | NavigationSpanAttributes
    | CheckpointSpanStartAttributes | TurnSpanStartAttributes | StepSpanAttributes
    | ToolSpanAttributes | HookSpanAttributes | SleepSpanAttributes
    | EventHandlerSpanStartAttributes | SessionWriteSpanAttributes
)
type HarnessSpanEventName[Name] = Never
type HarnessSpanEventAttributes[Name, EventName] = Never


class AgentTelemetrySpan[EndAttributes](Protocol):
    def start_span[Result](self, options: SpanOptions, callback: SpanCallback[Result]) -> Awaitable[Result]: ...
    def set_status(self, status: SpanStatus) -> None: ...
    def set_attributes(self, attributes: EndAttributes) -> None: ...
    def add_event(self, name: Never, attributes: Never = ...) -> None: ...


type AiTelemetrySpan[Name] = AgentTelemetrySpan[AiRequestEndAttributes]
type HarnessTelemetrySpan[Name] = AgentTelemetrySpan[HarnessSpanEndAttributes[Name]]


class _SpanDeclaration[Name, StartAttributes, EndAttributes](TypedDict):
    name: Name
    startAttributes: StartAttributes
    endAttributes: EndAttributes
    events: Mapping[Never, Never]


type AiSpan = _SpanDeclaration[Literal["pi.ai.request"], AiRequestStartAttributes, AiRequestEndAttributes]
type HarnessSpan = (
    _SpanDeclaration[Literal["pi.harness.run"], RunSpanStartAttributes, RunSpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.compaction"], CompactionSpanStartAttributes, CompactionSpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.navigation"], NavigationSpanStartAttributes, NavigationSpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.checkpoint"], CheckpointSpanStartAttributes, EmptySpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.turn"], TurnSpanStartAttributes, EmptySpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.step"], StepSpanStartAttributes, StepSpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.tool"], ToolSpanStartAttributes, ToolSpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.hook"], HookSpanStartAttributes, HookSpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.sleep"], SleepSpanStartAttributes, SleepSpanEndAttributes]
    | _SpanDeclaration[Literal["pi.harness.event_handler"], EventHandlerSpanStartAttributes, EmptySpanEndAttributes]
    | _SpanDeclaration[Literal["pi.session.write"], SessionWriteSpanStartAttributes, SessionWriteSpanEndAttributes]
)
