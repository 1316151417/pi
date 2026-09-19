"""Harness events ported from ``harness/agent-harness.ts`` (event payload union).

Events are the durable, JSON-serializable notification surface of an
:class:`AgentHarness`. Python represents the TS discriminated union as one
dataclass with a ``type`` discriminator and optional payload fields, which
keeps ``event.type`` checks and field access identical to the TypeScript shape.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Union

from .._pi_ai.types import AssistantMessage, JsonObject, ToolResultMessage, Usage
from .session.types import LaneConfiguration, SessionStats, UsageRow
from .utils.usage import empty_usage

__all__ = [
    "HarnessEventBus",
    "WatchHandle",
    "OperationError",
    "OperationStatus",
    "OperationKind",
    "HarnessEvent",
    "harness_event",
    "HarnessEventType",
    "EventListener",
    "SPECIAL_EVENT_TYPES",
    "LANE_CONFIG_PROPERTIES",
    "GLOBAL_CONFIG_PROPERTIES",
    "LaneSnapshotTool",
    "LaneQueuedItem",
    "LaneSnapshot",
    "LaneInfo",
    "CurrentOperationInfo",
    "SessionSnapshot",
    "ModelIdentity",
]

OperationStatus = str  # "running" | "open" | "aborting"
OperationKind = str  # "run" | "compaction" | "navigation"


@dataclass
class OperationError:
    """Failure payload recorded on operation result records."""

    code: str = "unknown"
    message: str = ""

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class ModelIdentity:
    provider: str = ""
    model_id: str = ""

    def to_json(self) -> dict:
        return {"provider": self.provider, "modelId": self.model_id}


@dataclass
class CurrentOperationInfo:
    id: str
    kind: OperationKind
    started_at: int
    status: OperationStatus
    captured_model: Optional[ModelIdentity] = None


@dataclass
class LaneSnapshotTool:
    """Running or settled tool entry inside a lane snapshot operation."""

    status: str  # "running" | "settled"
    tool_call_id: str
    tool_name: str
    args: Any = None
    result: Any = None
    is_error: Optional[bool] = None


@dataclass
class LaneQueuedItem:
    entry_id: str
    kind: str  # "steer" | "followUp" | "nextRun" | "write"
    type: str = "message"  # "message" | "custom"
    message: Any = None
    custom_type: Optional[str] = None
    data: Any = None


@dataclass
class LaneOperationState:
    """Live operation state carried on a lane snapshot."""

    id: str
    kind: OperationKind
    started_at: int
    from_tip_id: Optional[str]
    status: OperationStatus
    retry: Optional[dict] = None  # {"attempt", "maxAttempts", "nextAttemptAt"}
    deferred: Optional[dict] = None  # {"handle", "poll"}
    streaming_message: Optional[AssistantMessage] = None
    running_tools: List[LaneSnapshotTool] = field(default_factory=list)


@dataclass
class LaneSnapshot:
    lane: str
    transcript: List[Any] = field(default_factory=list)
    tip_id: Optional[str] = None
    last_result: Any = None
    configuration: Optional[LaneConfiguration] = None
    stats: Optional[SessionStats] = None
    operation: Optional[LaneOperationState] = None
    queues: List[LaneQueuedItem] = field(default_factory=list)
    faulted: bool = False

    def to_json(self) -> dict:
        configuration = self.configuration
        return {
            "lane": self.lane,
            "transcript": [_entry_to_json(entry) for entry in self.transcript],
            "tipId": self.tip_id,
            "lastResult": self.last_result.to_json() if hasattr(self.last_result, "to_json") else self.last_result,
            "configuration": {
                "model": {
                    "provider": configuration.model.get("provider", ""),
                    "modelId": configuration.model.get("modelId", ""),
                }
                if configuration
                else {"provider": "", "modelId": ""},
                "thinkingLevel": configuration.thinking_level if configuration else "off",
                "activeToolNames": list(configuration.active_tool_names) if configuration else [],
            },
            "stats": {
                "messageCount": self.stats.message_count if self.stats else 0,
                "usage": (self.stats.usage.to_json() if self.stats else empty_usage().to_json()),
            },
            "operation": _operation_to_json(self.operation),
            "queues": [
                {
                    "entryId": item.entry_id,
                    "kind": item.kind,
                    "type": item.type,
                    **({"message": item.message} if item.message is not None else {}),
                    **({"customType": item.custom_type} if item.custom_type else {}),
                    **({"data": item.data} if item.data is not None else {}),
                }
                for item in self.queues
            ],
            "faulted": self.faulted,
        }


def _operation_to_json(operation: Optional[LaneOperationState]) -> Optional[dict]:
    if operation is None:
        return None
    data: dict = {
        "id": operation.id,
        "kind": operation.kind,
        "startedAt": operation.started_at,
        "fromTipId": operation.from_tip_id,
        "status": operation.status,
        "runningTools": [
            {
                "status": tool.status,
                "toolCallId": tool.tool_call_id,
                "toolName": tool.tool_name,
                "args": tool.args,
                **({"result": tool.result} if tool.result is not None else {}),
                **({"isError": tool.is_error} if tool.is_error is not None else {}),
            }
            for tool in operation.running_tools
        ],
    }
    if operation.retry is not None:
        data["retry"] = operation.retry
    if operation.deferred is not None:
        data["deferred"] = operation.deferred
    if operation.streaming_message is not None:
        data["streamingMessage"] = (
            operation.streaming_message.to_json()
            if hasattr(operation.streaming_message, "to_json")
            else operation.streaming_message
        )
    return data


def _entry_to_json(entry: Any) -> Any:
    if hasattr(entry, "to_json"):
        return entry.to_json()
    return entry


@dataclass
class LaneInfo:
    name: str
    tip_id: Optional[str] = None
    operation: Optional[CurrentOperationInfo] = None


@dataclass
class SessionSnapshot:
    lanes: List[LaneInfo] = field(default_factory=list)
    faulted: bool = False


# ---------------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------------

SPECIAL_EVENT_TYPES = frozenset({"fault", "value_update", "usage", "config_update", "handler_error"})
LANE_CONFIG_PROPERTIES = frozenset({"model", "thinkingLevel", "activeTools"})
GLOBAL_CONFIG_PROPERTIES = frozenset(
    {
        "tools",
        "resources",
        "streamOptions",
        "retryPolicy",
        "compactionSettings",
        "steeringMode",
        "followUpMode",
    }
)


@dataclass
class HarnessEvent:
    """One harness event.

    ``type`` selects the payload; every other field is populated only for the
    event types that define it (mirrors the TypeScript discriminated union).
    """

    type: str
    lane: Optional[str] = None
    recovery: Optional[bool] = None

    # Operation lifecycle
    run_id: Optional[str] = None
    operation_id: Optional[str] = None
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    from_tip_id: Optional[str] = None
    tip_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[Any] = None
    reason: Optional[str] = None
    target_id: Optional[str] = None
    entry_id: Optional[str] = None
    at: Optional[str] = None

    # Suspend / retry
    deferred: Optional[Any] = None
    poll: Optional[int] = None
    step: Optional[str] = None
    attempt: Optional[int] = None
    max_attempts: Optional[int] = None
    delay_ms: Optional[int] = None
    not_before: Optional[int] = None
    error_message: Optional[str] = None
    success: Optional[bool] = None
    final_error: Optional[str] = None

    # Messages and turns
    message: Optional[Any] = None
    event: Optional[Any] = None  # AssistantMessageEvent on message_update
    frame: Optional[Any] = None
    turn_id: Optional[str] = None
    tool_results: Optional[List[Any]] = None
    steer: Optional[List[Any]] = None
    follow_up: Optional[List[Any]] = None

    # Tools
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    args: Optional[Any] = None
    partial_result: Optional[Any] = None
    result: Optional[Any] = None
    is_error: Optional[bool] = None
    terminate: Optional[bool] = None

    # State
    entry: Optional[Any] = None
    queues: Optional[List[Any]] = None
    row: Optional[UsageRow] = None
    totals: Optional[Usage] = None

    # value_update
    value: Optional[str] = None
    name: Optional[str] = None
    label: Optional[str] = None

    # config_update
    property: Optional[str] = None
    previous: Optional[Any] = None

    # fault / handler_error
    code: Optional[str] = None
    stack: Optional[str] = None
    kind: Optional[str] = None
    hook: Optional[str] = None

    def to_json(self) -> JsonObject:
        """Serialize to the TS wire shape (camelCase, omitting absent fields)."""
        data: JsonObject = {"type": self.type}
        mapping = {
            "lane": self.lane,
            "recovery": self.recovery,
            "runId": self.run_id,
            "operationId": self.operation_id,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "fromTipId": self.from_tip_id,
            "tipId": self.tip_id,
            "status": self.status,
            "reason": self.reason,
            "targetId": self.target_id,
            "entryId": self.entry_id,
            "at": self.at,
            "poll": self.poll,
            "step": self.step,
            "attempt": self.attempt,
            "maxAttempts": self.max_attempts,
            "delayMs": self.delay_ms,
            "notBefore": self.not_before,
            "errorMessage": self.error_message,
            "success": self.success,
            "finalError": self.final_error,
            "turnId": self.turn_id,
            "toolCallId": self.tool_call_id,
            "toolName": self.tool_name,
            "isError": self.is_error,
            "terminate": self.terminate,
            "value": self.value,
            "name": self.name,
            "label": self.label,
            "property": self.property,
            "code": self.code,
            "stack": self.stack,
            "kind": self.kind,
            "hook": self.hook,
        }
        for key, value in mapping.items():
            if value is not None:
                data[key] = value

        for key, value in (
            ("error", self.error),
            ("deferred", self.deferred),
            ("message", self.message),
            ("event", self.event),
            ("frame", self.frame),
            ("args", self.args),
            ("partialResult", self.partial_result),
            ("result", self.result),
            ("entry", self.entry),
            ("previous", self.previous),
            ("row", self.row),
            ("totals", self.totals),
        ):
            if value is None:
                continue
            data[key] = value.to_json() if hasattr(value, "to_json") else value

        for key, value in (
            ("toolResults", self.tool_results),
            ("steer", self.steer),
            ("followUp", self.follow_up),
            ("queues", self.queues),
        ):
            if value is None:
                continue
            data[key] = [item.to_json() if hasattr(item, "to_json") else item for item in value]
        return data

    def is_lane_scoped(self) -> bool:
        """Whether this event belongs to a lane (as opposed to a session-global event)."""
        return self.type not in SPECIAL_EVENT_TYPES


HarnessEventType = str
EventListener = Callable[[HarnessEvent], Union[Awaitable[None], None]]


# ---------------------------------------------------------------------------
# Event bus (ports ``harness/events.ts``)
# ---------------------------------------------------------------------------


class WatchHandle:
    """Snapshot plus a buffered event listener, mirroring the TS ``WatchHandle``."""

    def __init__(self, snapshot: Any, resnapshot_callback: Any, on_error: Any) -> None:
        self.snapshot = snapshot
        self._resnapshot_callback = resnapshot_callback
        self._on_error = on_error
        self._buffer: list = []
        self._listener: Optional[Any] = None
        self._unsubscribe_callback: Optional[Any] = None
        self._delivery_tail: Optional[asyncio.Task] = None
        self._epoch = 0
        self._resnapshot_state: Optional[dict] = None
        self._state = "buffering"

    def set_snapshot(self, snapshot: Any) -> None:
        self.snapshot = snapshot

    def start(self, listener: Any) -> None:
        if self._state != "buffering":
            raise RuntimeError("WatchHandle.start() may be called only once")
        self._state = "started"
        self._listener = listener
        buffered = self._buffer
        self._buffer = []
        for item in buffered:
            self._enqueue(item["event"], item["context"], item["epoch"])

    async def resnapshot(self, context: Any) -> Any:
        if self._state == "unsubscribed":
            raise RuntimeError("WatchHandle is unsubscribed")
        if self._resnapshot_callback is None:
            raise RuntimeError("WatchHandle does not support resnapshot")
        if self._resnapshot_state is not None:
            raise RuntimeError("WatchHandle resnapshot is already in progress")
        reached = asyncio.get_running_loop().create_future()
        state = {"phase": "dropping", "held": [], "reached": reached}
        self._epoch += 1
        self._resnapshot_state = state
        try:
            snapshot = await self._resnapshot_callback(context, self._mark_boundary)
            await reached
            self.snapshot = snapshot
            self._resnapshot_state = None
            for held in state["held"]:
                self.push(held["event"], held["context"])
            return snapshot
        except BaseException:
            self._resnapshot_state = None
            for held in state["held"]:
                self.push(held["event"], held["context"])
            raise

    def _mark_boundary(self) -> None:
        state = self._resnapshot_state
        if state is None or state["phase"] != "dropping":
            return
        state["phase"] = "holding"
        if not state["reached"].done():
            state["reached"].set_result(None)

    def unsubscribe(self) -> None:
        if self._state == "unsubscribed":
            return
        self._state = "unsubscribed"
        self._buffer = []
        self._listener = None
        if self._unsubscribe_callback is not None:
            self._unsubscribe_callback()
        self._unsubscribe_callback = None

    def push(self, event: Any, context: Any) -> None:
        if self._state == "unsubscribed":
            return
        state = self._resnapshot_state
        if state is not None and state["phase"] == "dropping":
            return
        if state is not None and state["phase"] == "holding":
            state["held"].append({"event": event, "context": context})
            return
        if self._state == "buffering":
            self._buffer.append({"event": event, "context": context, "epoch": self._epoch})
            return
        self._enqueue(event, context, self._epoch)

    def set_unsubscribe(self, callback: Any) -> None:
        self._unsubscribe_callback = callback

    def _enqueue(self, event: Any, context: Any, epoch: int) -> None:
        listener = self._listener
        if listener is None:
            return
        previous = self._delivery_tail

        async def _deliver() -> None:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if self._state == "started" and epoch == self._epoch:
                await _maybe_await(listener(event, context))

        async def _guarded() -> None:
            try:
                await _deliver()
            except BaseException as error:  # noqa: BLE001 - report through onError
                try:
                    await _maybe_await(self._on_error(error, event, context))
                except BaseException:  # noqa: BLE001
                    pass

        self._delivery_tail = asyncio.get_running_loop().create_task(_guarded())


#: Wire-key to field mapping for the typed event union.
_EVENT_FIELD_KEYS = {
    "lane": "lane",
    "recovery": "recovery",
    "runId": "run_id",
    "operationId": "operation_id",
    "startedAt": "started_at",
    "endedAt": "ended_at",
    "fromTipId": "from_tip_id",
    "tipId": "tip_id",
    "status": "status",
    "error": "error",
    "reason": "reason",
    "targetId": "target_id",
    "entryId": "entry_id",
    "at": "at",
    "deferred": "deferred",
    "poll": "poll",
    "step": "step",
    "attempt": "attempt",
    "maxAttempts": "max_attempts",
    "delayMs": "delay_ms",
    "notBefore": "not_before",
    "errorMessage": "error_message",
    "success": "success",
    "finalError": "final_error",
    "message": "message",
    "event": "event",
    "frame": "frame",
    "turnId": "turn_id",
    "toolResults": "tool_results",
    "steer": "steer",
    "followUp": "follow_up",
    "toolCallId": "tool_call_id",
    "toolName": "tool_name",
    "args": "args",
    "partialResult": "partial_result",
    "result": "result",
    "isError": "is_error",
    "terminate": "terminate",
    "entry": "entry",
    "queues": "queues",
    "row": "row",
    "totals": "totals",
    "value": "value",
    "name": "name",
    "label": "label",
    "property": "property",
    "previous": "previous",
    "code": "code",
    "stack": "stack",
    "kind": "kind",
    "hook": "hook",
}


def harness_event(payload: Any) -> Any:
    """Normalize one emitted payload into a :class:`HarnessEvent`.

    Producers may hand over either the typed event or its wire shape, exactly
    like the TypeScript API where an event literal is a plain object.
    """
    if isinstance(payload, HarnessEvent):
        return payload
    if not isinstance(payload, dict):
        return payload
    values: dict = {"type": payload.get("type")}
    for key, value in payload.items():
        if key == "type":
            continue
        field = _EVENT_FIELD_KEYS.get(key)
        if field is None:
            snake = _snake_case(key)
            field = snake if snake in HarnessEvent.__dataclass_fields__ else None
        if field in HarnessEvent.__dataclass_fields__:
            values[field] = value
    return HarnessEvent(**values)


def _snake_case(value: str) -> str:
    out = []
    for char in value:
        if char.isupper():
            out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


class HarnessEventBus:
    """Passive harness event bus with isolated handler failures."""

    def __init__(self) -> None:
        self._listeners: dict = {}
        self._watch_listeners: set = set()
        self._delivery_tail: Optional[asyncio.Task] = None
        self._closed_error: Optional[BaseException] = None

    def on(self, event_type: str, listener: EventListener) -> Any:
        """Subscribe to one event type; returns an unsubscribe callable."""
        if self._closed_error is not None:
            raise self._closed_error

        async def _wrapped(event: Any, context: Any) -> None:
            await _maybe_await(listener(event, context))

        listeners = self._listeners.setdefault(event_type, [])
        listeners.append(_wrapped)

        def _unsubscribe() -> None:
            try:
                self._listeners.get(event_type, []).remove(_wrapped)
            except ValueError:
                pass

        return _unsubscribe

    async def emit(self, event: Any, context: Any) -> None:
        await self.emit_batch([event], context)

    async def emit_batch(self, events: List[Any], context: Any) -> None:
        """Bind current recipients and append one contiguous batch to the delivery tail."""
        if self._closed_error is not None or not events:
            return
        bound = []
        for event in events:
            normalized = harness_event(event)
            bound.append(
                {
                    "payload": copy.deepcopy(normalized),
                    "recipients": self._snapshot_recipients(normalized),
                }
            )
        previous = self._delivery_tail

        async def _deliver_batch() -> None:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            for item in bound:
                await self._deliver(item["payload"], item["recipients"], True, context)

        task = asyncio.get_running_loop().create_task(_deliver_batch())
        self._delivery_tail = task
        await task

    def watch(self, snapshot: Any, filter_fn: Any, _context: Any, resnapshot: Any = None) -> WatchHandle:
        if self._closed_error is not None:
            raise self._closed_error
        return self._install_watcher(snapshot, filter_fn, resnapshot)

    async def watch_from_snapshot(self, capture: Any, filter_fn: Any, context: Any) -> WatchHandle:
        if self._closed_error is not None:
            raise self._closed_error

        async def _resnapshot(capture_context: Any, mark_boundary: Any) -> Any:
            snapshot = await capture(capture_context)
            mark_boundary()
            return snapshot

        watcher = self._install_watcher(None, filter_fn, _resnapshot)
        try:
            watcher.set_snapshot(await capture(context))
            return watcher
        except BaseException:
            watcher.unsubscribe()
            raise

    def close(self, error: BaseException) -> None:
        if self._closed_error is None:
            self._closed_error = error
        self._listeners.clear()
        self._watch_listeners.clear()

    def _install_watcher(self, snapshot: Any, filter_fn: Any, resnapshot: Any) -> WatchHandle:
        holder: dict = {}

        async def _on_error(error: BaseException, event: Any, context: Any) -> None:
            if getattr(event, "type", None) == "handler_error":
                return
            lane = getattr(event, "lane", None)
            await self.emit(
                HarnessEvent(
                    type="handler_error",
                    kind="event",
                    event=getattr(event, "type", None),
                    error=str(error),
                    lane=lane if isinstance(lane, str) else None,
                ),
                context,
            )

        async def _capture(context: Any, mark_boundary: Any) -> Any:
            marked = False

            def _mark() -> None:
                nonlocal marked
                if marked:
                    raise RuntimeError("Resnapshot boundary was already marked")
                marked = True
                mark_boundary()

            next_snapshot = await resnapshot(context, _mark)
            if not marked:
                raise RuntimeError("Resnapshot capture did not mark its boundary")
            return next_snapshot

        watcher = WatchHandle(snapshot, _capture if resnapshot is not None else None, _on_error)
        holder["watcher"] = watcher

        def _watch_listener(event: Any, context: Any) -> None:
            if filter_fn(event):
                watcher.push(event, context)

        self._watch_listeners.add(_watch_listener)
        watcher.set_unsubscribe(lambda: self._watch_listeners.discard(_watch_listener))
        return watcher

    def _snapshot_recipients(self, event: Any) -> List[Any]:
        return [*self._listeners.get(getattr(event, "type", None), []), *self._watch_listeners]

    async def _deliver(self, event: Any, recipients: List[Any], report_errors: bool, context: Any) -> None:
        for listener in recipients:
            try:
                await _maybe_await(listener(copy.deepcopy(event), context))
            except BaseException as error:  # noqa: BLE001 - isolated handler failures
                if not report_errors or getattr(event, "type", None) == "handler_error":
                    continue
                lane = getattr(event, "lane", None)
                handler_error = HarnessEvent(
                    type="handler_error",
                    kind="event",
                    event=getattr(event, "type", None),
                    error=str(error),
                    lane=lane if isinstance(lane, str) else None,
                )
                await self._deliver(handler_error, self._snapshot_recipients(handler_error), False, context)
