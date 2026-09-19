"""Runtime harness ported from ``harness/runtime/harness.ts``.

The harness manages lanes but is not itself a lane: it owns the session, the
models registry, the hook registry, the event bus, and the global config store.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.retry import RetryPolicy
from ..compaction.compaction import CompactionSettings, DEFAULT_COMPACTION_SETTINGS
from ..config import (
    DEFAULT_RETRY_POLICY,
    validate_compaction_settings,
    validate_retry_policy,
    validate_tool_names,
)
from ..events import HarnessEventBus
from ..hooks import HookRegistry
from ..messages import convert_to_llm
from ..result import HarnessClosed, HarnessFault, InvalidLane, UnknownTarget
from ..session.session import SessionInvariantError
from ..session.types import LaneConfiguration
from ..session.values import (
    branch_tip,
    delete_value,
    entry_label,
    lane_config,
    lane_state,
    session_name,
    set_value,
)
from .lane import Lane
from .restore import read_lane_storage, restore_lane_state, restore_session
from .types import Config, LaneState, SliceNotImplemented

__all__ = ["Harness", "OpenOperation", "create_agent_harness"]


@dataclass
class OpenOperation:
    """One operation that was still open when the session was restored."""

    lane: str
    operation_id: str
    kind: str
    started_at: int
    aborting: Optional[bool] = None

    def to_json(self) -> dict:
        data: dict = {
            "lane": self.lane,
            "operationId": self.operation_id,
            "kind": self.kind,
            "startedAt": self.started_at,
        }
        if self.aborting:
            data["aborting"] = True
        return data


@dataclass
class LaneInfo:
    """One lane's public execution summary."""

    name: str
    tip_id: Optional[str]
    operation: Any = None


class Harness:
    """Runtime implementation of ``AgentHarness``."""

    def __init__(
        self,
        options: Any,
        seed: LaneConfiguration,
        restored: Dict[str, LaneState],
    ) -> None:
        self.session = options.session
        self.models = options.models
        self.seed = seed
        self.events = HarnessEventBus()

        def _report_hook_error(error: BaseException, hook: str, lane: Optional[str], context: Any) -> None:
            payload: dict = {
                "type": "handler_error",
                "kind": "hook",
                "hook": hook,
                "error": str(error),
            }
            stack = getattr(error, "__traceback__", None)
            if stack is not None:
                import traceback

                payload["stack"] = "".join(traceback.format_tb(stack))
            if lane is not None:
                payload["lane"] = lane
            asyncio.ensure_future(self.events.emit(payload, context))

        self.hooks = HookRegistry(_report_hook_error)
        resources = getattr(options, "resources", None) or {}
        self._config = Config(
            tools=list(getattr(options, "tools", None) or []),
            resources=resources,
            stream_options=getattr(options, "stream_options", None) or {},
            retry_policy=getattr(options, "retry", None) or DEFAULT_RETRY_POLICY,
            compaction=getattr(options, "compaction", None) or DEFAULT_COMPACTION_SETTINGS,
            steering_mode=getattr(options, "steering_mode", None) or "all",
            follow_up_mode=getattr(options, "follow_up_mode", None) or "all",
            tool_execution=getattr(options, "tool_execution", None) or "parallel",
            tool_context=getattr(options, "tool_context", None),
            system_prompt=getattr(options, "system_prompt", None),
            to_provider_messages=getattr(options, "to_provider_messages", None)
            or (lambda messages, _context=None: convert_to_llm(messages)),
            entry_projectors=getattr(options, "entry_projectors", None) or {},
        )
        self.lanes_by_name: Dict[str, Lane] = {}
        self._close_promise: Optional[asyncio.Future] = None
        self._closed_error: Optional[BaseException] = None
        self._fault_error: Optional[BaseException] = None
        for name, state in restored.items():
            self.lanes_by_name[name] = self.build_lane(name, state)

    # ------------------------------------------------------------------
    # Lane access
    # ------------------------------------------------------------------

    async def lane(self, name: str, *args: Any) -> Lane:
        self.assert_open()
        if len(name) == 0 or "\u0000" in name:
            reason = (
                "lane name must not be empty"
                if len(name) == 0
                else "lane name must not contain \\u0000"
            )
            raise InvalidLane(
                lane=name, reason=reason, message=f"Invalid lane {name!r}: {reason}"
            )
        if len(args) == 1:
            options: Any = {}
            context = args[0]
        else:
            options = args[0]
            context = args[1]
        lane: Optional[Lane] = None
        delivery: Optional[Any] = None

        async def _mutation(mutator: Any, _context: Any) -> None:
            nonlocal lane, delivery
            self.assert_open()
            lane = self.lanes_by_name.get(name)
            if lane is not None:
                return
            stored = await read_lane_storage(mutator, name, context)
            if stored.kind == "lane":
                restored = await restore_lane_state(mutator, name, stored, context)
                lane = self.build_lane(name, restored)
                self.lanes_by_name[name] = lane
                return
            tip_id = (
                stored.tip.value
                if stored.kind == "branch"
                else getattr(options, "create_at", None)
            )
            if (
                stored.kind == "absent"
                and tip_id is not None
                and tip_id not in (await mutator.get_entries([tip_id], context))
            ):
                raise UnknownTarget(target_id=tip_id, message=f"Unknown target: {tip_id}")
            attached_configuration = LaneConfiguration(
                model=dict(self.seed.model),
                thinking_level=self.seed.thinking_level,
                active_tool_names=list(self.seed.active_tool_names),
            )
            state = LaneState(
                tip_id=tip_id,
                configuration=attached_configuration,
                inbox=[],
                last_operation_id=None,
                operation=None,
            )
            writes = [
                *([set_value(branch_tip(name), tip_id)] if stored.kind == "absent" else []),
                set_value(lane_config(name), attached_configuration),
                set_value(
                    lane_state(name),
                    {"currentOperationId": None, "lastOperationId": None, "inbox": []},
                ),
            ]
            await mutator.commit(writes, context)
            lane = self.build_lane(name, state)
            self.lanes_by_name[name] = lane
            delivery = self.events.emit_batch(
                [{"type": "lane_created", "lane": name, "at": tip_id}], context
            )

        try:
            await self.session.mutate(_mutation, context)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if self._closed_error is not None:
                raise self._closed_error
            if isinstance(error, (InvalidLane, UnknownTarget)):
                raise
            raise self.fault(error, context)
        if delivery is not None:
            await delivery
        if lane is None:
            raise self.fault(
                SessionInvariantError(f"Lane {name!r} was not published"), context
            )
        return lane

    async def lanes(self, context: Any) -> List[LaneInfo]:
        self.assert_open()
        executions = await asyncio.gather(
            *[lane.inspect_execution(context) for lane in self.lanes_by_name.values()]
        )
        return [
            LaneInfo(
                name=execution["lane"],
                tip_id=execution["tipId"],
                operation=execution["current"],
            )
            for execution in executions
        ]

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    async def get_name(self, context: Any) -> Optional[str]:
        self.assert_open()
        return await self.session.get_name(context)

    async def set_name(self, name: Optional[str], context: Any) -> None:
        self.assert_open()
        delivery: Optional[Any] = None

        async def _mutation(mutator: Any, _context: Any) -> None:
            nonlocal delivery
            self.assert_open()
            await mutator.commit(
                [
                    delete_value(session_name())
                    if name is None
                    else set_value(session_name(), name)
                ],
                context,
            )
            delivery = self.events.emit_batch(
                [{"type": "value_update", "value": "session_name", "name": name}], context
            )

        try:
            await self.session.mutate(_mutation, context)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if self._closed_error is not None:
                raise self._closed_error
            raise self.fault(error, context)
        if delivery is not None:
            await delivery

    async def get_label(self, target_id: str, context: Any) -> Optional[str]:
        self.assert_open()
        return await self.session.get_label(target_id, context)

    async def set_label(
        self, target_id: str, label: Optional[str], context: Any
    ) -> None:
        self.assert_open()
        delivery: Optional[Any] = None

        async def _mutation(mutator: Any, _context: Any) -> None:
            nonlocal delivery
            self.assert_open()
            address = entry_label(target_id)
            await mutator.commit(
                [
                    delete_value(address)
                    if label is None
                    else set_value(address, label)
                ],
                context,
            )
            delivery = self.events.emit_batch(
                [
                    {
                        "type": "value_update",
                        "value": "entry_label",
                        "targetId": target_id,
                        "label": label,
                    }
                ],
                context,
            )

        try:
            await self.session.mutate(_mutation, context)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if self._closed_error is not None:
                raise self._closed_error
            raise self.fault(error, context)
        if delivery is not None:
            await delivery

    # ------------------------------------------------------------------
    # Global configuration
    # ------------------------------------------------------------------

    async def get_tools(self, context: Any) -> List[Any]:
        return await self.get_config("tools", context)

    async def set_tools(self, tools: List[Any], context: Any) -> None:
        validate_tool_names(tools)
        await self.set_config(
            "tools", tools, lambda _previous, _value: {"type": "config_update", "property": "tools"}, context
        )

    async def get_resources(self, context: Any) -> Any:
        return await self.get_config("resources", context)

    async def set_resources(self, resources: Any, context: Any) -> None:
        await self.set_config(
            "resources",
            resources,
            lambda _previous, _value: {"type": "config_update", "property": "resources"},
            context,
        )

    async def get_stream_options(self, context: Any) -> Any:
        return await self.get_config("stream_options", context)

    async def set_stream_options(self, options: Any, context: Any) -> None:
        await self.set_config(
            "stream_options",
            options,
            lambda previous, value: {
                "type": "config_update",
                "property": "streamOptions",
                "previous": previous,
                "value": value,
            },
            context,
        )

    async def get_retry_policy(self, context: Any) -> RetryPolicy:
        return await self.get_config("retry_policy", context)

    async def set_retry_policy(self, policy: RetryPolicy, context: Any) -> None:
        validate_retry_policy(policy)
        await self.set_config(
            "retry_policy",
            policy,
            lambda previous, value: {
                "type": "config_update",
                "property": "retryPolicy",
                "previous": previous,
                "value": value,
            },
            context,
        )

    async def get_compaction_settings(self, context: Any) -> CompactionSettings:
        return await self.get_config("compaction", context)

    async def set_compaction_settings(
        self, compaction: CompactionSettings, context: Any
    ) -> None:
        validate_compaction_settings(compaction)
        await self.set_config(
            "compaction",
            compaction,
            lambda previous, value: {
                "type": "config_update",
                "property": "compactionSettings",
                "previous": previous,
                "value": value,
            },
            context,
        )

    async def get_steering_mode(self, context: Any) -> str:
        return await self.get_config("steering_mode", context)

    async def set_steering_mode(self, steering_mode: str, context: Any) -> None:
        await self.set_config(
            "steering_mode",
            steering_mode,
            lambda previous, value: {
                "type": "config_update",
                "property": "steeringMode",
                "previous": previous,
                "value": value,
            },
            context,
        )

    async def get_follow_up_mode(self, context: Any) -> str:
        return await self.get_config("follow_up_mode", context)

    async def set_follow_up_mode(self, follow_up_mode: str, context: Any) -> None:
        await self.set_config(
            "follow_up_mode",
            follow_up_mode,
            lambda previous, value: {
                "type": "config_update",
                "property": "followUpMode",
                "previous": previous,
                "value": value,
            },
            context,
        )

    async def watch_session(self, _context: Any) -> Any:
        raise SliceNotImplemented("watchSession")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def fault(self, cause: Any, context: Any) -> BaseException:
        if self._fault_error is not None:
            return self._fault_error
        if self._closed_error is not None:
            return self._closed_error
        normalized = cause if isinstance(cause, BaseException) else RuntimeError(str(cause))
        fault = HarnessFault("AgentHarness storage or invariant fault", normalized)
        self._fault_error = fault
        for lane in self.lanes_by_name.values():
            asyncio.ensure_future(lane.seal(fault))
        self.hooks.close(fault)
        asyncio.ensure_future(
            self.events.emit(
                {"type": "fault", "code": "harness_fault", "message": str(fault)}, context
            )
        )
        self.events.close(fault)
        return fault

    async def close(self, context: Any) -> None:
        if self._close_promise is not None:
            await self._close_promise
            return
        error = HarnessClosed()
        self._closed_error = error
        idle_callbacks = [lane.seal(error) for lane in self.lanes_by_name.values()]
        self.hooks.close(error)
        self.events.close(error)
        session_close = self.session.close(context)
        self._close_promise = asyncio.ensure_future(
            _settle(session_close, idle_callbacks)
        )
        await self._close_promise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def build_lane(self, name: str, state: LaneState) -> Lane:
        return Lane(
            name,
            self.session,
            self.models,
            self.hooks,
            state,
            lambda cause, context: self.fault(cause, context),
            lambda events, context: self.events.emit_batch(events, context),
            lambda snapshot, filter_fn, context, resnapshot: self.events.watch(
                snapshot, filter_fn, context, resnapshot
            ),
            lambda: self._config,
        )

    async def get_config(self, key: str, _context: Any) -> Any:
        self.assert_open()
        return getattr(self._config, key)

    async def set_config(
        self,
        key: str,
        value: Any,
        event: Any,
        context: Any,
    ) -> None:
        self.assert_open()
        previous = getattr(self._config, key)
        setattr(self._config, key, value)
        await self.events.emit(event(previous, value), context)

    def assert_open(self) -> None:
        if self._fault_error is not None:
            raise self._fault_error
        if self._closed_error is not None:
            raise self._closed_error


async def _settle(session_close: Any, idle_callbacks: List[Any]) -> None:
    await asyncio.gather(session_close, *idle_callbacks, return_exceptions=True)


async def create_agent_harness(options: Any, context: Any) -> Tuple[Harness, List[OpenOperation]]:
    """Attach runtime without starting provider, tool, hook, or timer effects."""
    tools = list(getattr(options, "tools", None) or [])
    validate_tool_names(tools)
    validate_retry_policy(getattr(options, "retry", None) or DEFAULT_RETRY_POLICY)
    validate_compaction_settings(
        getattr(options, "compaction", None) or DEFAULT_COMPACTION_SETTINGS
    )
    model = options.model
    active = getattr(options, "active_tool_names", None)
    seed = LaneConfiguration(
        model={"provider": model.provider, "modelId": model.id},
        thinking_level=getattr(options, "thinking_level", None) or "off",
        active_tool_names=list(active if active is not None else [tool.name for tool in tools]),
    )
    try:
        restored = await restore_session(options.session, context)
        open_operations: List[OpenOperation] = []
        for lane_name, state in restored.items():
            operation = state.operation
            if operation is None:
                continue
            open_operations.append(
                OpenOperation(
                    lane=lane_name,
                    operation_id=operation.meta.operation_id,
                    kind=operation.meta.intent.kind,
                    started_at=operation.meta.started_at,
                    aborting=True
                    if operation.state.control.status == "cancel_requested"
                    else None,
                )
            )
        return Harness(options, seed, restored), open_operations
    except asyncio.CancelledError:
        raise
    except BaseException as error:
        raise HarnessFault("AgentHarness storage or invariant fault", error) from error
