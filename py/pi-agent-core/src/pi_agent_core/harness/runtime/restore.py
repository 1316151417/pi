"""Lane storage restoration ported from ``harness/runtime/restore.ts``.

Reads the durable lane values (branch tip, lane config, lane state) and
reconstructs the in-process :class:`LaneState`, validating that any recorded
operation still matches its intent before it is handed to the lane runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..._chord.context import Context
from ..session.session import SessionInvariantError
from ..session.types import (
    InboxItem,
    Operation,
    revive_operation_meta,
    revive_operation_state,
)
from ..session.values import (
    branch_tip,
    branch_tip_inventory_prefix,
    lane_config,
    operation_meta,
    operation_state,
)
from ..session.values import lane_state as lane_state_value
from .types import LaneState

__all__ = [
    "ClassifiedLaneStorage",
    "read_lane_storage",
    "read_lane",
    "restore_lane",
    "restore_session",
    "restore_lane_state",
    "state_matches_intent",
]


@dataclass
class ClassifiedLaneStorage:
    """Durable lane storage classified as absent, branch-only, or fully configured."""

    kind: str  # "absent" | "branch" | "lane"
    tip: Any = None
    configuration: Any = None
    lane_state: Any = None



def _field(value: Any, snake: str, camel: str, default: Any = None) -> Any:
    """Read a field from either a dataclass instance or a JSON-replayed mapping.

    Durable values round-trip through JSONL as plain camelCase mappings, while
    same-process writes keep the dataclass. Both shapes must restore identically.
    """
    if value is None:
        return default
    if isinstance(value, dict):
        if camel in value:
            return value[camel]
        if snake in value:
            return value[snake]
        return default
    if hasattr(value, snake):
        return getattr(value, snake)
    return default


def _normalize_lane_configuration(value: Any) -> Any:
    """Return a LaneConfiguration dataclass from a stored value."""
    from ..session.types import LaneConfiguration

    if value is None:
        return None
    if isinstance(value, LaneConfiguration):
        return value
    model = _field(value, "model", "model")
    return LaneConfiguration(
        model=dict(model or {}),
        thinking_level=_field(value, "thinking_level", "thinkingLevel", "off"),
        active_tool_names=list(_field(value, "active_tool_names", "activeToolNames", []) or []),
    )


def _is_summary_state(state: Any) -> bool:
    return str(getattr(state, "at", "")).startswith("summary.")


def state_matches_intent(intent: Any, state: Any) -> bool:
    """Whether a recorded operation state is consistent with its intent."""
    kind = getattr(intent, "kind", None)
    if kind == "compaction":
        return _is_summary_state(state) and getattr(state.task.boundary, "kind", None) == "finish"
    if kind == "navigation":
        if getattr(state, "at", None) == "navigation.ready_to_commit":
            return (
                not getattr(intent, "summarize", False)
                and getattr(state, "target_id", None) == getattr(intent, "target_id", None)
                and getattr(state, "label", None) == getattr(intent, "label", None)
            )
        return (
            bool(getattr(intent, "summarize", False))
            and _is_summary_state(state)
            and getattr(state.task.boundary, "kind", None) == "commit_navigation"
            and getattr(state.task.boundary, "target_id", None) == getattr(intent, "target_id", None)
            and getattr(state.task.boundary, "label", None) == getattr(intent, "label", None)
            and getattr(state.task, "custom_instructions", None)
            == getattr(intent, "custom_instructions", None)
        )
    return getattr(state, "at", None) != "navigation.ready_to_commit" and (
        not _is_summary_state(state) or getattr(state.task.boundary, "kind", None) == "resume_checkpoint"
    )


def _classify_lane_storage(lane: str, tip: Any, configuration: Any, lane_state: Any) -> ClassifiedLaneStorage:
    if tip is None and configuration is None and lane_state is None:
        return ClassifiedLaneStorage(kind="absent")
    if tip is not None and configuration is None and lane_state is None:
        return ClassifiedLaneStorage(kind="branch", tip=tip)
    if tip is None:
        raise SessionInvariantError(f"Lane {lane!r} is missing branch.tip")
    if configuration is None:
        raise SessionInvariantError(f"Lane {lane!r} is missing lane.config")
    if lane_state is None:
        raise SessionInvariantError(f"Lane {lane!r} is missing lane.state")
    return ClassifiedLaneStorage(kind="lane", tip=tip, configuration=configuration, lane_state=lane_state)


async def read_lane_storage(reader: Any, lane: str, context: Context) -> ClassifiedLaneStorage:
    """Read the three durable lane values and classify their completeness."""
    tip = await reader.get_value(branch_tip(lane), context)
    configuration = await reader.get_value(lane_config(lane), context)
    lane_state = await reader.get_value(lane_state_value(lane), context)
    return _classify_lane_storage(lane, tip, configuration, lane_state)


async def restore_session(session: Any, context: Context) -> dict:
    """Restore every complete configured AgentLane in one coherent session read."""
    holder: dict = {}

    async def _mutation(reader, mutation_context):
        tips = await reader.scan_values(branch_tip_inventory_prefix(), mutation_context)
        configurations = await reader.scan_values(lane_config(""), mutation_context)
        states = await reader.scan_values(lane_state_value(""), mutation_context)

        tip_by_lane = {value.address.key: value for value in tips}
        configuration_by_lane = {value.address.key: value for value in configurations}
        state_by_lane = {value.address.key: value for value in states}
        names = set(tip_by_lane) | set(configuration_by_lane) | set(state_by_lane)

        restored: dict = {}
        for lane in names:
            stored = _classify_lane_storage(
                lane,
                tip_by_lane.get(lane),
                configuration_by_lane.get(lane),
                state_by_lane.get(lane),
            )
            if stored.kind != "lane":
                continue
            restored[lane] = await restore_lane_state(reader, lane, stored, mutation_context)
        holder["restored"] = restored
        return restored

    await session.mutate(_mutation, context)
    return holder.get("restored", {})


async def restore_lane(session: Any, lane: str, context: Context) -> LaneState:
    """Restore one configured lane without starting work or interpreting its state."""
    holder: dict = {}

    async def _mutation(reader, mutation_context):
        stored = await read_lane_storage(reader, lane, mutation_context)
        if stored.kind == "absent":
            raise SessionInvariantError(f"Lane {lane!r} is missing branch.tip")
        if stored.kind == "branch":
            raise SessionInvariantError(f"Lane {lane!r} is missing lane.config")
        state = await restore_lane_state(reader, lane, stored, mutation_context)
        holder["state"] = state
        return state

    await session.mutate(_mutation, context)
    return holder["state"]


async def restore_lane_state(
    reader: Any, lane: str, stored: ClassifiedLaneStorage, context: Context
) -> LaneState:
    """Rebuild the in-process lane state from classified durable storage."""
    lane_state_value_obj = stored.lane_state.value
    operation_id = _field(lane_state_value_obj, "current_operation_id", "currentOperationId")
    operation: Optional[Any] = None

    if operation_id is not None:
        meta = await reader.get_value(operation_meta(operation_id), context)
        state = await reader.get_value(operation_state(operation_id), context)
        if meta is None:
            raise SessionInvariantError(f"Operation {operation_id} is missing op.meta")
        if state is None:
            raise SessionInvariantError(f"Operation {operation_id} is missing op.state")
        meta_value = revive_operation_meta(meta.value)
        state_value = revive_operation_state(state.value)
        if meta_value.operation_id != operation_id:
            raise SessionInvariantError(
                f"Operation {operation_id} metadata names operation {meta_value.operation_id!r}"
            )
        if meta_value.lane != lane:
            raise SessionInvariantError(
                f"Operation {operation_id} belongs to lane {meta_value.lane!r}, not {lane!r}"
            )
        if not state_matches_intent(meta_value.intent, state_value):
            raise SessionInvariantError(
                f"Operation {operation_id} intent {meta_value.intent.kind} does not match state "
                f"{getattr(state_value, 'at', None)}"
            )
        operation = Operation(meta=meta_value, state=state_value)

    inbox = _field(lane_state_value_obj, "inbox", "inbox", []) or []
    return LaneState(
        tip_id=stored.tip.value,
        configuration=_normalize_lane_configuration(stored.configuration.value),
        inbox=[
            item
            if isinstance(item, InboxItem)
            else InboxItem(
                entry_id=_field(item, "entry_id", "entryId", ""),
                kind=_field(item, "kind", "kind", "write"),
            )
            for item in inbox
        ],
        last_operation_id=_field(lane_state_value_obj, "last_operation_id", "lastOperationId"),
        operation=operation,
    )
