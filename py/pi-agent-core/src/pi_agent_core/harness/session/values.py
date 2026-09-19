"""Value addresses ported from ``harness/session/values.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

__all__ = [
    "Value",
    "ValueList",
    "StoredValue",
    "ListElement",
    "ListCursor",
    "ListReadOptions",
    "ValueSetWrite",
    "ValueDeleteWrite",
    "ListAppendWrite",
    "ListDeleteWrite",
    "value",
    "list",
    "set_value",
    "delete_value",
    "append_list",
    "delete_list",
    "resolve_list_read_options",
    "branch_tip",
    "branch_tip_inventory_prefix",
    "lane_config",
    "lane_state",
    "operation_result",
    "operation_meta",
    "operation_state",
    "operation_tool_args",
    "operation_tool_memo",
    "operation_preparation",
    "operation_tool_args_prefix",
    "operation_tool_memo_prefix",
    "operation_preparation_prefix",
    "pending_entry",
    "pending_tool_output",
    "pending_tool_output_prefix",
    "pending_assistant_frames",
    "session_name",
    "entry_label",
]

T = TypeVar("T")


@dataclass(frozen=True)
class Value(Generic[T]):
    namespace: str
    key: str = ""
    kind: str = "value"


@dataclass(frozen=True)
class ValueList(Generic[T]):
    namespace: str
    key: str = ""
    kind: str = "list"


@dataclass
class StoredValue(Generic[T]):
    address: Value
    value: Any = None
    seq: int = 0


@dataclass
class ListElement(Generic[T]):
    seq: int
    value: Any = None


@dataclass
class ListCursor:
    seq: int


@dataclass
class ListReadOptions:
    cursor: Optional[ListCursor] = None
    order: str = "asc"  # "asc" | "desc"
    limit: Optional[int] = None


@dataclass
class ValueSetWrite:
    kind: str = "value"
    op: str = "set"
    namespace: str = ""
    key: str = ""
    value: Any = None


@dataclass
class ValueDeleteWrite:
    kind: str = "value"
    op: str = "delete"
    namespace: str = ""
    key: str = ""


@dataclass
class ListAppendWrite:
    kind: str = "list"
    op: str = "append"
    namespace: str = ""
    key: str = ""
    value: Any = None


@dataclass
class ListDeleteWrite:
    kind: str = "list"
    op: str = "delete"
    namespace: str = ""
    key: str = ""


def _validate_address(namespace: str, key: str) -> None:
    if len(namespace) == 0:
        raise TypeError("Value namespace must not be empty")
    if "\x00" in namespace:
        raise TypeError("Value namespace must not contain \\u0000")
    if "\x00" in key:
        raise TypeError("Value key must not contain \\u0000")


def value(namespace: str, key: str = "") -> Value:
    _validate_address(namespace, key)
    return Value(namespace=namespace, key=key)


def list(namespace: str, key: str = "") -> ValueList:  # noqa: A001 - mirrors TS naming
    _validate_address(namespace, key)
    return ValueList(namespace=namespace, key=key)


def set_value(address: Value, next_value: Any) -> ValueSetWrite:
    return ValueSetWrite(namespace=address.namespace, key=address.key, value=next_value)


def delete_value(address: Value) -> ValueDeleteWrite:
    return ValueDeleteWrite(namespace=address.namespace, key=address.key)


def append_list(address: ValueList, element: Any) -> ListAppendWrite:
    return ListAppendWrite(namespace=address.namespace, key=address.key, value=element)


def delete_list(address: ValueList) -> ListDeleteWrite:
    return ListDeleteWrite(namespace=address.namespace, key=address.key)


def resolve_list_read_options(options: Optional[ListReadOptions] = None) -> ListReadOptions:
    options = options or ListReadOptions()
    requested_limit = options.limit if options.limit is not None else 1000
    if not isinstance(requested_limit, int) or requested_limit <= 0:
        raise TypeError("List read limit must be a positive safe integer")
    return ListReadOptions(
        cursor=options.cursor,
        order=options.order or "asc",
        limit=min(requested_limit, 10000),
    )


# ---------------------------------------------------------------------------
# Predefined addresses
# ---------------------------------------------------------------------------


def branch_tip(branch: str) -> Value:
    return value("pi.branch.tip", branch)


def branch_tip_inventory_prefix() -> Value:
    return value("pi.branch.tip")


def lane_config(lane: str) -> Value:
    return value("pi.lane.config", lane)


def lane_state(lane: str) -> Value:
    return value("pi.lane.state", lane)


def operation_result(operation_id: str) -> Value:
    return value("pi.result", operation_id)


def operation_meta(operation_id: str) -> Value:
    return value("pi.op.meta", operation_id)


def operation_state(operation_id: str) -> Value:
    return value("pi.op.state", operation_id)


def operation_tool_args(operation_id: str, step_id: str, source_index: int) -> Value:
    return value("pi.op.tool_args", f"{operation_id}:{step_id}:{source_index}")


def operation_tool_memo(operation_id: str, invocation_id: str, name: str) -> Value:
    return value("pi.op.tool_memo", f"{operation_id}:{invocation_id}:{name}")


def operation_preparation(operation_id: str, task_id: str) -> Value:
    return value("pi.op.preparation", f"{operation_id}:{task_id}")


def operation_tool_args_prefix(operation_id: str, step_id: Optional[str] = None) -> Value:
    return value(
        "pi.op.tool_args",
        f"{operation_id}:" if step_id is None else f"{operation_id}:{step_id}:",
    )


def operation_tool_memo_prefix(operation_id: str, invocation_id: Optional[str] = None) -> Value:
    return value(
        "pi.op.tool_memo",
        f"{operation_id}:" if invocation_id is None else f"{operation_id}:{invocation_id}:",
    )


def operation_preparation_prefix(operation_id: str) -> Value:
    return value("pi.op.preparation", f"{operation_id}:")


def pending_entry(entry_id: str) -> Value:
    return value("pi.pending.entry", entry_id)


def pending_tool_output(operation_id: str, invocation_id: str) -> Value:
    return value("pi.pending.tool_output", f"{operation_id}:{invocation_id}")


def pending_tool_output_prefix(operation_id: str) -> Value:
    return value("pi.pending.tool_output", f"{operation_id}:")


def session_name() -> Value:
    return value("pi.session.name")


def entry_label(entry_id: str) -> Value:
    return value("pi.entry.label", entry_id)

def pending_assistant_frames(operation_id: str, response_entry_id: str) -> ValueList:
    """Durable list of assistant message frames streamed for one response entry."""
    return list("pi.pending.assistant_frame", f"{operation_id}:{response_entry_id}")
