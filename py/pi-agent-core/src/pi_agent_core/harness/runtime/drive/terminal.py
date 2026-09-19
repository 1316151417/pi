"""Operation terminal helpers ported from ``harness/runtime/drive/terminal.ts``.

Builds the mechanical operation-owned cleanup writes used by an owning
procedure's terminal transaction, plus the immutable result record for one
terminal decision.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from ...._chord.context import Context
from ...session.session import SessionInvariantError
from ...session.types import OperationError, OperationResultRecord, Write
from ...session.values import (
    delete_list,
    delete_value,
    operation_meta,
    operation_preparation_prefix,
    operation_state as operation_state_value,
    operation_tool_args_prefix,
    operation_tool_memo_prefix,
    pending_assistant_frames,
    pending_entry,
    pending_tool_output_prefix,
)

__all__ = ["operation_cleanup_writes", "operation_result_record"]


async def operation_cleanup_writes(
    reader: Any, operation_id: str, state: Any, context: Context
) -> List[Write]:
    """Build the mechanical operation-owned suffix used by a terminal transaction."""
    tool_arguments = await reader.scan_values(operation_tool_args_prefix(operation_id), context)
    tool_memos = await reader.scan_values(operation_tool_memo_prefix(operation_id), context)
    preparations = await reader.scan_values(operation_preparation_prefix(operation_id), context)
    tool_outputs = await reader.scan_values(pending_tool_output_prefix(operation_id), context)

    pending_ids: set = set()
    if getattr(state, "at", None) == "tools":
        batch = state.batch
        for call in batch.calls:
            if getattr(call, "status", None) == "outcome_ready":
                pending_ids.add(call.result_entry_id)

    frame_delete: Optional[Write] = None
    if getattr(state, "at", None) in ("assistant.effect_pending", "deferred.effect_pending"):
        frame_delete = delete_list(pending_assistant_frames(operation_id, state.response_entry_id))

    writes: List[Write] = [
        delete_value(operation_meta(operation_id)),
        delete_value(operation_state_value(operation_id)),
        *[delete_value(stored.address) for stored in tool_arguments],
        *[delete_value(stored.address) for stored in tool_memos],
        *[delete_value(stored.address) for stored in preparations],
        *[delete_value(stored.address) for stored in tool_outputs],
    ]
    if frame_delete is not None:
        writes.append(frame_delete)
    writes.extend(delete_value(pending_entry(pending_id)) for pending_id in pending_ids)
    return writes


def operation_result_record(
    meta: Any,
    status: str,
    tip_id: Optional[str],
    error: Optional[OperationError] = None,
) -> OperationResultRecord:
    """Construct the immutable observation record for one terminal decision."""
    if (status == "failed") != (error is not None):
        raise SessionInvariantError("Only a failed operation result may carry an error")
    return OperationResultRecord(
        operation_id=meta.operation_id,
        kind=meta.intent.kind,
        status=status,
        error=error,
        from_tip_id=meta.source_tip_id,
        tip_id=tip_id,
        started_at=meta.started_at,
        ended_at=int(time.time() * 1000),
    )
