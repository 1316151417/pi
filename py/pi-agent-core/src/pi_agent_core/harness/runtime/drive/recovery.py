"""Generation recovery ported from ``harness/runtime/drive/recovery.ts``.

Settles orphaned assistant requests from their bounded committed frame prefix
without another provider call, and synthetically settles cancelled effects.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from pi_ai.assistant_message_frame import reduce_assistant_message_frames
from pi_ai.types import AssistantMessage, Cost, Usage
from ...session.types import AssistantEffectPendingOperation
from ..progress import read_assistant_frames
from ..types import LaneReturn, ProcedureResult
from .response import publish_response

__all__ = [
    "ZERO_USAGE",
    "interrupted_assistant_message",
    "recover_assistant_generation",
    "recover_cancelled_assistant_effect",
]

#: Usage recorded for an interrupted request whose real outcome is unknown.
ZERO_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)

_INTERRUPTED_WARNING = (
    "Assistant request was interrupted. The preceding content is the latest committed "
    "partial; newer live output may be missing and the external outcome is unknown."
)


def interrupted_assistant_message(identity: Any, partial: Optional[AssistantMessage]) -> Any:
    """The synthetic settled message for an orphaned assistant request."""
    provider = _opt(identity, "provider")
    model_id = _opt(identity, "model_id", "modelId")
    if partial is None:
        return AssistantMessage(
            content=[],
            api="unknown",
            provider=provider,
            model=model_id,
            usage=ZERO_USAGE,
            stop_reason="error",
            error_message=_INTERRUPTED_WARNING,
            timestamp=int(time.time() * 1000),
        )
    resolved = partial.__class__(**vars(partial))
    resolved.usage = ZERO_USAGE
    resolved.stop_reason = "error"
    resolved.error_message = _INTERRUPTED_WARNING
    return resolved


async def recover_assistant_generation(lane: Any, drive: Any, generation: Any) -> ProcedureResult:
    """Settle an orphaned assistant request from its committed frame prefix."""

    async def _plan(_state: Any, _current: Any, _meta: Any, reader: Any) -> Any:
        return LaneReturn(
            result=await read_assistant_frames(
                reader, drive.operation_id, generation.response_entry_id, drive.context
            )
        )

    frames = await lane.continue_operation(generation, _plan, drive.context)
    if frames.kind == "cancel_requested":
        return ProcedureResult(kind="continue")

    message = interrupted_assistant_message(
        generation.generation_context.configuration.model,
        reduce_assistant_message_frames(frames.value),
    )
    await lane.emit_batch(
        [
            {
                "type": "message_start",
                "lane": lane.name,
                "runId": drive.operation_id,
                "message": message,
                "recovery": True,
            },
            {
                "type": "message_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "message": message,
                "entryId": generation.response_entry_id,
                "recovery": True,
            },
        ],
        drive.context,
    )
    return await publish_response(lane, drive, generation, message, {"recovery": True})


async def recover_cancelled_assistant_effect(lane: Any, drive: Any, effect: Any) -> ProcedureResult:
    """Synthetically settle one cancelled orphaned assistant or deferred effect."""

    async def _plan(_state: Any, _current: Any, _meta: Any, reader: Any) -> Any:
        return LaneReturn(
            result=await read_assistant_frames(
                reader, drive.operation_id, effect.response_entry_id, drive.context
            )
        )

    frames = await lane.settle_operation(effect, _plan, drive.context)
    identity = (
        effect.generation_context.configuration.model
        if effect.at == "assistant.effect_pending"
        else effect.configuration.model
    )
    message = interrupted_assistant_message(identity, reduce_assistant_message_frames(frames))
    await lane.emit_batch(
        [
            {
                "type": "message_start",
                "lane": lane.name,
                "runId": drive.operation_id,
                "message": message,
                "recovery": True,
            },
            {
                "type": "message_end",
                "lane": lane.name,
                "runId": drive.operation_id,
                "message": message,
                "entryId": effect.response_entry_id,
                "recovery": True,
            },
        ],
        drive.context,
    )
    return await publish_response(lane, drive, effect, message, {"recovery": True})


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping or an attribute."""
    if value is None:
        return None
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
