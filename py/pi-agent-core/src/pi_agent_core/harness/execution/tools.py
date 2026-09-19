"""Tool execution pipeline ported from ``harness/execution/tools.ts``.

Splits one tool call into effect-free phases so durable state can be published
between them: resolve+prepare arguments, apply the before-tool decision,
execute behind the drive gate, patch with the after-tool decision, and finally
produce the transcript tool-result message.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from ..._chord.context import Context, with_abort_signal
from pi_ai.types import TextContent, ToolResultMessage
from pi_ai.validation import ValidationError, validate_tool_arguments
from ..types import AgentHarnessTool, AgentHarnessToolInvocation, AgentHarnessToolUpdateCallback

__all__ = [
    "PreparedToolCall",
    "ImmediateToolOutcome",
    "BeforeToolDecision",
    "ClearedToolCall",
    "ExecutedToolCall",
    "AfterToolPatch",
    "FinalizedToolCall",
    "create_error_tool_result",
    "immediate_error",
    "prepare_tool_call",
    "apply_before_tool_decision",
    "execute_tool_call",
    "finalize_tool_call",
    "tool_result_from_message",
    "create_tool_result_message",
]


def create_error_tool_result(message: str) -> Any:
    """Build an error result without crossing the external tool-effect boundary."""
    from ...types import AgentToolResult

    return AgentToolResult(content=[TextContent(text=message)])


def _new_result(content: List[Any], details: Any = None, usage: Any = None, terminate: Any = None) -> Any:
    from ...types import AgentToolResult

    return AgentToolResult(content=content, details=details, usage=usage, terminate=terminate)


@dataclass
class PreparedToolCall:
    """A tool call whose tool exists and whose prepared arguments passed validation."""

    tool_call: Any
    tool: AgentHarnessTool
    args: Any


@dataclass
class ImmediateToolOutcome:
    """Synthetic result produced without crossing the external tool-effect boundary."""

    tool_call: Any
    result: Any
    is_error: bool = True
    terminate: bool = False
    kind: str = "immediate"


@dataclass
class BeforeToolDecision:
    """Aggregated decision from the before-tool hook pipeline."""

    args: Any = None
    block: Optional[dict] = None  # {"reason": str, "terminate": bool}


@dataclass
class ClearedToolCall:
    """A prepared call cleared for durable intent publication and execution."""

    tool_call: Any
    tool: AgentHarnessTool
    args: Any


@dataclass
class ExecutedToolCall:
    """Raw phase-two tool output before after-tool patching."""

    result: Any
    is_error: bool


@dataclass
class AfterToolPatch:
    """Aggregated patch from the after-tool hook pipeline."""

    content: Any = None
    details: Any = None
    is_error: Optional[bool] = None
    usage: Any = None
    terminate: Optional[bool] = None


@dataclass
class FinalizedToolCall:
    """Final tool output ready to become a durable tool-result message."""

    tool_call: Any
    result: Any
    is_error: bool
    terminate: bool


def immediate_error(tool_call: Any, message: str, terminate: bool = False) -> ImmediateToolOutcome:
    return ImmediateToolOutcome(
        tool_call=tool_call,
        result=create_error_tool_result(message),
        is_error=True,
        terminate=terminate,
    )


def prepare_tool_call(call: Any, tools: List[AgentHarnessTool]) -> Any:
    """Resolve a tool, apply its deterministic argument preparation, and validate the result."""
    tool = next((candidate for candidate in tools if candidate.name == call.name), None)
    if tool is None:
        return immediate_error(call, f"Tool {call.name!r} is unavailable")

    try:
        prepared_arguments = (
            tool.prepare_arguments(call.arguments) if tool.prepare_arguments is not None else call.arguments
        )
        prepared_call = call
        if prepared_arguments is not call.arguments:
            import copy

            prepared_call = copy.copy(call)
            prepared_call.arguments = prepared_arguments
        args = validate_tool_arguments(tool, prepared_call)
        return PreparedToolCall(tool_call=call, tool=tool, args=args)
    except (ValidationError, Exception) as error:  # noqa: BLE001 - expected tool validation failures
        return immediate_error(call, str(error))


def apply_before_tool_decision(prepared: PreparedToolCall, decision: Optional[BeforeToolDecision]) -> Any:
    """Apply an explicit hook decision and revalidate replacement arguments."""
    if decision is not None and decision.block:
        return immediate_error(
            prepared.tool_call,
            decision.block.get("reason", ""),
            decision.block.get("terminate") is True,
        )

    if decision is None or decision.args is None:
        return ClearedToolCall(tool_call=prepared.tool_call, tool=prepared.tool, args=prepared.args)

    try:
        import copy

        candidate = copy.copy(prepared.tool_call)
        candidate.arguments = decision.args
        validated_args = validate_tool_arguments(prepared.tool, candidate)
        return ClearedToolCall(tool_call=prepared.tool_call, tool=prepared.tool, args=validated_args)
    except (ValidationError, Exception) as error:  # noqa: BLE001
        return immediate_error(prepared.tool_call, str(error))


async def execute_tool_call(
    call: ClearedToolCall,
    gate: Any,
    on_update: AgentHarnessToolUpdateCallback,
    tool_context: Any,
    invocation: Optional[AgentHarnessToolInvocation],
    context: Context,
) -> ExecutedToolCall:
    """Execute one cleared external tool effect, converting expected tool throws to error output."""
    accepting_updates = True

    async def _invoke() -> ExecutedToolCall:
        nonlocal accepting_updates
        admitted_context = with_abort_signal(gate.signal, context)
        if admitted_context.signal is not None:
            admitted_context.signal.throw_if_aborted()

        def _on_update(partial: Any, options: Any = None) -> None:
            if accepting_updates and on_update is not None:
                on_update(partial, options)

        try:
            result = await call.tool.execute(
                call.tool_call.id,
                call.args,
                _on_update,
                tool_context,
                invocation,
                admitted_context,
            )
            return ExecutedToolCall(result=result, is_error=False)
        except Exception as error:  # noqa: BLE001 - expected tool failures become error output
            return ExecutedToolCall(result=create_error_tool_result(str(error)), is_error=True)
        finally:
            accepting_updates = False

    return await gate.admit(_invoke)


def finalize_tool_call(
    call: ClearedToolCall, executed: ExecutedToolCall, patch: Optional[AfterToolPatch]
) -> FinalizedToolCall:
    """Apply an after-tool patch field by field."""
    if patch is None:
        result = executed.result
    else:
        result = _new_result(
            content=executed.result.content if patch.content is None else patch.content,
            details=executed.result.details if patch.details is None else patch.details,
            usage=executed.result.usage if patch.usage is None else patch.usage,
            terminate=executed.result.terminate if patch.terminate is None else patch.terminate,
        )
    return FinalizedToolCall(
        tool_call=call.tool_call,
        result=result,
        is_error=executed.is_error if (patch is None or patch.is_error is None) else patch.is_error,
        terminate=result.terminate is True,
    )


def tool_result_from_message(message: ToolResultMessage, terminate: bool) -> Any:
    """Reconstruct the canonical tool result represented by a staged transcript message."""
    return _new_result(
        content=list(message.content),
        details=message.details,
        usage=message.usage,
        terminate=True if terminate else None,
    )


def create_tool_result_message(call: FinalizedToolCall) -> ToolResultMessage:
    """Convert finalized tool output to the provider-facing transcript message."""
    return ToolResultMessage(
        tool_call_id=call.tool_call.id,
        tool_name=call.tool_call.name,
        content=call.result.content or [],
        details=call.result.details,
        usage=call.result.usage,
        is_error=call.is_error,
        timestamp=int(time.time() * 1000),
    )
