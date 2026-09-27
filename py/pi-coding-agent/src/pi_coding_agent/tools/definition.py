"""Core tool-definition bridge from ``tool-definition-wrapper.ts``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pi_agent_core.types import AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolExecutionMode
from pi_ai.abort import AbortSignal
from pi_ai.types import JsonObject, Model


@dataclass
class ToolExecutionContext:
    cwd: str
    model: Model | None = None
    session_id: str | None = None
    session_file: str | None = None
    thinking_level: str | None = None


type ToolExecute = Callable[
    [str, object, AbortSignal | None, AgentToolUpdateCallback | None, ToolExecutionContext | None],
    Awaitable[AgentToolResult],
]


@dataclass
class ToolDefinition:
    name: str
    label: str
    description: str
    parameters: JsonObject
    execute: ToolExecute
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] = field(default_factory=list)
    constrained_sampling: JsonObject | None = None
    prepare_arguments: Callable[[object], object] | None = None
    execution_mode: ToolExecutionMode | None = None


def wrap_tool_definition(
    definition: ToolDefinition,
    context_factory: Callable[[], ToolExecutionContext] | None = None,
) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: object,
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        context = context_factory() if context_factory is not None else None
        return await definition.execute(tool_call_id, params, signal, on_update, context)

    return AgentTool(
        name=definition.name,
        label=definition.label,
        description=definition.description,
        parameters=definition.parameters,
        constrained_sampling=definition.constrained_sampling,
        prepare_arguments=definition.prepare_arguments,
        execution_mode=definition.execution_mode,
        execute=execute,
    )


__all__ = ["ToolExecutionContext", "ToolExecute", "ToolDefinition", "wrap_tool_definition"]
