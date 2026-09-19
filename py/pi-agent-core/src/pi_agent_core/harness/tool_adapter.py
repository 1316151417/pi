"""Adapter bridging harness tools onto the agent-core ``AgentTool`` shape."""

from __future__ import annotations

from typing import Any, List, Optional

from .._chord.context import BACKGROUND_CONTEXT, Context, with_abort_signal
from ..types import AgentTool, AgentToolResult
from .env.local import LocalExecutionEnv
from .types import AgentHarnessTool, AgentHarnessToolInvocation, ExecutionToolContext

__all__ = [
    "harness_tool_to_agent_tool",
    "default_harness_tools",
    "default_agent_harness_tools",
    "StaticToolContext",
    "DefaultToolContextFactory",
]


class StaticToolContext(ExecutionToolContext):
    """ExecutionToolContext bound to one env and cwd.

    This is the tool context an ``AgentHarness`` lane passes to the built-in
    filesystem and shell tools.
    """

    def __init__(self, env: LocalExecutionEnv) -> None:
        super().__init__(env=env, cwd=env.cwd, description="")


def harness_tool_to_agent_tool(
    tool: AgentHarnessTool,
    env: LocalExecutionEnv,
    context: Optional[Context] = None,
) -> AgentTool:
    """Wrap a harness tool so the core agent loop can execute it.

    The harness ``execute`` signature is
    ``(tool_call_id, params, on_update, tool_context, invocation, context)``;
    the core loop calls ``(tool_call_id, params, signal, on_update)``. The
    adapter supplies a static tool context and derives a chord Context from the
    run's abort signal.
    """
    base_context = context or BACKGROUND_CONTEXT

    async def execute(tool_call_id: str, params: Any, signal: Any, on_update: Any) -> AgentToolResult:
        run_context = base_context
        if signal is not None:
            run_context = with_abort_signal(signal, base_context)
        tool_context = StaticToolContext(env)

        def _harness_on_update(partial_result: Any, options: Any = None) -> None:
            if on_update is not None:
                on_update(partial_result)

        return await tool.execute(
            tool_call_id,
            params,
            _harness_on_update,
            tool_context,
            None,
            run_context,
        )

    return AgentTool(
        name=tool.name,
        label=tool.label,
        description=tool.description,
        parameters=tool.parameters,
        execute=execute,
        prepare_arguments=tool.prepare_arguments,
        execution_mode=tool.execution_mode,
    )


def default_harness_tools(env: LocalExecutionEnv, context: Optional[Context] = None) -> List[AgentTool]:
    """Bash, read, write, and edit tools bound to ``env`` for the core agent."""
    from .tools import create_bash_tool, create_edit_tool, create_read_tool, create_write_tool

    harness_tools = [
        create_bash_tool(),
        create_read_tool(),
        create_write_tool(),
        create_edit_tool(),
    ]
    return [harness_tool_to_agent_tool(tool, env, context) for tool in harness_tools]


def default_agent_harness_tools() -> List[AgentHarnessTool]:
    """Bash, read, write, and edit tools in the ``AgentHarness`` tool contract.

    Unlike :func:`default_harness_tools`, these keep the six-argument harness
    ``execute`` signature the lane runtime calls, so the harness owns context
    resolution instead of the adapter.
    """
    from .tools import create_bash_tool, create_edit_tool, create_read_tool, create_write_tool

    return [
        create_bash_tool(),
        create_read_tool(),
        create_write_tool(),
        create_edit_tool(),
    ]
