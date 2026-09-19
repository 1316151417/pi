"""Write tool ported from ``harness/tools/write.ts``."""

from __future__ import annotations

from typing import Any, Optional

from pi_ai.types import TextContent
from ...types import AgentToolResult
from ..._chord.context import Context
from ..types import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    ExecutionToolContext,
    get_or_throw,
)
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path

__all__ = ["create_write_tool"]

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


def create_write_tool() -> AgentHarnessTool:
    async def execute(
        _tool_call_id: str,
        params: Any,
        _on_update: Any,
        tool_context: ExecutionToolContext,
        _invocation: Optional[AgentHarnessToolInvocation],
        context: Context,
    ) -> AgentToolResult:
        env = tool_context.env
        path = params["path"]
        content = params["content"]

        async def _mutate() -> AgentToolResult:
            if context.signal is not None and context.signal.aborted:
                raise RuntimeError("Operation aborted")
            get_or_throw(await env.write_file(await resolve_tool_path(env, path, context), content, context))
            if context.signal is not None and context.signal.aborted:
                raise RuntimeError("Operation aborted")
            return AgentToolResult(content=[TextContent(text=f"Successfully wrote to {path}")])

        return await with_file_mutation_queue(env, path, _mutate, context)

    return AgentHarnessTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        parameters=WRITE_SCHEMA,
        execute=execute,
    )
