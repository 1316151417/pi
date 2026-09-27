"""Coding-agent write tool from ``core/tools/write.ts``."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from pi_agent_core.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_ai.abort import AbortSignal
from pi_ai.types import TextContent

from .definition import ToolDefinition, ToolExecutionContext, wrap_tool_definition
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_to_cwd

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}

class _WritePromptContribution(TypedDict):
    snippet: str
    guidelines: list[str]


WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION: _WritePromptContribution = {
    "snippet": "Create or overwrite files",
    "guidelines": ["Use write only for new files or complete rewrites."],
}


@dataclass
class WriteOperations:
    write_file: Callable[[str, str], Awaitable[None]]
    mkdir: Callable[[str], Awaitable[None]]


@dataclass
class WriteToolOptions:
    operations: WriteOperations | None = None


async def _local_write_file(path: str, content: str) -> None:
    write = asyncio.create_task(asyncio.to_thread(Path(path).write_text, content, encoding="utf-8"))
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        await write
        raise


async def _local_mkdir(directory: str) -> None:
    create = asyncio.create_task(asyncio.to_thread(Path(directory).mkdir, parents=True, exist_ok=True))
    try:
        await asyncio.shield(create)
    except asyncio.CancelledError:
        await create
        raise


_LOCAL_OPERATIONS = WriteOperations(_local_write_file, _local_mkdir)


def create_write_tool_definition(
    cwd: str, options: WriteToolOptions | None = None,
) -> ToolDefinition:
    operations = options.operations if options is not None and options.operations is not None else _LOCAL_OPERATIONS

    async def execute(
        _tool_call_id: str,
        params: object,
        signal: AbortSignal | None = None,
        _on_update: AgentToolUpdateCallback | None = None,
        context: ToolExecutionContext | None = None,
    ) -> AgentToolResult:
        data = cast(Mapping[str, object], params)
        path = cast(str, data["path"])
        content = cast(str, data["content"])
        absolute_path = resolve_to_cwd(path, context.cwd if context is not None and context.cwd else cwd)
        directory = os.path.dirname(absolute_path)

        async def mutate() -> AgentToolResult:
            def throw_if_aborted() -> None:
                if signal is not None and signal.aborted:
                    raise RuntimeError("Operation aborted")

            throw_if_aborted()
            await operations.mkdir(directory)
            throw_if_aborted()
            await operations.write_file(absolute_path, content)
            throw_if_aborted()
            return AgentToolResult(content=[TextContent(text=f"Successfully wrote to {path}")])

        return await with_file_mutation_queue(absolute_path, mutate)

    return ToolDefinition(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        parameters=WRITE_SCHEMA,
        constrained_sampling={"type": "json_schema", "strict": "prefer"},
        prompt_snippet=WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        prompt_guidelines=list(WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
        execute=execute,
    )


def create_write_tool(cwd: str, options: WriteToolOptions | None = None) -> AgentTool:
    return wrap_tool_definition(create_write_tool_definition(cwd, options))


__all__ = [
    "WRITE_SCHEMA", "WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION", "WriteOperations",
    "WriteToolOptions", "create_write_tool_definition", "create_write_tool",
]
