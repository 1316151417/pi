"""Shared coding-agent tool primitives."""

from dataclasses import dataclass
from typing import Literal

from pi_agent_core.types import AgentTool

from pi_agent_core.harness.utils.truncate import (
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH,
    TruncationOptions, TruncationResult, format_size, truncate_head,
    truncate_line, truncate_tail,
)

from .file_mutation_queue import with_file_mutation_queue
from .definition import ToolDefinition, ToolExecutionContext, wrap_tool_definition
from .path_utils import (
    expand_path, path_exists, resolve_read_path, resolve_read_path_async,
    resolve_to_cwd,
)
from .write import WriteOperations, WriteToolOptions, create_write_tool, create_write_tool_definition
from .read import ReadOperations, ReadToolOptions, create_read_tool, create_read_tool_definition
from .edit import EditOperations, EditToolOptions, create_edit_tool, create_edit_tool_definition
from .bash import (
    BashOperations, BashToolOptions, create_bash_tool,
    create_bash_tool_definition, create_local_bash_operations,
)
from .output_accumulator import OutputAccumulator, OutputAccumulatorOptions, OutputSnapshot

type ToolName = Literal["read", "bash", "edit", "write"]
ALL_TOOL_NAMES: frozenset[ToolName] = frozenset(("read", "bash", "edit", "write"))


@dataclass
class ToolsOptions:
    read: ReadToolOptions | None = None
    bash: BashToolOptions | None = None
    edit: EditToolOptions | None = None
    write: WriteToolOptions | None = None


def create_tool_definition(name: ToolName, cwd: str, options: ToolsOptions | None = None) -> ToolDefinition:
    chosen = options or ToolsOptions()
    if name == "read":
        return create_read_tool_definition(cwd, chosen.read)
    if name == "bash":
        return create_bash_tool_definition(cwd, chosen.bash)
    if name == "edit":
        return create_edit_tool_definition(cwd, chosen.edit)
    if name == "write":
        return create_write_tool_definition(cwd, chosen.write)
    raise ValueError(f"Unknown tool name: {name}")


def create_tool(name: ToolName, cwd: str, options: ToolsOptions | None = None) -> AgentTool:
    chosen = options or ToolsOptions()
    if name == "read":
        return create_read_tool(cwd, chosen.read)
    if name == "bash":
        return create_bash_tool(cwd, chosen.bash)
    if name == "edit":
        return create_edit_tool(cwd, chosen.edit)
    if name == "write":
        return create_write_tool(cwd, chosen.write)
    raise ValueError(f"Unknown tool name: {name}")


def create_coding_tool_definitions(cwd: str, options: ToolsOptions | None = None) -> list[ToolDefinition]:
    return [create_tool_definition(name, cwd, options) for name in ("read", "bash", "edit", "write")]


def create_coding_tools(cwd: str, options: ToolsOptions | None = None) -> list[AgentTool]:
    return [create_tool(name, cwd, options) for name in ("read", "bash", "edit", "write")]

__all__ = [
    "DEFAULT_MAX_BYTES", "DEFAULT_MAX_LINES", "GREP_MAX_LINE_LENGTH",
    "TruncationOptions", "TruncationResult", "format_size", "truncate_head",
    "truncate_line", "truncate_tail", "with_file_mutation_queue",
    "expand_path", "path_exists", "resolve_read_path",
    "resolve_read_path_async", "resolve_to_cwd",
    "ToolDefinition", "ToolExecutionContext", "wrap_tool_definition",
    "WriteOperations", "WriteToolOptions", "create_write_tool", "create_write_tool_definition",
    "ReadOperations", "ReadToolOptions", "create_read_tool", "create_read_tool_definition",
    "EditOperations", "EditToolOptions", "create_edit_tool", "create_edit_tool_definition",
    "BashOperations", "BashToolOptions", "create_bash_tool", "create_bash_tool_definition",
    "create_local_bash_operations", "OutputAccumulator", "OutputAccumulatorOptions",
    "OutputSnapshot", "ToolName", "ALL_TOOL_NAMES", "ToolsOptions",
    "create_tool_definition", "create_tool", "create_coding_tool_definitions", "create_coding_tools",
]
