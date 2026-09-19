"""Pico3 built-in kinds.

The TypeScript sources for this subpackage ship no barrel, so this module is
this port's aggregate: the six built-in kind tokens plus their module surfaces.
"""

from __future__ import annotations

from .collapse import (
    CollapseCheckpoint,
    CollapseFailure,
    CollapseHooks,
    CollapseInput,
    choose_through,
    collapse,
    collapse_config,
    collapse_kind,
)
from .entries import EntryKinds, core_entry, entries
from .frames import apply_frame, safe_parse
from .generation import (
    DisplayAssistantData,
    GenerationCheckpoint,
    GenerationFailure,
    GenerationHooks,
    GenerationInput,
    GenerationResult,
    empty_usage,
    generation,
    generation_config,
    generation_kind,
    retry_decision,
)
from .job import JobCheckpoint, JobFailure, JobInput, JobOutput, JobResult, job, job_kind
from .plugin import PluginCheckpoint, PluginFailure, PluginInput, plugin, plugin_kind
from .post_tools import (
    PostToolsHooks,
    PostToolsInput,
    PostToolsResult,
    post_tools,
    post_tools_config,
    post_tools_kind,
)
from .task_api import TaskApi
from .tool import (
    DEFAULT_BOUNDS,
    ToolCheckpoint,
    ToolHooks,
    ToolInput,
    ToolTaskResult,
    invalid,
    same_identity,
    tool,
    tool_kind,
)

__all__ = [
    # collapse
    "CollapseCheckpoint",
    "CollapseFailure",
    "CollapseHooks",
    "CollapseInput",
    "choose_through",
    "collapse",
    "collapse_config",
    "collapse_kind",
    # entries
    "EntryKinds",
    "core_entry",
    "entries",
    # frames
    "apply_frame",
    "safe_parse",
    # generation
    "DisplayAssistantData",
    "GenerationCheckpoint",
    "GenerationFailure",
    "GenerationHooks",
    "GenerationInput",
    "GenerationResult",
    "empty_usage",
    "generation",
    "generation_config",
    "generation_kind",
    "retry_decision",
    # job
    "JobCheckpoint",
    "JobFailure",
    "JobInput",
    "JobOutput",
    "JobResult",
    "job",
    "job_kind",
    # plugin
    "PluginCheckpoint",
    "PluginFailure",
    "PluginInput",
    "plugin",
    "plugin_kind",
    # post-tools
    "PostToolsHooks",
    "PostToolsInput",
    "PostToolsResult",
    "post_tools",
    "post_tools_config",
    "post_tools_kind",
    # task-api: the module stays importable as ``kinds.task_api``
    "TaskApi",
    # tool
    "DEFAULT_BOUNDS",
    "ToolCheckpoint",
    "ToolHooks",
    "ToolInput",
    "ToolTaskResult",
    "invalid",
    "same_identity",
    "tool",
    "tool_kind",
]
