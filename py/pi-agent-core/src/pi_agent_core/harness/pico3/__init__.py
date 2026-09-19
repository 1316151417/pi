"""Experimental Pico3 kernel API, mirroring ``harness/pico3/index.ts``.

This subpath is intentionally separate from the package root while the kernel
and its Chord integration are being validated.

Every TypeScript export keeps its name in ``snake_case`` form (``bashTool`` ->
``bash_tool``, ``defineSystemSection`` -> ``define_system_section``); only
durable JSON keys stay camelCase.

The ``kinds`` name is the built-in-kind map from ``harness.ts``, exactly as the
TypeScript barrel exports it; the ``kinds`` subpackage stays importable by path
(``harness.pico3.kinds.tool``).
"""

from __future__ import annotations

from .bash import BashTool, bash_tool
from .bounded import Bounded
from .context import derive_context, reorder_tool_results
from .delta import (
    Op,
    Path,
    PathError,
    Tracker,
    apply,
    apply_immutable,
    is_base,
    is_replace,
    op_append,
    op_delete,
    op_replace,
    op_set,
    op_splice,
    op_truncate,
    track,
)
from .hooks import HookRegistration, HookRunnerImpl, create_hook_runners
from .jsonl import JsonlStorage
from .memory import MemoryStorage
from .membrane import Membrane, PlainInputError, RevokedError
from .session import (
    CALLBACK_TX_METHODS,
    CORE_CONFIG_VALIDATORS,
    CORE_KINDS,
    LINE_KEY,
    STICKY_BASE_BUDGET,
    CommitChanges,
    CommitResult,
    ConversationIndex,
    Defaults,
    DocChange,
    Docs,
    NestedLineOperation,
    ScopedEvent,
    Session,
    TransactionControl,
    TransactionControlImpl,
    TxImpl,
    exact_object,
    finite_nonnegative,
    is_core_kind,
    plain,
    remove_where,
    validate_entry,
)
from .scheduler import Scheduler, SchedulerDeps, handler_for, validate_step
from .system import (
    DEFAULT_RETRY_POLICY,
    Canonical,
    Draft,
    EnvironmentInfo,
    FoldResult,
    ManagedEntryPlan,
    PreparationSnapshot,
    PrepareDraftResult,
    SectionRecord,
    SectionRegistry,
    SectionSeed,
    SectionState,
    SkillInfo,
    SystemEntryData,
    SystemSection,
    SystemSectionDraft,
    TakeSnapshotResult,
    ToolRegistry,
    define_system_section,
    effective_tools,
    fold_canonical,
    plan_managed_entry,
    prepare_draft,
    remove_section,
    same_snapshot,
    section_seed,
    system_sections,
    take_snapshot,
)
from .types import *  # noqa: F401,F403 - the barrel mirrors types.ts
from .types import __all__ as _types_all
from .view import WATCH_CAPACITY, ViewManager, Watch, WatchImpl, apply_envelope, touches_key
from .harness import (
    BUILTIN_KINDS,
    ConversationConfig,
    ConversationHandle,
    Harness,
    HarnessOptions,
    InputHandle,
    capture_active_transcript,
)
from .harness import entries as _harness_entries
from .harness import kinds
from .chord import (
    ChordViewBridge,
    ChordViewBridgeOptions,
    MutableReplicatedState,
    PicoConversationService,
    PublishedConversationView,
    ReplicatedState,
    ServiceDefinition,
    apply_tracked,
    attach_chord_view,
    create_pico_conversation_service,
    define_service,
    pico_conversation_service,
    pico_harness_service,
)
from .kinds.collapse import (
    CollapseCheckpoint,
    CollapseFailure,
    CollapseHooks,
    CollapseInput,
    choose_through,
    collapse,
    collapse_kind,
)
from .kinds.entries import entries
from .kinds.generation import (
    DisplayAssistantData,
    GenerationCheckpoint,
    GenerationFailure,
    GenerationHooks,
    GenerationInput,
    GenerationResult,
    empty_usage,
    generation,
    generation_kind,
    retry_decision,
)
from .kinds.job import JobCheckpoint, JobFailure, JobInput, JobOutput, JobResult, job, job_kind
from .kinds.plugin import PluginCheckpoint, PluginFailure, PluginInput, plugin, plugin_kind
from .kinds.post_tools import (
    PostToolsHooks,
    PostToolsInput,
    PostToolsResult,
    post_tools,
    post_tools_kind,
)
from .kinds.task_api import TaskApi, task_api
from .kinds.tool import (
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

_ = _harness_entries

__all__ = [
    # bash
    "BashTool",
    "bash_tool",
    # bounded
    "Bounded",
    # context
    "derive_context",
    "reorder_tool_results",
    # delta: the chord/delta essentials, shipped locally by this port
    "Op",
    "Path",
    "PathError",
    "Tracker",
    "apply",
    "apply_immutable",
    "is_base",
    "is_replace",
    "op_append",
    "op_delete",
    "op_replace",
    "op_set",
    "op_splice",
    "op_truncate",
    "track",
    # hooks
    "HookRegistration",
    "HookRunnerImpl",
    "create_hook_runners",
    # storage
    "JsonlStorage",
    "MemoryStorage",
    # membrane
    "Membrane",
    "PlainInputError",
    "RevokedError",
    # session: the line and its capability-checked transaction
    "CALLBACK_TX_METHODS",
    "STICKY_BASE_BUDGET",
    "Session",
    "TransactionControlImpl",
    "TxImpl",
    # scheduler
    "Scheduler",
    "SchedulerDeps",
    "handler_for",
    "validate_step",
    # harness
    "BUILTIN_KINDS",
    "ConversationConfig",
    "ConversationHandle",
    "Harness",
    "HarnessOptions",
    "InputHandle",
    "capture_active_transcript",
    # chord
    "ChordViewBridge",
    "ChordViewBridgeOptions",
    "MutableReplicatedState",
    "PicoConversationService",
    "PublishedConversationView",
    "ReplicatedState",
    "ServiceDefinition",
    "apply_tracked",
    "attach_chord_view",
    "create_pico_conversation_service",
    "define_service",
    "pico_conversation_service",
    "pico_harness_service",
    # kinds: the four remaining built-ins
    "CollapseCheckpoint",
    "CollapseFailure",
    "CollapseHooks",
    "CollapseInput",
    "choose_through",
    "collapse",
    "collapse_kind",
    "DisplayAssistantData",
    "GenerationCheckpoint",
    "GenerationFailure",
    "GenerationHooks",
    "GenerationInput",
    "GenerationResult",
    "empty_usage",
    "generation",
    "generation_kind",
    "retry_decision",
    "PostToolsHooks",
    "PostToolsInput",
    "PostToolsResult",
    "post_tools",
    "post_tools_kind",
    "DEFAULT_BOUNDS",
    "ToolCheckpoint",
    "ToolHooks",
    "ToolInput",
    "ToolTaskResult",
    "invalid",
    "same_identity",
    "tool",
    "tool_kind",
    # session, the primitives that do not need the line
    "CORE_CONFIG_VALIDATORS",
    "CORE_KINDS",
    "LINE_KEY",
    "CommitChanges",
    "CommitResult",
    "ConversationIndex",
    "Defaults",
    "DocChange",
    "Docs",
    "NestedLineOperation",
    "ScopedEvent",
    "TransactionControl",
    "exact_object",
    "finite_nonnegative",
    "is_core_kind",
    "plain",
    "remove_where",
    "validate_entry",
    # system
    "DEFAULT_RETRY_POLICY",
    "Canonical",
    "Draft",
    "EnvironmentInfo",
    "FoldResult",
    "ManagedEntryPlan",
    "PreparationSnapshot",
    "PrepareDraftResult",
    "SectionRecord",
    "SectionRegistry",
    "SectionSeed",
    "SectionState",
    "SkillInfo",
    "SystemEntryData",
    "SystemSection",
    "SystemSectionDraft",
    "TakeSnapshotResult",
    "ToolRegistry",
    "define_system_section",
    "effective_tools",
    "fold_canonical",
    "plan_managed_entry",
    "prepare_draft",
    "remove_section",
    "same_snapshot",
    "section_seed",
    "system_sections",
    "take_snapshot",
    # view
    "WATCH_CAPACITY",
    "ViewManager",
    "Watch",
    "WatchImpl",
    "apply_envelope",
    "touches_key",
    # kinds
    "entries",
    "kinds",
    "job_kind",
    "plugin_kind",
    "job",
    "JobCheckpoint",
    "JobFailure",
    "JobInput",
    "JobOutput",
    "JobResult",
    "plugin",
    "PluginCheckpoint",
    "PluginFailure",
    "PluginInput",
    "TaskApi",
    "task_api",
    # types.ts
    *_types_all,
]
