"""Core coding-agent presentation messages."""

from .messages import (
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX, BashExecutionMessage, BranchSummaryMessage,
    CompactionSummaryMessage, CustomMessage, bash_execution_to_text,
    convert_to_llm, create_branch_summary_message,
    create_compaction_summary_message, create_custom_message,
)
from .defaults import DEFAULT_THINKING_LEVEL, THINKING_LEVEL_OPTIONS
from .skills import Skill, format_skills_for_prompt
from .system_prompt import (
    BuildSystemPromptOptions, NormalizedBuildSystemPromptOptions, SystemPromptSections,
    SystemPromptState, build_system_prompt, build_system_prompt_sections,
    build_system_prompt_state, diff_system_prompt_sections,
    normalize_build_system_prompt_options,
)
from .bash_executor import BashExecutorOptions, BashResult, execute_bash_with_operations
from .session_entries import (
    CURRENT_SESSION_VERSION, SessionContext, build_context_entries,
    build_session_context, migrate_session_entries, parse_session_entries,
    session_entry_to_context_messages,
)
from .session_manager import (
    SessionManager, SessionTreeNode, assert_valid_session_id,
    find_most_recent_session, get_default_session_dir, load_entries_from_file,
)
from .agent_session import (
    AgentSession, CreateAgentSessionOptions, CreateAgentSessionResult,
    PromptOptions, create_agent_session,
)
from .model_config import ModelConfig, strip_json_comments
from .model_runtime import CreateModelRuntimeOptions, ModelRuntime
from .auth_storage import AuthStorage, RuntimeCredentials, read_stored_credential
from .model_resolver import (
    CliModelResult, InitialModelResult, ModelScopeDiagnostic, ModelScopeResult,
    ParsedModelResult, ScopedModel, find_exact_model_reference_match,
    find_initial_model, parse_model_pattern, resolve_cli_model,
    resolve_model_scope, resolve_model_scope_from_models, restore_model_from_session,
)
from .resources import (
    LoadedSkills, ResourceDiagnostic, load_project_context_files,
    load_skills, load_skills_from_dir,
)
from .compaction import CompactionResult, compact_session, estimate_context_tokens, should_compact_session

__all__ = [
    "BRANCH_SUMMARY_PREFIX", "BRANCH_SUMMARY_SUFFIX", "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX", "BashExecutionMessage", "BranchSummaryMessage",
    "CompactionSummaryMessage", "CustomMessage", "bash_execution_to_text",
    "convert_to_llm", "create_branch_summary_message",
    "create_compaction_summary_message", "create_custom_message",
    "DEFAULT_THINKING_LEVEL", "THINKING_LEVEL_OPTIONS", "Skill",
    "format_skills_for_prompt", "BuildSystemPromptOptions",
    "NormalizedBuildSystemPromptOptions", "SystemPromptSections", "SystemPromptState",
    "build_system_prompt", "build_system_prompt_sections", "build_system_prompt_state",
    "diff_system_prompt_sections", "normalize_build_system_prompt_options",
    "BashExecutorOptions", "BashResult", "execute_bash_with_operations",
    "CURRENT_SESSION_VERSION", "SessionContext", "build_context_entries",
    "build_session_context", "migrate_session_entries", "parse_session_entries",
    "session_entry_to_context_messages", "SessionManager", "SessionTreeNode",
    "assert_valid_session_id", "find_most_recent_session", "get_default_session_dir",
    "load_entries_from_file",
    "AgentSession", "CreateAgentSessionOptions", "CreateAgentSessionResult",
    "PromptOptions", "create_agent_session", "ModelConfig", "strip_json_comments",
    "CreateModelRuntimeOptions", "ModelRuntime", "AuthStorage", "RuntimeCredentials",
    "read_stored_credential",
    "CliModelResult", "InitialModelResult", "ModelScopeDiagnostic", "ModelScopeResult",
    "ParsedModelResult", "ScopedModel", "find_exact_model_reference_match",
    "find_initial_model", "parse_model_pattern", "resolve_cli_model",
    "resolve_model_scope", "resolve_model_scope_from_models", "restore_model_from_session",
    "LoadedSkills", "ResourceDiagnostic", "load_project_context_files",
    "load_skills", "load_skills_from_dir",
    "CompactionResult", "compact_session", "estimate_context_tokens", "should_compact_session",
]
