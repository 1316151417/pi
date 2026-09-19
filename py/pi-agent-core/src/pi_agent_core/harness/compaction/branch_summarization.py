"""Branch summarization ported from ``harness/compaction/branch-summarization.ts``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..._pi_ai.models import Models
from ..._pi_ai.text import content_text
from ..._pi_ai.types import Context as AiContext, Model, SimpleStreamOptions, TextContent, Usage, UserMessage
from ..._chord.context import Context
from ..messages import convert_to_llm, create_branch_summary_message, create_compaction_summary_message
from ..result import Result, err, ok
from ..types import BranchSummaryError
from ..utils.retry import RetryCallbacks, RetryPolicy
from .compaction import (
    SUMMARIZATION_SYSTEM_PROMPT,
    SummaryRequest,
    complete_simple_with_retries,
    create_summary_request_options,
    estimate_tokens,
)
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

__all__ = [
    "BranchSummaryResult",
    "BranchSummaryDetails",
    "BranchPreparation",
    "CollectEntriesResult",
    "GenerateBranchSummaryOptions",
    "PreparedBranchSummaryOptions",
    "BRANCH_SUMMARY_PREAMBLE",
    "BRANCH_SUMMARY_PROMPT",
    "collect_entries_for_branch_summary",
    "prepare_branch_entries",
    "generate_branch_summary",
    "generate_branch_summary_with_request",
    "FileOperations",
]


@dataclass
class BranchSummaryResult:
    """Generated branch summary data ready to be persisted as a branch-summary entry."""

    summary: str
    read_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    usage: Optional[Usage] = None


@dataclass
class BranchSummaryDetails:
    """File-operation details stored on generated branch summary entries."""

    read_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)


@dataclass
class BranchPreparation:
    """Prepared branch content for summarization."""

    messages: List[Any] = field(default_factory=list)
    file_ops: FileOperations = field(default_factory=create_file_ops)
    total_tokens: int = 0


@dataclass
class CollectEntriesResult:
    """Entries selected for branch summarization."""

    entries: List[Any] = field(default_factory=list)
    common_ancestor_id: Optional[str] = None


@dataclass
class GenerateBranchSummaryOptions:
    """Options for generating a branch summary."""

    models: Models = None  # type: ignore[assignment]
    model: Model = None  # type: ignore[assignment]
    custom_instructions: Optional[str] = None
    replace_instructions: bool = False
    reserve_tokens: int = 16384
    retry: Optional[RetryPolicy] = None
    callbacks: Optional[RetryCallbacks] = None


@dataclass
class PreparedBranchSummaryOptions:
    custom_instructions: Optional[str] = None
    replace_instructions: bool = False


async def collect_entries_for_branch_summary(
    branch: Any,
    session: Any,
    old_tip_id: Optional[str],
    target_id: str,
    context: Context,
) -> CollectEntriesResult:
    """Collect entries summarized before navigating to a different session tree entry."""
    if not old_tip_id:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)

    old_path = {
        entry.id
        for entry in await branch.find_entries({"start": old_tip_id}, context)
    }
    target_path = await branch.find_entries({"start": target_id}, context)

    common_ancestor_id: Optional[str] = None
    for entry in target_path:
        if entry.id in old_path:
            common_ancestor_id = entry.id
            break

    entries: List[Any] = []
    current: Optional[str] = old_tip_id
    while current and current != common_ancestor_id:
        entry = await session.get_entry(current, context)
        if entry is None:
            raise RuntimeError(f"Corrupt session: entry {current} not found")
        entries.append(entry)
        current = entry.parent_id
    entries.reverse()

    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


def _get_message_from_entry(entry: Any) -> Optional[Any]:
    entry_type = getattr(entry, "type", None)
    if entry_type == "message":
        return None if getattr(entry.message, "role", None) == "toolResult" else entry.message
    if entry_type == "branch_summary":
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
    if entry_type == "compaction":
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)
    return None


def prepare_branch_entries(entries: List[Any], token_budget: int = 0) -> BranchPreparation:
    """Prepare branch entries for summarization within an optional token budget."""
    messages: List[Any] = []
    file_ops = create_file_ops()
    total_tokens = 0

    for entry in entries:
        if getattr(entry, "type", None) != "branch_summary":
            continue
        details = getattr(entry, "details", None)
        if not isinstance(details, dict):
            continue
        read_files = details.get("readFiles")
        if isinstance(read_files, list):
            for path in read_files:
                if isinstance(path, str):
                    file_ops.read.add(path)
        modified_files = details.get("modifiedFiles")
        if isinstance(modified_files, list):
            for path in modified_files:
                if isinstance(path, str):
                    file_ops.edited.add(path)

    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        message = _get_message_from_entry(entry)
        if message is None:
            continue
        extract_file_ops_from_message(message, file_ops)

        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if getattr(entry, "type", None) in ("compaction", "branch_summary"):
                if total_tokens < token_budget * 0.9:
                    messages.insert(0, message)
                    total_tokens += tokens
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\n"
    "Summary of that exploration:\n\n"
)

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: List[Any],
    options: GenerateBranchSummaryOptions,
    context: Context,
) -> Result:
    """Generate a summary for abandoned branch entries."""
    model = options.model
    context_window = model.context_window or 128000
    preparation = prepare_branch_entries(entries, context_window - options.reserve_tokens)

    async def _request(ai_context, request_options, request_context):
        return await complete_simple_with_retries(
            options.models,
            model,
            ai_context,
            request_options,
            options.retry,
            options.callbacks,
            request_context,
        )

    return await generate_branch_summary_with_request(
        preparation,
        PreparedBranchSummaryOptions(
            custom_instructions=options.custom_instructions,
            replace_instructions=options.replace_instructions,
        ),
        _request,
        context,
    )


async def generate_branch_summary_with_request(
    preparation: BranchPreparation,
    options: PreparedBranchSummaryOptions,
    request: SummaryRequest,
    context: Context,
) -> Result:
    """Generate a prepared branch summary through a caller-owned one-request boundary."""
    custom_instructions = options.custom_instructions
    replace_instructions = options.replace_instructions
    messages = preparation.messages
    file_ops = preparation.file_ops

    if not messages:
        return ok(BranchSummaryResult(summary="No content to summarize", read_files=[], modified_files=[]))

    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)

    if replace_instructions and custom_instructions:
        instructions = custom_instructions
    elif custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"
    summarization_messages = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000))
    ]

    response = await request(
        AiContext(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        create_summary_request_options(SimpleStreamOptions(max_tokens=2048), context),
        context,
    )
    if response.stop_reason == "aborted":
        return err(BranchSummaryError("aborted", response.error_message or "Branch summary aborted"))
    if response.stop_reason == "error":
        return err(
            BranchSummaryError(
                "summarization_failed",
                f"Branch summary failed: {response.error_message or 'Unknown error'}",
            )
        )

    summary = BRANCH_SUMMARY_PREAMBLE + content_text(response.content)
    read_files, modified_files = compute_file_lists(file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        BranchSummaryResult(
            summary=summary or "No summary generated",
            usage=response.usage,
            read_files=read_files,
            modified_files=modified_files,
        )
    )
