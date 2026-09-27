"""Context compaction ported from ``harness/compaction/compaction.ts``."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from pi_ai.models import Models
from pi_ai.text import content_text
from pi_ai.types import (
    AssistantMessage,
    Context as AiContext,
    Model,
    SimpleStreamOptions,
    TextContent,
    Usage,
    UserMessage,
)
from ..._chord.context import Context
from ..messages import convert_to_llm, create_branch_summary_message, create_compaction_summary_message
from ..result import Result, err, ok
from ..session.context import build_context_entries, session_entry_to_context_messages
from ..types import CompactionError
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy, retry_assistant_call
from ..utils.usage import add_usage
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

__all__ = [
    "CompactionDetails",
    "CompactResult",
    "SummaryRequest",
    "CompactionSettings",
    "DEFAULT_COMPACTION_SETTINGS",
    "ContextUsageEstimate",
    "CutPointResult",
    "CompactionPreparation",
    "CompactGenerationOptions",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "SUMMARIZATION_PROMPT",
    "UPDATE_SUMMARIZATION_PROMPT",
    "TURN_PREFIX_SUMMARIZATION_PROMPT",
    "create_summary_request_options",
    "complete_simple_with_retries",
    "calculate_context_tokens",
    "get_last_assistant_usage",
    "estimate_context_tokens",
    "should_compact",
    "estimate_tokens",
    "find_turn_start_index",
    "find_cut_point",
    "generate_summary",
    "generate_summary_with_usage",
    "generate_summary_with_request",
    "prepare_compaction",
    "compact",
    "compact_with_request",
]


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return "[unserializable]"


# ---------------------------------------------------------------------------
# Entry <-> message projection helpers
# ---------------------------------------------------------------------------


def extract_file_operations(messages: List[Any], entries: List[Any], prev_compaction_index: int) -> FileOperations:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        previous = entries[prev_compaction_index]
        details = getattr(previous, "details", None)
        if isinstance(details, dict):
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
    for message in messages:
        extract_file_ops_from_message(message, file_ops)
    return file_ops


def _get_message_from_entry(entry: Any) -> Optional[Any]:
    entry_type = getattr(entry, "type", None)
    if entry_type == "message":
        return entry.message
    if entry_type == "branch_summary":
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
    if entry_type == "compaction":
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)
    return None


def _get_message_from_entry_for_compaction(entry: Any) -> Optional[Any]:
    if getattr(entry, "type", None) == "compaction":
        return None
    return _get_message_from_entry(entry)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class CompactionDetails:
    """File-operation details stored on generated compaction entries."""

    read_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"readFiles": self.read_files, "modifiedFiles": self.modified_files}


@dataclass
class CompactResult:
    """Generated compaction data ready to be persisted as a compaction entry."""

    summary: str
    tokens_before: int
    retained_tail: List[Any] = field(default_factory=list)
    usage: Optional[Usage] = None
    details: Any = None


#: One provider request producing a summary assistant message.
SummaryRequest = Callable[[AiContext, SimpleStreamOptions, Context], Awaitable[AssistantMessage]]


@dataclass
class CompactionSettings:
    """Compaction thresholds and retention settings."""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000

    def to_json(self) -> dict:
        return {
            "enabled": self.enabled,
            "reserveTokens": self.reserve_tokens,
            "keepRecentTokens": self.keep_recent_tokens,
        }


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


@dataclass
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: Optional[int]


@dataclass
class CutPointResult:
    """Cut point selected for compaction."""

    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


@dataclass
class CompactionPreparation:
    """Prepared inputs for a compaction run."""

    messages_to_summarize: List[Any] = field(default_factory=list)
    turn_prefix_messages: List[Any] = field(default_factory=list)
    retained_tail: List[Any] = field(default_factory=list)
    is_split_turn: bool = False
    tokens_before: int = 0
    previous_summary: Optional[str] = None
    file_ops: FileOperations = field(default_factory=create_file_ops)
    settings: CompactionSettings = field(default_factory=CompactionSettings)


@dataclass
class SummaryGenerationOptions:
    model: Model = None  # type: ignore[assignment]
    reserve_tokens: int = 16384
    custom_instructions: Optional[str] = None
    previous_summary: Optional[str] = None
    thinking_level: Optional[str] = None


@dataclass
class CompactGenerationOptions:
    model: Model = None  # type: ignore[assignment]
    custom_instructions: Optional[str] = None
    thinking_level: Optional[str] = None


# ---------------------------------------------------------------------------
# Request plumbing
# ---------------------------------------------------------------------------


def create_summary_request_options(options: SimpleStreamOptions, context: Context) -> SimpleStreamOptions:
    """Summaries are standalone requests: no cache writes, fresh session id."""
    from pi_ai.uuid_utils import uuidv7

    cloned = copy.copy(options)
    cloned.cache_retention = "none"
    cloned.session_id = cloned.session_id or uuidv7()
    if context.abort_signal is not None:
        cloned.signal = context.abort_signal
    return cloned


async def complete_simple_with_retries(
    models: Models,
    model: Model,
    ai_context: AiContext,
    options: SimpleStreamOptions,
    retry: Optional[RetryPolicy],
    callbacks: Optional[RetryCallbacks],
    context: Context,
) -> AssistantMessage:
    request_options = create_summary_request_options(options, context)

    async def _produce() -> AssistantMessage:
        stream = models.stream_simple(model, ai_context, request_options)
        collected = None
        async for _event in stream:
            collected = _event
        return await stream.result

    return await retry_assistant_call(_produce, retry, request_options.signal, callbacks)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def calculate_context_tokens(usage: Usage) -> int:
    return (
        usage.total_tokens
        or usage.input + usage.output + usage.cache_read + usage.cache_write
    )


def _get_assistant_usage(message: Any) -> Optional[Usage]:
    if getattr(message, "role", None) != "assistant":
        return None
    stop_reason = getattr(message, "stop_reason", None)
    usage = getattr(message, "usage", None)
    if stop_reason in ("aborted", "error") or usage is None:
        return None
    return usage if calculate_context_tokens(usage) > 0 else None


def get_last_assistant_usage(entries: List[Any]) -> Optional[Usage]:
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if getattr(entry, "type", None) == "message":
            usage = _get_assistant_usage(entry.message)
            if usage is not None:
                return usage
    return None


def _get_last_assistant_usage_info(messages: List[Any]) -> Optional[dict]:
    for index in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[index])
        if usage is not None:
            return {"usage": usage, "index": index}
    return None


def estimate_context_tokens(messages: List[Any]) -> ContextUsageEstimate:
    usage_info = _get_last_assistant_usage_info(messages)
    if usage_info is None:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(
            tokens=estimated, usage_tokens=0, trailing_tokens=estimated, last_usage_index=None
        )

    usage_tokens = calculate_context_tokens(usage_info["usage"])
    trailing = sum(
        estimate_tokens(messages[index]) for index in range(usage_info["index"] + 1, len(messages))
    )
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=usage_info["index"],
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


ESTIMATED_IMAGE_CHARS = 4800


def _estimate_text_and_image_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            chars += len(getattr(block, "text", "") or "")
        elif block_type == "image":
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: Any) -> int:
    """Estimate token count for one message using a conservative character heuristic."""
    role = getattr(message, "role", None)
    if role == "user":
        return -(-_estimate_text_and_image_content_chars(message.content) // 4)
    if role == "assistant":
        chars = 0
        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                chars += len(block.text)
            elif block_type == "thinking":
                chars += len(block.thinking)
            elif block_type == "toolCall":
                chars += len(block.name) + len(safe_json_stringify(block.arguments))
        return -(-chars // 4)
    if role in ("custom", "toolResult"):
        return -(-_estimate_text_and_image_content_chars(message.content) // 4)
    if role == "bashExecution":
        return -(-(len(message.command) + len(message.output)) // 4)
    if role in ("branchSummary", "compactionSummary"):
        return -(-len(message.summary) // 4)
    return 0


# ---------------------------------------------------------------------------
# Cut point selection
# ---------------------------------------------------------------------------


def _find_valid_cut_points(entries: List[Any], start_index: int, end_index: int) -> List[int]:
    cut_points: List[int] = []
    for index in range(start_index, end_index):
        entry = entries[index]
        entry_type = getattr(entry, "type", None)
        if entry_type == "message":
            role = getattr(entry.message, "role", None)
            if role in ("bashExecution", "custom", "branchSummary", "compactionSummary", "user", "assistant"):
                cut_points.append(index)
        if entry_type == "branch_summary":
            cut_points.append(index)
    return cut_points


def find_turn_start_index(entries: List[Any], entry_index: int, start_index: int) -> int:
    """Find the user-visible message that starts the turn containing an entry."""
    for index in range(entry_index, start_index - 1, -1):
        entry = entries[index]
        if getattr(entry, "type", None) == "branch_summary":
            return index
        if getattr(entry, "type", None) == "message":
            role = getattr(entry.message, "role", None)
            if role in ("user", "bashExecution"):
                return index
    return -1


def find_cut_point(
    entries: List[Any], start_index: int, end_index: int, keep_recent_tokens: int
) -> CutPointResult:
    """Find the compaction cut point keeping approximately the requested recent-token budget."""
    cut_points = _find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    accumulated_tokens = 0
    cut_index = cut_points[0]

    for index in range(end_index - 1, start_index - 1, -1):
        entry = entries[index]
        if getattr(entry, "type", None) != "message":
            continue
        accumulated_tokens += estimate_tokens(entry.message)
        if accumulated_tokens >= keep_recent_tokens:
            for candidate in cut_points:
                if candidate >= index:
                    cut_index = candidate
                    break
            break

    while cut_index > start_index:
        previous_entry = entries[cut_index - 1]
        if getattr(previous_entry, "type", None) == "compaction":
            break
        if getattr(previous_entry, "type", None) == "message":
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = getattr(cut_entry, "type", None) == "message" and getattr(
        cut_entry.message, "role", None
    ) == "user"
    turn_start_index = -1 if is_user_message else find_turn_start_index(entries, cut_index, start_index)
    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=not is_user_message and turn_start_index != -1,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user and an "
    "AI assistant, then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the "
    "structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- If something is no longer relevant, you may remove it, but err on the side of preserving

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


async def generate_summary_with_request(
    current_messages: List[Any],
    options: SummaryGenerationOptions,
    request: SummaryRequest,
    context: Context,
) -> Result:
    """Generate one summary through a caller-owned one-request boundary."""
    model = options.model
    reserve_tokens = options.reserve_tokens
    custom_instructions = options.custom_instructions
    previous_summary = options.previous_summary
    thinking_level = options.thinking_level

    max_tokens = min(
        int(0.8 * reserve_tokens),
        model.max_tokens if model.max_tokens > 0 else float("inf"),
    )
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    summarization_messages = [
        UserMessage(
            content=[TextContent(text=prompt_text)],
            timestamp=int(time.time() * 1000),
        )
    ]

    completion_options = SimpleStreamOptions(max_tokens=max_tokens)
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_options.reasoning = thinking_level

    response = await request(
        AiContext(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        create_summary_request_options(completion_options, context),
        context,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed",
                f"Summarization failed: {response.error_message or 'Unknown error'}",
            )
        )
    return ok({"text": content_text(response.content), "usage": response.usage})


async def generate_summary_with_usage(
    current_messages: List[Any],
    models: Models,
    model: Model,
    reserve_tokens: int,
    custom_instructions: Optional[str],
    previous_summary: Optional[str],
    thinking_level: Optional[str],
    retry: Optional[RetryPolicy],
    callbacks: Optional[RetryCallbacks],
    context: Context,
) -> Result:
    async def _request(ai_context, options, request_context):
        return await complete_simple_with_retries(
            models, model, ai_context, options, retry, callbacks, request_context
        )

    return await generate_summary_with_request(
        current_messages,
        SummaryGenerationOptions(
            model=model,
            reserve_tokens=reserve_tokens,
            custom_instructions=custom_instructions,
            previous_summary=previous_summary,
            thinking_level=thinking_level,
        ),
        _request,
        context,
    )


async def generate_summary(
    current_messages: List[Any],
    models: Models,
    model: Model,
    reserve_tokens: int,
    custom_instructions: Optional[str],
    previous_summary: Optional[str],
    thinking_level: Optional[str],
    retry: Optional[RetryPolicy],
    callbacks: Optional[RetryCallbacks],
    context: Context,
) -> Result:
    """Generate or update a conversation summary for compaction."""
    result = await generate_summary_with_usage(
        current_messages,
        models,
        model,
        reserve_tokens,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
        context,
    )
    return ok(result.value["text"]) if result.ok else err(result.error)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def prepare_compaction(path_entries: List[Any], settings: CompactionSettings) -> Result:
    """Prepare session entries for compaction, or return ``ok(None)`` when not applicable."""
    if not path_entries or getattr(path_entries[-1], "type", None) == "compaction":
        return ok(None)

    prev_compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if getattr(path_entries[index], "type", None) == "compaction":
            prev_compaction_index = index
            break

    previous_summary: Optional[str] = None
    compactable_entries = list(path_entries)
    if prev_compaction_index >= 0:
        previous = path_entries[prev_compaction_index]
        previous_summary = previous.summary
        from ..session.types import MessageEntry

        virtual_retained_entries = [
            MessageEntry(
                id=f"{previous.id}:retained:{index}",
                parent_id=previous.id if index == 0 else f"{previous.id}:retained:{index - 1}",
                seq=previous.seq,
                timestamp=getattr(message, "timestamp", 0),
                message=message,
            )
            for index, message in enumerate(previous.retained_tail)
        ]
        compactable_entries = [*virtual_retained_entries, *path_entries[prev_compaction_index + 1 :]]

    boundary_end = len(compactable_entries)

    context_messages = [
        message
        for entry in build_context_entries(path_entries)
        for message in session_entry_to_context_messages(entry)
    ]
    tokens_before = estimate_context_tokens(context_messages).tokens

    cut_point = find_cut_point(compactable_entries, 0, boundary_end, settings.keep_recent_tokens)
    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index

    messages_to_summarize: List[Any] = []
    for index in range(0, max(0, history_end)):
        message = _get_message_from_entry_for_compaction(compactable_entries[index])
        if message is not None:
            messages_to_summarize.append(message)

    turn_prefix_messages: List[Any] = []
    if cut_point.is_split_turn:
        for index in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            message = _get_message_from_entry_for_compaction(compactable_entries[index])
            if message is not None:
                turn_prefix_messages.append(message)

    retained_tail: List[Any] = []
    for index in range(cut_point.first_kept_entry_index, boundary_end):
        message = _get_message_from_entry_for_compaction(compactable_entries[index])
        if message is not None:
            retained_tail.append(message)

    file_ops = extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)
    if cut_point.is_split_turn:
        for message in turn_prefix_messages:
            extract_file_ops_from_message(message, file_ops)

    return ok(
        CompactionPreparation(
            messages_to_summarize=messages_to_summarize,
            turn_prefix_messages=turn_prefix_messages,
            retained_tail=retained_tail,
            is_split_turn=cut_point.is_split_turn,
            tokens_before=tokens_before,
            previous_summary=previous_summary,
            file_ops=file_ops,
            settings=settings,
        )
    )


async def _generate_turn_prefix_summary(
    messages: List[Any],
    model: Model,
    reserve_tokens: int,
    thinking_level: Optional[str],
    request: SummaryRequest,
    context: Context,
) -> Result:
    max_tokens = min(
        int(0.5 * reserve_tokens),
        model.max_tokens if model.max_tokens > 0 else float("inf"),
    )
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = (
        f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    )
    summarization_messages = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000))
    ]
    completion_options = SimpleStreamOptions(max_tokens=max_tokens)
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_options.reasoning = thinking_level

    response = await request(
        AiContext(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        create_summary_request_options(completion_options, context),
        context,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Turn prefix summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed",
                f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}",
            )
        )
    return ok({"text": content_text(response.content), "usage": response.usage})


async def compact_with_request(
    preparation: CompactionPreparation,
    options: CompactGenerationOptions,
    request: SummaryRequest,
    context: Context,
) -> Result:
    """Generate compaction data through a caller-owned boundary for each provider request."""
    model = options.model
    custom_instructions = options.custom_instructions
    thinking_level = options.thinking_level
    messages_to_summarize = preparation.messages_to_summarize
    turn_prefix_messages = preparation.turn_prefix_messages
    retained_tail = preparation.retained_tail
    is_split_turn = preparation.is_split_turn
    tokens_before = preparation.tokens_before
    previous_summary = preparation.previous_summary
    file_ops = preparation.file_ops
    settings = preparation.settings

    summary: str
    summary_usage: Optional[Usage]

    if is_split_turn and turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Optional[Usage] = None
        if messages_to_summarize:
            history_result = await generate_summary_with_request(
                messages_to_summarize,
                SummaryGenerationOptions(
                    model=model,
                    reserve_tokens=settings.reserve_tokens,
                    custom_instructions=custom_instructions,
                    previous_summary=previous_summary,
                    thinking_level=thinking_level,
                ),
                request,
                context,
            )
            if not history_result.ok:
                return err(history_result.error)
            history_text = history_result.value["text"]
            history_usage = history_result.value["usage"]

        turn_prefix_result = await _generate_turn_prefix_summary(
            turn_prefix_messages, model, settings.reserve_tokens, thinking_level, request, context
        )
        if not turn_prefix_result.ok:
            return err(turn_prefix_result.error)
        summary = (
            f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_result.value['text']}"
        )
        summary_usage = (
            add_usage(history_usage, turn_prefix_result.value["usage"])
            if history_usage is not None
            else turn_prefix_result.value["usage"]
        )
    else:
        summary_result = await generate_summary_with_request(
            messages_to_summarize,
            SummaryGenerationOptions(
                model=model,
                reserve_tokens=settings.reserve_tokens,
                custom_instructions=custom_instructions,
                previous_summary=previous_summary,
                thinking_level=thinking_level,
            ),
            request,
            context,
        )
        if not summary_result.ok:
            return err(summary_result.error)
        summary = summary_result.value["text"]
        summary_usage = summary_result.value["usage"]

    read_files, modified_files = compute_file_lists(file_ops)
    summary += format_file_operations(read_files, modified_files)
    details = CompactionDetails(read_files=read_files, modified_files=modified_files)

    return ok(
        CompactResult(
            summary=summary,
            tokens_before=tokens_before,
            usage=summary_usage,
            retained_tail=retained_tail,
            details=details,
        )
    )


async def compact(
    preparation: CompactionPreparation,
    models: Models,
    model: Model,
    custom_instructions: Optional[str],
    thinking_level: Optional[str],
    retry: Optional[RetryPolicy],
    callbacks: Optional[RetryCallbacks],
    context: Context,
) -> Result:
    """Generate compaction summary data from prepared session history."""

    async def _request(ai_context, request_options, request_context):
        return await complete_simple_with_retries(
            models, model, ai_context, request_options, retry, callbacks, request_context
        )

    return await compact_with_request(
        preparation,
        CompactGenerationOptions(
            model=model, custom_instructions=custom_instructions, thinking_level=thinking_level
        ),
        _request,
        context,
    )
