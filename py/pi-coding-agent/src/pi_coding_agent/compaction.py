"""Conversation checkpointing over the coding-agent JSONL session tree."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from pi_agent_core.harness.compaction.compaction import (
    SUMMARIZATION_PROMPT, SUMMARIZATION_SYSTEM_PROMPT,
    UPDATE_SUMMARIZATION_PROMPT, estimate_tokens, should_compact,
    CompactionSettings,
)
from pi_agent_core.harness.compaction.utils import serialize_conversation
from pi_ai.text import content_text
from pi_ai.types import Context, Model, SimpleStreamOptions, TextContent, UserMessage, Usage

from .messages import CodingAgentMessage, convert_to_llm
from .model_runtime import ModelRuntime
from .session_entries import SessionEntry, session_entry_to_context_messages
from .session_manager import SessionManager


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    usage: Usage


def estimate_context_tokens(messages: Sequence[CodingAgentMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if getattr(message, "role", None) != "assistant":
            continue
        usage = getattr(message, "usage", None)
        if usage is None or getattr(message, "stop_reason", None) in ("error", "aborted"):
            continue
        measured = usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write
        if measured > 0:
            return int(measured) + sum(estimate_tokens(item) for item in messages[index + 1:])
    return sum(estimate_tokens(message) for message in messages)


def _visible_messages(entry: SessionEntry) -> list[CodingAgentMessage]:
    return [message for message in session_entry_to_context_messages(entry) if getattr(message, "role", None) != "system"]


def _cut_index(entries: Sequence[SessionEntry], start: int, keep_recent_tokens: int) -> int | None:
    candidates = [
        index for index in range(start, len(entries))
        if any(getattr(message, "role", None) in ("user", "bashExecution", "custom", "branchSummary")
               for message in _visible_messages(entries[index]))
    ]
    if len(candidates) < 2:
        return None
    accumulated = 0
    for index in range(len(entries) - 1, start - 1, -1):
        accumulated += sum(estimate_tokens(message) for message in _visible_messages(entries[index]))
        if accumulated >= keep_recent_tokens:
            candidate = next((candidate for candidate in candidates if candidate >= index), candidates[-1])
            return candidate if candidate > start else None
    return None


async def compact_session(
    manager: SessionManager, runtime: ModelRuntime, model: Model,
    *, keep_recent_tokens: int = 20000, reserve_tokens: int = 16384,
    custom_instructions: str | None = None,
) -> CompactionResult | None:
    entries = manager.get_branch()
    if not entries or entries[-1].get("type") == "compaction":
        return None
    previous = next((entry for entry in reversed(entries) if entry.get("type") == "compaction"), None)
    start = 0
    previous_summary: str | None = None
    if previous is not None:
        previous_summary = str(previous.get("summary", ""))
        first_id = previous.get("firstKeptEntryId")
        start = next((index for index, entry in enumerate(entries) if entry.get("id") == first_id), entries.index(previous) + 1)
    cut = _cut_index(entries, start, keep_recent_tokens)
    if cut is None:
        return None
    messages = [message for entry in entries[start:cut] for message in _visible_messages(entry)]
    if not messages:
        return None
    tokens_before = estimate_context_tokens(manager.build_session_context().messages)
    conversation = serialize_conversation(convert_to_llm(messages))
    prompt = f"<conversation>\n{conversation}\n</conversation>\n\n"
    if previous_summary:
        prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt += UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        prompt += f"\n\nAdditional focus: {custom_instructions}"
    request = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[UserMessage(content=[TextContent(text=prompt)], timestamp=int(time.time() * 1000))],
    )
    max_tokens = min(int(reserve_tokens * 0.8), model.max_tokens if model.max_tokens > 0 else reserve_tokens)
    response = await runtime.stream_simple(model, request, SimpleStreamOptions(
        max_tokens=max_tokens, cache_retention="none", session_id=manager.get_session_id(),
    )).result
    if response.stop_reason in ("error", "aborted", "length"):
        raise RuntimeError(f"Summarization failed: {response.error_message or response.stop_reason}")
    if any(getattr(block, "type", None) == "toolCall" for block in response.content):
        raise RuntimeError("Summarization attempted to call a tool")
    summary = content_text(response.content)
    if not summary.strip():
        raise RuntimeError("Summarization returned an empty summary")
    first_kept_id = str(entries[cut]["id"])
    manager.append_compaction(summary, first_kept_id, tokens_before, usage=response.usage)
    return CompactionResult(summary, first_kept_id, tokens_before, response.usage)


def should_compact_session(messages: Sequence[CodingAgentMessage], model: Model) -> bool:
    return should_compact(estimate_context_tokens(messages), model.context_window, CompactionSettings())


__all__ = ["CompactionResult", "estimate_context_tokens", "compact_session", "should_compact_session"]
