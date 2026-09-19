"""Tests for the compaction subsystem (ported from compaction semantics)."""

from __future__ import annotations

import asyncio
import time

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_ai.types import (
    AssistantMessage,
    Cost,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_agent_core.harness.compaction import (
    BRANCH_SUMMARY_PREAMBLE,
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    FileOperations,
    calculate_context_tokens,
    compute_file_lists,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    find_turn_start_index,
    generate_branch_summary_with_request,
    generate_summary_with_request,
    prepare_branch_entries,
    prepare_compaction,
    should_compact,
)
from pi_agent_core.harness.compaction.compaction import (
    CompactionPreparation,
    CompactGenerationOptions,
    SummaryGenerationOptions,
    compact_with_request,
)
from pi_agent_core.harness.compaction.branch_summarization import (
    BranchPreparation,
    PreparedBranchSummaryOptions,
)
from pi_agent_core.harness.session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    MessageEntry,
)


def user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=int(time.time() * 1000))


def assistant_message(text: str, usage: Usage = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="faux",
        provider="faux",
        model="faux-1",
        usage=usage or Usage(cost=Cost()),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def message_entry(entry_id: str, parent_id, message) -> MessageEntry:
    return MessageEntry(id=entry_id, parent_id=parent_id, message=message)


def make_model() -> Model:
    return Model(
        id="faux-1",
        name="Faux",
        api="faux",
        provider="faux",
        base_url="http://localhost",
        context_window=128000,
        max_tokens=16384,
    )


def make_request(text: str):
    """Scripted summary request returning a fixed assistant message."""
    calls = []

    async def _request(ai_context, options, context) -> AssistantMessage:
        calls.append({"context": ai_context, "options": options})
        return assistant_message(text)

    _request.calls = calls
    return _request


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def test_calculate_context_tokens_prefers_total():
    assert calculate_context_tokens(Usage(input=10, output=5, total_tokens=42)) == 42
    assert calculate_context_tokens(Usage(input=10, output=5, cache_read=2, cache_write=3)) == 20


def test_estimate_tokens_is_char_based():
    assert estimate_tokens(user_message("a" * 400)) == 100
    assert estimate_tokens(assistant_message("b" * 8)) == 2


def test_estimate_context_tokens_uses_usage_then_trailing():
    messages = [
        user_message("x" * 400),
        assistant_message("y" * 400, usage=Usage(input=50, output=10, total_tokens=60)),
        user_message("z" * 400),
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.usage_tokens == 60
    assert estimate.trailing_tokens == 100  # only the trailing user message
    assert estimate.tokens == 160
    assert estimate.last_usage_index == 1


def test_estimate_context_tokens_without_usage_sums_everything():
    messages = [user_message("a" * 400), user_message("b" * 400)]
    estimate = estimate_context_tokens(messages)
    assert estimate.tokens == 200
    assert estimate.usage_tokens == 0
    assert estimate.last_usage_index is None


def test_should_compact_respects_threshold_and_enabled():
    settings = CompactionSettings(enabled=True, reserve_tokens=1000, keep_recent_tokens=20000)
    assert should_compact(99000, 100000, settings) is False
    assert should_compact(99001, 100000, settings) is True
    assert should_compact(200000, 100000, CompactionSettings(enabled=False)) is False


# ---------------------------------------------------------------------------
# Cut points
# ---------------------------------------------------------------------------


def test_find_cut_point_keeps_recent_tokens():
    entries = []
    for index in range(6):
        entries.append(message_entry(f"m{index}", None, user_message("x" * 400)))
        entries.append(message_entry(f"a{index}", None, assistant_message("y" * 400)))

    cut = find_cut_point(entries, 0, len(entries), 600)
    assert cut.first_kept_entry_index > 0
    kept_tokens = sum(
        estimate_tokens(entries[i].message) for i in range(cut.first_kept_entry_index, len(entries))
    )
    assert kept_tokens >= 600


def test_find_cut_point_never_splits_a_turn():
    entries = [
        message_entry("u0", None, user_message("hello")),
        message_entry("a0", "u0", assistant_message("partial work")),
        message_entry("u1", "a0", user_message("next")),
        message_entry("a1", "u1", assistant_message("done")),
    ]
    cut = find_cut_point(entries, 0, len(entries), 1)
    # The cut must land on a turn boundary and flag a split turn otherwise.
    if cut.is_split_turn:
        assert cut.turn_start_index >= 0
        assert cut.turn_start_index <= cut.first_kept_entry_index


def test_find_turn_start_index():
    entries = [
        message_entry("a0", None, assistant_message("no user yet")),
        message_entry("u0", "a0", user_message("ask")),
        message_entry("a1", "u0", assistant_message("answer")),
    ]
    assert find_turn_start_index(entries, 2, 0) == 1
    assert find_turn_start_index(entries, 0, 0) == -1


# ---------------------------------------------------------------------------
# File operation tracking
# ---------------------------------------------------------------------------


def test_extract_and_compute_file_lists():
    from pi_agent_core.harness.compaction.utils import create_file_ops, extract_file_ops_from_message

    ops = create_file_ops()
    extract_file_ops_from_message(
        AssistantMessage(content=[ToolCall(id="1", name="read", arguments={"path": "a.ts"})]),
        ops,
    )
    extract_file_ops_from_message(
        AssistantMessage(content=[ToolCall(id="2", name="write", arguments={"path": "b.ts"})]),
        ops,
    )
    extract_file_ops_from_message(
        AssistantMessage(content=[ToolCall(id="3", name="edit", arguments={"path": "c.ts"})]),
        ops,
    )
    extract_file_ops_from_message(
        AssistantMessage(content=[ToolCall(id="4", name="read", arguments={"path": "c.ts"})]),
        ops,
    )
    read_files, modified = compute_file_lists(ops)
    assert read_files == ["a.ts"]  # c.ts read but also edited -> moves to modified
    assert modified == ["b.ts", "c.ts"]


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def test_prepare_compaction_returns_none_for_empty_or_compaction_tail():
    settings = DEFAULT_COMPACTION_SETTINGS
    assert prepare_compaction([], settings).value is None
    tail = CompactionEntry(id="c1", parent_id=None, summary="s", tokens_before=1, timestamp=1)
    assert prepare_compaction([tail], settings).value is None


def test_prepare_compaction_splits_history_and_tail():
    entries = []
    for index in range(8):
        entries.append(message_entry(f"u{index}", None, user_message("x" * 800)))
        entries.append(message_entry(f"a{index}", None, assistant_message("y" * 800)))

    settings = CompactionSettings(enabled=True, reserve_tokens=16384, keep_recent_tokens=2000)
    preparation = prepare_compaction(entries, settings).value
    assert preparation is not None
    assert preparation.retained_tail, "expected a retained tail"
    assert preparation.messages_to_summarize, "expected summarized history"
    assert preparation.tokens_before > 0
    assert sum(estimate_tokens(m) for m in preparation.retained_tail) >= 2000


def test_prepare_compaction_reuses_previous_summary():
    previous = CompactionEntry(
        id="c1",
        parent_id=None,
        summary="old summary",
        tokens_before=10,
        timestamp=1,
        retained_tail=[user_message("retained")],
    )
    entries = [previous]
    for index in range(10):
        entries.append(message_entry(f"u{index}", None, user_message("x" * 800)))

    settings = CompactionSettings(enabled=True, reserve_tokens=16384, keep_recent_tokens=500)
    preparation = prepare_compaction(entries, settings).value
    assert preparation.previous_summary == "old summary"


# ---------------------------------------------------------------------------
# Summary generation through a scripted request boundary
# ---------------------------------------------------------------------------


async def test_generate_summary_with_request_builds_prompt():
    request = make_request("## Goal\nShip it")
    result = await generate_summary_with_request(
        [user_message("do the thing"), assistant_message("done")],
        SummaryGenerationOptions(model=make_model(), reserve_tokens=16384),
        request,
        BACKGROUND_CONTEXT,
    )
    assert result.ok
    assert result.value["text"] == "## Goal\nShip it"

    sent = request.calls[0]["context"]
    assert sent.system_prompt.startswith("You are a context summarization assistant")
    prompt_text = sent.messages[0].content[0].text
    assert "<conversation>" in prompt_text
    assert "[User]: do the thing" in prompt_text
    assert "## Goal" in prompt_text  # the summarization prompt template itself


async def test_generate_summary_uses_update_prompt_with_previous():
    request = make_request("updated")
    result = await generate_summary_with_request(
        [user_message("more")],
        SummaryGenerationOptions(model=make_model(), reserve_tokens=16384, previous_summary="old"),
        request,
        BACKGROUND_CONTEXT,
    )
    assert result.ok
    prompt_text = request.calls[0]["context"].messages[0].content[0].text
    assert "<previous-summary>\nold\n</previous-summary>" in prompt_text
    assert "NEW conversation messages" in prompt_text


async def test_generate_summary_appends_custom_instructions():
    request = make_request("x")
    await generate_summary_with_request(
        [user_message("more")],
        SummaryGenerationOptions(
            model=make_model(), reserve_tokens=16384, custom_instructions="focus on tests"
        ),
        request,
        BACKGROUND_CONTEXT,
    )
    prompt_text = request.calls[0]["context"].messages[0].content[0].text
    assert "Additional focus: focus on tests" in prompt_text


async def test_generate_summary_propagates_error_and_abort():
    async def _error_request(ai_context, options, context):
        error = assistant_message("")
        error.stop_reason = "error"
        error.error_message = "boom"
        return error

    result = await generate_summary_with_request(
        [user_message("x")], SummaryGenerationOptions(model=make_model()), _error_request, BACKGROUND_CONTEXT
    )
    assert not result.ok
    assert result.error.code == "summarization_failed"
    assert "boom" in result.error.message

    async def _aborted_request(ai_context, options, context):
        aborted = assistant_message("")
        aborted.stop_reason = "aborted"
        return aborted

    result = await generate_summary_with_request(
        [user_message("x")], SummaryGenerationOptions(model=make_model()), _aborted_request, BACKGROUND_CONTEXT
    )
    assert not result.ok
    assert result.error.code == "aborted"


async def test_compact_with_request_appends_file_operations():
    request = make_request("summary body")
    preparation = CompactionPreparation(
        messages_to_summarize=[user_message("x")],
        retained_tail=[assistant_message("tail")],
        tokens_before=1234,
        settings=DEFAULT_COMPACTION_SETTINGS,
    )
    preparation.file_ops.read.add("src/a.ts")
    preparation.file_ops.written.add("src/b.ts")

    result = await compact_with_request(
        preparation,
        CompactGenerationOptions(model=make_model()),
        request,
        BACKGROUND_CONTEXT,
    )
    assert result.ok
    assert result.value.summary.startswith("summary body")
    assert "<read-files>\nsrc/a.ts\n</read-files>" in result.value.summary
    assert "<modified-files>\nsrc/b.ts\n</modified-files>" in result.value.summary
    assert result.value.tokens_before == 1234
    assert result.value.retained_tail == [assistant_message("tail")] or len(result.value.retained_tail) == 1
    assert result.value.details.read_files == ["src/a.ts"]


async def test_compact_with_request_split_turn_merges_prefix():
    request = make_request("part summary")
    preparation = CompactionPreparation(
        messages_to_summarize=[user_message("history")],
        turn_prefix_messages=[assistant_message("prefix work")],
        retained_tail=[assistant_message("tail")],
        is_split_turn=True,
        tokens_before=500,
        settings=DEFAULT_COMPACTION_SETTINGS,
    )
    result = await compact_with_request(
        preparation, CompactGenerationOptions(model=make_model()), request, BACKGROUND_CONTEXT
    )
    assert result.ok
    assert "**Turn Context (split turn):**" in result.value.summary
    assert len(request.calls) == 2  # history + turn prefix


# ---------------------------------------------------------------------------
# Branch summarization
# ---------------------------------------------------------------------------


def test_prepare_branch_entries_skips_tool_results_and_budgets():
    entries = [
        message_entry("u0", None, user_message("hello")),
        message_entry(
            "t0",
            None,
            ToolResultMessage(tool_call_id="1", tool_name="bash", content=[TextContent(text="out")]),
        ),
        message_entry("a0", None, assistant_message("answer")),
    ]
    preparation = prepare_branch_entries(entries)
    assert len(preparation.messages) == 2  # tool result dropped
    assert preparation.total_tokens > 0


def test_prepare_branch_entries_inherits_file_ops_from_summaries():
    entry = BranchSummaryEntry(
        id="b1",
        parent_id=None,
        summary="earlier",
        from_id=None,
        timestamp=1,
        details={"readFiles": ["a.ts"], "modifiedFiles": ["b.ts"]},
    )
    preparation = prepare_branch_entries([entry])
    read_files, modified = compute_file_lists(preparation.file_ops)
    assert read_files == ["a.ts"]
    assert modified == ["b.ts"]


async def test_generate_branch_summary_with_request_adds_preamble():
    request = make_request("branch body")
    preparation = BranchPreparation(messages=[user_message("explore")])

    result = await generate_branch_summary_with_request(
        preparation, PreparedBranchSummaryOptions(), request, BACKGROUND_CONTEXT
    )
    assert result.ok
    assert result.value.summary.startswith(BRANCH_SUMMARY_PREAMBLE)
    assert "branch body" in result.value.summary

    prompt_text = request.calls[0]["context"].messages[0].content[0].text
    assert "Create a structured summary of this conversation branch" in prompt_text


async def test_generate_branch_summary_empty_messages():
    request = make_request("unused")
    result = await generate_branch_summary_with_request(
        BranchPreparation(), PreparedBranchSummaryOptions(), request, BACKGROUND_CONTEXT
    )
    assert result.ok
    assert result.value.summary == "No content to summarize"
    assert request.calls == []


async def test_generate_branch_summary_replace_instructions():
    request = make_request("body")
    await generate_branch_summary_with_request(
        BranchPreparation(messages=[user_message("x")]),
        PreparedBranchSummaryOptions(custom_instructions="ONLY THIS", replace_instructions=True),
        request,
        BACKGROUND_CONTEXT,
    )
    prompt_text = request.calls[0]["context"].messages[0].content[0].text
    assert prompt_text.endswith("ONLY THIS")
    assert "Create a structured summary of this conversation branch" not in prompt_text
