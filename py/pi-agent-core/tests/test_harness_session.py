"""Tests for the session storage subsystem (ported from session repo semantics)."""

from __future__ import annotations

import asyncio
import time

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core._pi_ai.types import TextContent, UserMessage, Usage
from pi_agent_core.harness.session import (
    BranchScan,
    EntryQuery,
    ForkOptions,
    LaneConfiguration,
    LaneState,
    MemorySessionRepo,
    MessageEntry,
    SessionBranchExistsError,
    SessionPendingAssistantMessageError,
    UsageRow,
    append_list,
    branch_tip,
    entry_label,
    insert_entry,
    insert_usage,
    lane_config,
    lane_state,
    list,
    session_name,
    set_value,
    value,
)
from pi_agent_core.harness.session.session import StorageBackedSession
from pi_agent_core.harness.session.memory import MemoryStorage


def user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=int(time.time() * 1000))


async def test_create_session_and_branch_flow():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)

    # Session starts with no branches.
    assert await session.branch("main", BACKGROUND_CONTEXT) is None

    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    assert main.name == "main"
    assert await main.get_tip_id(BACKGROUND_CONTEXT) is None

    # Append messages build a parent chain.
    first_id = await main.append_message(user_message("hello"), BACKGROUND_CONTEXT)
    second_id = await main.append_message(user_message("world"), BACKGROUND_CONTEXT)
    assert await main.get_tip_id(BACKGROUND_CONTEXT) == second_id

    entries = await main.find_entries(None, BACKGROUND_CONTEXT)
    assert [e.id for e in entries] == [second_id, first_id]  # newestFirst

    oldest = await main.find_entries(BranchScan(order="oldestFirst"), BACKGROUND_CONTEXT)
    assert [e.id for e in oldest] == [first_id, second_id]

    stats = await session.get_stats(BACKGROUND_CONTEXT)
    assert stats.message_count == 2


async def test_branch_from_existing_entry_forks_history():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    first_id = await main.append_message(user_message("one"), BACKGROUND_CONTEXT)
    second_id = await main.append_message(user_message("two"), BACKGROUND_CONTEXT)

    feature = await session.create_branch("feature", first_id, BACKGROUND_CONTEXT)
    assert await feature.get_tip_id(BACKGROUND_CONTEXT) == first_id
    third_id = await feature.append_message(user_message("three"), BACKGROUND_CONTEXT)

    feature_entries = await feature.find_entries(None, BACKGROUND_CONTEXT)
    assert [e.id for e in feature_entries] == [third_id, first_id]

    # Main untouched.
    assert await main.get_tip_id(BACKGROUND_CONTEXT) == second_id

    # Re-creating an existing branch fails.
    with pytest.raises(SessionBranchExistsError):
        await session.create_branch("feature", None, BACKGROUND_CONTEXT)


async def test_pending_assistant_message_rejected():
    from pi_agent_core._pi_ai.types import AssistantMessage

    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    pending = AssistantMessage(content=[], stop_reason="pending")
    with pytest.raises(SessionPendingAssistantMessageError):
        await main.append_message(pending, BACKGROUND_CONTEXT)


async def test_values_and_lists_roundtrip():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT

    await session.set_name("my session", context)
    assert await session.get_name(context) == "my session"

    marker = value("app", "marker")
    await session.set_value(marker, {"count": 3}, context)
    stored = await session.get_value(marker, context)
    assert stored.value == {"count": 3}
    assert stored.seq > 0

    await session.delete_value(marker, context)
    assert await session.get_value(marker, context) is None

    frames = list("app", "frames")
    await session.append_list(frames, "a", context)
    await session.append_list(frames, "b", context)
    elements = await session.read_list(frames, None, context)
    assert [e.value for e in elements] == ["a", "b"]

    await session.delete_list(frames, context)
    assert await session.read_list(frames, None, context) == []

    main = await session.create_branch("main", None, context)
    await main.append_message(user_message("x"), context)
    await session.set_label("some-entry", "checkpoint", context)
    assert await session.get_label("some-entry", context) == "checkpoint"


async def test_scan_entries_with_filters():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    ids = [await main.append_message(user_message(f"m{i}"), BACKGROUND_CONTEXT) for i in range(4)]

    all_entries = await session.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)
    assert [e.id for e in all_entries] == ids

    limited = await session.find_entries(EntryQuery(order="asc", limit=2), BACKGROUND_CONTEXT)
    assert [e.id for e in limited] == ids[:2]

    desc = await session.find_entries(EntryQuery(order="desc", limit=2), BACKGROUND_CONTEXT)
    assert [e.id for e in desc] == [ids[3], ids[2]]


async def test_custom_entries_and_scan():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    message_id = await main.append_message(user_message("hi"), BACKGROUND_CONTEXT)
    custom_id = await main.append_custom_entry("app.note", {"kind": "note"}, BACKGROUND_CONTEXT)

    custom_entries = await session.find_entries(
        EntryQuery(type="custom", custom_type="app.note"), BACKGROUND_CONTEXT
    )
    assert [e.id for e in custom_entries] == [custom_id]

    message_entries = await session.find_entries(EntryQuery(type="message"), BACKGROUND_CONTEXT)
    assert [e.id for e in message_entries] == [message_id]


async def test_usage_rows_accumulate_into_stats():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT
    main = await session.create_branch("main", None, context)
    await main.append_message(user_message("hi"), context)

    async def _mutation(mutator, mutation_context):
        writes = [
            insert_entry(
                MessageEntry(id="u-entry-1", parent_id=None, message=user_message("usage"))
            ),
            insert_usage(
                UsageRow(id="usage-1", usage=Usage(input=10, output=5, total_tokens=15))
            ),
        ]
        await mutator.commit(writes, mutation_context)

    await session.mutate(_mutation, context)
    stats = await session.get_stats(context)
    assert stats.message_count == 2
    assert stats.usage.input == 10
    assert stats.usage.output == 5



async def test_duplicate_entry_ids_rejected():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT

    async def _mutation(mutator, mutation_context):
        await mutator.commit(
            [insert_entry(MessageEntry(id="dup-1", parent_id=None, message=user_message("a")))],
            mutation_context,
        )

    await session.mutate(_mutation, context)

    async def _duplicate(mutator, mutation_context):
        await mutator.commit(
            [insert_entry(MessageEntry(id="dup-1", parent_id=None, message=user_message("b")))],
            mutation_context,
        )

    with pytest.raises(RuntimeError, match="Duplicate entry or usage id: dup-1"):
        await session.mutate(_duplicate, context)


async def test_missing_parent_rejected():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT

    async def _mutation(mutator, mutation_context):
        await mutator.commit(
            [insert_entry(MessageEntry(id="orphan", parent_id="ghost", message=user_message("x")))],
            mutation_context,
        )

    with pytest.raises(RuntimeError, match="Missing parent entry: ghost"):
        await session.mutate(_mutation, context)


async def test_mutator_commit_only_once():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT

    async def _mutation(mutator, mutation_context):
        await mutator.commit(
            [insert_entry(MessageEntry(id="once-1", parent_id=None, message=user_message("a")))],
            mutation_context,
        )
        with pytest.raises(RuntimeError, match="commit already attempted"):
            await mutator.commit(
                [insert_entry(MessageEntry(id="once-2", parent_id=None, message=user_message("b")))],
                mutation_context,
            )

    await session.mutate(_mutation, context)


async def test_mutations_are_serialized():
    repo = MemorySessionRepo()
    session = await repo.create(None, BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT
    interleaved: list = []

    async def _writer(index: int):
        async def _mutation(mutator, mutation_context):
            interleaved.append(f"start-{index}")
            await asyncio.sleep(0.01)
            interleaved.append(f"end-{index}")
            await mutator.commit(
                [insert_entry(MessageEntry(id=f"ser-{index}", parent_id=None, message=user_message(f"{index}")))],
                mutation_context,
            )

        await session.mutate(_mutation, context)

    await asyncio.gather(*[_writer(i) for i in range(4)])
    # No mutation observed another mutation's interior.
    for i in range(4):
        start = interleaved.index(f"start-{i}")
        end = interleaved.index(f"end-{i}")
        assert end == start + 1


async def test_fork_branch_copies_path():
    repo = MemorySessionRepo()
    context = BACKGROUND_CONTEXT
    session = await repo.create(None, context)
    main = await session.create_branch("main", None, context)

    # Configure the lane so branch forks are allowed.
    async def _configure(mutator, mutation_context):
        await mutator.commit(
            [
                set_value(branch_tip("main"), None),
                set_value(lane_config("main"), {"model": {"provider": "p", "modelId": "m"}}),
                set_value(lane_state("main"), {"currentOperationId": None, "lastOperationId": None, "inbox": []}),
            ],
            mutation_context,
        )

    await session.mutate(_configure, context)
    first_id = await main.append_message(user_message("one"), context)
    second_id = await main.append_message(user_message("two"), context)

    forked = await repo.fork(
        session.metadata,
        ForkOptions(scope="branch", branch="main"),
        context,
    )
    forked_main = await forked.branch("main", context)
    assert await forked_main.get_tip_id(context) == second_id
    entries = await forked_main.find_entries(None, context)
    assert len(entries) == 2

    # Fork at earlier position.
    earlier = await repo.fork(
        session.metadata,
        ForkOptions(scope="branch", branch="main", entry_id=first_id),
        context,
    )
    earlier_main = await earlier.branch("main", context)
    assert await earlier_main.get_tip_id(context) == first_id
    assert len(await earlier_main.find_entries(None, context)) == 1


async def test_fork_tree_copies_everything():
    repo = MemorySessionRepo()
    context = BACKGROUND_CONTEXT
    session = await repo.create(None, context)
    main = await session.create_branch("main", None, context)
    await main.append_message(user_message("one"), context)

    forked = await repo.fork(session.metadata, ForkOptions(scope="tree"), context)
    forked_main = await forked.branch("main", context)
    assert len(await forked_main.find_entries(None, context)) == 1
    assert forked.metadata.parent_session_id == session.metadata.id


async def test_repo_open_list_delete():
    repo = MemorySessionRepo()
    context = BACKGROUND_CONTEXT
    session = await repo.create(None, context)
    await session.close(context)

    listed = await repo.list(None, context)
    assert len(listed) == 1

    reopened = await repo.open(listed[0], context)
    await reopened.close(context)
    await repo.delete(listed[0], context)
    assert await repo.list(None, context) == []

    with pytest.raises(RuntimeError, match="Unknown session"):
        await repo.open(listed[0], context)
