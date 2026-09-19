"""Tests for JSONL session persistence (storage, repo, fork)."""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core._pi_ai.types import TextContent, UserMessage
from pi_agent_core.harness.env.local import create_local_execution_env
from pi_agent_core.harness.session import (
    CommitResult,
    CustomEntry,
    ForkOptions,
    LaneConfiguration,
    LaneState,
    MessageEntry,
    SessionMetadata,
    insert_entry,
    insert_usage,
    set_value,
    value,
)
from pi_agent_core.harness.session.jsonl import (
    JSONL_FORMAT_VERSION,
    JSONL_STORAGE_VERSION,
    JsonlSessionCreateOptions,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
    JsonlStorage,
    JsonlStorageHeader,
    JsonlStorageOptions,
    parse_jsonl_session_header,
    parse_jsonl_transaction,
    serialize_jsonl_transaction,
)
from pi_agent_core.harness.session.jsonl.io import publish_file_atomically
from pi_agent_core.harness.session.jsonl.repo import session_directory_name, session_file_name
from pi_agent_core.harness.session.values import session_name


def user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=int(time.time() * 1000))


def make_header(session_id: str = "s1", cwd: str = "/tmp/work") -> JsonlStorageHeader:
    return JsonlStorageHeader(
        v=JSONL_FORMAT_VERSION,
        kind="header",
        id=session_id,
        storage_version=JSONL_STORAGE_VERSION,
        created_at=int(time.time() * 1000),
        cwd=cwd,
    )


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def test_parse_jsonl_session_header_v4_and_v3():
    header = make_header()
    parsed = parse_jsonl_session_header(json.dumps(header.to_json()))
    assert parsed.ok
    assert parsed.value.format == "v4"
    assert parsed.value.header.id == "s1"

    legacy = {
        "type": "session",
        "version": 3,
        "id": "legacy-1",
        "timestamp": "2024-01-01T00:00:00.000Z",
        "cwd": "/tmp/work",
    }
    parsed = parse_jsonl_session_header(json.dumps(legacy))
    assert parsed.ok
    assert parsed.value.format == "v3-legacy"
    assert parsed.value.header.id == "legacy-1"

    assert not parse_jsonl_session_header("not json").ok
    assert not parse_jsonl_session_header('{"kind": "other"}').ok


def test_transaction_roundtrip():
    from pi_agent_core.harness.session.commit import CommittedWrite

    writes = [
        CommittedWrite(kind="value", op="set", seq=1, namespace="app", key="k", value={"a": 1}),
        CommittedWrite(kind="list", op="append", seq=2, namespace="app", key="l", value="x"),
    ]
    line = serialize_jsonl_transaction(writes)
    assert line.startswith("[")  # multiple writes serialize as one JSON array
    parsed = parse_jsonl_transaction(line)
    assert [w.kind for w in parsed] == ["value", "list"]
    assert parsed[0].value == {"a": 1}

    single = serialize_jsonl_transaction([writes[0]])
    assert json.loads(single)["kind"] == "value"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


async def test_storage_create_commit_reopen(workdir):
    env = create_local_execution_env(cwd=workdir)
    path = os.path.join(workdir, "session.jsonl")
    context = BACKGROUND_CONTEXT

    storage = await JsonlStorage.create(
        JsonlStorageOptions(file_system=env, path=path),
        make_header("s1", workdir),
        [],
        context,
    )
    result = await storage.commit(
        [
            insert_entry(MessageEntry(id="e1", parent_id=None, message=user_message("hello"))),
            set_value(session_name(), "my session"),
        ],
        context,
    )
    assert isinstance(result, CommitResult)
    assert result.first_seq == 1
    assert result.stats.message_count == 1
    await storage.close(context)

    # File layout: header line + one transaction line.
    raw = open(path).read().strip().split("\n")
    assert len(raw) == 2
    assert json.loads(raw[0])["kind"] == "header"

    reopened = await JsonlStorage.open(JsonlStorageOptions(file_system=env, path=path), context)
    entries = await reopened.get_entries(["e1"], context)
    assert "e1" in entries
    assert entries["e1"].message.content == "hello"
    stored_value = await reopened.get_value(session_name(), context)
    assert stored_value.value == "my session"
    stats = await reopened.get_stats(context)
    assert stats.message_count == 1
    await reopened.close(context)


async def test_storage_repairs_torn_trailing_line(workdir):
    env = create_local_execution_env(cwd=workdir)
    path = os.path.join(workdir, "torn.jsonl")
    context = BACKGROUND_CONTEXT

    storage = await JsonlStorage.create(
        JsonlStorageOptions(file_system=env, path=path), make_header("s1", workdir), [], context
    )
    await storage.commit(
        [insert_entry(MessageEntry(id="e1", parent_id=None, message=user_message("one")))], context
    )
    await storage.close(context)

    # Simulate a partially written trailing transaction.
    with open(path, "a") as handle:
        handle.write('{"kind": "value", "seq": 2, "op": "set", "namespace": "app", "key": "hal')

    repaired = await JsonlStorage.open(JsonlStorageOptions(file_system=env, path=path), context)
    assert (await repaired.get_entries(["e1"], context))["e1"].message.content == "one"
    # The torn line was dropped and the file rewritten without it.
    assert open(path).read().endswith("\n")
    assert "hal" not in open(path).read()
    await repaired.close(context)


async def test_storage_rejects_bad_version_and_closed_use(workdir):
    env = create_local_execution_env(cwd=workdir)
    path = os.path.join(workdir, "bad.jsonl")
    context = BACKGROUND_CONTEXT

    header = make_header("s1", workdir)
    header.storage_version = 99
    with open(path, "w") as handle:
        handle.write(f"{json.dumps(header.to_json())}\n")

    with pytest.raises(RuntimeError, match="unsupported storage version"):
        await JsonlStorage.open(JsonlStorageOptions(file_system=env, path=path), context)

    storage = await JsonlStorage.create(
        JsonlStorageOptions(file_system=env, path=os.path.join(workdir, "ok.jsonl")),
        make_header("s2", workdir),
        [],
        context,
    )
    await storage.close(context)
    with pytest.raises(RuntimeError, match="closed"):
        await storage.get_stats(context)


async def test_storage_usage_rows_accumulate(workdir):
    from pi_agent_core._pi_ai.types import Usage
    from pi_agent_core.harness.session import UsageRow

    env = create_local_execution_env(cwd=workdir)
    path = os.path.join(workdir, "usage.jsonl")
    context = BACKGROUND_CONTEXT

    storage = await JsonlStorage.create(
        JsonlStorageOptions(file_system=env, path=path), make_header("s1", workdir), [], context
    )
    await storage.commit(
        [insert_usage(UsageRow(id="u1", usage=Usage(input=10, output=20, total_tokens=30)))], context
    )
    stats = await storage.get_stats(context)
    assert stats.usage.input == 10
    rows = await storage.scan_usage(type("Q", (), {"from_seq": None, "to_seq": None, "order": "asc", "limit": None})(), context)
    assert [row.id for row in rows] == ["u1"]
    await storage.close(context)


# ---------------------------------------------------------------------------
# Repo
# ---------------------------------------------------------------------------


async def test_repo_create_list_open_delete(workdir):
    env = create_local_execution_env(cwd=workdir)
    root = os.path.join(workdir, "sessions")
    repo = JsonlSessionRepo(JsonlSessionRepoOptions(file_system=env, sessions_root=root))
    context = BACKGROUND_CONTEXT

    session = await repo.create(JsonlSessionCreateOptions(cwd=workdir, id="session-a"), context)
    assert session.metadata.id == "session-a"
    assert session.metadata.cwd == os.path.realpath(workdir) or session.metadata.cwd == workdir

    main = await session.create_branch("main", None, context)
    await main.append_message(user_message("persisted"), context)
    path = session.metadata.path
    assert os.path.exists(path)
    await session.close(context)

    listed = await repo.list(None, context)
    assert [m.id for m in listed] == ["session-a"]

    filtered = await repo.list(type("O", (), {"cwd": workdir})(), context)
    assert [m.id for m in filtered] == ["session-a"]

    reopened = await repo.open(listed[0], context)
    reopened_main = await reopened.branch("main", context)
    entries = await reopened_main.find_entries(None, context)
    assert len(entries) == 1
    assert entries[0].message.content == "persisted"
    await reopened.close(context)

    await repo.delete(listed[0], context)
    assert await repo.list(None, context) == []
    assert not os.path.exists(path)


async def test_repo_lists_across_directories_and_filters_by_cwd(workdir):
    env = create_local_execution_env(cwd=workdir)
    root = os.path.join(workdir, "sessions")
    repo = JsonlSessionRepo(JsonlSessionRepoOptions(file_system=env, sessions_root=root))
    context = BACKGROUND_CONTEXT

    dir_a = os.path.join(workdir, "a")
    dir_b = os.path.join(workdir, "b")
    os.makedirs(dir_a)
    os.makedirs(dir_b)
    await repo.create(JsonlSessionCreateOptions(cwd=dir_a, id="in-a"), context)
    await repo.create(JsonlSessionCreateOptions(cwd=dir_b, id="in-b"), context)

    all_sessions = await repo.list(None, context)
    assert {m.id for m in all_sessions} == {"in-a", "in-b"}

    only_a = await repo.list(type("O", (), {"cwd": dir_a})(), context)
    assert [m.id for m in only_a] == ["in-a"]


async def test_repo_fork_branch_scope_copies_path(workdir):
    env = create_local_execution_env(cwd=workdir)
    root = os.path.join(workdir, "sessions")
    repo = JsonlSessionRepo(JsonlSessionRepoOptions(file_system=env, sessions_root=root))
    context = BACKGROUND_CONTEXT

    session = await repo.create(JsonlSessionCreateOptions(cwd=workdir, id="source"), context)

    async def _configure(mutator, mutation_context):
        await mutator.commit(
            [
                set_value(
                    value("pi.lane.config", "main"),
                    {"model": {"provider": "faux", "modelId": "faux-1"}},
                ),
                set_value(
                    value("pi.lane.state", "main"),
                    {"currentOperationId": None, "lastOperationId": None, "inbox": []},
                ),
            ],
            mutation_context,
        )

    await session.mutate(_configure, context)
    main = await session.create_branch("main", None, context)
    first = await main.append_message(user_message("first"), context)
    second = await main.append_message(user_message("second"), context)
    await session.set_name("source session", context)
    await session.close(context)

    source_metadata = (await repo.list(None, context))[0]
    forked = await repo.fork(
        source_metadata, ForkOptions(scope="branch", branch="main", id="copy"), context
    )
    assert forked.metadata.id == "copy"
    assert forked.metadata.parent_session_id == "source"

    forked_main = await forked.branch("main", context)
    assert await forked_main.get_tip_id(context) == second
    entries = await forked_main.find_entries(None, context)
    assert [e.message.content for e in reversed(entries)] == ["first", "second"]
    assert await forked.get_name(context) == "source session"
    await forked.close(context)

    # Forking earlier keeps only the earlier prefix.
    earlier = await repo.fork(
        source_metadata,
        ForkOptions(scope="branch", branch="main", entry_id=first, id="earlier"),
        context,
    )
    earlier_main = await earlier.branch("main", context)
    assert await earlier_main.get_tip_id(context) == first
    assert len(await earlier_main.find_entries(None, context)) == 1
    await earlier.close(context)


async def test_repo_rejects_duplicate_ids_and_closed_repo(workdir):
    env = create_local_execution_env(cwd=workdir)
    root = os.path.join(workdir, "sessions")
    repo = JsonlSessionRepo(JsonlSessionRepoOptions(file_system=env, sessions_root=root))
    context = BACKGROUND_CONTEXT

    session = await repo.create(JsonlSessionCreateOptions(cwd=workdir, id="dup"), context)
    with pytest.raises(RuntimeError, match="already exists"):
        await repo.create(JsonlSessionCreateOptions(cwd=workdir, id="dup"), context)
    await session.close(context)

    await repo.close(context)
    with pytest.raises(RuntimeError, match="closed"):
        await repo.list(None, context)


def test_session_path_helpers():
    assert session_directory_name("/a/b") == "--a-b--"
    name = session_file_name(1700000000000, "session-1")
    assert name == "1700000000000_session-1.jsonl"
