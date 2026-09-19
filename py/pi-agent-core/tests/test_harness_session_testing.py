"""Tests for the ported ``harness/session/testing`` subsystem.

The subsystem is ported from ``packages/agent/src/harness/session/testing``:
the test-only ``Storage`` decorators, the benchmark workloads, and the
runner-independent conformance tables. These tests run the decorators against a
real in-memory ``Storage`` and run every ported conformance case against two
real fixtures (``MemoryStorage``/``MemorySessionRepo`` and
``JsonlStorage``/``JsonlSessionRepo``).
"""

from __future__ import annotations

import asyncio
import tempfile
from typing import Awaitable, Callable, List, Optional

import pytest

from pi_agent_core._pi_ai.types import TextContent, UserMessage
from pi_agent_core.harness.context import BACKGROUND_CONTEXT as CTX
from pi_agent_core.harness.env.local import create_local_execution_env
from pi_agent_core.harness.session.commit import insert_entry
from pi_agent_core.harness.session.jsonl import (
    JSONL_FORMAT_VERSION,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
    JsonlStorage,
    JsonlStorageHeader,
    JsonlStorageOptions,
)
from pi_agent_core.harness.session.memory import MemorySessionRepo, MemoryStorage
from pi_agent_core.harness.session.types import (
    EntryScan,
    ForkOptions,
    MessageEntry,
    StorageBranchScan,
    UsageScan,
)
from pi_agent_core.harness.session.values import branch_tip, lane_config, set_value, value
from pi_agent_core.harness.session import testing

NOW = 1_700_000_000_000

#: Cases blocked by a pre-existing divergence in the Python memory backend:
#: ``MemorySessionRepo`` hands out a ``StorageBackedSession`` whose ``close`` closes
#: the shared ``MemoryStorage``, while the TypeScript backend wraps it in a
#: ``MemorySessionFacade`` that keeps the storage open. Every case below closes a
#: source session and then reopens or forks it, so it passes against the JSONL
#: fixture but cannot pass against the memory fixture until that facade is ported.
# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


_current_memory_repo: Optional[MemorySessionRepo] = None
_pending_cleanups: List[Callable[[], Awaitable[None]]] = []


async def _memory_storage_fixture() -> testing.StorageFixture:
    storage = MemoryStorage(now=lambda: NOW)
    return testing.StorageFixture(storage=storage, dispose=lambda: storage.close(CTX))


async def _create_memory_repo() -> MemorySessionRepo:
    global _current_memory_repo
    _current_memory_repo = MemorySessionRepo(now=lambda: NOW)
    return _current_memory_repo


async def _close_memory_repo() -> None:
    global _current_memory_repo
    assert _current_memory_repo is not None
    repo, _current_memory_repo = _current_memory_repo, None
    await repo.close(CTX)


class JsonlRepoAdapter:
    """Scope every repository call to one conformance cwd, mirroring the TypeScript adapter."""

    def __init__(self, repo: JsonlSessionRepo, cwd: str) -> None:
        self._repo = repo
        self._cwd = cwd

    async def create(self, options, context):
        return await self._repo.create(
            JsonlSessionCreateOptions(
                cwd=self._cwd,
                id=None if options is None else options.id,
                parent_session_id=None if options is None else options.parent_session_id,
            ),
            context,
        )

    async def open(self, metadata, context):
        return await self._repo.open(metadata, context)

    async def list(self, options, context):
        return await self._repo.list(JsonlSessionListOptions(cwd=self._cwd), context)

    async def delete(self, metadata, context):
        return await self._repo.delete(metadata, context)

    async def fork(self, source, options, context):
        return await self._repo.fork(source, options, context)


async def _jsonl_storage_fixture() -> testing.StorageFixture:
    root = tempfile.TemporaryDirectory()
    env = create_local_execution_env(cwd=root.name)
    storage = await JsonlStorage.create(
        JsonlStorageOptions(file_system=env, path="session.jsonl", now=lambda: NOW),
        JsonlStorageHeader(
            v=JSONL_FORMAT_VERSION,
            kind="header",
            id="session",
            storage_version=1,
            created_at=NOW,
            cwd=root.name,
        ),
        [],
        CTX,
    )

    async def dispose() -> None:
        await storage.close(CTX)
        root.cleanup()

    return testing.StorageFixture(storage=storage, dispose=dispose)


async def _create_jsonl_repo() -> JsonlRepoAdapter:
    root = tempfile.TemporaryDirectory()
    env = create_local_execution_env(cwd=root.name)
    repo = JsonlSessionRepo(
        JsonlSessionRepoOptions(file_system=env, sessions_root="sessions", now=lambda: NOW)
    )

    async def cleanup() -> None:
        await repo.close(CTX)
        root.cleanup()

    _pending_cleanups.append(cleanup)
    return JsonlRepoAdapter(repo, root.name)


async def _close_jsonl_repo() -> None:
    cleanup = _pending_cleanups.pop()
    await cleanup()


def _memory_conformance_cases() -> List[testing.ConformanceCase]:
    cases = list(testing.create_storage_conformance(_memory_storage_fixture))
    cases += testing.create_session_repo_conformance(_create_memory_repo, _close_memory_repo)
    cases += testing.create_session_repo_streaming_fork_conformance(
        _create_memory_repo, _close_memory_repo
    )
    return cases


def _jsonl_conformance_cases() -> List[testing.ConformanceCase]:
    cases = list(testing.create_storage_conformance(_jsonl_storage_fixture))
    cases += testing.create_session_repo_conformance(_create_jsonl_repo, _close_jsonl_repo)
    cases += testing.create_session_repo_streaming_fork_conformance(
        _create_jsonl_repo, _close_jsonl_repo
    )
    return cases


def _case_params(cases: List[testing.ConformanceCase]):
    """One pytest parameter per conformance case, identified by group and name."""
    return [
        pytest.param(case, id=f"{case.group}::{case.name}") for case in cases
    ]


# ---------------------------------------------------------------------------
# conformance tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _case_params(_memory_conformance_cases()))
async def test_memory_conformance(case: testing.ConformanceCase) -> None:
    await case.run()


@pytest.mark.parametrize("case", _case_params(_jsonl_conformance_cases()))
async def test_jsonl_conformance(case: testing.ConformanceCase) -> None:
    await case.run()


def test_conformance_case_tables() -> None:
    storage_cases = testing.create_storage_conformance(_memory_storage_fixture)
    repo_cases = testing.create_session_repo_conformance(_create_memory_repo, _close_memory_repo)
    streaming = testing.create_session_repo_streaming_fork_conformance(
        _create_memory_repo, _close_memory_repo
    )
    fork = testing.create_session_repo_fork_conformance(_create_memory_repo, _close_memory_repo)
    coordination = testing.create_session_repo_fork_coordination_conformance(
        _create_memory_repo, _close_memory_repo
    )

    assert len(storage_cases) == 21
    assert len(repo_cases) == len(
        testing.create_session_repo_lifecycle_conformance(_create_memory_repo, _close_memory_repo)
    ) + len(
        testing.create_session_repo_ownership_conformance(_create_memory_repo, _close_memory_repo)
    ) + len(
        testing.create_session_repo_message_conformance(_create_memory_repo, _close_memory_repo)
    ) + len(fork)
    assert len(streaming) == 15
    assert len(coordination) == 3
    assert {case.group for case in storage_cases} == {
        "transactions",
        "values",
        "lists",
        "entry queries",
        "branch queries",
        "usage and stats",
        "serialization",
        "lifecycle",
    }
    assert all(isinstance(case, testing.ConformanceCase) for case in storage_cases + repo_cases)
    assert all(case.name and case.group for case in storage_cases + repo_cases)


# ---------------------------------------------------------------------------
# commit instrumentation and gating
# ---------------------------------------------------------------------------


async def test_instrumented_storage_records_commit_attempts() -> None:
    storage = InstrumentedStorageFixture()
    delegate = storage.delegate
    instrumented = storage.instrumented

    first = [insert_entry(_message_entry("first"))]
    second = [insert_entry(_message_entry("second", "first")), set_value(branch_tip("main"), "second")]

    await instrumented.commit(first, CTX)
    await instrumented.commit(second, CTX)

    attempts = instrumented.get_commit_attempts()
    assert len(attempts) == 2
    assert attempts[0] is first
    assert attempts[1] is second
    assert await delegate.get_entries(["first", "second"], CTX) != {}

    # Failed transactions are admitted too: the recorder wraps admission, not success.
    await _raises(lambda: instrumented.commit([insert_entry(_message_entry("first"))], CTX))
    assert len(instrumented.get_commit_attempts()) == 3

    instrumented.clear_commit_attempts()
    assert instrumented.get_commit_attempts() == []
    await instrumented.close(CTX)


async def test_gating_storage_parks_and_releases_commits() -> None:
    delegate = MemoryStorage(now=lambda: NOW)
    gating = testing.GatingStorage(delegate)
    gating.arm()

    first = asyncio.ensure_future(gating.commit([insert_entry(_message_entry("first"))], CTX))
    second = asyncio.ensure_future(
        gating.commit([insert_entry(_message_entry("second", "first"))], CTX)
    )
    await gating.wait_pending(2)

    assert gating.pending() == 2
    assert await delegate.get_entries(["first", "second"], CTX) == {}

    await gating.next()
    first_result = await first
    assert len(first_result.seqs) == 1
    assert gating.pending() == 1
    assert set(await delegate.get_entries(["first"], CTX)) == {"first"}
    assert await delegate.get_entries(["second"], CTX) == {}

    await gating.next()
    second_result = await second
    assert second_result.seqs[0] > first_result.seqs[0]
    assert gating.pending() == 0
    assert set(await delegate.get_entries(["first", "second"], CTX)) == {"first", "second"}
    await gating.close(CTX)


async def test_gating_storage_forwards_before_arm() -> None:
    delegate = MemoryStorage(now=lambda: NOW)
    gating = testing.GatingStorage(delegate)

    result = await gating.commit([insert_entry(_message_entry("direct"))], CTX)

    assert len(result.seqs) == 1
    assert gating.pending() == 0
    assert set(await delegate.get_entries(["direct"], CTX)) == {"direct"}
    await gating.close(CTX)


async def test_gating_storage_discard_rejects_every_commit() -> None:
    delegate = MemoryStorage(now=lambda: NOW)
    gating = testing.GatingStorage(delegate)
    gating.arm()

    parked = asyncio.ensure_future(gating.commit([insert_entry(_message_entry("parked"))], CTX))
    await gating.wait_pending()
    gating.discard()
    gating.discard()

    with pytest.raises(testing.CommitDiscarded) as parked_error:
        await parked
    assert "discarded" in str(parked_error.value)
    assert parked_error.value.name == "CommitDiscarded"

    with pytest.raises(testing.CommitDiscarded):
        await gating.commit([insert_entry(_message_entry("after"))], CTX)
    with pytest.raises(testing.CommitDiscarded):
        await gating.wait_pending()
    assert await delegate.get_entries(["parked", "after"], CTX) == {}
    await gating.close(CTX)


async def test_gating_storage_validates_counts() -> None:
    delegate = MemoryStorage(now=lambda: NOW)
    gating = testing.GatingStorage(delegate)

    with pytest.raises(ValueError):
        await gating.wait_pending(0)
    with pytest.raises(ValueError):
        await gating.next(0)
    with pytest.raises(ValueError):
        await gating.wait_pending(True)  # type: ignore[arg-type]
    await gating.close(CTX)


async def test_storage_decorator_forwards_every_operation() -> None:
    delegate = MemoryStorage(now=lambda: NOW)
    decorator = testing.StorageDecorator(delegate)

    commit = await decorator.commit(
        [
            insert_entry(_message_entry("entry")),
            set_value(value("test.value", "a"), 1),
        ],
        CTX,
    )
    assert len(commit.seqs) == 2
    assert set(await decorator.get_entries(["entry"], CTX)) == {"entry"}
    stored = await decorator.get_value(value("test.value", "a"), CTX)
    assert stored is not None and stored.value == 1
    assert [item.address.key for item in await decorator.scan_values(value("test.value", ""), CTX)] == ["a"]
    assert await decorator.read_list(value("test.list", "none"), None, CTX) == []
    assert [
        entry.id
        for entry in await decorator.scan_branch(StorageBranchScan(start="entry"), CTX)
    ] == ["entry"]
    assert [
        entry.id
        for entry in await decorator.scan_branch_structure(StorageBranchScan(start="entry"), CTX)
    ] == ["entry"]
    assert [
        entry.id for entry in await decorator.scan_entries(EntryScan(order="asc"), CTX)
    ] == ["entry"]
    assert await decorator.scan_usage(UsageScan(order="asc"), CTX) == []
    assert (await decorator.get_stats(CTX)).message_count == 1
    await decorator.close(CTX)

    with pytest.raises(RuntimeError):
        await delegate.get_stats(CTX)


# ---------------------------------------------------------------------------
# benchmark workloads
# ---------------------------------------------------------------------------


async def test_storage_benchmark_datasets_are_deterministic() -> None:
    datasets = testing.STORAGE_BENCHMARK_DATASETS
    assert [dataset.entry_count for dataset in datasets] == [1_000, 10_000, 100_000]
    assert [dataset.name for dataset in datasets] == [
        "synthetic linear branch: 1k entries, 256-byte payloads",
        "synthetic linear branch: 10k entries, 256-byte payloads",
        "synthetic linear branch: 100k entries, 256-byte payloads",
    ]
    assert all(dataset.payload_bytes == 256 for dataset in datasets)

    smallest = datasets[0]
    assert smallest.tip_id == "benchmark-entry-00000999"
    assert len(smallest.lookup_ids) == 100
    assert smallest.lookup_ids[0] == "benchmark-entry-00000000"
    assert smallest.lookup_ids[-1] == "benchmark-entry-00000999"
    assert testing.session_repo_benchmark_session_id(7) == "benchmark-session-00000007"


async def test_seed_storage_benchmark_feeds_read_and_write_scenarios() -> None:
    dataset = testing.STORAGE_BENCHMARK_DATASETS[0]
    storage = MemoryStorage(now=lambda: NOW)
    await testing.seed_storage_benchmark(storage, dataset)

    assert (await storage.get_stats(CTX)).message_count == dataset.entry_count
    # Seeding writes entries only; the branch tip value is the seed caller's concern.
    assert await storage.get_value(branch_tip("main"), CTX) is None
    assert len(
        await storage.scan_entries(EntryScan(order="desc", limit=1), CTX)
    ) == 1
    assert (await storage.scan_entries(EntryScan(order="desc", limit=1), CTX))[0].id == dataset.tip_id
    for scenario in testing.STORAGE_READ_BENCHMARK_SCENARIOS:
        result = await scenario.run(storage, dataset)
        assert result == scenario.expected_result(dataset)

    assert [scenario.name for scenario in testing.STORAGE_WRITE_BENCHMARK_SCENARIOS] == [
        "commit one message entry",
        "commit 100 message entries",
        f"commit mixed append ({dataset.name})",
    ]
    for scenario in testing.STORAGE_WRITE_BENCHMARK_SCENARIOS:
        target = MemoryStorage(now=lambda: NOW)
        if scenario.prepare is not None:
            await scenario.prepare(target)
        assert await scenario.run(target) == scenario.write_count
        await target.close(CTX)
    await storage.close(CTX)


async def test_generate_storage_benchmark_seed_transactions_batches() -> None:
    dataset = testing.STORAGE_BENCHMARK_DATASETS[0]
    transactions = list(testing.generate_storage_benchmark_seed_transactions(dataset))
    assert [len(transaction) for transaction in transactions] == [250, 250, 250, 250]

    fork_datasets = testing.SESSION_REPO_FORK_BENCHMARK_DATASETS
    assert [dataset.entry_count for dataset in fork_datasets] == [1_000, 10_000]


async def test_session_repo_catalog_read_benchmark() -> None:
    dataset = testing.SESSION_REPO_CATALOG_BENCHMARK_DATASETS[0]
    assert dataset.session_count == 100
    assert dataset.name == "synthetic catalog: 100 closed sessions"
    assert testing.SESSION_REPO_CATALOG_READ_BENCHMARK_SCENARIOS[0].expected_result(dataset) == 100

    repo = MemorySessionRepo(now=lambda: NOW)
    metadata = await testing.seed_session_repo_catalog_benchmark(repo, dataset)
    assert len(metadata) == 100
    assert metadata[0].id == "benchmark-session-00000000"
    assert metadata[-1].id == "benchmark-session-00000099"
    for scenario in testing.SESSION_REPO_CATALOG_READ_BENCHMARK_SCENARIOS:
        assert await scenario.run(repo) == scenario.expected_result(dataset)
    await repo.close(CTX)


async def test_session_repo_catalog_write_benchmark() -> None:
    scenarios = testing.SESSION_REPO_CATALOG_WRITE_BENCHMARK_SCENARIOS
    assert [scenario.name for scenario in scenarios] == [
        "create empty session",
        "open closed empty session",
        "delete closed empty session",
    ]
    assert all(scenario.expected_result == 1 for scenario in scenarios)

    # Every write scenario owns one independently prepared, initially empty repository.
    for scenario in scenarios:
        repo = MemorySessionRepo(now=lambda: NOW)
        operation = await scenario.prepare(repo)
        assert await operation.run() == scenario.expected_result
        await repo.close(CTX)


async def test_session_repo_fork_benchmark() -> None:
    dataset = testing.SESSION_REPO_FORK_BENCHMARK_DATASETS[0]
    repo = MemorySessionRepo(now=lambda: NOW)
    source = await testing.seed_session_repo_fork_benchmark(repo, dataset)

    assert source.id == "benchmark-session-00000000"
    assert source.parent_session_id is None
    assert len(await repo.list(None, CTX)) == 1
    for scenario in testing.SESSION_REPO_FORK_WRITE_BENCHMARK_SCENARIOS:
        assert await scenario.run(repo, source, dataset) == scenario.expected_result(dataset)

    forks = [metadata for metadata in await repo.list(None, CTX) if metadata.id != source.id]
    assert len(forks) == 1
    assert forks[0].id == "benchmark-session-00000001"
    assert forks[0].parent_session_id == source.id

    # The seeded source carries the whole synthetic branch and its lane configuration.
    check = await repo.fork(
        source,
        ForkOptions(id="benchmark-check", scope="branch", branch="main"),
        CTX,
    )
    assert (await check.get_stats(CTX)).message_count == dataset.entry_count
    assert (await check.get_value(branch_tip("main"), CTX)).value == dataset.tip_id
    assert (await check.get_value(lane_config("main"), CTX)).value["model"]["modelId"] == "benchmark"
    await repo.close(CTX)


# ---------------------------------------------------------------------------
# barrel surface
# ---------------------------------------------------------------------------


def test_testing_barrel_exports() -> None:
    assert sorted(testing.__all__) == [
        "CommitDiscarded",
        "ConformanceCase",
        "GatingStorage",
        "InstrumentedStorage",
        "SESSION_REPO_CATALOG_BENCHMARK_DATASETS",
        "SESSION_REPO_CATALOG_READ_BENCHMARK_SCENARIOS",
        "SESSION_REPO_CATALOG_WRITE_BENCHMARK_SCENARIOS",
        "SESSION_REPO_FORK_BENCHMARK_DATASETS",
        "SESSION_REPO_FORK_WRITE_BENCHMARK_SCENARIOS",
        "STORAGE_BENCHMARK_DATASETS",
        "STORAGE_READ_BENCHMARK_SCENARIOS",
        "STORAGE_WRITE_BENCHMARK_SCENARIOS",
        "SessionRepoCatalogBenchmarkDataset",
        "StorageDecorator",
        "StorageFixture",
        "create_session_repo_conformance",
        "create_session_repo_fork_behavior_conformance",
        "create_session_repo_fork_conformance",
        "create_session_repo_fork_coordination_conformance",
        "create_session_repo_fork_destination_reservation_conformance",
        "create_session_repo_fork_source_snapshot_conformance",
        "create_session_repo_lifecycle_conformance",
        "create_session_repo_message_conformance",
        "create_session_repo_ownership_conformance",
        "create_session_repo_streaming_fork_conformance",
        "create_storage_conformance",
        "generate_storage_benchmark_seed_transactions",
        "seed_session_repo_catalog_benchmark",
        "seed_session_repo_fork_benchmark",
        "seed_storage_benchmark",
        "session_repo_benchmark_session_id",
    ]
    assert all(hasattr(testing, name) for name in testing.__all__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class InstrumentedStorageFixture:
    """A delegate plus the instrumented view used by the decorator tests."""

    def __init__(self) -> None:
        self.delegate = MemoryStorage(now=lambda: NOW)
        self.instrumented = testing.InstrumentedStorage(self.delegate)


def _message_entry(entry_id: str, parent_id: Optional[str] = None) -> MessageEntry:
    return MessageEntry(
        id=entry_id,
        parent_id=parent_id,
        message=UserMessage(content=[TextContent(text=entry_id)], timestamp=1),
    )


async def _raises(operation: Callable[[], Awaitable[object]]) -> None:
    with pytest.raises(Exception):
        await operation()
