"""Session repository benchmark workloads ported from ``session/testing/benchmark/session-repo.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, List

from ....context import BACKGROUND_CONTEXT
from ...types import (
    ForkOptions,
    SessionCreateOptions,
    SessionMetadata,
    SessionRepo,
)
from ...values import branch_tip, lane_config, lane_state, set_value
from .datasets import STORAGE_BENCHMARK_DATASETS, StorageBenchmarkDataset
from .storage import generate_storage_benchmark_seed_transactions

__all__ = [
    "SessionRepoCatalogBenchmarkDataset",
    "SessionRepoCatalogWriteBenchmarkOperation",
    "SessionRepoCatalogReadBenchmarkScenario",
    "SessionRepoCatalogWriteBenchmarkScenario",
    "SessionRepoForkWriteBenchmarkScenario",
    "session_repo_benchmark_session_id",
    "SESSION_REPO_CATALOG_BENCHMARK_DATASETS",
    "seed_session_repo_catalog_benchmark",
    "SESSION_REPO_CATALOG_READ_BENCHMARK_SCENARIOS",
    "SESSION_REPO_CATALOG_WRITE_BENCHMARK_SCENARIOS",
    "SESSION_REPO_FORK_BENCHMARK_DATASETS",
    "seed_session_repo_fork_benchmark",
    "SESSION_REPO_FORK_WRITE_BENCHMARK_SCENARIOS",
]


@dataclass
class SessionRepoCatalogBenchmarkDataset:
    """One deterministic closed-session catalog shared by repository measurements."""

    name: str
    session_count: int


@dataclass
class SessionRepoCatalogWriteBenchmarkOperation:
    """One prepared write operation run against an already prepared repository."""

    run: Callable[[], Awaitable[int]]


@dataclass
class SessionRepoCatalogReadBenchmarkScenario:
    """A steady-state catalog read operation run against a pre-seeded repository."""

    name: str
    expected_result: Callable[[SessionRepoCatalogBenchmarkDataset], int]
    run: Callable[[SessionRepo], Awaitable[int]]


@dataclass
class SessionRepoCatalogWriteBenchmarkScenario:
    """A catalog write operation with one repository-local preparation step."""

    name: str
    expected_result: int
    prepare: Callable[[SessionRepo], Awaitable[SessionRepoCatalogWriteBenchmarkOperation]]


@dataclass
class SessionRepoForkWriteBenchmarkScenario:
    """A fork write operation run once against each independently prepared source."""

    name: str
    expected_result: Callable[[StorageBenchmarkDataset], int]
    run: Callable[[SessionRepo, SessionMetadata, StorageBenchmarkDataset], Awaitable[int]]


def session_repo_benchmark_session_id(index: int) -> str:
    """Package-internal deterministic session id shared by repository benchmark workloads."""
    return f"benchmark-session-{index:08d}"


_BENCHMARK_SESSION_ID = session_repo_benchmark_session_id(0)
_FORK_DESTINATION_SESSION_ID = session_repo_benchmark_session_id(1)


def _create_catalog_dataset(scale: str, session_count: int) -> SessionRepoCatalogBenchmarkDataset:
    return SessionRepoCatalogBenchmarkDataset(
        name=f"synthetic catalog: {scale} closed sessions",
        session_count=session_count,
    )


#: Deterministic closed-session catalogs shared by repository measurements.
SESSION_REPO_CATALOG_BENCHMARK_DATASETS: List[SessionRepoCatalogBenchmarkDataset] = [
    _create_catalog_dataset("100", 100),
    _create_catalog_dataset("1k", 1_000),
    _create_catalog_dataset("10k", 10_000),
]


async def seed_session_repo_catalog_benchmark(
    repo: SessionRepo, dataset: SessionRepoCatalogBenchmarkDataset
) -> List[SessionMetadata]:
    """Seed one deterministic catalog and return its durable metadata in creation order."""
    metadata: List[SessionMetadata] = []
    for index in range(dataset.session_count):
        session = await repo.create(
            SessionCreateOptions(id=session_repo_benchmark_session_id(index)), BACKGROUND_CONTEXT
        )
        metadata.append(session.metadata)
        await session.close(BACKGROUND_CONTEXT)
    return metadata


async def _list_sessions(repo: SessionRepo) -> int:
    return len(await repo.list(None, BACKGROUND_CONTEXT))


async def _prepare_create_empty_session(
    repo: SessionRepo,
) -> SessionRepoCatalogWriteBenchmarkOperation:
    async def run() -> int:
        session = await repo.create(SessionCreateOptions(id=_BENCHMARK_SESSION_ID), BACKGROUND_CONTEXT)
        return 1 if session.metadata.id == _BENCHMARK_SESSION_ID else 0

    return SessionRepoCatalogWriteBenchmarkOperation(run=run)


async def _prepare_open_closed_empty_session(
    repo: SessionRepo,
) -> SessionRepoCatalogWriteBenchmarkOperation:
    session = await repo.create(SessionCreateOptions(id=_BENCHMARK_SESSION_ID), BACKGROUND_CONTEXT)
    metadata = session.metadata
    await session.close(BACKGROUND_CONTEXT)

    async def run() -> int:
        reopened = await repo.open(metadata, BACKGROUND_CONTEXT)
        return 1 if reopened.metadata.id == metadata.id else 0

    return SessionRepoCatalogWriteBenchmarkOperation(run=run)


async def _prepare_delete_closed_empty_session(
    repo: SessionRepo,
) -> SessionRepoCatalogWriteBenchmarkOperation:
    session = await repo.create(SessionCreateOptions(id=_BENCHMARK_SESSION_ID), BACKGROUND_CONTEXT)
    metadata = session.metadata
    await session.close(BACKGROUND_CONTEXT)

    async def run() -> int:
        await repo.delete(metadata, BACKGROUND_CONTEXT)
        return 1

    return SessionRepoCatalogWriteBenchmarkOperation(run=run)


#: Shared catalog reads. Returning a number ensures each result is consumed.
SESSION_REPO_CATALOG_READ_BENCHMARK_SCENARIOS: List[SessionRepoCatalogReadBenchmarkScenario] = [
    SessionRepoCatalogReadBenchmarkScenario(
        name="list sessions",
        expected_result=lambda dataset: dataset.session_count,
        run=_list_sessions,
    ),
]

#: Shared catalog writes. Every invocation receives an equivalent independently prepared repository.
SESSION_REPO_CATALOG_WRITE_BENCHMARK_SCENARIOS: List[SessionRepoCatalogWriteBenchmarkScenario] = [
    SessionRepoCatalogWriteBenchmarkScenario(
        name="create empty session",
        expected_result=1,
        prepare=_prepare_create_empty_session,
    ),
    SessionRepoCatalogWriteBenchmarkScenario(
        name="open closed empty session",
        expected_result=1,
        prepare=_prepare_open_closed_empty_session,
    ),
    SessionRepoCatalogWriteBenchmarkScenario(
        name="delete closed empty session",
        expected_result=1,
        prepare=_prepare_delete_closed_empty_session,
    ),
]

#: Initial fork timing uses a bounded source because each iteration owns an equivalent seeded repository.
SESSION_REPO_FORK_BENCHMARK_DATASETS: List[StorageBenchmarkDataset] = [
    STORAGE_BENCHMARK_DATASETS[0],
    STORAGE_BENCHMARK_DATASETS[1],
]


def _lane_configuration() -> dict:
    """Lane configuration is a JSON value on the wire, so seed that same shape."""
    return {
        "model": {"provider": "benchmark", "modelId": "benchmark"},
        "thinkingLevel": "off",
        "activeToolNames": [],
    }


async def seed_session_repo_fork_benchmark(
    repo: SessionRepo, dataset: StorageBenchmarkDataset
) -> SessionMetadata:
    """Seed one deterministic open source session with a linear main branch."""
    session = await repo.create(SessionCreateOptions(id=_BENCHMARK_SESSION_ID), BACKGROUND_CONTEXT)
    for transaction in generate_storage_benchmark_seed_transactions(dataset):

        async def _commit_transaction(mutator, mutation_context, _transaction=transaction):
            await mutator.commit(_transaction, mutation_context)

        await session.mutate(_commit_transaction, BACKGROUND_CONTEXT)

    async def _configure_lane(mutator, mutation_context):
        await mutator.commit(
            [
                set_value(branch_tip("main"), dataset.tip_id),
                set_value(lane_config("main"), _lane_configuration()),
                set_value(lane_state("main"), {"currentOperationId": None, "lastOperationId": None, "inbox": []}),
            ],
            mutation_context,
        )

    await session.mutate(_configure_lane, BACKGROUND_CONTEXT)
    return session.metadata


async def _fork_open_current_branch(
    repo: SessionRepo, source: SessionMetadata, dataset: StorageBenchmarkDataset
) -> int:
    fork = await repo.fork(
        source,
        ForkOptions(id=_FORK_DESTINATION_SESSION_ID, scope="branch", branch="main"),
        BACKGROUND_CONTEXT,
    )
    matches = (
        fork.metadata.id == _FORK_DESTINATION_SESSION_ID
        and fork.metadata.parent_session_id == source.id
    )
    return dataset.entry_count if matches else 0


#: Shared fork writes. Every invocation receives an equivalent source repository.
SESSION_REPO_FORK_WRITE_BENCHMARK_SCENARIOS: List[SessionRepoForkWriteBenchmarkScenario] = [
    SessionRepoForkWriteBenchmarkScenario(
        name="fork open current branch",
        expected_result=lambda dataset: dataset.entry_count,
        run=_fork_open_current_branch,
    ),
]
