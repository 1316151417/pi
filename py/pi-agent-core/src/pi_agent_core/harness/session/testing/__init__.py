"""Testing and conformance subsystem ported from ``session/testing/index.ts``.

Barrel re-export of the benchmark datasets, conformance suites, and the test-only
``Storage`` decorators. Exported names mirror the TypeScript barrel one to one.
"""

from .benchmark.datasets import STORAGE_BENCHMARK_DATASETS
from .benchmark.session_repo import (
    SESSION_REPO_CATALOG_BENCHMARK_DATASETS,
    SESSION_REPO_CATALOG_READ_BENCHMARK_SCENARIOS,
    SESSION_REPO_CATALOG_WRITE_BENCHMARK_SCENARIOS,
    SESSION_REPO_FORK_BENCHMARK_DATASETS,
    SESSION_REPO_FORK_WRITE_BENCHMARK_SCENARIOS,
    SessionRepoCatalogBenchmarkDataset,
    seed_session_repo_catalog_benchmark,
    seed_session_repo_fork_benchmark,
    session_repo_benchmark_session_id,
)
from .benchmark.storage import (
    STORAGE_READ_BENCHMARK_SCENARIOS,
    STORAGE_WRITE_BENCHMARK_SCENARIOS,
    generate_storage_benchmark_seed_transactions,
    seed_storage_benchmark,
)
from .conformance.session_repo import (
    create_session_repo_conformance,
    create_session_repo_fork_behavior_conformance,
    create_session_repo_fork_conformance,
    create_session_repo_fork_coordination_conformance,
    create_session_repo_fork_destination_reservation_conformance,
    create_session_repo_fork_source_snapshot_conformance,
    create_session_repo_lifecycle_conformance,
    create_session_repo_message_conformance,
    create_session_repo_ownership_conformance,
    create_session_repo_streaming_fork_conformance,
)
from .conformance.storage import create_storage_conformance
from .gating_storage import CommitDiscarded, GatingStorage
from .instrumented_storage import InstrumentedStorage
from .storage_decorator import StorageDecorator
from .types import ConformanceCase, StorageFixture

__all__ = [
    "STORAGE_BENCHMARK_DATASETS",
    "SESSION_REPO_CATALOG_BENCHMARK_DATASETS",
    "SESSION_REPO_CATALOG_READ_BENCHMARK_SCENARIOS",
    "SESSION_REPO_CATALOG_WRITE_BENCHMARK_SCENARIOS",
    "SESSION_REPO_FORK_BENCHMARK_DATASETS",
    "SESSION_REPO_FORK_WRITE_BENCHMARK_SCENARIOS",
    "SessionRepoCatalogBenchmarkDataset",
    "seed_session_repo_catalog_benchmark",
    "seed_session_repo_fork_benchmark",
    "session_repo_benchmark_session_id",
    "generate_storage_benchmark_seed_transactions",
    "STORAGE_READ_BENCHMARK_SCENARIOS",
    "STORAGE_WRITE_BENCHMARK_SCENARIOS",
    "seed_storage_benchmark",
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
    "CommitDiscarded",
    "GatingStorage",
    "InstrumentedStorage",
    "StorageDecorator",
    "ConformanceCase",
    "StorageFixture",
]
