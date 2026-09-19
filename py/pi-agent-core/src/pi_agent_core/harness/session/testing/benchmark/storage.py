"""Storage benchmark workloads ported from ``session/testing/benchmark/storage.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Iterator, List, Optional

from pi_ai.types import Cost, TextContent, Usage, UserMessage
from ....context import BACKGROUND_CONTEXT
from ...commit import insert_entry, insert_usage
from ...types import EntryScan, MessageEntry, Storage, StorageBranchScan, UsageRow, Write
from ...values import branch_tip, set_value
from .datasets import STORAGE_BENCHMARK_DATASETS, StorageBenchmarkDataset, storage_benchmark_entry_id

__all__ = [
    "StorageReadBenchmarkScenario",
    "StorageWriteBenchmarkScenario",
    "generate_storage_benchmark_seed_transactions",
    "seed_storage_benchmark",
    "STORAGE_READ_BENCHMARK_SCENARIOS",
    "STORAGE_WRITE_BENCHMARK_SCENARIOS",
]

MESSAGE_TIMESTAMP = 1_650_000_000_000
SEED_BATCH_SIZE = 250
WRITE_BASELINE_DATASET = STORAGE_BENCHMARK_DATASETS[0]


def _create_entry(index: int, payload_bytes: int) -> MessageEntry:
    entry_id = storage_benchmark_entry_id(index)
    prefix = f"{entry_id}:"
    return MessageEntry(
        id=entry_id,
        parent_id=None if index == 0 else storage_benchmark_entry_id(index - 1),
        message=UserMessage(
            content=[TextContent(text=prefix + "x" * max(0, payload_bytes - len(prefix)))],
            timestamp=MESSAGE_TIMESTAMP,
        ),
    )


def _create_storage_benchmark_transaction(
    start_index: int, entry_count: int, payload_bytes: int
) -> List[Write]:
    return [
        insert_entry(_create_entry(start_index + offset, payload_bytes))
        for offset in range(entry_count)
    ]


def generate_storage_benchmark_seed_transactions(
    dataset: StorageBenchmarkDataset,
) -> Iterator[List[Write]]:
    """Generate deterministic seed transactions without claiming a production data distribution."""
    start_index = 0
    while start_index < dataset.entry_count:
        yield _create_storage_benchmark_transaction(
            start_index,
            min(SEED_BATCH_SIZE, dataset.entry_count - start_index),
            dataset.payload_bytes,
        )
        start_index += SEED_BATCH_SIZE


async def seed_storage_benchmark(storage: Storage, dataset: StorageBenchmarkDataset) -> None:
    """Seed one deterministic synthetic linear branch."""
    for transaction in generate_storage_benchmark_seed_transactions(dataset):
        await storage.commit(transaction, BACKGROUND_CONTEXT)


@dataclass
class StorageReadBenchmarkScenario:
    """A steady-state read operation run against a pre-seeded fixture."""

    name: str
    expected_result: Callable[[StorageBenchmarkDataset], int]
    run: Callable[[Storage, StorageBenchmarkDataset], Awaitable[int]]


@dataclass
class StorageWriteBenchmarkScenario:
    """A write operation run once against each independently prepared fixture."""

    name: str
    write_count: int
    run: Callable[[Storage], Awaitable[int]]
    prepare: Optional[Callable[[Storage], Awaitable[None]]] = None


_SINGLE_ENTRY_TRANSACTION = _create_storage_benchmark_transaction(0, 1, 256)
_HUNDRED_ENTRY_TRANSACTION = _create_storage_benchmark_transaction(0, 100, 256)
_APPENDED_ENTRY_ID = storage_benchmark_entry_id(WRITE_BASELINE_DATASET.entry_count)
_MIXED_APPEND_TRANSACTION: List[Write] = [
    *_create_storage_benchmark_transaction(
        WRITE_BASELINE_DATASET.entry_count, 1, WRITE_BASELINE_DATASET.payload_bytes
    ),
    set_value(branch_tip("main"), _APPENDED_ENTRY_ID),
    insert_usage(
        UsageRow(
            id="benchmark-usage",
            entry_id=_APPENDED_ENTRY_ID,
            adjustment=False,
            usage=Usage(
                input=1_000,
                output=250,
                cache_read=500,
                cache_write=0,
                total_tokens=1_250,
                cost=Cost(input=0.001, output=0.001, cache_read=0.0001, cache_write=0, total=0.0021),
            ),
        )
    ),
]


async def _scan_distributed_entries(
    storage: Storage, dataset: StorageBenchmarkDataset
) -> int:
    return len(await storage.get_entries(list(dataset.lookup_ids), BACKGROUND_CONTEXT))


async def _scan_latest_entries(storage: Storage, _dataset: StorageBenchmarkDataset) -> int:
    return len(await storage.scan_entries(EntryScan(order="desc", limit=50), BACKGROUND_CONTEXT))


async def _scan_full_branch_structure(
    storage: Storage, dataset: StorageBenchmarkDataset
) -> int:
    return len(
        await storage.scan_branch_structure(
            StorageBranchScan(start=dataset.tip_id, order="newestFirst"), BACKGROUND_CONTEXT
        )
    )


async def _commit_single_entry(storage: Storage) -> int:
    return len((await storage.commit(_SINGLE_ENTRY_TRANSACTION, BACKGROUND_CONTEXT)).seqs)


async def _commit_hundred_entries(storage: Storage) -> int:
    return len((await storage.commit(_HUNDRED_ENTRY_TRANSACTION, BACKGROUND_CONTEXT)).seqs)


async def _commit_mixed_append(storage: Storage) -> int:
    return len((await storage.commit(_MIXED_APPEND_TRANSACTION, BACKGROUND_CONTEXT)).seqs)


#: Shared read scenarios. Returning a number ensures each result is consumed.
STORAGE_READ_BENCHMARK_SCENARIOS: List[StorageReadBenchmarkScenario] = [
    StorageReadBenchmarkScenario(
        name="get 100 distributed entries",
        expected_result=lambda dataset: len(dataset.lookup_ids),
        run=_scan_distributed_entries,
    ),
    StorageReadBenchmarkScenario(
        name="scan latest 50 entries",
        expected_result=lambda dataset: min(50, dataset.entry_count),
        run=_scan_latest_entries,
    ),
    StorageReadBenchmarkScenario(
        name="scan full branch structure",
        expected_result=lambda dataset: dataset.entry_count,
        run=_scan_full_branch_structure,
    ),
]

#: Shared writes. Every invocation receives equivalent pre-benchmark state.
STORAGE_WRITE_BENCHMARK_SCENARIOS: List[StorageWriteBenchmarkScenario] = [
    StorageWriteBenchmarkScenario(
        name="commit one message entry",
        write_count=1,
        run=_commit_single_entry,
    ),
    StorageWriteBenchmarkScenario(
        name="commit 100 message entries",
        write_count=100,
        run=_commit_hundred_entries,
    ),
    StorageWriteBenchmarkScenario(
        name=f"commit mixed append ({WRITE_BASELINE_DATASET.name})",
        write_count=3,
        run=_commit_mixed_append,
        prepare=lambda storage: seed_storage_benchmark(storage, WRITE_BASELINE_DATASET),
    ),
]
