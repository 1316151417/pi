"""Synthetic storage datasets ported from ``session/testing/benchmark/datasets.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = ["StorageBenchmarkDataset", "storage_benchmark_entry_id", "STORAGE_BENCHMARK_DATASETS"]


@dataclass
class StorageBenchmarkDataset:
    """One deterministic synthetic linear branch shared by all storage measurements."""

    name: str
    entry_count: int
    payload_bytes: int
    lookup_ids: List[str] = field(default_factory=list)
    tip_id: str = ""


def storage_benchmark_entry_id(index: int) -> str:
    """Package-internal deterministic id shared by dataset and transaction generation."""
    return f"benchmark-entry-{index:08d}"


def _create_dataset(scale: str, entry_count: int) -> StorageBenchmarkDataset:
    lookup_count = min(100, entry_count)
    return StorageBenchmarkDataset(
        name=f"synthetic linear branch: {scale}, 256-byte payloads",
        entry_count=entry_count,
        payload_bytes=256,
        lookup_ids=[
            storage_benchmark_entry_id((index * (entry_count - 1)) // max(1, lookup_count - 1))
            for index in range(lookup_count)
        ],
        tip_id=storage_benchmark_entry_id(entry_count - 1),
    )


#: Deterministic synthetic linear branches shared by all storage measurements.
STORAGE_BENCHMARK_DATASETS: List[StorageBenchmarkDataset] = [
    _create_dataset("1k entries", 1_000),
    _create_dataset("10k entries", 10_000),
    _create_dataset("100k entries", 100_000),
]
