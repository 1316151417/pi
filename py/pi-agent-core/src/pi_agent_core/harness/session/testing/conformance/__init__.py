"""Conformance suites ported from ``session/testing/conformance/*``."""

from .session_repo import (
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
from .storage import create_storage_conformance

__all__ = [
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
]
