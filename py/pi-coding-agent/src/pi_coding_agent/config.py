"""Package asset paths used by coding-agent prompt construction."""

from __future__ import annotations

import os
from pathlib import Path


def get_package_dir() -> Path:
    override = os.environ.get("PI_PACKAGE_DIR")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2]


def get_readme_path() -> str:
    return str(get_package_dir() / "README.md")


def get_docs_path() -> str:
    return str(get_package_dir() / "docs")


def get_examples_path() -> str:
    return str(get_package_dir() / "examples")


def get_agent_dir() -> str:
    override = os.environ.get("PI_CODING_AGENT_DIR")
    return str(Path(override).expanduser().resolve()) if override else str(Path.home() / ".pi" / "agent")


def get_models_path() -> str:
    return str(Path(get_agent_dir()) / "models.json")


def get_auth_path() -> str:
    return str(Path(get_agent_dir()) / "auth.json")


def get_sessions_dir() -> str:
    return str(Path(get_agent_dir()) / "sessions")


def get_bin_dir() -> str:
    return str(Path(get_agent_dir()) / "bin")


__all__ = [
    "get_package_dir", "get_readme_path", "get_docs_path", "get_examples_path",
    "get_agent_dir", "get_models_path", "get_auth_path", "get_sessions_dir", "get_bin_dir",
]
