"""Provider-scoped environment overrides followed by the process environment."""

from __future__ import annotations

import os
from collections.abc import Mapping

__all__ = ["get_provider_env_value"]


def get_provider_env_value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    # The upstream /proc fallback repairs Bun's compiled process.env; Python's
    # os.environ does not have that runtime-specific failure mode.
    return (env.get(name) if env is not None else None) or os.environ.get(name) or None
