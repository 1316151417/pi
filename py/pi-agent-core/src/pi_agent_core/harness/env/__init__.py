"""Harness execution environments ported from ``harness/env/*``.

TypeScript ships one host-native ``ExecutionEnv`` implementation
(``nodejs.ts``'s ``NodeExecutionEnv``). The Python port's counterpart is
:mod:`local`, which drives the same host filesystem and child processes
directly and therefore replaces both the interface and its Node
implementation.
"""

from __future__ import annotations

from .local import LocalExecutionEnv, create_local_execution_env, errno_to_file_error_code

__all__ = [
    "LocalExecutionEnv",
    "create_local_execution_env",
    "errno_to_file_error_code",
]
