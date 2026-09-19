"""Assertion helpers for the ported conformance suites.

The TypeScript conformance tables assert with ``node:assert/strict``. Python has
no equivalent, so the four strict helpers used by those tables live here:

``strict_equal``
    Counterpart of ``strictEqual`` (same type and same value).
``deep_strict_equal``
    Counterpart of ``deepStrictEqual`` for the dataclass/JSON-like values the
    ports of ``Storage`` and ``SessionRepo`` return.
``ok``
    Counterpart of ``ok`` (truthiness).
``rejects``
    Counterpart of ``rejects`` (awaitable must raise).

This module is port support only; there is no TypeScript counterpart file in
``session/testing/``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Optional, Sequence

__all__ = ["strict_equal", "deep_strict_equal", "ok", "rejects", "strictly_increasing"]


def _failure(prefix: str, expected: Any, actual: Any, message: Optional[str]) -> AssertionError:
    detail = f"\n  expected: {expected!r}\n  actual:   {actual!r}"
    return AssertionError(f"{message}{detail}" if message else f"{prefix}{detail}")


def strict_equal(actual: Any, expected: Any, message: Optional[str] = None) -> None:
    """Require ``actual`` to be the same type and value as ``expected``."""
    if type(actual) is not type(expected) or actual != expected:
        raise _failure("Expected values to be strictly equal", expected, actual, message)


def deep_strict_equal(actual: Any, expected: Any, message: Optional[str] = None) -> None:
    """Require structural equality between dataclass or JSON-like values."""
    if actual != expected:
        raise _failure("Expected values to be deeply equal", expected, actual, message)


def ok(value: Any, message: Optional[str] = None) -> None:
    """Require a truthy value."""
    if not value:
        raise AssertionError(message or "Expected value to be truthy")


async def rejects(awaitable: Awaitable[Any], message: Optional[str] = None) -> None:
    """Require ``awaitable`` to raise."""
    try:
        await awaitable
    except Exception:  # noqa: BLE001 - any failure satisfies `rejects`
        return
    raise AssertionError(message or "Expected operation to reject")


def strictly_increasing(values: Sequence[int], message: Optional[str] = None) -> None:
    """Require every element to be strictly greater than its predecessor."""
    for index in range(1, len(values)):
        if not values[index - 1] < values[index]:
            rendered = ", ".join(str(value) for value in values)
            raise AssertionError(message or f"Expected {rendered} to be strictly increasing")
