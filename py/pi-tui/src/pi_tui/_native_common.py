"""Shared clipboard values without a dependency on any higher-level package."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal, final


@final
class Undefined:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNDEFINED"

    def __bool__(self) -> bool:
        return False


UNDEFINED = Undefined()
type ModifierKey = Literal["shift", "command", "control", "option"]
type ClipboardText = str | None | Undefined
type ClipboardImage = bytes | None | Undefined


async def run_clipboard[T](operation: Callable[[], T]) -> T:
    """The native worker keeps running if its Python waiter is cancelled."""
    try:
        return await asyncio.to_thread(operation)
    except MemoryError as error:
        raise RuntimeError("Out of memory") from error


def modifier_name(name: str) -> str:
    # Native get_value_string_utf8 writes at most 15 bytes plus a terminator;
    # its strcmp also treats an embedded NUL as the end of the name.
    if not isinstance(name, str):
        return ""
    return name.encode("utf-8", "replace")[:15].split(b"\0", 1)[0].decode("utf-8", "replace")
