"""Header conversion ported from ``src/utils/headers.ts``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

__all__ = ["headers_to_record", "provider_headers_to_record"]


class HeaderEntries(Protocol):
    def items(self) -> Iterable[tuple[str, str]]: ...


def headers_to_record(headers: HeaderEntries) -> dict[str, str]:
    return dict(headers.items())


def provider_headers_to_record(headers: Mapping[str, str | None] | None) -> dict[str, str] | None:
    if not headers:
        return None
    result = {key: value for key, value in headers.items() if value is not None}
    return result or None
