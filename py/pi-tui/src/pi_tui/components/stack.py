"""Constrained stack allocation from ``components/stack.ts``."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .._component import Component, Container
from ..layout_node import LayoutViewport, StackLayoutEntry, StackLayoutNode

MAX_SAFE_INTEGER = 9007199254740991


@dataclass(kw_only=True)
class StackEntryOptions:
    basis: int | float | Literal["auto"] | None = None
    grow: int | float | None = None
    shrink: int | float | None = None
    min_size: int | float | None = None
    max_size: int | float | None = None
    visible: Callable[[LayoutViewport], bool] | None = None


@dataclass(kw_only=True)
class StackEntry(StackEntryOptions):
    component: Component


type StackChild = Component | StackEntry


@dataclass
class StackOptions:
    gap: int | float | None = None
    align: Literal["stretch", "start", "center", "end"] | None = None


def _normalize_size(value: int | float | None, fallback: int) -> int:
    return fallback if value is None or not math.isfinite(value) else max(0, math.floor(value))


class Stack(Container, ABC):
    @property
    @abstractmethod
    def _layout_type(self) -> Literal["vstack", "hstack"]: ...

    def __init__(self, children: list[StackChild] | None = None, options: StackOptions | None = None) -> None:
        super().__init__()
        options = options or StackOptions()
        self._entries: list[StackLayoutEntry] = []
        self._gap = _normalize_size(options.gap, 0)
        self._align = options.align if options.align is not None else "stretch"
        for child in children if children is not None else []:
            if isinstance(child, StackEntry):
                self.add_child(child.component, child)
            else:
                self.add_child(child)

    def add_child(self, component: Component, options: StackEntryOptions | None = None) -> None:
        options = options or StackEntryOptions()
        super().add_child(component)
        self._entries.append(StackLayoutEntry(
            component=component, basis=options.basis,
            grow=None if options.grow is None else _normalize_size(options.grow, 0),
            shrink=None if options.shrink is None else _normalize_size(options.shrink, 1),
            min_size=None if options.min_size is None else _normalize_size(options.min_size, 0),
            max_size=None if options.max_size is None else _normalize_size(options.max_size, MAX_SAFE_INTEGER),
            visible=options.visible,
        ))

    def remove_child(self, component: Component) -> None:
        super().remove_child(component)
        for index, entry in enumerate(self._entries):
            if entry.component is component:
                del self._entries[index]
                break

    def clear(self) -> None:
        super().clear()
        self._entries.clear()

    def __pi_tui_layout_node__(self) -> StackLayoutNode:
        return StackLayoutNode(self._layout_type, self._entries, self._gap, self._align)


def visible_stack_entries(entries: Sequence[StackLayoutEntry], viewport: LayoutViewport) -> list[StackLayoutEntry]:
    result: list[StackLayoutEntry] = []
    # Array.filter fixes its length first but observes edits to later elements.
    for index in range(len(entries)):
        if index >= len(entries):
            continue
        entry = entries[index]
        visible = None if entry.visible is None else entry.visible(viewport)
        if visible is None or visible:
            result.append(entry)
    return result


def _clamp_size(size: int | float, entry: StackLayoutEntry) -> int:
    minimum = max(0, math.floor(0 if entry.min_size is None else entry.min_size))
    maximum = max(minimum, math.floor(MAX_SAFE_INTEGER if entry.max_size is None else entry.max_size))
    return max(minimum, min(maximum, max(0, math.floor(size))))


def _distribute(
    sizes: list[int], entries: Sequence[StackLayoutEntry], amount: int, mode: Literal["grow", "shrink"],
) -> None:
    remaining = amount
    while remaining > 0:
        candidates: list[tuple[StackLayoutEntry, int]] = []
        for index, entry in enumerate(entries):
            if mode == "grow":
                if (0 if entry.grow is None else entry.grow) > 0 and sizes[index] < (MAX_SAFE_INTEGER if entry.max_size is None else entry.max_size):
                    candidates.append((entry, index))
            elif (1 if entry.shrink is None else entry.shrink) > 0 and sizes[index] > (0 if entry.min_size is None else entry.min_size):
                candidates.append((entry, index))
        if not candidates:
            return
        total_weight = sum(
            (0 if entry.grow is None else entry.grow) if mode == "grow"
            else (1 if entry.shrink is None else entry.shrink) * max(1, sizes[index])
            for entry, index in candidates
        )
        distributed = 0
        for entry, index in candidates:
            if remaining <= 0:
                break
            weight = (0 if entry.grow is None else entry.grow) if mode == "grow" else (1 if entry.shrink is None else entry.shrink) * max(1, sizes[index])
            proposed = max(1, math.floor(remaining * weight / total_weight))
            capacity = (MAX_SAFE_INTEGER if entry.max_size is None else entry.max_size) - sizes[index] if mode == "grow" else sizes[index] - (0 if entry.min_size is None else entry.min_size)
            delta = min(remaining, proposed, capacity)
            if delta <= 0:
                continue
            sizes[index] += delta if mode == "grow" else -delta
            remaining -= delta
            distributed += delta
        if distributed == 0:
            return


def allocate_stack_sizes(
    entries: Sequence[StackLayoutEntry], intrinsic_sizes: Sequence[int | float],
    available_size: int | float | None, gap: int,
) -> list[int]:
    sizes = [
        _clamp_size(
            (intrinsic_sizes[index] if index < len(intrinsic_sizes) else 0)
            if entry.basis is None or entry.basis == "auto" else entry.basis,
            entry,
        ) for index, entry in enumerate(entries)
    ]
    if available_size is None:
        return sizes
    content_size = max(0, math.floor(available_size) - max(0, len(entries) - 1) * gap)
    total = sum(sizes)
    if total < content_size:
        _distribute(sizes, entries, content_size - total, "grow")
    elif total > content_size:
        _distribute(sizes, entries, total - content_size, "shrink")
    return sizes
