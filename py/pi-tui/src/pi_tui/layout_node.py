"""Structural layout metadata from ``layout-node.ts``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ._component import Component

LAYOUT_NODE = "__pi_tui_layout_node__"


@dataclass
class LayoutViewport:
    width: int
    height: int


@dataclass
class StackLayoutEntry:
    component: Component
    basis: float | Literal["auto"] | None = None
    grow: float | None = None
    shrink: float | None = None
    min_size: float | None = None
    max_size: float | None = None
    visible: Callable[[LayoutViewport], bool] | None = None


@dataclass
class StackLayoutNode:
    type: Literal["vstack", "hstack"]
    entries: Sequence[StackLayoutEntry]
    gap: int
    align: Literal["stretch", "start", "center", "end"]


class ScrollLayoutState(Protocol):
    @property
    def scroll_top(self) -> int: ...

    @property
    def primary(self) -> bool: ...

    @property
    def overscroll(self) -> Literal["chain", "contain"]: ...

    @property
    def viewport_height(self) -> int: ...

    def get_content_width(self, width: int) -> int: ...

    def update_layout(self, content_height: int, viewport_height: int, request_render: Callable[[], None]) -> None: ...


@dataclass
class ScrollLayoutNode:
    component: Component
    state: ScrollLayoutState
    type: Literal["scroll"] = "scroll"


type LayoutNode = StackLayoutNode | ScrollLayoutNode


class LayoutComponent(Component, Protocol):
    def __pi_tui_layout_node__(self) -> LayoutNode: ...


def get_layout_node(component: Component) -> LayoutNode | None:
    candidate = getattr(component, LAYOUT_NODE, None)
    return candidate() if callable(candidate) else None
