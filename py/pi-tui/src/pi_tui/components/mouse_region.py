"""A mouse handler wrapper that preserves the child's rendering."""

from __future__ import annotations

from collections.abc import Callable

from .._component import Component, TuiMouseDispatchResult, TuiMouseEvent, TuiMouseEventResult, dispatch_mouse_event

type MouseRegionHandler = Callable[[TuiMouseEvent], TuiMouseEventResult | None]


class MouseRegion:
    def __init__(self, child: Component, on_mouse: MouseRegionHandler) -> None:
        self._child = child
        self._on_mouse = on_mouse

    def render(self, width: int) -> list[str]:
        return self._child.render(width)

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | TuiMouseEventResult | None:
        result = dispatch_mouse_event(self._child, event)
        return result if result is not None else self._on_mouse(event)

    def invalidate(self) -> None:
        self._child.invalidate()
