"""Child container with padding, backgrounds and mouse coordinate translation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from .._component import Component, TuiMouseDispatchResult, TuiMouseEvent, dispatch_mouse_event
from .._javascript import js_repeat
from ..utils import apply_background_to_line, visible_width


@dataclass
class _RenderCache:
    child_lines: list[str]
    width: int
    bg_sample: str | None
    lines: list[str]


class Box:
    def __init__(self, padding_x: int = 1, padding_y: int = 1, bg_fn: Callable[[str], str] | None = None) -> None:
        self.children: list[Component] = []
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._bg_fn = bg_fn
        self._cache: _RenderCache | None = None
        self._mouse_width: int | None = None
        self._mouse_children: list[tuple[Component, int]] = []

    def add_child(self, component: Component) -> None:
        self.children.append(component)
        self._cache = None

    def remove_child(self, component: Component) -> None:
        for index, child in enumerate(self.children):
            if child is component:
                del self.children[index]
                self._cache = None
                break

    def clear(self) -> None:
        self.children = []
        self._cache = None

    def set_bg_fn(self, bg_fn: Callable[[str], str] | None = None) -> None:
        self._bg_fn = bg_fn

    def invalidate(self) -> None:
        self._cache = None
        for child in self.children:
            callback = getattr(child, "invalidate", None)
            if callback is not None:
                callback()

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
        content_width = max(1, event.width - self._padding_x * 2)
        content_y = event.y - self._padding_y
        content_x = event.x - self._padding_x
        if content_y < 0 or content_x < 0 or content_x >= content_width:
            return None
        mouse_children = self._mouse_children if self._mouse_width == content_width else [
            (component, len(component.render(content_width))) for component in self.children
        ]
        child_y = 0
        for child, child_height in mouse_children:
            if child_y <= content_y < child_y + child_height:
                return dispatch_mouse_event(child, replace(
                    event, x=content_x, y=content_y - child_y, width=content_width, height=child_height,
                ))
            child_y += child_height
        return None

    def render(self, width: int) -> list[str]:
        if not self.children:
            return []
        content_width = max(1, width - self._padding_x * 2)
        left_pad = js_repeat(" ", self._padding_x)
        child_lines: list[str] = []
        mouse_children: list[tuple[Component, int]] = []
        for child in self.children:
            lines = child.render(content_width)
            mouse_children.append((child, len(lines)))
            child_lines.extend(left_pad + line for line in lines)
        self._mouse_width = content_width
        self._mouse_children = mouse_children
        if not child_lines:
            return []
        bg_sample = self._bg_fn("test") if self._bg_fn else None
        cache = self._cache
        if cache is not None and cache.width == width and cache.bg_sample == bg_sample and cache.child_lines == child_lines:
            return cache.lines
        result: list[str] = []
        index = 0
        while index < self._padding_y:
            result.append(self._apply_bg("", width))
            index += 1
        result.extend(self._apply_bg(line, width) for line in child_lines)
        index = 0
        while index < self._padding_y:
            result.append(self._apply_bg("", width))
            index += 1
        self._cache = _RenderCache(child_lines, width, bg_sample, result)
        return result

    def _apply_bg(self, line: str, width: int) -> str:
        padded = line + js_repeat(" ", max(0, width - visible_width(line)))
        return apply_background_to_line(padded, width, self._bg_fn) if self._bg_fn else padded
