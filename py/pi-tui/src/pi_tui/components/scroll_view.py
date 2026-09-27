"""Scroll state and transient scrollbar behavior from ``scroll-view.ts``."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .._component import Component, Container
from .._timers import TimerHandle, set_timeout
from ..layout_node import ScrollLayoutNode

type ScrollViewScrollbar = Literal["hidden", "auto", "always"]


@dataclass
class ScrollViewOptions:
    axis: Literal["vertical"] | None = None
    follow: Literal["none", "end"] | None = None
    primary: bool | None = None
    overscroll: Literal["chain", "contain"] | None = None
    scrollbar: ScrollViewScrollbar | None = None
    scrollbar_track_style: Callable[[str], str] | None = None
    scrollbar_thumb_style: Callable[[str], str] | None = None
    scrollbar_hide_delay_ms: float | None = None


@dataclass
class ScrollViewScrollToOptions:
    disable_follow: bool | None = None


class ScrollView(Container):
    def __init__(self, component: Component, options: ScrollViewOptions | None = None) -> None:
        super().__init__()
        options = options or ScrollViewOptions()
        if options.axis is not None and options.axis != "vertical":
            raise ValueError(f"Unsupported ScrollView axis: {options.axis}")
        self._child = component
        self.children.append(component)
        self.follow_end = options.follow == "end"
        self._following_end = self.follow_end
        self.primary = False if options.primary is None else options.primary
        self.overscroll = "chain" if options.overscroll is None else options.overscroll
        self._current_scrollbar = "hidden" if options.scrollbar is None else options.scrollbar
        self.scrollbar_track_style = options.scrollbar_track_style or (lambda text: f"\x1b[90m{text}\x1b[39m")
        self.scrollbar_thumb_style = options.scrollbar_thumb_style or (lambda text: f"\x1b[37m{text}\x1b[39m")
        self._scrollbar_hide_delay_ms = max(0, math.floor(1000 if options.scrollbar_hide_delay_ms is None else options.scrollbar_hide_delay_ms))
        self._current_scroll_top = 0
        self._content_height = 0
        self._current_viewport_height = 0
        self._follow_suppressed_at_end = False
        self._request_render_callback: Callable[[], None] | None = None
        self._transient_scrollbar_visible = False
        self._scrollbar_active = False
        self._scrollbar_hide_timer: TimerHandle | None = None
        self._scrollbar_timer_generation = 0

    @property
    def scroll_top(self) -> int:
        return self._current_scroll_top

    @property
    def is_following_end(self) -> bool:
        return self._following_end

    @property
    def viewport_height(self) -> int:
        return self._current_viewport_height

    @property
    def scrollbar(self) -> ScrollViewScrollbar:
        return self._current_scrollbar

    @property
    def is_scrollbar_visible(self) -> bool:
        if self.scrollbar == "always":
            return self._current_viewport_height > 0
        return self.scrollbar == "auto" and self._content_height > self._current_viewport_height and self._transient_scrollbar_visible

    @property
    def is_scrollbar_active(self) -> bool:
        return self._scrollbar_active

    def set_scrollbar(self, scrollbar: ScrollViewScrollbar) -> None:
        if scrollbar == self._current_scrollbar:
            return
        self._current_scrollbar = scrollbar
        if scrollbar != "auto":
            self._hide_transient_scrollbar()
        elif self._scrollbar_active:
            self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def get_content_width(self, width: int) -> int:
        return width - 1 if self.scrollbar == "always" and width > 1 else width

    def _mark_scrollbar_activity(self) -> None:
        if self.scrollbar != "auto" or self._content_height <= self._current_viewport_height:
            return
        self._transient_scrollbar_visible = True
        self._scrollbar_timer_generation += 1
        if self._scrollbar_hide_timer is not None:
            self._scrollbar_hide_timer.cancel()
            self._scrollbar_hide_timer = None
        if self._scrollbar_active:
            return
        generation = self._scrollbar_timer_generation

        def hide() -> None:
            if generation != self._scrollbar_timer_generation:
                return
            self._scrollbar_hide_timer = None
            self._transient_scrollbar_visible = False
            if self._request_render_callback is not None:
                self._request_render_callback()

        self._scrollbar_hide_timer = set_timeout(hide, self._scrollbar_hide_delay_ms, unref=True)

    def _hide_transient_scrollbar(self) -> None:
        self._transient_scrollbar_visible = False
        self._scrollbar_timer_generation += 1
        if self._scrollbar_hide_timer is not None:
            self._scrollbar_hide_timer.cancel()
            self._scrollbar_hide_timer = None

    def set_scrollbar_active(self, active: bool) -> None:
        if active == self._scrollbar_active:
            return
        self._scrollbar_active = active
        self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def scroll_to(self, scroll_top: float, options: ScrollViewScrollToOptions | None = None) -> None:
        options = options or ScrollViewScrollToOptions()
        requested = math.trunc(scroll_top) if math.isfinite(scroll_top) else self._current_scroll_top
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        next_top = max(0, min(max_scroll_top, requested))
        next_suppressed = options.disable_follow is True and next_top == max_scroll_top
        next_following = not next_suppressed and self.follow_end and next_top == max_scroll_top
        if next_top == self._current_scroll_top and next_following == self._following_end and next_suppressed == self._follow_suppressed_at_end:
            return
        moved = next_top != self._current_scroll_top
        self._current_scroll_top = next_top
        self._following_end = next_following
        self._follow_suppressed_at_end = next_suppressed
        if moved:
            self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def scroll_by(self, lines: float) -> int:
        requested = math.trunc(lines) if math.isfinite(lines) else 0
        if requested == 0:
            return 0
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        start = max_scroll_top if self._following_end else self._current_scroll_top
        next_top = max(0, min(max_scroll_top, start + requested))
        moved = next_top - start
        was_following_end = self._following_end
        self._current_scroll_top = next_top
        self._following_end = self.follow_end and next_top == max_scroll_top
        self._follow_suppressed_at_end = False
        if moved != 0:
            self._mark_scrollbar_activity()
        if (moved != 0 or self._following_end != was_following_end) and self._request_render_callback is not None:
            self._request_render_callback()
        return requested - moved

    def scroll_to_start(self) -> None:
        next_following = self.follow_end and self._content_height <= self._current_viewport_height
        changed = self._current_scroll_top != 0 or self._following_end != next_following
        self._current_scroll_top = 0
        self._following_end = next_following
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def scroll_to_end(self) -> None:
        next_top = max(0, self._content_height - self._current_viewport_height)
        changed = self._current_scroll_top != next_top or self._following_end != self.follow_end
        self._current_scroll_top = next_top
        self._following_end = self.follow_end
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def update_layout(self, content_height: int, viewport_height: int, request_render: Callable[[], None]) -> None:
        self._content_height = max(0, math.floor(content_height))
        self._current_viewport_height = max(0, math.floor(viewport_height))
        self._request_render_callback = request_render
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        if self._following_end:
            self._current_scroll_top = max_scroll_top
        else:
            self._current_scroll_top = max(0, min(self._current_scroll_top, max_scroll_top))
        if self._current_scroll_top < max_scroll_top:
            self._follow_suppressed_at_end = False
        if self.follow_end and self._current_scroll_top == max_scroll_top and not self._follow_suppressed_at_end:
            self._following_end = True
        if self._content_height <= self._current_viewport_height:
            self._hide_transient_scrollbar()

    def add_child(self, component: Component) -> None:
        raise RuntimeError("ScrollView has exactly one child")

    def remove_child(self, component: Component) -> None:
        raise RuntimeError("ScrollView child cannot be removed")

    def clear(self) -> None:
        raise RuntimeError("ScrollView child cannot be cleared")

    def render(self, width: int) -> list[str]:
        content_width = self.get_content_width(width)
        lines = self._child.render(content_width)
        return lines if content_width == width else [line + " " for line in lines]

    def __pi_tui_layout_node__(self) -> ScrollLayoutNode:
        return ScrollLayoutNode(self._child, self)
