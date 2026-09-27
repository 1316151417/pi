"""Overlay, focus, input, scheduling and terminal queries from ``tui.ts``."""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Literal, cast

from ._component import (
    CURSOR_MARKER, VIEWPORT_TUI, Component, Container, Focusable, FocusableComponent,
    InputComponent, MouseComponent, OverlayAnchor, OverlayBounds, OverlayHandle,
    OverlayMargin, OverlayOptions, OverlayUnfocusOptions, SizeValue, TUI,
    TuiInputListener, TuiInputListenerResult, TuiMode, TuiMouseButton,
    TuiMouseDispatchResult, TuiMouseDispatchTarget, TuiMouseEvent,
    TuiMouseEventResult, TuiMouseEventType, TuiStopOptions, ViewportTUI,
    dispatch_mouse_event, is_focusable, is_viewport_tui, retarget_mouse_event,
)
from ._terminal_types import Terminal
from ._timers import TimerHandle, set_timeout
from .keys import is_key_release, matches_key
from .terminal_colors import (
    RgbColor, TerminalColorScheme, is_osc11_background_color_response,
    parse_osc11_background_color, parse_terminal_color_scheme_report,
)
from .terminal_image import CellDimensions, get_capabilities, is_image_line, set_cell_dimensions
from .utils import extract_segments, normalize_terminal_output, slice_by_column, slice_with_width, visible_width

type _Number = int | float
type AppKeybindings = Mapping[str, str | Sequence[str] | None]

DEFAULT_APP_KEYBINDINGS: dict[str, str | Sequence[str] | None] = {"tui.debug": "shift+ctrl+d"}
_SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"
_PERCENTAGE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")


@dataclass(eq=False)
class _SetEntry[T]:
    value: T
    present: bool = True


class _IdentitySet[T]:
    """Insertion-ordered identities with JavaScript's live Set iteration."""

    def __init__(self) -> None:
        self._entries: list[_SetEntry[T]] = []
        self._members: dict[int, _SetEntry[T]] = {}
        self._iterators = 0

    def add(self, value: T) -> None:
        if id(value) not in self._members:
            entry = _SetEntry(value)
            self._members[id(value)] = entry
            self._entries.append(entry)

    def discard(self, value: T) -> None:
        entry = self._members.pop(id(value), None)
        if entry is not None:
            entry.present = False
            if not self._iterators:
                self._entries = [item for item in self._entries if item.present]

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[T]:
        self._iterators += 1
        try:
            index = 0
            while index < len(self._entries):
                entry = self._entries[index]
                index += 1
                if entry.present:
                    yield entry.value
        finally:
            self._iterators -= 1
            if not self._iterators:
                self._entries = [item for item in self._entries if item.present]


def _array_values[T](values: Sequence[T]) -> Iterator[T]:
    # Array find/some/filter capture length but read the current entry at each index.
    length = len(values)
    for index in range(length):
        if index < len(values):
            yield values[index]


def _minimum(*values: _Number) -> _Number:
    return math.nan if any(math.isnan(value) for value in values) else min(values)


def _maximum(*values: _Number) -> _Number:
    return math.nan if any(math.isnan(value) for value in values) else max(values)


def _floor(value: _Number) -> _Number:
    return math.floor(value) if math.isfinite(value) else value


def _parse_size_value(value: SizeValue | None, reference_size: _Number) -> _Number | None:
    if value is None or isinstance(value, (int, float)):
        return value
    match = _PERCENTAGE.fullmatch(value)
    return _floor(reference_size * float(match[1]) / 100) if match else None


def _repeat_spaces(count: _Number) -> str:
    if math.isnan(count):
        return ""
    return " " * math.trunc(count)


@dataclass(eq=False)
class _OverlayStackEntry:
    component: Component
    options: OverlayOptions | None
    pre_focus: Component | None
    focus_order: int
    hidden: bool = False
    bounds: OverlayBounds | None = None


@dataclass
class _RenderedOverlayLayout:
    entry: _OverlayStackEntry
    row: _Number
    col: _Number
    width: _Number
    height: int


@dataclass
class _RenderedOverlay:
    entry: _OverlayStackEntry
    overlay_lines: list[str]
    row: _Number
    col: _Number
    width: _Number


@dataclass
class _OverlayLayout:
    width: _Number
    row: _Number
    col: _Number
    max_height: _Number | None


@dataclass
class _RestoreOverlay:
    pass


@dataclass
class _FocusTarget:
    target: Component | None


@dataclass
class _InactiveRestore:
    pass


@dataclass
class _EligibleRestore:
    overlay: _OverlayStackEntry


@dataclass
class _BlockedRestore:
    overlay: _OverlayStackEntry
    blocked_by: Component
    resume: _RestoreOverlay | _FocusTarget


type _OverlayFocusRestore = _InactiveRestore | _EligibleRestore | _BlockedRestore


@dataclass
class OverlayMouseDispatch:
    hit: bool
    result: TuiMouseDispatchResult | None = None


@dataclass
class CursorPosition:
    row: int
    col: int


class _QueryPromise[T]:
    """Immediate query admission with an awaitable independent of caller loops."""

    def __init__(self) -> None:
        self._future: Future[T] = Future()

    def resolve(self, value: T) -> None:
        if not self._future.done():
            self._future.set_result(value)

    def reject(self, error: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(error)

    def __await__(self) -> Generator[object, None, T]:
        return asyncio.shield(asyncio.wrap_future(self._future)).__await__()


@dataclass
class _PendingOsc11BackgroundQuery:
    settled: bool
    resolve: Callable[[RgbColor | None], None] | None
    timer: TimerHandle | None = None


def composite_tui_line(
    base_line: str,
    overlay_line: str,
    start_col: int,
    overlay_width: int,
    total_width: int,
) -> str:
    if is_image_line(base_line):
        return base_line
    after_start = start_col + overlay_width
    base = extract_segments(base_line, start_col, after_start, total_width - after_start, True)
    overlay = slice_with_width(overlay_line, 0, overlay_width, True)
    before_pad = _maximum(0, start_col - base.before_width)
    overlay_pad = _maximum(0, overlay_width - overlay.width)
    actual_before_width = _maximum(start_col, base.before_width)
    actual_overlay_width = _maximum(overlay_width, overlay.width)
    after_target = _maximum(0, total_width - actual_before_width - actual_overlay_width)
    after_pad = _maximum(0, after_target - base.after_width)
    result = (
        base.before + _repeat_spaces(before_pad) + _SEGMENT_RESET + overlay.text
        + _repeat_spaces(overlay_pad) + _SEGMENT_RESET + base.after + _repeat_spaces(after_pad)
    )
    return result if visible_width(result) <= total_width else slice_by_column(result, 0, total_width, True)


class _OverlayHandle:
    def __init__(self, owner: TuiBase, entry: _OverlayStackEntry) -> None:
        self._owner = owner
        self._entry = entry

    def hide(self) -> None:
        owner, entry = self._owner, self._entry
        if entry not in owner._overlay_stack:
            return
        owner._clear_overlay_focus_restore_for(entry)
        owner._retarget_overlay_pre_focus(entry)
        owner._overlay_stack.remove(entry)
        if owner._focused_component is entry.component:
            top_visible = owner._get_topmost_visible_overlay()
            owner.set_focus(top_visible.component if top_visible else entry.pre_focus)
        if not owner._overlay_stack:
            owner.terminal.hide_cursor()
        owner.request_render()

    def set_hidden(self, hidden: bool) -> None:
        owner, entry = self._owner, self._entry
        if entry.hidden is hidden:
            return
        entry.hidden = hidden
        if hidden:
            owner._clear_overlay_focus_restore_for(entry)
            if owner._focused_component is entry.component:
                top_visible = owner._get_topmost_visible_overlay()
                owner.set_focus(top_visible.component if top_visible else entry.pre_focus)
        elif not (entry.options and entry.options.non_capturing) and owner._is_overlay_visible(entry):
            owner._focus_order_counter += 1
            entry.focus_order = owner._focus_order_counter
            owner.set_focus(entry.component)
        owner.request_render()

    def is_hidden(self) -> bool:
        return self._entry.hidden

    def focus(self) -> None:
        owner, entry = self._owner, self._entry
        if entry not in owner._overlay_stack or not owner._is_overlay_visible(entry):
            return
        owner._focus_order_counter += 1
        entry.focus_order = owner._focus_order_counter
        owner.set_focus(entry.component)
        owner.request_render()

    def unfocus(self, options: OverlayUnfocusOptions | None = None) -> None:
        owner, entry = self._owner, self._entry
        focused = owner._focused_component is entry.component
        restore = owner._overlay_focus_restore
        pending_restore = not isinstance(restore, _InactiveRestore) and restore.overlay is entry
        if not focused and not pending_restore:
            return
        if isinstance(restore, _BlockedRestore) and restore.overlay is entry and owner._focused_component is restore.blocked_by:
            if options is not None:
                owner._overlay_focus_restore = _BlockedRestore(entry, restore.blocked_by, _FocusTarget(options.target))
            else:
                owner._clear_overlay_focus_restore()
            owner.request_render()
            return
        owner._clear_overlay_focus_restore_for(entry)
        if focused or options is not None:
            top_visible = owner._get_topmost_visible_overlay()
            fallback = top_visible.component if top_visible is not None and top_visible is not entry else entry.pre_focus
            owner.set_focus(options.target if options is not None else fallback)
        owner.request_render()

    def is_focused(self) -> bool:
        return self._owner._focused_component is self._entry.component

    def get_bounds(self) -> OverlayBounds | None:
        entry = self._entry
        if entry not in self._owner._overlay_stack or not self._owner._is_overlay_visible(entry) or entry.bounds is None:
            return None
        return replace(entry.bounds)


class TuiBase(Container, ABC):
    MIN_RENDER_INTERVAL_MS = 16

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
        *,
        keybindings: AppKeybindings | None = None,
    ) -> None:
        super().__init__()
        self.terminal = terminal
        self._log_directory = log_directory
        self._focused_component: Component | None = None
        self._input_listeners: _IdentitySet[TuiInputListener] = _IdentitySet()
        self.on_debug: Callable[[], None] | None = None
        self._render_requested = False
        self._immediate_render_scheduled = False
        self._render_timer: TimerHandle | None = None
        self._render_timer_generation = 0
        self._last_render_at = 0.0
        self._show_hardware_cursor = False if show_hardware_cursor is None else show_hardware_cursor
        self._clear_on_shrink = False
        self._full_redraw_count = 0
        self._stopped = False
        self._pending_osc11_background_replies = 0
        self._pending_osc11_background_queries: list[_PendingOsc11BackgroundQuery] = []
        self._terminal_color_scheme_listeners: _IdentitySet[Callable[[TerminalColorScheme], None]] = _IdentitySet()
        self._terminal_color_scheme_notifications_enabled = False
        self._focus_order_counter = 0
        self._overlay_stack: list[_OverlayStackEntry] = []
        self._rendered_overlay_layouts: list[_RenderedOverlayLayout] = []
        self._overlay_focus_restore: _OverlayFocusRestore = _InactiveRestore()
        self._keybindings = dict(DEFAULT_APP_KEYBINDINGS)
        if keybindings is not None:
            self._keybindings.update(keybindings)
        self._state_lock = threading.RLock()
        self._next_tick_callbacks: deque[Callable[[], None]] = deque()
        self._next_tick_scheduled = False
        try:
            self._event_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None

    @property
    @abstractmethod
    def mode(self) -> TuiMode:
        raise NotImplementedError

    @abstractmethod
    def _do_render(self) -> None:
        raise NotImplementedError

    def _reset_render_state(self) -> None:
        return None

    def _before_terminal_start(self) -> None:
        return None

    def _after_terminal_start(self) -> None:
        return None

    def _before_terminal_stop(self, options: TuiStopOptions) -> None:
        return None

    def _after_terminal_stop(self, options: TuiStopOptions) -> None:
        return None

    @property
    def has_overlay_entries(self) -> bool:
        return bool(self._overlay_stack)

    @property
    def full_redraws(self) -> int:
        return self._full_redraw_count

    def get_show_hardware_cursor(self) -> bool:
        return self._show_hardware_cursor

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        if self._show_hardware_cursor is enabled:
            return
        self._show_hardware_cursor = enabled
        if not enabled:
            self.terminal.hide_cursor()
        self.request_render()

    def get_clear_on_shrink(self) -> bool:
        return self._clear_on_shrink

    def set_clear_on_shrink(self, enabled: bool) -> None:
        self._clear_on_shrink = enabled

    def get_focused_component(self) -> Component | None:
        return self._focused_component

    def set_focus(self, component: Component | None) -> None:
        self._set_focus_internal(component, "clear")

    def _set_focus_internal(self, component: Component | None, overlay_focus_restore: Literal["clear", "preserve"]) -> None:
        previous_focus = self._focused_component
        next_focus = component
        previous_overlay = next((entry for entry in _array_values(self._overlay_stack) if entry.component is previous_focus and self._is_overlay_visible(entry)), None) if previous_focus is not None else None
        next_is_overlay = any(entry.component is next_focus for entry in _array_values(self._overlay_stack)) if next_focus is not None else False
        restore = self._get_visible_overlay_focus_restore()
        if next_focus is not None and not next_is_overlay:
            if isinstance(restore, _BlockedRestore) and restore.blocked_by is previous_focus:
                if isinstance(restore.resume, _FocusTarget) or not self._is_component_mounted(restore.blocked_by):
                    next_focus = self._resolve_blocked_overlay_focus_resume(restore)
                else:
                    self._overlay_focus_restore = _BlockedRestore(restore.overlay, next_focus, restore.resume)
            elif previous_overlay is not None and not isinstance(restore, _InactiveRestore) and restore.overlay is previous_overlay and not self._is_overlay_focus_ancestor(previous_overlay, next_focus):
                self._overlay_focus_restore = _BlockedRestore(previous_overlay, next_focus, _RestoreOverlay())
        elif next_focus is None:
            if isinstance(restore, _BlockedRestore) and restore.blocked_by is previous_focus:
                next_focus = self._resolve_blocked_overlay_focus_resume(restore)
            elif overlay_focus_restore == "clear":
                self._clear_overlay_focus_restore()
        if is_focusable(self._focused_component):
            self._focused_component.focused = False
        self._focused_component = next_focus
        if is_focusable(next_focus):
            next_focus.focused = True
        focused_overlay = next((entry for entry in _array_values(self._overlay_stack) if entry.component is next_focus and self._is_overlay_visible(entry)), None) if next_focus is not None else None
        if focused_overlay is not None:
            self._overlay_focus_restore = _EligibleRestore(focused_overlay)

    def _clear_overlay_focus_restore(self) -> None:
        self._overlay_focus_restore = _InactiveRestore()

    def _clear_overlay_focus_restore_for(self, overlay: _OverlayStackEntry) -> None:
        restore = self._overlay_focus_restore
        if not isinstance(restore, _InactiveRestore) and restore.overlay is overlay:
            self._clear_overlay_focus_restore()

    def _resolve_blocked_overlay_focus_resume(self, restore: _BlockedRestore) -> Component | None:
        if isinstance(restore.resume, _RestoreOverlay):
            return restore.overlay.component
        self._clear_overlay_focus_restore()
        return restore.resume.target

    def _get_visible_overlay_focus_restore(self) -> _OverlayFocusRestore:
        restore = self._overlay_focus_restore
        if isinstance(restore, _InactiveRestore):
            return restore
        if restore.overlay not in self._overlay_stack or not self._is_overlay_visible(restore.overlay):
            return _InactiveRestore()
        return restore

    def _is_overlay_focus_ancestor(self, entry: _OverlayStackEntry, component: Component) -> bool:
        visited: set[int] = set()
        current = entry.pre_focus
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if current is component:
                return True
            overlay = next((value for value in self._overlay_stack if value.component is current), None)
            current = overlay.pre_focus if overlay else None
        return False

    def _retarget_overlay_pre_focus(self, removed: _OverlayStackEntry) -> None:
        for overlay in self._overlay_stack:
            if overlay is not removed and overlay.pre_focus is removed.component:
                overlay.pre_focus = removed.pre_focus

    def _get_mounted_roots(self) -> Sequence[Component]:
        return self.children

    def _is_component_mounted(self, component: Component) -> bool:
        roots = self._get_mounted_roots()
        return any(self._contains_component(child, component) for child in _array_values(roots))

    def _contains_component(self, root: Component, target: Component) -> bool:
        if root is target:
            return True
        if not isinstance(root, Container):
            return False
        return any(self._contains_component(child, target) for child in _array_values(root.children))

    def show_overlay(self, component: Component, options: OverlayOptions | None = None) -> OverlayHandle:
        self._focus_order_counter += 1
        entry = _OverlayStackEntry(component, options, self._focused_component, self._focus_order_counter)
        self._overlay_stack.append(entry)
        if not (options and options.non_capturing) and self._is_overlay_visible(entry):
            self.set_focus(component)
        self.terminal.hide_cursor()
        self.request_render()
        return _OverlayHandle(self, entry)

    def hide_overlay(self) -> None:
        if not self._overlay_stack:
            return
        overlay = self._overlay_stack[-1]
        self._clear_overlay_focus_restore_for(overlay)
        self._retarget_overlay_pre_focus(overlay)
        self._overlay_stack.pop()
        if self._focused_component is overlay.component:
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(top_visible.component if top_visible else overlay.pre_focus)
        if not self._overlay_stack:
            self.terminal.hide_cursor()
        self.request_render()

    def has_overlay(self) -> bool:
        return any(self._is_overlay_visible(entry) for entry in _array_values(self._overlay_stack))

    def _is_overlay_focused(self) -> bool:
        return any(entry.component is self._focused_component and self._is_overlay_visible(entry) for entry in _array_values(self._overlay_stack))

    def _resolve_mouse_focus_target(self, component: Component) -> Component:
        index = len(self._overlay_stack) - 1
        while index >= 0:
            overlay = self._overlay_stack[index]
            if self._is_overlay_visible(overlay) and self._contains_component(overlay.component, component):
                return overlay.component
            index -= 1
        return component

    def _dispatch_mouse_to_overlay(self, event: TuiMouseEvent) -> OverlayMouseDispatch:
        index = len(self._rendered_overlay_layouts) - 1
        while index >= 0:
            layout = self._rendered_overlay_layouts[index]
            index -= 1
            if (event.screen_x < layout.col or event.screen_x >= layout.col + layout.width
                    or event.screen_y < layout.row or event.screen_y >= layout.row + layout.height):
                continue
            result = dispatch_mouse_event(layout.entry.component, replace(
                event, x=cast(int, event.screen_x - layout.col), y=cast(int, event.screen_y - layout.row),
                width=cast(int, layout.width), height=layout.height,
            ))
            if result is not None and result.focus:
                result = replace(result, focus_target=layout.entry.component)
            return OverlayMouseDispatch(True, result)
        return OverlayMouseDispatch(False)

    def _is_overlay_visible(self, entry: _OverlayStackEntry) -> bool:
        if entry.hidden:
            return False
        if entry.options is not None and entry.options.visible is not None:
            return entry.options.visible(self.terminal.columns, self.terminal.rows)
        return True

    def _get_topmost_visible_overlay(self) -> _OverlayStackEntry | None:
        topmost: _OverlayStackEntry | None = None
        for overlay in self._overlay_stack:
            if (overlay.options and overlay.options.non_capturing) or not self._is_overlay_visible(overlay):
                continue
            if topmost is None or overlay.focus_order > topmost.focus_order:
                topmost = overlay
        return topmost

    def invalidate(self) -> None:
        for root in self._get_mounted_roots():
            root.invalidate()
        for overlay in self._overlay_stack:
            overlay.component.invalidate()

    def start(self) -> None:
        self._stopped = False
        self._before_terminal_start()
        self.terminal.start(self._handle_terminal_input, self.request_render)
        self._after_terminal_start()
        self.terminal.hide_cursor()
        if self._terminal_color_scheme_notifications_enabled:
            self.terminal.write("\x1b[?2031h")
        if get_capabilities().images:
            self.terminal.write("\x1b[16t")
        self.request_render()

    def add_input_listener(self, listener: TuiInputListener) -> Callable[[], None]:
        self._input_listeners.add(listener)
        return lambda: self._input_listeners.discard(listener)

    def remove_input_listener(self, listener: TuiInputListener) -> None:
        self._input_listeners.discard(listener)

    def on_terminal_color_scheme_change(self, listener: Callable[[TerminalColorScheme], None]) -> Callable[[], None]:
        self._terminal_color_scheme_listeners.add(listener)
        return lambda: self._terminal_color_scheme_listeners.discard(listener)

    def set_terminal_color_scheme_notifications(self, enabled: bool) -> None:
        if self._terminal_color_scheme_notifications_enabled is enabled:
            return
        self._terminal_color_scheme_notifications_enabled = enabled
        if not self._stopped:
            self.terminal.write("\x1b[?2031h" if enabled else "\x1b[?2031l")

    def stop(self, options: TuiStopOptions | None = None) -> None:
        options = options or TuiStopOptions()
        with self._state_lock:
            self._stopped = True
            self._cancel_render_timer()
        if self._terminal_color_scheme_notifications_enabled:
            self.terminal.write("\x1b[?2031l")
        self._before_terminal_stop(options)
        self.terminal.show_cursor()
        self.terminal.stop()
        self._after_terminal_stop(options)

    def render_now(self, force: bool = False) -> None:
        with self._state_lock:
            if force:
                self._reset_render_state()
            self._render_requested = False
            self._cancel_render_timer()
            self._last_render_at = time.perf_counter() * 1000
            self._do_render()

    def _next_tick(self, callback: Callable[[], None]) -> None:
        self._next_tick_callbacks.append(callback)
        if self._next_tick_scheduled:
            return
        self._next_tick_scheduled = True

        def run() -> None:
            with self._state_lock:
                try:
                    while self._next_tick_callbacks:
                        self._next_tick_callbacks.popleft()()
                finally:
                    self._next_tick_scheduled = False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._event_loop
            if loop is not None and loop.is_running() and not loop.is_closed():
                loop.call_soon_threadsafe(run)
            else:
                set_timeout(run, 0)
        else:
            self._event_loop = loop
            loop.call_soon(run)

    def request_render(self, force: bool = False) -> None:
        with self._state_lock:
            if force:
                self._reset_render_state()
                self._request_immediate_render()
                return
            if self._render_requested:
                return
            self._render_requested = True
            self._next_tick(self._schedule_render)

    def _request_immediate_render(self) -> None:
        self._cancel_render_timer()
        self._render_requested = True
        if self._immediate_render_scheduled:
            return
        self._immediate_render_scheduled = True

        def render() -> None:
            self._immediate_render_scheduled = False
            if self._stopped or not self._render_requested:
                return
            self._cancel_render_timer()
            self._render_requested = False
            self._last_render_at = time.perf_counter() * 1000
            self._do_render()

        self._next_tick(render)

    def _cancel_render_timer(self) -> None:
        if self._render_timer is not None:
            self._render_timer_generation += 1
            self._render_timer.cancel()
            self._render_timer = None

    def _schedule_render(self) -> None:
        if self._stopped or self._render_timer is not None or not self._render_requested:
            return
        elapsed = time.perf_counter() * 1000 - self._last_render_at
        delay = max(0, self.MIN_RENDER_INTERVAL_MS - elapsed)
        generation = self._render_timer_generation

        def render() -> None:
            with self._state_lock:
                if generation != self._render_timer_generation:
                    return
                self._render_timer = None
                if self._stopped or not self._render_requested:
                    return
                self._render_requested = False
                self._last_render_at = time.perf_counter() * 1000
                self._do_render()
                if self._render_requested:
                    self._schedule_render()

        self._render_timer = set_timeout(render, delay)

    def _handle_terminal_input(self, data: str) -> None:
        with self._state_lock:
            if self._consume_osc11_background_response(data) or self._consume_terminal_color_scheme_report(data):
                return
            if len(self._input_listeners):
                current = data
                for listener in self._input_listeners:
                    result = listener(current)
                    if result is not None and result.consume:
                        return
                    if result is not None and result.data is not None:
                        current = result.data
                if not current:
                    return
                data = current
            if self._consume_cell_size_response(data):
                return
            debug_keys = self._keybindings.get("tui.debug")
            if isinstance(debug_keys, str):
                debug_keys = [debug_keys]
            if any(matches_key(data, key) for key in debug_keys or ()) and self.on_debug is not None:
                self.on_debug()
                return
            focused_overlay = next((entry for entry in self._overlay_stack if entry.component is self._focused_component), None)
            if focused_overlay is not None and not self._is_overlay_visible(focused_overlay):
                top_visible = self._get_topmost_visible_overlay()
                if top_visible is not None:
                    self.set_focus(top_visible.component)
                else:
                    self._set_focus_internal(focused_overlay.pre_focus, "preserve")
            focus_is_overlay = any(entry.component is self._focused_component for entry in self._overlay_stack)
            if not focus_is_overlay:
                restore = self._get_visible_overlay_focus_restore()
                if isinstance(restore, _EligibleRestore):
                    self.set_focus(restore.overlay.component)
                elif isinstance(restore, _BlockedRestore) and restore.blocked_by is not self._focused_component:
                    if isinstance(restore.resume, _RestoreOverlay):
                        self.set_focus(restore.overlay.component)
                    else:
                        self._clear_overlay_focus_restore()
                        self.set_focus(restore.resume.target)
            focused = self._focused_component
            handler = getattr(focused, "handle_input", None)
            if handler is not None:
                if is_key_release(data) and not getattr(focused, "wants_key_release", False):
                    return
                handler(data)
                self._request_immediate_render()

    def _consume_osc11_background_response(self, data: str) -> bool:
        if self._pending_osc11_background_replies <= 0 or not is_osc11_background_color_response(data):
            return False
        rgb = parse_osc11_background_color(data)
        self._pending_osc11_background_replies -= 1
        query = self._pending_osc11_background_queries.pop(0) if self._pending_osc11_background_queries else None
        if query is not None and not query.settled:
            query.settled = True
            if query.timer is not None:
                query.timer.cancel()
                query.timer = None
            if query.resolve is not None:
                query.resolve(rgb)
            query.resolve = None
        return True

    def _consume_terminal_color_scheme_report(self, data: str) -> bool:
        scheme = parse_terminal_color_scheme_report(data)
        if scheme is None:
            return False
        for listener in self._terminal_color_scheme_listeners:
            listener(scheme)
        return True

    def _consume_cell_size_response(self, data: str) -> bool:
        match = re.fullmatch(r"\x1b\[6;([0-9]+);([0-9]+)t", data)
        if match is None:
            return False
        height_px, width_px = float(match[1]), float(match[2])
        if height_px <= 0 or width_px <= 0:
            return True
        set_cell_dimensions(CellDimensions(width_px, height_px))
        self.invalidate()
        self.request_render()
        return True

    def _resolve_overlay_layout(self, options: OverlayOptions | None, overlay_height: _Number, term_width: int, term_height: int) -> _OverlayLayout:
        options = options or OverlayOptions()
        margin = options.margin
        if isinstance(margin, (int, float)):
            margin = OverlayMargin(margin, margin, margin, margin)
        margin = margin or OverlayMargin()
        top = _maximum(0, margin.top if margin.top is not None else 0)
        right = _maximum(0, margin.right if margin.right is not None else 0)
        bottom = _maximum(0, margin.bottom if margin.bottom is not None else 0)
        left = _maximum(0, margin.left if margin.left is not None else 0)
        available_width = _maximum(1, term_width - left - right)
        available_height = _maximum(1, term_height - top - bottom)
        width = _parse_size_value(options.width, term_width)
        if width is None:
            width = _minimum(80, available_width)
        if options.min_width is not None:
            width = _maximum(width, options.min_width)
        width = _maximum(1, _minimum(width, available_width))
        max_height = _parse_size_value(options.max_height, term_height)
        if max_height is not None:
            max_height = _maximum(1, _minimum(max_height, available_height))
        effective_height = _minimum(overlay_height, max_height) if max_height is not None else overlay_height
        if isinstance(options.row, str):
            match = _PERCENTAGE.fullmatch(options.row)
            row = top + _floor(_maximum(0, available_height - effective_height) * (float(match[1]) / 100)) if match else self._resolve_anchor_row("center", effective_height, available_height, top)
        elif options.row is not None:
            row = options.row
        else:
            row = self._resolve_anchor_row(options.anchor or "center", effective_height, available_height, top)
        if isinstance(options.col, str):
            match = _PERCENTAGE.fullmatch(options.col)
            col = left + _floor(_maximum(0, available_width - width) * (float(match[1]) / 100)) if match else self._resolve_anchor_col("center", width, available_width, left)
        elif options.col is not None:
            col = options.col
        else:
            col = self._resolve_anchor_col(options.anchor or "center", width, available_width, left)
        if options.offset_y is not None:
            row += options.offset_y
        if options.offset_x is not None:
            col += options.offset_x
        row = _maximum(top, _minimum(row, term_height - bottom - effective_height))
        col = _maximum(left, _minimum(col, term_width - right - width))
        return _OverlayLayout(width, row, col, max_height)

    def _resolve_anchor_row(self, anchor: OverlayAnchor, height: _Number, available_height: _Number, top: _Number) -> _Number:
        if anchor in ("top-left", "top-center", "top-right"):
            return top
        if anchor in ("bottom-left", "bottom-center", "bottom-right"):
            return top + available_height - height
        if anchor in ("left-center", "center", "right-center"):
            return top + _floor((available_height - height) / 2)
        return math.nan

    def _resolve_anchor_col(self, anchor: OverlayAnchor, width: _Number, available_width: _Number, left: _Number) -> _Number:
        if anchor in ("top-left", "left-center", "bottom-left"):
            return left
        if anchor in ("top-right", "right-center", "bottom-right"):
            return left + available_width - width
        if anchor in ("top-center", "center", "bottom-center"):
            return left + _floor((available_width - width) / 2)
        return math.nan

    def _composite_overlays(self, lines: list[str], term_width: int, term_height: int) -> list[str]:
        if not self._overlay_stack:
            self._rendered_overlay_layouts = []
            return lines
        result = list(lines)
        for entry in self._overlay_stack:
            entry.bounds = None
        rendered: list[_RenderedOverlay] = []
        min_lines_needed: _Number = len(result)
        visible = [entry for entry in _array_values(self._overlay_stack) if self._is_overlay_visible(entry)]
        visible.sort(key=lambda entry: entry.focus_order)
        for entry in visible:
            layout = self._resolve_overlay_layout(entry.options, 0, term_width, term_height)
            width, max_height = layout.width, layout.max_height
            overlay_lines = entry.component.render(cast(int, width))
            if max_height is not None and len(overlay_lines) > max_height:
                overlay_lines = overlay_lines[:math.trunc(max_height)]
            layout = self._resolve_overlay_layout(entry.options, len(overlay_lines), term_width, term_height)
            entry.bounds = OverlayBounds(cast(int, layout.row), cast(int, layout.col), cast(int, width), len(overlay_lines))
            rendered.append(_RenderedOverlay(entry, overlay_lines, layout.row, layout.col, width))
            min_lines_needed = _maximum(min_lines_needed, layout.row + len(overlay_lines))
        self._rendered_overlay_layouts = [
            _RenderedOverlayLayout(item.entry, item.row, item.col, item.width, len(item.overlay_lines)) for item in rendered
        ]
        working_height = _maximum(len(result), term_height, min_lines_needed)
        while len(result) < working_height:
            result.append("")
        viewport_start = _maximum(0, working_height - term_height)
        for item in rendered:
            for offset, line in enumerate(item.overlay_lines):
                index = viewport_start + item.row + offset
                if 0 <= index < len(result):
                    truncated = slice_by_column(line, 0, cast(int, item.width), True) if visible_width(line) > item.width else line
                    # Integral JS Numbers are ordinary array indices; fractional
                    # row positions have no corresponding line, as in JS.
                    if isinstance(index, float) and index.is_integer():
                        index = int(index)
                    result[cast(int, index)] = composite_tui_line(result[cast(int, index)], truncated, cast(int, item.col), cast(int, item.width), term_width)
        return result

    def _apply_line_resets(self, lines: list[str]) -> list[str]:
        for index, line in enumerate(lines):
            if not is_image_line(line):
                lines[index] = normalize_terminal_output(line) + _SEGMENT_RESET
        return lines

    def _extract_cursor_position(self, lines: list[str], height: int) -> CursorPosition | None:
        viewport_top = max(0, len(lines) - height)
        for row in range(len(lines) - 1, viewport_top - 1, -1):
            line = lines[row]
            index = line.find(CURSOR_MARKER)
            if index != -1:
                col = visible_width(line[:index])
                lines[row] = line[:index] + line[index + len(CURSOR_MARKER):]
                return CursorPosition(row, col)
        return None

    def query_terminal_background_color(self, *, timeout_ms: float) -> Awaitable[RgbColor | None]:
        promise: _QueryPromise[RgbColor | None] = _QueryPromise()
        with self._state_lock:
            try:
                query = _PendingOsc11BackgroundQuery(False, promise.resolve)

                def timeout() -> None:
                    with self._state_lock:
                        if query.settled:
                            return
                        query.settled = True
                        query.timer = None
                        if query.resolve is not None:
                            query.resolve(None)
                        query.resolve = None

                query.timer = set_timeout(timeout, timeout_ms)
                self._pending_osc11_background_queries.append(query)
                self._pending_osc11_background_replies += 1
                self.terminal.write("\x1b]11;?\x07")
            except Exception as error:
                promise.reject(error)
        return promise

    def query_terminal_color_scheme(self, *, timeout_ms: float) -> Awaitable[TerminalColorScheme | None]:
        promise: _QueryPromise[TerminalColorScheme | None] = _QueryPromise()
        with self._state_lock:
            try:
                settled = False
                timer: TimerHandle | None = None
                unsubscribe: Callable[[], None] = lambda: None

                def settle(scheme: TerminalColorScheme | None) -> None:
                    nonlocal settled, timer
                    with self._state_lock:
                        if settled:
                            return
                        settled = True
                        if timer is not None:
                            timer.cancel()
                            timer = None
                        unsubscribe()
                        promise.resolve(scheme)

                unsubscribe = self.on_terminal_color_scheme_change(settle)
                timer = set_timeout(lambda: settle(None), timeout_ms)
                self.terminal.write("\x1b[?996n")
            except Exception as error:
                promise.reject(error)
        return promise


__all__ = [
    "AppKeybindings", "CURSOR_MARKER", "DEFAULT_APP_KEYBINDINGS", "VIEWPORT_TUI",
    "Component", "Container", "CursorPosition", "Focusable", "FocusableComponent",
    "InputComponent", "MouseComponent", "OverlayAnchor", "OverlayBounds", "OverlayHandle",
    "OverlayMargin", "OverlayMouseDispatch", "OverlayOptions", "OverlayUnfocusOptions",
    "SizeValue", "TUI", "TuiBase", "TuiInputListener", "TuiInputListenerResult", "TuiMode",
    "TuiMouseButton", "TuiMouseDispatchResult", "TuiMouseDispatchTarget", "TuiMouseEvent",
    "TuiMouseEventResult", "TuiMouseEventType", "TuiStopOptions", "ViewportTUI",
    "composite_tui_line", "dispatch_mouse_event", "is_focusable", "is_viewport_tui",
    "retarget_mouse_event", "visible_width",
]
