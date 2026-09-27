"""Component contracts, mouse dispatch and Container from ``tui.ts``.

Kept separate from the render scheduler so leaf components need no terminal driver.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeGuard

from ._terminal_types import Terminal
from .terminal_colors import RgbColor, TerminalColorScheme

type TuiMouseEventType = Literal["press", "release", "move", "drag", "click", "wheel"]
type TuiMouseButton = Literal["left", "middle", "right", "none"]
type TuiMode = Literal["regular", "fullscreen"]
type OverlayAnchor = Literal[
    "center", "top-left", "top-right", "bottom-left", "bottom-right",
    "top-center", "bottom-center", "left-center", "right-center",
]
type SizeValue = int | float | str

CURSOR_MARKER = "\x1b_pi:c\x07"
VIEWPORT_TUI = "__pi_tui_viewport__"


@dataclass
class TuiMouseEvent:
    type: TuiMouseEventType
    button: TuiMouseButton
    x: int
    y: int
    screen_x: int
    screen_y: int
    width: int
    height: int
    shift: bool
    alt: bool
    ctrl: bool
    wheel_delta: int | None = None
    click_count: int | None = None


@dataclass(kw_only=True)
class TuiMouseEventResult:
    handled: bool | None = None
    capture: bool | None = None
    focus: bool | None = None
    render: bool | None = None


@dataclass
class TuiMouseDispatchTarget:
    component: Component
    origin_x: int
    origin_y: int
    width: int
    height: int


@dataclass(kw_only=True)
class TuiMouseDispatchResult(TuiMouseEventResult):
    handled: Literal[True] = True
    target: TuiMouseDispatchTarget
    focus_target: Component | None = None


class Component(Protocol):
    def render(self, width: int) -> list[str]: ...

    def invalidate(self) -> None: ...


class InputComponent(Component, Protocol):
    def handle_input(self, data: str) -> None: ...


class MouseComponent(Component, Protocol):
    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult | None: ...


class Focusable(Protocol):
    focused: bool


class FocusableComponent(Component, Focusable, Protocol):
    pass


def is_focusable(component: Component | None) -> TypeGuard[FocusableComponent]:
    return component is not None and hasattr(component, "focused")


def dispatch_mouse_event(component: Component, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
    handler = getattr(component, "handle_mouse", None)
    result: TuiMouseEventResult | None = handler(event) if handler is not None else None
    if result is None:
        return None
    if isinstance(result, TuiMouseDispatchResult):
        return result
    if not result.handled and not result.capture and not result.focus:
        return None
    return TuiMouseDispatchResult(
        capture=result.capture, focus=result.focus, render=result.render,
        focus_target=component if result.focus else None,
        target=TuiMouseDispatchTarget(
            component, event.screen_x - event.x, event.screen_y - event.y, event.width, event.height,
        ),
    )


def retarget_mouse_event(event: TuiMouseEvent, target: TuiMouseDispatchTarget) -> TuiMouseEvent:
    return replace(event, x=event.screen_x - target.origin_x, y=event.screen_y - target.origin_y,
                   width=target.width, height=target.height)


@dataclass
class TuiInputListenerResult:
    consume: bool | None = None
    data: str | None = None


type TuiInputListener = Callable[[str], TuiInputListenerResult | None]


@dataclass
class OverlayMargin:
    top: int | float | None = None
    right: int | float | None = None
    bottom: int | float | None = None
    left: int | float | None = None


@dataclass
class OverlayOptions:
    width: SizeValue | None = None
    min_width: int | float | None = None
    max_height: SizeValue | None = None
    anchor: OverlayAnchor | None = None
    offset_x: int | float | None = None
    offset_y: int | float | None = None
    row: SizeValue | None = None
    col: SizeValue | None = None
    margin: OverlayMargin | int | float | None = None
    visible: Callable[[int, int], bool] | None = None
    non_capturing: bool | None = None


@dataclass
class OverlayUnfocusOptions:
    target: Component | None


@dataclass
class OverlayBounds:
    row: int
    col: int
    width: int
    height: int


class OverlayHandle(Protocol):
    def hide(self) -> None: ...

    def set_hidden(self, hidden: bool) -> None: ...

    def is_hidden(self) -> bool: ...

    def focus(self) -> None: ...

    def unfocus(self, options: OverlayUnfocusOptions | None = None) -> None: ...

    def is_focused(self) -> bool: ...

    def get_bounds(self) -> OverlayBounds | None: ...


@dataclass
class TuiStopOptions:
    preserve_screen: bool | None = None


class TUI(Component, Protocol):
    children: list[Component]
    terminal: Terminal
    on_debug: Callable[[], None] | None

    @property
    def mode(self) -> TuiMode: ...

    @property
    def full_redraws(self) -> int: ...

    def add_child(self, component: Component) -> None: ...

    def remove_child(self, component: Component) -> None: ...

    def clear(self) -> None: ...

    def get_show_hardware_cursor(self) -> bool: ...

    def set_show_hardware_cursor(self, enabled: bool) -> None: ...

    def get_clear_on_shrink(self) -> bool: ...

    def set_clear_on_shrink(self, enabled: bool) -> None: ...

    def set_focus(self, component: Component | None) -> None: ...

    def show_overlay(self, component: Component, options: OverlayOptions | None = None) -> OverlayHandle: ...

    def hide_overlay(self) -> None: ...

    def has_overlay(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self, options: TuiStopOptions | None = None) -> None: ...

    def render_now(self, force: bool = False) -> None: ...

    def request_render(self, force: bool = False) -> None: ...

    def add_input_listener(self, listener: TuiInputListener) -> Callable[[], None]: ...

    def remove_input_listener(self, listener: TuiInputListener) -> None: ...

    def on_terminal_color_scheme_change(self, listener: Callable[[TerminalColorScheme], None]) -> Callable[[], None]: ...

    def set_terminal_color_scheme_notifications(self, enabled: bool) -> None: ...

    def query_terminal_background_color(self, *, timeout_ms: float) -> Awaitable[RgbColor | None]: ...

    def query_terminal_color_scheme(self, *, timeout_ms: float) -> Awaitable[TerminalColorScheme | None]: ...


class ViewportTUI(TUI, Protocol):
    __pi_tui_viewport__: Literal[True]

    def set_layout_root(self, component: Component | None) -> None: ...


def is_viewport_tui(tui: TUI) -> TypeGuard[ViewportTUI]:
    return getattr(tui, VIEWPORT_TUI, None) is True


class Container:
    def __init__(self) -> None:
        self.children: list[Component] = []
        self._mouse_width: int | None = None
        self._mouse_children: list[tuple[Component, int]] = []

    def add_child(self, component: Component) -> None:
        self.children.append(component)

    def remove_child(self, component: Component) -> None:
        for index, child in enumerate(self.children):
            if child is component:
                del self.children[index]
                break

    def clear(self) -> None:
        self.children = []

    def invalidate(self) -> None:
        for child in self.children:
            callback = getattr(child, "invalidate", None)
            if callback is not None:
                callback()

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
        if event.y < 0 or event.y >= event.height:
            return None
        mouse_children = self._mouse_children if self._mouse_width == event.width else [
            (component, len(component.render(event.width))) for component in self.children
        ]
        child_y = 0
        for child, child_height in mouse_children:
            if child_y <= event.y < child_y + child_height:
                result = dispatch_mouse_event(child, replace(event, y=event.y - child_y, height=child_height))
                if result is not None and result.focus and getattr(self, "handle_input", None):
                    return replace(result, focus_target=self)
                return result
            child_y += child_height
        return None

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        mouse_children: list[tuple[Component, int]] = []
        for child in self.children:
            child_lines = child.render(width)
            mouse_children.append((child, len(child_lines)))
            lines.extend(child_lines)
        self._mouse_width = width
        self._mouse_children = mouse_children
        return lines
