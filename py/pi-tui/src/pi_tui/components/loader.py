"""Animated loader with configurable frames from ``components/loader.ts``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .._component import TUI
from .._timers import IntervalHandle
from .text import Text

DEFAULT_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
DEFAULT_INTERVAL_MS = 80


@dataclass
class LoaderIndicatorOptions:
    frames: list[str] | None = None
    interval_ms: float | None = None


class Loader(Text):
    def __init__(
        self, ui: TUI, spinner_color_fn: Callable[[str], str], message_color_fn: Callable[[str], str],
        message: str = "Loading...", indicator: LoaderIndicatorOptions | None = None,
    ) -> None:
        super().__init__("", 1, 0)
        self._frames = DEFAULT_FRAMES.copy()
        self._interval_ms: float = DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self._interval_id: IntervalHandle | None = None
        self._ui = ui
        self._render_indicator_verbatim = False
        self._spinner_color_fn = spinner_color_fn
        self._message_color_fn = message_color_fn
        self._message = message
        self.set_indicator(indicator)

    def render(self, width: int) -> list[str]:
        return ["", *super().render(width)]

    def start(self) -> None:
        self._update_display()
        self._restart_animation()

    def stop(self) -> None:
        if self._interval_id is not None:
            self._interval_id.cancel()
            self._interval_id = None

    def set_message(self, message: str) -> None:
        self._message = message
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def set_indicator(self, indicator: LoaderIndicatorOptions | None = None) -> None:
        self._render_indicator_verbatim = indicator is not None
        self._frames = list(indicator.frames) if indicator is not None and indicator.frames is not None else DEFAULT_FRAMES.copy()
        self._interval_ms = indicator.interval_ms if indicator is not None and indicator.interval_ms is not None and indicator.interval_ms > 0 else DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self.start()

    def _restart_animation(self) -> None:
        self.stop()
        if len(self._frames) <= 1:
            return

        def advance() -> None:
            self._current_frame = (self._current_frame + 1) % len(self._frames)
            self._update_display()

        self._interval_id = IntervalHandle(advance, self._interval_ms)

    def _get_rendered_indicator(self) -> str:
        frame = self._frames[self._current_frame] if self._current_frame < len(self._frames) else ""
        return frame if self._render_indicator_verbatim else self._spinner_color_fn(frame)

    def _update_display(self) -> None:
        frame = self._get_rendered_indicator()
        indicator = frame + " " if frame else ""
        self.set_text(indicator + self._message_color_fn(self._message))
        if self._ui is not None:
            self._ui.request_render()
