"""Loader cancellation through the configurable select-cancel binding."""

from __future__ import annotations

from collections.abc import Callable

from .._component import TUI
from ..abort import AbortController, AbortSignal
from ..keybindings import get_keybindings
from .loader import Loader, LoaderIndicatorOptions


class CancellableLoader(Loader):
    def __init__(
        self, ui: TUI, spinner_color_fn: Callable[[str], str], message_color_fn: Callable[[str], str],
        message: str = "Loading...", indicator: LoaderIndicatorOptions | None = None,
    ) -> None:
        super().__init__(ui, spinner_color_fn, message_color_fn, message, indicator)
        self._abort_controller = AbortController()
        self.on_abort: Callable[[], None] | None = None

    @property
    def signal(self) -> AbortSignal:
        return self._abort_controller.signal

    @property
    def aborted(self) -> bool:
        return self._abort_controller.signal.aborted

    def handle_input(self, data: str) -> None:
        if get_keybindings().matches(data, "tui.select.cancel"):
            self._abort_controller.abort()
            if self.on_abort is not None:
                self.on_abort()

    def dispose(self) -> None:
        self.stop()
