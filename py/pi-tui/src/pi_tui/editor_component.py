"""Custom editor contracts from ``editor-component.ts``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ._component import Component
from .autocomplete import AutocompleteProvider


class EditorComponent(Component, Protocol):
    on_submit: Callable[[str], None] | None
    on_change: Callable[[str], None] | None

    def get_text(self) -> str: ...

    def set_text(self, text: str) -> None: ...

    def handle_input(self, data: str) -> None: ...


class HistoryEditorComponent(EditorComponent, Protocol):
    def add_to_history(self, text: str) -> None: ...


class InsertionEditorComponent(EditorComponent, Protocol):
    def insert_text_at_cursor(self, text: str) -> None: ...


class ExpandedTextEditorComponent(EditorComponent, Protocol):
    def get_expanded_text(self) -> str: ...


class AutocompleteEditorComponent(EditorComponent, Protocol):
    def set_autocomplete_provider(self, provider: AutocompleteProvider) -> None: ...


class StyledEditorComponent(EditorComponent, Protocol):
    border_color: Callable[[str], str] | None

    def set_padding_x(self, padding: int) -> None: ...

    def set_autocomplete_max_visible(self, max_visible: int) -> None: ...
