"""Configurable keybinding registry from ``tui/src/keybindings.ts``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .keys import KeyId, matches_key

# TS declaration merging permits downstream action IDs; Python uses open maps.
type Keybindings = dict[str, Literal[True]]
type Keybinding = str


@dataclass
class KeybindingDefinition:
    default_keys: KeyId | list[KeyId]
    description: str | None = None


type KeybindingDefinitions = dict[str, KeybindingDefinition]
type KeybindingsConfig = dict[str, KeyId | list[KeyId] | None]

TUI_KEYBINDINGS: KeybindingDefinitions = {
    "tui.editor.cursorUp": KeybindingDefinition("up", "Move cursor up"),
    "tui.editor.cursorDown": KeybindingDefinition("down", "Move cursor down"),
    "tui.editor.historyPrevious": KeybindingDefinition([], "Select previous prompt history entry"),
    "tui.editor.historyNext": KeybindingDefinition([], "Select next prompt history entry"),
    "tui.editor.cursorLeft": KeybindingDefinition(["left", "ctrl+b"], "Move cursor left"),
    "tui.editor.cursorRight": KeybindingDefinition(["right", "ctrl+f"], "Move cursor right"),
    "tui.editor.cursorWordLeft": KeybindingDefinition(["alt+left", "ctrl+left", "alt+b"], "Move cursor word left"),
    "tui.editor.cursorWordRight": KeybindingDefinition(["alt+right", "ctrl+right", "alt+f"], "Move cursor word right"),
    "tui.editor.cursorLineStart": KeybindingDefinition(["home", "ctrl+home", "ctrl+a"], "Move to line start"),
    "tui.editor.cursorLineEnd": KeybindingDefinition(["end", "ctrl+end", "ctrl+e"], "Move to line end"),
    "tui.editor.jumpForward": KeybindingDefinition("ctrl+]", "Jump forward to character"),
    "tui.editor.jumpBackward": KeybindingDefinition("ctrl+alt+]", "Jump backward to character"),
    "tui.editor.pageUp": KeybindingDefinition(["pageUp", "ctrl+pageUp"], "Page up"),
    "tui.editor.pageDown": KeybindingDefinition(["pageDown", "ctrl+pageDown"], "Page down"),
    "tui.editor.deleteCharBackward": KeybindingDefinition("backspace", "Delete character backward"),
    "tui.editor.deleteCharForward": KeybindingDefinition(["delete", "ctrl+d"], "Delete character forward"),
    "tui.editor.deleteWordBackward": KeybindingDefinition(["ctrl+w", "alt+backspace"], "Delete word backward"),
    "tui.editor.deleteWordForward": KeybindingDefinition(["alt+d", "alt+delete"], "Delete word forward"),
    "tui.editor.deleteToLineStart": KeybindingDefinition("ctrl+u", "Delete to line start"),
    "tui.editor.deleteToLineEnd": KeybindingDefinition("ctrl+k", "Delete to line end"),
    "tui.editor.yank": KeybindingDefinition("ctrl+y", "Yank"),
    "tui.editor.yankPop": KeybindingDefinition("alt+y", "Yank pop"),
    "tui.editor.undo": KeybindingDefinition("ctrl+-", "Undo"),
    "tui.input.newLine": KeybindingDefinition(["shift+enter", "ctrl+j"], "Insert newline"),
    "tui.input.submit": KeybindingDefinition("enter", "Submit input"),
    "tui.input.tab": KeybindingDefinition("tab", "Tab / autocomplete"),
    "tui.input.copy": KeybindingDefinition("ctrl+c", "Copy selection"),
    "tui.select.up": KeybindingDefinition("up", "Move selection up"),
    "tui.select.down": KeybindingDefinition("down", "Move selection down"),
    "tui.select.pageUp": KeybindingDefinition("pageUp", "Selection page up"),
    "tui.select.pageDown": KeybindingDefinition("pageDown", "Selection page down"),
    "tui.select.confirm": KeybindingDefinition("enter", "Confirm selection"),
    "tui.select.cancel": KeybindingDefinition(["escape", "ctrl+c"], "Cancel selection"),
    # These intentionally shadow the unmodified editor bindings in fullscreen.
    "tui.altScreen.pageUp": KeybindingDefinition("pageUp", "Scroll viewport up one page"),
    "tui.altScreen.pageDown": KeybindingDefinition("pageDown", "Scroll viewport down one page"),
    "tui.altScreen.halfPageUp": KeybindingDefinition([], "Scroll viewport up half a page"),
    "tui.altScreen.halfPageDown": KeybindingDefinition([], "Scroll viewport down half a page"),
    "tui.altScreen.lineUp": KeybindingDefinition([], "Scroll viewport up one line"),
    "tui.altScreen.lineDown": KeybindingDefinition([], "Scroll viewport down one line"),
    "tui.altScreen.previousPrompt": KeybindingDefinition(["ctrl+shift+up", "ctrl+up"], "Jump to previous semantic prompt"),
    "tui.altScreen.nextPrompt": KeybindingDefinition(["ctrl+shift+down", "ctrl+down"], "Jump to next semantic prompt"),
    "tui.altScreen.search": KeybindingDefinition("ctrl+shift+f", "Search the primary scroll view"),
    "tui.altScreen.searchNext": KeybindingDefinition(["enter", "ctrl+g"], "Select the next search match"),
    "tui.altScreen.searchPrevious": KeybindingDefinition(["shift+enter", "ctrl+shift+g"], "Select the previous search match"),
    "tui.altScreen.searchClose": KeybindingDefinition("escape", "Close transcript search"),
    "tui.altScreen.top": KeybindingDefinition("home", "Scroll viewport to top"),
    "tui.altScreen.bottom": KeybindingDefinition("end", "Scroll viewport to bottom"),
}

# These editor-specific compatibility branches are configurable in Python;
# keeping separate actions leaves the single-line Input defaults unchanged.
DEFAULT_EDITOR_KEYBINDINGS: KeybindingDefinitions = {
    "tui.editor.shiftedBackward": KeybindingDefinition("shift+backspace", "Delete character backward with Shift"),
    "tui.editor.shiftedForward": KeybindingDefinition("shift+delete", "Delete character forward with Shift"),
    "tui.editor.shiftedSpace": KeybindingDefinition("shift+space", "Insert space with Shift"),
    "tui.editor.enterFallback": KeybindingDefinition("enter", "Recognize Enter for the backslash submit fallback"),
}
TUI_KEYBINDINGS.update(DEFAULT_EDITOR_KEYBINDINGS)


@dataclass
class KeybindingConflict:
    key: KeyId
    keybindings: list[str]


def _normalize_keys(keys: KeyId | list[KeyId] | None) -> list[KeyId]:
    if keys is None:
        return []
    key_list = keys if isinstance(keys, list) else [keys]
    seen: set[KeyId] = set()
    result: list[KeyId] = []
    for key in key_list:
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _object_keys(record: Mapping[str, object]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    others: list[str] = []
    for key in record:
        if key.isascii() and key.isdigit() and len(key) <= 10 and (key == "0" or key[0] != "0") and int(key) < 0xFFFFFFFF:
            indexed.append((int(key), key))
        else:
            others.append(key)
    return [key for _, key in sorted(indexed)] + others


class KeybindingsManager:
    def __init__(self, definitions: KeybindingDefinitions, user_bindings: KeybindingsConfig | None = None) -> None:
        self._definitions = definitions
        self._user_bindings = {} if user_bindings is None else user_bindings
        self._keys_by_id: dict[Keybinding, list[KeyId]] = {}
        self._conflicts: list[KeybindingConflict] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self._keys_by_id.clear()
        self._conflicts = []
        user_claims: dict[KeyId, dict[Keybinding, None]] = {}
        for keybinding in _object_keys(self._user_bindings):
            keys = self._user_bindings[keybinding]
            if keybinding not in self._definitions:
                continue
            for key in _normalize_keys(keys):
                claimants = user_claims.setdefault(key, {})
                claimants[keybinding] = None
        for key, keybindings in user_claims.items():
            if len(keybindings) > 1:
                self._conflicts.append(KeybindingConflict(key, list(keybindings)))
        for id in _object_keys(self._definitions):
            definition = self._definitions[id]
            user_keys = self._user_bindings.get(id)
            keys = _normalize_keys(definition.default_keys) if user_keys is None else _normalize_keys(user_keys)
            self._keys_by_id[id] = keys

    def matches(self, data: str, keybinding: Keybinding) -> bool:
        keys = self._keys_by_id.get(keybinding, [])
        for key in keys:
            if matches_key(data, key):
                return True
        return False

    def get_keys(self, keybinding: Keybinding) -> list[KeyId]:
        return list(self._keys_by_id.get(keybinding, []))

    def get_definition(self, keybinding: Keybinding) -> KeybindingDefinition | None:
        return self._definitions.get(keybinding)

    def get_conflicts(self) -> list[KeybindingConflict]:
        return [KeybindingConflict(conflict.key, list(conflict.keybindings)) for conflict in self._conflicts]

    def set_user_bindings(self, user_bindings: KeybindingsConfig) -> None:
        self._user_bindings = user_bindings
        self._rebuild()

    def get_user_bindings(self) -> KeybindingsConfig:
        return {key: self._user_bindings[key] for key in _object_keys(self._user_bindings)}

    def get_resolved_bindings(self) -> KeybindingsConfig:
        resolved: KeybindingsConfig = {}
        for id in _object_keys(self._definitions):
            keys = self._keys_by_id.get(id, [])
            resolved[id] = keys[0] if len(keys) == 1 else list(keys)
        return resolved


_global_keybindings: KeybindingsManager | None = None


def set_keybindings(keybindings: KeybindingsManager) -> None:
    global _global_keybindings
    _global_keybindings = keybindings


def get_keybindings() -> KeybindingsManager:
    global _global_keybindings
    if _global_keybindings is None:
        _global_keybindings = KeybindingsManager(TUI_KEYBINDINGS)
    return _global_keybindings


__all__ = [
    "Keybindings", "Keybinding", "KeybindingDefinition", "KeybindingDefinitions", "KeybindingsConfig",
    "TUI_KEYBINDINGS", "DEFAULT_EDITOR_KEYBINDINGS", "KeybindingConflict", "KeybindingsManager", "set_keybindings", "get_keybindings",
]
