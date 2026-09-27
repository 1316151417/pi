"""Command, path, and fd-backed attachment completion from ``autocomplete.ts``."""

from __future__ import annotations

import asyncio
import inspect
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Protocol

from icu import Collator

from ._javascript import js_trim, utf16_length, utf16_slice
from .fuzzy import fuzzy_filter

_PATH_DELIMITERS = frozenset(" \t\"'=")
_COLLATOR = Collator.createInstance()


class AutocompleteAbortSignal(Protocol):
    @property
    def aborted(self) -> bool: ...

    def add_event_listener(self, event: str, callback: Callable[[], None], *, once: bool = False) -> None: ...

    def remove_event_listener(self, event: str, callback: Callable[[], None]) -> None: ...


@dataclass
class AutocompleteItem:
    value: str
    label: str
    description: str | None = None


@dataclass
class SlashCommand:
    name: str
    description: str | None = None
    argument_hint: str | None = None
    get_argument_completions: Callable[[str], list[AutocompleteItem] | None | Awaitable[list[AutocompleteItem] | None]] | None = None


@dataclass
class AutocompleteSuggestions:
    items: list[AutocompleteItem]
    prefix: str


@dataclass
class AutocompleteOptions:
    signal: AutocompleteAbortSignal
    force: bool = False


@dataclass
class CompletionResult:
    lines: list[str]
    cursor_line: int
    cursor_col: int


class AutocompleteProvider(Protocol):
    async def get_suggestions(
        self, lines: list[str], cursor_line: int, cursor_col: int, options: AutocompleteOptions,
    ) -> AutocompleteSuggestions | None: ...

    def apply_completion(
        self, lines: list[str], cursor_line: int, cursor_col: int, item: AutocompleteItem, prefix: str,
    ) -> CompletionResult: ...


class TriggeredAutocompleteProvider(AutocompleteProvider, Protocol):
    trigger_characters: list[str]


class FileAutocompleteProvider(AutocompleteProvider, Protocol):
    def should_trigger_file_completion(self, lines: list[str], cursor_line: int, cursor_col: int) -> bool: ...


@dataclass
class _FileEntry:
    path: str
    is_directory: bool


@dataclass
class _ScopedQuery:
    base_dir: str
    query: str
    display_base: str


def _to_display_path(value: str) -> str:
    return value.replace("\\", "/")


def _build_fd_path_query(query: str) -> str:
    normalized = _to_display_path(query)
    if "/" not in normalized:
        return normalized
    has_trailing_separator = normalized.endswith("/")
    trimmed = normalized.strip("/")
    if not trimmed:
        return normalized
    separator_pattern = r"[\\/]"
    segments = [re.sub(r"([.*+?^${}()|\[\]\\])", r"\\\1", part) for part in trimmed.split("/") if part]
    if not segments:
        return normalized
    pattern = separator_pattern.join(segments)
    return pattern + separator_pattern if has_trailing_separator else pattern


def _find_last_delimiter(text: str) -> int:
    for index in range(len(text) - 1, -1, -1):
        if text[index] in _PATH_DELIMITERS:
            return index
    return -1


def _extract_quoted_prefix(text: str) -> str | None:
    in_quotes = False
    quote_start = -1
    for index, char in enumerate(text):
        if char == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = index
    if not in_quotes:
        return None
    if quote_start > 0 and text[quote_start - 1] == "@":
        at_start = quote_start - 1
        if at_start != 0 and text[at_start - 1] not in _PATH_DELIMITERS:
            return None
        return text[at_start:]
    if quote_start != 0 and text[quote_start - 1] not in _PATH_DELIMITERS:
        return None
    return text[quote_start:]


def _parse_path_prefix(prefix: str) -> tuple[str, bool, bool]:
    if prefix.startswith('@"'):
        return prefix[2:], True, True
    if prefix.startswith('"'):
        return prefix[1:], False, True
    if prefix.startswith("@"):
        return prefix[1:], True, False
    return prefix, False, False


def _build_completion_value(path: str, *, is_at_prefix: bool, is_quoted_prefix: bool) -> str:
    prefix = "@" if is_at_prefix else ""
    if is_quoted_prefix or " " in path:
        return f'{prefix}"{path}"'
    return prefix + path


def _basename(path: str) -> str:
    return os.path.basename(path.rstrip(os.sep + (os.altsep or "")))


def _dirname(path: str) -> str:
    stripped = path.rstrip(os.sep + (os.altsep or ""))
    if not stripped and os.path.isabs(path):
        return os.path.splitdrive(path)[0] + os.sep
    return os.path.dirname(stripped) or "."


def _join(*parts: str) -> str:
    # Node path.join normalizes dot components and does not discard earlier
    # components when a later component starts with a separator.
    joined = os.sep.join(part for part in parts if part)
    normalized = os.path.normpath(joined)
    if joined.endswith(os.sep) and not normalized.endswith(os.sep):
        normalized += os.sep
    return normalized


async def _walk_directory_with_fd(
    base_dir: str, fd_path: str, query: str, max_results: int,
    signal: AutocompleteAbortSignal, max_depth: int | None = None,
) -> list[_FileEntry]:
    args = [
        "--base-directory", base_dir, "--max-results", str(max_results),
        "--type", "f", "--type", "d", "--follow", "--hidden",
        "--exclude", ".git", "--exclude", ".git/*", "--exclude", ".git/**",
    ]
    if max_depth is not None:
        args.extend(("--max-depth", str(max_depth)))
    if "/" in _to_display_path(query):
        args.append("--full-path")
    if query:
        args.append(_build_fd_path_query(query))
    if signal.aborted:
        return []
    try:
        child = await asyncio.create_subprocess_exec(
            fd_path, *args, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return []

    def on_abort() -> None:
        if child.returncode is None:
            try:
                child.kill()
            except ProcessLookupError:
                pass

    signal.add_event_listener("abort", on_abort, once=True)
    if signal.aborted:
        on_abort()
    try:
        stdout, _ = await child.communicate()
    except asyncio.CancelledError:
        on_abort()
        await child.wait()
        raise
    finally:
        signal.remove_event_listener("abort", on_abort)
    if signal.aborted or child.returncode != 0 or not stdout:
        return []
    results: list[_FileEntry] = []
    for line in js_trim(stdout.decode("utf-8", errors="replace")).split("\n"):
        if not line:
            continue
        display_line = _to_display_path(line)
        has_trailing_separator = display_line.endswith("/")
        normalized_path = display_line[:-1] if has_trailing_separator else display_line
        if normalized_path == ".git" or normalized_path.startswith(".git/") or "/.git/" in normalized_path:
            continue
        results.append(_FileEntry(display_line, has_trailing_separator))
    return results


class CombinedAutocompleteProvider:
    def __init__(
        self, commands: list[SlashCommand | AutocompleteItem] | None = None,
        *, base_path: str, fd_path: str | None = None,
    ) -> None:
        self._commands = commands if commands is not None else []
        self._base_path = base_path
        self._fd_path = fd_path

    async def get_suggestions(
        self, lines: list[str], cursor_line: int, cursor_col: int, options: AutocompleteOptions,
    ) -> AutocompleteSuggestions | None:
        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        text_before_cursor = utf16_slice(current_line, 0, cursor_col)
        at_prefix = self._extract_at_prefix(text_before_cursor)
        if at_prefix:
            raw_prefix, _, is_quoted_prefix = _parse_path_prefix(at_prefix)
            suggestions = await self._get_fuzzy_file_suggestions(raw_prefix, is_quoted_prefix, options.signal)
            return AutocompleteSuggestions(suggestions, at_prefix) if suggestions else None
        if not options.force and text_before_cursor.startswith("/"):
            space_index = text_before_cursor.find(" ")
            if space_index == -1:
                prefix = text_before_cursor[1:]
                command_items: list[AutocompleteItem] = []
                for command in self._commands:
                    name = command.name if isinstance(command, SlashCommand) else command.value
                    hint = command.argument_hint if isinstance(command, SlashCommand) else None
                    description = command.description or ""
                    full_description = (f"{hint} — {description}" if description else hint) if hint else description
                    command_items.append(AutocompleteItem(name, name, full_description or None))
                filtered = fuzzy_filter(command_items, prefix, lambda item: item.value)
                return AutocompleteSuggestions(filtered, text_before_cursor) if filtered else None
            command_name = text_before_cursor[1:space_index]
            argument_text = text_before_cursor[space_index + 1:]
            command = next((
                item for item in self._commands
                if (item.name if isinstance(item, SlashCommand) else item.value) == command_name
            ), None)
            if not isinstance(command, SlashCommand) or command.get_argument_completions is None:
                return None
            result = command.get_argument_completions(argument_text)
            argument_suggestions = await result if inspect.isawaitable(result) else result
            if not isinstance(argument_suggestions, list) or not argument_suggestions:
                return None
            return AutocompleteSuggestions(argument_suggestions, argument_text)
        path_match = self._extract_path_prefix(text_before_cursor, options.force)
        if path_match is None:
            return None
        suggestions = self._get_file_suggestions(path_match)
        return AutocompleteSuggestions(suggestions, path_match) if suggestions else None

    def apply_completion(
        self, lines: list[str], cursor_line: int, cursor_col: int, item: AutocompleteItem, prefix: str,
    ) -> CompletionResult:
        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        before_prefix = utf16_slice(current_line, 0, cursor_col - utf16_length(prefix))
        after_cursor = utf16_slice(current_line, cursor_col)
        quoted_prefix = prefix.startswith(('"', '@"'))
        adjusted_after = after_cursor[1:] if quoted_prefix and item.value.endswith('"') and after_cursor.startswith('"') else after_cursor
        slash_command = prefix.startswith("/") and not js_trim(before_prefix) and "/" not in prefix[1:]
        if slash_command:
            new_line = f"{before_prefix}/{item.value} {adjusted_after}"
            new_cursor = utf16_length(before_prefix) + utf16_length(item.value) + 2
        else:
            is_directory = item.label.endswith("/")
            suffix = " " if prefix.startswith("@") and not is_directory else ""
            new_line = before_prefix + item.value + suffix + adjusted_after
            cursor_offset = utf16_length(item.value) - (1 if is_directory and item.value.endswith('"') else 0)
            new_cursor = utf16_length(before_prefix) + cursor_offset + len(suffix)
        new_lines = lines.copy()
        if cursor_line >= len(new_lines):
            new_lines.extend([""] * (cursor_line + 1 - len(new_lines)))
        new_lines[cursor_line] = new_line
        return CompletionResult(new_lines, cursor_line, new_cursor)

    def _extract_at_prefix(self, text: str) -> str | None:
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix and quoted_prefix.startswith('@"'):
            return quoted_prefix
        token_start = _find_last_delimiter(text) + 1
        return text[token_start:] if text[token_start:token_start + 1] == "@" else None

    def _extract_path_prefix(self, text: str, force_extract: bool = False) -> str | None:
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix:
            return quoted_prefix
        path_prefix = text[_find_last_delimiter(text) + 1:]
        if force_extract:
            return path_prefix
        if "/" in path_prefix or path_prefix.startswith(".") or path_prefix.startswith("~/"):
            return path_prefix
        if not path_prefix and text.endswith(" "):
            return path_prefix
        return None

    def _expand_home_path(self, path: str) -> str:
        if path.startswith("~/"):
            expanded_path = _join(os.path.expanduser("~"), path[2:])
            return expanded_path + "/" if path.endswith("/") and not expanded_path.endswith("/") else expanded_path
        return os.path.expanduser("~") if path == "~" else path

    def _resolve_scoped_fuzzy_query(self, raw_query: str) -> _ScopedQuery | None:
        normalized_query = _to_display_path(raw_query)
        slash_index = normalized_query.rfind("/")
        if slash_index == -1:
            return None
        display_base = normalized_query[:slash_index + 1]
        query = normalized_query[slash_index + 1:]
        if display_base.startswith("~/"):
            base_dir = self._expand_home_path(display_base)
        elif display_base.startswith("/"):
            base_dir = display_base
        else:
            base_dir = _join(self._base_path, display_base)
        if not os.path.isdir(base_dir):
            return None
        return _ScopedQuery(base_dir, query, display_base)

    def _get_file_suggestions(self, prefix: str) -> list[AutocompleteItem]:
        try:
            raw_prefix, is_at_prefix, is_quoted_prefix = _parse_path_prefix(prefix)
            expanded_prefix = self._expand_home_path(raw_prefix) if raw_prefix.startswith("~") else raw_prefix
            is_root_prefix = raw_prefix in ("", "./", "../", "~", "~/", "/")
            absolute = raw_prefix.startswith("~") or expanded_prefix.startswith("/")
            if is_root_prefix or raw_prefix.endswith("/"):
                search_dir = expanded_prefix if absolute else _join(self._base_path, expanded_prefix)
                search_prefix = ""
            else:
                directory = _dirname(expanded_prefix)
                search_dir = directory if absolute else _join(self._base_path, directory)
                search_prefix = _basename(expanded_prefix)
            suggestions: list[AutocompleteItem] = []
            with os.scandir(search_dir) as entries:
                for entry in entries:
                    if not entry.name.lower().startswith(search_prefix.lower()):
                        continue
                    is_directory = entry.is_dir(follow_symlinks=False)
                    if not is_directory and entry.is_symlink():
                        try:
                            is_directory = entry.is_dir()
                        except OSError:
                            pass
                    name = entry.name
                    if raw_prefix.endswith("/"):
                        relative_path = raw_prefix + name
                    elif "/" in raw_prefix or "\\" in raw_prefix:
                        if raw_prefix.startswith("~/"):
                            directory = _dirname(raw_prefix[2:])
                            relative_path = "~/" + (name if directory == "." else _join(directory, name))
                        elif raw_prefix.startswith("/"):
                            directory = _dirname(raw_prefix)
                            relative_path = f"/{name}" if directory == "/" else f"{directory}/{name}"
                        else:
                            relative_path = _join(_dirname(raw_prefix), name)
                            if raw_prefix.startswith("./") and not relative_path.startswith("./"):
                                relative_path = "./" + relative_path
                    else:
                        relative_path = "~/" + name if raw_prefix.startswith("~") else name
                    relative_path = _to_display_path(relative_path)
                    path_value = relative_path + ("/" if is_directory else "")
                    value = _build_completion_value(path_value, is_at_prefix=is_at_prefix, is_quoted_prefix=is_quoted_prefix)
                    suggestions.append(AutocompleteItem(value, name + ("/" if is_directory else "")))

            def compare_items(left: AutocompleteItem, right: AutocompleteItem) -> int:
                # Keep the source's value-based check (quoted directories end in a quote).
                directory_order = int(right.value.endswith("/")) - int(left.value.endswith("/"))
                return directory_order or _COLLATOR.compare(left.label, right.label)

            suggestions.sort(key=cmp_to_key(compare_items))
            return suggestions
        except Exception:
            return []

    def _score_entry(self, file_path: str, query: str, is_directory: bool) -> int:
        lower_file_name = _basename(file_path).lower()
        lower_query = query.lower()
        if lower_file_name == lower_query:
            score = 100
        elif lower_file_name.startswith(lower_query):
            score = 80
        elif lower_query in lower_file_name:
            score = 50
        elif lower_query in file_path.lower():
            score = 30
        else:
            score = 0
        if is_directory and score > 0:
            score += 10
        return score

    async def _get_fuzzy_file_suggestions(
        self, query: str, is_quoted_prefix: bool, signal: AutocompleteAbortSignal,
    ) -> list[AutocompleteItem]:
        if not self._fd_path or signal.aborted:
            return []
        try:
            scoped_query = self._resolve_scoped_fuzzy_query(query)
            fd_base_dir = scoped_query.base_dir if scoped_query else self._base_path
            fd_query = scoped_query.query if scoped_query else query
            base_entries = await _walk_directory_with_fd(fd_base_dir, self._fd_path, fd_query, 100, signal, 1)
            recursive_entries = await _walk_directory_with_fd(fd_base_dir, self._fd_path, fd_query, 100, signal)
            seen_paths = {entry.path for entry in base_entries}
            entries = base_entries.copy()
            for entry in recursive_entries:
                if entry.path not in seen_paths:
                    seen_paths.add(entry.path)
                    entries.append(entry)
            if signal.aborted:
                return []
            scored_entries: list[tuple[_FileEntry, int]] = []
            for entry in entries:
                score = self._score_entry(entry.path, fd_query, entry.is_directory) if fd_query else 1
                if score > 0:
                    scored_entries.append((entry, score))

            def compare_entries(left: tuple[_FileEntry, int], right: tuple[_FileEntry, int]) -> int:
                score_diff = right[1] - left[1]
                if score_diff:
                    return score_diff
                left_path = left[0].path
                right_path = right[0].path
                depth_diff = len([part for part in _to_display_path(left_path).split("/") if part]) - len([
                    part for part in _to_display_path(right_path).split("/") if part
                ])
                if depth_diff:
                    return depth_diff
                length_diff = utf16_length(left_path) - utf16_length(right_path)
                return length_diff or _COLLATOR.compare(left_path, right_path)

            scored_entries.sort(key=cmp_to_key(compare_entries))
            suggestions: list[AutocompleteItem] = []
            for entry, _ in scored_entries[:20]:
                path_without_slash = entry.path[:-1] if entry.is_directory else entry.path
                display_path = (
                    _to_display_path(scoped_query.display_base) + _to_display_path(path_without_slash)
                    if scoped_query else path_without_slash
                )
                entry_name = _basename(path_without_slash)
                completion_path = display_path + ("/" if entry.is_directory else "")
                value = _build_completion_value(completion_path, is_at_prefix=True, is_quoted_prefix=is_quoted_prefix)
                suggestions.append(AutocompleteItem(value, entry_name + ("/" if entry.is_directory else ""), display_path))
            return suggestions
        except Exception:
            return []

    def should_trigger_file_completion(self, lines: list[str], cursor_line: int, cursor_col: int) -> bool:
        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        before = js_trim(utf16_slice(current_line, 0, cursor_col))
        return not (before.startswith("/") and " " not in before)
