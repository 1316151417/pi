"""Edit/diff engine ported from ``harness/tools/edit-diff.ts``."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

__all__ = [
    "Edit",
    "AppliedEditsResult",
    "FuzzyMatchResult",
    "detect_line_ending",
    "normalize_to_lf",
    "restore_line_endings",
    "normalize_for_fuzzy_match",
    "fuzzy_find_text",
    "apply_edits_to_normalized_content",
    "apply_replacements_preserving_unchanged_lines",
    "strip_bom",
    "generate_unified_patch",
    "generate_diff_string",
]


def detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


_SMART_SINGLE_QUOTES = re.compile("[\u2018\u2019\u201A\u201B]")
_SMART_DOUBLE_QUOTES = re.compile("[\u201C\u201D\u201E\u201F]")
_UNICODE_DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_UNICODE_SPACES = re.compile("[\u00A0\u2002-\u200A\u202F\u205F\u3000]")


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching (progressive transformations)."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = _SMART_SINGLE_QUOTES.sub("'", normalized)
    normalized = _SMART_DOUBLE_QUOTES.sub('"', normalized)
    normalized = _UNICODE_DASHES.sub("-", normalized)
    normalized = _UNICODE_SPACES.sub(" ", normalized)
    return normalized


_LINE_SPLIT = re.compile("[^\n]*\n|[^\n]+")


def _split_lines_with_endings(content: str) -> List[str]:
    return _LINE_SPLIT.findall(content) or []


@dataclass
class _LineSpan:
    start: int
    end: int


@dataclass
class Edit:
    old_text: str
    new_text: str


@dataclass
class _MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


@dataclass
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


@dataclass
class AppliedEditsResult:
    base_content: str
    new_content: str


def _get_line_spans(content: str) -> List[_LineSpan]:
    spans: List[_LineSpan] = []
    offset = 0
    for line in _split_lines_with_endings(content):
        spans.append(_LineSpan(start=offset, end=offset + len(line)))
        offset += len(line)
    return spans


def _get_replacement_line_range(lines: Sequence[_LineSpan], replacement: _MatchedEdit):
    replacement_start = replacement.match_index
    replacement_end = replacement.match_index + replacement.match_length

    start_line = -1
    for i, line in enumerate(lines):
        if line.start <= replacement_start < line.end:
            start_line = i
            break
    if start_line == -1:
        raise RuntimeError("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise RuntimeError("Replacement range is outside the base content.")

    return start_line, end_line + 1


def _apply_replacements(content: str, replacements: Sequence[_MatchedEdit], offset: int = 0) -> str:
    result = content
    for replacement in sorted(replacements, key=lambda r: r.match_index, reverse=True):
        match_index = replacement.match_index - offset
        result = (
            result[:match_index]
            + replacement.new_text
            + result[match_index + replacement.match_length :]
        )
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str, base_content: str, replacements: Sequence[_MatchedEdit]
) -> str:
    """Apply replacements matched against ``base_content`` to ``original_content``.

    Each replacement is widened to the lines it touches; touched lines are
    rewritten from the normalized base while unchanged lines keep their
    original bytes.
    """
    original_lines = _split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise RuntimeError(
            "Cannot preserve unchanged lines because the base content has a different line count."
        )

    groups: List[dict] = []
    for replacement in sorted(replacements, key=lambda r: r.match_index):
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        if groups and start_line < groups[-1]["end_line"]:
            groups[-1]["end_line"] = max(groups[-1]["end_line"], end_line)
            groups[-1]["replacements"].append(replacement)
            continue
        groups.append({"start_line": start_line, "end_line": end_line, "replacements": [replacement]})

    original_line_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_line_index : group["start_line"]])
        group_start_offset = base_lines[group["start_line"]].start
        group_end_offset = base_lines[group["end_line"] - 1].end
        result += _apply_replacements(
            base_content[group_start_offset:group_end_offset],
            group["replacements"],
            group_start_offset,
        )
        original_line_index = group["end_line"]
    result += "".join(original_lines[original_line_index:])
    return result


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find ``old_text`` in content: exact match first, then fuzzy match."""
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )
    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    if not fuzzy_old_text:
        return 0
    return fuzzy_content.count(fuzzy_old_text)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(
            f"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
    return RuntimeError(
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
    return RuntimeError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(f"oldText must not be empty in {path}.")
    return RuntimeError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _no_change_error(path: str, total_edits: int) -> RuntimeError:
    if total_edits == 1:
        return RuntimeError(
            f"No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        )
    return RuntimeError(f"No changes made to {path}. The replacements produced identical content.")


def apply_edits_to_normalized_content(
    normalized_content: str, edits: Sequence[Edit], path: str
) -> AppliedEditsResult:
    """Apply one or more exact-text replacements to LF-normalized content.

    All edits are matched against the same original content; replacements apply
    in reverse order so offsets stay stable. Fuzzy matches operate in
    normalized space, overlaying line-level changes onto the original content.
    """
    normalized_edits = [
        Edit(old_text=normalize_to_lf(edit.old_text), new_text=normalize_to_lf(edit.new_text))
        for edit in edits
    ]

    for i, edit in enumerate(normalized_edits):
        if len(edit.old_text) == 0:
            raise _empty_old_text_error(path, i, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, edit.old_text) for edit in normalized_edits]
    used_fuzzy_match = any(match.used_fuzzy_match for match in initial_matches)
    replacement_base_content = (
        normalize_for_fuzzy_match(normalized_content) if used_fuzzy_match else normalized_content
    )

    matched_edits: List[_MatchedEdit] = []
    for i, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit.old_text)
        if not match_result.found:
            raise _not_found_error(path, i, len(normalized_edits))

        occurrences = _count_occurrences(replacement_base_content, edit.old_text)
        if occurrences > 1:
            raise _duplicate_error(path, i, len(normalized_edits), occurrences)

        matched_edits.append(
            _MatchedEdit(
                edit_index=i,
                match_index=match_result.index,
                match_length=match_result.match_length,
                new_text=edit.new_text,
            )
        )

    matched_edits.sort(key=lambda m: m.match_index)
    for i in range(1, len(matched_edits)):
        previous = matched_edits[i - 1]
        current = matched_edits[i]
        if previous.match_index + previous.match_length > current.match_index:
            raise RuntimeError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    new_content = (
        apply_replacements_preserving_unchanged_lines(normalized_content, replacement_base_content, matched_edits)
        if used_fuzzy_match
        else _apply_replacements(replacement_base_content, matched_edits)
    )

    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))

    return AppliedEditsResult(base_content=base_content, new_content=new_content)


def strip_bom(content: str) -> tuple:
    """Strip UTF-8 BOM if present; returns ``(bom, text_without_bom)``."""
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


# ---------------------------------------------------------------------------
# Diff generation (difflib equivalents of the JS "diff" package)
# ---------------------------------------------------------------------------


@dataclass
class DiffPart:
    value: str
    added: bool = False
    removed: bool = False


def diff_lines(old_content: str, new_content: str) -> List[DiffPart]:
    """Line-level diff producing JS ``Diff.diffLines``-equivalent parts."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    parts: List[DiffPart] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(DiffPart(value="".join(old_lines[i1:i2])))
        elif tag == "delete":
            parts.append(DiffPart(value="".join(old_lines[i1:i2]), removed=True))
        elif tag == "insert":
            parts.append(DiffPart(value="".join(new_lines[j1:j2]), added=True))
        else:  # replace
            parts.append(DiffPart(value="".join(old_lines[i1:i2]), removed=True))
            parts.append(DiffPart(value="".join(new_lines[j1:j2]), added=True))
    return parts


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """Generate a standard unified patch."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=path,
        tofile=path,
        n=context_lines,
    )
    return "".join(diff)


def generate_diff_string(
    old_content: str, new_content: str, context_lines: int = 4
) -> tuple:
    """Display-oriented diff with line numbers and context.

    Returns ``(diff_string, first_changed_line)``; the line number is 1-indexed
    in the new file, or ``None`` when nothing changed.
    """
    parts = diff_lines(old_content, new_content)
    output: List[str] = []

    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: Optional[int] = None

    for i, part in enumerate(parts):
        raw = part.value.split("\n")
        if raw and raw[-1] == "":
            raw.pop()

        if part.added or part.removed:
            if first_changed_line is None:
                first_changed_line = new_line_num
            for line in raw:
                if part.added:
                    line_num = str(new_line_num).rjust(line_num_width)
                    output.append(f"+{line_num} {line}")
                    new_line_num += 1
                else:
                    line_num = str(old_line_num).rjust(line_num_width)
                    output.append(f"-{line_num} {line}")
                    old_line_num += 1
            last_was_change = True
        else:
            next_part_is_change = i < len(parts) - 1 and (parts[i + 1].added or parts[i + 1].removed)
            has_leading_change = last_was_change
            has_trailing_change = next_part_is_change

            if has_leading_change and has_trailing_change:
                if len(raw) <= context_lines * 2:
                    for line in raw:
                        line_num = str(old_line_num).rjust(line_num_width)
                        output.append(f" {line_num} {line}")
                        old_line_num += 1
                        new_line_num += 1
                else:
                    leading = raw[:context_lines]
                    trailing = raw[len(raw) - context_lines :]
                    skipped = len(raw) - len(leading) - len(trailing)
                    for line in leading:
                        line_num = str(old_line_num).rjust(line_num_width)
                        output.append(f" {line_num} {line}")
                        old_line_num += 1
                        new_line_num += 1
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped
                    new_line_num += skipped
                    for line in trailing:
                        line_num = str(old_line_num).rjust(line_num_width)
                        output.append(f" {line_num} {line}")
                        old_line_num += 1
                        new_line_num += 1
            elif has_leading_change:
                shown = raw[:context_lines]
                skipped = len(raw) - len(shown)
                for line in shown:
                    line_num = str(old_line_num).rjust(line_num_width)
                    output.append(f" {line_num} {line}")
                    old_line_num += 1
                    new_line_num += 1
                if skipped > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped
                    new_line_num += skipped
            elif has_trailing_change:
                skipped = max(0, len(raw) - context_lines)
                if skipped > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped
                    new_line_num += skipped
                for line in raw[skipped:]:
                    line_num = str(old_line_num).rjust(line_num_width)
                    output.append(f" {line_num} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                old_line_num += len(raw)
                new_line_num += len(raw)

            last_was_change = False

    return "\n".join(output), first_changed_line
