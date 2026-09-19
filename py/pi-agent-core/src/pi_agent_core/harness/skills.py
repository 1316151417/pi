"""Skills loading ported from ``harness/skills.ts``.

Includes a compact gitignore matcher covering the common pattern forms
(``*``, ``**``, negation, trailing-slash directories) instead of the npm
``ignore`` package.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .._chord.context import Context
from .types import ExecutionEnv, FileInfo, Skill

__all__ = [
    "SkillDiagnostic",
    "load_skills",
    "load_sourced_skills",
    "format_skill_invocation",
    "IgnoreMatcher",
]

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]


@dataclass
class SkillDiagnostic:
    type: str  # "warning"
    code: str  # "file_info_failed" | "list_failed" | "read_failed" | "parse_failed" | "invalid_metadata"
    message: str
    path: str


class IgnoreMatcher:
    """Minimal gitignore matcher supporting ``*``, ``**``, ``!``, and dir patterns."""

    def __init__(self) -> None:
        self._rules: List[Tuple[bool, bool, Any]] = []  # (negated, dir_only, regex)

    def add(self, patterns: Sequence[str]) -> None:
        for pattern in patterns:
            rule = self._compile(pattern)
            if rule is not None:
                self._rules.append(rule)

    @staticmethod
    def _compile(pattern: str) -> Optional[Tuple[bool, bool, Any]]:
        negated = pattern.startswith("!")
        body = pattern[1:] if negated else pattern
        dir_only = body.endswith("/")
        body = body.rstrip("/")
        if not body:
            return None
        anchored = body.startswith("/") or "/" in body
        body = body.lstrip("/")

        # Build regex: ** -> .*, * -> [^/]*, ? -> [^/]
        parts = ["^"] if anchored else ["(^|/)"]
        i = 0
        while i < len(body):
            ch = body[i]
            if ch == "*":
                if body[i : i + 2] == "**":
                    parts.append(".*")
                    i += 2
                    continue
                parts.append("[^/]*")
            elif ch == "?":
                parts.append("[^/]")
            else:
                parts.append(re.escape(ch))
            i += 1
        parts.append("/?" if dir_only else "$")
        # Non-anchored patterns also match paths nested under the pattern.
        if not anchored and not dir_only:
            regex = re.compile("".join(parts[:-1]) + "($|/.*$)")
        else:
            regex = re.compile("".join(parts))
        return negated, dir_only, regex

    def ignores(self, path: str, is_dir: bool = False) -> bool:
        ignored = False
        for negated, dir_only, regex in self._rules:
            if dir_only and not is_dir:
                continue
            if regex.search(path):
                ignored = not negated
        return ignored


def format_skill_invocation(skill: Skill, additional_instructions: Optional[str] = None) -> str:
    """Format a skill invocation prompt, optionally appending extra instructions."""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {_dirname_env_path(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    return f"{skill_block}\n\n{additional_instructions}" if additional_instructions else skill_block


async def load_skills(
    env: ExecutionEnv, dirs: Any, context: Context
) -> tuple:
    """Load skills from one or more directories.

    Traverses recursively, loads ``SKILL.md`` files and root ``.md`` files with
    skill frontmatter, honors ignore files, and returns diagnostics for invalid
    declared skill files. Missing input directories are skipped.
    """
    skills: List[Skill] = []
    diagnostics: List[SkillDiagnostic] = []
    for directory in dirs if isinstance(dirs, list) else [dirs]:
        root_info_result = await env.file_info(directory, context)
        if not root_info_result.ok:
            if root_info_result.error.code != "not_found":
                diagnostics.append(
                    SkillDiagnostic(
                        type="warning",
                        code="file_info_failed",
                        message=root_info_result.error.message,
                        path=directory,
                    )
                )
            continue
        root_info = root_info_result.value
        kind = await _resolve_kind(env, root_info, diagnostics, context)
        if kind != "directory":
            continue
        result = await _load_skills_from_dir(env, root_info.path, True, IgnoreMatcher(), root_info.path, context)
        skills.extend(result[0])
        diagnostics.extend(result[1])
    return skills, diagnostics


async def load_sourced_skills(env: ExecutionEnv, inputs: List[dict], map_skill, context: Context) -> tuple:
    """Load skills from source-tagged directories."""
    skills: List[dict] = []
    diagnostics: List[dict] = []
    for item in inputs:
        loaded_skills, loaded_diagnostics = await load_skills(env, item["path"], context)
        for skill in loaded_skills:
            mapped = map_skill(skill, item["source"], context) if map_skill else skill
            skills.append({"skill": mapped, "source": item["source"]})
        for diagnostic in loaded_diagnostics:
            diagnostics.append({**vars(diagnostic), "source": item["source"]})
    return skills, diagnostics


async def _load_skills_from_dir(
    env: ExecutionEnv,
    directory: str,
    include_root_files: bool,
    ignore_matcher: IgnoreMatcher,
    root_dir: str,
    context: Context,
) -> tuple:
    skills: List[Skill] = []
    diagnostics: List[SkillDiagnostic] = []

    dir_info_result = await env.file_info(directory, context)
    if not dir_info_result.ok:
        if dir_info_result.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(
                    type="warning", code="file_info_failed", message=dir_info_result.error.message, path=directory
                )
            )
        return skills, diagnostics
    dir_info = dir_info_result.value
    if await _resolve_kind(env, dir_info, diagnostics, context) != "directory":
        return skills, diagnostics

    await _add_ignore_rules(env, ignore_matcher, directory, root_dir, diagnostics, context)

    entries_result = await env.list_dir(directory, context)
    if not entries_result.ok:
        diagnostics.append(
            SkillDiagnostic(type="warning", code="list_failed", message=entries_result.error.message, path=directory)
        )
        return skills, diagnostics
    entries = entries_result.value

    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        kind = await _resolve_kind(env, entry, diagnostics, context)
        if kind != "file":
            continue
        rel_path = _relative_env_path(root_dir, entry.path)
        if ignore_matcher.ignores(rel_path):
            continue
        skill, file_diagnostics = await _load_skill_from_file(env, entry.path, dir_info.name, context)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)
        return skills, diagnostics

    for entry in sorted(entries, key=lambda e: e.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        kind = await _resolve_kind(env, entry, diagnostics, context)
        if not kind:
            continue
        rel_path = _relative_env_path(root_dir, entry.path)
        ignore_path = f"{rel_path}/" if kind == "directory" else rel_path
        if ignore_matcher.ignores(ignore_path, is_dir=kind == "directory"):
            continue

        if kind == "directory":
            nested_skills, nested_diagnostics = await _load_skills_from_dir(
                env, entry.path, False, ignore_matcher, root_dir, context
            )
            skills.extend(nested_skills)
            diagnostics.extend(nested_diagnostics)
            continue

        if kind != "file" or not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, file_diagnostics = await _load_skill_from_file(env, entry.path, dir_info.name, context)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)

    return skills, diagnostics


async def _add_ignore_rules(
    env: ExecutionEnv,
    matcher: IgnoreMatcher,
    directory: str,
    root_dir: str,
    diagnostics: List[SkillDiagnostic],
    context: Context,
) -> None:
    relative_dir = _relative_env_path(root_dir, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path_result = await env.join_path([directory, filename], context)
        if not ignore_path_result.ok:
            diagnostics.append(
                SkillDiagnostic(
                    type="warning",
                    code="file_info_failed",
                    message=ignore_path_result.error.message,
                    path=directory,
                )
            )
            continue
        ignore_path = ignore_path_result.value
        info = await env.file_info(ignore_path, context)
        if not info.ok:
            if info.error.code != "not_found":
                diagnostics.append(
                    SkillDiagnostic(
                        type="warning", code="file_info_failed", message=info.error.message, path=ignore_path
                    )
                )
            continue
        if info.value.kind != "file":
            continue
        content = await env.read_text_file(ignore_path, context)
        if not content.ok:
            diagnostics.append(
                SkillDiagnostic(type="warning", code="read_failed", message=content.error.message, path=ignore_path)
            )
            continue
        patterns = [
            prefixed
            for line in re.split(r"\r?\n", content.value)
            if (prefixed := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        if patterns:
            matcher.add(patterns)


def _prefix_ignore_pattern(line: str, prefix: str) -> Optional[str]:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None

    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


async def _load_skill_from_file(env: ExecutionEnv, file_path: str, parent_dir_name: str, context: Context) -> tuple:
    diagnostics: List[SkillDiagnostic] = []
    is_declared_skill = re.split(r"[\\/]+", file_path.rstrip("/\\"))[-1] == "SKILL.md"
    raw_content = await env.read_text_file(file_path, context)
    if not raw_content.ok:
        diagnostics.append(
            SkillDiagnostic(type="warning", code="read_failed", message=raw_content.error.message, path=file_path)
        )
        return None, diagnostics

    parsed = _parse_frontmatter(raw_content.value)
    if not parsed.ok:
        if is_declared_skill:
            diagnostics.append(
                SkillDiagnostic(type="warning", code="parse_failed", message=str(parsed.error), path=file_path)
            )
        return None, diagnostics

    frontmatter, body = parsed.value
    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else None
    if not is_declared_skill and (not description or description.strip() == ""):
        return None, diagnostics

    for error in _validate_description(description):
        diagnostics.append(
            SkillDiagnostic(type="warning", code="invalid_metadata", message=error, path=file_path)
        )

    frontmatter_name = frontmatter.get("name") if isinstance(frontmatter.get("name"), str) else None
    name = frontmatter_name or parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append(
            SkillDiagnostic(type="warning", code="invalid_metadata", message=error, path=file_path)
        )

    if not description or description.strip() == "":
        return None, diagnostics

    return (
        Skill(
            name=name,
            description=description,
            content=body,
            file_path=file_path,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


def _validate_name(name: str, parent_dir_name: str) -> List[str]:
    errors: List[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: Optional[str]) -> List[str]:
    errors: List[str] = []
    if not description or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


def _parse_simple_yaml(text: str) -> dict:
    """Parse the flat ``key: value`` YAML subset used by skill frontmatter."""
    result: dict = {}
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        result[key] = value
    return result


@dataclass
class _ParsedFrontmatter:
    ok: bool
    value: Any = None
    error: Any = None


def _parse_frontmatter(content: str) -> _ParsedFrontmatter:
    """Parse ``---``-delimited frontmatter into ``_ParsedFrontmatter``."""
    try:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.startswith("---"):
            return _ParsedFrontmatter(ok=True, value=({}, normalized))
        end_index = normalized.find("\n---", 3)
        if end_index == -1:
            return _ParsedFrontmatter(ok=True, value=({}, normalized))
        yaml_string = normalized[4:end_index]
        body = normalized[end_index + 4 :].strip()
        return _ParsedFrontmatter(ok=True, value=(_parse_simple_yaml(yaml_string), body))
    except Exception as error:  # noqa: BLE001
        return _ParsedFrontmatter(ok=False, error=error)


async def _resolve_kind(
    env: ExecutionEnv, info: FileInfo, diagnostics: List[SkillDiagnostic], context: Context
) -> Optional[str]:
    if info.kind in ("file", "directory"):
        return info.kind
    canonical_path = await env.canonical_path(info.path, context)
    if not canonical_path.ok:
        if canonical_path.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(
                    type="warning", code="file_info_failed", message=canonical_path.error.message, path=info.path
                )
            )
        return None
    target = await env.file_info(canonical_path.value, context)
    if not target.ok:
        if target.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(type="warning", code="file_info_failed", message=target.error.message, path=info.path)
            )
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None


def _dirname_env_path(path: str) -> str:
    normalized = path.rstrip("/\\")
    separator_index = max(normalized.rfind("/"), normalized.rfind("\\"))
    if separator_index == 2 and normalized[1] == ":":
        return normalized[:3]
    return "/" if separator_index <= 0 else normalized[:separator_index]


def _relative_env_path(root: str, path: str) -> str:
    normalized_root = root.replace("\\", "/").rstrip("/")
    normalized_path = path.replace("\\", "/").rstrip("/")
    if normalized_path == normalized_root:
        return ""
    if normalized_path.startswith(f"{normalized_root}/"):
        return normalized_path[len(normalized_root) + 1 :]
    return normalized_path.lstrip("/")
