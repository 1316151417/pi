"""Prompt templates ported from ``harness/prompt-templates.ts``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union

from .._chord.context import Context
from .types import ExecutionEnv, FileInfo

__all__ = [
    "PromptTemplateDiagnosticCode",
    "PromptTemplateDiagnostic",
    "PromptTemplate",
    "load_prompt_templates",
    "load_sourced_prompt_templates",
    "parse_command_args",
    "substitute_args",
    "format_prompt_template_invocation",
]

PromptTemplateDiagnosticCode = str  # "file_info_failed" | "list_failed" | "read_failed" | "parse_failed"


@dataclass
class PromptTemplate:
    """Reusable prompt fragment loaded from a markdown file."""

    name: str
    content: str
    description: str = ""


@dataclass
class PromptTemplateDiagnostic:
    type: str  # "warning"
    code: PromptTemplateDiagnosticCode
    message: str
    path: str


async def load_prompt_templates(
    env: ExecutionEnv, paths: Union[str, Sequence[str]], context: Context
) -> tuple:
    """Load prompt templates from one or more paths.

    Directory inputs load direct ``.md`` children non-recursively. File inputs
    load explicit ``.md`` files. Missing paths and non-markdown files are
    skipped; read and parse failures become diagnostics.
    """
    prompt_templates: List[PromptTemplate] = []
    diagnostics: List[PromptTemplateDiagnostic] = []
    for path in paths if isinstance(paths, list) else [paths]:
        info_result = await env.file_info(path, context)
        if not info_result.ok:
            if info_result.error.code != "not_found":
                diagnostics.append(
                    PromptTemplateDiagnostic(
                        type="warning",
                        code="file_info_failed",
                        message=info_result.error.message,
                        path=path,
                    )
                )
            continue
        info = info_result.value
        kind = await _resolve_kind(env, info, diagnostics, context)
        if kind == "directory":
            result = await _load_templates_from_dir(env, info.path, context)
            prompt_templates.extend(result[0])
            diagnostics.extend(result[1])
        elif kind == "file" and info.name.endswith(".md"):
            result = await _load_template_from_file(env, info.path, info.name, context)
            if result[0] is not None:
                prompt_templates.append(result[0])
            diagnostics.extend(result[1])
    return prompt_templates, diagnostics


async def load_sourced_prompt_templates(
    env: ExecutionEnv,
    inputs: List[dict],
    map_prompt_template,
    context: Context,
) -> tuple:
    """Load prompt templates from source-tagged paths."""
    prompt_templates: List[dict] = []
    diagnostics: List[dict] = []
    for item in inputs:
        templates, template_diagnostics = await load_prompt_templates(env, item["path"], context)
        for template in templates:
            mapped = map_prompt_template(template, item["source"], context) if map_prompt_template else template
            prompt_templates.append({"promptTemplate": mapped, "source": item["source"]})
        for diagnostic in template_diagnostics:
            diagnostics.append({**vars(diagnostic), "source": item["source"]})
    return prompt_templates, diagnostics


async def _load_templates_from_dir(env: ExecutionEnv, directory: str, context: Context) -> tuple:
    prompt_templates: List[PromptTemplate] = []
    diagnostics: List[PromptTemplateDiagnostic] = []
    entries_result = await env.list_dir(directory, context)
    if not entries_result.ok:
        diagnostics.append(
            PromptTemplateDiagnostic(
                type="warning", code="list_failed", message=entries_result.error.message, path=directory
            )
        )
        return prompt_templates, diagnostics

    for entry in sorted(entries_result.value, key=lambda e: e.name):
        kind = await _resolve_kind(env, entry, diagnostics, context)
        if kind != "file" or not entry.name.endswith(".md"):
            continue
        template, file_diagnostics = await _load_template_from_file(env, entry.path, entry.name, context)
        if template is not None:
            prompt_templates.append(template)
        diagnostics.extend(file_diagnostics)
    return prompt_templates, diagnostics


async def _load_template_from_file(
    env: ExecutionEnv, file_path: str, file_name: str, context: Context
) -> tuple:
    diagnostics: List[PromptTemplateDiagnostic] = []
    raw_content = await env.read_text_file(file_path, context)
    if not raw_content.ok:
        diagnostics.append(
            PromptTemplateDiagnostic(
                type="warning", code="read_failed", message=raw_content.error.message, path=file_path
            )
        )
        return None, diagnostics

    frontmatter, body = _parse_frontmatter(raw_content.value)
    first_line = next((line for line in body.split("\n") if line.strip()), None)
    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else ""
    if not description and first_line:
        description = first_line[:60]
        if len(first_line) > 60:
            description += "..."
    return (
        PromptTemplate(
            name=re.sub(r"\.md$", "", file_name, flags=re.IGNORECASE),
            description=description,
            content=body,
        ),
        diagnostics,
    )


async def _resolve_kind(
    env: ExecutionEnv, info: FileInfo, diagnostics: List[PromptTemplateDiagnostic], context: Context
) -> Optional[str]:
    if info.kind in ("file", "directory"):
        return info.kind
    canonical_path = await env.canonical_path(info.path, context)
    if not canonical_path.ok:
        if canonical_path.error.code != "not_found":
            diagnostics.append(
                PromptTemplateDiagnostic(
                    type="warning",
                    code="file_info_failed",
                    message=canonical_path.error.message,
                    path=info.path,
                )
            )
        return None
    target = await env.file_info(canonical_path.value, context)
    if not target.ok:
        if target.error.code != "not_found":
            diagnostics.append(
                PromptTemplateDiagnostic(
                    type="warning", code="file_info_failed", message=target.error.message, path=info.path
                )
            )
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None


def _parse_simple_yaml(text: str) -> dict:
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
        result[key] = value
    return result


def _parse_frontmatter(content: str) -> tuple:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    return _parse_simple_yaml(yaml_string), body


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def parse_command_args(args_string: str) -> List[str]:
    """Parse an argument string using simple shell-style single and double quotes."""
    args: List[str] = []
    current = ""
    in_quote: Optional[str] = None

    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: List[str]) -> str:
    """Substitute placeholders (``$1``, ``$@``, ``$ARGUMENTS``, ``${@:N}``, ``${@:N:L}``)."""
    result = re.sub(r"\$(\d+)", lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) - 1 < len(args) else "", content)

    def _slice(match: re.Match) -> str:
        start = int(match.group(1)) - 1
        if start < 0:
            start = 0
        if match.group(2):
            return " ".join(args[start : start + int(match.group(2))])
        return " ".join(args[start:])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", _slice, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


def format_prompt_template_invocation(template: PromptTemplate, args: Optional[List[str]] = None) -> str:
    """Format a prompt template invocation with positional arguments."""
    return substitute_args(template.content, args or [])
