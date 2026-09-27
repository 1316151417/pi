"""Structured system prompt construction from ``core/system-prompt.ts``."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from pi_ai.text import get_system_message_text
from pi_ai.types import SystemMessage

from .config import get_docs_path, get_examples_path, get_readme_path
from .skills import Skill, format_skills_for_prompt

type SystemPromptSections = dict[str, str]

_SECTION_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_DEFAULT_TOOLS = ("read", "bash", "edit", "write")
_PREAMBLE = (
    "You are an expert coding assistant operating inside pi, a coding agent harness. "
    "You help users by reading files, executing commands, editing code, and writing new files."
)
_DOCS_TEMPLATE = """Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: {readme}
- Additional docs: {docs}
- Examples: {examples} (extensions, custom tools, SDK)
- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md), environment variables (docs/environment-variables.md)
- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)"""


@dataclass
class BuildSystemPromptOptions:
    cwd: str
    custom_prompt: str | None = None
    force_system_prompt: str | None = None
    selected_tools: list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    tool_guidelines: dict[str, list[str]] | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    sections: dict[str, str] | None = None
    context_files: list[dict[str, str]] | None = None
    skills: list[Skill] | None = None


@dataclass
class NormalizedBuildSystemPromptOptions:
    cwd: str
    custom_prompt: str | None
    force_system_prompt: str | None
    selected_tools: list[str]
    tool_snippets: dict[str, str]
    tool_guidelines: dict[str, list[str]]
    prompt_guidelines: list[str]
    append_system_prompt: str
    sections: dict[str, str]
    context_files: list[dict[str, str]]
    skills: list[Skill]


@dataclass
class SystemPromptState:
    content: str
    sections: SystemPromptSections | None = None


def normalize_build_system_prompt_options(
    options: BuildSystemPromptOptions,
) -> NormalizedBuildSystemPromptOptions:
    return NormalizedBuildSystemPromptOptions(
        cwd=options.cwd,
        custom_prompt=options.custom_prompt,
        force_system_prompt=options.force_system_prompt,
        selected_tools=list(options.selected_tools if options.selected_tools is not None else _DEFAULT_TOOLS),
        tool_snippets=dict(options.tool_snippets or {}),
        tool_guidelines={name: list(rules) for name, rules in (options.tool_guidelines or {}).items()},
        prompt_guidelines=list(options.prompt_guidelines or []),
        append_system_prompt=options.append_system_prompt or "",
        sections=dict(options.sections or {}),
        context_files=[dict(item) for item in options.context_files or []],
        skills=[replace(skill) for skill in options.skills or []],
    )


def _render_project_context(files: list[dict[str, str]]) -> str:
    parts = ["Project-specific instructions and guidelines:"]
    parts.extend(
        f'<project_instructions path="{item["path"]}">\n{item["content"]}\n</project_instructions>'
        for item in files
    )
    return "\n\n".join(parts)


def _build_rules(
    selected_tools: list[str], tool_guidelines: dict[str, list[str]], prompt_guidelines: list[str],
) -> str:
    rules: list[str] = []
    seen: set[str] = set()

    def add(rule: str) -> None:
        normalized = rule.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            rules.append(normalized)

    has_bash = "bash" in selected_tools
    has_powershell = "powershell" in selected_tools
    if ((has_bash or has_powershell)
            and not any(name in selected_tools for name in ("grep", "find", "ls"))):
        if has_bash and has_powershell:
            add("Use bash or PowerShell for file operations like listing, searching, and finding files")
        elif has_powershell:
            add("Use PowerShell for file operations like listing, searching, and finding files")
        else:
            add("Use bash for file operations like ls, rg, find")
    for name in selected_tools:
        for rule in tool_guidelines.get(name, []):
            add(rule)
    for rule in prompt_guidelines:
        add(rule)
    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return "\n".join(f"- {rule}" for rule in rules)


def build_system_prompt_sections(options: BuildSystemPromptOptions) -> SystemPromptSections:
    resolved = normalize_build_system_prompt_options(options)
    for name in resolved.sections:
        if _SECTION_NAME.fullmatch(name) is None or name == "preamble":
            raise ValueError(f"Invalid system prompt section name: {name}")
    sections: SystemPromptSections = {}
    if resolved.custom_prompt:
        sections["preamble"] = resolved.custom_prompt
    else:
        sections["preamble"] = _PREAMBLE
        visible = [name for name in resolved.selected_tools if resolved.tool_snippets.get(name)]
        tools = "\n".join(f"- {name}: {resolved.tool_snippets[name]}" for name in visible) if visible else "(none)"
        sections["tools"] = (
            f"{tools}\n\nIn addition to the tools above, you may have access to other custom tools "
            "depending on the project."
        )
        sections["rules"] = _build_rules(
            resolved.selected_tools, resolved.tool_guidelines, resolved.prompt_guidelines,
        )
        sections["docs"] = _DOCS_TEMPLATE.format(
            readme=get_readme_path(), docs=get_docs_path(), examples=get_examples_path(),
        )
    if resolved.append_system_prompt:
        sections["addendum"] = resolved.append_system_prompt
    if resolved.context_files:
        sections["project_context"] = _render_project_context(resolved.context_files)
    read_tool: Literal["read", "bash"] | None = (
        "read" if "read" in resolved.selected_tools else
        "bash" if "bash" in resolved.selected_tools else None
    )
    if read_tool is not None and resolved.skills:
        skill_prompt = format_skills_for_prompt(resolved.skills, read_tool).strip()
        if skill_prompt:
            sections["skills"] = skill_prompt
    sections["cwd"] = resolved.cwd.replace("\\", "/")
    for name, content in resolved.sections.items():
        if content:
            sections[name] = content
    return {
        name: content if name == "preamble" else f"<{name}>\n{content}\n</{name}>"
        for name, content in sections.items()
    }


def build_system_prompt_state(options: BuildSystemPromptOptions) -> SystemPromptState:
    if options.force_system_prompt is not None:
        return SystemPromptState(options.force_system_prompt)
    return SystemPromptState("", build_system_prompt_sections(options))


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    state = build_system_prompt_state(options)
    return get_system_message_text(SystemMessage(content=state.content, sections=state.sections, timestamp=0))


def diff_system_prompt_sections(
    previous: dict[str, str | None], current: SystemPromptSections,
) -> dict[str, str | None] | None:
    patch: dict[str, str | None] = {}
    for name, text in current.items():
        if previous.get(name) != text:
            patch[name] = text
    for name in previous:
        if name not in current:
            patch[name] = None
    return patch or None


__all__ = [
    "BuildSystemPromptOptions", "NormalizedBuildSystemPromptOptions", "SystemPromptSections",
    "SystemPromptState", "normalize_build_system_prompt_options", "build_system_prompt_sections",
    "build_system_prompt_state", "build_system_prompt", "diff_system_prompt_sections",
]
