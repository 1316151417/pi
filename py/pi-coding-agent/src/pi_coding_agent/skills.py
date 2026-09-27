"""Skill metadata and prompt rendering from ``core/skills.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source_info: object
    disable_model_invocation: bool = False


def _escape_xml(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))


def format_skills_for_prompt(
    skills: list[Skill], file_read_tool: Literal["read", "bash"] = "read",
) -> str:
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""
    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description."
        if file_read_tool == "read" else
        "Use bash to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "", "<available_skills>",
    ]
    for skill in visible:
        lines.extend([
            "  <skill>",
            f"    <name>{_escape_xml(skill.name)}</name>",
            f"    <description>{_escape_xml(skill.description)}</description>",
            f"    <location>{_escape_xml(skill.file_path)}</location>",
            "  </skill>",
        ])
    lines.append("</available_skills>")
    return "\n".join(lines)


__all__ = ["Skill", "format_skills_for_prompt"]
