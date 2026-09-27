"""Project context and skill discovery from ``core/resource-loader.ts``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import get_agent_dir
from .skills import Skill

_CONTEXT_NAMES = ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class ResourceDiagnostic:
    type: str
    message: str
    path: str


@dataclass
class LoadedSkills:
    skills: list[Skill]
    diagnostics: list[ResourceDiagnostic]


def _context_from_dir(directory: Path) -> dict[str, str] | None:
    for name in _CONTEXT_NAMES:
        path = directory / name
        if path.is_file():
            return {"path": str(path), "content": path.read_text(encoding="utf-8-sig")}
    return None


def load_project_context_files(cwd: str, agent_dir: str | None = None) -> list[dict[str, str]]:
    resolved = Path(cwd).expanduser().resolve()
    global_context = _context_from_dir(Path(agent_dir or get_agent_dir()).expanduser().resolve())
    result = [global_context] if global_context else []
    ancestors = []
    while True:
        context = _context_from_dir(resolved)
        if context and not any(item["path"] == context["path"] for item in result):
            ancestors.insert(0, context)
        if resolved.parent == resolved:
            break
        resolved = resolved.parent
    return result + ancestors


def _frontmatter(content: str) -> dict[str, str]:
    normalized = content.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}
    end = normalized.find("\n---", 4)
    if end < 0:
        return {}
    fields: dict[str, str] = {}
    for line in normalized[4:end].split("\n"):
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(value[0]):
            value = value[1:-1]
        fields[key.strip()] = value
    return fields


def _load_skill(path: Path, source: str, diagnostics: list[ResourceDiagnostic]) -> Skill | None:
    try:
        metadata = _frontmatter(path.read_text(encoding="utf-8-sig"))
    except OSError as error:
        diagnostics.append(ResourceDiagnostic("warning", str(error), str(path)))
        return None
    name = metadata.get("name") or path.parent.name
    description = metadata.get("description", "")
    if not description.strip():
        if path.name == "SKILL.md":
            diagnostics.append(ResourceDiagnostic("warning", "description is required", str(path)))
        return None
    if len(name) > 64 or not _SKILL_NAME.fullmatch(name):
        diagnostics.append(ResourceDiagnostic("warning", "invalid skill name", str(path)))
    if len(description) > 1024:
        diagnostics.append(ResourceDiagnostic("warning", "description exceeds 1024 characters", str(path)))
    scope = "user" if source == "user" else "project" if source == "project" else "temporary"
    return Skill(
        name=name, description=description, file_path=str(path), base_dir=str(path.parent),
        source_info={"path": str(path), "source": "local", "scope": scope,
                     "origin": "top-level", "baseDir": str(path.parent)},
        disable_model_invocation=metadata.get("disable-model-invocation") == "true",
    )


def load_skills_from_dir(directory: str, source: str = "path") -> LoadedSkills:
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []
    root = Path(directory).expanduser().resolve()
    seen_dirs: set[Path] = set()

    def walk(path: Path, include_root_files: bool) -> None:
        if not path.is_dir():
            return
        canonical = path.resolve()
        if canonical in seen_dirs:
            return
        seen_dirs.add(canonical)
        declared = path / "SKILL.md"
        if declared.is_file():
            skill = _load_skill(declared, source, diagnostics)
            if skill:
                skills.append(skill)
            return
        try:
            children = list(path.iterdir())
        except OSError as error:
            diagnostics.append(ResourceDiagnostic("warning", str(error), str(path)))
            return
        for child in children:
            if child.name.startswith(".") or child.name == "node_modules":
                continue
            if child.is_dir():
                walk(child, False)
            elif include_root_files and child.is_file() and child.suffix == ".md":
                skill = _load_skill(child, source, diagnostics)
                if skill:
                    skills.append(skill)

    walk(root, True)
    return LoadedSkills(skills, diagnostics)


def load_skills(
    cwd: str, agent_dir: str | None = None, skill_paths: Sequence[str] = (),
    include_defaults: bool = True,
) -> LoadedSkills:
    agent = Path(agent_dir or get_agent_dir()).expanduser().resolve()
    project = Path(cwd).expanduser().resolve()
    locations: list[tuple[Path, str]] = []
    if include_defaults:
        locations.extend(((agent / "skills", "user"), (project / ".pi" / "skills", "project")))
    locations.extend((Path(path).expanduser().resolve(), "path") for path in skill_paths)
    result = LoadedSkills([], [])
    seen_files: set[Path] = set()
    names: dict[str, Skill] = {}
    for path, source in locations:
        if source in ("user", "project") and not path.exists():
            continue
        if path.is_dir():
            loaded = load_skills_from_dir(str(path), source)
        elif path.is_file() and path.suffix == ".md":
            diagnostics: list[ResourceDiagnostic] = []
            skill = _load_skill(path, source, diagnostics)
            loaded = LoadedSkills([skill] if skill else [], diagnostics)
        else:
            result.diagnostics.append(ResourceDiagnostic("warning", "skill path does not exist", str(path)))
            continue
        result.diagnostics.extend(loaded.diagnostics)
        for skill in loaded.skills:
            canonical = Path(skill.file_path).resolve()
            if canonical in seen_files:
                continue
            if skill.name in names:
                result.diagnostics.append(ResourceDiagnostic(
                    "collision", f'name "{skill.name}" collision', skill.file_path,
                ))
                continue
            seen_files.add(canonical)
            names[skill.name] = skill
            result.skills.append(skill)
    return result


__all__ = [
    "ResourceDiagnostic", "LoadedSkills", "load_project_context_files",
    "load_skills_from_dir", "load_skills",
]
