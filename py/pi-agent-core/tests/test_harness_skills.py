"""Tests for harness skills / system-prompt / messages."""

from __future__ import annotations

import os
import tempfile

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core.harness.env.local import create_local_execution_env
from pi_agent_core.harness.messages import (
    BashExecutionMessage,
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    bash_execution_to_text,
)
from pi_agent_core.harness.skills import format_skill_invocation, load_skills
from pi_agent_core.harness.system_prompt import format_skills_for_system_prompt
from pi_agent_core.harness.types import Skill


async def test_load_skills_discovers_skill_md(workdir):
    skill_dir = os.path.join(workdir, "my-skill")
    os.makedirs(skill_dir)
    with open(os.path.join(skill_dir, "SKILL.md"), "w") as handle:
        handle.write("---\ndescription: Does something useful.\n---\n\nInstructions here.\n")

    env = create_local_execution_env(cwd=workdir)
    skills, diagnostics = await load_skills(env, workdir, BACKGROUND_CONTEXT)
    assert len(skills) == 1
    assert skills[0].name == "my-skill"
    assert skills[0].description == "Does something useful."
    assert skills[0].content == "Instructions here."
    assert skills[0].file_path.endswith("SKILL.md")
    assert diagnostics == []


async def test_load_skills_nested_and_root_md(workdir):
    nested = os.path.join(workdir, "nested-skill")
    os.makedirs(nested)
    with open(os.path.join(nested, "SKILL.md"), "w") as handle:
        handle.write("---\ndescription: Nested.\n---\nBody\n")
    with open(os.path.join(workdir, "root-skill.md"), "w") as handle:
        handle.write("---\nname: root-skill\ndescription: Root level.\n---\nRoot body\n")
    with open(os.path.join(workdir, "plain.md"), "w") as handle:
        handle.write("# No frontmatter\n")

    env = create_local_execution_env(cwd=workdir)
    skills, _diagnostics = await load_skills(env, workdir, BACKGROUND_CONTEXT)
    names = [skill.name for skill in skills]
    assert "nested-skill" in names
    assert "root-skill" in names  # root .md files with frontmatter
    assert "plain" not in names


async def test_load_skills_honors_ignore_files(workdir):
    ignored = os.path.join(workdir, "ignored-skill")
    os.makedirs(ignored)
    with open(os.path.join(ignored, "SKILL.md"), "w") as handle:
        handle.write("---\ndescription: Ignored.\n---\n")
    with open(os.path.join(workdir, ".gitignore"), "w") as handle:
        handle.write("ignored-skill/\n")

    env = create_local_execution_env(cwd=workdir)
    skills, _diagnostics = await load_skills(env, workdir, BACKGROUND_CONTEXT)
    assert skills == []


async def test_load_skills_invalid_metadata_diagnostics(workdir):
    skill_dir = os.path.join(workdir, "Bad_Name")
    os.makedirs(skill_dir)
    with open(os.path.join(skill_dir, "SKILL.md"), "w") as handle:
        handle.write("---\nname: wrong-name\ndescription: X\n---\nBody\n")

    env = create_local_execution_env(cwd=workdir)
    skills, diagnostics = await load_skills(env, workdir, BACKGROUND_CONTEXT)
    # Invalid names still load but produce warnings.
    assert len(skills) == 1
    assert any(d.code == "invalid_metadata" for d in diagnostics)


async def test_load_skills_missing_dir_skipped(workdir):
    env = create_local_execution_env(cwd=workdir)
    skills, diagnostics = await load_skills(env, os.path.join(workdir, "missing"), BACKGROUND_CONTEXT)
    assert skills == []
    assert diagnostics == []


def test_format_skills_for_system_prompt():
    skills = [
        Skill(name="alpha", description="First <skill>", content="c1", file_path="/a/SKILL.md"),
        Skill(name="hidden", description="Nope", content="c2", file_path="/b/SKILL.md", disable_model_invocation=True),
    ]
    text = format_skills_for_system_prompt(skills)
    assert "<available_skills>" in text
    assert "<name>alpha</name>" in text
    assert "First &lt;skill&gt;" in text
    assert "<location>/a/SKILL.md</location>" in text
    assert "hidden" not in text

    assert format_skills_for_system_prompt([]) == ""


def test_format_skill_invocation():
    skill = Skill(name="alpha", description="d", content="Do the thing.", file_path="/skills/alpha/SKILL.md")
    text = format_skill_invocation(skill, "extra instructions")
    assert text.startswith('<skill name="alpha" location="/skills/alpha/SKILL.md">')
    assert "References are relative to /skills/alpha." in text
    assert text.endswith("</skill>\n\nextra instructions")


def test_bash_execution_to_text():
    message = BashExecutionMessage(
        command="ls -la",
        output="file.txt",
        exit_code=1,
        cancelled=False,
        truncated=True,
        full_output_path="/tmp/out.log",
        timestamp=1,
    )
    text = bash_execution_to_text(message)
    assert text.startswith("Ran `ls -la`")
    assert "```" in text
    assert "Command exited with code 1" in text
    assert "[Output truncated. Full output: /tmp/out.log]" in text

    cancelled = BashExecutionMessage(
        command="sleep", output="", exit_code=None, cancelled=True, truncated=False, timestamp=1
    )
    text = bash_execution_to_text(cancelled)
    assert "(no output)" in text
    assert "(command cancelled)" in text


def test_convert_to_llm_maps_custom_roles():
    from pi_ai.types import TextContent, UserMessage

    messages = [
        UserMessage(content="hi", timestamp=1),
        BashExecutionMessage(
            command="echo hi",
            output="hi",
            exit_code=0,
            cancelled=False,
            truncated=False,
            timestamp=2,
        ),
        create_branch_summary_message("branch summary", None, 3),
        create_compaction_summary_message("compaction summary", 500, 4),
        BashExecutionMessage(
            command="secret",
            output="",
            exit_code=0,
            cancelled=False,
            truncated=False,
            timestamp=5,
            exclude_from_context=True,
        ),
    ]
    converted = convert_to_llm(messages)
    assert len(converted) == 4
    assert all(isinstance(m, UserMessage) or getattr(m, "role") == "user" for m in converted)
    assert converted[0].content == "hi"
    assert "Ran `echo hi`" in converted[1].content[0].text
    assert "branch summary" in converted[2].content[0].text
    assert "compaction summary" in converted[3].content[0].text
    assert "secret" not in str(converted)
