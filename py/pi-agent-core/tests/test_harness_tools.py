"""Integration tests for the harness execution environment and tools."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core._chord.context import Context as ChordContext
from pi_agent_core.harness.env.local import create_local_execution_env
from pi_agent_core.harness.tools import (
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from pi_agent_core.harness.types import (
    AgentHarnessTool,
    ExecutionToolContext,
    get_or_throw,
)


def make_tool_context(env) -> ExecutionToolContext:
    return ExecutionToolContext(env=env, cwd=env.cwd, description="test")


async def test_fs_read_write_roundtrip(workdir):
    env = create_local_execution_env(cwd=workdir)
    context = BACKGROUND_CONTEXT

    write = await env.write_file("hello.txt", "line 1\nline 2\n", context)
    assert write.ok

    text = get_or_throw(await env.read_text_file("hello.txt", context))
    assert text == "line 1\nline 2\n"

    exists = get_or_throw(await env.exists("hello.txt", context))
    assert exists is True
    missing = get_or_throw(await env.exists("nope.txt", context))
    assert missing is False

    info = get_or_throw(await env.file_info("hello.txt", context))
    assert info.kind == "file"
    assert info.size > 0

    absolute = get_or_throw(await env.absolute_path("hello.txt", context))
    assert os.path.isabs(absolute)
    canonical = get_or_throw(await env.canonical_path("hello.txt", context))
    assert canonical == os.path.realpath(absolute)


async def test_fs_errors_encoded_as_results(workdir):
    env = create_local_execution_env(cwd=workdir)
    context = BACKGROUND_CONTEXT

    result = await env.read_text_file("missing.txt", context)
    assert not result.ok
    assert result.error.code == "not_found"

    result = await env.file_info(workdir, context)
    assert result.ok and result.value.kind == "directory"


async def test_exec_captures_output_and_exit_code(workdir):
    env = create_local_execution_env(cwd=workdir)
    context = BACKGROUND_CONTEXT

    result = await env.exec("printf 'out1\nout2\n'", None, context)
    assert result.ok
    assert result.value.exit_code == 0
    # Exit code surfaces through ShellExecResult (stdout delivered via capture view).

    failing = await env.exec("exit 3", None, context)
    assert failing.ok
    assert failing.value.exit_code == 3


async def test_exec_timeout_returns_error(workdir):
    env = create_local_execution_env(cwd=workdir)
    context = BACKGROUND_CONTEXT

    from pi_agent_core.harness.types import ShellExecOptions

    result = await env.exec("sleep 5", ShellExecOptions(timeout=0.2), context)
    assert not result.ok
    assert result.error.code == "timeout"


async def test_exec_on_update_streaming(workdir):
    env = create_local_execution_env(cwd=workdir)
    context = BACKGROUND_CONTEXT

    from pi_agent_core.harness.types import ShellExecOptions, ShellOutputCaptureOptions, ShellOutputLimits

    updates = []

    def on_update(update, _ctx):
        updates.append(update)

    options = ShellExecOptions(
        capture=ShellOutputCaptureOptions(limits=ShellOutputLimits(max_bytes=50 * 1024, max_lines=2000)),
        on_update=on_update,
    )
    result = await env.exec("printf 'alpha\nbeta\n'", options, context)
    assert result.ok
    assert updates, "expected at least one streaming update"
    combined = ""
    for update in updates:
        if update.kind == "replace":
            combined = update.output.text
        elif update.kind == "append":
            combined += update.text
        elif update.kind == "slide":
            combined = combined[update.drop :] + update.text
    assert "alpha" in combined
    assert "beta" in combined


async def test_write_tool(workdir):
    env = create_local_execution_env(cwd=workdir)
    tool = create_write_tool()
    result = await tool.execute(
        "call-1",
        {"path": "nested/dir/file.txt", "content": "hello world"},
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    assert [b.text for b in result.content] == ["Successfully wrote to nested/dir/file.txt"]
    assert get_or_throw(await env.read_text_file("nested/dir/file.txt", BACKGROUND_CONTEXT)) == "hello world"


async def test_read_tool_text(workdir):
    env = create_local_execution_env(cwd=workdir)
    (await env.write_file("a.txt", "\n".join(f"line {i}" for i in range(1, 8)), BACKGROUND_CONTEXT)).ok

    tool = create_read_tool()
    result = await tool.execute(
        "call-1",
        {"path": "a.txt"},
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    text = result.content[0].text
    assert "line 1" in text
    assert "line 7" in text

    partial = await tool.execute(
        "call-2",
        {"path": "a.txt", "offset": 2, "limit": 2},
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    text = partial.content[0].text
    assert "line 2" in text and "line 3" in text
    assert "line 4" not in text
    assert "4 more lines" in text
    assert "Use offset=4" in text

    with pytest.raises(Exception, match="not_found"):
        await tool.execute(
            "call-4",
            {"path": "nope.txt"},
            None,
            make_tool_context(env),
            None,
            BACKGROUND_CONTEXT,
        )


async def test_read_tool_truncation_hint(workdir):
    env = create_local_execution_env(cwd=workdir)
    big = "\n".join(f"line {i}" for i in range(1, 4001))
    (await env.write_file("big.txt", big, BACKGROUND_CONTEXT)).ok

    tool = create_read_tool()
    result = await tool.execute(
        "call-1",
        {"path": "big.txt"},
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    text = result.content[0].text
    assert "Use offset=" in text
    assert result.details is not None and result.details.get("truncation") is not None


async def test_edit_tool(workdir):
    env = create_local_execution_env(cwd=workdir)
    (await env.write_file("e.txt", "alpha\nbeta\ngamma\n", BACKGROUND_CONTEXT)).ok

    tool = create_edit_tool()
    result = await tool.execute(
        "call-1",
        {"path": "e.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]},
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    assert "Successfully replaced 1 block(s)" in result.content[0].text
    assert result.details is not None
    assert "BETA" in result.details["diff"]
    assert "beta" in result.details["patch"]
    updated = get_or_throw(await env.read_text_file("e.txt", BACKGROUND_CONTEXT))
    assert updated == "alpha\nBETA\ngamma\n"


async def test_edit_tool_multiple_and_errors(workdir):
    env = create_local_execution_env(cwd=workdir)
    (await env.write_file("m.txt", "one\ntwo\nthree\n", BACKGROUND_CONTEXT)).ok

    tool = create_edit_tool()
    result = await tool.execute(
        "call-1",
        {
            "path": "m.txt",
            "edits": [
                {"oldText": "one", "newText": "1"},
                {"oldText": "three", "newText": "3"},
            ],
        },
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    updated = get_or_throw(await env.read_text_file("m.txt", BACKGROUND_CONTEXT))
    assert updated == "1\ntwo\n3\n"

    with pytest.raises(RuntimeError, match="Could not find the exact text in m.txt"):
        await tool.execute(
            "call-2",
            {"path": "m.txt", "edits": [{"oldText": "zzz", "newText": "x"}]},
            None,
            make_tool_context(env),
            None,
            BACKGROUND_CONTEXT,
        )

    (await env.write_file("dup.txt", "x\nx\n", BACKGROUND_CONTEXT)).ok
    with pytest.raises(RuntimeError, match="2 occurrences"):
        await tool.execute(
            "call-3",
            {"path": "dup.txt", "edits": [{"oldText": "x", "newText": "y"}]},
            None,
            make_tool_context(env),
            None,
            BACKGROUND_CONTEXT,
        )


async def test_edit_tool_fuzzy_match(workdir):
    env = create_local_execution_env(cwd=workdir)
    (await env.write_file("f.txt", "hello — 'world'   \n", BACKGROUND_CONTEXT)).ok

    tool = create_edit_tool()
    await tool.execute(
        "call-1",
        {"path": "f.txt", "edits": [{"oldText": "hello - 'world'", "newText": "hi"}]},
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    updated = get_or_throw(await env.read_text_file("f.txt", BACKGROUND_CONTEXT))
    assert updated.startswith("hi")


async def test_edit_tool_prepare_arguments(workdir):
    from pi_agent_core.harness.tools.edit import prepare_edit_arguments

    legacy = prepare_edit_arguments({"path": "x", "oldText": "a", "newText": "b"})
    assert legacy["edits"] == [{"oldText": "a", "newText": "b"}]

    stringified = prepare_edit_arguments({"path": "x", "edits": '[{"oldText":"a","newText":"b"}]'})
    assert stringified["edits"] == [{"oldText": "a", "newText": "b"}]

    single = prepare_edit_arguments({"path": "x", "edits": {"oldText": "a", "newText": "b"}})
    assert single["edits"] == [{"oldText": "a", "newText": "b"}]


async def test_bash_tool(workdir):
    env = create_local_execution_env(cwd=workdir)
    tool = create_bash_tool()
    updates = []

    def on_update(partial, options=None):
        updates.append(partial)

    result = await tool.execute(
        "call-1",
        {"command": "printf 'hello\nworld\n'"},
        on_update,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    assert "hello" in result.content[0].text
    assert "world" in result.content[0].text

    with pytest.raises(RuntimeError, match="Command exited with code 1"):
        await tool.execute(
            "call-2",
            {"command": "echo oops >&2; exit 1"},
            None,
            make_tool_context(env),
            None,
            BACKGROUND_CONTEXT,
        )

    with pytest.raises(RuntimeError, match="timed out"):
        await tool.execute(
            "call-3",
            {"command": "sleep 2", "timeout": 0.2},
            None,
            make_tool_context(env),
            None,
            BACKGROUND_CONTEXT,
        )


async def test_bash_tool_spills_truncated_output(workdir):
    env = create_local_execution_env(cwd=workdir)
    tool = create_bash_tool()
    result = await tool.execute(
        "call-1",
        {"command": "seq 1 6000"},
        None,
        make_tool_context(env),
        None,
        BACKGROUND_CONTEXT,
    )
    text = result.content[0].text
    assert "Full output:" in text
    assert result.details is not None
    spill_path = result.details["fullOutputPath"]
    assert spill_path and os.path.exists(spill_path)
    spill_content = open(spill_path).read()
    assert spill_content.startswith("1\n2\n")
    assert "6000" in spill_content


async def test_file_mutation_queue_serializes(workdir):
    """Concurrent edits to the same file apply sequentially without interleaving."""
    env = create_local_execution_env(cwd=workdir)
    (await env.write_file("q.txt", "start\n", BACKGROUND_CONTEXT)).ok
    tool = create_edit_tool()

    async def do_edit(i):
        await tool.execute(
            f"call-{i}",
            {"path": "q.txt", "edits": [{"oldText": f"mark{i - 1}", "newText": f"mark{i}"}]},
            None,
            make_tool_context(env),
            None,
            BACKGROUND_CONTEXT,
        )

    (await env.write_file("q.txt", "mark0\n", BACKGROUND_CONTEXT)).ok
    await asyncio.gather(do_edit(1), do_edit(2), do_edit(3))
    final = get_or_throw(await env.read_text_file("q.txt", BACKGROUND_CONTEXT))
    assert final == "mark3\n"
