"""Tests for JSON streaming parse, proxy event reduction, prompt templates, shell capture."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core._pi_ai.json_parse import parse_json_with_repair, parse_streaming_json, repair_json
from pi_agent_core._pi_ai.types import AssistantMessage, TextContent, ToolCall, Usage, Cost
from pi_agent_core.harness.env.local import create_local_execution_env
from pi_agent_core.harness.prompt_templates import (
    PromptTemplate,
    format_prompt_template_invocation,
    load_prompt_templates,
    parse_command_args,
    substitute_args,
)
from pi_agent_core.harness.utils.shell_output import execute_shell_with_capture
from pi_agent_core.proxy import process_proxy_event, ProxyStreamOptions, stream_proxy


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_parse_streaming_json_handles_partial_input():
    assert parse_streaming_json(None) == {}
    assert parse_streaming_json("") == {}
    assert parse_streaming_json('{"a": 1}') == {"a": 1}
    assert parse_streaming_json('{"a": 1, "b": [1, 2') == {"a": 1, "b": [1, 2]}
    assert parse_streaming_json('{"command": "ls') == {"command": "ls"}
    assert parse_streaming_json("garbage") == {}


def test_repair_json_escapes_control_characters():
    repaired = repair_json('{"a": "line\nbreak"}')
    assert json.loads(repaired) == {"a": "line\nbreak"}

    repaired = repair_json('{"a": "bad \\q escape"}')
    assert json.loads(repaired) == {"a": "bad \\q escape"}


def test_parse_json_with_repair_roundtrip():
    assert parse_json_with_repair('{"x": 1}') == {"x": 1}
    assert parse_json_with_repair('{"x": "tab\there"}') == {"x": "tab\there"}


# ---------------------------------------------------------------------------
# Proxy event reduction
# ---------------------------------------------------------------------------


def make_partial() -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="faux",
        provider="faux",
        model="faux-1",
        usage=Usage(cost=Cost()),
        stop_reason="pending",
        timestamp=0,
    )


def test_process_proxy_event_text_and_thinking():
    partial = make_partial()
    start = process_proxy_event({"type": "start"}, partial)
    assert start.type == "start"

    event = process_proxy_event({"type": "text_start", "contentIndex": 0}, partial)
    assert event.type == "text_start"
    assert isinstance(partial.content[0], TextContent)

    event = process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": "hi"}, partial)
    assert event.delta == "hi"
    assert partial.content[0].text == "hi"

    event = process_proxy_event({"type": "text_end", "contentIndex": 0, "contentSignature": "sig"}, partial)
    assert event.content == "hi"
    assert partial.content[0].text_signature == "sig"

    process_proxy_event({"type": "thinking_start", "contentIndex": 1}, partial)
    process_proxy_event({"type": "thinking_delta", "contentIndex": 1, "delta": "think"}, partial)
    assert partial.content[1].thinking == "think"


def test_process_proxy_event_toolcall_streams_partial_json():
    partial = make_partial()
    process_proxy_event(
        {"type": "toolcall_start", "contentIndex": 0, "id": "t1", "toolName": "bash"}, partial
    )
    process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": '{"command"'}, partial)
    assert partial.content[0].arguments == {}
    process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": ': "ls"}'}, partial)
    assert partial.content[0].arguments == {"command": "ls"}

    event = process_proxy_event(
        {
            "type": "toolcall_end",
            "contentIndex": 0,
            "toolCall": {"id": "t1", "name": "bash", "arguments": {"command": "ls -la"}},
        },
        partial,
    )
    assert event.tool_call.arguments == {"command": "ls -la"}
    assert not hasattr(partial.content[0], "partial_json")


def test_process_proxy_event_done_and_error():
    partial = make_partial()
    event = process_proxy_event(
        {
            "type": "done",
            "reason": "stop",
            "usage": {"input": 5, "output": 7, "totalTokens": 12},
        },
        partial,
    )
    assert event.type == "done"
    assert partial.usage.input == 5
    assert partial.stop_reason == "stop"

    partial = make_partial()
    event = process_proxy_event(
        {"type": "error", "reason": "error", "errorMessage": "boom", "usage": {"input": 1}}, partial
    )
    assert event.type == "error"
    assert partial.error_message == "boom"


def test_process_proxy_event_type_mismatch_raises():
    partial = make_partial()
    with pytest.raises(RuntimeError, match="non-text content"):
        process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": "x"}, partial)


async def test_stream_proxy_reports_connection_errors():
    # No server listening: the stream must surface a terminal error event.
    from pi_agent_core._pi_ai.transcript import TranscriptContext

    stream = stream_proxy(
        type("M", (), {"id": "m", "api": "faux", "provider": "faux", "name": "m", "base_url": "", "reasoning": False, "input": [], "cost": Cost(), "context_window": 0, "max_tokens": 0})(),
        TranscriptContext(messages=[]),
        ProxyStreamOptions(auth_token="t", proxy_url="http://127.0.0.1:1"),
    )
    result = await stream.result
    assert result.stop_reason == "error"
    assert result.error_message


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


async def test_load_prompt_templates_from_dir_and_file(workdir):
    os.makedirs(os.path.join(workdir, "prompts"))
    with open(os.path.join(workdir, "prompts", "review.md"), "w") as handle:
        handle.write("---\ndescription: Review code\n---\nReview the diff.\n")
    with open(os.path.join(workdir, "prompts", "plain.md"), "w") as handle:
        handle.write("First line of the body\nMore text\n")
    with open(os.path.join(workdir, "prompts", "notes.txt"), "w") as handle:
        handle.write("not markdown\n")

    env = create_local_execution_env(cwd=workdir)
    templates, diagnostics = await load_prompt_templates(
        env, os.path.join(workdir, "prompts"), BACKGROUND_CONTEXT
    )
    by_name = {t.name: t for t in templates}
    assert set(by_name) == {"review", "plain"}
    assert by_name["review"].description == "Review code"
    assert by_name["review"].content == "Review the diff."
    # Description falls back to the first non-empty line, truncated at 60 chars.
    assert by_name["plain"].description == "First line of the body"

    single, _diag = await load_prompt_templates(
        env, os.path.join(workdir, "prompts", "review.md"), BACKGROUND_CONTEXT
    )
    assert [t.name for t in single] == ["review"]

    missing, diag = await load_prompt_templates(env, os.path.join(workdir, "nope"), BACKGROUND_CONTEXT)
    assert missing == [] and diag == []


def test_parse_command_args_and_substitute_args():
    assert parse_command_args('one two "three four"') == ["one", "two", "three four"]
    assert parse_command_args("a\t'b c'") == ["a", "b c"]

    content = "Run $1 against $2. All: $ARGUMENTS. Range: ${@:2}. Slice: ${@:2:2}. At: $@"
    args = ["alpha", "beta", "gamma", "delta"]
    result = substitute_args(content, args)
    assert "Run alpha against beta." in result
    assert "All: alpha beta gamma delta." in result
    assert "Range: beta gamma delta." in result
    assert "Slice: beta gamma." in result
    assert "At: alpha beta gamma delta" in result


def test_format_prompt_template_invocation():
    template = PromptTemplate(name="x", description="d", content="Do $1")
    assert format_prompt_template_invocation(template, ["it"]) == "Do it"


# ---------------------------------------------------------------------------
# Shell capture
# ---------------------------------------------------------------------------


async def test_execute_shell_with_capture_collects_output(workdir):
    env = create_local_execution_env(cwd=workdir)
    chunks = []

    class Options:
        pass

    options = Options()
    options.cwd = None
    options.env = None
    options.inherit_env = True
    options.timeout = None
    options.stdin = None
    options.on_chunk = lambda chunk, get_progress, ctx: chunks.append((chunk, get_progress().output))
    options.return_execution_errors = False

    result = await execute_shell_with_capture(env, "printf 'a\nb\nc\n'", options, BACKGROUND_CONTEXT)
    assert result.ok
    assert result.value.exit_code == 0
    assert "a" in result.value.output
    assert chunks, "expected incremental chunks"


async def test_execute_shell_with_capture_returns_execution_errors(workdir):
    env = create_local_execution_env(cwd=workdir)

    class Options:
        cwd = None
        env = None
        inherit_env = True
        timeout = 0.2
        stdin = None
        on_chunk = None
        return_execution_errors = True

    result = await execute_shell_with_capture(env, "sleep 5", Options(), BACKGROUND_CONTEXT)
    assert result.ok
    assert result.value.execution_error is not None
    assert result.value.execution_error.code == "timeout"
    assert result.value.cancelled is False
