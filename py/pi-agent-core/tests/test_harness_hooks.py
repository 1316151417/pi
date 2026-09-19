"""Tests for harness hooks, config validation, and the tool execution pipeline."""

from __future__ import annotations

import asyncio

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core.harness.config import (
    DEFAULT_RETRY_POLICY,
    validate_compaction_settings,
    validate_retry_policy,
    validate_tool_names,
)
from pi_agent_core.harness.compaction.compaction import CompactionSettings
from pi_agent_core.harness.execution import (
    AfterToolPatch,
    BeforeToolDecision,
    ClearedToolCall,
    create_gate,
    create_tool_result_message,
    execute_tool_call,
    finalize_tool_call,
    prepare_tool_call,
    tool_result_from_message,
)
from pi_agent_core.harness.hooks import (
    HookRegistry,
    apply_stream_options_patch,
    create_stream_options_patch,
)
from pi_agent_core.harness.types import (
    AgentHarnessStreamOptions,
    AgentHarnessStreamOptionsPatch,
    AgentHarnessTool,
    ExecutionToolContext,
)
from pi_agent_core.harness.utils.retry import RetryPolicy

ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def echo_tool(executed=None) -> AgentHarnessTool:
    async def execute(tool_call_id, params, on_update, tool_context, invocation, context):
        from pi_agent_core.types import AgentToolResult
        from pi_agent_core._pi_ai.types import TextContent

        if executed is not None:
            executed.append(params)
        return AgentToolResult(content=[TextContent(text=f"echo: {params['text']}")], details={"ok": True})

    return AgentHarnessTool(name="echo", label="echo", description="echo", parameters=ECHO_SCHEMA, execute=execute)


class _Call:
    def __init__(self, name: str, arguments: dict, call_id: str = "c1") -> None:
        self.id = call_id
        self.name = name
        self.arguments = arguments


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_retry_policy_and_validators():
    assert DEFAULT_RETRY_POLICY.enabled is True
    assert DEFAULT_RETRY_POLICY.max_retries == 3

    validate_retry_policy(RetryPolicy(max_retries=0, base_delay_ms=0))
    with pytest.raises(ValueError, match="Retry policy"):
        validate_retry_policy(RetryPolicy(max_retries=-1, base_delay_ms=0))

    validate_compaction_settings(CompactionSettings(reserve_tokens=0, keep_recent_tokens=0))
    with pytest.raises(ValueError, match="Compaction token counts"):
        validate_compaction_settings(CompactionSettings(reserve_tokens=-1, keep_recent_tokens=0))


def test_validate_tool_names_rejects_duplicates():
    validate_tool_names([echo_tool(), AgentHarnessTool(name="other", description="", parameters={})])
    with pytest.raises(TypeError, match="Duplicate tool name"):
        validate_tool_names([echo_tool(), echo_tool()])


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


async def test_hook_registry_before_run_accumulates_messages():
    registry = HookRegistry(lambda error, hook, lane, context: asyncio.sleep(0))
    from pi_agent_core._pi_ai.types import UserMessage

    registry.on("before_run", lambda event, ctx: {"messages": [UserMessage(content="extra", timestamp=1)]})
    registry.on("before_run", lambda event, ctx: None)

    result = await registry.aggregate("before_run", {"prompt": [], "lane": "main", "runId": "r1"}, BACKGROUND_CONTEXT)
    assert len(result["messages"]) == 1


async def test_hook_registry_transform_context_and_failures_reported():
    reported = []

    async def _report(error, hook, lane, context):
        reported.append((hook, str(error)))

    registry = HookRegistry(_report)

    def _boom(event, ctx):
        raise RuntimeError("hook failed")

    registry.on("transform_context", _boom)
    registry.on("transform_context", lambda event, ctx: {"systemPrompt": "replaced"})

    result = await registry.aggregate(
        "transform_context",
        {"messages": ["m"], "systemPrompt": "original", "lane": "main", "runId": "r1"},
        BACKGROUND_CONTEXT,
    )
    assert result["systemPrompt"] == "replaced"
    assert reported == [("transform_context", "hook failed")]


async def test_hook_registry_before_tool_block_wins_and_after_tool_aggregates():
    registry = HookRegistry(lambda error, hook, lane, context: asyncio.sleep(0))
    registry.on("before_tool", lambda event, ctx: {"args": {"text": "changed"}})
    registry.on("before_tool", lambda event, ctx: {"block": {"reason": "denied", "terminate": True}})
    registry.on("before_tool", lambda event, ctx: {"args": {"text": "ignored"}})

    result = await registry.aggregate(
        "before_tool", {"args": {"text": "original"}, "lane": "main", "runId": "r1"}, BACKGROUND_CONTEXT
    )
    assert result["args"] == {"text": "changed"}
    assert result["block"] == {"reason": "denied", "terminate": True}

    registry2 = HookRegistry(lambda error, hook, lane, context: asyncio.sleep(0))
    registry2.on("after_tool", lambda event, ctx: {"isError": True, "details": {"d": 1}})
    after = await registry2.aggregate(
        "after_tool",
        {"content": [], "isError": False, "lane": "main", "runId": "r1"},
        BACKGROUND_CONTEXT,
    )
    assert after == {"isError": True, "details": {"d": 1}}


async def test_hook_registry_before_tool_failure_blocks():
    registry = HookRegistry(lambda error, hook, lane, context: asyncio.sleep(0))

    def _boom(event, ctx):
        raise RuntimeError("hook exploded")

    registry.on("before_tool", _boom)
    result = await registry.aggregate("before_tool", {"args": {}, "lane": "main"}, BACKGROUND_CONTEXT)
    assert result["block"] == {"reason": "hook exploded"}


async def test_hook_registry_before_drive_fails_closed():
    registry = HookRegistry(lambda error, hook, lane, context: asyncio.sleep(0))

    def _boom(event, ctx):
        raise RuntimeError("closed")

    registry.on("before_drive", _boom)
    with pytest.raises(RuntimeError, match="closed"):
        await registry.aggregate("before_drive", {"lane": "main"}, BACKGROUND_CONTEXT)


async def test_hook_registry_structural_decline_and_conflict():
    reported = []
    registry = HookRegistry(lambda error, hook, lane, context: reported.append(str(error)))
    registry.on("before_compaction", lambda event, ctx: {"decline": True, "compaction": {"x": 1}})
    registry.on("before_compaction", lambda event, ctx: {"compaction": {"summary": "s"}})

    result = await registry.aggregate("before_compaction", {"lane": "main"}, BACKGROUND_CONTEXT)
    assert result == {"compaction": {"summary": "s"}}
    assert reported and "cannot return both decline and compaction" in reported[0]


async def test_hooks_close_rejects_registration():
    registry = HookRegistry(lambda error, hook, lane, context: asyncio.sleep(0))
    registry.close(RuntimeError("closed"))
    with pytest.raises(RuntimeError, match="closed"):
        registry.on("before_run", lambda event, ctx: None)


def test_stream_option_patch_roundtrip():
    base = AgentHarnessStreamOptions(timeout_ms=100, headers={"a": "1", "b": "2"})
    patch = AgentHarnessStreamOptionsPatch(headers={"a": None, "c": "3"})
    updated = apply_stream_options_patch(base, patch)
    assert updated.headers == {"b": "2", "c": "3"}
    assert updated.timeout_ms == 100

    derived = create_stream_options_patch(base, updated)
    assert derived.headers == {"a": None, "c": "3"}
    assert apply_stream_options_patch(base, derived).headers == updated.headers

    cleared = apply_stream_options_patch(base, AgentHarnessStreamOptionsPatch(headers=None))
    assert cleared.headers is None


# ---------------------------------------------------------------------------
# Tool execution pipeline
# ---------------------------------------------------------------------------


def test_prepare_tool_call_validates_and_reports_missing_tool():
    prepared = prepare_tool_call(_Call("echo", {"text": "hi"}), [echo_tool()])
    assert prepared.args == {"text": "hi"}

    missing = prepare_tool_call(_Call("nope", {}), [echo_tool()])
    assert missing.kind == "immediate"
    assert missing.is_error is True
    assert "unavailable" in missing.result.content[0].text

    invalid = prepare_tool_call(_Call("echo", {}), [echo_tool()])
    assert invalid.is_error is True
    assert "Validation failed" in invalid.result.content[0].text


def test_apply_before_tool_decision_block_and_revalidate():
    prepared = prepare_tool_call(_Call("echo", {"text": "hi"}), [echo_tool()])
    cleared = ClearedToolCall(tool_call=prepared.tool_call, tool=prepared.tool, args=prepared.args)

    from pi_agent_core.harness.execution import apply_before_tool_decision

    blocked = apply_before_tool_decision(prepared, BeforeToolDecision(block={"reason": "no", "terminate": True}))
    assert blocked.kind == "immediate"
    assert blocked.terminate is True

    rewritten = apply_before_tool_decision(prepared, BeforeToolDecision(args={"text": "other"}))
    assert rewritten.args == {"text": "other"}
    assert rewritten is not prepared  # a cleared call is returned

    invalid = apply_before_tool_decision(prepared, BeforeToolDecision(args={}))
    assert getattr(invalid, "kind", None) == "immediate"


async def test_execute_tool_call_runs_behind_gate():
    executed = []
    tool = echo_tool(executed)
    gate, _control = create_gate()
    context = ExecutionToolContext(env=None, cwd="", description="")

    call = ClearedToolCall(tool_call=_Call("echo", {"text": "hi"}), tool=tool, args={"text": "hi"})
    outcome = await execute_tool_call(call, gate, None, context, None, BACKGROUND_CONTEXT)
    assert outcome.is_error is False
    assert executed == [{"text": "hi"}]


async def test_execute_tool_call_converts_throws_to_error_output():
    async def _failing(tool_call_id, params, on_update, tool_context, invocation, context):
        raise RuntimeError("tool blew up")

    tool = AgentHarnessTool(name="boom", label="boom", description="", parameters={}, execute=_failing)
    gate, _control = create_gate()
    call = ClearedToolCall(tool_call=_Call("boom", {}), tool=tool, args={})
    outcome = await execute_tool_call(call, gate, None, None, None, BACKGROUND_CONTEXT)
    assert outcome.is_error is True
    assert outcome.result.content[0].text == "tool blew up"


async def test_execute_tool_call_blocked_by_closed_gate():
    tool = echo_tool()
    gate, control = create_gate()
    control.close(RuntimeError("gate closed"))
    call = ClearedToolCall(tool_call=_Call("echo", {"text": "hi"}), tool=tool, args={"text": "hi"})
    with pytest.raises(RuntimeError, match="gate closed"):
        await execute_tool_call(call, gate, None, None, None, BACKGROUND_CONTEXT)


def test_finalize_tool_call_applies_patch_field_by_field():
    from pi_agent_core._pi_ai.types import TextContent
    from pi_agent_core.types import AgentToolResult

    call = ClearedToolCall(tool_call=_Call("echo", {}), tool=None, args={})
    executed = type("E", (), {"result": AgentToolResult(content=[TextContent(text="raw")], details={"d": 1}), "is_error": False})()

    patched = finalize_tool_call(call, executed, AfterToolPatch(content=[TextContent(text="patched")], is_error=True))
    assert patched.result.content[0].text == "patched"
    assert patched.result.details == {"d": 1}  # untouched field survives
    assert patched.is_error is True
    assert patched.terminate is False

    message = create_tool_result_message(patched)
    assert message.tool_call_id == "c1"
    assert message.is_error is True

    restored = tool_result_from_message(message, True)
    assert restored.content[0].text == "patched"
    assert restored.terminate is True
