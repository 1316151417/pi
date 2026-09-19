"""Tests for the durable tool batch procedure ported from ``drive/tools.ts``.

The batch loop itself needs the full drive loop, which is still being written in
parallel, so these tests cover the procedure's pure shaping helpers with concrete
inputs and the dispatch decisions ``run_tools`` makes before any tool starts: the
exit paths, the recovery notice, the sequential/parallel split, active-tool
filtering, and tool-context resolution.
"""

from __future__ import annotations

import asyncio

import pytest

from pi_agent_core._pi_ai.types import (
    AssistantMessage,
    TextContent,
    ToolCall as ToolCallBlock,
    Usage,
)
from pi_agent_core.harness.execution.tools import FinalizedToolCall
from pi_agent_core.harness.runtime.drive import tools as tools_module
from pi_agent_core.harness.runtime.drive.tool_placement import ToolBatchSource
from pi_agent_core.harness.runtime.types import Config, LaneState, ProcedureResult
from pi_agent_core.harness.session.session import SessionInvariantError
from pi_agent_core.harness.session.types import (
    Control,
    LaneConfiguration,
    OperationMeta,
    RunSettings,
    ToolBatch,
    ToolCall,
    ToolsOperation,
)
from pi_agent_core.harness.session.values import StoredValue, operation_tool_memo
from pi_agent_core.harness.types import AgentHarnessTool
from pi_agent_core.types import AgentToolResult


class _Operation:
    """Minimal stand-in for the durable operation carried by a lane state."""

    def __init__(self, state: object) -> None:
        self.meta = OperationMeta(operation_id="op-1", lane="lane-1")
        self.state = state


class _Drive:
    """Minimal drive face used by ``run_tools`` before any effect is admitted."""

    def __init__(self) -> None:
        self.operation_id = "op-1"
        self.context = object()


class _Lane:
    """Lane face limited to the read/config/emit calls ``run_tools`` makes."""

    def __init__(self, state: LaneState, config: Config) -> None:
        self.name = "lane-1"
        self.state = state
        self._config = config
        self.emitted: list = []

    def read_config(self) -> Config:
        return self._config

    async def emit_batch(self, events: list, context: object) -> None:
        self.emitted.append(events)


def _batch(**overrides: object) -> ToolBatch:
    batch = ToolBatch(
        assistant_entry_id="assistant-1",
        configuration=LaneConfiguration(active_tool_names=["read", "write"]),
        turn_id="turn-1",
        calls=[ToolCall(source_index=0, result_entry_id="result-0", status="planned")],
    )
    for key, value in overrides.items():
        setattr(batch, key, value)
    return batch


def _run(batch: ToolBatch, **overrides: object) -> ToolsOperation:
    run = ToolsOperation(batch=batch)
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


def _lane_for(run: ToolsOperation, *, config: Config | None = None, owned: bool = True) -> _Lane:
    state = LaneState(
        tip_id="tip-1",
        configuration=LaneConfiguration(active_tool_names=["read", "write"]),
        operation=_Operation(run) if owned else None,
    )
    return _Lane(state, config or Config())


class _Reader:
    """Reader face limited to the single-value lookups memos perform."""

    def __init__(self, values: dict) -> None:
        self.values = values

    async def get_value(self, address: object, context: object) -> object:
        return self.values.get((address.namespace, address.key))


class _CommandLane(_Lane):
    """Lane that runs command plans against its own state, like the real mutation line."""

    def __init__(self, state: LaneState, config: Config, values: dict | None = None) -> None:
        super().__init__(state, config)
        self.reader = _Reader(values or {})
        self.committed: list = []

    async def command(self, plan, context: object) -> object:
        decision = plan(self.state, self.reader)
        if asyncio.iscoroutine(decision):
            decision = await decision
        if getattr(decision, "kind", None) == "return":
            return decision.result
        if getattr(decision, "kind", None) == "reject":
            raise decision.error
        self.committed.append(decision)
        return decision.materialize(None)


@pytest.fixture
def dispatch(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record which batch runner ``run_tools`` selected and with which inputs."""
    recorded: dict = {"materialize": 0}
    sources = object()

    async def _read_source(lane: object, drive: object, batch: object) -> object:
        return sources

    async def _materialize(lane: object, drive: object, run: object, read: object, recovery: bool) -> None:
        recorded["materialize"] += 1

    async def _sequential(lane, drive, run, read, execution, recovery) -> ProcedureResult:
        recorded["sequential"] = {
            "execution": execution,
            "recovery": recovery,
            "sources": read,
        }
        return ProcedureResult(kind="continue")

    async def _parallel(lane, drive, run, read, tools, tools_by_name, tool_context, recovery):
        recorded["parallel"] = {
            "tools": tools,
            "tools_by_name": tools_by_name,
            "tool_context": tool_context,
            "recovery": recovery,
            "sources": read,
        }
        return ProcedureResult(kind="continue")

    monkeypatch.setattr(tools_module, "read_tool_batch_source", _read_source)
    monkeypatch.setattr(tools_module, "materialize_ready", _materialize)
    monkeypatch.setattr(tools_module, "run_sequential", _sequential)
    monkeypatch.setattr(tools_module, "run_parallel", _parallel)
    recorded["sources"] = sources
    return recorded


# ---------------------------------------------------------------------------
# Call bookkeeping
# ---------------------------------------------------------------------------


def test_find_call_requires_source_index_and_result_entry_id() -> None:
    batch = _batch(
        calls=[
            ToolCall(source_index=0, result_entry_id="result-0", status="planned"),
            ToolCall(source_index=1, result_entry_id="result-1", status="planned"),
        ]
    )

    found = tools_module.find_call(batch, 0, "result-0")
    assert found is batch.calls[0]
    assert tools_module.find_call(batch, 0, "result-1") is None
    assert tools_module.find_call(batch, 1, "result-0") is None
    assert tools_module.find_call(batch, 2, "result-0") is None


def test_replace_call_swaps_one_call_and_keeps_the_rest() -> None:
    original = ToolCall(source_index=0, result_entry_id="result-0", status="planned")
    batch = _batch(calls=[original, ToolCall(source_index=1, result_entry_id="result-1")])
    replacement = ToolCall(
        source_index=1, result_entry_id="result-1", status="effect_pending", replay="safe"
    )

    replaced = tools_module.replace_call(batch, replacement)

    assert replaced is not batch
    assert replaced.calls[1] is replacement
    assert replaced.calls[0] is original
    assert batch.calls[1].status == "planned"
    assert replaced.assistant_entry_id == batch.assistant_entry_id
    assert replaced.turn_id == batch.turn_id


def test_current_batch_only_reads_the_tools_leaf() -> None:
    run = _run(_batch())
    owned = tools_module.current_batch(_lane_for(run))
    assert owned is not None
    assert owned.run is run
    assert owned.batch is run.batch

    assert tools_module.current_batch(_lane_for(run, owned=False)) is None
    assert tools_module.current_batch(_lane_for(_run(_batch(), at="checkpoint"))) is None


def test_validate_memo_name_rejects_unaddressable_names() -> None:
    tools_module.validate_memo_name("cache")

    with pytest.raises(TypeError, match="must not be empty"):
        tools_module.validate_memo_name("")
    with pytest.raises(TypeError, match="must not contain ':'"):
        tools_module.validate_memo_name("cache:slot")


# ---------------------------------------------------------------------------
# Synthetic outcomes
# ---------------------------------------------------------------------------


def test_synthetic_message_builds_an_error_tool_result() -> None:
    tool_call = ToolCallBlock(id="call-1", name="bash", arguments={})
    usage = Usage(input=3, output=5, total_tokens=8)

    message = tools_module.synthetic_message(
        tool_call, [TextContent(text="boom")], {"details": {"stage": "before"}, "usage": usage}
    )

    assert message.role == "toolResult"
    assert message.tool_call_id == "call-1"
    assert message.tool_name == "bash"
    assert message.is_error is True
    assert message.timestamp > 0
    assert [block.text for block in message.content] == ["boom"]
    assert message.details == {"stage": "before"}
    assert message.usage is usage

    bare = tools_module.synthetic_message(tool_call, [])
    assert bare.details is None
    assert bare.usage is None
    assert bare.content == []


def test_aborted_outcome_never_terminates() -> None:
    tool_call = ToolCallBlock(id="call-1", name="bash")

    outcome = tools_module.aborted_outcome(tool_call)

    assert outcome.tool_call is tool_call
    assert outcome.terminate is False
    assert [block.text for block in outcome.message.content] == [
        "Tool execution was cancelled before completion."
    ]
    assert outcome.message.is_error is True


def test_interrupted_outcome_keeps_the_checkpoint_and_appends_the_marker() -> None:
    tool_call = ToolCallBlock(id="call-1", name="bash")
    usage = Usage(input=1, output=1, total_tokens=2)
    checkpoint = AgentToolResult(
        content=[TextContent(text="partial output")], details={"lines": 4}, usage=usage
    )

    with_checkpoint = tools_module.interrupted_outcome(tool_call, checkpoint)

    assert [block.text for block in with_checkpoint.message.content] == [
        "partial output",
        tools_module.INTERRUPTION_MARKER,
    ]
    assert with_checkpoint.message.details == {"lines": 4}
    assert with_checkpoint.message.usage is usage
    assert with_checkpoint.terminate is False

    without_checkpoint = tools_module.interrupted_outcome(tool_call, None)

    assert [block.text for block in without_checkpoint.message.content] == [
        tools_module.INTERRUPTION_MARKER
    ]
    assert without_checkpoint.message.details is None


def test_truncated_outcome_names_the_tool_and_terminates_never() -> None:
    outcome = tools_module.truncated_outcome(ToolCallBlock(id="call-1", name="bash"))

    text = outcome.message.content[0].text
    assert text.startswith('Tool call "bash" was not executed')
    assert "output token limit" in text
    assert "Re-issue the tool call with complete arguments." in text
    assert outcome.terminate is False


def test_outcome_from_finalized_call_uses_the_finalized_fields() -> None:
    tool_call = ToolCallBlock(id="call-9", name="read")
    finalized = FinalizedToolCall(
        tool_call=tool_call,
        result=AgentToolResult(content=[TextContent(text="ok")], details={"n": 1}),
        is_error=False,
        terminate=True,
    )

    outcome = tools_module.outcome_from_finalized_call(finalized)

    assert outcome.tool_call is tool_call
    assert outcome.terminate is True
    assert outcome.message.tool_call_id == "call-9"
    assert outcome.message.tool_name == "read"
    assert outcome.message.is_error is False
    assert outcome.message.details == {"n": 1}


# ---------------------------------------------------------------------------
# run_tools dispatch
# ---------------------------------------------------------------------------


async def test_run_tools_dispatches_sequential_batches_with_filtered_tools(
    dispatch: dict,
) -> None:
    tools = [
        AgentHarnessTool(name="read", description=""),
        AgentHarnessTool(name="write", description=""),
        AgentHarnessTool(name="inactive", description=""),
    ]

    async def _tool_context(context: object) -> dict:
        return {"cwd": "/tmp"}

    config = Config(
        tools=tools,
        tool_context=_tool_context,
        tool_execution="sequential",
    )
    run = _run(
        _batch(configuration=LaneConfiguration(active_tool_names=["read"])),
        settings=RunSettings(tool_execution="sequential"),
    )

    result = await tools_module.run_tools(_lane_for(run, config=config), _Drive(), run)

    assert result.kind == "continue"
    assert "sequential" in dispatch
    assert "parallel" not in dispatch
    execution = dispatch["sequential"]["execution"]
    assert [tool.name for tool in execution.tools] == ["read"]
    assert list(execution.tools_by_name) == ["read"]
    assert execution.tool_context == {"cwd": "/tmp"}
    assert dispatch["sequential"]["recovery"] is False
    assert dispatch["sequential"]["sources"] is dispatch["sources"]


async def test_run_tools_dispatches_parallel_batches_by_default(dispatch: dict) -> None:
    tools = [
        AgentHarnessTool(name="read", description=""),
        AgentHarnessTool(name="inactive", description=""),
    ]
    context = {"cwd": "/work"}
    config = Config(tools=tools, tool_context=context)
    run = _run(
        _batch(configuration=LaneConfiguration(active_tool_names=["read", "inactive"]))
    )

    result = await tools_module.run_tools(_lane_for(run, config=config), _Drive(), run)

    assert result.kind == "continue"
    assert "parallel" in dispatch
    assert "sequential" not in dispatch
    assert [tool.name for tool in dispatch["parallel"]["tools"]] == ["read", "inactive"]
    assert set(dispatch["parallel"]["tools_by_name"]) == {"read", "inactive"}
    assert dispatch["parallel"]["tool_context"] is context


async def test_run_tools_announces_recovery_and_passes_the_flag(dispatch: dict) -> None:
    batch = _batch(
        calls=[
            ToolCall(source_index=0, result_entry_id="result-0", status="effect_pending"),
            ToolCall(source_index=1, result_entry_id="result-1", status="outcome_ready"),
        ]
    )
    run = _run(batch)
    lane = _lane_for(run)

    await tools_module.run_tools(lane, _Drive(), run)

    assert lane.emitted == [
        [
            {
                "type": "turn_start",
                "lane": "lane-1",
                "runId": "op-1",
                "turnId": "turn-1",
                "recovery": True,
            }
        ]
    ]
    assert dispatch["parallel"]["recovery"] is True


async def test_run_tools_continues_without_dispatch_when_the_batch_moved_on(
    dispatch: dict,
) -> None:
    run = _run(_batch())
    lane = _lane_for(run, owned=False)

    result = await tools_module.run_tools(lane, _Drive(), run)

    assert result.kind == "continue"
    assert "sequential" not in dispatch
    assert "parallel" not in dispatch
    assert dispatch["materialize"] == 1
    assert lane.emitted == []


async def test_run_tools_settles_cancelled_batches_without_execution_context(
    dispatch: dict,
) -> None:
    run = _run(_batch(), control=Control(status="cancel_requested"))

    result = await tools_module.run_tools(_lane_for(run), _Drive(), run)

    assert result.kind == "continue"
    assert "parallel" not in dispatch
    assert dispatch["sequential"]["execution"] is None


# ---------------------------------------------------------------------------
# Preparation and invocation capability
# ---------------------------------------------------------------------------


async def test_prepare_tool_invocation_truncates_length_limited_responses() -> None:
    call = ToolCall(source_index=0, result_entry_id="result-0", status="planned")
    sources = ToolBatchSource(
        assistant=AssistantMessage(stop_reason="length"),
        calls={0: ToolCallBlock(id="call-1", name="bash")},
    )

    prepared = await tools_module.prepare_tool_invocation(
        _lane_for(_run(_batch())), _Drive(), sources, call, []
    )

    assert prepared.kind == "outcome"
    assert prepared.cleared is None
    assert prepared.outcome.message.content[0].text.startswith(
        'Tool call "bash" was not executed'
    )


async def test_prepare_tool_invocation_reports_unavailable_tools() -> None:
    call = ToolCall(source_index=0, result_entry_id="result-0", status="planned")
    sources = ToolBatchSource(
        assistant=AssistantMessage(stop_reason="toolUse"),
        calls={0: ToolCallBlock(id="call-1", name="bash")},
    )

    prepared = await tools_module.prepare_tool_invocation(
        _lane_for(_run(_batch())), _Drive(), sources, call, []
    )

    assert prepared.kind == "outcome"
    assert prepared.outcome.message.is_error is True
    assert prepared.outcome.message.content[0].text == "Tool 'bash' is unavailable"


async def test_prepare_tool_invocation_rejects_invalid_source_indexes() -> None:
    call = ToolCall(source_index=3, result_entry_id="result-3", status="planned")
    sources = ToolBatchSource(assistant=AssistantMessage(stop_reason="toolUse"), calls={})

    with pytest.raises(SessionInvariantError, match="source index 3 is invalid"):
        await tools_module.prepare_tool_invocation(
            _lane_for(_run(_batch())), _Drive(), sources, call, []
        )


async def test_invocation_memo_reads_and_writes_only_while_effect_pending() -> None:
    address = operation_tool_memo("op-1", "result-0", "cache")
    call = ToolCall(source_index=0, result_entry_id="result-0", status="effect_pending")
    batch = _batch(calls=[call])
    lane = _CommandLane(
        _lane_for(_run(batch)).state,
        Config(),
        {("pi.op.tool_memo", "op-1:result-0:cache"): StoredValue(address=address, value={"hits": 2})},
    )

    capability = tools_module.invocation_capability(lane, _Drive(), batch, call)

    assert capability.invocation.invocation_id == "result-0"
    assert capability.invocation.operation_id == "op-1"
    assert capability.invocation.turn_id == "turn-1"
    assert await capability.invocation.get_memo("cache") == {"hits": 2}

    await capability.invocation.set_memo("cache", None)
    await capability.invocation.set_memo("cache", {"hits": 3})

    assert [write.op for write in lane.committed[0].writes] == ["delete"]
    assert lane.committed[1].writes[0].value == {"hits": 3}
    assert lane.committed[1].next is lane.state
    assert lane.committed[0].materialize(None) is None

    capability.expire()
    with pytest.raises(tools_module.ToolInvocationEnded, match="no longer owns"):
        await capability.invocation.get_memo("cache")


async def test_invocation_memo_rejects_calls_that_lost_their_effect() -> None:
    call = ToolCall(source_index=0, result_entry_id="result-0", status="effect_pending")
    batch = _batch(calls=[call])
    run = _run(batch)
    state = _lane_for(run).state
    lane = _CommandLane(state, Config())

    capability = tools_module.invocation_capability(lane, _Drive(), batch, call)
    batch.calls = [ToolCall(source_index=0, result_entry_id="result-0", status="placed")]

    with pytest.raises(tools_module.ToolInvocationEnded, match="no longer owns"):
        await capability.invocation.get_memo("cache")
    with pytest.raises(tools_module.ToolInvocationEnded, match="no longer owns"):
        await capability.invocation.set_memo("cache", {"hits": 1})
    assert lane.committed == []

