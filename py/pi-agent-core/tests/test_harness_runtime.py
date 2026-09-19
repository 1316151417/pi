"""Tests for the harness event bus, lane restoration, and AgentHarness surface."""

from __future__ import annotations

import asyncio

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core.harness.agent_harness import (
    AcquireLaneOptions,
    AgentHarnessNamespace,
    AgentHarnessOptions,
    NavigateOptions,
    OperationRequest,
    create_agent_harness,
)
from pi_agent_core.harness.events import HarnessEvent, HarnessEventBus
from pi_agent_core.harness.result import HarnessClosed
from pi_agent_core.harness.runtime.restore import (
    ClassifiedLaneStorage,
    read_lane_storage,
    restore_lane,
    restore_lane_state,
    restore_session,
    state_matches_intent,
)
from pi_agent_core.harness.session.session import SessionInvariantError
from pi_agent_core.harness.session.values import (
    branch_tip,
    lane_config,
    lane_state,
    operation_meta,
    operation_state,
    set_value,
)


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


async def test_event_bus_delivers_in_order_to_typed_listeners():
    bus = HarnessEventBus()
    received = []
    bus.on("run_start", lambda event, ctx: received.append(("run", event.run_id)))
    bus.on("turn_start", lambda event, ctx: received.append(("turn", event.turn_id)))

    await bus.emit(HarnessEvent(type="run_start", lane="main", run_id="r1"), BACKGROUND_CONTEXT)
    await bus.emit(HarnessEvent(type="turn_start", lane="main", run_id="r1", turn_id="t1"), BACKGROUND_CONTEXT)
    await bus.emit_batch(
        [
            HarnessEvent(type="message_end", lane="main", run_id="r1"),
            HarnessEvent(type="run_start", lane="main", run_id="r2"),
        ],
        BACKGROUND_CONTEXT,
    )

    assert received == [("run", "r1"), ("turn", "t1"), ("run", "r2")]


async def test_event_bus_isolates_handler_failures():
    bus = HarnessEventBus()
    errors = []
    delivered = []

    def _boom(event, ctx):
        raise RuntimeError("handler blew up")

    bus.on("run_start", _boom)
    bus.on("run_start", lambda event, ctx: delivered.append(event.run_id))
    bus.on("handler_error", lambda event, ctx: errors.append(event.error))

    await bus.emit(HarnessEvent(type="run_start", lane="main", run_id="r1"), BACKGROUND_CONTEXT)

    assert delivered == ["r1"]  # later listeners still receive the event
    assert errors == ["handler blew up"]


async def test_event_bus_unsubscribe_and_close():
    bus = HarnessEventBus()
    received = []
    unsubscribe = bus.on("run_start", lambda event, ctx: received.append(event.run_id))
    await bus.emit(HarnessEvent(type="run_start", run_id="r1"), BACKGROUND_CONTEXT)
    unsubscribe()
    await bus.emit(HarnessEvent(type="run_start", run_id="r2"), BACKGROUND_CONTEXT)
    assert received == ["r1"]

    bus.close(HarnessClosed())
    with pytest.raises(HarnessClosed):
        bus.on("run_start", lambda event, ctx: None)


async def test_watch_buffers_before_start_then_replays():
    bus = HarnessEventBus()
    watcher = bus.watch(None, lambda event: getattr(event, "lane", None) == "main", BACKGROUND_CONTEXT)

    await bus.emit(HarnessEvent(type="run_start", lane="main", run_id="r1"), BACKGROUND_CONTEXT)
    await bus.emit(HarnessEvent(type="run_start", lane="other", run_id="rX"), BACKGROUND_CONTEXT)

    received = []

    async def _listener(event, ctx):
        received.append(event.run_id)

    watcher.start(_listener)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert received == ["r1"]  # buffered pre-start event replayed, other lane filtered out

    watcher.unsubscribe()
    await bus.emit(HarnessEvent(type="run_start", lane="main", run_id="r2"), BACKGROUND_CONTEXT)
    await asyncio.sleep(0)
    assert received == ["r1"]


async def test_watch_resnapshot_holds_events_until_boundary():
    bus = HarnessEventBus()

    async def _resnapshot(context, mark_boundary):
        mark_boundary()
        return "fresh"

    watcher = bus.watch("initial", lambda event: True, BACKGROUND_CONTEXT, _resnapshot)

    async def _listener(event, ctx):
        pass

    watcher.start(_listener)
    assert await watcher.resnapshot(BACKGROUND_CONTEXT) == "fresh"
    assert watcher.snapshot == "fresh"


def test_watch_start_only_once():
    bus = HarnessEventBus()
    watcher = bus.watch(None, lambda event: True, BACKGROUND_CONTEXT)
    watcher.start(lambda event, ctx: None)
    with pytest.raises(RuntimeError, match="may be called only once"):
        watcher.start(lambda event, ctx: None)


# ---------------------------------------------------------------------------
# Lane restoration
# ---------------------------------------------------------------------------


def test_state_matches_intent_for_run_and_compaction():
    from pi_agent_core.harness.session.types import OperationError  # noqa: F401

    class _Boundary:
        kind = "resume_checkpoint"

    class _Task:
        boundary = _Boundary()
        custom_instructions = None

    class _State:
        at = "assistant.generating"
        task = _Task()

    class _Intent:
        kind = "run"

    assert state_matches_intent(_Intent(), _State()) is True

    class _SummaryState:
        at = "summary.finishing"
        task = _Task()

    class _CompactionIntent:
        kind = "compaction"

    _SummaryState.task.boundary.kind = "finish"
    assert state_matches_intent(_CompactionIntent(), _SummaryState()) is True


async def test_restore_session_and_lane(workdir):
    from pi_agent_core.harness.session import MemorySessionRepo, SessionCreateOptions

    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="s1"), BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT

    # Absent lane storage restores to an empty map.
    assert await restore_session(session, context) == {}

    configuration = {"model": {"provider": "faux", "modelId": "faux-1"}, "thinkingLevel": "off", "activeToolNames": []}
    lane_state_value = {"currentOperationId": None, "lastOperationId": None, "inbox": []}

    async def _configure(mutator, mutation_context):
        await mutator.commit(
            [
                set_value(branch_tip("main"), None),
                set_value(lane_config("main"), configuration),
                set_value(lane_state("main"), lane_state_value),
            ],
            mutation_context,
        )

    await session.mutate(_configure, context)

    restored = await restore_session(session, context)
    assert list(restored) == ["main"]
    assert restored["main"].tip_id is None
    config = restored["main"].configuration
    assert config.model == configuration["model"]
    assert config.thinking_level == configuration["thinkingLevel"]
    assert config.active_tool_names == configuration["activeToolNames"]
    assert restored["main"].operation is None

    branch_state = await restore_lane(session, "main", context)
    assert branch_state.tip_id is None


async def test_restore_lane_rejects_incomplete_storage(workdir):
    from pi_agent_core.harness.session import MemorySessionRepo, SessionCreateOptions

    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="s1"), BACKGROUND_CONTEXT)
    context = BACKGROUND_CONTEXT

    async def _configure(mutator, mutation_context):
        await mutator.commit([set_value(branch_tip("partial"), None)], mutation_context)

    await session.mutate(_configure, context)

    with pytest.raises(SessionInvariantError, match="missing lane.config"):
        await restore_lane(session, "partial", context)

    with pytest.raises(SessionInvariantError, match="missing branch.tip"):
        await restore_lane(session, "unknown", context)


async def test_restore_lane_state_validates_operation_identity(workdir):
    class _Metadata:
        def __init__(self, operation_id: str, lane: str) -> None:
            self.operation_id = operation_id
            self.lane = lane
            self.intent = _Intent()

    class _Intent:
        kind = "run"

    class _Value:
        def __init__(self, value: object) -> None:
            self.value = value

    class _State:
        at = "assistant.generating"
        task = None

    class _Stored:
        def __init__(self, value: object) -> None:
            self.value = value

    class _Reader:
        def __init__(self, meta, state) -> None:
            self._meta = meta
            self._state = state

        async def get_value(self, address, context):
            if address.namespace == "pi.op.meta":
                return self._meta
            if address.namespace == "pi.op.state":
                return self._state
            return None

    stored = ClassifiedLaneStorage(
        kind="lane",
        tip=_Stored(None),
        configuration=_Stored({"model": {}, "thinkingLevel": "off", "activeToolNames": []}),
        lane_state=_Stored(type("S", (), {"current_operation_id": "op-1", "last_operation_id": None, "inbox": []})()),
    )

    # Operation belongs to another lane -> invariant error.
    reader = _Reader(_Value(_Metadata("op-1", "other")), _Value(_State()))
    with pytest.raises(SessionInvariantError, match="belongs to lane"):
        await restore_lane_state(reader, "main", stored, BACKGROUND_CONTEXT)

    # Metadata names a different operation -> invariant error.
    reader = _Reader(_Value(_Metadata("op-2", "main")), _Value(_State()))
    with pytest.raises(SessionInvariantError, match="metadata names operation"):
        await restore_lane_state(reader, "main", stored, BACKGROUND_CONTEXT)

    # Missing state -> invariant error.
    reader = _Reader(_Value(_Metadata("op-1", "main")), None)
    with pytest.raises(SessionInvariantError, match="missing op.state"):
        await restore_lane_state(reader, "main", stored, BACKGROUND_CONTEXT)


# ---------------------------------------------------------------------------
# AgentHarness surface
# ---------------------------------------------------------------------------


def test_agent_harness_option_defaults():
    options = AgentHarnessOptions()
    assert options.tools == []
    assert options.steering_mode == "all"
    assert options.follow_up_mode == "all"
    assert options.tool_execution == "parallel"
    assert options.entry_projectors == {}


def test_operation_request_and_navigate_options():
    request = OperationRequest(kind="navigation", target_id="entry-1", options=NavigateOptions(summarize=True))
    assert request.kind == "navigation"
    assert request.options.summarize is True
    assert AcquireLaneOptions(create_at="tip-1").create_at == "tip-1"


async def test_create_agent_harness_wires_session_and_global_config():
    """createAgentHarness attaches runtime without starting any effects."""
    from pi_agent_core._pi_ai.models import Models
    from pi_agent_core._pi_ai.types import Model
    from pi_agent_core.harness.session import MemorySessionRepo, SessionCreateOptions

    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="s1"), BACKGROUND_CONTEXT)
    harness = await create_agent_harness(
        AgentHarnessOptions(
            session=session,
            models=Models(),
            model=Model(id="m1", name="m1", provider="faux", api="faux", base_url="", context_window=1000, max_tokens=100),
            thinking_level="low",
            active_tool_names=["bash"],
        ),
        BACKGROUND_CONTEXT,
    )
    assert harness["open"] == []
    runtime = harness["harness"]
    assert await runtime.get_name(BACKGROUND_CONTEXT) is None
    await runtime.set_name("named", BACKGROUND_CONTEXT)
    assert await runtime.get_name(BACKGROUND_CONTEXT) == "named"
    assert await runtime.get_tools(BACKGROUND_CONTEXT) == []
    assert await runtime.get_steering_mode(BACKGROUND_CONTEXT) == "all"
    await runtime.set_steering_mode("one-at-a-time", BACKGROUND_CONTEXT)
    assert await runtime.get_steering_mode(BACKGROUND_CONTEXT) == "one-at-a-time"
    await runtime.close(BACKGROUND_CONTEXT)


async def test_lane_creation_is_durable_and_idempotent():
    """Lane creation writes the seed configuration exactly once."""
    from pi_agent_core._pi_ai.models import Models
    from pi_agent_core._pi_ai.types import Model
    from pi_agent_core.harness.session import MemorySessionRepo, SessionCreateOptions

    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="s1"), BACKGROUND_CONTEXT)
    result = await create_agent_harness(
        AgentHarnessOptions(
            session=session,
            models=Models(),
            model=Model(id="m1", name="m1", provider="faux", api="faux", base_url="", context_window=1000, max_tokens=100),
            thinking_level="low",
        ),
        BACKGROUND_CONTEXT,
    )
    runtime = result["harness"]
    lane = await runtime.lane("main", BACKGROUND_CONTEXT)
    assert lane.name == "main"
    assert await lane.get_tip_id(BACKGROUND_CONTEXT) is None
    # The second acquisition returns the same in-process lane, not a new one.
    assert await runtime.lane("main", BACKGROUND_CONTEXT) is lane
    created = await runtime.lanes(BACKGROUND_CONTEXT)
    assert [info.name for info in created] == ["main"]
    with pytest.raises(Exception):
        await runtime.lane("", BACKGROUND_CONTEXT)
    await runtime.close(BACKGROUND_CONTEXT)


def test_agent_harness_namespace_exposes_create():
    assert callable(AgentHarnessNamespace.create)
