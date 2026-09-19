"""Round-trip fidelity for every durable value the harness writes to JSONL.

The durable value store keeps live dataclasses in-process but yields plain
camelCase mappings after a JSONL replay. These tests write through the real
serializer, read back through the real revivers, and assert the restored value
is the object the runtime expects to use — catching the whole class of bug where
a replayed mapping is later read as if it were still a dataclass.
"""

from __future__ import annotations

import json

import pytest

from pi_agent_core.harness.session.jsonl.io import to_json_value
from pi_agent_core.harness.session.types import (
    AssistantEffectPendingOperation,
    AssistantReadyOperation,
    AssistantRetryWaitOperation,
    CheckpointData,
    CheckpointOperation,
    Continuation,
    Control,
    DeferredEffectPendingOperation,
    DeferredSuspendedOperation,
    GenerationContext,
    LaneConfiguration,
    NavigationReadyToCommitOperation,
    NormalizedRetryPolicy,
    Operation,
    OperationIntent,
    OperationMeta,
    PendingEntry,
    ResultBoundary,
    RunSettings,
    StartingOperation,
    SummaryContext,
    SummaryDecidingOperation,
    SummaryEffectPendingOperation,
    SummaryReadyOperation,
    SummaryRetryWaitOperation,
    SummaryTask,
    ToolBatch,
    ToolCall,
    ToolsOperation,
    UsageRow,
    revive_operation_meta,
    revive_operation_state,
)

IDENTITY = LaneConfiguration(
    model={"provider": "faux", "modelId": "faux-1"}, thinking_level="low", active_tool_names=["bash"]
)
SETTINGS = RunSettings(
    compaction={"enabled": True},
    steering_mode="all",
    follow_up_mode="one-at-a-time",
    tool_execution="sequential",
)
RETRY = NormalizedRetryPolicy(max_attempts=3, base_delay_ms=10, max_agent_delay_ms=1000)
SCOPE = {
    "control": Control(status="cancel_requested", requested_at=42),
    "settings": SETTINGS,
    "latest_assistant_entry_id": "e-latest",
}


def _replay(value: object) -> object:
    """Round-trip a durable value through the real JSONL wire path."""
    return json.loads(json.dumps(to_json_value(value)))


def _all_leaves() -> list:
    task = SummaryTask(
        task_id="t1",
        reason="threshold",
        custom_instructions="be terse",
        boundary=ResultBoundary(kind="finish"),
    )
    summary_context = SummaryContext(
        result_entry_id="r1",
        configuration=IDENTITY,
        stream_options={"deferred": False},
        retry_policy=RETRY,
    )
    navigation_task = SummaryTask(
        task_id="t2",
        boundary=ResultBoundary(kind="commit_navigation", target_id="e-target", label="L"),
    )
    batch = ToolBatch(
        assistant_entry_id="a1",
        configuration=IDENTITY,
        turn_id="turn-1",
        calls=[
            ToolCall(source_index=0, result_entry_id="r0", status="planned"),
            ToolCall(
                source_index=1, result_entry_id="r1", status="outcome_ready", terminate=True
            ),
        ],
    )
    generation = GenerationContext(
        step_id="s1",
        trigger_entry_id="t-e",
        configuration=IDENTITY,
        stream_options={"transport": "sse"},
        retry_policy=RETRY,
        overflow_recovery_used=True,
    )
    return [
        StartingOperation(**SCOPE),
        CheckpointOperation(
            **SCOPE,
            continuation=Continuation(kind="need_assistant", overflow_recovery_used=True),
            trigger_entry_id="t-e",
        ),
        AssistantReadyOperation(**SCOPE, generation_context=generation, next_attempt=2),
        AssistantEffectPendingOperation(
            **SCOPE,
            generation_context=generation,
            attempt=2,
            response_entry_id="r-e",
            usage_id="u-e",
            intended_output_limit=4096,
            context_window=200000,
        ),
        AssistantRetryWaitOperation(
            **SCOPE,
            generation_context=generation,
            next_attempt=3,
            not_before=1780000000000,
            error_message="overloaded",
        ),
        ToolsOperation(**SCOPE, batch=batch),
        DeferredSuspendedOperation(
            **SCOPE,
            step_id="s3",
            source_entry_id="r-e",
            poll=4,
            configuration=IDENTITY,
            stream_options={"deferred": True},
        ),
        DeferredEffectPendingOperation(
            **SCOPE,
            step_id="s3",
            source_entry_id="r-e",
            poll=4,
            configuration=IDENTITY,
            stream_options={"deferred": True},
            response_entry_id="r-e2",
            usage_id="u-e2",
        ),
        SummaryDecidingOperation(**SCOPE, task=task),
        SummaryReadyOperation(
            **SCOPE, task=task, summary_context=summary_context, next_attempt=1
        ),
        SummaryEffectPendingOperation(
            **SCOPE,
            task=task,
            summary_context=summary_context,
            attempt=1,
            request={"index": 0, "usageId": "u9"},
            usage_ids=["u8"],
        ),
        SummaryRetryWaitOperation(
            **SCOPE,
            task=task,
            summary_context=summary_context,
            next_attempt=2,
            not_before=5,
            error_message="rate limited",
        ),
        NavigationReadyToCommitOperation(**SCOPE, target_id="e-target", label="L"),
        NavigationReadyToCommitOperation(**SCOPE, target_id=None, label=None),
    ]


@pytest.mark.parametrize("leaf", _all_leaves(), ids=lambda leaf: leaf.at)
def test_operation_state_survives_a_jsonl_replay(leaf):
    """Every leaf revives into itself, with its nested configuration typed."""
    revived = revive_operation_state(_replay(leaf))
    assert type(revived) is type(leaf)
    assert revived.at == leaf.at
    assert revived.control.status == "cancel_requested"
    assert revived.control.requested_at == 42
    assert revived.settings.tool_execution == "sequential"
    assert revived.latest_assistant_entry_id == "e-latest"
    assert revived == leaf


#: Leaves whose durable record embeds a full lane configuration.
_BEARING_CONFIGURATION = frozenset(
    {
        "assistant.ready",
        "assistant.effect_pending",
        "assistant.retry_wait",
        "tools",
        "deferred.suspended",
        "deferred.effect_pending",
        "summary.ready",
        "summary.effect_pending",
        "summary.retry_wait",
    }
)


@pytest.mark.parametrize(
    "leaf",
    [leaf for leaf in _all_leaves() if leaf.at in _BEARING_CONFIGURATION],
    ids=lambda leaf: leaf.at,
)
def test_replayed_nested_configuration_is_a_dataclass(leaf):
    """A replayed nested lane configuration is usable as an attribute holder."""
    revived = _replay_and_revive(leaf)

    def configurations(value):
        yield getattr(value, "configuration", None)
        generation_context = getattr(value, "generation_context", None)
        if generation_context is not None:
            yield generation_context.configuration
        summary_context = getattr(value, "summary_context", None)
        if summary_context is not None:
            yield summary_context.configuration
        batch = getattr(value, "batch", None)
        if batch is not None:
            yield batch.configuration

    found = [item for item in configurations(revived) if item is not None]
    assert found, f"{leaf.at} carries no configuration to check"
    for configuration in found:
        assert isinstance(configuration, LaneConfiguration)
        assert configuration.model["provider"] == "faux"
        assert configuration.model["modelId"] == "faux-1"
        assert configuration.thinking_level == "low"
        assert configuration.active_tool_names == ["bash"]


def test_operation_meta_survives_a_jsonl_replay():
    """Metadata revives with a typed intent, including the null tip and root target."""
    meta = OperationMeta(
        operation_id="op1",
        lane="main",
        source_tip_id=None,
        started_at=7,
        intent=OperationIntent(
            kind="navigation", target_id=None, summarize=True, label="L", custom_instructions="c"
        ),
    )
    revived = revive_operation_meta(_replay(meta))
    assert isinstance(revived, OperationMeta)
    assert revived == meta
    assert revived.intent.kind == "navigation"
    assert revived.intent.target_id is None
    assert revived.intent.summarize is True
    assert revived.source_tip_id is None


def test_operation_groups_meta_and_state():
    """The `Operation` record pairs the two durable halves."""
    meta = OperationMeta(operation_id="op1", lane="main", started_at=1)
    operation = Operation(meta=meta, state=StartingOperation(**SCOPE))
    assert operation.meta is meta
    assert operation.state.at == "starting"


def test_pending_entry_survives_a_jsonl_replay():
    """A queued message payload revives into the record the readers expect."""
    entry = PendingEntry(type="message", payload={"role": "user", "content": "hi"})
    revived = PendingEntry.from_value(_replay(entry))
    assert revived.type == "message"
    assert revived.payload == {"role": "user", "content": "hi"}
    assert revived.custom_type is None

    custom = PendingEntry(type="custom", custom_type="note", payload=[1, 2])
    revived_custom = PendingEntry.from_value(_replay(custom))
    assert revived_custom.custom_type == "note"
    assert revived_custom.payload == [1, 2]


def test_lane_configuration_survives_a_jsonl_replay():
    """A lane configuration revives with its active tool list intact."""
    revived = LaneConfiguration.from_value(_replay(IDENTITY))
    assert revived == IDENTITY
    assert revived.model == {"provider": "faux", "modelId": "faux-1"}


def test_usage_row_wire_shape_is_flat_and_camel_case():
    """A usage row serializes with the exact keys the TypeScript writer emits."""
    from pi_ai.types import Usage

    row = UsageRow(
        id="u1", usage=Usage(input=3, output=4, total_tokens=7), entry_id="e1", seq=9
    )
    wire = to_json_value(row)
    assert set(wire) == {"id", "seq", "usage", "adjustment", "entryId"}
    assert wire["entryId"] == "e1"
    assert wire["seq"] == 9
    assert wire["adjustment"] is False
    assert "totalTokens" in wire["usage"]


def _replay_and_revive(leaf):
    return revive_operation_state(_replay(leaf))


# ---------------------------------------------------------------------------
# Reading files written by the TypeScript implementation
#
# The TypeScript writer emits `JSON.stringify(committedWrite)`, where a committed
# entry/usage write is the record itself plus a `kind` tag — flat, not nested
# under an `entry`/`row` key. These fixtures are literal TypeScript output, typed
# out by hand from the TS type definitions in `session/commit.ts`, so a change to
# the reader's expected shape fails here rather than in the field.
# ---------------------------------------------------------------------------

TS_ENTRY_WRITE = (
    '{"kind":"entry","type":"message","id":"e1","parentId":null,"seq":1,'
    '"timestamp":1700,"message":{"role":"user","content":"hello from TS"}}'
)
TS_USAGE_WRITE = (
    '{"kind":"usage","seq":2,"id":"u1","usage":{"input":10,"output":5,'
    '"cacheRead":0,"cacheWrite":0,"totalTokens":15,"cost":{"input":0,"output":0,'
    '"cacheRead":0,"cacheWrite":0,"total":0}},"entryId":"e1","adjustment":false}'
)
TS_VALUE_SET_WRITE = (
    '{"kind":"value","seq":3,"op":"set","namespace":"lane.config","key":"main",'
    '"value":{"model":{"provider":"faux","modelId":"faux-1"},"thinkingLevel":"low",'
    '"activeToolNames":["bash"]}}'
)
TS_VALUE_DELETE_WRITE = (
    '{"kind":"value","seq":4,"op":"delete","namespace":"pending","key":"e9"}'
)
TS_OPERATION_STATE_WRITE = (
    '{"kind":"value","seq":5,"op":"set","namespace":"op.state","key":"op1",'
    '"value":{"at":"assistant.retry_wait","control":{"status":"running"},'
    '"settings":{"compaction":null,"steeringMode":"all","followUpMode":"all",'
    '"toolExecution":"parallel"},"latestAssistantEntryId":"e1",'
    '"generationContext":{"stepId":"s1","triggerEntryId":"e1",'
    '"configuration":{"model":{"provider":"faux","modelId":"faux-1"},'
    '"thinkingLevel":"low","activeToolNames":[]},"streamOptions":null,'
    '"retryPolicy":{"maxAttempts":3,"baseDelayMs":10,"maxAgentDelayMs":60},'
    '"overflowRecoveryUsed":false},"nextAttempt":2,"notBefore":1750,'
    '"errorMessage":"overloaded"}}'
)


def test_reads_a_flat_entry_write_written_by_typescript():
    """A TypeScript entry line parses into the entry object, not a nested wrapper."""
    from pi_agent_core.harness.session.jsonl.io import parse_jsonl_transaction

    writes = parse_jsonl_transaction(TS_ENTRY_WRITE)
    assert len(writes) == 1
    write = writes[0]
    assert write.kind == "entry"
    assert write.seq == 1
    assert write.timestamp == 1700
    assert write.id == "e1"
    assert write.parent_id is None
    assert write.entry.type == "message"
    assert write.entry.id == "e1"
    assert write.entry.message.role == "user"
    assert write.entry.message.content == "hello from TS"


def test_reads_a_flat_usage_write_written_by_typescript():
    """A TypeScript usage line parses into a UsageRow with its entry link."""
    from pi_agent_core.harness.session.jsonl.io import parse_jsonl_transaction

    write = parse_jsonl_transaction(TS_USAGE_WRITE)[0]
    assert write.kind == "usage"
    assert write.seq == 2
    row = write.entry
    assert row.id == "u1"
    assert row.entry_id == "e1"
    assert row.adjustment is False
    assert row.usage.input == 10
    assert row.usage.output == 5


def test_reads_value_writes_written_by_typescript():
    """Set and delete value lines parse with the same key presence rules."""
    from pi_agent_core.harness.session.jsonl.io import parse_jsonl_transaction

    set_write = parse_jsonl_transaction(TS_VALUE_SET_WRITE)[0]
    assert (set_write.op, set_write.namespace, set_write.key) == (
        "set",
        "lane.config",
        "main",
    )
    configuration = LaneConfiguration.from_value(set_write.value)
    assert configuration.model == {"provider": "faux", "modelId": "faux-1"}
    assert configuration.thinking_level == "low"
    assert configuration.active_tool_names == ["bash"]

    delete_write = parse_jsonl_transaction(TS_VALUE_DELETE_WRITE)[0]
    assert (delete_write.op, delete_write.namespace, delete_write.key) == (
        "delete",
        "pending",
        "e9",
    )


def test_reads_a_durable_operation_state_written_by_typescript():
    """A TypeScript operation-state line revives into the live leaf dataclass."""
    from pi_agent_core.harness.session.jsonl.io import parse_jsonl_transaction

    write = parse_jsonl_transaction(TS_OPERATION_STATE_WRITE)[0]
    revived = revive_operation_state(write.value)
    assert revived.at == "assistant.retry_wait"
    assert revived.next_attempt == 2
    assert revived.not_before == 1750
    assert revived.error_message == "overloaded"
    assert revived.latest_assistant_entry_id == "e1"
    assert revived.settings.steering_mode == "all"
    assert revived.generation_context.step_id == "s1"
    assert revived.generation_context.retry_policy.max_attempts == 3
    assert isinstance(revived.generation_context.configuration, LaneConfiguration)
    assert revived.generation_context.configuration.model["modelId"] == "faux-1"


def test_a_multi_write_transaction_line_written_by_typescript():
    """A batched TypeScript transaction line parses as an ordered write list."""
    from pi_agent_core.harness.session.jsonl.io import parse_jsonl_transaction

    line = "[" + ",".join(
        [TS_ENTRY_WRITE, TS_USAGE_WRITE, TS_VALUE_DELETE_WRITE]
    ) + "]"
    writes = parse_jsonl_transaction(line)
    assert [write.kind for write in writes] == ["entry", "usage", "value"]
    assert [write.seq for write in writes] == [1, 2, 4]


def test_committed_write_wire_shapes_match_the_typescript_writers():
    """Every committed-write kind emits exactly the TypeScript key set.

    TypeScript declares these as intersections (`Entry & { kind }`,
    `UsageRow & { kind }`) or closed interfaces, and serializes with
    `JSON.stringify`, so key presence is part of the format: a delete has no
    payload key at all while a set/append always carries one.
    """
    from pi_agent_core.harness.session.commit import CommittedWrite, commit_write
    from pi_agent_core.harness.session.types import (
        EntryWrite,
        MessageEntry,
        UsageRow,
        UsageWrite,
    )
    from pi_agent_core.harness.session.values import (
        ListAppendWrite,
        ListDeleteWrite,
        ValueDeleteWrite,
        ValueSetWrite,
    )

    message = MessageEntry(id="e1", parent_id=None, message={"role": "user", "content": "hi"})

    cases = [
        (EntryWrite(entry=message), {"kind", "type", "id", "parentId", "seq", "timestamp", "message"}),
        (
            UsageWrite(
                row=UsageRow(id="u1", usage=_usage(), entry_id="e1", adjustment=False)
            ),
            {"kind", "seq", "id", "usage", "adjustment", "entryId"},
        ),
        (ValueSetWrite(namespace="n", key="k", value={"a": 1}), {"kind", "seq", "op", "namespace", "key", "value"}),
        (ValueDeleteWrite(namespace="n", key="k"), {"kind", "seq", "op", "namespace", "key"}),
        (ListAppendWrite(namespace="n", key="k", value="item"), {"kind", "seq", "op", "namespace", "key", "value"}),
        (ListDeleteWrite(namespace="n", key="k"), {"kind", "seq", "op", "namespace", "key"}),
    ]
    from pi_agent_core.harness.session.jsonl.io import serialize_jsonl_transaction

    seen: set = set()
    for write, expected_keys in cases:
        # Serialize through the real transaction writer, then read the bytes back.
        committed = commit_write(write, 7, 1700)
        wire = json.loads(serialize_jsonl_transaction([committed]))
        assert set(wire) == expected_keys, (write.kind, write.op, set(wire) ^ expected_keys)
        assert wire["seq"] == 7
        assert wire["kind"] in ("entry", "usage", "value", "list")
        seen.add((wire["kind"], wire.get("op")))
    # `value set/delete` and `list append/delete` are four distinct kinds, not one.
    assert seen == {
        ("entry", None),
        ("usage", None),
        ("value", "set"),
        ("value", "delete"),
        ("list", "append"),
        ("list", "delete"),
    }


def _usage():
    from pi_ai.types import Usage

    return Usage(input=1, output=2, total_tokens=3)
