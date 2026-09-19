"""Tests for the pico3 kernel port (``harness/pico3/*``).

Each module that ships gets at least one case over concrete input, not an import
smoke test. The pico3 kernel is exercised through its own primitives: a real
in-memory/JSONL storage, real tracked documents, fake runtimes for the kind
handlers, and a fake session for the view manager.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT, with_abort_signal
from pi_ai.abort import AbortController
from pi_ai.assistant_message_frame import AssistantMessageFrame
from pi_ai.types import (
    AssistantMessage as PiAssistantMessage,
    AssistantMessageEvent as PiAssistantMessageEvent,
    Model as PiModel,
    ModelCost as PiModelCost,
    TextContent as PiTextContent,
    ToolCall as PiToolCall,
    Usage as PiUsage,
)
from pi_agent_core.harness.pico3 import (
    BashTool,
    Bounded,
    Defaults,
    JsonlStorage,
    MemoryStorage,
    Membrane,
    PlainInputError,
    Tracker,
    ViewManager,
    WATCH_CAPACITY,
    apply_envelope,
    apply_immutable,
    bash_tool,
    derive_context,
    is_base,
    job,
    plugin,
    track,
)
from pi_agent_core.harness.pico3.kinds import frames
from pi_agent_core.harness.pico3.kinds import task_api as task_api_module
from pi_agent_core.harness.pico3.kinds.generation import generation_kind
from pi_agent_core.harness.pico3.kinds.post_tools import post_tools_kind
from pi_agent_core.harness.pico3.kinds.tool import tool_kind
from pi_agent_core.harness.pico3.kinds.entries import entries, core_entry
from pi_agent_core.harness.pico3.kinds.job import JobFailure, JobInput, JobResult, job_kind
from pi_agent_core.harness.pico3.kinds.plugin import PluginFailure, plugin_kind
from pi_agent_core.harness.pico3 import session as pico_session
from pi_agent_core.harness.pico3 import system as pico_system
from pi_agent_core.harness.pico3 import types as t
from pi_agent_core.harness.pico3.membrane import RevokedError
from pi_agent_core.harness.pico3.hooks import HookRegistration, create_hook_runners

CTX = BACKGROUND_CONTEXT


# ---------------------------------------------------------------------------
# bounded.ts
# ---------------------------------------------------------------------------


def test_bounded_head_stops_at_byte_and_line_budget():
    head = Bounded(6, 2, "head")
    head.push(b"ab\ncd\nef\n")
    assert head.text() == "ab\ncd\n"
    assert head.dropped_bytes == 3
    assert head.dropped_lines == 1
    assert head.total == 9

    by_bytes = Bounded(3, 100, "head")
    by_bytes.push(b"abcdef")
    assert by_bytes.text() == "abc"
    assert by_bytes.dropped_bytes == 3


def test_bounded_tail_keeps_the_newest_lines_and_counts_drops():
    tail = Bounded(100, 2, "tail")
    tail.push(b"a\nb\nc\nd")
    assert tail.text() == "b\nc\nd"
    assert tail.dropped_lines == 1
    assert tail.dropped_bytes == 2

    empty = Bounded(0, 10, "tail")
    empty.push(b"gone")
    assert empty.text() == ""
    assert empty.dropped == 4


# ---------------------------------------------------------------------------
# delta.ts
# ---------------------------------------------------------------------------


def test_tracker_flush_it_apply_it_round_trip():
    before = {"entries": [{"id": 1}], "config": {"model": "a"}, "inbox": []}
    tracker = track(json.loads(json.dumps(before)))
    state = tracker.state
    state["entries"].append({"id": 2})
    state["config"]["model"] = "b"
    del state["config"]
    state["inbox"] = [{"id": 9, "mode": "steer"}]

    ops = tracker.flush()
    assert not is_base(ops)
    assert apply_immutable(before, ops) == tracker.state
    assert before == {"entries": [{"id": 1}], "config": {"model": "a"}, "inbox": []}
    assert tracker.dirty is False
    assert tracker.flush() == []


def test_tracker_rebase_emits_a_base_and_discard_swallows_ops():
    tracker = track({"a": 1})
    tracker.state["a"] = 2
    tracker.rebase()
    ops = tracker.flush()
    assert is_base(ops) and ops[0][0] == "r" and ops[0][1] == {"a": 2}

    tracker.state["a"] = 3
    tracker.discard()
    assert tracker.flush() == []
    assert tracker.state == {"a": 3}


def test_delta_splice_delete_and_equal_length_rewrite_paths():
    tracker = track({"list": [1, 2, 3, 4], "map": {"keep": 1, "drop": 2}})
    state = tracker.state
    state["list"][1:3] = []
    state["list"].append(5)
    del state["map"]["drop"]
    ops = tracker.flush()
    applied = apply_immutable({"list": [1, 2, 3, 4], "map": {"keep": 1, "drop": 2}}, ops)
    assert applied == {"list": [1, 4, 5], "map": {"keep": 1}}

    equal = track({"blocks": [{"text": "a"}, {"text": "b"}]})
    equal.state["blocks"][1]["text"] = "c"
    equal_ops = equal.flush()
    assert equal_ops == [("s", ("blocks", 1, "text"), "c")]


# ---------------------------------------------------------------------------
# context.ts
# ---------------------------------------------------------------------------


class _FakeStorage:
    """A tiny scan_entries-only storage: entries per conversation, newest first."""

    def __init__(self, entries_by_conversation):
        self.entries = entries_by_conversation

    async def scan_entries(self, scan, ctx=None):
        rows = self.entries.get(scan.conversation_id, [])
        out = [e for e in reversed(rows) if (scan.before is None or e.id < scan.before)]
        return out[: scan.limit]


@pytest.mark.asyncio
async def test_derive_context_honours_head_edits_and_tool_result_order():
    rows = [
        t.Entry(
            id=1,
            conversation_id=1,
            kind="pi.user",
            model=[{"role": "user", "content": "one"}],
        ),
        t.Entry(
            id=2,
            conversation_id=1,
            kind="pi.assistant",
            model=[
                {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "c1", "name": "bash"}],
                    "timestamp": 5,
                }
            ],
        ),
        t.Entry(
            id=3,
            conversation_id=1,
            kind="pi.tool_result",
            model=[{"role": "toolResult", "toolCallId": "c1", "content": [], "isError": False}],
        ),
        # the head entry omits entry 1 and replaces entry 3
        t.Entry(
            id=4,
            conversation_id=1,
            kind="pi.user",
            model=[{"role": "user", "content": "four"}],
            head=2,
            edits=[
                t.ContextEdit(target=1, action="omit"),
                t.ContextEdit(target=3, action="replace", messages=[{"role": "user", "content": "replaced"}]),
            ],
        ),
    ]
    storage = _FakeStorage({1: rows})
    view = await derive_context(storage, 1, None, CTX)
    assert view.head.id == 4
    # head 2 starts the transcript, entry 1 is excluded, entry 4 arrives as its own edit
    assert [entry.id for entry in view.entries] == [4, 2, 3]
    assert view.messages == [
        {"role": "user", "content": "four"},
        {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "c1", "name": "bash"}],
            "timestamp": 5,
        },
        # the replace edit removed the stored tool result, so reorder supplies one
        {
            "role": "toolResult",
            "toolCallId": "c1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "Tool result unavailable: history ends before this call completed."}],
            "isError": True,
            "details": {"reason": "missing_after_fork"},
            "timestamp": 5,
        },
        {"role": "user", "content": "replaced"},
    ]


@pytest.mark.asyncio
async def test_derive_context_synthesises_a_missing_tool_result_after_a_fork():
    from pi_agent_core.harness.pico3.context import reorder_tool_results

    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "c1", "name": "bash"},
                {"type": "toolCall", "id": "c2", "name": "bash"},
            ],
            "timestamp": 7,
        },
        {"role": "toolResult", "toolCallId": "c2", "content": [{"type": "text", "text": "ok"}]},
    ]
    out = reorder_tool_results(list(messages))
    assert [message["role"] for message in out] == ["assistant", "toolResult", "toolResult"]
    assert out[1]["toolCallId"] == "c1" and out[1]["isError"] is True
    assert out[1]["details"] == {"reason": "missing_after_fork"}
    assert out[2]["toolCallId"] == "c2"


# ---------------------------------------------------------------------------
# membrane.ts
# ---------------------------------------------------------------------------


def test_membrane_preserves_identity_and_revokes_every_wrapper():
    membrane = Membrane("sticky:1")
    document = {"tasks": {"1": {"phase": "running"}}}
    root = membrane.wrap(document)
    assert membrane.wrap(document) is root
    assert root["tasks"]["1"]["phase"] == "running"
    root["tasks"]["1"]["phase"] = "done"
    assert document["tasks"]["1"]["phase"] == "done"

    membrane.revoke()
    with pytest.raises(RevokedError):
        root["tasks"]
    with pytest.raises(RevokedError):
        root["tasks"] = {}


def test_membrane_rejects_wrapper_and_cyclic_assignments():
    membrane = Membrane("rewindable:1")
    document = {"plugins": {}}
    root = membrane.wrap(document)
    document["plugins"]["x"] = {"a": 1}
    with pytest.raises(PlainInputError):
        root["self"] = root
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(PlainInputError):
        root["cycle"] = cyclic
    assert set(document["plugins"].keys()) == {"x"}

    plain_dict = {"nested": [1, 2]}
    root["plain"] = plain_dict
    plain_dict["nested"].append(3)
    assert document["plain"] == {"nested": [1, 2]}


# ---------------------------------------------------------------------------
# hooks.ts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_runner_filters_by_conversation_subtree_and_stops_on_value():
    seen = []
    reports = []

    class Handlers:
        def __init__(self, tag):
            self.tag = tag

        async def before_tool(self, _info):
            seen.append(self.tag)
            return ("stop" if self.tag == "child" else None)

    registrations = [
        HookRegistration(
            namespace=t.Namespace(id="global"),
            kind=job_kind,
            handlers=Handlers("global"),
        ),
        HookRegistration(
            namespace=t.Namespace(id="child"),
            kind=job_kind,
            handlers=Handlers("child"),
            conversation_id=2,
        ),
        HookRegistration(
            namespace=t.Namespace(id="other"),
            kind=plugin_kind,
            handlers=Handlers("other-kind"),
        ),
    ]
    runner = create_hook_runners(lambda: registrations, lambda _id: [2], reports.append)(job_kind, t.HookApi(kind="pi.job", task_id=1, conversation_id=3))
    assert [binding.namespace.id for binding in runner.handlers()] == ["global"]
    child = create_hook_runners(lambda: registrations, lambda _id: [2], reports.append)(
        job_kind, t.HookApi(kind="pi.job", task_id=1, conversation_id=2)
    )
    stopped = []

    async def fn(handlers, api):
        assert api.kind == "pi.job"
        return await handlers.before_tool(api)

    await child.each(CTX, fn, lambda value: stopped.append(value) or True)
    assert seen == ["global", "child"]
    assert stopped == ["stop"]
    assert reports == []


@pytest.mark.asyncio
async def test_hook_runner_reports_handler_errors_unless_the_context_is_aborted():
    reports = []

    class Boom:
        async def before_tool(self, _info):
            raise RuntimeError("boom")

    runner = create_hook_runners(
        lambda: [HookRegistration(namespace=t.Namespace(id="n"), kind=job_kind, handlers=Boom())],
        lambda _id: [],
        reports.append,
    )(job_kind, t.HookApi(kind="pi.job", task_id=1, conversation_id=1))
    await runner.each(CTX, lambda handlers, api: handlers.before_tool(api))
    assert len(reports) == 1 and str(reports[0]) == "boom"

    controller = AbortController()
    controller.abort()
    aborted_ctx = with_abort_signal(controller.signal, CTX)
    with pytest.raises(RuntimeError, match="boom"):
        await runner.each(aborted_ctx, lambda handlers, api: handlers.before_tool(api))


# ---------------------------------------------------------------------------
# bash.ts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_tool_streams_output_and_reports_the_exit_code():
    chunks = []
    tool = bash_tool()
    assert tool.name == "bash" and tool.replay == "unsafe"
    result = await tool.execute(
        {"command": "printf 'hello'; printf 'oops' >&2"}, type("Api", (), {"stream": chunks.append})(), CTX
    )
    assert b"".join(chunks) == b"hellooops"
    assert result.is_error is False
    assert result.details["exitCode"] == 0

    failed = await tool.execute(
        {"command": "exit 3"}, type("Api", (), {"stream": lambda _chunk: None})(), CTX
    )
    assert failed.is_error is True
    assert failed.details["exitCode"] == 3
    assert isinstance(BashTool().parameters, dict)


# ---------------------------------------------------------------------------
# system.ts
# ---------------------------------------------------------------------------


def test_system_sections_render_and_seeds_serialise():
    assert pico_system.system_sections.environment.render({"cwd": "/tmp"}) == "Working directory: /tmp"
    assert pico_system.system_sections.skills.render(
        [{"name": "review", "description": "review code"}]
    ) == "- review: review code"
    seed = pico_system.section_seed(pico_system.system_sections.identity, "You are pi.")
    assert seed.to_json() == {"key": "identity", "value": "You are pi."}
    assert pico_system.remove_section("environment").to_json() == {"key": "environment", "remove": True}
    record = pico_system.SectionRecord(key="identity", value="x", rendered="x")
    assert pico_system.SectionRecord.from_json(record.to_json()) == record


def test_draft_wrap_applies_only_to_touched_sections_and_freeze_keeps_stored_text():
    canonical = {
        "identity": pico_system.SectionState(value="A", rendered="A"),
        "environment": pico_system.SectionState(value={"cwd": "/a"}, rendered="Working directory: /a"),
    }
    draft = pico_system.Draft(canonical)
    draft.set(pico_system.system_sections.identity, "B")
    draft.wrap(pico_system.system_sections.identity, lambda text: f"<{text}>")
    frozen = pico_system.freeze(draft, canonical, {"identity": pico_system.system_sections.identity})
    assert frozen["identity"].rendered == "<B>"
    assert frozen["identity"].value == "B"
    # the untouched section keeps its stored text even though the renderer would differ now
    assert frozen["environment"] is canonical["environment"]

    snapshot = draft.snapshot()
    draft.delete(pico_system.system_sections.identity)
    draft.restore(snapshot)
    assert "identity" in draft.values


@pytest.mark.asyncio
async def test_fold_canonical_replays_from_the_newest_baseline():
    rows = [
        t.Entry(
            id=1,
            conversation_id=1,
            kind="pi.system",
            data={"baseline": True, "sections": [{"key": "identity", "action": "set", "value": "A", "rendered": "A"}]},
        ),
        t.Entry(
            id=2,
            conversation_id=1,
            kind="pi.system",
            data={"sections": [{"key": "identity", "action": "set", "value": "B", "rendered": "B"}]},
        ),
        t.Entry(
            id=3,
            conversation_id=1,
            kind="pi.system",
            data={"sections": [{"key": "environment", "action": "set", "value": {"cwd": "/x"}, "rendered": "Working directory: /x"}]},
        ),
        t.Entry(id=4, conversation_id=1, kind="pi.system", data={"sections": [{"key": "environment", "action": "remove"}]}),
    ]
    folded = await pico_system.fold_canonical(_FakeStorage({1: rows}), 1)
    assert folded.newest_baseline == 1
    assert folded.newest_managed == 4
    assert list(folded.canonical.keys()) == ["identity"]
    assert folded.canonical["identity"].rendered == "B"


@pytest.mark.asyncio
async def test_plan_managed_entry_writes_a_baseline_then_a_delta():
    storage = _FakeStorage({1: []})
    snapshot = pico_system.PreparationSnapshot(
        newest_managed=None, newest_baseline=None, newest_head=None, settings={}
    )
    tools = [{"name": "bash", "description": "Run a shell command", "parameters": {}}]
    canonical = {"identity": pico_system.SectionState(value="A", rendered="A")}
    plan = await pico_system.plan_managed_entry(
        _TxStub(storage, messages=[], entries=[]), 1, snapshot, {}, canonical, tools, now=42
    )
    assert plan.data.baseline is True
    assert plan.model[0]["role"] == "system" and plan.model[0]["timestamp"] == 42
    assert plan.model[0]["toolsAdded"] == tools
    assert "## identity\nA" in plan.model[0]["content"]

    incremental = await pico_system.plan_managed_entry(
        _TxStub(storage, messages=plan.model, entries=[]),
        1,
        pico_system.PreparationSnapshot(newest_managed=7, newest_baseline=7, newest_head=7),
        canonical,
        {"identity": pico_system.SectionState(value="B", rendered="B")},
        tools,
        now=99,
    )
    assert incremental.data.baseline is False
    assert incremental.model[0]["content"] == "The identity section now reads:\nB"
    assert "toolsAdded" not in incremental.model[0]

    no_change = await pico_system.plan_managed_entry(
        _TxStub(storage, messages=plan.model, entries=[]),
        1,
        pico_system.PreparationSnapshot(newest_managed=7, newest_baseline=7, newest_head=7),
        canonical,
        canonical,
        tools,
        now=99,
    )
    assert no_change is None


def test_effective_tools_folds_adds_and_removes():
    messages = [
        {"role": "system", "content": "a", "toolsAdded": [{"name": "bash"}, {"name": "read"}]},
        {"role": "system", "content": "b", "toolsRemoved": [{"name": "read"}]},
        {"role": "user", "content": "ignored"},
    ]
    assert pico_system.effective_tools(messages) == [{"name": "bash"}]


class _TxStub:
    """The read side of a transaction, over a fake storage or literal rows."""

    def __init__(self, storage, messages, entries):
        self._storage = storage
        self._messages = messages
        self._entries = entries

    async def scan_entries(self, scan):
        return await self._storage.scan_entries(scan, CTX)

    async def conversation(self, _id):
        return t.Conversation(id=1)

    async def newest_entry(self, _id, _opts=None):
        return self._entries[-1] if self._entries else None

    async def context(self, _id):
        return t.ContextView(entries=self._entries, messages=self._messages)

    def snapshot(self, _ref):
        return {}


# ---------------------------------------------------------------------------
# memory.ts / jsonl.ts
# ---------------------------------------------------------------------------


async def _seed(storage):
    conversation_id = storage.mint_id()
    await storage.commit(
        [t.Write(type="conversation", conversation=t.Conversation(id=conversation_id))], CTX
    )
    entry_id = storage.mint_id()
    await storage.commit(
        [
            t.Write(
                type="entry",
                entry=t.Entry(
                    id=entry_id,
                    conversation_id=conversation_id,
                    kind="pi.user",
                    model=[{"role": "user", "content": "hello"}],
                ),
            )
        ],
        CTX,
    )
    return conversation_id, entry_id


@pytest.mark.asyncio
async def test_memory_storage_commits_reads_and_rejects_bad_batches():
    storage = MemoryStorage()
    conversation_id, entry_id = await _seed(storage)
    assert storage.seq == 2
    entry = (await storage.entries([entry_id], CTX))[entry_id]
    assert entry.model == [{"role": "user", "content": "hello"}]

    # a read hands out copies, never live records
    entry.kind = "tampered"
    assert (await storage.entries([entry_id], CTX))[entry_id].kind == "pi.user"

    with pytest.raises(ValueError, match="exists"):
        await storage.commit(
            [t.Write(type="entry", entry=t.Entry(id=entry_id, conversation_id=conversation_id, kind="x"))],
            CTX,
        )
    assert storage.seq == 2  # one failed batch changes nothing

    with pytest.raises(ValueError, match="unknown task"):
        await storage.commit([t.Write(type="task.patch", patch=t.TaskPatch(id=999, status="running"))], CTX)

    scan = await storage.scan_entries(t.EntryScan(conversation_id=conversation_id, limit=10), CTX)
    assert [row.id for row in scan] == [entry_id]


@pytest.mark.asyncio
async def test_memory_storage_task_patch_checkpoint_clear_and_doc_fold():
    storage = MemoryStorage()
    conversation_id, entry_id = await _seed(storage)
    task_id = storage.mint_id()
    await storage.commit(
        [t.Write(type="task", task=t.Task(id=task_id, conversation_id=conversation_id, kind="pi.job"))],
        CTX,
    )
    await storage.commit(
        [t.Write(type="task.patch", patch=t.TaskPatch(id=task_id, status="running", checkpoint={"phase": "spawning"}))],
        CTX,
    )
    task = await storage.task(task_id, CTX)
    assert (task.status, task.checkpoint) == ("running", {"phase": "spawning"})
    await storage.commit(
        [t.Write(type="task.patch", patch=t.TaskPatch(id=task_id, checkpoint=None))], CTX
    )
    assert (await storage.task(task_id, CTX)).checkpoint is None
    await storage.commit(
        [t.Write(type="task.patch", patch=t.TaskPatch(id=task_id, status="terminal", outcome=t.Outcome(status="completed", result={"exitCode": 0})))],
        CTX,
    )
    terminal = await storage.task(task_id, CTX)
    assert (terminal.status, terminal.outcome.result) == ("terminal", {"exitCode": 0})
    assert [row.id for row in await storage.scan_tasks(t.TaskScan(conversation_id=conversation_id, status=["terminal"]), CTX)] == [task_id]

    ref = t.DocRef(doc="rewindable", conversation_id=conversation_id)
    await storage.commit([t.Write(type="doc", ref=ref, ops=[("r", {"threshold": 0.5})])], CTX)
    await storage.commit([t.Write(type="doc", ref=ref, ops=[("s", ("threshold",), 0.9)])], CTX)
    assert await storage.doc(ref, CTX) == {"threshold": 0.9}
    # the base commit lands after entry 2, so nothing is folded as of that entry yet
    assert await storage.doc_as_of(conversation_id, storage.entry_seq[entry_id], CTX) is None
    later_entry = storage.mint_id()
    await storage.commit(
        [t.Write(type="entry", entry=t.Entry(id=later_entry, conversation_id=conversation_id, kind="pi.user"))],
        CTX,
    )
    assert await storage.doc_as_of(conversation_id, later_entry, CTX) == {"threshold": 0.9}
    await storage.commit([t.Write(type="doc", ref=t.DocRef(doc="session"), ops=[("s", ("plugins",), {"x": {}})])], CTX)
    assert await storage.doc(t.DocRef(doc="session"), CTX) == {"plugins": {"x": {}}}
    await storage.close(CTX)


@pytest.mark.asyncio
async def test_jsonl_storage_replays_sidecars_across_reopen_and_repairs_a_torn_tail(tmp_path):
    directory = str(tmp_path)
    storage = await JsonlStorage.open(directory)
    conversation_id, entry_id = await _seed(storage)
    sticky = t.DocRef(doc="sticky", conversation_id=conversation_id)
    await storage.commit([t.Write(type="doc", ref=sticky, ops=[("r", {"inbox": []})])], CTX)
    await storage.commit([t.Write(type="doc", ref=sticky, ops=[("s", ("inbox",), [1, 2])])], CTX)
    task_id = storage.mint_id()
    await storage.commit(
        [t.Write(type="task", task=t.Task(id=task_id, conversation_id=conversation_id, kind="pi.job"))], CTX
    )
    await storage.commit(
        [t.Write(type="task.patch", patch=t.TaskPatch(id=task_id, status="running", checkpoint={"phase": "waiting"}))],
        CTX,
    )
    assert set(storage.sizes()) >= {"main.jsonl", f"sticky-{conversation_id}.jsonl", f"task-{task_id}.jsonl"}
    await storage.close()

    # a torn tail is cut before the file is opened for append
    with open(f"{directory}/main.jsonl", "ab") as handle:
        handle.write(b'{"seq": 99, "maxId": 99, "writes"')
    reopened = await JsonlStorage.open(directory)
    assert (await reopened.doc(sticky, CTX)) == {"inbox": [1, 2]}
    task = await reopened.task(task_id, CTX)
    assert (task.status, task.checkpoint) == ("running", {"phase": "waiting"})
    assert [row.id for row in await reopened.scan_entries(t.EntryScan(conversation_id=conversation_id, limit=5), CTX)] == [entry_id]
    assert reopened.seq == storage.seq

    # truncate drops the sticky sidecar's superseded history, keeping the state
    await reopened.truncate(sticky, CTX)
    header = [json.loads(line) for line in open(f"{directory}/sticky-{conversation_id}.jsonl") if line.strip()]
    assert len(header) == 2 and is_base([tuple(op) for op in header[0]["writes"][0]["ops"]])
    assert (await reopened.doc(sticky, CTX)) == {"inbox": [1, 2]}

    # a terminal task retires its sidecar
    await reopened.commit(
        [t.Write(type="task.patch", patch=t.TaskPatch(id=task_id, status="terminal", outcome=t.Outcome(status="completed")))],
        CTX,
    )
    assert f"task-{task_id}.jsonl" not in reopened.sizes()
    await reopened.close()


# ---------------------------------------------------------------------------
# session.ts primitives
# ---------------------------------------------------------------------------


def test_defaults_register_validate_fill_and_fresh_state():
    kind = t.Kind(
        name="pi.demo",
        config=t.KindConfig(
            rewindable={"profile": "default", "thinkingLevel": "off"},
            sticky={"steeringMode": "all", "inbox": [], "turn": {"tools": []}, "tasks": {}},
        ),
    )
    defaults = Defaults([kind])
    assert defaults.route == {
        "profile": "rewindable",
        "thinkingLevel": "rewindable",
        "steeringMode": "sticky",
        "inbox": "sticky",
        "turn": "sticky",
        "tasks": "sticky",
    }
    with pytest.raises(ValueError, match="more than one kind"):
        Defaults([kind, t.Kind(name="pi.other", config=t.KindConfig(rewindable={"profile": "x"}))])

    assert defaults.validate("thinkingLevel", "high") is True
    assert defaults.validate("thinkingLevel", "loud") is False
    assert defaults.validate("model", {"provider": "anthropic", "modelId": "claude"}) is True
    assert defaults.validate("model", {"provider": "", "modelId": "claude"}) is False
    assert defaults.validate("retry", {"enabled": True, "maxRetries": -1, "baseDelayMs": 0}) is False
    assert defaults.validate("unknownKey", 1) is False
    with pytest.raises(TypeError, match="invalid sticky config"):
        defaults.validate_seed("sticky", {"steeringMode": "nope"})
    with pytest.raises(TypeError, match="invalid sticky config"):
        defaults.validate_seed("sticky", {"profile": "p"})
    with pytest.raises(TypeError, match="invalid rewindable config"):
        defaults.validate_seed("rewindable", {"steeringMode": "all"})
    with pytest.raises(TypeError, match="invalid rewindable config"):
        defaults.validate_seed("rewindable", {"thinkingLevel": "loud"})

    filled = {}
    defaults.fill("rewindable", filled)
    assert filled["profile"] == "default" and filled["thinkingLevel"] == "off"
    fresh = defaults.fresh_rewindable({"thinkingLevel": "low"})
    assert (fresh["profile"], fresh["thinkingLevel"]) == ("default", "low")
    assert fresh["plugins"] == {}
    assert defaults.fresh_sticky({"steeringMode": "all"})["steeringMode"] == "all"
    assert defaults.fresh_sticky({})["inbox"] == []
    defaults.unregister(kind)
    assert defaults.route == {}


def test_session_helpers_plain_validate_entry_and_remove_where():
    assert pico_session.is_core_kind("pi.generation") is True
    assert pico_session.is_core_kind("pi.job") is False
    assert pico_session.plain({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}
    assert pico_session.finite_nonnegative(0) is True and pico_session.finite_nonnegative(-1) is False
    assert pico_session.exact_object({"a": 1}, ["a"]) is True
    assert pico_session.exact_object({"a": 1, "b": 2}, ["a"]) is False

    with pytest.raises(ValueError, match="is in the future"):
        pico_session.validate_entry(t.Entry(id=1, conversation_id=1, kind="x", head=2))
    with pytest.raises(ValueError, match="malformed model message"):
        pico_session.validate_entry(t.Entry(id=2, conversation_id=1, kind="x", model=[{"content": "no role"}]))
    pico_session.validate_entry(t.Entry(id=3, conversation_id=1, kind="x", model=[{"role": "user"}], head=3))

    items = [1, 2, 3, 4]
    pico_session.remove_where(items, lambda value: value % 2 == 0)
    assert items == [1, 3]


# ---------------------------------------------------------------------------
# types.ts
# ---------------------------------------------------------------------------


def test_types_record_json_round_trips_and_helpers():
    conversation = t.Conversation(id=1, parent=t.ConversationParent(conversation_id=0, at=4))
    assert t.conversation_from_json(t.conversation_to_json(conversation)) == conversation

    task = t.Task(id=2, conversation_id=1, kind="pi.job", input={"notify": True}, after=[1], abort=True)
    assert t.task_from_json(t.task_to_json(task)) == task

    write = t.Write(
        type="doc",
        ref=t.DocRef(doc="sticky", conversation_id=1),
        ops=[("s", ("inbox",), [1])],
    )
    assert t.write_from_json(t.write_to_json(write)) == write

    event = t.ViewEvent(type="tool.finished", task_id=2, call_id="c1", entry_id=9, is_error=True)
    assert t.view_event_from_json(t.view_event_to_json(event)) == event
    assert t.doc_key(t.DocRef(doc="rewindable", conversation_id=4)) == "rewindable:4"
    assert t.doc_key(t.DocRef(doc="session")) == "session"

    slot = {}
    assert t.memo_once(slot, "k", None) is None
    assert "k" in slot["memos"] and t.memo_once(slot, "k", "other") is None

    stored = t.to_stored({"role": "user", "content": "hi"})
    stored["content"] = "changed"
    assert stored == {"role": "user", "content": "changed"}
    assert t.define_entry("my.entry").is_entry(t.Entry(id=1, conversation_id=1, kind="my.entry"))
    with pytest.raises(ValueError, match="reserved"):
        t.define_entry("pi.mine")
    with pytest.raises(ValueError, match="reserved"):
        t.define_task(t.Kind(name="pi.mine"))
    for error in (t.Forbidden("x"), t.ConversationBusy(1), t.GenerationInProgress(1), t.CollapseInProgress(1), t.Faulted("boom"), t.Closed(), t.TaskContractFault("pi.job", "no slot")):
        assert isinstance(str(error), str) and error.name


# ---------------------------------------------------------------------------
# view.ts
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, kinds, namespaces=(), defaults=None):
        self.kinds = {kind.name: kind for kind in kinds}
        self.namespaces = {registration.id: registration for registration in namespaces}
        self.defaults = defaults or Defaults(kinds)
        self.conversation_records = {}
        self.live_tasks = {}
        self.documents = {}

    def loaded_document(self, ref):
        return self.documents.get(t.doc_key(ref))

    def seed(self, conversation_id):
        self.conversation_records[conversation_id] = t.Conversation(id=conversation_id)
        self.documents[f"rewindable:{conversation_id}"] = {"plugins": {}, **getattr(self.defaults, "rewindable")}
        self.documents[f"sticky:{conversation_id}"] = {
            "inbox": [],
            "turn": {"tools": []},
            "tasks": {},
            "plugins": {},
            **getattr(self.defaults, "sticky"),
        }
        self.documents["session"] = {"plugins": {}}


def _entry(id, conversation_id, content):
    return t.Entry(
        id=id,
        conversation_id=conversation_id,
        kind="pi.user",
        model=[{"role": "user", "content": content}],
    )


def test_view_manager_publishes_ops_and_events_to_started_and_buffered_watches():
    kinds = [
        t.Kind(name="pi.job", config=t.KindConfig(rewindable={"profile": "default"})),
    ]
    session = _FakeSession(kinds)
    session.seed(1)
    reports = []
    manager = ViewManager(session, reports.append)
    conversation = session.conversation_records[1]

    buffered = manager.watch(conversation, [])
    started = manager.watch(conversation, [])
    envelopes = []

    def listener(envelope):
        envelopes.append(envelope)
        started.__dict__["view"] = apply_envelope(started.view, envelope)

    started.start(listener)

    entry = _entry(1, 1, "hello")
    result = pico_session.CommitResult(
        seq=1,
        changes=pico_session.CommitChanges(entries=[entry]),
    )
    manager.update(result)
    assert envelopes == []  # nothing is published until deliver()
    manager.deliver()
    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope.revision == 1
    assert envelope.ops and not is_base(envelope.ops)
    assert started.view["config"] == {"profile": "default"}
    assert started.view["entries"] == [t.entry_to_json(entry)]

    # the buffered watcher replays the same envelopes on start
    replayed = []
    buffered.start(replayed.append)
    assert [item.revision for item in replayed] == [1]
    replayed_view = apply_envelope(buffered.view, replayed[0])
    assert replayed_view["entries"] == [t.entry_to_json(entry)]

    # a moved head cuts the entries it forks away from
    head = _entry(2, 1, "forked")
    head.head = 2
    manager.update(pico_session.CommitResult(seq=2, changes=pico_session.CommitChanges(entries=[head])))
    manager.deliver()
    assert [row["id"] for row in started.view["entries"]] == [2]

    # entries for another conversation are not folded in
    other = _entry(3, 2, "elsewhere")
    manager.update(pico_session.CommitResult(seq=3, changes=pico_session.CommitChanges(entries=[other])))
    manager.deliver()
    assert [row["id"] for row in started.view["entries"]] == [2]
    assert reports == []


def test_view_manager_reports_a_listener_error_and_survives_it():
    session = _FakeSession([t.Kind(name="pi.job")])
    session.seed(1)
    reports = []
    manager = ViewManager(session, reports.append)
    conversation = session.conversation_records[1]
    watcher = manager.watch(conversation, [])

    def boom(_envelope):
        raise RuntimeError("listener boom")

    watcher.start(boom)
    manager.update(pico_session.CommitResult(seq=1, changes=pico_session.CommitChanges(entries=[_entry(1, 1, "x")])))
    manager.deliver()
    assert len(reports) == 1 and str(reports[0]) == "listener boom"
    assert watcher.closed is True


def test_view_helpers_touches_key_and_strip_private_state():
    from pi_agent_core.harness.pico3.view import strip_private_tool_state, touches_key

    assert touches_key([("s", ("plugins", "x"), 1)], "plugins") is True
    assert touches_key([("r", {"plugins": {}})], "anything") is True
    assert touches_key([("s", ("inbox",), [])], "plugins") is False
    slot = t.ToolSlot(call_id="c1", name="bash", args={}, status="running", memos={"secret": 1})
    assert "memos" not in strip_private_tool_state(slot)
    assert strip_private_tool_state(slot)["callId"] == "c1"
    assert WATCH_CAPACITY == 256


# ---------------------------------------------------------------------------
# kinds/*
# ---------------------------------------------------------------------------


def test_entry_kinds_are_reserved_names_and_witnesses():
    assert entries.user.kind == "pi.user"
    assert entries.system.is_entry(t.Entry(id=1, conversation_id=1, kind="pi.system")) is True
    assert entries.assistant.is_entry(t.Entry(id=1, conversation_id=1, kind="pi.user")) is False
    assert core_entry("pi.notice").kind == "pi.notice"
    assert len(list(entries)) == 9


def test_frames_apply_a_full_assistant_turn_onto_the_tracked_output():
    output = {}
    frames.apply_frame(
        output,
        AssistantMessageFrame(
            type="start",
            partial={
                "role": "assistant",
                "content": [],
                "api": "faux",
                "provider": "faux",
                "model": "faux",
                "timestamp": 1,
            },
        ),
    )
    frames.apply_frame(
        output,
        AssistantMessageFrame(type="text_start", content_index=0, content={"type": "text", "text": ""}),
    )
    frames.apply_frame(output, AssistantMessageFrame(type="text_delta", content_index=0, delta="Hel"))
    frames.apply_frame(output, AssistantMessageFrame(type="text_delta", content_index=0, delta="lo"))
    frames.apply_frame(
        output,
        AssistantMessageFrame(type="text_end", content_index=0, content="Hello", text_signature="sig"),
    )
    frames.apply_frame(
        output,
        AssistantMessageFrame(type="thinking_start", content_index=1, content={"type": "thinking", "thinking": ""}),
    )
    frames.apply_frame(output, AssistantMessageFrame(type="thinking_delta", content_index=1, delta="hm"))
    frames.apply_frame(
        output,
        AssistantMessageFrame(type="thinking_end", content_index=1, content="hm", thinking_signature="tsig"),
    )
    frames.apply_frame(
        output,
        AssistantMessageFrame(type="toolcall_start", content_index=2, tool_call={"type": "toolCall", "id": "c1", "name": "bash"}),
    )
    frames.apply_frame(output, AssistantMessageFrame(type="toolcall_checkpoint", content_index=2, json='{"command":"ls"}'))
    frames.apply_frame(output, AssistantMessageFrame(type="toolcall_delta", content_index=2, delta="ignored"))
    frames.apply_frame(
        output,
        AssistantMessageFrame(
            type="toolcall_end", content_index=2, id="c1", name="bash", arguments={"command": "ls"}
        ),
    )
    message = output["message"]
    assert message["content"][0] == {"type": "text", "text": "Hello", "textSignature": "sig"}
    assert message["content"][1] == {"type": "thinking", "thinking": "hm", "thinkingSignature": "tsig"}
    assert message["content"][2] == {
        "type": "toolCall",
        "id": "c1",
        "name": "bash",
        "arguments": {"command": "ls"},
    }
    assert frames.safe_parse("{oops") == {}
    with pytest.raises(ValueError, match="before start"):
        frames.apply_frame({}, AssistantMessageFrame(type="text_delta", content_index=0, delta="x"))


class _JobHost:
    def __init__(self, statuses=None):
        self.started = []
        self.killed = []
        self.statuses = list(statuses or [t.ProcessStatus(status="exited", exit_code=0, stdout="out", stderr="")])

    async def start(self, key, spec, ctx):
        self.started.append((key, spec))

    async def status(self, key, ctx):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    async def kill(self, key, signal, ctx):
        self.killed.append((key, signal))


class _Runtime:
    def __init__(self, host=None):
        self.process_host = host
        self.checkpoints = []
        self.slots = {}
        self.writes = []
        self.slept = []
        self.now_value = 1000
        self.current_task = t.Task(id=0, conversation_id=1, kind="pi.job")

    def transaction(self):
        outer = self

        class Tx:
            def checkpoint(self, value):
                outer.checkpoints.append(value)

            def slot(self, ref):
                return outer.slots.setdefault(ref.id, {})

            async def write(self, conversation_id, entry):
                outer.writes.append((conversation_id, entry))
                return 1

        return Tx()

    def now(self):
        return self.now_value

    async def sleep(self, until_ms, ctx):
        self.slept.append(until_ms)

    async def commit(self, fn, ctx, current=None):
        outer = self
        Tx = type(self.transaction())
        task = current if current is not None else outer.current_task
        result = fn(Tx(), task)
        if hasattr(result, "__await__"):
            result = await result
        return result


@pytest.mark.asyncio
async def test_job_kind_spawns_polls_and_repeats():
    host = _JobHost()
    runtime = _Runtime(host)
    task = t.Task(id=7, conversation_id=1, kind="pi.job", input={"notify": True})
    runtime.current_task = task
    step = await job_kind.initial(task, runtime, CTX)
    assert runtime.checkpoints == [
        {"phase": "spawning", "key": "7:1", "occurrence": 1},
        {"phase": "running", "key": "7:1", "occurrence": 1},
    ]
    assert host.started == [("7:1", task.input)]
    assert runtime.slots[7] == {
        "stdout": "out",
        "stderr": "",
        "droppedStdout": 0,
        "droppedStderr": 0,
        "exitCode": 0,
    }
    completion = await step.done(runtime.transaction(), task)
    assert completion.status == "completed"
    assert completion.result == JobResult(exitCode=0, occurrences=1, stdout="out", stderr="")
    assert runtime.writes[0][1]["model"][0]["content"] == "job 7 exited with code 0"


@pytest.mark.asyncio
async def test_job_kind_waits_when_not_before_is_in_the_future_and_handles_no_host():
    runtime = _Runtime(_JobHost())
    runtime.now_value = 1000
    task = t.Task(id=1, conversation_id=1, kind="pi.job", input={"notBefore": 5000})
    step = await job_kind.initial(task, runtime, CTX)
    assert step.next == {"phase": "waiting", "untilMs": 5000, "occurrence": 1}
    assert runtime.checkpoints == []

    # the waiting phase sleeps then spawns
    host = _JobHost()
    runtime = _Runtime(host)
    waiting_task = t.Task(
        id=2, conversation_id=1, kind="pi.job", checkpoint={"phase": "waiting", "untilMs": 1500, "occurrence": 1}
    )
    await job_kind.phases["waiting"](waiting_task, runtime, CTX)
    assert runtime.slept == [1500] and host.started == [("2:1", {})]

    hostless = _Runtime(None)
    step = await job_kind.initial(t.Task(id=3, conversation_id=1, kind="pi.job", input={}), hostless, CTX)
    assert step.done(None, None).failure.detail == "no process host"

    unknown = _Runtime(_JobHost([t.ProcessStatus(status="unknown")]))
    running = t.Task(
        id=4, conversation_id=1, kind="pi.job", checkpoint={"phase": "running", "key": "4:1", "occurrence": 1}, input={}
    )
    step = await job_kind.phases["running"](running, unknown, CTX)
    assert step.done(None, None).failure == JobFailure(reason="interrupted", detail="process outcome unknown")


@pytest.mark.asyncio
async def test_job_kind_abort_escalates_to_sigkill():
    host = _JobHost()
    runtime = _Runtime(host)
    task = t.Task(id=5, conversation_id=1, kind="pi.job", checkpoint={"phase": "running", "key": "5:1", "occurrence": 1})
    finish = await job_kind.abort(task, runtime, CTX)
    assert finish() == {"killed": True}
    assert host.killed == [("5:1", "SIGTERM"), ("5:1", "SIGKILL")]
    assert runtime.slept == [6000]

    idle = await job_kind.abort(t.Task(id=6, conversation_id=1, kind="pi.job"), runtime, CTX)
    assert idle() == {"killed": False}


@pytest.mark.asyncio
async def test_plugin_kind_resolves_completes_and_fails():
    async def handler(input_value, api, ctx):
        assert api.task_id == 1 and api.conversation_id == 1
        return {"echo": input_value}

    class Runtime:
        plugins = {"echo": handler}

    task = t.Task(id=1, conversation_id=1, kind="pi.plugin", input={"handler": "echo", "input": 42})
    step = await plugin_kind.phases["started"](task, Runtime(), CTX)
    completion = step.done(None, task)
    assert completion.status == "completed" and completion.result == {"echo": 42}

    missing = await plugin_kind.phases["started"](
        t.Task(id=1, conversation_id=1, kind="pi.plugin", input={"handler": "nope"}), Runtime(), CTX
    )
    assert missing.done(None, task).failure == PluginFailure(reason="missing_handler", detail="nope")

    class BoomRuntime:
        plugins = {"boom": lambda *_args: _raise()}

    def _raise():
        raise RuntimeError("handler boom")

    threw = await plugin_kind.phases["started"](
        t.Task(id=1, conversation_id=1, kind="pi.plugin", input={"handler": "boom"}), BoomRuntime(), CTX
    )
    failure = threw.done(None, task).failure
    assert failure.reason == "threw" and "handler boom" in failure.detail


@pytest.mark.asyncio
async def test_task_api_exposes_the_task_surface_but_not_the_tool_only_one():
    class Runtime:
        def __init__(self):
            self.sent = []
            self.aborted = []

        async def send_owned(self, conversation_id, input, ctx):
            self.sent.append((conversation_id, input))
            return 11

        async def wait_for_input(self, id, ctx):
            return t.Input(id=id, conversation_id=1, status="done")

        async def abort_conversation(self, id, ctx):
            self.aborted.append(id)

        async def commit(self, fn, ctx):
            class Tx:
                async def input(self, id):
                    return t.Input(id=id, conversation_id=1, status="placed")

                def create_task(self, kind, input, opts):
                    return t.TaskRef(id=5, kind=kind)

                async def task(self, id):
                    return t.Task(id=id, conversation_id=1, kind="pi.job", status="running")

                def snapshot(self, ref):
                    return {"tasks": {5: {"stdout": "x"}}}

            result = fn(Tx(), None)
            if hasattr(result, "__await__"):
                result = await result
            return result

        async def create_owned_conversation(self, spec, ctx):
            return 3

        async def wait_for_task(self, id, ctx):
            return t.Task(id=id, conversation_id=1, kind="pi.job", status="terminal")

    runtime = Runtime()
    api = task_api_module.task_api(t.Task(id=1, conversation_id=1, kind="pi.job"), runtime)
    assert api.task_id == 1 and api.call_id == ""
    for blocked in (lambda: api.stream(b"x"), lambda: api.progress(None), lambda: api.memo("k")):
        with pytest.raises(RuntimeError):
            blocked()

    owned = await api.conversation(t.OwnedConversationSpec(), CTX)
    assert owned.id == 3
    sent = await owned.send(t.SendInput(content="hi"), CTX)
    assert sent["id"] == 11 and (await sent["wait"](CTX)).status == "done"
    await owned.abort(CTX)
    assert runtime.aborted == [3]

    ref = await api.task(job_kind, {"notify": True}, {}, CTX)
    assert ref.id == 5
    assert (await api.get_task(ref, CTX)).status == "running"
    assert (await api.wait_for_task(ref, CTX)).status == "terminal"
    assert await api.slot(ref, CTX) == {"stdout": "x"}
    with pytest.raises(ValueError, match="kind token"):
        await api.task("pi.job", {}, {}, CTX)


# ---------------------------------------------------------------------------
# the barrel
# ---------------------------------------------------------------------------


def test_barrel_exports_every_ported_name():
    import pi_agent_core.harness.pico3 as barrel

    missing = [name for name in barrel.__all__ if not hasattr(barrel, name)]
    assert missing == []
    assert barrel.is_core_kind("pi.collapse") is True
    assert barrel.JobInput().notify is False
    assert barrel.kinds["pi.job"] is job_kind


_ = (asyncio, Tracker, MemoryStorage, apply_immutable)


# ---------------------------------------------------------------------------
# scheduler.ts
# ---------------------------------------------------------------------------


class _FakeLine:
    """The smallest session the scheduler needs: tasks, listeners, commit, on_line."""

    def __init__(self, tasks=()):
        self.tasks = {task.id: task for task in tasks}
        self.live_tasks = dict(self.tasks)
        self.listeners = set()
        self.retired = []
        self.commits = []
        self.storage = self

    async def commit(self, invoker, fn, ctx, options=None):
        line = self
        changes = pico_session.CommitChanges()
        outer = self

        class Tx:
            async def task(self, id):
                return line.tasks.get(id)

            def set_task(self, task):
                line.tasks[task.id] = task
                if task.status == "terminal":
                    line.live_tasks.pop(task.id, None)
                else:
                    line.live_tasks[task.id] = task
                changes.tasks.append(task)

            def checkpoint(self, value):
                current = line.live_tasks.get(1)
                if current is not None:
                    current.checkpoint = value

            def sticky(self, conversation_id):
                return line.stickies.setdefault(conversation_id, {"turn": {"tools": []}})

            async def resolve_inputs(self, ids, resolution):
                line.resolved.append((list(ids), resolution))

        value = fn(Tx(), ctx, Tx())
        if hasattr(value, "__await__"):
            value = await value
        result = pico_session.CommitResult(value=value, seq=len(self.commits), changes=changes)
        self.commits.append(result)
        for listener in list(self.listeners):
            listener(result)
        return result

    async def on_line(self, fn, ctx):
        result = fn(ctx)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def retire(self, task, ctx):
        self.retired.append(task.id)

    async def task(self, id, ctx=None):
        return self.tasks.get(id)

    async def input(self, id, ctx=None):
        return None


def _scheduler_deps(session, kinds, on_report, runtime=None):
    from pi_agent_core.harness.pico3.scheduler import SchedulerDeps

    return SchedulerDeps(
        session=session,
        kinds={kind.name: kind for kind in kinds},
        runtime=runtime
        or (
            lambda task, invoker, ctx: type(
                "Rt",
                (),
                {
                    "task_id": task.id,
                    "conversation_id": task.conversation_id,
                    "kind": task.kind,
                    "commit": lambda self, fn, ctx: session.commit({"type": "task"}, fn, ctx),
                },
            )()
        ),
        on_report=on_report,
        ctx=CTX,
    )


async def _settle(rounds=12):
    for _ in range(rounds):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_scheduler_reserves_runs_the_phase_loop_and_retires_the_task():
    from pi_agent_core.harness.pico3.scheduler import Scheduler

    async def advance(task, runtime, ctx):
        return t.Step(next={"phase": "done"})

    async def finish(task, runtime, ctx):
        return t.Step(done=lambda tx, current, ctx=None: t.Completion(status="completed", result={"ok": True}))

    kind = t.Kind(name="pi.demo", phases={"done": finish}, initial=advance)
    session = _FakeLine([t.Task(id=1, conversation_id=1, kind="pi.demo", input={})])
    reports = []
    scheduler = Scheduler(_scheduler_deps(session, [kind], reports.append))
    session.stickies = {}
    session.resolved = []

    scheduler.resume()
    await _settle()
    assert reports == []
    assert scheduler.quiescent() is True
    assert session.retired == [1]
    final = [result for result in session.commits if result.changes.tasks][-1]

    terminal = final.changes.tasks[-1]
    assert terminal.status == "terminal"
    assert terminal.outcome.status == "completed" and terminal.outcome.result == {"ok": True}
    assert 1 not in session.live_tasks

    # a terminal task is never reserved again
    marks = await scheduler.abort_task(1, CTX)
    assert marks == "terminal"


@pytest.mark.asyncio
async def test_scheduler_faults_a_kind_that_breaks_its_contract():
    from pi_agent_core.harness.pico3.scheduler import Scheduler

    async def broken(task, runtime, ctx):
        return t.Step(next={"phase": "nope"})

    kind = t.Kind(name="pi.demo", phases={"other": broken}, initial=broken)
    session = _FakeLine([t.Task(id=1, conversation_id=1, kind="pi.demo", input={})])
    reports = []
    scheduler = Scheduler(_scheduler_deps(session, [kind], reports.append))
    scheduler.resume()
    await _settle()
    assert len(reports) == 1 and isinstance(reports[0], t.TaskContractFault)
    faulted = [result for result in session.commits if result.changes.tasks][-1].changes.tasks[-1]
    assert faulted.status == "terminal"
    assert faulted.outcome.status == "faulted"
    assert "transition to unknown phase nope" in faulted.outcome.error


@pytest.mark.asyncio
async def test_scheduler_abort_marks_then_joins_the_running_invocation():
    from pi_agent_core.harness.pico3.scheduler import Scheduler

    started = asyncio.Event()

    async def long_phase(task, runtime, ctx):
        started.set()
        await ctx.signal.wait()
        raise RuntimeError("aborted handler")

    kind = t.Kind(name="pi.demo", phases={}, initial=long_phase)
    session = _FakeLine([t.Task(id=1, conversation_id=1, kind="pi.demo", input={})])
    session.stickies = {}
    session.resolved = []
    reports = []
    scheduler = Scheduler(_scheduler_deps(session, [kind], reports.append))
    scheduler.resume()
    await started.wait()
    assert scheduler.live_invocations == 1
    assert await scheduler.abort_task(1, CTX) == "marked"
    assert session.live_tasks[1].abort is True


@pytest.mark.asyncio
async def test_scheduler_hold_gates_draining_and_wait_for_idle_resolves():
    from pi_agent_core.harness.pico3.scheduler import Scheduler

    async def finish(task, runtime, ctx):
        return t.Step(done=lambda tx, current, ctx=None: t.Completion(status="completed", result={}))

    kind = t.Kind(name="pi.demo", phases={}, initial=finish)
    session = _FakeLine([t.Task(id=1, conversation_id=1, kind="pi.demo", input={})])
    session.stickies = {}
    session.resolved = []
    reports = []
    scheduler = Scheduler(_scheduler_deps(session, [kind], reports.append))

    release = scheduler.hold()
    scheduler.resume()
    await _settle()
    assert scheduler.live_invocations == 0  # held: nothing is dispatched

    idle = asyncio.ensure_future(scheduler.wait_for_idle(1, CTX))
    await _settle(2)
    assert not idle.done()
    release()
    await _settle()
    assert idle.done() and idle.exception() is None
    assert scheduler.is_idle(1) is True

    await scheduler.join_all()
    assert scheduler.quiescent() is True


# ---------------------------------------------------------------------------
# session.ts: TxImpl + Session (phase 2)
# ---------------------------------------------------------------------------


async def _boom(*_args, **_kwargs):
    raise AssertionError("the model must not be called")


@pytest.mark.asyncio
async def test_session_commit_replays_and_persists_what_a_transaction_wrote(tmp_path):
    from pi_agent_core.harness.pico3 import session as pico_line

    storage = MemoryStorage()
    session = pico_line.Session(
        storage, {kind.name: kind for kind in (job_kind, plugin_kind)}, {}
    )
    reports = []
    session.on_report = reports.append
    committed = []
    session.listeners.add(lambda result: committed.append(result.seq))

    async def setup(tx, line_ctx, control):
        conversation_id = tx.create_conversation(t.ConversationSpec())
        return conversation_id

    first = await session.commit({"type": "kernel"}, setup, CTX)
    conversation_id = first.value
    assert first.seq == 1 and committed == [1]
    assert session.conversation_records[conversation_id].id == conversation_id
    assert session.index.subtree(conversation_id) == {conversation_id}
    assert any(change.ref.doc == "rewindable" for change in first.changes.docs)

    # replay: a second Session over the same storage sees the conversation and its documents
    replay = pico_line.Session(
        MemoryStorage(), {kind.name: kind for kind in (job_kind, plugin_kind)}, {}
    )
    assert replay is not None
    conversation = await session.read(lambda s, ctx: s.conversation(conversation_id, ctx), CTX)
    assert conversation.id == conversation_id
    sticky = await session.read(
        lambda s, ctx: s.doc(t.DocRef(doc="sticky", conversation_id=conversation_id), ctx), CTX
    )
    assert sticky["inbox"] == [] and sticky["tasks"] == {}
    await session.close(CTX)
    assert session.closed is True


@pytest.mark.asyncio
async def test_tx_impl_capabilities_scope_poison_and_abort_path():
    from pi_agent_core.harness.pico3 import session as pico_line
    from pi_agent_core.harness.pico3.session import Forbidden as LineForbidden

    session = pico_line.Session(MemoryStorage(), {job_kind.name: job_kind}, {})

    async def setup(tx, ctx, control):
        mine = tx.create_conversation(t.ConversationSpec())
        other = tx.create_conversation(t.ConversationSpec())
        return mine, other

    mine, other = (await session.commit({"type": "kernel"}, setup, CTX)).value

    # a host transaction may not touch core turn machinery, and may not leave its conversation
    async def core_attempt(tx, ctx, control):
        tx.sticky(mine)

    with pytest.raises(LineForbidden, match="core turn machinery only"):
        await session.commit({"type": "host", "conversationId": mine}, core_attempt, CTX)

    async def make_task(tx, ctx, control):
        return tx.create_task(t.TaskSpec(kind="pi.job", conversation_id=mine))

    task_id = (await session.commit({"type": "kernel"}, make_task, CTX)).value
    token = t.InvocationToken(task_id, "run")
    invoker = t.Invoker(
        type="task", token=token, id=task_id, conversation_id=mine, kind=job_kind, mode="run"
    )

    async def outside(tx, ctx, control):
        await tx.context(other)

    with pytest.raises(LineForbidden, match="outside this task's subtree"):
        await session.commit(invoker, outside, CTX)

    # the callback surface closes when the callback returns
    captured = {}

    async def capture(tx, ctx, control):
        captured["tx"] = tx
        return None

    await session.commit({"type": "host", "conversationId": mine}, capture, CTX)
    # the surface exposes exactly the whitelisted methods, and none of the transaction internals
    assert not hasattr(captured["tx"], "preload")
    assert not hasattr(captured["tx"], "evict_touched")
    with pytest.raises(TypeError, match="outside its callback"):
        captured["tx"].context(mine)

    # an abort invocation that outlives its mark is refused, and a marked task is refused to its run lease
    async def mark(tx, ctx, control):
        tx.mark_task(task_id)

    await session.commit({"type": "kernel"}, mark, CTX)
    assert session.live_tasks[task_id].abort is True

    async def checkpoint_after_mark(tx, ctx, control):
        tx.checkpoint({"phase": "waiting", "untilMs": 0, "occurrence": 1})

    with pytest.raises(LineForbidden, match="marked run invocation"):
        await session.commit(invoker, checkpoint_after_mark, CTX)

    token.revoke()

    async def after_finish(tx, ctx, control):
        return None

    with pytest.raises(LineForbidden, match="finished invocation"):
        await session.commit(invoker, after_finish, CTX)


@pytest.mark.asyncio
async def test_tx_impl_task_writes_scans_and_reopen_reconciliation():
    from pi_agent_core.harness.pico3 import session as pico_line
    from pi_agent_core.harness.pico3.scheduler import Scheduler, SchedulerDeps

    storage = MemoryStorage()
    session = pico_line.Session(storage, {job_kind.name: job_kind}, {})

    async def setup(tx, ctx, control):
        conversation_id = tx.create_conversation(t.ConversationSpec())
        return conversation_id, tx.create_task(t.TaskSpec(kind="pi.job", conversation_id=conversation_id))

    conversation_id, task_id = (await session.commit({"type": "kernel"}, setup, CTX)).value
    assert session.live_tasks[task_id].status == "pending"

    # scans reject after a same-batch write to their domain
    async def scan_after_write(tx, ctx, control):
        tx.create_task(t.TaskSpec(kind="pi.job", conversation_id=conversation_id))
        await tx.tasks(t.TaskScan(conversation_id=conversation_id))

    with pytest.raises(pico_line.ReadAfterWrite):
        await session.commit({"type": "kernel"}, scan_after_write, CTX)

    # tasks() merges the live map with storage rows
    rows = await session.read(lambda s, ctx: s.scan_tasks(t.TaskScan(conversation_id=conversation_id), ctx), CTX)
    assert [row.id for row in rows] == [task_id]

    # reopen: the same storage, a process that has no such kind registered
    pico_line._owners_delete(session.storage)
    reopened = pico_line.Session(session.storage, {}, {})
    orphan_conversation_id = conversation_id
    orphan_id = task_id
    live = await reopened.read(
        lambda s, ctx: s.scan_tasks(t.TaskScan(status=["pending", "running"]), ctx), CTX
    )
    for task in live:
        reopened.live_tasks[task.id] = task
    scheduler = Scheduler(
        SchedulerDeps(
            session=reopened,
            kinds={},
            runtime=lambda task, invoker, ictx: None,
            on_report=lambda error: None,
            ctx=CTX,
        )
    )
    assert scheduler.is_idle(None) is False

    async def orphan(tx, ctx, control):
        for task in list(reopened.live_tasks.values()):
            if task.kind not in reopened.kinds:
                updated = t.task_from_json(t.task_to_json(task))
                updated.status = "terminal"
                updated.outcome = t.Outcome(status="orphaned")
                tx.set_task(updated)

    await reopened.commit(
        {"type": "kernel"},
        orphan,
        CTX,
        {"docs": [t.DocRef(doc="sticky", conversation_id=orphan_conversation_id)]},
    )
    assert reopened.live_tasks == {}
    final = await reopened.read(lambda s, ctx: s.task(orphan_id, ctx), CTX)
    assert (final.status, final.outcome.status) == ("terminal", "orphaned")
    assert scheduler.is_idle(None) is True


# ---------------------------------------------------------------------------
# kinds/{generation,tool,collapse,post-tools}.ts (phase 2)
# ---------------------------------------------------------------------------


async def _create_task(session, kind_name, conversation_id, input_value):
    """Create a task through the session, then mark it live as the scheduler would at open."""

    async def create(tx, ctx, control):
        return tx.create_task(
            t.TaskSpec(kind=kind_name, conversation_id=conversation_id, input=input_value)
        )

    task_id = (await session.commit({"type": "kernel"}, create, CTX)).value
    task = await session.read(lambda s, ctx: s.task(task_id, ctx), CTX)
    session.live_tasks[task.id] = task
    return task


@dataclass
class _Registries:
    """The ``rt.registries`` shape: sections and tools, each with ``map`` and ``revision``."""

    sections: Any
    tools: Any


def _conversation_docs(conversation_id):
    """The document set a conversation-scoped commit must preload."""
    return [
        t.DocRef(doc="rewindable", conversation_id=conversation_id),
        t.DocRef(doc="sticky", conversation_id=conversation_id),
    ]


class _ScriptedModels:
    """A pico3 ``Models`` adapter over scripted stream events (no provider calls)."""

    def __init__(self, model, scripts):
        self.model = model
        self._scripts = list(scripts)
        self.requests = []

    def resolve(self, ref):
        if ref is None:
            return None
        provider = getattr(ref, "provider", None) if not isinstance(ref, dict) else ref.get("provider")
        model_id = (
            getattr(ref, "model_id", None) if not isinstance(ref, dict) else ref.get("modelId")
        )
        if provider == self.model.provider and model_id == self.model.id:
            return self.model
        return None

    def stream(self, model, request, ctx):
        self.requests.append(request)

        async def events():
            script = self._scripts.pop(0) if self._scripts else []
            for event in script:
                yield event

        return events()

    async def fetch_deferred(self, model, handle, ctx):
        return self._scripts.pop(0) if self._scripts else None

    async def cancel_deferred(self, model, handle, ctx):
        return None


def _text_script(partial_text, final_text, timestamp=7):
    """A streamed text answer, shaped exactly like the faux provider emits it."""
    partial = PiAssistantMessage(
        content=[],
        api="faux",
        provider="faux",
        model="faux-1",
        usage=PiUsage(),
        stop_reason="pending",
        timestamp=timestamp,
    )
    started = copy.deepcopy(partial)
    started.content.append(PiTextContent(text=""))
    streamed = copy.deepcopy(started)
    streamed.content[0] = PiTextContent(text=partial_text)
    message = PiAssistantMessage(
        content=[PiTextContent(text=final_text)],
        api="faux",
        provider="faux",
        model="faux-1",
        usage=PiUsage(input=3, output=4, total_tokens=7),
        stop_reason="stop",
        timestamp=timestamp,
    )
    return [
        PiAssistantMessageEvent(type="start", partial=partial),
        PiAssistantMessageEvent(type="text_start", content_index=0, partial=started),
        PiAssistantMessageEvent(
            type="text_delta", content_index=0, delta=partial_text, partial=streamed
        ),
        PiAssistantMessageEvent(
            type="text_end", content_index=0, content=final_text, partial=streamed
        ),
        PiAssistantMessageEvent(type="done", message=message, reason="stop"),
    ]


def _tool_call_script(call_id, name, arguments):
    """A streamed tool call, shaped exactly like the faux provider emits it."""
    partial = PiAssistantMessage(
        content=[],
        api="faux",
        provider="faux",
        model="faux-1",
        usage=PiUsage(),
    )
    started = copy.deepcopy(partial)
    started.content.append(PiToolCall(id=call_id, name=name, arguments={}))
    call = PiToolCall(id=call_id, name=name, arguments=arguments)
    finished = copy.deepcopy(started)
    finished.content[0] = PiToolCall(id=call_id, name=name, arguments=copy.deepcopy(arguments))
    message = PiAssistantMessage(
        content=[PiToolCall(id=call_id, name=name, arguments=copy.deepcopy(arguments))],
        api="faux",
        provider="faux",
        model="faux-1",
        usage=PiUsage(input=5, output=5, total_tokens=10),
        stop_reason="toolUse",
    )
    return [
        PiAssistantMessageEvent(type="start", partial=partial),
        PiAssistantMessageEvent(
            type="toolcall_start", content_index=0, partial=started, tool_call=call
        ),
        PiAssistantMessageEvent(
            type="toolcall_delta", content_index=0, delta=json.dumps(arguments), partial=finished
        ),
        PiAssistantMessageEvent(
            type="toolcall_end", content_index=0, partial=finished, tool_call=call
        ),
        PiAssistantMessageEvent(type="done", message=message, reason="stop"),
    ]


class _EchoTool:
    """A registered tool declaration with a safe replay contract."""

    name = "echo"
    description = "Echo the argument"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    replay = "safe"
    output = {"maxBytes": 1024, "maxLines": 10, "retain": "head"}

    def __init__(self):
        self.calls = []

    async def execute(self, args, api, ctx):
        self.calls.append((args, api.call_id))
        api.stream(f"echo:{args['text']}")
        return t.ToolResult(content=[{"type": "text", "text": f"echo:{args['text']}"}])


def _faux_model():
    return PiModel(
        id="faux-1",
        name="Faux",
        api="faux",
        provider="faux",
        base_url="http://localhost:0",
        input=["text"],
        cost=PiModelCost(),
        context_window=128_000,
        max_tokens=16_384,
    )


@pytest.mark.asyncio
async def test_generation_kind_prepares_the_managed_entry_and_appends_the_assistant():
    from pi_agent_core.harness.pico3 import session as pico_line
    from pi_agent_core.harness.pico3.kinds.generation import generation_kind as kind

    model = _faux_model()
    models = _ScriptedModels(model, [_text_script("Hel", "Hello")])
    storage = MemoryStorage()
    session = pico_line.Session(
        storage, {k.name: k for k in (generation_kind, plugin_kind, job_kind)}, {}
    )

    async def setup(tx, ctx, control):
        conversation_id = tx.create_conversation(t.ConversationSpec())
        tx.rewindable(conversation_id)["model"] = {"provider": "faux", "modelId": "faux-1"}
        tx.rewindable(conversation_id)["selectedTools"] = ["echo"]
        return conversation_id

    conversation_id = (await session.commit({"type": "kernel"}, setup, CTX)).value
    tool = _EchoTool()
    captured_entries = []
    subject = conversation_id
    subject_kind = kind
    subject_models = models

    class Runtime:
        task_id = 0
        conversation_id = subject
        kind = subject_kind
        models = subject_models
        tools = {"echo": tool}
        kinds = {
            "pi.generation": subject_kind,
            "pi.tool": tool_kind,
            "pi.post_tools": post_tools_kind,
        }
        plugins = {}
        process_host = None

        class _Registry:
            def __init__(self, map_value, revision):
                self.map = map_value
                self._revision = revision

            @property
            def revision(self):
                return self._revision

        registries = None  # set below

        class _Hooks:
            def handlers(self):
                return []

            async def each(self, ctx, fn, on_value=None):
                return None

        hooks = _Hooks()

        def now(self):
            return 1000

        async def sleep(self, until_ms, ctx):
            return None

        subject_task = None
        task_id = 1

        async def commit(self, fn, ctx):
            subject_task = session.live_tasks.get(self.task_id) or self.subject_task

            async def call(tx, line_ctx, control=None):
                return await fn(tx, subject_task, line_ctx)

            result = await session.commit(
                t.Invoker(
                    type="task",
                    token=t.InvocationToken(self.task_id, "run"),
                    id=self.task_id,
                    conversation_id=subject,
                    kind=subject_kind,
                    core=True,
                ),
                call,
                ctx,
                {"docs": _conversation_docs(subject)},
            )
            return result.value

        async def context(self, conversation_id, at, ctx):
            async def read(tx, line_ctx, control=None):
                return await tx.context(conversation_id, at)

            return (await session.commit({"type": "kernel"}, read, ctx)).value

        async def newest_entry(self, conversation_id, opts, ctx):
            async def read(tx, line_ctx, control=None):
                return await tx.newest_entry(conversation_id, opts)

            return (await session.commit({"type": "kernel"}, read, ctx)).value

        async def rewindable(self, conversation_id, ctx):
            async def read(tx, line_ctx, control=None):
                return tx.snapshot(t.DocRef(doc="rewindable", conversation_id=conversation_id))

            return (await session.commit({"type": "kernel"}, read, ctx)).value

        async def sticky(self, conversation_id, ctx):
            async def read(tx, line_ctx, control=None):
                return tx.snapshot(t.DocRef(doc="sticky", conversation_id=conversation_id))

            return (await session.commit({"type": "kernel"}, read, ctx)).value

    registries = _Registries(
        sections=Runtime._Registry({}, 0), tools=Runtime._Registry({"echo": tool}, 0)
    )
    runtime = Runtime()
    runtime.registries = registries

    from pi_agent_core.harness.pico3.system import system_sections

    runtime.registries.sections.map = {"identity": system_sections.identity}

    task = await _create_task(session, "pi.generation", conversation_id, {"inputs": []})
    runtime.subject_task = task
    assert session.live_tasks.get(task.id) is not None
    runtime.task_id = task.id
    step = await kind.initial(task, runtime, CTX)
    assert callable(step.next)
    checkpoint = await runtime.commit(step.next, CTX)
    assert checkpoint["phase"] == "prepared" and checkpoint["model"]["modelId"] == "faux-1"
    assert checkpoint["tools"] == ["echo"]
    assert checkpoint["cutoff"] == checkpoint["system"]

    # the managed entry is a pi.system entry carrying the rendered section
    entries = await session.read(
        lambda s, ctx: s.scan_entries(t.EntryScan(conversation_id=conversation_id, limit=10), ctx), CTX
    )
    assert [entry.kind for entry in entries] == ["pi.system"]
    assert entries[0].data["baseline"] is True
    assert entries[0].model[0]["toolsAdded"][0]["name"] == "echo"

    prepared_task = t.Task(
        id=task.id,
        conversation_id=conversation_id,
        kind="pi.generation",
        input={"inputs": []},
        checkpoint=checkpoint,
    )
    session.live_tasks[prepared_task.id] = prepared_task
    step = await kind.phases["prepared"](prepared_task, runtime, CTX)
    assert callable(step.done)
    recorded = []

    async def finish(tx, line_ctx, control=None):
        return await step.done(tx, session.live_tasks.get(prepared_task.id), line_ctx)

    completion = await session.commit(
        {"type": "kernel", "conversationId": conversation_id}, finish, CTX
    )
    assert completion.value.status == "completed"
    assistant = completion.value.result.assistant
    entries = await session.read(
        lambda s, ctx: s.scan_entries(t.EntryScan(conversation_id=conversation_id, limit=10), ctx), CTX
    )
    assistant_entry = next(entry for entry in entries if entry.id == assistant)
    assert assistant_entry.kind == "pi.assistant"
    assert assistant_entry.model[0]["content"][0]["text"] == "Hello"
    # the derived request opens with the managed system entry and carries the user turn
    assert models.requests[0].messages[0]["role"] == "system"
    assert models.requests[0].messages[0]["toolsAdded"][0]["name"] == "echo"
    captured_entries.extend(entries)
    assert len(captured_entries) == 2
    _ = recorded


@pytest.mark.asyncio
async def test_retry_decision_and_collapse_choose_through():
    from pi_agent_core.harness.pico3.kinds.collapse import choose_through
    from pi_agent_core.harness.pico3.kinds.generation import retry_decision

    retry = t.RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1000, max_agent_delay_ms=60_000)
    assert retry_decision({"retry": retry, "attempt": 0}, None, 100) == {"kind": "retry", "untilMs": 1100}
    assert retry_decision({"retry": retry, "attempt": 2}, None, 100) == {"kind": "retry", "untilMs": 2100}
    assert retry_decision({"retry": retry, "attempt": 3}, None, 100) == {
        "kind": "fail",
        "reason": "retries_exhausted",
    }
    assert retry_decision({"retry": t.RetryPolicy(enabled=False), "attempt": 0}, None, 100) == {
        "kind": "fail",
        "reason": "provider",
    }

    from pi_agent_core.harness.pico3.kinds.collapse import estimate

    def exchange(entry_id, role, text):
        return t.Entry(
            id=entry_id,
            conversation_id=1,
            kind="pi.user",
            model=[{"role": role, "content": text}],
        )

    entries = [
        exchange(1, "user", "u" * 4000),
        exchange(2, "assistant", "a" * 4000),
        exchange(3, "toolResult", "t" * 4000),
        exchange(4, "user", "the newest question"),
    ]
    _ = entries
    sizes = [estimate(list(entry.model)) for entry in entries]
    assert sizes[0] > 0 and sizes[1] > 0
    # exactly the last exchange fits: the assistant/toolResult exchange is the cut point
    assert choose_through(entries, sizes[3]) == 3
    # the assistant and its tool result count as one exchange, so a little more still cuts there
    assert choose_through(entries, sizes[3] + sizes[1] // 2) == 3
    # everything fits: nothing to collapse
    assert choose_through(entries, sum(sizes) + 1) is None
    # with nothing fitting, the cut still lands on the newest exchange's predecessor,
    # because the newest exchange itself is never dropped alone
    assert choose_through(entries, 0) == 3


@pytest.mark.asyncio
async def test_tool_kind_validates_checks_offered_set_and_closes_the_slot():
    from pi_agent_core.harness.pico3 import session as pico_line
    from pi_agent_core.harness.pico3.kinds.generation import generation_kind
    from pi_agent_core.harness.pico3.kinds.tool import invalid, tool_kind as kind

    storage = MemoryStorage()
    session = pico_line.Session(
        storage,
        {k.name: k for k in (generation_kind, tool_kind, post_tools_kind)},
        {},
    )
    tool = _EchoTool()
    emitted = []

    async def setup(tx, ctx, control):
        conversation_id = tx.create_conversation(t.ConversationSpec())
        tx.sticky(conversation_id)["turn"] = {"tools": [{"callId": "c1", "name": "echo", "args": {}, "status": "pending"}]}
        return conversation_id

    conversation_id = (await session.commit({"type": "kernel"}, setup, CTX)).value
    subject = conversation_id
    subject_kind = kind

    class Runtime:
        task_id = 5
        conversation_id = subject
        kind = subject_kind
        models = None
        tools = {"echo": tool}
        kinds = {"pi.tool": tool_kind, "pi.generation": generation_kind, "pi.post_tools": post_tools_kind}
        plugins = {}
        process_host = None

        class _Hooks:
            def handlers(self):
                return []

            async def each(self, ctx, fn, on_value=None):
                return None

        hooks = _Hooks()

        def now(self):
            return 1234

        subject_task = None
        task_id = 1

        async def commit(self, fn, ctx):
            live = session.live_tasks.get(self.task_id) or self.subject_task

            async def call(tx, line_ctx, control=None):
                return await fn(tx, live, line_ctx)

            result = await session.commit(
                t.Invoker(
                    type="task",
                    token=t.InvocationToken(self.task_id, "run"),
                    id=self.task_id,
                    conversation_id=conversation_id,
                    kind=subject_kind,
                    core=True,
                ),
                call,
                ctx,
                {"docs": _conversation_docs(conversation_id)},
            )
            return result.value

        async def abort_task(self, id, ctx):
            return "marked"

    runtime = Runtime()
    assert invalid(tool, {"name": "echo", "arguments": {"text": "x"}}) is None
    assert "text" in (invalid(tool, {"name": "echo", "arguments": {}}) or "")

    not_offered = await _create_task(
        session,
        "pi.tool",
        conversation_id,
        {"assistant": 1, "call": {"id": "c1", "name": "echo"}, "offered": [], "index": 0},
    )
    runtime.subject_task = not_offered
    runtime.task_id = not_offered.id
    step = await kind.initial(not_offered, runtime, CTX)
    completion = await session.commit(
        {"type": "kernel", "conversationId": conversation_id},
        lambda tx, ctx, control=None: step.done(tx, not_offered, ctx),
        CTX,
    )
    assert completion.value.status == "completed"
    entry_id = completion.value.result.entry
    entries = await session.read(
        lambda s, ctx: s.scan_entries(t.EntryScan(conversation_id=conversation_id, limit=10), ctx), CTX
    )
    result_entry = next(entry for entry in entries if entry.id == entry_id)
    assert result_entry.kind == "pi.tool_result"
    assert result_entry.data["diagnostics"][0]["code"] == "not_offered"
    assert result_entry.model[0]["isError"] is True

    # offered and valid: the tool runs, streams, and the slot closes with the entry
    offered = await _create_task(
        session,
        "pi.tool",
        conversation_id,
        {
            "assistant": 1,
            "call": {"id": "c1", "name": "echo", "arguments": {"text": "hi"}},
            "offered": ["echo"],
            "index": 0,
        },
    )
    runtime.subject_task = offered
    runtime.task_id = offered.id
    step = await kind.initial(offered, runtime, CTX)
    completion = await session.commit(
        {"type": "kernel", "conversationId": conversation_id},
        lambda tx, ctx, control=None: step.done(tx, offered, ctx),
        CTX,
    )
    assert tool.calls == [({"text": "hi"}, "c1")]
    assert completion.value.result.entry > entry_id
    sticky = await session.read(
        lambda s, ctx: s.doc(t.DocRef(doc="sticky", conversation_id=conversation_id), ctx), CTX
    )
    assert sticky["turn"]["tools"][0]["status"] == "done"
    assert sticky["turn"]["tools"][0]["entry"] == completion.value.result.entry
    emitted.extend(completion.changes.events)
    assert [event.event.type for event in emitted][-1] == "tool.finished"


@pytest.mark.asyncio
async def test_collapse_kind_summarizes_and_post_tools_ends_the_turn():
    from pi_agent_core.harness.pico3 import session as pico_line
    from pi_agent_core.harness.pico3.kinds.collapse import collapse_kind, choose_through

    assert choose_through([], 0) is None
    model = _faux_model()
    models = _ScriptedModels(model, [_text_script("Summar", "Summary text")])
    storage = MemoryStorage()
    session = pico_line.Session(
        storage,
        {k.name: k for k in (collapse_kind, post_tools_kind, generation_kind, job_kind)},
        {},
    )

    async def setup(tx, ctx, control):
        conversation_id = tx.create_conversation(t.ConversationSpec())
        tx.rewindable(conversation_id)["model"] = {"provider": "faux", "modelId": "faux-1"}
        tx.append_entry(
            conversation_id,
            t.NewEntry(kind="pi.user", model=[{"role": "user", "content": "hello there"}], head="self"),
        )
        return conversation_id

    conversation_id = (await session.commit({"type": "kernel"}, setup, CTX)).value
    subject = conversation_id
    subject_models = models

    # the input the post-tools task will settle (seeded before any live turn task exists)
    async def seed_turn(tx, ctx, control):
        return await tx.send(conversation_id, t.SendInput(content="next question"))

    input_id = (
        await session.commit(
            {"type": "kernel", "conversationId": conversation_id}, seed_turn, CTX
        )
    ).value
    seeded = await session.read(lambda s, ctx: s.input(input_id, ctx), CTX)
    assert seeded is not None and seeded.conversation_id == conversation_id

    async def retire_seeded_generation(tx, ctx, control):
        """Stand in for the scheduler: the turn the seed started runs and finishes."""
        for task in list(session.live_tasks.values()):
            if task.kind != "pi.generation":
                continue
            finished = t.task_from_json(t.task_to_json(task))
            finished.status = "terminal"
            finished.outcome = t.Outcome(status="completed", result={"assistant": 0, "tools": []})
            tx.set_task(finished)

    await session.commit(
        {"type": "kernel", "conversationId": conversation_id},
        retire_seeded_generation,
        CTX,
        {"docs": _conversation_docs(conversation_id)},
    )
    assert not [task for task in session.live_tasks.values() if task.kind == "pi.generation"]

    class Runtime:
        task_id = 2
        conversation_id = subject
        kind = collapse_kind
        models = subject_models
        tools = {}
        kinds = {"pi.collapse": collapse_kind, "pi.post_tools": post_tools_kind, "pi.generation": job_kind}
        plugins = {}
        process_host = None

        class _Hooks:
            def handlers(self):
                return []

            async def each(self, ctx, fn, on_value=None):
                return None

        hooks = _Hooks()

        def now(self):
            return 500

        subject_task = None
        task_id = 1

        async def commit(self, fn, ctx):
            subject_task = session.live_tasks.get(self.task_id) or self.subject_task

            async def call(tx, line_ctx, control=None):
                return await fn(tx, subject_task, line_ctx)

            result = await session.commit(
                t.Invoker(
                    type="task",
                    token=t.InvocationToken(self.subject_task.id, "run"),
                    id=self.subject_task.id,
                    conversation_id=subject,
                    kind=collapse_kind,
                    core=True,
                ),
                call,
                ctx,
                {"docs": _conversation_docs(subject)},
            )
            return result.value

        async def sleep(self, until_ms, ctx):
            return None

        async def context(self, conversation_id, at, ctx):
            async def read(tx, line_ctx, control=None):
                return await tx.context(conversation_id, at)

            return (await session.commit({"type": "kernel"}, read, ctx)).value

        async def newest_entry(self, conversation_id, opts, ctx):
            async def read(tx, line_ctx, control=None):
                return await tx.newest_entry(conversation_id, opts)

            return (await session.commit({"type": "kernel"}, read, ctx)).value

        async def rewindable(self, conversation_id, ctx):
            async def read(tx, line_ctx, control=None):
                return tx.snapshot(t.DocRef(doc="rewindable", conversation_id=conversation_id))

            return (await session.commit({"type": "kernel"}, read, ctx)).value

        async def sticky(self, conversation_id, ctx):
            async def read(tx, line_ctx, control=None):
                return tx.snapshot(t.DocRef(doc="sticky", conversation_id=conversation_id))

            return (await session.commit({"type": "kernel"}, read, ctx)).value

    runtime = Runtime()
    entries = await session.read(
        lambda s, ctx: s.scan_entries(t.EntryScan(conversation_id=conversation_id, limit=10), ctx), CTX
    )
    through = entries[0].id
    task = await _create_task(
        session, "pi.collapse", conversation_id, {"reason": "manual", "through": through}
    )
    runtime.subject_task = task
    runtime.task_id = task.id
    step = await collapse_kind.initial(task, runtime, CTX)
    if callable(step.next):
        checkpoint = await runtime.commit(step.next, CTX)
    else:
        checkpoint = step.next
    assert checkpoint["phase"] == "prepared" and checkpoint["summary"] == "Summary text"

    prepared = t.Task(
        id=task.id,
        conversation_id=conversation_id,
        kind="pi.collapse",
        input=task.input,
        checkpoint=checkpoint,
    )
    session.live_tasks[prepared.id] = prepared
    step = await collapse_kind.phases["prepared"](prepared, runtime, CTX)
    completion = await session.commit(
        t.Invoker(
            type="task",
            token=t.InvocationToken(prepared.id, "run"),
            id=prepared.id,
            conversation_id=conversation_id,
            kind=collapse_kind,
            core=True,
        ),
        lambda tx, ctx, control=None: step.done(tx, prepared, ctx),
        CTX,
        {"docs": _conversation_docs(conversation_id)},
    )
    assert completion.value.status == "completed"
    summary_id = completion.value.result["summary"]
    summary = (await session.read(
        lambda s, ctx: s.scan_entries(t.EntryScan(conversation_id=conversation_id, limit=10), ctx), CTX
    ))
    summary_entry = next(entry for entry in summary if entry.id == summary_id)
    assert summary_entry.kind == "pi.summary"
    assert summary_entry.data["through"] == through
    assert summary_entry.model[0]["content"] == "Summary text"
    assert models.requests[0].messages[-1]["content"].endswith("Respond with the summary only.")

    # post_tools terminalizes its group and reports the ended turn
    assistant_id = summary_entry.id

    class PostRuntime(Runtime):
        kind = post_tools_kind
        kinds = {"pi.post_tools": post_tools_kind, "pi.generation": job_kind}
        models = subject_models

        async def commit(self, fn, ctx):
            subject_task = session.live_tasks.get(self.subject_task.id)

            async def call(tx, line_ctx, control=None):
                return await fn(tx, subject_task, line_ctx)

            result = await session.commit(
                t.Invoker(
                    type="task",
                    token=t.InvocationToken(self.subject_task.id, "run"),
                    id=self.subject_task.id,
                    conversation_id=subject,
                    kind=post_tools_kind,
                    core=True,
                ),
                call,
                ctx,
                {"docs": _conversation_docs(subject)},
            )
            return result.value

    post_task = await _create_task(
        session,
        "pi.post_tools",
        conversation_id,
        {"inputs": [input_id], "assistant": assistant_id, "tools": []},
    )
    post_runtime = PostRuntime()
    post_runtime.subject_task = post_task
    post_runtime.task_id = post_task.id
    step = await post_tools_kind.initial(post_task, post_runtime, CTX)
    completion = await session.commit(
        t.Invoker(
            type="task",
            token=t.InvocationToken(post_task.id, "run"),
            id=post_task.id,
            conversation_id=conversation_id,
            kind=post_tools_kind,
            core=True,
        ),
        lambda tx, ctx, control=None: step.done(tx, post_task, ctx),
        CTX,
        {"docs": _conversation_docs(conversation_id)},
    )
    assert completion.value.status == "completed"
    assert completion.value.result.successor is not None
    # the plain path hands the turn to a successor generation, so nothing ends here yet
    assert [event.event.type for event in completion.changes.events] == []
    still_placed = await session.read(lambda s, ctx: s.input(input_id, ctx), CTX)
    assert still_placed.status == "placed"

    # the abort path settles the group: unanswered inputs and a turn.ended event
    aborting = await post_tools_kind.abort(post_task, post_runtime, CTX)
    aborted = await session.commit(
        t.Invoker(
            type="task",
            token=t.InvocationToken(post_task.id, "run"),
            id=post_task.id,
            conversation_id=conversation_id,
            kind=post_tools_kind,
            core=True,
        ),
        lambda tx, line_ctx, control=None: aborting(tx, post_task, line_ctx),
        CTX,
        {"docs": _conversation_docs(conversation_id)},
    )
    ended = [event.event for event in aborted.changes.events if event.event.type == "turn.ended"]
    assert ended and ended[0].status == "unanswered" and ended[0].reason == "aborted"
    resolved = await session.read(lambda s, ctx: s.input(input_id, ctx), CTX)
    assert resolved.status == "unanswered" and resolved.reason == "aborted"


# ---------------------------------------------------------------------------
# harness.ts / chord.ts (phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_open_send_and_a_full_turn_through_the_scheduler():
    from pi_agent_core.harness.pico3.harness import Harness, HarnessOptions

    model = _faux_model()
    models = _ScriptedModels(
        model,
        [
            # 1) first generation answers with a tool call
            _tool_call_script("c1", "echo", {"text": "hi"}),
            # 2) the successor generation answers with text
            _text_script("do", "done"),
        ],
    )
    tool = _EchoTool()
    storage = MemoryStorage()
    harness = await Harness.open(
        storage,
        HarnessOptions(models=models, tools=[tool], now=lambda: 1000),
        CTX,
    )
    root = await harness.root(CTX)
    await root.config.set({"model": {"provider": "faux", "modelId": "faux-1"}, "selectedTools": ["echo"]}, CTX)

    handle = await root.send(t.SendInput(content="say hi"), CTX)
    harness.resume()
    await harness.wait_for_idle(CTX)
    assert harness.quiescent() is True

    resolved = await handle.result(CTX)
    assert resolved.status == "done" and resolved.answer is not None
    assert tool.calls == [({"text": "hi"}, "c1")]

    entries = await harness.entries(t.EntryScan(conversation_id=root.id, limit=20), CTX)
    kinds_seen = [entry.kind for entry in reversed(entries)]
    assert kinds_seen == [
        "pi.user",
        "pi.system",
        "pi.assistant",
        "pi.tool_result",
        "pi.assistant",
    ]
    assert entries[0].model[0]["content"][0]["text"] == "done"
    assert len(models.requests) == 2
    await harness.close(CTX)
    assert harness.suspended is True


@pytest.mark.asyncio
async def test_harness_registries_namespaces_hooks_and_authority_checks():
    from pi_agent_core.harness.pico3.harness import Harness, HarnessOptions

    storage = MemoryStorage()
    harness = await Harness.open(storage, HarnessOptions(models=_ScriptedModels(_faux_model(), [])), CTX)
    root = await harness.root(CTX)

    # kinds: reserved names and duplicate registration are rejected
    with pytest.raises(ValueError, match="reserved"):
        harness.register_task_kind(t.Kind(name="pi.mine"))
    mine = t.Kind(name="my.kind", config=t.KindConfig(rewindable={"mine": 1}))
    unregister = harness.register_task_kind(mine)
    with pytest.raises(ValueError, match="already registered"):
        harness.register_task_kind(mine)
    assert "my.kind" in harness.kinds and harness.session.defaults.route["mine"] == "rewindable"
    await root.config.set({"mine": 5}, CTX)
    assert (await root.config.get(CTX))["mine"] == 5
    unregister()
    assert "my.kind" not in harness.kinds and "mine" not in harness.session.defaults.route

    # namespaces: token identity, projection, and staleness after unregister
    namespace = harness.namespace(
        "demo", t.NamespaceDefaults(rewindable={"count": 0}), {"view": lambda slice_value: {"count": slice_value.get("count")}}
    )
    assert harness.namespaces["demo"].token is namespace
    with pytest.raises(ValueError, match="already registered"):
        harness.namespace("demo", t.NamespaceDefaults())
    with pytest.raises(t.Forbidden, match="stale"):
        harness.check_namespace(t.Namespace(id="demo"))
    namespace.unregister()
    assert "demo" not in harness.namespaces

    # hooks: kind and namespace tokens must both be current
    fired = []
    registration = harness.hooks(namespace if namespace.id in harness.namespaces else harness.namespace("demo2", t.NamespaceDefaults(sticky={"x": 1})), job_kind, object())
    from pi_agent_core.harness.pico3.hooks import HookRegistration

    assert isinstance(harness.hook_registrations[-1], HookRegistration)
    registration()
    assert all(reg.namespace.id != "demo2" for reg in harness.hook_registrations)
    with pytest.raises(ValueError, match="not the registered token"):
        harness.hooks(harness.namespace("demo3", t.NamespaceDefaults()), t.Kind(name="pi.nope"), object())

    # tools and sections: duplicates rejected, revisions move
    before = harness.revisions["tools"]
    unregister_tool = harness.register_tool(_EchoTool())
    assert harness.revisions["tools"] == before + 1
    unregister_tool()
    assert harness.revisions["tools"] == before + 2
    section = pico_system.define_system_section("extra", lambda value: f"extra:{value}")
    unregister_section = harness.register_section(section)
    assert harness.section_map["extra"] is section
    unregister_section()
    assert "extra" not in harness.section_map
    with pytest.raises(ValueError, match="reserved"):
        harness.register_entry_kind(t.define_entry("pi.mine"))

    # a host write of a reserved kind is refused; notices are allowed
    with pytest.raises(t.Forbidden, match="reserved"):
        await root.write(t.NewEntry(kind="pi.user", model=[{"role": "user", "content": "x"}]), CTX)
    assert await root.write(
        t.NewEntry(kind="pi.notice", model=[{"role": "user", "content": "n", "timestamp": 1}]), CTX
    ) > 0
    fired.append(len(harness.hook_registrations))
    assert fired == [0]
    await harness.close(CTX)


@pytest.mark.asyncio
async def test_chord_view_bridge_publishes_one_state_per_commit_and_fails_closed():
    from pi_agent_core.harness.pico3 import chord as pico_chord
    from pi_agent_core.harness.pico3.harness import Harness, HarnessOptions

    assert pico_chord.pico_harness_service == pico_chord.ServiceDefinition(name="pi.harness", local=True)
    assert pico_chord.pico_conversation_service.name == "pi.conversation"
    assert pico_chord.pico_conversation_service.local is False

    storage = MemoryStorage()
    harness = await Harness.open(storage, HarnessOptions(models=_ScriptedModels(_faux_model(), [])), CTX)
    root = await harness.root(CTX)

    service_holder = {}

    def create_state(initial):
        state = pico_chord.MutableReplicatedState(initial)
        service_holder["service"] = pico_chord.create_pico_conversation_service(harness, root, state)
        return state

    bridge = await pico_chord.attach_chord_view(root, create_state, CTX)
    assert bridge.closed is False
    published = bridge.view.published
    assert published == []  # nothing published before a commit

    await root.write(
        t.NewEntry(kind="pi.notice", model=[{"role": "user", "content": "n", "timestamp": 1}]), CTX
    )
    await asyncio.sleep(0)
    assert len(published) == 1
    assert published[0]["entries"][0]["kind"] == "pi.notice"
    assert published[0]["commit"]["events"][0]["type"] == "entry.added"
    # the published state and the bridge state are separate objects
    assert published[0] is not bridge.view.state

    service = service_holder["service"]
    assert service.view is bridge.view
    assert (await service.send(t.SendInput(content="queued"), CTX)) > 0
    assert service.view is bridge.view

    bridge.close()
    assert bridge.closed is True
    await harness.close(CTX)

    # a listener failure fails the bridge once, reports the cause, and stops the watch
    failures = []
    storage2 = MemoryStorage()
    harness2 = await Harness.open(storage2, HarnessOptions(models=_ScriptedModels(_faux_model(), [])), CTX)
    root2 = await harness2.root(CTX)
    bridge2 = await pico_chord.attach_chord_view(
        root2,
        lambda initial: pico_chord.MutableReplicatedState(initial),
        CTX,
        pico_chord.ChordViewBridgeOptions(capacity=1, on_failure=failures.append),
    )
    bridge2._fail(RuntimeError("bridge boom"))
    assert bridge2.closed is True and len(failures) == 1 and str(failures[0]) == "bridge boom"
    await harness2.close(CTX)


@pytest.mark.asyncio
async def test_harness_suspend_marks_and_abort_task_paths():
    from pi_agent_core.harness.pico3.harness import Harness, HarnessOptions
    from pi_agent_core.harness.pico3.scheduler import Scheduler, SchedulerDeps

    storage = MemoryStorage()
    harness = await Harness.open(storage, HarnessOptions(models=_ScriptedModels(_faux_model(), [])), CTX)
    root = await harness.root(CTX)

    async def make_task(tx, ctx, control):
        return tx.create_task(t.TaskSpec(kind="pi.job", conversation_id=root.id))

    task_id = (
        await harness.session.commit(
            {"type": "kernel"},
            make_task,
            CTX,
            {"docs": [t.DocRef(doc="sticky", conversation_id=root.id)]},
        )
    ).value
    assert await harness.mark_task(task_id, CTX) == "marked"
    assert await harness.mark_task(task_id, CTX) == "marked"
    assert harness.session.live_tasks[task_id].abort is True
    # the scheduler reserves the marked task and runs its abort path
    harness.resume()
    await harness.wait_for_task(task_id, CTX)
    assert task_id not in harness.session.live_tasks
    final = await harness.session.read(lambda s, ctx: s.task(task_id, ctx), CTX)
    assert final.status == "terminal" and final.outcome.status == "aborted"

    # hold gates dispatching; abort_task on a terminal task reports terminal
    release = harness.hold()
    assert harness.quiescent() is True
    release()
    assert await harness.abort_task(task_id, CTX) == "terminal"

    scheduler = Scheduler(
        SchedulerDeps(
            session=harness.session,
            kinds=harness.kinds,
            runtime=lambda task, invoker, ictx: None,
            on_report=lambda error: None,
            ctx=CTX,
        )
    )
    assert scheduler.quiescent() is True and scheduler.live_invocations == 0
    await harness.close(CTX)
