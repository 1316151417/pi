"""End-to-end AgentHarness run: prompt, tools, persistence, resume.

Drives the ported runtime the same way ``AgentHarness.create`` is used in
production: faux provider only, no network, no real API keys.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from pi_agent_core._chord.context import BACKGROUND_CONTEXT
from pi_agent_core._pi_ai.models import Provider, create_models
from pi_agent_core._pi_ai.providers.faux import (
    RegisterFauxProviderOptions,
    create_faux_core,
    faux_assistant_message,
)
from pi_agent_core.harness.agent_harness import AgentHarnessOptions, create_agent_harness
from pi_agent_core.harness.session import MemorySessionRepo, SessionCreateOptions


def _faux_models() -> tuple:
    """A ``Models`` registry holding one faux provider and its handle."""
    handle, functions = create_faux_core(RegisterFauxProviderOptions(provider="faux", api="faux"))
    models = create_models()
    models.set_provider(
        Provider(
            id="faux",
            name="Faux",
            models=handle.models,
            stream=functions["stream"],
            stream_simple=functions["stream_simple"],
            fetch_deferred=functions.get("fetch_deferred"),
            cancel_deferred=functions.get("cancel_deferred"),
        )
    )
    return models, handle


async def _new_harness(models, handle, **overrides):
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="s1"), BACKGROUND_CONTEXT)
    options = {
        "session": session,
        "models": models,
        "model": handle.get_model(),
        "thinking_level": "off",
    }
    options.update(overrides)
    return await create_agent_harness(AgentHarnessOptions(**options), BACKGROUND_CONTEXT)


async def _lane(harness, name: str = "main"):
    """Acquire a lane, surfacing the wrapped cause when the harness faults."""
    try:
        return await harness.lane(name, BACKGROUND_CONTEXT)
    except BaseException as error:  # noqa: BLE001
        import traceback as _tb

        cause = getattr(error, "cause", None)
        if isinstance(cause, BaseException):
            _tb.print_exception(type(cause), cause, cause.__traceback__)
        raise


async def test_prompt_runs_to_completion_and_records_the_turn():
    """One text-only turn settles and its entries are committed to the Branch."""
    models, handle = _faux_models()
    result = await _new_harness(models, handle)
    harness = result["harness"]
    lane = await _lane(harness)
    handle.set_responses([faux_assistant_message("hello there")])
    run = await lane.prompt("hi", BACKGROUND_CONTEXT)

    assert run.ok, getattr(run, "error", None)
    assert run.value.status == "completed"
    assert run.value.error is None
    transcript = [entry for entry in await lane.find_entries(None, BACKGROUND_CONTEXT)]
    roles = [getattr(getattr(entry, "message", None), "role", None) for entry in transcript]
    assert roles == ["assistant", "user"]
    assert run.value.tip_id is not None
    await harness.close(BACKGROUND_CONTEXT)


async def test_run_end_event_reports_the_committed_tip():
    """The event bus observes the same run lifecycle the result reports."""
    models, handle = _faux_models()
    result = await _new_harness(models, handle)
    harness = result["harness"]
    lane = await _lane(harness)

    seen: list = []
    harness.events.on("run_start", lambda event, _ctx: seen.append(("start", event.run_id)))
    harness.events.on("run_end", lambda event, _ctx: seen.append(("end", event.status)))
    watcher = await lane.watch(BACKGROUND_CONTEXT)
    assert watcher.snapshot["tipId"] is None

    handle.set_responses([faux_assistant_message("done")])
    run = await lane.prompt("hi", BACKGROUND_CONTEXT)

    assert run.ok
    assert [kind for kind, _value in seen] == ["start", "end"]
    assert seen[1][1] == "completed"
    refreshed = await watcher.resnapshot(BACKGROUND_CONTEXT)
    assert refreshed["tipId"] == run.value.tip_id
    watcher.unsubscribe()
    await harness.close(BACKGROUND_CONTEXT)


async def test_steer_message_is_drained_at_the_next_boundary():
    """A steered message queued mid-run is consumed by the next boundary."""
    models, handle = _faux_models()
    result = await _new_harness(models, handle)
    harness = result["harness"]
    lane = await _lane(harness)

    handle.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    pending = []

    async def _steer_soon():
        await asyncio.sleep(0)
        pending.append(await lane.steer("more", None, BACKGROUND_CONTEXT))

    await asyncio.gather(_steer_soon(), lane.prompt("hi", BACKGROUND_CONTEXT))
    assert pending and pending[0].ok
    transcript = await lane.find_entries(None, BACKGROUND_CONTEXT)
    user_texts = []
    for entry in transcript:
        message = getattr(entry, "message", None)
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", None)
        user_texts.append(
            content if isinstance(content, str) else "".join(b.text for b in content if getattr(b, "type", None) == "text")
        )
    assert "more" in user_texts
    await harness.close(BACKGROUND_CONTEXT)


async def test_abort_marks_the_run_aborted_without_losing_the_tip():
    """Cancelling an in-flight run settles it as aborted."""
    models, handle = _faux_models()
    result = await _new_harness(models, handle)
    harness = result["harness"]
    lane = await _lane(harness)

    handle.set_responses([faux_assistant_message("slow")])

    async def _abort_soon():
        await asyncio.sleep(0)
        return await lane.abort(BACKGROUND_CONTEXT)

    _run, aborted = await asyncio.gather(lane.prompt("hi", BACKGROUND_CONTEXT), _abort_soon())
    assert aborted.ok, getattr(aborted, "error", None)
    assert aborted.value["operationId"]
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution["current"] is None
    assert execution["lastOperationId"] is not None
    await harness.close(BACKGROUND_CONTEXT)


async def test_session_name_and_entry_labels_round_trip_through_the_harness():
    """Harness-level metadata helpers commit through the session mutation line."""
    models, handle = _faux_models()
    result = await _new_harness(models, handle)
    harness = result["harness"]
    lane = await _lane(harness)

    handle.set_responses([faux_assistant_message("hi")])
    await lane.prompt("hello", BACKGROUND_CONTEXT)
    tip = await lane.get_tip_id(BACKGROUND_CONTEXT)

    assert await harness.get_label(tip, BACKGROUND_CONTEXT) is None
    await harness.set_label(tip, "first turn", BACKGROUND_CONTEXT)
    assert await harness.get_label(tip, BACKGROUND_CONTEXT) == "first turn"
    await harness.set_label(tip, None, BACKGROUND_CONTEXT)
    assert await harness.get_label(tip, BACKGROUND_CONTEXT) is None
    await harness.close(BACKGROUND_CONTEXT)


async def test_jsonl_session_survives_a_reopen(workdir):
    """A JSONL-backed session reopens with its committed tip intact."""
    import os

    from pi_agent_core.harness.env.local import create_local_execution_env
    from pi_agent_core.harness.session.jsonl import (
        JsonlSessionCreateOptions,
        JsonlSessionRepo,
        JsonlSessionRepoOptions,
    )
    from pi_agent_core.harness.session.values import branch_tip

    models, handle = _faux_models()
    env = create_local_execution_env(cwd=workdir)
    repo = JsonlSessionRepo(
        JsonlSessionRepoOptions(
            file_system=env, sessions_root=os.path.join(workdir, "sessions")
        )
    )
    session = await repo.create(
        JsonlSessionCreateOptions(cwd=workdir, id="s1"), BACKGROUND_CONTEXT
    )
    result = await create_agent_harness(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=handle.get_model(),
            thinking_level="off",
        ),
        BACKGROUND_CONTEXT,
    )
    harness = result["harness"]
    lane = await _lane(harness)
    handle.set_responses([faux_assistant_message("persisted")])
    run = await lane.prompt("hi", BACKGROUND_CONTEXT)
    assert run.ok, getattr(run, "error", None)
    tip = run.value.tip_id
    assert tip is not None
    await harness.close(BACKGROUND_CONTEXT)

    listed = await repo.list(None, BACKGROUND_CONTEXT)
    reopened = await repo.open(next(m for m in listed if m.id == "s1"), BACKGROUND_CONTEXT)
    stored = await reopened.get_value(branch_tip("main"), BACKGROUND_CONTEXT)
    assert stored is not None
    assert stored.value == tip
    transcript = await reopened.scan_branch(
        __import__("pi_agent_core.harness.session.types", fromlist=["BranchScan"]).BranchScan(
            start=tip
        ),
        BACKGROUND_CONTEXT,
    )
    roles = [getattr(getattr(entry, "message", None), "role", None) for entry in transcript]
    assert roles == ["assistant", "user"]
    await reopened.close(BACKGROUND_CONTEXT)


@pytest.fixture
def workdir():
    import os

    with tempfile.TemporaryDirectory() as root:
        yield os.path.realpath(root)


async def test_tool_batch_runs_to_completion_and_places_results(workdir):
    """A tool-calling turn executes tools, places results, then finishes."""
    from pi_agent_core._pi_ai.providers.faux import faux_tool_call
    from pi_agent_core.harness.env.local import create_local_execution_env
    from pi_agent_core.harness.tool_adapter import (
        StaticToolContext,
        default_agent_harness_tools,
    )

    models, handle = _faux_models()
    env = create_local_execution_env(cwd=workdir)
    result = await _new_harness(
        models,
        handle,
        tools=default_agent_harness_tools(),
        tool_context=StaticToolContext(env),
    )
    harness = result["harness"]
    lane = await _lane(harness)

    handle.set_responses(
        [
            faux_assistant_message(
                faux_tool_call("write", {"path": "note.txt", "content": "from the tool batch\n"})
            ),
            faux_assistant_message(faux_tool_call("bash", {"command": "cat note.txt"})),
            faux_assistant_message("done"),
        ]
    )
    run = await lane.prompt("write a file then read it", BACKGROUND_CONTEXT)

    assert run.ok, getattr(run, "error", None)
    assert run.value.status == "completed", getattr(run.value.error, "message", None)
    entries = await lane.find_entries(None, BACKGROUND_CONTEXT)
    tool_results = [
        entry
        for entry in entries
        if getattr(getattr(entry, "message", None), "role", None) == "toolResult"
    ]
    assert [entry.message.tool_name for entry in tool_results] == ["bash", "write"]
    bash_output = next(
        block.text
        for block in tool_results[0].message.content
        if getattr(block, "type", None) == "text"
    )
    assert "from the tool batch" in bash_output
    # The tool batch is fully drained, so no arguments remain staged.
    assert (await lane.find_entries(None, BACKGROUND_CONTEXT))[0].type == "message"
    await harness.close(BACKGROUND_CONTEXT)


async def test_tool_batch_survives_a_reopen_mid_operation(workdir):
    """Tool arguments are staged durably, so a reopen restores the same batch."""
    import os

    from pi_agent_core._pi_ai.providers.faux import faux_tool_call
    from pi_agent_core.harness.env.local import create_local_execution_env
    from pi_agent_core.harness.session.jsonl import (
        JsonlSessionCreateOptions,
        JsonlSessionRepo,
        JsonlSessionRepoOptions,
    )
    from pi_agent_core.harness.tool_adapter import (
        StaticToolContext,
        default_agent_harness_tools,
    )

    models, handle = _faux_models()
    env = create_local_execution_env(cwd=workdir)
    repo = JsonlSessionRepo(
        JsonlSessionRepoOptions(
            file_system=env, sessions_root=os.path.join(workdir, "sessions")
        )
    )
    session = await repo.create(
        JsonlSessionCreateOptions(cwd=workdir, id="tb1"), BACKGROUND_CONTEXT
    )
    created = await create_agent_harness(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=handle.get_model(),
            thinking_level="off",
            tools=default_agent_harness_tools(),
            tool_context=StaticToolContext(env),
        ),
        BACKGROUND_CONTEXT,
    )
    harness = created["harness"]
    lane = await _lane(harness)
    handle.set_responses(
        [
            faux_assistant_message(
                faux_tool_call("write", {"path": "a.txt", "content": "alpha\n"})
            ),
            faux_assistant_message("done"),
        ]
    )
    run = await lane.prompt("write a.txt", BACKGROUND_CONTEXT)
    assert run.ok and run.value.status == "completed"
    tip = run.value.tip_id
    await harness.close(BACKGROUND_CONTEXT)

    listed = await repo.list(None, BACKGROUND_CONTEXT)
    reopened = await repo.open(next(m for m in listed if m.id == "tb1"), BACKGROUND_CONTEXT)
    from pi_agent_core.harness.session.values import branch_tip

    stored = await reopened.get_value(branch_tip("main"), BACKGROUND_CONTEXT)
    assert stored.value == tip
    await reopened.close(BACKGROUND_CONTEXT)


async def test_cli_self_test_covers_the_full_runtime(capsys):
    """`pi-agent-core --self-test` proves both runtimes in one command."""
    from pi_agent_core.cli import run_self_test

    assert await run_self_test() == 0
    output = capsys.readouterr().out
    assert "streaming, tool calling, transcript all OK" in output
    assert "harness tools (write/bash) executed on the real filesystem" in output
    assert "AgentHarness runtime ran a tool batch" in output


async def test_suspended_run_is_reported_as_an_open_operation_on_reopen(workdir):
    """A deferred run suspends durably, and reopen reports it as a resumable operation."""
    import os

    from pi_agent_core._pi_ai.providers.faux import (
        RegisterFauxProviderOptions,
        create_faux_core,
    )
    from pi_agent_core._pi_ai.models import Provider, create_models
    from pi_agent_core.harness.env.local import create_local_execution_env
    from pi_agent_core.harness.session.jsonl import (
        JsonlSessionCreateOptions,
        JsonlSessionRepo,
        JsonlSessionRepoOptions,
    )

    # One pending fetch keeps the handle pollable, so the resumed poll streams a
    # real deferred message instead of settling immediately without a start event.
    handle, functions = create_faux_core(
        RegisterFauxProviderOptions(
            provider="faux", api="faux", deferred={"pendingFetches": 1}
        )
    )
    models = create_models()
    models.set_provider(
        Provider(
            id="faux",
            name="Faux",
            models=handle.models,
            stream=functions["stream"],
            stream_simple=functions["stream_simple"],
            fetch_deferred=functions.get("fetch_deferred"),
            cancel_deferred=functions.get("cancel_deferred"),
        )
    )
    env = create_local_execution_env(cwd=workdir)
    repo = JsonlSessionRepo(
        JsonlSessionRepoOptions(
            file_system=env, sessions_root=os.path.join(workdir, "sessions")
        )
    )
    options = AgentHarnessOptions(
        session=await repo.create(
            JsonlSessionCreateOptions(cwd=workdir, id="d1"), BACKGROUND_CONTEXT
        ),
        models=models,
        model=handle.get_model(),
        thinking_level="off",
        stream_options={"deferred": True},
    )
    created = await create_agent_harness(options, BACKGROUND_CONTEXT)
    harness = created["harness"]
    lane = await _lane(harness)

    handle.set_responses([faux_assistant_message("will be deferred")])
    run = await lane.prompt("hi", BACKGROUND_CONTEXT)
    assert run.ok, getattr(run, "error", None)
    assert run.value["status"] == "suspended", run.value
    assert run.value["deferred"].id
    await harness.close(BACKGROUND_CONTEXT)

    listed = await repo.list(None, BACKGROUND_CONTEXT)
    reopened = await repo.open(next(m for m in listed if m.id == "d1"), BACKGROUND_CONTEXT)
    second = await create_agent_harness(
        AgentHarnessOptions(
            session=reopened,
            models=models,
            model=handle.get_model(),
            thinking_level="off",
        ),
        BACKGROUND_CONTEXT,
    )
    assert len(second["open"]) == 1
    operation = second["open"][0]
    assert operation.lane == "main"
    assert operation.kind == "run"
    assert operation.operation_id
    assert operation.aborting is None
    assert operation.to_json()["operationId"] == operation.operation_id
    # The restored lane keeps the suspended operation as its current one.
    restored_lane = await second["harness"].lane("main", BACKGROUND_CONTEXT)
    execution = await restored_lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution["current"]["id"] == operation.operation_id
    assert execution["current"]["status"] == "open"
    assert execution["current"]["kind"] == "run"

    # resume() drives one durable poll. The faux handle had a single pending
    # fetch, so the poll reports the handle again and the run stays suspended
    # under the same operation id — still resumable, with the poll advanced.
    before = await restored_lane.watch(BACKGROUND_CONTEXT)
    assert before.snapshot["operation"]["deferred"]["poll"] == 0
    before.unsubscribe()

    resumed = await restored_lane.resume(BACKGROUND_CONTEXT)
    assert resumed.ok, getattr(resumed, "error", None)
    assert resumed.value["operationId"] == operation.operation_id
    assert resumed.value["status"] == "suspended"
    assert resumed.value["deferred"].id == run.value["deferred"].id

    after = await restored_lane.watch(BACKGROUND_CONTEXT)
    assert after.snapshot["operation"]["id"] == operation.operation_id
    assert after.snapshot["operation"]["status"] == "open"
    assert after.snapshot["operation"]["deferred"]["poll"] == 1
    assert after.snapshot["tipId"] is not None
    after.unsubscribe()
    await second["harness"].close(BACKGROUND_CONTEXT)


async def test_lane_snapshot_mid_stream_reduces_the_committed_frame_prefix():
    """A snapshot taken while a response streams reports the reduced partial message.

    This is the only path that reaches the frame reducer from the lane, so it is
    the one that catches a stale import or a broken frame write.
    """
    from pi_agent_core._pi_ai.models import Provider, create_models
    from pi_agent_core._pi_ai.providers.faux import (
        RegisterFauxProviderOptions,
        create_faux_core,
    )

    # Slow streaming keeps the operation in assistant.effect_pending long enough
    # to snapshot it repeatedly, and frames are committed as the deltas arrive.
    handle, functions = create_faux_core(
        RegisterFauxProviderOptions(
            provider="faux",
            api="faux",
            tokens_per_second=30,
            token_size={"min": 1, "max": 1},
        )
    )
    models = create_models()
    models.set_provider(
        Provider(
            id="faux",
            name="Faux",
            models=handle.models,
            stream=functions["stream"],
            stream_simple=functions["stream_simple"],
        )
    )
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="mid"), BACKGROUND_CONTEXT)
    created = await create_agent_harness(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=handle.get_model(),
            thinking_level="off",
        ),
        BACKGROUND_CONTEXT,
    )
    harness = created["harness"]
    lane = await _lane(harness)
    final_text = "alpha bravo charlie delta echo foxtrot"
    handle.set_responses([faux_assistant_message(final_text)])

    observed: list = []
    stop = False

    async def _sample():
        while not stop:
            await asyncio.sleep(0.005)
            snapshot = await lane.watch(BACKGROUND_CONTEXT)
            operation = snapshot.snapshot["operation"]
            snapshot.unsubscribe()
            if operation is None:
                continue
            partial = operation.get("streamingMessage")
            if partial is not None:
                observed.append(partial)

    sampler = asyncio.ensure_future(_sample())
    prompt_result = await lane.prompt("hi", BACKGROUND_CONTEXT)
    stop = True
    await sampler

    assert prompt_result.ok, getattr(prompt_result, "error", None)
    # Frames must actually reach the durable list; without them the operation has
    # no bounded prefix to recover from after a crash.
    assert observed, "never observed a mid-stream operation snapshot"

    def _text(message) -> str:
        return "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )

    texts = [_text(message) for message in observed]
    assert all(message.role == "assistant" for message in observed)
    assert any(text for text in texts), f"no partial ever carried text: {texts[:5]}"
    # Every partial is a committed prefix of the eventual response, and the
    # prefix grows as deltas land.
    for text in texts:
        assert final_text.startswith(text), (text, final_text)
    assert max(len(text) for text in texts) < len(final_text), "streaming was not observed mid-flight"

    # The final committed entry is the whole response, not a sampled prefix.
    entries = await lane.find_entries(None, BACKGROUND_CONTEXT)
    assert entries[0].message.role == "assistant"
    assert _text(entries[0].message) == final_text
    await harness.close(BACKGROUND_CONTEXT)
