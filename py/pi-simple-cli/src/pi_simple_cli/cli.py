"""Executable demo CLI for the pi-agent-core Python port.

Run modes:

* ``pi``                       -> interactive chat with the faux model
* ``pi --provider anthropic``  -> real Anthropic API (ANTHROPIC_API_KEY)
* ``pi --provider openai``     -> real OpenAI API (OPENAI_API_KEY)
* ``pi --prompt "hi"``         -> one-shot, then exit
* ``pi --self-test``           -> offline self test with the faux provider

The demo ships a small tool set (echo, get_time) so tool-calling can be
exercised end to end. Shipped as the ``pi`` command by the ``pi-simple-cli``
package.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from typing import Any, List, Optional

from pi_agent_core import (
    Agent,
    AgentOptions,
    AgentInitialState,
    AgentTool,
    AgentToolResult,
    set_default_stream_fn,
)
from pi_ai.models import Models, create_models
from pi_ai.providers.anthropic import anthropic_provider
from pi_ai.providers.faux import (
    FauxModelDefinition,
    RegisterFauxProviderOptions,
    create_faux_core,
    faux_assistant_message,
    faux_tool_call,
    faux_text,
)
from pi_ai.providers.openai_completions import openai_provider
from pi_ai.types import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)
from pi_agent_core.harness.env.local import create_local_execution_env
from pi_agent_core.harness.session.jsonl import (
    JsonlSessionListOptions,
    JsonlSessionCreateOptions,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
)
from pi_agent_core.harness.tool_adapter import (
    StaticToolContext,
    default_agent_harness_tools,
    default_harness_tools,
)

FAUX_CHAT_SCRIPT = [
    faux_assistant_message("Hello from the faux model! I can echo text or tell the time."),
    faux_assistant_message(
        [faux_tool_call("get_time", {})],
        {"stopReason": "toolUse"},
    ),
    faux_assistant_message("Tools work end to end. Ask a real provider for a real conversation."),
    faux_assistant_message("That's everything the offline demo can show. Bye!"),
]

FAUX_HARNESS_SCRIPT = [
    faux_assistant_message(
        [faux_tool_call("write", {"path": "note.txt", "content": "written by pi-agent-core harness tools\n"})],
        {"stopReason": "toolUse"},
    ),
    faux_assistant_message(
        [faux_tool_call("bash", {"command": "cat note.txt && rm note.txt"})],
        {"stopReason": "toolUse"},
    ),
    faux_assistant_message("Harness tools (write/bash) executed against the real filesystem."),
]


def register_faux(models: Models) -> None:
    handle, streams = create_faux_core(
        RegisterFauxProviderOptions(api="faux", provider="faux", tokens_per_second=80)
    )
    handle.set_responses(list(FAUX_CHAT_SCRIPT))

    from pi_ai.models import Provider

    faux_provider_instance = Provider(
        id="faux",
        name="Faux",
        models=handle.models,
        stream=streams["stream"],
        stream_simple=streams["stream_simple"],
    )
    models.set_provider(faux_provider_instance)


def build_demo_tools() -> List[AgentTool]:
    async def echo(tool_call_id: str, params: Any, signal: Any, on_update: Any) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text=f"echo: {params['text']}")],
            details={"echoed": params["text"]},
        )

    async def get_time(tool_call_id: str, params: Any, signal: Any, on_update: Any) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))],
            details={},
        )

    return [
        AgentTool(
            name="echo",
            label="Echo",
            description="Echo the given text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            execute=echo,
        ),
        AgentTool(
            name="get_time",
            label="Get time",
            description="Get the current local time.",
            parameters={"type": "object", "properties": {}, "required": []},
            execute=get_time,
        ),
    ]


def build_models(
    provider: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Models:
    """Assemble the Models registry for one CLI invocation.

    ``--base-url`` (or ``OPENAI_BASE_URL``) re-points the OpenAI provider at any
    OpenAI-compatible endpoint — DeepSeek, vLLM, Ollama, LM Studio and friends —
    and lets ``--model`` name an id the built-in catalog does not know: a Model
    is synthesized against that endpoint instead of failing resolution.
    """
    models = create_models()
    if provider == "anthropic":
        models.set_provider(anthropic_provider())
    elif provider == "openai":
        from pi_ai.models import ProviderAuth
        from pi_ai.types import Model

        effective_base = base_url or os.environ.get("OPENAI_BASE_URL")
        openai = openai_provider(base_url=effective_base)
        if api_key:
            openai.auth = ProviderAuth(env_var="OPENAI_API_KEY", explicit_key=api_key)
        if effective_base and model_id and model_id not in {m.id for m in openai.get_models()}:
            openai.add_model(
                Model(
                    id=model_id,
                    name=model_id,
                    api="openai-completions",
                    provider="openai",
                    base_url=effective_base,
                    input=["text"],
                    context_window=128_000,
                    max_tokens=8_192,
                )
            )
        models.set_provider(openai)
    else:
        register_faux(models)
    return models


def resolve_model(models: Models, provider: str, model_id: Optional[str]) -> Any:
    available = models.get_models(provider)
    if not available:
        raise SystemExit(f"Provider {provider!r} exposes no models")
    model = next((m for m in available if m.id == model_id), None) if model_id else available[0]
    if model is None:
        ids = ", ".join(m.id for m in available)
        raise SystemExit(f"Unknown model {model_id!r} for provider {provider!r}. Available: {ids}")
    return model


def make_print_listener() -> Any:
    state = {"in_text": False, "in_thinking": False}

    async def listener(event: Any, signal: Any) -> None:
        etype = event.type
        if etype == "message_update":
            inner = event.assistant_message_event
            if inner is None:
                return
            if inner.type == "thinking_delta":
                if not state["in_thinking"]:
                    state["in_thinking"] = True
                    sys.stdout.write("\033[2m[thinking] ")
                sys.stdout.write(inner.delta or "")
                sys.stdout.flush()
            elif inner.type == "thinking_end":
                if state["in_thinking"]:
                    state["in_thinking"] = False
                    sys.stdout.write("\033[0m\n")
            elif inner.type == "text_delta":
                if state["in_thinking"]:
                    state["in_thinking"] = False
                    sys.stdout.write("\033[0m\n")
                state["in_text"] = True
                sys.stdout.write(inner.delta or "")
                sys.stdout.flush()
        elif etype == "message_end":
            message = event.message
            if isinstance(message, AssistantMessage):
                if state["in_text"] or state["in_thinking"]:
                    state["in_text"] = False
                    state["in_thinking"] = False
                    sys.stdout.write("\n")
                if message.stop_reason in ("error", "aborted") and message.error_message:
                    sys.stdout.write(f"\033[31m[{message.stop_reason}] {message.error_message}\033[0m\n")
                usage = message.usage
                if usage and (usage.input or usage.output):
                    sys.stdout.write(
                        f"\033[90m[{message.model} | in={usage.input} out={usage.output} "
                        f"cost=${usage.cost.total:.6f} | stop={message.stop_reason}]\033[0m\n"
                    )
                sys.stdout.flush()
        elif etype == "tool_execution_start":
            sys.stdout.write(f"\033[36m[{event.tool_name}({event.args})]\033[0m\n")
            sys.stdout.flush()

    return listener


def make_harness_event_listener() -> Any:
    """Render the AgentHarness runtime's event stream for the terminal."""
    quiet = {
        "usage",
        "value_update",
        "config_update",
        "queue_update",
        "turn_start",
        "handler_error",
    }

    async def listener(event: Any, _context: Any) -> None:
        etype = event.type
        if etype == "message_end":
            message = event.message
            if isinstance(message, AssistantMessage):
                sys.stdout.write("\n")
                if message.stop_reason in ("error", "aborted") and message.error_message:
                    sys.stdout.write(f"\033[31m[{message.stop_reason}] {message.error_message}\033[0m\n")
                usage = message.usage
                if usage and (usage.input or usage.output):
                    sys.stdout.write(
                        f"\033[90m[{message.model} | in={usage.input} out={usage.output} "
                        f"cost=${usage.cost.total:.6f} | stop={message.stop_reason}]\033[0m\n"
                    )
                sys.stdout.flush()
        elif etype == "run_end":
            if event.status != "completed":
                detail = f": {event.error.message}" if event.error is not None else ""
                sys.stdout.write(f"\033[31m[run {event.status}{detail}]\033[0m\n")
        elif etype == "compaction_start":
            sys.stdout.write(f"\033[90m[compacting ({event.reason})]\033[0m\n")
        elif etype == "retry_scheduled":
            sys.stdout.write(
                f"\033[33m[retry {event.attempt}/{event.max_attempts} in {event.delay_ms}ms: "
                f"{event.error_message}]\033[0m\n"
            )
        elif etype == "fault":
            sys.stdout.write(f"\033[31m[harness fault] {event.message}\033[0m\n")
        elif etype == "turn_end":
            for result in event.tool_results or []:
                sys.stdout.write(f"\033[36m[{result.tool_name} result]\033[0m\n")
        sys.stdout.flush()

    return listener


async def run_harness_repl(
    harness: Any,
    lane: Any,
    initial_prompt: Optional[str],
) -> None:
    """Interactive REPL driving the durable AgentHarness runtime."""
    sys.stdout.write(
        "pi (Python port, AgentHarness runtime). /quit /state /lanes /abort\n"
    )
    if initial_prompt is not None:
        await _harness_turn(lane, initial_prompt)
    loop = asyncio.get_running_loop()
    while True:
        sys.stdout.write("\x1b[1myou>\x1b[0m ")
        sys.stdout.flush()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/state":
            snapshot = await lane.watch(_ctx())
            sys.stdout.write(f"  lane={snapshot.snapshot['lane']} tip={snapshot.snapshot['tipId']}\n")
            for entry in snapshot.snapshot["transcript"]:
                sys.stdout.write(f"  {getattr(getattr(entry, 'message', None), 'role', entry.type)}\n")
            snapshot.unsubscribe()
            continue
        if text == "/lanes":
            for info in await harness.lanes(_ctx()):
                current = info.operation["id"] if info.operation else None
                sys.stdout.write(f"  {info.name} tip={info.tip_id} operation={current}\n")
            continue
        if text == "/abort":
            result = await lane.abort(_ctx())
            sys.stdout.write(f"  abort ok={result.ok}\n")
            continue
        await _harness_turn(lane, text)


async def _harness_turn(lane: Any, text: str) -> None:
    """Run one prompt and report a non-completed outcome verbatim."""
    try:
        result = await lane.prompt(text, _ctx())
    except BaseException as error:  # noqa: BLE001
        sys.stdout.write(f"\033[31m[harness error] {error}\033[0m\n")
        return
    if not result.ok:
        sys.stdout.write(f"\033[31m[rejected] {result.error}\033[0m\n")


async def run_harness(
    models: Models,
    model: Any,
    args: Any,
    tools: List[AgentTool],
    log_session: Any,
    tool_context: Any = None,
) -> int:
    """Drive the full AgentHarness runtime instead of the bare agent loop."""
    from pi_agent_core.harness.agent_harness import AgentHarnessOptions, create_agent_harness
    from pi_agent_core.harness.tool_adapter import (
        StaticToolContext,
        default_agent_harness_tools,
        default_harness_tools,
    )

    if log_session is None:
        raise SystemExit("--runtime harness requires --session or --resume")

    harness_tools = tools
    result = await create_agent_harness(
        AgentHarnessOptions(
            session=log_session,
            models=models,
            model=model,
            thinking_level=args.thinking,
            system_prompt=args.system_prompt,
            tools=harness_tools,
            tool_context=tool_context,
            resources={"skills": [], "prompt_templates": []},
        ),
        _ctx(),
    )
    harness = result["harness"]
    harness.events.on("message_start", make_harness_message_start_listener())
    harness.events.on("message_update", make_harness_message_update_listener())
    harness.events.on("message_end", make_harness_event_listener())
    for event_type in (
        "turn_start",
        "compaction_start",
        "retry_scheduled",
        "retry_start",
        "run_end",
        "fault",
        "run_resume",
        "run_suspend",
    ):
        harness.events.on(event_type, make_harness_event_listener())
    if result["open"]:
        for operation in result["open"]:
            sys.stdout.write(
                f"\033[33m[restored open {operation.kind} {operation.operation_id} on {operation.lane}]\033[0m\n"
            )
    lane = await harness.lane("main", _ctx())
    try:
        if args.prompt is not None:
            await _harness_turn(lane, args.prompt)
            return 0
        await run_harness_repl(harness, lane, None)
        return 0
    finally:
        await harness.close(_ctx())


def make_harness_message_start_listener() -> Any:
    """Prompt the user, for assistant messages only.

    Every committed message entry emits ``message_start``, including the user
    prompt, so the renderer filters on the message role.
    """

    async def listener(event: Any, _context: Any) -> None:
        if getattr(event.message, "role", None) != "assistant":
            return
        sys.stdout.write("\x1b[1massistant>\x1b[0m ")
        sys.stdout.flush()

    return listener


def make_harness_message_update_listener() -> Any:
    async def listener(event: Any, _context: Any) -> None:
        inner = event.event
        if inner is None:
            return
        if inner.type == "text_delta":
            sys.stdout.write(inner.delta or "")
            sys.stdout.flush()
        elif inner.type == "thinking_delta":
            sys.stdout.write(inner.delta or "")
            sys.stdout.flush()

    return listener


async def run_interactive(agent: Agent, initial_prompt: Optional[str]) -> None:
    sys.stdout.write("pi (Python port). /quit to exit, /reset to clear, /state for state.\n")
    if initial_prompt:
        await agent.prompt(initial_prompt)
    loop = asyncio.get_running_loop()
    while True:
        sys.stdout.write("\x1b[1myou>\x1b[0m ")
        sys.stdout.flush()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            agent.reset()
            sys.stdout.write("state reset\n")
            continue
        if text == "/state":
            for message in agent.state.messages:
                role = getattr(message, "role", "?")
                sys.stdout.write(f"  {role}\n")
            continue
        sys.stdout.write("\x1b[1massistant>\x1b[0m ")
        await agent.prompt(text)


async def run_self_test() -> int:
    models = build_models("faux")
    model = resolve_model(models, "faux", None)
    agent = Agent(
        AgentOptions(
            stream_fn=models.stream_simple,
            initial_state=AgentInitialState(
                system_prompt="You are a helpful assistant.", model=model, tools=build_demo_tools()
            ),
        )
    )
    agent.subscribe(make_print_listener())

    await agent.prompt("Hi!")
    await agent.prompt("What time is it?")
    await agent.prompt("Anything else?")

    roles = [getattr(m, "role", None) for m in agent.state.messages]
    tool_results = [m for m in agent.state.messages if isinstance(m, ToolResultMessage)]
    assert "toolResult" in roles, f"expected a tool result in transcript, got roles: {roles}"
    assert len(tool_results) == 1, f"expected exactly one tool result, got {len(tool_results)}"
    assert tool_results[0].tool_name == "get_time"
    assert agent.state.error_message is None
    sys.stdout.write("self-test passed: streaming, tool calling, transcript all OK\n")

    # Harness self test: real filesystem tools driven by scripted faux turns.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        handle, streams = create_faux_core(
            RegisterFauxProviderOptions(api="faux-harness", provider="faux-harness")
        )
        handle.set_responses(list(FAUX_HARNESS_SCRIPT))

        from pi_ai.models import Provider

        models.set_provider(
            Provider(
                id="faux-harness",
                name="Faux Harness",
                models=handle.models,
                stream=streams["stream"],
                stream_simple=streams["stream_simple"],
            )
        )
        harness_model = handle.get_model()
        env = create_local_execution_env(cwd=tmp)
        harness_agent = Agent(
            AgentOptions(
                stream_fn=models.stream_simple,
                initial_state=AgentInitialState(
                    system_prompt="You are a helpful assistant.",
                    model=harness_model,
                    tools=default_harness_tools(env),
                ),
            )
        )
        harness_agent.subscribe(make_print_listener())
        await harness_agent.prompt("write a note then read it back with bash")

        harness_results = [
            m
            for m in harness_agent.state.messages
            if isinstance(m, ToolResultMessage) and not m.is_error
        ]
        executed = [r.tool_name for r in harness_results]
        assert "write" in executed and "bash" in executed, f"harness tools did not run: {executed}"
        bash_text = next(r.content[0].text for r in harness_results if r.tool_name == "bash")
        assert "written by pi-agent-core harness tools" in bash_text
        assert harness_agent.state.error_message is None
        sys.stdout.write("self-test passed: harness tools (write/bash) executed on the real filesystem\n")

    await _self_test_agent_harness_runtime()
    return 0


async def _self_test_agent_harness_runtime() -> None:
    """Prove the durable AgentHarness runtime end to end on a JSONL session."""
    from pi_ai.models import Provider
    from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
    from pi_agent_core.harness.agent_harness import AgentHarnessOptions, create_agent_harness
    from pi_agent_core.harness.session.jsonl import (
        JsonlSessionCreateOptions,
        JsonlSessionRepo,
        JsonlSessionRepoOptions,
    )
    from pi_agent_core.harness.session.values import branch_tip
    from pi_agent_core.harness.tool_adapter import StaticToolContext, default_agent_harness_tools

    with tempfile.TemporaryDirectory() as tmp:
        handle, functions = create_faux_core(
            RegisterFauxProviderOptions(api="faux-runtime", provider="faux-runtime")
        )
        models = create_models()
        models.set_provider(
            Provider(
                id="faux-runtime",
                name="Faux Runtime",
                models=handle.models,
                stream=functions["stream"],
                stream_simple=functions["stream_simple"],
            )
        )
        env = create_local_execution_env(cwd=tmp)
        repo = JsonlSessionRepo(
            JsonlSessionRepoOptions(file_system=env, sessions_root=os.path.join(tmp, "sessions"))
        )
        session = await repo.create(JsonlSessionCreateOptions(cwd=tmp, id="self-test"), _ctx())
        created = await create_agent_harness(
            AgentHarnessOptions(
                session=session,
                models=models,
                model=handle.get_model(),
                system_prompt="You are a helpful assistant.",
                tools=default_agent_harness_tools(),
                tool_context=StaticToolContext(env),
            ),
            _ctx(),
        )
        harness = created["harness"]
        assert created["open"] == [], f"unexpected open operations: {created['open']}"
        events: List[str] = []
        harness.events.on("run_end", lambda event, _ctx: events.append(event.status))

        lane = await harness.lane("main", _ctx())
        handle.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call(
                        "write", {"path": "note.txt", "content": "durable harness runtime\n"}
                    )
                ),
                faux_assistant_message("wrote the note"),
            ]
        )
        run = await lane.prompt("write note.txt", _ctx())
        assert run.ok and run.value.status == "completed", f"run did not complete: {run}"
        assert events == ["completed"], f"unexpected run events: {events}"

        entries = await lane.find_entries(None, _ctx())
        roles = [getattr(getattr(entry, "message", None), "role", None) for entry in entries]
        assert roles.count("toolResult") == 1, f"tool result was not placed: {roles}"
        tip = run.value.tip_id
        await harness.close(_ctx())

        listed = await repo.list(None, _ctx())
        reopened = await repo.open(next(m for m in listed if m.id == "self-test"), _ctx())
        stored = await reopened.get_value(branch_tip("main"), _ctx())
        assert stored is not None and stored.value == tip, "reopened session lost the tip"
        await reopened.close(_ctx())
        await repo.close(_ctx())
        sys.stdout.write(
            "self-test passed: AgentHarness runtime ran a tool batch, settled the run, "
            "and the JSONL session reopened at the same tip\n"
        )


async def async_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pi", description="pi demo CLI (pi-agent-core Python port)")
    parser.add_argument("--provider", default="faux", choices=["faux", "anthropic", "openai"])
    parser.add_argument(
        "--base-url",
        default=None,
        help="override the provider endpoint (OpenAI-compatible APIs: DeepSeek, vLLM, Ollama, ...; "
        "also read from OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the provider (prefer the provider's environment variable in scripts)",
    )
    parser.add_argument("--model", default=None, help="model id (defaults to the provider's first model)")
    parser.add_argument("--prompt", default=None, help="one-shot prompt; omit for interactive REPL")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--thinking", default="off", choices=["off", "minimal", "low", "medium", "high"])
    parser.add_argument("--cwd", default=None, help="working directory for harness tools (defaults to cwd)")
    parser.add_argument(
        "--tools",
        default="harness",
        choices=["harness", "demo", "none"],
        help="harness: real bash/read/write/edit; demo: echo/get_time; none",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="persist this conversation to a JSONL session with the given id (requires --sessions-root)",
    )
    parser.add_argument(
        "--sessions-root",
        default=None,
        help="directory holding JSONL sessions (default: <cwd>/.pi-sessions)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the newest existing session for --cwd instead of creating a new one",
    )
    parser.add_argument(
        "--runtime",
        default="agent",
        choices=["agent", "harness"],
        help="agent: the bare agent loop; harness: the durable AgentHarness lane runtime",
    )
    parser.add_argument("--self-test", action="store_true", help="offline self test with the faux provider")
    args = parser.parse_args(argv)

    if args.self_test:
        return await run_self_test()

    models = build_models(args.provider, base_url=args.base_url, api_key=args.api_key, model_id=args.model)
    model = resolve_model(models, args.provider, args.model)

    if args.provider != "faux" and not models.is_configured(args.provider):
        env_var = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        raise SystemExit(f"Provider {args.provider!r} requires {env_var} to be set")

    if args.tools == "harness":
        env = create_local_execution_env(cwd=args.cwd)
        tools = (
            default_agent_harness_tools()
            if args.runtime == "harness"
            else default_harness_tools(env)
        )
    elif args.tools == "demo":
        tools = build_demo_tools()
    else:
        tools = []

    session_repo = None
    log_session = None
    if args.session or args.resume:
        workdir = args.cwd or os.getcwd()
        sessions_root = args.sessions_root or os.path.join(workdir, ".pi-sessions")
        session_repo = JsonlSessionRepo(
            JsonlSessionRepoOptions(file_system=env, sessions_root=sessions_root)
        )
        if args.resume:
            existing = await session_repo.list(JsonlSessionListOptions(cwd=workdir), _ctx())
            if not existing:
                raise SystemExit(f"No sessions to resume under {sessions_root}")
            log_session = await session_repo.open(existing[0], _ctx())
            sys.stdout.write(f"resumed session {log_session.metadata.id} ({log_session.metadata.path})\n")
        else:
            log_session = await session_repo.create(
                JsonlSessionCreateOptions(cwd=workdir, id=args.session), _ctx()
            )
            sys.stdout.write(f"session {log_session.metadata.id} -> {log_session.metadata.path}\n")

    if args.runtime == "harness":
        try:
            return await run_harness(
                models,
                model,
                args,
                tools,
                log_session,
                tool_context=StaticToolContext(env) if env is not None else None,
            )
        finally:
            if log_session is not None:
                await log_session.close(_ctx())
            if session_repo is not None:
                await session_repo.close(_ctx())

    agent = Agent(
        AgentOptions(
            stream_fn=models.stream_simple,
            initial_state=AgentInitialState(
                system_prompt=args.system_prompt,
                model=model,
                thinking_level=args.thinking,
                tools=tools,
            ),
        )
    )
    agent.subscribe(make_print_listener())
    if log_session is not None:
        agent.subscribe(make_session_listener(log_session))

    try:
        if args.prompt is not None:
            sys.stdout.write("\x1b[1massistant>\x1b[0m ")
            await agent.prompt(args.prompt)
            return 0
        await run_interactive(agent, None)
        return 0
    finally:
        if log_session is not None:
            await log_session.close(_ctx())
        if session_repo is not None:
            await session_repo.close(_ctx())


def _ctx():
    """Root chord context for CLI-driven session work."""
    from pi_agent_core._chord.context import BACKGROUND_CONTEXT

    return BACKGROUND_CONTEXT


def make_session_listener(session) -> Any:
    """Append finished messages to a JSONL session through its main branch."""
    branch_holder: dict = {}

    async def _branch():
        if "branch" not in branch_holder:
            branch = await session.branch("main", _ctx())
            if branch is None:
                branch = await session.create_branch("main", None, _ctx())
            branch_holder["branch"] = branch
        return branch_holder["branch"]

    async def listener(event: Any, signal: Any) -> None:
        if event.type != "message_end":
            return
        message = event.message
        role = getattr(message, "role", None)
        if role not in ("user", "assistant"):
            return
        if role == "assistant" and getattr(message, "stop_reason", None) in ("error", "aborted"):
            return
        try:
            branch = await _branch()
            await branch.append_message(message, _ctx())
        except Exception as error:  # noqa: BLE001 - persistence must not break the run
            sys.stderr.write(f"[session] failed to persist message: {error}\n")

    return listener


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
