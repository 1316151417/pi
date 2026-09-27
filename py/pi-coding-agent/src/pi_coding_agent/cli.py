"""Small executable entry for the coding-agent session path.

This keeps pi's one-shot text/JSON workflows and uses pi-tui for interactive
sessions. The package manager, extensions and RPC commands are separate layers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from pi_agent_core.types import ThinkingLevel
from pi_ai.types import AssistantMessage, TextContent

from .agent_session import CreateAgentSessionOptions, create_agent_session
from .interactive import run_interactive
from .model_resolver import resolve_cli_model
from .model_runtime import ModelRuntime
from .session_manager import SessionManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi-python", description="Python port of pi's core coding agent")
    parser.add_argument("messages", nargs="*", help="prompts to send in order")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--api-key")
    parser.add_argument("--system-prompt")
    parser.add_argument("--append-system-prompt")
    parser.add_argument("--print", "-p", action="store_true")
    parser.add_argument("--mode", choices=("text", "json"), default="text")
    parser.add_argument("--list-models", nargs="?", const="")
    parser.add_argument("--continue", "-c", dest="continue_session", action="store_true")
    parser.add_argument("--session")
    parser.add_argument("--session-dir")
    parser.add_argument("--no-session", action="store_true")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--tools", help="comma-separated active tool names")
    parser.add_argument("--exclude-tools", help="comma-separated tool names")
    parser.add_argument("--skill", action="append", default=[])
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--no-context-files", action="store_true")
    return parser


def _message_text(message: AssistantMessage) -> str:
    return "\n".join(block.text for block in message.content if isinstance(block, TextContent))


def _last_assistant(messages: Sequence[object]) -> AssistantMessage | None:
    return next((message for message in reversed(messages) if isinstance(message, AssistantMessage)), None)


def _json_event(event: object) -> str:
    if isinstance(event, dict):
        payload = event
    else:
        payload = {"type": getattr(event, "type", "unknown")}
        message = getattr(event, "message", None)
        if message is not None:
            payload["message"] = message.to_json() if hasattr(message, "to_json") else str(message)
        update = getattr(event, "assistant_message_event", None)
        if update is not None:
            payload["assistantMessageEvent"] = update.to_json()
    return json.dumps(payload, ensure_ascii=False, default=str)


async def run(argv: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    if parsed.session and parsed.continue_session:
        raise ValueError("--session and --continue cannot be combined")
    if parsed.no_session and (parsed.session or parsed.continue_session):
        raise ValueError("--no-session cannot be combined with --session or --continue")
    runtime = await ModelRuntime.create()
    if parsed.list_models is not None:
        search = parsed.list_models.lower()
        for model in runtime.get_models():
            reference = f"{model.provider}/{model.id}"
            if search in reference.lower() or search in model.name.lower():
                print(reference)
        return 0
    if parsed.api_key:
        if not parsed.provider:
            raise ValueError("--api-key requires --provider")
        await runtime.set_runtime_api_key(parsed.provider, parsed.api_key)
    cwd = os.getcwd()
    if parsed.no_session:
        manager = SessionManager.in_memory(cwd)
    elif parsed.session:
        if not Path(parsed.session).is_file():
            raise FileNotFoundError(f"Session not found: {parsed.session}")
        manager = SessionManager.open(parsed.session, parsed.session_dir)
    elif parsed.continue_session:
        manager = SessionManager.continue_recent(cwd, parsed.session_dir)
    else:
        manager = SessionManager.create(cwd, parsed.session_dir)
    model = None
    thinking = parsed.thinking
    if parsed.model:
        resolved = await resolve_cli_model(runtime, parsed.model, parsed.provider, parsed.thinking)
        if resolved.error:
            raise ValueError(resolved.error)
        if resolved.warning:
            print(f"Warning: {resolved.warning}", file=sys.stderr)
        model = resolved.model
        thinking = thinking or resolved.thinking_level
    elif parsed.provider:
        available = await runtime.get_available(parsed.provider)
        model = available[0] if available else None
    active_tools = parsed.tools.split(",") if parsed.tools else None
    excluded_tools = parsed.exclude_tools.split(",") if parsed.exclude_tools else []
    result = await create_agent_session(CreateAgentSessionOptions(
        cwd=manager.get_cwd(), models=runtime, model=model,
        thinking_level=thinking, session_manager=manager,
        tools=active_tools, exclude_tools=excluded_tools, no_tools=parsed.no_tools,
        custom_prompt=parsed.system_prompt, append_system_prompt=parsed.append_system_prompt,
        skill_paths=parsed.skill, no_skills=parsed.no_skills,
        no_context_files=parsed.no_context_files,
    ))
    session = result.session
    if result.model_fallback_message:
        print(f"Warning: {result.model_fallback_message}", file=sys.stderr)
    if session.model is None:
        raise RuntimeError("No model available. Set a provider API key or configure models.json.")
    if parsed.mode == "json":
        session.subscribe(lambda event: print(_json_event(event), flush=True))
    try:
        prompts = list(parsed.messages)
        if not sys.stdin.isatty():
            piped = sys.stdin.read().strip()
            if piped:
                prompts.insert(0, piped)
        if not prompts and not parsed.print and sys.stdin.isatty():
            if parsed.mode == "text":
                await run_interactive(session)
            else:
                while True:
                    try:
                        prompt = input("pi> ")
                    except (EOFError, KeyboardInterrupt):
                        break
                    if prompt.strip() in ("/exit", "/quit"):
                        break
                    if prompt.strip():
                        await session.prompt(prompt)
            return 0
        for prompt in prompts:
            await session.prompt(prompt)
        if parsed.mode == "text":
            answer = _last_assistant(session.messages)
            if answer is not None:
                if answer.stop_reason in ("error", "aborted"):
                    print(answer.error_message or f"Request {answer.stop_reason}", file=sys.stderr)
                    return 1
                print(_message_text(answer))
        return 0
    finally:
        session.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(run(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


__all__ = ["main", "run"]
