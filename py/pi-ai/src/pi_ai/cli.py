"""OAuth command line entry point from ``packages/ai/src/cli.ts``."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import cast

from ._javascript import javascript_json_parse, javascript_json_stringify, javascript_string
from ._json_runtime import JS_WHITESPACE
from .abort import AbortController
from .auth.types import (
    AuthDeviceCodeEvent, AuthEvent, AuthInfoEvent, AuthProgressEvent, AuthPrompt,
    AuthUrlEvent, ProviderAuthInteraction, SelectAuthPrompt,
)
from .providers.all import builtin_providers

AUTH_FILE = "auth.json"
_PROVIDERS = [provider for provider in builtin_providers() if provider.auth.oauth is not None]


async def _prompt(question: str) -> str:
    # Like readline.question, this prompt intentionally echoes secret input.
    return await asyncio.to_thread(input, question)


def _load_auth() -> dict[str, object]:
    path = Path(AUTH_FILE)
    if not path.exists():
        return {}
    try:
        return cast(dict[str, object], javascript_json_parse(path.read_bytes().decode("utf-8", errors="replace")))
    except Exception:
        return {}


def _selection_index(value: str) -> int | None:
    match = re.match(r"^[+-]?[0-9]+", value.lstrip(JS_WHITESPACE))
    return int(match[0]) - 1 if match is not None else None


async def _answer_prompt(prompt: AuthPrompt) -> str:
    if isinstance(prompt, SelectAuthPrompt):
        print(f"\n{prompt.message}")
        for index, option in enumerate(prompt.options):
            print(f"  {index + 1}. {option.label}")
        choice = _selection_index(await _prompt(f"Enter number (1-{len(prompt.options)}): "))
        if choice is None or choice < 0 or choice >= len(prompt.options):
            raise RuntimeError("Invalid selection")
        return prompt.options[choice].id
    placeholder = f" ({prompt.placeholder})" if prompt.placeholder else ""
    return await _prompt(f"{prompt.message}{placeholder}: ")


async def login(provider_id: str) -> None:
    provider = next((entry for entry in _PROVIDERS if entry.id == provider_id), None)
    if provider is None or provider.auth.oauth is None:
        raise RuntimeError(f"Unknown provider: {provider_id}")

    def notify(event: AuthEvent) -> None:
        if isinstance(event, AuthUrlEvent):
            print(f"\nOpen this URL in your browser:\n{event.url}")
            if event.instructions:
                print(event.instructions)
        elif isinstance(event, AuthDeviceCodeEvent):
            print(f"\nOpen this URL in your browser:\n{event.verification_uri}")
            print(f"Enter code: {event.user_code}")
        elif isinstance(event, (AuthInfoEvent, AuthProgressEvent)):
            print(event.message)

    credential = await provider.auth.oauth.login(ProviderAuthInteraction(
        signal=AbortController().signal, prompt=_answer_prompt, notify=notify,
    ))
    auth = _load_auth()
    auth[provider_id] = credential.to_json()
    Path(AUTH_FILE).write_text(cast(str, javascript_json_stringify(auth, space=2)), encoding="utf-8")
    print(f"\nCredentials saved to {AUTH_FILE}")


async def main(args: list[str] | None = None) -> None:
    args = sys.argv[1:] if args is None else args
    command = args[0] if args else None
    if not command or command in ("help", "--help", "-h"):
        providers = "\n".join(f"  {provider.id.ljust(20)} {provider.name}" for provider in _PROVIDERS)
        print("Usage: python -m pi_ai.cli <command> [provider]\n\nCommands:\n"
            "  login [provider]  Login to an OAuth provider\n  list              List available providers\n\n"
            f"Providers:\n{providers}")
        return
    if command == "list":
        for provider in _PROVIDERS:
            print(f"{provider.id.ljust(20)} {provider.name}")
        return
    if command == "login":
        provider_id = args[1] if len(args) > 1 else None
        if not provider_id:
            for index, provider in enumerate(_PROVIDERS):
                print(f"  {index + 1}. {provider.name}")
            choice = _selection_index(await _prompt(f"Enter number (1-{len(_PROVIDERS)}): "))
            provider_id = _PROVIDERS[choice].id if choice is not None and 0 <= choice < len(_PROVIDERS) else None
        if not provider_id or not any(provider.id == provider_id for provider in _PROVIDERS):
            raise RuntimeError(f"Unknown provider: {provider_id or ''}")
        await login(provider_id)
        return
    raise RuntimeError(f"Unknown command: {command}")


def entrypoint() -> None:
    try:
        asyncio.run(main())
    except Exception as error:
        print("Error:", javascript_string(error), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    entrypoint()
