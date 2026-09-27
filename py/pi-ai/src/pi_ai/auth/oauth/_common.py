"""Shared URL, manual prompt and JSON boundaries of the native OAuth flows."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass
from ada_url import URLSearchParams

from ...utils._javascript import javascript_string
from ..types import AuthPrompt, ModelAuth, OAuthCredential, ProviderAuthInteraction
from ._http import parse_url

JS_WHITESPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def search_param(query: str, name: str) -> str | None:
    scalar_query = query.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")
    return URLSearchParams(scalar_query).get(name)


@dataclass
class AuthorizationInput:
    code: str | None = None
    state: str | None = None


def parse_authorization_input(value: str, *, hash_state: bool = True) -> AuthorizationInput:
    value = value.strip(JS_WHITESPACE)
    if not value:
        return AuthorizationInput()
    try:
        parsed = parse_url(value)
    except (ValueError, TypeError):
        pass
    else:
        return AuthorizationInput(search_param(parsed.search, "code"), search_param(parsed.search, "state") if hash_state else None)
    if hash_state and "#" in value:
        parts = value.split("#", 2)
        return AuthorizationInput(parts[0], parts[1])
    if "code=" in value:
        return AuthorizationInput(search_param(value, "code"), search_param(value, "state") if hash_state else None)
    return AuthorizationInput(code=value)


def json_parse(text: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"Unexpected token {value} in JSON")

    return json.loads(text, parse_int=float, parse_float=float, parse_constant=reject_constant)


def property_value(value: object, key: str) -> object:
    if value is None:
        raise TypeError(f"Cannot read properties of null (reading '{key}')")
    return value.get(key) if isinstance(value, Mapping) else None


def js_truthy(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != 0 and not math.isnan(value)
    return True


def js_number(value: object, *, missing: bool = False) -> float:
    if value is None:
        return math.nan if missing else 0.0
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip(JS_WHITESPACE)
        if not text:
            return 0.0
        try:
            if re.fullmatch(r"0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+", text):
                return float(int(text, 0))
            if not re.fullmatch(r"[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|Infinity)", text):
                return math.nan
            return float(text)
        except (ValueError, OverflowError):
            return math.nan
    return js_number(javascript_string(value))


async def await_promise[T](future: asyncio.Future[T]) -> T:
    if future.done():
        # JS await still yields to promise reactions for already-settled input.
        await asyncio.sleep(0)
    return await asyncio.shield(future)


class ManualPrompt:
    def __init__(self, interaction: ProviderAuthInteraction, prompt: AuthPrompt, cancel_wait: Callable[[], None]) -> None:
        self.input: str | None = None
        self.error: BaseException | None = None
        pending = interaction.prompt(prompt)
        if isinstance(pending, Coroutine):
            pending = asyncio.Task(pending, loop=asyncio.get_running_loop(), eager_start=True)

        async def observe() -> None:
            try:
                self.input = await pending
            except BaseException as error:
                self.error = error
            cancel_wait()

        self.task = asyncio.create_task(observe())

    async def wait(self) -> None:
        await await_promise(self.task)


def defer_operation[T](operation: Awaitable[T]) -> asyncio.Task[T]:
    """Admit a returned JS promise before a caller's finally closes the listener."""
    if isinstance(operation, Coroutine):
        return asyncio.Task(operation, loop=asyncio.get_running_loop(), eager_start=True)

    async def wait() -> T:
        return await operation

    return asyncio.create_task(wait())


async def credential_to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)
