"""OAuth contracts retained by the TypeScript coding-agent extension surface."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .abort import AbortSignal
from .auth.types import OAuthCredentials


@dataclass
class OAuthPrompt:
    message: str
    placeholder: str | None = None
    allow_empty: bool | None = None


@dataclass
class OAuthAuthInfo:
    url: str
    instructions: str | None = None


@dataclass
class OAuthDeviceCodeInfo:
    user_code: str
    verification_uri: str
    interval_seconds: int | float | None = None
    expires_in_seconds: int | float | None = None


@dataclass
class OAuthSelectOption:
    id: str
    label: str


@dataclass
class OAuthSelectPrompt:
    message: str
    options: list[OAuthSelectOption]


@dataclass
class OAuthLoginCallbacks:
    on_auth: Callable[[OAuthAuthInfo], None]
    on_device_code: Callable[[OAuthDeviceCodeInfo], None]
    on_prompt: Callable[[OAuthPrompt], Awaitable[str]]
    on_select: Callable[[OAuthSelectPrompt], Awaitable[str | None]]
    on_progress: Callable[[str], None] | None = None
    on_manual_code_input: Callable[[], Awaitable[str]] | None = None
    signal: AbortSignal | None = None


__all__ = [
    "OAuthCredentials", "OAuthPrompt", "OAuthAuthInfo", "OAuthDeviceCodeInfo", "OAuthSelectOption",
    "OAuthSelectPrompt", "OAuthLoginCallbacks",
]
