"""Unmodified fetch passthrough from api/cloudflare-ai-binding.ts."""

from collections.abc import Awaitable
from typing import Protocol

from ..types import FetchFunction

CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL = "cloudflare-gateway-binding"


class AiBinding(Protocol):
    ai_gateway_log_id: str | None
    fetch: FetchFunction | None


def create_ai_binding_fetch(binding: AiBinding) -> FetchFunction:
    binding_fetch = getattr(binding, "fetch", None)
    if not callable(binding_fetch):
        raise TypeError("createAiBindingFetch: the AI binding does not expose fetch()")

    def fetch(*args: object, **kwargs: object) -> Awaitable[object]:
        return binding_fetch(*args, **kwargs)

    return fetch
