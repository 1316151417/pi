"""Direct remote-service transport from ``services/loopback.ts``."""

from __future__ import annotations

from collections.abc import Callable

from ..context import Context
from ..delta import Undefined
from ..types import (
    JsonValue, RemoteServiceTransport, ServiceCall, ServiceMode,
    ServiceProviderUpdate, ServiceSubscription,
)
from .provider import RemoteServiceProvider


class _LoopbackServiceTransport:
    def __init__(self, provider: RemoteServiceProvider) -> None:
        self._provider = provider

    async def invoke(self, call: ServiceCall, context: Context) -> JsonValue | Undefined:
        return await self._provider.invoke(call, context)

    async def subscribe(
        self, service_id: str, mode: ServiceMode,
        listener: Callable[[ServiceProviderUpdate, Context], None], context: Context,
    ) -> ServiceSubscription:
        return self._provider.subscribe(service_id, mode, listener)


def create_loopback_service_transport(provider: RemoteServiceProvider) -> RemoteServiceTransport:
    return _LoopbackServiceTransport(provider)
