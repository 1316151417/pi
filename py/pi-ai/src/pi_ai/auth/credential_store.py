"""Per-provider serialized in-memory credentials from ``credential-store.ts``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ..abort import operation_signal, race_with_abort_signal
from .types import AuthOperationOptions, Credential, CredentialInfo

__all__ = ["InMemoryCredentialStore"]


class InMemoryCredentialStore:
    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._chains: dict[str, asyncio.Task[None]] = {}

    def _enqueue[T](
        self, provider_id: str, task: Callable[[], Awaitable[T]], options: AuthOperationOptions | None,
    ) -> Awaitable[T]:
        signal = operation_signal(options.signal if options is not None else None)
        previous = self._chains.get(provider_id)

        async def run() -> T:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except BaseException:
                    pass
            signal.throw_if_aborted()
            return await task()

        queued = asyncio.get_running_loop().create_task(run())

        async def observe() -> None:
            try:
                await asyncio.shield(queued)
            except BaseException:
                pass

        tail = asyncio.get_running_loop().create_task(observe())
        self._chains[provider_id] = tail

        def cleanup(completed: asyncio.Task[None]) -> None:
            if self._chains.get(provider_id) is completed:
                del self._chains[provider_id]

        tail.add_done_callback(cleanup)
        return race_with_abort_signal(queued, signal)

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        if options is not None and options.signal is not None:
            options.signal.throw_if_aborted()
        return self._credentials.get(provider_id)

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        if options is not None and options.signal is not None:
            options.signal.throw_if_aborted()
        return [CredentialInfo(provider_id, credential.type) for provider_id, credential in self._credentials.items()]

    def modify(
        self, provider_id: str, fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Awaitable[Credential | None]:
        async def modify_credential() -> Credential | None:
            current = self._credentials.get(provider_id)
            next_credential = await fn(current)
            if options is not None and options.signal is not None:
                options.signal.throw_if_aborted()
            if next_credential is not None:
                self._credentials[provider_id] = next_credential
            return next_credential if next_credential is not None else current

        return self._enqueue(provider_id, modify_credential, options)

    def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[None]:
        async def delete_credential() -> None:
            self._credentials.pop(provider_id, None)

        return self._enqueue(provider_id, delete_credential, options)
