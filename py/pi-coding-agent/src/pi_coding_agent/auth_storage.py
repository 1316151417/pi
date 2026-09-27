"""JSON credential storage from ``core/auth-storage.ts``."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path

from pi_ai.auth.types import (
    ApiKeyCredential, AuthOperationOptions, Credential, CredentialInfo,
    OAuthCredential,
)

from .config import get_auth_path
from .paths import normalize_path
from .resolve_config_value import resolve_config_value

_STALE_SECONDS = 30.0


def _check_abort(options: AuthOperationOptions | None) -> None:
    if options is not None and options.signal is not None:
        options.signal.throw_if_aborted()


def _read_data(path: Path) -> dict[str, Credential]:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8").removeprefix("\ufeff"))
    except FileNotFoundError:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("Invalid auth.json: expected an object")
    result: dict[str, Credential] = {}
    for provider_id, raw in parsed.items():
        if not isinstance(provider_id, str) or not isinstance(raw, dict):
            raise ValueError(f'Invalid auth.json credential for provider "{provider_id}"')
        if raw.get("type") == "api_key":
            key = raw.get("key")
            env = raw.get("env")
            if key is not None and not isinstance(key, str):
                raise ValueError(f'Invalid auth.json credential for provider "{provider_id}"')
            if env is not None and (
                not isinstance(env, dict) or any(not isinstance(value, str) for value in env.values())
            ):
                raise ValueError(f'Invalid auth.json credential for provider "{provider_id}"')
            result[provider_id] = ApiKeyCredential.from_json(raw)
        elif raw.get("type") == "oauth":
            expires = raw.get("expires")
            if (
                not isinstance(raw.get("access"), str)
                or not isinstance(raw.get("refresh"), str)
                or isinstance(expires, bool)
                or not isinstance(expires, (int, float))
                or not math.isfinite(expires)
            ):
                raise ValueError(f'Invalid auth.json credential for provider "{provider_id}"')
            result[provider_id] = OAuthCredential.from_json(raw)
        else:
            raise ValueError(f'Invalid auth.json credential for provider "{provider_id}"')
    return result


def _write_data(path: Path, data: dict[str, Credential]) -> None:
    serialized = json.dumps(
        {name: credential.to_json() for name, credential in data.items()},
        ensure_ascii=False, indent=2,
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        stream.write(serialized)


class AuthStorage:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(normalize_path(path or get_auth_path()))
        self.lock_path = Path(str(self.path) + ".lock")

    async def _acquire(self, options: AuthOperationOptions | None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.path.exists():
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write("{}")
        deadline = time.monotonic() + _STALE_SECONDS
        retry = 0
        while True:
            _check_abort(options)
            try:
                self.lock_path.mkdir()
                return
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                    if age > _STALE_SECONDS:
                        self.lock_path.rmdir()
                        continue
                except (FileNotFoundError, OSError):
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Could not lock auth storage: {self.path}")
                delay = min(0.01 * (2 ** min(retry, 8)) * (1 + random.random()), remaining, 2.0)
                retry += 1
                await asyncio.sleep(delay)

    async def _with_lock[T](
        self, operation: Callable[[dict[str, Credential]], Awaitable[T]],
        options: AuthOperationOptions | None = None,
    ) -> T:
        await self._acquire(options)
        stop_heartbeat = asyncio.Event()

        async def heartbeat() -> None:
            while not stop_heartbeat.is_set():
                try:
                    await asyncio.wait_for(stop_heartbeat.wait(), 5.0)
                except TimeoutError:
                    try:
                        self.lock_path.touch()
                    except OSError:
                        pass

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            _check_abort(options)
            data = await asyncio.to_thread(_read_data, self.path)
            return await operation(data)
        finally:
            stop_heartbeat.set()
            try:
                await asyncio.shield(heartbeat_task)
            finally:
                try:
                    self.lock_path.rmdir()
                except OSError:
                    pass

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        async def read_data(data: dict[str, Credential]) -> Credential | None:
            _check_abort(options)
            return deepcopy(data.get(provider_id))

        credential = await self._with_lock(read_data, options)
        if isinstance(credential, ApiKeyCredential) and credential.key is not None:
            credential.key = resolve_config_value(credential.key, credential.env)
        return credential

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        async def list_data(data: dict[str, Credential]) -> list[CredentialInfo]:
            _check_abort(options)
            return [CredentialInfo(provider_id, credential.type) for provider_id, credential in data.items()]

        return await self._with_lock(list_data, options)

    def modify(
        self, provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Awaitable[Credential | None]:
        async def modify_data(data: dict[str, Credential]) -> Credential | None:
            current = data.get(provider_id)
            next_credential = await fn(deepcopy(current))
            _check_abort(options)
            if next_credential is None:
                return deepcopy(current)
            data[provider_id] = deepcopy(next_credential)
            await asyncio.to_thread(_write_data, self.path, data)
            return next_credential

        return self._with_lock(modify_data, options)

    def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[None]:
        async def delete_data(data: dict[str, Credential]) -> None:
            data.pop(provider_id, None)
            _check_abort(options)
            await asyncio.to_thread(_write_data, self.path, data)

        return self._with_lock(delete_data, options)

    @classmethod
    def create(cls, path: str | None = None) -> AuthStorage:
        return cls(path)


class RuntimeCredentials:
    """Non-persistent API-key overlay from ``core/runtime-credentials.ts``."""

    def __init__(self, store: AuthStorage) -> None:
        self.store = store
        self.overrides: dict[str, str] = {}

    def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        self.overrides[provider_id] = api_key

    def remove_runtime_api_key(self, provider_id: str) -> None:
        self.overrides.pop(provider_id, None)

    def has_runtime_api_key(self, provider_id: str) -> bool:
        return provider_id in self.overrides

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        _check_abort(options)
        override = self.overrides.get(provider_id)
        return ApiKeyCredential(key=override) if override else await self.store.read(provider_id, options)

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        entries = {entry.provider_id: entry for entry in await self.store.list(options)}
        _check_abort(options)
        for provider_id in self.overrides:
            entries[provider_id] = CredentialInfo(provider_id, "api_key")
        return list(entries.values())

    def modify(
        self, provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Awaitable[Credential | None]:
        return self.store.modify(provider_id, fn, options)

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        _check_abort(options)
        await self.store.delete(provider_id, options)
        self.overrides.pop(provider_id, None)


def read_stored_credential(provider_id: str, auth_path: str | None = None) -> Credential | None:
    try:
        return _read_data(Path(normalize_path(auth_path or get_auth_path()))).get(provider_id)
    except (OSError, ValueError):
        return None


__all__ = ["AuthStorage", "RuntimeCredentials", "read_stored_credential"]
