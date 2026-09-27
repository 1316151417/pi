"""Presentation-scoped Session routing and attachment leases from ``session-router.ts``."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast

from pi_chord.context import BACKGROUND_CONTEXT, Context
from pi_chord.delta import Undefined
from pi_chord.types import JsonValue, ServiceCall, ServiceProviderUpdate
from pi_protocol import RpcTarget, SessionTarget

from .errors import ServerDrainingError, SessionNotAttachedError
from .types import (
    PublishUpdate, RoutedSessionAttachment, RoutedSessionHandle, ServerHost,
    SessionMetadata,
)


class SessionCleanupError(ExceptionGroup):
    pass


@dataclass(eq=False)
class _HostedSession:
    id: str
    handle: RoutedSessionHandle
    attachments: set[_ClientAttachment] = field(default_factory=set)


@dataclass(eq=False)
class _ClientAttachment:
    id: str
    client: object
    session: _HostedSession
    operations: set[asyncio.Future[object]] = field(default_factory=set)
    acquiring: asyncio.Future[RoutedSessionAttachment] | None = None
    lease: RoutedSessionAttachment | None = None
    releasing: asyncio.Task[None] | None = None


@dataclass
class SessionRouterOptions[TMetadata: SessionMetadata]:
    host: ServerHost[TMetadata]
    server_id: str
    is_closing: Callable[[], bool]
    publish_attachment: Callable[[object, SessionTarget | None, Context], Awaitable[None]]
    report_error: Callable[[object], None]


class SessionRouter[TMetadata: SessionMetadata]:
    def __init__(self, options: SessionRouterOptions[TMetadata]) -> None:
        self._options = options
        self._hosted_sessions: dict[str, _HostedSession] = {}
        self._opening_sessions: dict[str, asyncio.Task[_HostedSession]] = {}
        self._attachments_by_client: dict[object, _ClientAttachment] = {}
        self._disconnected_clients: set[object] = set()
        self._client_operations: dict[object, asyncio.Future[None]] = {}
        self._close_promise: asyncio.Task[None] | None = None

    async def execute_service_call(
        self,
        call: ServiceCall,
        target: RpcTarget,
        client: object,
        publish: PublishUpdate,
        context: Context,
    ) -> JsonValue | Undefined:
        admitted = await self._run_for_client(
            client, lambda: self._start_service_call(client, target, call, publish, context),
        )
        return await admitted

    def attach_client(self, client: object, session_id: str, context: Context) -> asyncio.Future[None]:
        if self._options.is_closing():
            return _rejected(ServerDrainingError())
        return self._run_for_client(client, lambda: self._attach_client_now(client, session_id, context))

    def detach_client(self, client: object, context: Context) -> asyncio.Future[None]:
        async def detach() -> None:
            attachment = self._attachments_by_client.get(client)
            if attachment is not None:
                await self._release_attachment(attachment, context)
        return self._run_for_client(client, detach)

    async def remove_session(self, session_id: str, context: Context) -> None:
        if self._options.is_closing():
            raise ServerDrainingError()
        hosted = self._hosted_sessions.get(session_id)
        if hosted is None:
            return
        errors = await _all_errors([self._release_attachment(item, context) for item in list(hosted.attachments)])
        try:
            await hosted.handle.close(context)
        except Exception as error:
            errors.append(error)
        if self._hosted_sessions.get(session_id) is hosted:
            del self._hosted_sessions[session_id]
        _raise_errors(errors, f"Failed to close Session {session_id}")

    async def disconnect(self, client: object, context: Context) -> None:
        self._disconnected_clients.add(client)
        try:
            async def release() -> None:
                attachment = self._attachments_by_client.get(client)
                if attachment is not None:
                    await self._release_attachment(attachment, context, False)
            await self._run_for_client(client, release)
        finally:
            self._disconnected_clients.discard(client)

    def close(self, context: Context) -> asyncio.Task[None]:
        if self._close_promise is None:
            self._close_promise = asyncio.get_running_loop().create_task(self._close_internal(context))
        return self._close_promise

    async def _close_internal(self, context: Context) -> None:
        operations = list(self._client_operations.values())
        openings = list(self._opening_sessions.values())
        settled = await asyncio.gather(*operations, *openings, return_exceptions=True)
        errors: list[Exception] = []
        for result in settled:
            if isinstance(result, BaseException):
                self._options.report_error(result)
                if isinstance(result, SessionCleanupError):
                    errors.append(result)
        errors.extend(await _all_errors([
            self._release_attachment(attachment, context)
            for hosted in self._hosted_sessions.values()
            for attachment in list(hosted.attachments)
        ]))
        hosted_sessions = list(self._hosted_sessions.values())
        closing = await asyncio.gather(
            *(item.handle.close(context) for item in hosted_sessions), return_exceptions=True,
        )
        for hosted, result in zip(hosted_sessions, closing):
            if isinstance(result, BaseException):
                error = _to_exception(result)
                self._options.report_error(error)
                errors.append(error)
            elif self._hosted_sessions.get(hosted.id) is hosted:
                del self._hosted_sessions[hosted.id]
        self._attachments_by_client.clear()
        self._client_operations.clear()
        if errors:
            raise ExceptionGroup("Failed to close routed Sessions", errors)

    def _run_for_client[T](self, client: object, operation: Callable[[], Awaitable[T]]) -> asyncio.Task[T]:
        previous = self._client_operations.get(client)

        async def run() -> T:
            if previous is not None:
                try:
                    await previous
                except BaseException:
                    pass
            return await operation()

        result = asyncio.get_running_loop().create_task(run())

        async def settle() -> None:
            try:
                await result
            except BaseException:
                pass

        tail = asyncio.get_running_loop().create_task(settle())
        self._client_operations[client] = tail

        def remove(_completion: asyncio.Future[None]) -> None:
            if self._client_operations.get(client) is tail:
                del self._client_operations[client]

        tail.add_done_callback(remove)
        return result

    async def _attach_client_now(self, client: object, session_id: str, context: Context) -> None:
        if self._options.is_closing() or client in self._disconnected_clients:
            raise ServerDrainingError()
        current = self._attachments_by_client.get(client)
        if current is not None and current.session.id == session_id:
            return
        hosted = await self._acquire(session_id, context)
        if self._options.is_closing() or client in self._disconnected_clients:
            raise ServerDrainingError()
        if current is not None:
            await self._release_attachment(current, context, False)
        attachment = _ClientAttachment(str(uuid.uuid4()), client, hosted)
        hosted.attachments.add(attachment)
        try:
            acquiring = hosted.handle.attach_client(context)
            attachment.acquiring = asyncio.ensure_future(_await_value(acquiring))
            attachment.lease = await attachment.acquiring
        except Exception:
            hosted.attachments.discard(attachment)
            raise
        if (self._hosted_sessions.get(hosted.id) is not hosted
                or attachment not in hosted.attachments
                or client in self._disconnected_clients or self._options.is_closing()):
            await self._release_attachment(attachment, context)
            raise ServerDrainingError()
        self._attachments_by_client[client] = attachment
        await self._options.publish_attachment(client, {
            "serverId": self._options.server_id,
            "sessionId": session_id,
            "attachmentId": attachment.id,
        }, context)

    async def _start_service_call(
        self,
        client: object, target: RpcTarget, call: ServiceCall,
        publish: PublishUpdate, context: Context,
    ) -> asyncio.Future[JsonValue | Undefined]:
        attachment = self._require_attachment(client, target)
        lease = cast(RoutedSessionAttachment, attachment.lease)
        result = asyncio.ensure_future(lease.invoke_service(call, publish, context))
        attachment.operations.add(cast(asyncio.Future[object], result))
        result.add_done_callback(lambda _done: attachment.operations.discard(cast(asyncio.Future[object], result)))
        return result

    def _require_attachment(self, client: object, target: RpcTarget) -> _ClientAttachment:
        if self._options.is_closing() or client in self._disconnected_clients:
            raise ServerDrainingError()
        if "sessionId" not in target:
            raise SessionNotAttachedError()
        attachment = self._attachments_by_client.get(client)
        if (attachment is None or attachment.session.id != target["sessionId"]
                or attachment.id != target["attachmentId"]):
            raise SessionNotAttachedError()
        return attachment

    def _release_attachment(
        self, attachment: _ClientAttachment, context: Context, publish: bool = True,
    ) -> asyncio.Task[None]:
        if attachment.releasing is not None:
            return attachment.releasing

        async def release() -> None:
            errors: list[Exception] = []
            try:
                if attachment.operations:
                    await asyncio.gather(*list(attachment.operations), return_exceptions=True)
                try:
                    lease = attachment.lease
                    if lease is None and attachment.acquiring is not None:
                        lease = await attachment.acquiring
                    if lease is not None:
                        await _await_value(lease.release(context))
                except Exception as error:
                    errors.append(error)
                _raise_errors(errors, "Failed to release Session attachment")
            finally:
                await self._clear_attachment(attachment, context, publish)

        attachment.releasing = asyncio.get_running_loop().create_task(release())
        return attachment.releasing

    async def _clear_attachment(self, attachment: _ClientAttachment, context: Context, publish: bool) -> None:
        attachment.session.attachments.discard(attachment)
        if self._attachments_by_client.get(attachment.client) is attachment:
            del self._attachments_by_client[attachment.client]
            if publish:
                await self._options.publish_attachment(attachment.client, None, context)

    async def _acquire(self, session_id: str, context: Context) -> _HostedSession:
        existing = self._hosted_sessions.get(session_id)
        if existing is not None:
            return existing
        opening = self._opening_sessions.get(session_id)
        if opening is None:
            opening = asyncio.get_running_loop().create_task(self._open(session_id, context))
            self._opening_sessions[session_id] = opening
        try:
            return await opening
        finally:
            if self._opening_sessions.get(session_id) is opening:
                del self._opening_sessions[session_id]

    async def _open(self, session_id: str, context: Context) -> _HostedSession:
        metadata = await self._options.host.resolve_session(session_id, context)
        handle = await self._options.host.open_session(metadata, context)
        if self._options.is_closing():
            try:
                await handle.close(context)
            except Exception as error:
                self._options.report_error(error)
                raise SessionCleanupError(
                    "Failed to close routed Session acquired while draining",
                    [ServerDrainingError(), error],
                ) from error
            raise ServerDrainingError()
        hosted = _HostedSession(metadata.id, handle)
        self._hosted_sessions[hosted.id] = hosted
        terminated = getattr(handle, "terminated", None)
        if terminated is not None:
            async def observe() -> None:
                try:
                    error = await terminated
                except Exception as error:
                    self._invalidate(hosted, error)
                else:
                    self._invalidate(hosted, error)
            asyncio.get_running_loop().create_task(observe())
        return hosted

    def _invalidate(self, hosted: _HostedSession, error: Exception | None) -> None:
        if self._hosted_sessions.get(hosted.id) is not hosted:
            return
        del self._hosted_sessions[hosted.id]
        for attachment in list(hosted.attachments):
            closing = self._release_attachment(attachment, BACKGROUND_CONTEXT)

            def report(done: asyncio.Task[None]) -> None:
                try:
                    done.result()
                except Exception as release_error:
                    self._options.report_error(release_error)

            closing.add_done_callback(report)
        if error is not None:
            self._options.report_error(error)


async def _await_value[T](value: T | Awaitable[T]) -> T:
    return await value if inspect.isawaitable(value) else value


def _rejected[T](error: Exception) -> asyncio.Future[T]:
    result: asyncio.Future[T] = asyncio.get_running_loop().create_future()
    result.set_exception(error)
    return result


async def _all_errors(awaitables: list[Awaitable[object]]) -> list[Exception]:
    if not awaitables:
        return []
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    return [_to_exception(result) for result in results if isinstance(result, BaseException)]


def _to_exception(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else RuntimeError(str(error))


def _raise_errors(errors: list[Exception], message: str) -> None:
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup(message, errors)
