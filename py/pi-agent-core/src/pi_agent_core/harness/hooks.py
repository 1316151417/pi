"""Hook registry ported from ``harness/hooks.ts``.

Hooks are named aggregates invoked on a lane's serialized mutation line behind
its effect gate. Each aggregate folds handler results in registration order:
most hooks collect the last non-undefined field, ``before_run`` accumulates
injected prompt messages, ``before_tool`` lets the first block win, and
structural hooks (compaction/navigation) accept the first useful decision.

Handler failures are reported through the injected error reporter and never
abort an aggregate unless the hook is explicitly fail-closed
(``before_drive``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .._chord.context import Context, with_abort_signal
from .types import AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch

__all__ = [
    "HookErrorReporter",
    "HookRegistration",
    "HookRegistry",
    "apply_stream_options_patch",
    "create_stream_options_patch",
]

HookErrorReporter = Callable[[BaseException, str, Optional[str], Context], Any]

_SCALAR_OPTION_KEYS = (
    "transport",
    "timeout_ms",
    "max_retries",
    "max_retry_delay_ms",
    "cache_retention",
    "deferred",
)
_SCALAR_OPTION_KEYS_CAMEL = (
    "transport",
    "timeoutMs",
    "maxRetries",
    "maxRetryDelayMs",
    "cacheRetention",
    "deferred",
)


@dataclass
class HookRegistration:
    handler: Callable[[dict, Context], Any]
    id: Optional[str] = None


async def _maybe_await(value: Any) -> Any:
    import asyncio

    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


class HookRegistry:
    """Ordered hook registrations per hook name."""

    def __init__(self, report_error: HookErrorReporter) -> None:
        self._registrations: Dict[str, List[HookRegistration]] = {}
        self._report_error = report_error
        self._closed_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(
        self,
        name: str,
        handler: Callable[[dict, Context], Any],
        options: Optional[dict] = None,
    ) -> Callable[[], None]:
        """Register a handler; returns an unsubscribe callable."""
        if self._closed_error is not None:
            raise self._closed_error
        registrations = self._registrations.setdefault(name, [])
        registration = HookRegistration(handler=handler, id=(options or {}).get("id"))
        registrations.append(registration)

        def _unsubscribe() -> None:
            try:
                registrations.remove(registration)
            except ValueError:
                pass

        return _unsubscribe

    def has(self, name: str) -> bool:
        return len(self._registrations.get(name, [])) != 0

    def close(self, error: BaseException) -> None:
        if self._closed_error is None:
            self._closed_error = error

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def run_with_gate(
        self, name: str, event: dict, gate: Any, context: Context
    ) -> Any:
        """Invoke one accepted-operation aggregate after synchronously passing its effect gate."""

        async def _admitted() -> Any:
            admitted_context = with_abort_signal(gate.signal, context)
            if admitted_context.signal is not None:
                admitted_context.signal.throw_if_aborted()
            return await self._run_admitted(name, event, admitted_context)

        return await gate.admit(_admitted)

    async def run_tool_with_gate(
        self, name: str, event: dict, gate: Any, context: Context
    ) -> Any:
        """Invoke a tool-hook aggregate behind the drive gate."""

        async def _admitted() -> Any:
            admitted_context = with_abort_signal(gate.signal, context)
            if admitted_context.signal is not None:
                admitted_context.signal.throw_if_aborted()
            if name == "before_tool":
                return await self._before_tool(event, admitted_context)
            return await self._after_tool(event, admitted_context)

        return await gate.admit(_admitted)

    async def _run_admitted(self, name: str, event: dict, context: Context) -> Any:
        if self._closed_error is not None:
            raise self._closed_error
        return await self.aggregate(name, event, context)

    async def aggregate(self, name: str, event: dict, context: Context) -> Any:
        """Fold every registered handler for ``name`` into one aggregate result."""
        if name == "before_run":
            return await self._before_run(event, context)
        if name == "before_drive":
            await self._invoke_all_fail_closed(name, event, context)
            return None
        if name == "before_run_end":
            collected: dict = {}
            await self._invoke_all(
                name,
                event,
                lambda value: collected.update(
                    {"followUp": value["followUp"]}
                    if isinstance(value, dict) and value.get("followUp") is not None
                    else {}
                ),
                context,
            )
            return {"followUp": collected["followUp"]} if "followUp" in collected else None
        if name == "transform_context":
            return await self._transform_context(event, context)
        if name == "before_request":
            return await self._before_request(event, context)
        if name == "before_payload":
            return await self._before_payload(event, context)
        if name == "after_response":
            return await self._after_response(event, context)
        if name == "before_tool":
            return await self._before_tool(event, context)
        if name == "after_tool":
            return await self._after_tool(event, context)
        if name == "before_compaction":
            return await self._first_structural(name, event, "compaction", context)
        if name == "before_navigation":
            return await self._first_structural(name, event, "summary", context)
        return None

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    async def _before_run(self, event: dict, context: Context) -> Any:
        prompt = list(event.get("prompt") or [])
        injected: List[Any] = []
        for registration in self._registrations_for("before_run"):
            try:
                result = await _maybe_await(registration.handler({**event, "prompt": prompt}, context))
                if isinstance(result, dict) and result.get("messages") is not None:
                    injected.extend(result["messages"])
                    prompt = [*prompt, *result["messages"]]
            except BaseException as error:  # noqa: BLE001 - reported, never fatal
                await self._report(error, "before_run", event.get("lane"), context)
        return {"messages": injected} if injected else None

    async def _before_tool(self, event: dict, context: Context) -> Any:
        args = event.get("args")
        block: Optional[dict] = None
        for registration in self._registrations_for("before_tool"):
            try:
                result = await self._invoke_tool_registration(
                    "before_tool", registration, {**event, "args": args}, context
                )
                if isinstance(result, dict):
                    if result.get("args") is not None:
                        args = result["args"]
                    if result.get("block") is not None:
                        block = result["block"]
                        break
            except BaseException as error:  # noqa: BLE001
                normalized = error if isinstance(error, Exception) else Exception(str(error))
                await self._report(normalized, "before_tool", event.get("lane"), context)
                block = {"reason": str(normalized)}
                break
        aggregate: dict = {}
        if args is not event.get("args"):
            aggregate["args"] = args
        if block is not None:
            aggregate["block"] = block
        return aggregate or None

    async def _transform_context(self, event: dict, context: Context) -> Any:
        messages = event.get("messages")
        system_prompt = event.get("systemPrompt")
        for registration in self._registrations_for("transform_context"):
            try:
                result = await _maybe_await(
                    registration.handler(
                        {**event, "messages": messages, "systemPrompt": system_prompt}, context
                    )
                )
                if isinstance(result, dict):
                    if result.get("messages") is not None:
                        messages = result["messages"]
                    if result.get("systemPrompt") is not None:
                        system_prompt = result["systemPrompt"]
            except BaseException as error:  # noqa: BLE001
                await self._report(error, "transform_context", event.get("lane"), context)
        return {"messages": messages, "systemPrompt": system_prompt}

    async def _before_request(self, event: dict, context: Context) -> Any:
        stream_options = event.get("streamOptions") or AgentHarnessStreamOptions()
        changed = False
        for registration in self._registrations_for("before_request"):
            try:
                result = await _maybe_await(
                    registration.handler({**event, "streamOptions": stream_options}, context)
                )
                if isinstance(result, dict) and result.get("streamOptions") is not None:
                    stream_options = apply_stream_options_patch(stream_options, result["streamOptions"])
                    changed = True
            except BaseException as error:  # noqa: BLE001
                await self._report(error, "before_request", event.get("lane"), context)
        if not changed:
            return None
        return {"streamOptions": create_stream_options_patch(event.get("streamOptions"), stream_options)}

    async def _before_payload(self, event: dict, context: Context) -> Any:
        payload = event.get("payload")
        for registration in self._registrations_for("before_payload"):
            try:
                result = await _maybe_await(
                    registration.handler({**event, "payload": payload}, context)
                )
                if isinstance(result, dict) and result.get("payload") is not None:
                    payload = result["payload"]
            except BaseException as error:  # noqa: BLE001
                await self._report(error, "before_payload", event.get("lane"), context)
        return {"payload": payload}

    async def _after_response(self, event: dict, context: Context) -> Any:
        message = event.get("message")
        for registration in self._registrations_for("after_response"):
            try:
                result = await _maybe_await(
                    registration.handler({**event, "message": message}, context)
                )
                if isinstance(result, dict) and result.get("message") is not None:
                    message = result["message"]
            except BaseException as error:  # noqa: BLE001
                await self._report(error, "after_response", event.get("lane"), context)
        return {"message": message}

    async def _after_tool(self, event: dict, context: Context) -> Any:
        current = {
            "content": event.get("content"),
            "details": event.get("details"),
            "isError": event.get("isError"),
            "usage": event.get("usage"),
        }
        aggregate: dict = {}
        for registration in self._registrations_for("after_tool"):
            try:
                result = await self._invoke_tool_registration(
                    "after_tool", registration, {**event, **current}, context
                )
                if not isinstance(result, dict):
                    continue
                if result.get("content") is not None:
                    aggregate["content"] = result["content"]
                if result.get("details") is not None:
                    aggregate["details"] = result["details"]
                if result.get("isError") is not None:
                    aggregate["isError"] = result["isError"]
                if result.get("usage") is not None:
                    aggregate["usage"] = result["usage"]
                if result.get("terminate") is not None:
                    aggregate["terminate"] = result["terminate"]
                current = {
                    "content": result.get("content", current["content"]),
                    "details": result.get("details", current["details"]),
                    "isError": result.get("isError", current["isError"]),
                    "usage": result.get("usage", current["usage"]),
                }
            except BaseException as error:  # noqa: BLE001
                await self._report(error, "after_tool", event.get("lane"), context)
        return aggregate or None

    async def _first_structural(
        self, name: str, event: dict, result_field: str, context: Context
    ) -> Any:
        for registration in self._registrations_for(name):
            try:
                value = await _maybe_await(registration.handler(event, context))
                if value is None or not isinstance(value, dict):
                    continue
                if value.get("decline") is True and value.get(result_field) is not None:
                    await self._report(
                        Exception(f"{name} hook cannot return both decline and {result_field}"),
                        name,
                        event.get("lane"),
                        context,
                    )
                    continue
                if value.get("decline") is True or value.get(result_field) is not None:
                    return value
            except BaseException as error:  # noqa: BLE001
                await self._report(error, name, event.get("lane"), context)
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _invoke_tool_registration(
        self, name: str, registration: HookRegistration, event: dict, context: Context
    ) -> Any:
        # Tool hooks are wrapped in a telemetry span by the runtime; the registry
        # only needs to guarantee outcome reporting on failure.
        try:
            return await _maybe_await(registration.handler(event, context))
        except BaseException:
            raise

    async def _invoke_all_fail_closed(self, name: str, event: dict, context: Context) -> None:
        for registration in self._registrations_for(name):
            try:
                await _maybe_await(registration.handler(event, context))
            except BaseException as error:  # noqa: BLE001 - fail closed
                normalized = error if isinstance(error, Exception) else Exception(str(error))
                await self._report(normalized, name, event.get("lane"), context)
                raise normalized

    async def _invoke_all(self, name: str, event: dict, apply: Callable[[Any], None], context: Context) -> None:
        for registration in self._registrations_for(name):
            try:
                apply(await _maybe_await(registration.handler(event, context)))
            except BaseException as error:  # noqa: BLE001
                await self._report(error, name, event.get("lane"), context)

    def _registrations_for(self, name: str) -> List[HookRegistration]:
        return list(self._registrations.get(name, []))

    async def _report(
        self, error: BaseException, hook: str, lane: Optional[str], context: Context
    ) -> None:
        normalized = error if isinstance(error, Exception) else Exception(str(error))
        await _maybe_await(self._report_error(normalized, hook, lane, context))


# ---------------------------------------------------------------------------
# Stream option patches
# ---------------------------------------------------------------------------


def _option_value(options: Any, snake: str, camel: str) -> Any:
    if isinstance(options, dict):
        return options.get(camel, options.get(snake))
    return getattr(options, snake, None)


def _has_option(options: Any, snake: str, camel: str) -> bool:
    if isinstance(options, dict):
        return camel in options or snake in options
    if hasattr(options, "has"):
        return bool(options.has(snake))
    return hasattr(options, snake)


def _set_option(options: AgentHarnessStreamOptions, key: str, value: Any) -> None:
    setattr(options, key, value)


def apply_stream_options_patch(
    base: AgentHarnessStreamOptions, patch: Any
) -> AgentHarnessStreamOptions:
    """Apply a provider-hook stream option patch; ``None`` values delete keys."""
    next_options = AgentHarnessStreamOptions(**{**vars(base)})
    for snake, camel in zip(_SCALAR_OPTION_KEYS, _SCALAR_OPTION_KEYS_CAMEL):
        if not _has_option(patch, snake, camel):
            continue
        value = _option_value(patch, snake, camel)
        _set_option(next_options, snake, None if value is None else value)

    if _has_option(patch, "headers", "headers"):
        headers_patch = _option_value(patch, "headers", "headers")
        if headers_patch is None:
            next_options.headers = None
        else:
            headers = dict(next_options.headers or {})
            for key, value in headers_patch.items():
                if value is None:
                    headers.pop(key, None)
                else:
                    headers[key] = value
            next_options.headers = headers

    if _has_option(patch, "metadata", "metadata"):
        metadata_patch = _option_value(patch, "metadata", "metadata")
        if metadata_patch is None:
            next_options.metadata = None
        else:
            metadata = dict(next_options.metadata or {})
            for key, value in metadata_patch.items():
                if value is None:
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
            next_options.metadata = metadata

    return next_options


def create_stream_options_patch(
    base: Optional[AgentHarnessStreamOptions], value: AgentHarnessStreamOptions
) -> AgentHarnessStreamOptionsPatch:
    """Derive the minimal patch transforming ``base`` into ``value``."""
    provided: Dict[str, Any] = {}

    for snake, camel in zip(_SCALAR_OPTION_KEYS, _SCALAR_OPTION_KEYS_CAMEL):
        base_value = _option_value(base, snake, camel) if base is not None else None
        value_value = _option_value(value, snake, camel)
        if base_value != value_value:
            provided[snake] = value_value

    base_headers = base.headers if base is not None else None
    if base_headers != value.headers:
        if value.headers is None:
            provided["headers"] = None
        else:
            headers: Dict[str, Optional[str]] = {}
            for key in base_headers or {}:
                if key not in value.headers:
                    headers[key] = None
            for key, header in value.headers.items():
                if (base_headers or {}).get(key) != header:
                    headers[key] = header
            if base_headers is None and not headers:
                provided["headers"] = {}
            elif headers:
                provided["headers"] = headers

    base_metadata = base.metadata if base is not None else None
    if base_metadata != value.metadata:
        if value.metadata is None:
            provided["metadata"] = None
        else:
            metadata: Dict[str, Any] = {}
            for key in base_metadata or {}:
                if key not in value.metadata:
                    metadata[key] = None
            for key, metadata_value in value.metadata.items():
                if (base_metadata or {}).get(key) != metadata_value:
                    metadata[key] = metadata_value
            if base_metadata is None and not metadata:
                provided["metadata"] = {}
            elif metadata:
                provided["metadata"] = metadata

    return AgentHarnessStreamOptionsPatch(**provided)
