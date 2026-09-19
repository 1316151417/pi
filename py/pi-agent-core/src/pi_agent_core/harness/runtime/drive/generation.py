"""Assistant generation procedures ported from ``harness/runtime/drive/generation.ts``.

Prepares one approved assistant request, publishes its effect intent, performs
the provider call, and advances one durable retry wait.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from ...context import get_telemetry_context, with_abort_signal
from ...execution.assistant import (
    HarnessAssistantStreamConfig,
    stream_harness_assistant,
)
from ...hooks import apply_stream_options_patch
from ...session.session import SessionInvariantError
from ...session.types import (
    AssistantEffectPendingOperation,
    AssistantReadyOperation,
    AssistantRetryWaitOperation,
    OperationError,
    operation_scope_of,
)
from ..transcript import read_bounded_context
from ..types import CommitDecision, ContinueOperationResult, ProcedureResult
from .response import open_assistant_response, publish_configuration_failure, publish_response
from .retry import wait_until

__all__ = ["run_generation", "run_retry_wait", "AssistantEffectPending", "PreparedGeneration"]


#: Assistant effect-pending payload without the run-wide scope fields.
AssistantEffectPending = dict


class PreparedGeneration:
    """One resolved assistant request ready to be executed."""

    def __init__(
        self,
        model: Any,
        tools: List[Any],
        messages: List[Any],
        system_prompt: str,
        stream_options: Any,
        to_provider_messages: Any,
    ) -> None:
        self.kind = "ready"
        self.model = model
        self.tools = tools
        self.messages = messages
        self.system_prompt = system_prompt
        self.stream_options = stream_options
        self.to_provider_messages = to_provider_messages


def _configuration_error(code: str, details: Any) -> OperationError:
    return OperationError(
        code=code,
        message=(
            "The configured model is unavailable in this process"
            if code == "model_unavailable"
            else "One or more configured tools are unavailable in this process"
        ),
        details=details,
    )


async def resolve_system_prompt(lane: Any, context: Any) -> str:
    config = lane.read_config()
    system_prompt = config.system_prompt
    if system_prompt is None:
        return ""
    if isinstance(system_prompt, str):
        return system_prompt
    source = config.tool_context
    tool_context = await source(context) if callable(source) else source
    result = system_prompt(tool_context, context)
    if hasattr(result, "__await__"):
        result = await result
    return result


async def prepare_generation(lane: Any, drive: Any, generation: AssistantReadyOperation) -> Any:
    identity = generation.generation_context.configuration.model
    model = lane.models.get_model(
        _opt(identity, "provider"), _opt(identity, "model_id", "modelId")
    )
    if model is None:
        return _Bag(
            {"kind": "configuration_failure", "error": _configuration_error("model_unavailable", identity.to_json())}
        )

    config = lane.read_config()
    tools_by_name = {tool.name: tool for tool in config.tools}
    active = generation.generation_context.configuration.active_tool_names
    missing = [name for name in active if name not in tools_by_name]
    if missing:
        return _Bag(
            {
                "kind": "configuration_failure",
                "error": _configuration_error("configured_tools_unavailable", {"tools": missing}),
            }
        )
    tools = []
    for name in active:
        tool = tools_by_name.get(name)
        if tool is None:
            raise SessionInvariantError(
                f"Configured tool {name} disappeared during resolution"
            )
        from pi_ai.types import Tool as AiTool

        tools.append(
            AiTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                constrained_sampling=getattr(tool, "constrained_sampling", None),
            )
        )

    messages = await read_bounded_context(lane, drive, generation)
    if messages.kind == "cancel_requested":
        return _Bag({"kind": "cancel_requested"})
    system_prompt = await resolve_system_prompt(lane, drive.context)
    before_request = await lane.hooks.run_with_gate(
        "before_request",
        {
            "lane": lane.name,
            "runId": drive.operation_id,
            "model": model,
            "step": "assistant",
            "attempt": generation.next_attempt,
            "streamOptions": generation.generation_context.stream_options,
        },
        drive.gate,
        drive.context,
    )
    patch = None if before_request is None else _opt(before_request, "stream_options", "streamOptions")
    stream_options = (
        generation.generation_context.stream_options
        if patch is None
        else apply_stream_options_patch(generation.generation_context.stream_options, patch)
    )
    return PreparedGeneration(
        model=model,
        tools=tools,
        messages=messages.value,
        system_prompt=system_prompt,
        stream_options=stream_options,
        to_provider_messages=config.to_provider_messages,
    )


async def publish_generation_intent(
    lane: Any, drive: Any, ready: AssistantReadyOperation, prepared: PreparedGeneration
) -> ContinueOperationResult:
    at = _now()
    pending = {
        "generation_context": ready.generation_context,
        "attempt": ready.next_attempt,
        "response_entry_id": lane.session.id_generator.next(at),
        "usage_id": lane.session.id_generator.next(at),
        "intended_output_limit": prepared.model.max_tokens,
        "context_window": prepared.model.context_window,
    }

    def _plan(_state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        next_state = AssistantEffectPendingOperation(**operation_scope_of(current), **pending)

        def _events(_commit: Any) -> List[Any]:
            if ready.next_attempt != 1:
                return []
            return [
                {
                    "type": "turn_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "turnId": ready.generation_context.step_id,
                }
            ]

        return CommitDecision(
            writes=[],
            operation_state=next_state,
            materialize=lambda _commit: next_state,
            events=_events,
        )

    return await lane.continue_operation(ready, _plan, drive.context)


async def perform_generation(
    lane: Any, drive: Any, intent: AssistantEffectPendingOperation, prepared: PreparedGeneration
) -> Any:
    response = open_assistant_response(lane, drive, intent.response_entry_id)
    try:
        config = HarnessAssistantStreamConfig(
            model=prepared.model,
            system_prompt=prepared.system_prompt,
            tools=prepared.tools,
            thinking_level=intent.generation_context.configuration.thinking_level,
            stream_options=prepared.stream_options,
            to_provider_messages=prepared.to_provider_messages,
            request=_make_request(lane, drive, prepared),
            observer=response.observer,
            transform_context=_make_transform_context(lane, drive),
            before_payload=_make_before_payload(lane, drive),
            after_response=response.after_response,
        )
        return await stream_harness_assistant(prepared.messages, config, drive.context)
    finally:
        await response.close()


def _make_transform_context(lane: Any, drive: Any) -> Any:
    async def _transform(request_context: dict, context: Any) -> dict:
        result = await lane.hooks.run_with_gate(
            "transform_context",
            {"lane": lane.name, "runId": drive.operation_id, **request_context},
            drive.gate,
            context,
        )
        return {
            "messages": request_context["messages"]
            if result is None or _opt(result, "messages") is None
            else _opt(result, "messages"),
            "systemPrompt": request_context["systemPrompt"]
            if result is None or _opt(result, "system_prompt", "systemPrompt") is None
            else _opt(result, "system_prompt", "systemPrompt"),
        }

    return _transform


def _make_before_payload(lane: Any, drive: Any) -> Any:
    async def _before(payload: Any, request_model: Any, context: Any) -> Any:
        result = await lane.hooks.run_with_gate(
            "before_payload",
            {
                "lane": lane.name,
                "runId": drive.operation_id,
                "model": request_model,
                "payload": payload,
            },
            drive.gate,
            context,
        )
        return None if result is None else _opt(result, "payload")

    return _before


def _make_request(lane: Any, drive: Any, prepared: PreparedGeneration) -> Any:
    async def _request(ai_context: Any, options: dict, context: Any) -> Any:
        admitted = with_abort_signal(drive.gate.signal, context)

        def _admitted() -> Any:
            resolved = dict(options)
            resolved["sessionId"] = f"{lane.session.metadata.id}:{lane.name}"
            resolved["signal"] = admitted.signal
            return lane.models.stream_simple(prepared.model, ai_context, resolved)

        return drive.gate.admit(_admitted)

    return _request


async def run_retry_wait(lane: Any, drive: Any, generation: AssistantRetryWaitOperation) -> ProcedureResult:
    """Advance one durable assistant retry wait according to this pass's local wait policy."""
    if _now() < generation.not_before:
        if not drive.wait_for_retry:
            return ProcedureResult(
                kind="waiting",
                outcome={
                    "kind": "waiting",
                    "operationId": drive.operation_id,
                    "reason": "retry",
                    "notBefore": generation.not_before,
                },
            )
        await drive.gate.admit(lambda: wait_until(generation.not_before, drive.gate.signal))

    def _plan(_state: Any, current: Any, _meta: Any, _reader: Any) -> Any:
        next_state = AssistantReadyOperation(
            **operation_scope_of(current),
            generation_context=generation.generation_context,
            next_attempt=generation.next_attempt,
        )
        return CommitDecision(
            writes=[],
            operation_state=next_state,
            materialize=lambda _commit: ProcedureResult(kind="continue"),
            events=lambda _commit: [
                {
                    "type": "retry_start",
                    "lane": lane.name,
                    "runId": drive.operation_id,
                    "step": generation.generation_context.step_id,
                    "attempt": generation.next_attempt,
                }
            ],
        )

    result = await lane.continue_operation(generation, _plan, drive.context)
    return ProcedureResult(kind="continue") if result.kind == "cancel_requested" else result.value


async def run_generation(lane: Any, drive: Any, generation: Any) -> ProcedureResult:
    """Execute one ready assistant generation or advance its durable retry wait."""
    if generation.at == "assistant.retry_wait":
        return await run_retry_wait(lane, drive, generation)

    prepared = await prepare_generation(lane, drive, generation)
    if prepared.kind == "configuration_failure":
        return await publish_configuration_failure(lane, drive, generation, prepared.error)
    if prepared.kind == "cancel_requested":
        return ProcedureResult(kind="continue")

    intent = await publish_generation_intent(lane, drive, generation, prepared)
    if intent.kind == "cancel_requested":
        return ProcedureResult(kind="continue")
    response = await perform_generation(lane, drive, intent.value, prepared)
    return await publish_response(lane, drive, intent.value, response)


def _now() -> int:
    return int(time.time() * 1000)


class _Bag:
    """Attribute view over a plain mapping."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)


def _opt(value: Any, *names: str) -> Any:
    """Read the first present key from a mapping or an attribute."""
    if value is None:
        return None
    if isinstance(value, _Bag):
        value = value._values
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
