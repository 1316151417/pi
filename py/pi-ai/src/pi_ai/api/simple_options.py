"""Provider option preparation from ``packages/ai/src/api/simple-options.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ..types import Model, SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel, TranscriptContext
from ..utils.estimate import estimate_context_tokens

CONTEXT_SAFETY_TOKENS = 4096
MIN_MAX_TOKENS = 1
MIN_ANSWER_TOKENS = 1024
DEFAULT_THINKING_BUDGETS = ThinkingBudgets(minimal=1024, low=2048, medium=8192, high=16384)


def clamp_max_tokens_to_context(model: Model, context: TranscriptContext, max_tokens: int) -> int:
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context).tokens - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model, context: TranscriptContext, options: SimpleStreamOptions | None = None, api_key: str | None = None,
) -> StreamOptions:
    caller_sampling = options.sampling_params if options is not None else None
    sampling_params = (
        {**(model.sampling_params or {}), **(caller_sampling or {})}
        if model.sampling_params is not None or caller_sampling is not None
        else None
    )
    max_tokens = options.max_tokens if options is not None else None
    return StreamOptions(
        temperature=options.temperature if options is not None else None,
        sampling_params=sampling_params,
        max_tokens=clamp_max_tokens_to_context(model, context, model.max_tokens if max_tokens is None else max_tokens),
        signal=options.signal if options is not None else None,
        telemetry_context=options.telemetry_context if options is not None else None,
        api_key=api_key or (options.api_key if options is not None else None),
        fetch=options.fetch if options is not None else None,
        transport=options.transport if options is not None else None,
        cache_retention=options.cache_retention if options is not None else None,
        session_id=options.session_id if options is not None else None,
        headers=options.headers if options is not None else None,
        on_payload=options.on_payload if options is not None else None,
        on_response=options.on_response if options is not None else None,
        timeout_ms=options.timeout_ms if options is not None else None,
        websocket_connect_timeout_ms=options.websocket_connect_timeout_ms if options is not None else None,
        max_retries=options.max_retries if options is not None else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
        metadata=options.metadata if options is not None else None,
        env=options.env if options is not None else None,
    )


def clamp_reasoning(effort: ThinkingLevel | None) -> ThinkingLevel | None:
    return "high" if effort in ("xhigh", "max") else effort


def thinking_budget_for_level(reasoning_level: ThinkingLevel, custom_budgets: ThinkingBudgets | None = None) -> int:
    level = cast(str, clamp_reasoning(reasoning_level))
    custom = cast(int | None, getattr(custom_budgets, level)) if custom_budgets is not None else None
    return custom if custom is not None else cast(int, getattr(DEFAULT_THINKING_BUDGETS, level))


def clamp_thinking_budget_to_answer_room(thinking_budget: int, ceiling: int) -> int:
    return min(thinking_budget, max(0, ceiling - MIN_ANSWER_TOKENS))


@dataclass
class ThinkingTokenLimits:
    max_tokens: int
    thinking_budget: int


def adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> ThinkingTokenLimits:
    thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets)
    max_tokens = (
        model_max_tokens if base_max_tokens is None else min(base_max_tokens + thinking_budget, model_max_tokens)
    )
    if max_tokens <= thinking_budget:
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens)
    return ThinkingTokenLimits(max_tokens=max_tokens, thinking_budget=thinking_budget)
