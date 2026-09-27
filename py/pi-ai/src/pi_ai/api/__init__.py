"""Shared request preparation primitives for pi-ai provider adapters."""

from .cloudflare import (
    CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL,
    CLOUDFLARE_WORKERS_AI_BASE_URL,
)
from .cloudflare_ai_binding import AiBinding, CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL, create_ai_binding_fetch
from .constrained_sampling import (
    GrammarConstrainedSampling,
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    make_strict_json_schema,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .openai_prompt_cache import OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH, clamp_openai_prompt_cache_key
from .github_copilot_headers import (
    CopilotDynamicHeaderParams, build_copilot_dynamic_headers,
    has_copilot_vision_input, infer_copilot_initiator,
)
from .simple_options import (
    DEFAULT_THINKING_BUDGETS,
    MIN_ANSWER_TOKENS,
    ThinkingTokenLimits,
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
    clamp_reasoning,
    clamp_thinking_budget_to_answer_room,
    thinking_budget_for_level,
)
from .transform_messages import transform_messages

__all__ = [
    "AiBinding",
    "CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL",
    "CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL",
    "CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL",
    "CLOUDFLARE_WORKERS_AI_BASE_URL",
    "CLOUDFLARE_GATEWAY_BINDING_AUTH_SENTINEL",
    "CopilotDynamicHeaderParams",
    "build_copilot_dynamic_headers",
    "create_ai_binding_fetch",
    "has_copilot_vision_input",
    "infer_copilot_initiator",
    "DEFAULT_THINKING_BUDGETS",
    "GrammarConstrainedSampling",
    "GrammarToolInputJsonBuffer",
    "MIN_ANSWER_TOKENS",
    "OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH",
    "ThinkingTokenLimits",
    "adjust_max_tokens_for_thinking",
    "append_grammar_tool_input_json_delta",
    "build_base_options",
    "clamp_max_tokens_to_context",
    "clamp_openai_prompt_cache_key",
    "clamp_reasoning",
    "clamp_thinking_budget_to_answer_room",
    "create_grammar_tool_input_properties",
    "get_grammar_tool_input",
    "get_json_schema_tool_parameters",
    "make_strict_json_schema",
    "resolve_grammar_constrained_sampling",
    "resolve_json_schema_strict_sampling",
    "thinking_budget_for_level",
    "transform_messages",
]
