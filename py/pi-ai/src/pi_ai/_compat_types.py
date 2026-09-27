"""Provider compatibility and constrained-sampling contracts from ai/types.ts.

These TypedDict records retain wire field names, including camelCase and
``$var``. Optional fields remain absent unless supplied; these declarations do
not apply the provider-specific defaults described by the TypeScript API.
"""

from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from .types import ModelCost, ProviderId


ChatTemplateVariable = TypedDict(
    "ChatTemplateVariable",
    {
        "$var": Literal["thinking.enabled", "thinking.effort", "thinking.budget"],
        "omitWhenOff": NotRequired[bool],
    },
)

type ChatTemplateKwargValue = str | int | float | bool | None | ChatTemplateVariable
type ThinkingTokenBudgetField = Literal["thinking_token_budget", "thinking_budget", "thinking_budget_tokens"]
type SessionAffinityFormat = Literal["openai", "openai-nosession", "openrouter"]


class TextSignatureV1(TypedDict):
    v: Literal[1]
    id: str
    phase: NotRequired[Literal["commentary", "final_answer"]]


type GrammarFormat = Literal["openai_lark", "openai_regex"]


class GrammarVariants(TypedDict):
    openai_lark: NotRequired[str]
    openai_regex: NotRequired[str]


class JsonSchemaConstrainedSamplingConfig(TypedDict):
    type: Literal["json_schema"]
    strict: Literal["prefer", "require"]


class GrammarConstrainedSamplingConfig(TypedDict):
    type: Literal["grammar"]
    variants: GrammarVariants


type ConstrainedSamplingConfig = JsonSchemaConstrainedSamplingConfig | GrammarConstrainedSamplingConfig


class AnthropicAllowedFallbackModel(TypedDict):
    provider: "ProviderId"
    model: str
    cost: "ModelCost"


class OpenRouterSort(TypedDict):
    by: NotRequired[str]
    partition: NotRequired[str | None]


class OpenRouterMaxPrice(TypedDict):
    prompt: NotRequired[int | float | str]
    completion: NotRequired[int | float | str]
    image: NotRequired[int | float | str]
    audio: NotRequired[int | float | str]
    request: NotRequired[int | float | str]


class OpenRouterPercentileCutoffs(TypedDict):
    p50: NotRequired[int | float]
    p75: NotRequired[int | float]
    p90: NotRequired[int | float]
    p99: NotRequired[int | float]


class OpenRouterRouting(TypedDict):
    """Preferences sent in the OpenRouter request's provider field."""

    allow_fallbacks: NotRequired[bool]
    require_parameters: NotRequired[bool]
    data_collection: NotRequired[Literal["deny", "allow"]]
    zdr: NotRequired[bool]
    enforce_distillable_text: NotRequired[bool]
    order: NotRequired[list[str]]
    only: NotRequired[list[str]]
    ignore: NotRequired[list[str]]
    quantizations: NotRequired[list[str]]
    sort: NotRequired[str | OpenRouterSort]
    max_price: NotRequired[OpenRouterMaxPrice]
    preferred_min_throughput: NotRequired[int | float | OpenRouterPercentileCutoffs]
    preferred_max_latency: NotRequired[int | float | OpenRouterPercentileCutoffs]


class VercelGatewayRouting(TypedDict):
    only: NotRequired[list[str]]
    order: NotRequired[list[str]]


class OpenAICompletionsCompat(TypedDict):
    """Overrides for OpenAI-compatible Chat Completions transports."""

    supportsStore: NotRequired[bool]
    supportsDeveloperRole: NotRequired[bool]
    supportsReasoningEffort: NotRequired[bool]
    supportsUsageInStreaming: NotRequired[bool]
    supportsFinishReason: NotRequired[bool]
    maxTokensField: NotRequired[Literal["max_completion_tokens", "max_tokens"]]
    requiresToolResultName: NotRequired[bool]
    requiresAssistantAfterToolResult: NotRequired[bool]
    requiresThinkingAsText: NotRequired[bool]
    requiresReasoningContentOnAssistantMessages: NotRequired[bool]
    thinkingFormat: NotRequired[Literal[
        "openai", "openrouter", "deepseek", "together", "baseten", "zai", "qwen",
        "chat-template", "qwen-chat-template", "string-thinking", "ant-ling",
    ]]
    chatTemplateKwargs: NotRequired[dict[str, ChatTemplateKwargValue]]
    chatTemplateArgs: NotRequired[dict[str, ChatTemplateKwargValue]]
    openRouterRouting: NotRequired[OpenRouterRouting]
    vercelGatewayRouting: NotRequired[VercelGatewayRouting]
    zaiToolStream: NotRequired[bool]
    thinkingTokenBudgetField: NotRequired[ThinkingTokenBudgetField]
    supportsThinkingTokenBudget: NotRequired[bool]
    supportsOpenAIGrammarTools: NotRequired[bool]
    supportsMidConvoSystemMessages: NotRequired[bool]
    supportsMidConvoToolAdditions: NotRequired[bool]
    supportsStrictMode: NotRequired[bool]
    cacheControlFormat: NotRequired[Literal["anthropic"]]
    sendSessionAffinityHeaders: NotRequired[bool]
    sessionAffinityFormat: NotRequired[SessionAffinityFormat]
    supportsLongCacheRetention: NotRequired[bool]
    vllmPriority: NotRequired[int | float]


class OpenAIResponsesCompat(TypedDict):
    """Compatibility settings for OpenAI, Azure, and Codex Responses."""

    supportsDeveloperRole: NotRequired[bool]
    supportsMidConvoSystemMessages: NotRequired[bool]
    sessionAffinityFormat: NotRequired[SessionAffinityFormat]
    supportsLongCacheRetention: NotRequired[bool]
    supportsStrictMode: NotRequired[bool]
    supportsOpenAIGrammarTools: NotRequired[bool]
    supportsAdditionalTools: NotRequired[bool]
    supportsToolSearch: NotRequired[bool]
    supportsExplicitPromptCacheMode: NotRequired[bool]
    supportsMaxOutputTokens: NotRequired[bool]


class AnthropicMessagesCompat(TypedDict):
    supportsEagerToolInputStreaming: NotRequired[bool]
    supportsLongCacheRetention: NotRequired[bool]
    sendSessionAffinityHeaders: NotRequired[bool]
    sessionAffinityFormat: NotRequired[Literal["openrouter"]]
    supportsCacheControlOnTools: NotRequired[bool]
    supportsTemperature: NotRequired[bool]
    forceAdaptiveThinking: NotRequired[bool]
    allowEmptySignature: NotRequired[bool]
    supportsStrictTools: NotRequired[bool]
    supportsMidConvoEffort: NotRequired[bool]
    supportsMidConvoSystemMessages: NotRequired[bool]
    supportsMidConvoToolChanges: NotRequired[bool]
    allowedFallbackModels: NotRequired[list[AnthropicAllowedFallbackModel]]


class BedrockCompat(TypedDict):
    supportsStrictMode: NotRequired[bool]


class MistralConversationsCompat(TypedDict):
    supportsMidConvoSystemMessages: NotRequired[bool]


__all__ = [
    "AnthropicAllowedFallbackModel", "AnthropicMessagesCompat", "BedrockCompat",
    "ChatTemplateKwargValue", "ChatTemplateVariable", "ConstrainedSamplingConfig",
    "GrammarConstrainedSamplingConfig", "GrammarFormat", "GrammarVariants",
    "JsonSchemaConstrainedSamplingConfig", "MistralConversationsCompat",
    "OpenAICompletionsCompat", "OpenAIResponsesCompat", "OpenRouterMaxPrice",
    "OpenRouterPercentileCutoffs", "OpenRouterRouting", "OpenRouterSort",
    "SessionAffinityFormat", "TextSignatureV1", "ThinkingTokenBudgetField",
    "VercelGatewayRouting",
]
