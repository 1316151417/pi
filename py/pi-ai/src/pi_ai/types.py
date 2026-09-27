"""Core message and content types ported from @earendil-works/pi-ai ``src/types.ts``.

The TypeScript package models messages as plain JSON objects discriminated by a
``role`` field. This port uses dataclasses with an explicit ``role`` so the
structures stay serializable while remaining statically checkable. Custom
application messages can be any object exposing a ``role`` attribute.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, MutableSequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping, NotRequired, Optional, Sequence, TypedDict, Union, cast

from pi_telemetry import TelemetryContext

from .abort import AbortSignal
from ._model_wire import ModelFields, ModelList
from ._values import JSON_NULL, UNDEFINED, JsonNull, Undefined
from ._compat_types import (
    AnthropicAllowedFallbackModel, AnthropicMessagesCompat, BedrockCompat, ChatTemplateKwargValue,
    ChatTemplateVariable, ConstrainedSamplingConfig, GrammarConstrainedSamplingConfig, GrammarFormat,
    GrammarVariants, JsonSchemaConstrainedSamplingConfig, MistralConversationsCompat, OpenAICompletionsCompat,
    OpenAIResponsesCompat, OpenRouterMaxPrice, OpenRouterPercentileCutoffs, OpenRouterRouting, OpenRouterSort,
    SessionAffinityFormat, TextSignatureV1, ThinkingTokenBudgetField, VercelGatewayRouting,
)

if TYPE_CHECKING:
    from .event_stream import AssistantMessageEventStream

__all__ = [
    "TextContent",
    "ThinkingContent",
    "ImageContent",
    "ToolCall",
    "Usage",
    "Cost",
    "StopReason",
    "Tool",
    "ToolReference",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "AssistantMessageDiagnostic",
    "DiagnosticErrorInfo",
    "ToolResultMessage",
    "Message",
    "AgentMessage",
    "Model",
    "ModelCost",
    "ModelCostRates",
    "ModelCostTier",
    "ImagesModel",
    "model_from_json",
    "ThinkingLevel",
    "ModelThinkingLevel",
    "ThinkingBudgets",
    "Transport",
    "ToolChoice",
    "CacheRetention",
    "TranscriptContext",
    "Context",
    "JsonValue",
    "JsonObject",
    "SimpleStreamOptions",
    "StreamOptions",
    "ProviderResponse",
    "ProviderEnv",
    "ProviderHeaders",
    "FetchFunction",
    "DeferredHandle",
    "AssistantMessageEvent",
    "is_system_message",
    "is_user_message",
    "is_assistant_message",
    "is_tool_result_message",
    "message_to_json",
    "message_from_json",
    "content_block_to_json",
    "content_block_from_json",
]

JsonPrimitive = Union[None, bool, float, int, str]
JsonValue = Union[JsonPrimitive, list["JsonValue"], "JsonObject"]
JsonObject = dict[str, JsonValue]

type KnownApi = Literal[
    "openai-completions", "mistral-conversations", "openai-responses", "azure-openai-responses",
    "openai-codex-responses", "anthropic-messages", "bedrock-converse-stream", "google-generative-ai",
    "google-vertex", "pi-messages",
]
type Api = str
type KnownImagesApi = Literal["openrouter-images"]
type ImagesApi = str
type KnownProvider = Literal[
    "amazon-bedrock", "ant-ling", "anthropic", "google", "google-vertex", "openai", "azure-openai-responses",
    "openai-codex", "radius", "nvidia", "deepseek", "github-copilot", "xai", "groq", "cerebras", "openrouter",
    "vercel-ai-gateway", "zai", "zai-coding-cn", "mistral", "minimax", "minimax-cn", "moonshotai", "moonshotai-cn",
    "huggingface", "fireworks", "together", "baseten", "opencode", "opencode-go", "kimi-coding",
    "cloudflare-workers-ai", "cloudflare-ai-gateway", "qwen-token-plan", "qwen-token-plan-cn",
    "qwen-token-plan-individual", "xiaomi", "xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp",
]
type ProviderId = str
type KnownImagesProvider = Literal["openrouter"]
type ImagesProviderId = str
type ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
type ModelThinkingLevel = Literal["off"] | ThinkingLevel
type ThinkingLevelMap = dict[ModelThinkingLevel, str | None]
type Transport = Literal["sse", "websocket", "websocket-cached", "auto"]
type ToolChoice = Literal["auto", "none"]
type CacheRetention = Literal["none", "short", "long"]
ProviderEnv = dict[str, str]
ProviderHeaders = dict[str, str | None]
FetchFunction = Callable[..., Awaitable[object]]


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass
class TextContent:
    type: str = field(default="text", init=False)
    text: str = ""
    text_signature: Optional[str] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"type": "text", "text": self.text}
        if self.text_signature is not None:
            data["textSignature"] = self.text_signature
        return data


@dataclass
class ThinkingContent:
    type: str = field(default="thinking", init=False)
    thinking: str = ""
    thinking_signature: Optional[str] = None
    redacted: Optional[bool] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"type": "thinking", "thinking": self.thinking}
        if self.thinking_signature is not None:
            data["thinkingSignature"] = self.thinking_signature
        if self.redacted is not None:
            data["redacted"] = self.redacted
        return data


@dataclass
class ImageContent:
    type: str = field(default="image", init=False)
    data: str = ""
    mime_type: str = ""

    def to_json(self) -> JsonObject:
        return {"type": "image", "data": self.data, "mimeType": self.mime_type}


@dataclass
class ToolCall:
    type: str = field(default="toolCall", init=False)
    id: str = ""
    name: str = ""
    arguments: JsonObject = field(default_factory=dict)
    thought_signature: Optional[str] = None
    namespace: Optional[str] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "type": "toolCall",
            "id": self.id,
            "name": self.name,
            "arguments": copy.deepcopy(self.arguments),
        }
        if self.thought_signature is not None:
            data["thoughtSignature"] = self.thought_signature
        if self.namespace is not None:
            data["namespace"] = self.namespace
        return data


AssistantContentBlock = Union[TextContent, ThinkingContent, ToolCall]
UserContentBlock = Union[TextContent, ImageContent]
ToolResultContentBlock = Union[TextContent, ImageContent]


# ---------------------------------------------------------------------------
# Usage / cost
# ---------------------------------------------------------------------------


@dataclass
class Cost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0

    def to_json(self) -> JsonObject:
        return {
            "input": self.input,
            "output": self.output,
            "cacheRead": self.cache_read,
            "cacheWrite": self.cache_write,
            "total": self.total,
        }


EMPTY_COST = Cost()


def empty_cost() -> Cost:
    return Cost()


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: Optional[int] = None
    reasoning: Optional[int] = None
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "input": self.input,
            "output": self.output,
            "cacheRead": self.cache_read,
            "cacheWrite": self.cache_write,
            "totalTokens": self.total_tokens,
            "cost": self.cost.to_json(),
        }
        if self.cache_write_1h is not None:
            data["cacheWrite1h"] = self.cache_write_1h
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        return data


def empty_usage() -> Usage:
    return Usage()


def usage_to_json(usage: Optional[Usage]) -> Optional[JsonObject]:
    if usage is None:
        return None
    return usage.to_json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    """Tool declaration. ``parameters`` is a JSON-schema style dict."""

    name: str
    description: str
    parameters: JsonObject = field(default_factory=dict)
    constrained_sampling: Union[bool, JsonObject, None] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }
        if self.constrained_sampling is not None:
            data["constrainedSampling"] = copy.deepcopy(self.constrained_sampling)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Tool:
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=dict(data.get("parameters") or {}),
            constrained_sampling=copy.deepcopy(data.get("constrainedSampling")),
        )


@dataclass
class ToolReference:
    name: str

    def to_json(self) -> JsonObject:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

StopReason = str  # "pending" | "stop" | "length" | "toolUse" | "error" | "aborted" | "deferred"


@dataclass
class SystemMessage:
    role: str = field(default="system", init=False)
    content: Union[str, list[TextContent]] = ""
    sections: Optional[dict[str, Optional[str]]] = None
    tools_added: Optional[list[Tool]] = None
    tools_removed: Optional[list[ToolReference]] = None
    timestamp: int = 0

    def to_json(self) -> JsonObject:
        content_json: JsonValue
        if isinstance(self.content, str):
            content_json = self.content
        else:
            content_json = [block.to_json() for block in self.content]
        data: JsonObject = {"role": "system", "content": content_json, "timestamp": self.timestamp}
        if self.sections is not None:
            data["sections"] = dict(self.sections)
        if self.tools_added is not None:
            data["toolsAdded"] = [tool.to_json() for tool in self.tools_added]
        if self.tools_removed is not None:
            data["toolsRemoved"] = [ref.to_json() for ref in self.tools_removed]
        return data


@dataclass
class UserMessage:
    role: str = field(default="user", init=False)
    content: Union[str, list[UserContentBlock]] = ""
    timestamp: int = 0

    def to_json(self) -> JsonObject:
        content_json: JsonValue
        if isinstance(self.content, str):
            content_json = self.content
        else:
            content_json = [block.to_json() for block in self.content]
        return {"role": "user", "content": content_json, "timestamp": self.timestamp}


@dataclass
class DeferredHandle:
    provider: str
    model_id: str
    api: str
    id: str
    expires_at: Optional[int] = None
    poll_after_ms: Optional[int] = None
    data: Optional[JsonValue] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "provider": self.provider,
            "modelId": self.model_id,
            "api": self.api,
            "id": self.id,
        }
        if self.expires_at is not None:
            data["expiresAt"] = self.expires_at
        if self.poll_after_ms is not None:
            data["pollAfterMs"] = self.poll_after_ms
        if self.data is not None:
            data["data"] = copy.deepcopy(self.data)
        return data


@dataclass
class DiagnosticErrorInfo:
    message: str
    name: str | None = None
    stack: str | None = None
    code: str | int | float | None = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"message": self.message}
        if self.name is not None:
            data["name"] = self.name
        if self.stack is not None:
            data["stack"] = self.stack
        if self.code is not None:
            data["code"] = self.code
        return data


@dataclass
class AssistantMessageDiagnostic:
    type: str
    timestamp: int
    error: DiagnosticErrorInfo | None = None
    details: JsonObject | None = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"type": self.type, "timestamp": self.timestamp}
        if self.error is not None:
            data["error"] = self.error.to_json()
        if self.details is not None:
            data["details"] = self.details
        return data


@dataclass
class AssistantMessage:
    role: str = field(default="assistant", init=False)
    content: list[AssistantContentBlock] = field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    response_model: Optional[str] = None
    response_id: Optional[str] = None
    provider_thinking_level: Optional[str] = None
    diagnostics: Optional[list[AssistantMessageDiagnostic]] = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "pending"
    deferred: Optional[DeferredHandle] = None
    error_message: Optional[str] = None
    raw_stop_reason: Optional[str] = None
    end_turn: Optional[bool] = None
    timestamp: int = 0

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "role": "assistant",
            "content": [block.to_json() for block in self.content],
            "api": self.api,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_json(),
            "stopReason": self.stop_reason,
            "timestamp": self.timestamp,
        }
        if self.response_model is not None:
            data["responseModel"] = self.response_model
        if self.response_id is not None:
            data["responseId"] = self.response_id
        if self.provider_thinking_level is not None:
            data["providerThinkingLevel"] = self.provider_thinking_level
        if self.diagnostics is not None:
            data["diagnostics"] = [diagnostic.to_json() for diagnostic in self.diagnostics]
        if self.deferred is not None:
            data["deferred"] = self.deferred.to_json()
        if self.error_message is not None:
            data["errorMessage"] = self.error_message
        if self.raw_stop_reason is not None:
            data["rawStopReason"] = self.raw_stop_reason
        if self.end_turn is not None:
            data["endTurn"] = self.end_turn
        return data


@dataclass
class ToolResultMessage:
    role: str = field(default="toolResult", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultContentBlock] = field(default_factory=list)
    details: Optional[JsonValue] = None
    usage: Optional[Usage] = None
    is_error: bool = False
    timestamp: int = 0

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "role": "toolResult",
            "toolCallId": self.tool_call_id,
            "toolName": self.tool_name,
            "content": [block.to_json() for block in self.content],
            "isError": self.is_error,
            "timestamp": self.timestamp,
        }
        if self.details is not None:
            data["details"] = copy.deepcopy(self.details)
        if self.usage is not None:
            data["usage"] = self.usage.to_json()
        return data


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]
AgentMessage = Any  # Message union extended with app-specific custom messages.


def is_system_message(message: Any) -> bool:
    return getattr(message, "role", None) == "system"


def is_user_message(message: Any) -> bool:
    return getattr(message, "role", None) == "user"


def is_assistant_message(message: Any) -> bool:
    return getattr(message, "role", None) == "assistant"


def is_tool_result_message(message: Any) -> bool:
    return getattr(message, "role", None) == "toolResult"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def content_block_to_json(block: Any) -> JsonObject:
    return block.to_json()


def content_block_from_json(data: Mapping[str, Any]) -> Any:
    block_type = data.get("type")
    if block_type == "text":
        return TextContent(text=data.get("text", ""), text_signature=data.get("textSignature"))
    if block_type == "thinking":
        return ThinkingContent(
            thinking=data.get("thinking", ""),
            thinking_signature=data.get("thinkingSignature"),
            redacted=data.get("redacted"),
        )
    if block_type == "image":
        return ImageContent(data=data.get("data", ""), mime_type=data.get("mimeType", ""))
    if block_type == "toolCall":
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=cast(JsonObject, copy.deepcopy(data.get("arguments", {}))),
            thought_signature=data.get("thoughtSignature"),
            namespace=data.get("namespace"),
        )
    raise ValueError(f"Unknown content block type: {block_type!r}")


def message_to_json(message: Any) -> JsonObject:
    role = getattr(message, "role", None)
    if role == "system":
        return message.to_json()
    if role == "user":
        return message.to_json()
    if role == "assistant":
        return message.to_json()
    if role == "toolResult":
        return message.to_json()
    if hasattr(message, "to_json"):
        return message.to_json()
    raise ValueError(f"Cannot serialize message with role {role!r}")


def _usage_from_json(data: Optional[Mapping[str, Any]]) -> Usage:
    if data is None:
        return Usage()
    cost_data = data.get("cost") or {}
    return Usage(
        input=int(data.get("input", 0)),
        output=int(data.get("output", 0)),
        cache_read=int(data.get("cacheRead", 0)),
        cache_write=int(data.get("cacheWrite", 0)),
        cache_write_1h=data.get("cacheWrite1h"),
        reasoning=data.get("reasoning"),
        total_tokens=int(data.get("totalTokens", 0)),
        cost=Cost(
            input=float(cost_data.get("input", 0)),
            output=float(cost_data.get("output", 0)),
            cache_read=float(cost_data.get("cacheRead", 0)),
            cache_write=float(cost_data.get("cacheWrite", 0)),
            total=float(cost_data.get("total", 0)),
        ),
    )


def usage_from_json(data: Optional[Mapping[str, Any]]) -> Optional[Usage]:
    if data is None:
        return None
    return _usage_from_json(data)


def message_from_json(data: Mapping[str, Any]) -> Message:
    role = data.get("role")
    if role == "system":
        content = data.get("content", "")
        if isinstance(content, list):
            content = [content_block_from_json(block) for block in content if block.get("type") == "text"]
        sections = data.get("sections")
        return SystemMessage(
            content=content,
            sections=dict(sections) if sections is not None else None,
            tools_added=[Tool.from_json(t) for t in data.get("toolsAdded") or []] or None,
            tools_removed=[ToolReference(name=r["name"]) for r in data.get("toolsRemoved") or []] or None,
            timestamp=int(data.get("timestamp", 0)),
        )
    if role == "user":
        content = data.get("content", "")
        if isinstance(content, list):
            blocks: list[UserContentBlock] = []
            for block in content:
                if block.get("type") == "text":
                    blocks.append(TextContent(text=block.get("text", ""), text_signature=block.get("textSignature")))
                elif block.get("type") == "image":
                    blocks.append(ImageContent(data=block.get("data", ""), mime_type=block.get("mimeType", "")))
            content = blocks
        return UserMessage(content=content, timestamp=int(data.get("timestamp", 0)))
    if role == "assistant":
        content: list[AssistantContentBlock] = [content_block_from_json(block) for block in data.get("content", [])]
        deferred_data = data.get("deferred")
        diagnostics = None
        if data.get("diagnostics") is not None:
            diagnostics = []
            for diagnostic in data["diagnostics"]:
                error = diagnostic.get("error")
                diagnostics.append(AssistantMessageDiagnostic(
                    type=diagnostic["type"],
                    timestamp=diagnostic["timestamp"],
                    error=DiagnosticErrorInfo(
                        message=error["message"], name=error.get("name"),
                        stack=error.get("stack"), code=error.get("code"),
                    ) if error is not None else None,
                    details=copy.deepcopy(diagnostic.get("details")),
                ))
        return AssistantMessage(
            content=content,
            api=data.get("api", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            response_model=data.get("responseModel"),
            response_id=data.get("responseId"),
            provider_thinking_level=data.get("providerThinkingLevel"),
            diagnostics=diagnostics,
            usage=_usage_from_json(data.get("usage")),
            stop_reason=data.get("stopReason", "pending"),
            deferred=(
                DeferredHandle(
                    provider=deferred_data["provider"],
                    model_id=deferred_data["modelId"],
                    api=deferred_data["api"],
                    id=deferred_data["id"],
                    expires_at=deferred_data.get("expiresAt"),
                    poll_after_ms=deferred_data.get("pollAfterMs"),
                    data=deferred_data.get("data"),
                )
                if deferred_data
                else None
            ),
            error_message=data.get("errorMessage"),
            raw_stop_reason=data.get("rawStopReason"),
            end_turn=data.get("endTurn"),
            timestamp=int(data.get("timestamp", 0)),
        )
    if role == "toolResult":
        content: list[ToolResultContentBlock] = [content_block_from_json(block) for block in data.get("content", [])]
        return ToolResultMessage(
            tool_call_id=data.get("toolCallId", ""),
            tool_name=data.get("toolName", ""),
            content=content,
            details=copy.deepcopy(data.get("details")),
            usage=usage_from_json(data.get("usage")),
            is_error=bool(data.get("isError", False)),
            timestamp=int(data.get("timestamp", 0)),
        )
    raise ValueError(f"Unknown message role: {role!r}")


# ---------------------------------------------------------------------------
# Model / streaming types
# ---------------------------------------------------------------------------


@dataclass
class ModelCostRates(ModelFields):
    _wire_names: ClassVar[dict[str, str]] = {
        "input": "input", "output": "output", "cache_read": "cacheRead", "cache_write": "cacheWrite",
    }
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class ModelCostTier(ModelCostRates):
    _wire_names: ClassVar[dict[str, str]] = {
        **ModelCostRates._wire_names, "input_tokens_above": "inputTokensAbove",
    }
    input_tokens_above: float = 0.0


@dataclass
class ModelCost(ModelCostRates):
    _wire_names: ClassVar[dict[str, str]] = {**ModelCostRates._wire_names, "tiers": "tiers"}
    _optional_fields: ClassVar[frozenset[str]] = frozenset({"tiers"})
    tiers: MutableSequence[ModelCostTier] | None = None

    def _view_wire_field(self, name: str, value: object) -> object:
        if name == "tiers" and isinstance(value, list):
            cached = self.__dict__.get("_tiers_view")
            if cached is None or cached[0] is not value:
                cached = (value, ModelList(value, ModelCostTier._from_wire))
                object.__setattr__(self, "_tiers_view", cached)
            return cached[1]
        return value


@dataclass
class Model(ModelFields):
    _wire_names: ClassVar[dict[str, str]] = {
        "id": "id", "name": "name", "api": "api", "provider": "provider", "base_url": "baseUrl",
        "reasoning": "reasoning", "thinking_level_map": "thinkingLevelMap", "input": "input", "cost": "cost",
        "context_window": "contextWindow", "max_tokens": "maxTokens", "sampling_params": "samplingParams",
        "headers": "headers", "compat": "compat",
    }
    _optional_fields: ClassVar[frozenset[str]] = frozenset({"thinking_level_map", "sampling_params", "headers", "compat"})
    id: str
    name: str
    api: str
    provider: str
    base_url: str
    reasoning: bool = False
    thinking_level_map: Optional[dict[str, Optional[str]]] = None
    input: list[str] = field(default_factory=lambda: ["text", "image"])
    cost: ModelCost = field(default_factory=ModelCost)
    context_window: int = 0
    max_tokens: int = 0
    sampling_params: Optional[dict[str, object]] = None
    headers: Optional[dict[str, str]] = None
    compat: Optional[dict[str, object]] = None

    def _view_wire_field(self, name: str, value: object) -> object:
        if name == "cost" and isinstance(value, Mapping):
            cached = self.__dict__.get("_cost_view")
            if cached is None or cached[0] is not value:
                cached = (value, ModelCost._from_wire(cast(Mapping[str, object], value)))
                object.__setattr__(self, "_cost_view", cached)
            return cached[1]
        return value


@dataclass
class ImagesModel(ModelFields):
    _wire_names: ClassVar[dict[str, str]] = {
        key: value for key, value in Model._wire_names.items()
        if key not in {"reasoning", "context_window", "max_tokens", "compat"}
    } | {"output": "output"}
    _optional_fields: ClassVar[frozenset[str]] = frozenset({"thinking_level_map", "sampling_params", "headers"})
    id: str
    name: str
    api: str
    provider: str
    base_url: str
    input: list[str]
    output: list[str]
    cost: ModelCost
    thinking_level_map: dict[str, str | None] | None = None
    sampling_params: dict[str, object] | None = None
    headers: dict[str, str] | None = None

    _view_wire_field = Model._view_wire_field


def model_from_json(data: Mapping[str, object]) -> Model:
    """View a model object without filling missing fields or copying nested data."""
    return Model._from_wire(data)


def images_model_from_json(data: Mapping[str, object]) -> ImagesModel:
    return ImagesModel._from_wire(data)


@dataclass
class ThinkingBudgets:
    minimal: Optional[int] = None
    low: Optional[int] = None
    medium: Optional[int] = None
    high: Optional[int] = None


@dataclass
class ProviderResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class TranscriptContext:
    """Normalized request context produced by :func:`normalize_context`."""

    messages: list[Message] = field(default_factory=list)


@dataclass
class Context:
    system_prompt: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    tools: Optional[list[Tool]] = None


# ---------------------------------------------------------------------------
# Assistant message event protocol
# ---------------------------------------------------------------------------


@dataclass
class AssistantMessageEvent:
    """Event protocol for AssistantMessageEventStream (see pi-ai types.ts)."""

    type: str
    partial: Optional[AssistantMessage] = None
    content_index: Optional[int] = None
    delta: Optional[str] = None
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    reason: Optional[str] = None
    message: Optional[AssistantMessage] = None
    error: Optional[AssistantMessage] = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {"type": self.type}
        if self.partial is not None:
            data["partial"] = self.partial.to_json()
        if self.content_index is not None:
            data["contentIndex"] = self.content_index
        if self.delta is not None:
            data["delta"] = self.delta
        if self.content is not None:
            data["content"] = self.content
        if self.tool_call is not None:
            data["toolCall"] = self.tool_call.to_json()
        if self.reason is not None:
            data["reason"] = self.reason
        if self.message is not None:
            data["message"] = self.message.to_json()
        if self.error is not None:
            data["error"] = self.error.to_json()
        return data


def assistant_message_event_from_json(data: Mapping[str, Any]) -> AssistantMessageEvent:
    message_data = data.get("message") or data.get("error")
    return AssistantMessageEvent(
        type=data["type"],
        partial=message_from_json(data["partial"]) if data.get("partial") else None,
        content_index=data.get("contentIndex"),
        delta=data.get("delta"),
        content=data.get("content"),
        tool_call=content_block_from_json(data["toolCall"]) if data.get("toolCall") else None,
        reason=data.get("reason"),
        message=message_from_json(message_data) if message_data else None,
        error=message_from_json(data["error"]) if data.get("error") else None,
    )


# ---------------------------------------------------------------------------
# Provider request and streaming contracts
# ---------------------------------------------------------------------------


@dataclass
class ProviderRequestOptions[TModel]:
    """HTTP transport and lifecycle callbacks shared by provider requests.

    An on_payload callback can return None or UNDEFINED to keep the payload;
    return JSON_NULL to explicitly replace it with JSON null. Other values,
    including false, zero, and empty containers, replace the payload unchanged.
    """

    signal: Optional[AbortSignal] = None
    telemetry_context: TelemetryContext | None = None
    api_key: Optional[str] = None
    fetch: FetchFunction | None = None
    env: ProviderEnv | None = None
    on_payload: Callable[[object, TModel], object] | None = None
    on_response: Callable[[ProviderResponse, TModel], None | Awaitable[None]] | None = None
    headers: Optional[dict[str, Optional[str]]] = None
    timeout_ms: Optional[int] = None
    max_retries: Optional[int] = None
    max_retry_delay_ms: Optional[int] = None


@dataclass
class StreamOptions(ProviderRequestOptions[Model]):
    temperature: Optional[float] = None
    sampling_params: Optional[dict[str, object]] = None
    max_tokens: Optional[int] = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: Optional[str] = None
    websocket_connect_timeout_ms: int | None = None
    metadata: Optional[dict[str, object]] = None


class DeferredOptions(TypedDict):
    window: NotRequired[Literal["15m", "1h", "24h"]]


@dataclass
class SimpleStreamOptions(StreamOptions):
    tool_choice: Optional[ToolChoice] = None
    reasoning: Optional[ThinkingLevel] = None
    deferred: bool | DeferredOptions | None = None
    thinking_budgets: Optional[ThinkingBudgets] = None


@dataclass
class DeferredFetchOptions(ProviderRequestOptions[Model]):
    wait: int | float | None = None


@dataclass
class DeferredCancelOptions(ProviderRequestOptions[Model]):
    pass


ProviderStreamOptions = StreamOptions


@dataclass
class ImagesOptions(ProviderRequestOptions[ImagesModel]):
    metadata: dict[str, object] | None = None


ProviderImagesOptions = ImagesOptions
type ImagesInput = TextContent | ImageContent
type ImagesOutput = TextContent | ImageContent
type ImagesInputContent = TextContent | ImageContent
type ImagesOutputContent = TextContent | ImageContent
type ImagesStopReason = Literal["stop", "error", "aborted"]


@dataclass
class ImagesContext:
    input: list[ImagesInput]


@dataclass
class AssistantImages:
    api: ImagesApi
    provider: ImagesProviderId
    model: str
    output: list[ImagesOutput]
    stop_reason: ImagesStopReason
    timestamp: int | float
    usage: Usage | None = None
    response_id: str | None = None
    error_message: str | None = None

    def to_json(self) -> JsonObject:
        data: JsonObject = {
            "api": self.api, "provider": self.provider, "model": self.model,
            "output": [part.to_json() for part in self.output],
            "stopReason": self.stop_reason, "timestamp": self.timestamp,
        }
        if self.response_id is not None:
            data["responseId"] = self.response_id
        if self.usage is not None:
            data["usage"] = self.usage.to_json()
        if self.error_message is not None:
            data["errorMessage"] = self.error_message
        return data


type StreamFunction = Callable[[Model, TranscriptContext, StreamOptions | None], AssistantMessageEventStream]
type SimpleStreamFunction = Callable[[Model, TranscriptContext, SimpleStreamOptions | None], AssistantMessageEventStream]
type ImagesFunction = Callable[[ImagesModel, ImagesContext, ImagesOptions | None], Awaitable[AssistantImages]]
type DeferredFetchFunction = Callable[[Model, DeferredHandle, DeferredFetchOptions | None], AssistantMessageEventStream]
type DeferredCancelFunction = Callable[[Model, DeferredHandle, DeferredCancelOptions | None], Awaitable[None]]


@dataclass
class ProviderStreams:
    stream: StreamFunction
    stream_simple: SimpleStreamFunction
    fetch_deferred: DeferredFetchFunction | None = None
    cancel_deferred: DeferredCancelFunction | None = None


@dataclass
class ProviderImages:
    generate_images: ImagesFunction


__all__ += [
    "JSON_NULL", "UNDEFINED", "JsonNull", "Undefined",
    "AnthropicAllowedFallbackModel", "AnthropicMessagesCompat", "BedrockCompat", "ChatTemplateKwargValue",
    "ChatTemplateVariable", "ConstrainedSamplingConfig", "GrammarConstrainedSamplingConfig", "GrammarFormat",
    "GrammarVariants", "JsonSchemaConstrainedSamplingConfig", "MistralConversationsCompat", "OpenAICompletionsCompat",
    "OpenAIResponsesCompat", "OpenRouterMaxPrice", "OpenRouterPercentileCutoffs", "OpenRouterRouting", "OpenRouterSort",
    "SessionAffinityFormat", "TextSignatureV1", "ThinkingTokenBudgetField", "VercelGatewayRouting",
    "KnownApi", "Api", "KnownImagesApi", "ImagesApi", "KnownProvider", "ProviderId", "KnownImagesProvider",
    "ImagesProviderId", "ThinkingLevelMap", "ProviderRequestOptions", "DeferredOptions", "DeferredFetchOptions",
    "DeferredCancelOptions", "ProviderStreamOptions", "ImagesOptions", "ProviderImagesOptions", "ImagesInput",
    "ImagesOutput", "ImagesContext", "AssistantImages", "StreamFunction", "ImagesFunction", "ProviderStreams",
    "ProviderImages", "images_model_from_json", "SimpleStreamFunction", "DeferredFetchFunction", "DeferredCancelFunction",
    "ImagesInputContent", "ImagesOutputContent", "ImagesStopReason",
]


def _dataclass_field_names(dataclass_type: type) -> set[str]:
    return {f.name for f in fields(dataclass_type)}
