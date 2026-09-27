"""Python implementation of pi-ai core types, streaming, and shared utilities."""

from .abort import AbortController, AbortError, AbortSignal, operation_signal, race_with_abort_signal
from ._extension_oauth_types import (
    OAuthAuthInfo, OAuthDeviceCodeInfo, OAuthLoginCallbacks, OAuthPrompt,
    OAuthSelectOption, OAuthSelectPrompt,
)
from .auth import (
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthContext,
    AuthDeviceCodeEvent, AuthEvent, AuthInfoEvent, AuthInfoLink, AuthInteraction,
    AuthOperationOptions, AuthProgressEvent, AuthPrompt, AuthResolutionOverrides,
    AuthResult, AuthSelectOption, AuthType, AuthUrlEvent, Credential, CredentialInfo,
    CredentialStore, InMemoryCredentialStore, LazyOAuthOptions, ManualCodeAuthPrompt,
    ModelAuth, ModelsError, ModelsErrorCode, OAuthAuth, OAuthCredential,
    OAuthCredentials, ProviderAuth, ProviderAuthInteraction, SecretAuthPrompt,
    SelectAuthPrompt, TextAuthPrompt, default_provider_auth_context, env_api_key_auth,
    lazy_oauth, resolve_provider_auth,
)
from .event_stream import (
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
)
from .api.lazy import LazyApiCapabilities, lazy_api, lazy_stream
from .models import (
    CreateModelsOptions, CreateProviderOptions, Models, ModelsApiStreamOptions,
    ModelsDeferredCancelOptions, ModelsDeferredFetchOptions, ModelsPublication,
    ModelsRefreshOptions, ModelsRefreshResult, ModelsRequestTransforms,
    ModelsSimpleStreamOptions, MutableModels, Provider, RefreshModelsContext,
    calculate_cost, clamp_thinking_level, create_models, create_provider,
    get_supported_thinking_levels, has_api, models_are_equal,
)
from .api.openai_completions import OpenAICompletionsOptions
from .api.openai_responses import OpenAIResponsesOptions
from .api.azure_openai_responses import AzureOpenAIResponsesOptions
from .api.anthropic_messages import (
    AnthropicEffort, AnthropicNamedToolChoice, AnthropicOptions,
    AnthropicThinkingDisplay, AnthropicToolChoice,
)
from .api.google_shared import GoogleApiThinkingLevel, ResolvedGoogleThinkingLevel
from .api.google_generative_ai import GoogleOptions, GoogleThinkingOptions
from .api.pi_messages import PiMessagesEvent, PiMessagesOptions, PiMessagesResponseError, PiMessagesRewriteImpact
from .env_api_keys import (
    ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV,
    find_env_keys, get_env_api_key,
)
from .assistant_message_frame import (
    AssistantMessageFrame, AssistantMessageFrameEncoder, frame_from_json, reduce_assistant_message_frames,
)
from .json_parse import parse_json_with_repair, parse_streaming_json, partial_parse, repair_json
from .overflow import get_overflow_patterns, is_context_overflow, is_recoverable_length
from .images_models import (
    CreateImagesProviderOptions, ImagesModels, ImagesProvider, MutableImagesModels,
    create_images_models, create_images_provider,
)
from .providers.faux import (
    FauxContentBlock, FauxModelDefinition, FauxProviderHandle, FauxProviderRegistration,
    FauxProviderState, FauxResponseFactory, FauxResponseStep, RegisterFauxProviderOptions,
    create_faux_core, faux_assistant_message, faux_provider, faux_text, faux_thinking, faux_tool_call,
)
from .model_catalog import ModelCatalog, ModelGroups, flatten_model_catalog
from .models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions
from .session_resources import (
    SessionResourceCleanup,
    cleanup_session_resources,
    register_session_resource_cleanup,
)
from .text import content_text, get_system_message_text, render_system_message_update
from .transcript import (
    TranscriptMessages, TranscriptTools, ToolStateChanges, collapse_system_messages, declarations_equal,
    create_initial_system_message,
    get_declared_tools, get_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_tool_state_changes,
    has_non_additive_tool_changes, has_tool_redefinitions,
    normalize_context,
    resolve_transcript, resolve_transcript_tools,
    to_tool_declaration,
    without_initial_system_message,
)
from .types import (
    JSON_NULL, UNDEFINED, JsonNull, Undefined,
    AnthropicAllowedFallbackModel, AnthropicMessagesCompat, BedrockCompat, ChatTemplateKwargValue,
    ChatTemplateVariable, ConstrainedSamplingConfig, GrammarConstrainedSamplingConfig, GrammarFormat,
    GrammarVariants, JsonSchemaConstrainedSamplingConfig, MistralConversationsCompat, OpenAICompletionsCompat,
    OpenAIResponsesCompat, OpenRouterMaxPrice, OpenRouterPercentileCutoffs, OpenRouterRouting, OpenRouterSort,
    SessionAffinityFormat, TextSignatureV1, ThinkingTokenBudgetField, VercelGatewayRouting,
    Api, AssistantImages, CacheRetention, DeferredCancelFunction, DeferredCancelOptions, DeferredFetchFunction,
    DeferredFetchOptions, DeferredOptions, ImagesApi, ImagesContext, ImagesFunction, ImagesInputContent, ImagesModel,
    ImagesOptions, ImagesOutputContent, ImagesProviderId, ImagesStopReason, KnownApi, KnownImagesApi,
    KnownImagesProvider, KnownProvider, ModelCost, ModelCostRates, ModelCostTier, ModelThinkingLevel, ProviderId,
    ProviderImages, ProviderImagesOptions, ProviderRequestOptions, ProviderResponse, ProviderStreamOptions,
    ProviderStreams, SimpleStreamFunction, StreamFunction, ThinkingLevel, ThinkingLevelMap, ToolChoice, Transport,
    images_model_from_json, model_from_json,
    AssistantMessage,
    AssistantMessageDiagnostic,
    AssistantMessageEvent,
    Context,
    Cost,
    DeferredHandle,
    DiagnosticErrorInfo,
    FetchFunction,
    ImageContent,
    JsonObject,
    JsonValue,
    Message,
    Model,
    ProviderEnv,
    ProviderHeaders,
    SimpleStreamOptions,
    StreamOptions,
    SystemMessage,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UserMessage,
    AgentMessage, AssistantContentBlock, JsonPrimitive, StopReason, ToolResultContentBlock, UserContentBlock,
    content_block_from_json, content_block_to_json,
    is_assistant_message, is_system_message, is_tool_result_message, is_user_message,
    message_from_json,
    message_to_json,
)
from .uuid_utils import uuidv7
from .validation import validate_tool_arguments, validate_tool_call
from .utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    extract_diagnostic_error,
    format_thrown_value,
)
from .utils.typebox_helpers import StringEnumOptions, StringEnumSchema, string_enum
from .utils.retry import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS, RetryCallbacks, RetryPolicy,
    is_retryable_assistant_error, retry_assistant_call, retry_delay_ms,
)
