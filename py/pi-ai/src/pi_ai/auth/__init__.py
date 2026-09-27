"""Provider auth contracts, credential storage, resolution, and flow helpers."""

from .context import default_provider_auth_context
from .credential_store import InMemoryCredentialStore
from .helpers import LazyOAuthOptions, env_api_key_auth, lazy_oauth
from .resolve import AuthResolutionOverrides, ModelsError, ModelsErrorCode, resolve_provider_auth
from .types import (
    ApiKeyAuth,
    ApiKeyAuthInput,
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthDeviceCodeEvent,
    AuthEvent,
    AuthInfoEvent,
    AuthInfoLink,
    AuthInteraction,
    AuthOperationOptions,
    AuthProgressEvent,
    AuthPrompt,
    AuthResult,
    AuthSelectOption,
    AuthType,
    AuthUrlEvent,
    Credential,
    CredentialInfo,
    CredentialStore,
    ManualCodeAuthPrompt,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    OAuthCredentials,
    ProviderAuth,
    ProviderAuthInteraction,
    SecretAuthPrompt,
    SelectAuthPrompt,
    TextAuthPrompt,
)

__all__ = [
    "default_provider_auth_context", "InMemoryCredentialStore", "LazyOAuthOptions", "env_api_key_auth", "lazy_oauth",
    "AuthResolutionOverrides", "ModelsError", "ModelsErrorCode", "resolve_provider_auth", "ApiKeyAuth",
    "ApiKeyAuthInput", "ApiKeyCredential", "AuthCheck", "AuthContext", "AuthDeviceCodeEvent", "AuthEvent",
    "AuthInfoEvent", "AuthInfoLink", "AuthInteraction", "AuthOperationOptions", "AuthProgressEvent", "AuthPrompt",
    "AuthResult", "AuthSelectOption", "AuthType", "AuthUrlEvent", "Credential", "CredentialInfo", "CredentialStore",
    "ManualCodeAuthPrompt", "ModelAuth", "OAuthAuth", "OAuthCredential", "OAuthCredentials", "ProviderAuth",
    "ProviderAuthInteraction", "SecretAuthPrompt", "SelectAuthPrompt", "TextAuthPrompt",
]
