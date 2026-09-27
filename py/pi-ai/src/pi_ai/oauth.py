"""Type-only compatibility entry point for extension OAuth declarations."""

from ._extension_oauth_types import (
    OAuthAuthInfo, OAuthCredentials, OAuthDeviceCodeInfo, OAuthLoginCallbacks,
    OAuthPrompt, OAuthSelectOption, OAuthSelectPrompt,
)

__all__ = [
    "OAuthAuthInfo", "OAuthCredentials", "OAuthDeviceCodeInfo", "OAuthLoginCallbacks",
    "OAuthPrompt", "OAuthSelectOption", "OAuthSelectPrompt",
]
