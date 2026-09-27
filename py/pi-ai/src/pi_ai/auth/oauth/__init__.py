"""OAuth flow primitives shared by provider authentication implementations."""

from .pkce import PKCE, generate_pkce
from .device_code import (
    OAuthDeviceCodeComplete, OAuthDeviceCodeFailed, OAuthDeviceCodePending, OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult, OAuthDeviceCodeSlowDown, abortable_sleep, poll_oauth_device_code_flow,
)
from .oauth_page import oauth_error_html, oauth_success_html

__all__ = [
    "PKCE", "generate_pkce", "OAuthDeviceCodeComplete", "OAuthDeviceCodeFailed", "OAuthDeviceCodePending",
    "OAuthDeviceCodePollOptions", "OAuthDeviceCodePollResult", "OAuthDeviceCodeSlowDown", "abortable_sleep",
    "poll_oauth_device_code_flow", "oauth_error_html", "oauth_success_html",
]
