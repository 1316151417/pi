"""Provider authentication contracts from ``auth/types.ts``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

from ..abort import AbortSignal

if TYPE_CHECKING:
    from ..types import ProviderEnv, ProviderHeaders

type AuthType = Literal["api_key", "oauth"]


@dataclass
class ModelAuth:
    api_key: str | None = None
    headers: ProviderHeaders | None = None
    base_url: str | None = None

    def to_json(self) -> dict[str, object]:
        data: dict[str, object] = {}
        if self.api_key is not None:
            data["apiKey"] = self.api_key
        if self.headers is not None:
            data["headers"] = self.headers
        if self.base_url is not None:
            data["baseUrl"] = self.base_url
        return data


@dataclass
class ApiKeyCredential:
    type: Literal["api_key"] = field(default="api_key", init=False)
    key: str | None = None
    env: ProviderEnv | None = None

    def to_json(self) -> dict[str, object]:
        data: dict[str, object] = {"type": self.type}
        if self.key is not None:
            data["key"] = self.key
        if self.env is not None:
            data["env"] = self.env
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> ApiKeyCredential:
        return cls(key=cast(str | None, data.get("key")), env=cast("ProviderEnv | None", data.get("env")))


@dataclass
class OAuthCredentials:
    refresh: str
    access: str
    expires: int | float
    extra: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {"refresh": self.refresh, "access": self.access, "expires": self.expires, **self.extra}

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> OAuthCredentials:
        return cls(
            refresh=cast(str, data["refresh"]), access=cast(str, data["access"]),
            expires=cast(int | float, data["expires"]),
            extra={key: value for key, value in data.items() if key not in ("refresh", "access", "expires")},
        )


@dataclass
class OAuthCredential(OAuthCredentials):
    type: Literal["oauth"] = field(default="oauth", init=False)

    def to_json(self) -> dict[str, object]:
        return {"type": self.type, **super().to_json()}

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> OAuthCredential:
        return cls(
            refresh=cast(str, data["refresh"]), access=cast(str, data["access"]),
            expires=cast(int | float, data["expires"]),
            extra={key: value for key, value in data.items() if key not in ("type", "refresh", "access", "expires")},
        )


type Credential = ApiKeyCredential | OAuthCredential


@dataclass
class CredentialInfo:
    provider_id: str
    type: AuthType

    def to_json(self) -> dict[str, str]:
        return {"providerId": self.provider_id, "type": self.type}


@dataclass
class AuthOperationOptions:
    signal: AbortSignal | None = None


class CredentialStore(Protocol):
    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None: ...

    async def list(self, options: AuthOperationOptions | None = None) -> Sequence[CredentialInfo]: ...

    def modify(
        self, provider_id: str, fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Awaitable[Credential | None]: ...

    def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[None]: ...


class AuthContext(Protocol):
    async def env(self, name: str) -> str | None: ...

    async def file_exists(self, path: str) -> bool: ...


@dataclass
class AuthResult:
    auth: ModelAuth
    env: ProviderEnv | None = None
    source: str | None = None


@dataclass
class AuthCheck:
    type: AuthType
    source: str | None = None


@dataclass
class TextAuthPrompt:
    message: str
    placeholder: str | None = None
    signal: AbortSignal | None = None
    type: Literal["text"] = field(default="text", init=False)


@dataclass
class SecretAuthPrompt:
    message: str
    placeholder: str | None = None
    signal: AbortSignal | None = None
    type: Literal["secret"] = field(default="secret", init=False)


@dataclass
class AuthSelectOption:
    id: str
    label: str
    description: str | None = None


@dataclass
class SelectAuthPrompt:
    message: str
    options: Sequence[AuthSelectOption]
    signal: AbortSignal | None = None
    type: Literal["select"] = field(default="select", init=False)


@dataclass
class ManualCodeAuthPrompt:
    message: str
    placeholder: str | None = None
    signal: AbortSignal | None = None
    type: Literal["manual_code"] = field(default="manual_code", init=False)


type AuthPrompt = TextAuthPrompt | SecretAuthPrompt | SelectAuthPrompt | ManualCodeAuthPrompt


@dataclass
class AuthInfoLink:
    url: str
    label: str | None = None


@dataclass
class AuthInfoEvent:
    message: str
    links: Sequence[AuthInfoLink] | None = None
    type: Literal["info"] = field(default="info", init=False)


@dataclass
class AuthUrlEvent:
    url: str
    instructions: str | None = None
    type: Literal["auth_url"] = field(default="auth_url", init=False)


@dataclass
class AuthDeviceCodeEvent:
    user_code: str
    verification_uri: str
    interval_seconds: int | float | None = None
    expires_in_seconds: int | float | None = None
    type: Literal["device_code"] = field(default="device_code", init=False)


@dataclass
class AuthProgressEvent:
    message: str
    type: Literal["progress"] = field(default="progress", init=False)


type AuthEvent = AuthInfoEvent | AuthUrlEvent | AuthDeviceCodeEvent | AuthProgressEvent


@dataclass
class AuthInteraction:
    prompt: Callable[[AuthPrompt], Awaitable[str]]
    notify: Callable[[AuthEvent], None]
    signal: AbortSignal | None = None


@dataclass
class ProviderAuthInteraction:
    prompt: Callable[[AuthPrompt], Awaitable[str]]
    notify: Callable[[AuthEvent], None]
    signal: AbortSignal


@dataclass
class ApiKeyAuthInput:
    ctx: AuthContext
    signal: AbortSignal
    credential: ApiKeyCredential | None = None


@dataclass
class ApiKeyAuth:
    name: str
    resolve: Callable[[ApiKeyAuthInput], Awaitable[AuthResult | None]]
    login: Callable[[ProviderAuthInteraction], Awaitable[ApiKeyCredential]] | None = None
    check: Callable[[ApiKeyAuthInput], Awaitable[AuthCheck | None]] | None = None


@dataclass
class OAuthAuth:
    name: str
    login: Callable[[ProviderAuthInteraction], Awaitable[OAuthCredential]]
    refresh: Callable[[OAuthCredential, AbortSignal], Awaitable[OAuthCredential]]
    to_auth: Callable[[OAuthCredential], Awaitable[ModelAuth]]
    is_subscription: bool | None = None
    login_label: str | None = None


@dataclass
class ProviderAuth:
    api_key: ApiKeyAuth | None = None
    oauth: OAuthAuth | None = None


__all__ = [
    "ModelAuth", "ApiKeyCredential", "OAuthCredentials", "OAuthCredential", "Credential", "CredentialInfo",
    "AuthOperationOptions", "CredentialStore", "AuthContext", "AuthResult", "AuthCheck", "AuthType",
    "AuthPrompt", "TextAuthPrompt", "SecretAuthPrompt", "SelectAuthPrompt", "ManualCodeAuthPrompt",
    "AuthSelectOption", "AuthInfoLink", "AuthEvent", "AuthInfoEvent", "AuthUrlEvent", "AuthDeviceCodeEvent",
    "AuthProgressEvent", "AuthInteraction", "ProviderAuthInteraction", "ApiKeyAuthInput", "ApiKeyAuth",
    "OAuthAuth", "ProviderAuth",
]
