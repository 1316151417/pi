"""GitHub Copilot OAuth and model policies from ``oauth/github-copilot.ts``."""

from __future__ import annotations

import base64
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import cast

from ...abort import AbortSignal, any_signal
from ...providers.github_copilot_models import GITHUB_COPILOT_MODELS
from ...utils._javascript import javascript_string
from ...utils.sleep import sleep
from ..context import _JS_WHITESPACE
from ..types import (
    AuthDeviceCodeEvent, AuthProgressEvent, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction,
    TextAuthPrompt,
)
from ._http import OAuthHttpResponse, fetch, form_url_encode, parse_url, timeout_signal
from .device_code import (
    OAuthDeviceCodeComplete, OAuthDeviceCodeFailed, OAuthDeviceCodePending, OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult, OAuthDeviceCodeSlowDown, poll_oauth_device_code_flow,
)

_CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode("latin-1")
_COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
_COPILOT_API_VERSION = "2026-06-01"
_FLOAT_PREFIX = re.compile(r"[+-]?(?:Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)")


@dataclass
class _DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int | float
    interval: int | float | None = None


@dataclass
class _ModelCatalog:
    available_model_ids: list[str]
    policy_model_ids: list[str]


@dataclass
class _AccountModel:
    id: str
    picker_enabled: bool
    policy_state: object


@dataclass
class _RetryPolicy:
    max_retries: int
    max_elapsed_ms: int


def _normalize_domain(value: str) -> str | None:
    trimmed = value.strip(_JS_WHITESPACE)
    if not trimmed:
        return None
    try:
        return parse_url(trimmed if "://" in trimmed else f"https://{trimmed}").hostname
    except ValueError:
        return None


def _get_base_url(token: str | None = None, enterprise_domain: str | None = None) -> str:
    if token:
        match = re.search(r"proxy-ep=([^;]+)", token)
        if match is not None:
            host = re.sub(r"^proxy\.", "api.", match[1])
            return f"https://{host}"
    return f"https://copilot-api.{enterprise_domain}" if enterprise_domain else "https://api.individual.githubcopilot.com"


def _as_record(value: object) -> dict[str, object] | None:
    # JSON arrays also satisfy typeof === 'object', but have no named fields
    # used by these responses.
    return cast(dict[str, object], value) if isinstance(value, dict) else {} if isinstance(value, list) else None


def _parse_model_catalog(raw: object, allow_policy_fallback: bool) -> _ModelCatalog:
    root = _as_record(raw)
    data = root.get("data") if root is not None else None
    if not isinstance(data, list):
        raise RuntimeError("Invalid Copilot models response")
    models: list[_AccountModel] = []
    for raw_item in data:
        item = _as_record(raw_item)
        id = item.get("id") if item is not None else None
        if item is None or not isinstance(id, str):
            continue
        capabilities = _as_record(item.get("capabilities"))
        supports = _as_record(capabilities.get("supports")) if capabilities is not None else None
        if supports is not None and supports.get("tool_calls") is False:
            continue
        policy = _as_record(item.get("policy"))
        models.append(_AccountModel(id, item.get("model_picker_enabled") is True, policy.get("state") if policy is not None else None))
    picker_ids = [model.id for model in models if model.picker_enabled and model.policy_state != "disabled"]
    use_fallback = allow_policy_fallback and not picker_ids
    available_ids = picker_ids if picker_ids or not allow_policy_fallback else [model.id for model in models if model.policy_state == "enabled"]
    policy_ids = [
        model.id for model in models
        if model.policy_state == "unconfigured" and model.id in GITHUB_COPILOT_MODELS and (model.picker_enabled or use_fallback)
    ]
    return _ModelCatalog(available_ids, policy_ids)


async def _fetch_with_rate_limit_retry(
    url: str, signal: AbortSignal, retry_policy: _RetryPolicy, *, method: str = "GET",
    headers: Mapping[str, str] | None = None, body: str | None = None,
) -> OAuthHttpResponse:
    retry_budget = timeout_signal(retry_policy.max_elapsed_ms) if retry_policy.max_retries > 0 and retry_policy.max_elapsed_ms > 0 else None
    request_signal = any_signal(signal, retry_budget) if retry_budget is not None else signal
    assert request_signal is not None
    deadline = math.floor(time.time() * 1000) + retry_policy.max_elapsed_ms if retry_budget is not None else None
    retry = 0
    while True:
        response = await fetch(
            url, method=method, headers=headers, body=body, signal=any_signal(request_signal, timeout_signal(5000)),
        )
        if response.status != 429 or retry == retry_policy.max_retries:
            return response
        retry_after = response.headers.get("retry-after")
        delay_ms: int | float = 500 * 2**retry
        if retry_after:
            match = _FLOAT_PREFIX.match(retry_after.lstrip(_JS_WHITESPACE))
            if match is not None:
                delay_ms = float(match[0]) * 1000
            else:
                try:
                    date = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError, OverflowError):
                    try:
                        date = datetime.fromisoformat(retry_after.replace("Z", "+00:00"))
                    except ValueError:
                        return response
                try:
                    delay_ms = date.timestamp() * 1000 - math.floor(time.time() * 1000)
                except (OSError, OverflowError, ValueError):
                    return response
            if not math.isfinite(delay_ms):
                return response
        delay_ms = max(0, delay_ms)
        if deadline is not None and delay_ms >= deadline - math.floor(time.time() * 1000):
            return response
        await response.cancel_body()
        await sleep(delay_ms, request_signal)
        retry += 1


async def _fetch_models(token: str, enterprise_domain: str | None, signal: AbortSignal, policy: _RetryPolicy) -> _ModelCatalog:
    base_url = _get_base_url(token, enterprise_domain)
    allow_policy_fallback = base_url == "https://api.individual.githubcopilot.com"
    response = await _fetch_with_rate_limit_retry(f"{base_url}/models", signal, policy, headers={
        "Accept": "application/json", "Authorization": f"Bearer {token}", **_COPILOT_HEADERS,
        "X-GitHub-Api-Version": _COPILOT_API_VERSION,
    })
    if not response.ok:
        raise RuntimeError(f"{response.status} {response.status_text}: {await response.text()}")
    return _parse_model_catalog(await response.json(), allow_policy_fallback)


async def _fetch_json(
    url: str, *, signal: AbortSignal, method: str = "GET", headers: Mapping[str, str] | None = None,
    body: str | None = None,
) -> object:
    response = await fetch(url, method=method, headers=headers, body=body, signal=signal)
    if not response.ok:
        raise RuntimeError(f"{response.status} {response.status_text}: {await response.text()}")
    return await response.json()


async def _start_device_flow(domain: str, signal: AbortSignal) -> _DeviceCode:
    data = await _fetch_json(
        f"https://{domain}/login/device/code", method="POST", signal=signal,
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded", "User-Agent": "GitHubCopilotChat/0.35.0"},
        body=form_url_encode({"client_id": _CLIENT_ID, "scope": "read:user"}),
    )
    record = _as_record(data)
    if record is None:
        raise RuntimeError("Invalid device code response")
    device_code = record.get("device_code")
    user_code = record.get("user_code")
    verification_uri = record.get("verification_uri")
    interval = record.get("interval")
    expires = record.get("expires_in")
    if (
        not isinstance(device_code, str) or not isinstance(user_code, str) or not isinstance(verification_uri, str)
        or "interval" in record and type(interval) not in (int, float) or type(expires) not in (int, float)
    ):
        raise RuntimeError("Invalid device code response fields")
    try:
        uri = parse_url(verification_uri)
    except ValueError:
        raise RuntimeError("Untrusted verification_uri in device code response") from None
    if uri.protocol not in ("https:", "http:"):
        raise RuntimeError("Untrusted verification_uri in device code response")
    return _DeviceCode(device_code, user_code, uri.href, cast(int | float, expires), cast(int | float | None, interval))


async def _poll_for_github_access_token(domain: str, device: _DeviceCode, signal: AbortSignal) -> str:
    async def poll() -> OAuthDeviceCodePollResult[str]:
        raw = await _fetch_json(
            f"https://{domain}/login/oauth/access_token", method="POST", signal=signal,
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded", "User-Agent": "GitHubCopilotChat/0.35.0"},
            body=form_url_encode({
                "client_id": _CLIENT_ID, "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }),
        )
        record = _as_record(raw)
        access = record.get("access_token") if record is not None else None
        if isinstance(access, str):
            return OAuthDeviceCodeComplete(access)
        error = record.get("error") if record is not None else None
        if isinstance(error, str):
            assert record is not None
            if error == "authorization_pending":
                return OAuthDeviceCodePending()
            if error == "slow_down":
                interval = record.get("interval")
                return OAuthDeviceCodeSlowDown(cast(int | float, interval) if type(interval) in (int, float) else None)
            description = record.get("error_description")
            has_description = bool(description) if not isinstance(description, (list, dict)) else True
            if isinstance(description, float) and math.isnan(description):
                has_description = False
            suffix = f": {javascript_string(description)}" if has_description else ""
            return OAuthDeviceCodeFailed(f"Device flow failed: {error}{suffix}")
        return OAuthDeviceCodeFailed("Invalid device token response")

    return await poll_oauth_device_code_flow(OAuthDeviceCodePollOptions(
        poll=poll, signal=signal, interval_seconds=device.interval, expires_in_seconds=device.expires_in,
        wait_before_first_poll=True,
    ))


async def _refresh_access_token(refresh_token: str, enterprise_domain: str | None, signal: AbortSignal) -> OAuthCredential:
    domain = enterprise_domain or "github.com"
    raw = await _fetch_json(f"https://api.{domain}/copilot_internal/v2/token", signal=signal, headers={
        "Accept": "application/json", "Authorization": f"Bearer {refresh_token}", **_COPILOT_HEADERS,
    })
    record = _as_record(raw)
    if record is None:
        raise RuntimeError("Invalid Copilot token response")
    token = record.get("token")
    expires_at = record.get("expires_at")
    if not isinstance(token, str) or type(expires_at) not in (int, float):
        raise RuntimeError("Invalid Copilot token response fields")
    return OAuthCredential(
        refresh=refresh_token, access=token, expires=cast(int | float, expires_at) * 1000 - 5 * 60 * 1000,
        extra={"enterpriseUrl": enterprise_domain} if enterprise_domain is not None else {},
    )


async def _enable_model(token: str, model_id: str, enterprise_domain: str | None, signal: AbortSignal) -> bool:
    base_url = _get_base_url(token, enterprise_domain)
    try:
        response = await _fetch_with_rate_limit_retry(
            f"{base_url}/models/{model_id}/policy", signal, _RetryPolicy(2, 5000), method="POST",
            headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {token}", **_COPILOT_HEADERS,
                "openai-intent": "chat-policy", "x-interaction-type": "chat-policy",
            }, body='{"state":"enabled"}',
        )
    except Exception:
        if signal.aborted:
            raise
        return False
    if response.status == 429:
        raise RuntimeError(f"{response.status} {response.status_text}: {await response.text()}")
    return response.ok


async def _enable_models(token: str, ids: Sequence[str], enterprise_domain: str | None, signal: AbortSignal) -> list[str]:
    enabled: list[str] = []
    for model_id in ids:
        try:
            if await _enable_model(token, model_id, enterprise_domain, signal):
                enabled.append(model_id)
        except Exception:
            if signal.aborted:
                raise
            break
    return enabled


async def _login(interaction: ProviderAuthInteraction) -> OAuthCredential:
    value = await interaction.prompt(TextAuthPrompt(
        message="GitHub Enterprise URL/domain (blank for github.com)", placeholder="company.ghe.com",
    ))
    if interaction.signal.aborted:
        raise RuntimeError("Login cancelled")
    trimmed = value.strip(_JS_WHITESPACE)
    enterprise_domain = _normalize_domain(value)
    if trimmed and not enterprise_domain:
        raise RuntimeError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"
    device = await _start_device_flow(domain, interaction.signal)
    interaction.notify(AuthDeviceCodeEvent(
        user_code=device.user_code, verification_uri=device.verification_uri,
        interval_seconds=device.interval, expires_in_seconds=device.expires_in,
    ))
    access = await _poll_for_github_access_token(domain, device, interaction.signal)
    credentials = await _refresh_access_token(access, enterprise_domain, interaction.signal)
    models = await _fetch_models(credentials.access, enterprise_domain, interaction.signal, _RetryPolicy(2, 5000))
    enabled: list[str] = []
    if models.policy_model_ids:
        interaction.notify(AuthProgressEvent(message="Enabling models..."))
        enabled = await _enable_models(credentials.access, models.policy_model_ids, enterprise_domain, interaction.signal)
    return OAuthCredential(
        access=credentials.access, refresh=credentials.refresh, expires=credentials.expires,
        extra={**credentials.extra, "availableModelIds": list(dict.fromkeys([*models.available_model_ids, *enabled]))},
    )


def _enterprise_domain(credential: OAuthCredential) -> str | None:
    enterprise_url = credential.extra.get("enterpriseUrl")
    return _normalize_domain(enterprise_url) if isinstance(enterprise_url, str) and enterprise_url else None


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    domain = _enterprise_domain(credential)
    credentials = await _refresh_access_token(credential.refresh, domain, signal)
    models = await _fetch_models(credentials.access, domain, signal, _RetryPolicy(0, 0))
    return OAuthCredential(
        access=credentials.access, refresh=credentials.refresh, expires=credentials.expires,
        extra={**credentials.extra, "availableModelIds": models.available_model_ids},
    )


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access, base_url=_get_base_url(credential.access, _enterprise_domain(credential)))


github_copilot_oauth = OAuthAuth(
    name="GitHub Copilot", is_subscription=True, login=_login, refresh=_refresh, to_auth=_to_auth,
)

__all__ = ["github_copilot_oauth"]
