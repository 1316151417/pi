"""Profile file resolution from Anthropic SDK 0.124.0 core/credentials.ts.

Native Python is treated as the SDK's local-file-capable runtime. File I/O runs
on worker threads; no configuration is read until one of these functions runs.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .._javascript import javascript_json_parse, javascript_string
from .._json_runtime import JS_WHITESPACE
from .._values import UNDEFINED
from ._anthropic_credential_types import truthy

CONFIG_FILE_VERSION = "1.0"
CREDENTIALS_FILE_VERSION = "1.0"
type AnthropicConfig = dict[str, object]


@dataclass
class LoadedConfig:
    config: AnthropicConfig
    from_file: bool


def read_env(name: str) -> str | None:
    return os.environ.get(name, "").strip(JS_WHITESPACE) or None


def _validate_profile_name(name: str) -> None:
    if not name:
        raise RuntimeError("profile name is empty")
    if name in (".", ".."):
        raise RuntimeError(f'profile name "{name}" is not allowed')
    if "/" in name or "\\" in name:
        raise RuntimeError(f'profile name "{name}" must not contain path separators')
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise RuntimeError(f'profile name "{name}" contains disallowed characters (allowed: letters, digits, \'_\', \'.\', \'-\')')


async def read_utf8_file(path: str) -> str:
    return (await asyncio.to_thread(Path(path).read_bytes)).decode("utf-8", "replace")


def _root_config_path() -> str | None:
    configured = read_env("ANTHROPIC_CONFIG_DIR")
    if configured:
        return configured
    if sys.platform == "win32":
        app_data = read_env("APPDATA")
        if app_data:
            return os.path.join(app_data, "Anthropic")
        profile = read_env("USERPROFILE")
        return os.path.join(profile, "AppData", "Roaming", "Anthropic") if profile else None
    xdg_home = read_env("XDG_CONFIG_HOME")
    if xdg_home:
        return os.path.join(xdg_home, "anthropic")
    home = read_env("HOME")
    return os.path.join(home, ".config", "anthropic") if home else None


async def _active_profile_name() -> str | None:
    root = _root_config_path()
    if not root:
        return None
    profile = read_env("ANTHROPIC_PROFILE")
    if profile:
        return profile
    path = os.path.join(root, "active_config")
    try:
        return (await read_utf8_file(path)).strip(JS_WHITESPACE) or "default"
    except FileNotFoundError:
        return "default"
    except OSError as error:
        raise RuntimeError(f"failed to read {path}: {error}") from error


async def load_config_with_source(profile: str | None = None) -> LoadedConfig | None:
    root = _root_config_path()
    if root is None:
        return None
    profile_name = profile if profile is not None else await _active_profile_name()
    if profile_name is None:
        return None
    _validate_profile_name(profile_name)
    path = os.path.join(root, "configs", profile_name + ".json")
    try:
        raw = await read_utf8_file(path)
    except FileNotFoundError:
        organization = read_env("ANTHROPIC_ORGANIZATION_ID")
        token_file = read_env("ANTHROPIC_IDENTITY_TOKEN_FILE")
        rule = read_env("ANTHROPIC_FEDERATION_RULE_ID")
        if not rule or not organization:
            return None
        return LoadedConfig(from_file=False, config={
            "organization_id": organization,
            "workspace_id": read_env("ANTHROPIC_WORKSPACE_ID") or UNDEFINED,
            "base_url": read_env("ANTHROPIC_BASE_URL") or UNDEFINED,
            "authentication": {
                "type": "oidc_federation", "federation_rule_id": rule,
                "service_account_id": read_env("ANTHROPIC_SERVICE_ACCOUNT_ID") or UNDEFINED,
                "identity_token": {"source": "file", "path": token_file} if token_file else UNDEFINED,
                "scope": read_env("ANTHROPIC_SCOPE") or UNDEFINED,
            },
        })
    except OSError as error:
        raise RuntimeError(f"failed to read config file {path}: {error}") from error
    try:
        config = cast(AnthropicConfig, javascript_json_parse(raw))
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"failed to parse config file {path}: {error}") from error
    authentication = config.get("authentication", UNDEFINED)
    if not truthy(authentication):
        raise RuntimeError(f'config file {path} is missing "authentication"')
    auth = cast(dict[str, object], authentication)
    auth_type = auth.get("type", UNDEFINED)
    if auth_type not in ("oidc_federation", "user_oauth"):
        raise RuntimeError(f'authentication.type "{javascript_string(auth_type)}" is not a known authentication type')
    for key, env in (("organization_id", "ANTHROPIC_ORGANIZATION_ID"), ("workspace_id", "ANTHROPIC_WORKSPACE_ID"), ("base_url", "ANTHROPIC_BASE_URL")):
        if config.get(key) is None:
            config[key] = read_env(env) or UNDEFINED
    if auth.get("scope") is None:
        auth["scope"] = read_env("ANTHROPIC_SCOPE") or UNDEFINED
    if auth_type == "oidc_federation":
        if not truthy(auth.get("identity_token")):
            token_file = read_env("ANTHROPIC_IDENTITY_TOKEN_FILE")
            if token_file:
                auth["identity_token"] = {"source": "file", "path": token_file}
        if not truthy(auth.get("federation_rule_id")):
            auth["federation_rule_id"] = read_env("ANTHROPIC_FEDERATION_RULE_ID") or ""
        if auth.get("service_account_id") is None:
            auth["service_account_id"] = read_env("ANTHROPIC_SERVICE_ACCOUNT_ID") or UNDEFINED
    return LoadedConfig(config=config, from_file=True)


async def load_config(profile: str | None = None) -> AnthropicConfig | None:
    loaded = await load_config_with_source(profile)
    return loaded.config if loaded is not None else None


async def get_credentials_path(config: AnthropicConfig | None, profile: str | None = None) -> str | None:
    auth = cast(dict[str, object], config["authentication"]) if config is not None else {}
    path = auth.get("credentials_path")
    if truthy(path):
        return cast(str, path)
    root = _root_config_path()
    if not root:
        return None
    profile_name = profile if profile is not None else await _active_profile_name()
    if not profile_name:
        return None
    _validate_profile_name(profile_name)
    return os.path.join(root, "credentials", profile_name + ".json")


async def load_credentials() -> dict[str, object] | None:
    path = await get_credentials_path(await load_config())
    if not path:
        return None
    try:
        raw = await read_utf8_file(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(f"failed to read credentials file {path}: {error}") from error
    try:
        credentials = cast(dict[str, object], javascript_json_parse(raw))
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"failed to parse credentials file {path}: {error}") from error
    kind = credentials.get("type")
    if truthy(kind) and kind != "oauth_token":
        raise RuntimeError(f'credentials file {path} has unsupported type "{javascript_string(kind)}" (want "oauth_token")')
    return credentials
