"""Resolve literal, environment and command values from ``core/resolve-config-value.ts``."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping

from .shell import get_shell_config

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_COMMAND_CACHE: dict[str, str | None] = {}
type TemplatePart = tuple[str, str]


def _parse_template(config: str) -> list[TemplatePart]:
    parts: list[TemplatePart] = []

    def append_literal(value: str) -> None:
        if not value:
            return
        if parts and parts[-1][0] == "literal":
            parts[-1] = ("literal", parts[-1][1] + value)
        else:
            parts.append(("literal", value))

    index = 0
    while index < len(config):
        dollar = config.find("$", index)
        if dollar < 0:
            append_literal(config[index:])
            break
        append_literal(config[index:dollar])
        next_char = config[dollar + 1:dollar + 2]
        if next_char in ("$", "!"):
            append_literal(next_char)
            index = dollar + 2
            continue
        if next_char == "{":
            end = config.find("}", dollar + 2)
            if end < 0:
                append_literal("$")
                index = dollar + 1
                continue
            name = config[dollar + 2:end]
            if _ENV_NAME.fullmatch(name):
                parts.append(("env", name))
            else:
                append_literal(config[dollar:end + 1])
            index = end + 1
            continue
        match = _ENV_PREFIX.match(config[dollar + 1:])
        if match is not None:
            parts.append(("env", match.group()))
            index = dollar + 1 + len(match.group())
            continue
        append_literal("$")
        index = dollar + 1
    return parts


def get_config_value_env_var_names(config: str) -> list[str]:
    if config.startswith("!"):
        return []
    names: list[str] = []
    for kind, value in _parse_template(config):
        if kind == "env" and value not in names:
            names.append(value)
    return names


def get_config_value_env_var_name(config: str) -> str | None:
    parts = [] if config.startswith("!") else _parse_template(config)
    return parts[0][1] if len(parts) == 1 and parts[0][0] == "env" else None


def _resolve_env(name: str, env: Mapping[str, str] | None) -> str | None:
    return (env or {}).get(name) or os.environ.get(name) or None


def get_missing_config_value_env_var_names(config: str, env: Mapping[str, str] | None = None) -> list[str]:
    return [name for name in get_config_value_env_var_names(config) if _resolve_env(name, env) is None]


def is_command_config_value(config: str) -> bool:
    return config.startswith("!")


def is_config_value_configured(config: str, env: Mapping[str, str] | None = None) -> bool:
    return not get_missing_config_value_env_var_names(config, env)


def _execute_command_uncached(command_config: str) -> str | None:
    command = command_config[1:]
    if os.name == "nt":
        try:
            shell = get_shell_config()
            use_stdin = shell.command_transport == "stdin"
            completed = subprocess.run(
                [shell.shell, *shell.args, *([] if use_stdin else [command])],
                input=command if use_stdin else None, capture_output=True, text=True,
                timeout=10, check=False,
            )
            return completed.stdout.strip() or None if completed.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        completed = subprocess.run(
            command, shell=True, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _resolve(config: str, env: Mapping[str, str] | None, *, cached: bool) -> str | None:
    if config.startswith("!"):
        if cached and config in _COMMAND_CACHE:
            return _COMMAND_CACHE[config]
        value = _execute_command_uncached(config)
        if cached:
            _COMMAND_CACHE[config] = value
        return value
    resolved = ""
    for kind, value in _parse_template(config):
        if kind == "literal":
            resolved += value
        else:
            env_value = _resolve_env(value, env)
            if env_value is None:
                return None
            resolved += env_value
    return resolved


def resolve_config_value(config: str, env: Mapping[str, str] | None = None) -> str | None:
    return _resolve(config, env, cached=True)


def resolve_config_value_uncached(config: str, env: Mapping[str, str] | None = None) -> str | None:
    return _resolve(config, env, cached=False)


def resolve_config_value_or_throw(config: str, description: str, env: Mapping[str, str] | None = None) -> str:
    resolved = resolve_config_value_uncached(config, env)
    if resolved is not None:
        return resolved
    if config.startswith("!"):
        raise RuntimeError(f"Failed to resolve {description} from shell command: {config[1:]}")
    missing = get_missing_config_value_env_var_names(config, env)
    if len(missing) == 1:
        raise RuntimeError(f"Failed to resolve {description} from environment variable: {missing[0]}")
    if missing:
        raise RuntimeError(f"Failed to resolve {description} from environment variables: {', '.join(missing)}")
    raise RuntimeError(f"Failed to resolve {description}")


def resolve_headers(headers: Mapping[str, str] | None, env: Mapping[str, str] | None = None) -> dict[str, str] | None:
    if headers is None:
        return None
    resolved = {name: value for name, raw in headers.items() if (value := resolve_config_value(raw, env))}
    return resolved or None


def resolve_headers_or_throw(
    headers: Mapping[str, str] | None, description: str, env: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    if headers is None:
        return None
    resolved = {
        name: resolve_config_value_or_throw(raw, f'{description} header "{name}"', env)
        for name, raw in headers.items()
    }
    return resolved or None


def clear_config_value_cache() -> None:
    _COMMAND_CACHE.clear()


__all__ = [
    "get_config_value_env_var_name", "get_config_value_env_var_names",
    "get_missing_config_value_env_var_names", "is_command_config_value",
    "is_config_value_configured", "resolve_config_value", "resolve_config_value_uncached",
    "resolve_config_value_or_throw", "resolve_headers", "resolve_headers_or_throw",
    "clear_config_value_cache",
]
