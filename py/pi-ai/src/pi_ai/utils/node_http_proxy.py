"""Provider-scoped HTTP proxy resolution from utils/node-http-proxy.ts."""

from __future__ import annotations

import re

from ada_url import URL

from ..types import ProviderEnv
from ._javascript import javascript_json_stringify
from .provider_env import get_provider_env_value

DEFAULT_PROXY_PORTS = {"ftp": 21, "gopher": 70, "http": 80, "https": 443, "ws": 80, "wss": 443}
UNSUPPORTED_PROXY_PROTOCOL_MESSAGE = (
    "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL."
)
_WHITESPACE = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def _proxy_env(key: str, env: ProviderEnv | None) -> str:
    lower = key.lower()
    upper = key.upper()
    return (
        (env.get(lower) if env is not None else None)
        or (env.get(upper) if env is not None else None)
        or get_provider_env_value(lower) or get_provider_env_value(upper) or ""
    )


def _strip_brackets(host: str) -> str:
    return host[1:-1] if host.startswith("[") and host.endswith("]") else host


def _parse_port(value: str) -> int | None:
    match = re.match(r"[+-]?[0-9]+", value.lstrip(_WHITESPACE))
    if match is None:
        return None
    return int(match[0])


def _no_proxy_entry(entry: str) -> tuple[str, int] | None:
    trimmed = entry.strip(_WHITESPACE).lower()
    if not trimmed:
        return None
    if trimmed.startswith("["):
        closing = trimmed.find("]")
        if closing != -1:
            host, rest = trimmed[1:closing], trimmed[closing + 1:]
            return host, (_parse_port(rest[1:]) or 0) if rest.startswith(":") else 0
    if trimmed.count(":") > 1:
        return trimmed, 0
    colon = trimmed.rfind(":")
    if colon != -1:
        port = _parse_port(trimmed[colon + 1:])
        if port is not None:
            return trimmed[:colon], port
    return trimmed, 0


def _should_proxy(hostname: str, port: int, env: ProviderEnv | None) -> bool:
    no_proxy = _proxy_env("no_proxy", env).lower()
    if not no_proxy:
        return True
    if no_proxy == "*":
        return False
    target = _strip_brackets(hostname.lower())
    for entry in re.split("[," + _WHITESPACE + "]", no_proxy):
        parsed = _no_proxy_entry(entry)
        if parsed is None:
            continue
        host, entry_port = parsed
        if entry_port and entry_port != port:
            continue
        domain = _strip_brackets(host)
        if domain.startswith("*."):
            domain = domain[2:]
        elif domain.startswith((".", "*")):
            domain = domain[1:]
        if domain and (target == domain or target.endswith("." + domain)):
            return False
    return True


def resolve_http_proxy_url_for_target(target_url: str | URL, env: ProviderEnv | None = None) -> URL | None:
    try:
        target = target_url if isinstance(target_url, URL) else URL(
            target_url.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace"),
        )
    except ValueError:
        return None
    if not target.protocol or not target.host:
        return None
    protocol = target.protocol.split(":", 1)[0]
    hostname = _strip_brackets(target.hostname or re.sub(r":[0-9]*$", "", target.host))
    port = _parse_port(target.port) or DEFAULT_PROXY_PORTS.get(protocol, 0)
    if not _should_proxy(hostname, port, env):
        return None
    proxy = _proxy_env(protocol + "_proxy", env) or _proxy_env("all_proxy", env)
    if not proxy:
        return None
    if "://" not in proxy:
        proxy = protocol + "://" + proxy
    try:
        proxy_url = URL(proxy.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace"))
    except ValueError as error:
        raise RuntimeError(f"Invalid proxy URL {javascript_json_stringify(proxy)}: Invalid URL") from error
    if proxy_url.protocol not in ("http:", "https:"):
        raise RuntimeError(f"{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {proxy_url.protocol}")
    return proxy_url


__all__ = ["UNSUPPORTED_PROXY_PROTOCOL_MESSAGE", "resolve_http_proxy_url_for_target"]
