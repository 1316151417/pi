"""Token, file and error primitives from Anthropic SDK 0.124.0 credentials/types.ts."""

from __future__ import annotations

import asyncio
import math
import os
import sys
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .._javascript import javascript_json_parse, javascript_json_stringify, javascript_object_keys, javascript_string, utf16_length, utf16_slice
from .._random import random_base36
from .._values import JSON_NULL, UNDEFINED
from ..auth.oauth._http import parse_url
from ._anthropic_errors import AnthropicError

OAUTH_API_BETA_HEADER = "oauth-2025-04-20"
FEDERATION_BETA_HEADER = "oidc-federation-2026-04-01"
TOKEN_ENDPOINT = "/v1/oauth/token"
ADVISORY_REFRESH_THRESHOLD_IN_SECONDS = 120
MANDATORY_REFRESH_THRESHOLD_IN_SECONDS = 30
ADVISORY_REFRESH_BACKOFF_IN_SECONDS = 5


@dataclass
class AccessToken:
    token: str
    expires_at: int | float | None


@dataclass
class TokenProviderOptions:
    force_refresh: bool = False


type AccessTokenProvider = Callable[[TokenProviderOptions | None], Awaitable[AccessToken]]
type IdentityTokenProvider = Callable[[], str | Awaitable[str]]


class TokenResponse(Protocol):
    status: int
    ok: bool
    headers: Mapping[str, str]
    body: AsyncIterable[bytes] | None

    async def text(self) -> str: ...
    async def cancel_body(self) -> None: ...


class WorkloadIdentityError(AnthropicError):
    def __init__(
        self, message: str, status_code: int | None = None,
        body: object = None, request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.request_id = request_id


def truthy(value: object) -> bool:
    if value is None or value is UNDEFINED or value is JSON_NULL or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return value != "" if isinstance(value, str) else True


def property_value(value: object, key: str) -> object:
    if value is None or value is UNDEFINED:
        label = "null" if value is None else "undefined"
        raise TypeError(f"Cannot read properties of {label} (reading '{key}')")
    return value.get(key, UNDEFINED) if isinstance(value, Mapping) else UNDEFINED


def require_secure_token_endpoint(base_url: str) -> None:
    if not base_url:
        return
    try:
        url = parse_url(base_url)
    except Exception as error:
        raise WorkloadIdentityError(f'Invalid token endpoint base URL "{base_url}": {error}') from error
    if url.protocol == "https:":
        return
    if url.protocol == "http:" and url.hostname.lower().removeprefix("[").removesuffix("]") in ("localhost", "127.0.0.1", "::1"):
        return
    raise WorkloadIdentityError(f'Refusing to send credential over non-https token endpoint "{base_url}"')


def redact_sensitive(body: object) -> object:
    if body is None or body is UNDEFINED:
        return body
    if isinstance(body, str):
        try:
            parsed = javascript_json_parse(body)
        except (ValueError, TypeError):
            length = utf16_length(body)
            return body if length <= 2000 else utf16_slice(body, 2000) + f"... <{length - 2000} more chars>"
        return javascript_json_stringify(redact_sensitive(parsed))
    if isinstance(body, Mapping):
        return {key: body[key] for key in javascript_object_keys(body) if key in ("error", "error_description", "error_uri")}
    return None


async def parse_token_response(response: TokenResponse, request_id: str | None) -> dict[str, object]:
    chunks: list[bytes] = []
    received = 0
    body = response.body
    if body is not None:
        async for chunk in body:
            if received + len(chunk) > 1 << 20:
                remaining = (1 << 20) - received
                if remaining > 0:
                    chunks.append(chunk[:remaining])
                await response.cancel_body()
                break
            chunks.append(chunk)
            received += len(chunk)
    text = b"".join(chunks).decode("utf-8-sig", "replace")
    try:
        data = javascript_json_parse(text)
    except (ValueError, TypeError) as error:
        raise WorkloadIdentityError(
            f"Token endpoint returned non-JSON response (status {response.status})", response.status,
            redact_sensitive(text), request_id,
        ) from error
    if not truthy(property_value(data, "access_token")):
        raise WorkloadIdentityError(
            f"Token endpoint response missing access_token: {javascript_json_stringify(redact_sensitive(data))}",
            response.status, redact_sensitive(data), request_id,
        )
    kind = property_value(data, "token_type")
    if truthy(kind) and cast(str, kind).lower() != "bearer":
        raise WorkloadIdentityError(
            f'Token endpoint response: unsupported token_type "{javascript_string(kind)}" (want Bearer)',
            response.status, redact_sensitive(data), request_id,
        )
    return cast(dict[str, object], data)


async def check_credentials_file_safety(path: str, on_warn: Callable[[str], None] | None = None) -> None:
    if sys.platform == "win32":
        return
    try:
        resolved = await asyncio.to_thread(os.path.realpath, path, strict=True)
        info = await asyncio.to_thread(os.stat, resolved)
    except OSError:
        return
    mode = info.st_mode & 0o777
    if mode & 0o022:
        raise WorkloadIdentityError(
            f"Credentials file at {resolved} is group/world-writable (mode 0o{mode:o}); "
            f"this allows other local users to plant tokens. Run `chmod 600 {resolved}`.",
        )
    if mode & 0o044:
        raise WorkloadIdentityError(
            f"Credentials file at {resolved} is group/world-readable (mode 0o{mode:o}); "
            f"run `chmod 600 {resolved}` before retrying.",
        )
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        message = f"credentials file at {resolved} is owned by uid {info.st_uid} (current process uid {os.getuid()}); verify this is intentional."
        if on_warn is None:
            print("anthropic-sdk: " + message, file=sys.stderr)
        else:
            on_warn(message)


def _pretty_json(data: object) -> str:
    compact = javascript_json_stringify(data)
    if compact is None:
        raise TypeError("Cannot write undefined as JSON")
    result: list[str] = []
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(compact):
        if quoted:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
            result.append(character)
        elif character in "[{":
            result.append(character)
            depth += 1
            if compact[index + 1:index + 2] not in ("]", "}"):
                result.append("\n" + "  " * depth)
        elif character in "]}":
            depth -= 1
            if compact[index - 1:index] not in ("[", "{"):
                result.append("\n" + "  " * depth)
            result.append(character)
        elif character == ",":
            result.append(",\n" + "  " * depth)
        elif character == ":":
            result.append(": ")
        else:
            result.append(character)
    return "".join(result)


async def write_credentials_file_atomic(target_path: str, data: object) -> None:
    def write() -> None:
        directory = os.path.dirname(target_path) or "."
        missing: list[Path] = []
        current = Path(directory)
        while not current.exists():
            missing.append(current)
            if current.parent == current:
                break
            current = current.parent
        for path in reversed(missing):
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                if not path.is_dir():
                    raise
        temporary = f"{target_path}.{os.getpid()}.{random_base36()}.tmp"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                encoded = _pretty_json(data).encode("utf-8")
                offset = 0
                while offset < len(encoded):
                    offset += os.write(descriptor, encoded[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, target_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        try:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    await asyncio.to_thread(write)
