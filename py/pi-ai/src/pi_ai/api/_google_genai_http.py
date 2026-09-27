"""Native generateContentStream transport from @google/genai 2.21.0.

Copyright 2025 Google LLC. SPDX-License-Identifier: Apache-2.0.
Modified: translated the API-key streaming path to native Python.
Retry behavior derives from p-retry 4.6.2 and retry 0.13.1 (MIT);
their original notices are retained in ../licenses/.

Only the API-key path used by pi is exposed. The wire conversions, optional
SDK retry policy, per-attempt deadlines and Gemini SSE parser follow the
published Apache-2.0 JavaScript SDK. Python runtime headers identify Python.
"""

from __future__ import annotations

import asyncio
import codecs
import inspect
import logging
import math
import os
import platform
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, MutableMapping
from typing import Protocol, cast

import httpx
from ada_url import URLSearchParams

from .._javascript import javascript_json_parse, javascript_json_stringify, javascript_string
from .._json_runtime import JS_WHITESPACE
from .._values import JSON_NULL, UNDEFINED
from ..abort import AbortController, AbortSignal
from ..auth.oauth._http import OAuthHttpResponse, fetch, parse_url
from ._google_genai_convert import (
    contents, entries, generate_parameters, generate_response, get,
    move_response_schema, nullish, spread, truthy,
)

_logger = logging.getLogger(__name__)
_DEFAULT_RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]
_NETWORK_ERROR_MESSAGES = {
    "Failed to fetch", "NetworkError when attempting to fetch resource.",
    "The Internet connection appears to be offline.", "Network request failed",
}


class ApiError(Exception):
    name = "ApiError"

    def __init__(self, *, message: str, status: int | float | str) -> None:
        super().__init__(message)
        self.status = status

    @property
    def message(self) -> str:
        return str(self)


class CallableTool(Protocol):
    """Native equivalent of the SDK's callback-based CallableTool interface."""

    def tool(self) -> object | Awaitable[object]: ...
    def call_tool(self, function_calls: list[object]) -> list[object] | Awaitable[list[object]]: ...


def _number(value: object) -> float:
    if value is None or value is JSON_NULL:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if value is UNDEFINED:
        return math.nan
    text = javascript_string(value).strip(JS_WHITESPACE)
    if not text:
        return 0.0
    try:
        if text.lower().startswith(("0x", "0b", "0o")):
            return float(int(text, 0))
        if text not in ("Infinity", "+Infinity", "-Infinity") and re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text) is None:
            return math.nan
        return float(text)
    except ValueError:
        return math.nan


def _default(value: object, default: object) -> object:
    return default if nullish(value) else value


def _round(value: float) -> float:
    return float(math.floor(value + 0.5)) if math.isfinite(value) else value


def _timer_seconds(milliseconds: float) -> float:
    return (math.trunc(milliseconds) if 1 <= milliseconds <= 2147483647 else 1) / 1000


def _patch_http_options(base: Mapping[str, object], overrides: object) -> dict[str, object]:
    # The SDK clones the base through JSON, then shallow-merges object fields.
    result = cast(dict[str, object], javascript_json_parse(cast(str, javascript_json_stringify(base))))
    for key, value in entries(overrides).items():
        if _value_type(value) == "object":
            result[key] = {**spread(result.get(key, UNDEFINED)), **spread(value)}
        elif value is not UNDEFINED:
            result[key] = value
    return result


def _value_type(value: object) -> str:
    if value is UNDEFINED:
        return "undefined"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "function" if callable(value) else "object"


def _merge_extra_body(target: object, source: object) -> dict[str, object]:
    result = spread(target)
    for key, value in entries(source).items():
        previous = result.get(key, UNDEFINED)
        if truthy(value) and isinstance(value, Mapping) and truthy(previous) and isinstance(previous, Mapping):
            result[key] = _merge_extra_body(previous, value)
        else:
            if truthy(previous) and truthy(value) and _value_type(previous) != _value_type(value):
                _logger.warning('includeExtraBodyToRequestInit:deepMerge: Type mismatch for key "%s". Original type: %s, New type: %s. Overwriting.', key, _value_type(previous), _value_type(value))
            result[key] = value
    return result


def _append_header(headers: dict[str, str], key: str, value: object) -> None:
    key.encode("latin-1")
    if re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) is None:
        raise TypeError(f'Headers.append: "{key}" is an invalid header name.')
    text = javascript_string(value)
    text.encode("latin-1")
    text = text.strip(" \t\r\n")
    if any(character in text for character in ("\x00", "\r", "\n")):
        raise TypeError(f'Headers.append: "{text}" is an invalid header value.')
    lower = key.lower()
    headers[lower] = headers[lower] + ", " + text if lower in headers else text


class _Attempt:
    def __init__(self, timeout: object, caller: AbortSignal | None) -> None:
        self.signal: AbortSignal | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._caller = caller
        self._on_abort: Callable[[], None] | None = None
        milliseconds = _number(timeout)
        if not (truthy(timeout) and milliseconds > 0) and caller is None:
            return
        controller = AbortController()
        self.signal = controller.signal
        if truthy(timeout) and milliseconds > 0:
            self._timer = asyncio.get_running_loop().call_later(_timer_seconds(milliseconds), controller.abort)
        if caller is not None:
            self._on_abort = controller.abort
            if caller.aborted:
                controller.abort()
            else:
                caller.add_event_listener("abort", self._on_abort)

    def dispose(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._caller is not None and self._on_abort is not None:
            self._caller.remove_event_listener("abort", self._on_abort)


async def _throw_if_not_ok(response: OAuthHttpResponse) -> None:
    if response.ok:
        return
    if "application/json" in response.headers.get("content-type", ""):
        error_body = javascript_json_parse(await response.text())
    else:
        error_body = {"error": {"message": await response.text(), "code": response.status, "status": response.status_text}}
    message = cast(str, javascript_json_stringify(error_body))
    if 400 <= response.status < 600:
        raise ApiError(message=message, status=response.status)
    raise RuntimeError(message)


async def _stream_segments(response: OAuthHttpResponse) -> AsyncIterator[str]:
    if response.body is None:
        raise RuntimeError("Response body is empty")
    # TextDecoder's default strips one leading UTF-8 BOM across chunk splits.
    decoder = codecs.getincrementaldecoder("utf-8-sig")("replace")
    buffer = ""
    iterator = response.body.__aiter__()
    try:
        while True:
            try:
                data = await anext(iterator)
            except StopAsyncIteration:
                if buffer.strip(JS_WHITESPACE):
                    raise RuntimeError("Incomplete JSON segment at the end")
                break
            chunk = decoder.decode(data, final=False)
            try:
                parsed = javascript_json_parse(chunk)
                if isinstance(parsed, Mapping) and "error" in parsed:
                    error = javascript_json_parse(cast(str, javascript_json_stringify(parsed["error"])))
                    status, code = get(error, "status"), get(error, "code")
                    if 400 <= _number(code) < 600:
                        raise ApiError(message=f"got status: {javascript_string(status)}. {javascript_json_stringify(parsed)}", status=cast(int | float | str, code))
            except Exception as error:
                if getattr(error, "name", None) == "ApiError":
                    raise
            buffer += chunk
            while True:
                delimiter_index = -1
                delimiter_length = 0
                for delimiter in ("\n\n", "\r\r", "\r\n\r\n"):
                    index = buffer.find(delimiter)
                    if index != -1 and (delimiter_index == -1 or index < delimiter_index):
                        delimiter_index, delimiter_length = index, len(delimiter)
                if delimiter_index == -1:
                    break
                event = buffer[:delimiter_index].strip(JS_WHITESPACE)
                buffer = buffer[delimiter_index + delimiter_length:]
                if event.startswith("data:"):
                    # This SDK parses one complete data-prefixed segment. It
                    # neither folds multiple data lines nor treats [DONE] specially.
                    yield event[5:].strip(JS_WHITESPACE)
    finally:
        # The SDK only releases its reader lock: it does not cancel the body.
        # The response still owns its byte iterator, just as it owns the stream.
        del iterator


def _is_callable_tool(tool: object) -> bool:
    if nullish(tool) or isinstance(tool, (str, int, float, bool)):
        raise TypeError("Cannot use 'in' operator to search for 'callTool'")
    return callable(get(tool, "call_tool"))


async def _tool_declaration(tool: object) -> object:
    value = cast(Callable[[], object], get(tool, "tool"))()
    return await value if inspect.isawaitable(value) else value


class GoogleGenAI:
    def __init__(
        self, *, api_key: str, base_url: str | None = None,
        api_version: str | None = None, headers: Mapping[str, str] | None = None,
    ) -> None:
        enterprise = os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE")
        vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        use_enterprise = enterprise is not None and enterprise.strip(JS_WHITESPACE).lower() == "true"
        use_vertex = vertex is not None and vertex.strip(JS_WHITESPACE).lower() == "true"
        if enterprise is not None and vertex is not None and use_enterprise != use_vertex:
            _logger.warning("Warning: Both GOOGLE_GENAI_USE_ENTERPRISE and GOOGLE_GENAI_USE_VERTEXAI are set with conflicting values. The value of GOOGLE_GENAI_USE_ENTERPRISE will be used.")
        self.vertexai = use_enterprise if enterprise is not None else use_vertex
        if os.environ.get("GOOGLE_API_KEY", "").strip(JS_WHITESPACE) and os.environ.get("GEMINI_API_KEY", "").strip(JS_WHITESPACE):
            _logger.warning("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
        self.api_key = api_key
        # pi always supplies an explicit API key. It takes precedence over
        # implicit project/location, including when the environment enables Vertex.
        if self.vertexai and api_key and (os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip(JS_WHITESPACE) or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip(JS_WHITESPACE)):
            _logger.debug("The user provided Vertex AI API key will take precedence over the project/location from the environment variables.")
        if not self.vertexai and not api_key:
            _logger.warning("API key should be set when using the Gemini API.")
            _logger.warning("API key should be set when using the Gemini API.")
        selected_base = base_url or os.environ.get("GOOGLE_VERTEX_BASE_URL" if self.vertexai else "GOOGLE_GEMINI_BASE_URL", "").strip(JS_WHITESPACE)
        options: dict[str, object] = {
            "baseUrl": "https://aiplatform.googleapis.com/" if self.vertexai else "https://generativelanguage.googleapis.com/",
            "apiVersion": "v1beta1" if self.vertexai else "v1beta",
            "headers": self.get_default_headers(),
        }
        overrides: dict[str, object] = {}
        if selected_base:
            overrides["baseUrl"] = selected_base
        elif base_url is not None:
            overrides["baseUrl"] = base_url
        if api_version is not None:
            overrides["apiVersion"] = api_version
        if headers is not None:
            overrides["headers"] = headers
        self._http_options = _patch_http_options(options, overrides)

    def get_default_headers(self) -> dict[str, str]:
        version = f"google-genai-sdk/2.21.0 gl-python/{platform.python_version()}"
        return {"User-Agent": version, "x-goog-api-client": version, "Content-Type": "application/json"}

    async def _process_params(self, params: object) -> object:
        config = get(params, "config")
        tools = get(config, "tools")
        if not truthy(tools):
            return params
        if not isinstance(tools, list):
            raise TypeError("tools.map is not a function")

        async def transform(tool: object) -> object:
            return await _tool_declaration(tool) if _is_callable_tool(tool) else tool

        transformed = list(await asyncio.gather(*(transform(tool) for tool in tools)))
        new_config = {**spread(config), "tools": transformed}
        new_params = {"model": get(params, "model"), "contents": get(params, "contents"), "config": new_config}
        if any(isinstance(tool, Mapping) and "inputSchema" in tool for tool in tools):
            http_options = get(config, "httpOptions")
            new_headers = spread(get(http_options, "headers"))
            if not new_headers:
                new_headers = cast(dict[str, object], self.get_default_headers())
            previous = _default(new_headers.get("x-goog-api-client", UNDEFINED), "")
            new_headers["x-goog-api-client"] = (javascript_string(previous) + " mcp_used/unknown").lstrip(JS_WHITESPACE)
            new_config["httpOptions"] = {**spread(http_options), "headers": new_headers}
        return new_params

    async def generate_content_stream(self, params: object) -> AsyncIterator[dict[str, object]]:
        move_response_schema(params)
        config = get(params, "config")
        afc = get(config, "automaticFunctionCalling")
        disabled = truthy(get(afc, "disable"))
        tools = cast(list[object], _default(get(config, "tools"), []))
        if not disabled:
            disabled = not any(_is_callable_tool(tool) for tool in tools)
        maximum = get(afc, "maximumRemoteCalls")
        if not disabled and ((truthy(maximum) and (_number(maximum) < 0 or type(maximum) not in (int, float) or not math.isfinite(cast(float, maximum)) or _number(maximum) != math.trunc(_number(maximum)))) or (not nullish(maximum) and _number(maximum) == 0)):
            _logger.warning("Invalid maximumRemoteCalls value provided for automatic function calling. Disabled automatic function calling. Please provide a valid integer value greater than 0. maximumRemoteCalls provided: %s", javascript_string(maximum))
            disabled = True
        if disabled:
            return await self._generate_internal(await self._process_params(params))
        incompatible = [index for index, tool in enumerate(tools) if not _is_callable_tool(tool) and truthy(get(tool, "functionDeclarations")) and _number(get(get(tool, "functionDeclarations"), "length")) > 0]
        if incompatible:
            formatted = ", ".join(f"tools[{index}]" for index in incompatible)
            raise ValueError(f'Incompatible tools found at {formatted}. Automatic function calling with CallableTools (or MCP objects) and basic FunctionDeclarations" is not yet supported.')
        if truthy(get(get(get(config, "toolConfig"), "functionCallingConfig"), "streamFunctionCallArguments")) and not truthy(get(afc, "disable")):
            raise ValueError("Running in streaming mode with 'streamFunctionCallArguments' enabled, this feature is not compatible with automatic function calling (AFC). Please set 'config.automaticFunctionCalling.disable' to true to disable AFC or leave 'config.toolConfig.functionCallingConfig.streamFunctionCallArguments' to be undefined or set to false to disable streaming function call arguments feature.")
        afc_tools: dict[str, object] = {}
        for tool in tools:
            if _is_callable_tool(tool):
                declaration = await _tool_declaration(tool)
                for item in cast(list[object], _default(get(declaration, "functionDeclarations"), [])):
                    name = get(item, "name")
                    if not truthy(name):
                        raise ValueError("Function declaration name is required.")
                    if name in afc_tools:
                        raise ValueError(f"Duplicate tool declaration name: {javascript_string(name)}")
                    afc_tools[cast(str, name)] = tool
        return self._afc_stream(params, afc_tools, _number(_default(maximum, 10)))

    async def _afc_stream(self, params: object, tools: Mapping[str, object], maximum: float) -> AsyncIterator[dict[str, object]]:
        were_functions_called = False
        count = 0
        while count < maximum:
            if were_functions_called:
                count += 1
                were_functions_called = False
            response = await self._generate_internal(await self._process_params(params))
            function_responses: list[object] = []
            response_contents: list[object] = []
            try:
                async for chunk in response:
                    yield chunk
                    candidate = get(get(chunk, "candidates"), "0")
                    content = get(candidate, "content")
                    if truthy(get(chunk, "candidates")) and truthy(content):
                        response_contents.append(content)
                        for part in cast(list[object], _default(get(content, "parts"), [])):
                            call = get(part, "functionCall")
                            if count < maximum and truthy(call):
                                name = get(call, "name")
                                if not truthy(name):
                                    raise ValueError("Function call name was not returned by the model.")
                                if name not in tools:
                                    raise ValueError(f"Automatic function calling was requested, but not all the tools the model used implement the CallableTool interface. Available tools: [object Map Iterator], missing tool: {javascript_string(name)}")
                                value = cast(Callable[[list[object]], object], get(tools[cast(str, name)], "call_tool"))([call])
                                parts = await value if inspect.isawaitable(value) else value
                                function_responses.extend(cast(list[object], parts))
            finally:
                close = getattr(response, "aclose", None)
                if close is not None:
                    await close()
            if not function_responses:
                break
            were_functions_called = True
            yield {"candidates": [{"content": {"role": "user", "parts": function_responses}}]}
            cast(MutableMapping[str, object], params)["contents"] = contents(get(params, "contents")) + response_contents + [{"role": "user", "parts": function_responses}]

    async def _generate_internal(self, params: object) -> AsyncIterator[dict[str, object]]:
        model, payload = generate_parameters(params, vertex=self.vertexai)
        config = get(params, "config")
        override = get(config, "httpOptions")
        http_options = _patch_http_options(self._http_options, override) if truthy(override) else self._http_options
        base, version = http_options.get("baseUrl", UNDEFINED), http_options.get("apiVersion", UNDEFINED)
        if base is UNDEFINED or version is UNDEFINED:
            raise RuntimeError("HTTP options are not correctly set.")
        if not isinstance(base, str):
            raise TypeError("httpOptions.baseUrl.endsWith is not a function")
        elements = [base[:-1] if base.endswith("/") else base]
        if truthy(version):
            elements.append(javascript_string(version))
        elements.append(model + ":streamGenerateContent?alt=sse")
        url = parse_url("/".join(elements))
        query = URLSearchParams(url.search)
        if query.get("alt") != "sse":
            query.set("alt", "sse")
            url.search = str(query)
        body = cast(str, javascript_json_stringify(payload))
        extra_body = http_options.get("extraBody", UNDEFINED)
        if truthy(extra_body) and entries(extra_body):
            payload = _merge_extra_body(javascript_json_parse(body), extra_body)
            body = cast(str, javascript_json_stringify(payload))
        headers: dict[str, str] = {}
        configured_headers = http_options.get("headers", UNDEFINED)
        if truthy(configured_headers):
            for key, value in entries(configured_headers).items():
                _append_header(headers, key, value)
        timeout = http_options.get("timeout", UNDEFINED)
        milliseconds = _number(timeout)
        if truthy(timeout) and milliseconds > 0:
            seconds = float(math.ceil(milliseconds / 1000)) if math.isfinite(milliseconds) else milliseconds
            _append_header(headers, "X-Server-Timeout", seconds)
        if self.api_key.startswith("auth_tokens/"):
            raise ValueError("Ephemeral tokens are only supported by the live API.")
        if "x-goog-api-key" not in headers:
            _append_header(headers, "x-goog-api-key", self.api_key)
        caller = get(config, "abortSignal")
        signal = cast(AbortSignal | None, caller if truthy(caller) else None)
        response = await self._api_call(url.href, body, headers, http_options.get("retryOptions", UNDEFINED), timeout, signal)
        await _throw_if_not_ok(response)

        async def converted() -> AsyncIterator[dict[str, object]]:
            segments = _stream_segments(response)
            try:
                async for segment in segments:
                    # Response(string).json() round-trips through USVString.
                    scalar = segment.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
                    scalar = re.sub(r"[\ud800-\udfff]", "\ufffd", scalar)
                    item = generate_response(javascript_json_parse(scalar), vertex=self.vertexai)
                    item["sdkHttpResponse"] = {"headers": dict(sorted(response.headers.items()))}
                    yield item
            finally:
                await segments.aclose()

        return converted()

    async def _api_call(
        self, url: str, body: str, headers: Mapping[str, str], retry: object,
        timeout: object, caller: AbortSignal | None,
    ) -> OAuthHttpResponse:
        retry_statuses = _default(get(retry, "httpStatusCodes"), _DEFAULT_RETRY_STATUS_CODES)

        async def run_fetch() -> OAuthHttpResponse:
            attempt = _Attempt(timeout, caller)
            try:
                try:
                    response = await fetch(url, method="POST", headers=httpx.Headers(headers, encoding="latin-1"), body=body, signal=attempt.signal)
                except httpx.HTTPError as error:
                    raise TypeError("fetch failed") from error
            except BaseException:
                attempt.dispose()
                raise
            if not truthy(retry) or response.ok or response.status not in cast(list[int], retry_statuses):
                # As in the SDK, successful/non-retryable responses retain the
                # attempt timer and caller listener through body consumption.
                return response
            try:
                await _throw_if_not_ok(response)
            finally:
                attempt.dispose()
            return response

        if not truthy(retry):
            return await run_fetch()
        attempts = _number(_default(get(retry, "attempts"), 5))
        attempts = max(1.0, attempts) if not math.isnan(attempts) else attempts
        retries = attempts - 1
        initial = _round(_number(_default(get(retry, "initialDelay"), 1.0)) * 1000)
        maximum = _round(_number(_default(get(retry, "maxDelay"), 60.0)) * 1000)
        maximum = math.nan if math.isnan(initial) or math.isnan(maximum) else max(initial, maximum)
        factor = _number(_default(get(retry, "expBase"), 2))
        jitter = _number(_default(get(retry, "jitter"), 1)) > 0
        delays: list[float] = []
        index = 0
        while index < retries:
            try:
                power = math.pow(factor, index)
            except OverflowError:
                power = math.inf
            amount = _round((random.random() + 1 if jitter else 1) * (max(initial, 1) if not math.isnan(initial) else initial) * power)
            delays.append(math.nan if math.isnan(amount) or math.isnan(maximum) else min(amount, maximum))
            index += 1
        # JS numeric sort leaves incomparable NaNs in their relative positions.
        delays.sort()
        errors: list[Exception] = []
        attempt_number = 1
        while True:
            try:
                return await run_fetch()
            except Exception as error:
                if isinstance(error, TypeError) and str(error) not in _NETWORK_ERROR_MESSAGES:
                    raise
                setattr(error, "attempt_number", attempt_number)
                setattr(error, "retries_left", retries - (attempt_number - 1))
                if caller is not None and caller.aborted:
                    nested_error = getattr(error, "error", UNDEFINED)
                    if not nullish(nested_error) and isinstance(nested_error, BaseException):
                        raise nested_error
                    raise
                errors.append(error)
                if attempt_number > len(delays):
                    counts: dict[str, int] = {}
                    main_error, highest = error, 0
                    for previous in errors:
                        message = str(previous)
                        counts[message] = counts.get(message, 0) + 1
                        if counts[message] >= highest:
                            main_error, highest = previous, counts[message]
                    raise main_error
                await asyncio.sleep(_timer_seconds(delays[attempt_number - 1]))
                attempt_number += 1


__all__ = ["ApiError", "CallableTool", "GoogleGenAI"]
