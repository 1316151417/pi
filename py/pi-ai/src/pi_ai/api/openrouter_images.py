"""OpenRouter image generation from ``api/openrouter-images.ts``."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from collections.abc import Iterable, Mapping
from typing import cast

from ..auth.oauth._common import js_number
from ..types import (
    JSON_NULL, UNDEFINED, AssistantImages, Cost, ImageContent, ImagesContext,
    ImagesModel, ImagesOptions, ProviderResponse, TextContent, Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils._javascript import javascript_string
from ..utils.headers import headers_to_record, provider_headers_to_record
from ..utils.provider_retry import retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from ._openai_http import OpenAIHttpClient, OpenAIResponse, _truthy

_DATA_IMAGE = re.compile(r"^data:([^;]+);base64,([^\r\n\u2028\u2029]+)$")


def _property(value: object, key: str) -> object:
    if value is None or value is JSON_NULL or value is UNDEFINED:
        kind = "undefined" if value is UNDEFINED else "null"
        raise TypeError(f"Cannot read properties of {kind} (reading '{key}')")
    return value.get(key, UNDEFINED) if isinstance(value, Mapping) else UNDEFINED


def _parse_usage(raw: object, model: ImagesModel) -> Usage:
    prompt = _property(raw, "prompt_tokens")
    prompt = prompt if _truthy(prompt) else 0
    details = _property(raw, "prompt_tokens_details")
    cached = _property(details, "cached_tokens") if details is not None and details is not UNDEFINED else UNDEFINED
    cached = cached if _truthy(cached) else 0
    written = _property(details, "cache_write_tokens") if details is not None and details is not UNDEFINED else UNDEFINED
    written = written if _truthy(written) else 0
    cache_difference = js_number(cached) - js_number(written)
    cache_read = (math.nan if math.isnan(cache_difference) else max(0.0, cache_difference)) if js_number(written) > 0 else cached
    input_number = js_number(prompt) - js_number(cache_read) - js_number(written)
    input_tokens = math.nan if math.isnan(input_number) else max(0.0, input_number)
    output = _property(raw, "completion_tokens")
    output = output if _truthy(output) else 0
    cost_values = model.cost.to_json()
    cost = Cost(
        input=js_number(cost_values.get("input", UNDEFINED)) / 1_000_000 * input_tokens,
        output=js_number(cost_values.get("output", UNDEFINED)) / 1_000_000 * js_number(output),
        cache_read=js_number(cost_values.get("cacheRead", UNDEFINED)) / 1_000_000 * js_number(cache_read),
        cache_write=js_number(cost_values.get("cacheWrite", UNDEFINED)) / 1_000_000 * js_number(written),
    )
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write
    total_tokens: float | str = input_tokens
    for amount in (output, cache_read, written):
        if isinstance(total_tokens, str) or isinstance(amount, str):
            total_tokens = javascript_string(total_tokens) + javascript_string(amount)
        else:
            total_tokens += js_number(amount)
    return Usage(
        input=cast(int, input_tokens), output=cast(int, output), cache_read=cast(int, cache_read),
        cache_write=cast(int, written),
        total_tokens=cast(int, total_tokens), cost=cost,
    )


async def generate_images(
    model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None,
) -> AssistantImages:
    output = AssistantImages(
        api=model.api, provider=model.provider, model=model.id, output=[],
        stop_reason="stop", timestamp=time.time_ns() // 1_000_000,
    )
    try:
        api_key = options.api_key if options is not None else None
        if not api_key:
            raise RuntimeError(f"No API key for provider: {model.provider}")
        merged_headers = {**(model.headers or {}), **(options.headers or {} if options is not None else {})}
        client = OpenAIHttpClient(
            api_key=api_key, base_url=model.base_url, default_headers=provider_headers_to_record(merged_headers),
            custom_fetch=options.fetch if options is not None else None,
        )
        content: list[dict[str, object]] = []
        for item in context.input:
            if item.type == "text":
                content.append({"type": "text", "text": sanitize_surrogates(cast(TextContent, item).text)})
            else:
                picture = cast(ImageContent, item)
                content.append({"type": "image_url", "image_url": {"url": f"data:{picture.mime_type};base64,{picture.data}"}})
        params: object = {
            "model": model.id, "messages": [{"role": "user", "content": content}], "stream": False,
            "modalities": ["image", "text"] if "text" in model.output else ["image"],
        }
        replacement = options.on_payload(params, model) if options is not None and options.on_payload is not None else UNDEFINED
        if inspect.isawaitable(replacement):
            replacement = await replacement
        else:
            await asyncio.sleep(0)
        if replacement is not None and replacement is not UNDEFINED:
            params = None if replacement is JSON_NULL else replacement

        async def request() -> OpenAIResponse:
            # The JS resource reads body.stream before issuing any request,
            # including when onPayload explicitly replaces the body with null.
            streaming = _property(params, "stream")
            return await client.post(
                "/chat/completions", params, stream=_truthy(streaming),
                signal=options.signal if options is not None else None,
                timeout_ms=options.timeout_ms if options is not None else None,
            )

        result = await retry_provider_request(
            request, max_retries=options.max_retries if options is not None and options.max_retries is not None else 0,
            max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
            signal=options.signal if options is not None else None,
        )
        observed = options.on_response(
            ProviderResponse(status=result.response.status, headers=headers_to_record(result.response.headers)), model,
        ) if options is not None and options.on_response is not None else None
        if inspect.isawaitable(observed):
            await observed
        else:
            await asyncio.sleep(0)
        response = result.data
        response_id = _property(response, "id")
        output.response_id = None if response_id is UNDEFINED else cast(str | None, response_id)
        raw_usage = _property(response, "usage")
        if _truthy(raw_usage):
            output.usage = _parse_usage(raw_usage, model)
        choices = _property(response, "choices")
        if choices is None or choices is UNDEFINED:
            _property(choices, "0")
        choice = choices[0] if isinstance(choices, (list, tuple, str)) and len(choices) else (
            _property(choices, "0") if isinstance(choices, Mapping) else UNDEFINED
        )
        if _truthy(choice):
            message = _property(choice, "message")
            text = _property(message, "content")
            if isinstance(text, str) and text:
                output.output.append(TextContent(text=text))
            images = _property(message, "images")
            if images is None or images is UNDEFINED:
                images = []
            if isinstance(images, Mapping) or not isinstance(images, Iterable):
                raise TypeError("(choice.message.images ?? []) is not iterable")
            for image in images:
                image_url = _property(image, "image_url")
                if not isinstance(image_url, str):
                    image_url = _property(image_url, "url") if image_url is not None and image_url is not UNDEFINED else UNDEFINED
                if image_url is None or image_url is UNDEFINED:
                    continue
                if not isinstance(image_url, str):
                    raise TypeError("imageUrl?.startsWith is not a function")
                if not image_url.startswith("data:"):
                    continue
                matched = _DATA_IMAGE.fullmatch(image_url)
                if matched is not None:
                    output.output.append(ImageContent(mime_type=matched[1], data=matched[2]))
        return output
    except (Exception, asyncio.CancelledError) as error:
        output.stop_reason = "aborted" if options is not None and options.signal is not None and options.signal.aborted else "error"
        output.error_message = format_provider_error(normalize_provider_error(error))
        return output


__all__ = ["generate_images"]
