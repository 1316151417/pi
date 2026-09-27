"""Responses resource behavior from the pinned OpenAI JavaScript SDK 6.40.0."""

from collections.abc import Iterable, Mapping
from typing import cast

from .._javascript import javascript_string
from .._values import JSON_NULL, UNDEFINED
from ..abort import AbortSignal
from ._openai_http import OpenAIHttpClient, OpenAIResponse, _truthy


async def create_response(
    client: OpenAIHttpClient, body: object, *, signal: AbortSignal | None = None,
    timeout_ms: int | float | None = None,
) -> OpenAIResponse:
    if body is None or body is JSON_NULL or body is UNDEFINED:
        kind = "undefined" if body is UNDEFINED else "null"
        raise TypeError(f"Cannot read properties of {kind} (reading 'stream')")
    streaming = body.get("stream", False) if isinstance(body, Mapping) else False
    result = await client.post("/responses", body, stream=_truthy(streaming), signal=signal, timeout_ms=timeout_ms)
    response = result.data
    if response is None or response is UNDEFINED or isinstance(response, (str, int, float, bool)):
        raise TypeError(f"Cannot use 'in' operator to search for 'object' in {javascript_string(response)}")
    if isinstance(response, dict) and response.get("object") == "response":
        texts: list[str] = []
        outputs = cast(Iterable[Mapping[str, object]], response.get("output", UNDEFINED))
        for output in outputs:
            if output.get("type") != "message":
                continue
            for content in cast(Iterable[Mapping[str, object]], output.get("content", UNDEFINED)):
                if content.get("type") == "output_text":
                    text = content.get("text", UNDEFINED)
                    texts.append("" if text is None or text is UNDEFINED else javascript_string(text))
        response["output_text"] = "".join(texts)
    return result
