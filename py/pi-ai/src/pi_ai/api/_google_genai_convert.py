"""The generateContent converters in @google/genai 2.21.0.

Copyright 2025 Google LLC. SPDX-License-Identifier: Apache-2.0.
Modified: translated the generated streaming converters to native Python.

Wire field names and conversion order follow the published Apache-2.0 SDK.
Unknown fields are deliberately discarded by generated record converters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping
from typing import cast

from .._javascript import javascript_object_keys
from .._json_runtime import utf16_units
from .._values import JSON_NULL, UNDEFINED


def nullish(value: object) -> bool:
    return value is None or value is UNDEFINED or value is JSON_NULL


def truthy(value: object) -> bool:
    if nullish(value) or value is False:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return True


def get(value: object, key: str) -> object:
    """Property access with an undefined fallback and UTF-16 string indexes."""
    if isinstance(value, Mapping):
        return value.get(key, UNDEFINED)
    if isinstance(value, (list, tuple, str)):
        items = utf16_units(value) if isinstance(value, str) else value
        if key == "length":
            return len(items)
        if key.isascii() and key.isdecimal() and str(int(key)) == key:
            index = int(key)
            return items[index] if index < len(items) else UNDEFINED
        return UNDEFINED
    if isinstance(value, (int, float, bool)):
        return UNDEFINED
    return getattr(value, key, UNDEFINED) if not nullish(value) else UNDEFINED


def entries(value: object) -> dict[str, object]:
    if nullish(value):
        raise TypeError("Cannot convert undefined or null to object")
    if isinstance(value, Mapping):
        return {key: value[key] for key in javascript_object_keys(value)}
    if isinstance(value, (list, tuple, str)):
        sequence = utf16_units(value) if isinstance(value, str) else value
        return {str(index): item for index, item in enumerate(sequence)}
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def spread(value: object) -> dict[str, object]:
    return {} if nullish(value) else entries(value)


def _typeof(value: object) -> str:
    if value is UNDEFINED:
        return "undefined"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "function" if callable(value) else "object"


def _map(value: object, convert: Callable[[object], object]) -> object:
    return [convert(item) for item in value] if isinstance(value, list) else value


def _reject(key: str, *, vertex: bool) -> None:
    supported = "Gemini Developer API" if vertex else "Gemini Enterprise Agent Platform"
    unsupported = "Gemini Enterprise Agent Platform" if vertex else "Gemini Developer API"
    raise ValueError(f"{key} parameter is only supported in {supported} mode, not in {unsupported} mode.")


def _record(
    source: object, fields: tuple[str, ...], *, vertex: bool = False,
    transforms: Mapping[str, Callable[[object], object]] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields:
        reject = field.startswith("!")
        key = field[1:] if reject else field
        value = get(source, key)
        if reject:
            if value is not UNDEFINED:
                _reject(key, vertex=vertex)
        elif not nullish(value):
            result[key] = transforms[key](value) if transforms is not None and key in transforms else value
    return result


def _part_union(value: object) -> object:
    if nullish(value):
        raise ValueError("PartUnion is required")
    if _typeof(value) == "object":
        return value
    if isinstance(value, str):
        return {"text": value}
    raise ValueError(f"Unsupported part type: {_typeof(value)}")


def _parts(value: object) -> list[object]:
    if nullish(value) or (isinstance(value, list) and not value):
        raise ValueError("PartListUnion is required")
    return [_part_union(item) for item in value] if isinstance(value, list) else [_part_union(value)]


def _is_content(value: object) -> bool:
    return not nullish(value) and _typeof(value) == "object" and isinstance(get(value, "parts"), list)


def _content_union(value: object) -> object:
    if nullish(value):
        raise ValueError("ContentUnion is required")
    return value if _is_content(value) else {"role": "user", "parts": _parts(value)}


def contents(value: object) -> list[object]:
    if nullish(value) or (isinstance(value, list) and not value):
        raise ValueError("contents are required")
    if not isinstance(value, list):
        if isinstance(value, Mapping) and ("functionCall" in value or "functionResponse" in value):
            raise ValueError("To specify functionCall or functionResponse parts, please wrap them in a Content object, specifying the role for them")
        return [_content_union(value)]
    first_is_content = _is_content(value[0])
    for item in value:
        if _is_content(item) != first_is_content:
            raise ValueError("Mixing Content and Parts is not supported, please group the parts into a the appropriate Content objects and specify the roles for them")
        if not first_is_content and isinstance(item, Mapping) and ("functionCall" in item or "functionResponse" in item):
            raise ValueError("To specify functionCall or functionResponse parts, please wrap them, and any other parts, in Content objects as appropriate, specifying the role for them")
    return list(value) if first_is_content else [{"role": "user", "parts": _parts(value)}]


def _model(value: object, vertex: bool) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("model is required and must be a string")
    if any(part in value for part in ("..", "?", "&")):
        raise ValueError("invalid model parameter")
    if vertex:
        if value.startswith(("publishers/", "projects/", "models/")):
            return value
        if "/" in value:
            publisher, name = value.split("/")[:2]
            return f"publishers/{publisher}/models/{name}"
        return f"publishers/google/models/{value}"
    return value if value.startswith(("models/", "tunedModels/")) else f"models/{value}"


def _schema_type(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("type.toUpperCase is not a function")
    result = value.upper()
    return result if result in {"TYPE_UNSPECIFIED", "STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY", "OBJECT", "NULL"} else "TYPE_UNSPECIFIED"


def _schema(source: object) -> dict[str, object]:
    if nullish(source):
        raise TypeError("Cannot read properties of null or undefined")
    result: dict[str, object] = {}
    if truthy(get(source, "type")) and truthy(get(source, "anyOf")):
        raise ValueError("type and anyOf cannot be both populated.")
    any_of = get(source, "anyOf")
    if not nullish(any_of) and get(any_of, "length") == 2:
        if nullish(get(any_of, "0")):
            raise TypeError("Cannot read properties of null or undefined (reading 'type')")
        if get(get(any_of, "0"), "type") == "null":
            result["nullable"] = True
            source = get(any_of, "1")
        else:
            if nullish(get(any_of, "1")):
                raise TypeError("Cannot read properties of null or undefined (reading 'type')")
            if get(get(any_of, "1"), "type") == "null":
                result["nullable"] = True
                source = get(any_of, "0")
    type_list = get(source, "type")
    if isinstance(type_list, list):
        if "null" in type_list:
            result["nullable"] = True
        non_null = [item for item in type_list if item != "null"]
        if len(non_null) == 1:
            result["type"] = _schema_type(non_null[0])
        else:
            result["anyOf"] = [{"type": _schema_type(item)} for item in non_null]
    for field, value in entries(source).items():
        if nullish(value):
            continue
        if field == "type":
            if value == "null":
                raise ValueError("type: null can not be the only possible type for the field.")
            if not isinstance(value, list):
                result[field] = _schema_type(value)
        elif field == "items":
            result[field] = _schema(value)
        elif field == "anyOf":
            schemas: list[object] = []
            for item in cast(list[object], value):
                if get(item, "type") == "null":
                    result["nullable"] = True
                else:
                    schemas.append(_schema(item))
            result[field] = schemas
        elif field == "properties":
            result[field] = {key: _schema(item) for key, item in entries(value).items()}
        elif field != "additionalProperties":
            result[field] = value
    return result


def _tool_union(tool: object) -> object:
    if nullish(tool):
        raise TypeError("Cannot read properties of null or undefined (reading 'functionDeclarations')")
    declarations = get(tool, "functionDeclarations")
    if truthy(declarations):
        for declaration in cast(list[MutableMapping[str, object]], declarations):
            if nullish(declaration):
                raise TypeError("Cannot read properties of null or undefined (reading 'parameters')")
            for field in ("parameters", "response"):
                value = get(declaration, field)
                if truthy(value):
                    if "$schema" not in entries(value):
                        declaration[field] = _schema(value)
                    elif not truthy(get(declaration, field + "JsonSchema")):
                        declaration[field + "JsonSchema"] = value
                        del declaration[field]
    return tool


def move_response_schema(params: object) -> None:
    if nullish(params):
        raise TypeError("Cannot read properties of null or undefined (reading 'config')")
    config = get(params, "config")
    schema = get(config, "responseSchema")
    if truthy(config) and truthy(schema) and not truthy(get(config, "responseJsonSchema")) and "$schema" in entries(schema):
        record = cast(MutableMapping[str, object], config)
        record["responseJsonSchema"] = schema
        del record["responseSchema"]


def _content(value: object, vertex: bool) -> dict[str, object]:
    return _record(value, ("parts", "role"), transforms={"parts": lambda items: _map(items, lambda item: _part(item, vertex))})


def _part(value: object, vertex: bool) -> dict[str, object]:
    fields = (
        "mediaResolution", "!toolCall" if vertex else "toolCall", "!toolResponse" if vertex else "toolResponse",
        "audioTranscription", "codeExecutionResult", "executableCode", "fileData", "functionCall",
        "functionResponse", "inlineData", "text", "thought", "thoughtSignature", "videoMetadata",
        "!partMetadata" if vertex else "partMetadata", "mediaProcessing",
    )
    transforms: dict[str, Callable[[object], object]] = {} if vertex else {
        "fileData": lambda source: _record(source, ("!displayName", "fileUri", "mimeType")),
        "functionCall": lambda source: _record(source, ("args", "id", "name", "!partialArgs", "!willContinue")),
        "inlineData": lambda source: _record(source, ("data", "!displayName", "mimeType")),
    }
    return _record(value, fields, vertex=vertex, transforms=transforms)


def _tool(value: object, vertex: bool) -> dict[str, object]:
    fields = (
        "retrieval" if vertex else "!retrieval", "googleMaps", "mcpServers", "codeExecution", "computerUse",
        "enterpriseWebSearch" if vertex else "!enterpriseWebSearch", "exaAiSearch" if vertex else "!exaAiSearch",
        "functionDeclarations", "googleSearch", "googleSearchRetrieval", "parallelAiSearch" if vertex else "!parallelAiSearch",
        "urlContext", "!fileSearch" if vertex else "fileSearch",
    )
    transforms: dict[str, Callable[[object], object]] = {"functionDeclarations": lambda items: _map(items, lambda item: item)}
    if vertex:
        transforms["mcpServers"] = lambda items: _map(items, lambda item: _record(item, ("!name", "!streamableHttpTransport"), vertex=True))
        transforms["computerUse"] = lambda item: _record(item, ("enablePromptInjectionDetection", "environment", "excludedPredefinedFunctions", "!disabledSafetyPolicies"), vertex=True)
    else:
        transforms["mcpServers"] = lambda items: _map(items, lambda item: item)
        transforms["googleMaps"] = lambda item: _record(item, ("authConfig", "enableWidget", "!groundingTypes"), transforms={
            "authConfig": lambda auth: _record(auth, ("apiKey", "!apiKeyConfig", "!authType", "!googleServiceAccountConfig", "!httpBasicAuthConfig", "!oauthConfig", "!oidcConfig")),
        })
        transforms["googleSearch"] = lambda item: _record(item, ("!blockingConfidence", "!excludeDomains", "searchTypes", "timeRangeFilter"))
    return _record(_tool_union(value), fields, vertex=vertex, transforms=transforms)


def _tools(value: object, vertex: bool) -> list[object]:
    if nullish(value):
        raise ValueError("tools is required")
    if not isinstance(value, list):
        raise ValueError("tools is required and must be an array of Tools")
    return [_tool(item, vertex) for item in value]


def _tool_config(value: object, vertex: bool) -> dict[str, object]:
    transforms: dict[str, Callable[[object], object]] | None = None if vertex else {
        "functionCallingConfig": lambda item: _record(item, ("allowedFunctionNames", "mode", "!streamFunctionCallArguments")),
    }
    return _record(value, ("functionCallingConfig", "retrievalConfig", "!includeServerSideToolInvocations" if vertex else "includeServerSideToolInvocations"), vertex=vertex, transforms=transforms)


def _voice(value: object) -> dict[str, object]:
    return _record(value, ("replicatedVoiceConfig", "prebuiltVoiceConfig"), vertex=True, transforms={
        "replicatedVoiceConfig": lambda item: _record(item, ("mimeType", "voiceSampleAudio", "!consentAudio", "!voiceConsentSignature"), vertex=True),
    })


def _speech(value: object, vertex: bool) -> object:
    if isinstance(value, str):
        value = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": value}}}
    elif _typeof(value) != "object":
        raise ValueError(f"Unsupported speechConfig type: {_typeof(value)}")
    if not vertex:
        return value
    return _record(value, ("voiceConfig", "languageCode", "multiSpeakerVoiceConfig"), vertex=True, transforms={
        "voiceConfig": _voice,
        "multiSpeakerVoiceConfig": lambda item: _record(item, ("speakerVoiceConfigs",), vertex=True, transforms={
            "speakerVoiceConfigs": lambda items: _map(items, lambda speaker: _record(speaker, ("speaker", "voiceConfig"), vertex=True, transforms={"voiceConfig": _voice})),
        }),
    })


def _image(value: object, vertex: bool) -> dict[str, object]:
    if not vertex:
        return _record(value, ("aspectRatio", "imageSize", "!personGeneration", "!outputMimeType", "!outputCompressionQuality", "!imageOutputOptions", "!prominentPeople"))
    result = _record(value, ("aspectRatio", "imageSize", "personGeneration"), vertex=True)
    for source, target in (("outputMimeType", "mimeType"), ("outputCompressionQuality", "compressionQuality")):
        item = get(value, source)
        if not nullish(item):
            cast(dict[str, object], result.setdefault("imageOutputOptions", {}))[target] = item
    output = get(value, "imageOutputOptions")
    if not nullish(output):
        existing = result.get("imageOutputOptions", UNDEFINED)
        if existing is UNDEFINED:
            result["imageOutputOptions"] = output
        elif not truthy(output) or (_typeof(output) == "object" and not entries(output)):
            pass
        elif _typeof(output) == "object":
            cast(dict[str, object], existing).update(entries(output))
        else:
            raise ValueError("Cannot set value for an existing key. Key: imageOutputOptions")
    prominent = get(value, "prominentPeople")
    if not nullish(prominent):
        result["prominentPeople"] = prominent
    return result


def _cached_content(value: object, vertex: bool) -> str:
    if not isinstance(value, str):
        raise ValueError("name must be a string")
    if not vertex:
        return f"cachedContents/{value}" if "/" not in value else value
    if value.startswith("projects/"):
        return value
    if value.startswith("locations/"):
        return f"projects/undefined/{value}"
    if value.startswith("cachedContents/"):
        return f"projects/undefined/locations/undefined/{value}"
    if "/" not in value:
        return f"projects/undefined/locations/undefined/cachedContents/{value}"
    return value


def _config(source: object, parent: dict[str, object], vertex: bool) -> dict[str, object]:
    result: dict[str, object] = {}
    parent_fields = {"serviceTier", "systemInstruction", "safetySettings", "tools", "toolConfig", "labels", "cachedContent", "modelArmorConfig"}
    fields = (
        "serviceTier", "systemInstruction", "temperature", "topP", "topK", "candidateCount", "maxOutputTokens",
        "stopSequences", "responseLogprobs", "logprobs", "presencePenalty", "frequencyPenalty", "seed",
        "responseMimeType", "responseSchema", "responseJsonSchema", "routingConfig", "modelSelectionConfig",
        "safetySettings", "tools", "toolConfig", "labels", "cachedContent", "responseModalities", "mediaResolution",
        "speechConfig", "audioTimestamp", "thinkingConfig", "audioTranscriptionConfig", "imageConfig",
        "enableEnhancedCivicAnswers", "modelArmorConfig",
    )
    unsupported = {"enableEnhancedCivicAnswers"} if vertex else {"routingConfig", "modelSelectionConfig", "labels", "audioTimestamp", "modelArmorConfig"}
    for field in fields:
        value = get(source, field)
        if field in unsupported:
            if value is not UNDEFINED:
                _reject(field, vertex=vertex)
            continue
        if nullish(value):
            continue
        key = "modelConfig" if field == "modelSelectionConfig" else field
        if field == "systemInstruction":
            value = _content(_content_union(value), vertex)
        elif field == "responseSchema":
            value = _schema(value)
        elif field == "safetySettings":
            value = _map(value, (lambda item: item) if vertex else (lambda item: _record(item, ("category", "!method", "threshold"))))
        elif field == "tools":
            value = _tools(value, vertex)
        elif field == "toolConfig":
            value = _tool_config(value, vertex)
        elif field == "cachedContent":
            value = _cached_content(value, vertex)
        elif field == "speechConfig":
            value = _speech(value, vertex)
        elif field == "imageConfig":
            value = _image(value, vertex)
        (parent if field in parent_fields else result)[key] = value
    return result


def generate_parameters(params: object, *, vertex: bool) -> tuple[str, dict[str, object]]:
    body: dict[str, object] = {}
    model = get(params, "model")
    model_path = _model(model, vertex) if not nullish(model) else UNDEFINED
    content = get(params, "contents")
    if not nullish(content):
        body["contents"] = [_content(item, vertex) for item in contents(content)]
    config = get(params, "config")
    if not nullish(config):
        body["generationConfig"] = _config(config, body, vertex)
    if model_path is UNDEFINED:
        raise TypeError("Cannot convert undefined or null to object")
    return cast(str, model_path), body


def _candidate(value: object) -> dict[str, object]:
    return _record(value, ("content", "citationMetadata", "tokenCount", "finishReason", "groundingMetadata", "avgLogprobs", "index", "logprobsResult", "safetyRatings", "urlContextMetadata"), transforms={
        "citationMetadata": lambda item: {"citations": _map(get(item, "citationSources"), lambda entry: entry)} if not nullish(get(item, "citationSources")) else {},
        "safetyRatings": lambda items: _map(items, lambda item: item),
    })


def generate_response(source: object, *, vertex: bool) -> dict[str, object]:
    fields = ("sdkHttpResponse", "candidates", "createTime", "modelVersion", "promptFeedback", "responseId", "usageMetadata") if vertex else ("sdkHttpResponse", "candidates", "modelVersion", "promptFeedback", "responseId", "usageMetadata", "modelStatus")
    return _record(source, fields, transforms={"candidates": lambda items: _map(items, (lambda item: item) if vertex else _candidate)})
