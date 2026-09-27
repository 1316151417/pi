"""JSON-schema and grammar sampling helpers from ``api/constrained-sampling.ts``."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ..types import JsonObject, Tool
from ..utils._javascript import javascript_json_stringify

_MISSING = object()
_JS_WHITESPACE = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_UNSUPPORTED_STRICT_SCHEMA_KEYS = (
    "$ref", "$defs", "definitions", "allOf", "oneOf", "patternProperties", "dependentSchemas", "dependencies",
    "unevaluatedProperties", "propertyNames", "contains", "prefixItems", "not", "if", "then", "else",
)


class _UnsupportedStrictJsonSchemaError(ValueError):
    pass


def _is_structured_schema(schema: object) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    types = [schema_type] if isinstance(schema_type, str) else schema_type if isinstance(schema_type, list) else []
    return "object" in types or "array" in types or "properties" in schema or "items" in schema


def _schema_allows_null(schema: object) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type == "null" or isinstance(schema_type, list) and "null" in schema_type:
        return True
    enum = schema.get("enum")
    if schema.get("const", _MISSING) is None or isinstance(enum, list) and None in enum:
        return True
    variants = schema.get("anyOf")
    return isinstance(variants, list) and any(_schema_allows_null(variant) for variant in variants)


def _make_json_schema_node_strict(schema: object) -> None:
    if not isinstance(schema, dict):
        raise _UnsupportedStrictJsonSchemaError("boolean schemas are unsupported")
    for key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if key in schema:
            raise _UnsupportedStrictJsonSchemaError(f"{key} schemas are unsupported")
    if "anyOf" in schema:
        variants = schema["anyOf"]
        if not isinstance(variants, list) or not variants:
            raise _UnsupportedStrictJsonSchemaError("anyOf must contain at least one schema")
        for variant in variants:
            if _is_structured_schema(variant):
                raise _UnsupportedStrictJsonSchemaError("object and array unions are unsupported")
            _make_json_schema_node_strict(variant)
    if "items" in schema:
        if isinstance(schema["items"], list):
            raise _UnsupportedStrictJsonSchemaError("tuple schemas are unsupported")
        _make_json_schema_node_strict(schema["items"])
    is_object_schema = schema.get("type") == "object"
    if "properties" in schema and not is_object_schema:
        raise _UnsupportedStrictJsonSchemaError("properties require type object")
    if not is_object_schema:
        return
    if "additionalProperties" in schema and schema["additionalProperties"] is not False:
        raise _UnsupportedStrictJsonSchemaError("schema-valued or true additionalProperties is unsupported")
    if "properties" in schema and not isinstance(schema["properties"], dict):
        raise _UnsupportedStrictJsonSchemaError("object properties must be a schema map")
    if "required" in schema and (
        not isinstance(schema["required"], list) or any(not isinstance(key, str) for key in schema["required"])
    ):
        raise _UnsupportedStrictJsonSchemaError("object required must be a string array")
    properties = schema.get("properties", {})
    numeric_keys = sorted(
        (key for key in properties if len(key) <= 10 and key.isascii() and key.isdecimal()
         and str(int(key)) == key and int(key) < 2**32 - 1),
        key=int,
    )
    property_names = numeric_keys + [key for key in properties if key not in numeric_keys]
    required = set(schema.get("required", []))
    if any(key not in property_names for key in required):
        raise _UnsupportedStrictJsonSchemaError("required contains an unknown property")
    for key in property_names:
        property_schema = properties[key]
        _make_json_schema_node_strict(property_schema)
        if key not in required and not _schema_allows_null(property_schema):
            properties[key] = {"anyOf": [property_schema, {"type": "null"}]}
    schema["required"] = property_names
    schema["additionalProperties"] = False


def make_strict_json_schema(schema: JsonObject) -> dict[str, object]:
    cloned = copy.deepcopy(schema)
    if not isinstance(cloned, dict):
        raise _UnsupportedStrictJsonSchemaError("root schema must have type object")
    _make_json_schema_node_strict(cloned)
    if cloned.get("type") != "object":
        raise _UnsupportedStrictJsonSchemaError("root schema must have type object")
    return cast(dict[str, object], cloned)


def get_json_schema_tool_parameters(tool: Tool, strict: bool | None) -> JsonObject:
    return cast(JsonObject, make_strict_json_schema(tool.parameters)) if strict is True else tool.parameters


@dataclass
class GrammarConstrainedSampling:
    format: Literal["lark", "regex"]
    definition: str
    input_property: str


@dataclass
class GrammarToolInputJsonBuffer:
    input: str = ""
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: Mapping[str, object], input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer, input_property: str, next_input: str, close: bool,
) -> str | None:
    previous_units = buffer.input.encode("utf-16-le", errors="surrogatepass")
    next_units = next_input.encode("utf-16-le", errors="surrogatepass")
    if buffer.closed:
        if close and next_units == previous_units:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not next_units.startswith(previous_units):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')
    input_delta = next_units[len(previous_units):].decode("utf-16-le", errors="surrogatepass")
    if not close and not input_delta:
        return None
    delta = ""
    if not buffer.started:
        delta = "{" + cast(str, javascript_json_stringify(input_property)) + ':"'
        buffer.started = True
    delta += cast(str, javascript_json_stringify(input_delta))[1:-1]
    buffer.input = next_input
    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = tool.parameters
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")
    input_property = required[0]
    properties = schema.get("properties")
    property_schema = properties.get(input_property, _MISSING) if isinstance(properties, dict) else _MISSING
    if (property_schema is _MISSING or property_schema is None or property_schema is False
            or property_schema == 0 or property_schema == ""):
        raise ValueError(f"grammar constrained sampling requires a properties entry for {input_property}")
    if not isinstance(property_schema, dict) or property_schema.get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {input_property} must have type string")
    return input_property


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> bool | None:
    config = tool.constrained_sampling
    if not isinstance(config, dict) or config.get("type") != "json_schema":
        return None
    if supports_strict_mode:
        try:
            make_strict_json_schema(tool.parameters)
            return True
        except _UnsupportedStrictJsonSchemaError as error:
            if config.get("strict") != "require":
                return None
            raise ValueError(f'Tool "{tool.name}" requires JSON-schema constrained sampling, but {error}.') from error
    if config.get("strict") == "require":
        raise ValueError(f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.')
    return None


def resolve_grammar_constrained_sampling(tool: Tool, supports_openai_grammar_tools: bool) -> GrammarConstrainedSampling | None:
    config = tool.constrained_sampling
    if not isinstance(config, dict) or config.get("type") != "grammar":
        return None
    if not supports_openai_grammar_tools:
        return None
    variants = config.get("variants")
    if not isinstance(variants, dict):
        raise TypeError("grammar constrained sampling variants must be an object")
    lark_definition = variants.get("openai_lark")
    regex_definition = variants.get("openai_regex")
    has_lark_definition = isinstance(lark_definition, str) and bool(lark_definition.strip(_JS_WHITESPACE))
    has_regex_definition = isinstance(regex_definition, str) and bool(regex_definition.strip(_JS_WHITESPACE))
    if not has_lark_definition and not has_regex_definition:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.')
    try:
        return GrammarConstrainedSampling(
            format="lark" if has_lark_definition else "regex",
            definition=cast(str, lark_definition if has_lark_definition else regex_definition),
            input_property=_infer_grammar_input_property(tool),
        )
    except Exception as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.') from error


def create_grammar_tool_input_properties(tools: Sequence[Tool] | None, supports_openai_grammar_tools: bool) -> dict[str, str]:
    properties: dict[str, str] = {}
    for tool in tools or ():
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            properties[tool.name] = grammar.input_property
    return properties
