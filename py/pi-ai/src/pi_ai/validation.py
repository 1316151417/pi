"""Tool validation and coercion from pi-ai ``utils/validation.ts``.

The phases are optional-null normalization, in-place TypeBox conversion, pi's
JSON-schema conversion, and TypeBox validation. The TypeBox conversion's return
value is intentionally ignored. Python dictionary schemas use the source's
plain-JSON-schema path; JavaScript symbol metadata is not a JSON field.

Python dicts cannot be weakly referenced, so the identity cache retains compiled
schemas for this module's lifetime. TypeBox-specific non-JSON type-programming
limits are documented in ``_typebox_convert``; ECMAScript regular-expression
adaptation limits are documented in ``_typebox_regexp``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from ._javascript import javascript_json_stringify, javascript_string
from ._json_runtime import JS_WHITESPACE
from ._typebox_convert import convert
from ._typebox_primitives import entries, get_property, has_property, is_integer, number_from_string, strict_equal, structured_clone
from ._typebox_validator import LocalizedError, Validator
from ._values import UNDEFINED
from .types import Tool, ToolCall

__all__ = ["validate_tool_call", "validate_tool_arguments", "ValidationError"]


class ValidationError(ValueError):
    pass


_VALIDATORS: dict[int, tuple[object, Validator]] = {}


def _get_validator(schema: object) -> Validator:
    cached = _VALIDATORS.get(id(schema))
    if cached is not None and cached[0] is schema:
        return cached[1]
    validator = Validator(schema)
    # WeakMap.set rejects primitive keys, including boolean schemas. The
    # surrounding getSubSchemaValidator in pi catches that error when probing.
    if not isinstance(schema, (dict, list)):
        raise TypeError("Invalid value used as weak map key")
    _VALIDATORS[id(schema)] = schema, validator
    return validator


def _get_subschema_validator(schema: object) -> Validator | None:
    try:
        return _get_validator(schema)
    except Exception:
        return None


def _field(schema: object, key: str) -> object:
    if schema is None or schema is UNDEFINED:
        raise TypeError(f"Cannot read properties of {javascript_string(schema)} (reading '{key}')")
    return schema.get(key, UNDEFINED) if isinstance(schema, dict) else UNDEFINED


def _truthy(value: object) -> bool:
    return not (value is None or value is UNDEFINED or value is False or isinstance(value, str) and value == "" or isinstance(value, (int, float)) and not isinstance(value, bool) and (value == 0 or isinstance(value, float) and math.isnan(value)))


def _schema_types(schema: object) -> list[str]:
    field = _field(schema, "type")
    return [field] if isinstance(field, str) else [item for item in field if isinstance(item, str)] if isinstance(field, list) else []


def _matches_json_type(value: object, type_name: str) -> bool:
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return is_integer(value)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "null":
        return value is None
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _coerce_primitive_by_type(value: object, type_name: str) -> object:
    if type_name in ("number", "integer"):
        if value is None:
            return 0
        if isinstance(value, str) and value.strip(JS_WHITESPACE) != "":
            parsed = number_from_string(value)
            if math.isfinite(parsed) and (type_name == "number" or parsed.is_integer()):
                return parsed
        if isinstance(value, bool):
            return int(value)
    elif type_name == "boolean":
        if value is None:
            return False
        if value == "true":
            return True
        if value == "false":
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
    elif type_name == "string":
        if value is None:
            return ""
        if isinstance(value, (int, float, bool)):
            return javascript_string(value)
    elif type_name == "null":
        if strict_equal(value, "") or strict_equal(value, 0) or value is False:
            return None
    return value


def _coerce_with_union_schema(value: object, schemas: list[object]) -> object:
    for schema in schemas:
        validator = _get_subschema_validator(schema)
        if validator is not None and validator.check(value):
            return value
    for schema in schemas:
        candidate = _coerce_with_json_schema(structured_clone(value), schema)
        validator = _get_subschema_validator(schema)
        if validator is not None and validator.check(candidate):
            return candidate
    return value


def _coerce_with_json_schema(value: object, schema: object) -> object:
    next_value = value
    all_of = _field(schema, "allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            next_value = _coerce_with_json_schema(next_value, nested)
    for keyword in ("anyOf", "oneOf"):
        variants = _field(schema, keyword)
        if isinstance(variants, list):
            next_value = _coerce_with_union_schema(next_value, variants)
    types = _schema_types(schema)
    matched = len(types) > 1 and any(_matches_json_type(next_value, item) for item in types)
    if types and not matched:
        for typename in types:
            candidate = _coerce_primitive_by_type(next_value, typename)
            if not strict_equal(candidate, next_value):
                next_value = candidate
                break
    if "object" in types and isinstance(next_value, dict):
        properties = _field(schema, "properties")
        defined_keys = {name for name, _ in entries(properties)}
        if _truthy(properties):
            for name, child in entries(properties):
                if has_property(next_value, name, False):
                    next_value[name] = _coerce_with_json_schema(get_property(next_value, name), child)
        additional = _field(schema, "additionalProperties")
        if isinstance(additional, (dict, list)):
            for name, item in entries(next_value):
                if name not in defined_keys:
                    next_value[name] = _coerce_with_json_schema(item, additional)
    if "array" in types and isinstance(next_value, list):
        items = _field(schema, "items")
        if isinstance(items, list):
            for index in range(len(next_value)):
                if index < len(items) and _truthy(items[index]):
                    next_value[index] = _coerce_with_json_schema(next_value[index], items[index])
        elif isinstance(items, dict):
            for index in range(len(next_value)):
                next_value[index] = _coerce_with_json_schema(next_value[index], items)
    return next_value


def _normalize_optional_nulls(value: object, schema: object) -> None:
    if isinstance(value, list):
        items = _field(schema, "items")
        if isinstance(items, list):
            for index in range(len(value)):
                if index < len(items) and _truthy(items[index]):
                    _normalize_optional_nulls(value[index], items[index])
        elif _truthy(items):
            for item in value:
                _normalize_optional_nulls(item, items)
        return
    if not isinstance(value, dict):
        return
    properties = _field(schema, "properties")
    if not _truthy(properties):
        return
    required_value = _field(schema, "required")
    required = set(cast(Sequence[str], required_value)) if required_value is not None and required_value is not UNDEFINED else set()
    for name, child in entries(properties):
        if not has_property(value, name, False):
            continue
        remove = False
        item = get_property(value, name)
        if item is None and name not in required and not isinstance(_field(child, "$ref"), str):
            validator = _get_subschema_validator(child)
            remove = validator is not None and not validator.check(None)
        if remove:
            del value[name]
        else:
            _normalize_optional_nulls(item, child)


def _format_validation_path(error: LocalizedError) -> str:
    path = error.instance_path.removeprefix("/").replace("/", ".")
    if error.keyword == "required":
        required = error.params.get("requiredProperties")
        if isinstance(required, list) and required and required[0]:
            return f"{path}.{required[0]}" if path else str(required[0])
    return path or "root"


def collect_errors(value: object, schema: object, path: str = "$") -> list[tuple[str, str]]:
    """Preserved Python helper, returning TypeBox messages with legacy root paths."""
    result: list[tuple[str, str]] = []
    for error in Validator(schema).errors(value):
        suffix = _format_validation_path(error)
        result.append((path if suffix == "root" else f"{path}.{suffix}", error.message))
    return result


def _pretty_json(value: object) -> str:
    compact = javascript_json_stringify(value)
    if compact is None:
        return "undefined"
    pieces: list[str] = []
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(compact):
        if quoted:
            pieces.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            pieces.append(char)
        elif char in "[{":
            pieces.append(char)
            depth += 1
            if index + 1 < len(compact) and compact[index + 1] not in "]}":
                pieces.append("\n" + "  " * depth)
        elif char in "]}":
            depth -= 1
            if index and compact[index - 1] not in "[{":
                pieces.append("\n" + "  " * depth)
            pieces.append(char)
        elif char == ",":
            pieces.append(",\n" + "  " * depth)
        elif char == ":":
            pieces.append(": ")
        else:
            pieces.append(char)
    return "".join(pieces)


def validate_tool_call(tools: Sequence[Tool], tool_call: ToolCall) -> object:
    tool = next((tool for tool in tools if tool.name == tool_call.name), None)
    if tool is None:
        raise ValidationError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> object:
    args = structured_clone(tool_call.arguments)
    _normalize_optional_nulls(args, tool.parameters)
    convert(tool.parameters, args)
    validator = _get_validator(tool.parameters)
    coerced = _coerce_with_json_schema(args, tool.parameters)
    if not strict_equal(coerced, args):
        if isinstance(args, dict) and isinstance(coerced, (dict, list)):
            args.clear()
            args.update(entries(coerced))
        elif isinstance(args, list) and isinstance(coerced, list):
            # Object.keys deletion leaves the original array length unchanged.
            original_length = len(args)
            args[:] = coerced
            if len(args) < original_length:
                args.extend([UNDEFINED] * (original_length - len(args)))
        elif isinstance(args, list) and isinstance(coerced, dict):
            raise TypeError("A JavaScript array with named own properties has no plain Python list representation")
        else:
            # This early return, including returning unvalidated original args
            # when coercion fails, is part of pi's source behavior.
            return coerced if validator.check(coerced) else args
    if validator.check(args):
        return args
    errors = "\n".join(f"  - {_format_validation_path(error)}: {error.message}" for error in validator.errors(args)) or "Unknown validation error"
    raise ValidationError(f'Validation failed for tool "{tool_call.name}":\n{errors}\n\nReceived arguments:\n{_pretty_json(tool_call.arguments)}')
