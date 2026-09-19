"""Tool argument validation and coercion ported from pi-ai ``src/utils/validation.ts``.

TypeBox schemas are JSON Schema compatible, so this port validates plain JSON
schema dicts with the same coercion semantics:

1. ``normalize_optional_nulls`` drops ``null`` values for non-required properties.
2. ``Value.Convert``-style conversion rewrites primitives to the schema type.
3. A fallback coercion walks the schema for plain JSON-schema parameters.
4. Validation errors are reported as ``  - <path>: <message>`` lines like the TS
   implementation.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .types import Tool, ToolCall

__all__ = ["validate_tool_call", "validate_tool_arguments", "ValidationError"]


class ValidationError(ValueError):
    pass


# ---------------------------------------------------------------------------
# JSON type matching
# ---------------------------------------------------------------------------


def _schema_types(schema: Mapping[str, Any]) -> List[str]:
    type_field = schema.get("type")
    if isinstance(type_field, str):
        return [type_field]
    if isinstance(type_field, list):
        return [t for t in type_field if isinstance(t, str)]
    return []


def _matches_json_type(value: Any, type_name: str) -> bool:
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
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


def _js_strict_inequals(a: Any, b: Any) -> bool:
    """JS ``a !== b``: true when types differ or values differ (bool never equals number)."""
    if isinstance(a, bool) != isinstance(b, bool):
        return True
    return a != b or type(a) is not type(b)


# ---------------------------------------------------------------------------
# Primitive coercion (coercePrimitiveByType)
# ---------------------------------------------------------------------------


def _coerce_primitive_by_type(value: Any, type_name: str) -> Any:
    if type_name == "number":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip() != "":
            try:
                return float(value)
            except ValueError:
                return value
        if isinstance(value, bool):
            return 1 if value else 0
        return value
    if type_name == "integer":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip() != "":
            try:
                parsed = float(value)
                if parsed.is_integer():
                    return int(parsed)
            except ValueError:
                pass
            return value
        if isinstance(value, bool):
            return 1 if value else 0
        return value
    if type_name == "boolean":
        if value is None:
            return False
        if isinstance(value, str):
            if value == "true":
                return True
            if value == "false":
                return False
        if isinstance(value, int) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        return value
    if type_name == "string":
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return value
    if type_name == "null":
        if value == "" or value == 0 or value is False:
            return None
        return value
    return value


# ---------------------------------------------------------------------------
# Schema-walking coercion (coerceWithJsonSchema)
# ---------------------------------------------------------------------------


def _coerce_with_json_schema(value: Any, schema: Mapping[str, Any]) -> Any:
    next_value = value

    for nested in schema.get("allOf") or []:
        next_value = _coerce_with_json_schema(next_value, nested)

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            next_value = _coerce_with_union_schema(next_value, variants)

    schema_types = _schema_types(schema)
    matches_union_member = len(schema_types) > 1 and any(
        _matches_json_type(next_value, t) for t in schema_types
    )
    if len(schema_types) > 0 and not matches_union_member:
        for schema_type in schema_types:
            candidate = _coerce_primitive_by_type(next_value, schema_type)
            if _js_strict_inequals(candidate, next_value):
                next_value = candidate
                break

    if "object" in schema_types and isinstance(next_value, dict):
        _apply_schema_object_coercion(next_value, schema)

    if "array" in schema_types and isinstance(next_value, list):
        _apply_schema_array_coercion(next_value, schema)

    return next_value


def _coerce_with_union_schema(value: Any, schemas: Sequence[Mapping[str, Any]]) -> Any:
    for schema in schemas:
        if not collect_errors(value, schema):
            return value
    for schema in schemas:
        candidate = _coerce_with_json_schema(copy.deepcopy(value), schema)
        if not collect_errors(candidate, schema):
            return candidate
    return value


def _apply_schema_object_coercion(value: Dict[str, Any], schema: Mapping[str, Any]) -> None:
    properties = schema.get("properties")
    defined_keys = set(properties.keys()) if isinstance(properties, dict) else set()
    if isinstance(properties, dict):
        for key, property_schema in properties.items():
            if key not in value:
                continue
            value[key] = _coerce_with_json_schema(value[key], property_schema)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, property_value in list(value.items()):
            if key in defined_keys:
                continue
            value[key] = _coerce_with_json_schema(property_value, additional)


def _apply_schema_array_coercion(value: List[Any], schema: Mapping[str, Any]) -> None:
    items = schema.get("items")
    if isinstance(items, list):
        for index in range(len(value)):
            if index < len(items):
                value[index] = _coerce_with_json_schema(value[index], items[index])
        return
    if isinstance(items, dict):
        for index in range(len(value)):
            value[index] = _coerce_with_json_schema(value[index], items)


# ---------------------------------------------------------------------------
# normalizeOptionalNulls
# ---------------------------------------------------------------------------


def _normalize_optional_nulls(value: Any, schema: Mapping[str, Any]) -> None:
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index in range(len(value)):
                if index < len(items):
                    _normalize_optional_nulls(value[index], items[index])
        elif isinstance(items, dict):
            for item in value:
                _normalize_optional_nulls(item, items)
        return
    if not isinstance(value, dict):
        return
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    required = set(schema.get("required") or [])
    for key, property_schema in properties.items():
        if key not in value:
            continue
        if value[key] is None and key not in required and collect_errors(None, property_schema):
            del value[key]
        else:
            _normalize_optional_nulls(value[key], property_schema)


# ---------------------------------------------------------------------------
# Validation core (TypeBox Compile/Check/Errors equivalent)
# ---------------------------------------------------------------------------


def _enum_matches(value: Any, enum_values: Sequence[Any]) -> bool:
    for enum_value in enum_values:
        if not _js_strict_inequals(enum_value, value):
            return True
    return False


def collect_errors(value: Any, schema: Mapping[str, Any], path: str = "$") -> List[Tuple[str, str]]:
    """Return localized ``(instance_path, message)`` validation errors for ``value``."""
    errors: List[Tuple[str, str]] = []

    if "$ref" in schema:
        # Unsupported in the embedded-validator subset; treat as pass-through.
        return errors

    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        if not _enum_matches(value, enum_values):
            errors.append((path, f"Expected one of {json.dumps(enum_values)}"))
        return errors

    const_value = schema.get("const")
    if const_value is not None and _js_strict_inequals(value, const_value):
        errors.append((path, f"Expected const value {json.dumps(const_value)}"))
        return errors

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            if not any(not collect_errors(value, variant, path) for variant in variants):
                union_kind = "union" if key == "anyOf" else "intersection"
                errors.append((path, f"Value does not match {union_kind} schema"))
            return errors

    schema_types = _schema_types(schema)
    if schema_types and not any(_matches_json_type(value, t) for t in schema_types):
        errors.append((path, f"Expected {json.dumps(schema_types[0]) if len(schema_types) == 1 else schema_types}"))
        return errors

    if isinstance(value, dict):
        properties = schema.get("properties")
        required = schema.get("required") or []
        for name in required:
            if name not in value:
                errors.append((f"{path}.{name}", "Required property"))
        if isinstance(properties, dict):
            for key, sub_schema in properties.items():
                if key in value:
                    errors.extend(collect_errors(value[key], sub_schema, f"{path}.{key}"))
        additional = schema.get("additionalProperties")
        if additional is False and isinstance(properties, dict):
            for key in value:
                if key not in properties:
                    errors.append((path, f"Unexpected additional property '{key}'"))

    if isinstance(value, list):
        items = schema.get("items")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append((path, f"Array must have at least {min_items} items"))
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append((path, f"Array must have at most {max_items} items"))
        if isinstance(items, list):
            for index, item in enumerate(value):
                if index < len(items):
                    errors.extend(collect_errors(item, items[index], f"{path}[{index}]"))
        elif isinstance(items, dict):
            for index, item in enumerate(value):
                errors.extend(collect_errors(item, items, f"{path}[{index}]"))

    if isinstance(value, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append((path, f"String must be at least {min_length} characters"))
        if isinstance(max_length, int) and len(value) > max_length:
            errors.append((path, f"String must be at most {max_length} characters"))
        if isinstance(pattern, str):
            try:
                if not re.search(pattern, value):
                    errors.append((path, f"String does not match pattern {pattern!r}"))
            except re.error:
                pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append((path, f"Number must be greater or equal to {minimum}"))
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append((path, f"Number must be less or equal to {maximum}"))
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            errors.append((path, f"Number must be greater than {exclusive_minimum}"))
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            errors.append((path, f"Number must be less than {exclusive_maximum}"))

    return errors


# ---------------------------------------------------------------------------
# Value.Convert equivalent
# ---------------------------------------------------------------------------


def _convert_value(value: Any, schema: Mapping[str, Any]) -> Any:
    """Convert value types to the schema types (TypeBox Value.Convert semantics)."""
    if value is None:
        return None
    schema_types = _schema_types(schema)
    if schema_types and any(_matches_json_type(value, t) for t in schema_types):
        converted = value
    elif schema_types:
        converted = value
        for type_name in schema_types:
            candidate = _coerce_primitive_by_type(value, type_name)
            if _matches_json_type(candidate, type_name) and _js_strict_inequals(candidate, value):
                converted = candidate
                break
    else:
        converted = value

    if isinstance(converted, dict) and isinstance(schema.get("properties"), dict):
        for key, sub_schema in schema["properties"].items():
            if key in converted:
                converted[key] = _convert_value(converted[key], sub_schema)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            defined = set(schema["properties"].keys())
            for key in list(converted.keys()):
                if key not in defined:
                    converted[key] = _convert_value(converted[key], additional)
        return converted

    if isinstance(converted, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index in range(min(len(converted), len(items))):
                converted[index] = _convert_value(converted[index], items[index])
        elif isinstance(items, dict):
            for index in range(len(converted)):
                converted[index] = _convert_value(converted[index], items)
        return converted

    return converted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_tool_call(tools: Sequence[Tool], tool_call: ToolCall) -> Any:
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        raise ValidationError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def _format_validation_path(instance_path: str, message: str) -> str:
    """Mirror formatValidationPath: drop the leading '$', convert '/' to '.'."""
    if message == "Required property":
        return instance_path.removeprefix("$").lstrip(".") or "root"
    path = instance_path.removeprefix("$").lstrip(".")
    return path or "root"


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    """Validate tool call arguments against the tool's JSON schema, coercing primitives."""
    args = copy.deepcopy(tool_call.arguments)
    if not isinstance(args, dict):
        args = {} if args is None else args
    _normalize_optional_nulls(args, tool.parameters)
    args = _convert_value(args, tool.parameters)

    errors = collect_errors(args, tool.parameters)
    if not errors:
        return args

    coerced = _coerce_with_json_schema(copy.deepcopy(args), tool.parameters)
    coerced_errors = collect_errors(coerced, tool.parameters)
    if not coerced_errors:
        return coerced

    error_lines = "\n".join(
        f"  - {_format_validation_path(path, message)}: {message}" for path, message in errors
    )
    raise ValidationError(
        f'Validation failed for tool "{tool_call.name}":\n{error_lines}\n\n'
        f"Received arguments:\n{json.dumps(tool_call.arguments, indent=2)}"
    )
