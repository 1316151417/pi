"""Native TypeBox 1.3.27 Check/Errors machinery for JSON tool schemas.

The engine preserves the source's keyword guards, evaluation bookkeeping,
error ordering and eight-error limit. It does not execute generated JavaScript.
Schema dictionaries are cached by identity; callers should treat compiled
schemas as immutable, as with TypeBox's generated validators.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, cast

from ._javascript import javascript_string
from ._typebox_formats import format_test
from ._typebox_primitives import deep_equal, entries, get_property, grapheme_count, has_property, is_integer, is_number, is_schema, schema_keyword, value_hash
from ._typebox_refs import ReferenceStack
from ._typebox_regexp import compile_pattern
from ._values import UNDEFINED


@dataclass
class LocalizedError:
    keyword: str
    schema_path: str
    instance_path: str
    params: dict[str, object]

    @property
    def message(self) -> str:
        key, p = self.keyword, self.params
        if key in ("maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum"):
            return f"must be {p['comparison']} {javascript_string(p['limit'])}"
        if key in ("maxItems", "maxLength", "maxProperties", "minItems", "minLength", "minProperties"):
            unit = {"Items": "items", "Length": "characters", "Properties": "properties"}[key[3:]]
            relation = "more" if key.startswith("max") else "fewer"
            return f"must not have {relation} than {javascript_string(p['limit'])} {unit}"
        if key in ("dependencies", "dependentRequired"):
            names = ", ".join(cast(list[str], p["dependencies"]))
            return f"must have properties {names} when property {p['property']} is present"
        if key == "format":
            return f'must match format "{p["format"]}"'
        if key == "if":
            return f'must match "{p["failingKeyword"]}" schema'
        if key == "multipleOf":
            return f"must be multiple of {javascript_string(p['multipleOf'])}"
        if key == "pattern":
            return f'must match pattern "{p["pattern"]}"'
        if key == "propertyNames":
            return f"property names {', '.join(cast(list[str], p['propertyNames']))} are invalid"
        if key == "required":
            return f"must have required properties {', '.join(cast(list[str], p['requiredProperties']))}"
        if key == "type":
            typename = p["type"]
            return f"must be either {' or '.join(typename)}" if isinstance(typename, list) else f"must be {typename}"
        if key == "~refine":
            return javascript_string(p["message"])
        return {
            "additionalProperties": "must not have additional properties",
            "anyOf": "must match a schema in anyOf",
            "boolean": "schema is false",
            "const": "must be equal to constant",
            "contains": "must contain at least 1 valid item",
            "enum": "must be equal to one of the allowed values",
            "not": "must not be valid",
            "oneOf": "must match exactly one schema in oneOf",
            "unevaluatedItems": "must not have unevaluated items",
            "unevaluatedProperties": "must not have unevaluated properties",
            "uniqueItems": "must not have duplicate items",
        }.get(key, "an unknown validation error occurred")


@dataclass
class _Evaluation:
    indices: set[int] = field(default_factory=set)
    keys: set[str] = field(default_factory=set)


class _Context:
    def __init__(self) -> None:
        self.stack = [_Evaluation()]
        self.errors: list[LocalizedError] = []

    def merge(self, other: _Context) -> None:
        self.stack[-1].indices.update(other.stack[-1].indices)
        self.stack[-1].keys.update(other.stack[-1].keys)

    def append(self, error: LocalizedError) -> None:
        if len(self.errors) < 8:
            self.errors.append(error)


def _type(value: object, typename: str) -> bool:
    if typename == "object":
        return isinstance(value, dict)
    if typename == "array":
        return isinstance(value, list)
    if typename == "boolean":
        return isinstance(value, bool)
    if typename == "integer":
        return is_integer(value)
    if typename == "number":
        return is_number(value)
    if typename == "null":
        return value is None
    if typename == "string":
        return isinstance(value, str)
    if typename in ("undefined", "void"):
        return value is UNDEFINED
    if typename in ("bigint", "symbol"):
        return False  # These primitive kinds have no JSON representation.
    if typename == "constructor":
        return isinstance(value, type)
    if typename == "function":
        return callable(value)
    return True


def _has_unevaluated(schema: object) -> bool:
    return schema_keyword(schema, "unevaluatedItems") or schema_keyword(schema, "unevaluatedProperties") or any(
        _has_unevaluated(child) for _, child in entries(schema)
    )


def _properties_pattern(schema: dict[str, object]) -> str:
    patterns = [name for name, _ in entries(schema.get("patternProperties"))] if schema_keyword(schema, "patternProperties") else []
    if schema_keyword(schema, "properties"):
        patterns += ["^" + re.sub(r"([.*+?^${}()|\[\]\\])", r"\\\1", name) + "$" for name, _ in entries(schema["properties"])]
    return "(" + "|".join(patterns) + ")" if patterns else "(?!)"


_OBJECT_KEYS = ("required", "additionalProperties", "dependencies", "dependentRequired", "dependentSchemas", "patternProperties", "properties", "propertyNames", "minProperties", "maxProperties")
_ARRAY_KEYS = ("additionalItems", "contains", "items", "maxContains", "maxItems", "minContains", "minItems", "prefixItems", "uniqueItems")
_STRING_KEYS = ("maxLength", "minLength", "format", "pattern")
_NUMBER_KEYS = ("exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum", "multipleOf")
_GENERAL_KEYS = ("$ref", "$recursiveRef", "$dynamicRef", "const", "enum", "if", "not", "allOf", "anyOf", "oneOf")


class _Engine:
    def __init__(self, schema: object, compiled: bool, remotes: dict[str, object] | None = None) -> None:
        self.references = ReferenceStack(schema, remotes)
        self.compiled = compiled
        self.unevaluated = _has_unevaluated(schema)

    def evaluate(self, schema: object, value: object, context: _Context, collect: bool, schema_path: str = "#", path: str = "") -> bool:
        if collect and len(context.errors) >= 8:
            return False
        self.references.push(schema)
        try:
            if isinstance(schema, bool):
                if not schema and collect:
                    context.append(LocalizedError("boolean", schema_path, path, {}))
                return schema
            if not isinstance(schema, dict):
                return True
            keys = ["type"]
            if isinstance(value, dict):
                keys.extend(_OBJECT_KEYS)
            if isinstance(value, list):
                keys.extend(_ARRAY_KEYS)
            if isinstance(value, str):
                keys.extend(_STRING_KEYS)
            if is_number(value):
                keys.extend(_NUMBER_KEYS)
            keys.extend(_GENERAL_KEYS)
            if isinstance(value, list):
                keys.append("unevaluatedItems")
            if isinstance(value, (dict, list)):
                keys.append("unevaluatedProperties")
            valid = True
            for key in keys:
                if schema_keyword(schema, key):
                    passed = self.keyword(key, schema, value, context, collect, schema_path, path)
                    valid = passed and valid
                    if not collect and not passed:
                        return False
            if valid and schema_keyword(schema, "~refine"):
                valid = self.keyword("~refine", schema, value, context, collect, schema_path, path)
            return valid
        finally:
            self.references.pop(schema)

    def nested(self, schema: object, value: object, context: _Context, collect: bool, schema_path: str, path: str) -> bool:
        use_stack = collect or not self.compiled or self.unevaluated
        if use_stack:
            context.stack.append(_Evaluation())
        result = self.evaluate(schema, value, context, collect, schema_path, path)
        # The source intentionally short-circuits the Pop call on failure.
        if result and use_stack:
            context.stack.pop()
        return result

    def keyword(self, key: str, schema: dict[str, object], value: object, context: _Context, collect: bool, schema_path: str, path: str) -> bool:
        constraint = schema[key]

        def fail(params: dict[str, object] | None = None, keyword: str = key) -> bool:
            if collect:
                context.append(LocalizedError(keyword, schema_path, path, {} if params is None else params))
            return False

        if key == "type":
            variants = constraint if isinstance(constraint, list) else [constraint]
            return any(_type(value, cast(str, item)) for item in variants) or fail({"type": constraint})
        if key == "required":
            record = cast(dict[str, object], value)
            missing = [name for name in cast(list[str], constraint) if not has_property(record, name)]
            return not missing or fail({"requiredProperties": missing})
        if key in ("minProperties", "maxProperties", "minItems", "maxItems", "minLength", "maxLength"):
            size = grapheme_count(value) if isinstance(value, str) else len(cast(dict[str, object] | list[object], value))
            limit = cast(int | float, constraint)
            return (size >= limit if key.startswith("min") else size <= limit) or fail({"limit": constraint})
        if key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            number, limit = cast(int | float, value), cast(int | float, constraint)
            comparison = {"minimum": ">=", "maximum": "<=", "exclusiveMinimum": ">", "exclusiveMaximum": "<"}[key]
            passed = {">=": number >= limit, "<=": number <= limit, ">": number > limit, "<": number < limit}[comparison]
            return passed or fail({"comparison": comparison, "limit": limit})
        if key == "multipleOf":
            number, divisor = cast(int | float, value), cast(int | float, constraint)
            if divisor == 0:
                return fail({"multipleOf": divisor})
            inverse = 1 / divisor
            if is_integer(number) and is_integer(inverse):
                return True
            remainder = math.fmod(number, divisor)
            return min(abs(remainder), abs(remainder - divisor), abs(remainder + divisor)) < 1e-10 or fail({"multipleOf": divisor})
        if key == "format":
            return format_test(cast(str, constraint), cast(str, value)) or fail({"format": constraint})
        if key == "pattern":
            return compile_pattern(constraint).test(cast(str, value)) or fail({"pattern": constraint})
        if key in ("const", "enum"):
            passed = deep_equal(value, constraint) if key == "const" else any(deep_equal(value, item) for item in cast(list[object], constraint))
            return passed or fail({"allowedValue" if key == "const" else "allowedValues": constraint})
        if key == "properties":
            record = cast(dict[str, object], value)
            required = cast(list[str], schema["required"]) if schema_keyword(schema, "required") else []
            valid = True
            for name, child in entries(constraint):
                if not has_property(record, name):
                    continue
                item = get_property(record, name)
                passed = name not in required and item is UNDEFINED or self.nested(child, item, context, collect, f"{schema_path}/properties/{name}", f"{path}/{name}")
                if passed:
                    context.stack[-1].keys.add(name)
                valid = passed and valid
                if not collect and not passed:
                    break
            return valid
        if key in ("additionalProperties", "patternProperties"):
            valid = True
            bad: list[str] = []
            if key == "additionalProperties":
                if self.compiled and not collect:
                    if not self.unevaluated and (constraint is True or isinstance(constraint, dict) and not constraint):
                        return True
                    if constraint is False and schema_keyword(schema, "properties") and schema_keyword(schema, "required") and not schema_keyword(schema, "patternProperties") and len(entries(schema["properties"])) == len(cast(list[str], schema["required"])):
                        return len(cast(dict[str, object], value)) == len(cast(list[str], schema["required"]))
                matcher = compile_pattern(_properties_pattern(schema))
                candidates = [(name, item, constraint, f"{schema_path}/{key}") for name, item in entries(value) if not matcher.test(name)]
            else:
                candidates = []
                for pattern, child in entries(constraint):
                    matcher = compile_pattern(pattern)
                    candidates.extend((name, item, child, f"{schema_path}/{key}/{pattern}") for name, item in entries(value) if matcher.test(name))
            for name, item, child, child_path in candidates:
                passed = self.nested(child, item, context, collect, child_path, f"{path}/{name}")
                if passed:
                    context.stack[-1].keys.add(name)
                else:
                    bad.append(name)
                valid = passed and valid
                if not collect and not passed:
                    break
            if key == "additionalProperties" and not valid:
                return fail({"additionalProperties": bad})
            return valid
        if key in ("dependencies", "dependentRequired", "dependentSchemas"):
            record = cast(dict[str, object], value)
            valid = True
            for name, child in entries(constraint):
                if not has_property(record, name):
                    continue
                if isinstance(child, list):
                    passed = True
                    for dependency in child:
                        if not has_property(record, dependency):
                            fail({"property": name, "dependencies": child})
                            passed = False
                            if key == "dependencies" or not collect:
                                break
                else:
                    passed = self.evaluate(child, value, context, collect, f"{schema_path}/{key}/{name}", path)
                valid = passed and valid
                if not collect and not passed:
                    break
            return valid or not record
        if key == "propertyNames":
            bad = []
            for name, _ in entries(value):
                if not self.evaluate(constraint, name, context, collect, f"{schema_path}/propertyNames", path):
                    bad.append(name)
                    if not collect:
                        break
            return not bad or fail({"propertyNames": bad})
        if key in ("additionalItems", "items", "prefixItems"):
            array = cast(list[object], value)
            if key == "additionalItems":
                if not schema_keyword(schema, "items") or not isinstance(schema["items"], list):
                    return True
                indexed = [(index, constraint) for index in range(len(schema["items"]), len(array))]
            elif isinstance(constraint, list):
                indexed = list(enumerate(constraint[:len(array)]))
            else:
                start = len(cast(list[object], schema["prefixItems"])) if schema_keyword(schema, "prefixItems") else 0
                indexed = [(index, constraint) for index in range(start, len(array))]
            valid = True
            for index, child in indexed:
                child_path = f"{schema_path}/{key}/{index}" if isinstance(constraint, list) else f"{schema_path}/{key}"
                passed = self.nested(child, array[index], context, collect, child_path, f"{path}/{index}")
                if passed:
                    context.stack[-1].indices.add(index)
                valid = passed and valid
                if not passed and (not collect or key == "additionalItems"):
                    break
            return valid
        if key in ("contains", "minContains", "maxContains"):
            if key == "contains" and schema_keyword(schema, "minContains") and schema["minContains"] == 0:
                return True
            if key != "contains" and not schema_keyword(schema, "contains"):
                return True
            count = 0
            for index, item in enumerate(cast(list[object], value)):
                if self.evaluate(schema["contains"], item, context, False):
                    count += 1
                    if key != "maxContains":
                        context.stack[-1].indices.add(index)
                    if key == "contains" and self.compiled and not self.unevaluated and not collect:
                        break
            if key == "contains":
                return count > 0 or fail({"minContains": 1})
            if key == "minContains":
                return count >= cast(int | float, constraint) or fail({"minContains": constraint}, "contains")
            minimum = schema["minContains"] if schema_keyword(schema, "minContains") else 1
            return count <= cast(int | float, constraint) or fail({"minContains": minimum, "maxContains": constraint}, "contains")
        if key == "uniqueItems":
            if not constraint:
                return True
            seen: set[int] = set()
            duplicates: list[int] = []
            for index, item in enumerate(cast(list[object], value)):
                hashed = value_hash(item)
                if hashed in seen:
                    duplicates.append(index)
                    if not collect:
                        break
                seen.add(hashed)
            return not duplicates or fail({"duplicateItems": duplicates})
        if key in ("$ref", "$recursiveRef", "$dynamicRef"):
            target = self.references.ref(schema, key)
            if target is UNDEFINED or target is None:
                target = False
            if not is_schema(target):
                return False
            child_context = _Context() if key == "$ref" else context
            passed = self.evaluate(target, value, child_context, collect, "#", path)
            if key == "$ref":
                if passed:
                    context.merge(child_context)
                elif collect:
                    for error in child_context.errors:
                        context.append(error)
            return passed
        if key in ("allOf", "anyOf", "oneOf"):
            passed_contexts: list[_Context] = []
            failed_contexts: list[_Context] = []
            passing: list[int] = []
            variants = cast(list[object], constraint)
            for index, child in enumerate(variants):
                child_context = _Context()
                passed = self.evaluate(child, value, child_context, collect, f"{schema_path}/{key}/{index}", path)
                if passed:
                    passed_contexts.append(child_context)
                    passing.append(index)
                else:
                    failed_contexts.append(child_context)
                if not collect and self.compiled and not self.unevaluated and ((key == "allOf" and not passed) or (key == "anyOf" and passed)):
                    break
            valid = len(passing) == len(variants) if key == "allOf" else bool(passing) if key == "anyOf" else len(passing) == 1
            if valid:
                for child_context in passed_contexts:
                    context.merge(child_context)
                return True
            if collect and (key != "oneOf" or not passing):
                for child_context in failed_contexts:
                    for error in child_context.errors:
                        context.append(error)
            return False if key == "allOf" else fail({"passingSchemas": passing} if key == "oneOf" else {})
        if key == "if":
            then = schema["then"] if schema_keyword(schema, "then") else True
            otherwise = schema["else"] if schema_keyword(schema, "else") else True
            if not collect:
                condition = self.evaluate(constraint, value, context, False)
                return self.evaluate(then if condition else otherwise, value, context, False)
            true_context = _Context()
            condition = self.evaluate(constraint, value, true_context, True, f"{schema_path}/if", path)
            if condition:
                passed = self.evaluate(then, value, true_context, True, f"{schema_path}/then", path)
            else:
                passed = self.evaluate(otherwise, value, context, True, f"{schema_path}/else", path)
            if passed:
                context.merge(true_context)
                return True
            return fail({"failingKeyword": "then" if condition else "else"})
        if key == "not":
            child_context = _Context()
            passed = not self.evaluate(constraint, value, child_context, False)
            if passed and (collect or not self.compiled):
                context.merge(child_context)
            return passed or fail()
        if key in ("unevaluatedItems", "unevaluatedProperties"):
            indices = context.stack[-1].indices
            names = context.stack[-1].keys
            bad_items: list[int | str] = []
            candidates = list(enumerate(value)) if key == "unevaluatedItems" and isinstance(value, list) else entries(value)
            for name, item in candidates:
                seen = name in indices if key == "unevaluatedItems" else name in names
                child_context = _Context() if collect else context
                passed = seen or self.evaluate(constraint, item, child_context, collect, schema_path, path)
                if passed:
                    if key == "unevaluatedItems":
                        context.stack[-1].indices.add(cast(int, name))
                    elif not seen:
                        context.stack[-1].keys.add(cast(str, name))
                else:
                    bad_items.append(name)
                    if not collect:
                        break
            return not bad_items or fail({key: bad_items})
        if key == "~refine":
            valid = True
            for index, refinement in enumerate(cast(list[dict[str, object]], constraint)):
                passed = bool(cast(Callable[[object], object], refinement["check"])(value))
                if not passed:
                    message = cast(Callable[[object], object], refinement["error"])(value) if collect else ""
                    fail({"index": index, "message": message})
                valid = passed and valid
                if not collect and not passed:
                    break
            return valid
        raise AssertionError(f"Unhandled TypeBox keyword {key}")


def _prepare(schema: object, stack: ReferenceStack, visited: set[int]) -> None:
    if not isinstance(schema, dict) or id(schema) in visited:
        return
    visited.add(id(schema))
    stack.push(schema)
    try:
        if schema_keyword(schema, "pattern"):
            compile_pattern(schema["pattern"])
        if schema_keyword(schema, "patternProperties"):
            for pattern, _ in entries(schema["patternProperties"]):
                compile_pattern(pattern)
        if schema_keyword(schema, "additionalProperties"):
            additional = schema["additionalProperties"]
            ignored = not _has_unevaluated(stack.schema) and (additional is True or isinstance(additional, dict) and not additional)
            fast = additional is False and schema_keyword(schema, "properties") and schema_keyword(schema, "required") and not schema_keyword(schema, "patternProperties") and len(entries(schema["properties"])) == len(cast(list[str], schema["required"]))
            if not ignored and not fast:
                compile_pattern(_properties_pattern(schema))
        for keyword in ("$ref", "$recursiveRef", "$dynamicRef"):
            if schema_keyword(schema, keyword):
                _prepare(stack.ref(schema, keyword), stack, visited)
        for keyword in ("properties", "patternProperties", "dependencies", "dependentSchemas"):
            if schema_keyword(schema, keyword):
                for _, child in entries(schema[keyword]):
                    _prepare(child, stack, visited)
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems", "items"):
            if schema_keyword(schema, keyword):
                children = schema[keyword] if isinstance(schema[keyword], list) else [schema[keyword]]
                for child in children:
                    _prepare(child, stack, visited)
        for keyword in ("additionalItems", "additionalProperties", "contains", "if", "then", "else", "not", "propertyNames", "unevaluatedItems", "unevaluatedProperties"):
            if schema_keyword(schema, keyword):
                _prepare(schema[keyword], stack, visited)
    finally:
        stack.pop(schema)


class Validator:
    def __init__(self, schema: object) -> None:
        self.schema = schema
        self.compiled_schema = _snapshot(schema)
        _prepare(self.compiled_schema, ReferenceStack(self.compiled_schema), set())

    def check(self, value: object) -> bool:
        return _Engine(self.compiled_schema, True).evaluate(self.compiled_schema, value, _Context(), False)

    def errors(self, value: object) -> list[LocalizedError]:
        if self.check(value):
            return []
        context = _Context()
        _Engine(self.schema, False).evaluate(self.schema, value, context, True)
        return context.errors


def check(schema: object, value: object, context: dict[str, object] | None = None) -> bool:
    return _Engine(schema, False, context).evaluate(schema, value, _Context(), False)


def _snapshot(value: object, memo: dict[tuple[int, bool], object] | None = None, schema_position: bool = True) -> object:
    """Capture generated constants while retaining TypeBox's external objects."""
    memo = {} if memo is None else memo
    identity = id(value), schema_position
    if identity in memo:
        return memo[identity]
    if isinstance(value, dict):
        result: dict[str, object] = {}
        memo[identity] = result
        for key, child in entries(value):
            if schema_position and key in ("const", "pattern"):
                result[key] = child
            elif schema_position and key in ("enum", "~refine") and isinstance(child, list):
                result[key] = list(child)
            else:
                schema_map = schema_position and key in ("properties", "patternProperties", "dependencies", "dependentSchemas", "$defs", "definitions")
                result[key] = _snapshot(child, memo, not schema_map)
        return result
    if isinstance(value, list):
        array: list[object] = []
        memo[identity] = array
        array.extend(_snapshot(child, memo) for child in value)
        return array
    return value
