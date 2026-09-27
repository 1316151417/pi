"""TypeBox 1.3.27 Value.Convert for JSON-representable type schemas.

Plain JSON Schema has no ``~kind`` and is deliberately left alone. TypeBox's
non-JSON primitive kinds and deferred TypeScript type-programming actions are
not a Python schema-building API; conversions requiring those raise explicitly.

Remaining source branches: ``FromBigInt`` / BigInt literal conversions need a
distinct JS BigInt value model; ``FromIntersect -> Instantiate -> Call`` and
``InstantiateDeferred`` need the TypeBox deferred action engine (modifiers,
conditional/mapped types, intrinsics, index/key operations, interface/module,
generic calls and rest spreading). ``EvaluateDependent`` needs ExcludeOperation;
Extends comparisons involving constructor/function/infer/generic/deferred types
need their respective type-level inference implementations. Those inputs raise
UnsupportedTypeBoxSchema instead of being reported as successfully converted.
This module is not a complete port of TypeBox's public type-constructor API.
"""

from __future__ import annotations

import copy
import math
import re
from typing import cast

from ._javascript import javascript_string
from ._json_runtime import JS_WHITESPACE
from ._typebox_primitives import entries, is_number, number_from_string, strict_equal
from ._typebox_regexp import compile_pattern
from ._typebox_validator import check
from ._values import UNDEFINED


class UnsupportedTypeBoxSchema(TypeError):
    """A TypeBox type-programming operation has no JSON schema equivalent."""


def _kind(schema: object) -> object:
    return schema.get("~kind") if isinstance(schema, dict) else None


def _literal(value: object) -> dict[str, object]:
    typename = "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "string"
    return {"~kind": "Literal", "type": typename, "const": value}


def _try_number(value: object) -> object:
    if value is None or value is UNDEFINED:
        return 0
    if isinstance(value, bool):
        return int(value)
    if is_number(value):
        return value
    if isinstance(value, str):
        number = number_from_string(value)
        if math.isfinite(number):
            return number
        if value.lower() in ("false", "true"):
            return int(value.lower() == "true")
        # TryBigInt is used only after Number has failed, and only safe integers
        # can cross back into the JS Number domain.
        if compile_pattern(r"^-?(0|[1-9]\d*)n$", unicode=False).test(value):
            text = value[:-1]
            if not re.fullmatch(r"[+-]?[0-9]+", text.strip(JS_WHITESPACE)):
                raise ValueError(f"Cannot convert {text} to a BigInt")
            if len(text.lstrip("-+0")) < 17:
                integer = int(text)
                if abs(integer) <= 9007199254740991:
                    return float(integer)
    return value


def _primitive(kind: str, value: object) -> object:
    if kind in ("Number", "Integer"):
        number = _try_number(value)
        if kind == "Integer" and is_number(number):
            return math.copysign(0.0, number) if abs(number) < 1 else math.trunc(number)
        return number
    if kind == "Boolean":
        if value is None or value is UNDEFINED:
            return False
        if isinstance(value, bool):
            return value
        if is_number(value) and value in (0, 1):
            return value == 1
        if isinstance(value, str):
            if value.lower() in ("false", "true"):
                return value.lower() == "true"
            if value in ("0", "1"):
                return value == "1"
        return value
    if kind == "String":
        if value is UNDEFINED:
            return ""
        if value is None:
            return "null"
        return javascript_string(value) if isinstance(value, (str, bool)) or is_number(value) else value
    if kind in ("Null", "Undefined", "Void"):
        convertible = value is None or value is UNDEFINED or value is False or is_number(value) and value == 0
        if isinstance(value, str):
            convertible = value.lower() in ("undefined", "null") or value in ("", "0")
        return (None if kind == "Null" else UNDEFINED) if convertible else value
    if kind == "BigInt":
        raise UnsupportedTypeBoxSchema("TypeBox BigInt conversion produces a non-JSON JavaScript primitive")
    return value


def _template(pattern: str) -> dict[str, object]:
    sentinels = (r"-?(?:0|[1-9][0-9]*)n", ".*", r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", r"-?(?:0|[1-9][0-9]*)", "(?!)", "(", ")", "$", "|")

    def trim(index: int) -> int:
        while True:
            while index < len(pattern) and pattern[index] in JS_WHITESPACE:
                index += 1
            if pattern.startswith("/*", index):
                end = pattern.find("*/", index + 2)
                index = len(pattern) if end < 0 else end + 2
            elif pattern.startswith("//", index):
                end = pattern.find("\n", index + 2)
                index = len(pattern) if end < 0 else end
            else:
                return index

    def constant(token: str, index: int) -> int | None:
        index = trim(index)
        return index + len(token) if pattern.startswith(token, index) else None

    def base(index: int) -> tuple[dict[str, object], int] | None:
        for token, kind in zip(sentinels[:5], ("BigInt", "String", "Number", "Integer", "Never")):
            end = constant(token, index)
            if end is not None:
                return {"~kind": kind, "type": kind.lower()}, end
        start = constant("(", index)
        if start is not None:
            children, end = body(start)
            close = constant(")", end)
            if close is not None:
                return {"~kind": "Union", "anyOf": children}, close
        end = index
        while end < len(pattern) and not any(pattern.startswith(token, end) for token in sentinels):
            end += 1
        return (_literal(pattern[index:end]), end) if end > index and end < len(pattern) else None

    def body(index: int) -> tuple[list[dict[str, object]], int]:
        first = base(index)
        if first is None:
            return [], index
        child, end = first
        rest, end = body(end)
        result = [child, *rest]
        separator = constant("|", end)
        if separator is not None:
            after, end = body(separator)
            result.extend(after)
        return result, end

    start = constant("^", 0)
    if start is None:
        return {"~kind": "String", "type": "string"}
    types, end = body(start)
    if constant("$", end) is None or not types:
        return {"~kind": "String", "type": "string"}

    def values(schema: dict[str, object]) -> list[str] | None:
        if _kind(schema) == "Literal":
            return [javascript_string(schema["const"])]
        if _kind(schema) != "Union" or not schema.get("anyOf"):
            return None
        result: list[str] = []
        for child in cast(list[dict[str, object]], schema["anyOf"]):
            variants = values(child)
            if variants is None:
                return None
            result.extend(variants)
        return result

    variants = [""]
    for schema in types:
        choices = values(schema)
        if choices is None:
            return {"~kind": "String", "type": "string"}
        variants = [prefix + suffix for suffix in choices for prefix in variants]
    return _union([_literal(item) for item in variants])


def _extends(left: dict[str, object], right: dict[str, object]) -> bool:
    left, right = _evaluate(left), _evaluate(right)
    a, b = _kind(left), _kind(right)
    if a in ("Any", "Never") or b in ("Any", "Unknown"):
        return True
    if a == "Unknown":
        return b == "Unknown"
    if a == "Union":
        return all(_extends(child, right) for child in cast(list[dict[str, object]], left["anyOf"]))
    if b == "Union":
        return any(_extends(left, child) for child in cast(list[dict[str, object]], right["anyOf"]))
    if a == "Literal":
        if b == "Literal":
            return strict_equal(left.get("const"), right.get("const"))
        return b == {"boolean": "Boolean", "number": "Number", "string": "String"}.get(cast(str, left.get("type")))
    if a == "Integer" and b == "Number" or a == "Undefined" and b == "Void":
        return True
    if a == b and a in ("String", "Number", "Integer", "Boolean", "Null", "Undefined", "Void", "BigInt", "Symbol"):
        return True
    if a == "Array" and b == "Array":
        if "~immutable" in left and "~immutable" not in right:
            return False
        return _extends(cast(dict[str, object], left["items"]), cast(dict[str, object], right["items"]))
    if a == "Tuple" and b in ("Array", "Tuple"):
        children = cast(list[dict[str, object]], left["items"])
        if b == "Array":
            return all(_extends(child, cast(dict[str, object], right["items"])) for child in children)
        others = cast(list[dict[str, object]], right["items"])
        return len(children) == len(others) and all(_extends(child, other) for child, other in zip(children, others))
    if a in ("Object", "Record") and b == "Record":
        right_value = cast(dict[str, object], entries(right["patternProperties"])[0][1])
        children = entries(left["properties"]) if a == "Object" else entries(left["patternProperties"])[:1]
        return all(_extends(cast(dict[str, object], child), right_value) for _, child in children)
    if a == "Record" and b == "Object":
        return not right["properties"]
    if a == "Object" and b == "Object":
        properties = cast(dict[str, dict[str, object]], left["properties"])
        for name, other in entries(right["properties"]):
            other = cast(dict[str, object], other)
            if name not in properties:
                if "~optional" not in other:
                    return False
            elif not _extends(properties[name], other) or "~optional" in properties[name] and "~optional" not in other:
                return False
        return True
    supported = {None, "Array", "Tuple", "Record", "Object", "Literal", "Union", "Never", "Unknown", "Any", "String", "Number", "Integer", "Boolean", "Null", "Undefined", "Void", "BigInt", "Symbol", "Ref"}
    if a not in supported or b not in supported:
        raise UnsupportedTypeBoxSchema(f"TypeBox type-level comparison of {a!r} and {b!r} is not JSON schema validation")
    return False


def _union(types: list[dict[str, object]]) -> dict[str, object]:
    result: list[dict[str, object]] = []
    for schema in types:
        current = _evaluate(schema)
        kind = _kind(current)
        if kind in ("Any", "Unknown"):
            result = [current]
            break
        if kind == "Never":
            continue
        if kind == "Object":
            result.append(current)
            continue
        if any(_extends(current, previous) for previous in result):
            continue
        result = [previous for previous in result if not _extends(previous, current)]
        result.append(current)
    flattened: list[dict[str, object]] = []
    for current in result:
        flattened.extend(cast(list[dict[str, object]], current["anyOf"]) if _kind(current) == "Union" else [current])
    if len(flattened) == 1:
        return flattened[0]
    return {"~kind": "Union", "anyOf": flattened} if flattened else {"~kind": "Never", "not": {}}


def _narrow(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    a, b = _kind(left), _kind(right)
    if a in ("Never", "Any"):
        return left
    if a == "Unknown":
        return right
    if b in ("Never", "Any"):
        return right
    if b == "Unknown":
        return left
    if a in ("Object", "Tuple") or b in ("Object", "Tuple"):
        if a not in ("Object", "Tuple"):
            return right
        if b not in ("Object", "Tuple"):
            return left
        first = dict(entries(left["properties"])) if a == "Object" else dict(entries(left["items"]))
        second = dict(entries(right["properties"])) if b == "Object" else dict(entries(right["items"]))
        properties = dict(first)
        for name, child in second.items():
            if name not in properties:
                properties[name] = child
                continue
            before = cast(dict[str, object], properties[name])
            after = cast(dict[str, object], child)
            merged = dict(_intersection([before, after]))
            for modifier in ("~optional", "~readonly"):
                merged.pop(modifier, None)
                if modifier in before and modifier in after:
                    merged[modifier] = True
            properties[name] = merged
        required = [name for name, child in entries(properties) if not isinstance(child, dict) or "~optional" not in child]
        result: dict[str, object] = {"~kind": "Object", "type": "object", "properties": properties}
        if required:
            result["required"] = required
        return result
    left_inside, right_inside = _extends(left, right), _extends(right, left)
    return right if right_inside else left if left_inside else {"~kind": "Never", "not": {}}


def _intersection(types: list[dict[str, object]]) -> dict[str, object]:
    result: list[dict[str, object]] = []
    for schema in types:
        current = _evaluate(schema)
        variants = cast(list[dict[str, object]], current["anyOf"]) if _kind(current) == "Union" else [current]
        result = [_narrow(previous, choice) for choice in variants for previous in result] if result else list(variants)
    return _union(result)


def _evaluate(schema: dict[str, object]) -> dict[str, object]:
    kind = _kind(schema)
    if kind == "Enum":
        return _union([_literal(item) for item in cast(list[object], schema["enum"])])
    if kind == "TemplateLiteral":
        return _template(cast(str, schema["pattern"]))
    if kind == "Intersect":
        return _intersection(cast(list[dict[str, object]], schema["allOf"]))
    if kind == "Union":
        return _union(cast(list[dict[str, object]], schema["anyOf"]))
    if kind in ("Dependent", "Deferred", "Call", "Infer", "Generic", "Mapped"):
        raise UnsupportedTypeBoxSchema(f"TypeBox {kind} requires its TypeScript type-programming runtime")
    return schema


def _instantiate(schema: dict[str, object], context: dict[str, object], visited: tuple[str, ...] = ()) -> dict[str, object]:
    kind = _kind(schema)
    if kind == "Ref":
        ref = cast(str, schema["$ref"])
        if ref in visited or ref not in context:
            return schema
        result = dict(_instantiate(cast(dict[str, object], context[ref]), context, (*visited, ref)))
        for modifier in ("~optional", "~readonly", "~immutable"):
            if modifier in schema:
                result[modifier] = schema[modifier]
        return result
    if kind in ("Deferred", "Call", "Rest"):
        raise UnsupportedTypeBoxSchema(f"TypeBox {kind} requires its TypeScript type-programming runtime")
    result = dict(schema)
    if kind in ("Object", "Record"):
        key = "properties" if kind == "Object" else "patternProperties"
        result[key] = {name: _instantiate(cast(dict[str, object], child), context, visited) for name, child in entries(schema[key])}
        if kind == "Object":
            result.pop("required", None)
            required = [name for name, child in entries(result[key]) if not isinstance(child, dict) or "~optional" not in child]
            if required:
                result["required"] = required
    elif kind in ("Union", "Intersect", "Tuple"):
        key = {"Union": "anyOf", "Intersect": "allOf", "Tuple": "items"}[cast(str, kind)]
        result[key] = [_instantiate(child, context, visited) for child in cast(list[dict[str, object]], schema[key])]
    elif kind == "Array":
        result["items"] = _instantiate(cast(dict[str, object], schema["items"]), context, visited)
    return result


def convert(schema: object, value: object, context: dict[str, object] | None = None) -> object:
    if not isinstance(schema, dict):
        return value
    context = {} if context is None else context
    kind = _kind(schema)
    if kind in ("Number", "Integer", "Boolean", "String", "Null", "Undefined", "Void", "BigInt"):
        return _primitive(cast(str, kind), value)
    if kind == "Literal":
        if strict_equal(schema["const"], value):
            return value
        constant = schema["const"]
        primitive = "Boolean" if isinstance(constant, bool) else "Number" if is_number(constant) else "String" if isinstance(constant, str) else None
        if primitive is None:
            raise TypeError("Unreachable TypeBox literal kind")
        candidate = _primitive(primitive, value)
        return candidate if strict_equal(candidate, constant) else value
    if kind == "Array":
        return [convert(schema["items"], item, context) for item in (value if isinstance(value, list) else [value])]
    if kind == "Tuple" and isinstance(value, list):
        for index, child in enumerate(cast(list[object], schema["items"])[:len(value)]):
            value[index] = convert(child, value[index], context)
    elif kind in ("Object", "Record") and isinstance(value, dict):
        properties = schema["properties" if kind == "Object" else "patternProperties"]
        patterns = [(compile_pattern("^" + name + "$", unicode=False), child) for name, child in entries(properties)]
        keys = [name for name, _ in entries(value)]
        for pattern, child in patterns:
            for name in keys:
                optional_undefined = kind == "Object" and isinstance(child, dict) and "~optional" in child and value[name] is UNDEFINED
                if pattern.test(name) and not optional_undefined:
                    value[name] = convert(child, value[name], context)
        if isinstance(schema.get("additionalProperties"), (dict, list)):
            for pattern, _ in patterns:
                for name in keys:
                    if not pattern.test(name):
                        value[name] = convert(schema["additionalProperties"], value[name], context)
    elif kind == "Ref":
        ref = cast(str, schema["$ref"])
        if ref in context:
            return convert(context[ref], value, context)
    elif kind == "Cyclic":
        definitions = {**context, **dict(entries(schema["$defs"]))}
        return convert({"~kind": "Ref", "$ref": schema["$ref"]}, value, definitions)
    elif kind == "Union":
        variants = cast(list[object], schema["anyOf"])
        if any(check(child, value, context) for child in variants):
            return value
        candidates = [convert(child, copy.deepcopy(value), context) for child in variants]
        for candidate in candidates:
            if check(schema, candidate, context):
                return value if candidate is UNDEFINED else candidate
    elif kind in ("Enum", "TemplateLiteral", "Intersect"):
        evaluated = _evaluate(_instantiate(schema, context) if kind == "Intersect" else schema)
        return convert(evaluated, value, context)
    return value
