"""Validation of finite strict JSON values, matching chord/src/json.ts."""

import math


def is_json_value(value: object) -> bool:
    return _check(value, set(), 0)


def _check(value: object, ancestors: set[int], depth: int) -> bool:
    if depth > 512:
        return False
    if value is None or type(value) in (str, bool):
        return True
    if type(value) is int:
        # JavaScript numbers are IEEE-754 doubles, including JSON integer tokens.
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    if type(value) is float:
        return math.isfinite(value)
    if type(value) not in (list, dict):
        return False
    identity = id(value)
    if identity in ancestors:
        return False
    ancestors.add(identity)
    try:
        if isinstance(value, list):
            for item in value:
                if not _check(item, ancestors, depth + 1):
                    return False
        elif isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str or not _check(item, ancestors, depth + 1):
                    return False
        return True
    finally:
        ancestors.remove(identity)
