"""Lossless JSON records with the Python model's snake_case attribute spelling."""

import re
from collections.abc import Mapping
from typing import cast

from pi_ai._javascript import javascript_json_parse, javascript_json_stringify
from pi_ai.types import UNDEFINED


class Record(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        key = re.sub(r"_([a-z0-9])", lambda match: match[1].upper(), name)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: object) -> None:
        key = re.sub(r"_([a-z0-9])", lambda match: match[1].upper(), name)
        self[key] = value

    def to_json(self) -> dict[str, object]:
        return dict(self)


def field(value: object, snake: str, camel: str | None = None) -> object:
    if isinstance(value, Mapping):
        return value.get(camel if camel is not None else snake, UNDEFINED)
    return getattr(value, snake, UNDEFINED)


def parse(text: str) -> object:
    def records(value: object) -> object:
        if isinstance(value, dict):
            return Record((key, records(item)) for key, item in value.items())
        if isinstance(value, list):
            return [records(item) for item in value]
        return value
    return records(javascript_json_parse(text))


def stringify(value: object) -> object:
    result = javascript_json_stringify(value)
    # JSON.stringify(undefined) is an invalid SQLite bind value, not SQL NULL.
    return UNDEFINED if result is None else result


def parse_record(text: str) -> Record:
    return cast(Record, parse(text))


def object_spread(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    if isinstance(value, str):
        units = value.encode("utf-16-le", "surrogatepass")
        return {str(index // 2): units[index:index + 2].decode("utf-16-le", "surrogatepass")
                for index in range(0, len(units), 2)}
    return {}
