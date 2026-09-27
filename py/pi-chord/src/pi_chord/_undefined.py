"""A shared JavaScript undefined value, distinct from JSON null."""

from __future__ import annotations


class Undefined:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNDEFINED"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> Undefined:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Undefined:
        return self


UNDEFINED = Undefined()
