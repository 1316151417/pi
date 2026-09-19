"""Result helpers and tagged errors ported from ``harness/result.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

__all__ = [
    "Result",
    "ResultNamespace",
    "Result",
    "TaggedError",
    "TaggedErrorValue",
    "match_error",
    "LaneBusy",
    "OperationMismatch",
    "NoActiveRun",
    "NoActiveOperation",
    "NothingToResume",
    "NothingToCompact",
    "InvalidMessage",
    "InvalidNavigation",
    "UnknownSkill",
    "UnknownTemplate",
    "UnknownTarget",
    "InvalidLane",
    "Closed",
    "HarnessFault",
    "HarnessClosed",
]

TValue = TypeVar("TValue")
TError = TypeVar("TError")


@dataclass(frozen=True)
class Ok(Generic[TValue]):
    ok: bool = True
    value: Any = None


@dataclass(frozen=True)
class Err(Generic[TError]):
    ok: bool = False
    error: Any = None


#: Result of a fallible operation. Expected failures are returned as ``Err`` instead of raised.
Result = Any  # Ok | Err


class ResultNamespace:
    """Namespace mirroring the TS ``Result`` const object."""

    @staticmethod
    def ok(value: Any = None) -> Result:
        return Ok(value=value)

    @staticmethod
    def err(error: Any) -> Result:
        return Err(error=error)

    @staticmethod
    def is_ok(result: Result) -> bool:
        return bool(getattr(result, "ok", False))

    @staticmethod
    def is_err(result: Result) -> bool:
        return not bool(getattr(result, "ok", False))


def ok(value: Any = None) -> Result:
    """Create a successful :class:`Result`."""
    return Ok(value=value)


def err(error: Any) -> Result:
    """Create a failed :class:`Result`."""
    return Err(error=error)


def get_or_throw(result: Result) -> Any:
    """Return the success value or raise the failure error."""
    if not result.ok:
        raise result.error
    return result.value


def get_or_undefined(result: Result) -> Any:
    """Return the success value or ``None``."""
    return result.value if result.ok else None


def to_error(error: Any) -> Exception:
    """Normalize unknown thrown values into Exception instances."""
    if isinstance(error, Exception):
        return error
    if isinstance(error, BaseException):
        return Exception(str(error))
    if isinstance(error, str):
        return Exception(error)
    try:
        import json

        return Exception(json.dumps(error))
    except Exception:  # noqa: BLE001
        return Exception(str(error))


class TaggedError(Exception):
    """Error carrying a stable ``_tag`` for closed-set discrimination."""

    _tag: str = "TaggedError"

    def __init__(self, message: str = "", **props: Any) -> None:
        super().__init__(message)
        self.name = type(self)._tag
        for key, value in props.items():
            setattr(self, key, value)

    def to_json(self) -> dict:
        import json

        payload: dict = {"_tag": self._tag, "message": str(self)}
        for key, value in self.__dict__.items():
            if key == "_tag":
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)
        return payload


class LaneBusy(TaggedError):
    _tag = "LaneBusy"

    def __init__(self, lane: str, operation_id: str, operation_kind: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.operation_id = operation_id
        self.operation_kind = operation_kind


class OperationMismatch(TaggedError):
    _tag = "OperationMismatch"

    def __init__(
        self,
        lane: str,
        expected_operation_id: str,
        message: str,
        current_operation_id: str = None,
        last_operation_id: str = None,
    ) -> None:
        super().__init__(message)
        self.lane = lane
        self.expected_operation_id = expected_operation_id
        self.current_operation_id = current_operation_id
        self.last_operation_id = last_operation_id


class NoActiveRun(TaggedError):
    _tag = "NoActiveRun"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class NoActiveOperation(TaggedError):
    _tag = "NoActiveOperation"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class NothingToResume(TaggedError):
    _tag = "NothingToResume"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class NothingToCompact(TaggedError):
    _tag = "NothingToCompact"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class InvalidMessage(TaggedError):
    _tag = "InvalidMessage"

    def __init__(self, lane: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.reason = reason


class InvalidNavigation(TaggedError):
    _tag = "InvalidNavigation"

    def __init__(self, lane: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.reason = reason


class UnknownSkill(TaggedError):
    _tag = "UnknownSkill"

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.skill_name = name


class UnknownTemplate(TaggedError):
    _tag = "UnknownTemplate"

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.template_name = name


class UnknownTarget(TaggedError):
    _tag = "UnknownTarget"

    def __init__(self, target_id: str, message: str) -> None:
        super().__init__(message)
        self.target_id = target_id


class InvalidLane(TaggedError):
    _tag = "InvalidLane"

    def __init__(self, lane: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.reason = reason


class Closed(TaggedError):
    _tag = "Closed"


class HarnessFault(Exception):
    def __init__(self, message: str, cause: Any) -> None:
        super().__init__(message)
        self.name = "HarnessFault"
        self.cause = cause


class HarnessClosed(Exception):
    def __init__(self) -> None:
        super().__init__("AgentHarness was closed while the operation was active")
        self.name = "HarnessClosed"


def match_error(error: TaggedError, matchers: dict) -> Any:
    """Dispatch on ``error._tag`` to the matching handler."""
    matcher = matchers.get(error._tag)
    if matcher is None:
        raise KeyError(f"No matcher for tag {error._tag!r}")
    return matcher(error)
