"""Public fixture contracts from ``telemetry/src/testing/types.ts``.

No conformance cases or test runner integration are included in this port yet.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from types import TracebackType
from typing import Protocol

from ..types import RecordedTelemetrySpan, TelemetryContext


class TelemetryAdapterFixture(Protocol):
    @property
    def context(self) -> TelemetryContext: ...

    async def get_spans(self) -> Sequence[RecordedTelemetrySpan]: ...

    async def __aenter__(self) -> TelemetryAdapterFixture: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


type TelemetryAdapterFixtureFactory = Callable[[], Awaitable[TelemetryAdapterFixture]]


class TelemetryAdapterConformanceCase(Protocol):
    @property
    def group(self) -> str: ...

    @property
    def name(self) -> str: ...

    async def run(self) -> None: ...
