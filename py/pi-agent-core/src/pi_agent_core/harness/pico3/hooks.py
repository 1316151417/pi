"""Hook runners ported from ``harness/pico3/hooks.ts``.

A hook runner resolves the registrations that apply to one task invocation and
folds handler results in registration order: a non-``None`` value is offered to
``on_value``, which may return ``True`` to stop the fold. A throwing handler is
reported and skipped unless the context is already aborted, in which case the
error propagates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from ..._chord.context import Context
from .types import AnyKind, HookApi, HookBinding, HookInfo, HookResult, Id, Namespace

__all__ = ["HookRegistration", "HookRunnerImpl", "create_hook_runners"]


@dataclass
class HookRegistration:
    namespace: Namespace
    kind: AnyKind
    handlers: Any
    conversation_id: Optional[Id] = None
    subtree: bool = False


class HookRunnerImpl:
    """One kind's hook runner: the registrations that apply, and the fold over them."""

    def __init__(
        self,
        registrations: Callable[[], Sequence[HookRegistration]],
        ancestors: Callable[[Id], Sequence[Id]],
        on_report: Callable[[Any], None],
        kind: AnyKind,
        info: HookInfo,
    ) -> None:
        self._registrations = registrations
        self._ancestors = ancestors
        self._on_report = on_report
        self._kind = kind
        self._info = info

    def handlers(self) -> List[HookBinding]:
        out: List[HookBinding] = []
        for registration in self._registrations():
            if registration.kind is not self._kind and registration.kind.name != self._kind.name:
                continue
            if registration.conversation_id is not None:
                if registration.conversation_id == self._info.conversation_id:
                    pass
                elif registration.subtree is True and registration.conversation_id in self._ancestors(
                    self._info.conversation_id
                ):
                    pass
                else:
                    continue
            api = HookApi(
                kind=self._kind.name,
                task_id=self._info.task_id,
                conversation_id=self._info.conversation_id,
            )
            out.append(
                HookBinding(
                    handlers=registration.handlers,
                    namespace=registration.namespace,
                    api=api,
                )
            )
        return out

    async def each(
        self,
        ctx: Context,
        fn: Callable[[Any, HookApi], HookResult],
        on_value: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        for binding in self.handlers():
            try:
                value = fn(binding.handlers, binding.api)
                if hasattr(value, "__await__"):
                    value = await value
                if value is not None and on_value is not None and on_value(value) is True:
                    return
            except Exception as error:  # noqa: BLE001 - reported and skipped
                if ctx.abort_signal is not None and ctx.abort_signal.aborted:
                    raise
                self._on_report(error)


def create_hook_runners(
    registrations: Callable[[], Sequence[HookRegistration]],
    ancestors: Callable[[Id], Sequence[Id]],
    on_report: Callable[[Any], None],
) -> Callable[[AnyKind, HookInfo], HookRunnerImpl]:
    """Build the per-kind hook runner factory used by the runtime."""

    def runner(kind: AnyKind, info: HookInfo) -> HookRunnerImpl:
        return HookRunnerImpl(registrations, ancestors, on_report, kind, info)

    return runner
