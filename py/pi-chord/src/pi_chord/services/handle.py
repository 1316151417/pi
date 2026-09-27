"""Stable guarded service handles, ported from services/handle.ts."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .._undefined import UNDEFINED, Undefined


class ServiceSlot:
    def __init__(self, service_id: str, wrap_objects: bool) -> None:
        self._service_id = service_id
        self._wrap_objects = wrap_objects
        self._implementation: object | None = None

    def view(self, assert_access: Callable[[], None]) -> _ServiceView:
        return _ServiceView(self, assert_access, self._wrap_objects)

    def bind(self, implementation: object) -> None:
        self._implementation = implementation

    def unbind(self) -> None:
        self._implementation = None

    def resolve(self, property: str, assert_access: Callable[[], None]) -> object:
        assert_access()
        implementation = self._implementation
        if implementation is None:
            raise RuntimeError(f"Service {self._service_id} is disconnected")
        return _get_property(implementation, property)


class _ServiceView:
    def __init__(self, slot: ServiceSlot, assert_access: Callable[[], None], wrap_objects: bool) -> None:
        self._slot = slot
        self._assert_access = assert_access
        self._wrap_objects = wrap_objects
        self._members: dict[str, _ValueView] = {}

    def __getattr__(self, property: str) -> object:
        return self[property]

    def __getitem__(self, property: str) -> object:
        def resolve() -> object:
            return self._slot.resolve(property, self._assert_access)

        current = resolve()
        if not _is_object(current) or (not callable(current) and not self._wrap_objects):
            return current
        member = self._members.get(property)
        if member is None:
            member = _CallableValueView(resolve) if callable(current) else _ValueView(resolve)
            self._members[property] = member
        return member


class _ValueView:
    def __init__(self, resolve: Callable[[], object]) -> None:
        self._resolve = resolve
        self._children: dict[str, _CallableValueView] = {}

    def __getattr__(self, property: str) -> object:
        return self[property]

    def __getitem__(self, property: str) -> object:
        def resolve() -> object:
            parent = self._resolve()
            if not _is_object(parent):
                raise TypeError("Service member does not have properties")
            return _get_property(parent, property)

        current = resolve()
        if not callable(current):
            return current
        child = self._children.get(property)
        if child is None:
            child = _CallableValueView(resolve)
            self._children[property] = child
        return child


class _CallableValueView(_ValueView):
    def __call__(self, *args: object, **kwargs: object) -> object:
        value = self._resolve()
        if not callable(value):
            raise TypeError("Service member is not callable")
        # Python getattr already binds methods to their current implementation.
        return value(*args, **kwargs)


def _get_property(value: object, property: str) -> object:
    if isinstance(value, Mapping):
        return value.get(property, UNDEFINED)
    return getattr(value, property, UNDEFINED)


def _is_object(value: object) -> bool:
    return value is not None and not isinstance(value, (Undefined, str, bytes, bool, int, float, complex))
