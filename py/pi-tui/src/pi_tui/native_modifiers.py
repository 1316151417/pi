"""Native modifier-state lookup with the source's optional-helper fallback."""

from .native_platform import ModifierKey, Undefined, get_native_platform_helper


def is_native_modifier_pressed(key: ModifierKey) -> bool:
    helper = get_native_platform_helper()
    if isinstance(helper, Undefined):
        return False
    inspect = getattr(helper, "is_modifier_pressed", None)
    if not callable(inspect):
        return False
    try:
        return inspect(key) is True
    except Exception:
        return False


__all__ = ["ModifierKey", "is_native_modifier_pressed"]
