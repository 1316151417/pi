"""Vertical stack rendering from ``components/v-stack.ts``."""

from ..layout_node import LayoutViewport
from .stack import MAX_SAFE_INTEGER, Stack, StackChild, StackEntry, StackEntryOptions, StackOptions, allocate_stack_sizes, visible_stack_entries


class VStack(Stack):
    _layout_type = "vstack"

    def render(self, width: int) -> list[str]:
        viewport = LayoutViewport(max(1, width), MAX_SAFE_INTEGER)
        entries = visible_stack_entries(self._entries, viewport)
        rendered = [entry.component.render(viewport.width) for entry in entries]
        sizes = allocate_stack_sizes(entries, [len(lines) for lines in rendered], None, self._gap)
        lines: list[str] = []
        for index in range(len(entries)):
            if index > 0:
                lines.extend([""] * self._gap)
            child_lines = rendered[index][:sizes[index]]
            lines.extend(child_lines)
            lines.extend([""] * max(0, sizes[index] - len(child_lines)))
        return lines


__all__ = ["VStack", "StackChild", "StackEntry", "StackEntryOptions", "StackOptions"]
