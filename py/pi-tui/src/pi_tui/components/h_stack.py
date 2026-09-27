"""Horizontal stack rendering from ``components/h-stack.ts``."""

import math

from ..layout_node import LayoutViewport
from ..tui import composite_tui_line
from ..utils import visible_width
from .stack import MAX_SAFE_INTEGER, Stack, StackChild, StackEntry, StackEntryOptions, StackOptions, allocate_stack_sizes, visible_stack_entries


class HStack(Stack):
    _layout_type = "hstack"

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        viewport = LayoutViewport(safe_width, MAX_SAFE_INTEGER)
        entries = visible_stack_entries(self._entries, viewport)
        if not entries:
            return []
        intrinsic_widths = [max((visible_width(line) for line in entry.component.render(safe_width)), default=0) for entry in entries]
        widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, self._gap)
        rendered = [[] if widths[index] == 0 else entry.component.render(widths[index]) for index, entry in enumerate(entries)]
        height = max((len(lines) for lines in rendered), default=0)
        result = [""] * height
        x = 0
        for index, lines in enumerate(rendered):
            child_width = widths[index]
            offset = 0
            if self._align == "center":
                offset = math.floor((height - len(lines)) / 2)
            elif self._align == "end":
                offset = height - len(lines)
            for row, line in enumerate(lines):
                target = row + offset
                if 0 <= target < len(result):
                    result[target] = composite_tui_line(result[target], line, x, child_width, safe_width)
            x += child_width + self._gap
        return result


__all__ = ["HStack", "StackChild", "StackEntry", "StackEntryOptions", "StackOptions"]
