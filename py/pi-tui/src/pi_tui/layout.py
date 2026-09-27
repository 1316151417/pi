"""Viewport layout, clipping, image cropping, scrollbars and hit paths from ``layout.ts``."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from ._component import CURSOR_MARKER, Component
from ._javascript import utf16_length
from .components.scroll_view import ScrollView
from .components.stack import allocate_stack_sizes, visible_stack_entries
from .layout_node import LayoutViewport, ScrollLayoutNode, get_layout_node
from .terminal_image import crop_kitty_image_line, get_kitty_image_metadata, is_image_line
from .tui import composite_tui_line
from .utils import extract_ansi_code, get_active_background_ansi, get_grapheme_cell_range, slice_by_column, visible_width

_OSC133_ZONE_PREFIX = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")


@dataclass
class LayoutRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(eq=False)
class LayoutBox:
    component: Component
    rect: LayoutRect
    clip: LayoutRect
    children: list[LayoutBox]
    layer: int
    parent: LayoutBox | None = None
    lines: Sequence[str] | None = None
    line_offset: int | None = None
    scroll_view: ScrollView | None = None
    scroll_content_lines: Sequence[str] | None = None


@dataclass
class LayoutFrame:
    root: LayoutBox
    width: int
    height: int
    lines: list[str]
    primary_scroll_view: ScrollView | None = None


@dataclass
class ScrollbarGeometry:
    column: int
    track_top: int
    track_height: int
    thumb_top: int
    thumb_height: int
    max_scroll_top: int


@dataclass
class _LayoutContext:
    viewport: LayoutViewport
    request_render: Callable[[], None]
    render_cache: dict[int, tuple[Component, dict[int, list[str]]]] = field(default_factory=dict)
    primary_scroll_view: ScrollView | None = None


def _intersect(a: LayoutRect, b: LayoutRect) -> LayoutRect:
    x, y = max(a.x, b.x), max(a.y, b.y)
    right, bottom = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    return LayoutRect(x, y, max(0, right - x), max(0, bottom - y))


def _render_cached(context: _LayoutContext, component: Component, width: int) -> list[str]:
    safe_width = max(1, math.floor(width))
    entry = context.render_cache.get(id(component))
    if entry is None:
        widths: dict[int, list[str]] = {}
        context.render_cache[id(component)] = (component, widths)
    else:
        widths = entry[1]
    lines = widths.get(safe_width)
    if lines is None:
        lines = component.render(safe_width)
        widths[safe_width] = lines
    return lines


def _translate_box(box: LayoutBox, delta_y: int) -> None:
    box.rect.y += delta_y
    for child in box.children:
        _translate_box(child, delta_y)


def _update_clips(box: LayoutBox, parent_clip: LayoutRect) -> None:
    box.clip = _intersect(parent_clip, box.rect)
    for child in box.children:
        _update_clips(child, box.clip)


def _layout_component(
    context: _LayoutContext, component: Component, x: int, y: int,
    width: int, height: int | None, clip: LayoutRect,
) -> LayoutBox:
    safe_width = max(1, math.floor(width))
    node = get_layout_node(component)
    if node is None:
        lines = _render_cached(context, component, safe_width)
        allocated_height = len(lines) if height is None else max(0, math.floor(height))
        line_offset = 0
        if len(lines) > allocated_height and allocated_height > 0:
            cursor_line = next((index for index, line in enumerate(lines) if CURSOR_MARKER in line), -1)
            if cursor_line >= allocated_height:
                line_offset = cursor_line - allocated_height + 1
        rect = LayoutRect(x, y, safe_width, allocated_height)
        return LayoutBox(component, rect, _intersect(clip, rect), [], 0, lines=lines, line_offset=line_offset)
    if isinstance(node, ScrollLayoutNode):
        previous_scroll_top = node.state.scroll_top
        content_width = node.state.get_content_width(safe_width)
        child_box = _layout_component(context, node.component, x, y - previous_scroll_top, content_width, None, clip)
        content_height = child_box.rect.height
        viewport_height = content_height if height is None else max(0, math.floor(height))
        node.state.update_layout(content_height, viewport_height, context.request_render)
        _translate_box(child_box, previous_scroll_top - node.state.scroll_top)
        scroll_view = cast(ScrollView, node.state)
        if node.state.primary or context.primary_scroll_view is None:
            context.primary_scroll_view = scroll_view
        rect = LayoutRect(x, y, safe_width, viewport_height)
        child_clip = _intersect(clip, rect)
        box = LayoutBox(
            component, rect, child_clip, [child_box], 0, scroll_view=scroll_view,
            scroll_content_lines=_render_cached(context, node.component, content_width),
        )
        child_box.parent = box
        _update_clips(child_box, child_clip)
        return box
    entries = visible_stack_entries(node.entries, context.viewport)
    gap_total = max(0, len(entries) - 1) * node.gap
    if node.type == "vstack":
        intrinsic_heights = [entry.basis if isinstance(entry.basis, (int, float)) else len(_render_cached(context, entry.component, safe_width)) for entry in entries]
        sizes = allocate_stack_sizes(entries, intrinsic_heights, height, node.gap)
        natural_height = sum(sizes) + gap_total
        allocated_height = natural_height if height is None else max(0, math.floor(height))
        rect = LayoutRect(x, y, safe_width, allocated_height)
        box = LayoutBox(component, rect, _intersect(clip, rect), [], 0)
        child_y = y
        for index, entry in enumerate(entries):
            child = _layout_component(context, entry.component, x, child_y, safe_width, sizes[index], box.clip)
            child.parent = box
            box.children.append(child)
            child_y += sizes[index] + node.gap
        return box
    intrinsic_widths = [entry.basis if isinstance(entry.basis, (int, float)) else max((visible_width(line) for line in _render_cached(context, entry.component, safe_width)), default=0) for entry in entries]
    widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, node.gap)
    intrinsic_heights = [len(_render_cached(context, entry.component, max(1, widths[index]))) for index, entry in enumerate(entries)]
    allocated_height = max(intrinsic_heights, default=0) if height is None else max(0, height)
    rect = LayoutRect(x, y, safe_width, allocated_height)
    box = LayoutBox(component, rect, _intersect(clip, rect), [], 0)
    child_x = x
    for index, entry in enumerate(entries):
        natural_child_height = intrinsic_heights[index]
        child_height = allocated_height if node.align == "stretch" else min(allocated_height, natural_child_height)
        child_y = y
        if node.align == "center":
            child_y += math.floor((allocated_height - child_height) / 2)
        elif node.align == "end":
            child_y += allocated_height - child_height
        child_width = widths[index]
        if child_width == 0:
            child = LayoutBox(entry.component, LayoutRect(child_x, child_y, 0, child_height), LayoutRect(child_x, child_y, 0, 0), [], 0, parent=box)
        else:
            child = _layout_component(context, entry.component, child_x, child_y, child_width, child_height, box.clip)
            child.parent = box
        box.children.append(child)
        child_x += child_width + node.gap
    return box


def _replace_scrollbar_cell(line: str, column: int, total_width: int, replacement: str, preserve_target_background: bool) -> str:
    if is_image_line(line):
        return line
    grapheme_range = get_grapheme_cell_range(line, column)
    start = grapheme_range.start if grapheme_range else column
    end = grapheme_range.end if grapheme_range else column + 1
    before = slice_by_column(line, 0, start, True)
    target = slice_by_column(line, start, end - start, True)
    after = slice_by_column(line, end, max(0, total_width - end), True)
    target_prefix = ""
    target_index = 0
    while target_index < utf16_length(target):
        ansi = extract_ansi_code(target, target_index)
        if ansi is None:
            break
        target_prefix += ansi.code
        target_index += ansi.length
    before_padding = " " * max(0, start - visible_width(before))
    cell_padding_before = " " * max(0, column - start)
    cell_padding_after = " " * max(0, end - column - 1)
    target_style = "\x1b[0m\x1b]8;;\x07" + (get_active_background_ansi(target_prefix) if preserve_target_background else "")
    return before + before_padding + target_style + cell_padding_before + replacement + cell_padding_after + after


def get_scrollbar_geometry(box: LayoutBox, include_hidden_auto: bool = False) -> ScrollbarGeometry | None:
    scroll_view = box.scroll_view
    if scroll_view is None or box.rect.width <= 0 or box.rect.height <= 0:
        return None
    content_height = box.children[0].rect.height if box.children else len(box.scroll_content_lines) if box.scroll_content_lines is not None else 0
    track_height = box.rect.height
    reveal_hidden_auto = include_hidden_auto and getattr(scroll_view, "scrollbar", None) == "auto" and content_height > track_height
    if not getattr(scroll_view, "is_scrollbar_visible", False) and not reveal_hidden_auto:
        return None
    minimum_thumb_height = min(2, track_height)
    # An empty always-visible track divides by zero in JS and clamps Infinity
    # back to the track height.
    ratio = track_height * track_height / content_height if content_height != 0 else math.inf
    rounded = math.floor(ratio) + (1 if ratio - math.floor(ratio) >= 0.5 else 0) if math.isfinite(ratio) else ratio
    thumb_height = max(minimum_thumb_height, min(track_height, rounded))
    max_scroll_top = max(0, content_height - track_height)
    max_thumb_top = track_height - thumb_height
    thumb_ratio = 0 if max_scroll_top == 0 else scroll_view.scroll_top / max_scroll_top * max_thumb_top
    thumb_offset = math.floor(thumb_ratio) + (1 if thumb_ratio - math.floor(thumb_ratio) >= 0.5 else 0)
    column = box.rect.x + box.rect.width - 1
    if column < box.clip.x or column >= box.clip.x + box.clip.width:
        return None
    return ScrollbarGeometry(column, box.rect.y, track_height, box.rect.y + thumb_offset, cast(int, thumb_height), max_scroll_top)


def _paint_scrollbar(box: LayoutBox, screen: list[str], total_width: int) -> None:
    geometry = get_scrollbar_geometry(box)
    if geometry is None or box.scroll_view is None:
        return
    for offset in range(geometry.track_height):
        row = geometry.track_top + offset
        if row < box.clip.y or row >= box.clip.y + box.clip.height or row < 0 or row >= len(screen):
            continue
        is_thumb = geometry.thumb_top <= row < geometry.thumb_top + geometry.thumb_height
        replacement = box.scroll_view.scrollbar_thumb_style("█" if box.scroll_view.is_scrollbar_active else "┃") if is_thumb else box.scroll_view.scrollbar_track_style("│")
        screen[row] = _replace_scrollbar_cell(screen[row] if screen[row] is not None else "", geometry.column, total_width, replacement, box.scroll_view.scrollbar != "always")


def _paint_box(box: LayoutBox, screen: list[str], total_width: int) -> None:
    if box.lines is not None:
        offset = box.line_offset if box.line_offset is not None else 0
        first_row = max(box.rect.y, box.clip.y, 0)
        last_row = min(box.rect.y + box.rect.height, box.clip.y + box.clip.height, len(screen))
        for row in range(first_row, last_row):
            source_index = offset + row - box.rect.y
            if source_index < 0 or source_index >= len(box.lines):
                continue
            source_line = box.lines[source_index]
            if source_line is None:
                continue
            line = _OSC133_ZONE_PREFIX.sub("", source_line, count=1)
            metadata = get_kitty_image_metadata(line)
            if metadata is not None:
                clip_bottom = min(len(screen), box.clip.y + box.clip.height)
                visible_rows = min(metadata.rows, clip_bottom - row)
                if visible_rows < metadata.rows:
                    line = crop_kitty_image_line(line, 0, visible_rows)
            if box.rect.x == 0 and box.rect.width >= total_width and (is_image_line(line) or not screen[row]):
                screen[row] = line
            else:
                screen[row] = composite_tui_line(screen[row] if screen[row] is not None else "", line, box.rect.x, box.rect.width, total_width)
    for child in box.children:
        _paint_box(child, screen, total_width)
    if box.scroll_view is not None and box.scroll_content_lines is not None and box.scroll_view.scroll_top > 0 and box.rect.height > 0:
        for image_row in range(box.scroll_view.scroll_top - 1, -1, -1):
            image_line = box.scroll_content_lines[image_row] if image_row < len(box.scroll_content_lines) else ""
            metadata = get_kitty_image_metadata(image_line)
            if metadata is not None:
                hidden_rows = box.scroll_view.scroll_top - image_row
                if hidden_rows < metadata.rows:
                    visible_rows = min(box.rect.height, metadata.rows - hidden_rows)
                    cropped = crop_kitty_image_line(image_line, hidden_rows, visible_rows)
                    if box.rect.x == 0 and box.rect.width >= total_width:
                        # JS writes beyond the end extend the array with holes.
                        # None represents those undefined slots at this boundary.
                        if box.rect.y >= len(screen):
                            screen.extend([cast(str, None)] * (box.rect.y + 1 - len(screen)))
                        if box.rect.y >= 0:
                            screen[box.rect.y] = cropped
                break
            if image_line != "":
                break
    _paint_scrollbar(box, screen, total_width)


def render_layout_frame(root: Component, width: int, height: int, request_render: Callable[[], None]) -> LayoutFrame:
    safe_width, safe_height = max(1, math.floor(width)), max(1, math.floor(height))
    context = _LayoutContext(LayoutViewport(safe_width, safe_height), request_render)
    root_box = _layout_component(context, root, 0, 0, safe_width, safe_height, LayoutRect(0, 0, safe_width, safe_height))
    lines = [""] * safe_height
    _paint_box(root_box, lines, safe_width)
    return LayoutFrame(root_box, safe_width, safe_height, lines, context.primary_scroll_view)


def _contains_point(rect: LayoutRect, x: int, y: int) -> bool:
    return rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height


def get_layout_boxes_at(frame: LayoutFrame, x: int, y: int) -> list[LayoutBox]:
    result: list[tuple[LayoutBox, int]] = []

    def visit(box: LayoutBox, depth: int) -> None:
        if not _contains_point(box.clip, x, y):
            return
        result.append((box, depth))
        for child in box.children:
            visit(child, depth + 1)

    visit(frame.root, 0)
    result.sort(key=lambda item: (-item[0].layer, -item[1]))
    return [box for box, _ in result]


def get_scroll_view_box(frame: LayoutFrame, scroll_view: ScrollView) -> LayoutBox | None:
    def visit(box: LayoutBox) -> LayoutBox | None:
        if box.scroll_view is scroll_view:
            return box
        for child in box.children:
            match = visit(child)
            if match is not None:
                return match
        return None

    return visit(frame.root)


def get_scroll_views_at(frame: LayoutFrame, x: int, y: int) -> list[ScrollView]:
    result: list[tuple[ScrollView, int]] = []

    def visit(box: LayoutBox, depth: int) -> None:
        if not _contains_point(box.clip, x, y):
            return
        if box.scroll_view is not None and _contains_point(box.rect, x, y):
            result.append((box.scroll_view, depth))
        for child in box.children:
            visit(child, depth + 1)

    visit(frame.root, 0)
    result.sort(key=lambda item: -item[1])
    return [scroll_view for scroll_view, _ in result]


__all__ = [
    "LayoutRect", "LayoutBox", "LayoutFrame", "ScrollbarGeometry", "render_layout_frame",
    "get_scrollbar_geometry", "get_layout_boxes_at", "get_scroll_view_box", "get_scroll_views_at",
]
