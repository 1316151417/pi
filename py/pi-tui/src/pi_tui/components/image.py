"""Terminal image component with Kitty ID reuse and render caching."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..terminal_image import (
    ImageDimensions, ImageRenderOptions, allocate_image_id, get_capabilities,
    get_cell_dimensions, get_image_dimensions, image_fallback, render_image,
)
from ..utils import truncate_to_width


@dataclass
class ImageTheme:
    fallback_color: Callable[[str], str]


@dataclass
class ImageOptions:
    max_width_cells: int | None = None
    max_height_cells: int | None = None
    filename: str | None = None
    image_id: int | None = None


class Image:
    def __init__(
        self, base64_data: str, mime_type: str, theme: ImageTheme,
        options: ImageOptions | None = None, dimensions: ImageDimensions | None = None,
    ) -> None:
        self._base64_data = base64_data
        self._mime_type = mime_type
        self._theme = theme
        self._options = options if options is not None else ImageOptions()
        self._dimensions = dimensions or get_image_dimensions(base64_data, mime_type) or ImageDimensions(800, 600)
        self._image_id = self._options.image_id
        self._cached_lines: list[str] | None = None
        self._cached_width: int | None = None

    def get_image_id(self) -> int | None:
        return self._image_id

    def invalidate(self) -> None:
        self._cached_lines = None
        self._cached_width = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_width == width:
            return self._cached_lines
        max_width = max(1, min(width - 2, 60 if self._options.max_width_cells is None else self._options.max_width_cells))
        cell = get_cell_dimensions()
        default_max_height = max(1, math.ceil(max_width * cell.width_px / cell.height_px))
        max_height = default_max_height if self._options.max_height_cells is None else self._options.max_height_cells
        caps = get_capabilities()
        result = None
        if caps.images:
            if caps.images == "kitty" and self._image_id is None:
                self._image_id = allocate_image_id()
            result = render_image(self._base64_data, self._dimensions, ImageRenderOptions(
                max_width_cells=max_width, max_height_cells=max_height, image_id=self._image_id, move_cursor=False,
            ))
        if result is not None:
            if result.image_id:
                self._image_id = result.image_id
            if caps.images == "kitty":
                lines = [result.sequence]
                index = 0
                while index < result.rows - 1:
                    lines.append("")
                    index += 1
            else:
                lines = []
                index = 0
                while index < result.rows - 1:
                    lines.append("")
                    index += 1
                row_offset = result.rows - 1
                move_up = f"\x1b[{row_offset}A" if row_offset > 0 else ""
                lines.append(move_up + result.sequence)
        else:
            fallback = image_fallback(self._mime_type, self._dimensions, self._options.filename)
            lines = [truncate_to_width(self._theme.fallback_color(fallback), width)]
        self._cached_lines = lines
        self._cached_width = width
        return lines
