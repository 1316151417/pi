"""Terminal image protocols, sizing, and capability detection from terminal-image.ts."""

from __future__ import annotations

import base64
import math
import os
import random
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypedDict

type ImageProtocol = Literal["kitty", "iterm2"] | None
type Number = int | float


@dataclass
class TerminalCapabilities:
    images: ImageProtocol
    true_color: bool
    hyperlinks: bool


class CapabilityOverrides(TypedDict, total=False):
    images: ImageProtocol
    true_color: bool
    hyperlinks: bool


@dataclass
class CellDimensions:
    width_px: Number
    height_px: Number


@dataclass
class ImageDimensions:
    width_px: Number
    height_px: Number


@dataclass
class ImageRenderOptions:
    max_width_cells: Number | None = None
    max_height_cells: Number | None = None
    preserve_aspect_ratio: bool | None = None
    image_id: Number | None = None
    move_cursor: bool | None = None


@dataclass
class KittyImageOptions:
    columns: Number | None = None
    rows: Number | None = None
    image_id: Number | None = None
    move_cursor: bool | None = None


@dataclass
class ITerm2ImageOptions:
    width: Number | str | None = None
    height: Number | str | None = None
    name: str | None = None
    preserve_aspect_ratio: bool | None = None
    inline: bool | None = None


@dataclass
class ImageCellSize:
    columns: Number
    rows: Number


@dataclass
class KittyImageMetadata(ImageCellSize):
    image_id: Number
    width_px: Number
    height_px: Number


@dataclass
class _RegisteredKittyImageMetadata(KittyImageMetadata):
    transmission_generation: int


@dataclass
class KittyImagePlacement:
    image_id: Number
    transmission_generation: int
    transmission_bytes: int
    estimated_decoded_bytes: Number
    sequence: str
    replacement_line: str


@dataclass
class RenderedImage(ImageCellSize):
    sequence: str
    image_id: Number | None = None


_cached_capabilities: TerminalCapabilities | None = None
_capability_overrides: CapabilityOverrides = {}
_cell_dimensions = CellDimensions(9, 18)
_KITTY_PREFIX = "\x1b_G"
_ITERM2_PREFIX = "\x1b]1337;File="
_KITTY_COMMAND = re.compile(r"\x1b_G([^;]*);")
_KITTY_PLACEMENT_CONTROL_KEYS = frozenset(("i", "p", "x", "y", "w", "h", "X", "Y", "c", "r", "C", "U", "z", "P", "Q", "H", "V"))
_kitty_image_metadata: dict[Number, _RegisteredKittyImageMetadata] = {}
_kitty_transmission_generation = 0


def _number(value: Number | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"
    text = repr(value)
    if 1e-6 <= abs(value) < 1e21:
        fixed = format(Decimal(text), "f")
        return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    if "e" in text:
        coefficient, exponent = text.split("e")
        return f"{coefficient.removesuffix('.0')}e{int(exponent):+d}"
    return text.removesuffix(".0")


def _floor(value: Number) -> Number:
    return math.floor(value) if math.isfinite(value) else value


def _ceil(value: Number) -> Number:
    return math.ceil(value) if math.isfinite(value) else value


def _minimum(*values: Number) -> Number:
    return math.nan if any(math.isnan(value) for value in values) else min(values)


def _maximum(*values: Number) -> Number:
    return math.nan if any(math.isnan(value) for value in values) else max(values)


def _divide(numerator: Number, denominator: Number) -> float:
    if denominator == 0:
        if numerator == 0 or math.isnan(numerator):
            return math.nan
        return math.copysign(math.inf, numerator) * math.copysign(1, denominator)
    return numerator / denominator


def _truthy_number(value: Number | None) -> bool:
    return value is not None and value != 0 and not math.isnan(value)


def _decode_base64(value: str) -> bytes:
    # Buffer.from accepts URL-safe symbols, ignores whitespace/invalid symbols,
    # stops at padding, and discards a final single sextet.
    cleaned = re.sub(r"[^A-Za-z0-9+/_-]", "", value.split("=", 1)[0])
    if len(cleaned) % 4 == 1:
        cleaned = cleaned[:-1]
    return base64.b64decode(cleaned + "=" * (-len(cleaned) % 4), altchars=b"-_")


def get_cell_dimensions() -> CellDimensions:
    return _cell_dimensions


def set_cell_dimensions(dimensions: CellDimensions) -> None:
    global _cell_dimensions
    _cell_dimensions = dimensions


def _probe_tmux_hyperlinks() -> bool:
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#{client_termfeatures}"],
            encoding="utf-8",
            errors="replace",
            timeout=0.25,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return "hyperlinks" in [feature.strip() for feature in result.stdout.split(",")]
    except Exception:
        return False


def _detect_capabilities_from_environment(tmux_forwards_hyperlink: Callable[[], bool]) -> TerminalCapabilities:
    program = os.environ.get("TERM_PROGRAM", "").lower()
    emulator = os.environ.get("TERMINAL_EMULATOR", "").lower()
    term = os.environ.get("TERM", "").lower()
    color_term = os.environ.get("COLORTERM", "").lower()
    true_color_hint = color_term in ("truecolor", "24bit")
    if os.environ.get("TMUX") or term.startswith("tmux"):
        return TerminalCapabilities(None, true_color_hint, tmux_forwards_hyperlink())
    if term.startswith("screen"):
        return TerminalCapabilities(None, true_color_hint, False)
    if os.environ.get("KITTY_WINDOW_ID") or program == "kitty":
        return TerminalCapabilities("kitty", True, True)
    if program == "ghostty" or "ghostty" in term or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return TerminalCapabilities("kitty", True, True)
    if os.environ.get("WEZTERM_PANE") or program == "wezterm":
        return TerminalCapabilities("kitty", True, True)
    if program == "warpterminal" or os.environ.get("WARP_SESSION_ID") or os.environ.get("WARP_TERMINAL_SESSION_UUID"):
        return TerminalCapabilities("kitty", True, True)
    if os.environ.get("ITERM_SESSION_ID") or program == "iterm.app":
        return TerminalCapabilities("iterm2", True, True)
    if os.environ.get("WT_SESSION"):
        return TerminalCapabilities(None, True, True)
    if program in ("alacritty", "vscode", "zed"):
        return TerminalCapabilities(None, True, True)
    if emulator == "jetbrains-jediterm":
        return TerminalCapabilities(None, True, False)
    if sys.platform == "win32":
        return TerminalCapabilities(None, True, False)
    return TerminalCapabilities(None, true_color_hint, False)


def _parse_boolean_capability_override(value: str | None) -> bool | None:
    return True if value == "1" else False if value == "0" else None


def detect_capabilities(tmux_forwards_hyperlink: Callable[[], bool] = _probe_tmux_hyperlinks) -> TerminalCapabilities:
    hyperlinks = _parse_boolean_capability_override(os.environ.get("PI_HYPERLINKS"))
    detected = _detect_capabilities_from_environment(tmux_forwards_hyperlink if hyperlinks is None else lambda: hyperlinks)
    image_protocol = os.environ.get("PI_IMAGE_PROTOCOL", "").lower()
    if image_protocol in ("kitty", "iterm2"):
        detected.images = "kitty" if image_protocol == "kitty" else "iterm2"
    elif image_protocol in ("none", "0"):
        detected.images = None
    true_color = _parse_boolean_capability_override(os.environ.get("PI_TRUE_COLOR"))
    if true_color is not None:
        detected.true_color = true_color
    if hyperlinks is not None:
        detected.hyperlinks = hyperlinks
    return detected


def get_capabilities() -> TerminalCapabilities:
    global _cached_capabilities
    if _cached_capabilities is None:
        hyperlinks = _capability_overrides.get("hyperlinks")
        detected = detect_capabilities() if hyperlinks is None else detect_capabilities(lambda: hyperlinks)
        if "images" in _capability_overrides:
            detected.images = _capability_overrides["images"]
        if "true_color" in _capability_overrides:
            detected.true_color = _capability_overrides["true_color"]
        if "hyperlinks" in _capability_overrides:
            detected.hyperlinks = _capability_overrides["hyperlinks"]
        _cached_capabilities = detected
    return _cached_capabilities


def reset_capabilities_cache() -> None:
    global _cached_capabilities
    _cached_capabilities = None


def set_capability_overrides(overrides: CapabilityOverrides) -> None:
    global _capability_overrides, _cached_capabilities
    if _capability_overrides == overrides:
        return
    _capability_overrides = overrides.copy()
    _cached_capabilities = None


def set_capabilities(capabilities: TerminalCapabilities) -> None:
    global _cached_capabilities
    _cached_capabilities = capabilities


def is_image_line(line: str) -> bool:
    return _KITTY_PREFIX in line or _ITERM2_PREFIX in line


def allocate_image_id() -> int:
    return math.floor(random.random() * 0xFFFFFFFE) + 1


def encode_kitty(base64_data: str, options: KittyImageOptions | None = None) -> str:
    options = options or KittyImageOptions()
    parameters = ["a=T", "f=100", "q=2"]
    if options.move_cursor is False:
        parameters.append("C=1")
    if options.columns is not None and _truthy_number(options.columns):
        parameters.append(f"c={_number(options.columns)}")
    if options.rows is not None and _truthy_number(options.rows):
        parameters.append(f"r={_number(options.rows)}")
    if options.image_id is not None and _truthy_number(options.image_id):
        parameters.append(f"i={_number(options.image_id)}")
    encoded = base64_data.encode("utf-16-le", errors="surrogatepass")
    if len(encoded) <= 8192:
        return f"{_KITTY_PREFIX}{','.join(parameters)};{base64_data}\x1b\\"
    chunks: list[str] = []
    for offset in range(0, len(encoded), 8192):
        chunk = encoded[offset:offset + 8192].decode("utf-16-le", errors="surrogatepass")
        if offset == 0:
            controls = f"{','.join(parameters)},m=1"
        else:
            controls = "m=0" if offset + 8192 >= len(encoded) else "m=1"
        chunks.append(f"{_KITTY_PREFIX}{controls};{chunk}\x1b\\")
    return "".join(chunks)


def delete_kitty_image(image_id: Number) -> str:
    return f"{_KITTY_PREFIX}a=d,d=I,i={_number(image_id)},q=2\x1b\\"


def delete_all_kitty_images() -> str:
    return f"{_KITTY_PREFIX}a=d,d=A,q=2\x1b\\"


def delete_all_kitty_placements() -> str:
    return f"{_KITTY_PREFIX}a=d,d=a,q=2\x1b\\"


def encode_iterm2(base64_data: str, options: ITerm2ImageOptions | None = None) -> str:
    options = options or ITerm2ImageOptions()
    length = len(base64_data.encode("utf-16-le", errors="surrogatepass")) // 2
    if length and base64_data.endswith("="):
        length -= 1
        if length and base64_data.endswith("=="):
            length -= 1
    parameters = [f"inline={0 if options.inline is False else 1}", f"size={length * 3 // 4}"]
    if options.width is not None:
        parameters.append(f"width={_number(options.width)}")
    if options.height is not None:
        parameters.append(f"height={_number(options.height)}")
    if options.name:
        name = options.name.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")
        parameters.append(f"name={base64.b64encode(name.encode('utf-8')).decode('ascii')}")
    if options.preserve_aspect_ratio is False:
        parameters.append("preserveAspectRatio=0")
    return f"{_ITERM2_PREFIX}{';'.join(parameters)}:{base64_data}\x07"


def register_kitty_image_metadata(metadata: KittyImageMetadata) -> None:
    global _kitty_transmission_generation
    _kitty_transmission_generation += 1
    _kitty_image_metadata.pop(metadata.image_id, None)
    _kitty_image_metadata[metadata.image_id] = _RegisteredKittyImageMetadata(
        columns=metadata.columns,
        rows=metadata.rows,
        image_id=metadata.image_id,
        width_px=metadata.width_px,
        height_px=metadata.height_px,
        transmission_generation=_kitty_transmission_generation,
    )
    if len(_kitty_image_metadata) > 1000:
        del _kitty_image_metadata[next(iter(_kitty_image_metadata))]


def _get_registered_kitty_image_metadata(line: str) -> _RegisteredKittyImageMetadata | None:
    match = _KITTY_COMMAND.search(line)
    if match is None or not match[1]:
        return None
    image_id = re.search(r"(?:^|,)i=([0-9]+)(?:,|\Z)", match[1])
    return None if image_id is None else _kitty_image_metadata.get(float(image_id[1]))


def get_kitty_image_metadata(line: str) -> KittyImageMetadata | None:
    metadata = _get_registered_kitty_image_metadata(line)
    if metadata is None:
        return None
    return KittyImageMetadata(metadata.columns, metadata.rows, metadata.image_id, metadata.width_px, metadata.height_px)


def get_kitty_image_placement(line: str) -> KittyImagePlacement | None:
    match = _KITTY_COMMAND.search(line)
    metadata = _get_registered_kitty_image_metadata(line)
    if match is None or metadata is None:
        return None
    command_start = match.start()
    command_controls = match[1]
    while True:
        terminator = line.find("\x1b\\", command_start + len(_KITTY_PREFIX))
        if terminator == -1:
            return None
        transmission_end = terminator + 2
        if re.search(r"(?:^|,)m=1(?:,|\Z)", command_controls) is None:
            break
        command_start = transmission_end
        if not line.startswith(_KITTY_PREFIX, command_start):
            return None
        controls_end = line.find(";", command_start + len(_KITTY_PREFIX))
        if controls_end == -1:
            return None
        command_controls = line[command_start + len(_KITTY_PREFIX):controls_end]
    controls = [control for control in match[1].split(",") if control.split("=", 1)[0] in _KITTY_PLACEMENT_CONTROL_KEYS]
    sequence = f"{_KITTY_PREFIX}a=p,q=2,{','.join(controls)}\x1b\\"
    return KittyImagePlacement(
        image_id=metadata.image_id,
        transmission_generation=metadata.transmission_generation,
        transmission_bytes=len(line[match.start():transmission_end].encode("utf-16-le", errors="surrogatepass")) // 2,
        estimated_decoded_bytes=metadata.width_px * metadata.height_px * 4,
        sequence=sequence,
        replacement_line=line[:match.start()] + sequence + line[transmission_end:],
    )


def crop_kitty_image_line(line: str, hidden_rows: Number, visible_rows: Number) -> str:
    metadata = get_kitty_image_metadata(line)
    match = _KITTY_COMMAND.search(line)
    if metadata is None or match is None or hidden_rows < 0 or hidden_rows >= metadata.rows or visible_rows <= 0:
        return line
    cropped_rows = _minimum(visible_rows, metadata.rows - hidden_rows)
    if hidden_rows == 0 and cropped_rows == metadata.rows:
        return line
    source_y = _floor(_divide(metadata.height_px * hidden_rows, metadata.rows))
    source_end = _ceil(_divide(metadata.height_px * (hidden_rows + cropped_rows), metadata.rows))
    source_height = _maximum(1, _minimum(metadata.height_px, source_end) - source_y)
    controls = [control for control in match[1].split(",") if re.match(r"^[yhr]=", control) is None]
    controls.extend((f"y={_number(source_y)}", f"h={_number(source_height)}", f"r={_number(cropped_rows)}"))
    return f"{line[:match.start()]}{_KITTY_PREFIX}{','.join(controls)};{line[match.end():]}"


def calculate_image_cell_size(
    image_dimensions: ImageDimensions,
    max_width_cells: Number,
    max_height_cells: Number | None = None,
    cell_dimensions: CellDimensions | None = None,
) -> ImageCellSize:
    cell_dimensions = cell_dimensions or CellDimensions(9, 18)
    max_width = _maximum(1, _floor(max_width_cells))
    max_height = None if max_height_cells is None else _maximum(1, _floor(max_height_cells))
    image_width = _maximum(1, image_dimensions.width_px)
    image_height = _maximum(1, image_dimensions.height_px)
    width_scale = _divide(max_width * cell_dimensions.width_px, image_width)
    height_scale = width_scale if max_height is None else _divide(max_height * cell_dimensions.height_px, image_height)
    scale = _minimum(width_scale, height_scale)
    columns = _ceil(_divide(image_width * scale, cell_dimensions.width_px))
    rows = _ceil(_divide(image_height * scale, cell_dimensions.height_px))
    return ImageCellSize(_maximum(1, _minimum(max_width, columns)), _maximum(1, rows if max_height is None else _minimum(max_height, rows)))


def calculate_image_rows(
    image_dimensions: ImageDimensions,
    target_width_cells: Number,
    cell_dimensions: CellDimensions | None = None,
) -> Number:
    return calculate_image_cell_size(image_dimensions, target_width_cells, None, cell_dimensions).rows


def get_png_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = _decode_base64(base64_data)
        if len(buffer) < 24 or buffer[:4] != b"\x89PNG":
            return None
        return ImageDimensions(int.from_bytes(buffer[16:20], "big"), int.from_bytes(buffer[20:24], "big"))
    except Exception:
        return None


def get_jpeg_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = _decode_base64(base64_data)
        if len(buffer) < 2 or buffer[:2] != b"\xff\xd8":
            return None
        offset = 2
        while offset < len(buffer) - 9:
            if buffer[offset] != 0xFF:
                offset += 1
                continue
            marker = buffer[offset + 1]
            if 0xC0 <= marker <= 0xC2:
                return ImageDimensions(int.from_bytes(buffer[offset + 7:offset + 9], "big"), int.from_bytes(buffer[offset + 5:offset + 7], "big"))
            if offset + 3 >= len(buffer):
                return None
            length = int.from_bytes(buffer[offset + 2:offset + 4], "big")
            if length < 2:
                return None
            offset += 2 + length
        return None
    except Exception:
        return None


def get_gif_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = _decode_base64(base64_data)
        if len(buffer) < 10:
            return None
        signature = bytes(value & 0x7F for value in buffer[:6])
        if signature not in (b"GIF87a", b"GIF89a"):
            return None
        return ImageDimensions(int.from_bytes(buffer[6:8], "little"), int.from_bytes(buffer[8:10], "little"))
    except Exception:
        return None


def get_webp_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = _decode_base64(base64_data)
        if len(buffer) < 30:
            return None
        riff = bytes(value & 0x7F for value in buffer[:4])
        webp = bytes(value & 0x7F for value in buffer[8:12])
        if riff != b"RIFF" or webp != b"WEBP":
            return None
        chunk = bytes(value & 0x7F for value in buffer[12:16])
        if chunk == b"VP8 ":
            return ImageDimensions(int.from_bytes(buffer[26:28], "little") & 0x3FFF, int.from_bytes(buffer[28:30], "little") & 0x3FFF)
        if chunk == b"VP8L":
            bits = int.from_bytes(buffer[21:25], "little")
            return ImageDimensions((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
        if chunk == b"VP8X":
            return ImageDimensions(int.from_bytes(buffer[24:27], "little") + 1, int.from_bytes(buffer[27:30], "little") + 1)
        return None
    except Exception:
        return None


def get_image_dimensions(base64_data: str, mime_type: str) -> ImageDimensions | None:
    if mime_type == "image/png":
        return get_png_dimensions(base64_data)
    if mime_type == "image/jpeg":
        return get_jpeg_dimensions(base64_data)
    if mime_type == "image/gif":
        return get_gif_dimensions(base64_data)
    if mime_type == "image/webp":
        return get_webp_dimensions(base64_data)
    return None


def render_image(
    base64_data: str,
    image_dimensions: ImageDimensions,
    options: ImageRenderOptions | None = None,
) -> RenderedImage | None:
    options = options or ImageRenderOptions()
    capabilities = get_capabilities()
    if not capabilities.images:
        return None
    max_width = 80 if options.max_width_cells is None else options.max_width_cells
    size = calculate_image_cell_size(image_dimensions, max_width, options.max_height_cells, get_cell_dimensions())
    if capabilities.images == "kitty":
        if options.image_id is not None:
            register_kitty_image_metadata(KittyImageMetadata(size.columns, size.rows, options.image_id, image_dimensions.width_px, image_dimensions.height_px))
        sequence = encode_kitty(base64_data, KittyImageOptions(size.columns, size.rows, options.image_id, options.move_cursor))
        return RenderedImage(size.columns, size.rows, sequence, options.image_id)
    if capabilities.images == "iterm2":
        sequence = encode_iterm2(base64_data, ITerm2ImageOptions(
            width=size.columns,
            height="auto",
            preserve_aspect_ratio=True if options.preserve_aspect_ratio is None else options.preserve_aspect_ratio,
        ))
        return RenderedImage(size.columns, size.rows, sequence)
    return None


def hyperlink(text: str, url: str) -> str:
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


def image_fallback(mime_type: str, dimensions: ImageDimensions | None = None, filename: str | None = None) -> str:
    parts: list[str] = []
    if filename:
        home = os.path.expanduser("~")
        display = "~" + filename[len(home):] if home and (filename == home or filename.startswith((home + "/", home + "\\"))) else filename
        if get_capabilities().hyperlinks and os.path.isabs(filename):
            absolute = os.path.abspath(filename)
            if os.name != "nt":
                absolute = "/" + absolute.lstrip("/")
            url = Path(absolute).as_uri()
            if filename.endswith(("/", "\\") if os.name == "nt" else "/") and not url.endswith("/"):
                url += "/"
            parts.append(hyperlink(display, url))
        else:
            parts.append(display)
    parts.append(f"[{mime_type}]")
    if dimensions is not None:
        parts.append(f"{_number(dimensions.width_px)}x{_number(dimensions.height_px)}")
    return f"[Image: {' '.join(parts)}]"
