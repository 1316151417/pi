"""OSC background colors and color-scheme reports from terminal-colors.ts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

type TerminalColorScheme = Literal["dark", "light"]


@dataclass
class RgbColor:
    r: int | float
    g: int | float
    b: int | float


_OSC11_PATTERN = re.compile(r"^\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)\Z", re.IGNORECASE)
_COLOR_SCHEME_PATTERN = re.compile(r"^(?:\x1b\[\?997;(1|2)n)+\Z")
_JS_WHITESPACE = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def _parse_osc_hex_channel(channel: str) -> int | float | None:
    if re.fullmatch(r"[0-9a-f]+", channel, re.IGNORECASE) is None:
        return None
    maximum = math.inf if len(channel) >= 256 else float(16 ** len(channel)) - 1
    try:
        number = float(int(channel, 16))
    except OverflowError:
        number = math.inf
    scaled = number / maximum * 255
    if not math.isfinite(scaled):
        return scaled
    lower = math.floor(scaled)
    return lower + (1 if scaled - lower >= 0.5 else 0)


def is_osc11_background_color_response(data: str) -> bool:
    return _OSC11_PATTERN.search(data) is not None


def parse_osc11_background_color(data: str) -> RgbColor | None:
    match = _OSC11_PATTERN.search(data)
    if match is None:
        return None
    value = match[1].strip(_JS_WHITESPACE)
    if value.startswith("#"):
        hexadecimal = value[1:]
        if re.fullmatch(r"[0-9a-f]{6}", hexadecimal, re.IGNORECASE):
            return RgbColor(int(hexadecimal[:2], 16), int(hexadecimal[2:4], 16), int(hexadecimal[4:6], 16))
        if re.fullmatch(r"[0-9a-f]{12}", hexadecimal, re.IGNORECASE):
            red = _parse_osc_hex_channel(hexadecimal[:4])
            green = _parse_osc_hex_channel(hexadecimal[4:8])
            blue = _parse_osc_hex_channel(hexadecimal[8:12])
            return RgbColor(red, green, blue) if red is not None and green is not None and blue is not None else None
        return None
    channels = re.sub(r"^rgba?:", "", value, flags=re.IGNORECASE).split("/")
    if len(channels) < 3:
        return None
    red = _parse_osc_hex_channel(channels[0])
    green = _parse_osc_hex_channel(channels[1])
    blue = _parse_osc_hex_channel(channels[2])
    return RgbColor(red, green, blue) if red is not None and green is not None and blue is not None else None


def parse_terminal_color_scheme_report(data: str) -> TerminalColorScheme | None:
    match = _COLOR_SCHEME_PATTERN.search(data)
    if match is None:
        return None
    return "light" if match[1] == "2" else "dark"
