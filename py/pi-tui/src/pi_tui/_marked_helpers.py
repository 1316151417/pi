"""HTML, URL, and JavaScript value operations used by Marked."""

from __future__ import annotations

import math
from urllib.parse import quote

from ._javascript import from_utf16_units
from ._marked_regex import Match
from ._marked_rules import other


def js_truthy(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return len(value) != 0
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    return True


def js_string(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            if value == 0:
                return "0"
            representation = repr(value)
            if "e" in representation:
                mantissa, exponent_text = representation.split("e")
                exponent = int(exponent_text)
                if 1e-6 <= abs(value) < 1e21:
                    sign = "-" if mantissa.startswith("-") else ""
                    mantissa = mantissa.lstrip("-")
                    integer, _, fraction = mantissa.partition(".")
                    digits = integer + fraction
                    point = len(integer) + exponent
                    if point <= 0:
                        return sign + "0." + "0" * -point + digits
                    if point >= len(digits):
                        return sign + digits + "0" * (point - len(digits))
                    return sign + digits[:point] + "." + digits[point:]
                return mantissa + "e" + ("+" if exponent >= 0 else "-") + str(abs(exponent))
            return representation.removesuffix(".0")
        return str(value)
    if isinstance(value, list):
        return ",".join("" if item is None else js_string(item) for item in value)
    if isinstance(value, dict):
        return "[object Object]"
    return str(value)


_ESCAPE_REPLACEMENTS = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}


def escape_html_entities(html: str, encode: bool = False) -> str:
    test = other["escapeTest" if encode else "escapeTestNoEncode"]
    if not test.test(html):
        return html
    def substitute(match: Match) -> str:
        return _ESCAPE_REPLACEMENTS[match[0]]
    return from_utf16_units(other["escapeReplace" if encode else "escapeReplaceNoEncode"].replace(html, substitute))


def clean_url(href: str) -> str | None:
    try:
        return quote(from_utf16_units(href), safe=";/?:@&=+$,#-_.!~*'()", errors="strict").replace("%25", "%")
    except (UnicodeError, ValueError):
        return None
