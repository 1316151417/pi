"""String operations whose indexing is observable in the TypeScript TUI."""

from __future__ import annotations

import math

JS_WHITESPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def utf16_units(value: str) -> str:
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(encoded[index] | (encoded[index + 1] << 8)) for index in range(0, len(encoded), 2))


def from_utf16_units(value: str) -> str:
    # Also accept scalars, since a replacement string may contain an astral code point.
    return value.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="surrogatepass")


def utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def utf16_slice(value: str, start: int, end: int | None = None) -> str:
    return from_utf16_units(utf16_units(value)[start:end])


def js_trim(value: str) -> str:
    return value.strip(JS_WHITESPACE)


def js_repeat(value: str, count: int | float) -> str:
    if math.isnan(count):
        return ""
    if not math.isfinite(count):
        raise ValueError("Invalid count value")
    repetitions = math.trunc(count)
    if repetitions < 0:
        raise ValueError("Invalid count value")
    return value * repetitions
