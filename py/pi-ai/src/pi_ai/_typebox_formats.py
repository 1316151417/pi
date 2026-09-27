"""TypeBox 1.3.27 default format registry, including its IDNA algorithms.

Source: the package's complete ``build/format`` tree. The deliberately partial
Bidi/context rules in that source are retained, rather than replaced with a
stricter IDNA library. Unicode property tables come from regex's Unicode 16.0;
NFC normalization uses the Python runtime's Unicode database.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

import regex
from ada_url import URL

from ._javascript import utf16_length
from ._typebox_regexp import RegexSyntaxError, UnsupportedRegexError, compile_pattern, scalar_text
from ._values import UNDEFINED, Undefined

_DATE = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})")
_TIME = re.compile(r"([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.[0-9]+)?(?:([Zz])|([+-])([0-9]{2}):([0-9]{2}))?")
_DURATION = re.compile(r"P(([0-9]+Y([0-9]+M([0-9]+D)?)?|[0-9]+M([0-9]+D)?|[0-9]+D)(T([0-9]+H([0-9]+M([0-9]+S)?)?|[0-9]+M([0-9]+S)?|[0-9]+S))?|T([0-9]+H([0-9]+M([0-9]+S)?)?|[0-9]+M([0-9]+S)?|[0-9]+S)|[0-9]+W)")
_EMAIL = re.compile(r'''(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[^"\\]|\\[\x20-\x7e])*")@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*|\[(?:IPv6:[a-f0-9:]+|(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?:\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3})\])''', re.I | re.ASCII)
_IDN_EMAIL = regex.compile(r'''(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~\u0080-\U0010ffff-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~\u0080-\U0010ffff-]+)*|"(?:[^"\\]|\\[^\r\n\u2028\u2029])*")@[\p{L}\p{N}](?:[\p{L}\p{N}-]{0,62})(?<!-)(?:\.[\p{L}\p{N}](?:[\p{L}\p{N}-]{0,62})(?<!-))*''', regex.I | regex.VERSION0)
_IPV4_COMPONENT = r"(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
_IPV4 = rf"(?:{_IPV4_COMPONENT}\.){{3}}{_IPV4_COMPONENT}"
_IPV6 = rf"(?:(?:(?:[0-9a-f]{{1,4}}:){{6}}|::(?:[0-9a-f]{{1,4}}:){{5}}|(?:[0-9a-f]{{1,4}})?::(?:[0-9a-f]{{1,4}}:){{4}}|(?:(?:[0-9a-f]{{1,4}}:)?[0-9a-f]{{1,4}})?::(?:[0-9a-f]{{1,4}}:){{3}}|(?:(?:[0-9a-f]{{1,4}}:){{0,2}}[0-9a-f]{{1,4}})?::(?:[0-9a-f]{{1,4}}:){{2}}|(?:(?:[0-9a-f]{{1,4}}:){{0,3}}[0-9a-f]{{1,4}})?::[0-9a-f]{{1,4}}:|(?:(?:[0-9a-f]{{1,4}}:){{0,4}}[0-9a-f]{{1,4}})?::)(?:[0-9a-f]{{1,4}}:[0-9a-f]{{1,4}}|{_IPV4})|(?:(?:[0-9a-f]{{1,4}}:){{0,5}}[0-9a-f]{{1,4}})?::[0-9a-f]{{1,4}}|(?:(?:[0-9a-f]{{1,4}}:){{0,6}}[0-9a-f]{{1,4}})?::)"
_IPV4_PATTERN = re.compile(_IPV4)
_IPV6_PATTERN = re.compile(_IPV6, re.I | re.ASCII)

# These productions are the repeated subexpressions of the source URI and
# URI-reference regexes, including their empty authorities and empty ports.
_PERCENT = r"%[0-9a-f]{2}"
_USERINFO = rf"(?:(?:[-a-z0-9._~!$&'()*+,;=:]|{_PERCENT})*@)?"
_HOST = rf"(?:\[(?:{_IPV6}|v[0-9a-f]+\.[-a-z0-9._~!$&'()*+,;=:]+)\]|{_IPV4}|(?:[-a-z0-9._~!$&'()*+,;=]|{_PERCENT})*)"
_AUTHORITY = rf"//{_USERINFO}{_HOST}(?::[0-9]*)?"
_PCHAR = rf"(?:[-a-z0-9._~!$&'()*+,;=:@]|{_PERCENT})"
_SEGMENT_NC = rf"(?:[-a-z0-9._~!$&'()*+,;=@]|{_PERCENT})"
_PATH_ABEMPTY = rf"(?:/{_PCHAR}*)*"
_PATH_ABSOLUTE = rf"/(?:{_PCHAR}+{_PATH_ABEMPTY})?"
_HIERARCHY = rf"(?:{_AUTHORITY}{_PATH_ABEMPTY}|{_PATH_ABSOLUTE}|{_PCHAR}+{_PATH_ABEMPTY})?"
_RELATIVE = rf"(?:{_AUTHORITY}{_PATH_ABEMPTY}|{_PATH_ABSOLUTE}|{_SEGMENT_NC}+{_PATH_ABEMPTY})?"
_SCHEME = r"[a-z][a-z0-9+\-.]*:"
_QUERY_FRAGMENT = rf"(?:\?(?:[-a-z0-9._~!$&'()*+,;=:@/?]|{_PERCENT})*)?(?:\#(?:[-a-z0-9._~!$&'()*+,;=:@/?]|{_PERCENT})*)?"
_URI = re.compile(_SCHEME + _HIERARCHY + _QUERY_FRAGMENT, re.I | re.ASCII)
_URI_REFERENCE = re.compile(rf"(?:{_SCHEME}{_HIERARCHY}|{_RELATIVE}){_QUERY_FRAGMENT}", re.I | re.ASCII)
_URI_TEMPLATE = re.compile(r'''(?:(?:[^\x00-\x20"<>%\\^`{|}\x7f]|%[0-9a-f]{2})|\{[+#./;?&=,!@|]?(?:[a-z0-9_]|%[0-9a-f]{2})+(?:\.(?:[a-z0-9_]|%[0-9a-f]{2})+)*(?::[1-9][0-9]{0,3}|\*)?(?:,(?:[a-z0-9_]|%[0-9a-f]{2})+(?:\.(?:[a-z0-9_]|%[0-9a-f]{2})+)*(?::[1-9][0-9]{0,3}|\*)?)*\})*''', re.I | re.ASCII)
_JSON_POINTER = re.compile(r"(?:/(?:[^~/]|~0|~1)*)*")
_JSON_POINTER_FRAGMENT = re.compile(r"#(?:/(?:[a-z0-9_\-.!$&'()*+,;:=@]|%[0-9a-f]{2}|~0|~1)*)*", re.I | re.ASCII)
_RELATIVE_JSON_POINTER = re.compile(r"(?:0|[1-9][0-9]*)(?:#|(?:/(?:[^~/]|~0|~1)*)*)")
_UUID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", re.I | re.ASCII)
_IRI_INVALID = re.compile(r"[\x00-\x20<>\^`{|}\\]")
_IRI_REFERENCE_INVALID = re.compile(r"[\x00-\x20\x7f\\]|%(?![0-9a-fA-F]{2})")
_BAD_PERCENT = re.compile(r"%(?![0-9a-fA-F]{2})")
_IPV_FUTURE = re.compile(r"\[[vV][0-9a-fA-F]+\.[^\]]+\]")
_MALFORMED_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*//")

_MARK = regex.compile(r"[\p{Mn}\p{Mc}\p{Me}]", regex.VERSION0)
_NONSPACING = regex.compile(r"\p{Mn}", regex.VERSION0)
_LETTER = regex.compile(r"\p{L}", regex.VERSION0)
_GREEK = regex.compile(r"\p{Script=Greek}", regex.VERSION0)
_HEBREW = regex.compile(r"\p{Script=Hebrew}", regex.VERSION0)
_JAPANESE = regex.compile(r"[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]", regex.VERSION0)
_ARABIC_LETTER = regex.compile(r"[\p{Script=Arabic}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Mandaic}]", regex.VERSION0)
_VIRAMA = frozenset((0x094D, 0x09CD, 0x0A4D, 0x0ACD, 0x0B4D, 0x0BCD, 0x0C4D, 0x0CCD, 0x0D3B, 0x0D3C, 0x0D4D, 0x0DCA, 0x1B44, 0x1BAA, 0x1BAB, 0xA9C0, 0x11046, 0x1107F, 0x110B9, 0x11133, 0x11134, 0x111C0, 0x11235, 0x1134D, 0x11442, 0x114C2, 0x115BF, 0x1163F, 0x116B6, 0x11C3F, 0x11D44, 0x11D45))
_DISALLOWED = frozenset((0x0640, 0x07FA, 0x302E, 0x302F, 0x3031, 0x3032, 0x3033, 0x3034, 0x3035, 0x303B))
_PERMITTED = regex.compile(r"[\p{L}\p{Nd}\p{Mn}\p{Mc}\-+.,:/\u00b7\u0375\u05f3\u05f4\u200c\u200d\u30fb\u00df\u03c2\u06fd\u06fe\u0f0b\u3007]", regex.VERSION0)
_IGNORED_MAPPING = re.compile(r"[\u00ad\u034f\u180b-\u180d\u200b\ufe00-\ufe0f\U000e0100-\U000e01ef]")


def is_date(value: str) -> bool:
    matched = _DATE.fullmatch(value)
    if matched is None:
        return False
    year, month, day = (int(part) for part in matched.groups())
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days = (0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return 1 <= month <= 12 and 1 <= day <= days[month]


def is_time(value: str, strict_time_zone: bool = True) -> bool:
    matched = _TIME.fullmatch(value)
    if matched is None or strict_time_zone and matched[4] is None and matched[5] is None:
        return False
    hour, minute, second = (int(matched[index]) for index in (1, 2, 3))
    if hour > 23 or minute > 59 or second > 60:
        return False
    zone_hour, zone_minute = int(matched[6] or 0), int(matched[7] or 0)
    if matched[5] is not None and (zone_hour > 23 or zone_minute > 59):
        return False
    if second < 60:
        return True
    sign = -1 if matched[5] == "-" else 1
    return (hour * 60 + minute - sign * (zone_hour * 60 + zone_minute)) % 1440 == 1439


def is_date_time(value: str) -> bool:
    parts = re.split("[Tt]", value)
    return len(parts) == 2 and is_date(parts[0]) and is_time(parts[1])


def is_duration(value: str) -> bool:
    return _DURATION.fullmatch(value) is not None


def is_email(value: str) -> bool:
    return _EMAIL.fullmatch(value) is not None


def is_idn_email(value: str) -> bool:
    return _IDN_EMAIL.fullmatch(unicodedata.normalize("NFC", scalar_text(value))) is not None


def is_ipv4(value: str) -> bool:
    return _IPV4_PATTERN.fullmatch(value) is not None


def is_ipv6(value: str) -> bool:
    return _IPV6_PATTERN.fullmatch(value) is not None


def is_uri(value: str) -> bool:
    return _URI.fullmatch(value) is not None


def is_uri_reference(value: str) -> bool:
    return _URI_REFERENCE.fullmatch(value) is not None


def is_uri_template(value: str) -> bool:
    return _URI_TEMPLATE.fullmatch(value) is not None


def is_json_pointer(value: str) -> bool:
    return _JSON_POINTER.fullmatch(value) is not None


def is_json_pointer_uri_fragment(value: str) -> bool:
    return _JSON_POINTER_FRAGMENT.fullmatch(value) is not None


def is_relative_json_pointer(value: str) -> bool:
    return _RELATIVE_JSON_POINTER.fullmatch(value) is not None


def is_uuid(value: str) -> bool:
    return _UUID.fullmatch(value) is not None


def _can_parse_url(value: str, base: str | None = None) -> bool:
    value = re.sub(r"[\ud800-\udfff]", "\ufffd", scalar_text(value))
    try:
        URL(value, base)
        return True
    except (ValueError, TypeError):
        return False


def is_url(value: str) -> bool:
    return _can_parse_url(value)


def is_iri(value: str) -> bool:
    if _IRI_INVALID.search(value) is not None or _BAD_PERCENT.search(value) is not None:
        return False
    narrowed = _IPV_FUTURE.sub("[::1]", value, count=1) if utf16_length(value) < 2048 else value
    return _can_parse_url(narrowed)


def is_iri_reference(value: str) -> bool:
    return _IRI_REFERENCE_INVALID.search(value) is None and _MALFORMED_SCHEME.search(value) is None and _can_parse_url(value, "http://example.com")


def is_regex(value: str) -> bool:
    try:
        compile_pattern(value)
        return True
    except UnsupportedRegexError:
        # Syntax was fully checked before determining native matching limits.
        return True
    except RegexSyntaxError:
        return False


def _adapt(delta: int, points: int, first: bool) -> int:
    if first:
        delta //= 700
    else:
        signed = delta & 0xFFFFFFFF
        delta = (signed - 0x100000000 if signed >= 0x80000000 else signed) >> 1
    delta += delta // points
    k = 0
    while delta > 455:
        delta //= 35
        k += 36
    return k + (36 * delta) // (delta + 38)


def _decode_puny(value: str) -> str:
    output: list[int] = []
    n, index, bias = 128, 0, 72
    delimiter = value.rfind("-")
    if delimiter > 0:
        for char in value[:delimiter]:
            if ord(char) >= 128:
                raise ValueError("Invalid punycode")
            output.append(ord(char))
    cursor = 0 if delimiter < 0 else delimiter + 1
    while cursor < len(value):
        old_index, weight, k = index, 1, 36
        while True:
            if cursor >= len(value):
                raise ValueError("Invalid punycode")
            char = value[cursor]
            cursor += 1
            if "a" <= char <= "z":
                digit = ord(char) - 0x61
            elif "0" <= char <= "9":
                digit = ord(char) - 0x30 + 26
            else:
                raise ValueError("Invalid punycode")
            index += digit * weight
            threshold = 1 if k <= bias else 26 if k >= bias + 26 else k - bias
            if digit < threshold:
                break
            weight *= 36 - threshold
            k += 36
        length = len(output) + 1
        bias = _adapt(index - old_index, length, old_index == 0)
        n += index // length
        if n > 0x10FFFF:
            raise ValueError("Invalid punycode code point")
        index %= length
        output.insert(index, n)
        index += 1
    return "".join(chr(codepoint) for codepoint in output)


def _encode_puny(value: str) -> str:
    basic = "".join(char for char in value if ord(char) < 128)
    result = [basic, "-"] if basic else []
    points = [ord(char) for char in scalar_text(value)]
    n, delta, bias, handled = 128, 0, 72, len(basic)
    while handled < len(points):
        minimum = min(point for point in points if point >= n)
        delta += (minimum - n) * (handled + 1)
        n = minimum
        for point in points:
            if point < n:
                delta += 1
            if point == n:
                q, k = delta, 36
                while True:
                    threshold = 1 if k <= bias else 26 if k >= bias + 26 else k - bias
                    if q < threshold:
                        break
                    digit = threshold + (q - threshold) % (36 - threshold)
                    result.append(chr(digit + 0x61) if digit < 26 else chr(digit - 26 + 0x30))
                    q = (q - threshold) // (36 - threshold)
                    k += 36
                result.append(chr(q + 0x61) if q < 26 else chr(q - 26 + 0x30))
                bias = _adapt(delta, handled + 1, handled == len(basic))
                delta = 0
                handled += 1
        delta += 1
        n += 1
    return "".join(result)


def _bidi_class(char: str) -> str:
    point = ord(char)
    if 0x30 <= point <= 0x39 or 0x06F0 <= point <= 0x06F9:
        return "EN"
    if 0x0660 <= point <= 0x0669:
        return "AN"
    if _NONSPACING.search(char) is not None:
        return "NSM"
    if _HEBREW.search(char) is not None:
        return "R"
    if _ARABIC_LETTER.search(char) is not None:
        return "AL"
    return "L" if _LETTER.search(char) is not None else "ON"


def _has_rtl(value: str) -> bool:
    return any(_bidi_class(char) in ("R", "AL", "AN") for char in scalar_text(value))


def _has_bidi_chars(value: str) -> bool:
    if value.lower().startswith("xn--"):
        try:
            return _has_rtl(_decode_puny(value[4:].lower()))
        except (ValueError, OverflowError, ZeroDivisionError):
            return False
    return _has_rtl(value)


def _bidi_rule(value: str) -> bool:
    rtl = False
    allowed = {"L", "EN", "ES", "CS", "ET", "ON", "BN", "NSM"}
    european = arabic = False
    for index, char in enumerate(scalar_text(value)):
        kind = _bidi_class(char)
        if index == 0:
            if kind not in ("L", "R", "AL"):
                return False
            rtl = kind in ("R", "AL")
            if rtl:
                allowed = {"R", "AL", "AN", "EN", "ES", "CS", "ET", "ON", "BN", "NSM"}
        if kind not in allowed:
            return False
        european = european or kind == "EN"
        arabic = arabic or kind == "AN"
    return not (rtl and european and arabic)


def _unicode_label(value: str) -> bool:
    value = scalar_text(value)
    if any(ord(char) >= 128 for char in value) and len(_encode_puny(value)) + 4 > 63:
        return False
    if _has_rtl(value) and not _bidi_rule(value):
        return False
    chars = list(value)
    if chars[0] == "-" or chars[-1] == "-" or "".join(chars[2:]).startswith("--"):
        return False
    if _MARK.search(chars[0]) is not None:
        return False
    japanese = False
    for index, char in enumerate(chars):
        point = ord(char)
        if point in _DISALLOWED or _PERMITTED.search(char) is None:
            return False
        japanese = japanese or _JAPANESE.search(char) is not None
        previous = ord(chars[index - 1]) if index > 0 else None
        following = ord(chars[index + 1]) if index + 1 < len(chars) else None
        if point == 0x00B7 and (previous != 0x006C or following != 0x006C):
            return False
        if point == 0x0375 and (not following or _GREEK.search(chars[index + 1]) is None):
            return False
        if point in (0x05F3, 0x05F4) and (not previous or _HEBREW.search(chars[index - 1]) is None):
            return False
        if point == 0x200C and (not previous or previous < 0x80 and previous not in _VIRAMA):
            return False
        if point == 0x200D and (not previous or previous not in _VIRAMA):
            return False
    return "\u30fb" not in value or japanese


def _puny_label(value: str) -> bool:
    if not value.lower().startswith("xn--"):
        return False
    try:
        body = value[4:].lower()
        if body.rfind("-") == 0:
            return False
        decoded = _decode_puny(body)
        if not any(ord(char) >= 128 for char in decoded):
            return False
        return _unicode_label(decoded)
    except (ValueError, OverflowError, ZeroDivisionError):
        return False


def is_hostname(value: str) -> bool:
    if not value or utf16_length(value) > 253 or value.endswith("."):
        return False
    for label in value.split("."):
        if not 0 < utf16_length(label) <= 63:
            return False
        ascii_label = not label.startswith("-") and not label.endswith("-") and label[2:4] != "--" and re.fullmatch(r"[a-zA-Z0-9-]*", label) is not None
        if not _puny_label(label) and not ascii_label:
            return False
    return True


def is_idn_hostname(value: str) -> bool:
    if not value or " " in value:
        return False
    normalized = re.sub(r"[\uff01-\uff5e]", lambda match: chr(ord(match[0]) - 0xFEE0), scalar_text(value))
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = _IGNORED_MAPPING.sub("", normalized)
    normalized = re.sub(r"[\u002e\u3002\uff0e\uff61]", ".", normalized)
    if utf16_length(normalized) > 253:
        return False
    labels = normalized.split(".")
    bidi = any(_has_bidi_chars(label) for label in labels)
    return all(
        0 < utf16_length(label) <= 63 and (_puny_label(label) or _unicode_label(label)) and (not bidi or _bidi_rule(label))
        for label in labels
    )


type FormatCheck = Callable[[str], bool | None | Undefined]
_registry: dict[str, FormatCheck] = {}


def format_clear() -> None:
    _registry.clear()


def format_entries() -> list[tuple[str, FormatCheck]]:
    return list(_registry.items())


def format_set(name: str, check: FormatCheck) -> None:
    _registry[name] = check


def format_has(name: str) -> bool:
    return name in _registry


def format_get(name: str) -> FormatCheck | Undefined:
    return _registry.get(name, UNDEFINED)


def format_test(name: str, value: str) -> bool:
    check = _registry.get(name)
    if check is None:
        return True
    result = check(value)
    return True if result is None or isinstance(result, Undefined) else result


def format_reset() -> None:
    format_clear()
    _registry.update({
        "date-time": is_date_time, "date": is_date, "duration": is_duration, "email": is_email,
        "hostname": is_hostname, "idn-email": is_idn_email, "idn-hostname": is_idn_hostname,
        "ipv4": is_ipv4, "ipv6": is_ipv6, "iri-reference": is_iri_reference, "iri": is_iri,
        "json-pointer-uri-fragment": is_json_pointer_uri_fragment, "json-pointer": is_json_pointer,
        "regex": is_regex, "relative-json-pointer": is_relative_json_pointer, "time": is_time,
        "uri-reference": is_uri_reference, "uri-template": is_uri_template, "uri": is_uri,
        "url": is_url, "uuid": is_uuid,
    })


format_reset()

__all__ = [
    "FormatCheck", "format_clear", "format_entries", "format_set", "format_has", "format_get",
    "format_test", "format_reset", "is_date_time", "is_date", "is_duration", "is_email",
    "is_hostname", "is_idn_email", "is_idn_hostname", "is_ipv4", "is_ipv6", "is_iri_reference",
    "is_iri", "is_json_pointer_uri_fragment", "is_json_pointer", "is_regex", "is_relative_json_pointer",
    "is_time", "is_uri_reference", "is_uri_template", "is_uri", "is_url", "is_uuid",
]
