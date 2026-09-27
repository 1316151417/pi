"""ECMAScript pattern adapter used by TypeBox 1.3.27's native Python port.

Patterns are parsed as JavaScript syntax before translation to regex VERSION0.
Unicode properties use Unicode 16.0 aliases. Explicitly unsupported matching
constructs raise UnsupportedRegexError after syntax validation: backreferences
to captures nested inside repeated atoms, backreferences participating in
lookbehind, scoped ignore-case modifiers, repeat counts above 2**32-1, and
legacy malformed control escapes. Native engine resource limits also surface
as explicit UnsupportedRegexError failures.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol, cast

import regex

from ._javascript import javascript_string
from ._values import UNDEFINED

_WHITESPACE = r"\x09-\x0d\x20\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_WORD = r"[A-Za-z0-9_]"
_CATEGORIES = frozenset("""
C Other Cc Control cntrl Cf Format Cn Unassigned Co Private_Use Cs Surrogate
L Letter LC Cased_Letter Ll Lowercase_Letter Lm Modifier_Letter Lo Other_Letter Lt Titlecase_Letter Lu Uppercase_Letter
M Mark Combining_Mark Mc Spacing_Mark Me Enclosing_Mark Mn Nonspacing_Mark
N Number Nd Decimal_Number digit Nl Letter_Number No Other_Number
P Punctuation punct Pc Connector_Punctuation Pd Dash_Punctuation Pe Close_Punctuation Pf Final_Punctuation Pi Initial_Punctuation Po Other_Punctuation Ps Open_Punctuation
S Symbol Sc Currency_Symbol Sk Modifier_Symbol Sm Math_Symbol So Other_Symbol
Z Separator Zl Line_Separator Zp Paragraph_Separator Zs Space_Separator
""".split())
_SCRIPTS = frozenset("""
Adlm Adlam Aghb Caucasian_Albanian Ahom Arab Arabic Armi Imperial_Aramaic Armn Armenian Avst Avestan
Bali Balinese Bamu Bamum Bass Bassa_Vah Batk Batak Beng Bengali Bhks Bhaiksuki Bopo Bopomofo Brah Brahmi Brai Braille Bugi Buginese Buhd Buhid
Cakm Chakma Cans Canadian_Aboriginal Cari Carian Cham Cher Cherokee Chrs Chorasmian Copt Coptic Qaac Cpmn Cypro_Minoan Cprt Cypriot Cyrl Cyrillic
Deva Devanagari Diak Dives_Akuru Dogr Dogra Dsrt Deseret Dupl Duployan Egyp Egyptian_Hieroglyphs Elba Elbasan Elym Elymaic Ethi Ethiopic
Gara Garay Geor Georgian Glag Glagolitic Gong Gunjala_Gondi Gonm Masaram_Gondi Goth Gothic Gran Grantha Grek Greek Gujr Gujarati Gukh Gurung_Khema Guru Gurmukhi
Hang Hangul Hani Han Hano Hanunoo Hatr Hatran Hebr Hebrew Hira Hiragana Hluw Anatolian_Hieroglyphs Hmng Pahawh_Hmong Hmnp Nyiakeng_Puachue_Hmong Hrkt Katakana_Or_Hiragana Hung Old_Hungarian
Ital Old_Italic Java Javanese Kali Kayah_Li Kana Katakana Kawi Khar Kharoshthi Khmr Khmer Khoj Khojki Kits Khitan_Small_Script Knda Kannada Krai Kirat_Rai Kthi Kaithi
Lana Tai_Tham Laoo Lao Latn Latin Lepc Lepcha Limb Limbu Lina Linear_A Linb Linear_B Lisu Lyci Lycian Lydi Lydian
Mahj Mahajani Maka Makasar Mand Mandaic Mani Manichaean Marc Marchen Medf Medefaidrin Mend Mende_Kikakui Merc Meroitic_Cursive Mero Meroitic_Hieroglyphs Mlym Malayalam Modi Mong Mongolian Mroo Mro Mtei Meetei_Mayek Mult Multani Mymr Myanmar
Nagm Nag_Mundari Nand Nandinagari Narb Old_North_Arabian Nbat Nabataean Newa Nkoo Nko Nshu Nushu Ogam Ogham Olck Ol_Chiki Onao Ol_Onal Orkh Old_Turkic Orya Oriya Osge Osage Osma Osmanya Ougr Old_Uyghur
Palm Palmyrene Pauc Pau_Cin_Hau Perm Old_Permic Phag Phags_Pa Phli Inscriptional_Pahlavi Phlp Psalter_Pahlavi Phnx Phoenician Plrd Miao Prti Inscriptional_Parthian
Rjng Rejang Rohg Hanifi_Rohingya Runr Runic Samr Samaritan Sarb Old_South_Arabian Saur Saurashtra Sgnw SignWriting Shaw Shavian Shrd Sharada Sidd Siddham Sind Khudawadi Sinh Sinhala Sogd Sogdian Sogo Old_Sogdian Sora Sora_Sompeng Soyo Soyombo Sund Sundanese Sunu Sunuwar Sylo Syloti_Nagri Syrc Syriac
Tagb Tagbanwa Takr Takri Tale Tai_Le Talu New_Tai_Lue Taml Tamil Tang Tangut Tavt Tai_Viet Telu Telugu Tfng Tifinagh Tglg Tagalog Thaa Thaana Thai Tibt Tibetan Tirh Tirhuta Tnsa Tangsa Todr Todhri Toto Tutg Tulu_Tigalari
Ugar Ugaritic Vaii Vai Vith Vithkuqi Wara Warang_Citi Wcho Wancho Xpeo Old_Persian Xsux Cuneiform Yezi Yezidi Yiii Yi Zanb Zanabazar_Square Zinh Inherited Qaai Zyyy Common Zzzz Unknown
""".split())
_BINARY = frozenset("""
ASCII Any Assigned AHex ASCII_Hex_Digit Alpha Alphabetic Bidi_C Bidi_Control Bidi_M Bidi_Mirrored Cased CI Case_Ignorable
CWCF Changes_When_Casefolded CWCM Changes_When_Casemapped CWKCF Changes_When_NFKC_Casefolded CWL Changes_When_Lowercased CWT Changes_When_Titlecased CWU Changes_When_Uppercased
Dash Dep Deprecated DI Default_Ignorable_Code_Point Dia Diacritic EBase Emoji_Modifier_Base EComp Emoji_Component EMod Emoji_Modifier Emoji EPres Emoji_Presentation Ext Extender ExtPict Extended_Pictographic
Gr_Base Grapheme_Base Gr_Ext Grapheme_Extend Hex Hex_Digit IDC ID_Continue Ideo Ideographic IDS ID_Start IDSB IDS_Binary_Operator IDST IDS_Trinary_Operator Join_C Join_Control LOE Logical_Order_Exception Lower Lowercase Math NChar Noncharacter_Code_Point
Pat_Syn Pattern_Syntax Pat_WS Pattern_White_Space QMark Quotation_Mark Radical RI Regional_Indicator SD Soft_Dotted STerm Sentence_Terminal Term Terminal_Punctuation UIdeo Unified_Ideograph Upper Uppercase VS Variation_Selector WSpace White_Space space XIDC XID_Continue XIDS XID_Start
""".split())


class RegexSyntaxError(ValueError):
    """The pattern is not syntactically valid ECMAScript RegExp source."""


class UnsupportedRegexError(NotImplementedError):
    """A valid JS pattern whose matching semantics need a different engine."""


class RegexMatcher(Protocol):
    def test(self, value: str) -> bool: ...


def scalar_text(value: str) -> str:
    return value.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="surrogatepass")


def _code_units(value: str) -> str:
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(encoded[index] | encoded[index + 1] << 8) for index in range(0, len(encoded), 2))


@dataclass
class _Part:
    text: str
    captures: set[int] = field(default_factory=set)
    own_capture: int | None = None
    assertion: str | None = None
    character: str | None = None


@dataclass
class _Reference:
    key: int | str
    count_at_reference: int
    open_captures: frozenset[int]
    behind: bool


class _Parser:
    def __init__(self, source: str, unicode: bool) -> None:
        self.source = scalar_text(source) if unicode else _code_units(source)
        self.unicode = unicode
        self.index = 0
        self.capture_count = 0
        self.names: dict[str, list[int]] = {}
        self.capture_paths: dict[int, dict[int, int]] = {}
        self.paths: dict[int, int] = {}
        self.branch_counter = 0
        self.open_captures: set[int] = set()
        self.behind = 0
        self.behind_captures: set[int] = set()
        self.repeated_nested: set[int] = set()
        self.references: list[_Reference] = []
        self.unsupported: set[str] = set()
        self.multiline = False
        self.dot_all = False
        self.total_captures = 0
        self.has_named_groups = False
        cursor, in_class = 0, False
        while cursor < len(self.source):
            char = self.source[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "[" and not in_class:
                in_class = True
            elif char == "]" and in_class:
                in_class = False
            elif char == "(" and not in_class:
                named = self.source.startswith("(?<", cursor) and not self.source.startswith(("(?<=", "(?<!"), cursor)
                if named or not self.source.startswith("(?", cursor):
                    self.total_captures += 1
                self.has_named_groups = self.has_named_groups or named
            cursor += 1

    def error(self, message: str) -> RegexSyntaxError:
        return RegexSyntaxError(f"Invalid regular expression: {message} at {self.index}")

    def expression(self) -> _Part:
        self.branch_counter += 1
        branch = self.branch_counter
        self.paths[branch] = 0
        alternatives: list[str] = []
        captures: set[int] = set()
        while True:
            parts: list[str] = []
            while self.index < len(self.source) and self.source[self.index] not in "|)":
                part = self.term()
                parts.append(part.text)
                captures.update(part.captures)
            alternatives.append("".join(parts))
            if self.index >= len(self.source) or self.source[self.index] != "|":
                break
            self.index += 1
            self.paths[branch] += 1
        del self.paths[branch]
        return _Part("|".join(alternatives), captures)

    def term(self) -> _Part:
        char = self.source[self.index]
        self.index += 1
        if char in "*+?":
            raise self.error("Nothing to repeat")
        if char == "(":
            atom = self.group()
        elif char == "[":
            atom = self.character_class()
        elif char == "\\":
            atom = self.escape(False)
        elif char == ".":
            atom = _Part(r"[\s\S]" if self.dot_all else r"[^\n\r\u2028\u2029]")
        elif char == "^":
            atom = _Part(r"(?:\A|(?<=[\r\n\u2028\u2029]))" if self.multiline else r"\A", assertion="anchor")
        elif char == "$":
            atom = _Part(r"(?:\Z|(?=[\r\n\u2028\u2029]))" if self.multiline else r"\Z", assertion="anchor")
        elif char in "{}]" and self.unicode:
            raise self.error("Lone quantifier brackets")
        elif char == "{" and re.match(r"[0-9]+(?:,[0-9]*)?\}", self.source[self.index :]):
            raise self.error("Nothing to repeat")
        else:
            atom = _Part(regex.escape(char), character=char)
        if self.index == len(self.source):
            return atom
        quantifier = self.source[self.index]
        maximum: int | None
        if quantifier in "*+?":
            self.index += 1
            maximum = 1 if quantifier == "?" else None
        elif quantifier == "{":
            matched = re.match(r"\{([0-9]+)(?:,([0-9]*))?\}", self.source[self.index :])
            if matched is None:
                return atom
            minimum_text = matched[1].lstrip("0") or "0"
            if matched[2] is None:
                maximum_text = minimum_text
            elif matched[2]:
                maximum_text = matched[2].lstrip("0") or "0"
            else:
                maximum_text = None
            if maximum_text is not None and (len(minimum_text), minimum_text) > (len(maximum_text), maximum_text):
                raise self.error("numbers out of order in {} quantifier")
            minimum = int(minimum_text) if len(minimum_text) < 11 else 0x100000000
            maximum = (int(maximum_text) if len(maximum_text) < 11 else 0x100000000) if maximum_text is not None else None
            self.index += len(matched[0])
            quantifier = "{" + minimum_text + ("" if matched[2] is None else "," + (maximum_text or "")) + "}"
            if minimum > 0xFFFFFFFF or maximum is not None and maximum > 0xFFFFFFFF:
                self.unsupported.add("repeat counts above 2**32-1")
                quantifier = "{0}"
        else:
            return atom
        if atom.assertion is not None and (self.unicode or atom.assertion != "lookahead"):
            raise self.error("Invalid quantifier on assertion")
        if self.index < len(self.source) and self.source[self.index] == "?":
            self.index += 1
            quantifier += "?"
        if maximum is None or maximum > 1:
            self.repeated_nested.update(atom.captures - ({atom.own_capture} if atom.own_capture is not None else set()))
        return _Part(f"(?:{atom.text}){quantifier}", atom.captures)

    def group_name(self) -> str:
        start = self.index
        while self.index < len(self.source) and self.source[self.index] != ">":
            self.index += 1
        if self.index == len(self.source):
            raise self.error("Invalid capture group name")
        raw = self.source[start : self.index]
        self.index += 1

        def replace_escape(match: re.Match[str]) -> str:
            number = int(match[1] or match[2], 16)
            if number > 0x10FFFF:
                raise self.error("Invalid Unicode escape")
            return chr(number)

        name = scalar_text(re.sub(r"\\u(?:\{([0-9A-Fa-f]+)\}|([0-9A-Fa-f]{4}))", replace_escape, raw))
        if regex.fullmatch(r"[$_\p{ID_Start}][$_\u200c\u200d\p{ID_Continue}]*", name, flags=regex.VERSION0) is None:
            raise self.error("Invalid capture group name")
        return name

    def group(self) -> _Part:
        prefix = ""
        capture: int | None = None
        name: str | None = None
        assertion: str | None = None
        old_flags = self.multiline, self.dot_all
        if self.index < len(self.source) and self.source[self.index] == "?":
            self.index += 1
            if self.source.startswith(":", self.index):
                prefix, self.index = "?:", self.index + 1
            elif self.source.startswith(("=", "!"), self.index):
                prefix, self.index, assertion = "?" + self.source[self.index], self.index + 1, "lookahead"
            elif self.source.startswith(("<=", "<!"), self.index):
                prefix, self.index, assertion = "?" + self.source[self.index : self.index + 2], self.index + 2, "lookbehind"
                self.behind += 1
            elif self.source.startswith("<", self.index):
                self.index += 1
                name = self.group_name()
            else:
                flags = re.match(r"([ims]*)(?:-([ims]+))?:", self.source[self.index :])
                if flags is None or not flags[1] and not flags[2]:
                    raise self.error("Invalid group")
                enabled, disabled = flags[1], flags[2] or ""
                if len(set(enabled + disabled)) != len(enabled + disabled):
                    raise self.error("Invalid modifier flags")
                if "i" in enabled or "i" in disabled:
                    self.unsupported.add("scoped ignore-case modifiers")
                self.multiline = "m" in enabled or self.multiline and "m" not in disabled
                self.dot_all = "s" in enabled or self.dot_all and "s" not in disabled
                self.index += len(flags[0])
                prefix = "?:"
        if not prefix:
            self.capture_count += 1
            capture = self.capture_count
            self.capture_paths[capture] = dict(self.paths)
            self.open_captures.add(capture)
            if self.behind:
                self.behind_captures.add(capture)
            if name is not None:
                for previous in self.names.get(name, []):
                    previous_path = self.capture_paths[previous]
                    if not any(key in previous_path and previous_path[key] != value for key, value in self.paths.items()):
                        raise self.error("Duplicate capture group name")
                self.names.setdefault(name, []).append(capture)
            prefix = f"?P<_js{capture}>"
        inner = self.expression()
        if self.index == len(self.source) or self.source[self.index] != ")":
            raise self.error("Unterminated group")
        self.index += 1
        self.multiline, self.dot_all = old_flags
        if assertion == "lookbehind":
            self.behind -= 1
        if capture is not None:
            self.open_captures.remove(capture)
            inner.captures.add(capture)
        return _Part(f"({prefix}{inner.text})", inner.captures, capture, assertion)

    def escape(self, in_class: bool) -> _Part:
        if self.index == len(self.source):
            raise self.error("\\ at end of pattern")
        char = self.source[self.index]
        self.index += 1
        if char in "dDwWsS":
            content = "0-9" if char.lower() == "d" else "A-Za-z0-9_" if char.lower() == "w" else _WHITESPACE
            return _Part(f"[{'^' if char.isupper() else ''}{content}]")
        if char in "bB" and not in_class:
            boundary = f"(?:(?<={_WORD})(?!{_WORD})|(?<!{_WORD})(?={_WORD}))"
            return _Part(boundary if char == "b" else f"(?!{boundary})", assertion="boundary")
        if char in "fnrtv" or char == "b" and in_class:
            value = {"f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "b": "\b"}[char]
            return _Part(regex.escape(value), character=value)
        if char == "c":
            next_char = self.source[self.index : self.index + 1]
            if next_char and (next_char.isascii() and next_char.isalpha() or not self.unicode and in_class and next_char in "0123456789_"):
                self.index += 1
                value = chr(ord(next_char) % 32)
                return _Part(regex.escape(value), character=value)
            if self.unicode:
                raise self.error("Invalid Unicode escape")
            self.unsupported.add("legacy malformed control escapes")
            return _Part(r"\\c")
        if char in "xu":
            digits = 2 if char == "x" else 4
            matched = re.match(r"[0-9a-fA-F]{" + str(digits) + "}", self.source[self.index :])
            braced = char == "u" and self.unicode and self.source.startswith("{", self.index)
            if braced:
                matched = re.match(r"\{([0-9a-fA-F]+)\}", self.source[self.index :])
            if matched is None:
                if self.unicode:
                    raise self.error("Invalid Unicode escape")
                return _Part(char, character=char)
            codepoint = int(matched[1] if braced else matched[0], 16)
            if codepoint > 0x10FFFF:
                raise self.error("Invalid Unicode escape")
            self.index += len(matched[0])
            if self.unicode and not braced and 0xD800 <= codepoint <= 0xDBFF:
                following = re.match(r"\\u([dD][c-fC-F][0-9a-fA-F]{2})", self.source[self.index :])
                if following is not None:
                    codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + int(following[1], 16) - 0xDC00
                    self.index += len(following[0])
            value = chr(codepoint)
            return _Part(regex.escape(value), character=value)
        if char in "pP" and self.unicode:
            matched = re.match(r"\{([A-Za-z0-9_=]+)\}", self.source[self.index :])
            if matched is None:
                raise self.error("Invalid property name")
            prop = matched[1]
            if "=" in prop:
                name, value = prop.split("=", 1)
                valid = name in ("General_Category", "gc") and value in _CATEGORIES or name in ("Script", "sc", "Script_Extensions", "scx") and value in _SCRIPTS
            else:
                valid = prop in _CATEGORIES or prop in _BINARY
                if prop in _CATEGORIES:
                    prop = "General_Category=" + prop
            if not valid:
                raise self.error("Invalid property name")
            self.index += len(matched[0])
            return _Part(f"\\{char}{{{prop}}}")
        if char.isascii() and char.isdigit():
            digits = char
            while self.index < len(self.source) and self.source[self.index] in "0123456789":
                digits += self.source[self.index]
                self.index += 1
            if char == "0" and digits == "0":
                return _Part(r"\x00", character="\0")
            limit = str(self.total_captures)
            number = int(digits) if (len(digits), digits) <= (len(limit), limit) else self.total_captures + 1
            if in_class or char == "0" or not self.unicode and number > self.total_captures:
                if self.unicode:
                    raise self.error("Invalid class escape" if in_class else "Invalid decimal escape")
                return self.legacy_decimal(digits)
            ref = _Reference(number, self.capture_count, frozenset(self.open_captures), bool(self.behind))
            self.references.append(ref)
            return _Part(f"(?P=__js_reference_{len(self.references) - 1})")
        if char == "k":
            if not self.unicode and not self.has_named_groups:
                return _Part("k", character="k")
            if in_class:
                raise self.error("Invalid class escape")
            if self.source.startswith("<", self.index):
                self.index += 1
                name = self.group_name()
                self.references.append(_Reference(name, self.capture_count, frozenset(self.open_captures), bool(self.behind)))
                return _Part(f"(?P=__js_reference_{len(self.references) - 1})")
            raise self.error("Invalid named reference")
        if self.unicode and char not in r"^$\.*+?()[]{}|/" and not (in_class and char == "-"):
            raise self.error("Invalid escape")
        return _Part(regex.escape(char), character=char)

    def legacy_decimal(self, digits: str) -> _Part:
        if digits[0] not in "01234567":
            self.index -= len(digits) - 1
            return _Part(digits[0], character=digits[0])
        maximum = 3 if digits[0] in "0123" else 2
        count = 1
        while count < min(maximum, len(digits)) and digits[count] in "01234567":
            count += 1
        value = chr(int(digits[:count], 8))
        self.index -= len(digits) - count
        return _Part(regex.escape(value), character=value)

    def character_class(self) -> _Part:
        negative = self.source.startswith("^", self.index)
        if negative:
            self.index += 1
        units: list[_Part] = []
        while self.index < len(self.source) and self.source[self.index] != "]":
            char = self.source[self.index]
            self.index += 1
            unit = self.escape(True) if char == "\\" else _Part(regex.escape(char), character=char)
            if self.index < len(self.source) - 1 and self.source[self.index] == "-" and self.source[self.index + 1] != "]":
                self.index += 1
                end_char = self.source[self.index]
                self.index += 1
                end = self.escape(True) if end_char == "\\" else _Part(regex.escape(end_char), character=end_char)
                if unit.character is None or end.character is None:
                    if self.unicode:
                        raise self.error("Invalid character class range")
                    units.extend((unit, _Part(r"\-", character="-"), end))
                else:
                    if ord(unit.character) > ord(end.character):
                        raise self.error("Range out of order in character class")
                    units.append(_Part(f"[{regex.escape(unit.character)}-{regex.escape(end.character)}]"))
            else:
                units.append(unit)
        if self.index == len(self.source):
            raise self.error("Unterminated character class")
        self.index += 1
        union = "(?:" + "|".join(unit.text for unit in units) + ")"
        if negative:
            return _Part(f"(?!{union})[\\s\\S]" if units else r"[\s\S]")
        return _Part(union if units else r"(?!)")

    def parse(self) -> str:
        result = self.expression().text
        if self.index != len(self.source):
            raise self.error("Unmatched ')'")
        for index, reference in enumerate(self.references):
            key = reference.key
            if isinstance(key, int):
                if key > self.capture_count:
                    raise self.error("Invalid escape")
                groups = [key]
            else:
                if key not in self.names:
                    raise self.error("Invalid named capture referenced")
                groups = self.names[key]
            if reference.behind or self.behind_captures.intersection(groups):
                self.unsupported.add("backreferences inside or to captures inside lookbehind")
            if self.repeated_nested.intersection(groups):
                self.unsupported.add("backreferences to captures nested inside repeated atoms")
            text = ""
            for group in reversed(groups):
                if group <= reference.count_at_reference and group not in reference.open_captures:
                    text = f"(?(_js{group})\\g<_js{group}>|{text})"
            result = result.replace(f"(?P=__js_reference_{index})", text)
        return result


@dataclass
class _Matcher:
    expression: regex.Pattern[str]
    unicode: bool

    def test(self, value: str) -> bool:
        return self.expression.search(scalar_text(value) if self.unicode else _code_units(value)) is not None


@dataclass
class _SearchMatcher:
    search: Callable[[str], object]

    def test(self, value: str) -> bool:
        return self.search(value) is not None


@lru_cache(maxsize=512)
def _compile(source: str, unicode: bool) -> RegexMatcher:
    parser = _Parser(source, unicode)
    translated = parser.parse()
    try:
        compiled = regex.compile(translated, flags=regex.VERSION0)
    except (regex.error, OverflowError) as error:
        raise UnsupportedRegexError(f"ECMAScript pattern cannot be represented by the native regex engine: {error}") from error
    if parser.unsupported:
        raise UnsupportedRegexError("Unsupported ECMAScript matching semantics: " + "; ".join(sorted(parser.unsupported)))
    return _Matcher(compiled, unicode)


def compile_pattern(pattern: object, unicode: bool = True) -> RegexMatcher:
    existing_test = getattr(pattern, "test", None)
    if callable(existing_test):
        return cast(RegexMatcher, pattern)
    existing_search = getattr(pattern, "search", None)
    if callable(existing_search):
        return _SearchMatcher(existing_search)
    return _compile("" if pattern is UNDEFINED else javascript_string(pattern), unicode)


__all__ = ["RegexMatcher", "RegexSyntaxError", "UnsupportedRegexError", "compile_pattern"]
