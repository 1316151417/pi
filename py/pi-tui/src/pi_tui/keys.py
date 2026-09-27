"""Legacy terminal, Kitty, and modifyOtherKeys input from ``tui/src/keys.ts``."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Literal

from ._javascript import utf16_length

type Letter = Literal[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
type Digit = Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
type SymbolKey = Literal[
    "`", "-", "=", "[", "]", "\\", ";", "'", ",", ".", "/", "!", "@", "#", "$", "%",
    "^", "&", "*", "(", ")", "_", "+", "|", "~", "{", "}", ":", "<", ">", "?",
]
type SpecialKey = Literal[
    "escape", "esc", "enter", "return", "tab", "space", "backspace", "delete", "insert", "clear",
    "home", "end", "pageUp", "pageDown", "up", "down", "left", "right",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
]
type BaseKey = Letter | Digit | SymbolKey | SpecialKey
type ModifierName = Literal["ctrl", "shift", "alt", "super"]
# Python has no template-literal string types; the wire identifiers stay strings.
type KeyId = str
type KeyEventType = Literal["press", "repeat", "release"]

__all__ = [
    "KeyId", "KeyEventType", "Key", "set_kitty_protocol_active", "is_kitty_protocol_active",
    "is_key_release", "is_key_repeat", "matches_key", "parse_key", "decode_kitty_printable", "decode_printable_key",
]

_kitty_protocol_active = False
_last_event_type: KeyEventType = "press"


def set_kitty_protocol_active(active: bool) -> None:
    global _kitty_protocol_active
    _kitty_protocol_active = active


def is_kitty_protocol_active() -> bool:
    return _kitty_protocol_active


class Key:
    escape = "escape"
    esc = "esc"
    enter = "enter"
    return_ = "return"
    tab = "tab"
    space = "space"
    backspace = "backspace"
    delete = "delete"
    insert = "insert"
    clear = "clear"
    home = "home"
    end = "end"
    page_up = "pageUp"
    page_down = "pageDown"
    up = "up"
    down = "down"
    left = "left"
    right = "right"
    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    f11 = "f11"
    f12 = "f12"

    backtick = "`"
    hyphen = "-"
    equals = "="
    leftbracket = "["
    rightbracket = "]"
    backslash = "\\"
    semicolon = ";"
    quote = "'"
    comma = ","
    period = "."
    slash = "/"
    exclamation = "!"
    at = "@"
    hash = "#"
    dollar = "$"
    percent = "%"
    caret = "^"
    ampersand = "&"
    asterisk = "*"
    leftparen = "("
    rightparen = ")"
    underscore = "_"
    plus = "+"
    pipe = "|"
    tilde = "~"
    leftbrace = "{"
    rightbrace = "}"
    colon = ":"
    lessthan = "<"
    greaterthan = ">"
    question = "?"

    @staticmethod
    def ctrl(key: BaseKey) -> KeyId:
        return f"ctrl+{key}"

    @staticmethod
    def shift(key: BaseKey) -> KeyId:
        return f"shift+{key}"

    @staticmethod
    def alt(key: BaseKey) -> KeyId:
        return f"alt+{key}"

    @staticmethod
    def super(key: BaseKey) -> KeyId:
        return f"super+{key}"

    @staticmethod
    def ctrl_shift(key: BaseKey) -> KeyId:
        return f"ctrl+shift+{key}"

    @staticmethod
    def shift_ctrl(key: BaseKey) -> KeyId:
        return f"shift+ctrl+{key}"

    @staticmethod
    def ctrl_alt(key: BaseKey) -> KeyId:
        return f"ctrl+alt+{key}"

    @staticmethod
    def alt_ctrl(key: BaseKey) -> KeyId:
        return f"alt+ctrl+{key}"

    @staticmethod
    def shift_alt(key: BaseKey) -> KeyId:
        return f"shift+alt+{key}"

    @staticmethod
    def alt_shift(key: BaseKey) -> KeyId:
        return f"alt+shift+{key}"

    @staticmethod
    def ctrl_super(key: BaseKey) -> KeyId:
        return f"ctrl+super+{key}"

    @staticmethod
    def super_ctrl(key: BaseKey) -> KeyId:
        return f"super+ctrl+{key}"

    @staticmethod
    def shift_super(key: BaseKey) -> KeyId:
        return f"shift+super+{key}"

    @staticmethod
    def super_shift(key: BaseKey) -> KeyId:
        return f"super+shift+{key}"

    @staticmethod
    def alt_super(key: BaseKey) -> KeyId:
        return f"alt+super+{key}"

    @staticmethod
    def super_alt(key: BaseKey) -> KeyId:
        return f"super+alt+{key}"

    @staticmethod
    def ctrl_shift_alt(key: BaseKey) -> KeyId:
        return f"ctrl+shift+alt+{key}"

    @staticmethod
    def ctrl_shift_super(key: BaseKey) -> KeyId:
        return f"ctrl+shift+super+{key}"


_SYMBOL_KEYS = set("`-=[]\\;',./!@#$%^&*()_+|~{}:<>?")
_SHIFT = 1
_ALT = 2
_CTRL = 4
_SUPER = 8
_LOCK_MASK = 64 + 128
_CODEPOINTS = {"escape": 27, "tab": 9, "enter": 13, "space": 32, "backspace": 127, "kpEnter": 57414}
_ARROW_CODEPOINTS = {"up": -1, "down": -2, "right": -3, "left": -4}
_FUNCTIONAL_CODEPOINTS = {"delete": -10, "insert": -11, "pageUp": -12, "pageDown": -13, "home": -14, "end": -15}
_KITTY_FUNCTIONAL_KEY_EQUIVALENTS = {
    57399: 48, 57400: 49, 57401: 50, 57402: 51, 57403: 52, 57404: 53,
    57405: 54, 57406: 55, 57407: 56, 57408: 57,
    57409: 46, 57410: 47, 57411: 42, 57412: 45, 57413: 43, 57415: 61, 57416: 44,
    57417: _ARROW_CODEPOINTS["left"], 57418: _ARROW_CODEPOINTS["right"],
    57419: _ARROW_CODEPOINTS["up"], 57420: _ARROW_CODEPOINTS["down"],
    57421: _FUNCTIONAL_CODEPOINTS["pageUp"], 57422: _FUNCTIONAL_CODEPOINTS["pageDown"],
    57423: _FUNCTIONAL_CODEPOINTS["home"], 57424: _FUNCTIONAL_CODEPOINTS["end"],
    57425: _FUNCTIONAL_CODEPOINTS["insert"], 57426: _FUNCTIONAL_CODEPOINTS["delete"],
}

_LEGACY_KEY_SEQUENCES = {
    "up": ["\x1b[A", "\x1bOA"], "down": ["\x1b[B", "\x1bOB"],
    "right": ["\x1b[C", "\x1bOC"], "left": ["\x1b[D", "\x1bOD"],
    "home": ["\x1b[H", "\x1bOH", "\x1b[1~", "\x1b[7~"],
    "end": ["\x1b[F", "\x1bOF", "\x1b[4~", "\x1b[8~"],
    "insert": ["\x1b[2~"], "delete": ["\x1b[3~"],
    "pageUp": ["\x1b[5~", "\x1b[[5~"], "pageDown": ["\x1b[6~", "\x1b[[6~"],
    "clear": ["\x1b[E", "\x1bOE"],
    "f1": ["\x1bOP", "\x1b[11~", "\x1b[[A"], "f2": ["\x1bOQ", "\x1b[12~", "\x1b[[B"],
    "f3": ["\x1bOR", "\x1b[13~", "\x1b[[C"], "f4": ["\x1bOS", "\x1b[14~", "\x1b[[D"],
    "f5": ["\x1b[15~", "\x1b[[E"], "f6": ["\x1b[17~"], "f7": ["\x1b[18~"],
    "f8": ["\x1b[19~"], "f9": ["\x1b[20~"], "f10": ["\x1b[21~"],
    "f11": ["\x1b[23~"], "f12": ["\x1b[24~"],
}
_LEGACY_SHIFT_SEQUENCES = {
    "up": ["\x1b[a"], "down": ["\x1b[b"], "right": ["\x1b[c"], "left": ["\x1b[d"],
    "clear": ["\x1b[e"], "insert": ["\x1b[2$"], "delete": ["\x1b[3$"],
    "pageUp": ["\x1b[5$"], "pageDown": ["\x1b[6$"], "home": ["\x1b[7$"], "end": ["\x1b[8$"],
}
_LEGACY_CTRL_SEQUENCES = {
    "up": ["\x1bOa"], "down": ["\x1bOb"], "right": ["\x1bOc"], "left": ["\x1bOd"],
    "clear": ["\x1bOe"], "insert": ["\x1b[2^"], "delete": ["\x1b[3^"],
    "pageUp": ["\x1b[5^"], "pageDown": ["\x1b[6^"], "home": ["\x1b[7^"], "end": ["\x1b[8^"],
}
_LEGACY_SEQUENCE_KEY_IDS: dict[str, KeyId] = {
    "\x1bOA": "up", "\x1bOB": "down", "\x1bOC": "right", "\x1bOD": "left",
    "\x1bOH": "home", "\x1bOF": "end", "\x1b[E": "clear", "\x1bOE": "clear",
    "\x1bOe": "ctrl+clear", "\x1b[e": "shift+clear", "\x1b[2~": "insert",
    "\x1b[2$": "shift+insert", "\x1b[2^": "ctrl+insert", "\x1b[3$": "shift+delete",
    "\x1b[3^": "ctrl+delete", "\x1b[[5~": "pageUp", "\x1b[[6~": "pageDown",
    "\x1b[a": "shift+up", "\x1b[b": "shift+down", "\x1b[c": "shift+right", "\x1b[d": "shift+left",
    "\x1bOa": "ctrl+up", "\x1bOb": "ctrl+down", "\x1bOc": "ctrl+right", "\x1bOd": "ctrl+left",
    "\x1b[5$": "shift+pageUp", "\x1b[6$": "shift+pageDown", "\x1b[7$": "shift+home", "\x1b[8$": "shift+end",
    "\x1b[5^": "ctrl+pageUp", "\x1b[6^": "ctrl+pageDown", "\x1b[7^": "ctrl+home", "\x1b[8^": "ctrl+end",
    "\x1bOP": "f1", "\x1bOQ": "f2", "\x1bOR": "f3", "\x1bOS": "f4",
    "\x1b[11~": "f1", "\x1b[12~": "f2", "\x1b[13~": "f3", "\x1b[14~": "f4",
    "\x1b[[A": "f1", "\x1b[[B": "f2", "\x1b[[C": "f3", "\x1b[[D": "f4", "\x1b[[E": "f5",
    "\x1b[15~": "f5", "\x1b[17~": "f6", "\x1b[18~": "f7", "\x1b[19~": "f8",
    "\x1b[20~": "f9", "\x1b[21~": "f10", "\x1b[23~": "f11", "\x1b[24~": "f12",
    "\x1bb": "alt+left", "\x1bf": "alt+right", "\x1bp": "alt+up", "\x1bn": "alt+down",
}

_KITTY_CSI_U_REGEX = re.compile(r"\x1b\[([0-9]+)(?::([0-9]*))?(?::([0-9]+))?(?:;([0-9]+))?(?::([0-9]+))?u")
_KITTY_PRINTABLE_ALLOWED_MODIFIERS = _SHIFT | _LOCK_MASK


def _to_int32(value: int | float) -> int:
    if not math.isfinite(value) or value == 0:
        return 0
    unsigned = math.trunc(value) & 0xFFFFFFFF
    return unsigned - 0x100000000 if unsigned >= 0x80000000 else unsigned


def _from_char_code(value: int | float) -> str:
    return chr(_to_int32(value) & 0xFFFF)


def _normalize_kitty_functional_codepoint(codepoint: int | float) -> int | float:
    return _KITTY_FUNCTIONAL_KEY_EQUIVALENTS.get(codepoint, codepoint)


def _normalize_shifted_letter_identity_codepoint(codepoint: int | float, modifier: int | float) -> int | float:
    effective_modifier = _to_int32(modifier) & ~_LOCK_MASK
    if effective_modifier & _SHIFT and 65 <= codepoint <= 90:
        return codepoint + 32
    return codepoint


def _matches_legacy_modifier_sequence(data: str, key: str, modifier: int) -> bool:
    if modifier == _SHIFT:
        return data in _LEGACY_SHIFT_SEQUENCES[key]
    if modifier == _CTRL:
        return data in _LEGACY_CTRL_SEQUENCES[key]
    return False


@dataclass
class _ParsedKittySequence:
    codepoint: int | float
    modifier: int | float
    event_type: KeyEventType
    shifted_key: int | float | None = None
    base_layout_key: int | float | None = None


@dataclass
class _ParsedModifyOtherKeysSequence:
    codepoint: int | float
    modifier: int | float


def is_key_release(data: str) -> bool:
    if "\x1b[200~" in data:
        return False
    return any(marker in data for marker in (":3u", ":3~", ":3A", ":3B", ":3C", ":3D", ":3H", ":3F"))


def is_key_repeat(data: str) -> bool:
    if "\x1b[200~" in data:
        return False
    return any(marker in data for marker in (":2u", ":2~", ":2A", ":2B", ":2C", ":2D", ":2H", ":2F"))


def _parse_event_type(event_type_str: str | None) -> KeyEventType:
    if not event_type_str:
        return "press"
    event_type = float(event_type_str)
    if event_type == 2:
        return "repeat"
    if event_type == 3:
        return "release"
    return "press"


def _parse_kitty_sequence(data: str) -> _ParsedKittySequence | None:
    global _last_event_type
    csi_u_match = _KITTY_CSI_U_REGEX.fullmatch(data)
    if csi_u_match:
        codepoint = float(csi_u_match[1])
        shifted_key = float(csi_u_match[2]) if csi_u_match[2] else None
        base_layout_key = float(csi_u_match[3]) if csi_u_match[3] else None
        mod_value = float(csi_u_match[4]) if csi_u_match[4] else 1
        event_type = _parse_event_type(csi_u_match[5])
        _last_event_type = event_type
        return _ParsedKittySequence(codepoint, mod_value - 1, event_type, shifted_key, base_layout_key)

    arrow_match = re.fullmatch(r"\x1b\[1;([0-9]+)(?::([0-9]+))?([ABCD])", data)
    if arrow_match:
        mod_value = float(arrow_match[1])
        event_type = _parse_event_type(arrow_match[2])
        arrow_codes = {"A": -1, "B": -2, "C": -3, "D": -4}
        _last_event_type = event_type
        return _ParsedKittySequence(arrow_codes[arrow_match[3]], mod_value - 1, event_type)

    func_match = re.fullmatch(r"\x1b\[([0-9]+)(?:;([0-9]+))?(?::([0-9]+))?~", data)
    if func_match:
        key_num = float(func_match[1])
        mod_value = float(func_match[2]) if func_match[2] else 1
        event_type = _parse_event_type(func_match[3])
        func_codes = {
            2: _FUNCTIONAL_CODEPOINTS["insert"], 3: _FUNCTIONAL_CODEPOINTS["delete"],
            5: _FUNCTIONAL_CODEPOINTS["pageUp"], 6: _FUNCTIONAL_CODEPOINTS["pageDown"],
            7: _FUNCTIONAL_CODEPOINTS["home"], 8: _FUNCTIONAL_CODEPOINTS["end"],
        }
        codepoint = func_codes.get(key_num)
        if codepoint is not None:
            _last_event_type = event_type
            return _ParsedKittySequence(codepoint, mod_value - 1, event_type)

    home_end_match = re.fullmatch(r"\x1b\[1;([0-9]+)(?::([0-9]+))?([HF])", data)
    if home_end_match:
        mod_value = float(home_end_match[1])
        event_type = _parse_event_type(home_end_match[2])
        codepoint = _FUNCTIONAL_CODEPOINTS["home"] if home_end_match[3] == "H" else _FUNCTIONAL_CODEPOINTS["end"]
        _last_event_type = event_type
        return _ParsedKittySequence(codepoint, mod_value - 1, event_type)
    return None


def _matches_kitty_sequence(data: str, expected_codepoint: int, expected_modifier: int) -> bool:
    parsed = _parse_kitty_sequence(data)
    if parsed is None:
        return False
    actual_mod = _to_int32(parsed.modifier) & ~_LOCK_MASK
    expected_mod = _to_int32(expected_modifier) & ~_LOCK_MASK
    if actual_mod != expected_mod:
        return False
    normalized_codepoint = _normalize_shifted_letter_identity_codepoint(
        _normalize_kitty_functional_codepoint(parsed.codepoint), parsed.modifier,
    )
    normalized_expected_codepoint = _normalize_shifted_letter_identity_codepoint(
        _normalize_kitty_functional_codepoint(expected_codepoint), expected_modifier,
    )
    if normalized_codepoint == normalized_expected_codepoint:
        return True
    if parsed.base_layout_key is not None and parsed.base_layout_key == expected_codepoint:
        is_latin_letter = 97 <= normalized_codepoint <= 122
        is_known_symbol = _from_char_code(normalized_codepoint) in _SYMBOL_KEYS
        if not is_latin_letter and not is_known_symbol:
            return True
    return False


def _parse_modify_other_keys_sequence(data: str) -> _ParsedModifyOtherKeysSequence | None:
    match = re.fullmatch(r"\x1b\[27;([0-9]+);([0-9]+)~", data)
    if not match:
        return None
    return _ParsedModifyOtherKeysSequence(codepoint=float(match[2]), modifier=float(match[1]) - 1)


def _matches_modify_other_keys(data: str, expected_keycode: int, expected_modifier: int) -> bool:
    parsed = _parse_modify_other_keys_sequence(data)
    if parsed is None:
        return False
    return parsed.codepoint == expected_keycode and parsed.modifier == expected_modifier


def _is_windows_terminal_session() -> bool:
    return bool(os.environ.get("WT_SESSION")) and not any(os.environ.get(key) for key in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))


def _matches_raw_backspace(data: str, expected_modifier: int) -> bool:
    if data == "\x7f":
        return expected_modifier == 0
    if data != "\x08":
        return False
    return expected_modifier == (_CTRL if _is_windows_terminal_session() else 0)


def _raw_ctrl_char(key: str) -> str | None:
    char = key.lower()
    code = ord(char[0]) if char else math.nan
    if 97 <= code <= 122 or char in ("[", "\\", "]", "_"):
        return _from_char_code(_to_int32(code) & 0x1F)
    if char == "-":
        return chr(31)
    return None


def _matches_printable_modify_other_keys(data: str, expected_keycode: int, expected_modifier: int) -> bool:
    if expected_modifier == 0:
        return False
    parsed = _parse_modify_other_keys_sequence(data)
    if parsed is None or parsed.modifier != expected_modifier:
        return False
    return _normalize_shifted_letter_identity_codepoint(parsed.codepoint, parsed.modifier) == _normalize_shifted_letter_identity_codepoint(expected_keycode, expected_modifier)


def _format_key_name_with_modifiers(key_name: str, modifier: int | float) -> str | None:
    mods: list[str] = []
    effective_mod = _to_int32(modifier) & ~_LOCK_MASK
    supported_modifier_mask = _SHIFT | _CTRL | _ALT | _SUPER
    if effective_mod & ~supported_modifier_mask:
        return None
    if effective_mod & _SHIFT:
        mods.append("shift")
    if effective_mod & _CTRL:
        mods.append("ctrl")
    if effective_mod & _ALT:
        mods.append("alt")
    if effective_mod & _SUPER:
        mods.append("super")
    return "+".join([*mods, key_name]) if mods else key_name


@dataclass
class _ParsedKeyId:
    key: str
    ctrl: bool
    shift: bool
    alt: bool
    super: bool


def _parse_key_id(key_id: str) -> _ParsedKeyId | None:
    parts = key_id.lower().split("+")
    key = parts[-1]
    if not key:
        return None
    return _ParsedKeyId(key, "ctrl" in parts, "shift" in parts, "alt" in parts, "super" in parts)


def matches_key(data: str, key_id: KeyId) -> bool:
    parsed = _parse_key_id(key_id)
    if parsed is None:
        return False
    key = parsed.key
    modifier = 0
    if parsed.shift:
        modifier |= _SHIFT
    if parsed.alt:
        modifier |= _ALT
    if parsed.ctrl:
        modifier |= _CTRL
    if parsed.super:
        modifier |= _SUPER

    match key:
        case "escape" | "esc":
            if modifier != 0:
                return False
            return data == "\x1b" or _matches_kitty_sequence(data, _CODEPOINTS["escape"], 0) or _matches_modify_other_keys(data, _CODEPOINTS["escape"], 0)

        case "space":
            if not _kitty_protocol_active:
                if modifier == _CTRL and data == "\x00":
                    return True
                if modifier == _ALT and data == "\x1b ":
                    return True
            if modifier == 0:
                return data == " " or _matches_kitty_sequence(data, _CODEPOINTS["space"], 0) or _matches_modify_other_keys(data, _CODEPOINTS["space"], 0)
            return _matches_kitty_sequence(data, _CODEPOINTS["space"], modifier) or _matches_modify_other_keys(data, _CODEPOINTS["space"], modifier)

        case "tab":
            if modifier == _SHIFT:
                return data == "\x1b[Z" or _matches_kitty_sequence(data, _CODEPOINTS["tab"], _SHIFT) or _matches_modify_other_keys(data, _CODEPOINTS["tab"], _SHIFT)
            if modifier == 0:
                return data == "\t" or _matches_kitty_sequence(data, _CODEPOINTS["tab"], 0)
            return _matches_kitty_sequence(data, _CODEPOINTS["tab"], modifier) or _matches_modify_other_keys(data, _CODEPOINTS["tab"], modifier)

        case "enter" | "return":
            if modifier == _SHIFT:
                if _matches_kitty_sequence(data, _CODEPOINTS["enter"], _SHIFT) or _matches_kitty_sequence(data, _CODEPOINTS["kpEnter"], _SHIFT):
                    return True
                if _matches_modify_other_keys(data, _CODEPOINTS["enter"], _SHIFT):
                    return True
                if _kitty_protocol_active:
                    return data == "\x1b\r" or data == "\n"
                return False
            if modifier == _ALT:
                if _matches_kitty_sequence(data, _CODEPOINTS["enter"], _ALT) or _matches_kitty_sequence(data, _CODEPOINTS["kpEnter"], _ALT):
                    return True
                if _matches_modify_other_keys(data, _CODEPOINTS["enter"], _ALT):
                    return True
                if not _kitty_protocol_active:
                    return data == "\x1b\r"
                return False
            if modifier == 0:
                return (
                    data == "\r" or (not _kitty_protocol_active and data == "\n") or data == "\x1bOM"
                    or _matches_kitty_sequence(data, _CODEPOINTS["enter"], 0)
                    or _matches_kitty_sequence(data, _CODEPOINTS["kpEnter"], 0)
                )
            return (
                _matches_kitty_sequence(data, _CODEPOINTS["enter"], modifier)
                or _matches_kitty_sequence(data, _CODEPOINTS["kpEnter"], modifier)
                or _matches_modify_other_keys(data, _CODEPOINTS["enter"], modifier)
            )

        case "backspace":
            if modifier == _ALT:
                if data == "\x1b\x7f" or data == "\x1b\b":
                    return True
                return _matches_kitty_sequence(data, _CODEPOINTS["backspace"], _ALT) or _matches_modify_other_keys(data, _CODEPOINTS["backspace"], _ALT)
            if modifier == _CTRL:
                if _matches_raw_backspace(data, _CTRL):
                    return True
                return _matches_kitty_sequence(data, _CODEPOINTS["backspace"], _CTRL) or _matches_modify_other_keys(data, _CODEPOINTS["backspace"], _CTRL)
            if modifier == 0:
                return _matches_raw_backspace(data, 0) or _matches_kitty_sequence(data, _CODEPOINTS["backspace"], 0) or _matches_modify_other_keys(data, _CODEPOINTS["backspace"], 0)
            return _matches_kitty_sequence(data, _CODEPOINTS["backspace"], modifier) or _matches_modify_other_keys(data, _CODEPOINTS["backspace"], modifier)

        case "insert":
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["insert"] or _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["insert"], 0)
            if _matches_legacy_modifier_sequence(data, "insert", modifier):
                return True
            return _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["insert"], modifier)

        case "delete":
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["delete"] or _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["delete"], 0)
            if _matches_legacy_modifier_sequence(data, "delete", modifier):
                return True
            return _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["delete"], modifier)

        case "clear":
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["clear"]
            return _matches_legacy_modifier_sequence(data, "clear", modifier)

        case "home":
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["home"] or _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["home"], 0)
            if _matches_legacy_modifier_sequence(data, "home", modifier):
                return True
            return _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["home"], modifier)

        case "end":
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["end"] or _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["end"], 0)
            if _matches_legacy_modifier_sequence(data, "end", modifier):
                return True
            return _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["end"], modifier)

        case "pageup":
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["pageUp"] or _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["pageUp"], 0)
            if _matches_legacy_modifier_sequence(data, "pageUp", modifier):
                return True
            return _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["pageUp"], modifier)

        case "pagedown":
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["pageDown"] or _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["pageDown"], 0)
            if _matches_legacy_modifier_sequence(data, "pageDown", modifier):
                return True
            return _matches_kitty_sequence(data, _FUNCTIONAL_CODEPOINTS["pageDown"], modifier)

        case "up":
            if modifier == _ALT:
                return data == "\x1bp" or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["up"], _ALT)
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["up"] or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["up"], 0)
            if _matches_legacy_modifier_sequence(data, "up", modifier):
                return True
            return _matches_kitty_sequence(data, _ARROW_CODEPOINTS["up"], modifier)

        case "down":
            if modifier == _ALT:
                return data == "\x1bn" or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["down"], _ALT)
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["down"] or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["down"], 0)
            if _matches_legacy_modifier_sequence(data, "down", modifier):
                return True
            return _matches_kitty_sequence(data, _ARROW_CODEPOINTS["down"], modifier)

        case "left":
            if modifier == _ALT:
                return (
                    data == "\x1b[1;3D" or (not _kitty_protocol_active and data == "\x1bB")
                    or data == "\x1bb" or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["left"], _ALT)
                )
            if modifier == _CTRL:
                return data == "\x1b[1;5D" or _matches_legacy_modifier_sequence(data, "left", _CTRL) or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["left"], _CTRL)
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["left"] or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["left"], 0)
            if _matches_legacy_modifier_sequence(data, "left", modifier):
                return True
            return _matches_kitty_sequence(data, _ARROW_CODEPOINTS["left"], modifier)

        case "right":
            if modifier == _ALT:
                return (
                    data == "\x1b[1;3C" or (not _kitty_protocol_active and data == "\x1bF")
                    or data == "\x1bf" or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["right"], _ALT)
                )
            if modifier == _CTRL:
                return data == "\x1b[1;5C" or _matches_legacy_modifier_sequence(data, "right", _CTRL) or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["right"], _CTRL)
            if modifier == 0:
                return data in _LEGACY_KEY_SEQUENCES["right"] or _matches_kitty_sequence(data, _ARROW_CODEPOINTS["right"], 0)
            if _matches_legacy_modifier_sequence(data, "right", modifier):
                return True
            return _matches_kitty_sequence(data, _ARROW_CODEPOINTS["right"], modifier)

        case "f1" | "f2" | "f3" | "f4" | "f5" | "f6" | "f7" | "f8" | "f9" | "f10" | "f11" | "f12":
            if modifier != 0:
                return False
            return data in _LEGACY_KEY_SEQUENCES[key]

    if utf16_length(key) == 1 and ("a" <= key <= "z" or "0" <= key <= "9" or key in _SYMBOL_KEYS):
        codepoint = ord(key)
        raw_ctrl = _raw_ctrl_char(key)
        is_letter = "a" <= key <= "z"
        is_digit = "0" <= key <= "9"
        if modifier == _CTRL + _ALT and not _kitty_protocol_active and raw_ctrl:
            if data == "\x1b" + raw_ctrl:
                return True
        if modifier == _ALT and not _kitty_protocol_active and (is_letter or is_digit or key in _SYMBOL_KEYS):
            if data == "\x1b" + key:
                return True
        if modifier == _CTRL:
            if raw_ctrl and data == raw_ctrl:
                return True
            return _matches_kitty_sequence(data, codepoint, _CTRL) or _matches_printable_modify_other_keys(data, codepoint, _CTRL)
        if modifier == _SHIFT + _CTRL:
            return _matches_kitty_sequence(data, codepoint, _SHIFT + _CTRL) or _matches_printable_modify_other_keys(data, codepoint, _SHIFT + _CTRL)
        if modifier == _SHIFT:
            if is_letter and data == key.upper():
                return True
            return _matches_kitty_sequence(data, codepoint, _SHIFT) or _matches_printable_modify_other_keys(data, codepoint, _SHIFT)
        if modifier != 0:
            return _matches_kitty_sequence(data, codepoint, modifier) or _matches_printable_modify_other_keys(data, codepoint, modifier)
        return data == key or _matches_kitty_sequence(data, codepoint, 0)
    return False


def _format_parsed_key(codepoint: int | float, modifier: int | float, base_layout_key: int | float | None = None) -> str | None:
    normalized_codepoint = _normalize_kitty_functional_codepoint(codepoint)
    identity_codepoint = _normalize_shifted_letter_identity_codepoint(normalized_codepoint, modifier)
    is_latin_letter = 97 <= identity_codepoint <= 122
    is_digit = 48 <= identity_codepoint <= 57
    is_known_symbol = _from_char_code(identity_codepoint) in _SYMBOL_KEYS
    effective_codepoint = identity_codepoint if is_latin_letter or is_digit or is_known_symbol else (
        base_layout_key if base_layout_key is not None else identity_codepoint
    )
    key_name: str | None = None
    if effective_codepoint == _CODEPOINTS["escape"]:
        key_name = "escape"
    elif effective_codepoint == _CODEPOINTS["tab"]:
        key_name = "tab"
    elif effective_codepoint == _CODEPOINTS["enter"] or effective_codepoint == _CODEPOINTS["kpEnter"]:
        key_name = "enter"
    elif effective_codepoint == _CODEPOINTS["space"]:
        key_name = "space"
    elif effective_codepoint == _CODEPOINTS["backspace"]:
        key_name = "backspace"
    elif effective_codepoint == _FUNCTIONAL_CODEPOINTS["delete"]:
        key_name = "delete"
    elif effective_codepoint == _FUNCTIONAL_CODEPOINTS["insert"]:
        key_name = "insert"
    elif effective_codepoint == _FUNCTIONAL_CODEPOINTS["home"]:
        key_name = "home"
    elif effective_codepoint == _FUNCTIONAL_CODEPOINTS["end"]:
        key_name = "end"
    elif effective_codepoint == _FUNCTIONAL_CODEPOINTS["pageUp"]:
        key_name = "pageUp"
    elif effective_codepoint == _FUNCTIONAL_CODEPOINTS["pageDown"]:
        key_name = "pageDown"
    elif effective_codepoint == _ARROW_CODEPOINTS["up"]:
        key_name = "up"
    elif effective_codepoint == _ARROW_CODEPOINTS["down"]:
        key_name = "down"
    elif effective_codepoint == _ARROW_CODEPOINTS["left"]:
        key_name = "left"
    elif effective_codepoint == _ARROW_CODEPOINTS["right"]:
        key_name = "right"
    elif 48 <= effective_codepoint <= 57:
        key_name = _from_char_code(effective_codepoint)
    elif 97 <= effective_codepoint <= 122:
        key_name = _from_char_code(effective_codepoint)
    elif _from_char_code(effective_codepoint) in _SYMBOL_KEYS:
        key_name = _from_char_code(effective_codepoint)
    if not key_name:
        return None
    return _format_key_name_with_modifiers(key_name, modifier)


def parse_key(data: str) -> str | None:
    kitty = _parse_kitty_sequence(data)
    if kitty is not None:
        return _format_parsed_key(kitty.codepoint, kitty.modifier, kitty.base_layout_key)
    modify_other_keys = _parse_modify_other_keys_sequence(data)
    if modify_other_keys is not None:
        return _format_parsed_key(modify_other_keys.codepoint, modify_other_keys.modifier)
    if _kitty_protocol_active:
        if data == "\x1b\r" or data == "\n":
            return "shift+enter"
    legacy_sequence_key_id = _LEGACY_SEQUENCE_KEY_IDS.get(data)
    if legacy_sequence_key_id:
        return legacy_sequence_key_id
    if data == "\x1b":
        return "escape"
    if data == "\x1c":
        return "ctrl+\\"
    if data == "\x1d":
        return "ctrl+]"
    if data == "\x1f":
        return "ctrl+-"
    if data == "\x1b\x1b":
        return "ctrl+alt+["
    if data == "\x1b\x1c":
        return "ctrl+alt+\\"
    if data == "\x1b\x1d":
        return "ctrl+alt+]"
    if data == "\x1b\x1f":
        return "ctrl+alt+-"
    if data == "\t":
        return "tab"
    if data == "\r" or (not _kitty_protocol_active and data == "\n") or data == "\x1bOM":
        return "enter"
    if data == "\x00":
        return "ctrl+space"
    if data == " ":
        return "space"
    if data == "\x7f":
        return "backspace"
    if data == "\x08":
        return "ctrl+backspace" if _is_windows_terminal_session() else "backspace"
    if data == "\x1b[Z":
        return "shift+tab"
    if not _kitty_protocol_active and data == "\x1b\r":
        return "alt+enter"
    if not _kitty_protocol_active and data == "\x1b ":
        return "alt+space"
    if data == "\x1b\x7f" or data == "\x1b\b":
        return "alt+backspace"
    if not _kitty_protocol_active and data == "\x1bB":
        return "alt+left"
    if not _kitty_protocol_active and data == "\x1bF":
        return "alt+right"
    if not _kitty_protocol_active and utf16_length(data) == 2 and data[0] == "\x1b":
        code = ord(data[1])
        if 1 <= code <= 26:
            return "ctrl+alt+" + chr(code + 96)
        key = _from_char_code(code)
        if 97 <= code <= 122 or 48 <= code <= 57 or key in _SYMBOL_KEYS:
            return "alt+" + key
    if data == "\x1b[A":
        return "up"
    if data == "\x1b[B":
        return "down"
    if data == "\x1b[C":
        return "right"
    if data == "\x1b[D":
        return "left"
    if data == "\x1b[H" or data == "\x1bOH":
        return "home"
    if data == "\x1b[F" or data == "\x1bOF":
        return "end"
    if data == "\x1b[3~":
        return "delete"
    if data == "\x1b[5~":
        return "pageUp"
    if data == "\x1b[6~":
        return "pageDown"
    if utf16_length(data) == 1:
        code = ord(data)
        if 1 <= code <= 26:
            return "ctrl+" + chr(code + 96)
        if 32 <= code <= 126:
            return data
    return None


def decode_kitty_printable(data: str) -> str | None:
    match = _KITTY_CSI_U_REGEX.fullmatch(data)
    if not match:
        return None
    codepoint = float(match[1])
    if not math.isfinite(codepoint):
        return None
    shifted_key = float(match[2]) if match[2] else None
    mod_value = float(match[4]) if match[4] else 1
    modifier = mod_value - 1 if math.isfinite(mod_value) else 0
    if _to_int32(modifier) & ~_KITTY_PRINTABLE_ALLOWED_MODIFIERS:
        return None
    if _to_int32(modifier) & (_ALT | _CTRL):
        return None
    effective_codepoint = codepoint
    if _to_int32(modifier) & _SHIFT and shifted_key is not None:
        effective_codepoint = shifted_key
    effective_codepoint = _normalize_kitty_functional_codepoint(effective_codepoint)
    if not math.isfinite(effective_codepoint) or effective_codepoint < 32:
        return None
    try:
        return chr(int(effective_codepoint))
    except (ValueError, OverflowError):
        return None


def _decode_modify_other_keys_printable(data: str) -> str | None:
    parsed = _parse_modify_other_keys_sequence(data)
    if parsed is None:
        return None
    modifier = _to_int32(parsed.modifier) & ~_LOCK_MASK
    if modifier & ~_SHIFT:
        return None
    if not math.isfinite(parsed.codepoint) or parsed.codepoint < 32:
        return None
    try:
        return chr(int(parsed.codepoint))
    except (ValueError, OverflowError):
        return None


def decode_printable_key(data: str) -> str | None:
    kitty = decode_kitty_printable(data)
    return kitty if kitty is not None else _decode_modify_other_keys_printable(data)
