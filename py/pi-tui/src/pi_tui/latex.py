"""Render the source TUI's supported LaTeX subset as terminal Unicode text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from icu import UnicodeSet

from ._javascript import JS_WHITESPACE, from_utf16_units, js_trim, utf16_units
from .utils import visible_width

_SYMBOLS: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "varkappa": "ϰ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "ϕ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "ast": "∗",
    "star": "⋆",
    "circ": "∘",
    "bullet": "•",
    "oplus": "⊕",
    "ominus": "⊖",
    "otimes": "⊗",
    "oslash": "⊘",
    "odot": "⊙",
    "bigcirc": "○",
    "dagger": "†",
    "ddagger": "‡",
    "amalg": "⨿",
    "uplus": "⊎",
    "sqcap": "⊓",
    "sqcup": "⊔",
    "bowtie": "⋈",
    "Join": "⋈",
    "ltimes": "⋉",
    "rtimes": "⋊",
    "leftouterjoin": "⟕",
    "rightouterjoin": "⟖",
    "fullouterjoin": "⟗",
    "triangleleft": "◁",
    "triangleright": "▷",
    "wr": "≀",
    "cap": "∩",
    "cup": "∪",
    "bigcap": "⋂",
    "bigcup": "⋃",
    "bigwedge": "⋀",
    "bigvee": "⋁",
    "bigsqcup": "⨆",
    "biguplus": "⨄",
    "bigoplus": "⨁",
    "bigotimes": "⨂",
    "bigodot": "⨀",
    "setminus": "∖",
    "in": "∈",
    "notin": "∉",
    "ni": "∋",
    "subset": "⊂",
    "supset": "⊃",
    "subseteq": "⊆",
    "supseteq": "⊇",
    "sqsubset": "⊏",
    "sqsupset": "⊐",
    "sqsubseteq": "⊑",
    "sqsupseteq": "⊒",
    "prec": "≺",
    "preceq": "≼",
    "succ": "≻",
    "succeq": "≽",
    "ll": "≪",
    "gg": "≫",
    "le": "≤",
    "leq": "≤",
    "leqslant": "≤",
    "ge": "≥",
    "geq": "≥",
    "geqslant": "≥",
    "ne": "≠",
    "neq": "≠",
    "equiv": "≡",
    "approx": "≈",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "asymp": "≍",
    "doteq": "≐",
    "propto": "∝",
    "parallel": "∥",
    "perp": "⊥",
    "mid": "∣",
    "vdash": "⊢",
    "dashv": "⊣",
    "models": "⊨",
    "Vdash": "⊩",
    "Vvdash": "⊪",
    "nvdash": "⊬",
    "nvDash": "⊭",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "neg": "¬",
    "land": "∧",
    "wedge": "∧",
    "lor": "∨",
    "vee": "∨",
    "to": "→",
    "rightarrow": "→",
    "longrightarrow": "→",
    "leftarrow": "←",
    "longleftarrow": "←",
    "gets": "←",
    "leftrightarrow": "↔",
    "longleftrightarrow": "↔",
    "hookleftarrow": "↩",
    "hookrightarrow": "↪",
    "twoheadleftarrow": "↞",
    "twoheadrightarrow": "↠",
    "leftharpoonup": "↼",
    "leftharpoondown": "↽",
    "rightharpoonup": "⇀",
    "rightharpoondown": "⇁",
    "rightleftharpoons": "⇌",
    "leftrightharpoons": "⇋",
    "nearrow": "↗",
    "searrow": "↘",
    "swarrow": "↙",
    "nwarrow": "↖",
    "rightsquigarrow": "⇝",
    "leadsto": "⇝",
    "Rightarrow": "⇒",
    "Longrightarrow": "⇒",
    "Leftarrow": "⇐",
    "Longleftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "Longleftrightarrow": "⇔",
    "implies": "⇒",
    "iff": "⇔",
    "mapsto": "↦",
    "longmapsto": "↦",
    "uparrow": "↑",
    "downarrow": "↓",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "iint": "∬",
    "iiint": "∭",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "infty": "∞",
    "emptyset": "∅",
    "varnothing": "∅",
    "angle": "∠",
    "therefore": "∴",
    "because": "∵",
    "aleph": "ℵ",
    "beth": "ℶ",
    "gimel": "ℷ",
    "daleth": "ℸ",
    "top": "⊤",
    "bot": "⊥",
    "triangle": "△",
    "square": "□",
    "lozenge": "◊",
    "checkmark": "✓",
    "complement": "∁",
    "wp": "℘",
    "prime": "′",
    "ldots": "…",
    "dots": "…",
    "cdots": "⋯",
    "vdots": "⋮",
    "ddots": "⋱",
    "ell": "ℓ",
    "hbar": "ℏ",
    "Im": "ℑ",
    "Re": "ℜ",
    "langle": "⟨",
    "rangle": "⟩",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "Vert": "‖",
    "lVert": "‖",
    "rVert": "‖",
    "lbrace": "{",
    "rbrace": "}",
    "backslash": "\\",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "colon": ":",
}

_NAMED_OPERATORS = frozenset({
    "arccos",
    "arcsin",
    "arctan",
    "arg",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "deg",
    "det",
    "dim",
    "exp",
    "gcd",
    "hom",
    "inf",
    "ker",
    "lg",
    "lim",
    "liminf",
    "limsup",
    "ln",
    "log",
    "max",
    "min",
    "Pr",
    "sec",
    "sin",
    "sinh",
    "sup",
    "tan",
    "tanh",
})

_LIMIT_OPERATORS = frozenset({
    "argmax",
    "argmin",
    "inf",
    "injlim",
    "lim",
    "liminf",
    "limsup",
    "max",
    "min",
    "projlim",
    "sup",
})

_DISPLAY_LIMIT_SYMBOLS = frozenset({
    "bigcap",
    "bigcup",
    "bigodot",
    "bigoplus",
    "bigotimes",
    "bigsqcup",
    "biguplus",
    "bigvee",
    "bigwedge",
    "coprod",
    "int",
    "iint",
    "iiint",
    "oint",
    "prod",
    "sum",
})

_RELATION_COMMANDS = frozenset({
    "Leftarrow",
    "Leftrightarrow",
    "Longleftarrow",
    "Longleftrightarrow",
    "Longrightarrow",
    "Rightarrow",
    "Join",
    "Vdash",
    "Vvdash",
    "approx",
    "asymp",
    "bowtie",
    "cong",
    "dashv",
    "fullouterjoin",
    "doteq",
    "downarrow",
    "equiv",
    "ge",
    "geq",
    "geqslant",
    "gets",
    "gg",
    "hookleftarrow",
    "hookrightarrow",
    "iff",
    "implies",
    "in",
    "leadsto",
    "le",
    "leftarrow",
    "leftharpoondown",
    "leftharpoonup",
    "leftrightarrow",
    "leftrightharpoons",
    "leftouterjoin",
    "leq",
    "leqslant",
    "ll",
    "longleftarrow",
    "longleftrightarrow",
    "longmapsto",
    "longrightarrow",
    "ltimes",
    "mapsto",
    "mid",
    "models",
    "ne",
    "nearrow",
    "neq",
    "ni",
    "notin",
    "nvdash",
    "nvDash",
    "nwarrow",
    "parallel",
    "perp",
    "prec",
    "preceq",
    "propto",
    "rightharpoondown",
    "rightharpoonup",
    "rightleftharpoons",
    "rightouterjoin",
    "rightarrow",
    "rightsquigarrow",
    "rtimes",
    "searrow",
    "sim",
    "simeq",
    "sqsubset",
    "sqsubseteq",
    "sqsupset",
    "sqsupseteq",
    "subset",
    "subseteq",
    "succ",
    "succeq",
    "supset",
    "supseteq",
    "swarrow",
    "to",
    "triangleleft",
    "triangleright",
    "twoheadleftarrow",
    "twoheadrightarrow",
    "uparrow",
    "vdash",
})

_NEGATED_SYMBOLS: dict[str, str] = {
    "<": "≮",
    ">": "≯",
    "=": "≠",
    "∈": "∉",
    "∋": "∌",
    "∣": "∤",
    "∥": "∦",
    "∼": "≁",
    "≃": "≄",
    "≅": "≇",
    "≈": "≉",
    "≡": "≢",
    "≤": "≰",
    "≥": "≱",
    "≺": "⊀",
    "≻": "⊁",
    "⊂": "⊄",
    "⊃": "⊅",
    "⊆": "⊈",
    "⊇": "⊉",
    "⊢": "⊬",
    "⊨": "⊭",
    "↔": "↮",
    "←": "↚",
    "→": "↛",
    "⇒": "⇏",
    "⇐": "⇍",
    "⇔": "⇎",
    "≼": "⋠",
    "≽": "⋡",
}

_BLACKBOARD: dict[str, str] = {
    "C": "ℂ",
    "H": "ℍ",
    "N": "ℕ",
    "P": "ℙ",
    "Q": "ℚ",
    "R": "ℝ",
    "Z": "ℤ",
}

_SUPERSCRIPTS: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

_SUBSCRIPTS: dict[str, str] = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}

_SPACING_COMMANDS = frozenset({
    ",",
    ":",
    ";",
    " ",
    ">",
    "enspace",
    "enskip",
    "medspace",
    "quad",
    "qquad",
    "thickspace",
    "thinspace",
})

_NEGATIVE_SPACING_COMMANDS = frozenset({
    "!",
    "negmedspace",
    "negthickspace",
    "negthinspace",
})

_IGNORED_COMMANDS = frozenset({
    "displaystyle",
    "limits",
    "nolimits",
    "scriptstyle",
    "scriptscriptstyle",
    "textstyle",
})

_SIZE_COMMANDS = frozenset({
    "big",
    "Big",
    "bigg",
    "Bigg",
    "bigl",
    "Bigl",
    "biggl",
    "Biggl",
    "bigr",
    "Bigr",
    "biggr",
    "Biggr",
})

_PLAIN_WRAPPERS = frozenset({
    "emph",
    "mathcal",
    "mathbf",
    "mathfrak",
    "mathit",
    "mathrm",
    "mathnormal",
    "mathscr",
    "mathsf",
    "mathtt",
    "mathup",
    "mbox",
    "overbrace",
    "pmb",
    "smash",
    "substack",
    "text",
    "textbf",
    "textit",
    "textmd",
    "textnormal",
    "textrm",
    "textsc",
    "textsf",
    "textsl",
    "texttt",
    "textup",
    "underbrace",
    "bm",
    "boldsymbol",
})

_ACCENTS: dict[str, str] = {
    "acute": "́",
    "bar": "̅",
    "breve": "̆",
    "check": "̌",
    "ddot": "̈",
    "dot": "̇",
    "grave": "̀",
    "hat": "̂",
    "mathring": "̊",
    "overleftarrow": "⃖",
    "overleftrightarrow": "⃡",
    "overline": "̅",
    "overrightarrow": "⃗",
    "tilde": "̃",
    "underline": "̲",
    "vec": "⃗",
    "widehat": "̂",
    "widetilde": "̃",
}

_NEGATIVE_SPACE = "\u0000"
_NAMED_OPERATOR_START = "\U000f0004"
_NAMED_OPERATOR_END = "\U000f0005"
_NAMED_OPERATOR_END_UNITS = utf16_units(_NAMED_OPERATOR_END)
_LAYOUT_MARKER_START = "\U000f0000"
_LAYOUT_MARKER_END = "\U000f0001"
_LAYOUT_MARKER_PATTERN = re.compile(_LAYOUT_MARKER_START + r"([0-9]+)" + _LAYOUT_MARKER_END)
_TRAILING_LAYOUT_MARKER_PATTERN = re.compile(
    utf16_units(_LAYOUT_MARKER_START) + r"([0-9]+)" + utf16_units(_LAYOUT_MARKER_END) + r"$"
)
_PROTECTED_SPACE = "\U000f0002"
_SPACE_PATTERN = "[" + re.escape(JS_WHITESPACE) + "]"
_LETTER_OR_NUMBER = UnicodeSet("[[:L:][:N:]]")
_LETTER_OR_NUMBER.freeze()
_NUMBER = UnicodeSet("[:N:]")
_NUMBER.freeze()


def _format_script(value: str, kind: Literal["sub", "sup"]) -> str:
    value = js_trim(from_utf16_units(value))
    replacements = _SUBSCRIPTS if kind == "sub" else _SUPERSCRIPTS
    compact = re.sub(_SPACE_PATTERN + r"*([=+-])" + _SPACE_PATTERN + "*", r"\1", value)
    result = ""
    for character in compact:
        replacement = replacements.get(character)
        if replacement is None:
            break
        result += replacement
    else:
        return result
    prefix = "_" if kind == "sub" else "^"
    if len(value) == 1 or (kind == "sub" and re.fullmatch(r"[A-Za-z]+", value)):
        return prefix + value
    return f"{prefix}({value})"


def _format_fraction(numerator: str, denominator: str) -> str:
    numerator = js_trim(from_utf16_units(numerator))
    denominator = js_trim(from_utf16_units(denominator))
    simple_numerator = bool(numerator) and all(char == "." or _LETTER_OR_NUMBER.contains(char) for char in numerator)
    simple_denominator = (
        bool(denominator) and all(char == "." or _NUMBER.contains(char) for char in denominator)
    ) or len(denominator) == 1
    return f"{numerator if simple_numerator else f'({numerator})'}/{denominator if simple_denominator else f'({denominator})'}"


def _format_root(value: str, symbol: str = "√") -> str:
    value = js_trim(from_utf16_units(value))
    simple = bool(value) and all(char == "." or _LETTER_OR_NUMBER.contains(char) for char in value)
    return symbol + (value if simple else f"({value})")


def _normalize_output(value: str) -> str:
    value = from_utf16_units(value)
    value = "".join(
        " "
        if char == _NAMED_OPERATOR_START
        and index > 0
        and (value[index - 1] in ")]}" + _LAYOUT_MARKER_END or _LETTER_OR_NUMBER.contains(value[index - 1]))
        else char
        for index, char in enumerate(value)
    ).replace(_NAMED_OPERATOR_START, "")
    value = "".join(
        " "
        if char == _NAMED_OPERATOR_END
        and index + 1 < len(value)
        and (value[index + 1] in "√" + _LAYOUT_MARKER_START or _LETTER_OR_NUMBER.contains(value[index + 1]))
        else char
        for index, char in enumerate(value)
    ).replace(_NAMED_OPERATOR_END, "")
    lines = [js_trim(re.sub(r"[ \t]+", " ", line)) for line in value.split("\n")]
    return js_trim("\n".join(line for index, line in enumerate(lines) if line or 0 < index < len(lines) - 1))


@dataclass(slots=True)
class _FractionNode:
    numerator: str
    denominator: str


@dataclass(slots=True)
class _OperatorNode:
    operator: str
    lower: str | None = None
    upper: str | None = None


@dataclass(slots=True)
class _MatrixNode:
    lines: list[str]
    baseline: int


type _LayoutNode = _FractionNode | _OperatorNode | _MatrixNode


@dataclass(slots=True)
class _Layout:
    lines: list[str]
    width: int
    baseline: int


def _pad_layout_line(line: str, width: int, centered: bool = False) -> str:
    padding = max(0, width - visible_width(line))
    left = padding // 2 if centered else 0
    return " " * left + line + " " * (padding - left)


def _join_layouts(layouts: list[_Layout]) -> _Layout:
    if not layouts:
        return _Layout([""], 0, 0)
    baseline = max(layout.baseline for layout in layouts)
    below = max(len(layout.lines) - layout.baseline - 1 for layout in layouts)
    lines: list[str] = []
    for row in range(baseline + below + 1):
        line = ""
        for layout in layouts:
            source_row = row - baseline + layout.baseline
            line += (
                _pad_layout_line(layout.lines[source_row], layout.width)
                if 0 <= source_row < len(layout.lines)
                else " " * layout.width
            )
        lines.append(line.rstrip(JS_WHITESPACE))
    return _Layout(lines, sum(layout.width for layout in layouts), baseline)


def _render_layout(source: str, nodes: list[_LayoutNode]) -> _Layout:
    rendered_lines: list[str] = []
    first_baseline = 0
    for source_line in source.split("\n"):
        layouts: list[_Layout] = []
        position = 0
        previous_node: _LayoutNode | None = None
        for match in _LAYOUT_MARKER_PATTERN.finditer(source_line):
            index = match.start()
            # JS array lookup for an out-of-range numeric marker is undefined.
            digits = match[1].lstrip("0") or "0"
            node_index = int(digits) if len(digits) <= len(str(len(nodes))) else len(nodes)
            if node_index >= len(nodes):
                continue
            node = nodes[node_index]
            if index > position:
                sliced = source_line[position:index]
                trimmed = (sliced.lstrip(JS_WHITESPACE) if previous_node else sliced).rstrip(JS_WHITESPACE)
                preserve_leading_space = isinstance(previous_node, _MatrixNode) and bool(sliced) and sliced[0] in JS_WHITESPACE
                preserve_trailing_space = isinstance(node, _MatrixNode) and bool(sliced) and sliced[-1] in JS_WHITESPACE
                if trimmed:
                    text = (" " if preserve_leading_space else "") + trimmed + (" " if preserve_trailing_space else "")
                else:
                    text = " " if preserve_leading_space or preserve_trailing_space else ""
                layouts.append(_Layout([text], visible_width(text), 0))
            if isinstance(node, _FractionNode):
                numerator = _render_layout(node.numerator, nodes)
                denominator = _render_layout(node.denominator, nodes)
                content_width = max(numerator.width, denominator.width, 1)
                width = content_width + 2
                layouts.append(_Layout(
                    [
                        *(_pad_layout_line(line, width, True) for line in numerator.lines),
                        " " + "─" * content_width + " ",
                        *(_pad_layout_line(line, width, True) for line in denominator.lines),
                    ],
                    width,
                    len(numerator.lines),
                ))
            elif isinstance(node, _OperatorNode):
                content_width = max(
                    visible_width(node.operator),
                    0 if node.lower is None else visible_width(node.lower),
                    0 if node.upper is None else visible_width(node.upper),
                )
                lines: list[str] = []
                if node.upper is not None:
                    lines.append(_pad_layout_line(node.upper, content_width, True) + " ")
                lines.append(_pad_layout_line(node.operator, content_width, True) + " ")
                if node.lower is not None:
                    lines.append(_pad_layout_line(node.lower, content_width, True) + " ")
                layouts.append(_Layout(lines, content_width + 1, 0 if node.upper is None else 1))
            else:
                width = max((visible_width(line) for line in node.lines), default=0)
                layouts.append(_Layout([_pad_layout_line(line, width) for line in node.lines], width, node.baseline))
            position = match.end()
            previous_node = node
        if position < len(source_line):
            sliced = source_line[position:]
            trimmed = sliced.lstrip(JS_WHITESPACE) if previous_node else sliced
            text = " " + trimmed if isinstance(previous_node, _MatrixNode) and sliced[0] in JS_WHITESPACE else trimmed
            layouts.append(_Layout([text], visible_width(text), 0))
        line_layout = _join_layouts(layouts)
        if not rendered_lines:
            first_baseline = line_layout.baseline
        rendered_lines.extend(line_layout.lines)
    return _Layout(rendered_lines, max((visible_width(line) for line in rendered_lines), default=0), first_baseline)


class _LatexParser:
    def __init__(self, source: str, layout_nodes: list[_LayoutNode], display: bool) -> None:
        self.source = utf16_units(source)
        self.layout_nodes = layout_nodes
        self.display = display
        self.position = 0
        self.supported = True
        self.stack_fractions = True

    def render(self) -> str | None:
        rendered = self._parse_sequence()
        if not self.supported or self.position != len(self.source):
            return None
        return _normalize_output(rendered)

    def _parse_sequence(self, end_character: str | None = None) -> str:
        # Keep the accumulator in UTF-16 units, matching source[index] and
        # slice() even when one-token arguments split a surrogate pair.
        result = ""
        while self.position < len(self.source):
            character = self.source[self.position]
            if end_character and character == end_character:
                self.position += 1
                return from_utf16_units(result)
            if character == "}":
                self.supported = False
                return from_utf16_units(result)
            if character == "{":
                self.position += 1
                result += utf16_units(self._parse_sequence("}"))
                continue
            if character == "\\":
                command = self._parse_command()
                if command == _NEGATIVE_SPACE:
                    result = result.rstrip(JS_WHITESPACE)
                    if result.endswith(_NAMED_OPERATOR_END_UNITS):
                        result = result[:-len(_NAMED_OPERATOR_END_UNITS)]
                else:
                    result += utf16_units(command)
                continue
            if character in ("^", "_"):
                self.position += 1
                result = result.rstrip(JS_WHITESPACE)
                script = utf16_units(_format_script(self._parse_required_argument(False), "sub" if character == "_" else "sup"))
                if result.endswith(_NAMED_OPERATOR_END_UNITS):
                    result = result[:-len(_NAMED_OPERATOR_END_UNITS)] + script + _NAMED_OPERATOR_END_UNITS
                else:
                    result += script
                continue
            if character in JS_WHITESPACE:
                while self.position < len(self.source) and self.source[self.position] in JS_WHITESPACE:
                    self.position += 1
                result += " "
                continue
            if character in ("=", "<", ">"):
                result = result.rstrip(JS_WHITESPACE) + " " + character + " "
                self.position += 1
                continue
            if character == "&":
                self.position += 1
                continue
            if character == "~":
                self.position += 1
                result += " "
                continue
            if character == ".":
                marker = _TRAILING_LAYOUT_MARKER_PATTERN.search(result)
                node: _LayoutNode | None = None
                if marker is not None:
                    digits = marker[1].lstrip("0") or "0"
                    node_index = int(digits) if len(digits) <= len(str(len(self.layout_nodes))) else len(self.layout_nodes)
                    if node_index < len(self.layout_nodes):
                        node = self.layout_nodes[node_index]
                if isinstance(node, _MatrixNode):
                    node.lines[-1] += character
                    self.position += 1
                    continue
            result += character
            self.position += 1
        if end_character:
            self.supported = False
        return from_utf16_units(result)

    def _parse_command(self) -> str:
        self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""
        first = self.source[self.position]
        if first in ("\n", "\r"):
            self.position += 1
            if first == "\r" and self.source[self.position:self.position + 1] == "\n":
                self.position += 1
            return " "
        if "A" <= first <= "Z" or "a" <= first <= "z":
            start = self.position
            while self.position < len(self.source):
                character = self.source[self.position]
                if not ("A" <= character <= "Z" or "a" <= character <= "z"):
                    break
                self.position += 1
            command = self.source[start:self.position]
        else:
            command = first
            self.position += 1
        if command == "\\":
            return "\n"
        if command in _SPACING_COMMANDS:
            return " "
        if command in _NEGATIVE_SPACING_COMMANDS:
            return _NEGATIVE_SPACE
        if command in _IGNORED_COMMANDS:
            return ""
        if command in ("{", "}", "$", "%", "#", "_", "&"):
            return command
        if command == "|":
            return "‖"
        if command == "not":
            value = js_trim(from_utf16_units(self._parse_required_argument(False)))
            negated = _NEGATED_SYMBOLS.get(value)
            if negated is not None:
                return f" {negated} "
            if not value:
                self.supported = False
                return ""
            return f" {value[0]}\u0338{value[1:]} "
        if command in _LIMIT_OPERATORS:
            return self._parse_operator(command, "bracket", True, True)
        symbol = _SYMBOLS.get(command)
        if symbol is not None:
            if command in _DISPLAY_LIMIT_SYMBOLS:
                return self._parse_operator(symbol, "script", True)
            return f" {symbol} " if command in ("cdot", "times") or command in _RELATION_COMMANDS else symbol
        if command in _NAMED_OPERATORS:
            return _NAMED_OPERATOR_START + command + _NAMED_OPERATOR_END
        if command in _SIZE_COMMANDS:
            return ""
        if command in ("left", "middle", "right"):
            if self.source[self.position:self.position + 1] == ".":
                self.position += 1
            return ""
        if command in ("frac", "dfrac", "tfrac"):
            should_stack = self.display and self.stack_fractions and command != "tfrac"
            numerator = self._parse_required_argument(not should_stack)
            denominator = self._parse_required_argument(not should_stack)
            if should_stack:
                index = len(self.layout_nodes)
                self.layout_nodes.append(_FractionNode(_normalize_output(numerator), _normalize_output(denominator)))
                return f"{_LAYOUT_MARKER_START}{index}{_LAYOUT_MARKER_END}"
            return _format_fraction(numerator, denominator)
        if command == "sqrt":
            degree = self._parse_optional_argument()
            if degree is not None:
                degree = js_trim(degree)
            value = self._parse_required_argument()
            if degree is None or degree == "2":
                return _format_root(value)
            if degree == "3":
                return _format_root(value, "∛")
            if degree == "4":
                return _format_root(value, "∜")
            return _format_script(degree, "sup") + _format_root(value)
        if command in ("boxed", "fbox"):
            return "[" + js_trim(self._parse_required_argument()) + "]"
        if command in ("binom", "dbinom", "tbinom"):
            return f"({self._parse_required_argument()} choose {self._parse_required_argument()})"
        accent = _ACCENTS.get(command)
        if accent is not None:
            value = from_utf16_units(self._parse_required_argument())
            return value + accent if len(value) == 1 else f"{command}({value})"
        if command == "mathbb":
            value = from_utf16_units(self._parse_required_argument())
            return "".join(_BLACKBOARD.get(character, character) for character in value)
        if command == "operatorname":
            starred = self.source[self.position:self.position + 1] == "*"
            if starred:
                self.position += 1
            operator = js_trim(_normalize_output(self._parse_required_argument()))
            return self._parse_operator(operator, "bracket", starred, True)
        if command in ("mod", "bmod"):
            return " mod "
        if command in ("pmod", "pod"):
            value = js_trim(self._parse_required_argument())
            return f" (mod {value})" if command == "pmod" else f" ({value})"
        if command in ("overset", "stackrel"):
            upper = self._parse_required_argument()
            value = js_trim(self._parse_required_argument())
            return value + _format_script(upper, "sup")
        if command == "underset":
            lower = self._parse_required_argument()
            value = js_trim(self._parse_required_argument())
            return value + _format_script(lower, "sub")
        if command in _PLAIN_WRAPPERS:
            value = self._parse_required_argument()
            return value if command.startswith("text") or command == "mbox" else js_trim(value)
        if command == "begin":
            return self._parse_environment()
        if command == "end":
            self.supported = False
            return ""
        self.supported = False
        return "\\" + command

    def _parse_operator(
        self, operator: str, inline_lower_style: Literal["bracket", "script"], display_limits: bool, spaced: bool = False
    ) -> str:
        use_display_limits = display_limits
        modifier_position = self.position
        while modifier_position < len(self.source) and self.source[modifier_position] in " \t":
            modifier_position += 1
        modifier = re.match(r"\\(limits|nolimits)(?![A-Za-z])", self.source[modifier_position:])
        if modifier is not None:
            use_display_limits = modifier[1] == "limits"
            self.position = modifier_position + modifier.end()
        lower: str | None = None
        upper: str | None = None
        while True:
            script_position = self.position
            while script_position < len(self.source) and self.source[script_position] in " \t":
                script_position += 1
            kind = self.source[script_position:script_position + 1]
            if kind not in ("_", "^"):
                break
            self.position = script_position + 1
            value = _normalize_output(self._parse_required_argument(False)).replace(" ", "")
            if kind == "_":
                if lower is not None:
                    self.supported = False
                lower = value
            else:
                if upper is not None:
                    self.supported = False
                upper = value
        if self.display and use_display_limits and (lower is not None or upper is not None):
            index = len(self.layout_nodes)
            self.layout_nodes.append(_OperatorNode(operator, lower, upper))
            return f"{_LAYOUT_MARKER_START}{index}{_LAYOUT_MARKER_END}"
        rendered = operator
        if lower is not None:
            rendered += f"[{lower}]" if inline_lower_style == "bracket" else _format_script(lower, "sub")
        if upper is not None:
            rendered += _format_script(upper, "sup")
        return f" {rendered} " if spaced else rendered

    def _parse_required_argument(self, stack_fractions: bool = True) -> str:
        previous_stack_fractions = self.stack_fractions
        self.stack_fractions = previous_stack_fractions and stack_fractions
        value = self._parse_required_argument_value()
        self.stack_fractions = previous_stack_fractions
        return value

    def _parse_required_argument_value(self) -> str:
        while self.position < len(self.source) and self.source[self.position] in JS_WHITESPACE:
            self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""
        if self.source[self.position] == "{":
            self.position += 1
            return self._parse_sequence("}")
        if self.source[self.position] == "\\":
            return self._parse_command()
        value = self.source[self.position]
        self.position += 1
        return value

    def _parse_optional_argument(self) -> str | None:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.source[self.position:self.position + 1] != "[":
            return None
        end = self.source.find("]", self.position + 1)
        if end < 0:
            self.supported = False
            return None
        value = self.source[self.position + 1:end]
        self.position = end + 1
        return self._render_nested(value)

    def _read_raw_group(self) -> str | None:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.source[self.position:self.position + 1] != "{":
            self.supported = False
            return None
        self.position += 1
        start = self.position
        depth = 1
        while self.position < len(self.source):
            character = self.source[self.position]
            if character == "\\":
                self.position += 2
                continue
            if character == "{":
                depth += 1
            if character == "}":
                depth -= 1
            if depth == 0:
                value = self.source[start:self.position]
                self.position += 1
                return value
            self.position += 1
        self.supported = False
        return None

    def _split_environment_rows(self, body: str) -> list[str]:
        return re.split(r"\\\\(?:\[[^\]\n]*\])?", body)

    def _parse_environment(self) -> str:
        environment = self._read_raw_group()
        if not environment:
            return ""
        end_marker = "\\end{" + environment + "}"
        end = self.source.find(end_marker, self.position)
        if end < 0:
            self.supported = False
            return ""
        body = self.source[self.position:end]
        self.position = end + len(end_marker)
        if environment in ("equation", "equation*", "displaymath"):
            return js_trim(self._render_nested(body))
        if environment in (
            "aligned", "align", "align*", "alignedat", "alignat", "alignat*",
            "gather", "gathered", "multline", "multline*", "split",
        ):
            aligned_at = environment in ("alignedat", "alignat", "alignat*")
            aligned_body = re.sub("^" + _SPACE_PATTERN + r"*\{[^}]*\}", "", body, count=1) if aligned_at else body
            lines: list[str] = []
            for row in self._split_environment_rows(aligned_body):
                cells = row.split("&")
                source = " ".join("".join(cells[index:index + 2]) for index in range(0, len(cells), 2)) if aligned_at else "".join(cells)
                rendered = js_trim(self._render_nested(source))
                if rendered:
                    lines.append(rendered)
            return "\n".join(lines)
        if environment in ("cases", "cases*"):
            rows = [
                [js_trim(self._render_nested(cell, False)) for cell in row.split("&")]
                for row in self._split_environment_rows(body)
            ]
            rows = [row for row in rows if any(row)]
            case_lines: list[str] = []
            for index, row in enumerate(rows):
                value = re.sub("," + _SPACE_PATTERN + "*$", "", row[0] if row else "")
                condition = row[1] if len(row) > 1 else ""
                delimiter = "⎧" if index == 0 else "⎩" if index == len(rows) - 1 else "⎨"
                condition_prefix = " " if re.match(r"(?:if|when|for|otherwise)\b", condition, re.IGNORECASE | re.ASCII) else " if "
                case_lines.append(f"{delimiter} {value}" + (condition_prefix + condition if condition else ""))
            return "\n".join(case_lines)
        if environment in ("array", "matrix", "smallmatrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix"):
            matrix_body = re.sub("^" + _SPACE_PATTERN + r"*\{[^}]*\}", "", body, count=1) if environment == "array" else body
            return self._render_matrix(environment, matrix_body)
        self.supported = False
        return body

    def _render_matrix(self, environment: str, body: str) -> str:
        matrix = [
            [js_trim(self._render_nested(cell, False)) for cell in row.split("&")]
            for row in self._split_environment_rows(body)
        ]
        matrix = [row for row in matrix if any(row)]
        column_count = max((len(row) for row in matrix), default=0)
        column_widths = [
            max((visible_width(row[column] if column < len(row) else "") for row in matrix), default=0)
            for column in range(column_count)
        ]
        rows: list[str] = []
        for row in matrix:
            cells: list[str] = []
            for column in range(column_count):
                cell = row[column] if column < len(row) else ""
                cells.append(cell + _PROTECTED_SPACE * max(0, column_widths[column] - visible_width(cell)))
            rows.append(" │ ".join(cells))
        if environment in ("array", "matrix", "smallmatrix"):
            lines = rows
        else:
            delimiters = {
                "pmatrix": ("⎛", "⎞", "⎜", "⎟", "⎝", "⎠"),
                "bmatrix": ("⎡", "⎤", "⎢", "⎥", "⎣", "⎦"),
                "Bmatrix": ("⎧", "⎫", "⎨", "⎬", "⎩", "⎭"),
                "vmatrix": ("│", "│", "│", "│", "│", "│"),
                "Vmatrix": ("║", "║", "║", "║", "║", "║"),
            }
            delimiter = delimiters.get(environment)
            if delimiter is None:
                self.supported = False
                return "\n".join(rows)
            lines: list[str] = []
            for index, row in enumerate(rows):
                left = delimiter[0] if index == 0 else delimiter[4] if index == len(rows) - 1 else delimiter[2]
                right = delimiter[1] if index == 0 else delimiter[5] if index == len(rows) - 1 else delimiter[3]
                lines.append(f"{left} {row} {right}")
        if len(lines) <= 1:
            return lines[0] if lines else ""
        index = len(self.layout_nodes)
        self.layout_nodes.append(_MatrixNode(lines, 0))
        return f"{_LAYOUT_MARKER_START}{index}{_LAYOUT_MARKER_END}"

    def _render_nested(self, source: str, stack_fractions: bool = True) -> str:
        rendered = _LatexParser(source, self.layout_nodes, self.display and stack_fractions).render()
        if rendered is None:
            self.supported = False
            return source
        return rendered


@dataclass(frozen=True, slots=True)
class RenderLatexOptions:
    """Stack fractions and operator limits vertically when display is true."""

    display: bool = False


def render_latex(source: str, options: RenderLatexOptions | None = None) -> str | None:
    """Render supported LaTeX, or return None for unsupported/malformed syntax."""
    layout_nodes: list[_LayoutNode] = []
    rendered = _LatexParser(source, layout_nodes, options is not None and options.display is True).render()
    if rendered is None:
        return None
    if not layout_nodes:
        return rendered.replace(_PROTECTED_SPACE, " ")
    lines = _render_layout(rendered, layout_nodes).lines
    indentations = [len(line) - len(line.lstrip(JS_WHITESPACE)) for line in lines if js_trim(line)]
    if not indentations:
        return ""
    indentation = min(indentations)
    return "\n".join(line[indentation:].rstrip(JS_WHITESPACE) for line in lines).rstrip(JS_WHITESPACE).replace(_PROTECTED_SPACE, " ")


__all__ = ["RenderLatexOptions", "render_latex"]
