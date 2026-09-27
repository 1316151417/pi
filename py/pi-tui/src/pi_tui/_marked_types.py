"""Token and option records for the native Marked 18.0.11 implementation.

Extension callbacks use an explicit context in place of JavaScript's ``this``.
Walk callbacks receive only the token and can close over their Marked instance.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict, cast

if TYPE_CHECKING:
    from ._marked_hooks import Hooks
    from ._marked_lexer import Lexer, Tokenizer
    from ._marked_parser import Parser
    from ._marked_renderer import Renderer


type Alignment = Literal["center", "left", "right"] | None
type WalkTokensCallback = Callable[[Token], object]
type RendererObject = Mapping[str, Callable[..., object]]
type TokenizerObject = Mapping[str, Callable[..., object]]
type HooksObject = Mapping[str, Callable[..., object]]


@dataclass
class TableCell:
    text: str
    tokens: list[Token]
    header: bool
    align: Alignment


@dataclass
class TableRow:
    text: object


@dataclass
class Token:
    type: str
    raw: str
    text: str | None = None
    tokens: list[Token] | None = None
    depth: int = 0
    code_block_style: Literal["indented"] | None = None
    lang: str | None = None
    escaped: bool | None = None
    tag: str = ""
    href: str = ""
    title: str | None = None
    ordered: bool = False
    start: int | Literal[""] = ""
    loose: bool = False
    items: list[Token] = field(default_factory=list)
    task: bool = False
    checked: bool | None = None
    align: list[Alignment] = field(default_factory=list)
    header: list[TableCell] = field(default_factory=list)
    rows: list[list[TableCell]] = field(default_factory=list)
    pre: bool | None = None
    block: bool | None = None
    in_link: bool | None = None
    in_raw_block: bool | None = None
    pending: bool | None = None


class Tokens:
    Blockquote = Br = Checkbox = Code = Codespan = Def = Del = Em = Escape = Token
    Generic = Heading = Hr = HTML = Image = Link = List = ListItem = Paragraph = Token
    Space = Strong = Table = Tag = Text = Token
    TableCell = TableCell
    TableRow = TableRow


type MarkedToken = Token
type MaybePromise = Awaitable[None] | None


@dataclass
class LinkDefinition:
    href: str
    title: str | None = None


type Links = dict[str, LinkDefinition]


class TokensList(list[Token]):
    def __init__(self) -> None:
        super().__init__()
        self.links: dict[str, LinkDefinition] = {}


@dataclass
class TokenizerContext:
    lexer: Lexer


@dataclass
class RendererContext:
    parser: Parser


TokenizerThis = TokenizerContext
RendererThis = RendererContext


type TokenizerExtensionFunction = Callable[[TokenizerContext, str, list[Token]], Token | None | Literal[False]]
type TokenizerStartFunction = Callable[[TokenizerContext, str], int | float | None]
type RendererExtensionFunction = Callable[[RendererContext, Token], object]


@dataclass
class TokenizerExtension:
    """Extension source is a Python string; start offsets count UTF-16 units."""

    name: str
    level: Literal["block", "inline"]
    tokenizer: TokenizerExtensionFunction
    start: TokenizerStartFunction | None = None
    child_tokens: list[str] | None = None
    renderer: RendererExtensionFunction | None = None


@dataclass
class RendererExtension:
    name: str
    renderer: RendererExtensionFunction
    child_tokens: list[str] | None = None


type TokenizerAndRendererExtension = TokenizerExtension | RendererExtension


@dataclass
class _Extensions:
    renderers: dict[str, RendererExtensionFunction] = field(default_factory=dict)
    child_tokens: dict[str, list[str]] = field(default_factory=dict)
    block: list[TokenizerExtensionFunction] = field(default_factory=list)
    inline: list[TokenizerExtensionFunction] = field(default_factory=list)
    start_block: list[TokenizerStartFunction] = field(default_factory=list)
    start_inline: list[TokenizerStartFunction] = field(default_factory=list)


MarkedExtensions = _Extensions


class MarkedOptions:
    """Mutable options, retaining which keyword arguments were supplied.

    Retaining omitted keys makes per-call options and ``set_options`` shallow
    merges behave like their JavaScript object counterparts. ``async_`` is the
    Python spelling of the JavaScript ``async`` option.
    """

    async_: bool
    breaks: bool
    extensions: _Extensions | None
    gfm: bool
    hooks: Hooks | None
    pedantic: bool
    renderer: Renderer | None
    silent: bool
    tokenizer: Tokenizer | None
    walk_tokens: WalkTokensCallback | None
    em_strong_mask: Callable[[TokenizerContext, str], str | None] | None
    _provided: set[str]

    def __init__(self, **values: object) -> None:
        for name, value in _DEFAULT_VALUES.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_provided", set())
        for name, value in values.items():
            setattr(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name != "_provided":
            cast(set[str], self._provided).add(name)

    def as_mapping(self, *, only_provided: bool = False) -> dict[str, object]:
        names = self._provided if only_provided else self.__dict__
        return {name: getattr(self, name) for name in names if name != "_provided"}


class MarkedExtension(TypedDict, total=False):
    async_: bool
    breaks: bool
    extensions: list[TokenizerAndRendererExtension] | None
    gfm: bool
    hooks: HooksObject | None
    pedantic: bool
    renderer: RendererObject | None
    silent: bool
    tokenizer: TokenizerObject | None
    walk_tokens: WalkTokensCallback | None


_DEFAULT_VALUES: dict[str, object] = {
    "async_": False, "breaks": False, "extensions": None, "gfm": True,
    "hooks": None, "pedantic": False, "renderer": None, "silent": False,
    "tokenizer": None, "walk_tokens": None, "em_strong_mask": None,
}


def get_defaults() -> MarkedOptions:
    return MarkedOptions(**_DEFAULT_VALUES)


_defaults = get_defaults()


def change_defaults(new_defaults: MarkedOptions) -> None:
    global _defaults
    _defaults = new_defaults


def current_defaults() -> MarkedOptions:
    return _defaults


def option_values(options: MarkedOptions | Mapping[str, object] | None) -> dict[str, object]:
    return options.as_mapping(only_provided=True) if isinstance(options, MarkedOptions) else dict(options or {})


__all__ = [
    "Alignment", "HooksObject", "LinkDefinition", "Links", "MarkedExtension", "MarkedExtensions", "MarkedOptions", "MarkedToken", "MaybePromise",
    "RendererContext", "RendererExtension", "RendererExtensionFunction", "RendererObject",
    "RendererThis", "TableCell", "TableRow", "Token", "TokenizerAndRendererExtension", "TokenizerContext",
    "TokenizerExtension", "TokenizerExtensionFunction", "TokenizerObject", "TokenizerStartFunction",
    "TokenizerThis", "Tokens", "TokensList", "WalkTokensCallback", "change_defaults", "get_defaults",
]
