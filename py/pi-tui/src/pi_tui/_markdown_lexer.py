"""Native Marked 18.0.11 API used and re-exported by pi-tui.

Includes block/inline lexing and HTML parsing, renderers, hooks, extension
composition, token traversal, and async parse options. APIs use snake_case;
the reserved JavaScript ``async`` option is named ``async_``. Extension methods
receive their Python context explicitly instead of JavaScript ``this``.

The implementation uses the pinned Marked grammar rather than a different
Markdown engine or a JavaScript subprocess. Unicode property and casing data
come from ICU; matching the source runtime's Unicode data version is required
for identical behavior on newly assigned characters. See _marked_regex.py for
the upstream licenses and the scope of its regexp adapter.
"""

from ._marked_hooks import Hooks
from ._marked_instance import Marked
from ._marked_lexer import Lexer, Tokenizer
from ._marked_parser import Parser
from ._marked_renderer import Renderer, TextRenderer
from ._marked_types import (
    Alignment, HooksObject, LinkDefinition, Links, MarkedExtension, MarkedExtensions, MarkedOptions, MarkedToken, MaybePromise,
    RendererContext, RendererExtension, RendererExtensionFunction, RendererObject, RendererThis,
    TableCell, TableRow, Token, TokenizerAndRendererExtension, TokenizerContext,
    TokenizerExtension, TokenizerExtensionFunction, TokenizerObject,
    TokenizerStartFunction, TokenizerThis, Tokens, TokensList, WalkTokensCallback,
    change_defaults, get_defaults,
)

__all__ = [
    "Alignment", "Hooks", "HooksObject", "Lexer", "LinkDefinition", "Links", "Marked",
    "MarkedExtension", "MarkedExtensions", "MarkedOptions", "MarkedToken", "MaybePromise", "Parser", "Renderer", "RendererContext",
    "RendererExtension", "RendererExtensionFunction", "RendererObject", "RendererThis", "TableCell",
    "TableRow", "TextRenderer", "Token", "Tokenizer", "TokenizerAndRendererExtension",
    "TokenizerContext", "TokenizerExtension", "TokenizerExtensionFunction",
    "TokenizerObject", "TokenizerStartFunction", "TokenizerThis", "Tokens", "TokensList",
    "WalkTokensCallback", "change_defaults", "get_defaults",
]
