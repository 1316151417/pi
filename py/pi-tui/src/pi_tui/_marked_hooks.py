"""Marked 18.0.11 parser pipeline hooks."""

from __future__ import annotations

from collections.abc import Callable

from ._marked_lexer import Lexer
from ._marked_parser import Parser
from ._marked_types import MarkedOptions, Token, current_defaults


class Hooks:
    pass_through_hooks = {"preprocess", "postprocess", "process_all_tokens", "em_strong_mask"}
    pass_through_hooks_respect_async = {"preprocess", "postprocess", "process_all_tokens"}

    def __init__(self, options: MarkedOptions | None = None) -> None:
        self.options = options if options is not None else current_defaults()
        self.block: bool | None = None

    def preprocess(self, markdown: str) -> str:
        return markdown

    def postprocess(self, html: object) -> object:
        return html

    def process_all_tokens(self, tokens: list[Token]) -> list[Token]:
        return tokens

    def em_strong_mask(self, src: str) -> str:
        return src

    def provide_lexer(self, block: bool | None = None) -> Callable[[str, MarkedOptions | None], list[Token]]:
        return Lexer.lex if (self.block if block is None else block) else Lexer.lex_inline

    def provide_parser(self, block: bool | None = None) -> Callable[[list[Token], MarkedOptions | None], str]:
        return Parser.parse if (self.block if block is None else block) else Parser.parse_inline


__all__ = ["Hooks"]
