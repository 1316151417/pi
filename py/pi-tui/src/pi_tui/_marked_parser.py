"""Marked 18.0.11 block and inline token parsers."""

from __future__ import annotations

import sys
from typing import Protocol

from ._marked_helpers import js_string, js_truthy
from ._marked_renderer import Renderer, TextRenderer
from ._marked_types import MarkedOptions, RendererContext, Token, current_defaults


class _ParseCallable(Protocol):
    def __call__(self, tokens: list[Token], options: MarkedOptions | None = None) -> str: ...


class _ParseInlineCallable(Protocol):
    def __call__(self, tokens: list[Token], options: MarkedOptions | Renderer | TextRenderer | None = None,
                 *, renderer: Renderer | TextRenderer | None = None) -> str: ...


class _ParseMethod:
    def __get__(self, instance: Parser | None, owner: type[Parser]) -> _ParseCallable:
        def parse(tokens: list[Token], options: MarkedOptions | None = None) -> str:
            return (owner(options) if instance is None else instance)._parse(tokens)
        return parse


class _ParseInlineMethod:
    def __get__(self, instance: Parser | None, owner: type[Parser]) -> _ParseInlineCallable:
        def parse_inline(tokens: list[Token], options: MarkedOptions | Renderer | TextRenderer | None = None,
                         *, renderer: Renderer | TextRenderer | None = None) -> str:
            if instance is None:
                return owner(options if isinstance(options, MarkedOptions) else None)._parse_inline(tokens)
            selected = renderer if renderer is not None else options if isinstance(options, (Renderer, TextRenderer)) else None
            return instance._parse_inline(tokens, selected)
        return parse_inline


_BLOCK_METHODS = {name: name for name in ("space", "hr", "heading", "code", "table", "blockquote", "list", "checkbox", "html", "paragraph", "text")}
_BLOCK_METHODS["def"] = "def_"
_INLINE_METHODS = {name: name for name in ("html", "link", "image", "checkbox", "strong", "em", "codespan", "br", "text")}
_INLINE_METHODS.update({"escape": "text", "del": "del_"})


class Parser:
    parse = _ParseMethod()
    parse_inline = _ParseInlineMethod()

    def __init__(self, options: MarkedOptions | None = None) -> None:
        self.options = options if options is not None else current_defaults()
        if self.options.renderer is None:
            self.options.renderer = Renderer()
        self.renderer = self.options.renderer
        self.renderer.options = self.options
        self.renderer.parser = self
        self.text_renderer = TextRenderer()

    def _parse(self, tokens: list[Token]) -> str:
        return self._render(tokens, self.renderer, _BLOCK_METHODS)

    def _parse_inline(self, tokens: list[Token], renderer: Renderer | TextRenderer | None = None) -> str:
        return self._render(tokens, self.renderer if renderer is None else renderer, _INLINE_METHODS)

    def _render(self, tokens: list[Token], renderer: Renderer | TextRenderer, methods: dict[str, str]) -> str:
        self.renderer.parser = self
        output = ""
        for token in tokens:
            extensions = self.options.extensions
            extension = extensions.renderers.get(token.type) if extensions is not None else None
            if extension is not None:
                result = extension(RendererContext(self), token)
                if result is not False or token.type not in methods:
                    output += js_string(result) if js_truthy(result) else ""
                    continue
            method = methods.get(token.type)
            if method is None:
                message = f'Token with "{token.type}" type was not found.'
                if self.options.silent:
                    print(message, file=sys.stderr)
                    return ""
                raise RuntimeError(message)
            output += js_string(getattr(renderer, method)(token))
        return output


__all__ = ["Parser"]
