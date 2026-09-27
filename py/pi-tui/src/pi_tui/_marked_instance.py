"""Marked 18.0.11 instance API, extension composition, and async pipeline.

Python callbacks replacing a renderer, tokenizer, or hook method receive that
object as their first argument. Addon renderers/tokenizers receive their
RendererContext/TokenizerContext instead. walk_tokens callbacks take a token.

With a running asyncio loop, async parse calls return an already scheduled
Task. Outside a loop they return a coroutine for the caller to await. Async
walk callbacks in synchronous mode likewise require a running loop to start.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from types import MethodType
from typing import cast

from ._javascript import from_utf16_units, utf16_units
from ._marked_helpers import escape_html_entities, js_truthy
from ._marked_hooks import Hooks
from ._marked_lexer import Lexer, Tokenizer, _map_token_strings
from ._marked_parser import Parser
from ._marked_renderer import Renderer, TextRenderer
from ._marked_types import (
    MarkedExtension, MarkedOptions, RendererContext, RendererExtensionFunction,
    Token, TokenizerExtension, TokenizerAndRendererExtension, TokensList,
    WalkTokensCallback, _Extensions, get_defaults, option_values,
)


async def _resolve(value: object) -> object:
    seen: dict[int, object] = {}
    while inspect.isawaitable(value):
        if id(value) in seen:
            raise TypeError("Chaining cycle detected for promise")
        seen[id(value)] = value
        value = await value
    return value


def _start(value: object) -> object:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return value
    return asyncio.ensure_future(value)


def _call(callback: Callable[..., object], context: object, *args: object) -> object:
    function = callback.__func__ if isinstance(callback, MethodType) else callback
    return function(context, *args)


def _renderer_override(renderer: Renderer, callback: Callable[..., object], previous: Callable[..., object]) -> MethodType:
    def invoke(_self: object, *args: object) -> object:
        result = _call(callback, renderer, *args)
        if result is False:
            result = previous(*args)
        return result if js_truthy(result) else ""
    return MethodType(invoke, renderer)


def _tokenizer_override(tokenizer: Tokenizer, callback: Callable[..., object], previous: Callable[..., object]) -> MethodType:
    def invoke(_self: object, *args: object) -> object:
        source_args = tuple(from_utf16_units(value) if isinstance(value, str) else value for value in args)
        result = _call(callback, tokenizer, *source_args)
        if result is False:
            return previous(*args)
        if isinstance(result, Token):
            _map_token_strings([result], utf16_units)
        return result
    return MethodType(invoke, tokenizer)


def _extension_renderer(callback: RendererExtensionFunction, previous: RendererExtensionFunction) -> RendererExtensionFunction:
    def invoke(context: RendererContext, token: Token) -> object:
        result = callback(context, token)
        return previous(context, token) if result is False else result
    return invoke


def _hook_override(marked: Marked, hooks: Hooks, name: str, callback: Callable[..., object], previous: Callable[..., object]) -> MethodType:
    if name in Hooks.pass_through_hooks:
        def pass_through(_self: object, argument: object) -> object:
            if marked.defaults.async_ and name in Hooks.pass_through_hooks_respect_async:
                async def chained() -> object:
                    result = await _resolve(_call(callback, hooks, argument))
                    return await _resolve(previous(result))
                return _start(chained())
            return previous(_call(callback, hooks, argument))
        return MethodType(pass_through, hooks)
    def override(_self: object, *args: object) -> object:
        if marked.defaults.async_:
            async def chained() -> object:
                result = await _resolve(_call(callback, hooks, *args))
                return await _resolve(previous(*args)) if result is False else result
            return _start(chained())
        result = _call(callback, hooks, *args)
        return previous(*args) if result is False else result
    return MethodType(override, hooks)


def _walk_override(callback: WalkTokensCallback, previous: WalkTokensCallback | None) -> WalkTokensCallback:
    def walk(token: Token) -> object:
        values = [callback(token)]
        if previous is not None:
            result = previous(token)
            values.extend(result if isinstance(result, list) else [result])
        return values
    return walk


def _flatten_tokens(values: list[object]) -> list[Token]:
    flattened: list[Token] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_tokens(value))
        else:
            flattened.append(cast(Token, value))
    return flattened


def _input_type(value: object) -> str:
    if isinstance(value, bool):
        return "[object Boolean]"
    if isinstance(value, (int, float)):
        return "[object Number]"
    if isinstance(value, list):
        return "[object Array]"
    if callable(value):
        return "[object Function]"
    return "[object Object]"


class Marked:
    Parser = Parser
    Renderer = Renderer
    TextRenderer = TextRenderer
    Lexer = Lexer
    Tokenizer = Tokenizer
    Hooks = Hooks

    def __init__(self, *extensions: MarkedExtension | Mapping[str, object]) -> None:
        self.defaults = get_defaults()
        self.use(*extensions)

    def walk_tokens(self, tokens: list[Token], callback: WalkTokensCallback) -> list[object]:
        values: list[object] = []
        for token in tokens:
            result = callback(token)
            values.extend(_start(value) for value in (result if isinstance(result, list) else [result]))
            if token.type == "table":
                for cell in token.header:
                    values.extend(self.walk_tokens(cell.tokens, callback))
                for row in token.rows:
                    for cell in row:
                        values.extend(self.walk_tokens(cell.tokens, callback))
            elif token.type == "list":
                values.extend(self.walk_tokens(token.items, callback))
            else:
                extensions = self.defaults.extensions
                children = extensions.child_tokens.get(token.type) if extensions is not None else None
                if children is not None:
                    for name in children:
                        values.extend(self.walk_tokens(_flatten_tokens(getattr(token, name)), callback))
                elif token.tokens is not None:
                    values.extend(self.walk_tokens(token.tokens, callback))
        return values

    def use(self, *packs: MarkedExtension | Mapping[str, object], **values: object) -> Marked:
        """Register addons and partial method overrides, newest overrides first."""
        registry = self.defaults.extensions if self.defaults.extensions is not None else _Extensions()
        all_packs = [*packs, values] if values else list(packs)
        for pack in all_packs:
            updates = dict(pack)
            updates["async_"] = bool(self.defaults.async_ or updates.get("async_") or False)
            addons = pack.get("extensions")
            if addons is not None:
                for extension in cast(list[TokenizerAndRendererExtension], addons):
                    if not extension.name:
                        raise ValueError("extension name required")
                    if extension.renderer is not None:
                        previous_renderer = registry.renderers.get(extension.name)
                        registry.renderers[extension.name] = _extension_renderer(extension.renderer, previous_renderer) if previous_renderer is not None else extension.renderer
                    if isinstance(extension, TokenizerExtension):
                        if extension.level not in ("block", "inline"):
                            raise ValueError("extension level must be 'block' or 'inline'")
                        callbacks = registry.block if extension.level == "block" else registry.inline
                        callbacks.insert(0, extension.tokenizer)
                        if extension.start is not None:
                            starts = registry.start_block if extension.level == "block" else registry.start_inline
                            starts.append(extension.start)
                    if extension.child_tokens is not None:
                        registry.child_tokens[extension.name] = extension.child_tokens
                updates["extensions"] = registry
            renderer_pack = pack.get("renderer")
            if js_truthy(renderer_pack):
                renderer = self.defaults.renderer if self.defaults.renderer is not None else Renderer(self.defaults)
                for name, callback in cast(Mapping[str, Callable[..., object]], renderer_pack).items():
                    if not hasattr(renderer, name):
                        raise ValueError(f"renderer '{name}' does not exist")
                    if name in ("options", "parser"):
                        continue
                    setattr(renderer, name, _renderer_override(renderer, callback, getattr(renderer, name)))
                updates["renderer"] = renderer
            tokenizer_pack = pack.get("tokenizer")
            if js_truthy(tokenizer_pack):
                tokenizer = self.defaults.tokenizer if self.defaults.tokenizer is not None else Tokenizer(self.defaults)
                for name, callback in cast(Mapping[str, Callable[..., object]], tokenizer_pack).items():
                    if not hasattr(tokenizer, name):
                        raise ValueError(f"tokenizer '{name}' does not exist")
                    if name in ("options", "rules", "lexer"):
                        continue
                    setattr(tokenizer, name, _tokenizer_override(tokenizer, callback, getattr(tokenizer, name)))
                updates["tokenizer"] = tokenizer
            hooks_pack = pack.get("hooks")
            if js_truthy(hooks_pack):
                hooks = self.defaults.hooks if self.defaults.hooks is not None else Hooks()
                for name, callback in cast(Mapping[str, Callable[..., object]], hooks_pack).items():
                    if not hasattr(hooks, name):
                        raise ValueError(f"hook '{name}' does not exist")
                    if name in ("options", "block"):
                        continue
                    setattr(hooks, name, _hook_override(self, hooks, name, callback, getattr(hooks, name)))
                updates["hooks"] = hooks
            walk = pack.get("walk_tokens")
            if js_truthy(walk):
                updates["walk_tokens"] = _walk_override(cast(WalkTokensCallback, walk), self.defaults.walk_tokens)
            self.defaults = MarkedOptions(**{**self.defaults.as_mapping(), **updates})
        return self

    def set_options(self, options: MarkedOptions | Mapping[str, object] | None = None, **values: object) -> Marked:
        self.defaults = MarkedOptions(**{**self.defaults.as_mapping(), **option_values(options), **values})
        return self

    options = set_options

    def lexer(self, src: str, options: MarkedOptions | None = None) -> TokensList:
        return Lexer.lex(src, self.defaults if options is None else options)

    def parser(self, tokens: list[Token], options: MarkedOptions | None = None) -> str:
        return Parser.parse(tokens, self.defaults if options is None else options)

    def parse(self, src: str | None = None, options: MarkedOptions | Mapping[str, object] | None = None) -> object:
        return self._parse_markdown(src, options, True)

    def parse_inline(self, src: str | None = None, options: MarkedOptions | Mapping[str, object] | None = None) -> object:
        return self._parse_markdown(src, options, False)

    def _parse_markdown(self, src: object, options: MarkedOptions | Mapping[str, object] | None, block: bool) -> object:
        original = option_values(options)
        resolved = MarkedOptions(**{**self.defaults.as_mapping(), **original})
        error: Exception | None = None
        if self.defaults.async_ is True and original.get("async_") is False:
            error = RuntimeError("marked(): The async option was set to true by an extension. Remove async: false from the parse options object to return a Promise.")
        elif src is None:
            error = TypeError("marked(): input parameter is undefined or null")
        elif not isinstance(src, str):
            error = TypeError("marked(): input parameter is of type " + _input_type(src) + ", string expected")
        if error is not None:
            if resolved.async_:
                async def rejected() -> object:
                    return self._on_error(error, resolved.silent)
                return _start(rejected())
            return self._on_error(error, resolved.silent)
        source = cast(str, src)
        hooks = resolved.hooks
        if hooks is not None:
            hooks.options = resolved
            hooks.block = block
        if resolved.async_:
            return _start(self._parse_async(source, resolved, block))
        try:
            processed = hooks.preprocess(source) if hooks is not None else source
            lexer = hooks.provide_lexer(block) if hooks is not None else (Lexer.lex if block else Lexer.lex_inline)
            tokens = lexer(processed, resolved)
            if hooks is not None:
                tokens = hooks.process_all_tokens(tokens)
            if resolved.walk_tokens is not None:
                self.walk_tokens(tokens, resolved.walk_tokens)
            parser = hooks.provide_parser(block) if hooks is not None else (Parser.parse if block else Parser.parse_inline)
            html = parser(tokens, resolved)
            return hooks.postprocess(html) if hooks is not None else html
        except Exception as caught:
            return self._on_error(caught, resolved.silent)

    async def _parse_async(self, source: str, options: MarkedOptions, block: bool) -> object:
        try:
            hooks = options.hooks
            processed = await _resolve(hooks.preprocess(source)) if hooks is not None else source
            lexer = await _resolve(hooks.provide_lexer(block)) if hooks is not None else (Lexer.lex if block else Lexer.lex_inline)
            tokens = await _resolve(cast(Callable[..., object], lexer)(processed, options))
            processed_tokens = await _resolve(hooks.process_all_tokens(cast(list[Token], tokens))) if hooks is not None else tokens
            if options.walk_tokens is not None:
                await asyncio.gather(*(_resolve(value) for value in self.walk_tokens(cast(list[Token], processed_tokens), options.walk_tokens)))
            parser = await _resolve(hooks.provide_parser(block)) if hooks is not None else (Parser.parse if block else Parser.parse_inline)
            html = await _resolve(cast(Callable[..., object], parser)(processed_tokens, options))
            return await _resolve(hooks.postprocess(html)) if hooks is not None else html
        except Exception as caught:
            return self._on_error(caught, options.silent)

    @staticmethod
    def _on_error(error: Exception, silent: bool) -> str:
        message = str(error) + "\nPlease report this to https://github.com/markedjs/marked."
        error.args = (message, *error.args[1:])
        if silent:
            return "<p>An error occurred:</p><pre>" + escape_html_entities(message, True) + "</pre>"
        raise error


__all__ = ["Marked"]
