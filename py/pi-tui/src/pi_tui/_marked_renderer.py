"""Marked 18.0.11 HTML and text renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._javascript import from_utf16_units
from ._marked_helpers import clean_url, escape_html_entities, js_string
from ._marked_rules import other
from ._marked_types import MarkedOptions, TableCell, TableRow, Token, current_defaults

if TYPE_CHECKING:
    from ._marked_parser import Parser


class Renderer:
    def __init__(self, options: MarkedOptions | None = None) -> None:
        self.options = options if options is not None else current_defaults()
        self.parser: Parser

    def space(self, token: Token) -> str:
        return ""

    def code(self, token: Token) -> str:
        match = other["notSpaceStart"].exec(token.lang or "")
        lang = from_utf16_units(match[0]) if match is not None else ""
        code = (token.text or "").removesuffix("\n") + "\n"
        content = code if token.escaped else escape_html_entities(code, True)
        language = ' class="language-' + escape_html_entities(lang) + '"' if lang else ""
        return "<pre><code" + language + ">" + content + "</code></pre>\n"

    def blockquote(self, token: Token) -> str:
        return "<blockquote>\n" + self.parser.parse(token.tokens or []) + "</blockquote>\n"

    def html(self, token: Token) -> str:
        return token.text or ""

    def def_(self, token: Token) -> str:
        return ""

    def heading(self, token: Token) -> str:
        return f"<h{token.depth}>" + self.parser.parse_inline(token.tokens or []) + f"</h{token.depth}>\n"

    def hr(self, token: Token) -> str:
        return "<hr>\n"

    def list(self, token: Token) -> str:
        body = "".join(js_string(self.listitem(item)) for item in token.items)
        tag = "ol" if token.ordered else "ul"
        start = ' start="' + js_string(token.start) + '"' if token.ordered and token.start != 1 else ""
        return "<" + tag + start + ">\n" + body + "</" + tag + ">\n"

    def listitem(self, token: Token) -> str:
        return "<li>" + self.parser.parse(token.tokens or []) + "</li>\n"

    def checkbox(self, token: Token) -> str:
        return "<input " + ('checked="" ' if token.checked else "") + 'disabled="" type="checkbox"> '

    def paragraph(self, token: Token) -> str:
        return "<p>" + self.parser.parse_inline(token.tokens or []) + "</p>\n"

    def table(self, token: Token) -> str:
        header = js_string(self.tablerow(TableRow("".join(js_string(self.tablecell(cell)) for cell in token.header))))
        body = "".join(js_string(self.tablerow(TableRow("".join(js_string(self.tablecell(cell)) for cell in row)))) for row in token.rows)
        if body:
            body = "<tbody>" + body + "</tbody>"
        return "<table>\n<thead>\n" + header + "</thead>\n" + body + "</table>\n"

    def tablerow(self, token: TableRow) -> str:
        return "<tr>\n" + js_string(token.text) + "</tr>\n"

    def tablecell(self, token: TableCell) -> str:
        content = self.parser.parse_inline(token.tokens)
        tag = "th" if token.header else "td"
        start = "<" + tag + (' align="' + token.align + '"' if token.align else "") + ">"
        return start + content + "</" + tag + ">\n"

    def strong(self, token: Token) -> str:
        return "<strong>" + self.parser.parse_inline(token.tokens or []) + "</strong>"

    def em(self, token: Token) -> str:
        return "<em>" + self.parser.parse_inline(token.tokens or []) + "</em>"

    def codespan(self, token: Token) -> str:
        return "<code>" + escape_html_entities(token.text or "", True) + "</code>"

    def br(self, token: Token) -> str:
        return "<br>"

    def del_(self, token: Token) -> str:
        return "<del>" + self.parser.parse_inline(token.tokens or []) + "</del>"

    def link(self, token: Token) -> str:
        text = self.parser.parse_inline(token.tokens or [])
        href = clean_url(token.href)
        if href is None:
            return text
        title = ' title="' + escape_html_entities(token.title) + '"' if token.title else ""
        return '<a href="' + href + '"' + title + ">" + text + "</a>"

    def image(self, token: Token) -> str:
        text = self.parser.parse_inline(token.tokens, self.parser.text_renderer) if token.tokens is not None else token.text or ""
        href = clean_url(token.href)
        if href is None:
            return escape_html_entities(text)
        title = ' title="' + escape_html_entities(token.title) + '"' if token.title else ""
        return '<img src="' + href + '" alt="' + escape_html_entities(text) + '"' + title + ">"

    def text(self, token: Token) -> str:
        if token.tokens is not None:
            return self.parser.parse_inline(token.tokens)
        return (token.text or "") if token.escaped else escape_html_entities(token.text or "")


class TextRenderer:
    def strong(self, token: Token) -> str:
        return token.text or ""

    def em(self, token: Token) -> str:
        return token.text or ""

    def codespan(self, token: Token) -> str:
        return token.text or ""

    def del_(self, token: Token) -> str:
        return token.text or ""

    def html(self, token: Token) -> str:
        return token.text or ""

    def text(self, token: Token) -> str:
        return token.text or ""

    def link(self, token: Token) -> str:
        return token.text or ""

    def image(self, token: Token) -> str:
        return token.text or ""

    def br(self, token: Token | None = None) -> str:
        return ""

    def checkbox(self, token: Token) -> str:
        return token.raw


__all__ = ["Renderer", "TextRenderer"]
