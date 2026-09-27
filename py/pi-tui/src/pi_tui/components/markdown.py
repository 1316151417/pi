"""Styled terminal Markdown, using the native Marked 18.0.11 lexer port."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from .._javascript import JS_WHITESPACE, js_repeat, js_trim, utf16_units
from .._markdown_lexer import Marked, Token, Tokenizer, TokenizerContext, TokenizerExtension
from .._marked_regex import Regex
from ..latex import RenderLatexOptions, render_latex
from ..terminal_image import get_capabilities, hyperlink, is_image_line
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


_STRICT_STRIKETHROUGH = Regex(r"^(~~)(?=[^\s~])((?:\\.|[^\\])*?(?:\\.|[^\s~\\]))\1(?=[^~]|$)")
_PENDING_DOLLAR_MATH = Regex(r"\\[A-Za-z]+|[_^=+*/<>()[\]|±≤≥≠≈∈→⇒∞∫∑√-]")


class _StrictStrikethroughTokenizer(Tokenizer):
    def del_(self, src: str, masked_src: str = "", prev_char: str = "") -> Token | None:
        match = _STRICT_STRIKETHROUGH.exec(src)
        if match is None:
            return None
        return Token("del", match[0], match[2], self.lexer._inline_tokens(match[2]))


def _find_closing_delimiter(source: str, closing: str, start: int) -> int:
    index = source.find(closing, start)
    while index >= 0:
        backslashes = 0
        position = index - 1
        while position >= 0 and source[position] == "\\":
            backslashes += 1
            position -= 1
        if backslashes % 2 == 0:
            break
        index = source.find(closing, index + len(closing))
    return index


def _tokenize_inline_latex(_context: TokenizerContext, source: str, _tokens: list[Token]) -> Token | None:
    source = utf16_units(source)
    if source.startswith("$$"):
        opening = closing = "$$"
    elif source.startswith("\\("):
        opening, closing = "\\(", "\\)"
    elif source.startswith("\\["):
        opening, closing = "\\[", "\\]"
    elif source.startswith("$") and not Regex(r"^\$\s").test(source):
        opening = closing = "$"
    else:
        return None
    closing_index = _find_closing_delimiter(source, closing, len(opening))
    if closing_index >= 0 and opening == "$":
        text = source[len(opening):closing_index]
        following = source[closing_index + 1:]
        if (Regex(r"\s$").test(text) or Regex(r"^\d").test(following)
                or (Regex(r"^[A-Z_][A-Z0-9_]*(?:[^A-Za-z0-9_\s])?$").test(text)
                    and Regex(r"^[A-Za-z_][A-Za-z0-9_]*").test(following)) or "`" in text):
            return None
    if closing_index < 0:
        pending = source[len(opening):]
        if opening.startswith("\\") or _PENDING_DOLLAR_MATH.test(pending):
            return Token("latex", source, pending, pending=True)
        return None
    text = source[len(opening):closing_index]
    if not text or "\n" in text:
        return None
    return Token("latex", source[:closing_index + len(closing)], text)


def _tokenize_block_latex(_context: TokenizerContext, source: str, _tokens: list[Token]) -> Token | None:
    source = utf16_units(source)
    for pattern in (r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*?)\$\$[ \t]*(?:\n|$)",
                    r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*?)\\\][ \t]*(?:\n|$)"):
        match = Regex(pattern).exec(source)
        if match is not None and match[1]:
            return Token("latexBlock", match[0], js_trim(match[1]))
    match = Regex(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*)$").exec(source)
    if match is not None:
        return Token("latexBlock", match[0], match[1], pending=True)
    match = Regex(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*)$").exec(source)
    if match is not None and match[1] and _PENDING_DOLLAR_MATH.test(match[1]):
        return Token("latexBlock", match[0], match[1], pending=True)
    return None


def _start_block_latex(_context: TokenizerContext, source: str) -> int | None:
    match = Regex(r"(?:^|\n) {0,3}(?:\$\$|\\\[)").exec(utf16_units(source))
    return None if match is None else match.index + (1 if match[0].startswith("\n") else 0)


def _start_inline_latex(_context: TokenizerContext, source: str) -> int | None:
    source = utf16_units(source)
    indices = [index for marker in ("$", "\\(", "\\[") if (index := source.find(marker)) >= 0]
    return min(indices) if indices else None


def _trim_partial_closing_fences(tokens: list[Token]) -> None:
    token = tokens[-1] if tokens else None
    if token is not None and token.type == "list":
        _trim_partial_closing_fences((token.items[-1].tokens or []) if token.items else [])
        return
    if token is not None and token.type == "blockquote":
        _trim_partial_closing_fences(token.tokens or [])
        return
    if token is None or token.type != "code":
        return
    marker = Regex(r"^(`{3,}|~{3,})").exec(token.raw)
    last_line = token.raw.split("\n")[-1]
    if marker is None or not last_line or len(last_line) >= len(marker[1]) or last_line != marker[1][0] * len(last_line):
        return
    token.text = (token.text or "")[:-len(last_line)].removesuffix("\n")


_markdown_parser = Marked().set_options(tokenizer=_StrictStrikethroughTokenizer())
_markdown_parser.use(extensions=[
    TokenizerExtension("latexBlock", "block", _tokenize_block_latex, _start_block_latex),
    TokenizerExtension("latex", "inline", _tokenize_inline_latex, _start_inline_latex),
])


@dataclass
class DefaultTextStyle:
    color: Callable[[str], str] | None = None
    bg_color: Callable[[str], str] | None = None
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False


@dataclass
class MarkdownTheme:
    heading: Callable[[str], str]
    link: Callable[[str], str]
    link_url: Callable[[str], str]
    code: Callable[[str], str]
    code_block: Callable[[str], str]
    code_block_border: Callable[[str], str]
    quote: Callable[[str], str]
    quote_border: Callable[[str], str]
    hr: Callable[[str], str]
    list_bullet: Callable[[str], str]
    bold: Callable[[str], str]
    italic: Callable[[str], str]
    strikethrough: Callable[[str], str]
    underline: Callable[[str], str]
    highlight_code: Callable[[str, str | None], list[str]] | None = None
    code_block_indent: str | None = None


@dataclass
class MarkdownOptions:
    preserve_ordered_list_markers: bool = False
    preserve_backslash_escapes: bool = False
    transform: Callable[[str, int], str | None] | None = None
    render_latex: bool | None = None


@dataclass
class _InlineStyleContext:
    apply_text: Callable[[str], str]
    style_prefix: str


class Markdown:
    def __init__(self, text: str, padding_x: int, padding_y: int, theme: MarkdownTheme,
                 default_text_style: DefaultTextStyle | None = None, options: MarkdownOptions | None = None) -> None:
        self._text = text
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._theme = theme
        self._default_text_style = default_text_style
        self._options = replace(options) if options is not None else MarkdownOptions()
        self._default_style_prefix: str | None = None
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self._text = text
        self.invalidate()

    def invalidate(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_text == self._text and self._cached_width == width:
            return self._cached_lines
        content_width = max(1, width - self._padding_x * 2)
        text = self._options.transform(self._text, content_width) if self._options.transform is not None else self._text
        if text is None:
            text = self._text
        if not text or not js_trim(text):
            self._cached_text, self._cached_width, self._cached_lines = self._text, width, []
            return self._cached_lines
        tokens = _markdown_parser.lexer(text.replace("\t", "   "))
        _trim_partial_closing_fences(tokens)
        rendered_lines: list[str] = []
        for index, token in enumerate(tokens):
            next_type = tokens[index + 1].type if index + 1 < len(tokens) else None
            rendered_lines.extend(self._render_token(token, content_width, next_type))
        wrapped_lines: list[str] = []
        for line in rendered_lines:
            wrapped_lines.extend([line] if is_image_line(line) else wrap_text_with_ansi(line, content_width))
        left_margin = js_repeat(" ", self._padding_x)
        right_margin = js_repeat(" ", self._padding_x)
        bg_fn = self._default_text_style.bg_color if self._default_text_style is not None else None
        content_lines: list[str] = []
        for line in wrapped_lines:
            if is_image_line(line):
                content_lines.append(line)
                continue
            line_with_margins = left_margin + line + right_margin
            content_lines.append(apply_background_to_line(line_with_margins, width, bg_fn) if bg_fn is not None else
                                 line_with_margins + js_repeat(" ", max(0, width - visible_width(line_with_margins))))
        empty_line = js_repeat(" ", width)
        empty_lines: list[str] = []
        index = 0
        while index < self._padding_y:
            empty_lines.append(apply_background_to_line(empty_line, width, bg_fn) if bg_fn is not None else empty_line)
            index += 1
        result = empty_lines + content_lines + empty_lines
        self._cached_text, self._cached_width, self._cached_lines = self._text, width, result
        return result if result else [""]

    def _apply_default_style(self, text: str) -> str:
        style = self._default_text_style
        if style is None:
            return text
        if style.color is not None:
            text = style.color(text)
        if style.bold:
            text = self._theme.bold(text)
        if style.italic:
            text = self._theme.italic(text)
        if style.strikethrough:
            text = self._theme.strikethrough(text)
        if style.underline:
            text = self._theme.underline(text)
        return text

    def _get_default_style_prefix(self) -> str:
        if self._default_text_style is None:
            return ""
        if self._default_style_prefix is None:
            styled = self._apply_default_style("\0")
            index = styled.find("\0")
            self._default_style_prefix = styled[:index] if index >= 0 else ""
        return self._default_style_prefix

    @staticmethod
    def _get_style_prefix(style_fn: Callable[[str], str]) -> str:
        styled = style_fn("\0")
        index = styled.find("\0")
        return styled[:index] if index >= 0 else ""

    def _render_token(self, token: Token, width: int, next_token_type: str | None = None,
                      style_context: _InlineStyleContext | None = None) -> list[str]:
        lines: list[str] = []
        spaced = bool(next_token_type and next_token_type != "space")
        if token.type == "heading":
            heading_style = (lambda value: self._theme.heading(self._theme.bold(self._theme.underline(value)))) if token.depth == 1 else (lambda value: self._theme.heading(self._theme.bold(value)))
            context = _InlineStyleContext(heading_style, self._get_style_prefix(heading_style))
            heading_text = self._render_inline_tokens(token.tokens or [], context)
            lines.append((heading_style(js_repeat("#", token.depth) + " ") if token.depth >= 3 else "") + heading_text)
            if spaced:
                lines.append("")
        elif token.type == "paragraph":
            lines.append(self._render_inline_tokens(token.tokens or [], style_context))
            if spaced and next_token_type != "list":
                lines.append("")
        elif token.type == "text":
            lines.append(self._render_inline_tokens([token], style_context))
        elif token.type == "latexBlock":
            rendered = render_latex(token.text or "", RenderLatexOptions(display=True)) if not token.pending and self._options.render_latex is not False else None
            if rendered is None:
                rendered = js_trim(token.raw)
            lines.extend(self._apply_default_style(line) for line in rendered.split("\n"))
            if spaced:
                lines.append("")
        elif token.type == "code":
            indent = self._theme.code_block_indent if self._theme.code_block_indent is not None else "  "
            lines.append(self._theme.code_block_border("```" + (token.lang or "")))
            if self._theme.highlight_code is not None:
                lines.extend(indent + line for line in self._theme.highlight_code(token.text or "", token.lang))
            else:
                lines.extend(indent + self._theme.code_block(line) for line in (token.text or "").split("\n"))
            lines.append(self._theme.code_block_border("```"))
            if spaced:
                lines.append("")
        elif token.type == "list":
            lines.extend(self._render_list(token, 0, width, style_context))
        elif token.type == "table":
            lines.extend(self._render_table(token, width, next_token_type, style_context))
        elif token.type == "blockquote":
            quote_style = lambda value: self._theme.quote(self._theme.italic(value))
            prefix = self._get_style_prefix(quote_style)
            quote_width = max(1, width - 2)
            context = _InlineStyleContext(lambda value: value, prefix)
            children = token.tokens or []
            quote_lines: list[str] = []
            for index, child in enumerate(children):
                next_type = children[index + 1].type if index + 1 < len(children) else None
                quote_lines.extend(self._render_token(child, quote_width, next_type, context))
            while quote_lines and quote_lines[-1] == "":
                quote_lines.pop()
            for line in quote_lines:
                styled = quote_style(line.replace("\x1b[0m", "\x1b[0m" + prefix) if prefix else line)
                lines.extend(self._theme.quote_border("│ ") + part for part in wrap_text_with_ansi(styled, quote_width))
            if spaced:
                lines.append("")
        elif token.type == "hr":
            lines.append(self._theme.hr(js_repeat("─", min(width, 80))))
            if spaced:
                lines.append("")
        elif token.type == "html":
            lines.append(self._apply_default_style(js_trim(token.raw)))
        elif token.type == "space":
            lines.append("")
        elif isinstance(token.text, str):
            lines.append(token.text)
        return lines

    def _render_inline_tokens(self, tokens: list[Token], style_context: _InlineStyleContext | None = None) -> str:
        context = style_context if style_context is not None else _InlineStyleContext(self._apply_default_style, self._get_default_style_prefix())
        result = ""
        for token in tokens:
            text: str | None = None
            if token.type == "latex":
                text = render_latex(token.text or "") if not token.pending and self._options.render_latex is not False else None
                if text is None:
                    text = token.raw
            elif token.type == "escape":
                text = token.raw if self._options.preserve_backslash_escapes else token.text or ""
            elif token.type == "text":
                if token.tokens:
                    result += self._render_inline_tokens(token.tokens, context)
                else:
                    text = token.text or ""
            elif token.type == "paragraph":
                result += self._render_inline_tokens(token.tokens or [], context)
            elif token.type in ("strong", "em", "del"):
                content = self._render_inline_tokens(token.tokens or [], context)
                style = {"strong": self._theme.bold, "em": self._theme.italic, "del": self._theme.strikethrough}[token.type]
                result += style(content) + context.style_prefix
            elif token.type == "codespan":
                result += self._theme.code(token.text or "") + context.style_prefix
            elif token.type == "link":
                link_text = self._render_inline_tokens(token.tokens or [], context)
                styled = self._theme.link(self._theme.underline(link_text))
                if get_capabilities().hyperlinks:
                    result += hyperlink(styled, token.href) + context.style_prefix
                else:
                    comparison = token.href[7:] if token.href.startswith("mailto:") else token.href
                    result += styled
                    if token.text != token.href and token.text != comparison:
                        result += self._theme.link_url(" (" + token.href + ")")
                    result += context.style_prefix
            elif token.type == "br":
                result += "\n"
            elif token.type == "html":
                text = token.raw
            elif isinstance(token.text, str):
                text = token.text
            if text is not None:
                result += "\n".join(context.apply_text(segment) for segment in text.split("\n"))
        while context.style_prefix and result.endswith(context.style_prefix):
            result = result[:-len(context.style_prefix)]
        return result

    def _render_list(self, token: Token, depth: int, width: int, style_context: _InlineStyleContext | None) -> list[str]:
        lines: list[str] = []
        indent = js_repeat("    ", depth)
        start = token.start if isinstance(token.start, (int, float)) and not isinstance(token.start, bool) else 1
        for index, item in enumerate(token.items):
            bullet = f"{start + index}. " if token.ordered else "- "
            if self._options.preserve_ordered_list_markers:
                pattern = r"^(?: {0,3})(\d{1,9}[.)])[ \t]+" if token.ordered else r"^(?: {0,3})([-+*])(?:[ \t]+|(?=\r?\n|$))"
                match = Regex(pattern).exec(item.raw)
                if match is not None:
                    bullet = match[1] + " "
            marker = bullet + (("[x] " if item.checked else "[ ] ") if item.task else "")
            first_prefix = indent + self._theme.list_bullet(marker)
            continuation = indent + js_repeat(" ", visible_width(marker))
            item_width = max(1, width - visible_width(first_prefix))
            rendered_any = False
            for child in item.tokens or []:
                if child.type == "list":
                    lines.extend(self._render_list(child, depth + 1, width, style_context))
                    rendered_any = True
                    continue
                for line in self._render_token(child, item_width, None, style_context):
                    for part in wrap_text_with_ansi(line, item_width):
                        lines.append((continuation if rendered_any else first_prefix) + part)
                        rendered_any = True
            if not rendered_any:
                lines.append(first_prefix)
            if token.loose and index < len(token.items) - 1:
                lines.append("")
        return lines

    @staticmethod
    def _get_longest_word_width(text: str, max_width: int | None = None) -> int:
        longest = max((visible_width(word) for word in re.split("[" + re.escape(JS_WHITESPACE) + "]+", text) if word), default=0)
        return longest if max_width is None else min(longest, max_width)

    @staticmethod
    def _wrap_cell_text(text: str, max_width: int, style_prefix: str = "") -> list[str]:
        lines = wrap_text_with_ansi(text, max(1, max_width))
        return [line + ("\x1b[22;23;24;25;27;28;29;39m" if index < len(lines) - 1 else "") + style_prefix for index, line in enumerate(lines)]

    def _render_table(self, token: Token, available_width: int, next_token_type: str | None,
                      style_context: _InlineStyleContext | None) -> list[str]:
        count = len(token.header)
        if not count:
            return []
        overhead = 3 * count + 1
        available_cells = available_width - overhead
        spaced = bool(next_token_type and next_token_type != "space")
        if available_cells < count:
            lines = wrap_text_with_ansi(token.raw, available_width) if token.raw else []
            if spaced:
                lines.append("")
            return lines
        natural: list[int] = []
        min_words: list[int] = []
        for cell in token.header:
            text = self._render_inline_tokens(cell.tokens, style_context)
            natural.append(visible_width(text))
            min_words.append(max(1, self._get_longest_word_width(text, 30)))
        for row in token.rows:
            for index, cell in enumerate(row):
                text = self._render_inline_tokens(cell.tokens, style_context)
                natural[index] = max(natural[index], visible_width(text))
                min_words[index] = max(min_words[index], self._get_longest_word_width(text, 30))
        minimum = min_words
        min_cells = sum(minimum)
        if min_cells > available_cells:
            minimum = [1] * count
            remaining = available_cells - count
            if remaining > 0:
                total_weight = sum(max(0, width - 1) for width in min_words)
                growth = [math.floor(max(0, width - 1) / total_weight * remaining) if total_weight > 0 else 0 for width in min_words]
                minimum = [width + growth[index] for index, width in enumerate(minimum)]
                leftover = remaining - sum(growth)
                index = 0
                while leftover > 0 and index < count:
                    minimum[index] += 1
                    leftover -= 1
                    index += 1
            min_cells = sum(minimum)
        if sum(natural) + overhead <= available_width:
            widths = [max(width, minimum[index]) for index, width in enumerate(natural)]
        else:
            total_growth = sum(max(0, width - minimum[index]) for index, width in enumerate(natural))
            extra = max(0, available_cells - min_cells)
            widths = [width + (math.floor(max(0, natural[index] - width) / total_growth * extra) if total_growth > 0 else 0) for index, width in enumerate(minimum)]
            remaining = available_cells - sum(widths)
            while remaining > 0:
                grew = False
                for index in range(count):
                    if remaining <= 0:
                        break
                    if widths[index] < natural[index]:
                        widths[index] += 1
                        remaining -= 1
                        grew = True
                if not grew:
                    break
        border_cells = [js_repeat("─", width) for width in widths]
        lines = ["┌─" + "─┬─".join(border_cells) + "─┐"]
        prefix = style_context.style_prefix if style_context is not None else ""
        header_lines = [self._wrap_cell_text(self._render_inline_tokens(cell.tokens, style_context), widths[index], prefix) for index, cell in enumerate(token.header)]
        for line_index in range(max(len(cell) for cell in header_lines)):
            parts: list[str] = []
            for col, cell_lines in enumerate(header_lines):
                text = cell_lines[line_index] if line_index < len(cell_lines) else ""
                parts.append(self._theme.bold(text + js_repeat(" ", max(0, widths[col] - visible_width(text)))))
            lines.append("│ " + " │ ".join(parts) + " │")
        separator = "├─" + "─┼─".join(border_cells) + "─┤"
        lines.append(separator)
        for row_index, row in enumerate(token.rows):
            cell_lines = [self._wrap_cell_text(self._render_inline_tokens(cell.tokens, style_context), widths[index], prefix) for index, cell in enumerate(row)]
            for line_index in range(max((len(cell) for cell in cell_lines), default=0)):
                parts = []
                for col, fragments in enumerate(cell_lines):
                    text = fragments[line_index] if line_index < len(fragments) else ""
                    parts.append(text + js_repeat(" ", max(0, widths[col] - visible_width(text))))
                lines.append("│ " + " │ ".join(parts) + " │")
            if row_index < len(token.rows) - 1:
                lines.append(separator)
        lines.append("└─" + "─┴─".join(border_cells) + "─┘")
        if spaced:
            lines.append("")
        return lines


__all__ = ["DefaultTextStyle", "Markdown", "MarkdownOptions", "MarkdownTheme"]
