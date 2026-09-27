"""Native port of Marked 18.0.11 Lexer and Tokenizer.

See _marked_regex.py for upstream licenses. Internal scanner strings use UTF-16
units; public lexer results are Python Unicode strings.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MethodType
from typing import Literal, Protocol

from icu import Locale, UnicodeString

from . import _marked_rules as grammar
from ._javascript import JS_WHITESPACE, from_utf16_units, js_trim, utf16_length, utf16_units
from ._marked_regex import Match, Regex
from ._marked_types import (
    LinkDefinition, MarkedOptions, TableCell, Token, TokenizerContext,
    TokensList, current_defaults,
)


class _LexCallable(Protocol):
    def __call__(self, src: str, options: MarkedOptions | None = None) -> TokensList: ...


class _LexMethod:
    """Keep Marked's separate static and instance methods named lex."""

    def __get__(self, instance: Lexer | None, owner: type[Lexer]) -> _LexCallable:
        def lex(src: str, options: MarkedOptions | None = None) -> TokensList:
            return (owner(options) if instance is None else instance)._lex(src)
        return lex


class _RulesDescriptor:
    def __get__(self, instance: Lexer | None, owner: type[Lexer]) -> dict[str, dict[str, dict[str, Regex]]]:
        return {"block": grammar.block, "inline": grammar.inline}


@dataclass
class _LexerState:
    in_link: bool = False
    in_raw_block: bool = False
    link_emitted: bool = False
    top: bool = True


@dataclass
class _InlineQueueItem:
    src: str
    tokens: list[Token]


@dataclass
class _Rules:
    block: dict[str, Regex]
    inline: dict[str, Regex]
    other: dict[str, Regex] = field(default_factory=lambda: grammar.other)


def _map_token_strings(tokens: list[Token], convert: Callable[[str], str]) -> None:
    for token in tokens:
        token.raw = convert(token.raw)
        if token.text is not None:
            token.text = convert(token.text)
        token.href = convert(token.href)
        token.tag = convert(token.tag)
        if token.lang is not None:
            token.lang = convert(token.lang)
        if token.title is not None:
            token.title = convert(token.title)
        if token.tokens is not None:
            _map_token_strings(token.tokens, convert)
        _map_token_strings(token.items, convert)
        for row in (token.header, *token.rows):
            for cell in row:
                cell.text = convert(cell.text)
                _map_token_strings(cell.tokens, convert)


def _split_cells(table_row: str, count: int | None = None) -> list[str]:
    row = ""
    for offset, char in enumerate(table_row):
        if char != "|":
            row += char
            continue
        escaped = False
        current = offset - 1
        while current >= 0 and table_row[current] == "\\":
            escaped = not escaped
            current -= 1
        row += "|" if escaped else " |"
    cells = row.split(" |")
    if not js_trim(cells[0]):
        cells.pop(0)
    if cells and not js_trim(cells[-1]):
        cells.pop()
    if count:
        cells = cells[:count]
        cells.extend("" for _ in range(count - len(cells)))
    return [js_trim(cell).replace("\\|", "|") for cell in cells]


def _trim_trailing_blank_lines(text: str) -> str:
    lines = text.split("\n")
    end = len(lines) - 1
    while end >= 0 and grammar.other["blankLine"].test(lines[end]):
        end -= 1
    return text if len(lines) - end <= 2 else "\n".join(lines[:end + 1])


def _find_closing_bracket(text: str, brackets: str) -> int:
    if brackets[1] not in text:
        return -1
    level = 0
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 1
        elif text[index] == brackets[0]:
            level += 1
        elif text[index] == brackets[1]:
            level -= 1
            if level < 0:
                return index
        index += 1
    return -2 if level > 0 else -1


def _expand_tabs(line: str, indent: int = 0) -> str:
    column = indent
    expanded = ""
    for char in from_utf16_units(line):
        if char == "\t":
            added = 4 - column % 4
            expanded += " " * added
            column += added
        else:
            expanded += char
            column += 1
    return utf16_units(expanded)


def _lower(value: str) -> str:
    lowered = UnicodeString(utf16_units(value)).toLower(Locale.getRoot())
    # Read units separately: PyICU's UCS4 conversion rejects a string that
    # combines a non-BMP code point with an unpaired surrogate.
    return from_utf16_units("".join(lowered[index] for index in range(len(lowered))))


class Tokenizer:
    def __init__(self, options: MarkedOptions | None = None) -> None:
        self.options = options if options is not None else current_defaults()
        self.lexer: Lexer
        self.rules: _Rules

    def space(self, src: str) -> Token | None:
        cap = self.rules.block["newline"].exec(src)
        return Token("space", cap[0]) if cap is not None and cap[0] else None

    def code(self, src: str) -> Token | None:
        cap = self.rules.block["code"].exec(src)
        if cap is None:
            return None
        raw = cap[0] if self.options.pedantic else _trim_trailing_blank_lines(cap[0])
        return Token("code", raw, self.rules.other["codeRemoveIndent"].replace(raw, ""), code_block_style="indented")

    def fences(self, src: str) -> Token | None:
        cap = self.rules.block["fences"].exec(src)
        if cap is None:
            return None
        text = cap[3]
        indent_to_code = self.rules.other["indentCodeCompensation"].exec(cap[0])
        if indent_to_code is not None:
            lines: list[str] = []
            for node in text.split("\n"):
                indent_in_node = self.rules.other["beginningSpace"].exec(node)
                lines.append(node[len(indent_to_code[1]):] if indent_in_node is not None and len(indent_in_node[0]) >= len(indent_to_code[1]) else node)
            text = "\n".join(lines)
        lang = self.rules.inline["anyPunctuation"].replace(js_trim(cap[2]), "$1") if cap[2] else cap.group(2)
        return Token("code", cap[0], text, lang=lang)

    def heading(self, src: str) -> Token | None:
        cap = self.rules.block["heading"].exec(src)
        if cap is None:
            return None
        text = js_trim(cap[2])
        if self.rules.other["endingHash"].test(text):
            trimmed = text.rstrip("#")
            if self.options.pedantic or not trimmed or self.rules.other["endingSpaceChar"].test(trimmed):
                text = js_trim(trimmed)
        return Token("heading", cap[0].rstrip("\n"), text, self.lexer.inline(text), depth=len(cap[1]))

    def hr(self, src: str) -> Token | None:
        cap = self.rules.block["hr"].exec(src)
        return Token("hr", cap[0].rstrip("\n")) if cap is not None else None

    def blockquote(self, src: str) -> Token | None:
        cap = self.rules.block["blockquote"].exec(src)
        if cap is None:
            return None
        lines = cap[0].rstrip("\n").split("\n")
        raw = ""
        text = ""
        tokens: list[Token] = []
        while lines:
            in_blockquote = False
            current_lines: list[str] = []
            index = 0
            while index < len(lines):
                if self.rules.other["blockquoteStart"].test(lines[index]):
                    current_lines.append(lines[index])
                    in_blockquote = True
                elif not in_blockquote:
                    current_lines.append(lines[index])
                else:
                    break
                index += 1
            lines = lines[index:]
            current_raw = "\n".join(current_lines)
            current_text = self.rules.other["blockquoteSetextReplace2"].replace(
                self.rules.other["blockquoteSetextReplace"].replace(current_raw, "\n    $1"), ""
            )
            raw = raw + "\n" + current_raw if raw else current_raw
            text = text + "\n" + current_text if text else current_text
            top = self.lexer.state.top
            self.lexer.state.top = True
            self.lexer._block_tokens(current_text, tokens, True)
            self.lexer.state.top = top
            if not lines:
                break
            last_token = tokens[-1] if tokens else None
            if last_token is not None and last_token.type == "code":
                break
            if last_token is not None and last_token.type == "blockquote":
                continuation = "\n".join(lines)
                new_text = last_token.raw + "\n" + self.rules.other["blockquoteSetextReplace2"].replace(continuation, "")
                new_token = self.blockquote(new_text)
                if new_token is None:
                    raise RuntimeError("Marked blockquote continuation did not produce a blockquote")
                tokens[-1] = new_token
                raw += "\n" + continuation
                text = text[:len(text) - len(last_token.text or "")] + (new_token.text or "")
                break
            if last_token is not None and last_token.type == "list":
                new_text = last_token.raw + "\n" + "\n".join(lines)
                new_token = self.list(new_text)
                if new_token is None:
                    raise RuntimeError("Marked list continuation did not produce a list")
                tokens[-1] = new_token
                raw = raw[:len(raw) - len(last_token.raw)] + new_token.raw
                text = text[:len(text) - len(last_token.raw)] + new_token.raw
                lines = new_text[len(new_token.raw):].split("\n")
        return Token("blockquote", raw, text, tokens)

    def list(self, src: str) -> Token | None:
        cap = self.rules.block["list"].exec(src)
        if cap is None:
            return None
        bullet = js_trim(cap[1])
        is_ordered = len(bullet) > 1
        result = Token("list", "", ordered=is_ordered, start=int(bullet[:-1]) if is_ordered else "")
        bullet = r"\d{1,9}" + "\\" + bullet[-1] if is_ordered else "\\" + bullet
        if self.options.pedantic and not is_ordered:
            bullet = "[*+-]"
        item_regex = Regex(r"^( {0,3}" + bullet + r")((?:[\t ][^\n]*)?(?:\n|$))")
        ends_with_blank_line = False
        while src:
            cap = item_regex.exec(src)
            if cap is None or self.rules.block["hr"].test(src):
                break
            end_early = False
            item_contents = ""
            raw = cap[0]
            src = src[len(raw):]
            line = _expand_tabs(cap[2].split("\n", 1)[0], len(cap[1]))
            next_line = src.split("\n", 1)[0]
            blank_line = not js_trim(line)
            if self.options.pedantic:
                indent = 2
                item_contents = line.lstrip(JS_WHITESPACE)
            elif blank_line:
                indent = len(cap[1]) + 1
            else:
                non_space = self.rules.other["nonSpaceChar"].exec(line)
                indent = non_space.index if non_space is not None else -1
                indent = 1 if indent > 4 else indent
                item_contents = line[indent:]
                indent += len(cap[1])
            if blank_line and self.rules.other["blankLine"].test(next_line):
                raw += next_line + "\n"
                src = src[len(next_line) + 1:]
                end_early = True
            if not end_early:
                limit = max(0, min(3, indent - 1))
                prefix = "^ {0," + str(limit) + "}"
                next_bullet = Regex(prefix + r"(?:[*+-]|\d{1,9}[.)])((?:[ \t][^\n]*)?(?:\n|$))")
                hr = Regex(prefix + r"((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)")
                fences = Regex(prefix + r"(?:```|~~~)")
                heading = Regex(prefix + "#")
                html = Regex(prefix + r"<(?:[a-z].*>|!--)", "i")
                blockquote = Regex(prefix + ">")
                while src:
                    raw_line = src.split("\n", 1)[0]
                    next_line = self.rules.other["listReplaceNesting"].replace(raw_line, "  ") if self.options.pedantic else raw_line
                    next_without_tabs = next_line if self.options.pedantic else next_line.replace("\t", "    ")
                    if any(rule.test(next_line) for rule in (fences, heading, html, blockquote, next_bullet, hr)):
                        break
                    non_space = self.rules.other["nonSpaceChar"].exec(next_without_tabs)
                    if (non_space.index if non_space is not None else -1) >= indent or not js_trim(next_line):
                        item_contents += "\n" + next_without_tabs[indent:]
                    else:
                        if blank_line:
                            break
                        previous_non_space = self.rules.other["nonSpaceChar"].exec(line.replace("\t", "    "))
                        if (previous_non_space.index if previous_non_space is not None else -1) >= 4:
                            break
                        if any(rule.test(line) for rule in (fences, heading, hr)):
                            break
                        item_contents += "\n" + next_line
                    blank_line = not js_trim(next_line)
                    raw += raw_line + "\n"
                    src = src[len(raw_line) + 1:]
                    line = next_without_tabs[indent:]
            if not result.loose:
                if ends_with_blank_line:
                    result.loose = True
                elif self.rules.other["doubleBlankLine"].test(raw):
                    ends_with_blank_line = True
            result.items.append(Token("list_item", raw, item_contents, [], task=self.options.gfm and self.rules.other["listIsTask"].test(item_contents)))
            result.raw += raw
        if not result.items:
            return None
        result.items[-1].raw = result.items[-1].raw.rstrip(JS_WHITESPACE)
        result.items[-1].text = (result.items[-1].text or "").rstrip(JS_WHITESPACE)
        result.raw = result.raw.rstrip(JS_WHITESPACE)
        for item in result.items:
            self.lexer.state.top = False
            item.tokens = self.lexer._block_tokens(item.text or "", [])
            if not result.loose:
                result.loose = any(token.type == "space" and self.rules.other["anyLine"].test(token.raw) for token in item.tokens)
        for item in result.items:
            item_tokens = item.tokens if item.tokens is not None else []
            item_token = item_tokens[0] if item_tokens else None
            if item.task and item_token is not None and item_token.type in ("text", "paragraph"):
                item.text = self.rules.other["listReplaceTask"].replace(item.text or "", "")
                item_token.raw = self.rules.other["listReplaceTask"].replace(item_token.raw, "")
                item_token.text = self.rules.other["listReplaceTask"].replace(item_token.text or "", "")
                for queued in reversed(self.lexer.inline_queue):
                    if self.rules.other["listIsTask"].test(queued.src):
                        queued.src = self.rules.other["listReplaceTask"].replace(queued.src, "")
                        break
                task_raw = self.rules.other["listTaskCheckbox"].exec(item.raw)
                if task_raw is not None:
                    checkbox = Token("checkbox", task_raw[0] + " ", checked=task_raw[0] != "[ ]")
                    item.checked = checkbox.checked
                    if result.loose:
                        if item_tokens and item_tokens[0].type in ("paragraph", "text") and item_tokens[0].tokens is not None:
                            item_tokens[0].raw = checkbox.raw + item_tokens[0].raw
                            item_tokens[0].text = checkbox.raw + (item_tokens[0].text or "")
                            item_tokens[0].tokens.insert(0, checkbox)
                        else:
                            item_tokens.insert(0, Token("paragraph", checkbox.raw, checkbox.raw, [checkbox]))
                    else:
                        item_tokens.insert(0, checkbox)
            elif item.task:
                item.task = False
        if result.loose:
            for item in result.items:
                item.loose = True
                for token in item.tokens or []:
                    if token.type == "text":
                        token.type = "paragraph"
        return result

    def html(self, src: str) -> Token | None:
        cap = self.rules.block["html"].exec(src)
        if cap is None:
            return None
        raw = _trim_trailing_blank_lines(cap[0])
        return Token("html", raw, raw, pre=cap[1] in ("pre", "script", "style"), block=True)

    def def_(self, src: str) -> Token | None:
        cap = self.rules.block["def"].exec(src)
        if cap is None:
            return None
        tag = from_utf16_units(self.rules.other["multipleSpaceGlobal"].replace(_lower(cap[1]), " "))
        href = self.rules.inline["anyPunctuation"].replace(self.rules.other["hrefBrackets"].replace(cap[2], "$1"), "$1") if cap[2] else ""
        title = self.rules.inline["anyPunctuation"].replace(cap[3][1:-1], "$1") if cap[3] else cap.group(3)
        return Token("def", cap[0].rstrip("\n"), tag=tag, href=href, title=title)

    def table(self, src: str) -> Token | None:
        cap = self.rules.block["table"].exec(src)
        if cap is None or not self.rules.other["tableDelimiter"].test(cap[2]):
            return None
        headers = _split_cells(cap[1])
        aligns = self.rules.other["tableAlignChars"].replace(cap[2], "").split("|")
        rows = self.rules.other["tableRowBlankLine"].replace(cap[3], "").split("\n") if js_trim(cap[3]) else []
        if len(headers) != len(aligns):
            return None
        result = Token("table", cap[0].rstrip("\n"))
        for align in aligns:
            result.align.append(
                "right" if self.rules.other["tableAlignRight"].test(align)
                else "center" if self.rules.other["tableAlignCenter"].test(align)
                else "left" if self.rules.other["tableAlignLeft"].test(align) else None
            )
        result.header = [TableCell(text, self.lexer.inline(text), True, result.align[index]) for index, text in enumerate(headers)]
        result.rows = [[TableCell(text, self.lexer.inline(text), False, result.align[index]) for index, text in enumerate(_split_cells(row, len(headers)))] for row in rows]
        return result

    def lheading(self, src: str) -> Token | None:
        cap = self.rules.block["lheading"].exec(src)
        if cap is None:
            return None
        text = js_trim(cap[1])
        return Token("heading", cap[0].rstrip("\n"), text, self.lexer.inline(text), depth=1 if cap[2].startswith("=") else 2)

    def paragraph(self, src: str) -> Token | None:
        cap = self.rules.block["paragraph"].exec(src)
        if cap is None:
            return None
        text = cap[1][:-1] if cap[1].endswith("\n") else cap[1]
        return Token("paragraph", cap[0], text, self.lexer.inline(text))

    def text(self, src: str) -> Token | None:
        cap = self.rules.block["text"].exec(src)
        return Token("text", cap[0], cap[0], self.lexer.inline(cap[0])) if cap is not None else None

    def escape(self, src: str) -> Token | None:
        cap = self.rules.inline["escape"].exec(src)
        return Token("escape", cap[0], cap[1]) if cap is not None else None

    def tag(self, src: str) -> Token | None:
        cap = self.rules.inline["tag"].exec(src)
        if cap is None:
            return None
        if not self.lexer.state.in_link and self.rules.other["startATag"].test(cap[0]):
            self.lexer.state.in_link = True
        elif self.lexer.state.in_link and self.rules.other["endATag"].test(cap[0]):
            self.lexer.state.in_link = False
        if not self.lexer.state.in_raw_block and self.rules.other["startPreScriptTag"].test(cap[0]):
            self.lexer.state.in_raw_block = True
        elif self.lexer.state.in_raw_block and self.rules.other["endPreScriptTag"].test(cap[0]):
            self.lexer.state.in_raw_block = False
        return Token("html", cap[0], cap[0], block=False, in_link=self.lexer.state.in_link, in_raw_block=self.lexer.state.in_raw_block)

    def _output_link(self, cap: Match, link: LinkDefinition, raw: str) -> Token | None:
        text = self.rules.other["outputLinkReplace"].replace(cap[1], "$1")
        is_image = cap[0].startswith("!")
        self.lexer.state.in_link = True
        outer_link_emitted = self.lexer.state.link_emitted
        outer_raw_block = self.lexer.state.in_raw_block
        self.lexer.state.link_emitted = False
        tokens = self.lexer._inline_tokens(text)
        text_has_link = self.lexer.state.link_emitted
        self.lexer.state.link_emitted = outer_link_emitted
        self.lexer.state.in_link = False
        if not is_image:
            if text_has_link:
                self.lexer.state.in_raw_block = outer_raw_block
                return None
            self.lexer.state.link_emitted = True
        return Token("image" if is_image else "link", raw, text, tokens, href=link.href, title=link.title or None)

    def link(self, src: str) -> Token | None:
        cap = self.rules.inline["link"].exec(src)
        if cap is None:
            return None
        trimmed_url = js_trim(cap[2])
        if not self.options.pedantic and trimmed_url.startswith("<"):
            if not trimmed_url.endswith(">"):
                return None
            rtrim_slash = trimmed_url[:-1].rstrip("\\")
            if (len(trimmed_url) - len(rtrim_slash)) % 2 == 0:
                return None
        else:
            last_paren_index = _find_closing_bracket(cap[2], "()")
            if last_paren_index == -2:
                return None
            if last_paren_index > -1:
                start = 5 if cap[0].startswith("!") else 4
                link_len = start + len(cap[1]) + last_paren_index
                cap[2] = cap[2][:last_paren_index]
                cap[0] = js_trim(cap[0][:link_len])
                cap[3] = ""
        href = cap[2]
        title = ""
        if self.options.pedantic:
            link = self.rules.other["pedanticHrefTitle"].exec(href)
            if link is not None:
                href, title = link[1], link[3]
        else:
            title = cap[3][1:-1] if cap[3] else ""
        href = js_trim(href)
        if href.startswith("<"):
            href = href[1:] if self.options.pedantic and not trimmed_url.endswith(">") else href[1:-1]
        href = self.rules.inline["anyPunctuation"].replace(href, "$1") if href else href
        title = self.rules.inline["anyPunctuation"].replace(title, "$1") if title else title
        return self._output_link(cap, LinkDefinition(href, title), cap[0])

    def reflink(self, src: str, links: dict[str, LinkDefinition]) -> Token | None:
        cap = self.rules.inline["reflink"].exec(src) or self.rules.inline["nolink"].exec(src)
        if cap is None:
            return None
        link_string = self.rules.other["multipleSpaceGlobal"].replace(cap[2] or cap[1], " ")
        link = links.get(_lower(link_string))
        if link is None:
            return Token("text", cap[0][:1], cap[0][:1])
        return self._output_link(cap, link, cap[0])

    def em_strong(self, src: str, masked_src: str, prev_char: str = "") -> Token | None:
        match = self.rules.inline["emStrongLDelim"].exec(src)
        if match is None or not any(match[index] for index in (1, 2, 3, 4)):
            return None
        if match[4] and self.rules.other["unicodeAlphaNumeric"].test(prev_char):
            return None
        next_char = match[1] or match[3]
        if next_char and prev_char and not self.rules.inline["punctuation"].test(prev_char):
            return None
        left_length = len(from_utf16_units(match[0])) - 1
        delim_total = left_length
        mid_delim_total = 0
        delim_char = match[0][0]
        mid_run = prev_char == delim_char
        end_rule = self.rules.inline["emStrongRDelimAst" if delim_char == "*" else "emStrongRDelimUnd"]
        end_rule.last_index = 0
        masked_src = masked_src[-len(src) + left_length:]
        while (match := end_rule.exec(masked_src)) is not None:
            right_delim = next((match[index] for index in range(1, 7) if match[index]), "")
            if not right_delim:
                continue
            right_length = len(from_utf16_units(right_delim))
            if match[3] or match[4]:
                delim_total += right_length
                continue
            if match[5] or match[6]:
                if left_length % 3 and (left_length + right_length) % 3 == 0:
                    mid_delim_total += right_length
                    continue
                if mid_run:
                    break
            delim_total -= right_length
            if delim_total > 0:
                continue
            right_length = min(right_length, right_length + delim_total + mid_delim_total)
            last_char_length = utf16_length(from_utf16_units(match[0])[0])
            raw = src[:left_length + match.index + last_char_length + right_length]
            if min(left_length, right_length) % 2:
                text = raw[1:-1]
                return Token("em", raw, text, self.lexer._inline_tokens(text))
            text = raw[2:-2]
            return Token("strong", raw, text, self.lexer._inline_tokens(text))
        return None

    def codespan(self, src: str) -> Token | None:
        cap = self.rules.inline["code"].exec(src)
        if cap is None:
            return None
        text = cap[2].replace("\n", " ")
        if self.rules.other["nonSpaceChar"].test(text) and text.startswith(" ") and text.endswith(" "):
            text = text[1:-1]
        return Token("codespan", cap[0], text)

    def br(self, src: str) -> Token | None:
        cap = self.rules.inline["br"].exec(src)
        return Token("br", cap[0]) if cap is not None else None

    def del_(self, src: str, masked_src: str, prev_char: str = "") -> Token | None:
        match = self.rules.inline["delLDelim"].exec(src)
        if match is None:
            return None
        if match[1] and prev_char and not self.rules.inline["punctuation"].test(prev_char):
            return None
        left_length = len(from_utf16_units(match[0])) - 1
        delim_total = left_length
        end_rule = self.rules.inline["delRDelim"]
        end_rule.last_index = 0
        masked_src = masked_src[-len(src) + left_length:]
        while (match := end_rule.exec(masked_src)) is not None:
            right_delim = next((match[index] for index in range(1, 7) if match[index]), "")
            if not right_delim:
                continue
            right_length = len(from_utf16_units(right_delim))
            if right_length != left_length:
                continue
            if match[3] or match[4]:
                delim_total += right_length
                continue
            delim_total -= right_length
            if delim_total > 0:
                continue
            right_length = min(right_length, right_length + delim_total)
            last_char_length = utf16_length(from_utf16_units(match[0])[0])
            raw = src[:left_length + match.index + last_char_length + right_length]
            text = raw[left_length:-left_length]
            return Token("del", raw, text, self.lexer._inline_tokens(text))
        return None

    def autolink(self, src: str) -> Token | None:
        cap = self.rules.inline["autolink"].exec(src)
        if cap is None:
            return None
        text = cap[1]
        href = "mailto:" + text if cap[2] == "@" else text
        return Token("link", cap[0], text, [Token("text", text, text)], href=href)

    def url(self, src: str) -> Token | None:
        cap = self.rules.inline["url"].exec(src)
        if cap is None:
            return None
        if cap[2] == "@":
            text, href = cap[0], "mailto:" + cap[0]
        else:
            while True:
                previous = cap[0]
                backpedal = self.rules.inline["_backpedal"].exec(cap[0])
                cap[0] = backpedal[0] if backpedal is not None else ""
                if previous == cap[0]:
                    break
            text = cap[0]
            href = "http://" + text if cap[1] == "www." else text
        return Token("link", cap[0], text, [Token("text", text, text)], href=href)

    def inline_text(self, src: str) -> Token | None:
        cap = self.rules.inline["text"].exec(src)
        return Token("text", cap[0], cap[0], escaped=self.lexer.state.in_raw_block) if cap is not None else None


class Lexer:
    rules = _RulesDescriptor()
    lex = _LexMethod()

    def __init__(self, options: MarkedOptions | None = None) -> None:
        self.tokens = TokensList()
        self.options = options if options is not None else current_defaults()
        if self.options.tokenizer is None:
            self.options.tokenizer = Tokenizer()
        self.tokenizer = self.options.tokenizer
        self.tokenizer.options = self.options
        self.tokenizer.lexer = self
        self.inline_queue: list[_InlineQueueItem] = []
        self.state = _LexerState()
        mode = "pedantic" if self.options.pedantic else "gfm" if self.options.gfm else "normal"
        inline_mode = "breaks" if mode == "gfm" and self.options.breaks else mode
        self.tokenizer.rules = _Rules(grammar.block[mode], grammar.inline[inline_mode])

    def _lex(self, src: str) -> TokensList:
        source = grammar.other["carriageReturn"].replace(utf16_units(src), "\n")
        self._block_tokens(source, self.tokens)
        index = 0
        while index < len(self.inline_queue):
            queued = self.inline_queue[index]
            self._inline_tokens(queued.src, queued.tokens)
            index += 1
        self.inline_queue = []
        _map_token_strings(self.tokens, from_utf16_units)
        for link in self.tokens.links.values():
            link.href = from_utf16_units(link.href)
            if link.title is not None:
                link.title = from_utf16_units(link.title)
        return self.tokens

    @classmethod
    def lex_inline(cls, src: str, options: MarkedOptions | None = None) -> list[Token]:
        return cls(options).inline_tokens(src)

    def block_tokens(self, src: str, tokens: list[Token] | None = None, last_paragraph_clipped: bool = False) -> list[Token]:
        result = self._block_tokens(utf16_units(src), [] if tokens is None else tokens, last_paragraph_clipped)
        _map_token_strings(result, from_utf16_units)
        return result

    def _block_tokens(self, src: str, tokens: list[Token], last_paragraph_clipped: bool = False) -> list[Token]:
        src = utf16_units(src)
        self.tokenizer.lexer = self
        if self.options.pedantic:
            src = grammar.other["spaceLine"].replace(src.replace("\t", "    "), "")
        source_length = len(src) + 1
        while src:
            if len(src) >= source_length:
                self._infinite_loop_error(ord(src[0]))
                break
            source_length = len(src)
            token = self._extension_token("block", src, tokens)
            if token is not None:
                src = src[utf16_length(token.raw):]
                tokens.append(token)
                continue
            token = self.tokenizer.space(src)
            if token is not None:
                src = src[utf16_length(token.raw):]
                if utf16_length(token.raw) == 1 and tokens:
                    tokens[-1].raw += "\n"
                else:
                    tokens.append(token)
                continue
            token = self.tokenizer.code(src)
            if token is not None:
                src = src[utf16_length(token.raw):]
                if tokens and tokens[-1].type in ("paragraph", "text"):
                    last = tokens[-1]
                    last.raw += ("" if last.raw.endswith("\n") else "\n") + token.raw
                    last.text = (last.text or "") + "\n" + (token.text or "")
                    self.inline_queue[-1].src = last.text
                else:
                    tokens.append(token)
                continue
            token = None
            for tokenize in (self.tokenizer.fences, self.tokenizer.heading, self.tokenizer.hr, self.tokenizer.blockquote, self.tokenizer.list, self.tokenizer.html):
                token = tokenize(src)
                if token is not None:
                    break
            if token is not None:
                src = src[utf16_length(token.raw):]
                tokens.append(token)
                continue
            token = self.tokenizer.def_(src)
            if token is not None:
                src = src[utf16_length(token.raw):]
                if tokens and tokens[-1].type in ("paragraph", "text"):
                    last = tokens[-1]
                    last.raw += ("" if last.raw.endswith("\n") else "\n") + token.raw
                    last.text = (last.text or "") + "\n" + token.raw
                    self.inline_queue[-1].src = last.text
                elif from_utf16_units(token.tag) not in self.tokens.links:
                    self.tokens.links[from_utf16_units(token.tag)] = LinkDefinition(token.href, token.title)
                    tokens.append(token)
                continue
            token = self.tokenizer.table(src) or self.tokenizer.lheading(src)
            if token is not None:
                src = src[utf16_length(token.raw):]
                tokens.append(token)
                continue
            cut_src = self._clip_extensions("block", src)
            token = self.tokenizer.paragraph(cut_src) if self.state.top else None
            if token is not None:
                if last_paragraph_clipped and tokens and tokens[-1].type == "paragraph":
                    last = tokens[-1]
                    last.raw += ("" if last.raw.endswith("\n") else "\n") + token.raw
                    last.text = (last.text or "") + "\n" + (token.text or "")
                    self.inline_queue.pop()
                    self.inline_queue[-1].src = last.text
                else:
                    tokens.append(token)
                last_paragraph_clipped = len(cut_src) != len(src)
                src = src[utf16_length(token.raw):]
                continue
            token = self.tokenizer.text(src)
            if token is not None:
                src = src[utf16_length(token.raw):]
                if tokens and tokens[-1].type == "text":
                    last = tokens[-1]
                    last.raw += ("" if last.raw.endswith("\n") else "\n") + token.raw
                    last.text = (last.text or "") + "\n" + (token.text or "")
                    self.inline_queue.pop()
                    self.inline_queue[-1].src = last.text
                else:
                    tokens.append(token)
                continue
            self._infinite_loop_error(ord(src[0]))
            break
        self.state.top = True
        return tokens

    def inline(self, src: str, tokens: list[Token] | None = None) -> list[Token]:
        result = [] if tokens is None else tokens
        self.inline_queue.append(_InlineQueueItem(utf16_units(src), result))
        return result

    def _link_in_text(self, text: str) -> bool:
        if "[" not in text:
            return False
        rules = self.tokenizer.rules.inline
        for match in rules["blockSkip"].finditer(text):
            previous = text[match.index - 1:match.index] if match.index > 0 else ""
            if rules["link"].test(match[0]) and previous != "!":
                return True
        for match in rules["reflinkSearch"].finditer(text):
            value = match[0]
            ref_start = value.rfind("[")
            if value.startswith("!") or from_utf16_units(value[ref_start + 1:-1]) not in self.tokens.links:
                continue
            if ref_start > 1 and self._link_in_text(value[1:ref_start - 1]):
                continue
            return True
        return False

    def inline_tokens(self, src: str, tokens: list[Token] | None = None) -> list[Token]:
        result = self._inline_tokens(utf16_units(src), tokens)
        _map_token_strings(result, from_utf16_units)
        return result

    def _inline_tokens(self, src: str, tokens: list[Token] | None = None) -> list[Token]:
        src = utf16_units(src)
        self.tokenizer.lexer = self
        tokens = [] if tokens is None else tokens
        masked_src = src
        rules = self.tokenizer.rules.inline
        if self.tokens.links and "[" in src:
            reflink_search = rules["reflinkSearch"]
            def mask_reflink(match: Match) -> str:
                value = match[0]
                ref_start = value.rfind("[")
                if from_utf16_units(value[ref_start + 1:-1]) not in self.tokens.links:
                    return value
                if ref_start > 1 and not value.startswith("!"):
                    text = value[1:ref_start - 1]
                    if self._link_in_text(text):
                        return "[" + reflink_search.replace(text, mask_reflink) + "][" + "a" * (len(value) - ref_start - 2) + "]"
                return "[" + "a" * (len(value) - 2) + "]"
            masked_src = reflink_search.replace(masked_src, mask_reflink)
        masked_src = rules["anyPunctuation"].replace(masked_src, lambda match: "+" * len(match[0]))
        def mask_block(match: Match) -> str:
            offset = len(match[2])
            return match[0][:offset] + "[" + "a" * (len(match[0]) - offset - 2) + "]"
        masked_src = rules["blockSkip"].replace(masked_src, mask_block)
        if self.options.hooks is not None:
            hook = self.options.hooks.em_strong_mask
            masked = hook.__func__(TokenizerContext(self), from_utf16_units(masked_src)) if isinstance(hook, MethodType) else hook(from_utf16_units(masked_src))
            if masked is not None:
                masked_src = utf16_units(masked)
        elif self.options.em_strong_mask is not None:
            masked = self.options.em_strong_mask(TokenizerContext(self), from_utf16_units(masked_src))
            if masked is not None:
                masked_src = utf16_units(masked)
        keep_prev_char = False
        prev_char = ""
        source_length = len(src) + 1
        while src:
            if len(src) >= source_length:
                self._infinite_loop_error(ord(src[0]))
                break
            source_length = len(src)
            if not keep_prev_char:
                prev_char = ""
            keep_prev_char = False
            token = self._extension_token("inline", src, tokens)
            if token is not None:
                src = src[utf16_length(token.raw):]
                tokens.append(token)
                continue
            token = None
            for tokenize in (self.tokenizer.escape, self.tokenizer.tag, self.tokenizer.link):
                token = tokenize(src)
                if token is not None:
                    break
            if token is not None:
                src = src[utf16_length(token.raw):]
                tokens.append(token)
                continue
            token = self.tokenizer.reflink(src, self.tokens.links)
            if token is not None:
                src = src[utf16_length(token.raw):]
                if token.type == "text" and tokens and tokens[-1].type == "text":
                    tokens[-1].raw += token.raw
                    tokens[-1].text = (tokens[-1].text or "") + (token.text or "")
                else:
                    tokens.append(token)
                continue
            token = self.tokenizer.em_strong(src, masked_src, prev_char)
            if token is None:
                token = self.tokenizer.codespan(src)
            if token is None:
                token = self.tokenizer.br(src)
            if token is None:
                token = self.tokenizer.del_(src, masked_src, prev_char)
            if token is None:
                token = self.tokenizer.autolink(src)
            if token is None and not self.state.in_link:
                token = self.tokenizer.url(src)
            if token is not None:
                src = src[utf16_length(token.raw):]
                tokens.append(token)
                continue
            token = self.tokenizer.inline_text(self._clip_extensions("inline", src))
            if token is not None:
                src = src[utf16_length(token.raw):]
                if not token.raw.endswith("_"):
                    prev_char = utf16_units(token.raw)[-1:]
                keep_prev_char = True
                if tokens and tokens[-1].type == "text":
                    tokens[-1].raw += token.raw
                    tokens[-1].text = (tokens[-1].text or "") + (token.text or "")
                else:
                    tokens.append(token)
                continue
            self._infinite_loop_error(ord(src[0]))
            break
        return tokens

    def _extension_token(self, level: Literal["block", "inline"], src: str, tokens: list[Token]) -> Token | None:
        if self.options.extensions is None:
            return None
        callbacks = self.options.extensions.block if level == "block" else self.options.extensions.inline
        for callback in callbacks:
            token = callback(TokenizerContext(self), from_utf16_units(src), tokens)
            if isinstance(token, Token):
                _map_token_strings([token], utf16_units)
                return token
        return None

    def _clip_extensions(self, level: Literal["block", "inline"], src: str) -> str:
        if self.options.extensions is None:
            return src
        callbacks = self.options.extensions.start_block if level == "block" else self.options.extensions.start_inline
        start_index: int | None = None
        for callback in callbacks:
            start = callback(TokenizerContext(self), from_utf16_units(src[1:]))
            if isinstance(start, (int, float)) and not isinstance(start, bool) and start >= 0 and math.isfinite(start):
                start_index = int(start) if start_index is None else min(start_index, int(start))
        return src if start_index is None else src[:start_index + 1]

    def _infinite_loop_error(self, byte: int) -> None:
        message = f"Infinite loop on byte: {byte}"
        if self.options.silent:
            print(message, file=sys.stderr)
        else:
            raise RuntimeError(message)


__all__ = ["Lexer", "Tokenizer"]
