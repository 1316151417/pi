"""Ordered character matching from ``packages/tui/src/fuzzy.ts``."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ._javascript import JS_WHITESPACE, js_trim, utf16_units

_BOUNDARIES = frozenset(JS_WHITESPACE + "-_./:")
_TOKEN_SPLIT = re.compile("[" + re.escape(JS_WHITESPACE) + "/]+")


@dataclass
class FuzzyMatch:
    matches: bool
    score: float


def fuzzy_match(query: str, text: str) -> FuzzyMatch:
    query_lower = utf16_units(query.lower())
    text_lower = utf16_units(text.lower())

    def match_query(normalized_query: str) -> FuzzyMatch:
        if not normalized_query:
            return FuzzyMatch(True, 0)
        if len(normalized_query) > len(text_lower):
            return FuzzyMatch(False, 0)
        query_index = 0
        score = 0.0
        last_match_index = -1
        consecutive_matches = 0
        while query_index < len(normalized_query):
            index = text_lower.find(normalized_query[query_index], last_match_index + 1)
            if index == -1:
                break
            is_word_boundary = index == 0 or text_lower[index - 1] in _BOUNDARIES
            if last_match_index == index - 1:
                consecutive_matches += 1
                score -= consecutive_matches * 5
            else:
                consecutive_matches = 0
                if last_match_index >= 0:
                    score += (index - last_match_index - 1) * 2
            if is_word_boundary:
                score -= 10
            score += index * 0.1
            last_match_index = index
            query_index += 1
        if query_index < len(normalized_query):
            return FuzzyMatch(False, 0)
        if normalized_query == text_lower:
            score -= 100
        return FuzzyMatch(True, score)

    primary_match = match_query(query_lower)
    if primary_match.matches:
        return primary_match
    alpha_numeric = re.fullmatch(r"([a-z]+)([0-9]+)", query_lower)
    numeric_alpha = re.fullmatch(r"([0-9]+)([a-z]+)", query_lower)
    matched = alpha_numeric or numeric_alpha
    if matched is None:
        return primary_match
    swapped_match = match_query(matched[2] + matched[1])
    if not swapped_match.matches:
        return primary_match
    return FuzzyMatch(True, swapped_match.score + 5)


def fuzzy_filter[T](items: list[T], query: str, get_text: Callable[[T], str]) -> list[T]:
    if not js_trim(query):
        return items
    tokens = [token for token in _TOKEN_SPLIT.split(js_trim(query)) if token]
    if not tokens:
        return items
    results: list[tuple[T, float]] = []
    for item in items:
        text = get_text(item)
        total_score = 0.0
        for token in tokens:
            match = fuzzy_match(token, text)
            if not match.matches:
                break
            total_score += match.score
        else:
            results.append((item, total_score))
    results.sort(key=lambda result: result[1])
    return [item for item, _ in results]
