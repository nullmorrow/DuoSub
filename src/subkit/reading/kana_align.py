"""Align a reading to its surface: which kana belong to which kanji block.

``行った``/``いった`` → 行=い；``取り扱い``/``とりあつかい`` → 取=と, 扱=あつか；
``目覚め``/``めざめ`` → 目覚=めざ.  Kana in the surface act as anchors
(hiragana/katakana-insensitive).  Digits+kanji (``3人``) form one block.
The returned readings are slices of the *original* reading, so an explicit
katakana reading stays katakana.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional

from ..textutil import is_digit_char, is_kanji, kata_to_hira, normalize_halfwidth_kana


def _block_kind(ch: str) -> str:
    if is_kanji(ch) or is_digit_char(ch):
        return "K"
    return "L"


def blocks(surface: str) -> list[tuple[int, int, str]]:
    """Split a surface into runs of kanji-like (K) and literal (L) characters."""
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(surface):
        kind = _block_kind(surface[i])
        j = i
        while j < len(surface) and _block_kind(surface[j]) == kind:
            j += 1
        out.append((i, j, kind))
        i = j
    return out


_SMALL = "ぁぃぅぇぉっゃゅょゎー"


def _pattern(surface: str, lazy: bool) -> tuple[str, list[tuple[int, int]]]:
    pattern = ""
    groups = []
    for (a, b, kind) in blocks(surface):
        if kind == "K":
            pattern += "(.+?)" if lazy else "(.+)"
            groups.append((a, b))
        else:
            # Normalize only this literal anchor, never the indexed surface or
            # reading: a voiced half-width pair still occupies two source IDs.
            literal = surface[a:b]
            if any("ｦ" <= ch <= "ﾟ" for ch in literal):
                literal = unicodedata.normalize("NFC", normalize_halfwidth_kana(literal))
            lit = kata_to_hira(literal)
            for c in lit:
                if c in "〜～":
                    continue  # elongation marks (危な〜い) do not appear in readings
                pattern += re.escape(c) if c not in "ーｰ" else "[ーあいうえお]?"
    return pattern, groups


@lru_cache(maxsize=50000)
def _spans(surface: str, reading_hira: str, lazy: bool) -> Optional[tuple[tuple[int, int, int, int], ...]]:
    pattern, groups = _pattern(surface, lazy)
    m = re.fullmatch(pattern, reading_hira)
    if not m:
        return None
    out = []
    for gi, (a, b) in enumerate(groups, 1):
        s, e = m.span(gi)
        if reading_hira[s] in _SMALL:
            return None
        out.append((a, b, s, e))
    return tuple(out)


def align(surface: str, reading: str) -> Optional[list[tuple[int, int, str]]]:
    """Return [(start, end, reading)] for each kanji block of ``surface``; None if impossible."""
    if not surface or not reading:
        return None
    if not any(_block_kind(c) == "K" for c in surface):
        return []
    hira = kata_to_hira(reading)
    for lazy in (True, False):
        sp = _spans(surface, hira, lazy)
        if sp is not None:
            return [(a, b, reading[s:e]) for a, b, s, e in sp]
    return None


def okurigana_split(surface: str, reading: str) -> tuple[str, str]:
    """Strip trailing kana shared by surface and reading: (見て, みて) → (見, み)."""
    s, r = surface, reading
    while s and r and not is_kanji(s[-1]) and kata_to_hira(s[-1]) == kata_to_hira(r[-1]):
        s, r = s[:-1], r[:-1]
    return s, r
