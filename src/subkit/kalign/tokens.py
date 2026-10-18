"""Beat readings for MMS romaji and Japanese kana CTC vocabularies.

Romanisation follows tools/experiments/kalign/kana_romaji.py exactly. It is
alignment spelling, rather than a general-purpose Japanese transliterator.
"""

from __future__ import annotations

from collections.abc import Mapping


_BASE = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ゐ": "i", "ゑ": "e", "ん": "n", "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}
_YOON = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
_SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}


def kata_to_hira(text: str) -> str:
    """Convert full-width katakana, preserving other characters as in the prototype."""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def romaji(kana: str) -> str:
    """Return a-z alignment spelling; っ is silent, ー repeats the last vowel.

    Small kana replace the preceding vowel (yoon also adds y except in sh/ch/j).
    Preserve the prototype's spelling even for uncommon combinations: e.g.
    うぃ/うぇ/うぉ become i/e/o, while わぃ/わぇ/わぉ become wi/we/wo.
    ASCII letters are lower-cased; punctuation and unread kanji are dropped.
    """
    text = kata_to_hira(kana)
    out = ""
    i = 0
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if c == "っ":
            i += 1
            continue
        if c == "ー":
            out += next((ch for ch in reversed(out) if ch in "aeiou"), "")
            i += 1
            continue
        if nxt in _YOON and c in _BASE and c not in _YOON:
            stem = _BASE[c][:-1]
            out += stem + ("" if stem in ("sh", "ch", "j") else "y") + _YOON[nxt]
            i += 2
            continue
        if nxt in _SMALL_V and c in _BASE:
            out += _BASE[c][:-1] + _SMALL_V[nxt]
            i += 2
            continue
        if c in _BASE:
            out += _BASE[c]
        elif c.isascii() and c.isalpha():
            out += c.lower()
        i += 1
    return out


def particle_positions_many(texts: list[str]) -> list[set[int]]:
    """Visible offsets of particle は/へ, using one full dictionary for the batch.

    Sudachi is lazy and optional. Always close the owned dictionary, including
    when tokenization fails; empty input never opens one.
    """
    if not texts:
        return []
    try:
        from sudachipy import Dictionary, SplitMode
    except ImportError:
        return [set() for _ in texts]
    dictionary = Dictionary(dict="full")
    try:
        tokenizer = dictionary.create()
        return [{m.begin() for m in tokenizer.tokenize(text, SplitMode.C)
                 if m.surface() in ("は", "へ") and m.part_of_speech()[0] == "助詞"}
                for text in texts]
    finally:
        dictionary.close()


def particle_positions(text: str) -> set[int]:
    """Single-text convenience wrapper for :func:`particle_positions_many`."""
    return particle_positions_many([text])[0]


def sung_for_alignment(sung: str, is_particle: bool, from_ruby: bool) -> str:
    """Use わ/え for plain-text particles は/へ, preserving explicit ruby readings."""
    if is_particle and not from_ruby:
        return {"は": "わ", "へ": "え"}.get(sung, sung)
    return sung


def mms_tokens(sung: str, vocab: Mapping[str, int]) -> list[int]:
    """MMS token IDs; vocab must contain the emitted a-z letters (else KeyError)."""
    return [vocab[c] for c in romaji(sung)]


def kana_tokens(sung: str, vocab: Mapping[str, int]) -> list[int]:
    """Hiragana model IDs after katakana conversion; drop |, ー and unknowns.

    A small っ is retained when that model has a token for it. Unread kanji and
    Latin text do not become kana targets even in an extended vocabulary.
    """
    return [vocab[c] for c in kata_to_hira(sung) if "ぁ" <= c <= "ゟ" and c in vocab]


def last_vowel(sung: str) -> str:
    """Vowel row of the last kana (きょ is o); a final ん/っ has no vowel."""
    kana = [c for c in kata_to_hira(sung) if "ぁ" <= c <= "ゟ" or c == "ー"]
    while kana and kana[-1] == "ー":
        kana.pop()
    spelling = _BASE.get(kana[-1], "") if kana else ""
    return spelling[-1:] if spelling[-1:] in ("a", "i", "u", "e", "o") else ""


def word_starts(spans: list[tuple[int | None, bool]], text: str = "") -> list[bool]:
    """Beats that begin a new word and so never continue the previous vowel.

    ``spans`` gives each beat's visible offset and whether it reads a ruby (kanji) base.
    The first beat of a reading (今 of ない今: い｜ま) starts a word, and so does the
    first beat after a space or punctuation (ない␣␣いま).
    """
    import unicodedata

    out = [False] * len(spans)
    for i in range(1, len(spans)):
        (before, _ruby_before), (at, ruby) = spans[i - 1], spans[i]
        if before is None or at is None or at == before:
            continue
        gap = text[before:at]
        out[i] = ruby or any(c.isspace() or unicodedata.category(c).startswith("P") for c in gap)
    return out


def continuation_morae(sung: list[str], starts: list[bool] | None = None) -> list[bool]:
    """Mark vowel-only continuation beats within one line, ignoring punctuation.
    ``starts`` (from :func:`word_starts`) marks beats that begin a new word."""
    previous, out = "", []
    for i, text in enumerate(sung):
        hira = kata_to_hira(text)
        kana = "".join(c for c in hira if "ぁ" <= c <= "ゟ" or c == "ー")
        vowel = last_vowel(kana)
        only_kana = not any(c.isalnum() and not ("ぁ" <= c <= "ゟ" or c == "ー") for c in hira)
        is_vowel = len(kana) == 1 and kana in "あいうえおぁぃぅぇぉ"
        new_word = starts is not None and i < len(starts) and starts[i]
        continuation = bool(previous and only_kana and not new_word and (kana == "ー" or is_vowel and
                            (vowel == previous or vowel == "u" and previous in "ou" or
                             vowel == "i" and previous in "ei")))
        out.append(continuation)
        previous = previous if kana == "ー" else vowel
    return out
