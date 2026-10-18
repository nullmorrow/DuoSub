"""Conservative, lexicon-backed splitting of a word's ruby into individual kanji.

Kana written in the surface are fixed anchors, never extra ruby boxes.  Matching
normalizes full/half-width katakana and composed kana to hiragana; successful
``KanjiReading.reading`` values are therefore always normalized hiragana.  The
positions still refer to the original, unmodified surface string.

The lexicon's exact single-character readings license rendaku (k/s/t -> voiced,
h -> b/p) and a final tsu/chi/ku/ki becoming a small tsu before another kanji.
There is one explicit contraction: 日's attested にち licenses bound に before
another kanji (日本 = にほん); にっ follows the ordinary chi -> small-tsu rule.
No other reading is shortened by length or by guessing a prefix.

For written okurigana, an additional *exact* lexicon lookup of kanji + following
kana can prove the stem: 学ぶ/まなぶ -> 学/まな, 取り/とり -> 取/と.  The raw
single-character reading まなぶ or とり alone is not proof that an arbitrary
suffix may be removed.  Unknown/irregular words and ambiguous divisions return
None so the user can fill the individual boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Iterable, Protocol

from ..textutil import is_kana, is_kanji, kata_to_hira, normalize_halfwidth_kana


# These are failure limits, never truncation limits: exceeding one leaves the
# split unknown instead of accidentally reporting the first observed solution.
MAX_SURFACE_CHARS = 256
MAX_READING_CHARS = 2048
MAX_LOOKUP_READINGS = 512
MAX_CANDIDATES = 2048
MAX_TRANSITIONS = 100_000

_VOICED = dict(zip("かきくけこさしすせそたちつてとはひふへほ",
                   "がぎぐげござじずぜぞだぢづでどばびぶべぼ"))
_HALF_VOICED = dict(zip("はひふへほ", "ぱぴぷぺぽ"))
_SMALL_INITIAL = frozenset("ぁぃぅぇぉっゃゅょゎー")


class ReadingLookup(Protocol):
    def lookup_readings(self, surface: str) -> Iterable[str]: ...


@dataclass(frozen=True)
class KanjiReading:
    start: int
    end: int
    base: str
    reading: str


class _LimitExceeded(Exception):
    pass


def _normalized(text: str) -> str:
    return kata_to_hira(unicodedata.normalize("NFC", normalize_halfwidth_kana(text)))


def _written_kana(ch: str) -> bool:
    return is_kana(ch) or "ｦ" <= ch <= "ﾟ"


def _variants(readings: set[str], *, following_kanji: bool) -> tuple[str, ...]:
    out: set[str] = set()
    for reading in readings:
        heads = {reading}
        if reading[0] in _VOICED:
            heads.add(_VOICED[reading[0]] + reading[1:])
        if reading[0] in _HALF_VOICED:
            heads.add(_HALF_VOICED[reading[0]] + reading[1:])
        for value in heads:
            out.add(value)
            if following_kanji and value[-1] in "つちくき" and len(value) > 1:
                out.add(value[:-1] + "っ")
        if len(out) > MAX_CANDIDATES:
            raise _LimitExceeded
    return tuple(sorted(out, key=lambda value: (len(value), value)))


def split_kanji_reading(surface: str, reading: str,
                        analyzer: ReadingLookup) -> list[KanjiReading] | None:
    """Return the only complete per-kanji division, or None when unknown.

    ``start``/``end`` are Python string offsets into ``surface``.  Literal kana
    and other written characters must match exactly (kana script-insensitive).
    Every kanji must have a nonempty, lexicon-supported reading.  The caller may
    still offer empty editable boxes when this function returns None.

    A sparse dynamic program counts divisions, saturated at two.  It never
    enumerates all paths; work is bounded by MAX_TRANSITIONS plus the bounded
    lexicon lookups.  Lookup errors and any exceeded limit conservatively return
    None.  There is no model, tokenizer initialization, I/O or global cache here.
    """
    if (not isinstance(surface, str) or not isinstance(reading, str)
            or not surface or not reading or len(surface) > MAX_SURFACE_CHARS
            or len(reading) > MAX_READING_CHARS):
        return None
    normalized = _normalized(reading)
    if not normalized or len(normalized) > MAX_READING_CHARS:
        return None

    cache: dict[str, set[str]] = {}

    def lookup(text: str) -> set[str]:
        if text not in cache:
            values: set[str] = set()
            try:
                for count, raw in enumerate(analyzer.lookup_readings(text), 1):
                    if count > MAX_LOOKUP_READINGS:
                        raise _LimitExceeded
                    if not isinstance(raw, str) or not raw:
                        continue
                    if len(raw) > MAX_READING_CHARS:
                        raise _LimitExceeded
                    value = _normalized(raw)
                    if (value and value[0] not in _SMALL_INITIAL
                            and all(is_kana(ch) for ch in value)):
                        values.add(value)
            except _LimitExceeded:
                raise
            except Exception:
                raise _LimitExceeded from None
            cache[text] = values
        return cache[text]

    # (start, end, is_kanji, candidates); a literal run has exactly one choice.
    tokens: list[tuple[int, int, bool, tuple[str, ...]]] = []
    i, kanji_count = 0, 0
    try:
        while i < len(surface):
            if not is_kanji(surface[i]):
                end = i + 1
                while end < len(surface) and not is_kanji(surface[end]):
                    end += 1
                tokens.append((i, end, False, (_normalized(surface[i:end]),)))
                i = end
                continue
            kanji_count += 1
            ch = surface[i]
            candidates = set(lookup(ch))
            following_kanji = i + 1 < len(surface) and is_kanji(surface[i + 1])
            if ch == "日" and following_kanji and "にち" in candidates:
                candidates.add("に")
            end = i + 1
            while end < len(surface) and _written_kana(surface[end]) and not is_kanji(surface[end]):
                end += 1
            if end > i + 1:
                tail = _normalized(surface[i + 1:end])
                for full in lookup(surface[i:end]):
                    if len(full) > len(tail) and full.endswith(tail):
                        candidates.add(full[:-len(tail)])
            if not candidates:
                return None
            tokens.append((i, i + 1, True, _variants(candidates, following_kanji=following_kanji)))
            i += 1
    except _LimitExceeded:
        return None
    if not kanji_count:
        return None

    # Each position keeps a saturated count and its predecessor only when unique.
    # Retaining the per-token layers makes reconstruction iterative too.
    states: dict[int, tuple[int, tuple[int, str] | None]] = {0: (1, None)}
    layers: list[dict[int, tuple[int, tuple[int, str] | None]]] = []
    transitions = 0
    for _start, _end, _is_kanji, candidates in tokens:
        following: dict[int, tuple[int, tuple[int, str] | None]] = {}
        for position, (ways, _previous) in states.items():
            for candidate in candidates:
                transitions += 1
                if transitions > MAX_TRANSITIONS:
                    return None
                if not normalized.startswith(candidate, position):
                    continue
                end = position + len(candidate)
                if end in following:
                    following[end] = (min(2, following[end][0] + ways), None)
                else:
                    following[end] = (ways, (position, candidate) if ways == 1 else None)
        if not following:
            return None
        layers.append(following)
        states = following
    position = len(normalized)
    if position not in states or states[position][0] != 1:
        return None
    result: list[KanjiReading] = []
    for token, layer in zip(reversed(tokens), reversed(layers)):
        previous = layer[position][1]
        if previous is None:  # A unique final path must have unique ancestors.
            return None
        position, matched = previous
        start, end, kanji, _candidates = token
        if kanji:
            result.append(KanjiReading(start, end, surface[start:end], matched))
    result.reverse()
    return result
