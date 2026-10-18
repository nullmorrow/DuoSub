"""Reading engine: Japanese line → ruby entries with provenance and confidence.

Priority (highest first):
  1. ``!`` forced entries in the line's record (handled by the work-file layer)
  2. dictionary words (project/global, speaker-conditional)
  3. per-occurrence hints from the captions (``遠月(とおつき)``)
  4. manual record entries (work-file layer)
  5. context rules (heteronyms.py) / project ``prefer`` / resolvers (LLM, audio)
  6. analyzer reading (Sudachi)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Optional, Sequence

from ..textutil import has_kanji, is_digit_char, is_kana, is_kanji, kata_to_hira
from . import heteronyms as H
from .dictionary import ReadingDict
from .kana_align import align
from .numbers import COUNTERS, SPECIAL, counter_reading, digit_kanji_number, number_reading, parse_int


@dataclass
class Part:
    start: int
    end: int
    reading: str


@dataclass
class Entry:
    start: int
    end: int
    surface: str
    reading: str
    parts: list[Part]
    source: str                    # dict | hint | record | rule | prefer | analyzer | number | llm | audio
    confidence: float = 1.0
    alternatives: list[str] = field(default_factory=list)
    locked: bool = False
    forced: bool = False
    hidden: bool = False           # entry explicitly without ruby: 表記（-）
    note: str = ""
    rule_surface: Optional[str] = None  # parent Project word-split rule, when separate entries are emitted

    @property
    def needs_review(self) -> bool:
        return not self.locked and self.confidence < H.REVIEW_BELOW


@dataclass
class Morph:
    surface: str
    start: int
    end: int
    reading: str        # hiragana
    pos: tuple
    lemma: str


LOOKUP_CACHE_SIZE = 4096  # Bound retained popup surfaces; keep the open dictionary reusable.


class Analyzer:
    _instance: Optional["Analyzer"] = None

    def __init__(self, dict_name: str = "full") -> None:
        from sudachipy import Dictionary, SplitMode

        self._dict = Dictionary(dict=dict_name)
        self._tok = self._dict.create()
        self._mode = SplitMode.C

    @classmethod
    def get(cls, dict_name: str = "full") -> "Analyzer":
        if cls._instance is None:
            cls._instance = cls(dict_name)
        return cls._instance

    def tokenize(self, text: str, *, mode=None) -> list[Morph]:
        """Tokenize; wave-dash elongations (危な〜い, 許さ〜ん) are hidden from the
        analyzer and the offsets mapped back to the original text."""
        keep = [i for i, ch in enumerate(text) if not (ch in "〜～" and 0 < i < len(text) - 1
                                                        and not text[i - 1].isspace() and not text[i + 1].isspace())]
        if len(keep) == len(text):
            clean, index = text, None
        else:
            clean = "".join(text[i] for i in keep)
            index = keep + [len(text)]
        out = []
        for m in self._tok.tokenize(clean, self._mode if mode is None else mode):
            a, b = m.begin(), m.end()
            if index is not None:
                a2, b2 = index[a], index[b - 1] + 1 if b > a else index[a]
                surface = text[a2:b2]
            else:
                a2, b2, surface = a, b, m.surface()
            out.append(Morph(surface, a2, b2, kata_to_hira(m.reading_form()), tuple(m.part_of_speech()), m.dictionary_form()))
        return out

    def split_a(self, surface: str) -> list[tuple[str, str]]:
        """Shortest-unit (SplitMode.A) segmentation of a word: 上流階級 → 上流|階級."""
        from sudachipy import SplitMode

        return [(m.surface(), kata_to_hira(m.reading_form())) for m in self._tok.tokenize(surface, SplitMode.A)]

    def lookup_readings(self, surface: str) -> list[str]:
        """All lexicon readings for an exact surface (used for per-kanji splitting)."""
        return list(self._lookup_readings(surface))

    @lru_cache(maxsize=LOOKUP_CACHE_SIZE)
    def _lookup_readings(self, surface: str) -> tuple[str, ...]:
        try:
            ms = self._dict.lookup(surface)
        except Exception:
            return ()
        out = []
        for m in ms:
            r = kata_to_hira(m.reading_form())
            if r and r not in out:
                out.append(r)
        return tuple(out)


_KNUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _tokenizer_text(text: str) -> str:
    """Normalize colloquial elongations without changing character offsets."""
    return "".join("〜" if ch in "ーｰ" and i > 0
                   and ("ぁ" <= text[i - 1] <= "\u309f"
                        or (is_kanji(text[i - 1]) and not is_kana(text[i - 1])))
                   else ch for i, ch in enumerate(text))


def kanji_number(s: str) -> Optional[int]:
    """一, 三, 十, 二十, 三百, 五千 → int; None if not a plain kanji numeral."""
    if not s or any(ch not in "一二三四五六七八九十百千" for ch in s):
        return None
    total, cur = 0, 0
    for ch in s:
        if ch in _KNUM:
            cur = _KNUM[ch]
        else:
            mult = {"十": 10, "百": 100, "千": 1000}[ch]
            total += (cur or 1) * mult
            cur = 0
    return total + cur


def make_entry(start: int, surface: str, reading: str, source: str, *, confidence: float = 1.0,
               alternatives: Optional[list[str]] = None, locked: bool = False, groups: Optional[list[tuple[str, str]]] = None) -> Entry:
    parts: list[Part] = []
    if groups and len(groups) > 1:
        pos = start
        for gs, gr in groups:
            sub = align(gs, gr)
            if sub:
                parts.extend(Part(pos + a, pos + b, r) for a, b, r in sub)
            elif has_kanji(gs):
                parts.append(Part(pos, pos + len(gs), gr))
            pos += len(gs)
    else:
        sub = align(surface, reading)
        if sub is None:
            # irregular (熟字訓 with okurigana etc.): one group over the kanji span
            ks = [i for i, ch in enumerate(surface) if is_kanji(ch) or is_digit_char(ch)]
            if ks:
                parts = [Part(start + ks[0], start + ks[-1] + 1, reading)]
        else:
            parts = [Part(start + a, start + b, r) for a, b, r in sub]
    return Entry(start, start + len(surface), surface, reading, parts, source, confidence, alternatives or [], locked)


class ReadingEngine:
    def __init__(self, dictionary: Optional[ReadingDict] = None, analyzer: Optional[Analyzer] = None, *,
                 episode: Optional[str] = None, season: Optional[str] = None) -> None:
        self.dict = dictionary or ReadingDict()
        self.analyzer = analyzer or Analyzer.get()
        self.episode = episode
        self.season = season

    def _tokenize(self, text: str) -> list[Morph]:
        normalized = _tokenizer_text(text)
        morphs = self.analyzer.tokenize(normalized)
        if normalized == text:
            return morphs
        out = []
        for m in morphs:
            surface = text[m.start:m.end]
            # A trailing normalized mark is a symbol token, not the reading "きごう".
            reading = surface.replace("ｰ", "ー") if normalized[m.start:m.end] == "〜" else m.reading
            out.append(replace(m, surface=surface, reading=reading))
        return out

    def dictionary_readings(self, text: str, a: int, b: int) -> list[str]:
        """Popup-only kana readings of visible [a, b); automatic analysis is unchanged."""
        if not 0 <= a < b <= len(text):
            return []
        morphs = [m for m in self._tokenize(text) if m.start < b and a < m.end]

        def portion(surface: str, reading: str, start: int, end: int) -> str:
            if start == 0 and end == len(surface):
                return reading
            parts = align(surface, reading)
            if parts is None:
                return ""
            pieces, pos = [], start
            for x, y, r in parts:
                if y <= start or x >= end:
                    continue
                if x < start or y > end:
                    return ""  # A subrange of an unsplit kanji block has no reliable slice.
                pieces.extend((surface[pos:x], r))
                pos = y
            pieces.append(surface[pos:end])
            return kata_to_hira("".join(pieces))

        def combined(lemma: bool = False) -> list[str]:
            choices = [""]
            for m in morphs:
                x, y = max(a, m.start) - m.start, min(b, m.end) - m.start
                readings = [m.reading]
                if lemma:
                    readings = []
                    stem = m.lemma
                    while stem and is_kana(stem[-1]) and not is_kanji(stem[-1]):
                        stem = stem[:-1]
                    for r in self.analyzer.lookup_readings(m.lemma):
                        suffix = kata_to_hira(m.lemma[len(stem):])
                        if m.surface != m.lemma:
                            if not stem or not m.surface.startswith(stem) or (suffix and not r.endswith(suffix)):
                                continue
                            r = (r[:-len(suffix)] if suffix else r) + kata_to_hira(m.surface[len(stem):])
                        readings.append(r)
                readings = [portion(m.surface, r, x, y) for r in readings]
                readings = list(dict.fromkeys(r for r in readings if r and all(is_kana(c) and not is_kanji(c) for c in r)))
                choices = [prefix + r for prefix in choices for r in readings]
                if not choices:
                    break
            return choices if morphs and morphs[0].start <= a and morphs[-1].end >= b else []

        readings = combined() + self.analyzer.lookup_readings(text[a:b]) + combined(lemma=True)
        return list(dict.fromkeys(kata_to_hira(r) for r in readings
                                  if r and all(is_kana(c) and not is_kanji(c) for c in r)))

    # ----------------------------------------------------------------- main
    def analyze(self, text: str, *, speaker: Optional[str] = None, hints: Sequence = (),
                character: Optional[str] = None, line: Optional[str] = None) -> list[Entry]:
        entries: list[Entry] = []
        covered = [False] * len(text)

        def cover(a: int, b: int) -> None:
            for i in range(max(0, a), min(len(text), b)):
                covered[i] = True

        for a, b, w in self.dict.match(text, speaker, character=character, line=line,
                                     episode=self.episode, season=self.season):
            if w.separate_words:
                pos = a
                for surface, reading in w.groups:
                    e = make_entry(pos, surface, reading, "dict", locked=True)
                    e.rule_surface = w.surface
                    entries.append(e)
                    pos += len(surface)
            else:
                entries.append(make_entry(a, text[a:b], w.reading, "dict", locked=True, groups=w.groups))
            cover(a, b)
        for h in hints:
            if getattr(h, "start", -1) < 0:
                continue
            a, b = h.start, h.end
            if b > len(text) or text[a:b] != h.surface or any(covered[a:b]):
                continue
            entries.append(make_entry(a, h.surface, h.reading, "hint", confidence=0.98))
            cover(a, b)

        # A lone adjective followed by elongations uses its dictionary stem (硬ー → かた).
        for a, ch in enumerate(text):
            if not is_kanji(ch) or covered[a] or ch in self.dict.prefer or (a > 0 and is_kanji(text[a - 1])):
                continue
            b = a + 1
            while b < len(text) and text[b] in "ーｰ":
                b += 1
            if b == a + 1 or (b < len(text) and "ぁ" <= text[b] <= "\u309f"):
                continue
            reading = next((r[:-1] for r in self.analyzer.lookup_readings(ch + "い")
                            if r.endswith("い")), "")
            if reading:
                entries.append(make_entry(a, ch, reading, "analyzer", confidence=0.9))
                cover(a, a + 1)

        morphs = self._tokenize(text) if has_kanji(text) or any(is_digit_char(c) for c in text) else []
        i = 0
        while i < len(morphs):
            m = morphs[i]
            nxt = morphs[i + 1] if i + 1 < len(morphs) else None
            prv = morphs[i - 1] if i > 0 else None
            # digits with kanji units: 1000万 → いっせんまん (one group over the whole number)
            dk = digit_kanji_number(m.surface)
            if dk is not None and not any(covered[m.start:m.end]):
                r = number_reading(dk)
                e = make_entry(m.start, m.surface, r, "number", confidence=0.95)
                e.parts = [Part(m.start, m.end, r)]
                entries.append(e)
                cover(m.start, m.end)
                i += 1
                continue
            # numbers + counter (digits or kanji numerals)
            if (all(is_digit_char(c) for c in m.surface) or kanji_number(m.surface) is not None) and nxt and nxt.surface \
                    and (has_kanji(nxt.surface) or nxt.surface in ("つ",)):
                if not any(covered[m.start:nxt.end]):
                    e = self._number_entry(text, m, nxt)
                    if e:
                        entries.append(e)
                        cover(e.start, e.end)
                        i += 2
                        continue
            if not has_kanji(m.surface):
                i += 1
                continue
            if any(covered[m.start:m.end]):
                if not all(covered[m.start:m.end]):
                    entries.extend(self._residual(text, m, covered))
                i += 1
                continue
            entries.append(self._morph_entry(text, m, prv, nxt, speaker))
            i += 1
        entries.sort(key=lambda e: e.start)
        return entries

    # ------------------------------------------------------------- helpers
    def _morph_entry(self, text: str, m: Morph, prv: Optional[Morph], nxt: Optional[Morph], speaker: Optional[str]) -> Entry:
        reading = m.reading
        source = "analyzer"
        conf = 0.9
        alts: list[str] = []
        bad = not reading or reading == "*" or has_kanji(reading)
        if bad:
            reading = self._fallback_reading(m.surface)
            conf = 0.4
        ctx = H.Ctx(text, m.start, m.end, m.surface, reading, prv.surface if prv else "", nxt.surface if nxt else "",
                    nxt.reading if nxt else "", speaker)
        rule = H.RULES.get(m.surface)
        res = rule(ctx) if rule else None
        if res is not None:
            reading, conf, alts = res
            source = "rule"
        elif m.surface in self.dict.prefer:
            alts = [x for x in self.dict.candidates.get(m.surface, H.HETERONYMS.get(m.surface, [])) if x != self.dict.prefer[m.surface]]
            reading, conf, source = self.dict.prefer[m.surface], 0.85, "prefer"
        else:
            cands = self.dict.candidates.get(m.surface) or H.HETERONYMS.get(m.surface)
            if cands and len(cands) > 1:
                alts = [c for c in cands if c != reading]
                # analyzer agrees with the usual reading → fairly safe; otherwise review
                conf = 0.8 if reading == cands[0] else 0.6
        groups = self._compound_groups(m.surface, reading)
        return make_entry(m.start, m.surface, reading, source, confidence=conf, alternatives=alts, groups=groups)

    def _compound_groups(self, surface: str, reading: str) -> Optional[list[tuple[str, str]]]:
        """Long kanji compounds get one ruby group per sub-word (上流|階級), so each
        reading sits over its own kanji instead of one squeezed block."""
        kanji = sum(1 for c in surface if is_kanji(c))
        if kanji < 3 or len(reading) < 2 * kanji:
            return None
        try:
            subs = self.analyzer.split_a(surface)
        except Exception:
            return None
        if len(subs) < 2 or "".join(r for _s, r in subs) != kata_to_hira(reading) or "".join(s for s, _r in subs) != surface:
            return None
        return subs

    def _number_entry(self, text: str, num: Morph, counter: Morph) -> Optional[Entry]:
        kanji_num = kanji_number(num.surface)
        n = kanji_num if kanji_num is not None else parse_int(num.surface)
        if n is None:
            return None
        cs = counter.surface
        # longest counter key that the counter morph starts with
        key = None
        for k in sorted(set(COUNTERS) | {c for _, c in SPECIAL}, key=len, reverse=True):
            if cs.startswith(k):
                key = k
                break
        if key is None:
            return None
        reading, whole = counter_reading(n, key)
        if not reading:
            return None
        if whole:
            surface = num.surface + key
            e = make_entry(num.start, surface, reading, "number", confidence=0.85 if key == "日" else 0.95)
            if kanji_num is not None:
                return e  # 一つ → 一=ひと (normal okurigana alignment)
            if key == "つ":
                # ２つ → ruby ふた over the digit only; つ is visible
                e.parts = [Part(num.start, num.start + len(num.surface), reading[:-1])]
            else:
                e.parts = [Part(num.start, num.start + len(surface), reading)]
            return e
        # 年間 / 年生 / 週間 / 本目: keep the rest of the counter morph with its own reading
        rest = cs[len(key):]
        rest_reading = ""
        if rest:
            base = COUNTERS.get(key, ("", ""))[0]
            if base and counter.reading.startswith(base):
                rest_reading = counter.reading[len(base):]
            else:
                rest_reading = "".join(x.reading for x in self._tokenize(rest))
        if kanji_num is not None:
            # 一匹 → いっぴき: the numeral is kanji too, so it gets its own ruby
            from .numbers import number_prefix_reading

            num_r = number_prefix_reading(str(n), key)
            groups = [(num.surface, num_r), (key, reading)] + ([(rest, rest_reading)] if rest else [])
            surface = num.surface + cs
            return make_entry(num.start, surface, num_r + reading + rest_reading, "number", confidence=0.85, groups=groups)
        if rest:
            return make_entry(counter.start, cs, reading + rest_reading, "number", confidence=0.9,
                              groups=[(key, reading), (rest, rest_reading)])
        return make_entry(counter.start, key, reading, "number", confidence=0.9)

    def _residual(self, text: str, m: Morph, covered: list[bool]) -> list[Entry]:
        out = []
        i = m.start
        while i < m.end:
            if covered[i]:
                i += 1
                continue
            j = i
            while j < m.end and not covered[j]:
                j += 1
            piece = text[i:j]
            if has_kanji(piece):
                for sub in self._tokenize(piece):
                    if has_kanji(sub.surface):
                        e = make_entry(i + sub.start, sub.surface, sub.reading or self._fallback_reading(sub.surface), "analyzer", confidence=0.6)
                        out.append(e)
            i = j
        return out

    def _fallback_reading(self, surface: str) -> str:
        """Reading for words Sudachi cannot read: UniDic first, then each kanji alone."""
        r = _unidic_reading(surface)
        if r and not has_kanji(r):
            return r
        out = []
        for ch in surface:
            if is_kanji(ch):
                rs = self.analyzer.lookup_readings(ch)
                rs = [x for x in rs if x and not has_kanji(x)]
                if not rs:
                    u = _unidic_reading(ch)
                    rs = [u] if u and not has_kanji(u) else []
                out.append(rs[0] if rs else "？")
            else:
                out.append(kata_to_hira(ch))
        return "".join(out)


_TAGGER = None


def _unidic_reading(text: str) -> str:
    """Second opinion from fugashi + unidic-lite (installed); '' when unavailable."""
    global _TAGGER
    try:
        if _TAGGER is None:
            import fugashi

            _TAGGER = fugashi.Tagger()
        parts = []
        for w in _TAGGER(text):
            k = getattr(w.feature, "kana", None)
            if not k or k == "*":
                return ""
            parts.append(kata_to_hira(k))
        return "".join(parts)
    except Exception:
        return ""
