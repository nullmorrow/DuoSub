"""Alignment units built from a read-only Japanese source.

A unit is a phrase of one caption line, identified by exact offsets into the
source event text, so it can become a provenance-carrying run unchanged.
Each unit also records what kind of boundary precedes it — a new caption, a
new speaker, a new line, a space or sentence punctuation — and how much it
costs to cut there.  The aligner uses these costs to keep a caption's head
with its own sentence (cross-evaluation examples 9–13: 「七実様」「え〜？ 私」
「あっ」 must not be glued to the previous speaker's line).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from ..align.fragments import is_interjection as _ja_interjection
from ..languages import get_language
from ..model.source import Source
from . import cc
from .sdh_en import analyzer_for

SENT_END = get_language("ja").sentence_punctuation
_SPACE = " 　\t"
_PUNCT_STRIP = re.compile(r"[\s　、。，,.!！?？…‥・～〜~ー－\-―「」『』（）()｡]+")
_HONORIFIC_END = ("様", "さま", "さん", "君", "くん", "ちゃん", "先輩", "先生", "殿", "氏", "兄", "姉", "兄様", "姉様",
                  "お兄様", "お姉様", "兄さん", "姉さん")


def is_interjection(text: str, lang: str = "ja") -> bool:
    language = get_language(lang, text)
    if language.analyzer_id == "ja-cc":
        return _ja_interjection(text)
    words = language.filler_words
    if not words:
        return False
    core = "".join(c for c in unicodedata.normalize("NFC", text).casefold() if c.isalnum())
    core = re.sub(r"(.)\1+", r"\1", core)
    return bool(core) and core in {re.sub(r"(.)\1+", r"\1", w) for w in words}


def language_phrases(text: str, a: int, b: int, lang: str) -> list[tuple[int, int, str]]:
    """Non-Japanese boundaries; word spaces stay inside space-separated phrases."""
    language = get_language(lang, text[a:b])
    punctuation = language.phrase_punctuation
    out = []
    start, kind = a, "first"
    i = a
    while i < b:
        if start == i and text[i] in _SPACE:
            start += 1
            i += 1
            continue
        if not language.word_spaces and text[i] in _SPACE:
            out.append((start, i, kind))
            while i < b and text[i] in _SPACE:
                i += 1
            start, kind = i, "space"
            continue
        if text[i] in punctuation:
            # Decimal points and dots inside words are not sentence boundaries.
            if language.word_spaces and text[i] == "." and i + 1 < b and text[i + 1].isalnum():
                i += 1
                continue
            while i + 1 < b and text[i + 1] in punctuation:
                i += 1
            while i + 1 < b and text[i + 1] in "」』）)\"'”’":
                i += 1
            out.append((start, i + 1, kind))
            start, kind = i + 1, "punct"
        i += 1
    end = b
    while end > start and text[end - 1] in _SPACE:
        end -= 1
    if start < end:
        out.append((start, end, kind))
    return out


@dataclass
class Unit:
    idx: int
    src: str
    e: int                 # source event index
    line: int              # line index inside the event
    a: int                 # offsets into the event text
    b: int
    text: str
    start: int             # ms, original timeline (proportional inside the caption)
    end: int
    speaker: Optional[str] = None
    interj: bool = False
    lyric: bool = False
    boundary: str = "event"        # boundary before this unit: event | line_speaker | line | marker | space | punct
    protect: float = 0.0           # extra cost of cutting right before this unit
    wstart: float = 0.0            # warped to the timing-base timeline (the translation's by default)
    wend: float = 0.0
    tags: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(1, len(_PUNCT_STRIP.sub("", self.text)))


def split_range(text: str, a: int, b: int, lang: str = "ja") -> list[tuple[int, int, str]]:
    """Split text[a:b] into phrases at spaces and sentence punctuation: (a, b, kind_of_boundary_before)."""
    if get_language(lang).analyzer_id != "ja-cc":
        return language_phrases(text, a, b, lang)
    pieces: list[tuple[int, int, str]] = []
    i = a
    first = True
    while i < b:
        while i < b and text[i] in _SPACE:
            i += 1
        if i >= b:
            break
        j = i
        while j < b and text[j] not in _SPACE:
            j += 1
        # inside the chunk: break after runs of sentence-final punctuation followed by more text
        k = i
        start = i
        kind = "space" if not first else "first"
        while k < j:
            if text[k] in SENT_END:
                while k + 1 < j and text[k + 1] in SENT_END:
                    k += 1
                if k + 1 < j and text[k + 1] not in "」』）)":
                    pieces.append((start, k + 1, kind))
                    start, kind = k + 1, "punct"
            k += 1
        pieces.append((start, j, kind))
        first = False
        i = j
    # glue orphaned 1-character pieces (a particle split off by a stray space) to the previous piece
    merged: list[tuple[int, int, str]] = []
    for pa, pb, kind in pieces:
        core = _PUNCT_STRIP.sub("", text[pa:pb])
        if (merged and kind == "space" and len(core) <= 1 and not is_interjection(text[pa:pb])
                and not any(c in SENT_END for c in text[pa:pb]) and text[merged[-1][1] - 1] not in SENT_END):
            merged[-1] = (merged[-1][0], pb, merged[-1][2])
        else:
            merged.append((pa, pb, kind))
    # keep repeated sound words together: コツ コツ / ガシャン ガシャン！
    out: list[tuple[int, int, str]] = []
    for pa, pb, kind in merged:
        if out and kind == "space" and _repeat_pair(text[out[-1][0]:out[-1][1]], text[pa:pb]):
            out[-1] = (out[-1][0], pb, out[-1][2])
        else:
            out.append((pa, pb, kind))
    return out


def _kata_core(s: str) -> str:
    core = _PUNCT_STRIP.sub("", s)
    return core if core and all("ァ" <= c <= "ヺ" or c in "ーッ" for c in core) else ""


def _repeat_pair(prev: str, cur: str) -> bool:
    a, b = _kata_core(prev), _kata_core(cur)
    if not a or not b:
        return False
    s = a + b
    n = len(s)
    return any(n % k == 0 and n // k >= 2 and s == s[:k] * (n // k) for k in range(1, min(5, n // 2) + 1))


def _protect_cost(prev_text: str) -> float:
    """How much a cut right after ``prev_text`` would break a phrase that belongs together."""
    t = prev_text.rstrip(" 　")
    if not t or t[-1] in SENT_END or t[-1] in "、，,":
        return 0.0
    if t.endswith("の") or t.endswith("な"):
        return 1.0            # modifier → head noun: 七実君の｜お眼鏡にかなう王子様
    if t.endswith(_HONORIFIC_END):
        return 0.6            # vocative / name → its sentence: 七実様｜落ち着いて！
    return 0.0


def build_units(src: Source) -> list[Unit]:
    """Units of every dialogue/lyric line of an original-language source, in source order."""
    units: list[Unit] = []
    for ev in src.events:
        if src.roles.get(ev.i, "dialogue") not in ("dialogue", "top"):
            continue
        an = analyzer_for(src.lang)[1](ev.i, ev.text, src.format)
        pieces: list[tuple[int, int, int, str, Optional[str], bool]] = []   # (line, a, b, boundary, speaker, lyric)
        prev_speaker: Optional[str] = None
        first_line = True
        for li, line in enumerate(an.lines):
            if line.kind not in ("speech", "lyric"):
                continue
            ranges = cc.speech_ranges(ev.text, an, line)
            for ri, (ra, rb) in enumerate(ranges):
                for pi, (pa, pb, kind) in enumerate(split_range(ev.text, ra, rb, src.lang)):
                    if pi == 0 and ri == 0:
                        if first_line:
                            boundary = "event"
                        elif line.speaker and prev_speaker and line.speaker != prev_speaker:
                            boundary = "line_speaker"
                        else:
                            boundary = "line"
                    elif pi == 0:
                        boundary = "marker"      # a metadata span (≫, (笑い)…) separated this part
                    else:
                        boundary = kind
                    pieces.append((li, pa, pb, boundary, line.speaker, line.kind == "lyric"))
            first_line = False
            prev_speaker = line.speaker or prev_speaker
        if not pieces:
            continue
        weights = [max(1, len(_PUNCT_STRIP.sub("", ev.text[pa:pb]))) + 1 for _li, pa, pb, *_ in pieces]
        total = sum(weights)
        dur = max(100, ev.end - ev.start)
        t = float(ev.start)
        first = len(units)
        for (li, pa, pb, boundary, spk, lyric), w in zip(pieces, weights):
            seg = dur * w / total
            text = ev.text[pa:pb]
            u = Unit(len(units), src.id, ev.i, li, pa, pb, text, int(round(t)), int(round(t + seg)), spk,
                     is_interjection(text, src.lang), lyric, boundary)
            if boundary in ("space", "punct") and len(units) > first:
                u.protect = _protect_cost(units[-1].text) if boundary == "space" and get_language(src.lang).analyzer_id == "ja-cc" else 0.0
            units.append(u)
            t += seg
        has_content = any(not u.interj for u in units[first:])
        for u in units[first:]:
            u.tags["shares_cue"] = has_content
    return units
