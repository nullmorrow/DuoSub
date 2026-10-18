"""Karaoke: one timing source (the ``@K`` record) → JP, ruby and CN lines.

The ``@K`` record of a song row is a Comment holding the sung kana with \\k
durations — time it in Aegisub's karaoke mode::

    {\\k32}う{\\k25}ま{\\k40}れ{\\k30}か{\\k28}わ{\\k45}る

If a song row has no ``@K`` yet, a template is created from the line's reading
(furigana for kanji, kana as written) with evenly spread placeholder timings.

Mapping: the record's kana are aligned with the JP line's reading stream, so
every kanji block, kana and ruby syllable gets its time — repeated kana are
resolved by position (the old 抱きしめて/手 bug cannot happen).
CN lines: an optional ``@K中`` record maps Chinese segments to kana
(``緊抱我（で） 世界（せかい）``); otherwise the Chinese line is spread
proportionally over the sung time span.

Effect slot: ``{@fx:name}`` in the JP or CN song line (or ``[karaoke] effect``).
Built in: kf (fill sweep, default), ko (outline sweep), k (instant), none.
Any other name is kept as a slot: the release writes ``karaoke-fx:name`` into
the Effect field, so a later effects script can rebuild fancier karaoke from
the \\k timing — the ordering/timing work done here is reused.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from .ass.tags import KARAOKE_TAGS
from .ass.text import VisibleText, private_blocks, remove_tags_named, tokenize
from .textutil import is_kana, is_kanji, kata_to_hira

BUILTIN_FX = {"kf": "kf", "ko": "ko", "k": "k", "none": None}
_SMALL = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")


@dataclass
class Syl:
    text: str
    dur: int          # cs
    start: int = 0    # cs from line start
    end: int = 0


def parse_k(text: str) -> list[Syl]:
    sy: list[Syl] = []
    cur: Optional[Syl] = None
    for tok in tokenize(text):
        if tok.kind == "tags":
            for t in tok.tags:
                if t.name in KARAOKE_TAGS:
                    v = t.num()
                    cur = Syl("", int(v or 0))
                    sy.append(cur)
        elif tok.kind in ("text", "hardspace"):
            raw = tok.raw if tok.kind == "text" else " "
            if cur is None:
                cur = Syl("", 0)
                sy.append(cur)
            cur.text += raw
    t = 0
    for s in sy:
        s.start, s.end = t, t + s.dur
        t += s.dur
    return sy


def morae(kana: str) -> list[str]:
    out: list[str] = []
    for ch in kana:
        if out and (ch in _SMALL or ch in "ーっッ"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def reading_stream(text: str, pieces) -> list[tuple[int, int, str]]:
    """Slots over the visible JP text: (start, end, reading).  Ruby pieces are
    one slot for their kanji block; kana are their own slots; punctuation/space
    have empty readings."""
    slots: list[tuple[int, int, str]] = []
    covered = {}
    for p in pieces:
        covered[p.start] = p
    i = 0
    while i < len(text):
        if i in covered:
            p = covered[i]
            slots.append((p.start, p.end, kata_to_hira(p.reading)))
            i = p.end
            continue
        ch = text[i]
        if is_kana(ch):
            slots.append((i, i + 1, kata_to_hira(ch)))
        elif ch.isalnum() and not is_kanji(ch):
            slots.append((i, i + 1, ch.lower()))
        else:
            slots.append((i, i + 1, ""))
        i += 1
    return slots


def template_record(text: str, pieces, duration: int) -> str:
    syls: list[str] = []
    for a, b, r in reading_stream(text, pieces):
        if not r:
            continue
        syls.extend(morae(r) if any(is_kana(c) for c in r) else [r])
    if not syls:
        return ""
    each = max(1, duration // len(syls))
    return "".join(f"{{\\k{each}}}{s}" for s in syls)


@dataclass
class KaraTiming:
    slot_times: list[tuple[int, int, int, int]]   # (vis start, vis end, t0, t1) cs from line start
    piece_syls: dict[int, list[Syl]] = field(default_factory=dict)  # piece start → its syllables
    sung: tuple[int, int] = (0, 0)
    unmatched: int = 0


def align(text: str, pieces, syls: list[Syl]) -> KaraTiming:
    slots = reading_stream(text, pieces)
    stream = "".join(r for _a, _b, r in slots)
    # reading char → slot index
    owner = []
    for si, (_a, _b, r) in enumerate(slots):
        owner.extend([si] * len(r))
    ksyl_chars = []
    ksyl_owner = []
    for k, s in enumerate(syls):
        t = kata_to_hira(re.sub(r"\s+", "", s.text)).lower()
        ksyl_chars.append(t)
        ksyl_owner.extend([k] * len(t))
    kstream = "".join(ksyl_chars)
    sm = difflib.SequenceMatcher(None, stream, kstream, autojunk=False)
    slot_syls: dict[int, set[int]] = {}
    matched = 0
    for a, b, size in sm.get_matching_blocks():
        for d in range(size):
            si = owner[a + d]
            ki = ksyl_owner[b + d]
            slot_syls.setdefault(si, set()).add(ki)
            matched += 1
    timing = KaraTiming([], unmatched=len(kstream) - matched)
    if syls:
        voiced = [s for s in syls if s.text.strip()]
        if voiced:
            timing.sung = (voiced[0].start, voiced[-1].end)
    last_end = 0
    for si, (a, b, r) in enumerate(slots):
        ks = sorted(slot_syls.get(si, ()))
        if ks:
            t0, t1 = syls[ks[0]].start, syls[ks[-1]].end
            last_end = t1
        else:
            t0 = t1 = last_end
        timing.slot_times.append((a, b, t0, t1))
        for p in pieces:
            if p.start == a and ks:
                timing.piece_syls[a] = [syls[k] for k in ks]
    return timing


def _tag(kind: str, dur: int) -> str:
    return f"{{\\{kind}{max(0, int(dur))}}}"


def karaoke_text(raw: str, timing: KaraTiming, kind: str) -> str:
    """Rebuild a JP line with karaoke tags before each slot (keeps other tags)."""
    clean = remove_tags_named(raw, KARAOKE_TAGS)
    vis = VisibleText(clean)
    inserts: dict[int, str] = {}
    prev_end = 0
    first = True
    for a, b, t0, t1 in timing.slot_times:
        if b <= a:
            continue
        if first and t0 > 0:
            inserts[a] = _tag("k", t0)
            prev_end = t0
        first = False
        dur = max(0, t1 - prev_end)
        inserts[a] = inserts.get(a, "") + _tag(kind, dur)
        prev_end = max(prev_end, t1)
    from .ass.text import insert_at_visible

    return insert_at_visible(clean, inserts) if vis.text else clean


def ruby_karaoke(reading: str, syls: list[Syl], kind: str) -> str:
    """Karaoke text for one ruby line: delay to the first syllable, then per syllable."""
    if not syls:
        return reading
    out = [_tag("k", syls[0].start)] if syls[0].start > 0 else []
    rpos = 0
    for s in syls:
        n = len(kata_to_hira(s.text).strip())
        chunk = reading[rpos : rpos + n] if n else ""
        rpos += n
        out.append(_tag(kind, s.dur) + chunk)
    if rpos < len(reading):
        out.append(reading[rpos:])
    return "".join(out)


def cn_karaoke(raw: str, timing: KaraTiming, kind: str, mapping: Optional[list[tuple[str, str]]], syls: list[Syl]) -> str:
    """CN lyric line: mapped segments (@K中) or proportional spread over the sung span."""
    clean = remove_tags_named(raw, KARAOKE_TAGS)
    vis = VisibleText(clean).text
    if not vis:
        return clean
    spans: list[tuple[int, int, int, int]] = []  # (vis a, vis b, t0, t1)
    if mapping:
        kstream = "".join(kata_to_hira(s.text) for s in syls)
        owner = [k for k, s in enumerate(syls) for _ in kata_to_hira(s.text)]
        cur_v, cur_k = 0, 0
        for seg, kana in mapping:
            a = vis.find(seg, cur_v)
            kk = kstream.find(kata_to_hira(kana), cur_k)
            if a < 0 or kk < 0:
                continue
            k0, k1 = owner[kk], owner[min(len(owner) - 1, kk + len(kana) - 1)]
            spans.append((a, a + len(seg), syls[k0].start, syls[k1].end))
            cur_v, cur_k = a + len(seg), kk + len(kana)
    if not spans:
        t0, t1 = timing.sung
        chars = [i for i, ch in enumerate(vis) if not ch.isspace()]
        if not chars or t1 <= t0:
            return clean
        n = len(chars)
        for j, i in enumerate(chars):
            spans.append((i, i + 1, t0 + (t1 - t0) * j // n, t0 + (t1 - t0) * (j + 1) // n))
    inserts: dict[int, str] = {}
    prev = 0
    for k, (a, b, t0, t1) in enumerate(sorted(spans)):
        if k == 0 and t0 > 0:
            inserts[a] = _tag("k", t0)
            prev = t0
        inserts[a] = inserts.get(a, "") + _tag(kind, max(0, t1 - prev))
        prev = max(prev, t1)
    from .ass.text import insert_at_visible

    return insert_at_visible(clean, inserts)


def fx_slot(*texts: str) -> Optional[str]:
    for t in texts:
        for _pos, body, _tok in private_blocks(t or ""):
            m = re.match(r"^fx\s*[:：]\s*(\S+)", body)
            if m:
                return m.group(1)
    return None


_MAP = re.compile(r"([^\s（）()]+)[（(]([^（）()]*)[）)]")


def parse_cn_mapping(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _MAP.finditer(text or "")]
