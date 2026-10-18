"""Phrase-first lyric redistribution, using only derived beats and existing edits/links.

The two DPs are O(C*J) for phrase pairs and O(U*B**2) inside a pair
(at most five candidate cuts per beat). No models, media or Qt are involved.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from functools import lru_cache

from ..kalign.tokens import continuation_morae, word_starts
from ..textutil import kata_to_hira
from . import karaoke as kara, lyrics_map as lm
from .readings import visible_map

SEP = set(" 　、，,。.…・!！?？「」『』（）()~～—-")
SHINJITAI = dict(zip("変歩気涙戦桜楽読観帰続実円広鉄駅売図発様歳薬単満険験絵縁黒譲転伝辺隣闘覚触舞",
                    "变步气泪战樱乐读观归续实圆广铁驿卖图发样岁药单满险验绘缘黑让转传边邻斗觉触舞"))


@dataclass(frozen=True)
class Unit:
    text: str
    a: int
    b: int
    brk: bool


@dataclass(frozen=True)
class Reference:
    beat: int
    fraction: float


@dataclass(frozen=True)
class Segment:
    text: str
    a: int
    b: int
    start: Reference
    end: Reference


def translation_units(text: str) -> list[Unit]:
    """Keep punctuation/spacing in visible spans; Latin words are indivisible units."""
    matches = [m for m in re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|\S", text)
               if any(c not in SEP and not lm._ignored(c) for c in m.group())]
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append(Unit(m.group(), 0 if i == 0 else m.start(), end,
                        i + 1 == len(matches) or any(c.isspace() or c in SEP for c in text[m.end():end])))
    return out


def morae(sung: str) -> float:
    """Small kana join; closures count zero, long vowels and ん count one."""
    return max(0.5, sum(c not in "ぁぃぅぇぉゃゅょゎゕゖっ" for c in kata_to_hira(sung)))


STRONG_MERGE, WEAK_MERGE = 0.4, 0.1           # cost of pairing across a Japanese phrase break
STRONG_STRADDLE, WEAK_STRADDLE = 1.0, 0.3     # cost of one translation unit spanning it


def phrase_breaks(beats, original_text=None) -> dict[int, bool]:
    """Boundary index → strong?  A separator ending a word is strong; a rest of at least 150 ms
    between two words is weak (singers breathe inside phrases); a rest inside one word
    (between the beats of a kanji reading, e.g. ま｜ど of 窓) never breaks."""
    out = {}
    for i, beat in enumerate(beats[:-1]):
        # Ruby beats share a base span: its trailing separator belongs to the last one.
        if (beat.a, beat.b) == (beats[i + 1].a, beats[i + 1].b):
            continue
        tail = original_text[beat.a:beat.b] if original_text is not None else beat.base_text
        if tail and tail[-1] in SEP:
            out[i + 1] = True
        elif beats[i + 1].t0 - beat.t1 >= 150:
            out[i + 1] = False
    return out


def _base_text(beats) -> str:
    """The original text rebuilt from beat bases (offsets kept; uncovered characters are NUL)."""
    chars = {}
    for b in beats:
        for k, ch in enumerate(b.base_text or ""):
            chars[b.a + k] = ch
    return "".join(chars.get(i, "\0") for i in range(max(chars, default=-1) + 1))


def phrase_groups(beats, original_text=None):
    cuts = [0, *sorted(phrase_breaks(beats, original_text)), len(beats)]
    return list(zip(cuts, cuts[1:])) if beats else []


@lru_cache(maxsize=1)
def _converter():
    import opencc
    return opencc.OpenCC("t2s")


@lru_cache(maxsize=4096)
def norm_han(text, japanese=False):
    if japanese:
        text = "".join(SHINJITAI.get(c, c) for c in text)
    return _converter().convert(text)


ANCHOR_REACH = 0.25          # a shared character anchors only at about the same place in the phrase


def _anchors(units, beats, original_text):
    """Shared characters (勇氣 / 勇気) as anchors, unless they sit at clearly different places
    in the two phrases (產生… / …を生め): those are different words, and following them
    would crowd the translation onto a few beats."""
    bases = [original_text[b.a:b.b] if original_text is not None else b.base_text for b in beats]
    han = [{norm_han(c, True) for c in text if "一" <= c <= "鿿"} for text in bases]
    out = {}
    for i, unit in enumerate(units):
        if len(unit.text) == 1:
            at = (i + 0.5) / len(units)
            hits = [j for j, chars in enumerate(han) if norm_han(unit.text) in chars
                    and abs((j + 0.5) / len(beats) - at) <= ANCHOR_REACH]
            if hits:
                out[i] = hits
    return out


def _coalesce(groups, limit):
    """If the 1:3 topology is impossible, fold the shortest adjacent phrases first."""
    groups = list(groups)
    while len(groups) > limit:
        i = min(range(len(groups) - 1), key=lambda i: groups[i + 1][1] - groups[i][0])
        groups[i:i + 2] = [(groups[i][0], groups[i + 1][1])]
    return groups


def _phrase_pairs(cn, jp, weights, strong):
    """Pair translation and Japanese phrases; returns the Japanese breaks inside each pair
    as (boundary, straddle cost)."""
    cn = _coalesce(cn, 3 * len(jp))
    jp = _coalesce(jp, 3 * len(cn))
    cw = [z - a for a, z in cn]
    jw = [sum(weights[a:z]) for a, z in jp]
    tc, tj = sum(cw), sum(jw)
    best = [[math.inf] * (len(jp) + 1) for _ in range(len(cn) + 1)]
    prev = {}
    best[0][0] = 0.0
    for i in range(len(cn) + 1):
        for j in range(len(jp) + 1):
            if not math.isfinite(best[i][j]):
                continue
            for di, dj in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)):
                if i + di > len(cn) or j + dj > len(jp):
                    continue
                cost = abs(math.log((sum(cw[i:i + di]) / tc) / (sum(jw[j:j + dj]) / tj)))
                cost += STRONG_MERGE * (di - 1)
                cost += sum(STRONG_MERGE if strong.get(z, True) else WEAK_MERGE for _a, z in jp[j:j + dj - 1])
                value = best[i][j] + cost
                if value < best[i + di][j + dj]:
                    best[i + di][j + dj] = value
                    prev[i + di, j + dj] = i, j
    pairs, i, j = [], len(cn), len(jp)
    while i or j:
        pi, pj = prev[i, j]
        pairs.append((cn[pi][0], cn[i - 1][1], jp[pj][0], jp[j - 1][1],
                      [(z, STRONG_STRADDLE if strong.get(z, True) else WEAK_STRADDLE) for _a, z in jp[pj:j - 1]]))
        i, j = pi, pj
    return pairs[::-1]


def _spans(units, weights, breaks, anchors, splits):
    cuts = [(0, 0.0)]
    for j in range(len(weights)):
        if splits:
            cuts.extend((j, f) for f in (1 / 3, 1 / 2, 2 / 3))
        cuts.append((j + 1, 0.0))
    pos = [j + f for j, f in cuts]
    cum = [0.0]
    for w in weights:
        cum.append(cum[-1] + w)
    amounts = [cum[j] + (f * weights[j] if f else 0) for j, f in cuts]
    target = cum[-1] / len(units)
    costs = [[math.inf] * len(cuts) for _ in range(len(cuts))]
    for e in range(1, len(cuts)):
        for s in range(e):
            costs[s][e] = ((amounts[e] - amounts[s]) / target - 1) ** 2
            costs[s][e] += 0.35 * bool(cuts[e][1])
            costs[s][e] += sum(c for b, c in breaks if pos[s] < b < pos[e])
    best = [[math.inf] * len(cuts) for _ in range(len(units) + 1)]
    prev = [[-1] * len(cuts) for _ in best]
    best[0][0] = 0.0
    for u in range(1, len(units) + 1):
        hits = anchors.get(u - 1, ())
        for e in range(u, len(cuts)):
            for s in range(u - 1, e):
                if not math.isfinite(best[u - 1][s]):
                    continue
                cost = costs[s][e]
                if hits and not any(math.floor(pos[s]) <= j < math.ceil(pos[e]) for j in hits):
                    cost += 0.8
                value = best[u - 1][s] + cost
                if value < best[u][e]:
                    best[u][e], prev[u][e] = value, s
    out, e = [], len(cuts) - 1
    for u in range(len(units), 0, -1):
        s = prev[u][e]
        if s < 0:
            raise ValueError("翻译单位多于可分配的拍")
        out.append((cuts[s], cuts[e]))
        e = s
    return out[::-1]


def _whole_segments(units, weights, breaks, anchors):
    """Whole-beat proposal DP; merge adjacent units in place of fractional cuts.

    A merge costs 0.35 per extra unit, so shared-character anchors can justify
    merging even when there are enough beats. Complexity O(U**2 * B**2).
    """
    cum = [0.0]
    for w in weights:
        cum.append(cum[-1] + w)
    target = cum[-1] / len(units)
    best = [[math.inf] * (len(weights) + 1) for _ in range(len(units) + 1)]
    prev = {}
    best[0][0] = 0.0
    for u in range(1, len(units) + 1):
        for e in range(1, len(weights) + 1):
            for v in range(u):
                for s in range(e):
                    if not math.isfinite(best[v][s]):
                        continue
                    count = u - v
                    cost = (((cum[e] - cum[s]) / (target * count) - 1) ** 2) * count
                    cost += 0.35 * (count - 1) + sum(c for b, c in breaks if s < b < e)
                    cost += 0.8 * sum(bool(anchors.get(i)) and not any(s <= j < e for j in anchors[i])
                                      for i in range(v, u))
                    value = best[v][s] + cost
                    if value < best[u][e]:
                        best[u][e] = value
                        prev[u, e] = v, s
    out, u, e = [], len(units), len(weights)
    while u:
        v, s = prev[u, e]
        out.append((Unit("".join(x.text for x in units[v:u]), units[v].a, units[u - 1].b, False),
                    ((s, 0.0), (e, 0.0))))
        u, e = v, s
    return out[::-1]


def _note_reference(originals, notes, index, fraction, *, end=False):
    """Convert a note cut to a linked beat edge or an interior fraction (integer ms)."""
    if not fraction:
        return Reference(notes[index - 1][1] - 1, 1.0) if end else Reference(notes[index][0], 0.0)
    a, z = notes[index]
    instant = round(originals[a].t0 + fraction * (originals[z - 1].t1 - originals[a].t0))
    for j in range(a, z):
        beat = originals[j]
        if instant <= beat.t0:
            return Reference(j, 0.0)
        if instant == beat.t1 and not end and j + 1 < z and originals[j + 1].t0 == instant:
            return Reference(j + 1, 0.0)
        if instant <= beat.t1:
            return Reference(j, (instant - beat.t0) / (beat.t1 - beat.t0))
    return Reference(z - 1, 1.0)


def redistribute(originals, translation: str, *, original_text=None, splits=True,
                 follow_notes=False) -> list[Segment]:
    """Return visible spans and original beat/fraction references, preserving all text."""
    units = translation_units(translation)
    if not originals or not units:
        return []
    notes = []
    beat_breaks = phrase_breaks(originals, original_text)
    # A note never spans a word start (ない｜今: い｜い) or a phrase break.
    starts = word_starts([(b.a, b.ann is not None) for b in originals],
                         original_text if original_text is not None else _base_text(originals))
    starts = [s or i in beat_breaks for i, s in enumerate(starts)]
    continuations = continuation_morae([b.sung for b in originals], starts) if follow_notes else [False] * len(originals)
    for i, continuation in enumerate(continuations):
        if continuation and notes:
            notes[-1] = (notes[-1][0], i + 1)
        else:
            notes.append((i, i + 1))
    durs = [sum(max(0, b.t1 - b.t0) for b in originals[a:z]) for a, z in notes]
    mora = [sum(morae(b.sung) for b in originals[a:z]) for a, z in notes]
    mean_d, mean_m = sum(durs) / len(durs), sum(mora) / len(mora)
    weights = [0.6 * m / mean_m + 0.4 * (d / mean_d if mean_d else 1) for d, m in zip(durs, mora)]
    strong = {i: beat_breaks[a] for i, (a, _z) in enumerate(notes) if a in beat_breaks}
    cuts = [0, *sorted(strong), len(notes)]
    jp, cn, start = list(zip(cuts, cuts[1:])), [], 0
    for i, unit in enumerate(units):
        if unit.brk:
            cn.append((start, i + 1))
            start = i + 1
    out = []
    for ua, ub, ba, bb, breaks in _phrase_pairs(cn, jp, weights, strong):
        sub = units[ua:ub]
        capacity = (3 if splits else 1) * (bb - ba)
        if splits and len(sub) > capacity:
            # Even three-way splits have finite capacity; preserve excess text by merging.
            sub = [Unit("".join(u.text for u in sub[len(sub) * i // capacity:len(sub) * (i + 1) // capacity]),
                        sub[len(sub) * i // capacity].a, sub[len(sub) * (i + 1) // capacity - 1].b, False)
                   for i in range(capacity)]
        first, last = notes[ba][0], notes[bb - 1][1]
        beat_to_note = {j - first: i - ba for i in range(ba, bb) for j in range(*notes[i])}
        anchors = {u: list(dict.fromkeys(beat_to_note[j] for j in hits))
                   for u, hits in _anchors(sub, originals[first:last], original_text).items()}
        boundaries = [(b - ba, c) for b, c in breaks]
        can_split = splits and (not follow_notes or len(units) > len(notes) and len(sub) > bb - ba)
        if splits and (can_split or len(sub) <= bb - ba):
            assigned = zip(sub, _spans(sub, weights[ba:bb], boundaries, anchors, can_split))
        else:
            assigned = _whole_segments(sub, weights[ba:bb], boundaries, anchors)
        for unit, ((j0, f0), (j1, f1)) in assigned:
            if follow_notes:
                start_ref = _note_reference(originals, notes, ba + j0, f0)
                end_ref = _note_reference(originals, notes, ba + j1, f1, end=True)
            else:
                start_ref = Reference(ba + j0, f0)
                end_ref = Reference(ba + j1, f1) if f1 else Reference(ba + j1 - 1, 1.0)
            out.append(Segment(translation[unit.a:unit.b], unit.a, unit.b, start_ref, end_ref))
    return out


def mapped_beats(doc, translation, originals, segments):
    """Reuse lyric edits; only edges on original boundaries receive stored links."""
    whole = [lm.Segment(s.text, s.a, s.b, s.start.beat, s.end.beat + 1) for s in segments]
    beats, edits, view = lm.mapped_beats(doc, translation, originals, whole)
    for beat, segment in zip(beats, segments):
        beat.link = {}
        for edge, ref in (("start", segment.start), ("end", segment.end)):
            original = originals[ref.beat]
            value = round(original.t0 + ref.fraction * (original.t1 - original.t0))
            setattr(beat, "t0" if edge == "start" else "t1", value)
            if ref.fraction in (0.0, 1.0):
                beat.link[edge] = kara.beat_link(original, "start" if ref.fraction == 0 else "end")
            elif "auto_split" not in beat.flags:
                beat.flags.append("auto_split")      # timed with its original span (follow_links)
    return beats, edits, view


def _uncustomized(doc, original, translation):
    tracks = kara.tracks_of(doc, translation.id)
    if not tracks:
        return True
    beats = kara.effective(doc, translation).beats
    if any(b.by in ("tap", "drag") for b in beats):
        return False
    return all(b.by == "placeholder" for b in beats) or lm.current_mapping(doc, original, translation) == \
        lm.proportional_mapping(kara.effective(doc, original).beats, visible_map(translation)[0].text)


def redistribute_in_transaction(editor, tx, gids, *, only_uncustomized, follow_notes=True):
    """Write through the caller's transaction; twins and manual protection are per member."""
    doc = editor.doc
    report = {"updated": 0, "updated_members": 0, "skipped": {}}
    tx.keeps_text = True
    seen = set()
    for gid in dict.fromkeys(gids):
        group = doc.group(gid)
        original, translation = lm.member(doc, group, "original"), lm.member(doc, group, "translation")
        if group.kind != "song" or original is None or translation is None:
            report["skipped"][gid] = "需要歌曲的原声和翻译"
            continue
        originals = kara.effective(doc, original).beats
        segments = redistribute(originals, visible_map(translation)[0].text,
                                original_text=visible_map(original)[0].text, follow_notes=follow_notes)
        if not segments:
            report["skipped"][gid] = "没有可分配的拍"
            continue
        updated = False
        for mid in [translation.id] + [tid for tid in editor.karaoke_twins(translation.id)
                                      if doc.members[tid].group == gid]:
            if mid in seen:
                continue
            seen.add(mid)
            target = doc.members[mid]
            if only_uncustomized and not _uncustomized(doc, original, target):
                report["skipped"][mid] = "翻译拍已自定义"
                continue
            beats, edits, view = mapped_beats(doc, target, originals, segments)
            old = kara.tracks_of(doc, mid)
            track = kara.beats_to_track(target, beats, unit=view.unit,
                state="timed" if all(b.timed for b in beats) else "placeholder",
                kid=old[0]["id"] if old else None,
                effect=copy.deepcopy(old[0].get("effect")) if old else None, edits=edits)
            for previous in old[1:]:
                tx.remove_track(previous["id"])
            tx.set_track(track)
            updated = True
            report["updated_members"] += 1
        report["updated"] += int(updated)
    return report
