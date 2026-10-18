"""Pure lyric mapping validation; timing uses the existing karaoke edit/link format."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..textutil import kata_to_hira
from .readings import visible_map


@dataclass(frozen=True)
class Segment:
    text: str
    a: int
    b: int
    first: int
    last: int                         # exclusive original beat index


def _ignored(ch):
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def _kana(text):
    return kata_to_hira("".join(ch for ch in text if not ch.isspace())).casefold()


def member(doc, group, lane):
    return next((m for m in doc.members_of(group) if m.lane == lane and m.role != "comment"
                 and m.source_kind != "Comment" and visible_map(m)[0].text.strip()), None)


def disabled_reason(doc, group, languages):
    from ..languages import lane_label

    if group is None or group.kind != "song":
        return "（不是歌曲）"
    for lane in ("original", "translation"):
        if member(doc, group, lane) is None:
            return f"（这一组没有{lane_label(lane, languages)}）"
    return ""


def parse(text: str, originals, translation: str) -> list[Segment]:
    """Return canonical visible translation spans and consecutive whole original beats."""
    raw = []
    outside, inside, opening = "", "", ""
    for ch in text:
        if ch in "（(":
            if opening:
                raise ValueError("括号不成对")
            opening, inside = ch, ""
        elif ch in "）)":
            if not opening or (opening, ch) not in (("（", "）"), ("(", ")")):
                raise ValueError("括号不成对")
            if not _kana(inside) or not any(not _ignored(c) for c in outside):
                raise ValueError("有一段没有对应任何拍")
            raw.append((outside, inside))
            outside, opening = "", ""
        elif opening:
            inside += ch
        else:
            outside += ch
    if opening:
        raise ValueError("括号不成对")
    if any(not _ignored(ch) for ch in outside) or not raw:
        raise ValueError("有一段没有对应任何拍")
    positions = [i for i, ch in enumerate(translation) if not _ignored(ch)]
    expected = "".join(translation[i] for i in positions)
    supplied = "".join(ch for part, _ in raw for ch in part if not _ignored(ch))
    if supplied != expected:
        for op, a, b, c, d in SequenceMatcher(None, expected, supplied, autojunk=False).get_opcodes():
            if op != "equal":
                if c != d:
                    raise ValueError(f"翻译文字对不上：多了「{supplied[c:d]}」")
                raise ValueError(f"翻译文字对不上：少了「{expected[a:b]}」")
    out, at, count = [], 0, 0
    for part, kana in raw:
        first, target, sung = at, _kana(kana), ""
        while at < len(originals) and len(sung) < len(target):
            next_sung = _kana(originals[at].sung)
            if not target.startswith(sung + next_sung):
                raise ValueError(f"括号里的假名和原声拍对不上：第 {at + 1} 拍应该是「{originals[at].sung}」")
            sung += next_sung
            at += 1
        if sung != target:
            wanted = originals[at].sung if at < len(originals) else ""
            raise ValueError(f"括号里的假名和原声拍对不上：第 {at + 1} 拍应该是「{wanted}」")
        a = 0 if count == 0 else positions[count]
        count += sum(not _ignored(ch) for ch in part)
        b = positions[count] if count < len(positions) else len(translation)
        out.append(Segment(translation[a:b], a, b, first, at))
    if at != len(originals):
        raise ValueError(f"括号里的假名和原声拍对不上：第 {at + 1} 拍应该是「{originals[at].sung}」")
    return out


def format_mapping(segments, originals) -> str:
    return "".join(s.text + "（" + "".join(b.sung for b in originals[s.first:s.last]) + "）" for s in segments)


def auto_mapping(originals, translation: str, *, follow_notes: bool = True) -> str:
    """Phrase-aware proposal using whole beats (the text format cannot express splits)."""
    from .lyrics_auto import redistribute

    segments = [Segment(s.text, s.a, s.b, s.start.beat, s.end.beat + 1)
                for s in redistribute(originals, translation, splits=False, follow_notes=follow_notes)]
    return format_mapping(segments, originals)


def proportional_mapping(originals, translation: str) -> str:
    """Legacy average proposal, retained only to recognize uncustomized mappings."""
    units = list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\s]", translation))
    units = [u for u in units if any(not _ignored(ch) for ch in u.group())]
    n = min(len(units), len(originals))
    if not n:
        return ""
    segments = []
    for i in range(n):
        a = 0 if i == 0 else units[len(units) * i // n].start()
        b = units[len(units) * (i + 1) // n].start() if i + 1 < n else len(translation)
        segments.append(Segment(translation[a:b], a, b, len(originals) * i // n,
                                len(originals) * (i + 1) // n))
    return format_mapping(segments, originals)


def current_mapping(doc, original, translation):
    from . import karaoke as kara

    originals = kara.effective(doc, original).beats
    beats = kara.effective(doc, translation).beats
    keys = {b.key: i for i, b in enumerate(originals)}
    segments = []
    text = visible_map(translation)[0].text
    for b in beats:
        refs = []
        for edge in ("start", "end"):
            link = b.link.get(edge, {})
            key = (kara._ids_of_anchor(link.get("core", [])), link.get("k", 0), link.get("sung", ""))
            if key not in keys or link.get("edge") != edge:
                return ""
            refs.append(keys[key])
        segments.append(Segment(text[b.a:b.b], b.a, b.b, refs[0], refs[1] + 1))
    result = format_mapping(segments, originals)
    try:
        parse(result, originals, text)
    except ValueError:
        return ""
    return result


def mapped_beats(doc, translation, originals, segments):
    from . import karaoke as kara

    view = kara.effective(doc, translation)
    base = kara.derive(doc, translation, view.unit)
    boundaries = {s.a for s in segments[1:]}
    edits = [{"op": "split" if p.a in boundaries else "merge", "at": p.at}
             for b in base for p in b.parts][1:]
    beats = kara.derive(doc, translation, view.unit, edits)
    if len(beats) != len(segments) or any((b.a, b.b) != (s.a, s.b) for b, s in zip(beats, segments)):
        raise ValueError(kara.EDIT_MISSING)
    for b, s in zip(beats, segments):
        first, last = originals[s.first], originals[s.last - 1]
        b.t0, b.t1 = first.t0, last.t1
        b.by = "manual" if all(o.timed for o in originals[s.first:s.last]) else "placeholder"
        b.link = {"start": kara.beat_link(first, "start"), "end": kara.beat_link(last, "end")}
    return beats, edits, view
