"""Non-destructive analysis of Japanese closed captions (CC).

The baseline cleaner *deleted* speaker labels, ≪≫ markers, (sound
descriptions), ((flashback)) brackets, inline readings and ♪ lines.  This
analyzer deletes nothing: it reports those pieces as metadata spans with exact
offsets into the event text, classifies each line, and attaches speakers.
Anything it is unsure about stays dialogue and gets a doubt note
(REQUIREMENTS 二.2: "不确定…时，保留并提示，而不是自动丢弃").

Offsets refer to the raw event text (SRT: lines joined by "\\n"; ASS: the Text
field verbatim; override blocks ``{…}`` are never part of a span).

SRT formatting (``<font …>``, ``<i>``, ``<b>``, ``{\\an8}`` …) is not dialogue: each tag becomes a ``format``
span (kept, auditable, its attributes in ``data``), so the working text has no markup while every source
character still has a disposition.  In a cue whose every line starts with a dialogue dash (``-A`` / ``-B``,
two speakers) the dashes are ``marker`` spans with ``data.turn``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..model.source import Span
from ..textutil import is_kana, is_kanji

ANALYZER = {"name": "ja-cc", "version": 3}       # 2: SRT formatting tags, dialogue dashes
#                                                     3: katakana in brackets after kanji is text, not a reading

OPEN = "(（"
CLOSE = ")）"
MUSIC_CHARS = set("♪♫♬～〜~ー－-・ 　")
NOTE_CHARS = "♪♫♬"
_KANA_ONLY = re.compile(r"^[ぁ-ゖァ-ヺー・\s　]+$")
_KATA_ONLY = re.compile(r"^[ァ-ヺー・\s　]+$")
_MARKER_RE = re.compile(r"\[[A-Za-z]+\]|［[Ａ-Ｚａ-ｚA-Za-z]+］")
_FB_OPEN = ("((", "（（", "(（", "（(")
_FB_CLOSE = ("))", "））", ")）", "）)")
_MASK = "\ue000"     # stands in for characters of ASS override blocks while analysing
_BLANK = " 　\t" + _MASK
_SRT_TAG = re.compile(r"</?(?:font|i|b|u|s|span|ruby|rt|rp|c)\b[^<>\n]*>|\{\\[^{}\n]*\}", re.IGNORECASE)
DASHES = "-－‐‑–"


def format_tags(text: str) -> list[tuple[int, int, str]]:
    """SRT formatting markup in a cue: (start, end, raw)."""
    return [(m.start(), m.end(), m.group(0)) for m in _SRT_TAG.finditer(text)]


def _tag_data(raw: str) -> dict:
    if raw.startswith("{"):
        return {"tag": "ass", "raw": raw}
    m = re.match(r"<\s*(/?)\s*([A-Za-z]+)", raw)
    out: dict = {"tag": (m.group(1) + m.group(2).lower()) if m else raw}
    attrs = {k.lower(): v for k, v in re.findall(r"([A-Za-z-]+)\s*=\s*[\"']([^\"']*)[\"']", raw)}
    if attrs:
        out["attrs"] = attrs
    return out


# ----------------------------------------------------------------- speakers
_SPEAKER_SUFFIX = re.compile(r"^(.+?)・(?:回想|心の声|ナレーション|独白|声|電話|.*モノマネ|.*ものまね)$")
_DESC_WORDS = {
    "足音", "物音", "爆音", "轟音", "雑音", "水音", "羽音", "靴音", "波音", "銃声", "歓声", "悲鳴", "拍手", "鼓動", "笑い",
    "咳払い", "いびき", "寝息", "あくび", "舌打ち", "口笛", "効果音", "BGM", "ＢＧＭ", "バイブ", "サイレン", "チャイム",
}
_DESC_TAIL = re.compile(r"(の音|の声|音声|音楽|ため息|息遣い|ざわめき|どよめき|拍手|鼓動|悲鳴|鼻歌|笑い|モード|で|ながら)$")


def normalize_speaker(name: str) -> str:
    """``城一郎・回想`` / ``幸平創真・心の声`` → the speaker alone (the raw label is kept in the span)."""
    m = _SPEAKER_SUFFIX.match(name)
    return m.group(1) if m else name


def looks_like_description(label: str) -> bool:
    """A bracketed note describing a sound or a manner of speaking, not who speaks."""
    core = re.sub(r"[「『][^」』]*[」』]", "", label).strip()
    if not core or core in _DESC_WORDS or core.startswith("鼻歌") or _DESC_TAIL.search(core):
        return True
    return len(core) >= 3 and core.endswith(("音", "声"))


def _balanced(s: str, i: int) -> int:
    """Index after the bracket closing the one opening at s[i]; -1 if unbalanced."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] in OPEN:
            depth += 1
        elif s[j] in CLOSE:
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def _base_start(t: str, i: int, lo: int, taken: list[bool]) -> int:
    """Start of the kanji (or Latin / digit) word right before t[i]."""
    k = i
    while k > lo and not taken[k - 1] and (is_kanji(t[k - 1]) or (t[k - 1].isalnum() and not is_kana(t[k - 1]))):
        k -= 1
    return k


def _inline_readings(t: str, lo: int, hi: int, taken: list[bool]) -> list[tuple[int, int, int, int, str]]:
    """``遠月(とおつき)`` inside t[lo:hi]: (base_a, base_b, paren_a, paren_b, reading).  Only hiragana is a reading:
    katakana after a word (``眷属(ファミリア)``) is another name for it, kept as text (see :func:`_gloss_brackets`)."""
    out = []
    annotated_upto = lo
    i = lo
    while i < hi:
        if t[i] in OPEN and i > lo and not taken[i]:
            j = _balanced(t, i)
            inner = t[i + 1: j - 1] if 0 < j <= hi else ""
            if inner and _KANA_ONLY.match(inner) and not _KATA_ONLY.match(inner):
                k = i
                while k > annotated_upto and not taken[k - 1] and (
                        is_kanji(t[k - 1]) or (t[k - 1].isalnum() and not is_kana(t[k - 1]))):
                    k -= 1
                if k < i and any(is_kanji(c) for c in t[k:i]):
                    out.append((k, i, i, j, inner.replace(" ", "").replace("　", "")))
                    annotated_upto = j
                    i = j
                    continue
        i += 1
    return out


def _gloss_brackets(t: str, lo: int, hi: int, taken: list[bool]) -> list[tuple[int, int]]:
    """``眷属(ファミリア)`` / ``迷宮都市(オラリオ)``: katakana in brackets right after a kanji word — the name the
    characters say, written next to the word.  It is neither a reading nor a sound description: it stays text."""
    out = []
    i = lo
    while i < hi:
        if t[i] in OPEN and i > lo and not taken[i]:
            j = _balanced(t, i)
            inner = t[i + 1: j - 1] if 0 < j <= hi else ""
            if inner.strip(" 　") and _KATA_ONLY.match(inner) and any(is_kanji(c) for c in t[_base_start(t, i, lo, taken):i]):
                out.append((i, j))
                i = j
                continue
        i += 1
    return out


def parse_label(label: str) -> tuple[str, list[list[str]]]:
    """``有栖川(ありすがわ)樹璃(じゅり)`` → ("有栖川樹璃", [["有栖川","ありすがわ"],["樹璃","じゅり"]])."""
    taken = [False] * len(label)
    hints = _inline_readings(label, 0, len(label), taken)
    name, readings, pos = [], [], 0
    for ba, bb, pa, pb, reading in hints:
        name.append(label[pos:pa])
        readings.append([label[ba:bb], reading])
        pos = pb
    name.append(label[pos:])
    return normalize_speaker("".join(name).strip()), readings


# ----------------------------------------------------------------- results
@dataclass
class LineInfo:
    a: int                       # content range of the line in the event text
    b: int
    kind: str                    # speech | lyric | sfx | music | label | empty
    speaker: Optional[str] = None
    speaker_raw: Optional[str] = None
    flashback: bool = False
    offscreen: bool = False


@dataclass
class EventAnalysis:
    lines: list[LineInfo] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    doubts: list[str] = field(default_factory=list)


def split_lines(text: str, fmt: str) -> list[tuple[int, int]]:
    """Line ranges of an event text (SRT: \\n; ASS/SSA: \\N or \\n escapes)."""
    seps = [m.span() for m in re.finditer(r"\n" if fmt == "srt" else r"\\[Nn]|\n", text)]
    out, pos = [], 0
    for a, b in seps:
        out.append((pos, a))
        pos = b
    out.append((pos, len(text)))
    return out


def _masked(text: str) -> str:
    """Replace ASS override blocks by mask characters (same length → same offsets)."""
    return re.sub(r"\{[^}]*\}", lambda m: _MASK * len(m.group(0)), text)


def analyze_event(e: int, text: str, fmt: str = "srt") -> EventAnalysis:
    res = EventAnalysis()
    tags = format_tags(text) if fmt == "srt" else []
    if tags:
        chars = list(text)
        for a, b, _raw in tags:
            chars[a:b] = [_MASK] * (b - a)
        t = "".join(chars)
    else:
        t = _masked(text) if fmt != "srt" else text
    taken = [ch == _MASK for ch in t]
    for a, b, raw in tags:
        res.spans.append(Span(e, a, b, "format", raw, None, _tag_data(raw)))

    def span(a: int, b: int, kind: str, value: Optional[str] = None, base=None, data=None) -> None:
        if b <= a:
            return
        for k in range(a, b):
            taken[k] = True
        res.spans.append(Span(e, a, b, kind, value, base, data))

    pending_label: Optional[tuple[str, Optional[str], list]] = None   # label-only line waiting for its speech line
    lines = split_lines(text, fmt)
    dashed = _all_dashed(t, lines)
    for la, lb in lines:
        info = LineInfo(la, lb, "empty")
        p, q = la, lb
        while p < q and t[p] in _BLANK:
            p += 1
        while q > p and t[q - 1] in _BLANK:
            q -= 1
        if p >= q:
            res.lines.append(info)
            continue
        # dialogue dash of a two-speaker cue (-A / -B)
        if dashed and t[p] in DASHES:
            span(p, p + 1, "marker", t[p], data={"turn": True})
            p += 1
            while p < q and t[p] in _BLANK:
                p += 1
            if p >= q:
                res.lines.append(info)
                continue
        # leading off-screen markers ≪ ≫
        while p < q and t[p] in "≪≫":
            span(p, p + 1, "marker", t[p])
            info.offscreen = True
            p += 1
            while p < q and t[p] in " 　":
                p += 1
        # trailing continuation arrow
        if q > p and t[q - 1] == "➡":
            span(q - 1, q, "marker", "➡")
            q -= 1
            while q > p and t[q - 1] in " 　":
                q -= 1
        # whole line music (♪～)
        body = t[p:q]
        if body and all(c in MUSIC_CHARS for c in body) and any(c in NOTE_CHARS for c in body):
            for k in range(p, q):
                if not t[k].isspace() and not taken[k]:
                    span(k, k + 1, "music", t[k])
            info.kind = "music"
            res.lines.append(info)
            continue
        # speaker label at line start: (城一郎)… / （有栖川(ありすがわ)樹璃(じゅり)）…
        label_done = False
        if t[p] in OPEN and not t.startswith(_FB_OPEN, p):
            j = _balanced(t, p)
            if 0 < j <= q:
                raw_label = t[p + 1: j - 1]
                name, readings = parse_label(raw_label)
                if j == q:
                    # the whole line is one bracket: a sound description, or a label for the next line
                    info.kind = "sfx"
                    info.speaker_raw = raw_label
                    if readings or not looks_like_description(name):
                        pending_label = (name, raw_label, readings)
                        span(p, j, "speaker_label", name, data={"readings": readings, "raw": raw_label, "line_only": True})
                        info.kind = "label"
                    else:
                        span(p, j, "sfx", raw_label)
                    res.lines.append(info)
                    continue
                if readings or not looks_like_description(name):
                    span(p, j, "speaker_label", name, data={"readings": readings, "raw": raw_label})
                    info.speaker, info.speaker_raw = name, raw_label
                else:
                    span(p, j, "sfx", raw_label)       # (棒読みで)よろしく — manner, not a speaker
                label_done = True
                p = j
                while p < q and t[p] in " 　":
                    p += 1
                while p < q and t[p] in "≪≫":
                    span(p, p + 1, "marker", t[p])
                    info.offscreen = True
                    p += 1
                    while p < q and t[p] in " 　":
                        p += 1
        # ((flashback))
        if t.startswith(_FB_OPEN, p):
            span(p, p + 2, "flashback_mark", t[p:p + 2])
            info.flashback = True
            p += 2
        seg = t[p:q]
        opens = sum(seg.count(c) for c in OPEN)
        closes = sum(seg.count(c) for c in CLOSE)
        if q - p >= 2 and t[q - 2:q] in _FB_CLOSE and closes - opens >= 2:
            span(q - 2, q, "flashback_mark", t[q - 2:q])
            info.flashback = True
            q -= 2
        if p >= q:
            info.kind = "label" if label_done else "empty"
            res.lines.append(info)
            continue
        # lyric line: ♪ at either end
        if t[p] in NOTE_CHARS or t[q - 1] in NOTE_CHARS:
            info.kind = "lyric"
            while p < q and (t[p] in NOTE_CHARS or (t[p] in "～〜 　" and p + 1 < q and t[p + 1] in NOTE_CHARS)):
                if not t[p].isspace():
                    span(p, p + 1, "marker", t[p])
                p += 1
            while q > p and (t[q - 1] in NOTE_CHARS or t[q - 1] in "～〜" and q - 2 >= p and t[q - 2] in NOTE_CHARS):
                span(q - 1, q, "marker", t[q - 1])
                q -= 1
        # inner markers: [TEL], ≪ ≫, ➡
        for m in _MARKER_RE.finditer(t, p, q):
            if not any(taken[m.start():m.end()]):
                span(m.start(), m.end(), "marker", m.group(0))
        for k in range(p, q):
            if t[k] in "≪≫➡" and not taken[k]:
                span(k, k + 1, "marker", t[k])
        # inline readings 遠月(とおつき)
        for ba, bb, pa, pb, reading in _inline_readings(t, p, q, taken):
            span(pa, pb, "reading_hint", reading, base=[ba, bb])
        # 眷属(ファミリア): text — kept out of the description rule below without becoming a span
        for ga, gb in _gloss_brackets(t, p, q, taken):
            for k in range(ga, gb):
                taken[k] = True
        # remaining brackets: descriptions …(笑い)
        described = False
        k = p
        while k < q:
            if t[k] in OPEN and not taken[k]:
                j = _balanced(t, k)
                if 0 < j <= q:
                    span(k, j, "sfx", t[k + 1: j - 1])
                    described = True
                    k = j
                    continue
            k += 1
        rest = "".join(t[k] for k in range(p, q) if not taken[k]).strip(" 　")
        if not rest:
            info.kind = "sfx" if described else "empty"
        else:
            if info.kind != "lyric":
                info.kind = "speech"
            if described and all(is_kana(c) or c in "ー～〜…！？!? 　っッ" for c in rest) and not any("ぁ" <= c <= "ゖ" for c in rest):
                res.doubts.append(f"「{rest}」可能是效果音的拟声词，按台词保留")
        if info.kind in ("speech", "lyric") and info.speaker is None and pending_label is not None:
            info.speaker, info.speaker_raw = pending_label[0], pending_label[1]
        if info.kind in ("speech", "lyric"):
            pending_label = None
        info.a, info.b = la, lb
        res.lines.append(info)
    res.spans.sort(key=lambda s: (s.a, s.b))
    return res


def _all_dashed(t: str, lines: list[tuple[int, int]]) -> bool:
    """Every non-empty line of the cue starts with a dialogue dash followed by text (two or more lines)."""
    heads = []
    for la, lb in lines:
        k = la
        while k < lb and t[k] in _BLANK:
            k += 1
        if k < lb:
            heads.append((k, lb))
    return len(heads) >= 2 and all(t[k] in DASHES and k + 1 < lb and t[k + 1] not in DASHES + _BLANK for k, lb in heads)


def speech_ranges(text: str, analysis: EventAnalysis, line: LineInfo) -> list[tuple[int, int]]:
    """Maximal ranges of a line that are not metadata (the dialogue text, offsets into ``text``)."""
    covered = [False] * len(text)
    for s in analysis.spans:
        for k in range(s.a, s.b):
            covered[k] = True
    out: list[tuple[int, int]] = []
    k = line.a
    while k < line.b:
        if covered[k]:
            k += 1
            continue
        j = k
        while j < line.b and not covered[j]:
            j += 1
        a, b = k, j
        while a < b and text[a] in " 　\t":
            a += 1
        while b > a and text[b - 1] in " 　\t":
            b -= 1
        if a < b:
            out.append((a, b))
        k = j
    return out
