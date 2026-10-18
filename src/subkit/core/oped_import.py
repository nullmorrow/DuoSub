"""Read-only OPED parsing and conversion through the existing passage pipeline.

Times (including beats) are absolute episode milliseconds. The real OPED-DB
OP.song starts at 9.13 s and 絕對運命默示錄.song at 834.73 s, exactly their
NPZ start_sec; their last lyric ends equal NPZ end_sec (88.27 / 927.23 s).
Those files are NOT song-relative. NPZ features are neither loaded nor applied.
Visible offsets use the same ASS tokenizer as the rest of the core.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

from ..ass import AssDocument
from ..ass.document import DEFAULT_EVENT_FORMAT, Event
from ..ass.text import plain_text, tokenize
from ..encoding import decode_bytes
from ..model.document import Document, Group, Member
from ..reading.engine import ReadingEngine
from ..reading.kana_align import align
from ..textutil import is_kanji, kata_to_hira
from .songs import song_span


@dataclass
class SongLine:
    start_ms: int
    end_ms: int
    family: str
    original: str
    translation: str
    readings: list[tuple[int, int, str]]
    beats: list[tuple[str, int, int]]
    mapping: str | None
    source_file: str
    line_number: int
    translation_beats: list[tuple[str, int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    category: str = "NA"
    song_header: int = 0


@dataclass
class UnrecognisedLine:
    source_file: str
    line_number: int
    text: str
    reason: str
    block_number: int | None = None


@dataclass
class Song:
    family: str
    lines: list[SongLine]
    category: str = "NA"

    @property
    def start_ms(self):
        return min(line.start_ms for line in self.lines)

    @property
    def end_ms(self):
        return max(line.end_ms for line in self.lines)


@dataclass
class ImportResult:
    lines: list[SongLine] = field(default_factory=list)
    songs: list[Song] = field(default_factory=list)
    unrecognised: list[UnrecognisedLine] = field(default_factory=list)


@dataclass
class _Source:
    number: int
    raw: str
    event: Event | None


def _family(style):
    if style in ("CNOPED", "JPOPED", "FuriganaOPED"):
        return "OPED"
    if style.startswith("Furigana"):
        return None
    return style[:-2] if style.endswith(("CN", "JP")) and len(style) > 2 else None


def _norm(text):
    return kata_to_hira("".join(c for c in text if c.isalnum() or c == "ー"))


def _timed(text, start):
    """ASS karaoke chunks: (visible text, t0, t1, visible a, visible b).

    Consecutive timing tags include empty (also zero-duration) pauses. No
    clamping to the event end: old files contain real overrun/zero-time beats.
    """
    chunks, current, duration, at, pos = [], "", None, start, 0

    def finish():
        nonlocal current, at, pos
        if duration is not None:
            chunks.append((current, at, at + duration, pos, pos + len(current)))
            at += duration
        pos += len(current)
        current = ""

    for token in tokenize(text):
        if token.kind == "tags":
            for tag in token.tags:
                if tag.name.lower() in ("k", "kf", "ko"):
                    if not re.fullmatch(r"\d+", tag.arg):
                        raise ValueError("K 轴时长不是非负整数")
                    finish()
                    duration = int(tag.arg) * 10
        elif token.kind in ("text", "hardspace", "softbreak"):
            current += plain_text(token.raw)
    finish()
    return chunks


def _beats(chunks):
    return [(s.strip(), a, b) for s, a, b, _, _ in chunks]


def _explicit(original, text):
    readings, cursor = [], 0
    pattern = r"([^\s（）()]+)[（(]([^（）()]+)[）)]"
    for match in re.finditer(pattern, text):
        surface, kana = match.groups()
        kana = re.sub(r"[,，]\s*\d+(?:\.\d+)?\s*$", "", kana)
        at = original.find(surface, cursor)
        if at < 0:
            raise ValueError(f"振假名的文字找不到：{surface}")
        parts = align(surface, kana)
        if parts is None:
            raise ValueError(f"振假名和正文对不上：{surface}")
        readings.extend((at + a, at + b, reading) for a, b, reading in parts)
        cursor = at + len(surface)
    if not readings and any(is_kanji(c) for c in original):
        raise ValueError("振假名行没有可识别的读音")
    return readings


def _split_gap(original, a, b, kana, engine):
    """Split only when the engine agrees; peel agreeing ends for unusual lyrics."""
    parts = [(a + p.start, a + p.end, p.reading)
             for entry in engine.analyze(original[a:b]) for p in entry.parts]
    parts.sort()
    if len(parts) < 2 or parts[0][0] != a or parts[-1][1] != b:
        return [(a, b, kana)]
    if any(any(is_kanji(c) for c in original[x[1]:y[0]]) for x, y in zip(parts, parts[1:])):
        return [(a, b, kana)]
    left, right, prefix, suffix = 0, len(kana), [], []
    pending = list(parts)
    while len(pending) > 1:
        reading = pending[0][2]
        if _norm(kana[left:right]).startswith(_norm(reading)) and len(reading) < right - left:
            x, y, _ = pending.pop(0)
            prefix.append((x, y, kana[left:left + len(reading)]))
            left += len(reading)
            continue
        reading = pending[-1][2]
        if _norm(kana[left:right]).endswith(_norm(reading)) and len(reading) < right - left:
            x, y, _ = pending.pop()
            suffix.insert(0, (x, y, kana[right - len(reading):right]))
            right -= len(reading)
            continue
        break
    return prefix + [(pending[0][0], pending[-1][1], kana[left:right])] + suffix


def align_readings(original: str, kana: str, *, engine=None):
    """Kana/literal anchors, retaining visible offsets and explicit katakana."""
    positions = [i for i, c in enumerate(original) if c.isalnum() or c == "ー"]
    surface = "".join(original[i] for i in positions)
    reading = "".join(c for c in kana if c.isalnum() or c == "ー")
    aligned = align(surface, reading)
    if aligned is None:
        raise ValueError("原文和整行假名对不上")
    out = []
    for a, b, r in aligned:
        start, end = positions[a], positions[b - 1] + 1
        # Single kanji runs need no analyzer; split multi-word gaps conservatively.
        if end - start > 1:
            engine = engine or ReadingEngine()
            out.extend(_split_gap(original, start, end, r, engine))
        else:
            out.append((start, end, r))
    return out


def _merge_readings(preferred, fallback):
    return sorted(preferred + [r for r in fallback
                               if not any(r[0] < p[1] and p[0] < r[1] for p in preferred)])


def _ruby(original, chunks, ruby_sources, reject):
    readings, replacements = [], []
    # A ruby can span several separately tagged kanji, e.g. 二 + 人.
    for source in ruby_sources:
        try:
            timed = _timed(source.event.text, source.event.start * 10)
            indices = [i for i, c in enumerate(timed) if _norm(c[0])]
            sung = timed[indices[0]:indices[-1] + 1] if indices else []
            if not sung:
                continue                         # untimed rendered ruby is redundant
            first, last = sung[0][1], sung[-1][2]
            matches = []
            for i, chunk in enumerate(chunks):
                if chunk[1] != first or not any(is_kanji(c) for c in chunk[0]):
                    continue
                for j in range(i, len(chunks)):
                    if chunks[j][2] == last and any(is_kanji(c) for c in chunks[j][0]):
                        a, b = chunk[3], chunks[j][4]
                        raw_surface = original[a:b]
                        surface = raw_surface.strip()
                        a += len(raw_surface) - len(raw_surface.lstrip())
                        reading = "".join(c[0].strip() for c in sung)
                        parts = align(surface, reading)
                        suffix = ""
                        if parts is None:
                            # Old blocks include co-sung okurigana (合っ / あ).
                            suffix = re.search(r"[^\u3400-\u9fff々〆ヶ]+$", surface)
                            suffix = suffix.group() if suffix else ""
                            parts = align(surface, reading + suffix) if suffix else None
                        if parts is not None:
                            matches.append((i, j + 1, a, parts, suffix))
            if len(matches) != 1:
                raise ValueError("振假名的演唱时间不能唯一匹配日文汉字块")
            i, j, a, parts, suffix = matches[0]
            if any(i < old_j and old_i < j for old_i, old_j, _ in replacements):
                raise ValueError("同一汉字块有重复或重叠的振假名，保留第一行")
            readings.extend((a + x, a + y, reading) for x, y, reading in parts)
            if suffix:
                s, t0, t1, x, y = sung[-1]
                sung[-1] = (s + suffix, t0, t1, x, y)
            replacements.append((i, j, sung))
        except ValueError as error:
            reject(source, str(error))
    expanded, at = [], 0
    for i, j, sung in sorted(replacements):
        if i < at:
            continue
        expanded.extend(chunks[at:i])
        expanded.extend(sung)
        at = j
    expanded.extend(chunks[at:])
    return readings, expanded


def group_songs(lines: list[SongLine]) -> list[Song]:
    """Keep source order and delegate the gap/overlap boundary to song_span.

    A homogeneous legacy family is adapted to OP.JP; family changes delimit
    runs first, so arbitrary legacy names share the existing core song rule.
    """
    out = []
    for family, run in groupby(lines, key=lambda line: (line.source_file, line.family, line.song_header)):
        records = list(run)
        doc = Document("legacy-oped")
        for i, line in enumerate(records):
            gid, mid = str(i), "m" + str(i)
            doc.groups.append(Group(gid, "song", [mid]))
            doc.members[mid] = Member(mid, gid, "ja", "original", "song",
                                      line.start_ms, line.end_ms, style="OP.JP")
        at = 0
        while at < len(records):
            span = song_span(doc, str(at))
            out.append(Song(family[1], [records[int(gid)] for gid in span], records[at].category))
            at += len(span)
    return out


def _parse_legacy_text(text: str, source_file: str = "", *, engine=None) -> ImportResult:
    """Parse ASS documents or headerless ASS-event .txt/.song material."""
    result, sources, fmt = ImportResult(), [], list(DEFAULT_EVENT_FORMAT)

    def reject(source, reason):
        result.unrecognised.append(UnrecognisedLine(source_file, source.number, source.raw, reason))

    section = ""
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.lstrip("\ufeff").strip()
        if line.startswith("[") and line.endswith("]"):
            section = line.lower()
            continue
        if line.lower().startswith("format:") and section == "[events]":
            fmt = [s.strip() for s in line.split(":", 1)[1].split(",")]
            continue
        if not line.startswith(("Dialogue:", "Comment:")):
            if line and not section and not line.startswith(";"):
                reject(_Source(number, raw, None), "不是 ASS 歌词事件行")
            continue
        try:
            if len(line.split(":", 1)[1].split(",", len(fmt) - 1)) != len(fmt):
                raise ValueError("ASS 事件字段不完整")
            event = AssDocument.parse("[Events]\nFormat: " + ",".join(fmt) + "\n" + line).events[0]
            if event.end < event.start:
                raise ValueError("结束时间早于开始时间")
            sources.append(_Source(number, raw, event))
        except (ValueError, IndexError) as error:
            reject(_Source(number, raw, None), str(error))

    buckets, helpers = {}, {}
    for source in sources:
        e = source.event
        window = (e.start, e.end)
        family = _family(e.style)
        if e.style in ("K轴", "K轴测试", "振假名") or e.style.startswith("Furigana"):
            helpers.setdefault(window, []).append(source)
        elif family is not None and e.kind == "Dialogue":
            buckets.setdefault((*window, family), []).append(source)
        elif e.style in ("CNOPED", "JPOPED") and any(
                s.event.style == "K轴" and (s.event.start, s.event.end) == window for s in sources):
            continue                             # rendered backup of canonical K data
        else:
            reject(source, f"不是已知歌词样式：{e.style}")
    for window, extras in helpers.items():
        if any(s.event.style == "K轴" for s in extras) and not any(k[:2] == window for k in buckets):
            buckets[(*window, "OPED")] = []

    used_helpers = set()
    ordered = sorted(buckets.items(), key=lambda item: min(
        s.number for s in item[1] + helpers.get(item[0][:2], [])))
    for (start, end, family), bases in ordered:
        extras = helpers.get((start, end), [])
        peers = [k for k in buckets if k[:2] == (start, end)]
        if len(peers) > 1 and extras:
            for s in extras:
                if s.number not in used_helpers:
                    reject(s, "同一时间有多个歌词类别，辅助行无法唯一配对")
            used_helpers.update(s.number for s in extras)
            extras = []
        used_helpers.update(s.number for s in extras)
        cn = [s for s in bases if s.event.style == "CNOPED" or s.event.style.endswith("CN")]
        jp = [s for s in bases if s.event.style == "JPOPED" or s.event.style.endswith("JP")]
        chosen = (jp or cn or extras)[0]
        original = plain_text(jp[0].event.text) if jp else ""
        translation = plain_text(cn[0].event.text) if cn else ""
        for duplicate in jp[1:] + cn[1:]:
            reject(duplicate, "同一类别和时间有重复歌词行，保留第一行")
        beats, readings, mapping, kana, canonical = [], [], None, None, False
        rejected_k = set()
        for s in extras:
            if s.event.style != "K轴":
                continue
            try:
                origins = re.search(r"\{CNOrigin:(.*?),JPOrigin:(.*?),KanaOrigin:(.*?)\}", s.event.text)
                if not origins:
                    raise ValueError("K 轴行缺少 CNOrigin / JPOrigin / KanaOrigin")
                chunks = _timed(s.event.text.split(r"\N", 1)[0], start * 10)
                if not chunks:
                    raise ValueError("K 轴行没有可识别的拍")
                translation, original, kana = origins.groups()
                match = re.search(r"\{KaraOKMapping:(.*?)\}", s.event.text)
                mapping = match.group(1) if match else None
                beats, chosen, canonical = _beats(chunks), s, True
                break
            except ValueError as error:
                reject(s, str(error))
                rejected_k.add(s.number)
        if canonical:
            for s in extras:
                if s.event.style == "K轴" and s.number != chosen.number and s.number not in rejected_k:
                    reject(s, "同一类别和时间有重复 K 轴行，保留第一条有效行")
        for s in extras:
            if s.event.style == "振假名":
                try:
                    readings = _merge_readings(readings, _explicit(original, s.event.text))
                except ValueError as error:
                    reject(s, str(error))
        if jp and not canonical:
            for match in re.finditer(r"\{English:([^,{}]+),([^{}]+)\}", jp[0].event.text):
                surface, reading = match.groups()
                at = original.find(surface)
                if at >= 0:
                    readings = _merge_readings(readings, [(at, at + len(surface), reading)])
                else:
                    reject(jp[0], f"English 读音提示的文字找不到：{surface}")
        if not canonical:
            timing = next((s for s in extras if s.event.style == "K轴测试"), None)
            if timing:
                try:
                    beats = _beats(_timed(timing.event.text, start * 10))
                    kana = "".join(s for s, _, _ in beats)
                except ValueError as error:
                    reject(timing, str(error))
            elif jp:
                try:
                    chunks = _timed(jp[0].event.text, start * 10)
                    ruby = [s for s in extras if s.event.style.startswith("Furigana")]
                    derived, expanded = _ruby(original, chunks, ruby, reject)
                    readings = _merge_readings(readings, derived)
                    unresolved = [c for c in expanded if any(is_kanji(ch) for ch in c[0])]
                    if unresolved:
                        reject(jp[0], "日文 K 轴汉字块缺少同演唱时间的振假名：" +
                               " / ".join(c[0].strip() for c in unresolved))
                        expanded = [c for c in expanded if c not in unresolved]
                    beats = _beats(expanded)
                except ValueError as error:
                    reject(jp[0], str(error))
            elif cn:
                try:
                    beats = _beats(_timed(cn[0].event.text, start * 10))
                except ValueError as error:
                    reject(cn[0], str(error))
        uncovered = any(is_kanji(ch) and not any(a <= i < b for a, b, _ in readings)
                        for i, ch in enumerate(original))
        if kana is not None and original and uncovered:
            try:
                readings = _merge_readings(readings, align_readings(original, kana, engine=engine))
            except ValueError as error:
                reject(chosen, str(error))
        if not original and not translation:
            reject(chosen, "没有原声或翻译歌词文字")
            continue
        translation_beats = []
        if cn:
            try:
                translation_beats = _beats(_timed(cn[0].event.text, start * 10))
            except ValueError as error:
                reject(cn[0], str(error))
        result.lines.append(SongLine(start * 10, end * 10, family, original, translation,
                                     readings, beats, mapping, source_file, chosen.number,
                                     translation_beats=translation_beats))
    for extras in helpers.values():
        for s in extras:
            if s.number not in used_helpers:
                reject(s, "辅助歌词行找不到同时间的原声或翻译")
    result.unrecognised.sort(key=lambda line: line.line_number)
    result.songs = group_songs(result.lines)
    return result


LYRICS_PROMPT = '请帮我把一首日语歌整理成“歌词块”。我会给你三份资料：\n1. 日文歌词，每行一句；\n2. 中文歌词，每行一句，和日文一一对应；\n3. 用 Aegisub 打好的卡拉 OK 时间轴，只有假名，每行一句，是 ASS 字幕行（Dialogue 或 Comment 行，里面是 {\\kf数字}假名）。\n\n每一句输出一个歌词块，块与块之间空一行，格式：\n时间：开始 - 结束        （照抄第 3 份资料里这一行的开始和结束时间）\n原文：日文歌词\n读音：词（读音） 词（读音）……   （只写有汉字的词，送假名写在词里，读音用平假名；特殊读法按假名时间轴里实际唱的写）\n假名拍：{\\kf数字}假名{\\kf数字}假名……   （照抄第 3 份资料，不要改任何数字和假名）\n翻译：中文歌词            （照抄，不要改字）\n对应：中文一段（这一段唱的假名）中文一段（假名）……\n\n“对应”的规则：\n- 把中文按意思切成几段，每段后面的括号里写它对应的假名，必须是假名拍里连续的几拍，按顺序把这一句的假名全部用完；\n- 所有括号里的假名连起来，要和这一句假名拍里的假名完全一样（停顿不算）；\n- 中文不能改字、不能漏字；空格和标点跟在前一段后面；\n- 尽量让意思对得上：日文唱到“光”的时候，中文也在唱“陽光”。\n只有中文、没有日文的句子（例如英文歌词）只写“时间”和“翻译”。\n不确定的地方在那一块最后加一行“备注：……”。只输出歌词块，不要别的说明。\n\n第 1 份（日文歌词）：\n……\n第 2 份（中文歌词）：\n……\n第 3 份（假名时间轴）：\n……'


_FIELDS = ("时间", "原文", "读音", "假名拍", "翻译", "对应", "备注")
_CATEGORIES = {"片头曲": "OP", "片尾曲": "ED", "插曲": "IN"}


def _block_time(text):
    pattern = r"(\d+):([0-5]\d):([0-5]\d)\.(\d{1,3})"
    match = re.fullmatch(pattern + r"\s*-\s*" + pattern, text.strip())
    if match is None:
        raise ValueError(text)
    values = match.groups()
    def ms(v):
        h, m, s, fraction = v
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(fraction.ljust(3, "0"))
    start, end = ms(values[:4]), ms(values[4:])
    if end < start:
        raise ValueError(text)
    return start, end


def _add_member(doc, group, text, lane, lang, style, start, end, readings=()):
    from ..model.document import Annotation, Run
    from ..model.ids import new_id
    from .readings import anchor_of, visible_map

    m = Member(new_id("m_"), group.id, lang, lane, "song", start, end,
               style=style, runs=[Run(text, ins=new_id("i_"), b=len(text))])
    doc.members[m.id] = m
    group.members.append(m.id)
    vt, ids = visible_map(m)
    for a, b, reading in readings:
        if not 0 <= a < b <= len(ids) or not reading:
            raise ValueError("读音范围不在原文内")
        # Literal small/held kana following ruby are part of the sung syllable.
        from .karaoke import SMALL, HOLD
        while b < len(vt.text) and vt.text[b] in SMALL | HOLD:
            reading += vt.text[b]
            b += 1
        ann = Annotation(new_id("a_"), "ruby", m.id, anchor_of(ids[a:b]), vt.text[a:b],
                         reading, status="confirmed", origin="oped",
                         parts=[[0, b - a, reading]])
        doc.annotations[ann.id] = ann
    return m


def _timed_track(doc, m, timing, *, partial=False):
    """Match sung characters, retaining legacy boundaries as existing split/merge edits.

    Empty chunks carry elapsed time but have no character identity to store.
    A legacy chunk may span ruby and literal kana or split an English word.
    """
    from . import karaoke as kara

    parts = [p for b in kara.derive(doc, m) for p in b.parts]
    imported = [(s, a, b) for s, a, b in timing if _norm(s)]
    norm = lambda s: _norm(s).casefold()
    derived_text = "".join(norm(p.sung) for p in parts)
    imported_text = "".join(norm(s) for s, _, _ in imported)
    if derived_text != imported_text:
        if not partial:
            raise ValueError("假名拍和原文的读音对不上")
        from difflib import SequenceMatcher
        matched = {}
        for a, b, size in SequenceMatcher(None, derived_text, imported_text, autojunk=False).get_matching_blocks():
            matched.update((b + i, a + i) for i in range(size))
        spans, cursor = {}, 0
        for sung, start, end in imported:
            n = len(norm(sung))
            positions = [matched.get(i) for i in range(cursor, cursor + n)]
            if positions and positions[0] is not None and positions == list(range(positions[0], positions[0] + n)):
                spans[(positions[0], positions[-1] + 1)] = (start, end)
            cursor += n
        edges = {p for span in spans for p in span}
        edits, cursor = [], 0
        for previous, p in zip(parts, parts[1:]):
            cursor += len(norm(previous.sung))
            inside = any(a < cursor < b for a, b in spans)
            if cursor in edges or inside:
                edits.append({"op": "split" if cursor in edges else "merge", "at": p.at})
        beats = kara.derive(doc, m, edits=edits)
        cursor = 0
        for beat in beats:
            end = cursor + len(norm(beat.sung))
            if (cursor, end) in spans:
                beat.t0, beat.t1 = spans[(cursor, end)]
                beat.by = "memory"
            cursor = end
        return beats, edits
    boundaries, cursor = set(), 0
    for sung, _, _ in imported[:-1]:
        cursor += len(norm(sung))
        boundaries.add(cursor)
    edits, cursor = [], 0
    for previous, p in zip(parts, parts[1:]):
        cursor += len(norm(previous.sung))
        edits.append({"op": "split" if cursor in boundaries else "merge", "at": p.at})
    beats = kara.derive(doc, m, edits=edits)
    if [norm(b.sung) for b in beats] != [norm(s) for s, _, _ in imported]:
        raise ValueError("假名拍边界不能匹配原文的读音")
    for beat, (_, start, end) in zip(beats, imported):
        beat.t0, beat.t1, beat.by = start, end, "memory"
    return beats, edits


def _parse_block(fields, number, source_file, line_number, family, category, header, engine):
    from . import lyrics_map

    if not fields.get("时间"):
        raise ValueError(f"第 {number} 块没有“时间”")
    try:
        start, end = _block_time(fields["时间"])
    except ValueError:
        raise ValueError(f"第 {number} 块的时间写法不对：{fields['时间']}") from None
    original, translation = fields.get("原文", ""), fields.get("翻译", "")
    readings = []
    if fields.get("读音"):
        try:
            readings = _explicit(original, fields["读音"])
        except ValueError as error:
            if str(error).startswith("振假名的文字找不到："):
                word = str(error).split("：", 1)[1]
                raise ValueError(f"第 {number} 块“读音”里的「{word}」在原文里找不到") from None
            raise
    beats = _beats(_timed(fields.get("假名拍", ""), start))
    translated = _beats(_timed(translation, start))
    translation = plain_text(translation)
    if not original and not translation:
        raise ValueError("没有原声或翻译歌词文字")
    doc = Document("oped-block", groups=[Group("g", "song")])
    if original:
        m = _add_member(doc, doc.groups[0], original, "original", "ja", "OP.JP", start, end, readings)
        if beats:
            try:
                originals, _ = _timed_track(doc, m, beats)
            except ValueError:
                raise ValueError(f"第 {number} 块“假名拍”的假名和原文的读音对不上") from None
        else:
            from .karaoke import derive
            originals = derive(doc, m)
    else:
        originals = []
        if beats:
            raise ValueError(f"第 {number} 块“假名拍”的假名和原文的读音对不上")
    mapping = fields.get("对应") or None
    if mapping:
        lyrics_map.parse(mapping, originals, translation)
    return SongLine(start, end, family, original, translation, readings, beats, mapping,
                    source_file, line_number, translation_beats=translated,
                    notes=fields.get("备注", []), category=category, song_header=header)


def parse_text(text: str, source_file: str = "", *, engine=None) -> ImportResult:
    """Read either ASS material or validated lyric blocks, including mixed pasted events."""
    if not re.search(r"(?m)^\s*(?:#\s*(?:歌名|类别)|" + "|".join(_FIELDS) + r")\s*[:：]", text):
        return _parse_legacy_text(text, source_file, engine=engine)
    result, legacy, pending, metadata = ImportResult(), [], [], {}
    family, category, header, number = Path(source_file).stem or "OPED", "NA", 0, 0

    def flush():
        nonlocal number
        if not pending:
            return
        number += 1
        fields, invalid = {}, None
        for _, raw in pending:
            match = re.fullmatch(r"\s*([^:：]+)[:：](.*)", raw)
            if match is None or match[1].strip() not in _FIELDS:
                invalid = "不是已知歌词字段：" + raw
                break
            key, value = match[1].strip(), match[2].strip()
            if key == "备注":
                fields.setdefault(key, []).append(value)
            elif key in fields:
                invalid = "歌词字段重复：" + key
                break
            else:
                fields[key] = value
        try:
            if invalid:
                raise ValueError(invalid)
            result.lines.append(_parse_block(fields, number, source_file, pending[0][0],
                                            family, category, header, engine))
        except ValueError as error:
            result.unrecognised.append(UnrecognisedLine(source_file, pending[0][0],
                "\n".join(raw for _, raw in pending), str(error), number))
        pending.clear()

    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.lstrip("\ufeff").strip()
        legacy.append("")
        if not line:
            flush()
        elif line.startswith("#"):
            match = re.fullmatch(r"#\s*(歌名|类别)\s*[:：](.*)", line)
            if match:
                flush()
                if match[1] == "歌名":
                    family, category, header = match[2].strip(), "NA", header + 1
                else:
                    category = _CATEGORIES.get(match[2].strip(), "NA")
        elif line.startswith(("Dialogue:", "Comment:", "[", "Format:")):
            flush()
            legacy[-1] = raw
            metadata[line_number] = (family, category, header)
        else:
            pending.append((line_number, raw))
    flush()
    old = _parse_legacy_text("\n".join(legacy), source_file, engine=engine)
    for record in old.lines:
        fam, cat, head = metadata[record.line_number]
        if head:
            record.family, record.category, record.song_header = fam, cat, head
    result.lines.extend(old.lines)
    result.lines.sort(key=lambda record: record.line_number)
    result.unrecognised.extend(old.unrecognised)
    result.unrecognised.sort(key=lambda record: record.line_number)
    result.songs = group_songs(result.lines)
    return result


def build_passage(song: Song, name: str, category: str, version: str, *,
                  languages=None, styles=None, episode_id: str = "legacy-oped") -> dict:
    """Use inserted identities, confirmed readings, existing mapping edits and snapshot."""
    from ..languages import DEFAULT_LANGUAGES
    from ..model.ids import new_id
    from .merge import MergeSettings
    from .ops import Editor
    from .passages import snapshot

    languages = {**DEFAULT_LANGUAGES, **(languages or {})}
    styles = {**MergeSettings().styles, **(styles or {})}
    doc = Document(episode_id)
    editor = Editor(doc, {})
    mappings = {}
    for line in song.lines:
        group = Group(new_id("g_"), "song", state="confirmed")
        doc.groups.append(group)
        doc.structure_changed()
        for lane, text in (("original", line.original), ("translation", line.translation)):
            if not text:
                continue
            m = _add_member(doc, group, text, lane, languages[lane],
                            styles[f"{lane}/song"].format(fam=category), line.start_ms, line.end_ms,
                            line.readings if lane == "original" else ())
            timing = line.beats if lane == "original" else line.translation_beats
            if lane == "translation" and not line.original and not timing:
                timing = line.beats
            if timing:
                try:
                    beats, edits = _timed_track(doc, m, timing, partial=True)
                except ValueError as error:
                    raise ValueError(f"{line.source_file} 第 {line.line_number} 行 ({lane}): {error}") from error
                editor.set_karaoke(m.id, beats, edits=edits, effect={"preset": "kf", "params": {}})
        if line.mapping:
            mappings[group.id] = line.mapping
    doc.structure_changed()
    if mappings:
        editor.apply_lyrics_mapping(mappings)
    return snapshot(doc, [g.id for g in doc.groups], name=name, category=category,
                    version=version, episode_id=episode_id)


def decode_audio_features(path: Path | str, *, audio_track: int | None = 0, runner=None):
    """Decode the stored mpv audio id using PeakJob's pipeline, without Qt/caches."""
    import numpy as np
    from .. import proc
    from ..audio import RATE, compute_features, ffmpeg_audio_index

    runner = runner or proc.KillableRun()
    raw = runner.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                      "-map", f"0:a:{ffmpeg_audio_index(audio_track)}", "-ac", "1", "-ar", str(RATE),
                      "-f", "s16le", "-"], check=True).stdout
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return compute_features(samples, check_cancelled=runner.check_cancelled)


def build_audio_template(song: Song, media: Path | str | None = None, *,
                         audio_origin_ms: int | None = None, audio_track: int | None = 0, runner=None):
    """Align a sibling WAV via NPZ time metadata only, or use an episode media range.

    Missing sibling audio returns None. Explicit media defaults to episode time 0;
    callers can supply an origin for other cropped recordings. Old features never load.
    """
    import numpy as np
    from ..audio import HOP_MS
    from ..audio_match import template_from

    sibling = media is None
    if sibling:
        sources = {line.source_file for line in song.lines}
        if len(sources) != 1 or not next(iter(sources)):
            return None
        source = Path(next(iter(sources)))
        if source.suffix.lower() != ".song" or not source.with_suffix(".wav").is_file():
            return None
        media = source.with_suffix(".wav")
    media = Path(media)
    metadata_end = None
    if sibling and audio_origin_ms is None:
        npz = media.with_suffix(".npz")
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as saved:
                start, end = float(saved["start_sec"]), float(saved["end_sec"])
            if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
                raise ValueError("旧音频的起止时间不正确")
            audio_origin_ms, metadata_end = round(start * 1000), round(end * 1000)
        else:
            audio_origin_ms = song.start_ms
    origin = audio_origin_ms if audio_origin_ms is not None else 0
    features = decode_audio_features(media, audio_track=audio_track, runner=runner)
    duration = len(features.spectrogram) * HOP_MS
    if metadata_end is not None and abs(duration - (metadata_end - origin)) > HOP_MS:
        raise ValueError("旧音频时长和起止时间对不上")
    if sibling and metadata_end is None and audio_origin_ms == song.start_ms \
            and abs(duration - (song.end_ms - song.start_ms)) > HOP_MS:
        raise ValueError("旧音频缺少可核验的起止时间，请指定音频起点")
    if metadata_end is not None and song.end_ms > metadata_end:
        raise ValueError("歌词时间不在旧音频范围内")
    return template_from(features, song.start_ms - origin, song.end_ms - origin)


def store_passage(store, passage: dict, template=None, *, conflict: str = "skip") -> str | None:
    """Resolve the name/version pair, then delegate persistence to PassageStore.add."""
    import copy
    from ..model.ids import new_id

    if conflict not in ("replace", "new", "skip"):
        raise ValueError(conflict)
    entries = store.list()
    existing = next((p for p in entries if (p["name"], p["version"]) ==
                     (passage["name"], passage["version"])), None)
    if existing is not None:
        if conflict == "skip":
            return None
        if conflict == "replace":
            return store.add(passage, template, replace_id=existing["id"])
        passage = copy.deepcopy(passage)
        passage["id"] = new_id("psg_")
        number, base = 2, passage["version"]
        while any(p["name"] == passage["name"] and p["version"] == f"{base} ({number})" for p in entries):
            number += 1
        passage["version"] = f"{base} ({number})"
    return store.add(passage, template)


def parse(path: Path | str, *, engine=None) -> ImportResult:
    """Read the selected file only; never rewrite it or load sibling audio/NPZ."""
    path = Path(path)
    return parse_text(decode_bytes(path.read_bytes()).text, str(path), engine=engine)
