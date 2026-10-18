"""Headless season import and release export using the normal episode pipeline."""

from __future__ import annotations

import os
import re
from io import BytesIO
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .. import proc
from ..local_paths import local_value
from ..audio import AudioFeatures, load_features
from ..autotime import SNAP_LEAD_IN_MS, SNAP_LEAD_OUT_MS, _snap_activity, suggest_group_times
from .fontcheck import DEFAULT_FONT_LIBRARY, FontLibrary, font_runs, _family_key
from ..ass.document import AssDocument
from ..ass.text import plain_text
from ..align.fragments import is_interjection
from ..textutil import has_kanji, is_kana
from ..sources.cn import _is_positioned, _SONG_HINT, song_role, classify
from .merge import group_counts
from .ops import Editor
from .workspace import Episode, Project

Progress = Callable[[int, int, str, str], None]
_KEY = re.compile(r"^S\d{2}E\d{3}")


@dataclass(frozen=True)
class BatchOptions:
    mode: str = "5.1"
    require_video: bool = True
    adopt_suggested_times: bool = False
    hide_unpaired_interjections: bool = True
    lead_in_ms: int = SNAP_LEAD_IN_MS
    lead_out_ms: int = SNAP_LEAD_OUT_MS
    song_choices: dict[str, str] = field(default_factory=dict)
    assfonts: Optional[Path] = field(default_factory=lambda: Path(local_value("SUBKIT_ASSFONTS", "assfonts")))
    # 调轴: audio = like 时间轴 → 自动调轴（按视频音频） (whole episode / segments, both lanes together); original = dialogue on
    # the original file's times (5.1 only); translation = the merge's own timing (5.2: no adjustment).
    timing: str = "translation"

    def __post_init__(self) -> None:
        if self.mode not in ("5.1", "5.2"):
            raise ValueError(self.mode)
        if self.timing not in ("audio", "original", "translation") or (self.mode == "5.2" and self.timing == "original"):
            raise ValueError("无效的调轴方式")
        if any(value not in ("transcribe", "transcribe_vertical", "keep") for value in self.song_choices.values()):
            raise ValueError("无效的歌曲处理方式")


@dataclass
class EpisodeFiles:
    key: str
    video: Optional[Path]
    translation: Optional[Path]
    original: Optional[Path]
    status: str


@dataclass
class EpisodeResult:
    key: str
    status: str
    reason: str = ""
    output: Optional[Path] = None
    counts: dict[str, int] = field(default_factory=dict)
    unpaired: int = 0
    uncertain: int = 0
    effects: int = 0
    songs: dict[str, str] = field(default_factory=dict)
    hidden: int = 0
    hidden_interjections: int = 0
    warnings: list[str] = field(default_factory=list)
    adopted: int = 0
    timing: str = ""
    fonts_embedded: list[str] = field(default_factory=list)
    fonts_carried: list[str] = field(default_factory=list)
    fonts_carried_names: dict[str, list[str]] = field(default_factory=dict)
    fonts_missing: dict[str, list[str]] = field(default_factory=dict)
    fonts_recovered: list[str] = field(default_factory=list)
    assfonts_tail: str = ""


@dataclass
class BatchResult:
    episodes: list[EpisodeResult]
    report: Path
    project: Optional[Path]
    cancelled: bool = False


def scan_season(season: Path | str, options: BatchOptions = BatchOptions()) -> list[EpisodeFiles]:
    """Scan only direct children; candidate conflicts take precedence over missing files."""
    groups: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(Path(season).iterdir(), key=lambda p: p.name.casefold()):
        match = _KEY.match(path.name)
        if not path.is_file() or not match:
            continue
        ext = path.suffix.lower()
        lane = ("video" if ext in (".mkv", ".mp4") else
                "original" if options.mode == "5.1" and
                (ext == ".srt" or ext == ".ass" and "ja" in path.stem.casefold()) else
                "translation" if ext in (".ass", ".ssa") else None)
        if lane is not None:
            groups.setdefault(match[0], {k: [] for k in ("video", "translation", "original")})[lane].append(path)
    out = []
    for key, lanes in sorted(groups.items()):
        conflicts = [p.name for paths in lanes.values() if len(paths) > 1 for p in paths]
        status = ("有多个候选：" + "、".join(conflicts) if conflicts else
                  "缺视频" if options.require_video and not lanes["video"] else
                  "缺中文字幕" if not lanes["translation"] else
                  "缺日文字幕" if options.mode == "5.1" and not lanes["original"] else "就绪")
        one = lambda lane: lanes[lane][0] if len(lanes[lane]) == 1 else None
        out.append(EpisodeFiles(key, one("video"), one("translation"),
                                one("original") if options.mode == "5.1" else one("translation"), status))
    return out


def effect_styles(doc: AssDocument) -> list[str]:
    counts = Counter(ev.style for ev in doc.events if ev.is_dialogue)
    positioned = Counter(ev.style for ev in doc.events if ev.is_dialogue and _is_positioned(ev.text))
    return [style for style, count in counts.items() if positioned[style] * 2 >= count]


@dataclass(frozen=True)
class SongFamily:
    family: str
    styles: list[str]
    karaoke_lines: dict[str, int]
    fx_count: int
    fx_counts: dict[str, int]
    plain_lines: dict[str, int]
    has_drawing: bool
    own_furigana: bool
    vertical: bool

    @property
    def karaoke_count(self) -> int:
        return max((n for style, n in self.karaoke_lines.items() if not _furigana_style(style)), default=0)

    @property
    def line_count(self) -> int:
        return self.karaoke_count or max(self.plain_lines.values(), default=0)

    @property
    def fx_ratio(self) -> float:
        return max((self.fx_counts.get(style, 0) / n for style, n in self.karaoke_lines.items()
                    if n and not _furigana_style(style)), default=0.0)

    @property
    def default(self) -> str:
        if self.vertical:
            return "transcribe_vertical"
        return "keep" if (self.own_furigana or self.has_drawing or
                          self.fx_ratio > 50) else "transcribe"


def _furigana_style(style: str) -> bool:
    return any(hint in style.casefold() for hint in ("furi", "ruby", "假名"))


def song_families(doc: AssDocument) -> list[SongFamily]:
    """Read kara-templater originals and generated effects without clustering events."""
    styles = list(dict.fromkeys(ev.style for ev in doc.events
                  if _SONG_HINT.match(ev.style) or ev.effect.casefold() == "karaoke"))
    families: dict[str, list[str]] = {}
    for style in styles:
        prefix = re.match(r"^(OP|ED|IN)(\d*)", style, re.I)
        family = (prefix[0].upper() if prefix else "IN")
        families.setdefault(family, []).append(style)
    out = []
    style_map = doc.style_map()
    for family, names in families.items():
        events = [ev for ev in doc.events if ev.style in names]
        karaoke = Counter(ev.style for ev in events if ev.effect.casefold() == "karaoke")
        fx = [ev for ev in events if ev.effect.casefold() == "fx"]
        vertical = any(style_map[name].get("Fontname").startswith("@") and
                       style_map[name].num("Angle") % 360 in (90, 270)
                       for name in names if name in style_map and not _furigana_style(name))
        fx_counts = Counter(ev.style for ev in fx)
        plain = Counter(ev.style for ev in events if ev.is_dialogue and not _furigana_style(ev.style))
        out.append(SongFamily(family, names, {name: karaoke[name] for name in names}, len(fx),
                              dict(fx_counts), dict(plain),
                              any(re.search(r"\\p[1-9]\d*", ev.text) for ev in fx),
                              any(_furigana_style(name) for name in names), vertical))
    return out


def source_roles(doc: AssDocument, mode: str, *, strict: bool = True,
                 song_choices: Optional[dict[str, str]] = None) -> tuple[dict, list[str]]:
    """Select dialogue lanes and explicit per-event song roles; retain the source ledger."""
    families = song_families(doc)
    song_styles = {style for family in families for style in family.styles}
    choices = song_choices or {}
    kept = {style for family in families if choices.get(family.family, family.default) == "keep"
            for style in family.styles}
    effects = [style for style in effect_styles(doc) if style not in song_styles or style in kept]
    effects.extend(style for style in kept if style not in effects)
    candidates = [ev for ev in doc.events if ev.is_dialogue and ev.style not in song_styles
                  and ev.style not in effects]
    counts = Counter(ev.style for ev in candidates)
    jp = []
    if mode == "5.2":
        jp = [style for style in counts if "jp" in style.casefold() or "ja" in style.casefold()]
        if not jp:
            for style in counts:
                letters = [c for ev in candidates if ev.style == style for c in plain_text(ev.text) if c.isalpha()]
                if sum("ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" for c in letters) / max(1, len(letters)) > .5:
                    jp.append(style)
        if not jp and strict:
            raise ValueError("没有识别到日文对白样式")
    cn_doc = doc.copy()
    cn_doc.events = [ev for ev in candidates if ev.style not in jp]
    detected = classify(cn_doc, {})
    dialogue = [style for style, role in detected.style_roles.items() if role == "dialogue"]
    if not dialogue and strict:
        raise ValueError("没有识别到中文对白样式")
    roles, reasons = {}, {}
    for family in families:
        keep = choices.get(family.family, family.default) == "keep"
        for i, ev in enumerate(doc.events):
            if ev.style not in family.styles:
                continue
            if keep:
                roles[i] = "keep"
            elif _furigana_style(ev.style) or family.karaoke_count and ev.effect.casefold() == "fx":
                roles[i], reasons[i] = "drop", "song_transcribe"
            elif (ev.effect.casefold() == "karaoke" or not family.karaoke_count and ev.is_dialogue):
                role = song_role(ev.style, ev.text)
                if mode == "5.1" and role == "song_jp":
                    roles[i], reasons[i] = "drop", "song_original_lane"
                else:
                    roles[i] = role
    return {"dialogue": dialogue, "keep": effects, "drop": jp,
            "event_roles": roles, "excluded_reasons": reasons}, jp


def unused_path(path: Path) -> Path:
    candidate, index = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        index += 1
    return candidate


def _hide_effect_overlap(episode, intervals: list[tuple[int, int]]) -> int:
    editor = Editor(episode.document, episode.sources)
    hidden = 0
    covered = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if covered and start <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(end, covered[-1][1]))
        else:
            covered.append((start, end))
    for member in list(episode.document.members.values()):
        if member.lane != "original" or not member.text.strip():
            continue
        group = episode.document.group(member.group)
        if any(m.lane == "translation" and m.source_kind != "Comment" and
               plain_text(m.text).strip() for m in episode.document.members_of(group)):
            continue
        # Only song lyrics duplicate the kept effects. Real dialogue whose translation sits in a
        # positioned (kept) style is unpaired too and must stay visible (Nonbiri narration, Berserk うわあ).
        events = {(run.src, run.e) for run in member.runs if run.src in episode.sources and run.e is not None}
        if not any(ch in "♪♫♬" for src, e in events for ch in episode.sources[src].event(e).text):
            continue
        duration = member.end - member.start
        overlap = sum(max(0, min(member.end, end) - max(member.start, start)) for start, end in covered)
        if duration <= 0 or overlap * 2 < duration:
            continue
        length = sum(len(r.text) for r in member.runs if r.kind != "tag")
        cid = editor.hide_range(member.id, 0, length)
        if cid:
            hidden += 1
            # Hidden text has no displayed anchor; its automatic readings must not render.
            for aid in [a.id for a in episode.document.annotations.values() if a.member == member.id]:
                del episode.document.annotations[aid]
    return hidden


def _is_interjection_line(text: str) -> bool:
    """Only cries throughout the line; short repeated kana also count."""
    text = plain_text(text)
    if not text.strip() or has_kanji(text):
        return False
    separators = r'\s、。，,.!！?？…‥・「」『』（）()：:；;'
    tokens = re.findall(f"([^{separators}]+)([{separators}]*)", text)
    pieces = [piece for piece, _ in tokens]
    if not pieces:
        return False
    # The shared detector uses the ending punctuation as evidence for short katakana cries.
    if is_interjection(text) or all(is_interjection(piece + ending.strip(" \t\r\n　「」『』（）()"))
                                    for piece, ending in tokens):
        return True
    return (len(pieces) >= 2 and len(set(pieces)) == 1 and
            1 <= len(pieces[0]) <= 3 and all(is_kana(ch) for ch in pieces[0]))


def _hide_unpaired_interjections(episode) -> int:
    editor = Editor(episode.document, episode.sources)
    hidden = 0
    for group in episode.document.groups:
        if group.kind != "dialogue":
            continue
        members = [m for m in episode.document.members_of(group)
                   if m.source_kind != "Comment" and plain_text(m.text).strip()]
        if not members or any(m.lane != "original" or m.lang != "ja" or
                              m.role not in ("dialogue", "top", "alt") for m in members):
            continue
        if not _is_interjection_line(" ".join(m.text for m in members)):
            continue
        for member in members:
            length = sum(len(r.text) for r in member.runs if r.kind != "tag")
            if editor.hide_range(member.id, 0, length):
                hidden += 1
                for aid in [a.id for a in episode.document.annotations.values() if a.member == member.id]:
                    del episode.document.annotations[aid]
    return hidden


def adopt_suggestions(editor: Editor, features: AudioFeatures, options: BatchOptions) -> int:
    """Walk current neighbours in chronological order, as consecutive C-key presses."""
    doc = editor.doc
    groups = sorted((g for g in doc.groups if "hidden" not in g.flags and g.kind in ("dialogue", "song") and g.members),
                    key=doc.group_time)
    activity = features.snap_activity if features.snap_activity is not None else _snap_activity(features)
    changed = 0

    def apply(tx):
        nonlocal changed
        for i, group in enumerate(groups):
            members = list(doc.members_of(group))
            if group.kind != "dialogue" or group.state == "confirmed" or any(m.extra.get("time_manual") for m in members):
                continue
            if not any(m.source_kind != "Comment" and plain_text(m.text).strip() for m in members):
                continue
            start, end = doc.group_time(group)
            suggestion = suggest_group_times(activity, start, end,
                lead_in_ms=options.lead_in_ms, lead_out_ms=options.lead_out_ms,
                previous=doc.group_time(groups[i - 1]) if i else None,
                following=doc.group_time(groups[i + 1]) if i + 1 < len(groups) else None)
            if suggestion is None:
                continue
            a, b = suggestion
            if (start if a is None else a, end if b is None else b) == (start, end):
                continue
            editor.set_group_time(group.id, start=a, end=b, source="adopted", transaction=tx)
            changed += 1
    editor.history.run("全季自动采用建议时间", apply)
    return changed


def _font_entries(doc: AssDocument) -> list[tuple[str, list[str]]]:
    entries = []
    current = None
    for line in doc.raw_section("[Fonts]") or []:
        if line.lower().startswith("fontname:"):
            current = [line]
            entries.append((line.split(":", 1)[1].strip(), current))
        elif current is not None:
            current.append(line)
    return entries


def _ass_decode(lines: list[str]) -> bytes:
    """Decode ASS attachments, including the unpadded final two/three characters."""
    encoded = "".join(line.strip() for line in lines)
    if len(encoded) % 4 == 1:
        raise ValueError("ASS 字体附件尾组不完整")
    data = bytearray()
    bits = value = 0
    for char in encoded:
        digit = ord(char) - 33
        if not 0 <= digit <= 63:
            raise ValueError("ASS 字体附件编码无效")
        value = (value << 6) | digit
        bits += 6
        if bits >= 8:
            bits -= 8
            data.append((value >> bits) & 255)
            value &= (1 << bits) - 1
    return bytes(data)


def _ass_encode(data: bytes) -> list[str]:
    """ASS uses six-bit digits offset by 33, wrapped at 80 characters."""
    encoded = []
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        value = int.from_bytes(chunk.ljust(3, b"\0"), "big")
        encoded.extend(chr(((value >> shift) & 63) + 33)
                       for shift in (18, 12, 6, 0)[:len(chunk) + 1])
    text = "".join(encoded)
    return [text[i:i + 80] for i in range(0, len(text), 80)]


def _attachment_names(lines: list[str]) -> set[str]:
    """Read every language's family/full name from every sfnt/TTC face."""
    from fontTools.ttLib import TTCollection, TTFont

    names = set()
    with BytesIO(_ass_decode(lines[1:])) as stream:
        if stream.getvalue()[:4] == b"ttcf":
            font = TTCollection(stream)
            faces = font.fonts
        else:
            font = TTFont(stream)
            faces = [font]
        try:
            for face in faces:
                for record in face["name"].names:
                    if record.nameID in (1, 4, 16):
                        names.add(record.toUnicode())
        finally:
            font.close()
    return names


def _attachment_families(lines: list[str]) -> set[str]:
    try:
        return {_family_key(name) for name in _attachment_names(lines)}
    except Exception:
        return set()


def _attachment_cmaps(lines: list[str]) -> dict[str, set[int]]:
    """Map aliases to Unicode coverage, keeping TTC faces' families separate."""
    from fontTools.ttLib import TTCollection, TTFont

    coverage: dict[str, set[int]] = {}
    with BytesIO(_ass_decode(lines[1:])) as stream:
        font = TTCollection(stream) if stream.getvalue()[:4] == b"ttcf" else TTFont(stream)
        try:
            for face in font.fonts if isinstance(font, TTCollection) else [font]:
                chars = {cp for table in face["cmap"].tables if table.isUnicode()
                         for cp, glyph in table.cmap.items() if glyph != ".notdef"}
                for record in face["name"].names:
                    if record.nameID in (1, 4, 16):
                        coverage.setdefault(_family_key(record.toUnicode()), set()).update(chars)
        finally:
            font.close()
    return coverage


def _carried_gaps(doc: AssDocument, coverage: dict[str, set[int]]) -> list[str]:
    """Report unique missing characters in drawing order, including ruby events."""
    used: dict[str, dict[str, None]] = {}
    names = {}
    for event in doc.dialogues():
        # Hard spaces request no visible glyph; font_runs already skips blocks/drawings/breaks.
        for family, text in font_runs(event.copy(text=event.text.replace(r"\h", "")), doc):
            key = _family_key(family)
            if key in coverage:
                names.setdefault(key, family)
                used.setdefault(key, {}).update(dict.fromkeys(text))
    return [f"{names[key]}：字形不全：{len(missing)} 个字（{''.join(missing[:20])}）"
            for key, chars in used.items()
            if (missing := [ch for ch in chars if ord(ch) not in coverage[key]])]


def _subset_document(doc: AssDocument, excluded: set[str]) -> AssDocument:
    """Filter only the disposable subset input; mixed-font lines retain all glyphs."""
    copy = doc.copy()
    copy.drop_section("[Fonts]")
    copy.styles = [s for s in copy.styles if _family_key(s.fontname) not in excluded]
    copy.events = []
    for event in doc.dialogues():
        families = [name for name, text in font_runs(event, doc)
                    if text and name and _family_key(name) not in excluded]
        if not families:
            continue
        style = doc.style(event.style) or doc.style("Default")
        if style is not None and copy.style(style.name) is None:
            replacement = style.copy()
            replacement.set("Fontname", families[0])
            copy.upsert_style(replacement)

        def strip_block(match):
            block = re.sub(r"\\fn([^\\}()]*)", lambda m: "" if
                           _family_key(m[1]) in excluded else m[0], match[0])
            return re.sub(r"\\r([^\\}()]*)", lambda m: "\\r" if
                          doc.style(m[1]) is not None and
                          _family_key(doc.style(m[1]).fontname) in excluded else m[0], block)

        copy.events.append(event.copy(text=re.sub(r"\{[^}]*\}", strip_block, event.text)))
    return copy


def embed_fonts(path: Path, sources: list[AssDocument], executable: Path, result: EpisodeResult) -> None:
    """Subset all output fonts first, then carry unmatched source subsets with gap reports."""
    executable = Path(executable)
    database = executable.parent if (executable.parent / "fonts.json").is_file() else DEFAULT_FONT_LIBRARY.parent
    library = FontLibrary(database / "fonts.json")
    plain = AssDocument.load(path)
    logs = []
    missing = {}
    final_temp = None
    try:
        entries, produced = [], set()
        carried, carried_names = [], {}
        # libass uses the last definition; keep the exported styles untouched.
        usage = plain.copy()
        usage.styles = list(plain.style_map().values())
        requested = {s.fontname for s in usage.styles}
        requested.update(n for ev in usage.dialogues() for n, _ in font_runs(ev, usage) if n)
        root = Path(tempfile.mkdtemp(prefix="assfonts_batch_", dir=os.environ.get("TEMP")))
        staged = root / path.name
        embedded = []
        retry_limit = len({_family_key(n) for n in requested}) + 1
        for attempt in range(retry_limit):
            subset = _subset_document(usage, set(missing))
            if not subset.events:
                break
            subset.save(staged)
            dest = root / f"output-{attempt}"
            dest.mkdir()
            run = proc.run([str(executable), "-i", str(staged), "-o", str(dest), "-d", str(database), "-s"],
                           capture_output=True, timeout=120)
            log = (run.stdout + run.stderr).decode("utf-8", errors="replace")
            logs.append(log)
            absent = list(dict.fromkeys(re.findall(r'(?:Missing|Cannot find|not found).*?"([^"\n]+)"', log, re.I)))
            # A library file that cannot be read (e.g. an unmapped drive) aborts like a missing font;
            # assfonts names only the path there, so map it back through its own "Found font" lines.
            found = re.findall(r'Found font: "([^"\n]+)"[^\n]*?--> "([^"\n]+)"', log)
            for line in (l for l in log.splitlines() if "inaccessible" in l.lower()):
                for quoted in re.findall(r'"([^"\n]+)"', line):
                    absent.extend(name for name, file in found if file.casefold() == quoted.casefold())
            new = [name for name in dict.fromkeys(absent) if _family_key(name) not in missing]
            for name in absent:
                missing[_family_key(name)] = name
            if absent or "Found missing fonts" in log or "inaccessible" in log.lower():
                if attempt + 1 < retry_limit and new:
                    continue
                raise ValueError("assfonts 排除缺失或无法读取的字体后仍未完成")
            if run.returncode or "[ERROR]" in log:
                raise ValueError(f"assfonts 退出码 {run.returncode}：{log[-240:]}")
            files = sorted(p for p in dest.rglob("*") if p.suffix.lower() in (".ttf", ".otf", ".ttc"))
            if not files:
                raise ValueError(f"assfonts 退出码 {run.returncode}，没有生成子集字体")
            for font in files:
                data = font.read_bytes()
                lines = [f"fontname: {font.name}", *_ass_encode(data)]
                names = sorted(_attachment_names(lines), key=str.casefold)
                if not names:
                    raise ValueError(f"子集字体没有名称：{font.name}")
                entries.append((font.name, lines))
                produced.update(_family_key(n) for n in names)
                embedded.append(f"{font.name}（{'、'.join(names)}）")
            break
        seen, covered, coverage = set(), set(), {}
        for source in [plain, *sources]:
            for name, lines in _font_entries(source):
                content = tuple(line for line in lines[1:] if line.strip())
                key = (name, content)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    names = sorted(_attachment_names(lines), key=str.casefold)
                except Exception as exc:
                    names = []
                    result.warnings.append(f"原内嵌字体名称读取失败：{name}：{exc}")
                families = {_family_key(n) for n in names}
                if families & produced:
                    continue
                entries.append((name, lines))
                covered.update(families)
                if source is not plain:
                    carried.append(name)
                    carried_names[name] = names
                try:
                    for family, chars in _attachment_cmaps(lines).items():
                        coverage.setdefault(family, set()).update(chars)
                except Exception as exc:
                    result.warnings.append(f"原内嵌字体字形读取失败：{name}：{exc}")
        recovered = sorted((n for n in requested if _family_key(n) in covered), key=str.casefold)
        gaps = _carried_gaps(usage, coverage)
        result.fonts_missing = {name: library.lookup(name) for name in missing.values()}
        if entries:
            output = plain.copy()
            output.drop_section("[Fonts]")
            text = output.serialize()
            pos = text.lower().index("[events]")
            body = "\n".join(line for _name, lines in entries for line in lines)
            output = AssDocument.parse(text[:pos] + "[Fonts]\n" + body + "\n\n" + text[pos:])
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                final_temp = Path(stream.name)
            output.save(final_temp, newline=plain.newline)
            final_temp.replace(path)
        result.fonts_embedded = embedded
        result.fonts_carried = carried
        result.fonts_carried_names = carried_names
        result.fonts_recovered = recovered
        result.warnings.extend(gaps)
    except Exception as exc:
        result.warnings.append(f"assfonts 未完成，保留普通导出：{exc}")
        result.fonts_missing = {name: library.lookup(name) for name in missing.values()}
        logs.append(str(exc))
    finally:
        result.assfonts_tail = "\n".join("\n".join(logs).splitlines()[-40:])
        if final_temp is not None:
            final_temp.unlink(missing_ok=True)


def _timing_by_audio(episode: Episode, features: AudioFeatures) -> str:
    """「按视频的声音」: 时间轴 → 自动调轴（按视频音频） with its defaults (groups already within a frame, and segments
    without a reliable match, stay)."""
    from .autotime_doc import Settings, plan

    preview = plan(episode.document, episode.sources, features, Settings())
    groups, lines = Editor(episode.document, episode.sources).apply_autotime(preview)
    if not groups:
        return "按视频的声音：已经对得上或没有可信的偏差，没改"
    return f"按视频的声音：挪了 {groups} 组、{lines} 行（最多差 {preview.hint.max_offset_ms / 1000:.2f} 秒）"


def run_season(season: Path | str, target: Project | Path | str, options: BatchOptions = BatchOptions(), *,
               progress: Progress = lambda i, n, key, step: None,
               cancel_check: Callable[[], bool] = lambda: False,
               feature_provider: Callable[[Path, Optional[int], Path], AudioFeatures] = load_features) -> BatchResult:
    """Cancel at step boundaries; retain completed outputs and report every scanned episode.

    ``require_video=False`` supports subtitle-only validation. It is not the GUI default.
    Existing episode keys are skipped without opening or refreshing their files.
    """
    # Always load an independent instance, including when called with the window's Project.
    project = Project.open(target.root if isinstance(target, Project) else Path(target))
    project_root = project.root
    season = Path(season)
    files = scan_season(season, options)
    out_dir = season / "DuoSub"
    out_dir.mkdir(exist_ok=True)
    existing_keys = set(project.episode_ids())
    results = []
    cancelled = False
    for i, group in enumerate(files, 1):
        result = EpisodeResult(group.key, "跳过", group.status)
        results.append(result)
        if group.key in existing_keys:
            result.reason = "工程里已有这一集，未改动"
            continue
        if group.status != "就绪":
            continue
        if cancelled or cancel_check():
            cancelled = True
            result.reason = "已取消"
            continue

        def step(text: str) -> bool:
            nonlocal cancelled
            if cancel_check():
                cancelled = True
                result.reason = "已取消"
                return False
            progress(i, len(files), group.key, text)
            return True

        try:
            if not step("读取字幕"):
                continue
            source = AssDocument.load(group.translation)
            cfg, jp = source_roles(source, options.mode, song_choices=options.song_choices)
            families = song_families(source)
            result.songs = {family.family: options.song_choices.get(family.family, family.default) for family in families}
            transcribed = {style for family in families if result.songs[family.family] in ("transcribe", "transcribe_vertical")
                           for style in family.styles}
            song_intervals = [(ev.start * 10, ev.end * 10) for e, ev in enumerate(source.events)
                              if ev.style in transcribed and cfg["event_roles"].get(e) == "song_cn"]
            episode = project.add_episode(group.key, season=group.key[:3], number=int(group.key[4:]))
            if group.video is not None:
                episode.set_video(group.video)
            episode.add_source(group.translation, lang=project.languages["translation"], lane="translation", cn_cfg=cfg)
            episode.add_source(group.original, lang="ja", lane="original",
                               ass_styles=jp if options.mode == "5.2" else None,
                               song_intervals=song_intervals if options.mode == "5.1" else None)
            if not step("合并"):
                continue
            merge_settings = ({**episode.settings()["merge"], "pairing": "same_time"}
                              if options.mode == "5.2" else None)
            episode.merge(settings=merge_settings, persist=True, progress=lambda text: progress(i, len(files), group.key, "读音")
                          if text == "读音" else None)
            vertical_styles = {style for family in families
                               if result.songs[family.family] == "transcribe_vertical" and family.vertical
                               for style in family.styles}
            for member in episode.document.members.values():
                if member.role != "song":
                    continue
                event = next((episode.sources[r.src].event(r.e) for r in member.runs
                              if r.src in episode.sources and r.e is not None), None)
                if event is not None and event.style in vertical_styles:
                    member.style = event.style
                    member.margins = list(event.margins or [0, 0, 0])
                    member.extra["source_style"] = True
            intervals = [(ev.start * 10, ev.end * 10) for ev in source.events
                         if ev.is_dialogue and ev.style in cfg["keep"]]
            result.effects = len(intervals)
            # 5.2 takes its JP lines from the bilingual file itself, where they were shown.
            kept_song_styles = {style for family in families if result.songs[family.family] == "keep"
                                for style in family.styles}
            song_overlap = [(ev.start * 10, ev.end * 10) for ev in source.events
                            if ev.is_dialogue and ev.style in kept_song_styles]
            result.hidden = _hide_effect_overlap(episode, song_overlap) if options.mode == "5.1" else 0
            if options.hide_unpaired_interjections:
                result.hidden_interjections = _hide_unpaired_interjections(episode)
            features = None

            def audio():
                nonlocal features
                if features is None:
                    features = feature_provider(episode.video, (episode.data.get("media") or {}).get("audio_track"),
                                                project.cache_dir / "waveform")
                return features

            if options.timing == "audio":
                if episode.video is None:
                    result.timing = "按视频的声音：没有视频，没调"
                else:
                    if not step("读取音频"):
                        continue
                    audio()
                    if not step("按视频的声音调轴"):
                        continue
                    result.timing = _timing_by_audio(episode, features)
            elif options.timing == "original":
                if not step("按原声字幕的时间调轴"):
                    continue
                report = Editor(episode.document, episode.sources).unify_lane_times("original")
                result.timing = (f"按原声字幕的时间：{report['groups']} 组、{report['lines']} 行"
                                 + (f"（{report['shifted']} 行没有配对，按附近的时间差挪）" if report["shifted"] else ""))
            else:
                result.timing = "用翻译字幕的时间（没另外调）" if options.mode == "5.1" else "不调轴"
            if options.adopt_suggested_times:
                if not step("读取音频"):
                    continue
                if episode.video is None:
                    raise ValueError("采用建议时间需要视频音频")
                audio()
                if not step("采用建议时间"):
                    continue
                result.adopted = adopt_suggestions(Editor(episode.document, episode.sources), features, options)
            episode.save()
            result.counts = group_counts(episode.document)
            result.unpaired = sum(1 for m in episode.document.members.values()
                                  if m.text.strip() and m.role in ("dialogue", "top", "alt") and
                                  len({n.lane for n in episode.document.members_of(episode.document.group(m.group))
                                       if n.text.strip()}) == 1)
            if not step("导出"):
                continue
            ass, report = episode.build_export(mode="release")
            result.uncertain = report.uncertain
            result.warnings = list(report.warnings)
            result.warnings.append(f"隐藏的组：{report.hidden_groups} 组（没有导出）")
            stem = group.video.stem if group.video else group.key
            path = unused_path(out_dir / f"{stem}.zh-ja.ass")
            # Reserve the name atomically, including concurrent runs.
            while True:
                try:
                    with path.open("xb"):
                        pass
                    break
                except FileExistsError:
                    path = unused_path(out_dir / f"{stem}.zh-ja.ass")
            try:
                episode.write_export(ass, path)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            result.output, result.status, result.reason = path, "完成", ""
            if options.assfonts is not None:
                if not step("嵌入字体"):
                    continue
                sources = [source]
                if group.original != group.translation and group.original.suffix.lower() in (".ass", ".ssa"):
                    sources.append(AssDocument.load(group.original))
                embed_fonts(path, sources, options.assfonts, result)
        except Exception as exc:
            result.status, result.reason = "失败", str(exc)
    report_path = unused_path(out_dir / f"批量处理报告-{datetime.now():%Y%m%d-%H%M%S-%f}.txt")
    lines = [f"批量处理一季：{season.name}", f"模式：{options.mode}；工程：{project_root}；振假名大小按工程模板 JP.Ruby"]
    for result in results:
        lines.append(f"{result.key}：{result.status}" + (f"（{result.reason}）" if result.reason else ""))
        lines.append(f"  未配对行数：{result.unpaired}；待确认读音数：{result.uncertain}；"
                     f"特效原样保留：{result.effects}；隐藏日文行：{result.hidden}；采用建议时间：{result.adopted} 组")
        lines.append(f"  隐藏没有中文的叫喊和语气词：{result.hidden_interjections} 行")
        if result.timing:
            lines.append(f"  调轴：{result.timing}")
        for family, choice in result.songs.items():
            lines.append(f"  歌曲 {family}：" + ("原样保留" if choice == "keep" else
                                                   "转写为竖排并加振假名（右侧）" if choice == "transcribe_vertical" else
                                                   "转写并加振假名"))
        if result.output:
            lines.append(f"  输出：{result.output}")
        lines.append("  字体嵌入：" + ("、".join(result.fonts_embedded) or "无"))
        lines.append("  原内嵌字体带入：" + ("、".join(
            n + "（" + ("、".join(result.fonts_carried_names.get(n, [])) or "名称未解析") + "）"
            for n in result.fonts_carried) or "无"))
        lines.append("  仍缺字体：" + ("、".join(n for n in result.fonts_missing if n not in result.fonts_recovered) or "无"))
        for name, paths in result.fonts_missing.items():
            lines.append(f"  缺字体：{name}；字体库：" + ("、".join(paths) if paths else "字体库里也没有") +
                         ("；由原内嵌字体补回" if name in result.fonts_recovered else "；仍缺失" ))
        if result.assfonts_tail:
            lines.append("  assfonts 输出末尾：\n" + result.assfonts_tail)
        lines.extend(f"  {warning}" for warning in result.warnings)
    if options.assfonts is None:
        lines.append("assfonts 嵌入：已跳过")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BatchResult(results, report_path, project_root, cancelled)
