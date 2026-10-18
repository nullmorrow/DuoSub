"""The work file (work.ass): prepare (build from sources) and refresh (apply edits).

Conventions inside work.ass
---------------------------
* Every bilingual row has an id ``#12`` in the **Effect** field of its CN line,
  JP line, reading record and notes.  Generated lines carry ``#12:ruby`` etc.
  Rows are linked by id — never by time — so retiming in Aegisub is safe.
* ``@读音`` Comment = the reading record of the row's JP line (edit freely).
* ``@备注`` Comment = notes.  ``※…`` notes come from prepare (alignment doubts,
  omitted interjections) and stay until you delete them; ``⚙…`` notes are
  rewritten on every refresh (dictionary conflicts, ruby that still overlaps).
* The JP line's **Name/Actor** field holds the speaker (from the captions);
  speaker-specific dictionary rules (私→わたくし) use it.
* Lines without an id (signs, staff, the source's comments) are never touched.
* Generated lines (``#n:ruby`` …) are deleted and rebuilt on every refresh —
  do not edit them; edit the record instead.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .align.aligner import AlignRow, Aligner, Weights, estimate_offset, make_sim_fn, piecewise_offsets, prefetch_spans, realign_detached_blocks
from .align.embed import Embedder, resolve_backend
from .align.fragments import build_units
from .ass.document import AssDocument, Event, Style
from .ass.resample import resample
from .ass.text import VisibleText, plain_text
from .config import expand_path
from .project import Episode
from .reading.characters import CharacterTable
from .reading.dictionary import ReadingDict
from .reading.engine import Entry, Part, ReadingEngine, make_entry
from .reading.records import RecEntry, format_entry, parse_record
from .ruby.generate import RubyGenerator, RubyJob, RubyPiece, derive_ruby_style
from .ruby.measure import Measurer
from .sources.cn import clean_cn_text, load_cn, normalize_quotes
from .sources.jp import JpOptions, RubyHint, load_jp, speaker_reading_hints
from .textutil import has_kanji
from .gloss import parse_glosses
from .karaoke import BUILTIN_FX, align as kalign, cn_karaoke, fx_slot, karaoke_text, parse_cn_mapping, parse_k, ruby_karaoke, template_record
from .ass.tags import KARAOKE_TAGS
from .ass.text import remove_tags_named
from .pinyin import display as py_display, format_pinyin_record, line_pinyin, parse_pinyin_record
from .songs import SongMemory, rescale_k

ROW_EFFECT = re.compile(r"^#(\d+)(?::([A-Za-z]+))?\s*(.*)$")
STYLE_RECORD = "@读音"
STYLE_NOTE = "@备注"
STYLE_INFO = "@设置"
PERSIST_NOTE = "※"
TRANSIENT_NOTE = "⚙"
GENERATED_MARK = "———— 以下是自动生成的行（振假名/旁注/拼音），每次刷新都会重建，请不要编辑 ————"
DERIVED_SUFFIXES = ("Ruby", "Pinyin", "Gloss")

Log = Callable[[str], None]


def characters_path(ep: Episode) -> Path:
    cfg = ep.project.config
    return cfg.path(cfg["project"].get("characters", "dict/characters.tsv"))


def role_of_style(name: str) -> Optional[tuple[str, str]]:
    """'JP' → ('jp','dialogue'), 'CN.Top' → ('cn','dialogue'), 'OP.JP' → ('jp','song:OP'); derived → None."""
    parts = name.split(".")
    if parts[-1] in DERIVED_SUFFIXES or name.startswith("@"):
        return None
    if parts[0] in ("JP", "CN"):
        return parts[0].lower(), "dialogue"
    if len(parts) >= 2 and parts[1] in ("JP", "CN"):
        return parts[1].lower(), "song:" + parts[0]
    return None


def ruby_style_for(base: str) -> str:
    return base + ".Ruby"


@dataclass
class Row:
    id: int
    cn: Optional[Event] = None
    jp: Optional[Event] = None
    record: Optional[Event] = None
    notes: list[Event] = field(default_factory=list)
    generated: list[Event] = field(default_factory=list)
    order: int = 0


@dataclass
class Report:
    rows: int = 0
    records_written: int = 0
    manual_kept: int = 0
    review: int = 0
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    elapsed: float = 0.0


# ======================================================================== styles
COMMENT_STYLES = {
    STYLE_RECORD: "Style: @读音,Microsoft YaHei,30,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,10,10,10,1",
    STYLE_NOTE: "Style: @备注,Microsoft YaHei,30,&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,10,10,10,1",
    STYLE_INFO: "Style: @设置,Microsoft YaHei,30,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,10,10,10,1",
}


def ensure_styles(doc: AssDocument, scheme: Optional[AssDocument]) -> None:
    if scheme is not None:
        for st in scheme.styles:
            if doc.style(st.name) is None or doc.style(st.name).name != st.name:
                doc.upsert_style(Style(dict(st.values)))
    for name, line in COMMENT_STYLES.items():
        if doc.style(name) is None:
            tmp = AssDocument.parse("[V4+ Styles]\nFormat: " + ", ".join(doc.style_format) + "\n" + line + "\n")
            doc.styles.append(tmp.styles[0])
    # derived ruby styles for every JP-role style in use
    for ev in list(doc.events):
        role = role_of_style(ev.style)
        if role and role[0] == "jp":
            rname = ruby_style_for(ev.style)
            if doc.style(rname) is None:
                base = doc.style(ev.style)
                if base is not None:
                    doc.upsert_style(derive_ruby_style(base, rname))


# ======================================================================== rows
def collect_rows(doc: AssDocument) -> tuple[dict[int, Row], list[Event]]:
    rows: dict[int, Row] = {}
    for idx, ev in enumerate(doc.events):
        m = ROW_EFFECT.match(ev.effect or "")
        if not m:
            continue
        rid = int(m.group(1))
        kind = m.group(2)
        row = rows.setdefault(rid, Row(rid, order=idx))
        if kind:
            row.generated.append(ev)
        elif ev.style == STYLE_RECORD:
            row.record = ev
        elif ev.style == STYLE_NOTE:
            row.notes.append(ev)
        else:
            role = role_of_style(ev.style)
            if role is None:
                continue
            if role[0] == "jp":
                if row.jp is None:
                    row.jp = ev
            elif row.cn is None:
                row.cn = ev
    return rows, doc.events


def assign_new_ids(doc: AssDocument, state: dict) -> int:
    """Give ids to JP/CN lines the user added (or duplicated) in Aegisub."""
    seen: dict[tuple[int, str], int] = {}
    added = 0
    for ev in doc.events:
        role = role_of_style(ev.style)
        if role is None or not ev.is_dialogue:
            continue
        m = ROW_EFFECT.match(ev.effect or "")
        if m and not m.group(2):
            rid = int(m.group(1))
            if (rid, role[0]) in seen:
                # a split/duplicated line: give the copy its own row
                rid = state["next_id"]
                state["next_id"] += 1
                ev.effect = f"#{rid}"
                added += 1
            seen[(rid, role[0])] = rid
            state["next_id"] = max(state["next_id"], rid + 1)
        elif not m:
            rid = state["next_id"]
            state["next_id"] += 1
            ev.effect = f"#{rid}"
            added += 1
    return added


# ======================================================================== records
def _locate(text: str, surface: str, nth: int) -> int:
    pos = -1
    for _ in range(nth + 1):
        pos = text.find(surface, pos + 1)
        if pos < 0:
            return -1
    return pos


def _occurrence(text: str, surface: str, start: int) -> int:
    n, pos = 0, text.find(surface)
    while 0 <= pos < start:
        n += 1
        pos = text.find(surface, pos + 1)
    return n


def detect_manual_edits(text: str, current: str, last: str) -> list[dict]:
    """Entries the user changed compared with what the tool last wrote."""
    cur = parse_record(current)
    old = parse_record(last)
    edits: list[dict] = []

    def located(entries: list[RecEntry]) -> list[tuple[int, RecEntry]]:
        out, cursor = [], 0
        for e in entries:
            p = text.find(e.surface, cursor)
            if p < 0:
                p = text.find(e.surface)
            if p >= 0:
                out.append((p, e))
                cursor = p + len(e.surface)
        return out

    old_by_pos = {(p, e.surface): e for p, e in located(old)}
    cur_loc = located(cur)
    cur_keys = {(p, e.surface) for p, e in cur_loc}
    for p, e in cur_loc:
        o = old_by_pos.get((p, e.surface))
        changed = (
            o is None or o.reading != e.reading or o.hidden != e.hidden or o.forced != e.forced
            or o.scale != e.scale or (o.uncertain and not e.uncertain) or e.alternatives != o.alternatives and not e.uncertain
        )
        if changed:
            edits.append({"surface": e.surface, "nth": _occurrence(text, e.surface, p), "reading": e.reading,
                          "hidden": e.hidden, "forced": e.forced, "scale": e.scale, "mono": e.adjacent_to_next})
    for (p, s), o in old_by_pos.items():
        if (p, s) not in cur_keys and not any(cp <= p < cp + len(ce.surface) for cp, ce in cur_loc):
            # the user deleted this entry → no ruby here
            edits.append({"surface": s, "nth": _occurrence(text, s, p), "reading": "", "hidden": True,
                          "forced": False, "scale": None, "mono": False})
    return edits


def merge_manual(text: str, machine: list[Entry], manual: list[dict], conflicts: list[str]) -> tuple[list[Entry], list[dict]]:
    """Apply manual edits; dictionary entries win unless the edit is forced (!)."""
    entries = list(machine)
    alive: list[dict] = []
    for m in manual:
        pos = _locate(text, m["surface"], m.get("nth", 0))
        if pos < 0:
            continue  # the word is no longer in the line
        a, b = pos, pos + len(m["surface"])
        overlapping = [e for e in entries if e.start < b and a < e.end]
        if any(e.locked for e in overlapping) and not m.get("forced"):
            locked = next(e for e in overlapping if e.locked)
            if not m.get("hidden") and m.get("reading") != locked.reading:
                conflicts.append(f"词库优先：{m['surface']}（{m.get('reading')}）被词库 {locked.surface}（{locked.reading}）覆盖；要单行强制请写 !{m['surface']}（…）")
            alive.append(m)
            continue
        entries = [e for e in entries if not (e.start < b and a < e.end)]
        if m.get("hidden"):
            e = Entry(a, b, m["surface"], "", [], "record", hidden=True)
        else:
            e = make_entry(a, m["surface"], m["reading"], "record", confidence=1.0)
            e.forced = bool(m.get("forced"))
            if m.get("scale"):
                e.note = f"scale={m['scale']}"
        e.locked = False
        entries.append(e)
        alive.append(m)
    entries.sort(key=lambda e: e.start)
    return entries, alive


def format_record(text: str, entries: list[Entry], mono_after: set[int]) -> str:
    out = []
    for k, e in enumerate(entries):
        scale = None
        if e.note.startswith("scale="):
            scale = float(e.note.split("=", 1)[1])
        s = format_entry(e.surface, e.reading, alternatives=e.alternatives if e.needs_review else [],
                         uncertain=e.needs_review, forced=e.forced, from_dict=e.locked, hidden=e.hidden, scale=scale)
        out.append(s)
        if k + 1 < len(entries):
            out.append("" if e.start in mono_after else " ")
    return "".join(out)


# ======================================================================== prepare
def _video_height(ffprobe: str, video: Optional[Path]) -> Optional[int]:
    if not video:
        return None
    try:
        from .proc import run as _run_hidden
        out = _run_hidden([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                              "-of", "csv=p=0", str(video)], capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def prepare(ep: Episode, *, log: Log = print, embedding: Optional[str] = None, force: bool = False) -> Path:
    cfg = ep.project.config
    if ep.work_ass.exists() and not force:
        raise FileExistsError(f"{ep.work_ass} 已存在（用 --force 重新生成；旧文件会备份到 history/）")
    t0 = time.time()
    src = ep.sources()
    for k in ("video", "cn", "jp"):
        if src[k] is None:
            raise FileNotFoundError(f"找不到 {ep.id} 的 {k} 原料（检查 subproj.toml 的 [episodes]）")
        log(f"  {k}: {src[k]}")
    cn_cfg = cfg.section("cn", ep.season)
    jp_cfg = cfg.section("jp", ep.season)
    cn = load_cn(src["cn"], cn_cfg)
    jp_cues = load_jp(src["jp"], JpOptions.from_config(jp_cfg))
    for note in cn.notes:
        log(f"  ⚠ {note}")
    log(f"  中文 Style 角色：{cn.style_roles}；文字：{cn.lang}")
    tx, ty = cfg.playres
    if cn.doc.playres != (tx, ty):
        vh = _video_height(cfg["tools"]["ffprobe"], src["video"])
        resample(cn.doc, tx, ty, video_height=vh)
        log(f"  分辨率：{cn.doc.playres} ← 缩放（视频高 {vh}）")

    # ----------------------------------------------------------- the new document
    doc = AssDocument()
    doc.script_info = [
        (None, "; subkit 工作文件 — 在 Aegisub 中编辑：中日文本、时间、@读音 行。生成的振假名行每次刷新都会重建。"),
        ("Title", f"{cfg['project'].get('name', '')} {ep.id}"),
        ("ScriptType", "v4.00+"), ("WrapStyle", "0"), ("ScaledBorderAndShadow", "yes"),
        ("YCbCr Matrix", "TV.709"), ("PlayResX", str(tx)), ("PlayResY", str(ty)),
    ]
    scheme_path = cfg.path(cfg["project"]["styles"])
    scheme = AssDocument.load(scheme_path) if scheme_path.exists() else None
    if scheme is not None:
        doc.style_format = list(scheme.style_format)
    ours = {s.name for s in scheme.styles} if scheme else set()
    rename: dict[str, str] = {}
    for st in cn.doc.styles:
        if cn.style_roles.get(st.name, "keep") == "keep" or any(cn.roles[i] == "keep" and ev.style == st.name for i, ev in enumerate(cn.doc.events)):
            new = st.name if st.name not in ours and role_of_style(st.name) is None else "src." + st.name
            rename[st.name] = new
            doc.styles.append(Style({k: st.values.get(k, "") for k in st.values} | {"Name": new}))

    # ----------------------------------------------------------- alignment
    dlg = sorted(((i, ev) for i, ev in enumerate(cn.doc.events) if cn.roles.get(i) in ("dialogue", "top")),
                 key=lambda x: (x[1].start, x[1].end, x[0]))
    arows = [AlignRow(k, ev.start, ev.end, plain_text(ev.text, breaks=" ")) for k, (_i, ev) in enumerate(dlg)]
    units = build_units(jp_cues)
    off = estimate_offset(arows, units)
    warp = piecewise_offsets(arows, units, off)
    for u in units:
        u.wstart, u.wend = warp(u.start), warp(u.end)
    backend = resolve_backend(embedding or cfg["align"]["embedding"])
    emb = None
    if backend != "none":
        try:
            emb = Embedder(backend, cache_path=ep.project.cache_dir / "embeddings.sqlite3")
        except Exception as exc:  # pragma: no cover
            log(f"  向量模型不可用（{exc}），只用时间+规则对齐")
    sim = make_sim_fn(emb, arows, units) if emb else None
    aligner = Aligner(arows, units, weights=Weights(), sim_fn=sim)
    if emb:
        prefetch_spans(emb, aligner, aligner.w.max_units)
    res = aligner.run()
    for first, n, shift in realign_detached_blocks(arows, units, res, lambda r, u: Aligner(r, u, weights=Weights(), sim_fn=sim)):
        t = units[first].start
        log(f"  {t // 6000}:{t % 6000 / 100:05.2f} 起的一段日文和中文相差 {shift / 100:+.1f}s（两个版本剪辑不同，如次回预告），已单独对齐（{n} 个短语）")
    log(f"  对齐：偏移 {off/100:+.2f}s，向量={backend}，行={len(arows)}，日文短语={len(units)}，省略语气词/孤立={len(res.skipped)}")

    # map unit → cue line hints
    unit_hints: dict[int, list[RubyHint]] = {}
    cue_by_idx = {c.idx: c for c in jp_cues}
    for u in units:
        cue = cue_by_idx[u.cue]
        line = cue.speech_lines[u.line]
        lpos = line.text.find(u.text.replace(" ", " "))
        if lpos < 0:
            continue
        for h in line.hints:
            if h.start >= 0 and lpos <= h.start and h.end <= lpos + len(u.text):
                unit_hints.setdefault(u.idx, []).append(RubyHint(h.start - lpos, h.end - lpos, h.surface, h.reading, h.source))

    jp_sep = jp_cfg.get("space", "  ")
    state = {"version": 1, "episode": ep.id, "next_id": 1, "rows": {}, "speaker_hints": speaker_reading_hints(jp_cues)}
    # characters: map caption labels to canonical names, collect unknown ones
    chars = CharacterTable.load(characters_path(ep))
    label_counts = Counter(u.speaker for u in units if u.speaker)
    added_chars = chars.append_unknown(dict(label_counts), state["speaker_hints"])
    if added_chars:
        log(f"  新角色写入 {characters_path(ep).name}：{'、'.join(added_chars)}（请补全假名/中文名/私的读法）")
    for u in units:
        if u.speaker:
            ch = chars.lookup(u.speaker)
            if ch is not None:
                u.speaker = ch.name
    events: list[tuple[int, int, list[Event]]] = []  # (start, order, events)

    def new_id() -> int:
        rid = state["next_id"]
        state["next_id"] += 1
        return rid

    def jp_text_for(idxs: list[int]) -> tuple[str, list[list]]:
        parts, hints, pos = [], [], 0
        for k, ui in enumerate(idxs):
            if k:
                parts.append(jp_sep)
                pos += len(jp_sep)
            for h in unit_hints.get(ui, []):
                hints.append([h.surface, h.reading, pos + h.start])
            parts.append(units[ui].text)
            pos += len(units[ui].text)
        return "".join(parts), hints

    lowconf = 0
    for k, (src_i, ev) in enumerate(dlg):
        rid = new_id()
        top = cn.roles[src_i] == "top"
        idxs = res.rows[k]
        jp_text, hints = jp_text_for(idxs)
        speakers = Counter(units[i].speaker for i in idxs if units[i].speaker)
        speaker = speakers.most_common(1)[0][0] if speakers else ""
        cn_ev = ev.copy(style="CN.Top" if top else "CN", layer=1, effect=f"#{rid}",
                        text=normalize_quotes(clean_cn_text(ev.text, linebreak=cn_cfg["linebreak"], space=cn_cfg["space"]), cn_cfg.get("quotes", "")))
        jp_ev = ev.copy(style="JP.Top" if top else "JP", layer=0, effect=f"#{rid}", name=speaker, text=normalize_quotes(jp_text, jp_cfg.get("quotes", "")))
        group = [cn_ev, jp_ev]
        doubt = _boundary_doubt(units, idxs, res.confidence[k] if k < len(res.confidence) else 1.0)
        if doubt:
            lowconf += 1
            group.append(Event(kind="Comment", start=ev.start, end=ev.end, style=STYLE_NOTE, effect=f"#{rid}",
                               text=f"{PERSIST_NOTE}{doubt}"))
        if k in res.swapped_rows:
            group.append(Event(kind="Comment", start=ev.start, end=ev.end, style=STYLE_NOTE, effect=f"#{rid}",
                               text=f"{PERSIST_NOTE}日文和相邻行交换了顺序（中日语序相反时按中文的顺序对应）：请确认"))
        state["rows"][str(rid)] = {"hints": hints, "speaker": speaker}
        events.append((ev.start, len(events), group))
    # skipped units: fillers → note on nearest row; content → orphan JP rows
    row_spans = [(dlg[k][1].start, dlg[k][1].end) for k in range(len(dlg))]
    orphan_buf: list[int] = []

    def flush_orphans() -> None:
        if not orphan_buf:
            return
        rid = new_id()
        t0_, t1_ = int(units[orphan_buf[0]].wstart), int(units[orphan_buf[-1]].wend)
        text, hints = jp_text_for(orphan_buf)
        spk = units[orphan_buf[0]].speaker or ""
        group = [
            Event(kind="Dialogue", layer=1, start=t0_, end=t1_, style="CN", effect=f"#{rid}", text=""),
            Event(kind="Dialogue", layer=0, start=t0_, end=t1_, style="JP", effect=f"#{rid}", name=spk, text=text),
            Event(kind="Comment", start=t0_, end=t1_, style=STYLE_NOTE, effect=f"#{rid}",
                  text=f"{PERSIST_NOTE}这段日文找不到对应的中文行（中文没有翻译？），已单独成行；可删除或并入相邻行"),
        ]
        state["rows"][str(rid)] = {"hints": hints, "speaker": spk}
        events.append((t0_, len(events), group))
        orphan_buf.clear()

    omitted: dict[int, list[str]] = {}
    unit_row = {ui: k for k, r in enumerate(res.rows) for ui in r}
    for ui in res.skipped:
        u = units[ui]
        if u.interj:
            # A filler the aligner left out (the CN line has no exclamation and its time does
            # not fit) is dropped — the Utena finals drop those too — but listed in a note on
            # the row that took the rest of its caption (else the row closest in time).
            flush_orphans()
            best = next((unit_row[nb] for nb in (ui + 1, ui - 1) if nb in unit_row and units[nb].cue == u.cue), None)
            if best is None and row_spans:
                best = min(range(len(row_spans)), key=lambda k: abs((row_spans[k][0] + row_spans[k][1]) / 2 - (u.wstart + u.wend) / 2))
            if best is not None:
                omitted.setdefault(best, []).append(u.text)
        else:
            if orphan_buf and units[orphan_buf[-1]].cue != u.cue and u.wstart - units[orphan_buf[-1]].wend > 100:
                flush_orphans()
            orphan_buf.append(ui)
    flush_orphans()
    for k, words in omitted.items():
        st, en = row_spans[k]
        for (_s, _o, group) in events:
            if group and group[0].effect == f"#{k + 1}":
                group.append(Event(kind="Comment", start=st, end=en, style=STYLE_NOTE, effect=f"#{k + 1}",
                                   text=f"{PERSIST_NOTE}已省略中文没有翻译的语气词：{'  '.join(words)}（需要的话复制到日文行）"))
                break

    # songs: pair source JP/CN lyric lines by overlap
    song_jp = [ev for i, ev in enumerate(cn.doc.events) if cn.roles.get(i) == "song_jp"]
    song_cn = [ev for i, ev in enumerate(cn.doc.events) if cn.roles.get(i) == "song_cn"]
    used_cn: set[int] = set()
    total = max((ev.end for ev in cn.doc.events), default=0)
    for sj in song_jp:
        fam = _song_family(sj.style, sj.start, total)
        partner = None
        for ci, sc in enumerate(song_cn):
            if ci in used_cn:
                continue
            ov = min(sj.end, sc.end) - max(sj.start, sc.start)
            if ov > 0.5 * max(1, min(sj.duration, sc.duration)):
                partner, _ = sc, used_cn.add(ci)
                break
        rid = new_id()
        group = []
        if partner is not None:
            group.append(partner.copy(style=f"{fam}.CN", layer=1, effect=f"#{rid}"))
        group.append(sj.copy(style=f"{fam}.JP", layer=0, effect=f"#{rid}"))
        state["rows"][str(rid)] = {"hints": [], "speaker": ""}
        events.append((sj.start, len(events), group))
    for ci, sc in enumerate(song_cn):
        if ci not in used_cn:
            rid = new_id()
            events.append((sc.start, len(events), [sc.copy(style=f"{_song_family(sc.style, sc.start, total)}.CN", layer=1, effect=f"#{rid}")]))
            state["rows"][str(rid)] = {"hints": [], "speaker": ""}
    # kept lines (signs, staff, source comments)
    for i, ev in enumerate(cn.doc.events):
        if cn.roles.get(i) == "keep":
            events.append((ev.start, len(events), [ev.copy(style=rename.get(ev.style, ev.style))]))

    events.sort(key=lambda x: (x[0], x[1]))
    doc.events = [Event(kind="Comment", start=0, end=0, style=STYLE_INFO,
                        text=f"subkit {ep.id} — 视频：{src['video'].name}；中文：{src['cn'].name}；日文：{src['jp'].name}")]
    for _s, _o, group in events:
        doc.events.extend(group)
    ensure_styles(doc, scheme)
    # Aegisub opens the video/audio automatically from Project Garbage
    doc._sections = []
    ep.work_dir.mkdir(parents=True, exist_ok=True)
    if ep.work_ass.exists():
        _backup(ep)
    doc.save(ep.work_ass)
    garbage = f"[Aegisub Project Garbage]\nAudio File: {src['video']}\nVideo File: {src['video']}\n"
    with open(ep.work_ass, "a", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\n" + garbage)
    ep.save_state(state)
    log(f"  prepare 完成：{len(dlg)} 行对白，{lowconf} 行对齐待确认，{sum(len(v) for v in omitted.values())} 个语气词省略，{time.time() - t0:.1f}s")
    return ep.work_ass


def _boundary_doubt(units, idxs: list[int], conf: float) -> Optional[str]:
    """A row boundary that cuts through one Japanese caption is where alignment
    mistakes concentrate (Utena: flags 7.5% of rows, 43% of them really wrong)."""
    if not idxs:
        return None
    first, last = units[idxs[0]], units[idxs[-1]]
    msgs = []
    if idxs[0] > 0:
        prev = units[idxs[0] - 1]
        if prev.cue == first.cue and prev.line == first.line:
            msgs.append(f"「{prev.text}｜{first.text}」")
    if idxs[-1] + 1 < len(units):
        nxt = units[idxs[-1] + 1]
        if nxt.cue == last.cue and nxt.line == last.line:
            msgs.append(f"「{last.text}｜{nxt.text}」")
    if not msgs:
        return None
    level = "较可能有误" if conf < 0.78 else "请确认"
    return f"对齐{level}：这行和相邻行的分界在同一句日文中间 {' '.join(msgs)}"


def _song_family(style: str, start: Optional[int] = None, total: Optional[int] = None) -> str:
    """OP / ED / IN from the source style name (opjp, edcn, shokeki_edjp); a shared
    ``OPED`` style is decided by time (first half of the episode = OP)."""
    low = style.lower()
    op = re.search(r"(^|[^a-z])op", low) is not None
    ed = re.search(r"(^|[^a-z])ed|oped", low) is not None
    if op and not ed:
        return "OP"
    if ed and not op:
        return "ED"
    if op and ed and start is not None and total:
        return "OP" if start < total / 2 else "ED"
    return "IN"


def _backup(ep: Episode) -> None:
    hist = ep.work_dir / "history"
    hist.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(ep.work_ass, hist / f"work-{stamp}.ass")
    olds = sorted(hist.glob("work-*.ass"))
    for p in olds[:-30]:
        p.unlink(missing_ok=True)


# ======================================================================== refresh
STYLE_K = "@K"
STYLE_K_CN = "@K中"
STYLE_PINYIN = "@拼音"
EXTRA_RECORD_STYLES = {STYLE_K, STYLE_K_CN, STYLE_PINYIN}


def _row_extra_records(doc: AssDocument) -> dict[int, dict[str, Event]]:
    out: dict[int, dict[str, Event]] = {}
    for ev in doc.events:
        if ev.style in EXTRA_RECORD_STYLES:
            m = ROW_EFFECT.match(ev.effect or "")
            if m and not m.group(2):
                out.setdefault(int(m.group(1)), {})[ev.style] = ev
    return out


def _hints_for(text: str, rs: dict) -> list[RubyHint]:
    hints = []
    for surface, reading, pos in rs.get("hints", []):
        p = text.find(surface, max(0, pos - 3))
        if p < 0:
            p = text.find(surface)
        if p >= 0:
            hints.append(RubyHint(p, p + len(surface), surface, reading, "cc"))
    return hints


def _jp_entries(engine: ReadingEngine, text: str, row: Row, rs: dict, excluded: list[tuple[int, int]],
                rep: Report) -> tuple[list[Entry], list[str], str]:
    """Machine analysis + manual edits → (entries, conflicts, record text)."""
    speaker = (row.jp.name if row.jp else "") or rs.get("speaker") or None
    machine = engine.analyze(text, speaker=speaker, hints=_hints_for(text, rs))
    machine = [e for e in machine if not any(a < e.end and e.start < b for a, b in excluded)]
    manual = list(rs.get("manual", []))
    current = row.record.text if row.record is not None else None
    last = rs.get("record")
    if current is not None and last is not None and current != last:
        edits = detect_manual_edits(text, current, last)
        keys = {(e["surface"], e["nth"]) for e in edits}
        manual = [m for m in manual if (m["surface"], m.get("nth", 0)) not in keys] + edits
    conflicts: list[str] = []
    entries, manual = merge_manual(text, machine, manual, conflicts)
    rs["manual"] = manual
    rep.manual_kept += len(manual)
    mono_after = set()
    if current:
        for e in parse_record(current):
            if e.adjacent_to_next:
                p = text.find(e.surface)
                if p >= 0:
                    mono_after.add(p)
    record = format_record(text, entries, mono_after) if entries else ""
    rs["record"] = record
    return entries, conflicts, record


def _pieces(entries: list[Entry]) -> list[RubyPiece]:
    out = []
    for word, e in enumerate(entries):
        if e.hidden:
            continue
        scale = float(e.note.split("=", 1)[1]) if e.note.startswith("scale=") else None
        for p in e.parts:
            out.append(RubyPiece(p.start, p.end, p.reading, scale, word=word))
    return out


def _derived_style(doc: AssDocument, base: str, suffix: str) -> str:
    name = f"{base}.{suffix}"
    if doc.style(name) is None:
        b = doc.style(base)
        family = doc.style(f"{base.split('.')[0]}.{suffix}")
        if family is not None:
            doc.upsert_style(family.copy(name))
        elif b is not None:
            doc.upsert_style(derive_ruby_style(b, name, ratio=0.5 if suffix != "Gloss" else 0.45))
    return name


def refresh(ep: Episode, *, log: Log = print) -> Report:
    t0 = time.time()
    cfg = ep.project.config
    rep = Report()
    doc = AssDocument.load(ep.work_ass)
    state = ep.load_state()
    state.setdefault("rows", {})
    state.setdefault("next_id", 1)
    added = assign_new_ids(doc, state)
    if added:
        log(f"  新增行：{added}")
    rows, _ = collect_rows(doc)
    extras = _row_extra_records(doc)
    rep.rows = len(rows)
    gdict = Path(expand_path(cfg["tools"]["global_dict"]))
    rdict = ReadingDict.load(gdict, cfg.path(cfg["project"]["dict"]))
    rdict.merge_characters(CharacterTable.load(characters_path(ep)))
    engine = ReadingEngine(rdict)
    scheme_path = cfg.path(cfg["project"]["styles"])
    ensure_styles(doc, AssDocument.load(scheme_path) if scheme_path.exists() else None)
    for name in (STYLE_K, STYLE_K_CN, STYLE_PINYIN):
        if doc.style(name) is None:
            st = (doc.style(STYLE_RECORD) or Style.new(name)).copy(name)
            doc.upsert_style(st)
    kcfg = cfg.data.get("karaoke", {})
    songs = SongMemory(cfg.path(cfg["project"].get("songs", "dict/songs.json")))
    pcfg = cfg.data.get("pinyin", {})
    pinyin_on = bool(pcfg.get("enabled", False))
    pinyin_styles = set(pcfg.get("styles", ["CN"]))

    jobs: list[RubyJob] = []
    out_records: dict[int, list[Event]] = {}
    notes: dict[int, list[str]] = {}
    kara: dict[int, dict] = {}
    for rid, row in sorted(rows.items(), key=lambda kv: kv[1].order):
        rs = state["rows"].setdefault(str(rid), {})
        recs: list[Event] = []
        nts: list[str] = []
        ex = extras.get(rid, {})
        # ------------------------------------------------ JP line
        jp = row.jp
        if jp is not None and jp.is_dialogue:
            text = VisibleText(jp.text).text
            glosses = parse_glosses(jp.text)
            excluded = [(g.start, g.end) for g in glosses if g.target]
            entries, conflicts, record = ([], [], "")
            if text.strip():
                entries, conflicts, record = _jp_entries(engine, text, row, rs, excluded, rep)
            nts.extend(conflicts)
            rep.conflicts.extend(conflicts)
            if record:
                recs.append(jp.copy(kind="Comment", style=STYLE_RECORD, effect=f"#{rid}", text=record))
                rep.records_written += 1
                rep.review += sum(1 for e in entries if e.needs_review)
            role = role_of_style(jp.style)
            is_song = bool(role and role[1].startswith("song"))
            mem = songs.get(text) if is_song and text.strip() else None
            if mem and mem.get("record") and not rs.get("song_seeded") and record:
                # same lyric corrected in another episode: reuse those readings as manual edits
                edits = detect_manual_edits(text, mem["record"], record)
                rs["song_seeded"] = True
                if edits:
                    rs["manual"] = rs.get("manual", []) + edits
                    saved_record, row.record = row.record, None
                    entries, conflicts, record = _jp_entries(engine, text, row, rs, excluded, rep)
                    row.record = saved_record
                    recs = [jp.copy(kind="Comment", style=STYLE_RECORD, effect=f"#{rid}", text=record)] if record else []
            pieces = _pieces(entries)
            if pieces:
                jobs.append(RubyJob(jp, _derived_style(doc, jp.style, "Ruby"), pieces, effect=f"#{rid}:ruby", prefer_center=True))
            if glosses:
                gp = [RubyPiece(g.start, g.end, g.text, g.scale) for g in glosses]
                jobs.append(RubyJob(jp, _derived_style(doc, jp.style, "Gloss"), gp, effect=f"#{rid}:gloss"))
            if is_song:
                krec = ex.get(STYLE_K)
                if krec is None and mem and mem.get("k"):
                    krec = jp.copy(kind="Comment", style=STYLE_K, effect=f"#{rid}", text=rescale_k(mem["k"], jp.end - jp.start))
                    rs["k_template"] = ""
                if krec is None:
                    tmpl = template_record(text, pieces, jp.end - jp.start)
                    if tmpl:
                        krec = jp.copy(kind="Comment", style=STYLE_K, effect=f"#{rid}", text=tmpl)
                        rs["k_template"] = tmpl
                if krec is not None and krec.text == rs.get("k_template"):
                    nts.append("K 轴是自动均分的占位：请选中 @K 行，在 Aegisub 的卡拉 OK 模式里按演唱打轴")
                if krec is not None:
                    kara[rid] = {"krec": krec, "pieces": pieces, "text": text}
                    recs.append(krec.copy(start=jp.start, end=jp.end))
                    if STYLE_K_CN in ex:
                        recs.append(ex[STYLE_K_CN].copy(start=jp.start, end=jp.end))
                # remember this lyric's work for later episodes
                timed = krec is not None and krec.text != rs.get("k_template")
                if rs.get("manual") or timed or STYLE_K_CN in ex:
                    songs.put(text, record=record if rs.get("manual") else "", k=krec.text if timed else "",
                              k_cn=ex[STYLE_K_CN].text if STYLE_K_CN in ex else "", episode=ep.id)
        # ------------------------------------------------ CN line
        cn = row.cn
        if cn is not None and cn.is_dialogue:
            ctext = VisibleText(cn.text).text
            cglosses = parse_glosses(cn.text)
            with_pinyin = pinyin_on and cn.style in pinyin_styles and ctext.strip()
            if cglosses:
                gp = [RubyPiece(g.start, g.end, g.text, g.scale) for g in cglosses]
                gj = RubyJob(cn, _derived_style(doc, cn.style, "Gloss"), gp, effect=f"#{rid}:gloss")
                if with_pinyin:
                    gj.level, gj.below_style = 1, _derived_style(doc, cn.style, "Pinyin")
                jobs.append(gj)
            if with_pinyin:
                items = line_pinyin(ctext)
                auto = {i: s for i, s, _a in items}
                manual = {int(k): v for k, v in rs.get("py_manual", {}).items()}
                cur = ex[STYLE_PINYIN].text if STYLE_PINYIN in ex else None
                if cur is not None and cur != rs.get("py_record"):
                    # the user edited the record: entries that differ from automatic become manual
                    for i, syl in parse_pinyin_record(cur, ctext).items():
                        if syl and auto.get(i) != syl:
                            manual[i] = syl
                        else:
                            manual.pop(i, None)
                manual = {i: s for i, s in manual.items() if i in auto}
                rs["py_manual"] = {str(k): v for k, v in manual.items()}
                mode = pcfg.get("mode", "PY")
                ppieces = [RubyPiece(i, i + 1, py_display(manual.get(i, s), mode)) for i, s, _a in items if manual.get(i, s)]
                if ppieces:
                    jobs.append(RubyJob(cn, _derived_style(doc, cn.style, "Pinyin"), ppieces, effect=f"#{rid}:py"))
                prec = format_pinyin_record(items, ctext, manual)
                rs["py_record"] = prec
                if prec:
                    recs.append(cn.copy(kind="Comment", style=STYLE_PINYIN, effect=f"#{rid}", text=prec))
        out_records[rid] = recs
        notes[rid] = nts

    measurer = Measurer(doc, ffmpeg=cfg["tools"]["ffmpeg"], cache_path=ep.project.cache_dir / "render.sqlite3")
    rcfg = cfg["ruby"]
    gen = RubyGenerator(doc, measurer, gap=float(rcfg["gap"]), min_scale=float(rcfg["min_scale"]), overhang=float(rcfg["overhang"]))
    gen.run(jobs)
    rep.fonts = sorted(measurer.font_log)
    gen_by_row: dict[int, list[Event]] = {}
    ruby_jobs: dict[int, RubyJob] = {}
    for j in jobs:
        rid = int(j.effect.split(":")[0][1:])
        gen_by_row.setdefault(rid, []).extend(j.out)
        if j.effect.endswith(":ruby"):
            ruby_jobs[rid] = j
        notes.setdefault(rid, []).extend(j.warnings)
        rep.warnings.extend(j.warnings)

    # ------------------------------------------------ karaoke (after ruby placement)
    default_fx = str(kcfg.get("effect", "kf"))
    for rid, kd in kara.items():
        row = rows[rid]
        syls = parse_k(kd["krec"].text)
        timing = kalign(kd["text"], kd["pieces"], syls)
        if timing.unmatched:
            notes.setdefault(rid, []).append(f"K 轴有 {timing.unmatched} 个假名对不上日文读音（检查 @K 行的假名）")
        fx = fx_slot(row.jp.text, row.cn.text if row.cn else "") or default_fx
        kind = BUILTIN_FX.get(fx, "kf")
        if kind is None:
            continue
        row.jp.text = karaoke_text(row.jp.text, timing, kind)
        rjob = ruby_jobs.get(rid)
        if rjob is not None:
            for ev, (piece, _x, _sc, _ov) in zip(rjob.out, rjob.placements):
                syl = timing.piece_syls.get(piece.start, [])
                head, _sep, reading = ev.text.partition("}")
                ev.text = head + "}" + ruby_karaoke(remove_tags_named(reading, KARAOKE_TAGS), syl, kind)
        if row.cn is not None and kcfg.get("cn", True):
            ex = extras.get(rid, {})
            mapping = parse_cn_mapping(ex[STYLE_K_CN].text) if STYLE_K_CN in ex else None
            row.cn.text = cn_karaoke(row.cn.text, timing, kind, mapping, syls)

    # ------------------------------------------------ rebuild events
    out: list[Event] = []
    tail: list[Event] = []
    emitted: set[int] = set()
    for ev in doc.events:
        if ev.style == STYLE_INFO and GENERATED_MARK in ev.text:
            continue
        m = ROW_EFFECT.match(ev.effect or "")
        if not m:
            out.append(ev)
            continue
        rid = int(m.group(1))
        if rid in emitted:
            continue
        emitted.add(rid)
        row = rows.get(rid)
        if row is None:
            continue
        base = row.jp or row.cn
        if row.cn is not None:
            out.append(row.cn)
        if row.jp is not None:
            out.append(row.jp)
        for r in out_records.get(rid, []):
            out.append(r.copy(start=base.start, end=base.end))
        for n in row.notes:
            if not n.text.startswith(TRANSIENT_NOTE):
                out.append(n.copy(start=base.start, end=base.end))
        for msg in notes.get(rid, []):
            out.append(Event(kind="Comment", start=base.start, end=base.end, style=STYLE_NOTE, effect=f"#{rid}",
                             text=f"{TRANSIENT_NOTE}{msg}"))
        tail.extend(gen_by_row.get(rid, []))
    if tail:
        # generated lines live in one block at the end, so the Aegisub grid shows only what you edit
        out.append(Event(kind="Comment", start=0, end=0, style=STYLE_INFO, text=GENERATED_MARK))
        out.extend(tail)
    doc.events = out
    _backup(ep)
    doc.save(ep.work_ass)
    ep.save_state(state)
    songs.save()
    rep.elapsed = time.time() - t0
    return rep
