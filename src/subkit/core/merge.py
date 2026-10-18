"""Initial merge: read-only sources → aligned working document (FORMAT-v1 §6).

Every character of both sources ends up somewhere auditable:

* translation (CN) dialogue rows and original (JP) phrase units are aligned by
  :mod:`subkit.align.dp2` (order-preserving) and become dialogue Groups;
* an original-language unit without a partner becomes its own Group at its
  source position (displayed); in compact mode a stand-alone untranslated filler
  becomes a restorable ``hidden_filler`` Comment of the adjacent Group;
* all other events of the translation file (signs, song lyrics, credits, the
  file's own Comment lines) become their own Groups, unchanged;
* structural metadata of the original captions (speaker labels, ≪≫, readings…)
  stays in the source's spans.

Groups are interleaved by time with a k-way merge that never changes the order
inside any single source stream (FORMAT-v1 §7).
"""

from __future__ import annotations

import copy
import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..align.dp2 import Aligner2, AlignOutput, Block, Row, Weights2, realign_detached, time_warp
from ..ass.text import plain_text, tokenize
from ..languages import lane_label
from ..model.document import Comment, Document, Group, Member, Run
from ..model.ids import new_id
from ..model.source import Source
from .units import Unit, build_units


@dataclass
class MergeSettings:
    filler_mode: str = "keep"                 # keep (show every filler) | compact (hide stand-alone untranslated fillers)
    jp_sep: str = "  "                        # derived separator between JP lines / captions inside one member
    semantic: bool = False                    # use the embedding model (optional, user-enabled)
    weights: dict[str, Any] = field(default_factory=dict)
    styles: dict[str, str] = field(default_factory=lambda: {
        "translation/dialogue": "CN", "translation/top": "CN.Top", "translation/alt": "CN.Alt",
        "original/dialogue": "JP", "original/top": "JP.Top", "original/alt": "JP.Alt",
        "translation/song": "{fam}.CN", "original/song": "{fam}.JP",
    })
    pairing: str = "align"                   # align | same_time (already-paired bilingual import)
    # Several translation lines on one original line (the fansub split one sentence): first = one group per
    # translation line, the original whole with the first (user 2026-10-14: move its tail with 结尾→下一组);
    # cut = also cut the original at its sentence breaks; keep = one multi-line group (the old behaviour).
    joint: str = "first"


@dataclass
class MergeResult:
    document: Document
    units: list[Unit]
    rows: list[Row]
    alignment: AlignOutput
    log: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ helpers
def dialogue_lanes(doc: Document, group: Group) -> set[str]:
    """Lanes with visible, non-comment dialogue text; shared by results and filtering."""
    return {m.lane for m in doc.members_of(group)
            if m.lane in ("original", "translation") and m.role != "comment"
            and m.source_kind != "Comment" and plain_text(m.text).strip()}


def group_counts(doc: Document) -> dict[str, int]:
    counts = dict(total=len(doc.groups), dialogue=0, both=0, translation_only=0,
                  original_only=0, song=0, sign_info=0)
    for group in doc.groups:
        if group.kind == "dialogue":
            counts["dialogue"] += 1
            lanes = dialogue_lanes(doc, group)
            key = ("both" if len(lanes) == 2 else "translation_only" if lanes == {"translation"}
                   else "original_only" if lanes == {"original"} else None)
            if key:
                counts[key] += 1
        elif group.kind == "song":
            counts["song"] += 1
        elif group.kind in ("sign", "info"):
            counts["sign_info"] += 1
    return counts


def runs_of_event(src: Source, e: int) -> list[Run]:
    """The full event text as provenance runs (override blocks as tag runs) — lossless."""
    text = src.events[e].text
    metadata = {k for sp in src.spans_of(e) for k in range(sp.a, sp.b)}
    runs: list[Run] = []
    for tok in tokenize(text):
        kind = "tag" if tok.kind in ("tags", "comment") else "text"
        start = tok.start
        while start < tok.end:
            if start in metadata:
                start += 1
                continue
            end = start + 1
            while end < tok.end and end not in metadata:
                end += 1
            if runs and runs[-1].kind == kind == "text" and runs[-1].b == start:
                last = runs[-1]
                runs[-1] = last.copy(text=last.text + text[start:end], b=end)
            else:
                runs.append(Run(text[start:end], kind, src.id, e, start, end))
            start = end
    if not runs and text == "":
        return []
    return runs


def _only_reading_hints(src: Source, e: int, a: int, b: int) -> bool:
    """True when every non-space character of text[a:b] lies inside a reading_hint span (the bracket)."""
    covered = [(s.a, s.b) for s in src.spans_of(e) if s.kind == "reading_hint"]
    if not covered:
        return False
    text = src.events[e].text
    return all(ch in " 　\t" or any(lo <= i < hi for lo, hi in covered) for i, ch in enumerate(text[a:b], a))


def units_runs(units: list[Unit], src: Source, sep: str) -> list[Run]:
    """Runs for a contiguous list of units: one run per uninterrupted source range."""
    runs: list[Run] = []
    prev: Optional[Unit] = None
    for u in units:
        text = src.events[u.e].text
        same_line = prev is not None and prev.e == u.e and prev.line == u.line
        if same_line and text[prev.b:u.a].strip(" 　\t") == "":
            last = runs[-1]
            runs[-1] = last.copy(text=text[last.a:u.b], b=u.b)
        elif same_line and _only_reading_hints(src, u.e, prev.b, u.a):
            # 風（うず）を: the bracket became a reading; the text continues without a separator.
            runs.append(Run(text[u.a:u.b], "text", src.id, u.e, u.a, u.b))
        else:
            if runs:
                runs.append(Run(sep, "sep"))
            runs.append(Run(text[u.a:u.b], "text", src.id, u.e, u.a, u.b))
        prev = u
    return runs


def song_family(style: str, start: int, total: int) -> str:
    import re

    low = style.lower()
    op = re.search(r"(^|[^a-z])op", low) is not None
    ed = re.search(r"(^|[^a-z])ed|oped", low) is not None
    if op and not ed:
        return "OP"
    if ed and not op:
        return "ED"
    if op and ed and total:
        return "OP" if start < total / 2 else "ED"
    return "IN"


def build_rows(cn: Source) -> list[Row]:
    """Translation dialogue rows (dialogue + top), interleaved by time per style stream (no reordering inside a style)."""
    streams: dict[str, list[Row]] = {}
    for ev in cn.events:
        role = cn.roles.get(ev.i)
        if role not in ("dialogue", "top"):
            continue
        streams.setdefault(ev.style or "", []).append(
            Row(0, ev.start, ev.end, plain_text("".join(r.text for r in runs_of_event(cn, ev.i)), breaks=" ").strip(), ev.i, ev.style or ""))
    merged = list(heapq.merge(*streams.values(), key=lambda r: r.start))
    for k, r in enumerate(merged):
        r.idx = k
    return merged


def _same_time_pairing(rows: list[Row], units: list[Unit], ja: Source, *, cancel_check) -> AlignOutput:
    """Pair whole source events at matching times, consuming both sequences once."""
    lines: dict[int, list[int]] = {}
    for u in units:
        u.wstart, u.wend = u.start, u.end
        lines.setdefault(u.e, []).append(u.idx)
    original = list(lines.items())
    for e, indices in original:
        units[indices[0]].wstart = ja.events[e].start
        units[indices[-1]].wend = ja.events[e].end
    blocks: list[Block] = []
    cursor = 0
    for row in rows:
        cancel_check()
        partner = None
        for i in range(cursor, len(original)):
            cancel_check()
            event = ja.events[original[i][0]]
            if abs(row.start - event.start) <= 10 and abs(row.end - event.end) <= 10:
                partner = i
                break
        if partner is None:
            blocks.append(Block("empty_row", [row.idx], []))
            continue
        # Consuming preceding lines prevents a later pair from crossing this one.
        for i in range(cursor, partner):
            blocks.append(Block("unpaired", [], original[i][1]))
        blocks.append(Block("pair", [row.idx], original[partner][1]))
        cursor = partner + 1
    for _e, indices in original[cursor:]:
        cancel_check()
        blocks.append(Block("unpaired", [], indices))
    return AlignOutput(blocks)


def align(cn: Source, ja: Source, settings: MergeSettings, *,
          sim_factory: Optional[Callable[[list[Row], list[Unit]], Optional[Callable[[str, str], float]]]] = None,
          log: Callable[[str], None] = lambda s: None, summary=None, cancel_check=lambda: None) -> tuple[list[Unit], list[Row], AlignOutput]:
    cancel_check()
    units = build_units(ja)
    rows = build_rows(cn)
    cancel_check()
    if settings.pairing == "same_time":
        out = _same_time_pairing(rows, units, ja, cancel_check=cancel_check)
        counts = Counter(block.kind for block in out.blocks)
        if summary is not None:
            summary.update(offset_ms=0, translation_rows=len(rows), original_units=len(units), cuts=[],
                           pairing="same_time", pairs=counts["pair"],
                           translation_only=counts["empty_row"], original_only=counts["unpaired"])
        log(f"按相同时间配对：配对 {counts['pair']} 组，只有中文 {counts['empty_row']} 行，"
            f"只有日文 {counts['unpaired']} 行")
        return units, rows, out
    warp, off, wrep = time_warp(rows, units)          # offset, drift and jumps between the two releases
    for u in units:
        u.wstart, u.wend = warp(u.start), warp(u.end)
    w = Weights2(**settings.weights)
    cancel_check()
    sim = sim_factory(rows, units) if (settings.semantic and sim_factory) else None
    make = lambda r, u: Aligner2(r, u, weights=w, sim_fn=sim, filler_mode=settings.filler_mode, cancel_check=cancel_check)  # noqa: E731
    cancel_check()
    out = make(rows, units).run()
    cancel_check()
    realign_detached(rows, units, out, make)
    cancel_check()
    out.offset = off
    if summary is not None:
        summary.update(offset_ms=off, translation_rows=len(rows), original_units=len(units),
                       cuts=[(at, after) for at, before, after in wrep.jumps] if wrep else [])
        summary["cuts"].extend((units[first].start, shift) for first, n, shift in out.reanchored)
    langs = {"original": ja.lang, "translation": cn.lang}
    log(f"对齐：偏移 {off / 1000:+.2f}s，{lane_label('translation', langs)} {len(rows)} 行，"
        f"{lane_label('original', langs)}短语 {len(units)} 个")
    if wrep is not None and (wrep.jumps or any(abs(b - a) >= 400 for _s, _e, a, b in wrep.segments)):
        for line in wrep.lines():
            log(line)
    for first, n, shift in out.reanchored:
        t = units[first].start
        log(f"  {t // 60000}:{t % 60000 / 1000:06.3f} 起的一段原声字幕与翻译相差 {shift / 1000:+.1f}s（两个版本剪辑不同），已单独对齐 {n} 段")
    return units, rows, out


# ------------------------------------------------------------------ document
def build_document(episode: str, cn: Source, ja: Source, units: list[Unit], rows: list[Row], out: AlignOutput,
                   settings: MergeSettings, *, cancel_check=lambda: None) -> Document:
    doc = Document(episode=episode)
    st = settings.styles
    total = max((e.end for e in cn.events), default=0)

    def add_group(kind: str, members: list[Member], origin: dict[str, Any]) -> Group:
        g = Group(new_id("g_"), kind, [m.id for m in members], [], "auto", origin)
        for m in members:
            m.start, m.end = max(0, m.start), max(0, m.end)
            m.group = g.id
            doc.members[m.id] = m
        return g

    def cn_member(e: int, role: str, layer: Optional[int] = None) -> Member:
        ev = cn.events[e]
        speakers = Counter(s.value for s in cn.spans_of(e) if s.kind == "speaker_label" and s.value).most_common(1)
        return Member(new_id("m_"), "", cn.lang, "translation", role, ev.start, ev.end, layer=1 if layer is None else layer,
                      style=st.get(f"translation/{role}", "CN"), margins=[0, 0, 0], effect="",
                      speaker={"character": None, "label": speakers[0][0]} if speakers else None,
                      runs=runs_of_event(cn, e))

    def jp_member(us: list[Unit], start: int, end: int, role: str) -> Member:
        spk = Counter(u.speaker for u in us if u.speaker).most_common(1)
        return Member(new_id("m_"), "", ja.lang, "original", role, int(start), int(end), layer=0,
                      style=st.get(f"original/{role}", "JP"),
                      speaker={"character": None, "label": spk[0][0]} if spk else None,
                      runs=units_runs(us, ja, settings.jp_sep))

    # 1. dialogue stream (alignment order)
    dialogue_groups: list[tuple[int, Group]] = []
    pending_hidden: list[Block] = []
    for bl in out.blocks:
        cancel_check()
        us = [units[k] for k in bl.units]
        if bl.kind == "unpaired" and bl.hidden:
            pending_hidden.append(bl)
            continue
        members: list[Member] = []
        role = "dialogue"
        if bl.rows:
            roles = [cn.roles.get(rows[r].e, "dialogue") for r in bl.rows]
            role = "top" if all(x == "top" for x in roles) else "dialogue"
            for r, x in zip(bl.rows, roles):
                members.append(cn_member(rows[r].e, x))
        if us:
            if bl.rows:
                start, end = rows[bl.rows[0]].start, max(rows[r].end for r in bl.rows)
            else:
                start, end = us[0].wstart, us[-1].wend
            members.append(jp_member(us, start, end, role))
        origin = {"by": "aligner", "method": "subkit-align-v2", "block": bl.kind, "doubts": list(bl.doubts)}
        g = add_group("dialogue", members, origin)
        for k, hb in enumerate(pending_hidden):
            _attach_hidden(doc, g, hb, units, ja, settings, before=True, at=k)
        pending_hidden = []
        dialogue_groups.append((doc.group_time(g)[0], g))
    if pending_hidden and dialogue_groups:
        for hb in pending_hidden:
            _attach_hidden(doc, dialogue_groups[-1][1], hb, units, ja, settings, before=False)

    # 2. other translation-file streams: signs, songs, credits, comments (one stream per style)
    other: dict[str, list[tuple[int, Group]]] = {}
    song_orig = [(cn, ev) for ev in cn.events if cn.roles.get(ev.i) == "song_original"]
    song_orig.extend((ja, ev) for ev in ja.events if ja.roles.get(ev.i) == "song_original")
    song_tl = [ev for ev in cn.events if cn.roles.get(ev.i) == "song_translation"]
    used_tl: set[int] = set()
    for song_source, ev in song_orig:
        cancel_check()
        partner = None
        for tl in song_tl:
            if tl.i in used_tl:
                continue
            if song_source is cn and song_family(ev.style or "", ev.start, total) != song_family(tl.style or "", tl.start, total):
                continue
            if song_source is cn and ev.effect.casefold() == "karaoke":
                if abs(ev.start - tl.start) > 10 or abs(ev.end - tl.end) > 10:
                    continue
                partner = tl
                used_tl.add(tl.i)
                break
            ov = min(ev.end, tl.end) - max(ev.start, tl.start)
            if ov > 0.5 * max(1, min(ev.end - ev.start, tl.end - tl.start)):
                partner = tl
                used_tl.add(tl.i)
                break
        fam = song_family((partner.style if partner is not None and song_source is ja else ev.style) or "", ev.start, total)
        ms = []
        if partner is not None:      # song lines keep their source layer (fansub effects stack glow under main)
            m = cn_member(partner.i, "song", layer=partner.layer or 0)
            m.style = st["translation/song"].format(fam=fam)
            ms.append(m)
        mj = Member(new_id("m_"), "", ja.lang, "original", "song", ev.start, ev.end, layer=ev.layer or 0,
                    style=st["original/song"].format(fam=fam), runs=runs_of_event(song_source, ev.i))
        ms.append(mj)
        g = add_group("song", ms, {"by": "import", "source_style": ev.style})
        other.setdefault("song:" + song_source.id + ":" + (ev.style or ""), []).append((ev.start, g))
    for ev in cn.events:
        cancel_check()
        role = cn.roles.get(ev.i)
        if role in ("dialogue", "top", "song_original", "excluded") or (role == "song_translation" and ev.i in used_tl):
            continue
        if role == "song_translation":
            fam = song_family(ev.style or "", ev.start, total)
            m = cn_member(ev.i, "song", layer=ev.layer or 0)
            m.style = st["translation/song"].format(fam=fam)
            g = add_group("song", [m], {"by": "import", "source_style": ev.style})
        else:
            kind = "info" if role == "comment" else "sign"
            m = Member(new_id("m_"), "", cn.lang, "translation", "comment" if role == "comment" else "sign",
                       ev.start, ev.end, layer=ev.layer or 0, style=ev.style or "Default",
                       margins=list(ev.margins or [0, 0, 0]), effect=ev.effect or "", runs=runs_of_event(cn, ev.i),
                       source_kind=ev.kind, speaker={"character": None, "label": ev.name} if ev.name else None)
            m.extra["source_style"] = True
            g = add_group(kind, [m], {"by": "import", "source_style": ev.style})
        other.setdefault(f"{role}:{ev.style or ''}", []).append((ev.start, g))

    streams: list[list[tuple[int, Group]]] = [dialogue_groups] + list(other.values())
    doc.groups = [g for _t, g in heapq.merge(*streams, key=lambda x: x[0])]
    doc.structure_changed()
    return doc


def _attach_hidden(doc: Document, g: Group, bl: Block, units: list[Unit], ja: Source, settings: MergeSettings,
                   *, before: bool, at: int = 0) -> None:
    us = [units[k] for k in bl.units]
    jp = [m for m in doc.members_of(g) if m.lane == "original"]
    if before:
        position = {"before": jp[0].id} if jp else {"only": True}
    else:
        position = {"after": jp[-1].id} if jp else {"only": True}
    c = Comment(new_id("c_"), g.id, "hidden_filler", ja.lang, units_runs(us, ja, settings.jp_sep), position, "tool", True)
    doc.comments[c.id] = c
    if before:
        g.comments.insert(at, c.id)          # several hidden blocks keep their source order
    else:
        g.comments.append(c.id)


def _carry_tombstones(previous: Optional[Document], doc: Document) -> None:
    """Remove deleted source ranges from rebuilt members and hidden comments."""
    if previous is None or not previous.tombstones:
        return
    doc.tombstones = copy.deepcopy(previous.tombstones)
    ranges: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for stone in doc.tombstones:
        ranges.setdefault((stone.src, stone.e), []).append((stone.a, stone.b))

    def kept(runs: list[Run]) -> list[Run]:
        result = []
        for run in runs:
            if (run.src, run.e) not in ranges:
                result.append(run)
                continue
            pieces = [(run.a, run.b)]
            for a, b in ranges.get((run.src, run.e), ()):
                pieces = [(lo, hi) for start, end in pieces
                          for lo, hi in ((start, min(end, a)), (max(start, b), end)) if lo < hi]
            result.extend(run.slice(a - run.a, b - run.a) for a, b in pieces)
        while result and (result[0].kind == "sep" or not result[0].text.strip()):
            result.pop(0)
        while result and (result[-1].kind == "sep" or not result[-1].text.strip()):
            result.pop()
        return result

    for collection in (doc.members, doc.comments):
        for key, obj in list(collection.items()):
            obj.runs = kept(obj.runs)
            if not obj.runs:
                del collection[key]
    for group in doc.groups:
        group.members = [mid for mid in group.members if mid in doc.members]
        group.comments = [cid for cid in group.comments if cid in doc.comments]
        for cid in group.comments:
            position = doc.comments[cid].position
            for edge in ("before", "after"):
                if position.get(edge) not in group.members:
                    position.pop(edge, None)
    doc.groups = [g for g in doc.groups if g.members or g.comments]
    doc.structure_changed()


JOINT_MODES = ("first", "cut", "keep")
_BREAKS = frozenset(" 　、。，,！？!?…‥")


def joint_cuts(text: str, starts: list[int], end: int) -> Optional[list[int]]:
    """Visible cut positions (one fewer than the translation lines) after sentence breaks of the original, chosen so
    the parts' share of characters follows the lines' share of time; None when there are too few breaks."""
    n = len(starts)
    cands = [k for k in range(1, len(text)) if text[k - 1] in _BREAKS and text[k] not in _BREAKS
             and text[:k].strip(" 　") and text[k:].strip(" 　")]
    if n < 2 or len(cands) < n - 1:
        return None
    sung = [0]
    for ch in text:
        sung.append(sung[-1] + (ch not in _BREAKS))
    total, span = max(1, sung[-1]), max(1, end - starts[0])
    target = [(starts[i + 1] - starts[0]) / span for i in range(n - 1)]
    cost = [[float("inf")] * len(cands) for _ in range(n - 1)]
    back = [[-1] * len(cands) for _ in range(n - 1)]
    for i in range(n - 1):
        for j, k in enumerate(cands):
            here = abs(sung[k] / total - target[i])
            if i == 0:
                cost[i][j] = here
                continue
            p = min(range(j), key=lambda p: cost[i - 1][p], default=None)
            if p is not None and cost[i - 1][p] < float("inf"):
                cost[i][j], back[i][j] = cost[i - 1][p] + here, p
    j = min(range(len(cands)), key=lambda j: cost[n - 2][j])
    if cost[n - 2][j] == float("inf"):
        return None
    out = []
    for i in range(n - 2, -1, -1):
        out.append(cands[j])
        j = back[i][j]
    return out[::-1]


def split_joint_groups(doc: Document, mode: str = "first") -> int:
    """One group per translation line where the aligner put several on one original line (see MergeSettings.joint).
    The groups holding the original are marked 边界拿不准 (it needs a look).  Returns the number of groups added."""
    if mode not in JOINT_MODES:
        raise ValueError(mode)
    if mode == "keep":
        return 0
    from .ops import Editor
    from .readings import visible_map

    editor, added = Editor(doc, {}), 0
    for g in [g for g in doc.groups if g.kind == "dialogue"]:
        live = [doc.members[m] for m in g.members if doc.members[m].role != "comment"]
        cn = [m for m in live if m.lane == "translation"]
        jp = [m for m in live if m.lane == "original"]
        if len(cn) < 2 or len(jp) != 1:
            continue
        cuts = joint_cuts(visible_map(jp[0])[0].text, [m.start for m in cn], jp[0].end) if mode == "cut" else None
        origin = dict(g.origin or {})
        made = []
        for i in range(len(cn) - 1, 0, -1):          # from the end: earlier cut positions stay valid
            made.append(editor.split(g.id, jp_at=cuts[i - 1] if cuts else None, cn_member_index=i))
        if not cuts:                                  # the original stays with the first line only
            jp[0].start, jp[0].end = cn[0].start, cn[0].end
        doubts = list(dict.fromkeys([*origin.get("doubts", []), "joint_split", "close_boundary"]))
        for gid in [g.id, *made]:
            group = doc.group(gid)
            has_original = any(doc.members[m].lane == "original" for m in group.members)
            group.origin = {**origin, "block": "joint_split",
                            "doubts": doubts if has_original else [d for d in doubts if d != "close_boundary"]}
        added += len(made)
    editor.history.done.clear()
    return added


def merge_episode(episode: str, cn: Source, ja: Source, settings: Optional[MergeSettings] = None, *,
                  sim_factory=None, log: Callable[[str], None] = lambda s: None,
                  character_book=None, previous: Optional[Document] = None, cancel_check=lambda: None) -> MergeResult:
    settings = settings or MergeSettings()
    lines: list[str] = []

    def _log(s: str) -> None:
        lines.append(s)
        log(s)

    summary = {}
    units, rows, out = align(cn, ja, settings, sim_factory=sim_factory, log=_log, summary=summary, cancel_check=cancel_check)
    doc = build_document(episode, cn, ja, units, rows, out, settings, cancel_check=cancel_check)
    from .people import preserve_user_speakers

    cancel_check()
    _carry_tombstones(previous, doc)
    preserve_user_speakers(previous, doc)
    split_joint_groups(doc, settings.joint)
    if character_book is not None:
        character_book.link_document(doc, sources={cn.id: cn, ja.id: ja})
    summary["groups"] = group_counts(doc)
    return MergeResult(doc, units, rows, out, lines, summary)
