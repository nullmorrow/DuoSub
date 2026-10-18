"""Project passage library and identity-independent, non-mutating suggestions."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..ass.text import VisibleText
from ..audio_match import Match, Template, load_template, save_template
from ..model.document import Annotation, Document, Group, Member, Run
from ..model.ids import new_id
from ..model.serialize import read_json, write_json_atomic
from . import karaoke as kara
from .convert import member_target
from .merge import MergeSettings
from .readings import anchor_of, anchor_positions, pieces_of, visible_map
from .workspace import Project, now

CATEGORIES = ("OP", "ED", "IN", "NA")
CONTINUITY_GAP_MS = 8000

if TYPE_CHECKING:
    from .ops import Editor, Transaction


class PassageStore:
    def __init__(self, project: Project) -> None:
        self.path = project.root / project.data.get("dictionaries", {}).get("passages", "dict/passages.json")
        self.templates = project.root / "dict" / "passages"

    def _read(self) -> dict:
        if not self.path.exists():
            return {"format": "subkit-passages", "format_version": 1, "passages": []}
        data = read_json(self.path)
        if data.get("format") != "subkit-passages" or data.get("format_version") != 1:
            raise ValueError("unsupported passage library format")
        return data

    def list(self) -> list[dict]:
        return copy.deepcopy(self._read()["passages"])

    def get(self, id: str) -> dict:
        for p in self.list():
            if p["id"] == id:
                return p
        raise KeyError(id)

    def _template_path(self, id: str) -> Path:
        if not id.startswith("psg_") or Path(id).name != id or any(c in id for c in "/\\:"):
            raise ValueError("invalid passage id")
        return self.templates / f"{id}.npz"

    def add(self, passage: dict, template: Template | None, replace_id: str | None = None) -> str:
        data = self._read()
        p = copy.deepcopy(passage)
        id = replace_id or p["id"]
        path = self._template_path(id)
        entries = data["passages"]
        old = next((i for i, item in enumerate(entries) if item["id"] == id), None)
        if replace_id is not None and old is None:
            raise KeyError(id)
        if replace_id is None and old is not None:
            raise ValueError("duplicate passage id")
        p["id"] = id
        self.templates.mkdir(parents=True, exist_ok=True)
        if template is not None:
            save_template(path, template)
        else:
            path.unlink(missing_ok=True)
        if old is None:
            entries.append(p)
        else:
            entries[old] = p
        write_json_atomic(self.path, data)
        return id

    def rename(self, id: str, name: str, version: str) -> None:
        data = self._read()
        p = next((p for p in data["passages"] if p["id"] == id), None)
        if p is None:
            raise KeyError(id)
        p.update(name=name, version=version)
        write_json_atomic(self.path, data)

    def remove(self, id: str) -> None:
        data = self._read()
        if not any(p["id"] == id for p in data["passages"]):
            raise KeyError(id)
        path = self._template_path(id)
        data["passages"] = [p for p in data["passages"] if p["id"] != id]
        write_json_atomic(self.path, data)
        path.unlink(missing_ok=True)

    def has_audio(self, id: str) -> bool:
        return self._template_path(id).is_file()

    def template(self, id: str) -> Template:
        self.get(id)
        return load_template(self._template_path(id))


def _range(anchor: list[dict], ids: list) -> tuple[int, int] | None:
    return anchor_positions(Annotation("", "ruby", "", anchor, "", ""), ids)


def _offsets(anchor: list[dict], ids: list) -> list[list[int]] | None:
    """Keep discontinuous core keys as ordered visible ranges."""
    out: list[list[int]] = []
    for segment in anchor:
        pos = _range([segment], ids)
        if pos is None:
            return None
        if out and out[-1][1] == pos[0]:
            out[-1][1] = pos[1]
        else:
            out.append(list(pos))
    return out or None


def _format(m: Member, ids: list) -> list[dict]:
    out = []
    for span in m.format:
        ranges = _offsets(span.get("anchor", []), ids)
        if ranges is not None:
            for a, b in ranges:
                out.append({"a": a, "b": b, **copy.deepcopy({k: v for k, v in span.items() if k != "anchor"})})
    return out


def _annotations(doc: Document, m: Member, ids: list) -> list[dict]:
    out = []
    for ann in doc.annotations_of(m.id):
        pos = anchor_positions(ann, ids)
        if pos is not None:
            out.append({"kind": ann.kind, "a": pos[0], "b": pos[1], "text": ann.text,
                        "parts": copy.deepcopy(pieces_of(ann)), "status": ann.status})
    return out


def _karaoke(doc: Document, m: Member, members: list[Member], start: int, idx: kara.KIndex) -> tuple[dict | None, str]:
    if not idx.has(m):
        return None, "none"
    view = kara.effective(doc, m, idx=idx)
    tracks = kara.tracks_of(doc, m.id)
    effect = copy.deepcopy(tracks[0].get("effect", {"preset": view.kind, "params": {}})) if tracks else {"preset": view.kind, "params": {}}
    ids = visible_map(m)[1]
    edits = []
    for edit in kara.edits_of(doc, m, idx):
        core = _offsets(edit.get("at", {}).get("core", []), ids)
        if core is not None:
            item = copy.deepcopy(edit)
            item["at"]["core"] = core
            edits.append(item)
    # Auto K timing ("suggested", not yet confirmed) is real timing: it is recorded with its state and stays
    # unconfirmed where it is applied (it used to be dropped, leaving the lyrics untimed in the next episode).
    timed = [b for b in view.beats if b.timed and b.t1 is not None and "placeholder" not in b.flags]
    raw = kara.beats_to_track(m, timed, effect=effect)["beats"]
    beats = []
    for beat in raw:
        core = _offsets(beat["core"], ids)
        if core is None:
            continue
        beat["core"] = core
        beat["t0"] -= start
        beat["t1"] -= start
        links = {}
        for edge, ref in beat.pop("link", {}).items():
            for mi, original in enumerate(members):
                if original.lane != "original":
                    continue
                offsets = _offsets(ref.get("core", []), visible_map(original)[1])
                if offsets is not None:
                    links[edge] = {**copy.deepcopy(ref), "core": offsets, "member_index": mi}
                    break
        if links:
            beat["link"] = links
        beats.append(beat)
    summary = "done" if view.beats and len(beats) == len(view.beats) else "partial"
    return {"unit": view.unit, "state": view.state, "effect": effect, "edits": edits, "beats": beats}, summary


def snapshot(doc: Document, gids: Iterable[str], *, name: str, category: str, version: str, episode_id: str) -> dict:
    if category not in CATEGORIES:
        raise ValueError(category)
    selected = set(gids)
    groups = [(g, [doc.members[mid] for mid in g.members
                   if doc.members[mid].role != "comment" and visible_map(doc.members[mid])[0].text.strip()])
              for g in doc.groups if g.id in selected]
    groups = [(g, ms) for g, ms in groups if ms]
    if not groups:
        raise ValueError("先在列表里选中这一段的几组。")
    start = min(m.start for _, ms in groups for m in ms)
    end = max(m.end for _, ms in groups for m in ms)
    summary = {"groups": len(groups), "lines": sum(len(ms) for _, ms in groups), "readings": 0,
               "readings_skipped": 0, "unconfirmed_lines": 0, "karaoke": {"none": 0, "done": 0, "partial": 0}}
    saved = []
    idx = kara.KIndex(doc)
    for group, members in groups:
        stored = []
        for m in members:
            vt, ids = visible_map(m)
            annotations = _annotations(doc, m, ids)
            # A group marked confirmed was reviewed with its furigana on screen: its readings count as confirmed.
            if group.state == "confirmed":
                for a in annotations:
                    if a["status"] not in ("confirmed", "hidden"):
                        a["status"] = "confirmed"
            kept = [a for a in annotations if a["status"] in ("confirmed", "hidden")]
            skipped = len(annotations) - len(kept)
            k, state = _karaoke(doc, m, members, start, idx)
            summary["readings"] += len(kept)
            summary["readings_skipped"] += skipped
            summary["unconfirmed_lines"] += bool(skipped)
            summary["karaoke"][state] += 1
            stored.append({"lane": m.lane, "lang": m.lang, "role": m.role, "layer": m.layer,
                           "style": m.style, "margins": list(m.margins), "start": m.start - start,
                           "end": m.end - start, "text": "".join(r.text for r in m.runs if r.kind != "tag"),
                           "visible": vt.text, "inline_tags": "".join(r.text for r in m.runs if r.kind == "tag"),
                           "overrides": copy.deepcopy(m.overrides), "format": _format(m, ids),
                           "annotations": kept, "karaoke": k})
        saved.append({"kind": group.kind, "members": stored})
    return {"id": new_id("psg_"), "name": name, "category": category, "version": version,
            "episode_id": episode_id, "created_at": now(), "start": start, "duration": end - start,
            "groups": saved, "summary": summary}


@dataclass
class PlanAction:
    id: str
    kind: str
    label: str
    checked: bool
    target_member_id: str | None
    target_group_id: str | None
    data: dict[str, Any]
    conflict: str = ""


@dataclass
class PlanLine:
    group_index: int
    member_index: int
    text: str
    target_member_id: str | None
    target_group_id: str | None
    status: str
    start: int
    end: int
    actions: list[PlanAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    readings_differ: bool = False
    coverage: str = "unknown"


@dataclass
class PassagePlan:
    id: str
    name: str
    category: str
    version: str
    at_ms: int
    score: float
    covered: tuple[int, int]
    drift_ms: int
    scale: float
    lines: list[PlanLine] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    already_applied: bool = False
    styles: dict[str, str] = field(default_factory=lambda: MergeSettings().styles)


def already_applied(doc: Document, passage_id: str, at_ms: int) -> bool:
    return any(g.extra.get("passage", {}).get("id") == passage_id
               and abs(g.extra["passage"].get("at", -1000000000) - at_ms) <= 2000 for g in doc.groups)


def _time(ms: int) -> str:
    sign = "-" if ms < 0 else ""
    ms = abs(ms)
    return f"{sign}{ms // 3600000}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}.{ms % 1000:03}"


def _same_reading(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in ("kind", "a", "b", "text", "status", "parts"))


def _actions(doc: Document, passage: dict, line: PlanLine, stored: dict, target: Member,
             factor: float, at_ms: int, available: set[str] | None, target_members: dict[int, str | None],
             styles: Mapping[str, str]) -> None:
    def add(kind: str, label: str, data: dict, conflict: str = "", checked: bool = True) -> None:
        line.actions.append(PlanAction(f"{passage['id']}:{line.group_index}:{line.member_index}:{kind}",
                            kind, label, checked and not bool(conflict), target.id, target.group,
                            copy.deepcopy(data), conflict))

    if abs(target.start - line.start) > 10 or abs(target.end - line.end) > 10:
        add("time", f"改时间 {_time(target.start)}–{_time(target.end)} → {_time(line.start)}–{_time(line.end)}（{line.start - target.start:+d}毫秒）",
            {"start": line.start, "end": line.end}, checked=line.status == "matched")
    if line.status != "matched":
        line.notes.append(f"文字和库里不同：库里「{stored['visible']}」")
        return
    ids = visible_map(target)[1]
    current = _annotations(doc, target, ids)
    safe, conflicts, labels = [], [], []
    for ann in stored["annotations"]:
        overlapping = [a for a in current if a["kind"] == ann["kind"] and a["a"] < ann["b"] and ann["a"] < a["b"]]
        protected = [a for a in overlapping if a["status"] in ("confirmed", "hidden")]
        if any(_same_reading(a, ann) for a in protected) and all(_same_reading(a, ann) for a in protected):
            continue
        bad = [a for a in protected if not _same_reading(a, ann)]
        (conflicts if bad else safe).append(ann)
        for old in bad:
            labels.append(f"会覆盖你确认过的读音：{stored['visible'][ann['a']:ann['b']]}({old['text']}) → ({ann['text']})")
    for kind, anns, conflict in (("readings", safe, ""), ("readings_conflict", conflicts, "\n".join(labels))):
        if anns:
            examples = " …".join(f"{stored['visible'][a['a']:a['b']]}({a['text']})" for a in anns[:3])
            if len(anns) > 3:
                examples += "等"
            add(kind, f"补读音 {len(anns)} 处：{examples}", {"annotations": anns}, conflict)
    k = stored.get("karaoke")
    if k and k["beats"]:
        protected_beats = any(b.get("by") in ("tap", "drag", "manual") for t in kara.tracks_of(doc, target.id) for b in t.get("beats", []))
        payload = copy.deepcopy(k)
        for beat in payload["beats"]:
            beat["t0"] = _at(beat["t0"], factor, at_ms)
            beat["t1"] = _at(beat["t1"], factor, at_ms)
        payload["group_members"] = passage["groups"][line.group_index]["members"]
        payload["target_members"] = target_members
        line.readings_differ = any(not any(_same_reading(a, old) for old in current) for a in stored["annotations"])
        payload["readings_differ"] = line.readings_differ
        add("karaoke", f"套用 K 轴（{len(k['beats'])} 拍）", payload, "会覆盖你打过的 K 轴" if protected_beats else "")
    group = next(g for g in doc.groups if g.id == target.group)
    final_style = target.style
    if passage["groups"][line.group_index]["kind"] == "song" and group.kind == "dialogue":
        family = passage["category"]
        converted = member_target(target, "song", styles, family)
        if converted is not None:
            final_style = converted[1]
        add("convert", f"转为歌曲（{family}）", {"family": family})
    if stored["style"] != final_style and (available is None or stored["style"] in available):
        add("style", f"样式 {final_style} → {stored['style']}", {"style": stored["style"]})
    if stored["overrides"] != target.overrides or stored["format"] != _format(target, ids):
        add("position", "位置和整行格式", {"overrides": stored["overrides"], "format": stored["format"]},
            "会覆盖你设的位置" if target.overrides or target.format else "")


def _at(relative: int, factor: float, at_ms: int) -> int:
    """Round once at the passage origin (ties to even, like existing K-axis links)."""
    return at_ms + round(relative * factor)


def _coverage(start: int, end: int, intervals: tuple[tuple[int, int], ...]) -> str:
    """Use the least certain coverage touched by the whole template-relative line."""
    if any(a <= start <= end <= b for a, b in intervals):
        return "direct"
    runs = []
    for a, b in intervals:
        if runs and a - runs[-1][1] <= CONTINUITY_GAP_MS:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return "inferred" if any(a <= start <= end <= b for a, b in runs) else "unknown"


def plan(doc: Document, passage: dict, match: Match, *, scale: bool = False,
         template_styles: Iterable[str] | None = None, styles: Mapping[str, str] | None = None) -> PassagePlan:
    styles = MergeSettings().styles if styles is None else styles
    factor = match.scale if scale else 1.0
    result = PassagePlan(passage["id"], passage["name"], passage["category"], passage["version"],
                         match.start_ms, match.score, tuple(match.covered), match.drift_ms, factor,
                         already_applied=already_applied(doc, passage["id"], match.start_ms), styles=dict(styles))
    intervals = match.covered_intervals if match.covered_intervals is not None else (match.covered,)
    available = None if template_styles is None else set(template_styles)
    candidates = [doc.members[mid] for g in doc.groups for mid in g.members if doc.members[mid].role != "comment"]
    visible = {m.id: visible_map(m)[0].text for m in candidates}
    used: set[str] = set()
    stored_lines = []
    # Resolve all exact siblings before deciding where to insert missing members.
    for gi, group in enumerate(passage["groups"]):
        options = []
        for stored in group["members"]:
            start = _at(stored["start"], factor, match.start_ms)
            exact = [m for m in candidates if m.id not in used and m.lane == stored["lane"]
                     and visible[m.id] == stored["visible"] and abs(m.start - start) <= 3000]
            options.append((start, exact))
        # The lines of one stored group belong together: prefer the target group that holds most of them, so an
        # identical caption of the same lyric elsewhere (a Japanese CC line beside the fansub's EDJP+EDTC pair)
        # does not take the original away from its translation.
        hits = Counter(gid for _start, exact in options for gid in {m.group for m in exact})
        home = None
        if len(options) > 1 and hits and max(hits.values()) > 1:
            top = max(hits.values())
            home = min((gid for gid, n in hits.items() if n == top),
                       key=lambda gid: sum(min((abs(m.start - s) for m in exact if m.group == gid), default=0)
                                           for s, exact in options))
        for mi, stored in enumerate(group["members"]):
            start = _at(stored["start"], factor, match.start_ms)
            end = _at(stored["end"], factor, match.start_ms)
            coverage = _coverage(stored["start"], stored["end"], intervals)
            exact = [m for m in options[mi][1] if m.id not in used]
            if home is not None and any(m.group == home for m in exact):
                exact = [m for m in exact if m.group == home]
            target = min(exact, key=lambda m: abs(m.start - start), default=None)
            if target is not None:
                twins = [m for m in exact if abs(m.start - target.start) <= 20 and abs(m.end - target.end) <= 20]
                target = min(twins, key=lambda m: (m.layer != stored["layer"], abs(m.start - start)))
            if target:
                used.add(target.id)
            line = PlanLine(gi, mi, stored["text"], target.id if target else None, target.group if target else None,
                            "matched" if target else "missing_group", start, end, coverage=coverage)
            result.lines.append(line)
            stored_lines.append(stored)
    for line, stored in zip(result.lines, stored_lines):
        if line.status == "matched":
            continue
        overlapping = [m for m in candidates if m.id not in used and m.lane == stored["lane"]
                       and visible[m.id].strip() and line.end > line.start
                       and min(m.end, line.end) - max(m.start, line.start) >= (line.end - line.start) * 0.5]
        target = min(overlapping, key=lambda m: (abs(m.start - line.start), m.layer != stored["layer"]), default=None)
        if target:
            used.add(target.id)
            line.target_member_id, line.target_group_id, line.status = target.id, target.group, "text_differs"
    for line, stored in zip(result.lines, stored_lines):
        if line.target_member_id:
            targets = {l.member_index: l.target_member_id for l in result.lines if l.group_index == line.group_index and l.status == "matched"}
            _actions(doc, passage, line, stored, doc.members[line.target_member_id], factor, match.start_ms, available, targets, styles)
        else:
            siblings = [l for l in result.lines if l.group_index == line.group_index and l.status == "matched"]
            line.target_group_id = siblings[0].target_group_id if siblings else None
            line.status = "missing_member" if siblings else "missing_group"
            kind = "insert_member" if siblings else "insert_group"
            data = {"member": copy.deepcopy(stored), "start": line.start, "end": line.end,
                    "group_index": line.group_index, "member_index": line.member_index,
                    "kind": passage["groups"][line.group_index]["kind"], "at_ms": match.start_ms, "scale": factor}
            data["group_members"] = copy.deepcopy(passage["groups"][line.group_index]["members"])
            data["target_members"] = {l.member_index: l.target_member_id for l in siblings}
            # Keep the shared safe action; unknown lines require their own opt-in.
            checked = line.coverage != "unknown"
            if not siblings:
                data["members"] = [{"member": copy.deepcopy(s), "start": l.start, "end": l.end, "member_index": l.member_index}
                                   for l, s in zip(result.lines, stored_lines) if l.group_index == line.group_index
                                   and (l is line if not checked else l.coverage != "unknown")]
            aid = f"{passage['id']}:{line.group_index}:{line.member_index if siblings or not checked else 'group'}:{kind}"
            line.actions.append(PlanAction(aid, kind, "补上这一行" if siblings else "补上这一组", checked,
                                           None, line.target_group_id, data))
    unknown = sum(l.coverage == "unknown" and l.target_member_id is None for l in result.lines)
    a = _at(match.covered[0], factor, match.start_ms)
    b = _at(match.covered[1], factor, match.start_ms)
    extra = sum(1 for m in candidates if m.id not in used and visible[m.id].strip() and m.start < b and m.end > a)
    if extra:
        result.info.append(f"这一段里还有 {extra} 行库里没有，不会改动。")
    if unknown:
        result.info.append(f"覆盖未知：{unknown} 行默认不勾选，可手动勾选。")
    return result


@dataclass
class ApplySummary:
    time_lines: int = 0
    readings: int = 0
    karaoke_lines: int = 0
    inserted_lines: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)

    def skip(self, action_id: str, reason: str) -> None:
        item = {"id": action_id, "reason": reason}
        if item not in self.skipped:          # once per line, not once per beat edge
            self.skipped.append(item)


def _anchor(ranges: list[list[int]], ids: list) -> list[dict]:
    selected = []
    for a, b in ranges:
        if not 0 <= a < b <= len(ids) or any(i is None for i in ids[a:b]):
            raise ValueError("invalid passage visible offsets")
        selected.extend(ids[a:b])
    return anchor_of(selected)


def _apply_readings(tx: Transaction, m: Member, annotations: list[dict], summary: ApplySummary,
                    action_id: str, *, force: bool = False) -> None:
    vt, ids = visible_map(m)
    for saved in annotations:
        anchor = _anchor([[saved["a"], saved["b"]]], ids)
        overlapping = []
        for ann in tx.doc.annotations_of(m.id):
            pos = anchor_positions(ann, ids)
            if ann.kind == saved["kind"] and pos and pos[0] < saved["b"] and saved["a"] < pos[1]:
                overlapping.append(ann)
        protected = [a for a in overlapping if a.status in ("confirmed", "hidden")]
        current = _annotations(tx.doc, m, ids)
        if protected and not force:
            protected_data = [a for a in current if a["status"] in ("confirmed", "hidden")
                              and a["kind"] == saved["kind"] and a["a"] < saved["b"] and saved["a"] < a["b"]]
            if any(not _same_reading(saved, a) for a in protected_data):
                summary.skip(action_id, "confirmed reading changed since planning")
            continue
        for ann in overlapping:
            tx.annotation(ann.id)
            del tx.doc.annotations[ann.id]
        tx.add_annotation(Annotation(new_id("a_"), saved["kind"], m.id, anchor,
                          vt.text[saved["a"]:saved["b"]], saved["text"], status=saved["status"],
                          origin="passage", parts=copy.deepcopy(saved["parts"])))
        summary.readings += 1


def _prepare_karaoke(doc: Document, m: Member, saved: dict) -> tuple[list, list[dict]] | None:
    ids = visible_map(m)[1]
    edits = copy.deepcopy(saved.get("edits", []))
    for edit in edits:
        edit["at"]["core"] = _anchor(edit["at"]["core"], ids)
    derived = kara.derive(doc, m, saved["unit"], edits)
    if saved.get("state") in ("timed", "suggested") and [b.sung for b in derived] != [b["sung"] for b in saved["beats"]]:
        return None
    keys = {b.key: (i, b) for i, b in enumerate(derived)}
    selected, previous = [], -1
    # Partial recordings contain only timed beats; all stored keys must survive in order.
    for old in saved["beats"]:
        core = kara._ids_of_anchor(_anchor(old["core"], ids))
        found = keys.get((core, old["k"], old["sung"]))
        if found is None or found[0] <= previous:
            return None
        previous, beat = found
        beat.t0, beat.t1, beat.by = old["t0"], old["t1"], "memory"
        selected.append(beat)
    return selected, edits


def _apply_position(tx: Transaction, m: Member, saved: dict) -> None:
    ids = visible_map(m)[1]
    m = tx.member(m.id)
    m.overrides = copy.deepcopy(saved["overrides"])
    m.format = [{"anchor": _anchor([[span["a"], span["b"]]], ids),
                 **copy.deepcopy({k: v for k, v in span.items() if k not in ("a", "b")})}
                for span in saved["format"]]


def _shift_beats(tx: Transaction, m: Member, delta: int) -> None:
    mine = {i for i, *_ in m.identities() if i is not None}
    for kid, track in list(tx.doc.karaoke.items()):
        if not any(set(kara._ids_of_anchor(b["core"])) & mine for b in track.get("beats", [])):
            continue
        track = tx.track(kid)
        for beat in track["beats"]:
            if set(kara._ids_of_anchor(beat["core"])) & mine:
                beat["t0"] += delta
                beat["t1"] += delta


def apply(editor: Editor, plan: PassagePlan, chosen_ids: Iterable[str], *,
          styles: Mapping[str, str] | None = None) -> ApplySummary:
    from ..model.validate import validate

    doc = editor.doc
    chosen = set(chosen_ids)
    selected: dict[str, tuple[PlanLine, PlanAction]] = {}
    for line in plan.lines:
        for action in line.actions:
            if action.id in chosen:
                selected.setdefault(action.id, (line, action))
    summary = ApplySummary()
    if not selected:
        return summary
    styles = plan.styles if styles is None else styles

    def errors() -> Counter:
        return Counter((i.code, i.message, repr(i.where)) for i in validate(doc, editor.sources).errors)

    before = errors()

    def run(tx: Transaction) -> ApplySummary:
        touched: set[str] = set()
        targets = {(l.group_index, l.member_index): l.target_member_id for l in plan.lines if l.status == "matched"}
        active = []
        for line, action in selected.values():
            if action.target_member_id:
                m = doc.members.get(action.target_member_id)
                if m is None or m.group != action.target_group_id:
                    summary.skip(action.id, "target no longer exists in its planned group")
                    continue
                if action.kind != "time" and visible_map(m)[0].text != VisibleText(line.text).text:
                    summary.skip(action.id, "target text changed since planning")
                    continue
            active.append((line, action))

        converted = set()
        for line, action in active:
            if action.kind == "convert" and action.target_group_id not in converted:
                gid = action.target_group_id
                editor._convert_transaction(tx, [gid], "song", styles, action.data["family"])
                converted.add(gid)
                touched.add(gid)
        for line, action in active:
            if action.kind in ("readings", "readings_conflict"):
                _apply_readings(tx, doc.members[action.target_member_id], action.data["annotations"],
                                summary, action.id, force=action.kind == "readings_conflict")
                touched.add(action.target_group_id)

        prepared = {}
        for line, action in active:
            if action.kind != "karaoke":
                continue
            m = doc.members[action.target_member_id]
            if not action.conflict and any(b.get("by") in ("tap", "drag", "manual")
                    for t in kara.tracks_of(doc, m.id) for b in t.get("beats", [])):
                summary.skip(action.id, "manual karaoke changed since planning")
                continue
            result = _prepare_karaoke(doc, m, action.data)
            if result is None:
                summary.skip(action.id, "sung sequence differs after applying readings")
            else:
                prepared[action.id] = result
        replaced = {a.target_member_id for _, a in active if a.id in prepared}
        for line, action in active:
            if action.kind == "time":
                m = tx.member(action.target_member_id)
                delta = action.data["start"] - m.start
                if m.id not in replaced:
                    _shift_beats(tx, m, delta)
                m.start, m.end = action.data["start"], action.data["end"]
                summary.time_lines += 1
                touched.add(m.group)

        pending_links = []

        def write_karaoke(m: Member, saved: dict, gi: int, action_id: str, prepared=None) -> None:
            result = prepared if prepared is not None else _prepare_karaoke(doc, m, saved)
            if result is None:
                summary.skip(action_id, "sung sequence differs after applying readings")
                return
            beats, edits = result
            old = kara.tracks_of(doc, m.id)
            state = "suggested" if saved.get("state") == "suggested" else "timed"
            track = kara.beats_to_track(m, beats, kid=old[0]["id"] if old else None,
                                       unit=saved["unit"], state=state, effect=copy.deepcopy(saved["effect"]), edits=edits)
            for other in old[1:]:
                tx.remove_track(other["id"])
            tx.set_track(track)
            pending_links.append((track, saved, gi, action_id))
            summary.karaoke_lines += 1
            touched.add(m.group)

        for line, action in active:
            if action.id in prepared:
                write_karaoke(doc.members[action.target_member_id], action.data, line.group_index,
                              action.id, prepared[action.id])
        for line, action in active:
            if action.kind == "style":
                tx.member(action.target_member_id).style = action.data["style"]
                touched.add(action.target_group_id)
        for line, action in active:
            if action.kind == "position":
                m = doc.members[action.target_member_id]
                if not action.conflict and (m.overrides or m.format):
                    summary.skip(action.id, "position changed since planning")
                    continue
                _apply_position(tx, m, action.data)
                touched.add(m.group)

        def insert(g: Group, item: dict, gi: int, action_id: str) -> None:
            saved = item["member"]
            text = saved["text"]
            runs = [Run(text, ins=new_id("i_"), b=len(text))]
            tags = saved.get("inline_tags", "")
            if tags:
                runs.insert(0, Run(tags, "tag", ins=new_id("i_"), b=len(tags)))
            m = Member(new_id("m_"), g.id, saved["lang"], saved["lane"], saved["role"],
                       item["start"], item["end"], layer=saved["layer"], style=saved["style"],
                       margins=copy.deepcopy(saved["margins"]), runs=runs)
            tx.add_member(m)
            g = tx.group(g.id)
            index = next((i for i, mid in enumerate(g.members) if m.lane == "translation"
                          and doc.members[mid].lane == "original"), len(g.members))
            g.members.insert(index, m.id)
            targets[gi, item["member_index"]] = m.id
            _apply_readings(tx, m, saved["annotations"], summary, action_id)
            _apply_position(tx, m, saved)
            k = copy.deepcopy(saved.get("karaoke"))
            if k and k["beats"]:
                for b in k["beats"]:
                    b["t0"], b["t1"] = _at(b["t0"], plan.scale, plan.at_ms), _at(b["t1"], plan.scale, plan.at_ms)
                write_karaoke(m, k, gi, action_id)
            summary.inserted_lines += 1
            touched.add(g.id)

        inserted_groups = {}
        for line, action in active:
            if action.kind == "insert_member":
                g = tx.group(action.target_group_id)
                if g is None:
                    summary.skip(action.id, "insertion group no longer exists")
                    continue
                insert(g, action.data, line.group_index, action.id)
            elif action.kind == "insert_group":
                g = inserted_groups.get(line.group_index)
                if g is None:
                    gid = new_id("g_")
                    tx.group(gid)
                    order = tx.order()
                    g = Group(gid, action.data["kind"], origin={"by": "passage", "passage": plan.id})
                    start = min(item["start"] for item in action.data["members"])
                    index = next((i for i, other in enumerate(order) if doc.group_time(other)[0] > start), len(order))
                    order.insert(index, g)
                    inserted_groups[line.group_index] = g
                    doc.structure_changed()
                for item in action.data["members"]:
                    insert(g, item, line.group_index, action.id)

        for gi, g in inserted_groups.items():
            members = [mid for (group_index, _), mid in sorted(targets.items())
                       if group_index == gi and mid is not None]
            g.members[:] = sorted(members, key=lambda mid: doc.members[mid].lane != "translation")

        for track, saved, gi, action_id in pending_links:
            m = doc.members[track["member"]]
            for beat, old in zip(track["beats"], saved["beats"]):
                for edge, ref in old.get("link", {}).items():
                    original = doc.members.get(targets.get((gi, ref["member_index"])))
                    if original is None or original.lane != "original" or original.group != m.group:
                        summary.skip(action_id, "karaoke link target is absent from the target group")
                        continue
                    link = copy.deepcopy(ref)
                    link.pop("member_index")
                    link["core"] = _anchor(link["core"], visible_map(original)[1])
                    beat.setdefault("link", {})[edge] = link
        marker = {"id": plan.id, "at": plan.at_ms, "score": plan.score, "applied_at": now()}
        for gid in touched:
            tx.group(gid).extra["passage"] = copy.deepcopy(marker)
        tx.keeps_text = summary.inserted_lines == 0
        if errors() - before:
            raise ValueError("passage application introduced document validation errors")
        return summary

    return editor.history.run(f"套用重复段落《{plan.name}》", run)
