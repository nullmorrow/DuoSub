"""Validators for the red lines (FORMAT-v1 §7, §8 and reference integrity).

* :func:`check_integrity` — unique IDs, every reference resolves, every member is
  in exactly one group, source runs reproduce the source text exactly, annotation
  anchors point at characters that exist in their member.
* :func:`check_order`     — for every source, the source runs read in document
  order are non-decreasing in (event, offset).  Unpaired members and hidden
  comments are part of the sequence; tombstones are not.
* :func:`ledger`          — every non-whitespace character of every source event
  is accounted for exactly once (paired / unpaired / comment / metadata /
  deleted); missing or duplicated characters are errors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .document import Comment, Document, Group, Member, Run, SEP_TEXTS
from .source import Source

LEDGER_CATEGORIES = ("paired", "unpaired", "comment", "metadata", "deleted")


@dataclass
class Issue:
    level: str          # error | warning | info
    code: str
    message: str
    where: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    ledger: dict[str, dict[str, int]] = field(default_factory=dict)       # source → category → count
    order_exceptions: list[str] = field(default_factory=list)              # accepted (flagged) groups

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, code: str, message: str, **where: Any) -> None:
        self.issues.append(Issue(level, code, message, where))

    def summary(self) -> str:
        c = Counter(i.code for i in self.errors)
        return "ok" if not c else ", ".join(f"{k}×{v}" for k, v in sorted(c.items()))


def validate(doc: Document, sources: dict[str, Source]) -> Report:
    from ..gloss import parse_glosses

    rep = Report()
    check_integrity(doc, sources, rep)
    check_order(doc, sources, rep)
    ledger(doc, sources, rep)
    for member in doc.members.values():
        warnings = []
        parse_glosses(member.text, warnings)
        for warning in warnings:
            rep.add("warning", "gloss.syntax", warning, member=member.id, group=member.group)
    return rep


# ------------------------------------------------------------------ integrity
def _check_run(run: Run, sources: dict[str, Source], rep: Report, where: dict[str, Any]) -> None:
    if run.kind == "sep":
        if run.src is not None or run.ins is not None:
            rep.add("error", "run.sep_provenance", "sep run must not carry provenance", **where)
        if run.text not in SEP_TEXTS and run.text.strip(" 　") != "":
            rep.add("error", "run.sep_text", f"sep run text must be whitespace or a break, got {run.text!r}", **where)
        return
    if run.src is not None:
        src = sources.get(run.src)
        if src is None:
            rep.add("error", "run.unknown_source", f"unknown source {run.src}", **where)
            return
        if run.e is None or not (0 <= run.e < len(src.events)):
            rep.add("error", "run.bad_event", f"bad event {run.src}:{run.e}", **where)
            return
        text = src.events[run.e].text
        if not (0 <= run.a <= run.b <= len(text)):
            rep.add("error", "run.bad_range", f"range {run.a}-{run.b} outside event {run.src}:{run.e}", **where)
            return
        if text[run.a:run.b] != run.text:
            rep.add("error", "run.text_mismatch",
                    f"run text {run.text!r} != source {text[run.a:run.b]!r} ({run.src}:{run.e} {run.a}-{run.b})", **where)
        return
    if run.ins is not None:
        if run.b - run.a != len(run.text):
            rep.add("error", "run.ins_length", "insertion run offsets do not match its text", **where)
        return
    rep.add("error", "run.no_provenance", f"text run without provenance: {run.text!r}", **where)


def check_integrity(doc: Document, sources: dict[str, Source], rep: Report) -> None:
    seen: set[str] = set()
    for g in doc.groups:
        if g.id in seen:
            rep.add("error", "id.duplicate", f"duplicate group id {g.id}", group=g.id)
        seen.add(g.id)
    owner: dict[str, str] = {}
    for g in doc.groups:
        for mid in g.members:
            if mid not in doc.members:
                rep.add("error", "ref.member", f"group {g.id} references missing member {mid}", group=g.id)
                continue
            if mid in owner:
                rep.add("error", "ref.member_twice", f"member {mid} in groups {owner[mid]} and {g.id}", group=g.id)
            owner[mid] = g.id
            if doc.members[mid].group != g.id:
                rep.add("error", "ref.member_group", f"member {mid}.group={doc.members[mid].group} but listed in {g.id}",
                        group=g.id)
        for cid in g.comments:
            c = doc.comments.get(cid)
            if c is None:
                rep.add("error", "ref.comment", f"group {g.id} references missing comment {cid}", group=g.id)
            elif c.group != g.id:
                rep.add("error", "ref.comment_group", f"comment {cid}.group={c.group} but listed in {g.id}", group=g.id)
            else:
                pos = c.position or {}
                anchor = pos.get("before") or pos.get("after")
                if anchor and anchor not in g.members:
                    rep.add("warning", "ref.comment_position", f"comment {cid} display position points outside its group",
                            group=g.id)
    for mid, m in doc.members.items():
        if mid not in owner:
            rep.add("error", "ref.orphan_member", f"member {mid} is in no group", member=mid)
        if m.end < m.start:
            rep.add("error", "time.negative", f"member {mid} ends before it starts", member=mid)
        for k, r in enumerate(m.runs):
            _check_run(r, sources, rep, {"member": mid, "run": k})
    listed = {cid for g in doc.groups for cid in g.comments}
    for cid, c in doc.comments.items():
        if cid not in listed:
            rep.add("error", "ref.orphan_comment", f"comment {cid} is in no group", comment=cid)
        for k, r in enumerate(c.runs):
            _check_run(r, sources, rep, {"comment": cid, "run": k})
    for aid, a in doc.annotations.items():
        m = doc.members.get(a.member)
        if m is None:
            rep.add("error", "ref.annotation", f"annotation {aid} on missing member {a.member}", annotation=aid)
            continue
        have = {ident for ident, _ch, _ri, _k in m.identities() if ident is not None}
        for seg in a.anchor:
            if "src" in seg:
                need = [("s", seg["src"], seg["e"], o) for o in range(seg["a"], seg["b"])]
            else:
                need = [("i", seg["ins"], o) for o in range(seg["a"], seg["b"])]
            if not need or any(n not in have for n in need):
                rep.add("warning", "annotation.detached",
                        f"annotation {aid} ({a.base}→{a.text}) points at characters no longer in member {a.member}",
                        annotation=aid, member=a.member)
                break


# ------------------------------------------------------------------ order
def comment_key(c: Comment) -> Optional[tuple[str, int, int]]:
    """Source position (src, event, offset) of a comment's first source character."""
    for r in c.runs:
        if r.src is not None and r.kind != "sep":
            return (r.src, r.e, r.a)
    return None


def group_sequence(doc: Document, g: Group) -> list[tuple[str, Any]]:
    """Members and comments of a group in reading order: ('m', Member) / ('c', Comment).

    Hidden text (comments) is placed by its *source position* among the members' text of the
    same source (FORMAT-v1 §6.4): it is where the words were spoken, whatever editing did to the
    members around it.  Comments without source text fall back to their recorded ``position``."""
    members = [doc.members[m] for m in g.members if m in doc.members]
    comments = [doc.comments[c] for c in g.comments if c in doc.comments]
    out: list[tuple[str, Any]] = [("m", m) for m in members]
    for c in comments:
        key = comment_key(c)
        idx = None
        if key is not None:
            for k, (kind, obj) in enumerate(out):
                if kind != "m":
                    continue
                first = next(((r.src, r.e, r.a) for r in obj.runs if r.src == key[0] and r.kind != "sep"), None)
                if first is not None and first > key:
                    idx = k
                    break
        if idx is None:
            pos = c.position or {}
            if key is None and pos.get("before") in g.members:
                idx = next(k for k, (kind, obj) in enumerate(out) if kind == "m" and obj.id == pos["before"])
            else:
                idx = len(out)
        out.insert(idx, ("c", c))
    return out


def group_runs(doc: Document, g: Group, src_id: str) -> list[tuple[int, int, str]]:
    """(event, offset, owner) of the group's runs from ``src_id`` in reading order: members' runs as
    they are, hidden comments interleaved at their source position (run level, so a comment can sit
    inside a member whose text spans several captions)."""
    flat: list[tuple[int, int, str]] = []
    for mid in g.members:
        m = doc.members.get(mid)
        if m is None:
            continue
        flat.extend((r.e, r.a, m.id) for r in m.runs if r.src == src_id and r.kind != "sep")
    for cid in g.comments:
        c = doc.comments.get(cid)
        if c is None:
            continue
        cr = [(r.e, r.a, c.id) for r in c.runs if r.src == src_id and r.kind != "sep"]
        if not cr:
            continue
        k = next((i for i, x in enumerate(flat) if (x[0], x[1]) > (cr[0][0], cr[0][1])), len(flat))
        flat[k:k] = cr
    return flat


def source_positions(doc: Document, src_id: str) -> Iterable[tuple[int, int, int, str]]:
    """(group index, event, offset, owner id) of every source run of ``src_id`` in document order."""
    for gi, g in enumerate(doc.groups):
        for e, a, owner in group_runs(doc, g, src_id):
            yield gi, e, a, owner


def stream_of(src: Source, e: int) -> str:
    """Order is checked per stream (FORMAT-v1 §7): one stream per (style, role) for ASS/SSA sources
    (fansub files keep all OP lyrics in a block at the end and slip positioned sign lines into the
    dialogue style out of time order), the whole file for SRT."""
    if src.format in ("ass", "ssa"):
        return f"{src.events[e].style or ''}/{src.roles.get(e, '')}"
    return ""


def check_order(doc: Document, sources: dict[str, Source], rep: Report) -> None:
    for sid, src in sources.items():
        if src.lane == "reference":
            continue
        last: dict[str, tuple[int, int, int, str]] = {}
        for gi, e, a, owner in source_positions(doc, sid):
            if not (0 <= e < len(src.events)):
                continue
            st = stream_of(src, e)
            prev = last.get(st)
            if prev is not None and (e, a) < (prev[1], prev[2]):
                g_now, g_prev = doc.groups[gi], doc.groups[prev[0]]
                if "order_exception" in g_now.flags or "order_exception" in g_prev.flags:
                    for gid in (g_prev.id, g_now.id):
                        if gid not in rep.order_exceptions:
                            rep.order_exceptions.append(gid)
                else:
                    label = f"{sid}[{st}]" if st else sid
                    rep.add("error", "order",
                            f"{label}: {sid}:{e}@{a} (group #{gi + 1}) comes after {sid}:{prev[1]}@{prev[2]} (group #{prev[0] + 1})",
                            group=g_now.id, previous_group=g_prev.id, source=sid, stream=st)
            if prev is None or (e, a) >= (prev[1], prev[2]):
                last[st] = (gi, e, a, owner)


# ------------------------------------------------------------------ ledger
def ledger(doc: Document, sources: dict[str, Source], rep: Report) -> None:
    cover: dict[tuple[str, int], list[list[str]]] = {}

    def slot(sid: str, e: int) -> Optional[list[list[str]]]:
        src = sources.get(sid)
        if src is None or not (0 <= e < len(src.events)):
            return None
        key = (sid, e)
        if key not in cover:
            cover[key] = [[] for _ in src.events[e].text]
        return cover[key]

    def mark(sid: str, e: int, a: int, b: int, cat: str) -> None:
        s = slot(sid, e)
        if s is None:
            return
        for k in range(max(0, a), min(b, len(s))):
            s[k].append(cat)

    for g in doc.groups:
        paired = len(doc.group_langs(g)) >= 2
        for mid in g.members:
            m = doc.members.get(mid)
            if m is None:
                continue
            for r in m.runs:
                if r.src is not None and r.kind != "sep":
                    mark(r.src, r.e, r.a, r.b, "paired" if paired else "unpaired")
        for cid in g.comments:
            c = doc.comments.get(cid)
            if c is None:
                continue
            for r in c.runs:
                if r.src is not None and r.kind != "sep":
                    mark(r.src, r.e, r.a, r.b, "comment")
    as_text = {(r.get("src"), r.get("e"), r.get("a"), r.get("b")) for r in doc.rules if r.get("kind") == "span_text"}
    for sid, src in sources.items():
        for sp in src.spans:
            if (sid, sp.e, sp.a, sp.b) in as_text:
                continue                 # the user put this bracket back into the text: a member covers it
            mark(sid, sp.e, sp.a, sp.b, "metadata")
    for t in doc.tombstones:
        mark(t.src, t.e, t.a, t.b, "deleted")

    for sid, src in sources.items():
        if src.lane == "reference":
            continue
        counts = {c: 0 for c in LEDGER_CATEGORIES} | {"missing": 0, "duplicate": 0}
        missing_at: list[tuple[int, int]] = []
        dup_at: list[tuple[int, int]] = []
        for e, ev in enumerate(src.events):
            s = cover.get((sid, e))
            for k, ch in enumerate(ev.text):
                if ch.isspace():
                    continue
                cats = s[k] if s else []
                if not cats:
                    counts["missing"] += 1
                    missing_at.append((e, k))
                elif len(cats) > 1:
                    counts["duplicate"] += 1
                    dup_at.append((e, k))
                else:
                    counts[cats[0]] += 1
        rep.ledger[sid] = counts
        for label, where in (("missing", missing_at), ("duplicate", dup_at)):
            if not where:
                continue
            by_event: dict[int, list[int]] = defaultdict(list)
            for e, k in where:
                by_event[e].append(k)
            for e, ks in list(by_event.items())[:50]:
                text = src.events[e].text
                chars = "".join(text[k] for k in ks)
                rep.add("error", f"ledger.{label}", f"{sid}:{e} {label} characters {chars!r} in {text!r}",
                        source=sid, event=e, offsets=ks)
