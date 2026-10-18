"""Character formatting — a whole line or part of it in another font, size, weight or colour (emotion, shouting…),
plus any other ASS override tags (rotation, shear, scaling, borders, \\t animation …) typed as they are.

* a whole line: ``member.overrides`` keys ``fn`` ``fs`` ``b`` ``i`` ``c1`` and ``raw`` (override tags as text);
  written as line tags, and tags of the same names in the source text are left out of the output — the source
  itself is never changed;
* part of a line: ``member.format`` spans ``{"anchor": [...], "props": {...}}`` anchored to the characters like
  readings, so the formatting follows the text through edits.  Export opens the span with its tags, drops the
  source's own tags of those names inside the span, and afterwards restores what the source has in effect there.

Colours are stored as ASS colours ``&HBBGGRR&``; sizes in script pixels (PlayRes).
"""

from __future__ import annotations

import copy

from typing import Any, Optional

from ..ass.state import animated_names, check_raw, names_of, restore_tags, strip_inside
from ..ass.tags import canonical, parse_block
from ..ass.state import LINE_ONLY
from ..ass.text import insert_at_visible
from ..model.document import Member
from .readings import anchor_of, visible_map

FMT_PROPS = ("fn", "fs", "b", "i", "c1")
PROPS = FMT_PROPS + ("raw",)
TAG_OF = {"fn": "fn", "fs": "fs", "b": "b", "i": "i", "c1": "1c"}

__all__ = ["PROPS", "FMT_PROPS", "TAG_OF", "tags", "char_props", "with_span", "apply_spans", "line_props",
           "shown_props", "check_raw", "tag_names", "spans_for", "merge_formats", "group_raw_changes"]


def tags(props: dict[str, Any]) -> str:
    out = []
    for k in FMT_PROPS:
        if k not in props or props[k] in (None, ""):
            continue
        v = props[k]
        if k == "fn":
            out.append(f"\\fn{v}")
        elif k == "fs":
            out.append(f"\\fs{float(v):g}")
        elif k in ("b", "i"):
            out.append(f"\\{k}{1 if v else 0}")
        elif k == "c1":
            out.append(f"\\1c{v}")
    if props.get("raw"):
        out.append(str(props["raw"]))
    return "".join(out)


def tag_names(props: dict[str, Any]) -> set[str]:
    """Every tag spelling the properties set (to leave the source's own tags of those names out)."""
    names: set[str] = set()
    for k in FMT_PROPS:
        if props.get(k) not in (None, ""):
            names |= {TAG_OF[k]} | ({"c"} if k == "c1" else set())
    if props.get("raw"):
        names |= names_of(str(props["raw"])) - {"t"}       # the source's own animations stay
    if names & {"fad", "fade"}:
        names |= {"fad", "fade"}                             # one fade per line: either spelling replaces both
    return names


def group_raw_changes(doc, mid: str, raw: Optional[str]) -> dict[str, dict[str, Any]]:
    """Explicit group ASS tags, preserving unrelated manual tags and all source text.

    Use Editor.set_overrides once with this mapping so the bilingual edit has one undo step.
    Empty input restores each member's source tags, just like clearing a line's raw setting.
    """
    normalized, error = check_raw(raw or "", span=False)
    if error:
        raise ValueError(error)
    names = tag_names({"raw": normalized})
    changes = {}
    for m in doc.members_of(doc.group_of(mid)):
        if m.lane not in ("original", "translation") or m.role == "comment" or m.source_kind == "Comment":
            continue
        old = str(m.overrides.get("raw") or "")
        kept = "".join(t.raw for t in parse_block(old) if canonical(t.name) not in names)
        new = kept + normalized if normalized else ""
        if new != old:
            changes[m.id] = {"raw": new or None}
    return changes


def _ids_of(segs: list[dict]) -> list[tuple]:
    out: list[tuple] = []
    for s in segs:
        if "src" in s:
            out.extend(("s", s["src"], s["e"], o) for o in range(s["a"], s["b"]))
        else:
            out.extend(("i", s["ins"], o) for o in range(s["a"], s["b"]))
    return out


def char_props(m: Member, ids: Optional[list] = None) -> list[Optional[dict]]:
    """Span properties of every visible character of a member (None = no span)."""
    if ids is None:
        ids = visible_map(m)[1]
    by_id: dict[tuple, dict] = {}
    for sp in m.format or ():
        for ident in _ids_of(sp.get("anchor", [])):
            by_id[ident] = sp.get("props") or {}
    return [dict(by_id[i]) if i in by_id else None for i in ids]


def with_span(m: Member, a: int, b: int, change: dict[str, Any]) -> list[dict]:
    """The member's spans after applying ``change`` to visible [a, b) (a value None removes that property)."""
    _vt, ids = visible_map(m)
    cur = char_props(m, ids)
    for i in range(max(0, a), min(len(ids), b)):
        d = dict(cur[i] or {})
        for k, v in change.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v
        cur[i] = d or None
    return _spans(ids, cur)


def _spans(ids: list, cur: list[Optional[dict]]) -> list[dict]:
    spans: list[dict] = []
    i = 0
    while i < len(ids):
        if not cur[i] or ids[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < len(ids) and cur[j + 1] == cur[i] and ids[j + 1] is not None:
            j += 1
        spans.append({"anchor": anchor_of(ids[i:j + 1]), "props": cur[i]})
        i = j + 1
    return spans


def spans_for(m: Member, spans: list[dict]) -> list[dict]:
    """Keep only a destination member's character identities in each incoming span."""
    ids = {ident for ident, *_ in m.identities() if ident is not None}
    out = []
    for sp in spans:
        anchor = anchor_of([i for i in _ids_of(sp.get("anchor", [])) if i in ids])
        if anchor:
            out.append({**copy.deepcopy(sp), "anchor": anchor})
    return out


def merge_formats(first: Member, second: Member) -> tuple[list[dict], bool]:
    """Second-line text overrides become character spans; first-line placement/animation wins."""
    line_only = LINE_ONLY | {"t"}

    def settings(m: Member) -> tuple:
        raw = tuple(t.raw for t in parse_block(str(m.overrides.get("raw") or ""))
                    if canonical(t.name) in line_only)
        return (m.overrides.get("an"), m.overrides.get("pos"), m.overrides.get("move"), raw)

    conflict = settings(first) != settings(second)
    defaults = line_props(second.overrides)
    if defaults.get("raw"):
        # These tags cannot be scoped to the second line's characters.
        raw = "".join(t.raw for t in parse_block(str(defaults["raw"]))
                      if canonical(t.name) not in line_only)
        if raw:
            defaults["raw"] = raw
        else:
            defaults.pop("raw")
    ids = visible_map(second)[1]
    cur = char_props(second, ids)
    for i, props in enumerate(cur):
        combined = {**defaults, **(props or {})}
        if defaults.get("raw"):
            names = tag_names({k: v for k, v in (props or {}).items() if k in FMT_PROPS})
            raw = "".join(t.raw for t in parse_block(defaults["raw"]) if t.name not in names)
            raw += str((props or {}).get("raw") or "")
            if raw:
                combined["raw"] = raw
            else:
                combined.pop("raw", None)
        cur[i] = combined or None
    return copy.deepcopy(first.format) + _spans(ids, cur), conflict


def _restore_names(p: dict[str, Any]) -> list[str]:
    names = [TAG_OF[k] for k in FMT_PROPS if p.get(k) not in (None, "")]
    if p.get("raw"):
        names += sorted(animated_names(str(p["raw"])) - set(names))
    return names


def apply_spans(m: Member, ids_shown: list, text: str, style) -> str:
    """The event text with the member's spans: each opens with its tags, the source's own tags of those names
    inside it are left out, and after it the values the source has in effect there come back."""
    if not m.format:
        return text
    props = char_props(m, ids_shown)
    n = len(props)
    runs: list[tuple[int, int, dict]] = []
    i = 0
    while i < n:
        p = props[i]
        if not p:
            i += 1
            continue
        j = i
        while j + 1 < n and props[j + 1] == p:
            j += 1
        runs.append((i, j + 1, p))
        i = j + 1
    inserts: dict[int, str] = {}
    for a, b, p in runs:          # restores are computed on the source text, before anything is changed
        inserts[a] = inserts.get(a, "") + "{" + tags(p) + "}"
        if b < n:
            inserts[b] = "{" + restore_tags(text, b, _restore_names(p), style) + "}" + inserts.get(b, "")
    out = text
    for a, b, p in runs:
        out = strip_inside(out, a, b, tag_names(p))
    return insert_at_visible(out, inserts)


def line_props(overrides: dict[str, Any]) -> dict[str, Any]:
    """The whole-line formatting among a member's overrides (raw tags included)."""
    return {k: overrides[k] for k in PROPS if overrides.get(k) not in (None, "")}


def shown_props(m: Member, style, vis: Optional[tuple[int, int]]) -> dict[str, Any]:
    """What a format bar shows: the selection's span (first character), else the whole line, over the style."""
    props: dict[str, Any] = {"fn": style.fontname if style else "", "fs": style.fontsize if style else 48.0,
                             "b": 1 if (style and style.bold) else 0, "i": 1 if (style and style.italic) else 0,
                             "c1": None, "raw": ""}
    if vis:
        props["raw"] = ""
        for k, v in m.overrides.items():
            if k in FMT_PROPS:
                props[k] = v
        cp = char_props(m)
        props.update((cp[vis[0]] if vis[0] < len(cp) else None) or {})
    else:
        props.update({k: v for k, v in m.overrides.items() if k in props})
    return props
