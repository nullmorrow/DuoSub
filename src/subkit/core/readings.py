"""Reading annotations (ruby) for Japanese-language members.

The baseline reading engine (Sudachi + rules + dictionary + character table +
caption hints) produces *suggestions*.  They are stored as Annotations anchored
to character identities (FORMAT-v1 §6.6) with status ``auto`` or
``uncertain``.  Confirmed, hidden or manual readings already on the member are
never touched (FORMAT-v1 §10: automation cannot override confirmed records).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..ass.text import VisibleText
from ..model.document import Annotation, Document, Member
from ..languages import get_language
from ..model.ids import new_id
from ..model.source import Source

def ruby_parts(base: str, reading: str) -> list[list]:
    """Where the ruby goes: one piece per kanji block, okurigana and other kana stay bare.

    ``合わ``/``あわ`` → [[0, 1, "あ"]]; ``落ち着いて``/``おちついて`` → [[0, 1, "お"], [2, 3, "つ"]].
    Typing only the kanji part (``合わ``/``あ``) works too.  Irregular readings (熟字訓 with
    okurigana) become one piece over the kanji span, with shared trailing kana removed.
    Returns [] when the base has no kanji (nothing to annotate)."""
    from ..reading.kana_align import align, okurigana_split
    from ..textutil import is_digit_char, is_kanji

    if not base or not reading:
        return []
    sub = align(base, reading)
    if sub:
        return [[a, b, r] for a, b, r in sub]
    if any(ch.isascii() and ch.isalnum() or "Ａ" <= ch <= "ｚ" or "０" <= ch <= "９" for ch in base):
        return [[0, len(base), reading]]       # A級（えーきゅう）, ５日（いつか）: the whole word (old tool's rule)
    ks = [i for i, ch in enumerate(base) if is_kanji(ch) or is_digit_char(ch)]
    if not ks:
        return []
    k0, k1 = ks[0], ks[-1] + 1
    head = base[:k0]
    r = reading[len(head):] if reading.startswith(head) else reading
    _s, r = okurigana_split(base[k0:], r)
    return [[k0, k1, r or reading]]


def placement_text(base: str, reading: str) -> str:
    """``合わ``/``あわ`` → ``合(あ)わ``: how the ruby will sit, for the reading popup."""
    out, pos = [], 0
    for a, b, r in ruby_parts(base, reading):
        out.append(base[pos:a])
        out.append(f"{base[a:b]}({r})")
        pos = b
    out.append(base[pos:])
    return "".join(out)


def pieces_of(a: "Annotation") -> list[list]:
    """Ruby pieces of an annotation (stored parts, or computed for older documents)."""
    return a.parts if a.parts else ruby_parts(a.base, a.text)


def _annotation_view(a: Annotation) -> Annotation:
    """Repair only the exact old automatic half-width fallback, without saving it."""
    if (a.kind != "ruby" or a.origin != "analyzer" or a.status not in ("auto", "uncertain")
            or a.mono or not a.parts or not any("ｦ" <= ch <= "ﾟ" for ch in a.base)):
        return a
    from ..reading.kana_align import align
    from ..textutil import is_digit_char, is_kanji

    kanji = [i for i, ch in enumerate(a.base) if is_kanji(ch) or is_digit_char(ch)]
    if not kanji or a.parts != [[kanji[0], kanji[-1] + 1, a.text]]:
        return a
    aligned = align(a.base, a.text)
    if aligned is None:
        return a
    parts = [[start, end, reading] for start, end, reading in aligned]
    return replace(a, parts=parts) if parts != a.parts else a


def display_annotations(doc: Document, member: Member, annotations: list[Annotation]) -> list[Annotation]:
    """Read-only compatibility views shared by the player, editor and review.

    Existing timed lyrics retain stored parts: changing them here could make
    effective karaoke beats appear to need redistribution. Explicit reading
    refresh uses the corrected aligner and keeps its existing review workflow.
    """
    views = [_annotation_view(a) for a in annotations]
    if all(view is original for view, original in zip(views, annotations)):
        return annotations
    from .karaoke import has_karaoke

    if has_karaoke(doc, member):
        return annotations
    vt, ids = visible_map(member)
    for i, (view, original) in enumerate(zip(views, annotations)):
        if view is not original:
            pos = anchor_positions(original, ids)
            if pos is None or vt.text[pos[0]:pos[1]] != original.base:
                views[i] = original
    return views


KEEP_STATUS = ("confirmed", "hidden")
KEEP_ORIGIN = ("manual", "exception", "legacy")
_ORIGIN = {"dict": "dict", "hint": "cc_hint", "rule": "rule", "prefer": "rule", "analyzer": "analyzer",
           "number": "rule", "record": "manual", "llm": "analyzer", "audio": "voice"}


def visible_map(m: Member) -> tuple[VisibleText, list[Optional[tuple]]]:
    """VisibleText of the member text plus the character identity of every visible character."""
    text = m.text
    vt = VisibleText(text)
    spans = []
    pos = 0
    for r in m.runs:
        spans.append((pos, pos + len(r.text), r))
        pos += len(r.text)
    ids: list[Optional[tuple]] = []
    k = 0
    for ch in vt.chars:
        while k < len(spans) and spans[k][1] <= ch.raw_start:
            k += 1
        if k >= len(spans):
            ids.append(None)
            continue
        a, _b, r = spans[k]
        ids.append(r.identity(ch.raw_start - a) if r.kind in ("text",) else None)
    return vt, ids


def plain_to_visible(m: Member) -> list[Optional[int]]:
    """For every character of the editing text (including grey gloss marks, line breaks written out) its visible
    index (None for the characters of a line break and the second character of a hard space)."""
    vt = VisibleText(m.text)
    by_raw = {c.raw_start: i for i, c in enumerate(vt.chars)}
    out: list[Optional[int]] = []
    from .editor_text import is_private_run

    raw = 0
    for r in m.runs:
        if r.kind != "tag" or is_private_run(r):
            out.extend(by_raw.get(raw + k) for k in range(len(r.text)))
        raw += len(r.text)
    return out


def plain_range_to_visible(m: Member, a: int, b: int) -> Optional[tuple[int, int]]:
    """Editing-text range [a, b) → visible range, or None if it covers no visible character."""
    pv = plain_to_visible(m)
    vis = [v for v in pv[max(0, a):max(a, b)] if v is not None]
    return (min(vis), max(vis) + 1) if vis else None


def reading_spans(doc: Document, m: Member, engine=None) -> list[dict]:
    """The member's visible text cut into words for range editing (「迷宮」 + 「都市」 → one reading
    「迷宮都市（めいきゅうとし）」): annotated words with their reading, kanji runs without a reading (the
    analyzer's reading when an engine is given), and single other characters (kana read as themselves).
    Each span: {"a", "b", "reading" (None if unknown), "ann" (annotation id or None)}."""
    from ..textutil import kata_to_hira

    if get_language(m.lang).annotation_system != "ruby":
        return []
    vt, ids = visible_map(m)
    text = vt.text
    owner: dict[int, tuple[tuple[int, int], Annotation]] = {}
    for a in doc.annotations.values():
        if a.member != m.id or a.kind != "ruby" or a.status == "hidden" or not a.text:
            continue
        pos = anchor_positions(a, ids)
        if pos:
            for i in range(*pos):
                owner[i] = (pos, a)
    entries = []
    if engine is not None and text.strip():
        try:
            entries = _analyze_member(m, engine, text)
        except Exception:  # pragma: no cover - a reading hint is optional
            entries = []
    out: list[dict] = []
    i = 0
    while i < len(text):
        if i in owner:
            (pa, pb), a = owner[i]
            out.append({"a": pa, "b": pb, "reading": a.text, "ann": a.id})
            i = pb
            continue
        ch = text[i]
        if "一" <= ch <= "鿿" or ch in "々〆ヶ":
            j = i
            while j < len(text) and j not in owner and ("一" <= text[j] <= "鿿" or text[j] in "々〆ヶ"):
                j += 1
            out.append({"a": i, "b": j, "reading": _engine_reading(entries, text, i, j), "ann": None})
            i = j
            continue
        out.append({"a": i, "b": i + 1, "reading": kata_to_hira(ch) if ("ぁ" <= ch <= "ヿ") else ch, "ann": None})
        i += 1
    return out


def _engine_reading(entries, text: str, a: int, b: int) -> Optional[str]:
    """Reading of the kanji run text[a:b] from analyzer entries (via their kanji-block parts)."""
    from ..textutil import kata_to_hira

    out = []
    k = a
    for e in entries:
        if e.end <= k or e.start >= b or not e.reading:
            continue
        parts = ruby_parts(e.surface, e.reading)
        for pa, pb, t in parts:
            ga, gb = e.start + pa, e.start + pb
            if ga == k and gb <= b:
                out.append(kata_to_hira(t))
                k = gb
        if k >= b:
            break
    return "".join(out) if k >= b else None


def visible_run_positions(m: Member) -> list[Optional[tuple[int, int]]]:
    """(run index, offset in run) of every visible character (as VisibleText sees them)."""
    vt = VisibleText(m.text)
    spans, pos = [], 0
    for ri, r in enumerate(m.runs):
        spans.append((pos, pos + len(r.text), ri))
        pos += len(r.text)
    out: list[Optional[tuple[int, int]]] = []
    k = 0
    for ch in vt.chars:
        while k < len(spans) and spans[k][1] <= ch.raw_start:
            k += 1
        out.append((spans[k][2], ch.raw_start - spans[k][0]) if k < len(spans) else None)
    return out


def anchor_of(ids: list[Optional[tuple]]) -> list[dict]:
    """Compress identities to anchor segments [{"src","e","a","b"}] / [{"ins","a","b"}]."""
    segs: list[dict] = []
    for ident in ids:
        if ident is None:
            continue
        if ident[0] == "s":
            _s, src, e, off = ident
            if segs and segs[-1].get("src") == src and segs[-1].get("e") == e and segs[-1]["b"] == off:
                segs[-1]["b"] = off + 1
            else:
                segs.append({"src": src, "e": e, "a": off, "b": off + 1})
        else:
            _i, ins, off = ident
            if segs and segs[-1].get("ins") == ins and segs[-1]["b"] == off:
                segs[-1]["b"] = off + 1
            else:
                segs.append({"ins": ins, "a": off, "b": off + 1})
    return segs


def anchor_positions(a: Annotation, ids: list[Optional[tuple]]) -> Optional[tuple[int, int]]:
    """Visible [start, end) of an annotation in a member, or None if its characters are not contiguous there."""
    want = []
    for seg in a.anchor:
        if "src" in seg:
            want.extend(("s", seg["src"], seg["e"], o) for o in range(seg["a"], seg["b"]))
        else:
            want.extend(("i", seg["ins"], o) for o in range(seg["a"], seg["b"]))
    if not want:
        return None
    try:
        start = ids.index(want[0])
    except ValueError:
        return None
    if ids[start:start + len(want)] != want:
        return None
    return start, start + len(want)


@dataclass
class _Hint:
    start: int
    end: int
    surface: str
    reading: str


def _cc_hints(m: Member, src: Optional[Source], vt: VisibleText, ids: list) -> list[_Hint]:
    if src is None:
        return []
    pos = {ident: i for i, ident in enumerate(ids) if ident is not None}
    out = []
    for sp in src.spans:
        if sp.kind != "reading_hint" or not sp.base:
            continue
        if any(("s", src.id, sp.e, o) in pos for o in range(sp.a, sp.b)):
            continue                     # the bracket is shown as text (the user said it is not a reading)
        want = [("s", src.id, sp.e, o) for o in range(sp.base[0], sp.base[1])]
        if all(w in pos for w in want):
            a = pos[want[0]]
            if [pos[w] for w in want] == list(range(a, a + len(want))):
                out.append(_Hint(a, a + len(want), vt.text[a:a + len(want)], sp.value or ""))
    return out


def _analyze_member(m: Member, engine, text: str, *, hints=()):
    speaker = m.speaker or {}
    context = {}
    if getattr(getattr(engine, "dict", None), "character_book", None) is not None:
        context = {"character": speaker.get("character"), "line": m.id}
    return engine.analyze(text, speaker=speaker.get("label"), hints=hints, **context)


def suggest_readings(m: Member, engine, *, source: Optional[Source] = None, covered: Optional[set[int]] = None,
                     window: Optional[tuple[int, int]] = None) -> list[Annotation]:
    """Fresh automatic ruby suggestions for a member (not inserted).  The whole line is analysed (context matters
    for segmentation); ``window`` keeps only words overlapping visible [a, b); ``covered`` positions are skipped."""
    if get_language(m.lang).annotation_system != "ruby":
        return []
    vt, ids = visible_map(m)
    if not vt.text.strip():
        return []
    covered = covered or set()
    out: list[Annotation] = []
    for e in _analyze_member(m, engine, vt.text, hints=_cc_hints(m, source, vt, ids)):
        if e.hidden or not e.reading or e.reading == e.surface:
            continue
        if window is not None and (e.end <= window[0] or e.start >= window[1]):
            continue
        if any(i in covered for i in range(e.start, e.end)):
            continue
        seg = anchor_of(ids[e.start:e.end])
        if not seg or any(x is None for x in ids[e.start:e.end]):
            continue
        cands = [e.reading] + [x for x in e.alternatives if x != e.reading]
        parts = [[p.start - e.start, p.end - e.start, p.reading] for p in e.parts] or ruby_parts(e.surface, e.reading)
        if not parts:
            continue                       # no kanji: nothing to annotate
        ann = Annotation(new_id("a_"), "ruby", m.id, seg, e.surface, e.reading,
                         status="uncertain" if e.needs_review else "auto",
                         origin=_ORIGIN.get(e.source, "analyzer"), candidates=cands, parts=parts)
        if e.locked:
            parent = getattr(e, "rule_surface", None)
            ann.rule = f"word_split:{parent}" if parent else f"dict:{e.surface}"
        out.append(ann)
    return out


def kept_positions(doc: Document, m: Member, ids: list, *, kind: str = "ruby") -> tuple[list[Annotation], set[int]]:
    """Readings an automatic refresh must not touch (confirmed / hidden / manual …) and the characters they cover."""
    keep = [a for a in doc.annotations.values() if a.member == m.id and a.kind == kind
            and (a.status in KEEP_STATUS or a.origin in KEEP_ORIGIN)]
    cov: set[int] = set()
    for a in keep:
        p = anchor_positions(a, ids)
        if p:
            cov.update(range(*p))
    return keep, cov


def word_split_plan(doc: Document, m: Member, engine, surface: str, *, source=None) -> dict:
    """Only refresh automatic readings overlapping exact occurrences of a word rule."""
    vt, ids = visible_map(m)
    occurrences = []
    pos = 0
    while surface and (pos := vt.text.find(surface, pos)) >= 0:
        occurrences.append((pos, pos + len(surface)))
        pos += len(surface)
    if not occurrences:
        return {"remove": [], "add": [], "items": []}
    keep, kept = kept_positions(doc, m, ids)
    fixed_ids = {a.id for a in keep}
    old = [(anchor_positions(a, ids), a) for a in doc.annotations.values()
           if a.member == m.id and a.kind == "ruby"]
    removed = [ann for rng, ann in old if rng and ann.id not in fixed_ids
               and any(rng[0] < b and a < rng[1] for a, b in occurrences)]
    removed_ids = {a.id for a in removed}
    covered = set()
    freed = set(i for a, b in occurrences for i in range(a, b))
    for rng, ann in old:
        if rng:
            (freed if ann.id in removed_ids else covered).update(range(*rng))
    added = suggest_readings(m, engine, source=source, covered=covered, window=(min(freed), max(freed) + 1))
    added = [ann for ann in added if (rng := anchor_positions(ann, ids))
             and any(i in freed for i in range(*rng))]

    def signature(anns):
        return [(anchor_positions(a, ids), a.base, a.text, a.parts, a.status, a.origin,
                 a.rule, a.candidates, a.layout, a.extra)
                for a in sorted(anns, key=lambda a: anchor_positions(a, ids) or (-1, -1))]

    def overlaps(ann, a, b):
        rng = anchor_positions(ann, ids)
        return rng and rng[0] < b and a < rng[1]

    items = []
    for a, b in occurrences:
        before = [ann for ann in removed if overlaps(ann, a, b)]
        after = [ann for ann in added if overlaps(ann, a, b)]
        changes = [] if signature(before) == signature(after) else [{
            "before": [{"base": ann.base, "reading": ann.text} for ann in before],
            "after": [{"base": ann.base, "reading": ann.text} for ann in after]}]
        items.append({"member": m.id, "at": [a, b], "text": vt.text,
                      "changes": changes, "kept": sorted(kept & set(range(a, b)))})
    if signature(removed) == signature(added):
        removed, added = [], []
    return {"remove": removed, "add": added, "items": items}


def annotate_member(doc: Document, m: Member, engine, *, source: Optional[Source] = None) -> int:
    """(Re)create automatic ruby suggestions for one member; returns how many were written."""
    if get_language(m.lang).annotation_system != "ruby":
        return 0
    vt, ids = visible_map(m)
    if not vt.text.strip():
        return 0
    keep, kept_cov = kept_positions(doc, m, ids)
    for a in [a for a in doc.annotations.values() if a.member == m.id and a.kind == "ruby" and a not in keep]:
        del doc.annotations[a.id]
    n = 0
    for ann in suggest_readings(m, engine, source=source, covered=kept_cov):
        doc.annotations[ann.id] = ann
        n += 1
    return n


def annotate_document(doc: Document, sources: dict[str, Source], engine, *, lang: str = "ja", cancel_check=lambda: None) -> int:
    total = 0
    for g in doc.groups:
        cancel_check()
        for mid in g.members:
            m = doc.members[mid]
            if get_language(m.lang).annotation_system != "ruby" or m.role in ("sign", "info", "comment"):
                continue
            src = next((sources[r.src] for r in m.runs if r.src in sources), None)
            total += annotate_member(doc, m, engine, source=src)
    return total
