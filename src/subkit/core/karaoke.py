"""K 轴 (karaoke) — beats, their timing, and the ``\\k`` tags of the export (docs/KARAOKE.md).

Same relationships and output as the old tool and the user's finished files:

* the original-language lyric is split into sung **beats**: a kanji block sings its reading (``潔`` = い|さ|ぎ|よ),
  kana are one beat each, small kana always join the previous beat, and in the default "syllable" unit
  ``っ ー ん`` join it too (``なっ`` ``ガー`` ``ホン``; the "mora" unit keeps them separate); a Latin word is one
  beat; punctuation and spaces join the previous beat (at the start of a line, the next one);
* the base line gets one tag per slot (a kanji block gets the total of its beats);
* every ruby piece waits with an empty syllable and then sings its beats (``{\\kf199}{\\kf24}い``);
* the translation lyric (e.g. Chinese) is one beat per character / per Latin word;
* silence between beats is an empty syllable.

Timing is stored per beat with **absolute** milliseconds and keyed by the identities of the characters it
sings (plus its index inside a kanji block and the kana it sings).  The beat structure itself is always derived
from the current text and readings, so edits never leave stale timing behind: a beat whose characters and kana
are unchanged keeps its time; a kanji block whose reading changed keeps its time span, split evenly and
flagged ``reading_changed``; new characters get placeholder time and are flagged.
"""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from ..ass.tags import KARAOKE_TAGS
from ..ass.text import tokenize
from ..model.document import Document, Member
from ..model.ids import new_id
from ..textutil import is_kana, is_kanji, kata_to_hira
from .readings import anchor_of, anchor_positions, pieces_of, visible_map

SMALL = set("ぁぃぅぇぉゃゅょゎゕゖァィゥェォャュョヮヵヶ")
HOLD = set("っッーんン")                 # join the previous beat in the "syllable" unit
TIMED = ("tap", "drag", "import", "memory", "onset", "manual", "align")
KINDS = ("kf", "k", "ko", "none")
EDIT_MISSING = "拆分/合并的位置找不到了（字或读音改过）"
LINK_MISSING = "吸附的原声拍找不到了（字、读音或拆合改过）"


@dataclass
class BeatPart:
    """One editable character/kana, retaining its base and ruby provenance."""
    a: int
    b: int
    core: tuple
    head: Optional[tuple]
    sung: str
    at: dict
    ann: Optional[str] = None
    piece: Optional[int] = None
    r: Optional[tuple[int, int]] = None


@dataclass
class Beat:
    a: int                                  # visible [a, b) of the base characters (member text)
    b: int
    slot: int                               # beats of one kanji block / kana / word share a slot
    k: int = 0                              # index of the beat in its slot
    n: int = 1
    core: tuple = ()                        # identities that key the timing (the slot's own characters)
    head: Optional[tuple] = None            # identity of the slot's first character (leading punctuation included)
    sung: str = ""                          # the kana / text this beat sings (hiragana for kana)
    ann: Optional[str] = None               # ruby annotation and piece this beat belongs to
    piece: Optional[int] = None
    r: Optional[tuple[int, int]] = None     # offsets in the piece reading
    t0: Optional[int] = None                # absolute ms
    t1: Optional[int] = None
    by: str = ""
    flags: list[str] = field(default_factory=list)
    parts: tuple[BeatPart, ...] = ()
    link: dict[str, dict] = field(default_factory=dict)  # own edge -> original beat key and edge
    base_text: str = ""                      # derived visible text, never persisted

    @property
    def key(self) -> tuple:
        return (self.core, self.k, self.sung)

    @property
    def timed(self) -> bool:
        return self.t0 is not None and self.by in TIMED


@dataclass(frozen=True)
class Rest:
    """A gap after a beat; timing remains on its two neighbouring beat edges."""
    after: int
    t0: int
    t1: int


def rests_of(beats: list[Beat]) -> list[Rest]:
    return [Rest(i, left.t1, right.t0) for i, (left, right) in enumerate(zip(beats, beats[1:]))
            if left.t1 is not None and right.t0 is not None and left.t1 < right.t0]


def move_edge(beats: list[Beat], index: int, edge: str, ms: int, *, shift: bool = False) -> list[Beat]:
    """Shared edges move together; the two sides of a rest never move each other."""
    if edge == "end" and index + 1 < len(beats) and beats[index].t1 == beats[index + 1].t0:
        index, edge = index + 1, "start"
    out = copy.deepcopy(beats)
    b = out[index]
    if edge == "start":
        previous = out[index - 1] if index else None
        shared = previous is not None and previous.t1 == b.t0
        lo = previous.t0 + 10 if shared else previous.t1 if previous else -10 ** 9
        ms = max(0, lo, min(b.t1 - 10, ms))
        delta = ms - b.t0
        if shift and (previous is None or shared):
            for later in out[index:]:
                later.t0 += delta
                later.t1 += delta
        else:
            b.t0 = ms
        if shared:
            previous.t1 = ms
    elif edge == "end":
        hi = out[index + 1].t0 if index + 1 < len(out) else 10 ** 9
        b.t1 = max(b.t0 + 10, min(hi, ms))
    else:
        raise ValueError("unknown beat edge")
    for old, changed in zip(beats, out):
        for own_edge, field in (("start", "t0"), ("end", "t1")):
            if getattr(old, field) != getattr(changed, field):
                changed.by = "drag"
                changed.flags = [f for f in changed.flags if f != "check"]
                changed.link.pop(own_edge, None)
    return out


def view_signature(view: Optional["KView"]) -> tuple:
    """What the export depends on (part of the preview cache key)."""
    if view is None:
        return ()
    return (view.kind, json.dumps(view.params, sort_keys=True, ensure_ascii=False), view.state, tuple(view.issues),
            tuple((b.slot, b.k, b.t0, b.t1, b.ann, b.piece, b.r, b.head,
                   tuple((p.core, p.ann, p.piece, p.r, p.sung) for p in b.parts)) for b in view.beats))


def kana_beats(kana: str, unit: str = "syllable") -> list[tuple[int, int]]:
    """Split a reading into beats: [(start, end)] offsets.  Spaces join the previous beat."""
    out: list[list[int]] = []
    for i, ch in enumerate(kana):
        joins = ch in SMALL or ch.isspace() or (unit == "syllable" and ch in HOLD)
        if out and joins:
            out[-1][1] = i + 1
        elif not ch.isspace():
            out.append([i, i + 1])
    return [(a, b) for a, b in out]


def _is_word(ch: str) -> bool:
    return (ch.isascii() and (ch.isalnum() or ch in "'’")) or ("０" <= ch <= "９") or ("Ａ" <= ch <= "ｚ")


@lru_cache(maxsize=1)
def _english_dictionary():
    """Optional local dictionary; never download or install a dependency."""
    try:
        import pyphen

        return pyphen.Pyphen(lang="en_US", left=1, right=1)
    except (ImportError, KeyError, OSError):
        return None


def english_syllables(word: str) -> list[str]:
    """Sung syllables: dictionary break positions when they give as many pieces as the vowel rules.

    Hyphenation dictionaries are made for typesetting and leave words like away, over or alone
    whole; a sung beat needs those split, so fewer dictionary pieces fall back to the vowel rules.
    """
    if not re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", word):
        return [word]
    vowels = _vowel_syllables(word)
    dictionary = _english_dictionary()
    if dictionary is not None:
        pieces = dictionary.inserted(word).split("-")
        if all(pieces) and "".join(pieces) == word and len(pieces) >= len(vowels):
            return pieces
    return vowels


def _vowel_syllables(word: str) -> list[str]:
    lower = word.lower()
    end = len(word)
    # Final e is silent except consonant + le; -ed only adds a beat after t/d.
    if lower.endswith("ed") and len(word) > 3 and lower[-3] not in "td":
        end -= 2
    elif lower.endswith("e") and not (lower.endswith("le") and len(word) > 2
                                      and lower[-3] not in "aeiouy"):
        end -= 1
    nuclei = list(re.finditer(r"[aeiouy]+", lower[:end]))
    if len(nuclei) < 2:
        return [word]
    onsets = {"bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "ph", "pl", "pr",
              "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw", "th", "tr", "tw", "wh"}
    cuts = []
    for first, second in zip(nuclei, nuclei[1:]):
        a, z = first.end(), second.start()
        cluster = lower[a:z]
        cut = a if len(cluster) <= 1 else z - (2 if cluster[-2:] in onsets else 1)
        # These suffixes keep their sung onset together (ac|tion, vi|sion).
        for suffix in ("tion", "sion", "ing"):
            start = len(word) - len(suffix)
            if lower.endswith(suffix) and a <= start <= z:
                cut = start
        cuts.append(cut)
    bounds = [0] + sorted(set(c for c in cuts if 0 < c < len(word))) + [len(word)]
    return [word[a:z] for a, z in zip(bounds, bounds[1:])]


def english_split_boundaries(doc: Document, m: Member, view: "KView") -> list[dict]:
    """Only new, word-internal syllable boundaries; keep existing manual edits."""
    if m.extra.get("karaoke_off") or m.role == "comment" or m.source_kind == "Comment":
        return []
    text = visible_map(m)[0].text
    starts = {_boundary_id(b.parts[0].at) for b in view.beats if b.parts}
    parts = {p.a: p for b in view.beats for p in b.parts if p.ann is None}
    out = []
    for match in re.finditer(r"(?<![\w'’])[A-Za-z]+(?:['’][A-Za-z]+)*(?![\w'’])", text):
        pieces = english_syllables(match.group())
        if len(pieces) < 2:
            continue
        pos = match.start()
        positions = [pos]
        for piece in pieces[:-1]:
            pos += len(piece)
            positions.append(pos)
        # A lyric mapping may have merged several words into a single beat.
        positions.append(match.end())
        for pos in positions:
            part = parts.get(pos)
            if part is not None and _boundary_id(part.at) not in starts:
                if part.at not in out:
                    out.append(copy.deepcopy(part.at))
    return out


def derive(doc: Document, m: Member, unit: str = "syllable", edits: Optional[list[dict]] = None,
           *, issues: Optional[list[str]] = None) -> list[Beat]:
    """Beat structure of a member from its current text and ruby (no times)."""
    vt, ids = visible_map(m)
    text = vt.text
    pieces: dict[int, tuple[int, str, int, str]] = {}
    for a in doc.annotations.values():
        if a.member != m.id or a.kind != "ruby" or a.status == "hidden" or not a.text:
            continue
        pos = anchor_positions(a, ids)
        if pos is None:
            continue
        for k, (pa, pb, t) in enumerate(pieces_of(a)):
            pieces[pos[0] + pa] = (pos[0] + pb, a.id, k, t)
    beats: list[Beat] = []
    slot = -1
    lead: Optional[int] = None              # punctuation at the start of the line joins the first beat
    i = 0

    def add(beat: Beat) -> None:
        nonlocal lead
        beat.head = ids[beat.a] if beat.a < len(ids) else None
        if lead is not None:
            beat.a = lead
            beat.head = ids[lead]
            lead = None
        beats.append(beat)

    while i < len(text):
        ch = text[i]
        if i in pieces:
            end, aid, pk, reading = pieces[i]
            slot += 1
            parts = kana_beats(reading, unit) or [(0, len(reading))]
            core = tuple(ids[i:end])
            for k, (r0, r1) in enumerate(parts):
                add(Beat(i, end, slot, k=k, n=len(parts), core=core, sung=kata_to_hira(reading[r0:r1]).replace(" ", ""),
                         ann=aid, piece=pk, r=(r0, r1)))
            i = end
            continue
        if is_kana(ch):
            joins = ch in SMALL or (unit == "syllable" and ch in HOLD)
            if joins and beats and beats[-1].b == i:
                beats[-1].b = i + 1
                if beats[-1].ann is None:
                    beats[-1].sung += kata_to_hira(ch)
                i += 1
                continue
            slot += 1
            add(Beat(i, i + 1, slot, core=(ids[i],), sung=kata_to_hira(ch)))
            i += 1
            continue
        if _is_word(ch):
            j = i
            while j < len(text) and _is_word(text[j]):
                j += 1
            slot += 1
            add(Beat(i, j, slot, core=tuple(ids[i:j]), sung=text[i:j].lower()))
            i = j
            continue
        if is_kanji(ch) or (not ch.isspace() and ch.isalpha()):
            slot += 1
            b = Beat(i, i + 1, slot, core=(ids[i],), sung=ch)
            if is_kanji(ch):
                b.flags.append("no_reading")
            add(b)
            i += 1
            continue
        # punctuation, space, symbols: join the previous beat (or the next one at the start)
        if beats and beats[-1].b == i:
            for bt in beats:
                if bt.slot == beats[-1].slot:
                    bt.b = max(bt.b, i + 1)
        elif not beats:
            lead = i if lead is None else lead
        i += 1
    _add_parts(doc, beats, text, ids)
    beats = _apply_edits(beats, edits, issues) if edits else beats
    for beat in beats:
        beat.base_text = text[beat.a:beat.b]
    return beats


def _boundary_id(at: dict) -> tuple:
    return (_ids_of_anchor(at.get("core", [])), at.get("r"), at.get("reading"))


def _add_parts(doc: Document, beats: list[Beat], text: str, ids: list) -> None:
    for b in beats:
        parts = []
        if b.ann is not None:
            reading = pieces_of(doc.annotations[b.ann])[b.piece][2]
            for r0 in range(*b.r):
                if reading[r0].isspace() and parts:
                    parts[-1].r = (parts[-1].r[0], r0 + 1)
                    continue
                at = {"core": anchor_of(list(b.core)), "r": r0,
                      "reading": kata_to_hira(reading)}
                parts.append(BeatPart(b.a, b.b, b.core, b.head, kata_to_hira(reading[r0]).replace(" ", ""),
                                      at, b.ann, b.piece, (r0, r0 + 1)))
        else:
            chars = [i for i in range(b.a, b.b) if is_kana(text[i]) or _is_word(text[i])
                     or is_kanji(text[i]) or text[i].isalpha()]
            for j, i in enumerate(chars):
                core = (ids[i],)
                parts.append(BeatPart(b.a if j == 0 else i, chars[j + 1] if j + 1 < len(chars) else b.b,
                                      core, b.head if j == 0 else ids[i],
                                      kata_to_hira(text[i]).lower(), {"core": anchor_of(list(core))}))
        b.parts = tuple(parts)


def boundary_key(doc: Document, m: Member, at: int, reading_offset: Optional[int] = None) -> dict:
    """Identity key before a visible base character, or at a ruby piece's kana offset."""
    ids = visible_map(m)[1]
    if not 0 <= at < len(ids):
        raise ValueError(EDIT_MISSING)
    for b in derive(doc, m):
        for p in b.parts:
            if reading_offset is None:
                if p.ann is None and p.core == (ids[at],):
                    return copy.deepcopy(p.at)
                if p.ann is not None and p.a <= at < p.b and p.r[0] == 0:
                    return copy.deepcopy(p.at)
            elif p.ann is not None and p.a <= at < p.b and p.r[0] == reading_offset:
                return copy.deepcopy(p.at)
    raise ValueError(EDIT_MISSING)


def _apply_edits(beats: list[Beat], edits: list[dict], issues: Optional[list[str]]) -> list[Beat]:
    parts = [p for b in beats for p in b.parts]
    positions = {_boundary_id(p.at): i for i, p in enumerate(parts) if all(x is not None for x in p.core)}
    starts = {_boundary_id(b.parts[0].at) for b in beats if b.parts}
    for edit in edits:
        key = _boundary_id(edit.get("at", {}))
        i = positions.get(key)
        if i is None or i == 0 or edit.get("op") not in ("split", "merge"):
            if issues is not None and EDIT_MISSING not in issues:
                issues.append(EDIT_MISSING)
            continue
        if edit["op"] == "split":
            starts.add(key)
        else:
            starts.discard(key)
    original = {tuple(_boundary_id(p.at) for p in b.parts): b for b in beats}
    part_flags = {id(p): b.flags for b in beats for p in b.parts}
    groups: list[list[BeatPart]] = []
    for p in parts:
        if not groups or _boundary_id(p.at) in starts:
            groups.append([])
        groups[-1].append(p)
    out = []
    for ps in groups:
        old = original.get(tuple(_boundary_id(p.at) for p in ps))
        core = tuple(dict.fromkeys(i for p in ps for i in p.core))
        if old is not None:
            out.append(replace(old, core=core, flags=list(old.flags)))
            continue
        same_ruby = ps[0].ann is not None and all((p.ann, p.piece) == (ps[0].ann, ps[0].piece) for p in ps)
        out.append(Beat(min(p.a for p in ps), max(p.b for p in ps), 0, core=core, head=ps[0].head,
                        sung="".join(p.sung for p in ps), ann=ps[0].ann if same_ruby else None,
                        piece=ps[0].piece if same_ruby else None,
                        r=(ps[0].r[0], ps[-1].r[1]) if same_ruby else None, parts=tuple(ps),
                        flags=list(dict.fromkeys(f for p in ps for f in part_flags[id(p)]))))
    # A base ruby block cannot carry separate tags at its internal reading boundaries.
    parents = list(range(len(out)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    ruby_owner = {}
    for i, b in enumerate(out):
        for p in b.parts:
            if p.ann is not None:
                key = (p.ann, p.piece)
                if key in ruby_owner:
                    parents[root(i)] = root(ruby_owner[key])
                else:
                    ruby_owner[key] = i
    slots: dict[int, list[Beat]] = {}
    for i, b in enumerate(out):
        slots.setdefault(root(i), []).append(b)
    for slot, bs in enumerate(slots.values()):
        for k, b in enumerate(bs):
            b.slot, b.k, b.n = slot, k, len(bs)
    return out


def remap_edits(doc: Document, source: Member, target: Member, edits: list[dict]) -> list[dict]:
    """Twins have different character identities; map keys by corresponding visible positions."""
    src_ids, dst_ids = visible_map(source)[1], visible_map(target)[1]
    mapping = dict(zip(src_ids, dst_ids))
    out = []
    for edit in edits:
        core = _ids_of_anchor(edit.get("at", {}).get("core", []))
        if not core or any(i not in mapping or mapping[i] is None for i in core):
            continue
        e = copy.deepcopy(edit)
        e["at"]["core"] = anchor_of([mapping[i] for i in core])
        out.append(e)
    return out


# ------------------------------------------------------------------ stored timing
def tracks_of(doc: Document, mid: str, *, dormant: bool = False) -> list[dict[str, Any]]:
    """Active tracks; ``dormant=True`` also includes saved, inactive tracks."""
    return [t for t in doc.karaoke.values() if t.get("member") == mid and (dormant or not t.get("dormant"))]


def _ids_of_anchor(segs: list[dict]) -> tuple:
    out = []
    for s in segs:
        if "src" in s:
            out.extend(("s", s["src"], s["e"], o) for o in range(s["a"], s["b"]))
        else:
            out.extend(("i", s["ins"], o) for o in range(s["a"], s["b"]))
    return tuple(out)


def beat_link(beat: Beat, edge: str) -> dict:
    """A JSON-safe reference using the same identity key as stored timing."""
    return {"core": anchor_of(list(beat.core)), "k": beat.k, "sung": beat.sung, "edge": edge}


def linked_times(doc: Document, gid: str) -> dict[tuple, Beat]:
    """Only effective original beats in this group can be link targets."""
    idx = KIndex(doc)
    return {b.key: b for m in doc.members_of(doc.group(gid))
            if m.lane == "original" and not m.extra.get("karaoke_off")
            for b in effective(doc, m, idx=idx).beats}


def follow_links(track: dict, originals: dict[tuple, Beat]) -> bool:
    """Move linked edges and interpolate interior edges from their previous positions."""
    edges = [(b, edge, field) for b in track.get("beats", ())
             for edge, field in (("start", "t0"), ("end", "t1"))]
    anchors = []
    resolved = {}
    changed = False
    for i, (b, edge, field) in enumerate(edges):
        link = b.get("link", {}).get(edge)
        if not link:
            continue
        key = (_ids_of_anchor(link.get("core", [])), int(link.get("k", 0)), link.get("sung", ""))
        original = originals.get(key)
        target_edge = link.get("edge")
        if original is None or target_edge not in ("start", "end"):
            del b["link"][edge]
            if not b["link"]:
                b.pop("link")
            issues = track.setdefault("issues", [])
            if LINK_MISSING not in issues:
                issues.append(LINK_MISSING)
            changed = True
            continue
        value = original.t0 if target_edge == "start" else original.t1
        resolved[(id(b), edge)] = original
        anchors.append((i, int(b[field]), int(value)))
    updates = {i: value for i, _old, value in anchors}
    for (i, old_a, new_a), (j, old_z, new_z) in zip(anchors, anchors[1:]):
        for k in range(i + 1, j):
            b, _edge, field = edges[k]
            # Equal-time anchors have no temporal ratio; preserve their ordering evenly.
            ratio = (int(b[field]) - old_a) / (old_z - old_a) if old_z > old_a else (k - i) / (j - i)
            updates[k] = int(round(new_a + (new_z - new_a) * ratio))
    for i, value in updates.items():
        b, _edge, field = edges[i]
        if int(b[field]) != value:
            b[field] = value
            changed = True
    # A fully mapped placeholder becomes timed once its entire original span is tapped.
    ordered = list(originals.values())
    positions = {b.key: i for i, b in enumerate(ordered)}
    for bi, b in enumerate(track.get("beats", ())):
        first = resolved.get((id(b), "start"))
        last = resolved.get((id(b), "end"))
        # Fractional edges of an automatic split have no direct link. Their enclosing linked edges
        # still identify the original span that must be timed before promoting the placeholder.
        # Other unlinked pieces (e.g. English syllables) still wait to be tapped.
        if b.get("by") == "placeholder" and b.get("auto_split"):
            if first is None:
                left = next((i for i, _old, _new in reversed(anchors) if i <= 2 * bi), None)
                if left is not None:
                    owner, edge, _field = edges[left]
                    first = resolved.get((id(owner), edge))
            if last is None:
                right = next((i for i, _old, _new in anchors if i >= 2 * bi + 1), None)
                if right is not None:
                    owner, edge, _field = edges[right]
                    last = resolved.get((id(owner), edge))
        if b.get("by") == "placeholder" and first is not None and last is not None:
            a, z = positions[first.key], positions[last.key]
            if a <= z and all(o.timed for o in ordered[a:z + 1]):
                b["by"] = "manual"
                changed = True
    if track.get("state") == "placeholder" and track.get("beats") and all(
            b.get("by") in TIMED for b in track["beats"]):
        track["state"] = "timed"
        changed = True
    return changed


class KIndex:
    """All stored beats of a document: (core identities, k, sung) → beat; which members have a track."""

    def __init__(self, doc: Document) -> None:
        self.beats: dict[tuple, dict] = {}
        self.by_core: dict[tuple, list[dict]] = {}
        self.ids: set = set()
        self.members: set[str] = set()
        self.edits: list[tuple[str, dict]] = []
        for t in doc.karaoke.values():
            if t.get("dormant"):
                continue
            self.members.add(t.get("member"))
            for edit in t.get("edits", []):
                self.edits.append((t.get("member"), edit))
                self.ids.update(_ids_of_anchor(edit.get("at", {}).get("core", [])))
            for sb in t.get("beats", ()):
                core = _ids_of_anchor(sb.get("core", []))
                self.beats[(core, int(sb.get("k", 0)), sb.get("sung", ""))] = sb
                self.by_core.setdefault(core, []).append(sb)
                self.ids.update(core)

    def __bool__(self) -> bool:
        return bool(self.members)

    def has(self, m: Member) -> bool:
        if m.extra.get("karaoke_off"):
            return False
        if m.id in self.members:
            return True
        return bool(self.ids) and any(ident in self.ids for ident, *_ in m.identities() if ident is not None)


def stored_index(doc: Document) -> KIndex:
    return KIndex(doc)


def has_karaoke(doc: Document, m: Member, idx: Optional[KIndex] = None) -> bool:
    return (idx or KIndex(doc)).has(m)


def edits_of(doc: Document, m: Member, idx: Optional[KIndex] = None) -> list[dict]:
    """Edits belonging to this member, plus edits whose characters have moved here."""
    if m.extra.get("karaoke_off"):
        return []
    idx = idx if idx is not None else KIndex(doc)
    ids = {i for i, *_ in m.identities() if i is not None}
    out = {}
    # This member's latest decision takes precedence over a track left behind by a text move.
    for mid, edit in sorted(idx.edits, key=lambda item: item[0] == m.id):
        key = _boundary_id(edit.get("at", {}))
        if mid != m.id and (not key[0] or not set(key[0]) <= ids):
            continue
        out[key] = copy.deepcopy(edit)
    return list(out.values())


@dataclass
class KView:
    member: str
    beats: list[Beat]
    state: str                       # none | placeholder | suggested | timed | confirmed
    unit: str
    kind: str                        # kf | k | ko | none
    issues: list[str] = field(default_factory=list)

    params: dict = field(default_factory=dict)

    @property
    def tag_kind(self):
        return self.kind if self.kind in KINDS else "kf"

    @property
    def complete(self) -> bool:
        return self.state != "suggested" and bool(self.beats) and all(b.timed for b in self.beats)


def placeholder_eligible(doc: Document, m: Member) -> bool:
    """Song lyrics without source karaoke may render derived, unstored placeholder beats."""
    return (not m.extra.get("karaoke_off") and m.lane in ("original", "translation")
            and m.role != "comment" and m.source_kind != "Comment" and not has_k_tags(m)
            and doc.group_of(m.id).kind == "song" and bool(visible_map(m)[0].text.strip()))


def effective(doc: Document, m: Member, *, idx: Optional[KIndex] = None, default_kind: str = "kf",
              unit: Optional[str] = None) -> KView:
    """Derived beats with their timing (stored, or placeholder where nothing is stored)."""
    off = bool(m.extra.get("karaoke_off"))
    tr = [] if off else tracks_of(doc, m.id)
    t0 = tr[0] if tr else {}
    unit = unit or t0.get("unit") or "syllable"
    kind = ((t0.get("effect") or {}).get("preset")) or default_kind
    idx = KIndex(doc) if idx is None else idx
    issues: list[str] = list(dict.fromkeys(issue for t in tr for issue in t.get("issues", [])))
    beats = derive(doc, m, unit, edits_of(doc, m, idx), issues=issues)
    # A converted dialogue must not borrow another member's active timing through shared identities.
    by_core = {} if off else idx.by_core
    for b in beats:
        sb = None if off else idx.beats.get(b.key)
        if sb is not None:
            b.t0, b.t1, b.by = int(sb["t0"]), int(sb["t1"]), sb.get("by", "manual")
            b.link = copy.deepcopy(sb.get("link", {}))
            for flag in ("check", "auto_split"):
                if sb.get(flag):
                    b.flags.append(flag)
    # a kanji block whose reading changed: keep its old time span, split evenly, flag it
    slots: dict[int, list[Beat]] = {}
    for b in beats:
        slots.setdefault(b.slot, []).append(b)
    for bs in slots.values():
        if all(b.t0 is not None for b in bs):
            continue
        old = by_core.get(bs[0].core) or []
        if old and bs[0].ann is not None:
            a, z = min(int(x["t0"]) for x in old), max(int(x["t1"]) for x in old)
            for k, b in enumerate(bs):
                b.t0 = a + (z - a) * k // len(bs)
                b.t1 = a + (z - a) * (k + 1) // len(bs)
                b.by = "placeholder"
                b.flags.append("reading_changed")
    timed_any = any(b.t0 is not None for b in beats)
    if any(b.t0 is None for b in beats):
        _fill_placeholders(beats, m.start, m.end)
        if timed_any:
            issues.append("有新字或改过的字还没打拍")
    if any("reading_changed" in b.flags for b in beats):
        issues.append("读音改了，拍数变了，需要重打")
    if not tr and not timed_any:
        state = "placeholder" if placeholder_eligible(doc, m) else "none"
    elif not timed_any:
        state = "placeholder"
    else:
        state = t0.get("state", "timed") if all(b.timed for b in beats) else "placeholder"
    if any(b.t0 is not None and (b.t0 < m.start - 20 or b.t1 > m.end + 20) for b in beats):
        issues.append("有拍落在这一行的时间外")
    if state == "suggested":
        issues.append("自动建议还没确认")
    return KView(m.id, beats, state, unit, kind, issues, copy.deepcopy((t0.get("effect") or {}).get("params") or {}))


def unfinished(doc: Document) -> list[tuple[str, list[str]]]:
    """Release check: lines whose karaoke is not finished (placeholder beats, changed readings, beats outside)."""
    idx = KIndex(doc)
    out = []
    for g in doc.groups:
        for mid in g.members:
            m = doc.members[mid]
            if not idx.has(m) and not placeholder_eligible(doc, m):
                continue
            v = effective(doc, m, idx=idx)
            if v.state == "none":
                continue
            issues = list(v.issues)
            if not v.complete and not issues:
                issues.append("自动建议还没确认" if v.state == "suggested" else "有占位拍（还没打）")
            if issues:
                out.append((mid, issues))
    return out


def _fill_placeholders(beats: list[Beat], start: int, end: int) -> None:
    """Untimed beats share the time between their timed neighbours evenly (marked placeholder)."""
    n = len(beats)
    i = 0
    while i < n:
        if beats[i].t0 is not None:
            i += 1
            continue
        j = i
        while j < n and beats[j].t0 is None:
            j += 1
        a = beats[i - 1].t1 if i > 0 else start
        z = beats[j].t0 if j < n else end
        if z < a:
            z = a
        cnt = j - i
        for k in range(cnt):
            b = beats[i + k]
            b.t0 = a + (z - a) * k // cnt
            b.t1 = a + (z - a) * (k + 1) // cnt
            b.by = "placeholder"
            b.flags.append("placeholder")
        i = j


def beats_to_track(m: Member, beats: list[Beat], *, state: str = "timed", unit: str = "syllable",
                   kid: Optional[str] = None, effect: Optional[dict] = None,
                   edits: Optional[list[dict]] = None) -> dict[str, Any]:
    """The stored form (FORMAT-v1 r2 §6.7): only the timing, keyed by characters."""
    out = []
    for b in beats:
        if b.t0 is None or any(x is None for x in b.core):
            continue
        out.append({"core": anchor_of(list(b.core)), "k": b.k, "sung": b.sung, "t0": max(0, int(b.t0)), "t1": max(0, int(b.t1)),
                    "by": b.by or "manual"})
        for flag in ("check", "auto_split"):
            if flag in b.flags:
                out[-1][flag] = True
        if b.link:
            out[-1]["link"] = copy.deepcopy(b.link)
    tr = {"id": kid or new_id("k_"), "member": m.id, "state": state, "unit": unit, "beats": out}
    if effect:
        tr["effect"] = effect
    if edits:
        tr["edits"] = copy.deepcopy(edits)
    return tr


# ------------------------------------------------------------------ import of existing \k tags
_K_NAME = re.compile(r"^(k|K|kf|ko)$")


def parse_k_runs(text: str) -> list[tuple[str, int]]:
    """Syllables of a tagged text: [(visible text, duration cs)] (text before the first tag has duration 0)."""
    syl: list[list] = []
    for tok in tokenize(text):
        if tok.kind == "tags":
            for t in tok.tags:
                if t.name in KARAOKE_TAGS:
                    v = t.num()
                    syl.append(["", int(v or 0)])
        elif tok.kind in ("text", "hardspace"):
            if not syl:
                syl.append(["", 0])
            syl[-1][0] += tok.raw if tok.kind == "text" else " "
    return [(s, d) for s, d in syl]


def has_k_tags(m: Member) -> bool:
    return any(r.kind == "tag" and re.search(r"\\(?:k|K|kf|ko)\d", r.text) for r in m.runs)


def import_from_tags(doc: Document, m: Member, unit: str = "syllable") -> list[Beat]:
    """Beats timed from the \\k tags already in the member's text (e.g. a fansub's karaoke)."""
    syl = parse_k_runs(m.text)
    beats = derive(doc, m, unit, edits_of(doc, m))
    if not syl or not beats:
        return beats
    _vt, ids = visible_map(m)
    pos = {ident: i for i, ident in enumerate(ids) if ident is not None}
    syl_of: list[int] = []                        # visible character → syllable index
    spans: list[tuple[int, int]] = []
    t = m.start
    for si, (txt, d) in enumerate(syl):
        spans.append((t, t + d * 10))
        syl_of.extend([si] * len(txt))
        t += d * 10
    slots: dict[int, list[Beat]] = {}
    for b in beats:
        slots.setdefault(b.slot, []).append(b)
    # the slots a syllable covers share its time evenly (\kf sweeps look the same), then each slot's beats
    by_syl: dict[int, list[int]] = {}
    for sl, bs in slots.items():
        c = next((pos[i] for i in bs[0].core if i in pos), None)
        if c is not None and c < len(syl_of):
            by_syl.setdefault(syl_of[c], []).append(sl)
    for si, sls in by_syl.items():
        a, z = spans[si]
        for j, sl in enumerate(sls):
            sa, sz = a + (z - a) * j // len(sls), a + (z - a) * (j + 1) // len(sls)
            bs = slots[sl]
            for k, b in enumerate(bs):
                b.t0 = sa + (sz - sa) * k // len(bs)
                b.t1 = sa + (sz - sa) * (k + 1) // len(bs)
                b.by = "import"
    return beats


# ------------------------------------------------------------------ export
def _cs(t_ms: float) -> int:
    return int((int(round(t_ms)) + 5) // 10)


def base_inserts(view: KView, ids_shown: list, start_cs: int, *,
                 display_starts=None, display_ends=None) -> dict[int, str]:
    """Tags to insert before visible characters of the *shown* text (display rules may hide characters)."""
    if view.kind == "none" or not view.beats:
        return {}
    pos = {ident: i for i, ident in enumerate(ids_shown) if ident is not None}
    slots: dict[int, list[Beat]] = {}
    for b in view.beats:
        slots.setdefault(b.slot, []).append(b)
    if display_starts and any(a < z for a, z in zip(display_starts, display_ends)):
        # Two source characters can share one displayed glyph (ｶ + ﾞ → ガ).
        # Group their display beats without touching the stored track or ruby.
        parent = {s: s for s in slots}
        def root(s):
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s
        occupied = {}
        for s, bs in slots.items():
            indices = {pos[ident] for b in bs for ident in b.core if ident in pos}
            for i in indices:
                for at in range(display_starts[i], display_ends[i+1]):
                    if at in occupied:
                        a, z = root(s), root(occupied[at])
                        parent[max(a, z)] = min(a, z)
                    else:
                        occupied[at] = s
        merged = {}
        for s, bs in slots.items():
            merged.setdefault(root(s), []).extend(bs)
        slots = merged
    out: dict[int, str] = {}
    cur = 0                                          # cs relative to the event start
    for s in sorted(slots):
        bs = slots[s]
        first = bs[0]
        if display_starts:
            def shown(identity):
                i = pos.get(identity)
                return i is not None and display_ends[i+1] > display_starts[i]
            if not any(shown(identity) for b in bs for identity in b.core):
                continue
        at = pos.get(first.head) if first.head is not None else None
        if at is not None and display_starts and display_ends[at+1] <= display_starts[at]:
            at = None
        if at is None:
            at = next((pos[i] for b in bs for i in b.core if i in pos and
                       (not display_starts or display_ends[pos[i]+1] > display_starts[pos[i]])), None)
        if at is None:
            continue
        a = max(0, _cs(min(b.t0 for b in bs)) - start_cs)
        z = max(a, _cs(max(b.t1 for b in bs)) - start_cs)
        tag = ""
        if a > cur:
            tag += f"{{\\{view.tag_kind}{a - cur}}}"
            cur = a
        tag += f"{{\\{view.tag_kind}{max(0, z - cur)}}}"
        cur = max(cur, z)
        out[at] = out.get(at, "") + tag
    return out


def ruby_text(view: KView, ann: str, piece: int, reading: str, start_cs: int) -> Optional[str]:
    """Karaoke text of one ruby piece: wait, then its beats (``{\\kf199}{\\kf24}い``)."""
    if view.kind == "none":
        return None
    segments = []
    for b in view.beats:
        if b.ann == ann and b.piece == piece:
            segments.append((b.t0, b.t1, b.r))
            continue
        # A merged beat can sing across pieces and bare kana. Allocate its span by sung characters.
        total = sum(len(p.sung) for p in b.parts) or 1
        used = 0
        merged = []
        for p in b.parts:
            end = used + len(p.sung)
            if p.ann == ann and p.piece == piece:
                a = b.t0 + (b.t1 - b.t0) * used // total
                z = b.t0 + (b.t1 - b.t0) * end // total
                if merged and merged[-1][1] == a and merged[-1][2][1] == p.r[0]:
                    sa, _sz, sr = merged[-1]
                    merged[-1] = (sa, z, (sr[0], p.r[1]))
                else:
                    merged.append((a, z, p.r))
            used = end
        segments.extend(merged)
    if not segments:
        return None
    out = []
    cur = 0
    used = 0
    for t0, t1, r in segments:
        a = max(0, _cs(t0) - start_cs)
        z = max(a, _cs(t1) - start_cs)
        if a > cur:
            out.append(f"{{\\{view.tag_kind}{a - cur}}}")
            cur = a
        r0, r1 = r or (used, len(reading))
        out.append(f"{{\\{view.tag_kind}{max(0, z - cur)}}}{reading[r0:r1]}")
        used = r1
        cur = max(cur, z)
    if used < len(reading):
        out.append(reading[used:])
    return "".join(out)
