"""Editing operations on the working document, with undo/redo.

Every operation runs inside a :class:`Transaction` that records the
pre-image of everything it touches (members, groups, comments, annotations,
the group order, tombstones).  Undo restores those pre-images exactly; redo
re-applies the post-images.  Operations never reorder text of one language:
pairing changes move a *boundary* between neighbouring groups, so the
original-language sequence keeps its source order (FORMAT-v1 §7).

Deleting source text writes a tombstone (FORMAT-v1 §6.5) so that later merges
and refreshes cannot bring it back.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional

from ..model.document import Annotation, Comment, Document, Group, Member, Run, Tombstone
from ..languages import get_language
from ..model.ids import new_id

if TYPE_CHECKING:
    from .autotime_doc import AutoTimePlan
    from .convert import ConvertPlan
    from .passages import ApplySummary, PassagePlan
    from .position import VerticalResult

SENT_END = get_language("ja").sentence_punctuation
SPACES = " 　\t"
AUTO_KARAOKE_MIN_MS = 100  # Clipping below a perceptible beat needs review.
_MISSING = object()


def _snap(x: Any) -> Any:
    """Deep copy, except the 'did not exist' marker (its identity matters)."""
    return x if x is _MISSING else copy.deepcopy(x)


def _respread_squeezed(beats, low: int, high: float, minimum: int) -> None:
    """Beats squeezed against a clipping bound get room again (stored beat dicts, in time order).

    Clipping clamps each edge on its own, so beats pushed past a neighbour's boundary collapse onto it (zero
    length, invisible).  The squeezed beats get ``minimum`` ms each from the bound, first out of the gap before
    the next beat; only when that gap is too small the next beat gives way (keeping ``minimum`` if it can,
    otherwise everything shares its span equally).  All of them stay flagged for review."""
    timed = [b for b in beats if b.get("t0") is not None and b.get("t1") is not None]
    for run, bound, sign in ((timed, low, 1), (timed[::-1], high, -1)):
        near, far = ("t0", "t1") if sign > 0 else ("t1", "t0")
        n = 0
        while n < len(run) and run[n]["t1"] - run[n]["t0"] < minimum and run[n][near] == bound:
            n += 1
        if not n or n == len(run):
            continue
        need = n * minimum
        if abs(run[n][near] - bound) >= need:
            share = minimum                                   # the gap before the next beat is enough
        elif abs(run[n][far] - bound) >= need + minimum:
            share = minimum
            run[n][near] = bound + sign * need
        else:
            share = abs(run[n][far] - bound) // (n + 1)
            run[n][near] = bound + sign * n * share
            run[n]["check"] = True
        edge = bound
        for b in run[:n]:
            b[near] = edge
            edge += sign * share
            b[far] = edge
            b["check"] = True


# ------------------------------------------------------------------ transactions
@dataclass
class Transaction:
    doc: Document
    label: str
    op: str = field(default_factory=lambda: new_id("op_"))
    pre_members: dict[str, Any] = field(default_factory=dict)
    pre_groups: dict[str, Any] = field(default_factory=dict)
    pre_comments: dict[str, Any] = field(default_factory=dict)
    pre_annotations: dict[str, Any] = field(default_factory=dict)
    pre_karaoke: dict[str, Any] = field(default_factory=dict)
    pre_order: Optional[list[str]] = None
    pre_tombstones: Optional[list[Tombstone]] = None
    pre_rules: Optional[list[dict]] = None
    post: Optional[dict[str, Any]] = None
    keeps_text: bool = False          # no member text changes (e.g. dialogue ↔ song): listeners skip reading refresh
    people_changed: bool = False      # character readings are captured in this transaction
    reading_rules_changed: bool = False  # invalidate cached engines on do/undo/redo
    pre_external: list[tuple[Callable[[], Any], Callable[[Any], None], Any]] = field(default_factory=list)

    def external(self, get: Callable[[], Any], put: Callable[[Any], None]) -> None:
        """Project-level state participating in this undo step."""
        self.pre_external.append((get, put, copy.deepcopy(get())))

    def member(self, mid: str) -> Optional[Member]:
        if mid not in self.pre_members:
            self.pre_members[mid] = _snap(self.doc.members.get(mid, _MISSING))
        return self.doc.members.get(mid)

    def group(self, gid: str) -> Optional[Group]:
        if gid not in self.pre_groups:
            try:
                g = self.doc.group(gid)
            except KeyError:
                g = None
            self.pre_groups[gid] = _snap(g if g is not None else _MISSING)
        try:
            return self.doc.group(gid)
        except KeyError:
            return None

    def comment(self, cid: str) -> Optional[Comment]:
        if cid not in self.pre_comments:
            self.pre_comments[cid] = _snap(self.doc.comments.get(cid, _MISSING))
        return self.doc.comments.get(cid)

    def annotation(self, aid: str) -> Optional[Annotation]:
        if aid not in self.pre_annotations:
            self.pre_annotations[aid] = _snap(self.doc.annotations.get(aid, _MISSING))
        return self.doc.annotations.get(aid)

    def track(self, kid: str) -> Optional[dict]:
        """A karaoke track (FORMAT-v1 r2 §6.7) — recorded for undo before it is changed."""
        if kid not in self.pre_karaoke:
            self.pre_karaoke[kid] = _snap(self.doc.karaoke.get(kid, _MISSING))
        return self.doc.karaoke.get(kid)

    def set_track(self, t: dict) -> None:
        self.track(t["id"])
        self.doc.karaoke[t["id"]] = t

    def remove_track(self, kid: str) -> None:
        self.track(kid)
        self.doc.karaoke.pop(kid, None)

    def order(self) -> list[Group]:
        if self.pre_order is None:
            self.pre_order = [g.id for g in self.doc.groups]
            for g in self.doc.groups:
                self.group(g.id)
        return self.doc.groups

    def tombstones(self) -> list[Tombstone]:
        if self.pre_tombstones is None:
            self.pre_tombstones = list(self.doc.tombstones)
        return self.doc.tombstones

    def rules(self) -> list[dict]:
        """The episode's rules (e.g. reading rules) — recorded for undo before they change."""
        if self.pre_rules is None:
            self.pre_rules = copy.deepcopy(self.doc.rules)
        return self.doc.rules

    # -- bookkeeping for new objects
    def add_member(self, m: Member) -> None:
        self.pre_members.setdefault(m.id, _MISSING)
        m.start, m.end = max(0, m.start), max(0, m.end)
        self.doc.members[m.id] = m

    def add_comment(self, c: Comment) -> None:
        self.pre_comments.setdefault(c.id, _MISSING)
        self.doc.comments[c.id] = c

    def add_annotation(self, a: Annotation) -> None:
        self.pre_annotations.setdefault(a.id, _MISSING)
        self.doc.annotations[a.id] = a

    def capture_post(self) -> None:
        d = self.doc
        # Normalize all touched working times before recording the redo image.
        for mid in self.pre_members:
            m = d.members.get(mid)
            if m is not None:
                m.start, m.end = max(0, m.start), max(0, m.end)
        for kid in self.pre_karaoke:
            for beat in d.karaoke.get(kid, {}).get("beats", ()):
                for edge in ("t0", "t1"):
                    if beat.get(edge) is not None:
                        beat[edge] = max(0, beat[edge])
        groups_by_id = {g.id: g for g in d.groups}
        self.post = {
            "members": {k: _snap(d.members.get(k, _MISSING)) for k in self.pre_members},
            "groups": {k: _snap(groups_by_id.get(k, _MISSING)) for k in self.pre_groups},
            "comments": {k: _snap(d.comments.get(k, _MISSING)) for k in self.pre_comments},
            "annotations": {k: _snap(d.annotations.get(k, _MISSING)) for k in self.pre_annotations},
            "karaoke": {k: _snap(d.karaoke.get(k, _MISSING)) for k in self.pre_karaoke},
            "rules": copy.deepcopy(d.rules) if self.pre_rules is not None else None,
            "order": [g.id for g in d.groups] if self.pre_order is not None else None,
            "tombstones": list(d.tombstones) if self.pre_tombstones is not None else None,
            "external": [copy.deepcopy(get()) for get, _put, _pre in self.pre_external],
        }

    def _apply(self, members, groups, comments, annotations, order, tombstones, karaoke=None, rules=None,
               external=None) -> None:
        d = self.doc
        if rules is not None:
            d.rules = copy.deepcopy(rules)
        for k, v in (karaoke or {}).items():
            if v is _MISSING:
                d.karaoke.pop(k, None)
            else:
                d.karaoke[k] = copy.deepcopy(v)
        for k, v in members.items():
            if v is _MISSING:
                d.members.pop(k, None)
            else:
                d.members[k] = copy.deepcopy(v)
        for k, v in comments.items():
            if v is _MISSING:
                d.comments.pop(k, None)
            else:
                d.comments[k] = copy.deepcopy(v)
        for k, v in annotations.items():
            if v is _MISSING:
                d.annotations.pop(k, None)
            else:
                d.annotations[k] = copy.deepcopy(v)
        current = {g.id: g for g in d.groups}
        for k, v in groups.items():
            if v is _MISSING:
                current.pop(k, None)
            else:
                current[k] = copy.deepcopy(v)
        ids = order if order is not None else [g.id for g in d.groups]
        d.groups = [current[i] for i in ids if i in current]
        if tombstones is not None:
            d.tombstones = list(tombstones)
        for (_get, put, _pre), value in zip(self.pre_external, external or []):
            put(copy.deepcopy(value))
        d.structure_changed()
        d.revision += 1

    def undo(self) -> None:
        self._apply(self.pre_members, self.pre_groups, self.pre_comments, self.pre_annotations,
                    self.pre_order, self.pre_tombstones, self.pre_karaoke, self.pre_rules,
                    [pre for _get, _put, pre in self.pre_external])

    def redo(self) -> None:
        p = self.post or {}
        self._apply(p["members"], p["groups"], p["comments"], p["annotations"], p["order"], p["tombstones"],
                    p.get("karaoke"), p.get("rules"), p.get("external"))

    def touched_groups(self) -> set[str]:
        return set(self.pre_groups) | {getattr(m, "group", None) for m in self.pre_members.values() if m is not _MISSING} - {None}


class History:
    """Undo/redo stack.  ``run(label, fn)`` executes fn(tx) as one undoable step."""

    def __init__(self, doc: Document, limit: int = 500) -> None:
        self.doc = doc
        self.done: list[Transaction] = []
        self.undone: list[Transaction] = []
        self.limit = limit
        self.listeners: list[Callable[[Transaction, str], None]] = []

    def run(self, label: str, fn: Callable[[Transaction], Any]) -> Any:
        tx = Transaction(self.doc, label)
        try:
            result = fn(tx)
        except Exception:
            tx.undo()           # leave the document exactly as before
            raise
        tx.capture_post()
        self.doc.revision += 1
        self.done.append(tx)
        del self.done[:-self.limit]
        self.undone.clear()
        for cb in self.listeners:
            cb(tx, "do")
        return result

    def undo(self) -> Optional[Transaction]:
        if not self.done:
            return None
        tx = self.done.pop()
        tx.undo()
        self.undone.append(tx)
        for cb in self.listeners:
            cb(tx, "undo")
        return tx

    def redo(self) -> Optional[Transaction]:
        if not self.undone:
            return None
        tx = self.undone.pop()
        tx.redo()
        self.done.append(tx)
        for cb in self.listeners:
            cb(tx, "redo")
        return tx


# ------------------------------------------------------------------ run helpers
def _normalize_runs(runs: list[Run], source_text: Callable[[str, int], str],
                    deleted: Optional[Callable[[str, int, int, int], bool]] = None) -> list[Run]:
    """Merge neighbouring runs of one source event when only whitespace separates them
    (the whitespace is source text too, so it becomes part of the run) — unless the user
    deleted that whitespace (``deleted`` reports a tombstone over the gap)."""
    out: list[Run] = []
    for r in runs:
        if out and r.kind == "text" and out[-1].kind == "text" and r.src is not None and out[-1].src == r.src \
                and out[-1].e == r.e:
            text = source_text(r.src, r.e)
            gap = text[out[-1].b:r.a]
            if out[-1].b <= r.a and gap.strip(SPACES) == "" and "\n" not in gap and "\\N" not in gap \
                    and not (gap and deleted is not None and deleted(r.src, r.e, out[-1].b, r.a)):
                last = out[-1]
                out[-1] = last.copy(text=text[last.a:r.b], b=r.b)
                continue
        if out and r.kind == "text" and out[-1].kind == "text" and r.ins is not None and out[-1].ins == r.ins \
                and out[-1].b == r.a:
            last = out[-1]
            out[-1] = last.copy(text=last.text + r.text, b=r.b)
            continue
        if out and r.kind == "sep" and out[-1].kind == "sep":
            continue
        out.append(r)
    while out and out[0].kind == "sep":
        out.pop(0)
    while out and out[-1].kind == "sep":
        out.pop()
    return out


def _visible_index(runs: list[Run]) -> list[tuple[int, int]]:
    """(run index, offset in run) for every visible character (tags skipped)."""
    out = []
    for ri, r in enumerate(runs):
        if r.kind == "tag":
            continue
        for k in range(len(r.text)):
            out.append((ri, k))
    return out


def split_runs(runs: list[Run], pos: int) -> tuple[list[Run], list[Run]]:
    """Split runs at visible position ``pos`` (tags before the cut stay on the left)."""
    left: list[Run] = []
    right: list[Run] = []
    seen = 0
    for r in runs:
        if r.kind == "tag":
            (left if seen < pos or (seen == pos and not right and pos > 0) else right).append(r)
            continue
        n = len(r.text)
        if seen + n <= pos:
            left.append(r)
        elif seen >= pos:
            right.append(r)
        else:
            k = pos - seen
            if r.kind == "sep":
                left.append(r.copy(text=r.text[:k]))
                right.append(r.copy(text=r.text[k:]))
            else:
                left.append(r.slice(0, k))
                right.append(r.slice(k, n))
        seen += n
    return left, right


def phrase_breaks(member: Member) -> list[int]:
    """Visible positions where a phrase starts (after spaces, separators or sentence punctuation)."""
    text = "".join(r.text if r.kind != "tag" else "" for r in member.runs)
    kinds = []
    for r in member.runs:
        if r.kind == "tag":
            continue
        kinds.extend([r.kind] * len(r.text))
    if get_language(member.lang).analyzer_id != "ja-cc":
        from .units import language_phrases

        out = [a for a, _b, kind in language_phrases(text, 0, len(text), member.lang) if a > 0 and kind != "first"]
        for i, kind in enumerate(kinds):
            if kind != "sep" or i > 0 and kinds[i - 1] == "sep":
                continue
            j = i
            while j < len(text) and (kinds[j] == "sep" or text[j] in SPACES):
                j += 1
            if i > 0 and j < len(text):
                out.append(j)
        return sorted(set(out))
    out = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in SPACES or kinds[i] == "sep":
            j = i
            while j < n and (text[j] in SPACES or kinds[j] == "sep"):
                j += 1
            if 0 < i and j < n:
                out.append(j)
            i = j
            continue
        if text[i] in SENT_END:
            j = i
            while j + 1 < n and text[j + 1] in SENT_END:
                j += 1
            if j + 1 < n and text[j + 1] not in SPACES and text[j + 1] not in "」』）)":
                out.append(j + 1)
            i = j + 1
            continue
        i += 1
    return sorted(set(out))


def _strip_edges(runs: list[Run]) -> list[Run]:
    """Drop leading/trailing whitespace characters of source runs (they stay in the source, not lost)."""
    runs = [r for r in runs]
    while runs and runs[0].kind in ("text", "sep") and runs[0].text.strip(SPACES) == "":
        runs.pop(0)
    while runs and runs[-1].kind in ("text", "sep") and runs[-1].text.strip(SPACES) == "":
        runs.pop()
    if runs and runs[0].kind == "text":
        t = runs[0].text
        k = len(t) - len(t.lstrip(SPACES))
        if k:
            runs[0] = runs[0].slice(k, len(t)) if runs[0].src or runs[0].ins else runs[0].copy(text=t[k:])
    if runs and runs[-1].kind == "text":
        t = runs[-1].text
        k = len(t) - len(t.rstrip(SPACES))
        if k:
            runs[-1] = runs[-1].slice(0, len(t) - k) if runs[-1].src or runs[-1].ins else runs[-1].copy(text=t[:len(t) - k])
    return runs


# ------------------------------------------------------------------ editor facade
class Editor:
    """All user-facing edits of one document.  ``sources`` maps source ids to Source objects
    (needed to rebuild runs from the read-only source text)."""

    def __init__(self, doc: Document, sources: dict, *, jp_sep: str = "  ", character_book=None,
                 reading_engine_factory=None, pinyin_words_factory=None) -> None:
        self.doc = doc
        self.sources = sources
        self.history = History(doc)
        self.jp_sep = jp_sep
        self.merge_note: Optional[str] = None
        self.character_book = character_book
        if character_book is not None:
            character_book.set_sources(sources)
        self.reading_engine_factory = reading_engine_factory
        self.pinyin_words_factory = pinyin_words_factory

    # -------------------------------------------------------------- speakers
    def same_label_candidates(self, mid: str, *, whole_group: bool = True,
                              shown_label: Optional[str] = None, preferred_lane: str = "original") -> list[str]:
        from .people import needs_speaker, normalize_speaker, shown_speaker_label

        member = self.doc.members[mid]
        label = normalize_speaker(shown_label if shown_label is not None else
            shown_speaker_label(self.doc, self.doc.group(member.group), preferred_lane, self.character_book))
        if not label or not any(needs_speaker(self.doc.group(member.group), m)
                                for m in self.doc.members_of(self.doc.group(member.group))):
            return []
        excluded = set(self.doc.group(member.group).members) if whole_group else {mid}
        book = self.character_book
        return [target for group in self.doc.groups for target in group.members
                if target not in excluded
                and (self.doc.members[target].speaker or {}).get("by") != "user"
                and normalize_speaker((self.doc.members[target].speaker or {}).get("label", "")) == label
                and needs_speaker(group, self.doc.members[target])
                and (book is None or not book.speaker_ignored(self.doc.members[target]))]

    def fill_down_candidates(self, mid: str, *, shown_label: Optional[str] = None,
                             preferred_lane: str = "original") -> list[str]:
        """Compute following groups against the original speaker, before any edits."""
        from .people import needs_speaker, normalize_speaker, shown_speaker_label

        book = self.character_book
        group = self.doc.group(self.doc.members[mid].group)
        def identity(g, label=None):
            members = [m for m in self.doc.members_of(g) if needs_speaker(g, m)
                       and (book is None or not book.speaker_ignored(m))]
            members.sort(key=lambda m: m.lane != preferred_lane)
            cid = next(((m.speaker or {}).get("character") for m in members
                        if (m.speaker or {}).get("character")), None)
            if cid:
                return ("character", cid)
            label = shown_speaker_label(self.doc, g, preferred_lane, book) if label is None else label
            return ("label", normalize_speaker(label)) if label else None

        if not any(needs_speaker(group, m) for m in self.doc.members_of(group)):
            return []
        previous = identity(group, shown_label)
        end = self.doc.group_time(group)[1]
        affected = []
        for candidate in self.doc.groups[self.doc.groups.index(group) + 1:]:
            members = self.doc.members_of(candidate)
            start, next_end = self.doc.group_time(candidate)
            if (not any(needs_speaker(candidate, m) for m in members) or start - end > 10000
                    or any((m.speaker or {}).get("by") == "user"
                           or (m.speaker or {}).get("character") == "nobody" for m in members)):
                break
            current = identity(candidate)
            if current == ("character", "crowd") or (current is not None and current != previous):
                break
            affected.append(candidate.id)
            end = next_end
        return affected

    def _assign_speaker(self, tx: Transaction, mids: Iterable[str], character: Optional[str], *,
                        whole_group: bool = True, same_label: bool = False, fill_down: bool = False,
                        shown_label: Optional[str] = None, preferred_lane: str = "original") -> list[str]:
        from .people import normalize_speaker, shown_speaker_label

        selected = list(dict.fromkeys(mids))
        labels = list(dict.fromkeys(shown_label if shown_label is not None else
            shown_speaker_label(self.doc, self.doc.group(self.doc.members[mid].group),
                                preferred_lane, self.character_book) for mid in selected)) if same_label else []
        candidates = list(dict.fromkeys(target for mid in selected
            for target in self.same_label_candidates(mid, whole_group=whole_group,
                shown_label=shown_label, preferred_lane=preferred_lane))) if same_label else []
        following = {gid for mid in selected for gid in self.fill_down_candidates(mid,
            shown_label=shown_label, preferred_lane=preferred_lane)} if fill_down else set()
        if whole_group:
            gids = {self.doc.members[mid].group for mid in selected}
            selected = [mid for g in self.doc.groups if g.id in gids for mid in g.members]
            candidate_gids = {self.doc.members[mid].group for mid in candidates}
            candidates = [mid for g in self.doc.groups if g.id in candidate_gids for mid in g.members
                          if (self.doc.members[mid].speaker or {}).get("by") != "user"]
        selected = list(dict.fromkeys([*selected, *candidates,
            *(mid for g in self.doc.groups if g.id in following for mid in g.members)]))
        book = self.character_book
        if book is not None and (labels or character == "nobody"):
            if not any(get == book.snapshot for get, _put, _pre in tx.pre_external):
                tx.external(book.snapshot, book.restore)
            for label in labels:
                key = normalize_speaker(label)
                if not key:
                    continue
                book.not_person = [value for value in book.not_person if normalize_speaker(value) != key]
                if character == "nobody":
                    book.crowd = [value for value in book.crowd if normalize_speaker(value) != key]
                    book.not_person.append(label)
                elif character == "crowd":
                    if not any(normalize_speaker(value) == key for value in book.crowd):
                        book.crowd.append(label)
                else:
                    book.crowd = [value for value in book.crowd if normalize_speaker(value) != key]
                    if character is None:
                        for ch in book.characters:
                            ch.aliases = [value for value in ch.aliases if normalize_speaker(value) != key]
                    else:
                        ch = book.character(character)
                        names = [*(ch.full_name(lang) for lang in ch.names), *ch.aliases]
                        if not any(normalize_speaker(value) == key for value in names):
                            ch.aliases.append(label)
        tx.keeps_text = True
        for mid in selected:
            m = tx.member(mid)
            m.speaker = {**(m.speaker or {"label": ""}), "character": character, "by": "user"}
            m.rev += 1
        if book is not None and (labels or character == "nobody"):
            book.link_document(self.doc, tx=tx)
        return selected

    def set_speaker(self, mids: Iterable[str], character: Optional[str], *, whole_group: bool = True,
                    same_label: bool = False, fill_down: bool = False,
                    shown_label: Optional[str] = None, preferred_lane: str = "original") -> int:
        if self.character_book is not None and character not in (None, "crowd", "nobody"):
            self.character_book.character(character)
        selected = list(dict.fromkeys(mids))
        for mid in selected:
            self.doc.members[mid]     # validate all IDs before entering the transaction

        def apply(tx: Transaction) -> int:
            assigned = self._assign_speaker(tx, selected, character, whole_group=whole_group,
                same_label=same_label, fill_down=fill_down, shown_label=shown_label, preferred_lane=preferred_lane)
            self._refresh_character_readings(tx, self.character_book, None if same_label else assigned)
            return len(assigned)

        return self.history.run("说话人", apply)

    def edit_characters(self, change: Callable, *, book=None, replacements: Optional[dict] = None):
        book = book if book is not None else self.character_book
        if book is None:
            raise ValueError("A project CharacterBook is required")

        def apply(tx: Transaction):
            tx.keeps_text = True
            tx.external(book.snapshot, book.restore)
            old_words = book.words()
            old_pinyin = {lang: book.pinyin_words(lang) for lang in {m.lang for m in self.doc.members.values()}
                          if get_language(lang).annotation_system == "pinyin"}
            old_refs = {ch.id: copy.deepcopy(ch.self_refs) for ch in book.characters}
            result = change(book)
            for m in self.doc.members.values():
                if m.speaker and m.speaker.get("character") in (replacements or {}):
                    tx.member(m.id)
                    m.speaker = {**m.speaker, "character": replacements[m.speaker["character"]]}
                    m.rev += 1
            book.link_document(self.doc, tx=tx)
            dictionary_changed = old_words != book.words() or any(
                words != book.pinyin_words(lang) for lang, words in old_pinyin.items())
            refs_changed = {ch.id for ch in book.characters if old_refs.get(ch.id) != ch.self_refs}
            mids = None if dictionary_changed else [m.id for m in self.doc.members.values()
                if m.id in tx.pre_members or (m.speaker or {}).get("character") in refs_changed]
            self._refresh_character_readings(tx, book, mids)
            return result

        return self.history.run("人物表", apply)

    def relink_speakers(self, book=None) -> int:
        book = book if book is not None else self.character_book
        if book is None:
            raise ValueError("A project CharacterBook is required")

        def apply(tx: Transaction) -> int:
            tx.keeps_text = True
            tx.external(book.snapshot, book.restore)
            changed = book.link_document(self.doc, tx=tx)
            self._refresh_character_readings(tx, book)
            return changed

        return self.history.run("说话人", apply)

    def merge_characters(self, source: str, target: str, book=None) -> None:
        book = book if book is not None else self.character_book
        if book is None:
            raise ValueError("A project CharacterBook is required")
        book.merge_characters(source, target, editor=self)

    def _refresh_character_readings(self, tx: Transaction, book, mids=None) -> None:
        """Book/speaker changes refresh automatic annotations in the same undo step."""
        if book is None:
            return
        from .readings import kept_positions, suggest_readings, visible_map
        from ..reading.dictionary import ReadingDict
        from ..reading.engine import ReadingEngine

        if self.reading_engine_factory is not None:
            engine = self.reading_engine_factory()
        else:
            rd = ReadingDict()
            rd.merge_character_book(book)
            engine = ReadingEngine(rd, episode=self.doc.episode)
        tx.people_changed = True
        for mid in (self.doc.members if mids is None else mids):
            m = self.doc.members[mid]
            if get_language(m.lang).annotation_system == "pinyin" and m.role not in ("sign", "info", "comment"):
                from .pinyin_notes import episode_rules

                words = self.pinyin_words_factory(m.lang) if self.pinyin_words_factory is not None else (
                    book.pinyin_words(m.lang) if any(a.kind == "pinyin" and a.member == mid
                                                    for a in self.doc.annotations.values()) else None)
                if words is not None:
                    self._tx_pinyin(tx, [mid], episode_rules(self.doc, words))
                continue
            if get_language(m.lang).annotation_system != "ruby" or m.role in ("sign", "info", "comment"):
                continue
            _vt, ids = visible_map(m)
            keep, covered = kept_positions(self.doc, m, ids)
            previous = [ann for ann in self.doc.annotations.values()
                        if ann.member == mid and ann.kind == "ruby" and ann not in keep]
            source = next((self.sources[r.src] for r in m.runs if r.src in self.sources), None)
            suggested = list(suggest_readings(m, engine, source=source, covered=covered))
            # Stable suggestions keep their identities and the preview layout cache.
            def signature(ann):
                return {key: value for key, value in vars(ann).items() if key != "id"}
            if [signature(a) for a in previous] == [signature(a) for a in suggested]:
                continue
            tx.member(mid)
            for ann in previous:
                tx.annotation(ann.id)
                del self.doc.annotations[ann.id]
            for ann in suggested:
                tx.add_annotation(ann)

    # -------------------------------------------------------------- utilities
    def _src_text(self, sid: str, e: int) -> str:
        return self.sources[sid].events[e].text

    def _deleted(self, sid: str, e: int, a: int, b: int) -> bool:
        return any(t.src == sid and t.e == e and t.a < b and a < t.b for t in self.doc.tombstones)

    def _norm(self, runs: list[Run]) -> list[Run]:
        return _normalize_runs(_strip_edges(runs), self._src_text, self._deleted)

    def lane_member(self, g: Group, lane: str) -> Optional[Member]:
        ms = [m for m in self.doc.members_of(g) if m.lane == lane and m.role not in ("comment",)]
        return ms[-1] if ms else None

    def _join(self, a: list[Run], b: list[Run]) -> list[Run]:
        if a and b:
            return self._norm(a + [Run(self.jp_sep, "sep")] + b)
        return self._norm(a + b)

    def _insert_by_source(self, runs: list[Run], extra: list[Run]) -> list[Run]:
        """Put ``extra`` (source text) back at its source position among ``runs``."""
        key = _first_key(extra)
        if key is None:
            return self._join(runs, extra)
        k = len(runs)
        for i, r in enumerate(runs):
            if r.src == key[0] and r.kind != "sep" and (r.src, r.e, r.a) > key:
                k = i
                break
        left, right = runs[:k], runs[k:]
        out = self._join(left, extra) if left else list(extra)
        return self._join(out, right) if right else self._norm(out)

    def neighbours(self, gid: str) -> tuple[Optional[Group], Optional[Group]]:
        i = self.doc.group_index(gid)
        prev = next((g for g in reversed(self.doc.groups[:i]) if g.kind == "dialogue"), None)
        nxt = next((g for g in self.doc.groups[i + 1:] if g.kind == "dialogue"), None)
        return prev, nxt

    # -------------------------------------------------------------- pairing: move a boundary
    def move_phrase(self, gid: str, lane: str, direction: str, *, at: Optional[int] = None) -> bool:
        """Move text across the boundary with the neighbouring dialogue group.

        direction: 'head_to_prev' (first phrase of this group → previous group),
                   'tail_to_next' (last phrase → next group),
                   'take_from_prev' (last phrase of previous group → this group),
                   'take_from_next' (first phrase of next group → this group).
        ``at``: a visible cursor position instead of a phrase boundary (Ctrl+Shift+↑/↓)."""
        g = self.doc.group(gid)
        prev, nxt = self.neighbours(gid)
        if direction == "head_to_prev":
            return self._move(g, prev, lane, from_head=True, at=at) if prev else False
        if direction == "tail_to_next":
            return self._move(g, nxt, lane, from_head=False, at=at) if nxt else False
        if direction == "take_from_prev":
            return self._move(prev, g, lane, from_head=False, at=at) if prev else False
        if direction == "take_from_next":
            return self._move(nxt, g, lane, from_head=True, at=at) if nxt else False
        raise ValueError(direction)

    def _move(self, src_g: Group, dst_g: Group, lane: str, *, from_head: bool, at: Optional[int]) -> bool:
        m = self.lane_member(src_g, lane)
        if m is None:
            return False
        vis = len("".join(r.text for r in m.runs if r.kind != "tag"))
        if at is None:
            breaks = phrase_breaks(m)
            if not breaks:
                cut = vis if from_head else 0          # the whole text moves
            else:
                cut = breaks[0] if from_head else breaks[-1]
        else:
            cut = max(0, min(vis, at))
        if (from_head and cut == 0) or (not from_head and cut == vis):
            return False

        from .textformat import spans_for

        def apply(tx: Transaction) -> None:
            mm = tx.member(m.id)
            left, right = split_runs(mm.runs, cut)
            moved, kept = (left, right) if from_head else (right, left)
            moved, kept = self._norm(moved), self._norm(kept)
            if not moved:
                return
            dm = self.lane_member(dst_g, lane)
            if dm is None:
                tx.group(dst_g.id)
                s, e = self.doc.group_time(dst_g)
                dm = Member(new_id("m_"), dst_g.id, mm.lang, lane, mm.role if mm.role != "top" else "dialogue", s, e,
                            layer=mm.layer, style=mm.style, speaker=copy.deepcopy(mm.speaker))
                tx.add_member(dm)
                tg = tx.group(dst_g.id)
                tg.members.append(dm.id)
            else:
                dm = tx.member(dm.id)
            dm.runs = self._join(moved, dm.runs) if not from_head else self._join(dm.runs, moved)
            dm.format = spans_for(dm, dm.format + mm.format)
            dm.rev += 1
            # annotations follow their characters (identity anchored); re-home them
            ids = {ident for ident, *_ in dm.identities() if ident is not None}
            for a in list(self.doc.annotations.values()):
                if a.member == m.id and _anchor_ids(a) <= ids:
                    tx.annotation(a.id).member = dm.id
            # hidden text spoken on the moved side of the cut moves with it (keeps the source order)
            from ..model.validate import comment_key
            sg, dg = tx.group(src_g.id), tx.group(dst_g.id)
            cut_key = _first_key(moved) if not from_head else _last_key(moved)
            if cut_key is not None:
                for cid in list(sg.comments):
                    ck = comment_key(self.doc.comments[cid])
                    if ck is None or ck[0] != cut_key[0]:
                        continue
                    on_moved_side = ck >= cut_key if not from_head else ck < cut_key
                    if on_moved_side:
                        com = tx.comment(cid)
                        sg.comments.remove(cid)
                        com.group = dg.id
                        com.position = {}
                        if from_head:
                            dg.comments.append(cid)
                        else:
                            dg.comments.insert(0, cid)
            if kept:
                mm.runs = kept
                mm.format = spans_for(mm, mm.format)
                mm.rev += 1
            else:
                tg = tx.group(src_g.id)
                tg.members.remove(mm.id)
                del self.doc.members[mm.id]
                for c in list(tg.comments):
                    com = tx.comment(c)
                    if com.position.get("before") == mm.id or com.position.get("after") == mm.id:
                        com.position = {"only": True}
            for gid in (src_g.id, dst_g.id):
                tx.group(gid).state = "auto"
            self.doc.structure_changed()

        self.history.run("移动分界", apply)
        return True

    # -------------------------------------------------------------- merge / split / link
    def merge_with_next(self, gid: str) -> Optional[str]:
        """Join this group with the next dialogue group (all members kept, original text joined in order)."""
        from .textformat import merge_formats

        self.merge_note = None
        g = self.doc.group(gid)
        _prev, nxt = self.neighbours(gid)
        if nxt is None:
            return None

        def apply(tx: Transaction) -> None:
            order = tx.order()
            ga, gb = tx.group(g.id), tx.group(nxt.id)
            ja, jb = self.lane_member(ga, "original"), self.lane_member(gb, "original")
            if ja is not None and jb is not None:
                ma, mb = tx.member(ja.id), tx.member(jb.id)
                ma.format, conflict = merge_formats(ma, mb)
                if conflict:
                    self.merge_note = "合并：第二行的位置／动画设置没有保留"
                ma.runs = self._join(ma.runs, mb.runs)
                ma.start, ma.end = min(ma.start, mb.start), max(ma.end, mb.end)
                ma.rev += 1
                for a in self.doc.annotations.values():
                    if a.member == mb.id:
                        tx.annotation(a.id).member = ma.id
                for cid in gb.comments:
                    com = tx.comment(cid)
                    if com.position.get("before") == mb.id or com.position.get("after") == mb.id:
                        com.position = {"after": ma.id}
                gb.members.remove(mb.id)
                del self.doc.members[mb.id]
            for mid in gb.members:
                tx.member(mid).group = ga.id
            # translation members first in their own order, then the original member
            ga.members = [x for x in ga.members + gb.members]
            ga.members.sort(key=lambda mid: 0 if self.doc.members[mid].lane == "translation" else 1)
            for cid in gb.comments:
                tx.comment(cid).group = ga.id
            ga.comments += gb.comments
            ga.state = "auto"
            order.remove(next(x for x in order if x.id == gb.id))
            self.doc.structure_changed()

        self.history.run("合并", apply)
        return g.id

    def split(self, gid: str, *, jp_at: Optional[int] = None, cn_member_index: Optional[int] = None,
              cn_at: Optional[int] = None) -> Optional[str]:
        """Split a group in two.  Original text is cut at visible position ``jp_at``; translation
        members are distributed (first ``cn_member_index`` members stay) or one member's text is
        cut at ``cn_at`` (an explicit user decision; times are split proportionally)."""
        from .textformat import spans_for

        g = self.doc.group(gid)

        def apply(tx: Transaction) -> str:
            order = tx.order()
            ga = tx.group(g.id)
            gb = Group(new_id("g_"), ga.kind, [], [], "auto", {"by": "user", "op": "split"})
            tx.pre_groups.setdefault(gb.id, _MISSING)
            cn = [self.doc.members[m] for m in ga.members if self.doc.members[m].lane == "translation"]
            jp = self.lane_member(ga, "original")
            move_cn: list[str] = []
            if cn_at is not None and cn:
                mm = tx.member(cn[0].id)
                left, right = split_runs(mm.runs, cn_at)
                vis = max(1, len("".join(r.text for r in mm.runs if r.kind != "tag")))
                t_cut = mm.start + int((mm.end - mm.start) * cn_at / vis)
                nb = Member(new_id("m_"), gb.id, mm.lang, mm.lane, mm.role, t_cut, mm.end, layer=mm.layer,
                            style=mm.style, margins=list(mm.margins), runs=_strip_edges(right),
                            overrides=copy.deepcopy(mm.overrides))
                nb.format = spans_for(nb, mm.format)
                mm.runs, mm.end = _strip_edges(left), t_cut
                mm.format = spans_for(mm, mm.format)
                mm.rev += 1
                tx.add_member(nb)
                move_cn = [nb.id]
                gb.members.append(nb.id)
            elif cn_member_index is not None:
                move_cn = [m.id for m in cn[cn_member_index:]]
            if jp is not None and jp_at is not None:
                mj = tx.member(jp.id)
                left, right = split_runs(mj.runs, jp_at)
                left, right = self._norm(left), self._norm(right)
                if right:
                    t_cut = mj.end
                    if move_cn:
                        t_cut = min(self.doc.members[x].start for x in move_cn)
                    nj = Member(new_id("m_"), gb.id, mj.lang, mj.lane, mj.role, t_cut, mj.end, layer=mj.layer,
                                style=mj.style, speaker=copy.deepcopy(mj.speaker), runs=right,
                                overrides=copy.deepcopy(mj.overrides))
                    nj.format = spans_for(nj, mj.format)
                    tx.add_member(nj)
                    gb.members.append(nj.id)
                    mj.runs = left
                    mj.format = spans_for(mj, mj.format)
                    mj.end = max(mj.start, t_cut)
                    mj.rev += 1
                    ids_b = {ident for ident, *_ in nj.identities() if ident is not None}
                    for a in list(self.doc.annotations.values()):
                        if a.member == mj.id and _anchor_ids(a) <= ids_b:
                            tx.annotation(a.id).member = nj.id
            if jp is not None and jp_at is not None and gb.members:
                from ..model.validate import comment_key
                nj_first = next((_first_key(self.doc.members[x].runs) for x in gb.members
                                 if self.doc.members[x].lane == "original"), None)
                if nj_first is not None:
                    for cid in list(ga.comments):
                        ck = comment_key(self.doc.comments[cid])
                        if ck is not None and ck[0] == nj_first[0] and ck >= nj_first:
                            com = tx.comment(cid)
                            ga.comments.remove(cid)
                            com.group = gb.id
                            com.position = {}
                            gb.comments.append(cid)
            for mid in move_cn:
                if mid in ga.members:
                    ga.members.remove(mid)
                    tx.member(mid).group = gb.id
                    if mid not in gb.members:
                        gb.members.insert(0, mid)
            gb.members.sort(key=lambda mid: 0 if self.doc.members[mid].lane == "translation" else 1)
            for mid in gb.members:
                self.doc.members[mid].group = gb.id
            idx = next(i for i, x in enumerate(order) if x.id == ga.id)
            order.insert(idx + 1, gb)
            ga.state = "auto"
            self.doc.structure_changed()
            return gb.id

        return self.history.run("拆分", apply)

    def unlink(self, gid: str) -> Optional[str]:
        """Separate the languages of a group into two adjacent groups (order unchanged)."""
        g = self.doc.group(gid)
        jp = [m for m in g.members if self.doc.members[m].lane == "original"]
        cn = [m for m in g.members if self.doc.members[m].lane == "translation"]
        if not jp or not cn:
            return None

        def apply(tx: Transaction) -> str:
            order = tx.order()
            ga = tx.group(g.id)
            gb = Group(new_id("g_"), ga.kind, list(jp), [], "auto", {"by": "user", "op": "unlink"})
            tx.pre_groups.setdefault(gb.id, _MISSING)
            ga.members = list(cn)
            for mid in jp:
                tx.member(mid).group = gb.id
            for cid in list(ga.comments):
                com = tx.comment(cid)
                if com.lang == self.doc.members[jp[0]].lang:
                    ga.comments.remove(cid)
                    gb.comments.append(cid)
                    com.group = gb.id
            idx = next(i for i, x in enumerate(order) if x.id == ga.id)
            # the language whose text starts first stays first
            jp_start = min(self.doc.members[m].start for m in jp)
            cn_start = min(self.doc.members[m].start for m in cn)
            order.insert(idx + (1 if cn_start <= jp_start else 0), gb)
            ga.state = "auto"
            self.doc.structure_changed()
            return gb.id

        return self.history.run("解除配对", apply)

    def link_with_next(self, gid: str) -> Optional[str]:
        """Join two adjacent groups (Link): same as merge, offered for single-language groups."""
        return self.merge_with_next(gid)

    # -------------------------------------------------------------- text
    def duplicate_groups(self, gids: Iterable[str]) -> list[str]:
        """Copy selected groups in document order; new characters are independent insertions."""
        from .duplicate import apply

        requested = set(gids)
        selected = {g.id for g in self.doc.groups if g.id in requested}
        if not selected:
            return []
        return self.history.run("复制一组", lambda tx: apply(tx, selected))

    def add_missing_language(self, texts: Mapping[str, str], lane: str, *, languages=None, styles=None,
                             allow_empty: bool = False) -> list[str]:
        """Insert missing lane members and automatic readings in one undo step.

        Empty editor rows are opt-in; ordinary paste/add dialogs still skip them.
        """
        from ..languages import DEFAULT_LANGUAGES, lane_label
        from .merge import MergeSettings, song_family
        from .editor_text import editing_runs
        from .readings import suggest_readings
        from .pinyin_notes import episode_rules
        from ..reading.engine import ReadingEngine

        if lane not in DEFAULT_LANGUAGES:
            raise ValueError(lane)
        languages = {**DEFAULT_LANGUAGES, **(languages or {})}
        styles = {**MergeSettings().styles, **(styles or {})}
        for gid in texts:
            self.doc.group(gid)
        pending = [(g, texts[g.id]) for g in self.doc.groups if g.id in texts
                   and (allow_empty or texts[g.id].strip())
                   and self.lane_member(g, lane) is None]
        if not pending:
            return []
        lang = languages[lane]
        engine = (self.reading_engine_factory() if self.reading_engine_factory else ReadingEngine()) \
            if get_language(lang).annotation_system == "ruby" and any(text.strip() for _g, text in pending) else None

        def apply(tx: Transaction) -> list[str]:
            tx.keeps_text = True       # readings are already captured here, including their identities
            inserted = []
            for g, text in pending:
                siblings = [m for m in self.doc.members_of(g) if m.role != "comment"] or list(self.doc.members_of(g))
                if not siblings:
                    raise ValueError("这一组没有可以取时间的字幕")
                ref = siblings[-1]
                role = "song" if g.kind == "song" else (
                    ref.role if g.kind == "dialogue" else "info" if g.kind == "info" else "sign")
                family = song_family(ref.style, ref.start, max((m.end for m in self.doc.members.values()), default=0))
                fallback = ref.style if g.kind in ("sign", "info") else styles[f"{lane}/dialogue"]
                style = styles.get(f"{lane}/{role}", fallback).format(fam=family)
                m = Member(new_id("m_"), g.id, lang, lane, role, ref.start, ref.end,
                           layer=ref.layer if role == "song" else int(lane == "translation"), style=style,
                           margins=[0, 0, 0], runs=editing_runs(text, new_id("i_")))
                tx.add_member(m)
                tx.group(g.id)
                at = next((i for i, mid in enumerate(g.members) if self.doc.members[mid].lane == "translation"), len(g.members)) \
                    if lane == "original" else len(g.members)
                g.members.insert(at, m.id)
                if engine is not None and role in ("dialogue", "top", "alt", "song"):
                    for ann in suggest_readings(m, engine):
                        tx.add_annotation(ann)
                if text.strip() and self.pinyin_words_factory is not None:
                    words = self.pinyin_words_factory(lang) if get_language(lang).annotation_system == "pinyin" else None
                    if words is not None:
                        self._tx_pinyin(tx, [m.id], episode_rules(self.doc, words))
                inserted.append(m.id)
            self.doc.structure_changed()
            return inserted

        return self.history.run("补上" + lane_label(lane, languages), apply)

    def delete_range(self, mid: str, a: int, b: int) -> None:
        """Delete visible characters [a, b) of a member; source characters get tombstones."""
        old = self.visible_text(mid)
        self.replace_text(mid, old[:a] + old[b:], span=(a, b), _history_label="删除文字")

    def visible_text(self, mid: str) -> str:
        return "".join(r.text for r in self.doc.members[mid].runs if r.kind != "tag")

    def replace_editor_text(self, mid: str, text: str) -> bool:
        """Commit text-box input, including editable hidden glosses, in one transaction."""
        from .editor_text import editing_text, split_private_marks

        if editing_text(self.doc.members[mid]) == text:
            return False
        ordinary, marks = split_private_marks(text)
        return self.replace_text(mid, ordinary, _gloss_marks=marks)

    def replace_text(self, mid: str, new_text: str, *, span: Optional[tuple[int, int]] = None,
                     _gloss_marks: Optional[list[tuple[int, str]]] = None,
                     _history_label: str = "编辑文字") -> bool:
        """Replace a member's visible text with ``new_text`` as ONE undoable edit.  Unchanged
        characters keep their identity (provenance, readings); removed source characters get
        tombstones; typed characters become an insertion run. ``span`` constrains the diff to
        editing-text offsets [a, b), preserving every character outside an explicit selection."""
        import difflib

        old = self.visible_text(mid)
        if old == new_text and _gloss_marks is None:
            return False
        if span is None:
            ops_ = difflib.SequenceMatcher(None, old, new_text, autojunk=False).get_opcodes()
            position_ops = ops_
        else:
            a, b = span
            end = len(new_text) - (len(old) - b)
            if not (0 <= a <= b <= len(old)) or old[:a] != new_text[:a] or old[b:] != new_text[end:]:
                raise ValueError("Replacement must stay inside the selected span")
            # Repeated text outside a selected range must keep its own identities.
            ops_ = [(tag, a + i, a + j, a + k, a + l) for tag, i, j, k, l in
                    difflib.SequenceMatcher(None, old[a:b], new_text[a:end], autojunk=False).get_opcodes()]
            position_ops = [("equal", 0, a, 0, a), *ops_, ("equal", b, len(old), end, len(new_text))]

        def tag_position(pos: int) -> int:
            for tag, i1, i2, j1, j2 in position_ops:
                if i1 <= pos <= i2:
                    if tag == "equal":
                        return j1 + pos - i1
                    if tag in ("insert", "delete"):
                        return j1
                    return j1 + (pos - i1) * (j2 - j1) // (i2 - i1)
            return len(new_text)

        def apply(tx: Transaction) -> None:
            from .editor_text import is_private_run
            from .render_origin import run_source, source_id

            mm = tx.member(mid)
            sid = source_id(mm)
            origin = next((r for r in mm.runs if run_source(mm, r) == sid), None) if sid else None
            if _gloss_marks is not None:
                mm.runs = [r for r in mm.runs if not is_private_run(r)]
            # The text boxes hide ASS controls. Editing their visible text must
            # not delete those controls or turn them into source tombstones.
            tags, runs, offset = {}, [], 0
            for run in mm.runs:
                if run.kind == "tag" and not is_private_run(run):
                    tags.setdefault(tag_position(offset), []).append(run)
                else:
                    runs.append(run)
                    if run.kind != "tag":
                        offset += len(run.text)
            mm.runs = runs
            tx.keeps_text = old == new_text
            stones = tx.tombstones()
            now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            ins_id = new_id("i_")
            ins_off = 0
            pieces = {}
            for tag, i1, i2, j1, j2 in ops_:
                if tag in ("replace", "insert"):
                    pieces[(i1, j1)] = new_text[j1:j2]
            for tag, i1, i2, j1, j2 in reversed(ops_):
                if tag in ("replace", "delete"):
                    left, rest = split_runs(mm.runs, i1)
                    cut, right = split_runs(rest, i2 - i1)
                    for r in cut:
                        # Deleted spaces get tombstones too, or normalising would bring them back.
                        if r.src is not None and r.kind != "sep" and r.text:
                            stones.append(Tombstone(r.src, r.e, r.a, r.b, r.text, tx.op, mid, "user", now))
                    mm.runs = left + right
                if tag in ("replace", "insert"):
                    text = new_text[j1:j2]
                    left, right = split_runs(mm.runs, i1)
                    mm.runs = left + [Run(text, "text", ins=ins_id + f".{j1}", a=0, b=len(text))] + right
            for pos, controls in sorted(tags.items(), reverse=True):
                left, right = split_runs(mm.runs, pos)
                mm.runs = left + controls + right
            if origin is not None and source_id(mm) != sid:
                # Retain the source style/coordinate system without marking new
                # characters (or freshly typed ASS coordinates) as source text.
                mm.runs.insert(0, Run("", "tag", ins=new_id("i_"), render_src=sid,
                                      render_e=origin.e if origin.src is not None else origin.render_e))
            mm.runs = _normalize_runs(mm.runs, self._src_text, self._deleted)
            if _gloss_marks is not None:
                by_position = {}
                for pos, mark in _gloss_marks:
                    by_position.setdefault(pos, []).append(Run(mark, "tag", ins=new_id("i_"), b=len(mark)))
                for pos, tags in reversed(list(by_position.items())):
                    left, right = split_runs(mm.runs, pos)
                    # A Raw/gloss marker at the first character must not wrap
                    # the hidden ASS prefix and make it literal Raw content.
                    while right and right[0].kind == "tag" and not is_private_run(right[0]):
                        left.append(right.pop(0))
                    mm.runs = left + tags + right
            mm.rev += 1
            self._drop_detached(tx, mm)

        self.history.run(_history_label, apply)
        return True

    def reannotate(self, mids: list[str], engine, sources: dict, *, label: str = "刷新自动读音") -> int:
        """Fresh automatic readings for members (one undo step).  Confirmed / hidden / typed readings are kept.
        Returns how many suggestions were written."""
        from .readings import kept_positions, suggest_readings, visible_map

        def apply(tx: Transaction) -> int:
            n = 0
            for mid in mids:
                m = self.doc.members.get(mid)
                if m is None or get_language(m.lang).annotation_system != "ruby":
                    continue
                _vt, ids = visible_map(m)
                keep, cov = kept_positions(self.doc, m, ids)
                keep_ids = {x.id for x in keep}
                for a in [x for x in self.doc.annotations.values() if x.member == mid and x.kind == "ruby"
                          and x.id not in keep_ids]:
                    tx.annotation(a.id)
                    del self.doc.annotations[a.id]
                src = next((sources[r.src] for r in m.runs if r.src in sources), None)
                for ann in suggest_readings(m, engine, source=src, covered=cov):
                    tx.add_annotation(ann)
                    n += 1
            return n

        return self.history.run(label, apply)

    # ---- a bracket in the original subtitles that is not a reading (眷属(ファミリア)): back into the text
    def _bracket_of(self, aid: str):
        """The source bracket (reading_hint span) an annotation came from, or None."""
        a = self.doc.annotations.get(aid)
        if a is None or not a.anchor:
            return None
        seg = a.anchor[0]
        sid, e = seg.get("src"), seg.get("e")
        src = self.sources.get(sid) if sid else None
        if src is None:
            return None
        lo = min(s["a"] for s in a.anchor if s.get("src") == sid and s.get("e") == e)
        hi = max(s["b"] for s in a.anchor if s.get("src") == sid and s.get("e") == e)
        for sp in src.spans:
            if sp.kind == "reading_hint" and sp.e == e and sp.base and sp.base[0] >= lo and sp.base[1] <= hi:
                return src, sp
        # analysed again by a newer rule (the span is gone): the bracket right after the word in the source text
        from ..model.source import Span
        from .cc import CLOSE, OPEN

        t = src.events[e].text if 0 <= e < len(src.events) else ""
        if hi < len(t) and t[hi] in OPEN:
            depth = 0
            for j in range(hi, len(t)):
                depth += 1 if t[j] in OPEN else -1 if t[j] in CLOSE else 0
                if depth == 0:
                    return src, Span(e, hi, j + 1, "reading_hint", t[hi + 1:j], [lo, hi])
        return None

    def bracket_candidates(self, *, katakana_only: bool = True) -> list[str]:
        """Annotations that came from brackets in the original subtitles (by default only katakana ones, which
        ja-cc 3 no longer reads as readings)."""
        import re as _re

        kata = _re.compile(r"^[ァ-ヺー・\s　]+$")
        out = []
        for a in self.doc.annotations.values():
            if a.origin != "cc_hint":
                continue
            hit = self._bracket_of(a.id)
            if hit is None:
                continue
            src, sp = hit
            inner = src.events[sp.e].text[sp.a + 1:sp.b - 1]
            if not katakana_only or kata.match(inner):
                out.append(a.id)
        return out

    def _tx_bracket_to_text(self, tx: "Transaction", aid: str, engine, now: str) -> bool:
        from .readings import kept_positions, suggest_readings, visible_map

        hit = self._bracket_of(aid)
        if hit is None:
            return False
        src, sp = hit
        a = self.doc.annotations[aid]
        mm = tx.member(a.member)
        if mm is None:
            return False
        ev_text = src.events[sp.e].text
        bb = sp.base[1]
        i = next((k for k, r in enumerate(mm.runs) if r.src == src.id and r.e == sp.e and r.kind != "sep"
                  and r.a < bb <= r.b), None)
        if i is None:
            return False
        r = mm.runs[i]
        if r.b > bb:                                  # (not expected: the bracket was cut out right after the word)
            mm.runs[i:i + 1] = [r.slice(0, bb - r.a), r.slice(bb - r.a, r.b - r.a)]
        j = i + 1
        while j < len(mm.runs) and mm.runs[j].kind == "sep":    # the tool's separator where the bracket was
            j += 1
        nxt = mm.runs[j] if j < len(mm.runs) else None
        if nxt is not None and nxt.src == src.id and nxt.e == sp.e and nxt.a == sp.b:
            del mm.runs[i + 1:j]
        mm.runs.insert(i + 1, Run(ev_text[sp.a:sp.b], "text", src.id, sp.e, sp.a, sp.b))
        mm.rev += 1
        tx.annotation(aid)
        del self.doc.annotations[aid]
        tx.rules().append({"kind": "span_text", "src": src.id, "e": sp.e, "a": sp.a, "b": sp.b,
                           "text": ev_text[sp.a:sp.b], "was": "reading_hint", "by": "user", "at": now})
        if engine is not None:                        # an ordinary reading for the word (けんぞく)
            _vt, ids = visible_map(mm)
            try:
                p0 = ids.index(("s", src.id, sp.e, sp.base[0]))
            except ValueError:
                return True
            _keep, cov = kept_positions(self.doc, mm, ids)
            for ann in suggest_readings(mm, engine, source=src, covered=cov,
                                        window=(p0, p0 + sp.base[1] - sp.base[0])):
                tx.add_annotation(ann)
        return True

    def bracket_to_text(self, aids: list[str], engine=None) -> int:
        """Readings that came from brackets in the original subtitles but are not readings (眷属(ファミリア)): the
        bracket goes back into the line as the source has it, the ruby is removed and the word gets an ordinary
        reading suggestion.  One undo step.  Returns how many were changed."""
        aids = [x for x in dict.fromkeys(aids) if x in self.doc.annotations]
        done = [0]

        def apply(tx: Transaction) -> None:
            now = self._now()
            for aid in aids:
                done[0] += int(self._tx_bracket_to_text(tx, aid, engine, now))
            self.doc.structure_changed()

        self.history.run("括号不是注音：放回正文" + (f" ×{len(aids)}" if len(aids) > 1 else ""), apply)
        return done[0]

    def add_reading_rule(self, surface: str, reading: str) -> None:
        """An episode reading rule: this word is read like this everywhere in the episode (one ruby group;
        spaces in both split it into groups, like the project dictionary).  Undoable; stored in the document."""
        def apply(tx: Transaction) -> None:
            rules = tx.rules()
            key = surface.replace(" ", "")
            rules[:] = [r for r in rules if not (r.get("kind") == "reading" and r.get("surface", "").replace(" ", "") == key)]
            rules.append({"kind": "reading", "surface": surface, "reading": reading})
        self.history.run("本集读音规则", apply)

    def remove_reading_rule(self, surface: str) -> None:
        key = surface.replace(" ", "")

        def apply(tx: Transaction) -> None:
            rules = tx.rules()
            rules[:] = [r for r in rules if not (r.get("kind") == "reading" and r.get("surface", "").replace(" ", "") == key)]
        self.history.run("删除本集读音规则", apply)

    def add_annotation(self, mid: str, a: int, b: int, text: str, *, kind: str = "ruby", engine=None,
                       source=None) -> Optional[str]:
        """A manual reading over **visible** characters [a, b) of a member (as rendered: a line break is not
        a character; editing-text offsets are converted with ``readings.plain_range_to_visible``)."""
        from .readings import anchor_of, visible_map

        m = self.doc.members[mid]
        if kind == "ruby" and get_language(m.lang).annotation_system != "ruby":
            return None
        vt, ids = visible_map(m)
        plain = vt.text
        if not (0 <= a < b <= len(ids)):
            return None
        seg = anchor_of(ids[a:b])
        if not seg or any(x is None for x in ids[a:b]):
            return None

        pinyin = None
        if kind == "pinyin":
            from .pinyin_notes import normalize_rule

            if get_language(m.lang).annotation_system != "pinyin":
                return None
            _word, text = normalize_rule(plain[a:b], text)
            pinyin = text.split()

        def apply(tx: Transaction) -> str:
            tx.keeps_text = True
            tx.member(mid)
            freed: set[int] = set()
            for old in [x for x in self.doc.annotations.values() if x.member == mid and x.kind == kind]:
                ov = _anchor_ids(old) & {x for x in ids[a:b] if x is not None}
                if ov:
                    freed.update(i for i, ident in enumerate(ids) if ident in _anchor_ids(old))
                    tx.annotation(old.id)
                    del self.doc.annotations[old.id]
            from .readings import kept_positions, ruby_parts, suggest_readings
            if pinyin is not None:
                first = None
                for k, syllable in enumerate(pinyin):
                    ann = Annotation(new_id("a_"), kind, mid, anchor_of([ids[a + k]]), plain[a + k], syllable,
                                     status="confirmed", origin="manual", candidates=[syllable],
                                     parts=[[0, 1, syllable]])
                    tx.add_annotation(ann)
                    first = first or ann.id
                return first
            ann = Annotation(new_id("a_"), kind, mid, seg, plain[a:b], text, status="confirmed", origin="manual",
                             candidates=[text], parts=ruby_parts(plain[a:b], text) if kind == "ruby" else [])
            tx.add_annotation(ann)
            # a word cut in two (e.g. a reading for 都市 inside 迷宮都市): the rest gets fresh suggestions
            rest = sorted(i for i in freed if not (a <= i < b))
            if rest and engine is not None and kind == "ruby":
                _keep, cov = kept_positions(self.doc, m, ids)
                cov.update(range(a, b))
                for sug in suggest_readings(m, engine, source=source, covered=cov, window=(rest[0], rest[-1] + 1)):
                    tx.add_annotation(sug)
            return ann.id

        return self.history.run("添加读音", apply)

    def _word_split_engine(self, project=None, *, surface=None, parts=None):
        from ..reading.engine import ReadingEngine

        base = self.reading_engine_factory() if self.reading_engine_factory else None
        engine = (project.reading_engine(episode=self.doc.episode, season=getattr(base, "season", None))
                  if project is not None else base or ReadingEngine())
        # A factory can return a cached engine. Preview must not mutate it.
        dictionary = copy.copy(engine.dict)
        dictionary.words = dict(engine.dict.words)
        dictionary.word_splits = dict(engine.dict.word_splits)
        if parts is not None:
            dictionary.merge_json({"word_splits": {surface: parts}}, "project")
        rules = {r["surface"]: r["reading"] for r in self.doc.rules if r.get("kind") == "reading"}
        if rules:
            dictionary.merge_json({"words": rules}, "episode")
        return ReadingEngine(dictionary, engine.analyzer, episode=engine.episode, season=engine.season)

    def preview_word_split_rule(self, surface: str, parts: list[dict], *, project=None) -> list[dict]:
        """Every exact occurrence, showing automatic updates and protected positions."""
        from ..reading.dictionary import normalize_word_split
        from .readings import word_split_plan

        parts = normalize_word_split(surface, parts)
        engine = self._word_split_engine(project, surface=surface, parts=parts)
        out = []
        for member in self.doc.members.values():
            if get_language(member.lang).annotation_system != "ruby" or member.role in ("sign", "info", "comment"):
                continue
            source = next((self.sources[r.src] for r in member.runs if r.src in self.sources), None)
            out.extend(word_split_plan(self.doc, member, engine, surface, source=source)["items"])
        return out

    def _tx_word_split_refresh(self, tx, surface: str, engine) -> int:
        from .readings import word_split_plan

        count = 0
        for member in self.doc.members.values():
            if get_language(member.lang).annotation_system != "ruby" or member.role in ("sign", "info", "comment"):
                continue
            source = next((self.sources[r.src] for r in member.runs if r.src in self.sources), None)
            plan = word_split_plan(self.doc, member, engine, surface, source=source)
            if plan["remove"] or plan["add"]:
                tx.member(member.id)
                for ann in plan["remove"]:
                    tx.annotation(ann.id)
                    del self.doc.annotations[ann.id]
                for ann in plan["add"]:
                    tx.add_annotation(ann)
                    count += 1
        return count

    @staticmethod
    def _tx_reading_dictionary(tx, project):
        from ..reading.dictionary import write_dictionary_bytes

        path = project.dict_path("readings")

        def put(data):
            current = path.read_bytes() if path.exists() else None
            if current == data:
                return
            if data is None:
                path.unlink(missing_ok=True)
            else:
                write_dictionary_bytes(path, data)

        tx.external(lambda: path.read_bytes() if path.exists() else None, put)
        tx.reading_rules_changed = True
        return path

    def split_reading_words(self, mid: str, a: int, b: int, parts: list[dict], *,
                            scope: str = "here", project=None) -> int:
        """Explicit words here, optionally remembered for this Project, as one undo step."""
        from ..reading.dictionary import ReadingDict, normalize_word_split
        from .readings import anchor_of, anchor_positions, ruby_parts, visible_map

        if scope not in ("here", "project") or (scope == "project" and project is None):
            raise ValueError("拆词范围必须是只改这里或整个工程")
        member = self.doc.members[mid]
        if get_language(member.lang).annotation_system != "ruby":
            raise ValueError("按指定词拆开只适用于日文读音")
        vt, ids = visible_map(member)
        if not 0 <= a < b <= len(ids) or any(identity is None for identity in ids[a:b]):
            raise ValueError("拆词范围已改变，请重新选中")
        surface = vt.text[a:b]
        parts = normalize_word_split(surface, parts)
        old = []
        for ann in self.doc.annotations.values():
            rng = anchor_positions(ann, ids) if ann.member == mid and ann.kind == "ruby" else None
            if rng and rng[0] < b and a < rng[1]:
                if rng[0] < a or rng[1] > b:
                    raise ValueError("请先选中整个读音范围，再按指定词拆开")
                old.append((rng, ann))

        def apply(tx):
            tx.keeps_text = True
            tx.member(mid)
            if scope == "project":
                path = self._tx_reading_dictionary(tx, project)
                ReadingDict().set_word_split(path, surface, parts)
            for _rng, ann in old:
                tx.annotation(ann.id)
                del self.doc.annotations[ann.id]
            count = 0
            pos = a
            for part in parts:
                end = pos + len(part["surface"])
                pieces = ruby_parts(part["surface"], part["reading"])
                if pieces:
                    layout = next((copy.deepcopy(ann.layout) for (x, y), ann in old if x < end and pos < y), {})
                    tx.add_annotation(Annotation(
                        new_id("a_"), "ruby", mid, anchor_of(ids[pos:end]), part["surface"], part["reading"],
                        status="confirmed", origin="manual", candidates=[part["reading"]], parts=pieces, layout=layout))
                    count += 1
                pos = end
            if scope == "project":
                count += self._tx_word_split_refresh(tx, surface, self._word_split_engine(project))
            return count

        return self.history.run("按指定词拆开" + ("（工程规则）" if scope == "project" else ""), apply)

    def remove_word_split_rule(self, surface: str, *, project) -> int:
        """Delete a Project split and refresh its automatic occurrences, in one undo step."""
        from ..reading.dictionary import ReadingDict

        if surface not in ReadingDict.load(project.dict_path("readings")).word_splits:
            return 0

        def apply(tx):
            tx.keeps_text = True
            path = self._tx_reading_dictionary(tx, project)
            ReadingDict().remove_word_split(path, surface)
            return self._tx_word_split_refresh(tx, surface, self._word_split_engine(project))

        return self.history.run("删除工程拆词规则", apply)

    def split_kanji_readings(self, mid: str, a: int, b: int, readings: list[str]) -> int:
        from .kanji_readings import split

        return split(self, mid, a, b, readings)

    def _tx_pinyin(self, tx: Transaction, mids: Iterable[str], rules: Mapping[str, str]) -> int:
        from .pinyin_notes import suggest_pinyin
        from .readings import kept_positions, visible_map

        tx.keeps_text = True
        n = 0
        for mid in dict.fromkeys(mids):
            m = self.doc.members.get(mid)
            if m is None or get_language(m.lang).annotation_system != "pinyin":
                continue
            tx.member(mid)
            _vt, ids = visible_map(m)
            keep, covered = kept_positions(self.doc, m, ids, kind="pinyin")
            for old in list(self.doc.annotations.values()):
                if old.member == mid and old.kind == "pinyin" and old not in keep:
                    tx.annotation(old.id)
                    del self.doc.annotations[old.id]
            for ann in suggest_pinyin(m, rules, covered=covered):
                tx.add_annotation(ann)
                n += 1
        return n

    def reannotate_pinyin(self, mids: Iterable[str], rules: Optional[Mapping[str, str]] = None) -> int:
        from .pinyin_notes import episode_rules

        return self.history.run("拼音", lambda tx: self._tx_pinyin(tx, mids, episode_rules(self.doc, rules)))

    def set_pinyin_mode(self, project, mode: str) -> int:
        from .pinyin_notes import MODES, episode_rules

        if mode not in MODES:
            raise ValueError(mode)
        old = project.pinyin_mode
        if old == mode:
            return 0

        def put(value) -> None:
            settings = project.data.setdefault("settings", {})
            present, pinyin = value
            if present:
                settings["pinyin"] = pinyin
            else:
                settings.pop("pinyin", None)
            project.save()

        def apply(tx: Transaction) -> int:
            tx.keeps_text = True
            tx.external(lambda: ("pinyin" in project.data.get("settings", {}),
                                 project.data.get("settings", {}).get("pinyin")), put)
            project.set_pinyin_mode(mode)
            if old == "off" and mode != "off":
                return self._tx_pinyin(tx, self.doc.members, episode_rules(self.doc, project.pinyin_words()))
            return 0

        return self.history.run("拼音设置…", apply)

    def preview_pinyin_rule(self, surface: str, reading: str) -> list[dict]:
        """Every matching occurrence, with automatic changes and protected positions."""
        from .pinyin_notes import normalize_rule
        from .readings import anchor_positions, kept_positions, visible_map

        word, reading = normalize_rule(surface, reading)
        syllables = reading.split()
        out = []
        for m in self.doc.members.values():
            if get_language(m.lang).annotation_system != "pinyin":
                continue
            vt, ids = visible_map(m)
            _keep, covered = kept_positions(self.doc, m, ids, kind="pinyin")
            current = {}
            for ann in self.doc.annotations.values():
                pos = anchor_positions(ann, ids) if ann.member == m.id and ann.kind == "pinyin" else None
                if pos:
                    current[pos[0]] = ann.text
            start = 0
            while (start := vt.text.find(word, start)) >= 0:
                changes = [{"at": start + k, "old": current.get(start + k), "new": syl}
                           for k, syl in enumerate(syllables)
                           if start + k not in covered and ids[start + k] is not None and current.get(start + k) != syl]
                out.append({"member": m.id, "at": [start, start + len(word)], "text": vt.text,
                            "changes": changes, "kept": sorted(covered & set(range(start, start + len(word))))})
                start += 1
        return out

    def set_pinyin_rule(self, surface: str, reading: str, *, project=None,
                        words: Optional[Mapping[str, str]] = None) -> int:
        """Word rule and its current-episode refresh, in one undo step."""
        from .pinyin_notes import episode_rules, normalize_rule

        word, syllables = normalize_rule(surface, reading)
        mids = [item["member"] for item in self.preview_pinyin_rule(word, syllables)]
        project_words = dict(words or {})

        def apply(tx: Transaction) -> int:
            tx.keeps_text = True
            if project is None:
                rules = tx.rules()
                rules[:] = [r for r in rules if not (r.get("kind") == "pinyin" and r.get("surface") == word)]
                rules.append({"kind": "pinyin", "surface": word, "reading": syllables})
                active_words = project_words
            else:
                path = project.pinyin_path

                def put(data) -> None:
                    if data is None:
                        path.unlink(missing_ok=True)
                    else:
                        tmp = path.with_name(path.name + ".tmp")
                        tmp.write_bytes(data)
                        tmp.replace(path)

                tx.external(lambda: path.read_bytes() if path.exists() else None, put)
                project.set_pinyin_word(word, syllables)
                active_words = project.pinyin_words()
            return self._tx_pinyin(tx, mids, episode_rules(self.doc, active_words))

        return self.history.run(f"拼音：{word}", apply)

    def apply_pinyin_rule(self, mid: str, a: int, b: int, reading: str, *, scope: str = "here", project=None):
        from .readings import visible_map

        if scope == "here":
            return self.add_annotation(mid, a, b, reading, kind="pinyin")
        if scope not in ("episode", "project") or (scope == "project" and project is None):
            raise ValueError(scope)
        m = self.doc.members[mid]
        if get_language(m.lang).annotation_system != "pinyin" or not 0 <= a < b <= len(visible_map(m)[0].text):
            return 0
        return self.set_pinyin_rule(visible_map(m)[0].text[a:b], reading,
                                    project=project if scope == "project" else None,
                                    words=project.pinyin_words() if project is not None else None)

    def regenerate_readings(self, mid: str, a: int, b: int, engine, *, source=None,
                            include_confirmed: bool = False) -> int:
        """Run the reading analysis again for visible characters [a, b) of a member (one undo step).

        Readings you confirmed, hid or typed are kept unless ``include_confirmed`` (then they are replaced by
        fresh suggestions too).  Returns how many suggestions were written."""
        from .readings import anchor_positions, kept_positions, suggest_readings, visible_map

        m = self.doc.members[mid]
        _vt, ids = visible_map(m)
        if get_language(m.lang).annotation_system != "ruby":
            return 0
        a, b = max(0, a), min(len(ids), b)
        if a >= b:
            return 0
        keep, _cov = kept_positions(self.doc, m, ids)
        keep_ids = set() if include_confirmed else {x.id for x in keep}

        def apply(tx: Transaction) -> int:
            covered: set[int] = set()
            for old in [x for x in self.doc.annotations.values() if x.member == mid and x.kind == "ruby"]:
                pos = anchor_positions(old, ids)
                if pos is None:
                    continue
                if pos[1] <= a or pos[0] >= b or old.id in keep_ids:
                    covered.update(range(*pos))          # outside the window, or kept: leave it
                    continue
                tx.annotation(old.id)
                del self.doc.annotations[old.id]
            n = 0
            for ann in suggest_readings(m, engine, source=source, covered=covered, window=(a, b)):
                tx.add_annotation(ann)
                n += 1
            return n

        return self.history.run("重新生成读音", apply)

    def hide_annotation(self, aid: str) -> None:
        def apply(tx: Transaction) -> None:
            a = tx.annotation(aid)
            a.status = "hidden"
            a.origin = "manual"
        self.history.run("不标注", apply)

    def insert_text(self, mid: str, pos: int, text: str) -> None:
        old = self.visible_text(mid)
        self.replace_text(mid, old[:pos] + text + old[pos:], span=(pos, pos), _history_label="输入文字")

    # ---- deleting: the source text of what is deleted gets tombstones (it never comes back by itself)
    def _tx_drop_member(self, tx: Transaction, mid: str, now: str) -> None:
        mm = tx.member(mid)
        stones = tx.tombstones()
        for r in mm.runs:
            if r.src is not None and r.kind != "sep" and r.text.strip(SPACES):
                stones.append(Tombstone(r.src, r.e, r.a, r.b, r.text, tx.op, mid, "user", now))
        del self.doc.members[mid]
        for aid in [a.id for a in self.doc.annotations.values() if a.member == mid]:
            tx.annotation(aid)
            del self.doc.annotations[aid]

    def _tx_drop_comment(self, tx: Transaction, cid: str, now: str) -> None:
        c = tx.comment(cid)
        stones = tx.tombstones()
        for r in c.runs:
            if r.src is not None and r.kind != "sep" and r.text.strip(SPACES):
                stones.append(Tombstone(r.src, r.e, r.a, r.b, r.text, tx.op, None, "user", now))
        del self.doc.comments[cid]

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def delete_member(self, mid: str) -> None:
        """Delete one line (only this member); its source text gets tombstones."""
        m = self.doc.members[mid]
        g = self.doc.group(m.group)

        def apply(tx: Transaction) -> None:
            tg = tx.group(g.id)
            self._tx_drop_member(tx, mid, self._now())
            tg.members.remove(mid)
            if not tg.members and not tg.comments:
                order = tx.order()
                order.remove(next(x for x in order if x.id == tg.id))
            self.doc.structure_changed()

        self.history.run("删除这一行", apply)

    def delete_group(self, gid: str) -> None:
        self.delete_groups([gid])

    def delete_groups(self, gids: list[str]) -> None:
        """Delete whole groups (both languages, with their hidden fillers) in one undo step."""
        live = {g.id for g in self.doc.groups}
        gids = [g for g in dict.fromkeys(gids) if g in live]
        if not gids:
            return

        def apply(tx: Transaction) -> None:
            now = self._now()
            order = tx.order()                              # records every group for undo
            by_id = {g.id: g for g in order}                # (doc.group() would use an index that goes stale here)
            for gid in gids:
                tg = by_id[gid]
                for mid in list(tg.members):
                    self._tx_drop_member(tx, mid, now)
                for cid in list(tg.comments):
                    self._tx_drop_comment(tx, cid, now)
                order.remove(tg)
            self.doc.structure_changed()

        self.history.run("删除整个 Group" if len(gids) == 1 else f"删除 {len(gids)} 个 Group", apply)

    def delete_lane(self, gids: list[str], lane: str) -> int:
        """Delete the lines of one language ("original" / "translation") in these groups, with that language's
        hidden fillers, in one undo step.  A group left with nothing is removed.  Returns the number of lines."""
        live = {g.id for g in self.doc.groups}
        gids = [g for g in dict.fromkeys(gids) if g in live]
        mids = [m for gid in gids for m in self.doc.group(gid).members if self.doc.members[m].lane == lane]
        if not mids:
            return 0

        def is_lane(c) -> bool:          # hidden text of that language (notes stay)
            if c.kind not in ("hidden_filler", "user_hidden"):
                return False
            src = next((self.sources.get(r.src) for r in c.runs if r.src), None)
            if src is not None:
                return src.lane == lane
            return bool(c.lang) and ((get_language(c.lang).annotation_system == "pinyin") == (lane == "translation"))

        def apply(tx: Transaction) -> None:
            now = self._now()
            order = tx.order()                              # records every group for undo
            by_id = {g.id: g for g in order}                # (doc.group() would use an index that goes stale here)
            for gid in gids:
                tg = by_id[gid]
                for mid in [m for m in tg.members if self.doc.members[m].lane == lane]:
                    self._tx_drop_member(tx, mid, now)
                    tg.members.remove(mid)
                for cid in [c for c in tg.comments if is_lane(self.doc.comments[c])]:
                    self._tx_drop_comment(tx, cid, now)
                    tg.comments.remove(cid)
                if not tg.members and not tg.comments:
                    order.remove(next(x for x in order if x.id == gid))
            self.doc.structure_changed()

        self.history.run(f"删除{'日文' if lane == 'original' else '中文'}行 ×{len(mids)}", apply)
        return len(mids)

    def restore_comment(self, cid: str) -> None:
        """Hidden filler / user-hidden text → displayed text again, at its recorded position."""
        c = self.doc.comments[cid]

        def apply(tx: Transaction) -> None:
            com = tx.comment(cid)
            tg = tx.group(com.group)
            anchor = com.position.get("before") or com.position.get("after")
            target = self.doc.members.get(anchor) if anchor else None
            if target is None:
                target = self.lane_member(tg, "original") if com.lang and com.lang != "zh" else None
            if target is None:
                s, e = self.doc.group_time(tg)
                target = Member(new_id("m_"), tg.id, com.lang or "ja", "original", "dialogue", s, e, style="JP")
                tx.add_member(target)
                tg.members.append(target.id)
            else:
                target = tx.member(target.id)
            if not any(r.src is not None for r in com.runs) and com.position.get("before") == target.id:
                target.runs = self._join(list(com.runs), target.runs)
            else:
                target.runs = self._insert_by_source(target.runs, list(com.runs))
            target.rev += 1
            tg.comments.remove(cid)
            del self.doc.comments[cid]
            self.doc.structure_changed()

        self.history.run("恢复显示", apply)

    def hide_range(self, mid: str, a: int, b: int) -> Optional[str]:
        """Hide visible characters [a, b) as a restorable user_hidden Comment (only at the edges)."""
        m = self.doc.members[mid]
        vis = len("".join(r.text for r in m.runs if r.kind != "tag"))
        if a != 0 and b != vis:
            return None                        # hiding the middle would need two comments; split first

        def apply(tx: Transaction) -> str:
            mm = tx.member(mid)
            left, rest = split_runs(mm.runs, a)
            cut, right = split_runs(rest, b - a)
            c = Comment(new_id("c_"), mm.group, "user_hidden", mm.lang, self._norm(cut),
                        {"before": mid} if a == 0 else {"after": mid}, "user", True)
            tx.add_comment(c)
            tg = tx.group(mm.group)
            tg.comments.append(c.id)
            mm.runs = self._norm(left + right)
            mm.rev += 1
            return c.id

        return self.history.run("隐藏", apply)

    # -------------------------------------------------------------- time
    @staticmethod
    def _record_time_source(member: Member, before: tuple[int, int], source: str) -> None:
        changed = [edge for edge, old in zip(("start", "end"), before)
                   if getattr(member, edge) != old]
        if not changed:
            return
        if "time_src" not in member.extra and member.extra.get("time_manual"):
            member.extra["time_src"] = {"start": "manual", "end": "manual"}
        sources = member.extra.setdefault("time_src", {})
        for edge in changed:
            sources[edge] = source
        if source in ("manual", "adopted"):
            member.extra["time_manual"] = True

    def shift_group(self, gid: str, delta: int) -> None:
        self._shift_groups([gid], delta, "平移 Group")

    def shift_groups(self, gids: Iterable[str], delta: int) -> None:
        """Shift a selection once, including beats shared by several selected groups."""
        gids = list(dict.fromkeys(gids))
        if gids and delta:
            self._shift_groups(gids, delta, f"整体平移 {len(gids)} 组")

    def _shift_groups(self, gids: list[str], delta: int, label: str) -> None:
        """Move a whole group in time (all members, same delta; times below zero clamped).  Karaoke beats sung by
        the group's characters move with it (they are absolute times)."""
        from .karaoke import _ids_of_anchor

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            mine = set()
            for gid in gids:
                tg = tx.group(gid)
                for mid in tg.members:
                    mm = tx.member(mid)
                    before = (mm.start, mm.end)
                    mm.start = max(0, mm.start + delta)
                    mm.end = max(0, mm.end + delta)
                    self._record_time_source(mm, before, "manual")
                    mine.update(ident for ident, *_ in mm.identities() if ident is not None)
            for kid, t in list(self.doc.karaoke.items()):
                beats = t.get("beats", ())
                if not any(set(_ids_of_anchor(b.get("core", []))) & mine for b in beats):
                    continue
                tt = tx.track(kid)
                for b in tt["beats"]:
                    if set(_ids_of_anchor(b.get("core", []))) & mine:
                        b["t0"] = max(0, b["t0"] + delta)
                        b["t1"] = max(0, b["t1"] + delta)
            self._follow_karaoke_links(tx, gids)
        self.history.run(label, apply)

    def apply_autotime(self, plan: AutoTimePlan,
                       chosen_group_ids: Optional[Iterable[str]] = None) -> tuple[int, int]:
        """Apply a preview in one undo step; refuse previews made before another edit."""
        from .karaoke import _ids_of_anchor

        if plan.episode != self.doc.episode or plan.revision != self.doc.revision:
            raise ValueError("auto-timing preview is stale")
        chosen = ({g.group_id for g in plan.changes} if chosen_group_ids is None else set(chosen_group_ids))
        changes = [g for g in plan.changes if g.group_id in chosen]
        for g in changes:
            members = self.doc.group(g.group_id).members
            for c in g.members:
                m = self.doc.members[c.member_id]
                if c.member_id not in members or (m.start, m.end) != (c.before_start, c.before_end):
                    raise ValueError("auto-timing preview is stale")
        if not changes:
            return (0, 0)
        mine = {ident for g in changes for m in self.doc.members_of(self.doc.group(g.group_id))
                for ident, *_ in m.identities() if ident is not None}
        beat_changes = []
        for kid, track in self.doc.karaoke.items():
            for i, beat in enumerate(track.get("beats", ())):
                if set(_ids_of_anchor(beat.get("core", []))) & mine:
                    t0, t1 = max(0, plan.mapping.map(beat["t0"])), max(0, plan.mapping.map(beat["t1"]))
                    if t1 < t0:
                        raise ValueError("timing map reverses a karaoke beat")
                    if (t0, t1) != (beat["t0"], beat["t1"]):
                        beat_changes.append((kid, i, t0, t1))

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            for g in changes:
                for c in g.members:
                    m = tx.member(c.member_id)
                    before = (m.start, m.end)
                    m.start, m.end = max(0, c.start), max(0, c.end)
                    self._record_time_source(m, before, "autotime")
            for kid, i, t0, t1 in beat_changes:
                beat = tx.track(kid)["beats"][i]
                beat["t0"], beat["t1"] = t0, t1
        self.history.run("自动调轴", apply)
        return len(changes), sum(len(g.members) for g in changes)

    # -------------------------------------------------------------- dialogue / song conversion
    def convert_to_song(self, gids: Iterable[str], family: str,
                        styles: Optional[Mapping[str, str]] = None) -> ConvertPlan:
        return self._convert(gids, "song", family=family, styles=styles)

    def convert_to_dialogue(self, gids: Iterable[str], styles: Optional[Mapping[str, str]] = None) -> ConvertPlan:
        return self._convert(gids, "dialogue", styles=styles)

    def convert_to_sign(self, gids: Iterable[str], styles: Optional[Mapping[str, str]] = None) -> ConvertPlan:
        return self._convert(gids, "sign", styles=styles)

    def _convert(self, gids: Iterable[str], to: str, *, family: Optional[str] = None,
                 styles: Optional[Mapping[str, str]] = None) -> ConvertPlan:
        from .convert import plan_conversion
        from .merge import MergeSettings

        styles = MergeSettings().styles if styles is None else styles
        plan = plan_conversion(self.doc, gids, to, family=family, styles=styles)
        if not plan.gids:
            return plan

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            self._convert_transaction(tx, plan.gids, to, styles, plan.family)
        self.history.run({"song": "转为歌曲", "dialogue": "转为普通台词", "sign": "转为屏幕字"}[to], apply)
        return plan

    def _convert_transaction(self, tx: Transaction, gids: Iterable[str], to: str,
                             styles: Mapping[str, str], family: Optional[str]) -> None:
        from .convert import member_target
        from .karaoke import tracks_of

        gids = list(gids)
        for gid in gids:
            g = tx.group(gid)
            if to == "sign" or (to == "dialogue" and g.kind == "sign"):
                self._convert_sign_transaction(tx, g, to, styles)
                continue
            old = g.extra.get("before_convert") or {}
            kind = old.get("kind") if to == "dialogue" and old.get("kind") == "dialogue" else to
            g.extra["before_convert"] = {"kind": g.kind}
            g.kind = kind
            for mid in g.members:
                m = self.doc.members[mid]
                if to == "dialogue" and m.role != "song":
                    continue
                target = member_target(m, to, styles, family)
                if target is None:
                    continue
                m = tx.member(mid)
                m.extra["before_convert"] = {"role": m.role, "style": m.style}
                m.role, m.style = target
                if to == "song":
                    m.extra.pop("karaoke_off", None)
                else:
                    m.extra["karaoke_off"] = True
                for t in tracks_of(self.doc, mid, dormant=True):
                    track = tx.track(t["id"])
                    if to == "song":
                        track.pop("dormant", None)
                    else:
                        track["dormant"] = True
                m.rev += 1
        if to == "song":
            self._follow_karaoke_links(tx, gids)

    def _convert_sign_transaction(self, tx: Transaction, g: Group, to: str,
                                  styles: Mapping[str, str]) -> None:
        from dataclasses import asdict
        from ..ass.tags import canonical, parse_block
        from ..reading.engine import ReadingEngine
        from .readings import anchor_positions, suggest_readings, visible_map

        saved_kind = g.extra.pop("before_sign_convert", None)
        if saved_kind is None:
            g.extra["before_sign_convert"] = {"kind": g.kind}
        g.kind = to
        engine = None
        for mid in g.members:
            m = self.doc.members[mid]
            if m.role not in (("sign",) if to == "dialogue" else ("dialogue", "top", "alt")):
                continue
            m = tx.member(mid)
            saved = m.extra.pop("before_sign_convert", None)
            if saved is not None and saved_kind and saved_kind["kind"] == to:
                for key in ("role", "style", "margins", "overrides", "rev"):
                    setattr(m, key, copy.deepcopy(saved[key]))
                if saved["source_style"] is None:
                    m.extra.pop("source_style", None)
                else:
                    m.extra["source_style"] = saved["source_style"]
                for aid, original in saved.get("generated", {}).items():
                    ann = self.doc.annotations.get(aid)
                    if ann is not None and asdict(ann) == original:
                        tx.annotation(aid)
                        del self.doc.annotations[aid]
                continue
            saved = {key: copy.deepcopy(getattr(m, key))
                     for key in ("role", "style", "margins", "overrides", "rev")}
            saved["source_style"] = m.extra.get("source_style")
            m.extra["before_sign_convert"] = saved
            if to == "dialogue":
                m.role, m.style = "dialogue", styles[f"{m.lane}/dialogue"]
                m.margins = [0, 0, 0]
                m.extra.pop("source_style", None)
                for key in ("an", "pos", "move"):
                    m.overrides.pop(key, None)
                if m.overrides.get("raw"):
                    m.overrides["raw"] = "".join(t.raw for t in parse_block(m.overrides["raw"])
                                                  if canonical(t.name) not in ("an", "pos", "move", "org"))
                m.overrides["source_position_off"] = True
                if m.lane == "original" and get_language(m.lang).annotation_system == "ruby":
                    if engine is None:
                        engine = self.reading_engine_factory() if self.reading_engine_factory else ReadingEngine()
                    _vt, ids = visible_map(m)
                    covered = set()
                    for ann in self.doc.annotations.values():
                        if ann.member == mid:
                            positions = anchor_positions(ann, ids)
                            if positions:
                                covered.update(range(*positions))
                    source = next((self.sources[r.src] for r in m.runs if r.src in self.sources), None)
                    saved["generated"] = {}
                    for ann in suggest_readings(m, engine, source=source, covered=covered):
                        tx.add_annotation(ann)
                        saved["generated"][ann.id] = asdict(ann)
            else:
                m.role = "sign"
                event = next((self.sources[r.src].events[r.e] for r in m.runs if r.src in self.sources), None)
                if event is not None:
                    m.style = event.style or "Default"
                    m.margins = list(event.margins or [0, 0, 0])
                m.overrides = {}
                m.extra["source_style"] = True
            m.rev += 1

    def apply_passage(self, plan: PassagePlan, chosen_ids: Iterable[str], *,
                      styles: Optional[Mapping[str, str]] = None) -> ApplySummary:
        from .passages import apply

        return apply(self, plan, chosen_ids, styles=styles)

    # -------------------------------------------------------------- karaoke
    def _follow_karaoke_links(self, tx: Transaction, gids: Iterable[str]) -> None:
        from .karaoke import follow_links, linked_times, tracks_of

        for gid in set(gids):
            linked = [track for m in self.doc.members_of(self.doc.group(gid))
                      if m.lane == "translation" and not m.extra.get("karaoke_off")
                      for track in tracks_of(self.doc, m.id)
                      if any(b.get("link") for b in track.get("beats", ()))]
            if not linked:
                continue
            originals = linked_times(self.doc, gid)
            for track in linked:
                updated = copy.deepcopy(track)
                if follow_links(updated, originals):
                    tx.set_track(updated)

    def karaoke_twins(self, mid: str) -> list[str]:
        """Other copies of the same lyric (e.g. the glow layer a fansub draws under it): same lane, same text,
        same time.  They get the same karaoke."""
        from .readings import visible_map

        m = self.doc.members[mid]
        text = visible_map(m)[0].text
        out = []
        for o in self.doc.members.values():
            if o.id == mid or o.lane != m.lane or o.role != m.role:
                continue
            if abs(o.start - m.start) <= 20 and abs(o.end - m.end) <= 20 and visible_map(o)[0].text == text:
                out.append(o.id)
        return out

    def set_karaoke(self, mid: str, beats, *, state: str = "timed", unit: str = "syllable",
                    effect: Optional[dict] = None, label: str = "K 轴", twins: bool = False,
                    edits: Optional[list[dict]] = None) -> str:
        """Write the timing of a member's beats (replaces its track; one undo step).  ``twins``: copies of the
        same lyric get the same beat times.  Returns the track id."""
        import copy as _copy

        from .karaoke import beats_to_track, derive, edits_of, remap_edits, tracks_of

        beats = _copy.deepcopy(list(beats))
        for b in beats:
            if b.by == "tap" or state == "confirmed":
                b.flags = [f for f in b.flags if f != "check"]
        m = self.doc.members[mid]
        edits = edits_of(self.doc, m) if edits is None else edits
        targets = [(mid, beats, edits)]
        if twins:
            for tid in self.karaoke_twins(mid):
                tm = self.doc.members[tid]
                te = remap_edits(self.doc, m, tm, edits)
                tb = derive(self.doc, tm, unit, te)
                if [b.sung for b in tb] != [b.sung for b in beats]:
                    continue
                for b, src in zip(tb, beats):
                    b.t0, b.t1, b.by = src.t0, src.t1, src.by
                    b.link = _copy.deepcopy(src.link)
                    b.flags = list(dict.fromkeys(b.flags + src.flags))
                targets.append((tid, tb, te))
        tracks = []
        for tmid, tbeats, tedits in targets:
            old = tracks_of(self.doc, tmid)
            kid = old[0]["id"] if old else None
            eff = effect if effect is not None else (_copy.deepcopy(old[0].get("effect")) if old else None)
            tracks.append((old, beats_to_track(self.doc.members[tmid], tbeats, state=state, unit=unit, kid=kid,
                                               effect=eff, edits=tedits)))
            if old and old[0].get("issues"):
                tracks[-1][1]["issues"] = _copy.deepcopy(old[0]["issues"])

        def apply(tx: Transaction) -> str:
            tx.keeps_text = True
            for old, t in tracks:
                for o in old[1:]:
                    tx.remove_track(o["id"])
                tx.set_track(t)
            self._follow_karaoke_links(tx, [self.doc.members[t["member"]].group for _old, t in tracks
                                           if self.doc.members[t["member"]].lane == "original"])
            return tracks[0][1]["id"]

        return self.history.run(label, apply)

    def apply_auto_karaoke(self, results, *, translations: bool, overwrite: bool,
                           label: str = "自动 K 轴", requested_ids=None, follow_notes: bool = True) -> dict:
        """Store requested original alignments and linked translations in one undo step."""
        return self._apply_auto_karaoke(results, translations=translations, overwrite=overwrite, label=label,
                                        requested_ids=requested_ids, follow_notes=follow_notes)

    def adopt_auto_karaoke(self, mid: str, beats, unit: str, translations: bool, *, follow_notes: bool = True) -> dict:
        """Explicit single-line adoption may replace the selected line's existing timing."""
        values = [(b.t0, b.t1, 1.0, "check" in b.flags) for b in beats]
        return self._apply_auto_karaoke([{"mid": mid, "beats": values}], translations=translations,
                                       overwrite=True, label="采用自动 K 轴", units={mid: unit}, follow_notes=follow_notes)

    def _redistribute_auto_karaoke_translations(self, tx: Transaction, gids: set[str], *, follow_notes=True):
        """Redistribute eligible translations within the automatic timing undo step."""
        from .lyrics_auto import redistribute_in_transaction

        return redistribute_in_transaction(self, tx, gids, only_uncustomized=True, follow_notes=follow_notes)

    def _apply_auto_karaoke(self, results, *, translations: bool, overwrite: bool, label: str,
                            units: Optional[dict[str, str]] = None, requested_ids=None, follow_notes: bool = True) -> dict:
        from . import karaoke as kara

        rows = results.lines if hasattr(results, "lines") else results.get("lines", []) if isinstance(results, dict) else results
        report = {"updated": 0, "skipped": {}, "lines_with_check": 0, "widened_lines": 0,
                  "widened": [], "translations_skipped": "", "clipped": 0}
        tracks, seen = {}, set()
        units = units or {}
        for row in rows:
            mid = row.mid if hasattr(row, "mid") else row["mid"]
            values = row.beats if hasattr(row, "beats") else row["beats"]
            if mid in seen:
                continue
            m = self.doc.members.get(mid)
            if m is None or m.lane != "original" or self.doc.group(m.group).kind != "song":
                report["skipped"][mid] = "不是歌曲原声行"
                continue
            targets = [mid] + self.karaoke_twins(mid)
            if requested_ids is not None:
                targets = [tid for tid in targets if tid in requested_ids]
            views = {tid: kara.effective(self.doc, self.doc.members[tid]) for tid in targets}
            if not overwrite and any(v.state == "confirmed" or any(b.by in ("tap", "drag", "manual")
                                                                       for b in v.beats) for v in views.values()):
                report["skipped"][mid] = "已确认或手调过"
                continue
            view = views[mid]
            unit = units.get(mid, view.unit)
            edits = kara.edits_of(self.doc, m)
            source = kara.derive(self.doc, m, unit, edits)
            if len(source) != len(values):
                raise ValueError("自动 K 轴拍数与当前歌词不一致")
            for beat, value in zip(source, values):
                a, z, _conf, check = value
                if a < 0 or z < a:
                    raise ValueError("自动 K 轴时间无效")
                beat.t0, beat.t1, beat.by = int(a), int(z), "align"
                if check:
                    beat.flags.append("check")
            for tid in targets:
                if tid in seen:
                    continue
                target = self.doc.members[tid]
                te = edits if tid == mid else kara.remap_edits(self.doc, m, target, edits)
                tb = source if tid == mid else kara.derive(self.doc, target, unit, te)
                if [b.sung for b in tb] != [b.sung for b in source]:
                    continue
                if tid != mid:
                    for b, src in zip(tb, source):
                        b.t0, b.t1, b.by = src.t0, src.t1, src.by
                        b.flags = list(dict.fromkeys(b.flags + src.flags))
                old = kara.tracks_of(self.doc, tid)
                track = kara.beats_to_track(target, tb, state="suggested", unit=unit,
                    kid=old[0]["id"] if old else None,
                    effect=copy.deepcopy(old[0].get("effect")) if old else None, edits=te)
                if old and old[0].get("issues"):
                    track["issues"] = copy.deepcopy(old[0]["issues"])
                tracks[tid] = (old, track)
                seen.add(tid)
                report["updated"] += 1
                report["lines_with_check"] += int(any("check" in b.flags for b in tb))
        if not tracks:
            return report

        order = sorted((g for g in self.doc.groups if g.members and "hidden" not in g.flags
                        and g.kind in ("dialogue", "song")), key=self.doc.group_time)
        before = {g.id: self.doc.group_time(g) for g in order}

        def apply(tx: Transaction) -> dict:
            tx.keeps_text = True
            for old, track in tracks.values():
                for previous in old[1:]:
                    tx.remove_track(previous["id"])
                tx.set_track(track)
            gids = {self.doc.members[mid].group for mid in tracks}
            self._follow_karaoke_links(tx, gids)
            if translations:
                redistributed = self._redistribute_auto_karaoke_translations(tx, gids, follow_notes=follow_notes)
                if redistributed is None:
                    report["translations_skipped"] = "翻译拍重新分配尚未可用，已跳过"
                else:
                    report["translation_report"] = redistributed
            # Followed translation beats may extend farther than the new original beats.
            members_by_group = {gid: [m for m in self.doc.members_of(self.doc.group(gid))
                                     if m.id in tracks or m.lane == "translation"] for gid in gids}
            wanted = dict(before)
            bounds = {gid: [0, float("inf")] for gid in gids}
            for gid, members in members_by_group.items():
                beats = [b for m in members for b in kara.effective(self.doc, m).beats if b.timed]
                if gid in before and beats:
                    wanted[gid] = (min(before[gid][0], min(b.t0 for b in beats)),
                                   max(before[gid][1], max(b.t1 for b in beats)))
            for left, right in zip(order, order[1:]):
                if before[left.id][1] > before[right.id][0] or wanted[left.id][1] <= wanted[right.id][0]:
                    continue
                # A shared boundary inside the old gap protects both widening neighbours.
                boundary = max(before[left.id][1], min(before[right.id][0], wanted[right.id][0]))
                if left.id in bounds:
                    bounds[left.id][1] = boundary
                if right.id in bounds:
                    bounds[right.id][0] = boundary
            clipped = set()
            clip_before = {track["id"]: [(b["t0"], b["t1"]) for b in track.get("beats", ())]
                           for members in members_by_group.values() for m in members
                           for track in kara.tracks_of(self.doc, m.id)}

            def clip_member(m):
                low, high = bounds[m.group]
                for track in kara.tracks_of(self.doc, m.id):
                    updated = copy.deepcopy(track)
                    changed = False
                    for beat in updated.get("beats", ()):
                        beat_changed = False
                        for field in ("t0", "t1"):
                            value = max(low, min(high, beat[field]))
                            if value != beat[field]:
                                beat[field] = value
                                edge = "start" if field == "t0" else "end"
                                # An independently clipped translation edge no longer follows its old target.
                                if m.lane == "translation" and edge in beat.get("link", {}):
                                    del beat["link"][edge]
                                    if not beat["link"]:
                                        beat.pop("link")
                                beat_changed = changed = True
                                clipped.add(m.id)
                        if beat_changed and beat["t1"] - beat["t0"] < AUTO_KARAOKE_MIN_MS:
                            beat["check"] = True
                    if changed and m.lane != "translation":
                        _respread_squeezed(updated.get("beats", ()), low, high, AUTO_KARAOKE_MIN_MS)
                    if changed:
                        tx.set_track(updated)

            # Clip originals first, then move their linked/fractional translation edges.
            for members in members_by_group.values():
                for m in members:
                    if m.id in tracks:
                        clip_member(m)
            self._follow_karaoke_links(tx, gids)
            for members in members_by_group.values():
                for m in members:
                    if m.lane == "translation":
                        clip_member(m)
            # Link following also shortens translations; count/flag those lines too.
            for members in members_by_group.values():
                for m in members:
                    for track in kara.tracks_of(self.doc, m.id):
                        updated = copy.deepcopy(track)
                        changed = False
                        for beat, old in zip(updated.get("beats", ()), clip_before[track["id"]]):
                            if (beat["t0"], beat["t1"]) != old:
                                clipped.add(m.id)
                                if beat["t1"] - beat["t0"] < AUTO_KARAOKE_MIN_MS and not beat.get("check"):
                                    beat["check"] = changed = True
                        if changed:
                            tx.set_track(updated)
            report["clipped"] = len(clipped)
            report["lines_with_check"] = sum(any("check" in b.flags for b in kara.effective(
                self.doc, self.doc.members[mid]).beats) for mid in tracks)
            for gid, members in members_by_group.items():
                beats = [b for m in members for b in kara.effective(self.doc, m).beats if b.timed]
                if not beats:
                    continue
                a, z = min(b.t0 for b in beats), max(b.t1 for b in beats)
                if any(a < m.start or z > m.end for m in members):
                    # Widen to one shared edge: a translation widened on its own would end before
                    # (and fade out before) the original it is sung with.
                    a = min([a] + [m.start for m in members])
                    z = max([z] + [m.end for m in members])
                for m in members:
                    if a < m.start or z > m.end:
                        tx.member(m.id)
                        m.start, m.end = min(m.start, a), max(m.end, z)
                        report["widened"].append(m.id)
            report["widened_lines"] = len(report["widened"])
            return report

        return self.history.run(label, apply)

    def add_rest(self, mid: str, key: tuple) -> bool:
        """Shorten the selected beat, leaving its successor's start untouched."""
        from .karaoke import effective

        view = effective(self.doc, self.doc.members[mid])
        i = next((i for i, b in enumerate(view.beats) if b.key == key), None)
        if i is None or i + 1 == len(view.beats):
            return False
        beat = view.beats[i]
        duration = beat.t1 - beat.t0
        if duration < 2:
            return False
        beat.t1 -= 200 if duration >= 400 else max(1, duration // 3)
        beat.by = "manual"
        beat.flags = [f for f in beat.flags if f != "check"]
        beat.link.pop("end", None)
        self.set_karaoke(mid, view.beats, state=view.state, unit=view.unit, twins=True, label="后面加空拍")
        return True

    def remove_rest(self, mid: str, left_key: tuple, right_key: tuple) -> bool:
        from .karaoke import effective, rests_of

        view = effective(self.doc, self.doc.members[mid])
        rest = next((r for r in rests_of(view.beats)
                     if (view.beats[r.after].key, view.beats[r.after + 1].key) == (left_key, right_key)), None)
        if rest is None:
            return False
        beat = view.beats[rest.after]
        beat.t1, beat.by = rest.t1, "manual"
        beat.flags = [f for f in beat.flags if f != "check"]
        beat.link.pop("end", None)
        self.set_karaoke(mid, view.beats, state=view.state, unit=view.unit, twins=True, label="删掉空拍")
        return True

    def redistribute_translation(self, gids: Iterable[str], *, only_uncustomized: bool = False,
                                 label: str = "重新分配翻译拍", follow_notes: bool = True) -> dict:
        """Redistribute selected song translations, including twins, in one undo step."""
        from .lyrics_auto import redistribute_in_transaction

        gids = list(dict.fromkeys(gids))
        return self.history.run(label, lambda tx: redistribute_in_transaction(
            self, tx, gids, only_uncustomized=only_uncustomized, follow_notes=follow_notes))

    def apply_lyrics_mapping(self, mappings: Mapping[str, str]) -> int:
        """Replace translation taps/edits with mapped segments, all rows in one undo step."""
        from . import karaoke as kara
        from .lyrics_map import mapped_beats, member, parse
        from .readings import visible_map

        tracks = {}
        for gid, text in mappings.items():
            group = self.doc.group(gid)
            original, translation = member(self.doc, group, "original"), member(self.doc, group, "translation")
            if group.kind != "song" or original is None or translation is None:
                raise ValueError("歌词对应需要歌曲的原声和翻译")
            originals = kara.effective(self.doc, original).beats
            segments = parse(text, originals, visible_map(translation)[0].text)
            for mid in [translation.id] + [tid for tid in self.karaoke_twins(translation.id)
                                          if self.doc.members[tid].group == gid]:
                target = self.doc.members[mid]
                beats, edits, view = mapped_beats(self.doc, target, originals, segments)
                old = kara.tracks_of(self.doc, mid)
                tracks[mid] = (old, kara.beats_to_track(target, beats, unit=view.unit,
                    state="timed" if all(b.timed for b in beats) else "placeholder",
                    kid=old[0]["id"] if old else None,
                    effect=copy.deepcopy(old[0].get("effect")) if old else None, edits=edits))

        if not tracks:
            return 0

        def apply(tx: Transaction) -> int:
            tx.keeps_text = True
            for old, track in tracks.values():
                for previous in old[1:]:
                    tx.remove_track(previous["id"])
                tx.set_track(track)
            return len(mappings)

        return self.history.run("歌词对应", apply)

    def split_english_syllables(self, gid: str) -> int:
        """Split both song lanes through stored edits, preserving linked outer edges."""
        from . import karaoke as kara

        group = self.doc.group(gid)
        if group.kind != "song":
            return 0
        tracks, replacements = {}, {}
        for m in self.doc.members_of(group):
            if m.lane not in ("original", "translation") or m.role == "comment" or m.source_kind == "Comment" \
                    or m.extra.get("karaoke_off"):
                continue
            view = kara.effective(self.doc, m)
            boundaries = kara.english_split_boundaries(self.doc, m, view)
            if not boundaries:
                continue
            edits = kara.edits_of(self.doc, m)
            keys = {kara._boundary_id(at) for at in boundaries}
            edits = [e for e in edits if kara._boundary_id(e.get("at", {})) not in keys]
            edits += [{"op": "split", "at": at} for at in boundaries]
            beats = kara.derive(self.doc, m, view.unit, edits)
            owner = {kara._boundary_id(p.at): b for b in view.beats for p in b.parts}
            pieces = {}
            for b in beats:
                old = owner[kara._boundary_id(b.parts[0].at)]
                pieces.setdefault(old.key, []).append(b)
            for old in view.beats:
                children = pieces[old.key]
                if len(children) == 1:
                    b = children[0]
                    b.t0, b.t1, b.by, b.link = old.t0, old.t1, old.by, copy.deepcopy(old.link)
                    continue
                weights = [sum(p.sung.isascii() and p.sung.isalpha() for p in b.parts) for b in children]
                weights = [max(1, w) for w in weights]
                total, used = sum(weights), 0
                for b, weight in zip(children, weights):
                    b.t0 = old.t0 + (old.t1 - old.t0) * used // total
                    used += weight
                    b.t1 = old.t0 + (old.t1 - old.t0) * used // total
                    b.by = "placeholder"
                children[0].link = {e: copy.deepcopy(v) for e, v in old.link.items() if e == "start"}
                children[-1].link.update({e: copy.deepcopy(v) for e, v in old.link.items() if e == "end"})
                if m.lane == "original":
                    replacements[old.key] = (children[0], children[-1])
            old_tracks = kara.tracks_of(self.doc, m.id)
            track = kara.beats_to_track(m, beats, unit=view.unit, state="placeholder",
                kid=old_tracks[0]["id"] if old_tracks else None,
                effect=copy.deepcopy(old_tracks[0].get("effect")) if old_tracks else None, edits=edits)
            if old_tracks and old_tracks[0].get("issues"):
                track["issues"] = copy.deepcopy(old_tracks[0]["issues"])
            tracks[m.id] = (old_tracks, track)
        if not tracks:
            return 0

        def apply(tx: Transaction) -> int:
            tx.keeps_text = True
            for old_tracks, track in tracks.values():
                for old in old_tracks[1:]:
                    tx.remove_track(old["id"])
                tx.set_track(track)
            # Retarget links before following them, so splitting the original cannot orphan mappings.
            for m in self.doc.members_of(group):
                if m.lane != "translation":
                    continue
                for track in kara.tracks_of(self.doc, m.id):
                    for b in track.get("beats", []):
                        for link in b.get("link", {}).values():
                            key = (kara._ids_of_anchor(link.get("core", [])), int(link.get("k", 0)),
                                   link.get("sung", ""))
                            if key in replacements:
                                tx.track(track["id"])
                                edge = link["edge"]
                                link.update(kara.beat_link(replacements[key][0 if edge == "start" else 1], edge))
            self._follow_karaoke_links(tx, [gid])
            return len(tracks)

        return self.history.run("英文按音节拆开", apply)

    def split_beat(self, mid: str, at: dict, t_ms: Optional[int] = None) -> bool:
        """Add an identity-keyed boundary and its timing, including glow twins, in one undo step."""
        return self._edit_beat(mid, at, "split", t_ms)

    def merge_beats(self, mid: str, at: dict) -> bool:
        """Remove the boundary before a beat and keep the two beats' combined span."""
        return self._edit_beat(mid, at, "merge")

    def _edit_beat(self, mid: str, at: dict, op: str, t_ms: Optional[int] = None) -> bool:
        from .karaoke import _boundary_id, derive, edits_of, effective

        m = self.doc.members[mid]
        if m.extra.get("karaoke_off"):
            return False
        view = effective(self.doc, m)
        key = _boundary_id(at)

        def span(b) -> tuple:
            return tuple(_boundary_id(p.at) for p in b.parts)

        index = next((i for i, b in enumerate(view.beats) if key in span(b)), None)
        if index is None:
            return False
        old = view.beats[index]
        offset = span(old).index(key)
        if op == "split":
            if offset == 0:
                return False
            if old.timed and old.t1 - old.t0 < 20:
                raise ValueError("这一拍不足 20 毫秒，拆开后每拍至少需要 10 毫秒")
            weight = [max(1, len(p.sung)) for p in old.parts]
            cut = old.t0 + (old.t1 - old.t0) * sum(weight[:offset]) // sum(weight) if t_ms is None else int(t_ms)
            if old.t1 - old.t0 >= 20:
                cut = max(old.t0 + 10, min(old.t1 - 10, cut))
            else:
                cut = max(old.t0, min(old.t1, cut))
            changed = {span(old)[:offset]: (old.t0, cut, "manual" if old.timed else "placeholder"),
                       span(old)[offset:]: (cut, old.t1, "manual" if old.timed else "placeholder")}
            changed_links = {span(old)[:offset]: {e: v for e, v in old.link.items() if e == "start"},
                             span(old)[offset:]: {e: v for e, v in old.link.items() if e == "end"}}
        else:
            if offset != 0 or index == 0:
                return False
            first = view.beats[index - 1]
            changed = {span(first) + span(old): (first.t0, old.t1,
                                                "manual" if first.timed and old.timed else "placeholder")}
            changed_links = {span(first) + span(old):
                             {**{e: v for e, v in first.link.items() if e == "start"},
                              **{e: v for e, v in old.link.items() if e == "end"}}}
        edits = [e for e in edits_of(self.doc, m) if _boundary_id(e.get("at", {})) != key]
        edits.append({"op": op, "at": copy.deepcopy(at)})
        beats = derive(self.doc, m, view.unit, edits)
        times = {span(b): (b.t0, b.t1, b.by) for b in view.beats}
        times.update(changed)
        flags = {span(b): b.flags for b in view.beats}
        links = {span(b): b.link for b in view.beats}
        links.update(changed_links)
        for b in beats:
            b.t0, b.t1, b.by = times[span(b)]
            b.link = copy.deepcopy(links.get(span(b), {}))
            b.flags = list(dict.fromkeys(b.flags + flags.get(span(b), []))) if span(b) not in changed else b.flags
        self.set_karaoke(mid, beats, unit=view.unit, state="timed", edits=edits, twins=True,
                         label="拆开拍" if op == "split" else "合并拍")
        return True

    def import_karaoke(self, mid: str, unit: str = "syllable") -> Optional[str]:
        """Read the \\k tags the source line already has into a track (state timed, beats by "import")."""
        from .karaoke import has_k_tags, import_from_tags

        m = self.doc.members[mid]
        if not has_k_tags(m):
            return None
        beats = import_from_tags(self.doc, m, unit)
        return self.set_karaoke(mid, beats, state="timed", unit=unit, label="读入原字幕的 K 轴")

    def clear_karaoke(self, mid: str) -> None:
        from .karaoke import tracks_of

        old = tracks_of(self.doc, mid)

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            for o in old:
                tx.remove_track(o["id"])
            if self.doc.members[mid].lane == "original":
                self._follow_karaoke_links(tx, [self.doc.members[mid].group])
        self.history.run("清除 K 轴", apply)

    def set_karaoke_state(self, mids: list[str], *, state: Optional[str] = None, unit: Optional[str] = None,
                          label: str = "K 轴") -> None:
        """Track fields that do not touch the beats: state (e.g. confirmed) or the split unit."""
        from .karaoke import tracks_of

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            for mid in mids:
                for t in tracks_of(self.doc, mid):
                    tt = tx.track(t["id"])
                    if state is not None:
                        tt["state"] = state
                        if state == "confirmed":
                            for beat in tt.get("beats", []):
                                beat.pop("check", None)
                    if unit is not None:
                        tt["unit"] = unit
            if unit is not None:
                self._follow_karaoke_links(tx, [self.doc.members[mid].group for mid in mids
                                               if self.doc.members[mid].lane == "original"])
        self.history.run(label, apply)

    def set_karaoke_effect(self, mids: list[str], preset: str, params: Optional[dict] = None) -> None:
        """Change the effect of members' karaoke (k / kf / ko / none / a preset name); the beats never change."""
        from .karaoke import tracks_of

        def apply(tx: Transaction) -> None:
            for mid in mids:
                for t in tracks_of(self.doc, mid):
                    tt = tx.track(t["id"])
                    tt["effect"] = {"preset": preset, "params": copy.deepcopy(params or {})}
        self.history.run("K 轴特效", apply)

    def apply_karaoke_effect(self, mid: str, preset: str, params: Optional[dict] = None,
                              *, scope: str = "line", gids: Iterable[str] = ()) -> None:
        """Apply one effect transaction, including glow twins; preserve every existing beat."""
        from .karaoke import beats_to_track, edits_of, effective, tracks_of
        from ..karaoke_fx.presets import validate_params

        params = validate_params(preset, params or {})
        member = self.doc.members[mid]
        if scope == "line":
            mids = [mid]
        elif scope in ("selection", "song"):
            chosen = set(gids)
            if scope == "song":
                from .songs import song_span

                chosen = set(song_span(self.doc, member.group))
                if member.style.split(".", 1)[0] not in ("OP", "ED", "IN") or not chosen:
                    raise ValueError("这一行不属于 OP / ED / 插曲")
            mids = [m for g in self.doc.groups if g.id in chosen and g.kind == "song" for m in g.members
                    if self.doc.members[m].role == "song" and self.doc.members[m].source_kind != "Comment"]
        else:
            raise ValueError(scope)
        mids = list(dict.fromkeys(mids + [t for m in mids for t in self.karaoke_twins(m)]))

        def apply(tx):
            tx.keeps_text = True
            for target in mids:
                m = self.doc.members[target]
                effect = {"preset": preset, "params": copy.deepcopy(params)}
                tracks = tracks_of(self.doc, target)
                if tracks:
                    for track in tracks:
                        tx.track(track["id"])["effect"] = effect
                else:
                    view = effective(self.doc, m)
                    tx.set_track(beats_to_track(m, view.beats, state="placeholder", unit=view.unit,
                                               effect=effect, edits=edits_of(self.doc, m)))
        if mids:
            self.history.run("K 轴特效", apply)

    def set_member_time(self, mid: str, start: Optional[int] = None, end: Optional[int] = None) -> None:
        def apply(tx: Transaction) -> None:
            mm = tx.member(mid)
            before = (mm.start, mm.end)
            if start is not None:
                mm.start = max(0, int(start))
            if end is not None:
                mm.end = max(0, int(end))
            if mm.end < mm.start:
                mm.end = mm.start
            self._record_time_source(mm, before, "manual")
        self.history.run("调整时间", apply)

    def set_group_edge(self, gid: str, *, start: Optional[int] = None, end: Optional[int] = None) -> None:
        """Change the group's start/end edge: members that shared the old edge move with it."""
        g = self.doc.group(gid)
        s0, e0 = self.doc.group_time(g)

        def apply(tx: Transaction) -> None:
            tg = tx.group(gid)
            for mid in tg.members:
                mm = tx.member(mid)
                before = (mm.start, mm.end)
                if start is not None and mm.start == s0:
                    mm.start = max(0, min(int(start), mm.end))
                if end is not None and mm.end == e0:
                    mm.end = max(int(end), mm.start)
                self._record_time_source(mm, before, "manual")
        self.history.run("调整 Group 边界", apply)

    def set_group_time(self, gid: str, *, start: Optional[int] = None, end: Optional[int] = None,
                       target: str = "group", label: Optional[str] = None,
                       source: str = "manual", transaction: Optional[Transaction] = None) -> None:
        """Aegisub-style timing: set the start (left click) or end (right click) of the group's lines.

        ``target``: ``group`` (every line of the group gets this time — the user's finished files have one time
        per group), ``translation`` or ``original`` (only that lane).  If the new start passes the end (or the new
        end passes the start), the line keeps its duration and moves there instead of collapsing.
        """
        if source not in ("manual", "adopted", "autotime"):
            raise ValueError("unknown timing source")

        def apply(tx: Transaction) -> None:
            tg = tx.group(gid)
            for mid in tg.members:
                m = self.doc.members[mid]
                if m.role == "comment" or (target != "group" and m.lane != target):
                    continue
                mm = tx.member(mid)
                before = (mm.start, mm.end)
                dur = max(10, mm.end - mm.start)
                if start is not None:
                    mm.start = max(0, int(start))
                    if mm.end <= mm.start:
                        mm.end = mm.start + dur
                if end is not None:
                    mm.end = max(0, int(end))
                    if mm.end <= mm.start:
                        mm.start = max(0, mm.end - dur)
                self._record_time_source(mm, before, source)
        if transaction is not None:
            apply(transaction)
        else:
            self.history.run(label or ("设开始时间" if start is not None else "设结束时间"), apply)

    def set_exact_group_time(self, gid: str, start: int, end: int, *, target: str = "group",
                             shift_beats: bool = True) -> int:
        from .precise_time import set_time

        return set_time(self, gid, start, end, target=target, shift_beats=shift_beats)

    def set_group_times(self, changes: Iterable[tuple[str, Optional[int], Optional[int]]], *,
                        label: str, source: str = "adopted", avoid_new_overlap: bool = False) -> None:
        """Apply reviewed group edges in one undo step, using normal timing semantics.

        ``avoid_new_overlap``: chronologically adjacent visible dialogue/song groups that did not overlap before do
        not overlap afterwards (see :meth:`_without_new_overlaps`)."""
        changes = [(gid, start, end) for gid, start, end in changes
                   if start is not None or end is not None]
        if source not in ("manual", "adopted", "autotime"):
            raise ValueError("unknown timing source")
        if avoid_new_overlap:
            changes = self._without_new_overlaps(changes)
        if not changes:
            return

        def apply(tx: Transaction) -> None:
            for gid, start, end in changes:
                self.set_group_time(gid, start=start, end=end, source=source, transaction=tx)
        self.history.run(label, apply)

    def unify_lane_times(self, lane: str, *, include_protected: bool = False) -> dict:
        """Dialogue takes the timing of one subtitle file (统一两种语言的时间 / batch 「按…字幕的时间」).

        Lines of ``lane`` go back to the times of their own source events (merging already gave paired
        original lines the translation's times).  In a group with both lanes, the other lane's lines are
        stretched onto that span, keeping their order and proportions (one line simply takes the reference
        start and end).  Other-lane lines without a partner move by the local difference between the lanes,
        fitted on the paired groups like auto-timing's reference map.  Songs (see
        :meth:`align_song_translation_times`), signs, hidden groups and — unless ``include_protected`` —
        confirmed or hand-timed groups stay.  One undo step; returns {"groups", "lines", "shifted", "protected"}."""
        from .. import autotime

        if lane not in ("original", "translation"):
            raise ValueError("unknown lane")
        other = "translation" if lane == "original" else "original"

        def lines(group, which):
            return [m for m in self.doc.members_of(group) if m.lane == which and m.role in ("dialogue", "top", "alt")
                    and m.source_kind != "Comment" and m.end > m.start and m.text.strip()]

        def source_time(m):
            events = {(r.src, r.e) for r in m.runs if r.src in self.sources and r.e is not None}
            events = [self.sources[src].events[e] for src, e in events if 0 <= e < len(self.sources[src].events)]
            return (min(ev.start for ev in events), max(ev.end for ev in events)) if events else (m.start, m.end)

        changes: dict[str, tuple[int, int]] = {}
        samples, unpaired, protected, groups = [], [], 0, 0
        for g in self.doc.groups:
            if g.kind != "dialogue" or "hidden" in g.flags:
                continue
            ref, oth = lines(g, lane), lines(g, other)
            if not ref and not oth:
                continue
            own = {m.id: source_time(m) for m in ref}
            if ref and oth:
                a, z = min(s for s, _e in own.values()), max(e for _s, e in own.values())
                s0, e0 = min(m.start for m in oth), max(m.end for m in oth)
                samples += [(s0, a), (e0, z)]            # protected pairs still measure the difference
            if not include_protected and (g.state == "confirmed" or any(
                    m.extra.get("time_manual") for m in self.doc.members_of(g))):
                protected += 1
                continue
            if not ref:
                unpaired.extend(oth)
                continue
            moved = False
            for m in ref:
                if own[m.id] != (m.start, m.end) and own[m.id][1] > own[m.id][0]:
                    changes[m.id] = own[m.id]
                    moved = True
            if oth:
                scale = (z - a) / (e0 - s0) if e0 > s0 else 1.0
                for m in oth:
                    start, end = a + round((m.start - s0) * scale), a + round((m.end - s0) * scale)
                    if (start, end) != (m.start, m.end):
                        changes[m.id] = (start, max(start, end))
                        moved = True
            groups += moved
        shifted = 0
        if unpaired and len(samples) >= 2:
            mapping = autotime.reference_map(samples, duration_ms=max(m.end for m in self.doc.members.values()))
            for m in unpaired:
                segment = next((s for s in mapping.segments if s.start_ms <= m.start < s.end_ms), mapping.segments[-1])
                if segment.confidence == "low":
                    continue
                start, end = mapping.map(m.start), mapping.map(m.end)
                if (start, end) != (m.start, m.end) and end >= start >= 0:
                    changes[m.id] = (start, end)
                    shifted += 1
        report = {"groups": groups, "lines": len(changes), "shifted": shifted, "protected": protected}
        if not changes:
            return report

        def apply(tx: Transaction) -> dict:
            for mid, (start, end) in changes.items():
                mm = tx.member(mid)
                before = (mm.start, mm.end)
                mm.start, mm.end = start, end
                self._record_time_source(mm, before, "autotime")
            return report

        return self.history.run("统一两种语言的时间", apply)

    def align_song_translation_times(self, gids: Iterable[str]) -> int:
        """Song groups: every lyric line takes the original's start and end (one time per group, so the
        translation fades with the original), widened only to keep the group's timed beats inside.
        Returns how many lines changed; one undo step."""
        from . import karaoke as kara

        changes = []
        for gid in dict.fromkeys(gids):
            group = next((g for g in self.doc.groups if g.id == gid), None)
            if group is None or group.kind != "song":
                continue
            lines = [m for m in self.doc.members_of(group) if m.lane in ("original", "translation")
                     and m.role != "comment" and m.source_kind != "Comment"]
            originals = [m for m in lines if m.lane == "original"]
            if not originals or len(originals) == len(lines):
                continue
            beats = [b for m in lines for b in kara.effective(self.doc, m).beats if b.timed]
            start = min([m.start for m in originals] + [b.t0 for b in beats])
            end = max([m.end for m in originals] + [b.t1 for b in beats])
            changes.extend((m.id, start, end) for m in lines if (m.start, m.end) != (start, end))
        if not changes:
            return 0

        def apply(tx: Transaction) -> int:
            for mid, start, end in changes:
                mm = tx.member(mid)
                before = (mm.start, mm.end)
                mm.start, mm.end = start, end
                self._record_time_source(mm, before, "manual")
            return len(changes)

        return self.history.run("翻译时间对齐原声", apply)

    def song_fade(self, gids: Iterable[str]) -> tuple[Optional[tuple[int, int]], bool]:
        """The fade most lyric lines of the groups show (None when they have no simple \\fad) and whether
        any line already overrides the source's fade."""
        from collections import Counter

        from ..ass.text import line_tag

        def fad(text: str) -> Optional[tuple[int, int]]:
            tag = line_tag(text, "fad")
            try:
                values = [int(float(v)) for v in (tag.arg if tag else "").split(",")]
            except ValueError:
                return None
            return (values[0], values[1]) if len(values) == 2 else None

        seen, overridden = Counter(), False
        for gid in dict.fromkeys(gids):
            group = next((g for g in self.doc.groups if g.id == gid), None)
            if group is None or group.kind != "song":
                continue
            for m in self.doc.members_of(group):
                if m.role == "comment" or m.source_kind == "Comment":
                    continue
                raw = str(m.overrides.get("raw") or "")
                own = "{" + raw + "}" if raw else ""
                if own and (line_tag(own, "fad") or line_tag(own, "fade")):
                    overridden = True
                    seen[fad(own)] += 1
                else:
                    seen[fad("".join(r.text for r in m.runs if r.kind == "tag"))] += 1
        return (seen.most_common(1)[0][0] if seen else None), overridden

    def set_song_fade(self, gids: Iterable[str], fade: Optional[tuple[int, int]]) -> int:
        """Line fade of every lyric line in the song groups: ``(in, out)`` milliseconds written as a line tag
        that replaces the source's own \\fad / \\fade, or None to go back to the source's fade (the source
        itself is never changed).  Returns how many lines changed; one undo step."""
        from ..ass.tags import parse_block

        if fade is not None:
            fade = (max(0, int(fade[0])), max(0, int(fade[1])))
        changes = []
        for gid in dict.fromkeys(gids):
            group = next((g for g in self.doc.groups if g.id == gid), None)
            if group is None or group.kind != "song":
                continue
            for m in self.doc.members_of(group):
                if m.role == "comment" or m.source_kind == "Comment":
                    continue
                old = str(m.overrides.get("raw") or "")
                raw = "".join(t.raw for t in parse_block(old) if t.name not in ("fad", "fade"))
                if fade is not None:
                    raw += f"\\fad({fade[0]},{fade[1]})"
                if raw != old:
                    changes.append((m.id, raw))
        if not changes:
            return 0

        def apply(tx: Transaction) -> int:
            for mid, raw in changes:
                mm = tx.member(mid)
                if raw:
                    mm.overrides["raw"] = raw
                else:
                    mm.overrides.pop("raw", None)
                mm.rev += 1
            return len(changes)

        return self.history.run("淡入淡出", apply)

    def _without_new_overlaps(self, changes: list[tuple[str, Optional[int], Optional[int]]],
                              gap: int = 40) -> list[tuple[str, Optional[int], Optional[int]]]:
        """Trim changed edges so that neighbours which did not overlap keep not overlapping (existing overlaps stay).

        The changed edge gives way, keeping ``gap`` ms when there is room (touching otherwise); when both the earlier
        end and the later start changed, the earlier end gives way.  An edge that cannot give way is dropped."""
        order = sorted((g for g in self.doc.groups if g.members and "hidden" not in g.flags
                        and g.kind in ("dialogue", "song")), key=self.doc.group_time)
        before = {g.id: self.doc.group_time(g) for g in order}
        want = {gid: [start, end] for gid, start, end in changes}

        def now(gid: str) -> tuple[int, int]:
            s, e = before[gid]
            w = want.get(gid, (None, None))
            return (s if w[0] is None else w[0], e if w[1] is None else w[1])

        for a, b in zip(order, order[1:]):
            if before[a.id][1] > before[b.id][0]:
                continue
            (a0, a1), (b0, b1) = now(a.id), now(b.id)
            if a1 > b0 and a.id in want and want[a.id][1] is not None:
                limit = b0 - gap if b0 - gap > a0 else b0
                want[a.id][1] = limit if limit > a0 else None
                (a0, a1), (b0, b1) = now(a.id), now(b.id)
            if a1 > b0 and b.id in want and want[b.id][0] is not None:
                limit = a1 + gap if a1 + gap < b1 else a1
                want[b.id][0] = limit if limit < b1 else None
        return [(gid, *want[gid]) for gid, _s, _e in changes if any(v is not None for v in want[gid])]

    # -------------------------------------------------------------- position / line tags
    def set_overrides(self, changes: dict[str, dict[str, Any]], label: str = "位置") -> None:
        """Merge per-member override changes {member id: {key: value or None (= remove)}} in one undo step."""
        def apply(tx: Transaction) -> None:
            for mid, ch in changes.items():
                mm = tx.member(mid)
                for k, v in ch.items():
                    if v is None:
                        mm.overrides.pop(k, None)
                    else:
                        mm.overrides[k] = v
                mm.rev += 1
        self.history.run(label, apply)

    def set_vertical(self, gids: Iterable[str], where: str = "top", *,
                     styles: Optional[Mapping[str, str]] = None) -> VerticalResult:
        """Use template top/bottom styles, skipping source-positioned/protected members.

        Returns VerticalResult with changed groups/members and explicit skip reasons.
        """
        from .merge import MergeSettings
        from .position import PositionSkip, VerticalResult, source_positioned

        if where not in ("top", "bottom", "alt"):
            raise ValueError(where)
        styles = {**MergeSettings().styles, **(styles or {})}
        role = {"top": "top", "bottom": "dialogue", "alt": "alt"}[where]
        selected = set(gids)
        result = VerticalResult()
        changes = []
        for g in self.doc.groups:
            if g.id not in selected:
                continue
            if g.kind != "dialogue":
                result.skipped.append(PositionSkip(g.id, None, "歌曲、屏幕字或制作信息，不改位置"))
                continue
            for m in self.doc.members_of(g):
                reason = ("歌曲、屏幕字或注释，不改位置" if m.role not in ("dialogue", "alt", "top")
                          else "原字幕自带位置，不改位置" if source_positioned(m, self.sources) else "")
                if reason:
                    result.skipped.append(PositionSkip(g.id, m.id, reason))
                    continue
                style = styles[f"{m.lane}/{role}"]
                if m.role == role and m.style == style and not any(k in m.overrides for k in ("an", "pos", "move")):
                    continue
                changes.append((m.id, style))
                result.members.append(m.id)
                if g.id not in result.gids:
                    result.gids.append(g.id)
        live = {g.id for g in self.doc.groups}
        result.skipped.extend(PositionSkip(gid, None, "组不存在，不改位置") for gid in sorted(selected - live))
        if not changes:
            return result

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            for mid, style in changes:
                m = tx.member(mid)
                m.role, m.style = role, style
                for key in ("an", "pos", "move"):
                    m.overrides.pop(key, None)
                m.rev += 1
        self.history.run({"top": "移到顶部", "bottom": "移到底部", "alt": "底部第二人"}[where], apply)
        return result

    # -------------------------------------------------------------- character formatting
    def set_format(self, mid: str, change: dict[str, Any], a: Optional[int] = None, b: Optional[int] = None,
                   *, label: str = "格式") -> None:
        """Font / size / bold / italic / colour / raw override tags for the whole line (a, b = None) or for visible
        [a, b).  A value None removes that property.  Raw tags are checked (ValueError with the reason).
        One undo step."""
        from .textformat import PROPS, check_raw, with_span

        change = {k: v for k, v in change.items() if k in PROPS}
        if "raw" in change:
            raw, err = check_raw(str(change["raw"] or ""), span=a is not None and b is not None)
            if err:
                raise ValueError(err)
            change["raw"] = raw or None

        def apply(tx: Transaction) -> None:
            mm = tx.member(mid)
            if a is None or b is None:
                for k, v in change.items():
                    if v is None:
                        mm.overrides.pop(k, None)
                    else:
                        mm.overrides[k] = v
            else:
                mm.format = with_span(mm, a, b, change)
            mm.rev += 1
        self.history.run(label, apply)

    def clear_format(self, mid: str) -> None:
        """Remove every font / size / bold / italic / colour / raw tag setting of a line (position is kept)."""
        from .textformat import PROPS

        def apply(tx: Transaction) -> None:
            mm = tx.member(mid)
            for k in PROPS:
                mm.overrides.pop(k, None)
            mm.format = []
            mm.rev += 1
        self.history.run("清除格式", apply)

    # -------------------------------------------------------------- review state
    def set_group_flag(self, gids: Iterable[str], flag: str, value: bool) -> None:
        """Change review flags without touching members, in one undo step."""
        if flag not in ("hidden", "marked"):
            raise ValueError(flag)
        live = {g.id: g for g in self.doc.groups}
        gids = [gid for gid in dict.fromkeys(gids) if gid in live and (flag in live[gid].flags) != value]
        if not gids:
            return

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            for gid in gids:
                group = tx.group(gid)
                group.flags = [f for f in group.flags if f != flag]
                if value:
                    group.flags.append(flag)
        label = ("隐藏" if value else "取消隐藏") if flag == "hidden" else ("标记待复查" if value else "取消标记")
        self.history.run(label, apply)

    def hide_many(self, gids: Iterable[str], value: bool = True) -> None:
        self.set_group_flag(gids, "hidden", value)

    def mark_many(self, gids: Iterable[str], value: bool = True) -> None:
        self.set_group_flag(gids, "marked", value)

    def set_mark_note(self, gids: Iterable[str], note: str) -> None:
        live = {g.id: g for g in self.doc.groups}
        gids = [gid for gid in dict.fromkeys(gids) if gid in live and live[gid].extra.get("mark_note", "") != note]
        if not gids:
            return

        def apply(tx: Transaction) -> None:
            tx.keeps_text = True
            for gid in gids:
                group = tx.group(gid)
                if note:
                    group.extra["mark_note"] = note
                else:
                    group.extra.pop("mark_note", None)
        self.history.run("编辑标记备注", apply)

    def confirm(self, gid: str, value: bool = True) -> None:
        def apply(tx: Transaction) -> None:
            tx.group(gid).state = "confirmed" if value else "auto"
        self.history.run("确认" if value else "取消确认", apply)

    def confirm_many(self, gids: list[str], value: bool = True) -> None:
        """Mark several groups confirmed / not confirmed in one undo step."""
        live = {g.id for g in self.doc.groups}
        gids = [g for g in dict.fromkeys(gids) if g in live]
        if not gids:
            return

        def apply(tx: Transaction) -> None:
            for gid in gids:
                tx.group(gid).state = "confirmed" if value else "auto"
        self.history.run(f"{'确认' if value else '取消确认'} ×{len(gids)}", apply)

    # -------------------------------------------------------------- annotations
    def fix_self_reference(self, mismatch: dict) -> str:
        from .self_check import fix

        return fix(self, mismatch)

    def keep_self_reference(self, mismatch: dict, *, add_exception: bool = False, line_exception: bool = False, book=None) -> str:
        from .self_check import keep

        return keep(self, mismatch, add_exception=add_exception, line_exception=line_exception, book=book)

    def set_reading(self, aid: str, text: str, *, status: str = "confirmed", origin: str = "manual") -> None:
        """Change a reading.  The ruby pieces are recomputed so okurigana stays bare
        (合わ: typing あわ or あ both give 合(あ)わ)."""
        from .readings import ruby_parts

        if self.doc.annotations[aid].kind == "pinyin":
            from .pinyin_notes import normalize_pinyin

            reading = normalize_pinyin(text)

            def apply_pinyin(tx: Transaction) -> None:
                tx.keeps_text = True
                ann = tx.annotation(aid)
                tx.member(ann.member)
                ann.text, ann.parts = reading, [[0, 1, reading]]
                ann.status, ann.origin = status, origin
                ann.candidates = list(dict.fromkeys([reading] + ann.candidates))
            self.history.run(f"拼音：{self.doc.annotations[aid].base}", apply_pinyin)
            return

        def apply(tx: Transaction) -> None:
            from ..textutil import kata_to_hira

            a = tx.annotation(aid)
            reading = text
            parts = ruby_parts(a.base, reading)
            # a reading typed for the kanji only (合わ ← あ): store the whole-word reading
            if parts and len(parts) == 1:
                tail = a.base[parts[0][1]:]
                if tail and all(not ("一" <= c <= "鿿") for c in tail) and not kata_to_hira(reading).endswith(kata_to_hira(tail)):
                    reading = reading + kata_to_hira(tail)
            a.text = reading
            a.parts = ruby_parts(a.base, reading)
            if reading not in a.candidates:
                a.candidates = [reading] + a.candidates
            a.status = status
            a.origin = origin
        self.history.run("修改读音", apply)

    def _drop_detached(self, tx: Transaction, m: Member) -> None:
        ids = {ident for ident, *_ in m.identities() if ident is not None}
        for a in list(self.doc.annotations.values()):
            if a.member == m.id and not _anchor_ids(a) <= ids:
                tx.annotation(a.id)
                del self.doc.annotations[a.id]


def _first_key(runs: list[Run]) -> Optional[tuple]:
    return next(((r.src, r.e, r.a) for r in runs if r.src is not None and r.kind != "sep"), None)


def _last_key(runs: list[Run]) -> Optional[tuple]:
    return next(((r.src, r.e, r.b) for r in reversed(runs) if r.src is not None and r.kind != "sep"), None)


def _anchor_ids(a: Annotation) -> set:
    out = set()
    for seg in a.anchor:
        if "src" in seg:
            out.update(("s", seg["src"], seg["e"], o) for o in range(seg["a"], seg["b"]))
        else:
            out.update(("i", seg["ins"], o) for o in range(seg["a"], seg["b"]))
    return out
