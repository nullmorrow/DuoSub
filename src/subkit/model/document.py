"""The working document (FORMAT-v1 §6): ordered Groups of Members whose text is
made of provenance-carrying runs, plus Comments, Annotations, Karaoke and Tombstones.

Character identity
------------------
Every character of a member's visible text has an identity:

* ``("s", source_id, event_index, offset)`` — it came from a source file;
* ``("i", insertion_id, offset)``           — the user typed it.

Annotations (ruby / pinyin / gloss) and karaoke syllables are anchored to
identities, so they follow the characters wherever editing moves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

FORMAT_DOCUMENT = "subkit-document"

RUN_KINDS = ("text", "tag", "sep")
GROUP_KINDS = ("dialogue", "song", "sign", "info")
MEMBER_ROLES = ("dialogue", "top", "alt", "song", "sign", "info", "comment")
COMMENT_KINDS = ("hidden_filler", "user_hidden", "note")
ANNOTATION_KINDS = ("ruby", "pinyin", "gloss")
ANNOTATION_STATUS = ("auto", "uncertain", "confirmed", "hidden")
SEP_TEXTS = (" ", "  ", "　", "\\N", "\\n", "\\h")

Identity = tuple  # ("s", src, e, off) | ("i", ins, off)


@dataclass
class Run:
    text: str
    kind: str = "text"                 # text | tag | sep
    src: Optional[str] = None          # source id (cn1, ja1 …)
    e: Optional[int] = None            # source event index
    a: int = 0                         # offset range in the source event text / insertion text
    b: int = 0
    ins: Optional[str] = None          # insertion id for user-typed text
    render_src: Optional[str] = None   # retained ASS context only; never character provenance
    render_e: Optional[int] = None     # source event whose rendering context the copy retains

    @property
    def is_source(self) -> bool:
        return self.src is not None

    def identity(self, k: int) -> Optional[Identity]:
        """Identity of the k-th character of this run (None for sep runs)."""
        if self.kind == "sep":
            return None
        if self.src is not None:
            return ("s", self.src, self.e, self.a + k)
        if self.ins is not None:
            return ("i", self.ins, self.a + k)
        return None

    def copy(self, **kw: Any) -> "Run":
        d = dict(text=self.text, kind=self.kind, src=self.src, e=self.e, a=self.a, b=self.b, ins=self.ins,
                 render_src=self.render_src, render_e=self.render_e)
        d.update(kw)
        return Run(**d)

    def slice(self, i: int, j: int) -> "Run":
        """Sub-run for characters [i, j) of this run's text (text/tag runs with provenance)."""
        return self.copy(text=self.text[i:j], a=self.a + i, b=self.a + j)


@dataclass
class Member:
    id: str
    group: str
    lang: str
    lane: str                          # original (spoken language) | translation
    role: str                          # dialogue | top | alt | song | sign | info | comment
    start: int                         # ms
    end: int
    layer: int = 0
    style: str = ""
    margins: list[int] = field(default_factory=lambda: [0, 0, 0])
    effect: str = ""
    speaker: Optional[dict[str, Any]] = None
    runs: list[Run] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    format: list[dict[str, Any]] = field(default_factory=list)   # part-of-line formatting (core.textformat)
    source_kind: str = "Dialogue"      # Dialogue | Comment (as in the source / export)
    rev: int = 0                       # revision counter (bumped by every edit)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(r.text for r in self.runs)

    def visible_runs(self) -> Iterator[tuple[int, Run]]:
        for i, r in enumerate(self.runs):
            if r.kind != "tag":
                yield i, r

    def identities(self) -> list[tuple[Optional[Identity], str, int, int]]:
        """(identity, char, run index, offset in run) for every character outside tags."""
        out = []
        for ri, r in enumerate(self.runs):
            if r.kind == "tag":
                continue
            for k, ch in enumerate(r.text):
                out.append((r.identity(k), ch, ri, k))
        return out


@dataclass
class Comment:
    id: str
    group: str
    kind: str                          # hidden_filler | user_hidden | note
    lang: Optional[str] = None
    runs: list[Run] = field(default_factory=list)
    position: dict[str, Any] = field(default_factory=dict)   # {"before": m} | {"after": m} | {"only": True}
    by: str = "tool"
    restorable: bool = True
    text: str = ""                     # note text
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return "".join(r.text for r in self.runs) if self.runs else self.text


@dataclass
class Group:
    id: str
    kind: str = "dialogue"
    members: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    state: str = "auto"                # auto | confirmed
    origin: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotation:
    id: str
    kind: str                          # ruby | pinyin | gloss
    member: str
    anchor: list[dict[str, Any]]       # [{"src","e","a","b"}] | [{"ins","a","b"}]
    base: str
    text: str
    status: str = "auto"
    origin: str = "analyzer"
    rule: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    mono: bool = False
    parts: list[list[Any]] = field(default_factory=list)   # [[a, b, text], …] relative to ``base`` (上流|階級)
    layout: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tombstone:
    src: str
    e: int
    a: int
    b: int
    text: str
    op: str
    member: Optional[str] = None
    by: str = "user"
    at: str = ""


@dataclass
class Document:
    episode: str
    revision: int = 0
    groups: list[Group] = field(default_factory=list)
    members: dict[str, Member] = field(default_factory=dict)
    comments: dict[str, Comment] = field(default_factory=dict)
    annotations: dict[str, Annotation] = field(default_factory=dict)
    karaoke: dict[str, dict[str, Any]] = field(default_factory=dict)
    tombstones: list[Tombstone] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    _gindex: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _gindex_valid: bool = field(default=False, repr=False, compare=False)

    # ------------------------------------------------------------ indexes
    def structure_changed(self) -> None:
        self._gindex_valid = False

    def group_index(self, group_id: str) -> int:
        if not self._gindex_valid:
            self._gindex = {g.id: i for i, g in enumerate(self.groups)}
            self._gindex_valid = True
        return self._gindex[group_id]

    def group(self, group_id: str) -> Group:
        return self.groups[self.group_index(group_id)]

    def group_of(self, member_id: str) -> Group:
        return self.group(self.members[member_id].group)

    def members_of(self, group: Group, lang: Optional[str] = None, lane: Optional[str] = None) -> list[Member]:
        out = [self.members[m] for m in group.members]
        if lang is not None:
            out = [m for m in out if m.lang == lang]
        if lane is not None:
            out = [m for m in out if m.lane == lane]
        return out

    def annotations_of(self, member_id: str) -> list[Annotation]:
        return [a for a in self.annotations.values() if a.member == member_id]

    def group_langs(self, group: Group) -> set[str]:
        """Languages that have visible text in this group (comment-only members excluded)."""
        return {m.lang for m in self.members_of(group) if m.role != "comment" and _has_visible(m)}

    def group_time(self, group: Group) -> tuple[int, int]:
        ms = self.members_of(group)
        if not ms:
            return (0, 0)
        return (min(m.start for m in ms), max(m.end for m in ms))

    def hidden_count(self, group: Group) -> int:
        return sum(1 for c in group.comments if self.comments[c].kind in ("hidden_filler", "user_hidden"))


def _has_visible(m: Member) -> bool:
    return any(r.kind == "text" and r.text.strip() for r in m.runs)
