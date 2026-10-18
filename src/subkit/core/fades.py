"""Read-only fade inheritance for paired dialogue, shared by rendering and the editor.

Only a unique, valid top-level ``fad`` or ``fade`` is inherited. Different
timelines or translation fades need a human decision; no source or member is
changed. Reasons are stable identifiers for an editor to explain its result.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..ass.tags import Tag, parse_block
from ..ass.text import plain_text, tokenize
from ..languages import get_language
from ..model.document import Document, Group, Member

FADE_TAGS = {"fad", "fade"}


@dataclass(frozen=True)
class FadeInheritance:
    raw: str = ""
    donors: tuple[str, ...] = ()
    reason: str = ""


def _tags(text: str) -> list[Tag]:
    return [tag for token in tokenize(text) if token.kind == "tags"
            for tag in token.tags if tag.name in FADE_TAGS]


def _own_fade(member: Member) -> bool:
    return bool(_tags(member.text) or
                any(t.name in FADE_TAGS for t in parse_block(str(member.overrides.get("raw") or ""))) or
                any(t.name in FADE_TAGS for span in member.format
                    for t in parse_block(str((span.get("props") or {}).get("raw") or ""))))


def _fade(member: Member) -> tuple[str, tuple | None]:
    # Whole-line user overrides replace both source fade spellings, as in apply_overrides.
    tags = [t for t in parse_block(str(member.overrides.get("raw") or "")) if t.name in FADE_TAGS]
    tags = tags or _tags(member.text)
    if not tags or len({t.name for t in tags}) != 1:
        return "", None
    values = []
    for tag in tags:
        args = tag.float_args()
        if (not tag.paren or tag.raw.count("(") != 1 or tag.raw.count(")") != 1
                or args is None or len(args) != (2 if tag.name == "fad" else 7)):
            return "", None
        if not all(math.isfinite(v) and v >= 0 for v in args):
            return "", None
        if tag.name == "fade" and (any(v > 255 for v in args[:3]) or args[3:] != sorted(args[3:])):
            return "", None
        values.append((tag.name, tuple(args)))
    if any(value != values[0] for value in values[1:]):
        return "", None
    return tags[0].raw, values[0]


def _time(member: Member) -> tuple[int, int]:
    # Same quantisation/clamping as export.ms_to_cs and _layout_member.
    return tuple(max(0, (int(round(t)) + 5) // 10) for t in (member.start, member.end))


def _visible(member: Member) -> bool:
    return (member.role in ("dialogue", "top", "alt") and member.source_kind != "Comment"
            and bool(plain_text(member.text).strip()))


def _group_fades(doc: Document, group: Group) -> dict[str, FadeInheritance]:
    targets = [m for m in doc.members_of(group) if m.lane == "original"
               and get_language(m.lang).code == "ja" and _visible(m)]
    if not targets:
        return {}
    if group.kind != "dialogue" or "hidden" in group.flags:
        return {m.id: FadeInheritance(reason="not_visible_dialogue") for m in targets}
    donors = [m for m in doc.members_of(group) if m.lane == "translation" and _visible(m)]
    donor_ids = tuple(m.id for m in donors)
    fades = [_fade(m) for m in donors]
    out = {}
    for member in targets:
        reason, raw = "", ""
        if _own_fade(member):
            reason = "own_fade"
        elif not donors:
            reason = "no_translation"
        elif any(value is None for _raw, value in fades):
            reason = "missing_or_invalid_translation_fade"
        elif any(value != fades[0][1] for _raw, value in fades[1:]):
            reason = "conflicting_translation_fades"
        elif _time(member)[1] <= _time(member)[0] or any(_time(m) != _time(member) for m in donors):
            reason = "different_times"
        else:
            raw, reason = fades[0][0], "inherited"
        out[member.id] = FadeInheritance(raw, donor_ids if raw else (), reason)
    return out


def inherited_fades(doc: Document) -> dict[str, FadeInheritance]:
    """Per-Japanese-member derived fade and explanation, without changing the document."""
    out = {}
    for group in doc.groups:
        out.update(_group_fades(doc, group))
    return out


def query_fade_inheritance(doc: Document, mid: str) -> FadeInheritance:
    """Explain one member's inheritance for an editor, including explicit ``fad(0,0)`` opt-outs."""
    member = doc.members.get(mid)
    group = next((g for g in doc.groups if member is not None and mid in g.members), None)
    if group is None:
        return FadeInheritance(reason="missing_member")
    return _group_fades(doc, group).get(mid, FadeInheritance(reason="not_japanese_dialogue"))
