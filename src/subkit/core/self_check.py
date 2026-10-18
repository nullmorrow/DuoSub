"""Scoped Japanese self-reference checks and single-transaction decisions (Qt-free)."""

from __future__ import annotations

import copy

from ..languages import get_language
from ..model.document import Annotation
from ..model.ids import new_id
from ..textutil import kata_to_hira
from .cc import normalize_speaker
from .people import needs_speaker
from .readings import anchor_of, anchor_positions, pieces_of, ruby_parts, visible_map


def _expected(character, line, episode, season):
    selected = {}
    for ref in character.self_refs:
        scope = ref.get("scope") or {}
        if "line" in scope:
            priority, applies = 3, scope["line"] == line
        elif "episodes" in scope:
            priority, applies = 2, episode in scope["episodes"]
        elif "season" in scope:
            priority, applies = 1, scope["season"] == season
        else:
            priority, applies = 0, not scope
        surface, reading = ref.get("surface", ""), ref.get("reading", "")
        if applies and surface and reading and priority >= selected.get(surface, (-1,))[0]:
            selected[surface] = (priority, reading)
    return {surface: reading for surface, (_priority, reading) in selected.items()}


def _annotations(doc, member, ids):
    return [(ann, pos) for ann in doc.annotations_of(member.id) if ann.kind == "ruby"
            and (pos := anchor_positions(ann, ids)) is not None]


def _current(text, at, annotations):
    a, b = at
    overlapping = [(ann, pos) for ann, pos in annotations if pos[0] < b and a < pos[1]]
    protected = any(ann.status in ("confirmed", "hidden") for ann, _pos in overlapping)
    for ann, pos in overlapping:
        if pos == at and ann.status != "hidden":
            return ann.text or None, protected
    pieces = sorted((pos[0] + x, pos[0] + y, reading)
                    for ann, pos in overlapping if ann.status != "hidden"
                    for x, y, reading in pieces_of(ann))
    reading, cursor = [], a
    for x, y, value in pieces:
        if y <= a or x >= b:
            continue
        if x < cursor or y > b:
            return None, protected  # a whole-word ruby cannot be divided by guessing
        gap = text[cursor:x]
        if any(not ("ぁ" <= c <= "ヿ") for c in gap):
            return None, protected
        reading.extend((kata_to_hira(gap), value))
        cursor = y
    tail = text[cursor:b]
    if any(not ("ぁ" <= c <= "ヿ") for c in tail):
        return None, protected
    return "".join(reading) + kata_to_hira(tail) or None, protected


def check(doc, book, episode=None, season=None) -> dict[str, list[dict]]:
    """Return ``mismatches`` and separate ``unknown_speakers`` occurrences of 私.

    Confirmed/hidden ruby is a human decision and is not listed again. ``at`` is
    a visible [start, end) pair; IDs, rather than mutable model objects, are returned.
    """
    result = {"mismatches": [], "unknown_speakers": []}
    characters = {ch.id: ch for ch in book.characters}
    for group in doc.groups:
        for member in doc.members_of(group):
            if (get_language(member.lang).annotation_system != "ruby"
                    or not needs_speaker(group, member) or book.speaker_ignored(member)
                    or member.source_kind == "Comment"):
                continue
            cid = (member.speaker or {}).get("character")
            ch = characters.get(cid)
            label = (member.speaker or {}).get("label", "")
            if cid in ("crowd", "nobody") or any(label and normalize_speaker(label) == normalize_speaker(ignored)
                                     for ignored in book.not_person):
                continue
            expected = _expected(ch, member.id, episode or doc.episode, season) if ch else {"私": None}
            vt, ids = visible_map(member)
            anns = _annotations(doc, member, ids)
            for surface, reading in expected.items():
                start = 0
                while (a := vt.text.find(surface, start)) >= 0:
                    b = a + len(surface)
                    start = b
                    if vt.chars[a].line != vt.chars[b - 1].line:
                        continue
                    current, protected = _current(vt.text, (a, b), anns)
                    item = {"member": member.id, "group": group.id, "at": [a, b],
                            "surface": surface, "current": current, "expected": reading,
                            "speaker": cid if ch else None, "time": member.start}
                    if ch is None:
                        result["unknown_speakers"].append(item)
                    elif not protected and kata_to_hira(current or "") != kata_to_hira(reading):
                        result["mismatches"].append(item)
    return result


def _decide(editor, item, *, keep=False, add_exception=False, line_exception=False, book=None):
    member = editor.doc.members[item["member"]]
    if get_language(member.lang).annotation_system != "ruby" or (
            member.speaker or {}).get("character") != item["speaker"]:
        raise ValueError("Self-reference speaker or language has changed; check again")
    vt, ids = visible_map(member)
    a, b = item["at"]
    if (not 0 <= a < b <= len(ids) or vt.text[a:b] != item["surface"]
            or member.group != item["group"] or any(x is None for x in ids[a:b])):
        raise ValueError("Self-reference occurrence is no longer valid")
    anns = _annotations(editor.doc, member, ids)
    current, _protected = _current(vt.text, (a, b), anns)
    if current != item["current"]:
        raise ValueError("Self-reference reading has changed; check again")
    reading = current if keep else item["expected"]
    if not keep and not reading:
        raise ValueError("An expected reading is required")
    book = book if book is not None else editor.character_book
    if add_exception or line_exception:
        if not keep or book is None or not current:
            raise ValueError("A current reading and CharacterBook are required for an exception")
        character = book.character(item["speaker"])
        if (member.speaker or {}).get("character") != character.id:
            raise ValueError("Self-reference speaker has changed; check again")
    overlapping = [(ann, pos) for ann, pos in anns if pos[0] < b and a < pos[1]]
    for ann, pos in overlapping:
        for x, y, _value in pieces_of(ann):
            x, y = pos[0] + x, pos[0] + y
            if x < a < y or x < b < y:
                raise ValueError("Cannot split a whole-word ruby; edit its reading range first")

    def apply(tx):
        tx.keeps_text = True
        tx.member(member.id)
        for ann, pos in overlapping:
            tx.annotation(ann.id)
            del editor.doc.annotations[ann.id]
            # Preserve ruby outside the chosen occurrence instead of deleting the whole word.
            for x, y, value in pieces_of(ann):
                x, y = pos[0] + x, pos[0] + y
                if y <= a or x >= b:
                    remainder = copy.deepcopy(ann)
                    remainder.id = new_id("a_")
                    remainder.anchor = anchor_of(ids[x:y])
                    remainder.base, remainder.text = vt.text[x:y], value
                    remainder.parts = [[0, y - x, value]]
                    tx.add_annotation(remainder)
        annotation = Annotation(new_id("a_"), "ruby", member.id, anchor_of(ids[a:b]),
                                vt.text[a:b], reading or "", status="confirmed", origin="manual",
                                parts=ruby_parts(vt.text[a:b], reading or ""))
        tx.add_annotation(annotation)
        if add_exception or line_exception:
            tx.external(book.snapshot, book.restore)
            scope = {"line": member.id} if line_exception else {"episodes": [editor.doc.episode]}
            refs = [ref for ref in character.self_refs if not
                    (ref.get("surface") == item["surface"] and ref.get("scope") == scope)]
            ref = {"surface": item["surface"], "reading": current, "scope": scope}
            if line_exception:
                ref["display_hint"] = {"episode": editor.doc.episode,
                                       "number": editor.doc.group_index(member.group) + 1,
                                       "text": vt.text[:12]}
            refs.append(ref)
            book.update_character(character.id, self_refs=refs)
            tx.people_changed = True
        return annotation.id

    return editor.history.run("自称检查：保留" if keep else "自称检查：改读音", apply)


def fix(editor, item):
    return _decide(editor, item)


def keep(editor, item, *, add_exception=False, line_exception=False, book=None):
    return _decide(editor, item, keep=True, add_exception=add_exception, line_exception=line_exception, book=book)
