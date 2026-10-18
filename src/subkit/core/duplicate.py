"""Exact group copies with new character identities (one Editor transaction).

The visible result is copied; the original source-character ledger is untouched.
Unlike passage memories this includes every member, annotation and local format.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ..model.document import Group, Run
from ..model.ids import new_id
from . import karaoke as kara
from .readings import anchor_of

if TYPE_CHECKING:
    from .ops import Transaction


def _runs(runs: list[Run]) -> tuple[list[Run], dict]:
    copied, mapping = [], {}
    for old in runs:
        if old.kind == "sep":
            copied.append(old.copy())
            continue
        new = Run(old.text, old.kind, ins=new_id("i_"), a=0, b=len(old.text),
                  render_src=old.src or old.render_src, render_e=old.e if old.src is not None else old.render_e)
        copied.append(new)
        for i in range(len(old.text)):
            ident = old.identity(i)
            if ident is not None:
                mapping[ident] = new.identity(i)
    return copied, mapping


def _anchor(anchor: list[dict], mapping: dict) -> list[dict]:
    ids = kara._ids_of_anchor(anchor)
    return anchor_of([mapping[i] for i in ids]) if ids and all(i in mapping for i in ids) else []


def _track(track: dict, mid: str, mapping: dict) -> dict:
    out = copy.deepcopy(track)
    out["id"], out["member"] = new_id("k_"), mid
    out["beats"] = []
    for beat in track.get("beats", []):
        core = _anchor(beat.get("core", []), mapping)
        if core:
            b = copy.deepcopy(beat)
            b["core"] = core
            b.pop("link", None)  # A copy must not follow the original-language beats.
            out["beats"].append(b)
    if "edits" in out:
        out["edits"] = []
        for edit in track["edits"]:
            core = _anchor(edit.get("at", {}).get("core", []), mapping)
            if core:
                e = copy.deepcopy(edit)
                e["at"]["core"] = core
                out["edits"].append(e)
    return out


def apply(tx: Transaction, selected: set[str]) -> list[str]:
    doc = tx.doc
    tx.keeps_text = True  # The session must not regenerate the copied automatic readings.
    order = tx.order()
    original_groups = list(order)
    annotations = list(doc.annotations.values())
    index = kara.KIndex(doc)
    created, new_order = [], []
    for group in original_groups:
        new_order.append(group)
        if group.id not in selected:
            continue
        g = Group(new_id("g_"), group.kind, state="auto",
                  origin={"by": "user", "op": "duplicate", "from": group.id},
                  flags=[f for f in group.flags if f not in ("hidden", "marked")],
                  extra=copy.deepcopy(group.extra))
        g.extra.pop("mark_note", None)
        tx.group(g.id)  # Capture the absent pre-image before inserting the group.
        member_ids = {}
        for mid in group.members:
            original = doc.members[mid]
            m = copy.deepcopy(original)
            m.id, m.group, m.rev = new_id("m_"), g.id, 0
            m.runs, mapping = _runs(original.runs)
            m.format = []
            for span in original.format:
                anchor = _anchor(span.get("anchor", []), mapping)
                if anchor:
                    m.format.append({**copy.deepcopy(span), "anchor": anchor})
            tx.add_member(m)
            member_ids[mid] = m.id
            g.members.append(m.id)
            annotation_ids = {}
            for old in annotations:
                if old.member != mid:
                    continue
                anchor = _anchor(old.anchor, mapping)
                if anchor:
                    ann = copy.deepcopy(old)
                    ann.id, ann.member, ann.anchor = new_id("a_"), m.id, anchor
                    tx.add_annotation(ann)
                    annotation_ids[old.id] = ann.id
            saved = m.extra.get("before_sign_convert", {})
            if "generated" in saved:
                # Conversion keeps pre-images of automatically generated ruby.
                # Its cleanup must refer to the copy, never the original member.
                generated = {}
                for aid, before in saved["generated"].items():
                    if aid in annotation_ids:
                        before_image = copy.deepcopy(before)
                        before_image.update(id=annotation_ids[aid], member=m.id,
                                            anchor=_anchor(before_image.get("anchor", []), mapping))
                        generated[annotation_ids[aid]] = before_image
                saved["generated"] = generated
            # Materialize the effective current track: timing/edits may have followed
            # characters here from another member, rather than being owned by mid.
            if index.has(original):
                view = kara.effective(doc, original, idx=index)
                track = kara.beats_to_track(original, view.beats, state=view.state, unit=view.unit,
                                           effect={"preset": view.kind, "params": view.params},
                                           edits=kara.edits_of(doc, original, index))
                if view.issues:
                    track["issues"] = list(view.issues)
                tx.set_track(_track(track, m.id, mapping))
            # Converted lyrics also retain their inactive track for later restoration.
            for track in kara.tracks_of(doc, mid, dormant=True):
                if track.get("dormant") or original.extra.get("karaoke_off"):
                    tx.set_track(_track(track, m.id, mapping))
        for cid in group.comments:
            c = copy.deepcopy(doc.comments[cid])
            c.id, c.group = new_id("c_"), g.id
            c.runs, _ = _runs(c.runs)
            for key in ("before", "after"):
                if c.position.get(key) in member_ids:
                    c.position[key] = member_ids[c.position[key]]
            tx.add_comment(c)
            g.comments.append(c.id)
        new_order.append(g)
        created.append(g.id)
    order[:] = new_order
    doc.structure_changed()
    return created
