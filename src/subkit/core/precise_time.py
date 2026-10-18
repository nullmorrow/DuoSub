"""Set both group edges without changing the time of an unselected language."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ass.text import tokenize
from ..ass.tags import canonical
from . import karaoke as kara
from .readings import visible_map

if TYPE_CHECKING:
    from .ops import Editor


def set_time(editor: Editor, gid: str, start: int, end: int, *, target: str = "group",
             shift_beats: bool = True) -> int:
    if target not in ("group", "original", "translation"):
        raise ValueError("unknown timing target")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ValueError("结束要晚于开始，开始不能早于 0")
    doc = editor.doc
    group = doc.group(gid)
    members = [m for m in doc.members_of(group)
               if m.role != "comment" and (target == "group" or m.lane == target)]
    if not members:
        return 0
    shifts = {i: start - m.start for m in members for i in visible_map(m)[1] if i is not None}
    index = kara.KIndex(doc)
    raw_tracks = []
    if group.kind == "song":
        for m in members:
            if kara.has_k_tags(m) and not index.has(m) and not m.extra.get("karaoke_off"):
                # Capture before changing the row's start. Even with the box off,
                # source \k beats must keep their old absolute times.
                preset = next((canonical(tag.name) for tok in tokenize(m.text) if tok.kind == "tags"
                               for tag in tok.tags if canonical(tag.name) in ("k", "kf", "ko")), "kf")
                raw_tracks.append(kara.beats_to_track(m, kara.import_from_tags(doc, m), state="timed",
                                                      effect={"preset": preset}))

    def apply(tx):
        tx.keeps_text = True
        for track in raw_tracks:
            tx.set_track(track)
        editor.set_group_time(gid, start=start, end=end, target=target, transaction=tx)
        for m in members:
            # Accepting the values is an explicit manual decision, including an
            # unchanged edge; later automatic timing must respect both edges.
            doc.members[m.id].extra.update(time_manual=True, time_src={"start": "manual", "end": "manual"})
        if shift_beats:
            for kid, track in list(doc.karaoke.items()):
                if track.get("dormant"):
                    continue
                changed = []
                for bi, beat in enumerate(track.get("beats", [])):
                    core = kara._ids_of_anchor(beat.get("core", []))
                    if core and all(i in shifts for i in core):
                        deltas = {shifts[i] for i in core}
                        if len(deltas) == 1 and (delta := next(iter(deltas))):
                            changed.append((bi, delta))
                if not changed:
                    continue
                copied = tx.track(kid)
                for bi, delta in changed:
                    beat = copied["beats"][bi]
                    for edge in ("t0", "t1"):
                        if beat.get(edge) is not None:
                            beat[edge] = max(0, beat[edge] + delta)
                    # Like dragging a beat: a manual move must not be reverted by
                    # a future follow-links pass against an unchanged language.
                    beat.pop("link", None)
        return len(members)

    return editor.history.run("设置时间", apply)
