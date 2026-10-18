"""Per-song alternating lyric slots; document times and beats stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..ass.document import Event, Style
from ..ass.text import VisibleText
from ..core.songs import song_span
from .base import Parameter


PARAMETERS = {
    "ktv": Parameter("choice", "off", "排列：左上 / 右下 · 上下两行居中",
                     choices=("off", "corners", "stacked")),
    "line_advance_ms": Parameter("int", 600, "换行提前（毫秒）", 0, 10000),
}


def effect_params(params):
    return {k: v for k, v in params.items() if k not in PARAMETERS}


@dataclass(frozen=True)
class Slot:
    original: str
    index: int
    mode: str
    start: int
    end: int
    an: int = 2
    x: float = 0
    y: float = 0


def song_slots(doc, views):
    """The first explicit track layout in a song span applies to the song."""
    songs = []
    visited = set()
    groups = {g.id: g for g in doc.groups}
    for group in doc.groups:
        if group.kind != "song" or "hidden" in group.flags or group.id in visited:
            continue
        span = song_span(doc, group.id)
        visited.update(span)
        if span:
            songs.append([[doc.members[mid] for mid in groups[gid].members
                           if doc.members[mid].role == "song" and doc.members[mid].source_kind != "Comment"]
                          for gid in span if "hidden" not in groups[gid].flags])
    slots = {}
    for song in songs:
        settings = next((views[m.id].params for members in song for m in members
                         if m.id in views and "ktv" in views[m.id].params), {})
        mode = PARAMETERS["ktv"].validate(settings.get("ktv", "off"))
        advance = PARAMETERS["line_advance_ms"].validate(settings.get("line_advance_ms", 600))
        if mode == "off":
            continue
        occupied = [0, 0]
        index = 0
        for members in song:
            originals = [m for m in members if m.lane == "original"]
            translations = [m for m in members if m.lane == "translation"]
            # A translation-only song line still occupies a slot.
            for i, original in enumerate(originals or translations):
                slot = index % 2
                start = max(0, original.start - advance, occupied[slot])
                assignment = Slot(original.id, slot, mode, start, original.end)
                slots[original.id] = assignment
                if originals:
                    following = translations if len(originals) == 1 else translations[i:i+1]
                    for m in following:
                        slots[m.id] = assignment
                occupied[slot] = max(occupied[slot], original.end)
                index += 1
    return slots


def position_slots(doc, slots, out, styles, generator, ruby_gap):
    """Keep the template's relative lyric/translation rows within each slot band.

    Bottom slots retain the song margins. Top slots translate that same measured
    band to the top song margin, including room for the original's furigana.
    """
    bands = {}
    for mid, slot in slots.items():
        bands.setdefault(slot.original, []).append(mid)
    positioned = {}
    for original, mids in bands.items():
        metrics = {}
        tops, bottoms, margins = [], [], []
        for mid in mids:
            m = doc.members[mid]
            name = styles[mid]
            st = out.style(name) or Style.new(name)
            ev = Event(style=name, text=m.text)
            lines = VisibleText(m.text).line_count
            top = generator.block_top(ev, lines)
            height = generator.line_height(name) * lines
            metrics[mid] = (top + height, st)
            tops.append(top)
            bottoms.append(top + height)
            margins.append(st.num("MarginV"))
            if any(a.member == mid and a.kind == "ruby" and a.status != "hidden" and a.text
                   for a in doc.annotations.values()):
                ruby = name + ".Ruby"
                if out.style(ruby) is None:
                    from ..ruby.generate import derive_ruby_style
                    out.styles.append(derive_ruby_style(st, ruby))
                tops.append(top - generator.line_height(ruby) - ruby_gap)
        slot = slots[original]
        shift = (min(margins) - min(tops) if slot.index == 0 else
                 out.playres[1] - min(margins) - max(bottoms))
        for mid, (y, st) in metrics.items():
            an = 2 if slot.mode == "stacked" else 1 if slot.index == 0 else 3
            x = out.playres[0]/2 if an == 2 else st.num("MarginL") if an == 1 else out.playres[0]-st.num("MarginR")
            positioned[mid] = replace(slots[mid], an=an, x=x, y=y+shift)
    return positioned
