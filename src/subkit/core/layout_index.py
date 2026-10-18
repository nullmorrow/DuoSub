"""Where every subtitle line, character and ruby piece is drawn — in script coordinates.

Used by the player to find what the user clicked (观剧模式): the exported
preview document plus ``ExportReport.owners`` say which member / annotation each
event belongs to; base lines are measured with libass (the same probes and cache
as the ruby layout), ruby lines carry the position the ruby generator chose.

Static rotation, scale, shear and perspective use transformed ink quads; each
visual line has its own shear frame. move/animated lines use their static start
state. Mixed local transforms and automatic wrapping are not modelled;
libass collision shifting of overlapping unpositioned lines is not
emulated (bilingual styles normally stack by margins, so lines do not collide).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..ass.document import AssDocument, Event
from ..ass.text import VisibleText
from ..ruby.generate import RubyGenerator
from ..ruby.measure import LineRequest, Measurer
from .transform import (LineTransform, Point, Quad, line_quad, line_quads, measurement_copy,
                        source_transform_tags, transform_of)


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float, pad: float = 0.0) -> bool:
        return self.x0 - pad <= x <= self.x1 + pad and self.y0 - pad <= y <= self.y1 + pad

    def union(self, o: "Rect") -> "Rect":
        return Rect(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1))

    def moved(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def distance(self, x: float, y: float) -> float:
        dx = max(self.x0 - x, 0.0, x - self.x1)
        dy = max(self.y0 - y, 0.0, y - self.y1)
        return (dx * dx + dy * dy) ** 0.5

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def corners(self) -> Quad:
        return ((self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1))


def in_quad(quad: Quad, x: float, y: float, pad: float) -> bool:
    # Offset each edge by the existing square padding (identical for a rect).
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(quad, (*quad[1:], quad[0])))
    if abs(area) < 1e-9:
        for a, b in zip(quad, (*quad[1:], quad[0])):
            lo, hi = 0.0, 1.0
            for start, end, click in zip(a, b, (x, y)):
                delta = end - start
                if abs(delta) < 1e-12:
                    if abs(click - start) > pad:
                        hi = -1.0
                else:
                    p, q = (click - pad - start) / delta, (click + pad - start) / delta
                    lo, hi = max(lo, min(p, q)), min(hi, max(p, q))
            if lo <= hi:
                return True
        return False
    sign = 1 if area > 0 else -1
    for a, b in zip(quad, (*quad[1:], quad[0])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if sign * (dx * (y - a[1]) - dy * (x - a[0])) < -pad * (abs(dx) + abs(dy)) - 1e-9:
            return False
    return True


def unproject(quad: Quad, rect: Rect, x: float, y: float) -> Optional[Point]:
    """Invert rectangle -> quad homography, including libass perspective."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    dx, dy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    g = h = 0.0
    if abs(dx) + abs(dy) > 1e-9:
        a, b, c, d = x1 - x2, x3 - x2, y1 - y2, y3 - y2
        det = a * d - b * c
        if abs(det) < 1e-12:
            return None
        g, h = (dx * d - b * dy) / det, (a * dy - dx * c) / det
    a, b = x1 - x0 + g * x1 - x * g, x3 - x0 + h * x3 - x * h
    c, d = y1 - y0 + g * y1 - y * g, y3 - y0 + h * y3 - y * h
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    u, v = ((x - x0) * d - b * (y - y0)) / det, (a * (y - y0) - (x - x0) * c) / det
    return rect.x0 + u * (rect.x1 - rect.x0), rect.y0 + v * (rect.y1 - rect.y0)


@dataclass
class LineGeom:
    index: int                      # event index in the exported document
    member: str
    group: str
    kind: str                       # base | ruby | pinyin
    layer: int
    rect: Rect
    chars: list[Optional[Rect]] = field(default_factory=list)   # base: per visible character (None = no ink)
    ids: list = field(default_factory=list)                     # base: identity per visible character
    ann: Optional[str] = None                                   # ruby: annotation id
    text: str = ""
    quad: Optional[Quad] = None
    quads: tuple[Quad, ...] = ()     # individual outlines for multi-line shear
    _frames: list[tuple[Rect, Quad, list[int]]] = field(default_factory=list, repr=False)
    layout_origins: tuple[Point, ...] = ()  # measured origins matching _frames (precise path)
    layout_box: Optional[Rect] = None  # unrotated layout advances/height, not ink
    line_height: float = 0.0

    def __post_init__(self) -> None:
        if self.quad is None:
            self.quad = self.rect.corners
        if not self.quads:
            self.quads = (self.quad,)


@dataclass
class Hit:
    line: LineGeom
    char: Optional[int] = None      # visible index (base lines)

    @property
    def identity(self) -> Optional[tuple]:
        if self.char is None or self.char >= len(self.line.ids):
            return None
        return self.line.ids[self.char]


class LayoutIndex:
    def __init__(self, ass: AssDocument, owners: Sequence[dict[str, Any]], *, measure_cache: Optional[Path] = None,
                 video_height: Optional[int] = None, runner=None, renderer=None) -> None:
        self.ass = ass
        self.owners = list(owners)
        self.measurer = Measurer(ass, cache_path=measure_cache, runner=runner, renderer=renderer)
        self.gen = RubyGenerator(ass, self.measurer)
        self._geo: dict[int, LineGeom] = {}
        self.playres = ass.playres
        self.measure_cache = measure_cache
        layout_y = float(ass.info("LayoutResY", "0") or 0)
        self.perspective_distance = 312.5 * self.playres[1] / (layout_y if layout_y > 0 else video_height or self.playres[1])

    def close(self) -> None:
        self.measurer.close()

    # ------------------------------------------------------------ queries
    def active(self, t_ms: float) -> list[int]:
        t = t_ms / 10.0
        return [i for i, e in enumerate(self.ass.events)
                if e.kind == "Dialogue" and e.start <= t < e.end and i < len(self.owners)
                and self.owners[i].get("kind") != "gloss"]

    def events_of_group(self, gid: str) -> list[int]:
        return [i for i, o in enumerate(self.owners) if o.get("group") == gid and o.get("kind") != "gloss"]

    def geometry(self, indices: Sequence[int]) -> list[LineGeom]:
        pending = [i for i in dict.fromkeys(indices) if i not in self._geo]
        precise = []
        for i in pending:
            ev = self.ass.events[i]
            tf = transform_of(self.ass, ev)
            # Any transform value (scales and a style Angle included) takes the precise probe: the plain path
            # places scaled lines by the style's metrics (9 px off at \fscy200) and draws a style's Angle.
            if source_transform_tags(ev.text) or tf != LineTransform(animated=tf.animated):
                precise.append(i)
        need = [i for i in pending if i not in precise and self.owners[i].get("kind") == "base"]
        if need:
            reqs = []
            for i in need:
                ev = self.ass.events[i]
                an, x = self.gen._anchor(ev)
                reqs.append(LineRequest(ev.style, ev.text, an, x))
            for i, boxes in zip(need, self.measurer.line_char_boxes(reqs)):
                self._geo[i] = self._base_geom(i, boxes)
        if precise:
            self._transformed_geoms(precise)
        for i in indices:
            if i not in self._geo and self.owners[i].get("kind") in ("ruby", "pinyin"):
                o = self.owners[i]
                ev = self.ass.events[i]
                x, y, w, h = float(o["x"]), float(o["y"]), float(o["w"]), float(o["h"])
                self._geo[i] = LineGeom(i, o["member"], o["group"], o["kind"], ev.layer, Rect(x - w / 2, y - h, x + w / 2, y),
                                        ann=o.get("ann"), text=VisibleText(ev.text).text)
        return [self._geo[i] for i in indices if i in self._geo and self._geo[i].rect is not None]

    def precise_geometry(self, indices: Sequence[int]) -> list[LineGeom]:
        """Capture measured frames and origins even for untransformed lines."""
        if indices:
            self._transformed_geoms(list(dict.fromkeys(indices)))
        return [self._geo[i] for i in indices]

    def layout_boxes(self, indices: Sequence[int]) -> list[Rect]:
        """Measured unrotated block bounds for changing the alignment point.

        Hard breaks align individually; the widest advance defines the block.
        Like precise_geometry, automatic wrapping/local transforms are not modelled.
        """
        pending = [i for i in dict.fromkeys(indices)
                   if i not in self._geo or self._geo[i].layout_box is None]
        if pending:
            self._transformed_geoms(pending, measure_layout=True)
        return [self._geo[i].layout_box for i in indices]

    def _transformed_geoms(self, indices: Sequence[int], *, measure_layout: bool = False) -> None:
        # Neutralise style Angle as well as event rotation; scales stay baked in.
        doc, _ev = measurement_copy(self.ass, self.ass.events[indices[0]])
        events = [measurement_copy(doc, self.ass.events[i])[1] for i in indices]
        with Measurer(doc, cache_path=self.measure_cache, raster_scale=4, runner=self.measurer.runner, renderer=self.measurer.renderer) as m:
            gen = RubyGenerator(doc, m)
            reqs = []
            for ev in events:
                an, x = gen._anchor(ev)
                reqs.append(LineRequest(ev.style, ev.text, an, x))
            # Matching top-left probes reveal layout advances without assuming
            # that the ink bearing is zero. Each hard break aligns separately.
            left_reqs = [LineRequest(r.style, r.text, 7, r.x) for r in reqs]
            # Width probes span the canvas: probing right alignment at a left
            # margin clips glyphs and underestimates advances (and vice versa).
            width_left = [LineRequest(r.style, r.text, 7, 0) for r in reqs] if measure_layout else []
            width_right = [LineRequest(r.style, r.text, 9, self.playres[0]) for r in reqs] if measure_layout else []
            measured = m.line_char_boxes(reqs + left_reqs + width_left + width_right)
            for n, (i, ev) in enumerate(zip(indices, events)):
                vt = VisibleText(ev.text)
                top = gen.block_top(ev, vt.line_count)
                boxes, left_boxes = measured[n], measured[n + len(indices)]
                o = self.owners[i]
                g = self._base_geom(i, boxes, event=ev, gen=gen)
                g.kind, g.ann = o["kind"], o.get("ann")
                anchor = self.anchor(i)
                tf = transform_of(self.ass, self.ass.events[i])
                lh = gen._metrics[gen.metric_key(ev)][0]
                g.line_height = lh
                if measure_layout:
                    width_left_boxes = measured[n + 2 * len(indices)]
                    width_right_boxes = measured[n + 3 * len(indices)]
                    widths = [self.playres[0] + width_left_boxes[k].x0 - width_right_boxes[k].x0
                              for k in range(len(boxes))
                              if width_left_boxes[k] is not None and width_right_boxes[k] is not None]
                    width = max(widths, default=0.0)
                    left = anchor["x"] - ((anchor["an"] - 1) % 3) * width / 2
                    g.layout_box = Rect(left, top, left + width, top + vt.line_count * lh)
                frames, origins, char_indices = [], [], []
                for line in range(vt.line_count):
                    ks = [k for k, r in enumerate(g.chars) if r is not None and vt.line_of(k) == line]
                    if not ks:
                        continue
                    rect = g.chars[ks[0]]
                    for k in ks[1:]:
                        rect = rect.union(g.chars[k])
                    left = min(left_boxes[k].x0 for k in ks if left_boxes[k] is not None)
                    origins.append((anchor["x"] + rect.x0 - left, top + line * lh))
                    frames.append(rect)
                    char_indices.append(ks)
                if not frames:
                    frames, origins, char_indices = [g.rect], [(anchor["x"], top)], [[]]
                quads = line_quads(frames, (anchor["x"], anchor["y"]), tf, layout_origins=origins,
                                   measured_scale=(tf.fscx or 100, tf.fscy or 100),
                                   perspective_distance=self.perspective_distance)
                g.quads = tuple(quads)
                g._frames = list(zip(frames, quads, char_indices))
                g.layout_origins = tuple(origins)
                if len(quads) == 1:
                    g.quad = quads[0]
                elif not tf.fax and not tf.fay:
                    g.quad = line_quad(g.rect, (anchor["x"], anchor["y"]), tf,
                                       measured_scale=(tf.fscx or 100, tf.fscy or 100),
                                       perspective_distance=self.perspective_distance)
                else:
                    # Four corners cannot describe independent shear frames.
                    # quads are authoritative for hit-testing and outlines.
                    points = [p for q in quads for p in q]
                    g.quad = Rect(min(p[0] for p in points), min(p[1] for p in points),
                                  max(p[0] for p in points), max(p[1] for p in points)).corners
                self._geo[i] = g

    def _base_geom(self, i: int, boxes, *, event: Optional[Event] = None,
                   gen: Optional[RubyGenerator] = None) -> LineGeom:
        ev = event or self.ass.events[i]
        gen = gen or self.gen
        o = self.owners[i]
        vt = VisibleText(ev.text)
        top = gen.block_top(ev, vt.line_count)
        chars = [Rect(b.x0, top + b.y0, b.x1, top + b.y1) if b is not None else None for b in boxes]
        rect = None
        for r in chars:
            if r is not None:
                rect = r if rect is None else rect.union(r)
        if rect is None:   # nothing drawn (empty / spaces only): a thin box at the anchor so it can still be selected
            _an, x = gen._anchor(ev)
            lh = gen.line_height(ev.style)
            rect = Rect(x - lh / 2, top, x + lh / 2, top + lh)
        return LineGeom(i, o["member"], o["group"], "base", ev.layer, rect, chars, list(o.get("ids") or []),
                        text=vt.text)

    def anchor(self, i: int) -> dict[str, Any]:
        """Where event i is anchored now (script coordinates): {"an", "x", "y", "move"}.

        A line positioned by its style is converted to the equivalent \\pos, so dragging it starts exactly where
        it is drawn."""
        from ..ass.document import Style
        from ..ass.text import line_tag

        ev = self.ass.events[i]
        an, x = self.gen._anchor(ev)
        mv = line_tag(ev.text, "move")
        if mv is not None:
            a = mv.float_args() or []
            if len(a) >= 4:
                return {"an": an, "x": a[0], "y": a[1], "move": a}
        pos = line_tag(ev.text, "pos")
        if pos is not None:
            a = pos.float_args() or []
            if len(a) >= 2:
                return {"an": an, "x": a[0], "y": a[1], "move": None}
        st = self.ass.style(ev.style) or Style.new(ev.style)
        mv_ = ev.margin_v or int(st.num("MarginV"))
        v = (an - 1) // 3
        py = self.playres[1]
        y = py - mv_ if v == 0 else py / 2.0 if v == 1 else float(mv_)
        return {"an": an, "x": x, "y": float(y), "move": None}

    # ------------------------------------------------------------ hit testing
    @staticmethod
    def hit(geoms: Sequence[LineGeom], x: float, y: float, *, pad: float = 6.0) -> Optional[Hit]:
        """Top-most line under (x, y) (libass draws by layer, then file order), and the nearest character."""
        if not all(math.isfinite(v) for v in (x, y, pad)):
            return None
        pad = max(0.0, pad)
        for g in sorted(geoms, key=lambda g: (g.layer, g.index), reverse=True):
            plain = not g._frames and g.quads == (g.rect.corners,)
            inside = g.rect.contains(x, y, pad) if plain else any(in_quad(q, x, y, pad) for q in g.quads)
            if not inside:
                continue
            if g.kind != "base":
                return Hit(g)
            best, best_d = None, 1e18
            frames = g._frames or [(g.rect, g.quad, list(range(len(g.chars))))]
            for rect, quad, ks in frames:
                if not plain and not in_quad(quad, x, y, pad):
                    continue
                local = (x, y) if plain else unproject(quad, rect, x, y)
                if local is None:
                    continue
                for k in ks:
                    r = g.chars[k]
                    if r is None:
                        continue
                    d = r.distance(*local)
                    if d < best_d:
                        best, best_d = k, d
            return Hit(g, best)
        return None
