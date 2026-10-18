"""Generate ruby Dialogue lines for many base lines at once (one measurement batch).

Vertical placement is computed from measured style metrics (line height and
the ink top of a tall reference glyph), so every line of a style gets its ruby
at exactly the same height, independent of which glyphs the line contains.

Ruby follows what is done to its base characters: size (\\fs) and scaling (\\fscx \\fscy) are measured; rotation
(\\frz \\frx \\fry) and shear (\\fax \\fay) are copied with the base line's rotation origin (\\org, or its anchor);
colours / alpha / blur, \\t animations of those, \\move, \\fad and \\clip are copied.  Positions are computed in
the unrotated layout, exactly like libass does for the base line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

from ..ass.document import AssDocument, Event, Style
from ..ass.state import LAYOUT_ANIMATED, PAINT, TRANSFORM, filter_t, split_t, state_at
from ..ass.tags import canonical
from ..ass.text import VisibleText, leading_tags, line_tag, remove_tags_named, tokenize
from ..textutil import is_digit_char, is_kanji, is_space
from .layout import Item, layout_line
from .measure import Box, LineRequest, Measurer

LINE_COPY = ("fad", "fade", "clip", "iclip")          # line-level tags the ruby lines take as they are
FOLLOW_IN_T = set(TRANSFORM) | set(PAINT) | {"clip", "iclip"}
REF_GLYPH = "国"
REF_RUBY = "あ"


def _num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


@dataclass
class RubyPiece:
    start: int          # visible index range of the base
    end: int
    reading: str
    scale: Optional[float] = None   # fixed \fscx from the record
    tag: object = None
    k_text: Optional[str] = None    # karaoke text written instead of the reading (measurement uses the reading)
    word: object = None            # same word across parts; core annotations use tag when omitted


@dataclass
class RubyJob:
    event: Event
    ruby_style: str
    pieces: list[RubyPiece]
    effect: str = ""                # Effect field for generated lines
    level: int = 0                  # 1 = stacked above another annotation band (gloss above pinyin)
    below_style: str = ""           # style of the band below (its line height decides the lift)
    out: list[Event] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    placements: list[tuple[RubyPiece, float, float, bool]] = field(default_factory=list)  # piece, x, scale, overlap
    offset: float = 0.0             # signed px at 1080p, applied in the base line's unrotated coordinates
    max_width_percent: Optional[float] = None
    font_context: Optional[Event] = None  # vertical line/ruby font before display-only substitutions
    prefer_center: bool = False    # horizontal furigana: only neighbouring ruby constrains centring
    base_advances: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)  # measured cap basis; centres keep ink boxes
    geoms: list[dict] = field(default_factory=list)   # per generated line: ann (piece.tag), x, y (n2 anchor), w, h


def derive_ruby_style(base: Style, name: str, ratio: float = 0.5) -> Style:
    st = base.copy(name)
    st.set("Fontsize", round(base.fontsize * ratio, 1))
    st.set("Outline", round(max(0.0, base.outline * 0.6), 2))
    st.set("Shadow", round(base.shadow * ratio, 2))
    st.set("Spacing", 0)
    st.set("Alignment", 2)
    for k in ("MarginL", "MarginR", "MarginV"):
        st.set(k, 0)
    return st


class RubyGenerator:
    def __init__(self, doc: AssDocument, measurer: Measurer, *, gap: float = 2.0, min_scale: float = 60.0,
                 overhang: float = 0.5) -> None:
        self.doc = doc
        self.m = measurer
        self.gap = gap
        self.min_scale = min_scale
        self.overhang = overhang
        self._metrics: dict[str, tuple[float, float, float]] = {}

    # ----------------------------------------------------------- metrics
    @staticmethod
    def _parse_key(key: str) -> tuple[str, str, str]:
        """``"Style@60^150"`` → ("Style", "60", "150"): the style at font size 60 and \\fscy150 (either optional)."""
        rest, _, sy = key.partition("^")
        s, _, fs = rest.partition("@")
        return s, fs, sy

    def _style_metrics(self, styles: list[str]) -> None:
        """line height, reference ink top and ink bottom per style (relative to the box top).

        A key ``"Style@60"`` measures the style at another font size (a line or part of it set to \\fs60),
        ``"Style^150"`` (or ``"Style@60^150"``) with a vertical scale \\fscy150."""
        need = [s for s in styles if s not in self._metrics]
        if not need:
            return
        reqs = []
        for key in need:
            s, fs_txt, sy_txt = self._parse_key(key)
            if s.endswith(".Pinyin"):
                glyph = "g"   # latin: the descender must clear the base line
            elif s.endswith(".Ruby") or s.endswith(".Gloss"):
                glyph = REF_RUBY
            else:
                glyph = REF_GLYPH
            pre = (f"\\fs{fs_txt}" if fs_txt else "") + (f"\\fscy{sy_txt}" if sy_txt else "")
            reqs.append(LineRequest(s, (f"{{{pre}}}" if pre else "") + f"{glyph}\\N{glyph}", 8, self.m.playres_x / 2))
        res = self.m.line_char_boxes(reqs)
        for key, boxes in zip(need, res):
            s, fs_txt, sy_txt = self._parse_key(key)
            b0, b1 = boxes[0], boxes[1] if len(boxes) > 1 else None
            st = self.doc.style(s)
            fs = float(fs_txt) if fs_txt else (st.fontsize if st else 48)
            fs *= (float(sy_txt) / 100.0) if sy_txt else ((st.scale_y if st else 100.0) / 100.0)
            if b0 and b1:
                self._metrics[key] = (b1.y0 - b0.y0, b0.y0, b0.y1)
            else:
                self._metrics[key] = (fs * 1.2, fs * 0.15, fs * 1.0)

    def char_scales(self, ev: Event) -> list[tuple[float, float, float]]:
        """(font size, \\fscx, \\fscy) in effect at every visible character (tags in the text; the style's values
        otherwise)."""
        st = self.doc.style(ev.style)
        base = (st.fontsize if st else 48.0, st.scale_x if st else 100.0, st.scale_y if st else 100.0)
        cur, out = list(base), []
        for tok in tokenize(ev.text):
            if tok.kind == "tags":
                for t in tok.tags:
                    if t.name in ("fs", "fscx", "fscy"):
                        k = ("fs", "fscx", "fscy").index(t.name)
                        v = t.num()
                        cur[k] = base[k] if v is None or (k == 0 and v <= 0) else v      # empty = the style's
                    elif t.name == "r":
                        cur = list(base)
            elif tok.kind == "text":
                out.extend([tuple(cur)] * len(tok.raw))
            elif tok.kind in ("hardspace", "softbreak"):
                out.append(tuple(cur))
        return out

    def char_sizes(self, ev: Event) -> list[float]:
        """Font size in effect at every visible character (\\fs in the text; the style's size otherwise)."""
        return [s[0] for s in self.char_scales(ev)]

    def metric_key(self, ev: Event) -> str:
        """Metrics of the line: the style, or the style at the tallest size (font size × \\fscy) used in the line."""
        st = self.doc.style(ev.style)
        base_fs = st.fontsize if st else 48.0
        base_sy = st.scale_y if st else 100.0
        scales = self.char_scales(ev)
        if not scales:
            return ev.style
        fs, _sx, sy = max(scales, key=lambda s: s[0] * s[2])
        key = ev.style
        if abs(fs - base_fs) >= 0.01:
            key += f"@{fs:g}"
        if abs(sy - base_sy) >= 0.01:
            key += f"^{sy:g}"
        return key

    def vertical(self, ev: Event) -> tuple[float, str] | None:
        """Quarter-turn lines using ASS vertical glyphs, at the first character."""
        st = self.doc.style(ev.style) or Style.new(ev.style)
        font, angle = st.get("Fontname"), st.num("Angle")
        for tag in leading_tags(ev.text):
            name = canonical(tag.name)
            if name == "r":
                reset = self.doc.style(tag.arg) or st
                font, angle = reset.get("Fontname"), reset.num("Angle")
            elif name == "fn":
                font = tag.arg or st.get("Fontname")
            elif name == "frz":
                angle = tag.num() if tag.num() is not None else st.num("Angle")
        turn = angle % 360
        return (angle, font) if font.startswith("@") and any(abs(turn-v) < 1e-6 for v in (90, 270)) else None

    def vertical_style(self, ev: Event, name: str) -> str:
        vertical = self.vertical(ev)
        if vertical is None:
            return name
        import hashlib
        font = vertical[1]
        target = name + ".Vertical" + hashlib.sha1(font.encode("utf-8")).hexdigest()[:8] + ".Ruby"
        if self.doc.style(target) is None:
            st = (self.doc.style(name) or derive_ruby_style(self.doc.style(ev.style) or Style.new(ev.style), name)).copy(target)
            st.set("Fontname", font)
            st.set("Angle", 0)
            self.doc.styles.append(st)
        return target

    # ----------------------------------------------------------- main
    def run(self, jobs: list[RubyJob]) -> None:
        horizontal = [j for j in jobs if self.vertical(j.font_context or j.event) is None]
        vertical = [j for j in jobs if self.vertical(j.font_context or j.event) is not None]
        self._run(horizontal)
        if vertical:
            from ..core.transform import measurement_copy
            original_styles = {id(j): j.ruby_style for j in vertical}
            for j in vertical:
                j.ruby_style = self.vertical_style(j.font_context or j.event, j.ruby_style)
            measured_doc, _ev = measurement_copy(self.doc, vertical[0].event)
            with Measurer(measured_doc, ffmpeg=self.m.ffmpeg, raster_scale=self.m.raster_scale,
                          runner=self.m.runner, renderer=self.m.renderer) as m:
                gen = RubyGenerator(self.doc, m, gap=self.gap, min_scale=self.min_scale, overhang=self.overhang)
                gen._run(vertical)
                self.m.font_log.update(m.font_log)
            for j in vertical:
                j.ruby_style = original_styles[id(j)]
                for event in j.out:
                    event.style = j.ruby_style

    def _run(self, jobs: list[RubyJob]) -> None:
        jobs = [j for j in jobs if j.pieces]
        if not jobs:
            return
        line_reqs = []
        for j in jobs:
            an, x = self._anchor(j.event)
            line_reqs.append(LineRequest(j.event.style, j.event.text, an,
                                         self.m.playres_x / 2 if self.vertical(j.font_context or j.event) else x))
        all_boxes = self.m.line_char_boxes(line_reqs)
        for i, (j, req, boxes) in enumerate(zip(jobs, line_reqs, all_boxes)):
            if self.vertical(j.font_context or j.event):
                delta = self._anchor(j.event)[1] - req.x
                all_boxes[i] = [Box(b.x0+delta, b.x1+delta, b.y0, b.y1) if b is not None else None
                                for b in boxes]
        capped = [(j, p) for j in jobs if j.max_width_percent is not None
                  for p in j.pieces if p.scale is None]
        advances = self.m.span_advances([(j.event.style, j.event.text, p.start, p.end) for j, p in capped]) if capped else []
        for (j, p), width in zip(capped, advances):
            if width is not None:
                j.base_advances[p.start, p.end] = width
            else:
                j.warnings.append(f"无法测量汉字占位宽度，本段沿用笔画宽度限宽：{VisibleText(j.event.text).text[p.start:p.end]}")
        texts = [(j.ruby_style, p.reading) for j in jobs for p in j.pieces]
        ruby_boxes = self.m.text_boxes(texts)
        self._style_metrics(sorted({j.event.style for j in jobs} | {self.metric_key(j.event) for j in jobs}
                                   | {j.ruby_style for j in jobs} | {j.below_style for j in jobs if j.below_style}))
        k = 0
        for j, boxes in zip(jobs, all_boxes):
            rb = ruby_boxes[k : k + len(j.pieces)]
            k += len(j.pieces)
            self._layout_job(j, boxes, rb)

    def _anchor(self, ev: Event) -> tuple[int, float]:
        st = self.doc.style(ev.style) or Style.new(ev.style)
        an_tag = line_tag(ev.text, "an")
        an = int(an_tag.num() or st.alignment) if an_tag else st.alignment
        pos = line_tag(ev.text, "pos") or line_tag(ev.text, "move")
        px = None
        if pos:
            a = pos.float_args()
            if a:
                px = a[0]
        return an, self.m.anchor_x(ev.style, an, ev.margin_l, ev.margin_r, px)

    def block_top(self, ev: Event, n_lines: int) -> float:
        """Top of the event's text block (the probes of :class:`Measurer` are relative to it)."""
        key = self.metric_key(ev)
        self._style_metrics([key])
        st = self.doc.style(ev.style) or Style.new(ev.style)
        lh = self._metrics[key][0]
        an, _x = self._anchor(ev)
        pos = line_tag(ev.text, "pos") or line_tag(ev.text, "move")
        py = None
        if pos:
            a = pos.float_args()
            if a and len(a) >= 2:
                py = a[1]
        v = (an - 1) // 3  # 0 bottom, 1 middle, 2 top
        mv = ev.margin_v or int(st.num("MarginV"))
        total = n_lines * lh
        if v == 0:
            bottom = py if py is not None else self.m.playres_y - mv
            box_top = bottom - total
        elif v == 1:
            center = py if py is not None else self.m.playres_y / 2.0
            box_top = center - total / 2.0
        else:
            box_top = py if py is not None else mv
        return box_top

    def line_height(self, style: str) -> float:
        self._style_metrics([style])
        return self._metrics[style][0]

    def _line_tops(self, ev: Event, n_lines: int) -> list[float]:
        """Ink top of the reference glyph for each visual line of the event."""
        key = self.metric_key(ev)
        self._style_metrics([key])
        lh, ref_top, _ = self._metrics[key]
        box_top = self.block_top(ev, n_lines)
        return [box_top + i * lh + ref_top for i in range(n_lines)]

    def _merge_word_parts(self, pairs: list[tuple[RubyPiece, Item]], j: RubyJob,
                          base_widths: dict[int, float], outline: float) -> tuple[list[tuple[RubyPiece, Item]], dict[int, list[Item]]]:
        """Lay out one contiguous annotated word before applying interword caps.

        Internal reading parts are output anchors, not separate word boundaries.
        Kana between annotated blocks and visual line breaks keep their positions.
        A shared manual scale wins; conflicting manual scales keep the original path.
        Internal gaps stay fixed when the composite word is compressed.
        """
        ordered = sorted(pairs, key=lambda pair: pair[1].target)
        result, merged = [], {}
        index = 0
        while index < len(ordered):
            end = index + 1
            word = ordered[index][1].word
            while (end < len(ordered) and word is not None and ordered[end][1].word == word
                   and ordered[end - 1][0].end == ordered[end][0].start):
                end += 1
            group = ordered[index:end]
            manual = {it.manual_scale for _p, it in group if it.manual_scale is not None}
            if len(group) > 1 and len(manual) <= 1:
                pieces = [p for p, _it in group]
                parts = [it for _p, it in group]
                gap_width = self.gap * (len(parts) - 1)
                key = RubyPiece(min(p.start for p in pieces), max(p.end for p in pieces),
                                "".join(p.reading for p in pieces), word=word)
                item = Item(min(it.base_x0 for it in parts), max(it.base_x1 for it in parts),
                            sum(it.width for it in parts) + gap_width,
                            manual_scale=next(iter(manual), None), key=key, word=word, gap_width=gap_width)
                if j.max_width_percent is not None and item.manual_scale is None:
                    ink = sum(it.width - 2 * outline for it in parts) + gap_width
                    if ink > 0:
                        item.max_scale = sum(base_widths[id(p)] for p in pieces) * j.max_width_percent / ink
                result.append((key, item))
                merged[id(item)] = parts
            else:
                result.extend(group)
            index = end
        return result, merged

    def _word_caps(self, pairs: list[tuple[RubyPiece, Item]], word_gap: float) -> set[int]:
        """Fit touching words at their own centres before allowing collision drift.

        Each side owns the space up to its kanji boundary. A short reading
        lends its unused share to its neighbour; automatic scales and the
        minimum remain independent. The ordinary solver handles any space
        still missing at the floor or with a fixed manual scale.
        """
        ordered = sorted(pairs, key=lambda pair: pair[1].target)
        widths = {id(it): it.w(it.manual_scale if it.manual_scale is not None
                  else min(100.0, max(self.min_scale, it.max_scale))) for _p, it in ordered}
        limited: set[int] = set()
        for (lp, left), (rp, right) in zip(ordered, ordered[1:]):
            if (lp.end != rp.start or left.word is None or right.word is None
                    or left.word == right.word):
                continue
            lw, rw = widths[id(left)], widths[id(right)]
            if lw + rw + 2 * word_gap <= 2 * (right.target - left.target) + 1e-6:
                continue
            boundary = (left.base_x1 + right.base_x0) / 2.0
            left_share = max(0.0, 2 * (boundary - left.target) - word_gap)
            right_share = max(0.0, 2 * (right.target - boundary) - word_gap)
            budgets = (left_share + max(0.0, right_share - rw),
                       right_share + max(0.0, left_share - lw))
            for item, budget in zip((left, right), budgets):
                ceiling = 100.0 * (budget - item.gap_width) / max(1e-6, item.width - item.gap_width)
                if item.manual_scale is None and ceiling < min(100.0, item.max_scale):
                    item.max_scale = ceiling
                    limited.add(id(item))
        return limited

    def _layout_job(self, j: RubyJob, boxes, ruby_boxes) -> None:
        ev = j.event
        vis = VisibleText(ev.text)
        text = vis.text
        rst = self.doc.style(j.ruby_style)
        vertical = self.vertical(j.font_context or ev)
        ruby_ink = {id(p): rb for p, rb in zip(j.pieces, ruby_boxes) if rb is not None}
        vertical_bounds = []
        rfs = rst.fontsize if rst else 24.0
        r_out = rst.outline if rst else 1.0
        covered: dict[int, int] = {}
        for pi, p in enumerate(j.pieces):
            for c in range(p.start, p.end):
                covered[c] = pi
        by_line: dict[int, list[tuple[RubyPiece, Item]]] = {}
        st_base = self.doc.style(ev.style)
        base_fs = st_base.fontsize if st_base else 48.0
        base_sx = st_base.scale_x if st_base else 100.0
        base_sy = st_base.scale_y if st_base else 100.0
        scales = self.char_scales(ev)
        line_v = max((s[0] * s[2] for s in scales), default=base_fs * base_sy)
        # ruby follows the size and scaling of its kanji (a word set larger, a line stretched …)
        ratio: dict[int, tuple[float, float, float]] = {}       # size ratio, x scale ratio, y scale ratio
        ink_top: dict[int, float] = {}      # pieces in another height than the line's tallest: their own glyph top
        base_widths: dict[int, float] = {}
        width_caps: dict[int, float] = {}
        for p, rb in zip(j.pieces, ruby_boxes):
            bs = [boxes[c] for c in range(p.start, min(p.end, len(boxes))) if c < len(boxes) and boxes[c] is not None]
            if not bs or rb is None:
                j.warnings.append(f"无法测量：{text[p.start:p.end]}")
                continue
            x0 = min(b.x0 for b in bs)
            x1 = max(b.x1 for b in bs)
            fs_p, sx_p, sy_p = scales[p.start] if p.start < len(scales) else (base_fs, base_sx, base_sy)
            rr = (fs_p / base_fs if base_fs else 1.0, sx_p / base_sx if base_sx else 1.0, sy_p / base_sy if base_sy else 1.0)
            ratio[id(p)] = rr
            if abs(fs_p * sy_p - line_v) >= 0.01:
                ink_top[id(p)] = min(b.y0 for b in bs)
            item = Item(x0, x1, rb.width * rr[0] * rr[1] + 2 * r_out, manual_scale=p.scale, key=p)
            base_widths[id(p)] = j.base_advances.get((p.start, p.end), x1 - x0)
            if j.max_width_percent is not None and p.scale is None:
                ink_width = rb.width * rr[0] * rr[1]
                if ink_width > 0:
                    item.max_scale = base_widths[id(p)] * j.max_width_percent / ink_width
            width_caps[id(p)] = item.max_scale
            if j.prefer_center and not vertical:
                item.word = p.word if p.word is not None else p.tag
            # Furigana already has its own width ceiling and annotation
            # collision constraints. Bare base neighbours must not push it
            # off-centre; vertical text and other annotation bands retain
            # their existing overhang limits.
            if not j.prefer_center or vertical:
                item.left_limit = self._limit(text, boxes, covered, p, -1, x0, rfs)
                item.right_limit = self._limit(text, boxes, covered, p, +1, x1, rfs)
            by_line.setdefault(vis.line_of(p.start), []).append((p, item))
        tops = self._line_tops(ev, vis.line_count)
        box_top = self.block_top(ev, vis.line_count)
        lh_r, _rt, r_ink_bottom = self._metrics[j.ruby_style]
        r_sx = rst.scale_x if rst else 100.0
        r_sy = rst.scale_y if rst else 100.0
        line_tags = "".join(t.raw for t in (line_tag(ev.text, n) for n in LINE_COPY) if t is not None)
        mv = line_tag(ev.text, "move")
        mv_args = (mv.float_args() or []) if mv is not None else []
        org_tag = line_tag(ev.text, "org")
        fay_line = any(canonical(t.name) == "fay" for tok in tokenize(ev.text) if tok.kind == "tags" for t in tok.tags)
        for line_no, pairs in sorted(by_line.items()):
            word_gap = max(self.gap, 0.3 * rfs)
            merged = {}
            word_limited = set()
            if j.prefer_center and not vertical:
                pairs, merged = self._merge_word_parts(pairs, j, base_widths, r_out)
                word_limited = self._word_caps(pairs, word_gap)
            items = [it for _p, it in pairs]
            layout_line(items, gap=self.gap, min_scale=self.min_scale, word_gap=word_gap)
            restored = []
            for item in items:
                parts = merged.get(id(item))
                if parts is None:
                    restored.append(item)
                    continue
                left = item.x - item.w() / 2.0
                for part in parts:
                    part.scale, part.overlap = item.scale, item.overlap
                    part.manual_scale, part.max_scale = item.manual_scale, item.max_scale
                    part.x = left + part.w() / 2.0
                    left += part.w() + self.gap
                    if id(item) in word_limited:
                        word_limited.add(id(part))
                    width_caps[id(part.key)] = item.max_scale
                    restored.append(part)
            items = restored
            base_top = tops[min(line_no, len(tops) - 1)]
            target_bottom = base_top - self.gap
            if j.level and j.below_style in self._metrics:
                lh_b, top_b, bottom_b = self._metrics[j.below_style]
                target_bottom -= (bottom_b - top_b) + self.gap  # lift above the lower band's ink
            for it in items:
                p: RubyPiece = it.key
                r, rx, ry = ratio.get(id(p), (1.0, 1.0, 1.0))
                tb = (box_top + ink_top[id(p)] - self.gap) if id(p) in ink_top else target_bottom
                y = tb + (lh_r - r_ink_bottom) * r * ry
                x = it.x
                if vertical:
                    span = [b for b in boxes[p.start:p.end] if b is not None]
                    rb = ruby_ink[id(p)]
                    if vertical[0] % 360 == 90:
                        y = box_top + max(b.y1 for b in span) + self.gap + (st_base.outline if st_base else 0) + r_out + (lh_r - rb.y0) * r * ry
                    else:
                        y = box_top + min(b.y0 for b in span) - self.gap - (st_base.outline if st_base else 0) - r_out + (lh_r - rb.y1) * r * ry
                    x -= (rb.cx - self.m.playres_x / 2) * r * rx * it.scale / 100
                if j.offset:
                    y += (1 if vertical and vertical[0] % 360 == 90 else -1) * j.offset * self.m.playres_y / 1080.0
                w = max(1.0, (it.width - 2 * r_out) * it.scale / 100.0)
                follow, needs_org, dropped = self._follow(ev.text, p.start)
                if fay_line:
                    y += self._fay_shift(ev.text, boxes, vis, p.start, it.x - w / 2)
                if dropped:
                    j.warnings.append(f"动画里改了字号/缩放（\\t 里的 {'、'.join(sorted(dropped))}）：振假名不跟着变：{text[p.start:p.end]}")
                if vertical:
                    follow += f"\\fn{vertical[1]}\\frz{_num(vertical[0])}"
                    needs_org = True
                tags = ["\\an2"]
                if len(mv_args) >= 4:
                    dx, dy = mv_args[2] - mv_args[0], mv_args[3] - mv_args[1]
                    timing = "".join("," + _num(v) for v in mv_args[4:6])
                    tags.append(f"\\move({x:.1f},{y:.1f},{x + dx:.1f},{y + dy:.1f}{timing})")
                else:
                    tags.append(f"\\pos({x:.1f},{y:.1f})")
                if needs_org:
                    if org_tag is not None:
                        tags.append(org_tag.raw)
                    else:
                        ox, oy = self.origin(ev, vis.line_count)
                        tags.append(f"\\org({ox:.1f},{oy:.1f})")
                        if len(mv_args) >= 4 and (mv_args[2], mv_args[3]) != (mv_args[0], mv_args[1]):
                            j.warnings.append(f"移动中旋转的行没有 \\org：振假名只在开始时对齐：{text[p.start:p.end]}")
                fscx = r_sx * rx * it.scale / 100.0
                capped = it.manual_scale is None and it.max_scale < 100.0
                if capped and it.scale > self.min_scale + 1e-6:
                    fscx = math.floor(fscx * 100.0 + 1e-8) / 100.0
                if capped:
                    tags.append(f"\\fscx{_num(fscx)}")
                elif abs(fscx - r_sx) >= 0.5:
                    tags.append(f"\\fscx{fscx:.0f}")
                if abs(ry - 1.0) >= 0.005:
                    tags.append(f"\\fscy{r_sy * ry:.0f}")
                if abs(r - 1.0) >= 0.01:
                    tags.append(f"\\fs{rfs * r:.1f}")
                txt = "{" + "".join(tags) + follow + line_tags + "}" + (p.k_text or p.reading)
                j.out.append(ev.copy(kind="Dialogue", style=j.ruby_style, effect=j.effect, text=txt, name=""))
                j.placements.append((p, it.x, it.scale, it.overlap))
                j.geoms.append({"ann": p.tag, "x": round(x, 2), "y": round(y, 2), "w": round(w, 2),
                                "h": round(lh_r * r * ry, 2), "base_start": p.start, "base_end": p.end})
                if vertical:
                    rb = ruby_ink[id(p)]
                    vertical_bounds.append(Box(x - w / 2 - r_out, x + w / 2 + r_out,
                                               y - lh_r * r * ry + rb.y0 * r * ry - r_out,
                                               y - lh_r * r * ry + rb.y1 * r * ry + r_out))
                if it.overlap:
                    j.warnings.append(f"振假名放不下（压缩到下限后伸到了汉字上方）：{text[p.start:p.end]}（{p.reading}）")
                if p.scale is None and width_caps[id(p)] < self.min_scale - 1e-6:
                    actual = (it.width - 2 * r_out) * it.scale / max(1e-6, base_widths[id(p)])
                    j.warnings.append(f"振假名太长，压到下限仍是汉字占位宽度的 {actual:.0f}%：{text[p.start:p.end]}（{p.reading}）")
                if id(it) in word_limited and it.max_scale < self.min_scale - 1e-6:
                    j.warnings.append(f"相邻词注音过密，已保留缩放下限并退避：{text[p.start:p.end]}（{p.reading}）")

        if vertical and vertical_bounds:
            padding = (st_base.outline + abs(st_base.shadow)) if st_base else 0
            base_bounds = [Box(b.x0-padding, b.x1+padding, box_top+b.y0-padding, box_top+b.y1+padding)
                           for b in boxes if b is not None]
            self._keep_vertical_inside(j, base_bounds + vertical_bounds, vertical[0], vis.line_count)

    def _keep_vertical_inside(self, j: RubyJob, bounds: list[Box], angle: float, n_lines: int) -> None:
        """Translate the exported line and its shared origin by the screen overflow."""
        ev = j.event
        ox, oy = self.origin(ev, n_lines)
        org = line_tag(ev.text, "org")
        if org is not None and len(org.float_args() or []) == 2:
            ox, oy = org.float_args()
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        points = [(ox + c*(x-ox) + s*(y-oy), oy - s*(x-ox) + c*(y-oy))
                  for b in bounds for x in (b.x0, b.x1) for y in (b.y0, b.y1)]
        shifts = []
        for axis, size in enumerate((self.m.playres_x, self.m.playres_y)):
            lo, hi = min(p[axis] for p in points), max(p[axis] for p in points)
            delta = -lo + self.gap if lo < 0 else size - hi - self.gap if hi > size else 0
            shifts.append(math.copysign(math.ceil(abs(delta)*10)/10, delta) if delta else 0)
            if hi - lo > size:
                j.warnings.append("竖排振假名整行超过画面尺寸，无法完全内移")
        dx, dy = shifts
        if not (dx or dy):
            return
        ax, ay = self.origin(ev, n_lines)
        def moved(event, x, y):
            move = line_tag(event.text, "move")
            args = move.float_args() if move is not None else []
            if len(args) >= 4:
                args[:4] = [args[0]+dx, args[1]+dy, args[2]+dx, args[3]+dy]
                pos = "\\move(" + ",".join(_num(v) for v in args) + ")"
            else:
                pos = f"\\pos({_num(x+dx)},{_num(y+dy)})"
            prefix = pos + f"\\org({_num(ox+dx)},{_num(oy+dy)})"
            return event.copy(text="{" + prefix + "}" + remove_tags_named(event.text, {"pos", "move", "org"}))
        j.event.text = moved(ev, ax, ay).text
        for i, event in enumerate(j.out):
            args = line_tag(event.text, "pos") or line_tag(event.text, "move")
            x, y = args.float_args()[:2]
            j.out[i] = moved(event, x, y)
            j.geoms[i]["x"] = round(j.geoms[i]["x"] + dx, 2)
            j.geoms[i]["y"] = round(j.geoms[i]["y"] + dy, 2)
        j.warnings.append(f"竖排振假名靠近画面边缘，整行内移 {_num(math.hypot(dx, dy))} px")

    # ----------------------------------------------------------- following the base line
    def origin(self, ev: Event, n_lines: int) -> tuple[float, float]:
        """Where libass rotates the base line around when it has no \\org: its anchor point (the \\pos / \\move
        start, else the alignment point of the text block)."""
        an, x = self._anchor(ev)
        top = self.block_top(ev, n_lines)
        lh = self._metrics[self.metric_key(ev)][0]
        v = (an - 1) // 3          # 0 bottom, 1 middle, 2 top
        y = top + n_lines * lh if v == 0 else top + n_lines * lh / 2.0 if v == 1 else top
        return x, y

    @staticmethod
    def _follow(text: str, vis: int) -> tuple[str, bool, set[str]]:
        """Tags a ruby line copies from the characters it sits on: rotation / shear / paint in effect there and the
        \\t animations of those.  Returns (tags, needs the base's rotation origin, animated names it cannot follow)."""
        st, ts = state_at(text, vis, TRANSFORM + PAINT)
        out = [t.raw for c, t in st.items()]
        needs_org = any(c in TRANSFORM for c in st)
        dropped: set[str] = set()
        for t in ts:
            kept, gone = filter_t(t, FOLLOW_IN_T)
            dropped |= gone & LAYOUT_ANIMATED
            if kept:
                out.append(kept)
                needs_org = needs_org or any(canonical(x.name) in TRANSFORM for x in split_t(t)[1])
        return "".join(out), needs_org, dropped

    @staticmethod
    def _fay_shift(text: str, boxes, vis: VisibleText, start: int, ruby_x0: float) -> float:
        """libass moves glyphs down by \\fay × advance, accumulated from the start of each line.  The ruby line
        starts its own accumulation at 0, so it is moved down by what the base line has accumulated where the ruby
        starts (in the unrotated layout, like the base)."""
        line = vis.line_of(start)
        idx = [c for c in range(len(boxes)) if boxes[c] is not None and vis.line_of(c) == line]
        if not idx:
            return 0.0
        shift, prev = 0.0, None
        fay_at = {c: (state_at(text, c, ("fay",))[0].get("fay")) for c in idx}
        for c in idx:
            if prev is not None:
                t = fay_at[prev]
                shift += ((t.num() or 0.0) if t is not None else 0.0) * (boxes[c].x0 - boxes[prev].x0)
            if c >= start:
                t = fay_at[c]
                return shift + ((t.num() or 0.0) if t is not None else 0.0) * (ruby_x0 - boxes[c].x0)
            prev = c
        return shift

    def _limit(self, text: str, boxes, covered: dict[int, int], p: RubyPiece, direction: int, edge: float, rfs: float) -> float:
        """How far the ruby's visible extent may reach toward a neighbour."""
        n = len(text)
        c = p.start - 1 if direction < 0 else p.end
        overhang = self.overhang * rfs
        if c < 0 or c >= n:
            return edge - 2 * rfs if direction < 0 else edge + 2 * rfs  # line edge: free space
        ch = text[c]
        if c in covered:
            return float("-inf") if direction < 0 else float("inf")  # neighbour has ruby: separation decides
        if is_kanji(ch) or is_digit_char(ch):
            return edge  # never overhang an un-annotated kanji / digit
        if is_space(ch):
            return edge - overhang * 2 if direction < 0 else edge + overhang * 2
        b = boxes[c] if c < len(boxes) else None
        if direction < 0:
            lim = edge - overhang
            return max(lim, b.x0) if b else lim
        lim = edge + overhang
        return min(lim, b.x1) if b else lim
