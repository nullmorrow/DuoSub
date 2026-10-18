"""Discrete XY trembling; ruby and its base share one deterministic motion unit."""
from __future__ import annotations

import hashlib
import math

from .base import Effect, EventLimit, Parameter, paint, position_at, rounded_ms


def _units(glyphs):
    """Join only explicit base anchors, including a ruby piece spanning kanji."""
    parents = list(range(len(glyphs)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    anchors = {}
    for i, g in enumerate(glyphs):
        for anchor in g.anchor_ids:
            if anchor in anchors:
                a, b = root(i), root(anchors[anchor])
                parents[max(a, b)] = min(a, b)
            else:
                anchors[anchor] = i
    return [root(i) for i in range(len(glyphs))]


class FearEffect(Effect):
    def generate(self, ctx):
        p = ctx.params
        glyphs = [g for g in ctx.glyphs if g.beats]
        if not glyphs:
            return []
        start = rounded_ms(max(0, ctx.start - p["lead_in_ms"]))
        # Shared boundaries keep every ruby piece locked to its base, even when
        # the kana and kanji have different final mora timings.
        last = max(z for g in glyphs for _, _, z in g.beats)
        end = rounded_ms(last + p["keep_after_ms"] if p["keep_after_ms"] else max(ctx.end, last))
        if end <= start:
            return []
        interval = math.ceil(p["interval_ms"] / 10) * 10
        movement_cuts = {rounded_ms(ctx.start + t) for g in glyphs if g.movement
                         for t in g.movement[2:] if start < rounded_ms(ctx.start + t) < end}
        # Long/ruby-heavy lines use a slower cadence, never a partial effect.
        slots = ctx.event_cap // len(glyphs)
        if slots < 1:
            raise EventLimit()
        required = math.ceil((end - start) / slots / 10) * 10
        interval = max(interval, required)
        while interval <= 200:
            cuts = sorted(set(range(start, end, interval)) | movement_cuts | {end})
            if len(cuts)-1 <= slots:
                break
            interval += 10
        else:
            raise EventLimit()
        if interval > math.ceil(p["interval_ms"] / 10) * 10:
            stamp = f"{ctx.start//3600000}:{ctx.start//60000%60:02d}:{ctx.start%60000/1000:05.2f}"
            ctx.warnings.append(f"{stamp} 惊恐颤抖：长句已放慢为每 {interval} 毫秒移动一次")
        units = _units(glyphs)
        seed = ctx.rng.getrandbits(128)
        fi, fo = min(p["fade_in_ms"], (end - start) // 2), min(p["fade_out_ms"], (end - start) // 2)
        events = []
        for a, z in zip(cuts, cuts[1:]):
            frame = (a-start)//interval
            offsets = {}
            for g, unit in zip(glyphs, units):
                if p["whole_line"]:
                    unit = 0
                if unit not in offsets:
                    digest = hashlib.sha256(f"{seed}:{unit}:{frame}".encode("ascii")).digest()
                    # Integer script pixels preserve the same raster translation
                    # for the small ruby and the much larger base glyph.
                    offsets[unit] = tuple(int.from_bytes(digest[i:i+4], "big") % (2*amplitude+1) - amplitude
                                          for i, amplitude in ((0, p["jitter_x"]), (4, p["jitter_y"])))
                dx, dy = offsets[unit]
                x0, y0 = position_at(ctx, g, a)
                x1, y1 = position_at(ctx, g, z)
                t0, t1 = 0, z-a
                if g.movement:
                    # ASS move times have millisecond precision, including when
                    # an original endpoint falls between centisecond cuts.
                    t0 = max(0, ctx.start+g.movement[2]-a)
                    t1 = min(z-a, ctx.start+g.movement[3]-a)
                pos = (f"\\move({x0+dx:.2f},{y0+dy:.2f},{x1+dx:.2f},{y1+dy:.2f},{t0},{t1})"
                       if (x0, y0) != (x1, y1) else f"\\pos({x0+dx:.2f},{y0+dy:.2f})")
                first = rounded_ms(min(t for _, t, _ in g.beats))
                # Keep fades on the line clock, not restarted on every fragment.
                tags = paint(ctx, g, a) + f"\\fade(255,0,255,{start-a},{start+fi-a},{end-fo-a},{end-a})"
                if first > a:
                    tags += f"\\alpha&H{p['unsung_alpha']:02X}&"
                    if first < z:
                        tags += f"\\t({max(0, first-a-1)},{first-a},\\alpha&H00&)"
                else:
                    tags += "\\alpha&H00&"
                ctx.emit(events, g, a, z, tags, position=pos)
        return events


EFFECTS = [FearEffect("tremble", "惊恐颤抖", {
    "jitter_x": Parameter("int", 3, "左右幅度（像素）", 0, 30),
    "jitter_y": Parameter("int", 3, "上下幅度（像素）", 0, 30),
    "interval_ms": Parameter("int", 40, "移动间隔（毫秒）", 20, 200),
    "unsung_alpha": Parameter("int", 208, "未唱文字透明度（0 不透明，255 透明）", 0, 255),
    "whole_line": Parameter("bool", False, "整行一起抖"),
})]
