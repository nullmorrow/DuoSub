"""Original beat-driven motion and vector effects, without external templates."""
from __future__ import annotations

import math
import re
from dataclasses import replace

from .base import Effect, Parameter, color, emit_scaled, paint, position_at, rounded_ms, transform


def _origin(ms):
    return rounded_ms(ms)


def _move(x0, y0, x1, y1, a, z):
    return f"\\move({x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f},{max(0, round(a))},{max(1, round(z))})"


def _span(ctx, g):
    p = ctx.params
    last = max(z for _, _, z in g.beats)
    start = max(0, ctx.start - p["lead_in_ms"])
    end = last + p["keep_after_ms"] if p["keep_after_ms"] else max(ctx.end, last)
    return start, end


def _paint(ctx, g, start, end):
    p = ctx.params
    fi = min(p["fade_in_ms"], max(0, (end-start)//2))
    fo = min(p["fade_out_ms"], max(0, (end-start)//2))
    return paint(ctx, g, start) + f"\\fad({fi},{fo})"


def _path(shape, size):
    """Centred paths drawn in local pixels; cubic curves are our own geometry."""
    if shape == "星":
        points = []
        for i in range(10):
            angle = -math.pi/2 + i*math.pi/5
            r = size * (1 if i % 2 == 0 else .42)
            points.append(f"{r*math.cos(angle):.2f} {r*math.sin(angle):.2f}")
        return "m " + points[0] + " l " + " ".join(points[1:] + points[:1])
    if shape == "心":
        coords = [(0, .9), (-1.5, -.05), (-.9, -1.4), (0, -.45),
                  (.9, -1.4), (1.5, -.05), (0, .9)]
    elif shape == "花瓣":
        coords = [(0, -1), (1.2, -.5), (1.2, .5), (0, 1),
                  (-.45, .35), (-.45, -.35), (0, -1)]
    else:  # circle: four cubic arcs
        k = .55228475
        coords = [(0, -1), (k, -1), (1, -k), (1, 0), (1, k), (k, 1), (0, 1),
                  (-k, 1), (-1, k), (-1, 0), (-1, -k), (-k, -1), (0, -1)]
    pts = [f"{x*size:.2f} {y*size:.2f}" for x, y in coords]
    return "m " + pts[0] + " b " + " ".join(pts[1:])


class MotionEffect(Effect):
    def generate(self, ctx):
        events = []
        p = ctx.params
        for g in ctx.glyphs:
            if not g.beats:
                continue
            start, end = _span(ctx, g)
            if self.name == "bounce":
                self._bounce(ctx, events, g, start, end)
                continue
            tags = _paint(ctx, g, start, end)
            if self.name == "fly":
                self._fly(ctx, events, g, start, end)
                continue
            if self.name in ("shake", "morph"):
                g = replace(g, tags=re.sub(r"\\org\([^)]*\)", "", g.tags))
                cx, cy = (g.box[0]+g.box[2])/2, (g.box[1]+g.box[3])/2
                if self.name == "shake":
                    tags += f"\\org({cx:.2f},{cy:.2f})"
                scale_peaks = []
                for index, a, z in g.beats:
                    t, duration = max(0, a-_origin(start)), max(1, z-a)
                    if self.name == "shake":
                        steps = 2*p["count"]
                        for j in range(steps):
                            angle = p["angle"] * (1 if j % 2 == 0 else -1)
                            tags += transform(t+duration*j/steps, t+duration*(j+1)/steps, f"\\frz{angle:g}")
                        tags += transform(t+duration, t+duration+1, "\\frz0")
                    else:
                        amp = p["amplitude"]/100
                        scale_peaks.append((a, 1, 1))
                        if p["mode"] == "波浪":
                            peaks = [(1+amp/2, 1-amp/2), (1-amp/2, 1+amp/2),
                                     (1+amp/4, 1-amp/4), (1, 1)]
                            if index % 2:
                                peaks = [(y, x) for x, y in peaks]
                            for j, (x, y) in enumerate(peaks):
                                scale_peaks.append((a+duration*(j+1)/4, x, y))
                            continue
                        if p["mode"] == "拉伸":
                            x, y = 1+amp, 1-amp*2/3
                        else:
                            x, y = 1-amp*2/3, 1+amp
                        scale_peaks += [(a+duration/2, x, y), (z, 1, 1)]
                if self.name == "morph":
                    emit_scaled(ctx, events, g, start, end, scale_peaks, bottom=True)
                    continue
            ctx.emit(events, g, start, end, tags)
        if self.name == "particles":
            self._particles(ctx, events)
        return events

    def _bounce(self, ctx, events, g, start, end):
        p = ctx.params
        start, end = _origin(start), _origin(end)
        cursor = start
        knots = {start: 0, end: 0}
        for _, a, z in sorted(g.beats, key=lambda beat: beat[1]):
            a, z = max(_origin(a), cursor), min(_origin(z), end)
            if z <= a:
                continue
            for i in range(2*p["count"]+1):
                b = _origin(a+(z-a)*i/(2*p["count"]))
                knots[b] = p["height"] if i % 2 else 0
            cursor = z
        times = sorted(knots)
        cuts = set(times)
        if g.movement:
            _, _, a, z = g.movement
            cuts.update(t for t in (_origin(ctx.start+a), _origin(ctx.start+z)) if start < t < end)
        def height(t):
            for a, z in zip(times, times[1:]):
                if a <= t <= z:
                    f = (t-a)/(z-a)
                    return knots[a]*(1-f)+knots[z]*f
            return knots[times[-1]]
        fi, fo = min(p["fade_in_ms"], (end-start)//2), min(p["fade_out_ms"], (end-start)//2)
        cuts = sorted(cuts)
        for a, z in zip(cuts, cuts[1:]):
            x0, y0 = position_at(ctx, g, a)
            x1, y1 = position_at(ctx, g, z)
            y0, y1 = y0-height(a), y1-height(z)
            position = (_move(x0, y0, x1, y1, 0, z-a) if (x0,y0) != (x1,y1) else
                        f"\\pos({x0:.2f},{y0:.2f})")
            tags = paint(ctx, g, a) + f"\\fade(255,0,255,{start-a},{start+fi-a},{end-fo-a},{end-a})"
            ctx.emit(events, g, a, z, tags, position=position)

    def _fly(self, ctx, events, g, start, end):
        p = ctx.params
        # Stagger characters, including multiple characters sharing one beat.
        # Ruby takes the rank of the closest base character in the same beat.
        base = [glyph for glyph in ctx.glyphs if glyph.owner.get("kind") != "ruby"]
        if g.owner.get("kind") == "ruby":
            indices = {i for i, _, _ in g.beats}
            matching = [glyph for glyph in base if indices.intersection(i for i, _, _ in glyph.beats)]
            parent = min(matching or base, key=lambda glyph: abs(glyph.x-g.x)) if base else g
            rank = base.index(parent) if base else 0
        else:
            rank = base.index(g)
        start, end = _origin(start), _origin(end)
        arrival = _origin(min(ctx.start, min(a for _, a, _ in g.beats)))
        stagger = min(40, max(0, (arrival-start)//max(1, len(base))))
        entry = _origin(min(arrival, start + rank*stagger))
        distance = max(24, ctx.style.fontsize*1.5)
        dx, dy = {"左": (-distance, 0), "右": (distance, 0),
                  "上": (0, -distance), "下": (0, distance)}[p["direction"]]
        exit_start = _origin(max(arrival, end-min(p["fade_out_ms"], max(0, end-arrival))))
        if arrival > entry:
            x0, y0 = position_at(ctx, g, entry)
            x1, y1 = position_at(ctx, g, arrival)
            ctx.emit(events, g, entry, arrival,
                f"\\1c{color(ctx.style.get('SecondaryColour'))}\\fad({min(p['fade_in_ms'], arrival-entry)},0)",
                position=_move(x0+dx, y0+dy, x1, y1, 0, arrival-entry))
        # Singing has no repeated entry fade or premature exit fade.
        ctx.emit(events, g, arrival, exit_start, paint(ctx, g, arrival))
        if end > exit_start:
            dx = dy = 0
            if p["after"] == "散开":
                angle = ctx.rng.uniform(0, math.tau)
                dx, dy = math.cos(angle)*distance, math.sin(angle)*distance
            elif p["after"] == "落下":
                dy = distance
            x0, y0 = position_at(ctx, g, exit_start)
            x1, y1 = position_at(ctx, g, end)
            ctx.emit(events, g, exit_start, end,
                     f"\\1c{color(ctx.style.get('PrimaryColour'))}\\fad(0,{end-exit_start})",
                     position=_move(x0, y0, x1+dx, y1+dy, 0, end-exit_start))

    def _particles(self, ctx, events):
        p = ctx.params
        beats = {}
        for g in ctx.glyphs:
            for index, a, z in g.beats:
                beats.setdefault((index, a, z), []).append(g)
        for (_, a, _), glyphs in sorted(beats.items()):
            for _ in range(p["count"]):
                g = ctx.rng.choice(glyphs)
                x = ctx.rng.uniform(g.box[0], g.box[2])
                y = ctx.rng.uniform(g.box[1], g.box[3])
                angle = ctx.rng.uniform(0, math.tau)
                distance = ctx.rng.uniform(.25, 1)*p["spread"]
                dx, dy = math.cos(angle)*distance, math.sin(angle)*distance
                rotation = ctx.rng.uniform(-20, 20)
                style = ctx.style
                # Measured glyph height also keeps ruby particles proportionate.
                size = max(1, min(style.fontsize, g.box[3]-g.box[1])*.14)
                particle = replace(g, tags="", text=_path(p["shape"], size), x=x, y=y, movement=None)
                duration = p["duration_ms"]
                tags = (f"\\p1\\an7\\fscx100\\fscy100\\bord0\\shad0\\blur0.4"
                        f"\\1c{color(p['color'])}\\fad({min(30, duration//3)},{duration//2})"
                        + transform(0, duration, f"\\frz{rotation:.2f}"))
                ctx.emit(events, particle, a, a+duration, tags, layer=max(0, g.event.layer)+1,
                         position=_move(x, y, x+dx, y+dy, a-_origin(a), duration+a-_origin(a)))


EFFECTS = [
    MotionEffect("bounce", "跳动", {"height": Parameter("float", 12, "高度（像素）", 0, 200),
                                    "count": Parameter("int", 1, "次数", 1, 10)}),
    MotionEffect("shake", "抖动", {"angle": Parameter("float", 8, "角度", 0, 45),
                                  "count": Parameter("int", 3, "次数", 1, 20)}),
    MotionEffect("particles", "星光 / 粒子", {
        "shape": Parameter("choice", "星", "形状：星 / 心 / 圆 / 花瓣", choices=("星", "心", "圆", "花瓣")),
        "count": Parameter("int", 6, "数量", 1, 100),
        "color": Parameter("color", "#FFFFFF", "颜色"),
        "spread": Parameter("float", 40, "扩散距离", 0, 500),
        "duration_ms": Parameter("int", 600, "持续（毫秒）", 10, 5000)}),
    MotionEffect("fly", "飞入飞出", {
        "direction": Parameter("choice", "左", "飞入方向", choices=("左", "右", "上", "下")),
        "after": Parameter("choice", "散开", "唱完：散开 / 落下 / 淡出", choices=("散开", "落下", "淡出"))}),
    MotionEffect("morph", "字形变化", {
        "mode": Parameter("choice", "拉伸", "方式", choices=("拉伸", "挤压", "波浪")),
        "amplitude": Parameter("float", 30, "幅度", 0, 80)}),
]
