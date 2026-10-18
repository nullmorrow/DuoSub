"""Original implementations of the first five effects; no external templates."""
from __future__ import annotations

import colorsys

from .base import Effect, Parameter, color, emit_scaled, paint, rounded_ms, transform


class BasicEffect(Effect):
    def generate(self, ctx):
        p = ctx.params
        events = []
        unsung = color(ctx.style.get("SecondaryColour"))
        sung = color(ctx.style.get("PrimaryColour"))
        for char_index, g in enumerate(ctx.glyphs):
            if not g.beats:
                continue
            below = self.name == "glow" or (self.name == "wipe" and p["soft_edge"])
            layer = max(1 if below else 0, g.event.layer)
            start = max(0, ctx.start - p["lead_in_ms"])
            last = max(z for _, _, z in g.beats)
            end = max(ctx.end, last) if not p["keep_after_ms"] else last + p["keep_after_ms"]
            if self.name == "fade":
                start = min(g.beats[0][1], start + char_index * p["stagger_ms"])
                end = last + p["fade_out_ms"]
            fi, fo = min(p["fade_in_ms"], (end-start)//2), min(p["fade_out_ms"], (end-start)//2)
            tags = (f"\\1c{unsung}" if self.name in ("color", "wipe") else paint(ctx, g, start))
            tags += f"\\fad({fi},{fo})"
            peaks = []
            glow_spans = []
            wipe_tags = ""
            total = sum(max(1, z-a) for _, a, z in g.beats)
            progress = 0
            for index, a, z in g.beats:
                # All transform times are relative to the rounded ASS event start.
                origin = ((round(start) + 5) // 10) * 10
                t, duration = max(0, a-origin), max(1, z-a)
                target = sung
                if self.name == "color":
                    if p["rainbow"]:
                        rgb = colorsys.hsv_to_rgb((index * .137) % 1, .8, 1)
                        target = color("#" + "".join(f"{round(c*255):02X}" for c in rgb))
                    elif p["colors"]:
                        target = color(p["colors"][index % len(p["colors"] )])
                if self.name == "color":
                    tags += transform(t, t + duration, f"\\1c{target}")
                if self.name == "pop":
                    peak = max(1, min(100, duration//3))
                    peaks += [(a, 1, 1), (a+peak, p['scale']/100, p['scale']/100),
                              (a+peak+min(p["return_ms"], max(1, duration-peak)), 1, 1)]
                elif self.name == "glow":
                    tags += transform(t, t+min(100, duration), f"\\3c{color(p['glow_color'])}")
                    if not p["keep_glow"]:
                        tags += transform(t+duration, t+duration+100, "\\3c" + color(ctx.style.get("OutlineColour")))
                    glow_spans.append((rounded_ms(a), rounded_ms(end if p["keep_glow"] else z)))
                elif self.name == "wipe":
                    x0, y0, x1, y1 = g.box
                    pad = max(2, ctx.style.outline + ctx.style.shadow + 2)
                    empty = (f"{x0-pad:.2f},{y1+pad:.2f},{x1+pad:.2f},{y1+pad:.2f}" if p["direction"] == "up" else
                             f"{x0-pad:.2f},{y0-pad:.2f},{x0-pad:.2f},{y1+pad:.2f}")
                    # One glyph-wide overlay; later beats extend the previous
                    # coverage instead of restarting an empty clip.
                    progress += duration
                    f = progress/total
                    edge = (f"{x0-pad:.2f},{y1+pad-(y1-y0+2*pad)*f:.2f},{x1+pad:.2f},{y1+pad:.2f}" if p["direction"] == "up" else
                            f"{x0-pad:.2f},{y0-pad:.2f},{x0-pad+(x1-x0+2*pad)*f:.2f},{y1+pad:.2f}")
                    if not wipe_tags:
                        wipe_tags = f"\\1c{sung}\\clip({empty})"
                    origin = rounded_ms(min(b for _, b, _ in g.beats))
                    wipe_tags += transform(max(0, a-origin), max(1, z-origin), f"\\clip({edge})")
            if self.name == "glow":
                # Contiguous/overlapping glow windows share one layer event,
                # avoiding repeated fades and double-drawn retained glow.
                spans = []
                for a, z in sorted(glow_spans):
                    if spans and a <= spans[-1][1]:
                        spans[-1] = (spans[-1][0], max(z, spans[-1][1]))
                    else:
                        spans.append((a, z))
                for a, z in spans:
                    ctx.emit(events, g, a, z,
                             f"\\1c{color(p['glow_color'])}\\3c{color(p['glow_color'])}\\bord{p['glow_size']:g}"
                             f"\\blur{p['glow_size']:g}\\fad(50,50)", layer=layer-1)
            if self.name == "wipe":
                tags = f"\\1c{unsung}\\fad({fi},{fo})"
                origin = rounded_ms(min(a for _, a, _ in g.beats))
                wipe_tags += f"\\fade(255,0,255,{rounded_ms(start)-origin},{rounded_ms(start)+fi-origin},{rounded_ms(end)-fo-origin},{rounded_ms(end)-origin})"
                ctx.emit(events, g, origin, end, wipe_tags, layer=layer+1)
                if p["soft_edge"]:
                    ctx.emit(events, g, origin, end, wipe_tags + "\\blur2", layer=layer-1)
            if self.name == "pop":
                emit_scaled(ctx, events, g, start, end, peaks)
            else:
                ctx.emit(events, g, start, end, tags, layer=layer)
        return events


EFFECTS = [
    BasicEffect("pop", "放大弹出", {"scale": Parameter("int", 150, "放大到 %", 120, 250),
                                  "return_ms": Parameter("int", 200, "回落时间", 1, 3000)}),
    BasicEffect("glow", "发光", {"glow_color": Parameter("color", "#FFFFFF", "发光颜色"),
                                "glow_size": Parameter("float", 4, "光晕大小", 0, 30),
                                "keep_glow": Parameter("bool", False, "唱完保持发光")}),
    BasicEffect("color", "变色 / 彩虹", {"colors": Parameter("colors", [], "颜色"),
                                        "rainbow": Parameter("bool", False, "彩虹")}),
    BasicEffect("fade", "逐字淡入淡出", {"stagger_ms": Parameter("int", 40, "每字间隔（毫秒）", 0, 1000)}),
    BasicEffect("wipe", "擦除", {"soft_edge": Parameter("bool", False, "柔边"),
                                "direction": Parameter("choice", "right", "方向：左→右 / 下→上", choices=("right", "up"))}),
]
