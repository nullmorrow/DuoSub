"""Qt-free, deterministic ASS effect contract. All times in the context are milliseconds."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from ..ass.document import Event, Style


class ExportError(ValueError):
    pass


class EventLimit(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    type: str
    default: Any
    label: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()

    def validate(self, value):
        valid = ((self.type == "bool" and isinstance(value, bool)) or
                 (self.type == "int" and type(value) is int) or
                 (self.type == "float" and type(value) in (int, float)) or
                 (self.type == "choice" and value in self.choices) or
                 (self.type == "color" and isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value)) or
                 (self.type == "colors" and isinstance(value, list) and
                  all(isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v) for v in value)))
        if not valid or (self.minimum is not None and not self.minimum <= value <= self.maximum):
            raise ExportError(f"卡拉 OK 特效参数无效：{self.label}")
        return copy.deepcopy(value)


COMMON = {
    "lead_in_ms": Parameter("int", 300, "提前出现（毫秒）", 0, 10000),
    "keep_after_ms": Parameter("int", 0, "唱完后保留（毫秒）", 0, 10000),
    "fade_in_ms": Parameter("int", 150, "淡入（毫秒）", 0, 10000),
    "fade_out_ms": Parameter("int", 150, "淡出（毫秒）", 0, 10000),
    "follow_furigana": Parameter("bool", True, "振假名跟着"),
    "follow_translation": Parameter("bool", True, "翻译行跟着"),
}


@dataclass
class FxGlyph:
    event: Event
    box: tuple[float, float, float, float]
    tags: str
    text: str
    x: float
    y: float
    beats: list[tuple[int, int, int]]       # beat index, absolute start/end
    owner: dict = field(default_factory=dict)
    scale_x: float = 100
    scale_y: float = 100
    movement: tuple | None = None         # dx, dy, original relative start/end
    anchor_ids: tuple = ()                # base identities; ruby uses its whole base piece


@dataclass
class FxContext:
    member: str
    start: int
    end: int
    style: Style
    glyphs: list[FxGlyph]
    params: dict
    alignment: int = 2
    position: tuple[float, float] = (0, 0)
    rng: random.Random = field(default_factory=random.Random)
    owners: list[dict] = field(default_factory=list)
    event_cap: int = 2000
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def seeded(member: str, params: dict) -> random.Random:
        raw = json.dumps([member, params], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return random.Random(int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest(), "big"))

    def emit(self, events, glyph, start, end, tags="", *, layer=None, position=None):
        # Match export.ms_to_cs; do not manufacture zero-duration dialogue lines.
        a, z = rounded_ms(start) // 10, rounded_ms(end) // 10
        if z <= a:
            return
        if len(events) >= self.event_cap:
            raise EventLimit()
        explicit_position = position is not None
        position = position or f"\\pos({glyph.x:.2f},{glyph.y:.2f})"
        if glyph.movement is not None and not explicit_position:
            dx, dy, t0, t1 = glyph.movement
            first, last = self.start+t0, self.start+t1
            f = min(1, max(0, (a*10-first)/max(1, last-first)))
            x, y = position_at(self, glyph, a*10)
            position = (f"\\move({x:.2f},{y:.2f},{glyph.x+dx:.2f},{glyph.y+dy:.2f},"
                        f"{max(0, first-a*10)},{max(1, last-a*10)})" if f < 1 else
                        f"\\pos({x:.2f},{y:.2f})")
        head = "\\an7" + position + glyph.tags + tags
        events.append(glyph.event.copy(start=a, end=z, effect="", layer=max(0, glyph.event.layer if layer is None else layer),
                                       text="{" + head + "}" + glyph.text))
        self.owners.append(dict(glyph.owner))


@dataclass
class Effect:
    name: str
    display_name: str
    parameters: dict[str, Parameter] = field(default_factory=dict)

    @property
    def schema(self):
        return {**COMMON, **self.parameters}

    def resolve(self, values=None):
        values = values or {}
        unknown = set(values) - self.schema.keys()
        if unknown:
            raise ExportError(f"卡拉 OK 特效参数无效：{sorted(unknown)[0]}")
        return {k: p.validate(values.get(k, p.default)) for k, p in self.schema.items()}

    def generate(self, ctx: FxContext) -> list[Event]:
        raise NotImplementedError


def color(value: str) -> str:
    if value.startswith("#"):
        return "&H" + value[5:7] + value[3:5] + value[1:3] + "&"
    return "&H" + value.upper().replace("&H", "").replace("&", "")[-6:] + "&"


def transform(a, z, tags):
    return f"\\t({max(0, int(a))},{max(1, int(z))},{tags})"


def rounded_ms(t):
    """One shared ASS boundary for both sides of each segment."""
    return max(0, (round(t) + 5) // 10) * 10


def position_at(ctx, glyph, t):
    if glyph.movement is None:
        return glyph.x, glyph.y
    dx, dy, a, z = glyph.movement
    f = min(1, max(0, (t-ctx.start-a)/max(1, z-a)))
    return glyph.x+dx*f, glyph.y+dy*f


def paint(ctx, glyph, start):
    """Carry completed highlight history into the event's initial state.

    Replaying a past beat with t(0,1) leaves one unsung frame in libass.
    Only future beats need transforms; the first started beat is permanent.
    """
    origin = rounded_ms(start)
    first = min(a for _, a, _ in glyph.beats)
    if first <= origin:
        return "\\1c" + color(ctx.style.get("PrimaryColour"))
    return ("\\1c" + color(ctx.style.get("SecondaryColour")) +
            transform(first-origin-1, first-origin, "\\1c" + color(ctx.style.get("PrimaryColour"))))


def emit_scaled(ctx, events, glyph, start, end, peaks, *, bottom=False):
    """Compensate an7 placement so scaling fixes the measured ink centre/bottom.

    ASS cannot animate pos with t; split linear scale legs into events and move
    their top-left anchors by the matching linear compensation instead.
    """
    start, end = rounded_ms(start), rounded_ms(end)
    knots = {start: (1, 1), end: (1, 1)}
    for t, x, y in peaks:
        knots[min(end, max(start, rounded_ms(t)))] = (x, y)
    times = sorted(knots)

    def scale(t):
        for a, z in zip(times, times[1:]):
            if a <= t <= z:
                f = (t-a)/(z-a)
                return tuple(knots[a][i]*(1-f) + knots[z][i]*f for i in (0, 1))
        return knots[times[-1]]

    cuts = set(times)
    if glyph.movement:
        _, _, a, z = glyph.movement
        cuts.update(t for t in (rounded_ms(ctx.start+a), rounded_ms(ctx.start+z)) if start < t < end)
    cx = (glyph.box[0]+glyph.box[2])/2
    cy = glyph.box[3] if bottom else (glyph.box[1]+glyph.box[3])/2

    def position(t, sx, sy):
        x, y = cx+(glyph.x-cx)*sx, cy+(glyph.y-cy)*sy
        mx, my = position_at(ctx, glyph, t)
        x, y = x+mx-glyph.x, y+my-glyph.y
        return x, y

    fi = min(ctx.params["fade_in_ms"], (end-start)//2)
    fo = min(ctx.params["fade_out_ms"], (end-start)//2)
    cuts = sorted(cuts)
    for a, z in zip(cuts, cuts[1:]):
        sx, sy = scale(a)
        ex, ey = scale(z)
        x0, y0 = position(a, sx, sy)
        x1, y1 = position(z, ex, ey)
        tags = paint(ctx, glyph, a)
        # Full-line fades retain their original time origin across split events.
        tags += f"\\fade(255,0,255,{start-a},{start+fi-a},{end-fo-a},{end-a})"
        tags += f"\\fscx{glyph.scale_x*sx:g}\\fscy{glyph.scale_y*sy:g}"
        tags += transform(0, z-a, f"\\fscx{glyph.scale_x*ex:g}\\fscy{glyph.scale_y*ey:g}")
        pos = (f"\\move({x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f},0,{z-a})" if (x0,y0) != (x1,y1) else
               f"\\pos({x0:.2f},{y0:.2f})")
        ctx.emit(events, glyph, a, z, tags, position=pos)
