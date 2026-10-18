"""Free-transform session and drag geometry, independent of Qt and rendering.

The caller captures precise LineGeom frames once at entry (including plain
lines when shear is wanted). Drags only use that data and current overrides.
Preview changes are relative to current overrides; changes() is relative to
entry and can be passed to Editor.set_overrides as one history operation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.text import line_tag
from subkit.core.export import apply_overrides
from subkit.core.layout_index import LineGeom, Rect, in_quad, unproject
from subkit.core.transform import (Point, Quad, TransformState, apply_group_transform,
                                   line_quad, source_transform_tags, transform_of)

HINT = "自由变换：拖手柄调整，Enter 确定，Esc 取消，Backspace 清除变形"
DISABLED_3D_REASON = "3D 旋转过的行不能再缩放、斜切或移动旋转中心（先 Backspace 清除变形）"
_CORNERS = ("nw", "ne", "se", "sw")
_EDGES = ("n", "e", "s", "w")


@dataclass(frozen=True)
class TransformMember:
    member: str
    source_event: Event             # before member overrides, with exported style name
    style: Style
    anchor: Mapping[str, Any]       # entry LayoutIndex.anchor
    overrides: Mapping[str, Any]
    frames: tuple[Rect, ...]        # unrotated, entry-scaled ink boxes
    layout_origins: tuple[Point, ...]
    measured_scale: Point
    source_tags: frozenset[str]
    original: bool = False

    @classmethod
    def from_geom(cls, geom: LineGeom, source_event: Event, style: Style,
                  anchor: Mapping[str, Any], overrides: Mapping[str, Any], *,
                  measured_scale: Point, original: bool = False,
                  layout_origins: Optional[Sequence[Point]] = None) -> TransformMember:
        """Capture a base line; origins may also be supplied by the caller's probe.

        Plain watch geometry has no layout origins. Capture it through the
        precise LayoutIndex path before entry; ink bearings cannot replace
        measured layout origins, especially for centre/right alignment.
        """
        if geom.kind != "base":
            raise ValueError("transform members must be base lines")
        frames = tuple(frame[0] for frame in geom._frames) or (geom.rect,)
        origins = tuple(geom.layout_origins if layout_origins is None else layout_origins)
        return deepcopy(cls(geom.member, source_event, style, anchor, overrides, frames, origins,
                            measured_scale, source_transform_tags(source_event.text), original))


@dataclass(frozen=True)
class Handle:
    name: str
    point: Point
    enabled: bool = True
    reason: str = ""


@dataclass(frozen=True)
class Drag:
    handle: str
    start: Point
    current: Point
    shift: bool = False
    alt: bool = False
    ctrl: bool = False
    screen_scale: Point = (1.0, 1.0)


def _merge(values: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(values))
    for key, value in change.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = deepcopy(value)
    return out


def _point(point: Point) -> Point:
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise ValueError("coordinates must be finite points")
    return float(point[0]), float(point[1])


class TransformTool:
    hint = HINT

    def __init__(self, ass: AssDocument, members: Sequence[TransformMember], *,
                 reference: Optional[str] = None, perspective_distance: float = 312.5) -> None:
        if not members or len({m.member for m in members}) != len(members):
            raise ValueError("a transform selection needs distinct members")
        if not math.isfinite(perspective_distance) or perspective_distance <= 0:
            raise ValueError("3D: perspective distance must be positive")
        self._ass = ass.copy()
        self._members = deepcopy(tuple(members))
        self.reference = reference or next((m.member for m in members if m.original), members[0].member)
        if self.reference not in {m.member for m in members}:
            raise ValueError("reference must belong to the selection")
        for member in self._members:
            if not member.frames or len(member.frames) != len(member.layout_origins):
                raise ValueError("each untransformed frame needs a measured layout origin")
            if any(not math.isfinite(v) or v <= 0 for v in member.measured_scale):
                raise ValueError("measured scales must be positive")
            for origin in member.layout_origins:
                _point(origin)
            self._ass.styles = [s for s in self._ass.styles if s.name != member.style.name]
            self._ass.styles.append(member.style.copy())
        self._entry = {m.member: deepcopy(dict(m.overrides)) for m in self._members}
        self._current = deepcopy(self._entry)
        self.mode_3d = False
        self.perspective_distance = perspective_distance

    @property
    def current_overrides(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._current)

    def toggle_3d(self) -> bool:
        self.mode_3d = not self.mode_3d
        return self.mode_3d

    def _states(self) -> list[TransformState]:
        states = []
        for member in self._members:
            overrides = self._current[member.member]
            event = member.source_event.copy(text=apply_overrides(member.source_event.text, overrides)[0])
            tf = transform_of(self._ass, event)
            anchor = deepcopy(dict(member.anchor))
            # Keep the measured style anchor unless the exported event has an
            # explicit position. Removing an override exposes source positions.
            entry_event = member.source_event.copy(
                text=apply_overrides(member.source_event.text, self._entry[member.member])[0])
            source_position = line_tag(member.source_event.text, "move") or line_tag(member.source_event.text, "pos")
            source_an = line_tag(member.source_event.text, "an")
            anchor["an"] = int(overrides.get("an") or
                               (source_an.num() if source_an else member.style.alignment))
            position = line_tag(event.text, "move") or line_tag(event.text, "pos")
            if position is not None:
                values = position.float_args() or []
                if len(values) not in (2, 4, 6):
                    raise ValueError("invalid member position")
                anchor.update(x=values[0], y=values[1], move=values if len(values) > 2 else None)
            elif source_position is None and event.text == entry_event.text:
                anchor = deepcopy(dict(member.anchor))
            offset = (member.layout_origins[0][0] - member.anchor["x"],
                      member.layout_origins[0][1] - member.anchor["y"])
            origin = (anchor["x"] + offset[0] * tf.fscx / member.measured_scale[0],
                      anchor["y"] + offset[1] * tf.fscy / member.measured_scale[1])
            states.append(TransformState(member.member, tf, member.style, anchor, overrides, origin,
                                         self._entry[member.member], member.source_tags))
        return states

    def states(self) -> list[TransformState]:
        """Current placement/transform snapshots for overlays following the preview."""
        return self._states()

    def _box(self) -> tuple[Rect, TransformState]:
        states = self._states()
        reference = next(s for s in states if s.member == self.reference)
        box = None
        for member, state in zip(self._members, states):
            sx = state.transform.fscx / member.measured_scale[0]
            sy = state.transform.fscy / member.measured_scale[1]
            for frame in member.frames:
                current = Rect(state.anchor["x"] + (frame.x0 - member.anchor["x"]) * sx,
                               state.anchor["y"] + (frame.y0 - member.anchor["y"]) * sy,
                               state.anchor["x"] + (frame.x1 - member.anchor["x"]) * sx,
                               state.anchor["y"] + (frame.y1 - member.anchor["y"]) * sy)
                box = current if box is None else box.union(current)
        return box, reference

    def box(self) -> Quad:
        rect, reference = self._box()
        tf = reference.transform
        return line_quad(rect, (reference.anchor["x"], reference.anchor["y"]), tf,
                         layout_origin=reference.layout_origin,
                         measured_scale=(tf.fscx or 100, tf.fscy or 100),
                         perspective_distance=self.perspective_distance)

    def pivot(self) -> Point:
        reference = next(s for s in self._states() if s.member == self.reference)
        if reference.transform.org is not None:
            return reference.transform.org
        # Project the centre too: averaging corners is wrong under perspective.
        rect, _state = self._box()
        tf = reference.transform
        return line_quad((rect.cx, rect.cy, rect.cx, rect.cy),
                         (reference.anchor["x"], reference.anchor["y"]), tf,
                         layout_origin=reference.layout_origin,
                         measured_scale=(tf.fscx or 100, tf.fscy or 100),
                         perspective_distance=self.perspective_distance)[0]

    def handles(self, tolerance: float = 6.0) -> tuple[Handle, ...]:
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("handle tolerance must be finite and nonnegative")
        q = self.box()
        disabled = any(s.transform.frx or s.transform.fry for s in self._states())
        def handle(name: str, point: Point, blocked: bool) -> Handle:
            return Handle(name, point, not blocked, DISABLED_3D_REASON if blocked else "")
        out = [handle(name, p, disabled and not self.mode_3d) for name, p in zip(_CORNERS, q)]
        out += [handle(name, ((a[0]+b[0])/2, (a[1]+b[1])/2), disabled)
                for name, a, b in zip(_EDGES, q, (*q[1:], q[0]))]
        centre = (sum(p[0] for p in q)/4, sum(p[1] for p in q)/4)
        for name, p in zip(_CORNERS, q):
            dx, dy = p[0]-centre[0], p[1]-centre[1]
            length = math.hypot(dx, dy) or 1
            distance = 2.5 * tolerance
            out.append(handle("rotate_"+name, (p[0]+dx/length*distance, p[1]+dy/length*distance), False))
        out.append(handle("pivot", self.pivot(), disabled))
        return tuple(out)

    def hit_test(self, point: Point, tolerance: float = 6.0) -> Optional[Handle]:
        point = _point(point)
        handles = self.handles(tolerance)
        # Pivot wins where it overlaps a tiny box; disabled hits retain reason.
        for handle in (handles[-1], *handles[:-1]):
            if math.dist(point, handle.point) <= tolerance:
                return handle
        if in_quad(self.box(), *point, 0):
            return Handle("inside", point)
        return None

    def operation(self, drag: Drag) -> dict[str, Any]:
        start, current = _point(drag.start), _point(drag.current)
        handle = next((h for h in self.handles() if h.name == drag.handle), None)
        if drag.handle != "inside" and handle is None:
            raise ValueError("unknown transform handle")
        if handle is not None and not handle.enabled:
            raise ValueError("3D: " + handle.reason)
        dx, dy = current[0]-start[0], current[1]-start[1]
        if drag.handle == "inside":
            return {"kind": "move", "dx": dx, "dy": dy}
        if drag.handle == "pivot":
            pivot = self.pivot()
            return {"kind": "set_pivot", "pivot": (pivot[0]+dx, pivot[1]+dy)}
        pivot = self.pivot()
        if drag.handle.startswith("rotate_"):
            before = math.atan2(start[1]-pivot[1], start[0]-pivot[0])
            after = math.atan2(current[1]-pivot[1], current[0]-pivot[0])
            delta = (math.degrees(before-after)+180) % 360 - 180
            if drag.shift:
                angle = next(s.transform.frz for s in self._states() if s.member == self.reference)
                delta = round((angle+delta)/15)*15-angle
            return {"kind": "rotate", "delta": delta, "pivot": pivot}
        if drag.handle in _CORNERS and self.mode_3d:
            scale = _point(drag.screen_scale)
            if min(scale) <= 0:
                raise ValueError("script-to-screen scales must be positive")
            return {"kind": "rotate3d", "dfrx": dy*scale[1]*.5,
                    "dfry": dx*scale[0]*.5, "pivot": pivot}
        rect, reference = self._box()
        q = self.box()
        local_start = unproject(q, rect, *start)
        local_current = unproject(q, rect, *current)
        if local_start is None or local_current is None:
            raise ValueError("cannot drag a collapsed transform box")
        angle = reference.transform.frz
        if drag.handle in _EDGES and drag.ctrl:
            index = _EDGES.index(drag.handle)
            fixed = tuple((q[(index+2)%4][k]+q[(index+3)%4][k])/2 for k in (0, 1))
            opposite = ((rect.x0+rect.x1)/2, rect.y1 if index == 0 else rect.y0) if index%2 == 0 else (
                rect.x0 if index == 1 else rect.x1, (rect.y0+rect.y1)/2)
            axis = 1 if index%2 == 0 else 0
            denominator = local_start[axis]-opposite[axis]
            if abs(denominator) < 1e-9:
                raise ValueError("cannot shear a collapsed transform box")
            coefficient = (local_current[1-axis]-local_start[1-axis])/denominator
            return {"kind": "shear", "kx" if axis == 1 else "ky": coefficient,
                    "fixed": fixed, "frame_angle": angle}
        if drag.alt:
            fixed = pivot
            local_fixed = unproject(q, rect, *pivot)
            if local_fixed is None:
                raise ValueError("cannot scale about this pivot")
        elif drag.handle in _CORNERS:
            index = (_CORNERS.index(drag.handle)+2)%4
            fixed, local_fixed = q[index], rect.corners[index]
        else:
            index = _EDGES.index(drag.handle)
            fixed = tuple((q[(index+2)%4][k]+q[(index+3)%4][k])/2 for k in (0, 1))
            local_fixed = tuple((rect.corners[(index+2)%4][k]+rect.corners[(index+3)%4][k])/2 for k in (0, 1))
        axes = (0, 1) if drag.handle in _CORNERS else ((0,) if drag.handle in ("e", "w") else (1,))
        factors = [1.0, 1.0]
        for axis in axes:
            denominator = local_start[axis]-local_fixed[axis]
            if abs(denominator) < 1e-9:
                raise ValueError("cannot scale a collapsed transform box")
            factors[axis] = max(0.0, (local_current[axis]-local_fixed[axis])/denominator)
        if drag.shift and drag.handle in _CORNERS:
            # Use the axis with the larger relative change, preserving ratio.
            factor = max(factors, key=lambda value: abs(value-1))
            factors = [factor, factor]
        return {"kind": "scale", "sx": factors[0], "sy": factors[1],
                "fixed": fixed, "frame_angle": angle}

    def preview(self, drag: Drag) -> dict[str, dict[str, Any]]:
        return apply_group_transform(self._states(), self.operation(drag))

    def end_drag(self, drag: Drag) -> dict[str, dict[str, Any]]:
        changes = self.preview(drag)
        self._apply(changes)
        return changes

    def _apply(self, changes: Mapping[str, Mapping[str, Any]]) -> None:
        self._current = {mid: _merge(values, changes.get(mid, {})) for mid, values in self._current.items()}

    def clear(self) -> dict[str, dict[str, Any]]:
        changes = apply_group_transform(self._states(), {"kind": "clear"})
        self._apply(changes)
        return changes

    def cancel(self) -> None:
        self._current = deepcopy(self._entry)

    def changes(self) -> dict[str, dict[str, Any]]:
        out = {}
        for mid, values in self._current.items():
            entry = self._entry[mid]
            change = {key: deepcopy(values.get(key)) for key in entry.keys() | values.keys()
                      if entry.get(key) != values.get(key)}
            if change:
                out[mid] = change
        return out

    def readout(self) -> str:
        tf = next(s.transform for s in self._states() if s.member == self.reference)
        text = (f"旋转 {tf.frz:.1f}° · 缩放 {tf.fscx:g}% × {tf.fscy:g}% · "
                f"斜切 {tf.fax:.2f} / {tf.fay:.2f}")
        if self.mode_3d:
            text += f" · X {tf.frx:g}° Y {tf.fry:g}°".replace("-", "−")
        return text
