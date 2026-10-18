"""Static ASS transform geometry, in script coordinates (no Qt).

An ink box is not a layout box. Measurer retains fscx/fscy, drops text
rotation/shear, and still renders the style's Angle. Use measurement_copy
before measuring, and pass the scales already present in the resulting box.

Uniform text on one visual line has an affine 2D transform. Shears need the
layout left/top, not the ink left/top; each hard break starts a new frame.
A quad encloses transformed ink; it does not describe the glyph contours.
Mixed font metrics / local transform changes need separate glyph frames.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Protocol, Sequence

from ..ass.document import AssDocument, Event, Style
from ..ass.state import check_raw
from ..ass.tags import canonical, parse_block
from ..ass.text import remove_tags_named, tokenize

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


class InkBox(Protocol):
    x0: float
    y0: float
    x1: float
    y1: float


BoxLike = tuple[float, float, float, float] | InkBox
_NUMERIC = ("frz", "frx", "fry", "fscx", "fscy", "fax", "fay")
_UNROTATED = {"fr", "frz", "frx", "fry", "fax", "fay"}
_NUMBER = re.compile(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


@dataclass(frozen=True)
class LineTransform:
    frz: float = 0.0
    frx: float = 0.0
    fry: float = 0.0
    fscx: float = 100.0
    fscy: float = 100.0
    fax: float = 0.0
    fay: float = 0.0
    org: Optional[Point] = None
    animated: bool = False


def _defaults(style: Style) -> dict[str, float]:
    return {"frz": style.num("Angle"), "frx": 0.0, "fry": 0.0,
            "fscx": max(0.0, style.scale_x), "fscy": max(0.0, style.scale_y),
            "fax": 0.0, "fay": 0.0}


def transform_of(ass: AssDocument, event: Event) -> LineTransform:
    """Values at the first drawn character; t targets are ignored and flagged.

    Consecutive leading blocks participate, fr aliases frz, and r resets to
    the requested style. org is line-level: its first valid value wins,
    even when it appears after the first character (as in libass).
    """
    style = ass.style(event.style) or Style.new(event.style)
    defaults = _defaults(style)
    values = dict(defaults)
    org = None
    leading = True
    animated = False
    for tok in tokenize(event.text):
        if tok.kind == "tags":
            for tag in tok.tags:
                name = canonical(tag.name)
                if name == "org" and org is None:
                    args = tag.float_args()
                    if args is not None and len(args) == 2 and all(math.isfinite(a) for a in args):
                        org = (args[0], args[1])
                if name == "t":
                    animated = True
                if not leading:
                    continue
                if name == "r":
                    reset = ass.style(tag.arg.strip()) if tag.arg.strip() else style
                    defaults = _defaults(reset or style)
                    values = dict(defaults)
                elif name in _NUMERIC:
                    # libass accepts .5 and 1e-1; Tag.num's narrower grammar
                    # is used elsewhere and must not silently change here.
                    match = _NUMBER.match(tag.arg)
                    number = float(match[1]) if match else defaults[name]
                    if math.isfinite(number):
                        values[name] = max(0.0, number) if name in ("fscx", "fscy") else number
        elif tok.kind in ("text", "hardspace", "softbreak", "drawing"):
            leading = False
    return LineTransform(**values, org=org, animated=animated)


def measurement_copy(ass: AssDocument, event: Event) -> tuple[AssDocument, Event]:
    """A private measurement document/event with rotation/shear neutralised.

    Scale and font tags remain: Measurer already measures them. Neither the
    source event nor the source styles are edited. Animation is not sampled.
    """
    # Other episode events are discarded; never deep-copy them for one probe.
    doc = copy.deepcopy(ass, {id(ass.events): []})
    for style in doc.styles:
        style.set("Angle", 0)
    ev = event.copy(text=remove_tags_named(event.text, _UNROTATED))
    doc.events = [ev]
    return doc, ev


def _corners(box: BoxLike) -> Quad:
    if hasattr(box, "x0"):
        x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
    else:
        x0, y0, x1, y1 = box
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 < x0 or y1 < y0:
        raise ValueError("invalid ink box")
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def line_quad(ink_box: BoxLike, anchor: Point, transform: LineTransform, *,
              layout_origin: Optional[Point] = None,
              measured_scale: Point = (100.0, 100.0),
              perspective_distance: float = 312.5) -> Quad:
    """Transform TL/TR/BR/BL corners of one visual line's unrotated ink box.

    anchor is the actual an/pos/move-start alignment point, also the default
    rotation origin. layout_origin is that visual line's layout left/top,
    in the SAME already-scaled coordinates as ink_box. It is mandatory for
    shear: an ink box cannot reveal the advance width or font ascent.
    measured_scale identifies fscx/fscy already baked into the box (including
    style scales); passing transform's scales avoids applying them twice.

    libass applies independent x/y shear in font coordinates, then x/y scale,
    then Z, X, Y rotation and perspective (libass calc_transform_matrix).
    Positive frz is counterclockwise on the screen. perspective_distance
    is in script pixels: 312.5 * PlayResY / layout_height, where layout_height
    is libass's LayoutResY, else the video's storage height, else PlayResY.
    The default assumes matching script/layout sizes and square pixels.
    A box crossing the camera plane cannot be represented by a finite quad.
    """
    if (transform.fax or transform.fay) and layout_origin is None:
        raise ValueError("shear requires a measured layout origin, not the ink corner")
    mx, my = measured_scale
    numbers = (*anchor, mx, my, transform.frz, transform.frx, transform.fry, perspective_distance,
               transform.fscx, transform.fscy,
               transform.fax, transform.fay, *(layout_origin or anchor), *(transform.org or anchor))
    if not all(math.isfinite(v) for v in numbers) or mx <= 0 or my <= 0:
        raise ValueError("geometry and measured scales must be finite; measured scales must be positive")
    if perspective_distance <= 0:
        raise ValueError("3D: perspective distance must be positive")
    ax, ay = anchor
    left, top = layout_origin or anchor
    ox, oy = transform.org or anchor
    sx, sy = transform.fscx / mx, transform.fscy / my
    angle = math.radians(transform.frz)
    cosine, sine = math.cos(angle), math.sin(angle)
    rx, ry = math.radians(transform.frx), math.radians(transform.fry)
    cx, sinx, cy, siny = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    out: list[Point] = []
    for x, y in _corners(ink_box):
        # Both shears read the original coordinates: no fax*fay cross term.
        tx = ax + sx * (x - ax) + transform.fax * transform.fscx / my * (y - top)
        ty = ay + sy * (y - ay) + transform.fay * transform.fscy / mx * (x - left)
        dx, dy = tx - ox, ty - oy
        xz, yz = cosine * dx + sine * dy, -sine * dx + cosine * dy
        if transform.frx or transform.fry:
            # libass uses 26.6 coordinates and a 20000-unit camera distance.
            depth = xz * siny - yz * sinx * cy
            denominator = perspective_distance + depth
            if denominator <= 0:
                raise ValueError("3D: ink box crosses the perspective plane")
            projection = perspective_distance / denominator
            xz, yz = (xz * cy + yz * sinx * siny) * projection, yz * cx * projection
        out.append((ox + xz, oy + yz))
    return out[0], out[1], out[2], out[3]


def line_quads(ink_boxes: Sequence[BoxLike], anchor: Point, transform: LineTransform, *,
               layout_origins: Sequence[Point], measured_scale: Point = (100.0, 100.0),
               perspective_distance: float = 312.5) -> list[Quad]:
    """One quad per visual line; N resets shear, not the shared rotation origin.

    Multi-line/sheared ink generally has no exact four-corner outline. Keep
    these frames separate instead of shearing the union of all ink boxes.
    """
    if len(ink_boxes) != len(layout_origins):
        raise ValueError("each visual line needs its own layout origin")
    return [line_quad(box, anchor, transform, layout_origin=origin, measured_scale=measured_scale,
                      perspective_distance=perspective_distance)
            for box, origin in zip(ink_boxes, layout_origins)]


@dataclass(frozen=True)
class TransformState:
    """A member snapshot, using the exported event and LayoutIndex.anchor.

    layout_origin is the unrotated, already-scaled layout left/top of one
    visual line (required for shear). Keep baseline_overrides from tool entry
    across subsequent snapshots so reset restores the exact original export.
    source_tags: the transform tags the member's source text has anywhere
    (source_transform_tags); a value equal to the style's is only left out
    when the source has no tag of that name, because leaving it out lets the
    source tag back into the export.
    The input mappings and source event are never modified.
    """
    member: str
    transform: LineTransform
    style: Style
    anchor: Mapping[str, Any]
    overrides: Mapping[str, Any]
    layout_origin: Optional[Point] = None
    baseline_overrides: Optional[Mapping[str, Any]] = None
    source_tags: frozenset[str] = frozenset()


_EDITABLE = ("frz", "frx", "fry", "fscx", "fscy", "fax", "fay")
_CHANGE_KEYS = ("an", "pos", "move", "raw")


def source_transform_tags(text: str) -> frozenset[str]:
    """Editable transform tags and \\org (canonical names) anywhere in a source text, outside \\t animations."""
    return frozenset(canonical(tag.name) for tok in tokenize(text) if tok.kind == "tags"
                     for tag in tok.tags if canonical(tag.name) in (*_EDITABLE, "org"))


def _number(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        raise ValueError("transform values must be finite")
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _point(value: Sequence[float]) -> Point:
    if len(value) != 2:
        raise ValueError("a point needs two coordinates")
    return tuple(float(_number(float(v))) for v in value)


def _rotate_vector(point: Point, angle: float) -> Point:
    radians = math.radians(angle)
    c, s = math.cos(radians), math.sin(radians)
    x, y = point
    return c * x + s * y, -s * x + c * y


def _about(point: Point, origin: Point, angle: float) -> Point:
    x, y = _rotate_vector((point[0] - origin[0], point[1] - origin[1]), angle)
    return origin[0] + x, origin[1] + y


def _repivot(anchor: Point, old: Point, new: Point, angle: float) -> Point:
    # R*A + (I-R)*org must stay constant when only the origin changes.
    d = old[0] - new[0], old[1] - new[1]
    x, y = _rotate_vector(d, -angle)
    return anchor[0] + x - d[0], anchor[1] + y - d[1]


def _merge_raw(raw: str, old: LineTransform, transform: LineTransform, style: Style,
               source_tags: frozenset[str] = frozenset(), *, clear: bool = False) -> str:
    normal, error = check_raw(raw, span=False)
    if error:
        raise ValueError(error)
    changed = {name for name in _EDITABLE
               if not math.isclose(getattr(old, name), getattr(transform, name), abs_tol=1e-9, rel_tol=0)}
    if old.org != transform.org:
        changed.add("org")
    if clear:
        # Clear also removes redundant raw tags and neutralises source tags
        # after the first character, which the static transform cannot see.
        changed.update((*_EDITABLE, "org"))
    # t is one opaque tag: animation targets and all other raw bytes survive; a transform tag the operation did not
    # change stays exactly as typed (rewriting it could override a source tag elsewhere in the line).
    kept = [tag.raw for tag in parse_block(normal)
            if canonical(tag.name) not in changed]
    defaults = _defaults(style)
    for name in _EDITABLE:
        if name not in changed:
            continue
        value = getattr(transform, name)
        digits = 2 if name in ("fax", "fay") else 1
        number = _number(value, digits)
        if name not in source_tags and (math.isclose(value, defaults[name], abs_tol=1e-9, rel_tol=0)
                                        or float(number) == defaults[name]):
            continue
        kept.append(f"\\{name}{number}")
    if "org" in changed and transform.org is not None:
        x, y = transform.org
        kept.append(f"\\org({_number(x)},{_number(y)})")
    result, error = check_raw("".join(kept), span=False)
    if error:
        raise ValueError(error)
    return result


def apply_group_transform(states: Sequence[TransformState], op: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return Editor.set_overrides changes from immutable member snapshots.

    op['kind']: move(dx,dy), rotate(delta,pivot), set_pivot(pivot),
    scale(sx,sy,fixed,frame_angle=0), shear(kx OR ky,fixed,frame_angle=0),
    rotate3d(dfrx,dfry,pivot), clear (style transforms, positions kept), or reset (entry snapshot).
    Scale factors multiply each member's own axes; the anchors in
    the drawn frame follow the common affine. Nonuniform scaling of members
    with different rotations therefore is not a rigid screen-space affine.
    Shear is a screen-space coefficient in the stated frame, composed with
    existing scale/shear; it needs one visual line's layout origin. Multi-line
    ASS resets shear at each hard break, so it cannot shear the whole block
    as a single rigid frame. Existing 3D members keep their origin during
    rotate/rotate3d (the requested pivot is ignored); move/clear/reset are allowed,
    while set_pivot/scale/shear raise a ValueError beginning with '3D:'.
    """
    kind = op["kind"]
    if kind not in {"move", "rotate", "rotate3d", "set_pivot", "scale", "shear", "clear", "reset"}:
        raise ValueError(f"unknown transform operation: {kind}")
    fixed = _point(op["fixed"]) if kind in ("scale", "shear") else (0.0, 0.0)
    pivot = _point(op["pivot"]) if kind in ("rotate", "rotate3d", "set_pivot") else (0.0, 0.0)
    angle = float(op.get("frame_angle", 0.0))
    dx, dy = float(op.get("dx", 0.0)), float(op.get("dy", 0.0))
    delta = float(op.get("delta", 0.0))
    dfrx, dfry = float(op.get("dfrx", 0.0)), float(op.get("dfry", 0.0))
    sx, sy = float(op.get("sx", 1.0)), float(op.get("sy", 1.0))
    kx, ky = float(op.get("kx", 0.0)), float(op.get("ky", 0.0))
    if not all(math.isfinite(v) for v in (angle, dx, dy, delta, dfrx, dfry, sx, sy, kx, ky)):
        raise ValueError("transform values must be finite")
    if kind == "scale" and (sx < 0 or sy < 0):
        raise ValueError("ASS scales cannot be negative")
    if kind == "shear" and "kx" in op and "ky" in op:
        raise ValueError("shear changes one axis at a time")

    def affine(point: Point) -> Point:
        x, y = _rotate_vector((point[0] - fixed[0], point[1] - fixed[1]), -angle)
        if kind == "scale":
            x, y = sx * x, sy * y
        else:
            x, y = x + kx * y, y + ky * x
        x, y = _rotate_vector((x, y), angle)
        return fixed[0] + x, fixed[1] + y

    changes: dict[str, dict[str, Any]] = {}
    for state in states:
        if state.member in changes:
            raise ValueError("duplicate transform member")
        if kind == "reset":
            baseline = state.overrides if state.baseline_overrides is None else state.baseline_overrides
            raw, error = check_raw(str(baseline.get("raw") or ""), span=False)
            if error:
                raise ValueError(error)
            # Do not normalise the snapshot: reset must be byte-identical.
            change = {key: baseline.get(key) for key in _CHANGE_KEYS}
        elif kind == "clear":
            new = LineTransform(**_defaults(state.style), animated=state.transform.animated)
            if "org" in state.source_tags and (new.frz or new.frx or new.fry):
                # still rotated (the style's Angle): an \org at the anchor (libass's default) keeps the source's own
                # \org out of the export; without any rotation the source \org has no visible effect
                new = replace(new, org=(float(state.anchor["x"]), float(state.anchor["y"])))
            raw = _merge_raw(str(state.overrides.get("raw") or ""), state.transform, new,
                             state.style, state.source_tags, clear=True)
            change = {"raw": raw or None}
        else:
            tf = state.transform
            if not all(math.isfinite(getattr(tf, name)) for name in _NUMERIC):
                raise ValueError("transform values must be finite")
            if tf.fscx < 0 or tf.fscy < 0:
                raise ValueError("ASS scales cannot be negative")
            existing_3d = bool(tf.frx or tf.fry)
            if existing_3d and kind in {"set_pivot", "scale", "shear"}:
                raise ValueError(f"3D: {kind} cannot compensate an existing perspective transform")
            if tf.org is not None:
                _point(tf.org)
            anchor = float(state.anchor["x"]), float(state.anchor["y"])
            _point(anchor)
            move = state.anchor.get("move")
            if move is not None and len(move) not in (4, 6):
                raise ValueError("move needs four coordinates and optional start/end times")
            points = [anchor] if move is None else [(float(move[0]), float(move[1])),
                                                       (float(move[2]), float(move[3]))]
            for point in points:
                _point(point)
            new = tf
            if kind == "move":
                positions = [(x + dx, y + dy) for x, y in points]
                if tf.org is not None:
                    new = replace(tf, org=_point((tf.org[0] + dx, tf.org[1] + dy)))
            elif kind in ("rotate", "rotate3d", "set_pivot"):
                # Re-pivot only while the old geometry is still affine.
                positions = (points if existing_3d else
                             [_repivot(p, tf.org or p, pivot, tf.frz) for p in points])
                new = replace(tf, frz=tf.frz + (delta if kind == "rotate" else 0),
                              frx=tf.frx + (dfrx if kind == "rotate3d" else 0),
                              fry=tf.fry + (dfry if kind == "rotate3d" else 0),
                              org=tf.org if existing_3d else pivot)
            else:
                org = affine(tf.org) if tf.org is not None else None
                if kind == "scale":
                    new = replace(tf, fscx=tf.fscx * sx, fscy=tf.fscy * sy, org=org)
                    positions = []
                    for p in points:
                        drawn = _about(p, tf.org or p, tf.frz)
                        target = affine(drawn)
                        positions.append(_about(target, org, -tf.frz) if org is not None else target)
                else:
                    if state.layout_origin is None:
                        raise ValueError("shear requires a measured layout origin")
                    if not tf.fscx or not tf.fscy:
                        raise ValueError("cannot shear a zero-scale line")
                    layout = state.layout_origin
                    _point(layout)
                    # Compose the screen affine in the member's font frame.
                    def local(v: Point) -> Point:
                        screen = _rotate_vector(v, tf.frz - angle)
                        screen = screen[0] + kx * screen[1], screen[1] + ky * screen[0]
                        return _rotate_vector(screen, angle - tf.frz)
                    column_x = local((tf.fscx, tf.fscy * tf.fay))
                    column_y = local((tf.fscx * tf.fax, tf.fscy))
                    ax, by = column_x[0], column_y[1]
                    if ax <= 0 or by <= 0:
                        raise ValueError("shear would require unsupported negative/zero ASS scales")
                    new = replace(tf, fscx=ax, fscy=by, fax=column_y[0] / ax,
                                  fay=column_x[1] / by, org=org)
                    offset = layout[0] - anchor[0], layout[1] - anchor[1]
                    new_offset = offset[0] * ax / tf.fscx, offset[1] * by / tf.fscy
                    positions = []
                    for p in points:
                        old_layout = p[0] + offset[0], p[1] + offset[1]
                        target = affine(_about(old_layout, tf.org or p, tf.frz))
                        new_origin = org if org is not None else affine(p)
                        new_layout = _about(target, new_origin, -tf.frz)
                        # With implicit org, org follows the compensated anchor.
                        if org is None:
                            correction = _rotate_vector(new_offset, tf.frz)
                            positions.append((target[0] - correction[0], target[1] - correction[1]))
                        else:
                            positions.append((new_layout[0] - new_offset[0], new_layout[1] - new_offset[1]))
            raw = _merge_raw(str(state.overrides.get("raw") or ""), tf, new, state.style, state.source_tags)
            an = int(state.anchor["an"])
            if not 1 <= an <= 9:
                raise ValueError("invalid ASS alignment")
            change = {"an": an, "raw": raw or None, "pos": None, "move": None}
            if move is None:
                change["pos"] = list(_point(positions[0]))
            else:
                change["move"] = [*_point(positions[0]), *_point(positions[1]), *move[4:]]
        changes[state.member] = {k: v for k, v in change.items() if state.overrides.get(k) != v}
    return changes
