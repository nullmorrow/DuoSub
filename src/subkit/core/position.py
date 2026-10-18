"""Numeric subtitle positions from measured layout snapshots (no Qt)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..ass.document import Style
from ..ass.state import check_raw
from ..ass.tags import canonical, parse_block
from ..ass.text import tokenize
from ..model.document import Member
from ..model.source import Source
from .layout_index import LayoutIndex, Rect
from .transform import LineTransform, transform_of

Changes = dict[str, dict[str, Any]]


def source_positioned(member: Member, sources: Mapping[str, Source]) -> bool:
    """Inspect whole source events, including tags omitted from provenance runs."""
    from .render_origin import source_events

    events = source_events(member)
    return any(canonical(tag.name) in {"an", "pos", "move"}
               for sid, e in events if sid in sources and 0 <= e < len(sources[sid].events)
               for tok in tokenize(sources[sid].events[e].text)
               if tok.kind == "tags" for tag in tok.tags)


@dataclass(frozen=True)
class PositionTarget:
    member: str
    lane: str
    role: str
    style: Style
    margins: Sequence[int]
    overrides: Mapping[str, Any]
    source_positioned: bool
    anchor: Mapping[str, Any]
    layout_box: Rect
    transform: LineTransform

    @classmethod
    def from_layout(cls, index: LayoutIndex, i: int, member: Member,
                    sources: Mapping[str, Source]) -> "PositionTarget":
        """Capture a base member from its current exported preview event."""
        owner = index.owners[i]
        if owner.get("kind") != "base" or owner.get("member") != member.id:
            raise ValueError("position target must match a base event")
        ev = index.ass.events[i]
        return cls(member.id, member.lane, member.role,
                   copy.deepcopy(index.ass.style(ev.style) or Style.new(ev.style)),
                   tuple(member.margins), copy.deepcopy(member.overrides), source_positioned(member, sources),
                   copy.deepcopy(index.anchor(i)), copy.deepcopy(index.layout_boxes([i])[0]),
                   transform_of(index.ass, ev))


@dataclass(frozen=True)
class PositionSkip:
    group: str
    member: Optional[str]
    reason: str


@dataclass
class VerticalResult:
    gids: list[str] = field(default_factory=list)  # groups with changed members
    members: list[str] = field(default_factory=list)
    skipped: list[PositionSkip] = field(default_factory=list)


def _number(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("position values must be finite")
    return float(f"{value:.1f}")


def _alignment(an: int) -> int:
    if isinstance(an, bool) or int(an) != an or not 1 <= an <= 9:
        raise ValueError("invalid ASS alignment")
    return int(an)


def _select(targets: Sequence[PositionTarget], scope: str,
            reference: Optional[PositionTarget | str] = None) -> tuple[list[PositionTarget], Optional[PositionTarget]]:
    if scope not in ("group", "line"):
        raise ValueError("scope must be group or line")
    targets = list(targets)
    if len({t.member for t in targets}) != len(targets):
        raise ValueError("duplicate position member")
    if reference is None:
        ref = targets[0] if targets else None
    else:
        mid = reference.member if isinstance(reference, PositionTarget) else reference
        ref = next((t for t in targets if t.member == mid), None)
        if ref is None:
            raise ValueError("reference is not a position target")
    return (targets if scope == "group" else [ref] if ref else []), ref


def _org_raw(target: PositionTarget, org: tuple[float, float]) -> str:
    raw, error = check_raw(str(target.overrides.get("raw") or ""), span=False)
    if error:
        raise ValueError(error)
    # Keep unrelated tags and opaque animations exactly as typed.
    kept = [tag.raw for tag in parse_block(raw) if canonical(tag.name) != "org"]
    x, y = (_number(v) for v in org)
    return "".join(kept) + f"\\org({x:g},{y:g})"


def _position(target: PositionTarget, an: int, dx: float, dy: float, *, move_org: bool) -> dict[str, Any]:
    an = _alignment(an)
    dx, dy = float(dx), float(dy)
    _number(dx), _number(dy)
    change: dict[str, Any] = {"an": an, "pos": None, "move": None}
    move = target.anchor.get("move")
    if move is None:
        change["pos"] = [_number(float(target.anchor["x"]) + dx),
                         _number(float(target.anchor["y"]) + dy)]
    else:
        if len(move) not in (4, 6) or not all(math.isfinite(float(v)) for v in move):
            raise ValueError("move needs four coordinates and optional start/end times")
        change["move"] = [_number(float(v) + (dx if k % 2 == 0 else dy)) for k, v in enumerate(move[:4])]
        change["move"].extend(move[4:])
    tf = target.transform
    if move_org and tf.org is not None:
        change["raw"] = _org_raw(target, (tf.org[0] + dx, tf.org[1] + dy))
    elif not move_org and tf.org is None and (tf.frz or tf.frx or tf.fry):
        # libass otherwise repivots to the new alignment point.
        change["raw"] = _org_raw(target, (float(target.anchor["x"]), float(target.anchor["y"])))
    elif not move_org and tf.org is not None:
        # Position overrides also strip source org; retain its effective value.
        raw = str(target.overrides.get("raw") or "")
        if not any(canonical(tag.name) == "org" for tag in parse_block(raw)):
            change["raw"] = _org_raw(target, tf.org)
    return {k: v for k, v in change.items() if target.overrides.get(k) != v}


def set_xy(targets: Sequence[PositionTarget], reference: PositionTarget | str,
           x: Optional[float] = None, y: Optional[float] = None, scope: str = "group") -> Changes:
    """Move all selected anchors by the reference delta; keep move times."""
    selected, ref = _select(targets, scope, reference)
    if ref is None or x is None and y is None:
        return {}
    dx = 0.0 if x is None else _number(float(x)) - float(ref.anchor["x"])
    dy = 0.0 if y is None else _number(float(y)) - float(ref.anchor["y"])
    return {t.member: _position(t, t.anchor["an"], dx, dy, move_org=True) for t in selected}


def set_alignment(targets: Sequence[PositionTarget], an: int, scope: str = "group", *,
                  reference: Optional[PositionTarget | str] = None) -> Changes:
    """Keep the layout block in place; hard-break lines re-align within it.

    For line scope, reference defaults to the first target (focused line).
    """
    an = _alignment(an)
    selected, _ = _select(targets, scope, reference)
    changes = {}
    for t in selected:
        if an == t.anchor["an"]:
            changes[t.member] = {}
            continue
        box = t.layout_box
        if not all(math.isfinite(v) for v in (box.x0, box.y0, box.x1, box.y1)) or box.x1 < box.x0 or box.y1 < box.y0:
            raise ValueError("invalid layout block box")
        x = box.x0 + (an - 1) % 3 * (box.x1 - box.x0) / 2
        y = box.y1 - (an - 1) // 3 * (box.y1 - box.y0) / 2
        changes[t.member] = _position(t, an, x - float(t.anchor["x"]), y - float(t.anchor["y"]), move_org=False)
    return changes


def preset_side(targets: Sequence[PositionTarget], side: str, playres: tuple[int, int]) -> Changes:
    """Use each member's own margin and alignment row, preserving stacking."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    selected, _ = _select(targets, "group")
    changes = {}
    for t in selected:
        row = (_alignment(t.anchor["an"]) - 1) // 3
        margin_index, field = (0, "MarginL") if side == "left" else (1, "MarginR")
        margin = t.margins[margin_index] or t.style.num(field)
        x = margin if side == "left" else playres[0] - margin
        # Only horizontal alignment changes, so the vertical anchor is identical.
        changes[t.member] = _position(t, row * 3 + (1 if side == "left" else 3),
                                      x - float(t.anchor["x"]), 0, move_org=True)
    return changes


def restore(targets: Sequence[PositionTarget]) -> Changes:
    """Expose the source/style position; leave raw and format overrides intact."""
    selected, _ = _select(targets, "group")
    return {t.member: {k: None for k in ("an", "pos", "move") if k in t.overrides} for t in selected}
