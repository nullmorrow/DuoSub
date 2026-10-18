"""Step playback timing from preview ASS events, in milliseconds (no Qt)."""

from __future__ import annotations

import math
from typing import Iterable

from subkit.ass import Event
from subkit.ass.tags import parse_block
from subkit.ass.text import tokenize
from subkit.model.document import Member


def _fade_in(text: str, duration: float) -> float:
    fade = 0.0
    for token in tokenize(text):
        for tag in token.tags:
            args = tag.args()
            value = 0.0
            if tag.name == "fad" and len(args) == 2:
                numbers = tag.float_args()
                if numbers is not None:
                    value = numbers[0]
            elif tag.name == "fade" and len(args) == 7:
                numbers = tag.float_args()
                if numbers is not None and numbers[0] > numbers[1]:
                    value = numbers[4]
            elif tag.name == "t" and len(args) in (3, 4):
                try:
                    start, end = float(args[0]), float(args[1])
                except ValueError:
                    continue
                if math.isfinite(start) and start <= 300 and any(
                    inner.name in ("alpha", "1a", "2a", "3a", "4a")
                    for inner in parse_block(args[-1])
                ):
                    value = end
            if math.isfinite(value):
                fade = max(fade, value)
    return min(fade, 2000.0, max(0.0, duration))


def fully_visible_time(gid: str, start: float, end: float, frame_ms: float, *,
                       events: Iterable[Event] = (), owners: Iterable[dict] = (),
                       members: Iterable[Member] = ()) -> float:
    """Use initial/lead-in events only; ASS event times are centiseconds."""
    rows = [(event.start * 10, event.end * 10, event.text)
            for event, owner in zip(events, owners)
            if owner.get("group") == gid and event.is_dialogue and event.end > event.start]
    if not rows:
        rows = [(member.start, member.end, member.text) for member in members
                if member.group == gid and member.source_kind == "Dialogue" and member.end > member.start]
    stop = start
    if rows:
        earliest = min(a for a, _, _ in rows)
        stop = max(a + _fade_in(text, b - a) for a, b, text in rows
                   if a < start or a <= earliest + frame_ms)
    return max(start, min(stop, end - frame_ms))
