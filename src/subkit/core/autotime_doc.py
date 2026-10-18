"""Pure auto-timing previews over the working document and immutable sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .. import autotime
from ..ass.tags import POSITION_TAGS, canonical, parse_block
from ..ass.text import tokenize
from ..audio import AudioFeatures
from ..model.document import Document, Group, Member
from ..model.source import Source, SourceEvent

MIN_SHIFT_MS = 40               # about one frame; smaller map changes leave a group untouched


@dataclass(frozen=True)
class Settings:
    reference: str = "audio"
    method: str = "piecewise_rate"
    snap: str = "off"
    lead_in_ms: int = autotime.SNAP_LEAD_IN_MS
    lead_out_ms: int = autotime.SNAP_LEAD_OUT_MS
    include_protected: bool = False
    include_low_confidence: bool = False

    def __post_init__(self) -> None:
        if self.reference not in ("audio", "original", "translation"):
            raise ValueError("unknown timing reference")
        if self.method not in autotime.METHODS or self.snap not in ("off", "start", "both", "keep_duration"):
            raise ValueError("unknown timing method or snap mode")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
               for v in (self.lead_in_ms, self.lead_out_ms)):
            raise ValueError("expected nonnegative integer lead times")


@dataclass(frozen=True)
class MemberTimeChange:
    member_id: str
    before_start: int
    before_end: int
    start: int
    end: int


@dataclass(frozen=True)
class GroupTimeChange:
    group_id: str
    members: tuple[MemberTimeChange, ...]


@dataclass(frozen=True)
class ExcludedGroup:
    group_id: str
    reason: str                     # protected | low_confidence | hidden


@dataclass(frozen=True)
class TimingHint:
    max_offset_ms: float
    confidence: str

    @property
    def show(self) -> bool:
        return self.max_offset_ms >= 150 and self.confidence != "low"


@dataclass(frozen=True)
class AutoTimePlan:
    episode: str
    revision: int
    settings: Settings
    mapping: autotime.TimingMap
    changes: tuple[GroupTimeChange, ...]
    excluded: tuple[ExcludedGroup, ...]
    quality: autotime.TimingQuality
    hint: TimingHint

    @property
    def segments(self) -> list[autotime.Segment]:
        return self.mapping.segments

    @property
    def group_count(self) -> int:
        return len(self.changes)

    @property
    def member_count(self) -> int:
        return sum(len(g.members) for g in self.changes)

    @property
    def protected_count(self) -> int:
        return sum(g.reason == "protected" for g in self.excluded)

    @property
    def low_confidence_count(self) -> int:
        return sum(g.reason == "low_confidence" for g in self.excluded)


def _events(member: Member, sources: Mapping[str, Source]) -> list[SourceEvent]:
    return [sources[sid].events[e] for sid, e in sorted(
        {(r.src, r.e) for r in member.runs if r.src in sources and r.e is not None})
        if 0 <= e < len(sources[sid].events)]


def _positioned(member: Member, sources: Mapping[str, Source]) -> bool:
    if any(member.overrides.get(k) is not None for k in POSITION_TAGS):
        return True
    if any(canonical(t.name) in POSITION_TAGS for t in parse_block(str(member.overrides.get("raw") or ""))):
        return True
    if member.overrides.get("source_position_off"):
        return False
    texts = [member.text]
    texts.extend(e.text for e in _events(member, sources))
    return any(canonical(tag.name) in POSITION_TAGS
               for text in texts for tok in tokenize(text) if tok.kind == "tags" for tag in tok.tags)


def speech_member(group: Group, member: Member, sources: Mapping[str, Source]) -> bool:
    return ("hidden" not in group.flags and group.kind == "dialogue" and member.role in ("dialogue", "top", "alt")
            and member.source_kind != "Comment" and member.end > member.start
            and any(r.kind == "text" and r.text.strip() for r in member.runs)
            and not _positioned(member, sources))


def estimation_intervals(doc: Document, sources: Mapping[str, Source]) -> list[tuple[int, int]]:
    return [(m.start, m.end) for g in doc.groups for m in doc.members_of(g)
            if speech_member(g, m, sources)]


def _clamp_neighbours(doc: Document, mapped: dict[str, list[tuple[int, int]]],
                      planned: dict[str, list[tuple[int, int]]]) -> None:
    """Clamp snap expansions against chronological dialogue/song neighbours."""
    groups = [g for g in doc.groups if "hidden" not in g.flags and g.kind in ("dialogue", "song") and g.id in mapped]
    groups.sort(key=lambda g: min(s for s, e in mapped[g.id]))
    # An overlapping short group must not hide a longer group's next neighbour.
    while True:
        before = {g.id: tuple(planned[g.id]) for g in groups}
        for index, previous in enumerate(groups):
            for following in groups[index + 1:]:
                left, right = planned[previous.id], planned[following.id]
                base_left, base_right = mapped[previous.id], mapped[following.id]
                base_end = max(e for s, e in base_left)
                base_start = min(s for s, e in base_right)
                if base_end > base_start:
                    continue                     # preserve mapped simultaneous speech
                right_start = min(s for s, e in right)
                if max(e for s, e in left) <= right_start:
                    continue
                for i, (start, end) in enumerate(left):
                    bound = max(base_left[i][1], right_start)
                    if end > bound:
                        left[i] = (start, bound) if start < bound else base_left[i]
                left_end = max(e for s, e in left)
                for i, (start, end) in enumerate(right):
                    bound = min(base_right[i][0], left_end)
                    if start < bound:
                        right[i] = (bound, end) if bound < end else base_right[i]
        if all(tuple(planned[g.id]) == before[g.id] for g in groups):
            break


def plan(doc: Document, sources: Mapping[str, Source], features: AudioFeatures | None,
         settings: Settings = Settings(), *, mapping_override: autotime.TimingMap | None = None,
         cancel_check: Callable[[], None] | None = None) -> AutoTimePlan:
    """Fit one map for all members, leaving source events and the document untouched."""
    check = cancel_check or (lambda: None)
    check()
    intervals = estimation_intervals(doc, sources)
    if mapping_override is not None:
        mapping = mapping_override
    elif settings.reference == "audio":
        if features is None:
            raise ValueError("audio features are required")
        mapping = autotime.estimate(features, intervals, method=settings.method)
    else:
        samples = []
        for g in doc.groups:
            if "hidden" in g.flags:
                continue
            for m in doc.members_of(g, lane=settings.reference):
                events = _events(m, sources)
                if events and m.role != "comment":
                    samples.extend(((m.start, min(e.start for e in events)),
                                    (m.end, max(e.end for e in events))))
        mapping = autotime.reference_map(samples, method=settings.method, intervals=intervals,
                                        duration_ms=max([m.end for m in doc.members.values()] + [1]))
    check()
    measured = autotime.quality(mapping)
    confidence = ("low" if any(s.confidence == "low" for s in mapping.segments)
                  else "medium" if any(s.confidence == "medium" for s in mapping.segments) else "high")
    changes, excluded = [], []
    mapped_groups, planned_groups = {}, {}
    for g in doc.groups:
        check()
        if "hidden" in g.flags:
            excluded.append(ExcludedGroup(g.id, "hidden"))
            continue
        members = doc.members_of(g)
        if not members:
            continue
        mapped_groups[g.id] = [(m.start, m.end) for m in members]
        planned_groups[g.id] = list(mapped_groups[g.id])
        if not settings.include_protected and (g.state == "confirmed" or any(
                m.extra.get("time_manual") for m in members)):
            excluded.append(ExcludedGroup(g.id, "protected"))
            continue
        start, end = doc.group_time(g)
        relevant = [s for s in mapping.segments if s.start_ms < end and s.end_ms > start]
        if not relevant:
            relevant = [mapping.segments[0] if end < mapping.segments[0].start_ms else mapping.segments[-1]]
        if not settings.include_low_confidence and any(s.confidence == "low" for s in relevant):
            excluded.append(ExcludedGroup(g.id, "low_confidence"))
            continue
        times = [(mapping.map(m.start), mapping.map(m.end)) for m in members]
        # A release cut inside a line must not produce an inverted duration.
        if any(e < s for s, e in times):
            excluded.append(ExcludedGroup(g.id, "low_confidence"))
            continue
        mapped_groups[g.id] = list(times)
        if settings.snap != "off":
            if features is None:
                raise ValueError("audio features are required for snapping")
            eligible = [i for i, m in enumerate(members) if speech_member(g, m, sources)
                        and times[i][1] > times[i][0]]
            mapped = list(times)
            snapped = autotime.snap_lines(features, [times[i] for i in eligible], mode=settings.snap,
                                         lead_in_ms=settings.lead_in_ms, lead_out_ms=settings.lead_out_ms)
            for i, value in zip(eligible, snapped):
                times[i] = value
            # Snapped speech must also keep its edge order relative to skipped lines.
            orders = [sorted(range(len(members)), key=lambda i: mapped[i][edge]) for edge in (0, 1)]
            while True:
                conflicts = set()
                for edge, order in enumerate(orders):
                    for a, b in zip(order, order[1:]):
                        if (times[a][edge] > times[b][edge] or mapped[a][edge] == mapped[b][edge]
                                and times[a][edge] != times[b][edge]):
                            conflicts.update((a, b))
                changed = [i for i in conflicts if times[i] != mapped[i]]
                if not changed:
                    break
                for i in changed:
                    times[i] = mapped[i]
        planned_groups[g.id] = times
    if settings.snap != "off":
        _clamp_neighbours(doc, mapped_groups, planned_groups)
    for g in doc.groups:
        check()
        if g.id not in planned_groups:
            continue
        members, times = doc.members_of(g), planned_groups[g.id]
        # Sub-frame mapping noise leaves groups untouched; deliberate snaps are kept.
        if settings.snap == "off" and all(abs(s - m.start) < MIN_SHIFT_MS and abs(e - m.end) < MIN_SHIFT_MS
                                          for m, (s, e) in zip(members, times)):
            continue
        member_changes = tuple(MemberTimeChange(m.id, m.start, m.end, s, e)
                               for m, (s, e) in zip(members, times) if (m.start, m.end) != (s, e))
        if member_changes:
            changes.append(GroupTimeChange(g.id, member_changes))
    return AutoTimePlan(doc.episode, doc.revision, settings, mapping, tuple(changes), tuple(excluded),
                        measured, TimingHint(measured.max_offset_ms, confidence))
