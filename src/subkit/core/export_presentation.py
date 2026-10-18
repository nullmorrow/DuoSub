"""Release-only cleanup and file ordering, after layout and effect generation."""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..ass.text import line_tag, tokenize

ANNOTATIONS = frozenset(("ruby", "pinyin"))
ANNOTATION_ORDERS = ("end", "after_parent")


@dataclass
class CleanText:
    text: str
    spaces: int
    bounds: list[int]
    kept: list[int]


def clean_text(text: str) -> CleanText:
    """Trim literal ASCII spaces at visual line edges without touching ASS tags.

    Drawings and explicit hard spaces occupy a position; full-width spaces,
    tabs and soft-break escapes are kept. Hard line breaks are never removed.
    The boundary map is only for final output metadata, never layout input.
    """
    line = []
    lines = [line]
    visible, wrap = 0, 0
    tokens = tokenize(text)
    for token in tokens:
        if token.kind == "tags":
            value = next((tag.num() for tag in reversed(token.tags) if tag.name == "q"), None)
            if value in (0, 1, 2, 3):
                wrap = int(value)
        elif token.kind == "break" or token.kind == "softbreak" and wrap == 2:
            if token.kind == "softbreak":
                visible += 1
            line = []
            lines.append(line)
        elif token.kind == "text":
            for offset, char in enumerate(token.raw, token.start):
                line.append((offset, visible, char == " "))
                visible += 1
        elif token.kind in ("hardspace", "softbreak"):
            line.append((None, visible, False))
            visible += 1
        elif token.kind == "drawing" and token.raw:
            line.append((None, None, False))
    raw_removed, removed = set(), set()
    for line in lines:
        first, last = 0, len(line)
        while first < last and line[first][2]:
            first += 1
        while last > first and line[last - 1][2]:
            last -= 1
        for offset, index, _space in line[:first] + line[last:]:
            raw_removed.add(offset)
            removed.add(index)
    bounds = [0]
    for i in range(visible):
        bounds.append(bounds[-1] + int(i not in removed))
    # A karaoke tag is attached to the next parsed glyph, including a space.
    # Removing that glyph must consume its duration before the next one, rather
    # than attaching the same syllable to the next letter and singing it early.
    waits, pending = {}, None
    for token in tokens:
        if token.kind == "tags":
            pending = next((tag.name for tag in reversed(token.tags)
                            if tag.name in ("k", "K", "kf", "ko")), pending)
        elif token.kind == "text":
            for offset in range(token.start, token.end):
                if offset in raw_removed and pending is not None:
                    waits[offset] = "{\\" + pending + "0}"
                pending = None
        elif token.kind in ("break", "hardspace", "softbreak", "drawing"):
            pending = None
    return CleanText("".join(waits.get(i, "") if i in raw_removed else char for i, char in enumerate(text)),
                     len(removed), bounds, [i for i in range(visible) if i not in removed])


def empty_dialogue(text: str) -> bool:
    """No body/drawing; explicit hard/full-width spaces retain their intent."""
    return not any(token.kind == "hardspace"
                   or token.kind == "drawing" and token.raw.strip()
                   or token.kind == "text" and token.raw.strip(" \t\r\n")
                   for token in tokenize(text))


def _ordered_indices(owners, order):
    body = [i for i, owner in enumerate(owners) if owner.get("kind") not in ANNOTATIONS]
    annotations = [i for i, owner in enumerate(owners) if owner.get("kind") in ANNOTATIONS]
    if order == "end":
        return body + annotations
    last = {owners[i].get("member"): i for i in body if owners[i].get("kind") == "base"}
    buckets, orphans = {}, []
    for i in annotations:
        at = last.get(owners[i].get("member"))
        if at is None:
            orphans.append(i)
        else:
            buckets.setdefault(at, []).append(i)
    result = []
    for i in body:
        result.append(i)
        result.extend(buckets.get(i, ()))
    return result + orphans


def preserve_positioned_order(ass, owners) -> int:
    """Encode original compositing order where moving lines could change it.

    Only complete time-connected components whose events all have explicit
    pos/move may be split into layers: they do not participate in automatic
    collision placement. Other components keep a shared layer. Relative order
    of original layers is retained; source/cached events are never passed here.
    Only final release copies are normalized; preview and source layers stay intact.
    """
    if not any(owner.get("kind") in ANNOTATIONS for owner in owners):
        return 0
    ranks = [{index: rank for rank, index in enumerate(_ordered_indices(owners, order))}
             for order in ANNOTATION_ORDERS]
    layers = {}
    for i, event in enumerate(ass.events):
        if event.is_dialogue and event.end > event.start and not empty_dialogue(event.text):
            layers.setdefault(event.layer, []).append(i)
    protected = set()

    def positioned(event):
        for name, sizes in (("pos", (2,)), ("move", (4, 6))):
            tag = line_tag(event.text, name)
            if tag is not None:
                values = tag.float_args() or []
                if len(values) in sizes and all(math.isfinite(value) for value in values):
                    return True
        return False

    def protect(component):
        original = sorted(component)
        if not any(sorted(original, key=rank.get) != original for rank in ranks):
            return
        if not all(positioned(ass.events[i]) for i in original):
            return
        protected.update(original)

    for indices in layers.values():
        component, end = [], -1
        for i in sorted(indices, key=lambda i: (ass.events[i].start, ass.events[i].end, i)):
            event = ass.events[i]
            if component and event.start >= end:
                protect(component)
                component = []
            component.append(i)
            end = max(end, event.end) if len(component) > 1 else event.end
        if component:
            protect(component)
    if not protected:
        return 0
    # Compact keys avoid overflowing ASS's integer Layer field. A common key
    # retains collision grouping for every component that was not split.
    keys = {i: (event.layer, i + 1 if i in protected else 0)
            for i, event in enumerate(ass.events) if event.is_dialogue}
    levels = {key: rank for rank, key in enumerate(sorted(set(keys.values())))}
    changed = 0
    for i, key in keys.items():
        event = ass.events[i]
        layer = levels[key]
        changed += int(event.layer != layer)
        event.layer = layer
    return changed


def finish_release(ass, report, annotation_order: str) -> None:
    """Modify assembled copies only; retain event/owner pairing and stable order."""
    if annotation_order not in ANNOTATION_ORDERS:
        raise ValueError(f"未知注音行顺序：{annotation_order}")
    if len(ass.events) != len(report.owners):
        raise ValueError("导出事件与来源对应数量不一致")
    report.order_layer_events = getattr(report, "order_layer_events", 0) + preserve_positioned_order(ass, report.owners)
    pairs, base_maps = [], {}
    base_counts = {}
    for owner in report.owners:
        if owner.get("kind") == "base":
            mid = owner.get("member")
            base_counts[mid] = base_counts.get(mid, 0) + 1
    for event, original_owner in zip(ass.events, report.owners):
        owner = dict(original_owner)
        if event.is_dialogue:
            cleaned = clean_text(event.text)
            report.edge_spaces_removed += cleaned.spaces
            if empty_dialogue(cleaned.text):
                report.empty_events_removed += 1
                kind = owner.get("kind")
                if kind == "ruby":
                    report.ruby_lines -= 1
                elif kind == "pinyin":
                    report.pinyin_lines -= 1
                else:
                    report.events -= 1
                continue
            event.text = cleaned.text
            ids = owner.get("ids")
            if ids is not None and len(ids) == len(cleaned.bounds) - 1:
                owner["ids"] = [ids[i] for i in cleaned.kept]
            if owner.get("kind") == "base":
                if base_counts[owner.get("member")] == 1:
                    base_maps[owner.get("member")] = cleaned.bounds
                if cleaned.spaces and "replacement_ranges" in owner:
                    owner["replacement_ranges"] = [(cleaned.bounds[a], cleaned.bounds[b])
                                                    for a, b in owner["replacement_ranges"]]
        pairs.append((event, owner))
    for event, owner in pairs:
        if owner.get("kind") in ANNOTATIONS:
            bounds = base_maps.get(owner.get("member"))
            if bounds and "base_start" in owner and "base_end" in owner:
                owner["base_start"], owner["base_end"] = bounds[owner["base_start"]], bounds[owner["base_end"]]
    ordered = [pairs[i] for i in _ordered_indices([owner for _event, owner in pairs], annotation_order)]
    ass.events = [event for event, _owner in ordered]
    report.owners = [owner for _event, owner in ordered]
    report.annotation_order = annotation_order
