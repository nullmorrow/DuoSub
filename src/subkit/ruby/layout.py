"""Ruby placement on one visual line: shift first, compress only when needed.

Model (inspired by JLREQ / JIS X 4051 ruby rules):

* each ruby item wants to be centred over its base (ink box of the base chars);
* it may overhang a neighbouring kana / punctuation / space by up to
  ``overhang`` ruby-character widths, but never an un-annotated kanji;
* a ruby centre drifts at most ``MAX_SHIFT_RATIO`` of its base width, so it
  never wanders over to a neighbour's base;
* neighbouring ruby must not collide: ``gap`` px apart inside a word,
  ``word_gap`` px across explicit word identities (or inferred from base
  spacing when unknown), measured on the visible extent;
* solve the minimum-squared-displacement placement under those constraints
  (ordered 1-D clustering with bounds — exact for this convex problem);
* only if the rubies cannot fit, compress — by *progressive filling*: all
  rubies of a crowded run share one common \\fscx (a word's ruby looks
  uniform), the scale is raised as far as the space allows, rubies that are
  not the bottleneck keep rising to 100%.  Rubies no wider than their base
  (a one-kana ruby) are compressed only if nothing else helps.
* if a run does not fit even at ``min_scale``, its bounds are relaxed (the
  ruby may overhang a kanji or drift further) rather than letting ruby glyphs
  collide; such items are reported as ``overlap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

INF = float("inf")


@dataclass
class Item:
    base_x0: float
    base_x1: float
    width: float               # visible width at 100% (ink + 2*outline)
    left_limit: float = -INF   # leftmost x the visible extent may reach
    right_limit: float = INF
    manual_scale: Optional[float] = None
    key: object = None
    max_scale: float = 100.0   # automatic cap, including centred word budgets when supplied
    # results
    x: float = 0.0
    scale: float = 100.0
    overlap: bool = False      # a bound had to be broken (does not fit even compressed)
    relaxed: bool = False      # internal: bounds ignored while placing
    word: object = None        # explicit word identity; None keeps the geometric fallback

    gap_width: float = 0.0      # fixed internal spacing of a composite word

    @property
    def target(self) -> float:
        return (self.base_x0 + self.base_x1) / 2.0

    def w(self, scale: Optional[float] = None) -> float:
        return (self.width - self.gap_width) * (self.scale if scale is None else scale) / 100.0 + self.gap_width


MAX_SHIFT_RATIO = 0.20  # a ruby centre may drift at most this share of its base width
WORD_GAP: float = 0.0   # extra gap between ruby of different words (set per call by layout_line)


def _pair_gap(a: Item, b: Item, gap: float) -> float:
    """Use explicit word boundaries when known, otherwise infer them from the bases."""
    if a.word is not None and b.word is not None:
        return gap if a.word == b.word else max(gap, WORD_GAP)
    touching = (b.base_x0 - a.base_x1) < 0.4 * min(a.base_x1 - a.base_x0, b.base_x1 - b.base_x0)
    return gap if touching else max(gap, WORD_GAP)


def _drift(it: Item) -> float:
    return max(4.0, MAX_SHIFT_RATIO * (it.base_x1 - it.base_x0))


def _center_bounds(it: Item, width: Optional[float] = None, *, strict: bool = False) -> tuple[float, float]:
    if it.relaxed and not strict:
        return -INF, INF
    h = (it.w() if width is None else width) / 2.0
    d = _drift(it)
    return max(it.left_limit + h, it.target - d), min(it.right_limit - h, it.target + d)


def _place(items: list[Item], gap: float) -> None:
    """Minimum squared displacement placement with separation + bound constraints."""
    clusters: list[dict] = []
    for i, it in enumerate(items):
        lo, hi = _center_bounds(it)
        clusters.append({"members": [i], "offs": [0.0], "sum": it.target, "n": 1, "lo": lo, "hi": hi})
        while True:
            c = clusters[-1]
            X = c["sum"] / c["n"]
            X = min(max(X, c["lo"]), c["hi"]) if c["lo"] <= c["hi"] else (c["lo"] + c["hi"]) / 2.0
            c["X"] = X
            if len(clusters) == 1:
                break
            p = clusters[-2]
            last = p["members"][-1]
            first = c["members"][0]
            need = items[last].w() / 2.0 + items[first].w() / 2.0 + _pair_gap(items[last], items[first], gap)
            if (p["X"] + p["offs"][-1]) + need <= X + 1e-6:
                break
            # merge c into p
            base_off = p["offs"][-1] + need
            for m, o in zip(c["members"], c["offs"]):
                p["members"].append(m)
                p["offs"].append(base_off + o)
                p["sum"] += items[m].target - (base_off + o)
                p["n"] += 1
                mlo, mhi = _center_bounds(items[m])
                p["lo"] = max(p["lo"], mlo - (base_off + o))
                p["hi"] = min(p["hi"], mhi - (base_off + o))
            clusters.pop()
    for c in clusters:
        for m, o in zip(c["members"], c["offs"]):
            items[m].x = c["X"] + o


def _violations(items: list[Item], gap: float) -> list[int]:
    bad = []
    for i, it in enumerate(items):
        h = it.w() / 2.0
        lo, hi = _center_bounds(it, strict=True)
        if it.x < lo - 0.5 or it.x > hi + 0.5:
            bad.append(i)
        if i + 1 < len(items):
            nx = items[i + 1]
            if it.x + h + _pair_gap(it, nx, gap) > nx.x - nx.w() / 2.0 + 0.5:
                bad.extend([i, i + 1])
    return sorted(set(bad))


# ------------------------------------------------------------- compression

def _segments(items: list[Item], gap: float) -> list[list[int]]:
    """Runs of items that could touch each other at their natural width."""
    segs: list[list[int]] = []
    prev_reach = -INF
    for i, it in enumerate(items):
        h = it.w() / 2.0
        d = _drift(it)
        left = max(it.left_limit, it.target - d - h)
        if segs and prev_reach + _pair_gap(items[i - 1], it, gap) > left:
            segs[-1].append(i)
        else:
            segs.append([i])
        prev_reach = min(it.right_limit, it.target + d + h)
    return segs


def _first_conflict(items: list[Item], seg: list[int], sc: dict[int, float], gap: float) -> list[int]:
    """Greedy leftmost placement; returns the chain of items that cannot fit (empty = feasible)."""
    pushed: list[bool] = []
    right = -INF
    prev: Optional[int] = None
    for n, i in enumerate(seg):
        it = items[i]
        width = it.w(sc[i])
        h = width / 2.0
        lo, hi = _center_bounds(it, width)
        by_prev = False
        if prev is not None:
            p = right + _pair_gap(items[prev], it, gap) + h
            if p > lo:
                lo, by_prev = p, True
        pushed.append(by_prev)
        if lo > hi + 1e-6:
            j = n
            while j > 0 and pushed[j]:
                j -= 1
            return seg[j : n + 1]
        right = lo + h
        prev = i
    return []


def _progressive(items: list[Item], seg: list[int], sc: dict[int, float], active: list[int], gap: float,
                 min_scale: float) -> bool:
    """Raise one common scale for ``active`` as far as feasible, freeze the bottleneck
    items, keep raising the rest.  Updates ``sc`` and returns True on success."""
    work = dict(sc)
    active = list(active)
    ceilings = {i: items[i].scale for i in active}

    def feasible(s: float, bump: Optional[int] = None) -> bool:
        trial = dict(work)
        for i in active:
            trial[i] = min(s, ceilings[i])
        if bump is not None:
            trial[bump] = min(ceilings[bump], s + 0.5)
        return not _first_conflict(items, seg, trial, gap)

    while active:
        if feasible(100.0):
            for i in active:
                work[i] = ceilings[i]
            break
        if not feasible(min_scale):
            return False
        lo, hi = min_scale, 100.0
        for _ in range(32):
            mid = (lo + hi) / 2.0
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        blocked = [i for i in active if not feasible(lo, bump=i)] or list(active)
        for i in active:
            work[i] = min(lo, ceilings[i])
        active = [i for i in active if i not in blocked]
    sc.update(work)
    return True


def _compress(items: list[Item], seg: list[int], gap: float, min_scale: float) -> None:
    sc = {i: items[i].scale for i in seg}
    if not _first_conflict(items, seg, sc, gap):
        return
    free = [i for i in seg if items[i].manual_scale is None]
    for _attempt in range(len(seg) + 1):
        free = [i for i in free if not items[i].relaxed]
        short = [i for i in free if items[i].width <= items[i].base_x1 - items[i].base_x0]
        wide = [i for i in free if i not in short]
        if _progressive(items, seg, sc, wide, gap, min_scale) or _progressive(items, seg, sc, wide + short, gap, min_scale):
            break
        # over-full even at min_scale: let the tightest chain break its bounds (overhang a
        # kanji / drift) at minimum width instead of colliding, then fit the rest.
        trial = dict(sc)
        for i in free:
            trial[i] = min_scale
        chain = _first_conflict(items, seg, trial, gap)
        if not chain:
            break
        for i in chain:
            items[i].relaxed = True
            if items[i].manual_scale is None:
                sc[i] = min_scale
    for i in seg:
        items[i].scale = sc[i]


def layout_line(items: list[Item], *, gap: float = 2.0, min_scale: float = 60.0, word_gap: float = 0.0) -> list[Item]:
    """Place (and if needed compress) the ruby items of one visual line in place."""
    global WORD_GAP
    WORD_GAP = word_gap
    items.sort(key=lambda it: it.target)
    for it in items:
        it.scale = it.manual_scale if it.manual_scale is not None else min(100.0, max(min_scale, it.max_scale))
        it.relaxed = False
        it.overlap = False
    for seg in _segments(items, gap):
        _compress(items, seg, gap, min_scale)
    _place(items, gap)
    for i in _violations(items, gap):
        items[i].overlap = True
    for it in items:
        it.relaxed = False
    return items
