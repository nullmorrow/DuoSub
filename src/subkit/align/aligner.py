"""CN rows ↔ JP phrase units alignment.

1. Estimate the JP→CN time offset (global, then piecewise per window).
2. Dynamic programming over (rows, units).  A row takes a contiguous run of
   0..K units; units may be skipped (interjections cheaply, content dearly);
   two neighbouring rows may take their runs crosswise (the A2→A1 case).
   Each choice is scored by time fit + cross-lingual semantic similarity.
3. Per-row confidence from the score margin of the local alternatives.

The DP is exact for its scoring model, so improving the alignment means
improving the features/weights — tuned against the Utena gold set.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .fragments import Unit


@dataclass
class AlignRow:
    idx: int
    start: int   # cs
    end: int
    text: str    # CN visible text


@dataclass
class Weights:
    """Tuned on Utena E01–E05, verified on E06–E10 (see _tool/eval/tune_align.py)."""
    sem: float = 8.0          # weight of (similarity - sem_base)
    sem_base: float = 0.30    # similarity regarded as "unrelated"
    time: float = 2.0         # per-unit time fit weight
    gap_scale: float = 60.0   # cs: distance at which a non-overlapping unit is fully penalised
    empty_row: float = 1.0    # penalty for a dialogue row that receives no JP
    skip_content: float = 1.0 # penalty per skipped content unit (× length weight); skipped content becomes an orphan line
    skip_interj: float = 0.05 # penalty for dropping a stand-alone filler
    interj_attach: float = 0.15  # bonus/penalty when attaching a stand-alone filler next to content
    swap: float = 1.2         # penalty for a crosswise (A2→A1) assignment
    window: float = 250.0     # cs: a unit may only go to rows within this distance
    max_units: int = 6        # max units per row
    max_swap_units: int = 3


@dataclass
class AlignResult:
    rows: list[list[int]]                 # unit indices per row
    skipped: list[int]                    # dropped/orphan unit indices
    swapped_rows: set[int] = field(default_factory=set)
    confidence: list[float] = field(default_factory=list)
    offset_cs: float = 0.0


# --------------------------------------------------------------- time warp
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


_BIN = 10  # cs per activity bin


def _activity(intervals: Sequence[tuple[float, float]], length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.float32)
    for a, b in intervals:
        i0, i1 = max(0, int(a // _BIN)), min(length, int(math.ceil(b / _BIN)))
        if i1 > i0:
            arr[i0:i1] = 1.0
    return arr


def _best_lag(cn: np.ndarray, jp: np.ndarray, lo: int, hi: int, center: int = 0) -> tuple[int, float]:
    """Lag (bins, added to JP) maximising overlap; ties go to the lag nearest ``center``."""
    best, best_lag = -1.0, center
    n = len(cn)
    for lag in range(lo, hi + 1):
        if lag >= 0:
            s = float(np.dot(cn[lag:], jp[: n - lag]))
        else:
            s = float(np.dot(cn[: n + lag], jp[-lag:]))
        s -= 1e-3 * abs(lag - center)
        if s > best:
            best, best_lag = s, lag
    return best_lag, best


def estimate_offset(rows: Sequence[AlignRow], units: Sequence[Unit], *, search: int = 1500) -> float:
    """Global offset (cs) to add to JP times, by cross-correlating speech activity."""
    if not rows or not units:
        return 0.0
    length = int(max(rows[-1].end, units[-1].end) // _BIN) + 2 * search // _BIN + 10
    cn = _activity([(r.start, r.end) for r in rows], length)
    jp = _activity([(u.start, u.end) for u in units], length)
    lag, _ = _best_lag(cn, jp, -search // _BIN, search // _BIN)
    return float(lag * _BIN)


def piecewise_offsets(rows: Sequence[AlignRow], units: Sequence[Unit], global_off: float, *,
                      window: int = 12000, search: int = 300) -> Callable[[float], float]:
    """Refine the offset per ~2-minute window (drift / differently cut releases), smoothed."""
    if not rows or not units:
        return lambda t: t + global_off
    length = int(max(rows[-1].end, units[-1].end) // _BIN) + 2 * search // _BIN + 10
    cn = _activity([(r.start, r.end) for r in rows], length)
    g = int(round(global_off / _BIN))
    centers, offs = [], []
    t = 0
    t_end = max(rows[-1].end, units[-1].end)
    while t < t_end:
        sel = [(u.start, u.end) for u in units if t <= u.start < t + window]
        if len(sel) >= 6:
            jp = _activity(sel, length)
            lag, score = _best_lag(cn, jp, g - search // _BIN, g + search // _BIN, center=g)
            if score > 0:
                centers.append(t + window / 2)
                offs.append(lag * _BIN)
        t += window
    if not centers:
        return lambda tt: tt + global_off
    sm = []
    for i in range(len(offs)):
        win = sorted(offs[max(0, i - 1) : i + 2])
        sm.append(win[len(win) // 2])
    cs = np.array(centers, dtype=float)
    os_ = np.array(sm, dtype=float)

    def warp(tt: float) -> float:
        return tt + float(np.interp(tt, cs, os_))

    return warp


# --------------------------------------------------------------- features
_CN_INTERJ = set("嗯啊哦噢诶欸誒咦唉哎哈呵哼呃嘿喂哇呜嗚唔嗨呀咿喔哟喲呦呼嘘噓噗咳哇")
_CN_INTERJ_LEAD = re.compile(r"^[\s「『“]*([嗯啊哦噢诶欸誒咦唉哎呃嘿喂哇呜嗚唔嗨呀咿喔哟喲呦呼嘘噓噗咳]|哈+|呵+|哼+)[\s，,、。！!？?…~～]")


def cn_has_interjection(text: str) -> bool:
    """True when the CN line itself carries an exclamation, so a JP filler may stay with it."""
    core = re.sub(r"[\s，。！？、…,.!?~～—\-「」『』“”]+", "", text)
    if not core:
        return False
    if _CN_INTERJ_LEAD.match(text.strip() + " "):
        return True
    return len(core) <= 3 and all(c in _CN_INTERJ for c in core)


class Aligner:
    def __init__(self, rows: Sequence[AlignRow], units: Sequence[Unit], *, weights: Weights | None = None,
                 sim_fn: Optional[Callable[[str, str], float]] = None) -> None:
        self.rows = list(rows)
        self.units = list(units)
        self.w = weights or Weights()
        self.sim_fn = sim_fn
        self._sim_cache: dict[tuple[int, int, int], float] = {}
        self.row_interj = [cn_has_interjection(r.text) for r in self.rows]

    # unit length weight: short units matter less
    def lw(self, u: Unit) -> float:
        return min(1.0, 0.35 + u.length / 8.0)

    def time_fit(self, u: Unit, r: AlignRow) -> float:
        dur = max(10.0, u.wend - u.wstart)
        ov = _overlap(u.wstart, u.wend, r.start, r.end)
        if ov > 0:
            return min(1.0, ov / dur)
        gap = r.start - u.wend if u.wend <= r.start else u.wstart - r.end
        return -min(1.0, gap / self.w.gap_scale)

    def sem(self, i: int, a: int, b: int) -> float:
        if self.sim_fn is None or b <= a:
            return self.w.sem_base
        key = (i, a, b)
        v = self._sim_cache.get(key)
        if v is None:
            jp = " ".join(self.units[k].text for k in range(a, b))
            v = self.sim_fn(self.rows[i].text, jp)
            self._sim_cache[key] = v
        return v

    def assign_score(self, i: int, a: int, b: int) -> float:
        w = self.w
        r = self.rows[i]
        if b <= a:
            return -w.empty_row
        score = 0.0
        content = sum(1 for k in range(a, b) if not self.units[k].interj)
        for k in range(a, b):
            u = self.units[k]
            if u.interj and content and not u.tags.get("shares_cue"):
                # a stand-alone filler next to real content: keep it only if the CN line also exclaims
                score += w.interj_attach if self.row_interj[i] else -w.interj_attach
            else:
                score += w.time * self.lw(u) * self.time_fit(u, r)
        if self.sim_fn is not None and content:
            # similarity of the content part only (interjections add noise)
            a2, b2 = a, b
            while a2 < b2 and self.units[a2].interj:
                a2 += 1
            while b2 > a2 and self.units[b2 - 1].interj:
                b2 -= 1
            score += w.sem * (self.sem(i, a2, b2) - w.sem_base)
        return score

    def skip_score(self, k: int) -> float:
        u = self.units[k]
        if u.interj and not u.tags.get("shares_cue"):
            return -self.w.skip_interj
        return -self.w.skip_content * self.lw(u)

    # ----------------------------------------------------------- the DP
    def run(self) -> AlignResult:
        n, m, w = len(self.rows), len(self.units), self.w
        NEG = -1e18
        # candidate unit window per row
        lo = [0] * n
        hi = [0] * n
        j0 = 0
        for i, r in enumerate(self.rows):
            while j0 < m and self.units[j0].wend < r.start - w.window:
                j0 += 1
            j1 = j0
            while j1 < m and self.units[j1].wstart <= r.end + w.window:
                j1 += 1
            lo[i], hi[i] = j0, j1
        dp = np.full((n + 1, m + 1), NEG)
        back: dict[tuple[int, int], tuple] = {}
        dp[0][0] = 0.0
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] + self.skip_score(j - 1)
            back[(0, j)] = ("skip",)
        for i in range(n):
            # skip transitions within row i boundary state are handled when processing (i, j)
            row_lo, row_hi = lo[i], hi[i]
            for j in range(0, m + 1):
                base = dp[i][j]
                if base <= NEG / 2:
                    continue
                # (a) row i gets units j..j+k
                for k in range(0, w.max_units + 1):
                    b = j + k
                    if b > m:
                        break
                    if k > 0 and (j < row_lo or b > row_hi):
                        if j < row_lo:
                            break
                        break
                    s = base + self.assign_score(i, j, b)
                    if s > dp[i + 1][b]:
                        dp[i + 1][b] = s
                        back[(i + 1, b)] = ("assign", j)
                # (c) crosswise: row i gets later run, row i+1 earlier run
                if (w.swap > 0 and i + 1 < n and self.sim_fn is not None
                        and max(lo[i], lo[i + 1]) <= j < min(hi[i], hi[i + 1])):
                    for k1 in range(1, w.max_swap_units + 1):
                        for k2 in range(1, w.max_swap_units + 1):
                            c, b = j + k1, j + k1 + k2
                            if b > m:
                                break
                            if j < max(lo[i], lo[i + 1]) or b > min(hi[i], hi[i + 1]):
                                continue
                            s = base + self.assign_score(i, c, b) + self.assign_score(i + 1, j, c) - w.swap
                            if s > dp[i + 2][b]:
                                dp[i + 2][b] = s
                                back[(i + 2, b)] = ("swap", j, c)
            # (b) skips after row i has been placed: dp[i+1][j] -> dp[i+1][j+1]
            for j in range(0, m):
                if dp[i + 1][j] <= NEG / 2:
                    continue
                s = dp[i + 1][j] + self.skip_score(j)
                if s > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = s
                    back[(i + 1, j + 1)] = ("skip",)
        # backtrack
        rows: list[list[int]] = [[] for _ in range(n)]
        skipped: list[int] = []
        swapped: set[int] = set()
        i, j = n, m
        while i > 0 or j > 0:
            op = back.get((i, j))
            if op is None:
                # unreachable state (should not happen); dump remaining units as skipped
                skipped.extend(range(0, j))
                break
            if op[0] == "skip":
                skipped.append(j - 1)
                j -= 1
            elif op[0] == "assign":
                a = op[1]
                rows[i - 1] = list(range(a, j))
                i, j = i - 1, a
            else:
                a, c = op[1], op[2]
                rows[i - 2] = list(range(c, j))
                rows[i - 1] = list(range(a, c))
                swapped.update({i - 2, i - 1})
                i, j = i - 2, a
        skipped.sort()
        res = AlignResult(rows, skipped, swapped)
        res.confidence = self._confidence(res)
        return res

    def _confidence(self, res: AlignResult) -> list[float]:
        """Boundary margins: how much worse the solution gets when one unit moves across
        the boundary to the neighbouring row (both rows re-scored) or is dropped/added."""
        n = len(res.rows)
        spans = [(r[0], r[-1] + 1) if r else None for r in res.rows]
        margins = [[] for _ in range(n)]
        skipped = set(res.skipped)
        for i in range(n):
            if spans[i] is None:
                continue
            a, b = spans[i]
            base_i = self.assign_score(i, a, b)
            # next row with units
            j = i + 1
            while j < n and spans[j] is None:
                j += 1
            if j < n and i not in res.swapped_rows:
                c, d = spans[j]
                if c == b:  # adjacent: move one unit across the boundary
                    base = base_i + self.assign_score(j, c, d)
                    alts = []
                    if b - a > 1:
                        alts.append(self.assign_score(i, a, b - 1) + self.assign_score(j, c - 1, d))
                    if d - c > 1:
                        alts.append(self.assign_score(i, a, b + 1) + self.assign_score(j, c + 1, d))
                    if alts:
                        m = base - max(alts)
                        margins[i].append(m)
                        margins[j].append(m)
            # dropping the edge units / taking a neighbouring skipped unit
            alts = []
            if b - a > 1:
                alts.append(self.assign_score(i, a + 1, b) + self.skip_score(a))
                alts.append(self.assign_score(i, a, b - 1) + self.skip_score(b - 1))
            if a - 1 in skipped:
                alts.append(self.assign_score(i, a - 1, b) - self.skip_score(a - 1))
            if b in skipped:
                alts.append(self.assign_score(i, a, b + 1) - self.skip_score(b))
            if alts:
                margins[i].append(base_i - max(alts))
        conf = []
        for i in range(n):
            if spans[i] is None:
                conf.append(0.5)
            elif not margins[i]:
                conf.append(0.95)
            else:
                m = min(margins[i])
                conf.append(1.0 / (1.0 + math.exp(-1.5 * m)))
        return conf


def realign_detached_blocks(rows: Sequence[AlignRow], units: Sequence[Unit], res: AlignResult,
                            make_aligner: Callable[[list[AlignRow], list[Unit]], "Aligner"], *,
                            pause: int = 2000, min_unmatched: float = 0.6) -> list[tuple[int, int, float]]:
    """Re-anchor JP blocks that found no partner because the two releases cut them differently.

    Example: TV captions put the next-episode preview after the ending song (26:11)
    while the BD subtitles have it at 24:42 — 90 s away, far outside the offset search.
    A block (units after a pause of ``pause`` cs or more) whose content is mostly
    unmatched is aligned again, on its own, against the CN rows that are still empty
    between the rows its neighbouring blocks took.  Returns (first unit, units placed,
    offset cs) per re-anchored block."""
    if not units:
        return []
    blocks: list[list[int]] = [[0]]
    for k in range(1, len(units)):
        if units[k].start - units[k - 1].end >= pause:
            blocks.append([])
        blocks[-1].append(k)
    unit_row = {ui: k for k, r in enumerate(res.rows) for ui in r}
    skipped = set(res.skipped)
    done: list[tuple[int, int, float]] = []
    for block in blocks:
        content = [ui for ui in block if not units[ui].interj]
        if not content or sum(ui in skipped for ui in content) < min_unmatched * len(content):
            continue
        inside = set(block)
        before = [unit_row[ui] for ui in range(block[0]) if ui in unit_row]
        after = [unit_row[ui] for ui in range(block[-1] + 1, len(units)) if ui in unit_row]
        lo_k = max(before) if before else -1
        hi_k = min(after) if after else len(rows)
        cands = [k for k in range(lo_k + 1, hi_k) if all(ui in inside for ui in res.rows[k])]
        if not cands:
            continue
        crow = [rows[k] for k in cands]
        t_end = max(max(r.end for r in crow), units[block[-1]].end)
        length = int(t_end // _BIN) + 20
        cn_act = _activity([(r.start, r.end) for r in crow], length)
        jp_act = _activity([(units[ui].start, units[ui].end) for ui in block], length)
        lag_lo = int((crow[0].start - units[block[-1]].end) // _BIN) - 50
        lag_hi = int((crow[-1].end - units[block[0]].start) // _BIN) + 50
        lag, score = _best_lag(cn_act, jp_act, lag_lo, lag_hi, center=int((crow[0].start - units[block[0]].start) // _BIN))
        if score <= 0:
            continue
        shift = lag * _BIN
        sub_units = []
        for ui in block:
            u = units[ui]
            sub_units.append(Unit(len(sub_units), u.cue, u.line, u.start, u.end, u.text, u.interj, u.speaker,
                                  u.start + shift, u.end + shift, dict(u.tags)))
        sub = make_aligner(crow, sub_units).run()
        placed = sum(1 for r in sub.rows for p in r if not sub_units[p].interj)
        if placed < 0.5 * len(content):
            continue
        for ui in block:
            if ui in unit_row:
                k = unit_row.pop(ui)
                res.rows[k] = [x for x in res.rows[k] if x != ui]
            skipped.discard(ui)
        for j, k in enumerate(cands):
            res.rows[k] = [block[p] for p in sub.rows[j]]
            for ui in res.rows[k]:
                unit_row[ui] = k
            if k < len(res.confidence) and j < len(sub.confidence):
                res.confidence[k] = sub.confidence[j]
            if j in sub.swapped_rows:
                res.swapped_rows.add(k)
        skipped.update(block[p] for p in sub.skipped)
        for ui, su in zip(block, sub_units):
            units[ui].wstart, units[ui].wend = su.wstart, su.wend
        done.append((block[0], placed, shift))
    res.skipped = sorted(skipped)
    return done


def make_sim_fn(embedder, rows: Sequence[AlignRow], units: Sequence[Unit]) -> Callable[[str, str], float]:
    """Similarity with batched pre-computation of rows and single units; spans computed lazily."""
    cache: dict[str, np.ndarray] = {}
    texts = [r.text for r in rows] + [u.text for u in units]
    vecs = embedder.embed(texts)
    for t, v in zip(texts, vecs):
        cache[t] = v

    def vec(t: str) -> np.ndarray:
        v = cache.get(t)
        if v is None:
            v = embedder.embed([t])[0]
            cache[t] = v
        return v

    def sim(cn: str, jp: str) -> float:
        return float(np.dot(vec(cn), vec(jp)))

    return sim


def prefetch_spans(embedder, aligner: Aligner, max_units: int) -> None:
    """Embed all candidate spans in one batch (much faster than lazy single calls)."""
    spans = set()
    for i, r in enumerate(aligner.rows):
        # same band as the DP
        idx = [k for k, u in enumerate(aligner.units) if u.wend >= r.start - aligner.w.window and u.wstart <= r.end + aligner.w.window]
        if not idx:
            continue
        a0, b0 = idx[0], idx[-1] + 1
        for a in range(a0, b0):
            for b in range(a + 2, min(b0, a + max_units) + 1):
                spans.add(" ".join(aligner.units[k].text for k in range(a, b)))
    if spans:
        embedder.embed(sorted(spans))
