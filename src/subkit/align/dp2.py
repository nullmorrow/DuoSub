"""Order-preserving CN↔JP alignment (v2).

Differences from the baseline aligner (docs/AUDIT-baseline.md B1–B3, A6):

* **No swaps.**  The output is a monotone segmentation of both sequences:
  translation rows (Chinese) and original-language phrase units (Japanese) are each read in
  source order and cut into consecutive blocks.  Crossed word order
  (CN A2→A1, JP A1→A2) becomes one *joint* block (several CN rows + the JP in
  its own order) instead of a swapped assignment.
* **Nothing is dropped.**  A unit that fits no row becomes an *unpaired* block
  at its source position (it is displayed); only the optional compact filler
  mode marks stand-alone, untranslated fillers as *hidden* (still kept, as a
  same-group Comment, restorable).
* **Cut costs.**  Cutting inside a caption costs something, more at protected
  boundaries (modifier の / vocative 様); a block that takes the head of the next
  caption while leaving its rest behind ("straddling") pays extra; mixing
  speakers across captions pays extra.  This keeps sentence heads with their
  own sentence (cross-evaluation examples 9–13).

All times are milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..core.units import Unit
from .aligner import cn_has_interjection

NEG = -1e18


@dataclass
class Row:
    idx: int
    start: int
    end: int
    text: str            # visible text (tags removed, line breaks as spaces)
    e: int               # source event index
    style: str = ""


@dataclass
class Weights2:
    sem: float = 8.0
    sem_base: float = 0.30
    time: float = 2.0
    gap_scale: float = 600.0        # ms: distance at which a non-overlapping unit is fully penalised
    window: float = 2500.0          # ms: a unit may only go to rows within this distance
    empty_row: float = 1.0          # a CN row without JP
    unpaired_content: float = 1.0   # per content unit left without a CN row (× length weight)
    unpaired_filler: float = 0.05   # a filler left on its own (displayed in "keep" mode)
    interj_attach: float = 0.15     # stand-alone filler attached next to content
    cut_space: float = 0.30         # cutting a caption line at a space
    cut_punct: float = 0.20         # … after sentence punctuation inside a line
    cut_line: float = 0.15          # … between two lines of the same speaker
    cut_marker: float = 0.15        # … at an inner ≫ / (description)
    straddle: float = 0.80          # a block that takes only part of a neighbouring caption
    mix_speaker: float = 1.00       # different known speakers from different captions in one block (tuned, Utena E01–05)
    protect_scale: float = 2.0      # multiplier for protected boundaries (modifier の, vocative 様) (tuned)
    joint: float = 2.0              # per extra CN row in a joint block
    joint_min_overlap: float = 0.3  # each row of a joint block must overlap the JP span by this share (or 300 ms)
    max_units: int = 6
    max_joint_rows: int = 3
    max_joint_units: int = 8
    band: int = 40                  # DP band (units) around each row's window
    doubt_margin: float = 1.5       # flag a boundary whose best one-unit alternative scores within this margin
                                    # (Utena E01–10 with LaBSE: flags 5.3 % of rows, 35 % of them differ from the user's final)
    doubt_margin_time: float = 0.75 # same without the semantic model (flags 4.5 %, 46 % of them differ)


@dataclass
class Block:
    kind: str                       # pair | joint | unpaired | empty_row
    rows: list[int]
    units: list[int]
    hidden: bool = False            # compact mode: stand-alone filler not rendered (kept as Comment)
    score: float = 0.0
    doubts: list[str] = field(default_factory=list)


@dataclass
class AlignOutput:
    blocks: list[Block]
    offset: float = 0.0
    reanchored: list[tuple[int, int, float]] = field(default_factory=list)


# ------------------------------------------------------------------ time warp (ms)
_BIN = 100


def _activity(intervals: Sequence[tuple[float, float]], length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.float32)
    for a, b in intervals:
        i0, i1 = max(0, int(a // _BIN)), min(length, int(math.ceil(b / _BIN)))
        if i1 > i0:
            arr[i0:i1] = 1.0
    return arr


def _best_lag(cn: np.ndarray, jp: np.ndarray, lo: int, hi: int, center: int = 0) -> tuple[int, float]:
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


def estimate_offset(rows: Sequence[Row], units: Sequence[Unit], search: int = 15000) -> float:
    if not rows or not units:
        return 0.0
    length = int(max(max(r.end for r in rows), units[-1].end) // _BIN) + 2 * search // _BIN + 10
    cn = _activity([(r.start, r.end) for r in rows], length)
    jp = _activity([(u.start, u.end) for u in units], length)
    lag, _ = _best_lag(cn, jp, -search // _BIN, search // _BIN)
    return float(lag * _BIN)


def piecewise_offsets(rows: Sequence[Row], units: Sequence[Unit], global_off: float, *,
                      window: int = 120000, search: int = 3000) -> Callable[[float], float]:
    if not rows or not units:
        return lambda t: t + global_off
    t_end = max(max(r.end for r in rows), units[-1].end)
    length = int(t_end // _BIN) + 2 * search // _BIN + 10
    cn = _activity([(r.start, r.end) for r in rows], length)
    g = int(round(global_off / _BIN))
    centers, offs = [], []
    t = 0
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
        win = sorted(offs[max(0, i - 1): i + 2])
        sm.append(win[len(win) // 2])
    cs = np.array(centers, dtype=float)
    os_ = np.array(sm, dtype=float)
    return lambda tt: tt + float(np.interp(tt, cs, os_))


@dataclass
class WarpReport:
    """How the translation's times relate to the original's: offsets (ms) along the episode, jumps, drift."""
    offset: float                                         # median offset (CN − JP), ms
    segments: list[tuple[float, float, float, float]]     # (JP start, JP end, offset at start, offset at end)
    jumps: list[tuple[float, float, float]]               # (JP time, offset before, offset after)
    method: str = "track"

    def lines(self) -> list[str]:
        def t(ms: float) -> str:
            ms = max(0.0, ms)
            return f"{int(ms) // 60000}:{ms % 60000 / 1000:04.1f}"

        out = []
        for a, b, o0, o1 in self.segments:
            drift = "" if abs(o1 - o0) < 800 else f"（逐渐变到 {o1 / 1000:+.1f}s）"
            out.append(f"  {t(a)}–{t(b)}：时间差 {o0 / 1000:+.1f}s{drift}")
        for at, o0, o1 in self.jumps:
            out.append(f"  {t(at)} 处时间差突变 {o0 / 1000:+.1f}s → {o1 / 1000:+.1f}s（两个版本这里多/少了一段）")
        return out


def _ncc_all(cn: np.ndarray, win: np.ndarray, lo: int, hi: int, start: int) -> np.ndarray:
    """Pearson correlation of ``win`` (JP activity from bin ``start``) with ``cn`` at every lag in [lo, hi]."""
    n = len(win)
    a = start + lo
    b = start + hi + n
    pad_l, pad_r = max(0, -a), max(0, b - len(cn))
    seg = cn[max(0, a): min(len(cn), b)]
    if pad_l or pad_r:
        seg = np.concatenate([np.zeros(pad_l, np.float32), seg, np.zeros(pad_r, np.float32)])
    w = win - win.mean()
    wn = float(np.sqrt((w * w).sum()))
    if wn <= 1e-6:
        return np.zeros(hi - lo + 1, np.float32)
    raw = np.correlate(seg, w, mode="valid")[: hi - lo + 1]
    c1 = np.concatenate([[0.0], np.cumsum(seg, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum(seg.astype(np.float64) ** 2)])
    k = np.arange(hi - lo + 1)
    s1 = c1[k + n] - c1[k]
    s2 = c2[k + n] - c2[k]
    var = np.maximum(s2 - s1 * s1 / n, 1e-9)
    return (raw / (wn * np.sqrt(var))).astype(np.float32)


SPEED_RATIOS = (1.0, 24000 / 23976, 23976 / 24000, 25 / 24, 24 / 25, 25000 / 23976, 23976 / 25000)


def _onsets(times: Sequence[float], length: int, sigma: float = 2.0) -> np.ndarray:
    """Line start times as a smoothed impulse train (100 ms bins) — far less ambiguous than on/off activity."""
    arr = np.zeros(length, np.float32)
    for t in times:
        k = int(t // _BIN)
        if 0 <= k < length:
            arr[k] += 1.0
    x = np.arange(-8, 9)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return np.convolve(arr, (g / g.sum()).astype(np.float32), mode="same").astype(np.float32)


def track_offsets(rows: Sequence[Row], units: Sequence[Unit], *, window: int = 120000, step: int = 15000,
                  search: int = 180000, jump_cost: float = 1.6, tol: int = 300, min_lines: int = 6
                  ) -> Optional[tuple[Callable[[float], float], WarpReport]]:
    """Follow the CN − JP time difference through the episode.

    * a speed difference (frame-rate conversion: 24 ↔ 23.976, 25 ↔ 24, 25 ↔ 23.976) is detected first by trying the
      common ratios on the whole episode;
    * then, for 2-minute windows of the JP lines (every 15 s), the correlation of line start times with the CN file
      is computed for every lag within ±3 min, and a Viterbi pass over all lags follows the offset: small changes
      (≤ 0.3 s per step) are free drift, a jump costs a fixed penalty, so only consistent evidence moves it;
    * inside a run without a jump the offset is interpolated; a jump switches at the line boundary that fits both
      sides best.
    Returns None when there is too little to go on (the caller falls back to the global estimate)."""
    first: dict = {}
    for u in units:                                  # one start per JP caption (phrases inside it are estimates)
        first[u.e] = min(first.get(u.e, u.start), u.start)
    cue_starts = sorted(first.values())
    if len(rows) < min_lines or len(cue_starts) < min_lines:
        return None
    t_end = max(max(r.end for r in rows), units[-1].end)
    sb = search // _BIN
    length = int(t_end * 1.06 // _BIN) + 2 * sb + window // _BIN + 10
    cn = _onsets([r.start for r in rows], length)
    # 1. speed ratio (whole episode, ±search)
    best_ratio, best_score = 1.0, -1.0
    for ratio in SPEED_RATIOS:
        jp_all = _onsets([t * ratio for t in cue_starts], length)
        sc = float(_ncc_all(cn, jp_all[sb: length - sb], -sb, sb, sb).max()) if length > 2 * sb + 10 else 0.0
        if sc > best_score + (0.02 if ratio != 1.0 else 0.0):
            best_ratio, best_score = ratio, sc
    scaled = [t * best_ratio for t in cue_starts]
    jp = _onsets(scaled, length)
    # 2. emissions for every window and lag
    nw = window // _BIN
    starts = np.array(scaled)
    centres: list[float] = []
    emis: list[Optional[np.ndarray]] = []
    t = 0.0
    while t < t_end * best_ratio:
        i0 = int(np.searchsorted(starts, t))
        i1 = int(np.searchsorted(starts, t + window))
        b0 = int(t // _BIN)
        if i1 - i0 >= min_lines and b0 + nw <= length:
            emis.append(1.0 - _ncc_all(cn, jp[b0: b0 + nw], -sb, sb, b0))
        else:
            emis.append(None)                       # too few lines here: no evidence either way
        centres.append(t + window / 2)
        t += step
    if sum(e is not None for e in emis) < 3:
        return None
    # 3. Viterbi over all lags: stay within ±tol (drift) for free, or jump for a fixed cost
    nl = 2 * sb + 1
    d = max(1, tol // _BIN)
    from numpy.lib.stride_tricks import sliding_window_view

    cost = np.zeros(nl) if emis[0] is None else emis[0].astype(np.float64)
    move_cost = 0.03 * np.abs(np.arange(-d, d + 1))
    backs: list[np.ndarray] = []
    for e in emis[1:]:
        padded = np.concatenate([np.full(d, np.inf), cost, np.full(d, np.inf)])
        view = sliding_window_view(padded, 2 * d + 1) + move_cost   # small moves cost a little: no wandering
        k_stay = view.argmin(axis=1)
        stay = view[np.arange(nl), k_stay]
        jump_from = int(np.argmin(cost))
        jump = cost[jump_from] + jump_cost
        use_jump = jump < stay
        back = np.where(use_jump, jump_from, np.arange(nl) + k_stay - d)
        cost = np.where(use_jump, jump, stay) + (0.0 if e is None else e)
        backs.append(back)
    lag_i = int(np.argmin(cost))
    path = [lag_i]
    for back in reversed(backs):
        lag_i = int(back[lag_i])
        path.append(lag_i)
    path.reverse()
    lags = [(k - sb) * _BIN for k in path]
    # 4. runs (no jump inside) and the switch points between them
    runs: list[list[int]] = [[0]]
    for i in range(1, len(lags)):
        if abs(lags[i] - lags[i - 1]) > tol:
            runs.append([])
        runs[-1].append(i)
    cn_starts = np.array(sorted(r.start for r in rows))

    def hits(ts: list[float], lag: float) -> list[float]:
        out = []
        for x in ts:
            j = int(np.searchsorted(cn_starts, x + lag))
            near = min((abs(cn_starts[k] - (x + lag)) for k in (j - 1, j) if 0 <= k < len(cn_starts)), default=1e9)
            out.append(1.0 if near <= 400 else 0.0)
        return out

    bounds: list[float] = []
    for ra, rb in zip(runs, runs[1:]):
        la, lb = lags[ra[-1]], lags[rb[0]]
        lo_t, hi_t = centres[ra[-1]] - window / 2, centres[rb[0]] + window / 2
        ts = [x for x in scaled if lo_t <= x < hi_t]
        if not ts:
            bounds.append((centres[ra[-1]] + centres[rb[0]]) / 2)
            continue
        fa, fb = hits(ts, la), hits(ts, lb)
        best, at = -1.0, 0
        for cut in range(len(ts) + 1):
            sc = sum(fa[:cut]) + sum(fb[cut:])
            if sc > best:
                best, at = sc, cut
        bounds.append(ts[at] - 50 if at < len(ts) else ts[-1] + 50)
    run_c = [np.array([centres[i] for i in r]) for r in runs]
    run_l = [np.array([float(lags[i]) for i in r]) for r in runs]
    bnd = np.array(bounds)
    ratio = best_ratio

    def warp(tt: float) -> float:
        tau = tt * ratio
        k = int(np.searchsorted(bnd, tau)) if len(bnd) else 0
        return tau + float(np.interp(tau, run_c[k], run_l[k]))

    # report in JP time: offset(t) = warp(t) − t (includes the speed difference)
    edges = [0.0] + [x / ratio for x in bounds] + [float(t_end)]
    segs = []
    for k in range(len(runs)):                       # offsets where the run has evidence (not at the far edges)
        a = max(edges[k] + 1, run_c[k][0] / ratio)
        z = max(a, min(edges[k + 1] - 1, run_c[k][-1] / ratio))
        segs.append((edges[k], edges[k + 1], warp(a) - a, warp(z) - z))
    jumps = [(edges[k + 1], segs[k][3], segs[k + 1][2]) for k in range(len(runs) - 1)]
    med = float(np.median([warp(x) - x for x in cue_starts]))
    rep = WarpReport(med, segs, jumps, method=f"track×{ratio:.4f}" if ratio != 1.0 else "track")
    return warp, rep


def time_warp(rows: Sequence[Row], units: Sequence[Unit]) -> tuple[Callable[[float], float], float, Optional[WarpReport]]:
    """JP time → CN time.  The tracker when it finds enough to go on; otherwise the global offset with ±3 s local
    corrections (the earlier method)."""
    tr = track_offsets(rows, units)
    if tr is not None:
        warp, rep = tr
        return warp, rep.offset, rep
    off = estimate_offset(rows, units)
    return piecewise_offsets(rows, units, off), off, None


# ------------------------------------------------------------------ aligner
class Aligner2:
    def __init__(self, rows: Sequence[Row], units: Sequence[Unit], *, weights: Optional[Weights2] = None,
                 sim_fn: Optional[Callable[[str, str], float]] = None, filler_mode: str = "keep",
                 cancel_check: Callable[[], None] = lambda: None) -> None:
        self.cancel_check = cancel_check
        self.rows = list(rows)
        self.units = list(units)
        self.w = weights or Weights2()
        self.sim_fn = sim_fn
        self.filler_mode = filler_mode            # keep | compact
        self.row_interj = [cn_has_interjection(r.text) for r in self.rows]
        self._sim: dict[tuple, float] = {}
        m = len(self.units)
        boundary_cost = {"event": 0.0, "line_speaker": 0.0, "line": self.w.cut_line, "marker": self.w.cut_marker,
                         "space": self.w.cut_space, "punct": self.w.cut_punct}
        self.cut = [0.0] * (m + 1)
        for b in range(1, m):
            u = self.units[b]
            self.cut[b] = boundary_cost.get(u.boundary, 0.0) + (self.w.protect_scale * u.protect if u.boundary == "space" else 0.0)
        self.ev = [u.e for u in self.units]

    # -------------------------------------------------------------- features
    def lw(self, u: Unit) -> float:
        return min(1.0, 0.35 + u.length / 8.0)

    def time_fit(self, u: Unit, s: float, e: float) -> float:
        dur = max(100.0, u.wend - u.wstart)
        ov = max(0.0, min(u.wend, e) - max(u.wstart, s))
        if ov > 0:
            return min(1.0, ov / dur)
        gap = s - u.wend if u.wend <= s else u.wstart - e
        return -min(1.0, gap / self.w.gap_scale)

    def _content_span(self, a: int, b: int) -> tuple[int, int]:
        while a < b and self.units[a].interj:
            a += 1
        while b > a and self.units[b - 1].interj:
            b -= 1
        return a, b

    def sem(self, key: tuple, cn: str, a: int, b: int) -> float:
        if self.sim_fn is None or b <= a:
            return self.w.sem_base
        k = (key, a, b)
        v = self._sim.get(k)
        if v is None:
            jp = " ".join(self.units[x].text for x in range(a, b))
            v = self.sim_fn(cn, jp)
            self._sim[k] = v
        return v

    def _units_score(self, spans: list[tuple[float, float]], j: int, b: int, interj_ok: bool) -> float:
        """Time fit of units [j, b) against the rows' own intervals (not their hull: a unit lying in the
        gap between two rows of a joint block fits neither)."""
        w = self.w
        score = 0.0
        content = any(not self.units[k].interj for k in range(j, b))
        for k in range(j, b):
            u = self.units[k]
            if u.interj and content and not u.tags.get("shares_cue"):
                score += w.interj_attach if interj_ok else -w.interj_attach
            else:
                score += w.time * self.lw(u) * max(self.time_fit(u, s, e) for s, e in spans)
        return score

    def block_penalty(self, j: int, b: int) -> float:
        """Cut at the end + straddling + mixed speakers for a block made of units [j, b)."""
        w = self.w
        p = self.cut[b] if b < len(self.units) else 0.0
        if b - j >= 1 and self.ev[j] != self.ev[b - 1]:
            if b < len(self.units) and self.ev[b] == self.ev[b - 1]:
                p += w.straddle
            if j > 0 and self.ev[j - 1] == self.ev[j]:
                p += w.straddle
            spk = {self.units[k].speaker for k in range(j, b) if self.units[k].speaker}
            if len(spk) > 1:
                p += w.mix_speaker
        return p

    def assign_score(self, i: int, j: int, b: int) -> float:
        if b <= j:
            return -self.w.empty_row
        r = self.rows[i]
        score = self._units_score([(r.start, r.end)], j, b, self.row_interj[i])
        a2, b2 = self._content_span(j, b)
        if self.sim_fn is not None and b2 > a2:
            score += self.w.sem * (self.sem(("r", i), r.text, a2, b2) - self.w.sem_base)
        return score - self.block_penalty(j, b)

    def joint_score(self, i: int, nr: int, j: int, b: int) -> float:
        rs = self.rows[i: i + nr]
        score = self._units_score([(r.start, r.end) for r in rs], j, b, any(self.row_interj[i: i + nr]))
        a2, b2 = self._content_span(j, b)
        if self.sim_fn is not None and b2 > a2:
            cn = " ".join(r.text for r in rs)
            score += self.w.sem * (self.sem(("j", i, nr), cn, a2, b2) - self.w.sem_base)
        return score - self.w.joint * (nr - 1) - self.block_penalty(j, b)

    def unpaired_score(self, j: int, b: int) -> float:
        w = self.w
        score = 0.0
        for k in range(j, b):
            u = self.units[k]
            if u.interj and not u.tags.get("shares_cue"):
                score -= w.unpaired_filler
            else:
                score -= w.unpaired_content * self.lw(u)
        return score - self.block_penalty(j, b)

    # -------------------------------------------------------------- windows
    def _windows(self) -> tuple[list[int], list[int]]:
        n, m, w = len(self.rows), len(self.units), self.w
        lo, hi = [0] * n, [0] * n
        j0 = 0
        for i, r in enumerate(self.rows):
            while j0 < m and self.units[j0].wend < r.start - w.window:
                j0 += 1
            j1 = j0
            while j1 < m and self.units[j1].wstart <= r.end + w.window:
                j1 += 1
            lo[i], hi[i] = j0, j1
        # windows must be monotone for the band logic
        for i in range(1, n):
            lo[i] = max(lo[i], lo[i - 1]) if lo[i] < lo[i - 1] else lo[i]
            hi[i] = max(hi[i], hi[i - 1])
        return lo, hi

    def _joint_covers(self, i: int, nr: int, j: int, b: int) -> bool:
        """A joint block is only for JP that really runs across all its rows (crossed order, one long
        phrase); it must not be a way to swallow CN rows that have no JP at all."""
        js = min(self.units[k].wstart for k in range(j, b))
        je = max(self.units[k].wend for k in range(j, b))
        for r in self.rows[i: i + nr]:
            ov = max(0.0, min(je, r.end) - max(js, r.start))
            if ov < min(300.0, self.w.joint_min_overlap * max(1, r.end - r.start)):
                return False
        return True

    def collect_texts(self) -> tuple[set[str], set[str]]:
        """Dry run that records every (CN, JP) text the DP will compare, for batch embedding."""
        cn_t: set[str] = set()
        jp_t: set[str] = set()
        real, base = self.sim_fn, self.w.sem_base

        def rec(cn: str, jp: str) -> float:
            cn_t.add(cn)
            jp_t.add(jp)
            return base

        self.sim_fn, self._sim = rec, {}
        try:
            self.run()
        finally:
            self.sim_fn, self._sim = real, {}
        return cn_t, jp_t

    def _in_window(self, s: float, e: float, j: int, b: int) -> bool:
        w = self.w
        return all(self.units[k].wend >= s - w.window and self.units[k].wstart <= e + w.window for k in range(j, b))

    # -------------------------------------------------------------- DP
    def run(self) -> AlignOutput:
        n, m, w = len(self.rows), len(self.units), self.w
        dp = np.full((n + 1, m + 1), NEG)
        bk = np.zeros((n + 1, m + 1), dtype=np.int8)        # 1 assign, 2 joint, 3 unpaired
        bj = np.zeros((n + 1, m + 1), dtype=np.int32)       # previous j
        br = np.zeros((n + 1, m + 1), dtype=np.int8)        # rows consumed by a joint step
        dp[0][0] = 0.0
        lo, hi = self._windows()
        K = w.max_units

        def j_range(i: int) -> range:
            if n == 0:
                return range(0, m + 1)
            # Bridge the previous window to this one so single-sided runs
            # remain reachable, without widening every later row to the origin.
            a = 0 if i == 0 else max(0, lo[i - 1] - w.band)
            b = m if i >= n else min(m, hi[i] + w.band)
            return range(a, max(a, b) + 1)

        for i in range(n + 1):
            self.cancel_check()
            jr = j_range(i)
            # unpaired runs stay on row-state i (push in increasing j)
            for j in jr:
                base = dp[i][j]
                if base <= NEG / 2:
                    continue
                for b in range(j + 1, min(m, j + K) + 1):
                    if self.ev[b - 1] != self.ev[j]:
                        break                     # one caption per unpaired block
                    s = base + self.unpaired_score(j, b)
                    if s > dp[i][b]:
                        dp[i][b], bk[i][b], bj[i][b] = s, 3, j
            if i >= n:
                break
            r = self.rows[i]
            for j in jr:
                base = dp[i][j]
                if base <= NEG / 2:
                    continue
                # (a) row i takes units [j, b)
                for b in range(j, min(m, j + K) + 1):
                    if b > j and not self._in_window(r.start, r.end, b - 1, b):
                        break
                    s = base + self.assign_score(i, j, b)
                    if s > dp[i + 1][b]:
                        dp[i + 1][b], bk[i + 1][b], bj[i + 1][b] = s, 1, j
                # (b) joint: rows [i, i+nr) take units [j, b)
                for nr in range(2, w.max_joint_rows + 1):
                    if i + nr > n:
                        break
                    s0 = r.start
                    e0 = max(x.end for x in self.rows[i: i + nr])
                    for b in range(j + 1, min(m, j + w.max_joint_units) + 1):
                        if not self._in_window(s0, e0, b - 1, b):
                            break
                        if not self._joint_covers(i, nr, j, b):
                            continue
                        s = base + self.joint_score(i, nr, j, b)
                        if s > dp[i + nr][b]:
                            dp[i + nr][b], bk[i + nr][b], bj[i + nr][b], br[i + nr][b] = s, 2, j, nr
        # backtrack
        blocks: list[Block] = []
        i, j = n, m
        if dp[n][m] <= NEG / 2:
            return self._fallback()
        while i > 0 or j > 0:
            kind = int(bk[i][j])
            pj = int(bj[i][j])
            if kind == 3:
                blocks.append(Block("unpaired", [], list(range(pj, j))))
                j = pj
            elif kind == 1:
                blocks.append(Block("pair" if j > pj else "empty_row", [i - 1], list(range(pj, j))))
                i, j = i - 1, pj
            elif kind == 2:
                nr = int(br[i][j])
                blocks.append(Block("joint", list(range(i - nr, i)), list(range(pj, j))))
                i, j = i - nr, pj
            else:
                raise RuntimeError(f"alignment backtrack failed at ({i},{j})")
        blocks.reverse()
        self._mark(blocks)
        self._margins(blocks)
        return AlignOutput(blocks)

    def _fallback(self) -> AlignOutput:
        """Keep both source sequences intact when the search has no path."""
        blocks = []
        i = j = 0
        while i < len(self.rows) or j < len(self.units):
            self.cancel_check()
            if i < len(self.rows) and (j == len(self.units)
                                      or self.rows[i].start <= self.units[j].wstart):
                blocks.append(Block("empty_row", [i], [], doubts=["alignment_fallback"]))
                i += 1
            else:
                blocks.append(Block("unpaired", [], [j], doubts=["alignment_fallback"]))
                j += 1
        return AlignOutput(blocks)

    def _margins(self, blocks: list[Block]) -> None:
        """Close calls: moving one unit across the boundary of two neighbouring pairs scores almost as well."""
        for x in range(len(blocks) - 1):
            A, B = blocks[x], blocks[x + 1]
            if A.kind != "pair" or B.kind != "pair" or not A.units or not B.units or A.units[-1] + 1 != B.units[0]:
                continue
            i, j = A.rows[0], B.rows[0]
            a0, a1, b1 = A.units[0], B.units[0], B.units[-1] + 1
            base = self.assign_score(i, a0, a1) + self.assign_score(j, a1, b1)
            alts = []
            if a1 - a0 > 1:
                alts.append(self.assign_score(i, a0, a1 - 1) + self.assign_score(j, a1 - 1, b1))
            if b1 - a1 > 1:
                alts.append(self.assign_score(i, a0, a1 + 1) + self.assign_score(j, a1 + 1, b1))
            limit = self.w.doubt_margin if self.sim_fn is not None else self.w.doubt_margin_time
            if alts and base - max(alts) < limit:
                for bl in (A, B):
                    if "close_boundary" not in bl.doubts:
                        bl.doubts.append("close_boundary")

    # -------------------------------------------------------------- post
    def _mark(self, blocks: list[Block]) -> None:
        for bl in blocks:
            if bl.kind == "unpaired" and self.filler_mode == "compact":
                us = [self.units[k] for k in bl.units]
                if us and all(u.interj and not u.tags.get("shares_cue") for u in us):
                    bl.hidden = True
            if bl.kind in ("pair", "joint") and bl.units:
                # only cuts inside a caption *line* are doubtful; a cut between two lines of one caption
                # (often two speakers) is where the translation usually splits too
                a, b = bl.units[0], bl.units[-1] + 1
                weak = ("space", "punct", "marker")
                if (a > 0 and self.ev[a - 1] == self.ev[a] and self.units[a].boundary in weak) or                         (b < len(self.units) and self.ev[b] == self.ev[b - 1] and self.units[b].boundary in weak):
                    bl.doubts.append("cut_inside_cue")
            if bl.kind == "joint":
                bl.doubts.append("joint_rows")


def realign_detached(rows: Sequence[Row], units: Sequence[Unit], out: AlignOutput,
                     make: Callable[[list[Row], list[Unit]], Aligner2], *, pause: int = 20000,
                     min_unmatched: float = 0.6) -> None:
    """Re-anchor caption blocks after a long pause that found no partner because the two
    releases cut them differently (TV preview after a moved ending song, 90 s away)."""
    if not units:
        return
    groups: list[list[int]] = [[0]]
    for k in range(1, len(units)):
        if units[k].start - units[k - 1].end >= pause:
            groups.append([])
        groups[-1].append(k)
    for ublock in groups:
        content = [k for k in ublock if not units[k].interj]
        inside = set(ublock)
        unpaired = {k for bl in out.blocks if bl.kind == "unpaired" for k in bl.units}
        if not content or sum(k in unpaired for k in content) < min_unmatched * len(content):
            continue
        # blocks touching this unit block, and the empty rows between its neighbours
        idx = [bi for bi, bl in enumerate(out.blocks) if any(k in inside for k in bl.units)]
        if not idx:
            continue
        b0, b1 = idx[0], idx[-1]
        # extend over adjacent empty rows on both sides (rows that could be the partners)
        while b0 > 0 and out.blocks[b0 - 1].kind == "empty_row":
            b0 -= 1
        while b1 + 1 < len(out.blocks) and out.blocks[b1 + 1].kind == "empty_row":
            b1 += 1
        span = out.blocks[b0: b1 + 1]
        if any(bl.kind in ("pair", "joint") and not set(bl.units) <= inside for bl in span):
            continue
        cand_rows = [r for bl in span for r in bl.rows]
        if not cand_rows:
            continue
        crow = [rows[r] for r in cand_rows]
        length = int(max(max(r.end for r in crow), units[ublock[-1]].end) // _BIN) + 20
        cn_act = _activity([(r.start, r.end) for r in crow], length)
        jp_act = _activity([(units[k].start, units[k].end) for k in ublock], length)
        lag_lo = int((crow[0].start - units[ublock[-1]].end) // _BIN) - 50
        lag_hi = int((crow[-1].end - units[ublock[0]].start) // _BIN) + 50
        lag, score = _best_lag(cn_act, jp_act, lag_lo, lag_hi, center=int((crow[0].start - units[ublock[0]].start) // _BIN))
        if score <= 0:
            continue
        shift = lag * _BIN
        sub_units = []
        for k in ublock:
            u = units[k]
            su = Unit(**{**u.__dict__, "idx": len(sub_units), "tags": dict(u.tags)})
            su.wstart, su.wend = u.start + shift, u.end + shift
            sub_units.append(su)
        sub = make([Row(x, r.start, r.end, r.text, r.e, r.style) for x, r in enumerate(crow)], sub_units).run()
        placed = sum(1 for bl in sub.blocks if bl.kind in ("pair", "joint") for k in bl.units if not sub_units[k].interj)
        if placed < 0.5 * len(content):
            continue
        new_blocks = []
        for bl in sub.blocks:
            new_blocks.append(Block(bl.kind, [cand_rows[r] for r in bl.rows], [ublock[k] for k in bl.units],
                                    bl.hidden, bl.score, bl.doubts + ["reanchored"]))
        # unit blocks of this caption block that were outside the span stay as they were
        out.blocks[b0: b1 + 1] = new_blocks
        for k, su in zip(ublock, sub_units):
            units[k].wstart, units[k].wend = su.wstart, su.wend
        out.reanchored.append((ublock[0], placed, shift))
