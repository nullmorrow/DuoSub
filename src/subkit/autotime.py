"""Qt-free subtitle-to-audio time maps. Times and offsets are milliseconds.

Correlation aligns subtitle occupancy with a speech-band contrast heuristic,
not a speech recognizer. Music can therefore only supply weak evidence.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, field
import json
from typing import Callable, Iterable

import numpy as np

from .audio import AudioFeatures, BAND_EDGES_HZ, HOP_MS

METHODS = ("shift", "piecewise", "piecewise_rate")
MAX_RATE_DELTA = 0.05
RATE_FAMILIES = (1.0, 25 / 23.976, 23.976 / 25, 25 / 24, 24 / 25)
RATE_TOLERANCE = 0.0015
RATE_MIN_SPAN_MS = 300000
# Two loss units equal one saturated outlier. Synthetic weak clock slopes
# cannot buy a rate; sustained 0.1% / PAL drift over an episode can.
RATE_PENALTY = 2.0
# Gankutsuou E01 finished CN and JP: median raw lag +160 ms, IQR 80 ms,
# 35 reliable windows each. This measures occupancy-vs-energy bias, not
# individual line lead-in/out. Other subtitle conventions may differ.
CORRELATION_BIAS_MS = 160
SNAP_LEAD_IN_MS = 270
SNAP_LEAD_OUT_MS = 0
HINT_SEARCH_MS = 600
HINT_RUN_ON = True


@dataclass(frozen=True)
class WindowLag:
    start_ms: int
    end_ms: int
    lag_ms: float
    score: float
    confidence: str

    @property
    def centre_ms(self) -> float:
        return (self.start_ms + self.end_ms) / 2


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int
    offset_ms: float
    rate: float = 1.0
    confidence: str = "low"
    score: float = 0.0

    def map(self, t_ms: float) -> int:
        return int(round(self.start_ms + self.offset_ms + (t_ms - self.start_ms) * self.rate))


@dataclass
class TimingMap:
    segments: list[Segment]
    method: str = "piecewise_rate"
    windows: list[WindowLag] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.method not in METHODS or not self.segments:
            raise ValueError("invalid timing map")
        for i, s in enumerate(self.segments):
            if (s.end_ms <= s.start_ms or not np.isfinite(s.offset_ms)
                    or not np.isfinite(s.rate) or abs(s.rate - 1) > MAX_RATE_DELTA + 1e-9
                    or s.confidence not in ("high", "medium", "low") or not np.isfinite(s.score)):
                raise ValueError("invalid segment")
            if i and self.segments[i - 1].end_ms != s.start_ms:
                raise ValueError("segments must be contiguous")

    def map(self, t_ms: float) -> int:
        if not np.isfinite(t_ms):
            raise ValueError("expected finite time")
        i = max(0, bisect_right([s.start_ms for s in self.segments], t_ms) - 1)
        return self.segments[i].map(t_ms)

    def to_json(self) -> str:
        return json.dumps({"version": 1, "method": self.method,
                           "segments": [asdict(s) for s in self.segments],
                           "windows": [asdict(w) for w in self.windows]}, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> TimingMap:
        data = json.loads(value)
        if data.get("version") != 1:
            raise ValueError("unsupported timing-map version")
        return cls([Segment(**s) for s in data["segments"]], data["method"],
                   [WindowLag(**w) for w in data.get("windows", [])])


@dataclass(frozen=True)
class TimingQuality:
    max_offset_ms: float
    residual_mad_ms: float
    low_fraction: float
    grade: str


def quality(mapping: TimingMap) -> TimingQuality:
    """Separate release offset from the scatter of reliable window residuals."""
    maximum = max(abs(s.offset_ms + (t - s.start_ms) * (s.rate - 1))
                  for s in mapping.segments for t in (s.start_ms, s.end_ms))
    windows = [w for w in mapping.windows
               if mapping.segments[0].start_ms <= w.centre_ms <= mapping.segments[-1].end_ms]
    reliable = [w for w in windows if w.confidence != "low"]
    low_fraction = 1 - len(reliable) / len(windows) if windows else 1.0
    residuals = np.array([w.lag_ms - (mapping.map(w.centre_ms) - w.centre_ms)
                          for w in reliable])
    mad = float(np.median(np.abs(residuals - np.median(residuals)))) if reliable else 0.0
    grade = "bad" if low_fraction >= 0.5 or mad > 200 else "good" if mad <= 80 else "fair"
    return TimingQuality(float(maximum), mad, low_fraction, grade)


def speech_activity(features: AudioFeatures) -> np.ndarray:
    """Positive 250..3500 Hz log-energy above a local three-second 20% floor.

    Floor estimation uses 100 ms bins to avoid a frame-by-frame percentile
    matrix. The returned float32 curve retains the original 10 ms framing.
    """
    spectrum = np.asarray(features.spectrogram)
    if spectrum.ndim != 2 or spectrum.shape[1] != len(BAND_EDGES_HZ) - 1:
        raise ValueError("expected audio-v3 spectrogram")
    if not len(spectrum):
        return np.empty(0, np.float32)
    bands = (BAND_EDGES_HZ[:-1] >= 250) & (BAND_EDGES_HZ[1:] <= 3500)
    energy = spectrum[:, bands].mean(axis=1, dtype=np.float32)
    coarse = np.array([energy[i:i + 10].mean() for i in range(0, len(energy), 10)], np.float32)
    padded = np.pad(coarse, (15, 15), mode="edge")
    floor = np.percentile(np.lib.stride_tricks.sliding_window_view(padded, 31), 20, axis=1)
    floor = np.interp(np.arange(len(energy)), np.arange(len(coarse)) * 10 + 4.5, floor)
    contrast = np.maximum(energy - floor, 0)
    return (np.convolve(np.pad(contrast, (2, 2), mode="edge"), np.ones(5) / 5,
                        mode="valid") / 255).astype(np.float32)


def _intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    for start, end in intervals:
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("expected finite, positive-duration intervals")
        result.append((int(start), int(end)))
    return sorted(result)


def window_lags(activity: np.ndarray, intervals: Iterable[tuple[int, int]], *,
                window_ms: int = 120000, step_ms: int = 30000,
                max_lag_ms: int = 60000, rate: float = 1.0) -> list[WindowLag]:
    """Pearson-normalized FFT correlation; positive lag means move later.

    Padding permits the full lag range; candidates with less than 75% real
    audio coverage are rejected. Competing peaks over one second away lower
    confidence. Intervals are an occupancy union, never a line-count weight.
    """
    a = np.asarray(activity, dtype=np.float64)
    if a.ndim != 1 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("expected finite nonnegative activity")
    if window_ms < 1000 or step_ms <= 0 or max_lag_ms < 0 or abs(rate - 1) > MAX_RATE_DELTA + 1e-9:
        raise ValueError("invalid correlation settings")
    items = _intervals(intervals)
    if not items or not len(a):
        return []
    end = max(e for _, e in items)
    width = max(100, window_ms // HOP_MS)
    radius = max_lag_ms // HOP_MS
    starts = list(range(0, max(1, end - window_ms + 1), step_ms))
    last = max(0, end - window_ms)
    if starts[-1] != last:
        starts.append(last)
    result = []
    for start in starts:
        template = np.zeros(width, np.float64)
        for lo, hi in items:
            if hi <= start or lo >= start + window_ms:
                continue
            left = max(0, int(np.ceil(((lo - start) * rate) / HOP_MS)))
            right = min(width, int(np.ceil(((hi - start) * rate) / HOP_MS)))
            template[left:right] = 1
        var = float(np.sum((template - template.mean()) ** 2))
        if var < width * 0.015:
            result.append(WindowLag(start, start + window_ms, 0, 0, "low"))
            continue
        audio_start = start // HOP_MS - radius
        count = width + radius * 2
        chunk = np.zeros(count)
        lo, hi = max(0, audio_start), min(len(a), audio_start + count)
        if hi > lo:
            chunk[lo - audio_start:hi - audio_start] = a[lo:hi]
        nfft = 1 << (count + width - 2).bit_length()
        cross = np.fft.irfft(np.fft.rfft(chunk, nfft) * np.fft.rfft(template[::-1], nfft), nfft)
        cross = cross[width - 1:count]
        sums = np.r_[0, np.cumsum(chunk)]
        squares = np.r_[0, np.cumsum(chunk * chunk)]
        sums = sums[width:] - sums[:-width]
        variance = np.maximum(0, squares[width:] - squares[:-width] - sums * sums / width)
        denom = np.sqrt(var * variance)
        scores = np.divide(cross - sums * template.mean(), denom,
                           out=np.zeros_like(cross), where=denom > 1e-12)
        candidate_starts = audio_start + np.arange(len(scores))
        overlap = np.minimum(candidate_starts + width, len(a)) - np.maximum(candidate_starts, 0)
        scores[overlap < width * 0.75] = -1
        best = int(np.argmax(scores))
        score = max(0.0, float(scores[best]))
        rivals = scores.copy()
        rivals[max(0, best - 100):best + 101] = -1
        margin = score - max(0.0, float(rivals.max()))
        confidence = "high" if score >= 0.3 and margin >= 0.025 else "medium" if score >= 0.12 and margin >= 0.012 else "low"
        # Rate warping is about the window's start, so express lag at centre.
        lag = (best - radius) * HOP_MS + (rate - 1) * window_ms / 2
        result.append(WindowLag(start, start + window_ms, lag, score, confidence))
    return result


def _loss(residual: np.ndarray) -> float:
    return float(np.minimum((residual / 350) ** 2, 2).sum())


def _line(x: np.ndarray, y: np.ndarray, weights: np.ndarray, allow_rate: bool) -> tuple[float, float, np.ndarray]:
    unit_intercept = float(np.median(y))
    unit_good = np.abs(y - unit_intercept) <= 500
    if unit_good.any():
        unit_intercept = float(np.average(y[unit_good], weights=weights[unit_good]))
    unit_residual = y - unit_intercept
    slope = 0.0
    if allow_rate and len(x) >= 3 and np.ptp(x) >= RATE_MIN_SPAN_MS:
        # Stratified samples bound the robust seed; refinement uses all points.
        sample = np.linspace(0, len(x) - 1, min(64, len(x)), dtype=int)
        left, right = np.triu_indices(len(sample), 1)
        dx = x[sample[right]] - x[sample[left]]
        dy = y[sample[right]] - y[sample[left]]
        distant = dx > max(1, np.ptp(x) / 4)
        if distant.any():
            slope = float(np.clip(np.median(dy[distant] / dx[distant]),
                                  -MAX_RATE_DELTA, MAX_RATE_DELTA))
    intercept = float(np.median(y - slope * x))
    residual = y - (intercept + slope * x)
    good = np.abs(residual) <= 500
    if allow_rate and good.sum() >= 3 and np.ptp(x[good]) >= RATE_MIN_SPAN_MS:
        xx, yy, ww = x[good], y[good], weights[good]
        centre = np.average(xx, weights=ww)
        v = np.sum(ww * (xx - centre) ** 2)
        slope = float(np.clip(np.sum(ww * (xx - centre) * yy) / v, -MAX_RATE_DELTA, MAX_RATE_DELTA))
        intercept = float(np.average(yy - slope * xx, weights=ww))
    residual = y - (intercept + slope * x)
    family = min(RATE_FAMILIES, key=lambda rate: abs(1 + slope - rate))
    good = np.abs(residual) <= 500
    if (not allow_rate or good.sum() < 3 or np.ptp(x[good]) < RATE_MIN_SPAN_MS
            or abs(1 + slope - family) > family * RATE_TOLERANCE
            or _loss(unit_residual) - _loss(residual) <= RATE_PENALTY):
        return unit_intercept, 0.0, unit_residual
    # Retain the fitted value within the physical family, including clock drift;
    # rounding to its nominal ratio would erase genuine 23.976/24 differences.
    return intercept, slope, residual


def fit(windows: Iterable[WindowLag], intervals: Iterable[tuple[int, int]] = (), *,
        method: str = "piecewise_rate", duration_ms: int | None = None) -> TimingMap:
    """Robust penalized segmentation; weak windows inherit reliable neighbours.

    Fit once, then search only near local residual changes. Robust slope seeds
    use at most 64 stratified points, never per-interval pairwise matrices.
    Bounded residual costs prevent a music outlier from buying a new segment.
    Boundaries prefer the largest subtitle gap around the observed change.
    Each segment is monotonic; a negative release cut may jump backwards.
    """
    if method not in METHODS:
        raise ValueError("unknown timing method")
    all_windows = sorted(windows, key=lambda w: w.centre_ms)
    if any(not np.isfinite(w.lag_ms) or not np.isfinite(w.score) for w in all_windows):
        raise ValueError("expected finite window evidence")
    items = _intervals(intervals)
    end = max(1, duration_ms or max([w.end_ms for w in all_windows] + [e for _, e in items] + [1]))
    reliable = [w for w in all_windows if w.confidence != "low"]
    if not reliable:
        return TimingMap([Segment(0, end, 0)], method, all_windows)
    x = np.array([w.centre_ms for w in reliable])
    y = np.array([w.lag_ms for w in reliable])
    weights = np.maximum(0.01, [w.score for w in reliable])
    n = len(x)
    whole = _line(x, y, weights, method == "piecewise_rate")
    models = {(0, n): whole}
    # A global rate can partly explain a cut. Test its local residual changes
    # before accepting it, with medians so isolated music peaks do not split.
    residual = whole[2]
    candidates = [0]
    if method != "shift":
        for edge in range(1, n - 3):
            before = np.median(residual[max(0, edge - 4):edge])
            after = np.median(residual[edge:edge + 4])
            if abs(after - before) > 50:
                candidates.append(edge)
    candidates.append(n)
    cost = {0: -6.0}
    parents = {}
    for pos, hi in enumerate(candidates[1:], 1):
        cost[hi] = np.inf
        for lo in candidates[:pos]:
            if lo and hi - lo < 4:
                continue
            model = models.get((lo, hi))
            if model is None:
                model = _line(x[lo:hi], y[lo:hi], weights[lo:hi], method == "piecewise_rate")
            b, slope, residual = model
            loss = _loss(residual) + (RATE_PENALTY if slope else 0)
            candidate = cost[lo] + loss + 6.0
            if candidate < cost[hi]:
                cost[hi], parents[hi] = candidate, lo
                models[(lo, hi)] = model
    ranges = []
    hi = n
    while hi:
        lo = int(parents[hi])
        ranges.append((lo, hi))
        hi = lo
    ranges.reverse()
    bounds = [0]
    union = []
    for lo, hi in items:
        if union and lo <= union[-1][1]:
            union[-1] = (union[-1][0], max(hi, union[-1][1]))
        else:
            union.append((lo, hi))
    for (_, before), (after, _) in zip(ranges, ranges[1:]):
        change = (x[before - 1] + x[after]) / 2
        gaps = [(b[0] - a[1], (a[1] + b[0]) // 2) for a, b in zip(union, union[1:])
                if abs((a[1] + b[0]) / 2 - change) <= 60000]
        boundary = max(gaps)[1] if gaps else int(change)
        bounds.append(max(bounds[-1] + 1, min(end - 1, boundary)))
    bounds.append(end)
    segments = []
    for i, (lo, hi) in enumerate(ranges):
        b, slope, residual = models[(lo, hi)]
        good = np.abs(residual) <= 500
        score = float(np.median(weights[lo:hi][good])) if good.any() else 0.0
        high_count = sum(w.confidence == "high" for w, keep in zip(reliable[lo:hi], good) if keep)
        confidence = "high" if good.sum() >= 4 and high_count >= 3 and score >= 0.3 else "medium" if good.sum() >= 4 else "low"
        segments.append(Segment(bounds[i], bounds[i + 1], b + slope * bounds[i],
                                1 + slope, confidence, score))
    trusted = [i for i, s in enumerate(segments) if s.confidence != "low"]
    for i, s in enumerate(segments):
        if s.confidence == "low" and trusted:
            neighbour = segments[min(trusted, key=lambda j: abs(j - i))]
            offset = neighbour.offset_ms + (s.start_ms - neighbour.start_ms) * (neighbour.rate - 1)
            # Borrow the boundary offset, never a speed without this segment's
            # own five-minute reliable span and significance evidence.
            segments[i] = Segment(s.start_ms, s.end_ms, offset, 1.0, "low", s.score)
        elif s.confidence == "low":
            segments[i] = Segment(s.start_ms, s.end_ms, 0, 1, "low", s.score)
    return TimingMap(segments, method, all_windows)


def estimate(features: AudioFeatures, intervals: Iterable[tuple[int, int]], *,
             method: str = "piecewise_rate", correlation_bias_ms: float = CORRELATION_BIAS_MS) -> TimingMap:
    if not np.isfinite(correlation_bias_ms):
        raise ValueError("expected finite correlation bias")
    items = _intervals(intervals)
    activity = speech_activity(features)
    windows = window_lags(activity, items)
    if method == "piecewise_rate" and windows and np.median([w.score for w in windows]) < 0.4:
        # PAL drift smears a 120 s occupancy window by five seconds. Search
        # a few sparse pilot windows before robust segmentation interprets
        # those smeared, unrelated peaks as release cuts.
        pilots = [(float(rate), window_lags(activity, items, step_ms=240000, rate=float(rate)))
                  for rate in np.linspace(0.95, 1.05, 11)]
        rate, pilot = max(pilots, key=lambda pair: np.median([w.score for w in pair[1]]))
        if np.median([w.score for w in pilot]) > np.median([w.score for w in windows]) + 0.03:
            windows = window_lags(activity, items, rate=rate)
    reliable = [w for w in windows if w.confidence != "low"]
    if method == "piecewise_rate" and len(reliable) >= 3:
        _, slope, residual = _line(np.array([w.centre_ms for w in reliable]),
                                  np.array([w.lag_ms for w in reliable]),
                                  np.array([w.score for w in reliable]), True)
        if abs(slope) > 0.002:
            refined = window_lags(activity, items, rate=1 + slope)
            # Retain the original evidence if rate compensation does not help.
            if np.median([w.score for w in refined]) > np.median([w.score for w in windows]):
                windows = refined
    windows = [WindowLag(w.start_ms, w.end_ms, w.lag_ms - correlation_bias_ms, w.score, w.confidence)
               for w in windows]
    return fit(windows, items, method=method, duration_ms=len(activity) * HOP_MS)


def reference_map(samples: Iterable[tuple[int, int]], *, method: str = "piecewise_rate",
                  intervals: Iterable[tuple[int, int]] = (), duration_ms: int | None = None) -> TimingMap:
    """Map current times to a lane's immutable source times, using the same fit."""
    windows = []
    for current, source in samples:
        if not np.isfinite(current) or not np.isfinite(source):
            raise ValueError("expected finite reference samples")
        windows.append(WindowLag(int(current), int(current), source - current, 1.0, "high"))
    return fit(windows, intervals, method=method, duration_ms=duration_ms)


def speech_segments(activity: np.ndarray, start: int, end: int, *,
                    vad_segments: np.ndarray | None = None) -> list[tuple[int | None, int | None]]:
    """Local contrast runs, joining syllable gaps <=80 ms, lasting >=100 ms.

    Energy is only a speech heuristic: flat background and low contrast abstain.
    An edge cut off by the search region is not an observed onset/offset.
    """
    if vad_segments is not None:
        # An empty VAD result means no speech; only None uses the old detector.
        return [(int(a) if a > start else None, int(b) if b < end else None)
                for a, b in vad_segments if b > start and a < end]
    lo = max(0, start // HOP_MS)
    hi = min(len(activity), (end + HOP_MS - 1) // HOP_MS)
    curve = activity[lo:hi]
    if len(curve) < 10:
        return []
    floor, peak = np.percentile(curve, (20, 90))
    if peak < 0.035 or peak - floor < 0.025:
        return []
    active = curve > floor + max(0.018, (peak - floor) * 0.3)
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs: list[tuple[int, int]] = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        if runs and a - runs[-1][1] <= 8:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return [((a + lo) * HOP_MS if a > 0 else None,
             (b + lo) * HOP_MS if b < len(curve) else None)
            for a, b in runs if b - a >= 10]


def _speech_edges(activity: np.ndarray, start: int, end: int, search_ms: int,
                  lead_in_ms: int, lead_out_ms: int, *,
                  vad_segments: np.ndarray | None = None) -> tuple[int | None, int | None]:
    runs = speech_segments(activity, start - search_ms - 200, end + search_ms + 200,
                           vad_segments=vad_segments)
    runs = [(a, b) for a, b in runs if (b is None or b > start) and (a is None or a < end)]
    onsets = [a for a, _ in runs if a is not None
              and abs(a - lead_in_ms - start) <= search_ms]
    offsets = [b for _, b in runs if b is not None
               and abs(b + lead_out_ms - end) <= search_ms]
    onset = min(onsets, key=lambda t: (abs(t - start - lead_in_ms), t)) if onsets else None
    offset = min(offsets, key=lambda t: (abs(t - end + lead_out_ms), t)) if offsets else None
    if onset is not None and offset is not None and offset <= onset:
        return None, None
    return onset, offset


def _snap_activity(features: AudioFeatures) -> np.ndarray:
    # Use local line contrast rather than the mapping curve's moving floor:
    # a sustained utterance must not acquire an offset as that floor rises.
    spectrum = np.asarray(features.spectrogram)
    if spectrum.ndim != 2 or spectrum.shape[1] != len(BAND_EDGES_HZ) - 1:
        raise ValueError("expected audio-v3 spectrogram")
    bands = (BAND_EDGES_HZ[:-1] >= 250) & (BAND_EDGES_HZ[1:] <= 3500)
    return spectrum[:, bands].mean(axis=1, dtype=np.float32) / 255


def hint_speech_edges(activity, start, end, *, search_ms=HINT_SEARCH_MS,
                      previous=None, following=None, runs=None, run_on=HINT_RUN_ON,
                      lead_in_ms=0, lead_out_ms=0):
    """Independent observed edges; continuous neighbour speech uses a valley.

    Calibration and import snapping still use observed edges only. A clipped
    run proves continuity only inside the region actually searched.
    """
    left = min(start, start + lead_in_ms) - search_ms - 200
    right = max(end, end - lead_out_ms) + search_ms + 200
    start, end = start + lead_in_ms, end - lead_out_ms
    if runs is None:
        runs = speech_segments(activity, left, right)
    starts = [a for a, _ in runs if a is not None and abs(a - start) <= search_ms]
    ends = [b for _, b in runs if b is not None and abs(b - end) <= search_ms]
    onset = min(starts, key=lambda t: (abs(t - start), t)) if starts else None
    offset = min(ends, key=lambda t: (abs(t - end), t)) if ends else None

    def continuous(a, b):
        return left <= a <= b < right and any(
            (lo is None or lo < a) and (hi is None or hi > b) for lo, hi in runs)

    def valley(a, b):
        lo = max(0, (a + HOP_MS - 1) // HOP_MS)
        hi = min(len(activity), b // HOP_MS + 1)
        if hi <= lo:
            return None
        values = np.asarray(activity[lo:hi])
        return (lo + int(np.argmin(values))) * HOP_MS

    if run_on and previous is not None and continuous(previous[1], start):
        onset = valley(start - 300, start + 300)
    if run_on and following is not None and continuous(end, following[0]):
        offset = valley(end - 300, end + 300)
    return onset, offset


@dataclass(frozen=True)
class HintCalibration:
    lead_in_ms: int
    lead_out_ms: int
    source: str                       # episode | project | autotime
    onset_samples: int
    offset_samples: int


def calibrate_hint_padding(activity: np.ndarray, dialogue_groups: Iterable[tuple[int, int]], *,
                           last_project: tuple[int, int] | None = None,
                           autotime_values: tuple[int, int] = (SNAP_LEAD_IN_MS, SNAP_LEAD_OUT_MS),
                           vad_segments: np.ndarray | None = None) -> HintCalibration:
    """Median episode padding from nearest observed edges within +/-600 ms.

    Each boundary needs at least 20 group samples. Otherwise return the last
    project pair, then the autotime pair, unchanged. Empty/reversed groups are
    skipped. No settings are persisted.
    Equidistant edges prefer the earlier time; medians round to integer ms.
    """
    onsets, offsets = _hint_padding_samples(activity, dialogue_groups, vad_segments=vad_segments)
    return _hint_calibration(onsets, offsets, last_project, autotime_values)


def _hint_padding_samples(activity, times, *, vad_segments=None):
    onsets, offsets = [], []
    for start, end in _intervals((s, e) for s, e in times if e > s):
        runs = speech_segments(activity, start - 800, end + 800, vad_segments=vad_segments)
        starts = [a for a, _ in runs if a is not None and abs(a - start) <= 600]
        ends = [b for _, b in runs if b is not None and abs(b - end) <= 600]
        if starts:
            onsets.append(min(starts, key=lambda t: (abs(t - start), t)) - start)
        if ends:
            offsets.append(end - min(ends, key=lambda t: (abs(t - end), t)))
    return onsets, offsets


def _hint_calibration(onsets, offsets, last_project, autotime_values):
    if len(onsets) >= 20 and len(offsets) >= 20:
        values = (int(round(np.clip(np.median(onsets), 0, 600))),
                  int(round(np.clip(np.median(offsets), -200, 600))))
        source = "episode"
    else:
        values = last_project if last_project is not None else autotime_values
        source = "project" if last_project is not None else "autotime"
    return HintCalibration(*values, source, len(onsets), len(offsets))


def calibrate_manual_hint_padding(inputs, habit: tuple[int, int]) -> HintCalibration:
    """Pool manual boundary samples across current/cached project episodes."""
    onsets, offsets = [], []
    for activity, doc in inputs:
        for group in doc.groups:
            if group.kind != "dialogue" or not group.members:
                continue
            start, end = doc.group_time(group)
            manual = {}
            for edge, time in (("start", start), ("end", end)):
                manual[edge] = any(
                    getattr(m, edge) == time and (
                        m.extra.get("time_src", {}).get(edge) == "manual"
                        or edge not in m.extra.get("time_src", {}) and (
                            group.state == "confirmed"
                            or "time_src" not in m.extra and m.extra.get("time_manual")))
                    for m in doc.members_of(group))
            if any(manual.values()):
                a, b = _hint_padding_samples(activity, [(start, end)])
                if manual["start"]:
                    onsets.extend(a)
                if manual["end"]:
                    offsets.extend(b)
    result = _hint_calibration(onsets, offsets, None, habit)
    return HintCalibration(result.lead_in_ms, result.lead_out_ms,
                           "manual" if result.source == "episode" else "habit",
                           result.onset_samples, result.offset_samples)


def clamp_suggested_times(suggested: tuple[int | None, int | None], current: tuple[int, int], *,
                          previous: tuple[int, int] | None = None,
                          following: tuple[int, int] | None = None) -> tuple[int | None, int | None] | None:
    """Clamp to neighbours' current times, allowing pre-existing overlaps.

    Neighbours are the adjacent dialogue/song groups in chronological order.
    An overlap exempts only that side. Touching is allowed; empty spans abstain.
    """
    start, end = suggested
    if start is not None and previous is not None and not (previous[0] < current[1] and current[0] < previous[1]):
        start = max(start, previous[1])
    if end is not None and following is not None and not (following[0] < current[1] and current[0] < following[1]):
        end = min(end, following[0])
    effective_start = current[0] if start is None else start
    effective_end = current[1] if end is None else end
    return (start, end) if (start is not None or end is not None) and effective_end > effective_start else None


def suggest_group_times(activity, start: int, end: int, *,
                        lead_in_ms: int = SNAP_LEAD_IN_MS, lead_out_ms: int = SNAP_LEAD_OUT_MS,
                        previous: tuple[int, int] | None = None,
                        following: tuple[int, int] | None = None, runs=None,
                        search_ms: int = HINT_SEARCH_MS) -> tuple[int | None, int | None] | None:
    """Suggest padded, neighbour-safe edges; abstain if adoption would drift."""
    def compute(current, observed=None):
        a, b = hint_speech_edges(activity, *current, lead_in_ms=lead_in_ms,
                                 lead_out_ms=lead_out_ms, previous=previous, following=following,
                                 runs=observed, search_ms=search_ms)
        return clamp_suggested_times(
            (max(0, a - lead_in_ms) if a is not None else None,
             b + lead_out_ms if b is not None else None), current,
            previous=previous, following=following)

    current = (start, end)
    suggested = compute(current, runs)
    if suggested is None:
        return None
    adopted = tuple(c if s is None else s for c, s in zip(current, suggested))
    # Clamping or a clipped window can expose a different edge on the next search.
    again = compute(adopted)
    if again is not None and any(s is not None and s != c for c, s in zip(adopted, again)):
        return None
    return suggested


def snap_lines(features: AudioFeatures, lines: Iterable[tuple[int, int]], *,
               mode: str = "both", lead_in_ms: int = SNAP_LEAD_IN_MS, lead_out_ms: int = SNAP_LEAD_OUT_MS,
               search_ms: int = 600,
               skip: Callable[[tuple[int, int]], bool] | None = None,
               vad_segments: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Return mapped intervals nudged to nearby speech; input remains untouched.

    The search bound applies to the final subtitle edges, including padding.
    keep_duration shifts both edges by the onset delta. Missing edges stay put.
    The caller supplies exclusions for signs, songs and protected lines. Crossing
    proposals fall back to their mapped times, preserving existing edge order.
    Existing overlaps (including equal group spans) stay allowed. New overlaps
    cut the earlier end to the later start. If that leaves less than 300 ms,
    restore the earlier pre-snap end; restore conflicting proposals as needed.
    Excluded lines remain untouched, and output retains source order.
    """
    if mode not in ("start", "both", "keep_duration"):
        raise ValueError("unknown snap mode")
    if any(not np.isfinite(v) or v < 0 or int(v) != v
           for v in (lead_in_ms, lead_out_ms, search_ms)):
        raise ValueError("expected nonnegative integer snap settings")
    raw_lines = list(lines)
    # Validate without sorting: source order is part of the contract.
    _intervals(raw_lines)
    original = [(int(s), int(e)) for s, e in raw_lines]
    activity = _snap_activity(features)
    result = []
    excluded = set()
    for i, line in enumerate(original):
        start, end = line
        if skip is not None and skip(line):
            excluded.add(i)
            result.append((start, end))
            continue
        onset, offset = _speech_edges(activity, start, end, search_ms, lead_in_ms, lead_out_ms,
                                      vad_segments=vad_segments)
        new_start = onset - lead_in_ms if onset is not None else start
        new_end = end
        if mode == "both" and offset is not None:
            new_end = offset + lead_out_ms
        elif mode == "keep_duration":
            new_end += new_start - start
        result.append((new_start, new_end) if new_end > new_start else (start, end))
    # Compare all chronological neighbours for each edge, without reordering
    # the returned lines. Reverting can expose another crossing, so repeat.
    orders = [sorted(range(len(original)), key=lambda i: original[i][edge]) for edge in (0, 1)]
    restored = set(excluded)
    while True:
        conflicts = set()
        for edge, order in enumerate(orders):
            for a, b in zip(order, order[1:]):
                if (result[a][edge] > result[b][edge]
                        or original[a][edge] == original[b][edge]
                        and result[a][edge] != result[b][edge]):
                    conflicts.update((a, b))
        changed = [i for i in conflicts if result[i] != original[i]]
        for i in changed:
            result[i] = original[i]
            restored.add(i)
        adjusted = False
        for a, b in ((a, b) for a in orders[0] for b in orders[0]
                     if original[a][1] <= original[b][0]):
            if result[a][1] <= result[b][0]:
                continue
            if a not in restored and result[b][0] - result[a][0] >= 300:
                result[a] = (result[a][0], result[b][0])
            else:
                end = original[a][1]
                result[a] = (result[a][0], end) if end > result[a][0] else original[a]
                if result[a][1] > result[b][0]:
                    # Reverting an end alone can still collide with an early onset.
                    result[b] = original[b]
                    restored.add(b)
            adjusted = True
        if not changed and not adjusted:
            break
    return result
