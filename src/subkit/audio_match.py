"""Qt-free repeated-passage location by band-centred Pearson correlation.

32 averaged bands retain changing harmony while per-band window centring removes
stationary timbre and constant dB gain. Ten-second segments vote with <=60 ms
residuals around a shift or a linear drift (up to 3%). Segment threshold 0.55,
single-segment threshold 0.70; see the synthetic and real-media evaluations.
Global search averages four frames (40 ms), retaining eight spaced peaks per
segment, then evaluates the original Pearson score within 80 ms at 10 ms hops.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

from subkit.audio import AudioFeatures, CACHE_VERSION, HOP_MS

SEGMENT_THRESHOLD = 0.55
SINGLE_THRESHOLD = 0.70
CONSENSUS_MS = 60
MAX_SCALE_DELTA = 0.03
COARSE_FRAMES = 4
MAX_PEAKS = 8
TOO_SHORT = "段落太短（不到 3 秒），没法按音频找。"
VERSION_MISMATCH = "这一段的音频特征是旧版本算的，请在原来那一集重新记录。"


@dataclass
class Template:
    spectrogram: np.ndarray
    onset: np.ndarray
    hop_ms: int = HOP_MS
    features: str = CACHE_VERSION


@dataclass
class Match:
    start_ms: int
    score: float
    covered: tuple[int, int]
    segments: list[tuple[int, int, float]]
    drift_ms: int
    scale: float
    covered_intervals: tuple[tuple[int, int], ...] | None = None


@dataclass
class SearchResult:
    matches: list[Match]
    best_score: float


def _validate(t: Template) -> None:
    if t.features != CACHE_VERSION:
        raise ValueError(VERSION_MISMATCH)
    if t.hop_ms != HOP_MS:
        raise ValueError("unsupported feature hop")
    _spectrum(t.spectrogram)
    if t.onset.shape != (len(t.spectrogram),) or t.onset.dtype != np.float32:
        raise ValueError("expected float32 onset matching spectrogram")
    if not np.all(np.isfinite(t.onset)):
        raise ValueError("expected finite onset")


def _spectrum(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 128 or a.dtype != np.uint8:
        raise ValueError("expected n x 128 uint8 spectrogram")
    return a


def template_from(features: AudioFeatures, start_ms: int, end_ms: int) -> Template:
    a = _spectrum(features.spectrogram)
    if start_ms < 0 or end_ms < start_ms or end_ms > len(a) * HOP_MS:
        raise ValueError("template range outside audio")
    lo, hi = int(np.ceil(start_ms / HOP_MS)), int(np.ceil(end_ms / HOP_MS))
    if end_ms - start_ms < 3000 or (hi - lo) * HOP_MS < 3000:
        raise ValueError(TOO_SHORT)
    t = Template(a[lo:hi].copy(), features.onset[lo:hi].copy())
    _validate(t)
    return t


def save_template(path: str | Path, t: Template) -> None:
    """Atomic replacement needs a sibling file on the destination filesystem."""
    _validate(t)
    path = Path(path)
    scratch = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".passage-", suffix=".npz", delete=False) as out:
            scratch = Path(out.name)
            np.savez_compressed(out, spectrogram=t.spectrogram, onset=t.onset,
                                hop_ms=np.int64(t.hop_ms), features=np.str_(t.features))
        os.replace(scratch, path)
    finally:
        if scratch is not None and scratch.exists():
            scratch.unlink()


def load_template(path: str | Path) -> Template:
    with np.load(path, allow_pickle=False) as data:
        t = Template(data["spectrogram"].copy(), data["onset"].copy(),
                     int(data["hop_ms"].item()), str(data["features"].item()))
    _validate(t)
    return t


def _reduce(a: np.ndarray) -> np.ndarray:
    return a.reshape(len(a), 32, 4).mean(axis=2, dtype=np.float32)


class EpisodeIndex:
    """Cache short-block FFTs and window statistics; no Qt or PCM decoding."""

    def __init__(self, spectrogram: np.ndarray):
        self.count = len(_spectrum(spectrogram))
        a = _reduce(spectrogram)
        if self.count:
            a -= a.mean(axis=0)
        self._audio = a
        self._sums = np.pad(np.cumsum(a, axis=0, dtype=np.float64), ((1, 0), (0, 0)))
        self._squares = np.r_[0.0, np.cumsum(np.sum(a * a, axis=1, dtype=np.float64))]
        count = self.count // COARSE_FRAMES
        self._coarse = a[:count * COARSE_FRAMES].reshape(count, COARSE_FRAMES, 32).mean(axis=1)
        self._coarse_sums = np.pad(np.cumsum(self._coarse, axis=0, dtype=np.float64), ((1, 0), (0, 0)))
        self._coarse_squares = np.r_[0.0, np.cumsum(np.sum(self._coarse * self._coarse, axis=1, dtype=np.float64))]
        self._ffts: dict[int, np.ndarray] = {}
        self._coarse_norms: dict[int, np.ndarray] = {}
        self._block_fft(1024)

    def _block_fft(self, size: int) -> np.ndarray:
        if size not in self._ffts:
            step = size // 2
            blocks = (len(self._coarse) + step - 1) // step
            if not blocks:
                self._ffts[size] = np.empty((0, 32, size // 2 + 1), np.complex64)
                return self._ffts[size]
            padded = np.pad(self._coarse, ((0, blocks * step + step - len(self._coarse)), (0, 0)))
            windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=0)[::step][:blocks]
            self._ffts[size] = np.fft.rfft(windows, axis=2)
        return self._ffts[size]

    def _correlate(self, a: np.ndarray) -> np.ndarray:
        m = len(a)
        energy = float(np.sum(a * a, dtype=np.float64))
        if energy <= 1e-8:
            return np.zeros(len(self._coarse) - m + 1, np.float32)
        if m not in self._coarse_norms:
            sums = self._coarse_sums[m:] - self._coarse_sums[:-m]
            variance = self._coarse_squares[m:] - self._coarse_squares[:-m] - np.sum(sums * sums, axis=1) / m
            self._coarse_norms[m] = np.sqrt(np.maximum(variance, 0))
        size = max(1024, 1 << (2 * m - 1).bit_length())
        kernel = np.fft.rfft(a.T, n=size, axis=1)
        # Contract bands directly, without an episode-sized complex product.
        product = np.einsum("bck,ck->bk", self._block_fft(size), kernel.conj())
        dot = np.fft.irfft(product, n=size, axis=1)[:, :size // 2].reshape(-1)[:len(self._coarse) - m + 1]
        denom = self._coarse_norms[m] * np.sqrt(energy)
        scores = np.zeros_like(dot)
        np.divide(dot, denom, out=scores, where=denom > 1e-8)
        return np.clip(scores, 0, 1)

    def _peaks(self, segment: np.ndarray) -> list[tuple[int, float]]:
        a = _reduce(segment)
        a -= a.mean(axis=0)
        energy = float(np.sum(a * a, dtype=np.float64))
        if energy <= 1e-8:
            return []
        m = len(a)
        count = m // COARSE_FRAMES
        coarse = a[:count * COARSE_FRAMES].reshape(count, COARSE_FRAMES, 32).mean(axis=1)
        coarse -= coarse.mean(axis=0)
        scores = self._correlate(coarse)
        peaks = np.flatnonzero((scores >= np.r_[-1, scores[:-1]]) & (scores > np.r_[scores[1:], -1]))
        selected: list[int] = []
        for at in peaks[np.argsort(-scores[peaks], kind="stable")]:
            if all(abs(int(at) - other) * COARSE_FRAMES >= m // 2 for other in selected):
                selected.append(int(at))
                if len(selected) == MAX_PEAKS:
                    break
        found = []
        for peak in selected:
            lo = max(0, peak * COARSE_FRAMES - 8)
            hi = min(self.count - m, peak * COARSE_FRAMES + 8)
            if hi < lo:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(self._audio[lo:hi + m], m, axis=0)
            dot = np.einsum("pcm,mc->p", windows, a)
            sums = self._sums[lo + m:hi + m + 1] - self._sums[lo:hi + 1]
            variance = self._squares[lo + m:hi + m + 1] - self._squares[lo:hi + 1] - np.sum(sums * sums, axis=1) / m
            denom = np.sqrt(np.maximum(variance, 0) * energy)
            fine = np.zeros(len(dot), np.float32)
            np.divide(dot, denom, out=fine, where=denom > 1e-8)
            at = int(np.argmax(fine))
            found.append((lo + at, float(np.clip(fine[at], 0, 1))))
        return found

    def find(self, template: Template) -> SearchResult:
        _validate(template)
        n = len(template.spectrogram)
        if not n or n > self.count or n * HOP_MS < 3000:
            return SearchResult([], 0.0)
        bounds = [(0, n)] if n < 1500 else [(i, min(i + 1000, n)) for i in range(0, n, 1000)]
        # Avoid a tiny tail; equal subdivision keeps every segment useful.
        if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < 500:
            edges = np.rint(np.linspace(0, n, len(bounds) - 1 + 1)).astype(int)
            bounds = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
        threshold = SINGLE_THRESHOLD if len(bounds) == 1 else SEGMENT_THRESHOLD
        candidates: list[tuple[int, float, float, float]] = []
        best = 0.0
        for number, (lo, hi) in enumerate(bounds):
            for at, score in self._peaks(template.spectrogram[lo:hi]):
                best = max(best, score)
                if score >= threshold:
                    candidates.append((number, (lo + hi) / 2, float(at - lo), score))
        if not candidates:
            return SearchResult([], best)
        tolerance = CONSENSUS_MS / HOP_MS
        values = np.array(candidates)
        groups, positions, offsets, strengths = values.T
        dx = positions[None, :] - positions[:, None]
        slopes = np.zeros_like(dx)
        np.divide(offsets[None, :] - offsets[:, None], dx, out=slopes, where=dx != 0)
        pairs = np.triu((groups[:, None] != groups[None, :]) & (np.abs(slopes) <= MAX_SCALE_DELTA), 1)
        rows, cols = np.nonzero(pairs)
        hypotheses = np.column_stack((np.r_[offsets, offsets[rows] - slopes[rows, cols] * positions[rows]],
                                      np.r_[np.zeros(len(offsets)), slopes[rows, cols]]))
        proposed: list[Match] = []
        seen: set[tuple[int, ...]] = set()
        # Bound temporary consensus arrays, and fit each agreeing set only once.
        for begin in range(0, len(hypotheses), 256):
            batch = hypotheses[begin:begin + 256]
            residual = np.abs(offsets - batch[:, :1] - batch[:, 1:] * positions)
            chosen = np.full((len(batch), len(bounds)), -1, np.int32)
            for number in range(len(bounds)):
                indices = np.flatnonzero(groups == number)
                if not len(indices):
                    continue
                eligible = np.where(residual[:, indices] <= tolerance, strengths[indices], -1)
                at = np.argmax(eligible, axis=1)
                chosen[:, number] = np.where(eligible[np.arange(len(batch)), at] >= 0, indices[at], -1)
            for selection in chosen:
                key = tuple(selection[selection >= 0].tolist())
                if key in seen or len(key) < min(2, len(bounds)):
                    continue
                seen.add(key)
                agreeing = [candidates[i] for i in key]
                proposed_match = self._fit(agreeing, bounds, tolerance)
                if proposed_match is not None:
                    proposed.append(proposed_match)
        matches: list[Match] = []
        for match in sorted(proposed, key=lambda m: (-len(m.segments), -m.score)):
            lo = match.start_ms + match.covered[0] * match.scale
            hi = match.start_ms + match.covered[1] * match.scale
            if all(hi <= other.start_ms + other.covered[0] * other.scale or
                   lo >= other.start_ms + other.covered[1] * other.scale for other in matches):
                matches.append(match)
        return SearchResult(sorted(matches, key=lambda m: m.start_ms), best)

    @staticmethod
    def _fit(agreeing: list[tuple[int, float, float, float]], bounds: list[tuple[int, int]],
             tolerance: float) -> Match | None:
        intercept, slope = agreeing[0][2], 0.0
        x = np.array([c[1] for c in agreeing])
        y = np.array([c[2] for c in agreeing])
        if len(agreeing) > 1:
            centred = x - x.mean()
            slope = float(np.dot(centred, y - y.mean()) / np.dot(centred, centred))
            intercept = float(y.mean() - slope * x.mean())
        if abs(slope) > MAX_SCALE_DELTA or np.max(np.abs(y - intercept - slope * x)) > tolerance:
            return None
        intervals = []
        for lo, hi in sorted(bounds[c[0]] for c in agreeing):
            if intervals and lo <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(hi, intervals[-1][1]))
            else:
                intervals.append((lo, hi))
        lo, hi = intervals[0][0], intervals[-1][1]
        return Match(int(round(intercept * HOP_MS)),
                     float(np.median([c[3] for c in agreeing])),
                     (lo * HOP_MS, hi * HOP_MS),
                     [(bounds[c[0]][0] * HOP_MS, int(round(c[2] * HOP_MS)), c[3]) for c in agreeing],
                     int(round(slope * (hi - lo) * HOP_MS)), 1.0 + slope,
                     tuple((a * HOP_MS, b * HOP_MS) for a, b in intervals))
