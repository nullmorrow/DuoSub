"""Qt-free features of decoded mono audio and non-destructive beat suggestions.

Frame i is centred at i * HOP_MS, including the final partial frame. Spectrogram
rows are time, columns are 128 log-spaced bands (0..8 kHz); uint8 maps -80..0 dB
to 0..255. Onsets use unquantized magnitudes, not the display spectrogram.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

RATE = 16000
HOP_MS = 10
DETAIL_HOP_MS = 2
HOP_SAMPLES = RATE * HOP_MS // 1000
WINDOW_SAMPLES = 512
SPECTRUM_BINS = WINDOW_SAMPLES // 2 + 1
DISPLAY_BANDS = 128
# Mel-like edges; narrow low-frequency bands share a bin when FFT resolution is coarser.
BAND_EDGES_HZ = 700 * np.expm1(np.linspace(0, np.log1p((RATE / 2) / 700), DISPLAY_BANDS + 1))
_LINEAR_HZ = np.fft.rfftfreq(WINDOW_SAMPLES, 1 / RATE)
_BAND_START = np.minimum(np.searchsorted(_LINEAR_HZ, BAND_EDGES_HZ[:-1]), SPECTRUM_BINS - 1)
_BAND_STOP = np.minimum(np.maximum(_BAND_START + 1, np.searchsorted(_LINEAR_HZ, BAND_EDGES_HZ[1:])), SPECTRUM_BINS)
_BAND_STOP[-1] = SPECTRUM_BINS  # Include the 8 kHz Nyquist bin.
CACHE_VERSION = "audio-v4"  # Bump when feature layout, scaling or framing changes.


def ffmpeg_audio_index(aid: Optional[int]) -> int:
    """Convert the stored 1-based mpv audio id to an ffmpeg audio stream index."""
    return aid - 1 if aid is not None and aid >= 1 else 0


def waveform_cache_file(video: Path, track: Optional[int], cache: Path) -> Path:
    """Shared waveform cache identity; track is the stored 1-based mpv audio id.

    The identity is the file's size and modified time, not its path, so a video moved to another folder
    keeps its cache; an entry under the older path-based name is renamed on first use."""
    st = video.stat()
    key = hashlib.sha1(f"{st.st_size}|{st.st_mtime_ns}|{ffmpeg_audio_index(track)}|{CACHE_VERSION}".encode("utf-8")).hexdigest()[:20]
    path = cache / f"{key}.npz"
    if not path.exists():
        adopt_legacy(legacy_waveform_cache_file(video, track, cache), path)
    return path


def legacy_waveform_cache_file(video: Path, track: Optional[int], cache: Path, *,
                               recorded: Optional[Path] = None) -> Path:
    """The pre-0.21 path-based name; ``recorded`` is where the video was when the entry was written."""
    st = video.stat()
    key = hashlib.sha1(f"{Path(recorded or video).resolve()}|{st.st_size}|{st.st_mtime_ns}|"
                       f"{ffmpeg_audio_index(track)}|{CACHE_VERSION}".encode("utf-8")).hexdigest()[:20]
    return cache / f"{key}.npz"


def adopt_legacy(old: Path, new: Path) -> bool:
    """Rename a cache entry (file or folder) to its new name; nothing happens when that is not possible."""
    if new.exists() or not old.exists():
        return False
    try:
        os.replace(old, new)
    except OSError:
        return False
    return True


def load_features(video: Path, track: Optional[int], cache: Path) -> AudioFeatures:
    """Decode once, storing exactly the waveform panel's feature format."""
    from . import proc

    path = waveform_cache_file(video, track, cache)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return AudioFeatures(z["mins"], z["maxs"], z["spectrogram"], z["onset"])
    raw = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                    "-map", f"0:a:{ffmpeg_audio_index(track)}", "-ac", "1", "-ar", str(RATE),
                    "-f", "s16le", "-"], capture_output=True, check=True, timeout=180).stdout
    samples = np.frombuffer(raw, dtype="<i2")
    features = compute_features(samples.astype(np.float32) / 32768.0)
    features.samples = samples
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            scratch = Path(stream.name)
            np.savez_compressed(stream, mins=features.mins, maxs=features.maxs,
                                spectrogram=features.spectrogram, onset=features.onset)
        scratch.replace(path)
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    return features


@dataclass
class AudioFeatures:
    mins: np.ndarray
    maxs: np.ndarray
    spectrogram: np.ndarray
    onset: np.ndarray
    samples: Optional[np.ndarray] = None  # decoded int16 PCM; never stored in the feature cache
    snap_activity: Optional[np.ndarray] = None  # speech hint curve; episode lifetime only


def compute_features(samples: np.ndarray, *, check_cancelled: Optional[Callable[[], None]] = None) -> AudioFeatures:
    """Analyse normalized 16 kHz mono PCM; bounded FFT chunks avoid episode-sized FFTs."""
    a = np.asarray(samples, dtype=np.float32)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("expected finite mono PCM")
    count = (len(a) + HOP_SAMPLES - 1) // HOP_SAMPLES
    mins, maxs = np.empty(count, np.float32), np.empty(count, np.float32)
    spectrum = np.empty((count, DISPLAY_BANDS), np.uint8)
    onset = np.empty(count, np.float32)
    full = len(a) // HOP_SAMPLES
    blocks = a[:full * HOP_SAMPLES].reshape(full, HOP_SAMPLES)
    mins[:full], maxs[:full] = blocks.min(axis=1), blocks.max(axis=1)
    if full < count:
        mins[-1], maxs[-1] = a[full * HOP_SAMPLES:].min(), a[full * HOP_SAMPLES:].max()
    if not count:
        return AudioFeatures(mins, maxs, spectrum, onset)
    window = np.hanning(WINDOW_SAMPLES).astype(np.float32)
    padded = np.pad(a, (WINDOW_SAMPLES // 2, WINDOW_SAMPLES // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW_SAMPLES)[::HOP_SAMPLES][:count]
    previous = np.zeros(SPECTRUM_BINS, np.float32)
    for start in range(0, count, 2048):
        if check_cancelled is not None:
            check_cancelled()
        stop = min(count, start + 2048)
        magnitude = np.abs(np.fft.rfft(frames[start:stop] * window, axis=1)).astype(np.float32)
        magnitude *= 2.0 / window.sum()
        flux = np.empty_like(magnitude)
        flux[0] = magnitude[0] - previous
        flux[1:] = np.diff(magnitude, axis=0)
        onset[start:stop] = np.maximum(flux, 0).mean(axis=1)
        previous = magnitude[-1].copy()
        sums = np.pad(np.cumsum(magnitude, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
        bands = (sums[:, _BAND_STOP] - sums[:, _BAND_START]) / (_BAND_STOP - _BAND_START)
        db = 20 * np.log10(np.maximum(bands, 1e-4))
        spectrum[start:stop] = np.rint(np.clip((db + 80) / 80, 0, 1) * 255).astype(np.uint8)
    return AudioFeatures(mins, maxs, spectrum, onset)


def pixel_peaks(samples: np.ndarray, start_ms: float, span_ms: float, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalized min/max in each half-open pixel column; empty columns are silence."""
    a = np.asarray(samples)
    if a.ndim != 1 or a.dtype != np.int16:
        raise ValueError("expected int16 mono PCM")
    if width < 0 or not np.isfinite(start_ms) or not np.isfinite(span_ms) or span_ms <= 0:
        raise ValueError("expected a finite range and nonnegative width")
    lo, hi = np.zeros(width, np.float32), np.zeros(width, np.float32)
    if not width or not len(a):
        return lo, hi
    # Ceil assigns a sample on a pixel boundary to the column on its right.
    edges = np.ceil((start_ms + np.arange(width + 1) * span_ms / width) * RATE / 1000).astype(np.int64)
    edges = np.clip(edges, 0, len(a))
    valid = edges[1:] > edges[:-1]
    positions = edges[:-1][valid]
    if len(positions):
        segment = a[int(positions[0]):int(edges[-1])]
        indices = positions - positions[0]
        lo[valid] = np.minimum.reduceat(segment, indices) / 32768.0
        hi[valid] = np.maximum.reduceat(segment, indices) / 32768.0
    return lo, hi


def window_spectrogram(samples: np.ndarray, start_ms: float, span_ms: float, *,
                       check_cancelled: Optional[Callable[[], None]] = None) -> np.ndarray:
    """Hann STFT centred at start + i * 2 ms, with the coarse display's bands/scaling.

    Only copy the requested window plus its FFT halo, zero-padding at episode edges.
    The final partial hop is included; PCM frame centres round to the nearest sample.
    """
    a = np.asarray(samples)
    if a.ndim != 1 or a.dtype != np.int16:
        raise ValueError("expected int16 mono PCM")
    if not np.isfinite(start_ms) or not np.isfinite(span_ms) or span_ms <= 0:
        raise ValueError("expected a finite positive range")
    if check_cancelled is not None:
        check_cancelled()
    count = int(np.ceil(span_ms / DETAIL_HOP_MS))
    spectrum = np.empty((count, DISPLAY_BANDS), np.uint8)
    first = int(round(start_ms * RATE / 1000)) - WINDOW_SAMPLES // 2
    hop = RATE * DETAIL_HOP_MS // 1000
    stop = first + (count - 1) * hop + WINDOW_SAMPLES
    local = np.zeros(stop - first, np.float32)
    lo, hi = max(0, first), min(len(a), stop)
    if hi > lo:
        local[lo - first:hi - first] = a[lo:hi].astype(np.float32) / 32768.0
    frames = np.lib.stride_tricks.sliding_window_view(local, WINDOW_SAMPLES)[::hop][:count]
    window = np.hanning(WINDOW_SAMPLES).astype(np.float32)
    for start in range(0, count, 512):
        if check_cancelled is not None:
            check_cancelled()
        end = min(count, start + 512)
        magnitude = np.abs(np.fft.rfft(frames[start:end] * window, axis=1)).astype(np.float32)
        magnitude *= 2.0 / window.sum()
        sums = np.pad(np.cumsum(magnitude, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
        bands = (sums[:, _BAND_STOP] - sums[:, _BAND_START]) / (_BAND_STOP - _BAND_START)
        db = 20 * np.log10(np.maximum(bands, 1e-4))
        spectrum[start:end] = np.rint(np.clip((db + 80) / 80, 0, 1) * 255).astype(np.uint8)
    if check_cancelled is not None:
        check_cancelled()
    return spectrum


def _onset_array(onset: np.ndarray) -> np.ndarray:
    a = np.asarray(onset, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError("expected a one-dimensional onset curve")
    return np.where(np.isfinite(a) & (a > 0), a, 0)


def onset_peaks(onset: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    """Prominent local-maximum times in [start, end), at least 100 ms apart.

    A plateau contributes its first frame; strength ties prefer the earlier peak.
    """
    a = _onset_array(onset)
    lo = max(0, int(np.ceil(start_ms / HOP_MS)))
    hi = min(len(a), int(np.ceil(end_ms / HOP_MS)))
    if hi <= lo:
        return np.empty(0, np.int64)
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    # Compare against a one-second neighbourhood, including outside the query.
    threshold = np.maximum(uniform_filter1d(a, 101, mode="nearest") * 1.5,
                           maximum_filter1d(a, 101, mode="nearest") * 0.25)
    left = np.r_[0, a[:-1]]
    right = np.r_[a[1:], 0]
    candidates = np.flatnonzero((a[lo:hi] > left[lo:hi]) & (a[lo:hi] >= right[lo:hi])
                               & (a[lo:hi] >= threshold[lo:hi])) + lo
    selected: list[int] = []
    for i in sorted(candidates.tolist(), key=lambda i: (-float(a[i]), i)):
        at = bisect_left(selected, i)
        if (at and i - selected[at - 1] < 10) or (at < len(selected) and selected[at] - i < 10):
            continue
        selected.insert(at, i)
    return np.asarray(selected, dtype=np.int64) * HOP_MS


def offset_edges(features: AudioFeatures, onsets: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    """Observed drops below 25% of post-onset peak amplitude for >=120 ms.

    Require >=60 ms of preceding sound and peak amplitude >=0.02. Search stops
    at the next onset; a truncated quiet run cannot supply an offset. This is
    an energy hint, not voice recognition or a proposal to change beat lengths.
    """
    lo = max(0, int(np.ceil(start_ms / HOP_MS)))
    hi = min(len(features.mins), int(np.ceil(end_ms / HOP_MS)))
    envelope = np.maximum(np.abs(features.mins[lo:hi]), np.abs(features.maxs[lo:hi]))
    result = []
    for i, onset in enumerate(onsets):
        a = max(0, int(onset // HOP_MS) - lo)
        b = min(len(envelope), int(onsets[i + 1] // HOP_MS) - lo) if i + 1 < len(onsets) else len(envelope)
        curve = envelope[a:b]
        if len(curve) < 18:
            continue
        peak = float(curve.max())
        if peak < 0.02:
            continue
        quiet = curve < peak * 0.25
        edges = np.diff(np.r_[False, quiet, False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if left >= 6 and right - left >= 12 and np.count_nonzero(~quiet[:left]) >= 6:
                result.append((lo + a + int(left)) * HOP_MS)
                break
    return np.unique(np.asarray(result, np.int64))


class BeatStart(Protocol):
    t0: Optional[int]


def suggest_beats(onset: np.ndarray, start_ms: float, end_ms: float,
                  beats: Sequence[BeatStart]) -> list[int]:
    """Return one ordered start per beat, without changing beats or their document.

    Dynamic programming balances peak strength against squared distance from the
    current starts (or uniform placeholders). Current starts remain candidates so
    silence or too few peaks never force duplicate boundaries. End is exclusive;
    an interval too short for distinct integer-ms starts returns no suggestion.
    """
    n = len(beats)
    lo, hi = int(np.ceil(start_ms)), int(np.ceil(end_ms)) - 1
    if not n or lo < 0 or hi - lo + 1 < n:
        return []
    targets = np.array([float(b.t0) if b.t0 is not None else start_ms + (end_ms - start_ms) * i / n
                        for i, b in enumerate(beats)])
    if not np.all(np.isfinite(targets)):
        raise ValueError("expected finite beat starts")
    targets = np.clip(targets, lo, hi)
    fallback = []
    for i, t in enumerate(targets):
        lower = fallback[-1] + 1 if fallback else lo
        fallback.append(max(lower, min(hi - (n - i - 1), int(round(t)))))
    a = _onset_array(onset)
    peaks = onset_peaks(a, start_ms, end_ms)
    if not len(peaks):
        return fallback
    candidates = np.unique(np.r_[peaks, fallback])
    strength = np.zeros(len(candidates), np.float64)
    peak_positions = np.searchsorted(candidates, peaks)
    values = a[peaks // HOP_MS]
    strength[peak_positions] = values / values.max()
    scale = max(60.0, (end_ms - start_ms) / n / 2)
    parents = np.full((n, len(candidates)), -1, np.int32)
    score = strength - 0.35 * ((candidates - targets[0]) / scale) ** 2
    for i in range(1, n):
        best = np.maximum.accumulate(score)
        improved = np.r_[True, score[1:] > best[:-1]]
        owners = np.maximum.accumulate(np.where(improved, np.arange(len(score)), 0))
        parents[i, 1:] = owners[:-1]
        score = np.r_[-np.inf, best[:-1]] + strength - 0.35 * ((candidates - targets[i]) / scale) ** 2
    at = int(np.argmax(score))
    result = [int(candidates[at])]
    for i in range(n - 1, 0, -1):
        at = int(parents[i, at])
        result.append(int(candidates[at]))
    return result[::-1]
