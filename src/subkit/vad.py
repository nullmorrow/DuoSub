"""Cancellable, CPU-only Silero ONNX analysis; no torch import or downloads.

Probabilities are cached separately from waveform features. Product callers must
explicitly opt into the evaluated segments; missing results retain energy hints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
from pathlib import Path
import tempfile
from typing import Callable
from zipfile import BadZipFile

import numpy as np

from .audio import AudioFeatures, HOP_MS, RATE, ffmpeg_audio_index

FRAME_SAMPLES = 512
FRAME_MS = 32
CACHE_VERSION = "silero-probabilities-v1"


@dataclass(frozen=True)
class VadResult:
    probabilities: np.ndarray
    duration_ms: float
    model_version: str


def packaged_model() -> tuple[Path, str]:
    """Locate package data without importing silero_vad (which imports torch)."""
    package = metadata.distribution("silero-vad")
    path = Path(package.locate_file("silero_vad/data/silero_vad.onnx"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, f"silero-vad-{package.version}:{digest}"


def cache_file(video: Path, track: int | None, cache: Path, model_version: str) -> Path:
    stat = video.stat()
    identity = (f"{video.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
                f"{ffmpeg_audio_index(track)}|{model_version}|{CACHE_VERSION}")
    return cache / (hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24] + ".vad.npz")


def _validate(result: VadResult) -> None:
    p = result.probabilities
    if (p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1))
            or not np.isfinite(result.duration_ms) or result.duration_ms < 0
            or len(p) != int(np.ceil(result.duration_ms / FRAME_MS))):
        raise ValueError("invalid VAD probabilities")


def load_cache(path: Path, model_version: str) -> VadResult | None:
    try:
        # Own the stream even if numpy rejects a damaged ZIP before returning.
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as values:
            result = VadResult(values["probabilities"].astype(np.float32),
                               float(values["duration_ms"]), str(values["model_version"]))
        _validate(result)
        return result if result.model_version == model_version else None
    except (FileNotFoundError, ValueError, KeyError, EOFError, BadZipFile):
        return None


def save_cache(path: Path, result: VadResult, check_cancelled: Callable[[], None]) -> None:
    _validate(result)
    check_cancelled()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, probabilities=result.probabilities,
                                duration_ms=result.duration_ms, model_version=result.model_version)
        check_cancelled()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def analyze(samples: np.ndarray, *, cache_path: Path | None = None,
            check_cancelled: Callable[[], None] = lambda: None) -> VadResult:
    """Normalized 16 kHz mono PCM, one sequential recurrent step per 32 ms.

    Each invocation owns its recurrent state. Final partial frames are zero
    padded; duration remains the exact sample count, never the padded duration.
    """
    check_cancelled()
    audio = np.asarray(samples)
    if (audio.ndim != 1 or audio.dtype.kind != "f" or not np.all(np.isfinite(audio))
            or np.any(np.abs(audio) > 1)):
        raise ValueError("expected normalized finite mono PCM")
    model, version = packaged_model()
    if cache_path is not None:
        cached = load_cache(cache_path, version)
        check_cancelled()
        if cached is not None and cached.duration_ms == len(audio) * 1000 / RATE:
            return cached
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, 64), np.float32)
    count = (len(audio) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    probabilities = np.empty(count, np.float32)
    rate = np.array(RATE, dtype=np.int64)
    try:
        for frame in range(count):
            check_cancelled()
            chunk = np.zeros((1, FRAME_SAMPLES), np.float32)
            part = audio[frame * FRAME_SAMPLES:(frame + 1) * FRAME_SAMPLES]
            chunk[0, :len(part)] = part
            inputs = np.concatenate((context, chunk), axis=1)
            output, state = session.run(None, {"input": inputs, "state": state, "sr": rate})
            probabilities[frame] = float(output[0, 0])
            context = inputs[:, -64:].copy()
        check_cancelled()
    finally:
        del session
    result = VadResult(probabilities, len(audio) * 1000 / RATE, version)
    _validate(result)
    if cache_path is not None:
        save_cache(cache_path, result, check_cancelled)
    return result


def segments(result: VadResult, *, start_threshold: float = 0.5, end_threshold: float = 0.35,
             min_speech_ms: int = 100, join_ms: int = 128) -> np.ndarray:
    """Hysteresis with bounded silence joining; discard short runs after joining."""
    _validate(result)
    if not 0 <= end_threshold < start_threshold <= 1 or min_speech_ms < 0 or join_ms < 0:
        raise ValueError("invalid VAD segment settings")
    runs = []
    start = None
    for i, probability in enumerate(result.probabilities):
        if start is None and probability >= start_threshold:
            start = i * FRAME_MS
        elif start is not None and probability < end_threshold:
            runs.append((start, min(i * FRAME_MS, result.duration_ms)))
            start = None
    if start is not None:
        runs.append((start, result.duration_ms))
    joined = []
    for a, b in runs:
        if joined and a - joined[-1][1] <= join_ms:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return np.array([(int(round(a)), int(round(b))) for a, b in joined
                     if b - a >= min_speech_ms], dtype=np.int64).reshape(-1, 2)


def refine_edges(runs: np.ndarray, features: AudioFeatures, *, radius_ms: int = 150) -> np.ndarray:
    """Use local energy slopes and onset strength, bounded by the VAD edges."""
    from .autotime import _snap_activity

    energy = _snap_activity(features)
    if not len(energy):
        return runs.copy()
    smoothed = np.convolve(np.pad(energy, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    slope = np.diff(smoothed, prepend=smoothed[0])
    refined = []
    for start, end in runs:
        pair = []
        for edge, sign in ((start, 1), (end, -1)):
            lo = max(0, int(np.ceil((edge - radius_ms) / HOP_MS)))
            hi = min(len(energy), int((edge + radius_ms) // HOP_MS) + 1)
            curve = sign * slope[lo:hi]
            if not len(curve) or np.ptp(smoothed[lo:hi]) < 0.025 or curve.max() < 0.002:
                pair.append(int(edge))
                continue
            score = curve.copy()
            if sign == 1:
                onset = features.onset[lo:hi]
                if len(onset) and onset.max() > 0:
                    score += curve.max() * 0.25 * onset / onset.max()
            times = np.arange(lo, hi) * HOP_MS
            score *= 1 - 0.25 * np.abs(times - edge) / max(1, radius_ms)
            pair.append(int(times[int(np.argmax(score))]))
        a, b = max(0, pair[0]), min(len(energy) * HOP_MS, pair[1])
        if refined:
            a = max(a, refined[-1][1])
        if b <= a:
            a, b = int(start), int(end)
            # Restoring a short run must also undo a previous extended offset.
            if refined:
                previous_start, previous_end = refined[-1]
                refined[-1] = (previous_start, min(previous_end, a))
        refined.append((a, b))
    return np.array(refined, dtype=np.int64).reshape(-1, 2)
