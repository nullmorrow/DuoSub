"""Audio waveform with the groups' time bars (translation lane above, original lane below).

Timing works like Aegisub: with a group selected, **left click = start**, **right click = end** (applied at
once, undoable).  The current group's start line is blue and its end line red; both can be dragged.
Click the time ruler (or middle click) to seek.  Pink lines are video keyframes (scene cuts); clicks and
drags snap to them and to neighbouring lines' edges (Shift = no snapping).  Ctrl+wheel zooms time, Shift+wheel zooms amplitude, the
wheel scrolls.  The bars in the lanes can still be dragged: edges change one line, the middle moves the group.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QRunnable, QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout, QLayout, QLabel, QMenu, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget, QWidgetAction)

from subkit import proc
from subkit.audio import (AudioFeatures, CACHE_VERSION, DETAIL_HOP_MS, HOP_MS, RATE,
                          compute_features, ffmpeg_audio_index, offset_edges, onset_peaks, pixel_peaks,
                          waveform_cache_file as _waveform_cache_file, window_spectrogram)
from subkit.autotime import (HINT_SEARCH_MS, _snap_activity, calibrate_hint_padding,
                            calibrate_manual_hint_padding,
                            hint_speech_edges, speech_segments, suggest_group_times)
from subkit.model.document import Document

from .. import pool as job_pool

BLOCK_MS = HOP_MS       # one peak pair / feature frame per 10 ms
DETAIL_SPAN_MS = 6000
START_COLOR = QColor("#3d8bfd")
END_COLOR = QColor("#ff5c5c")
LANE_H = 26
RULER_H = 18

# Spectrum colour maps (low -> high energy). The default is calm on the dark background.
SPECTRUM_PALETTES = {
    "deep_sea": ("深海青（推荐）", ("#0b1016", "#0d2a35", "#0f4c5c", "#2a7f88", "#6fb7b3", "#d8f0e8")),
    "soft_gray": ("柔和灰", ("#0f1318", "#1f2730", "#3a4654", "#66778a", "#a3b1bf", "#e8edf2")),
    "magma": ("暮光", ("#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf")),
    "cividis": ("色盲友好", ("#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8678", "#a59c74",
                            "#c3b369", "#e1cc55", "#fee838")),
    "classic": ("经典蓝绿黄", None),
}
DEFAULT_SPECTRUM_PALETTE = "deep_sea"
DEFAULT_SPECTRUM_CONTRAST, DEFAULT_SPECTRUM_BRIGHTNESS = 50, 60   # 0-100 each
_palette_tables: dict[tuple, list[int]] = {}


def spectrum_levels(contrast: int, brightness: int) -> np.ndarray:
    """Index remap: contrast hides the noise floor (black point up), brightness lifts the mid tones (gamma)."""
    contrast, brightness = max(0, min(100, int(contrast))), max(0, min(100, int(brightness)))
    black, white = 0.45 * contrast / 100, 1 - 0.15 * contrast / 100
    x = np.clip((np.arange(256) / 255 - black) / (white - black), 0, 1) ** (1.0 - 0.006 * brightness)
    return x * 255


def spectrum_color_table(key: str, contrast: int = 0, brightness: int = 0) -> list[int]:
    if key not in SPECTRUM_PALETTES:
        key = DEFAULT_SPECTRUM_PALETTE
    cache_key = (key, int(contrast), int(brightness))
    if cache_key not in _palette_tables:
        stops = SPECTRUM_PALETTES[key][1]
        levels = spectrum_levels(contrast, brightness)
        if stops is None:
            table = [QColor(min(255, max(0, 3 * i - 256)), min(255, 2 * i), min(255, max(24, 255 - 2 * i))).rgb()
                     for i in np.rint(levels).astype(int)]
        else:
            rgb = np.array([[int(s[k:k + 2], 16) for k in (1, 3, 5)] for s in stops], dtype=float)
            xs = np.linspace(0, 255, len(stops))
            table = [QColor(*(int(round(np.interp(v, xs, rgb[:, c]))) for c in range(3))).rgb() for v in levels]
        _palette_tables[cache_key] = table
    return _palette_tables[cache_key]
HABIT_INITIAL_MS = (100, 160)  # X2b: measured hand-timed Utena + Overlord


class _PeakSignals(QObject):
    done = Signal(object, int)       # AudioFeatures, load generation
    failed = Signal(object, int)


class _AudioJob(QRunnable):
    """Keep signal sources alive and cancel queued/running tool work before teardown."""

    def __init__(self) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.runner = proc.KillableRun()
        self.finished = threading.Event()
        self.cancelled = False
        self._publish_lock = threading.Lock()

    def cancel(self) -> None:
        with self._publish_lock:
            self.cancelled = True
        self.runner.cancel()

    def _emit(self, signal, *args) -> None:
        with self._publish_lock:
            if not self.cancelled:
                signal.emit(*args)

    def _save(self, path: Path, write) -> None:
        self.runner.check_cancelled()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as fh:
                temp = Path(fh.name)
                write(fh)
            with self._publish_lock:
                if self.cancelled:
                    raise proc.CancelledError()
                temp.replace(path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


def _audio_error(error) -> tuple[str, str]:
    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    lines = (stderr or "").strip().splitlines()
    reason = lines[-1] if lines else str(error)
    command = getattr(error, "cmd", None)
    details = subprocess.list2cmdline([str(part) for part in command]) if isinstance(command, (list, tuple)) else str(command or "")
    return reason[:80], details + "\n" + (stderr or str(error))


def _decode_samples(runner: proc.KillableRun, video: Path, track: Optional[int]) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-map", f"0:a:{ffmpeg_audio_index(track)}", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    raw = runner.run(cmd, check=True).stdout
    runner.check_cancelled()
    return np.frombuffer(raw, dtype="<i2")


class SampleJob(_AudioJob):
    """Lazy PCM decode when the episode's coarse features came from cache."""

    def __init__(self, video: Path, track: Optional[int], generation: int) -> None:
        super().__init__()
        self.video, self.track, self.generation = video, track, generation
        self.signals = _PeakSignals()

    def run(self) -> None:
        try:
            samples = _decode_samples(self.runner, self.video, self.track)
            self._emit(self.signals.done, samples, self.generation)
        except proc.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - external-tool failures
            self._emit(self.signals.failed, exc, self.generation)
        finally:
            self.finished.set()


class _DetailSignals(QObject):
    done = Signal(object, object, int, int)  # spectrum, (start, span, width), episode, request
    failed = Signal(str, object, int, int)


class SpectrogramJob(_AudioJob):
    def __init__(self, samples: np.ndarray, key: tuple, generation: int, request: int) -> None:
        super().__init__()
        self.samples, self.key = samples, key
        self.generation, self.request = generation, request
        self.signals = _DetailSignals()

    def run(self) -> None:
        try:
            self.runner.check_cancelled()
            spectrum = window_spectrogram(self.samples, self.key[0], self.key[1],
                                          check_cancelled=self.runner.check_cancelled)
            self._emit(self.signals.done, spectrum, self.key, self.generation, self.request)
        except proc.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - reported without replacing coarse data
            self._emit(self.signals.failed, str(exc), self.key, self.generation, self.request)
        finally:
            self.samples = None
            self.finished.set()


class PeakJob(_AudioJob):
    def __init__(self, video: Path, track: Optional[int], cache: Path, generation: int = 0) -> None:
        super().__init__()
        self.video, self.track, self.cache = video, track, cache
        self.generation = generation
        self.signals = _PeakSignals()

    def run(self) -> None:
        try:
            self.runner.check_cancelled()
            f = _waveform_cache_file(self.video, self.track, self.cache)
            if f.exists():
                with np.load(f, allow_pickle=False) as z:
                    features = AudioFeatures(z["mins"], z["maxs"], z["spectrogram"], z["onset"])
                self._emit(self.signals.done, features, self.generation)
                return
            samples = _decode_samples(self.runner, self.video, self.track)
            features = compute_features(samples.astype(np.float32) / 32768.0,
                                        check_cancelled=self.runner.check_cancelled)
            features.samples = samples
            self._save(f, lambda fh: np.savez_compressed(fh, mins=features.mins, maxs=features.maxs,
                                                        spectrogram=features.spectrogram, onset=features.onset))
            self._emit(self.signals.done, features, self.generation)
        except proc.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - reported in the status bar
            self._emit(self.signals.failed, exc, self.generation)
        finally:
            self.finished.set()


class _KeySignals(QObject):
    done = Signal(object, int)       # keyframe times (ms), load generation
    failed = Signal(str)


class KeyframeJob(_AudioJob):
    """Video keyframes (encoders put them on scene cuts) — drawn as lines on the waveform, like Aegisub."""

    def __init__(self, video: Path, cache: Path, generation: int = 0) -> None:
        super().__init__()
        self.video, self.cache = video, cache
        self.generation = generation
        self.signals = _KeySignals()

    def run(self) -> None:
        try:
            self.runner.check_cancelled()
            st = self.video.stat()
            key = hashlib.sha1(f"{self.video}|{st.st_size}|{int(st.st_mtime)}|kf1".encode()).hexdigest()[:20]
            f = self.cache / f"{key}.keyframes.npy"
            if f.exists():
                self._emit(self.signals.done, np.load(f, allow_pickle=False), self.generation)
                return
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                   "-of", "csv=p=0", str(self.video)]
            out = self.runner.run(cmd, text=True, check=True).stdout
            ks = []
            for line in out.splitlines():
                self.runner.check_cancelled()
                t, _, flags = line.partition(",")
                if "K" in flags:
                    try:
                        ks.append(float(t) * 1000.0)
                    except ValueError:
                        pass
            arr = np.unique(np.round(np.array(ks, dtype=np.float64)))
            self._save(f, lambda fh: np.save(fh, arr))
            self._emit(self.signals.done, arr, self.generation)
        except proc.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover
            self._emit(self.signals.failed, str(exc))
        finally:
            self.finished.set()


def fmt_ms(ms: float) -> str:
    ms = max(0, int(round(ms)))
    return f"{ms // 3600000}:{ms // 60000 % 60:02d}:{ms % 60000 / 1000:05.2f}"


class WaveformView(QWidget):
    help_anchor = "timing"
    seekRequested = Signal(int)                   # ms
    selectionChanged = Signal(int, int)
    timeClicked = Signal(int, str)                # ms, "start" | "end"   (Aegisub left / right click)
    memberTimeEdited = Signal(str, int, int)      # member id, start, end
    groupShifted = Signal(str, int)               # group id, delta ms
    groupClicked = Signal(str)
    viewChanged = Signal()                        # zoom / scroll changed (toolbar shows the visible span)
    beatsEdited = Signal(str, object)             # member id, list of Beat (K 轴 edited on the waveform)
    beatSplit = Signal(str, object, int)          # member id, part boundary key, click ms
    beatMerge = Signal(str, object)               # member id, beat boundary key
    suggestionsChanged = Signal(bool)
    playRange = Signal(int, int)                  # ms, ms (click on a beat)
    audioStatus = Signal(str)
    featuresLoaded = Signal()
    timingSuggestionChanged = Signal(bool)
    kSelectionChanged = Signal()

    MIN_SPAN, MAX_SPAN = 1000.0, 120000.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None
        self.spectrogram: Optional[np.ndarray] = None  # time x frequency, uint8 (-80..0 dB)
        self.onset: Optional[np.ndarray] = None        # 10 ms hop, same time origin as peaks
        self.display_mode = "waveform"
        self.snap_onset = False
        self.hint_lines = False
        self.lead_in_ms, self.lead_out_ms = 270, 0
        self.dialogue_records = {}
        self.dialogue_padding = (165, 250)
        self._hint_cache: dict[tuple, dict] = {}
        self._hint_group_keys: dict[str, tuple] = {}
        self.k_suggestions: Optional[tuple[str, list, str]] = None
        self._load_generation = 0
        self._jobs = []
        self._features: Optional[AudioFeatures] = None
        self.samples: Optional[np.ndarray] = None
        self._audio_source: Optional[tuple[Path, Optional[int]]] = None
        self._audio_cache: Optional[Path] = None
        self.spectrum_palette = DEFAULT_SPECTRUM_PALETTE
        self.spectrum_contrast, self.spectrum_brightness = DEFAULT_SPECTRUM_CONTRAST, DEFAULT_SPECTRUM_BRIGHTNESS
        self._hints_failed = False
        self._sample_job = None
        self._samples_failed = False
        self._detail_job = None
        self._detail_key = None
        self._detail_request = 0
        self._detail_cache: dict[tuple, np.ndarray] = {}
        self.viewChanged.connect(self._refresh_detail)
        self.viewChanged.connect(self._refresh_hints)
        self.doc: Optional[Document] = None
        self._current_gid: Optional[str] = None
        self.t0, self.span = 0.0, 12000.0          # visible window: start ms, length ms
        self.amp = 1.0                             # amplitude zoom
        self.follow = True
        self.keyframes: Optional[np.ndarray] = None
        self.snap = True                           # snap clicks/drags to keyframes and neighbouring lines
        self.timing_mode = "aegisub"
        self.karaoke_mode = False
        self.kviews: dict[str, tuple[str, object]] = {}   # lane → (member id, KView) of the current song group
        self.k_active: Optional[tuple[str, int]] = None   # (lane, beat index) being tapped
        self.k_selected: Optional[tuple[str, tuple]] = None  # member id, stable beat key
        self.k_rest_selected: Optional[tuple[str, tuple, tuple]] = None
        self.playhead = 0.0
        self.sel: Optional[tuple[float, float]] = None
        self._drag: Optional[dict] = None
        self._bars: list[tuple[QRectF, str, str, str]] = []   # rect, member id, group id, lane
        self.status_text = "（还没有加载音频）"
        self.btn_retry = QToolButton(self, text="重新读取音频")
        self.btn_retry.help_anchor = "audio-read"
        self.btn_retry.clicked.connect(self.retry_audio)
        self.btn_retry.hide()

    # ------------------------------------------------------------ data
    def load(self, video: Optional[Path], track: Optional[int], cache: Path) -> None:
        self.cancel_jobs()
        self.clear_suggestions()
        self._load_generation += 1
        generation = self._load_generation
        self._features = None
        self.mins = self.maxs = self.spectrogram = self.onset = None
        self.keyframes = None
        self._audio_cache = cache
        self._hints_failed = False
        self.setToolTip("")
        self.btn_retry.hide()
        if video is None or not Path(video).exists():
            self.status_text = "（没有视频/音频文件）"
            self.update()
            return
        self._audio_source = (Path(video), track)
        self.status_text = "正在读取音频波形…"
        job = PeakJob(Path(video), track, cache, generation)
        # Direct QObject slots preserve queued delivery after the worker finishes.
        job.signals.done.connect(self._loaded)
        job.signals.failed.connect(self._fail)
        job_pool.start(job)
        kj = KeyframeJob(Path(video), cache, generation)
        kj.signals.done.connect(self._keyframes_loaded)
        self._jobs = [job, kj]
        job_pool.start(kj)
        self.update()

    def retry_audio(self) -> None:
        if self._audio_source is None or self._audio_cache is None:
            return
        source, cache = self._audio_source, self._audio_cache
        self.cancel_jobs()
        self._audio_source = source
        try:
            _waveform_cache_file(*source, cache).unlink(missing_ok=True)
        except OSError as exc:
            self._fail(exc)
            return
        self.load(*source, cache)

    def cancel_jobs(self) -> None:
        self._load_generation += 1
        jobs, self._jobs = self._jobs, []
        deadline = time.monotonic() + 2.0
        for job in jobs:
            job.cancel()
            if job_pool.try_take(job):
                if isinstance(job, SpectrogramJob):
                    job.samples = None
                job.finished.set()
        for job in jobs:
            job.finished.wait(max(0.0, deadline - time.monotonic()))
        self.samples = None
        if self._features is not None:
            self._features.samples = None
            self._features.snap_activity = None
        self._audio_source = None
        self._sample_job = self._detail_job = None
        self._samples_failed = False
        self._detail_key = None
        self._detail_request += 1
        self._detail_cache.clear()
        self._hint_cache.clear()
        self._hint_group_keys.clear()
        self._features = None
        self.timingSuggestionChanged.emit(False)

    def closeEvent(self, e) -> None:  # noqa: N802
        self.cancel_jobs()
        super().closeEvent(e)

    def _keyframes_loaded(self, arr, generation: Optional[int] = None) -> None:
        if generation is not None and generation != self._load_generation:
            return
        self.keyframes = arr
        self._refresh_hints()

    def snap_ms(self, ms: float, *, exclude_group: Optional[str] = None, px: float = 7.0,
                edge: Optional[str] = None, beat: bool = False) -> float:
        """Nearest keyframe or neighbouring line edge within ``px`` pixels (Aegisub-style snapping)."""
        tol = px * self.span / max(1, self.width())
        best, best_d = ms, float("inf")
        if self.snap and self.keyframes is not None and len(self.keyframes):
            i = int(np.searchsorted(self.keyframes, ms))
            for j in (i - 1, i):
                if 0 <= j < len(self.keyframes):
                    d = abs(float(self.keyframes[j]) - ms)
                    if d <= tol and d < best_d:
                        best, best_d = float(self.keyframes[j]), d
        if self.snap and self.doc is not None:
            lo, hi = ms - tol, ms + tol
            for g in self.doc.groups:
                if g.id == exclude_group or g.kind not in ("dialogue", "song"):
                    continue
                for mid in g.members:
                    m = self.doc.members[mid]
                    for t in (m.start, m.end):
                        if lo <= t <= hi and abs(t - ms) < best_d:
                            best, best_d = float(t), abs(t - ms)
        if self.snap_onset:
            hints = self._nearby_hints()
            kinds = ("onset", "offset") if beat else (edge,) if self._is_dialogue(exclude_group) else ()
            hint_tol = 6 * self.span / max(1, self.width())
            for kind in kinds:
                for result in hints:
                    for t in result.get(kind, ()):
                        distance = abs(float(t) - ms)
                        if distance <= hint_tol and distance < best_d:
                            best, best_d = float(t), distance
        return best

    def _is_dialogue(self, gid: Optional[str]) -> bool:
        try:
            return self.doc is not None and self.doc.group(gid or self.current_gid).kind == "dialogue"
        except KeyError:
            return False

    def _loaded(self, features: AudioFeatures, generation: Optional[int] = None) -> None:
        if generation is not None and generation != self._load_generation:
            return
        self._hint_cache.clear()
        self._hint_group_keys.clear()
        if features.snap_activity is None:
            try:
                features.snap_activity = _snap_activity(features)
            except Exception:
                self._hints_failed = True
        self._features = features
        self.samples = features.samples
        self.mins, self.maxs = features.mins, features.maxs
        self.spectrogram, self.onset = features.spectrogram, features.onset
        self.status_text = ""
        self.setToolTip("")
        self.btn_retry.hide()
        self._refresh_detail()
        self._refresh_hints()
        self.update()
        self.featuresLoaded.emit()

    def _visible_detail_key(self) -> Optional[tuple]:
        if self.display_mode == "spectrogram" and self.samples is not None and self.span < DETAIL_SPAN_MS and self.width() > 0:
            return (self.t0, self.span, self.width())
        return None

    def _refresh_detail(self) -> None:
        # Keep running signal sources alive; discard completed, superseded jobs on the GUI thread.
        self._jobs = [job for job in self._jobs
                      if not isinstance(job, (SampleJob, SpectrogramJob)) or not job.finished.is_set()
                      or job is self._sample_job or job is self._detail_job]
        if (self.span < DETAIL_SPAN_MS and self.samples is None and self._audio_source is not None
                and self._features is not None and self._sample_job is None and not self._samples_failed):
            job = SampleJob(*self._audio_source, self._load_generation)
            job.signals.done.connect(self._samples_loaded)
            job.signals.failed.connect(self._samples_fail)
            self._sample_job = job
            self._jobs.append(job)
            job_pool.start(job)
        key = self._visible_detail_key()
        if key == self._detail_key:
            return
        self._detail_request += 1
        self._detail_key = key
        if self._detail_job is not None:
            job, self._detail_job = self._detail_job, None
            job.cancel()
            if job_pool.try_take(job):
                job.samples = None
                job.finished.set()
        if key is None or key in self._detail_cache:
            return
        job = SpectrogramJob(self.samples, key, self._load_generation, self._detail_request)
        job.signals.done.connect(self._detail_loaded)
        job.signals.failed.connect(self._detail_fail)
        self._detail_job = job
        self._jobs.append(job)
        job_pool.start(job)

    def _samples_loaded(self, samples: np.ndarray, generation: int) -> None:
        if generation != self._load_generation:
            return
        self.samples = samples
        if self._features is not None:
            self._features.samples = samples
        self._sample_job = None
        self._refresh_detail()
        self.update()

    def _samples_fail(self, error, generation: int) -> None:
        if generation == self._load_generation:
            self._sample_job = None
            self._samples_failed = True
            self.setToolTip(_audio_error(error)[1])
            self.audioStatus.emit("读取放大细节失败，显示粗波形")
            self.update()

    def _detail_loaded(self, spectrum: np.ndarray, key: tuple, generation: int, request: int) -> None:
        if (generation != self._load_generation or request != self._detail_request
                or key != self._visible_detail_key()):
            return
        self._detail_job = None
        self._detail_cache[key] = spectrum
        while len(self._detail_cache) > 4:
            del self._detail_cache[next(iter(self._detail_cache))]
        self.update()

    def _detail_fail(self, error: str, key: tuple, generation: int, request: int) -> None:
        if generation == self._load_generation and request == self._detail_request:
            self._detail_job = None
            self.audioStatus.emit("读取放大细节失败，显示粗波形")
            self.update()

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self.btn_retry.adjustSize()
        self.btn_retry.move(max(0, (self.width() - self.btn_retry.width()) // 2), self.height() // 2 + 20)
        self._refresh_detail()

    def audio_features(self) -> Optional[AudioFeatures]:
        return self._features

    @property
    def current_gid(self) -> Optional[str]:
        return self._current_gid

    @current_gid.setter
    def current_gid(self, gid: Optional[str]) -> None:
        if gid != self._current_gid:
            self.k_selected = self.k_rest_selected = None
            self.kSelectionChanged.emit()
            self.clear_suggestions()
        self._current_gid = gid
        self._refresh_hints()

    def _group_hint_key(self, group) -> tuple:
        start, end = self.doc.group_time(group)
        return (group.id, start, end, self.lead_in_ms, self.lead_out_ms)

    def _group_hints(self, group) -> dict:
        try:
            return self._compute_group_hints(group)
        except Exception:
            self._hints_failed = True
            self._hint_cache.clear()
            self._hint_group_keys.clear()
            return {"kind": group.kind, "start": [], "end": [], "onset": [], "offset": [], "suggested": None}

    def _hint_neighbours(self, group):
        groups = sorted((g for g in self.doc.groups if "hidden" not in g.flags and g.kind in ("dialogue", "song") and g.members),
                        key=lambda g: self.doc.group_time(g))
        index = next(i for i, g in enumerate(groups) if g.id == group.id)
        return (self.doc.group_time(groups[index - 1]) if index else None,
                self.doc.group_time(groups[index + 1]) if index + 1 < len(groups) else None)

    def set_dialogue_records(self, records) -> None:
        self.dialogue_records = records
        if records and self.doc is not None:
            from ..dialogue_alignment import prefetch
            prefetch(self.doc, records)
        self._hint_cache.clear()
        self._hint_group_keys.clear()
        self._refresh_hints()

    def dialogue_edge(self, group):
        from ..dialogue_alignment import group_edge
        return group_edge(self.doc, group, self.dialogue_records)

    def _compute_group_hints(self, group) -> dict:
        if "hidden" in group.flags:
            return {"kind": group.kind, "start": [], "end": [], "onset": [], "offset": [], "suggested": None}
        # Validate every time: text/ruby edits must invalidate cached edges.
        edge = self.dialogue_edge(group) if self.dialogue_records else None
        if edge is not None:
            from subkit.kalign.dialogue import suggest_times
            previous, following = self._hint_neighbours(group)
            times, edges = [], []
            if previous is not None:
                times.append(previous)
                edges.append(None)
            index = len(times)
            times.append(self.doc.group_time(group))
            edges.append(edge)
            if following is not None:
                times.append(following)
                edges.append(None)
            values = suggest_times(edges, times, self.keyframes if self.keyframes is not None else [],
                                   lead_in_ms=self.dialogue_padding[0], lead_out_ms=self.dialogue_padding[1],
                                   previous="current")
            return {"gid": group.id, "kind": group.kind, "start": [edge.speech_start],
                    "end": [edge.speech_end], "onset": [], "offset": [],
                    "suggested": values[index], "source": "aligned"}
        key = self._group_hint_key(group)
        old = self._hint_group_keys.get(group.id)
        neighbours = self._hint_neighbours(group) if group.kind == "dialogue" else (None, None)
        cached = self._hint_cache.get(key) if old == key else None
        if cached is not None and cached["kind"] == group.kind and cached.get("neighbours") == neighbours:
            return cached
        _, start, end, lead_in, lead_out = key
        result = cached if cached is not None and cached["kind"] == group.kind else {}
        runs = result.get("runs")
        result.update({"gid": group.id, "kind": group.kind, "start": [], "end": [], "onset": [],
                       "offset": [], "suggested": None, "neighbours": neighbours})
        if self._hints_failed or self._features is None:
            return result
        if group.kind == "dialogue":
            activity = self._features.snap_activity
            if runs is None:
                runs = speech_segments(activity, min(start, start + lead_in) - HINT_SEARCH_MS - 200,
                                       max(end, end - lead_out) + HINT_SEARCH_MS + 200)
            result["runs"] = runs
            a, b = hint_speech_edges(activity, start, end, runs=runs,
                                     lead_in_ms=lead_in, lead_out_ms=lead_out,
                                     previous=neighbours[0], following=neighbours[1])
            if a is not None:
                result["start"] = [a]
            if b is not None:
                result["end"] = [b]
            result["suggested"] = suggest_group_times(
                activity, start, end, lead_in_ms=lead_in, lead_out_ms=lead_out,
                previous=neighbours[0], following=neighbours[1], runs=runs)
        else:
            peaks = onset_peaks(self._features.onset, start, end)
            result["onset"] = peaks
            result["offset"] = offset_edges(self._features, peaks, start, end)
        # Replace only after computation: no empty frame or stale-snap interval.
        self._hint_cache[key] = result
        self._hint_group_keys[group.id] = key
        if old is not None and old != key:
            self._hint_cache.pop(old, None)
        return result

    def _nearby_hints(self) -> list[dict]:
        if not (self.hint_lines or self.snap_onset) or self.doc is None or (not self.dialogue_records and (self._features is None or self._hints_failed)):
            return []
        start, end = self.t0 - 800, self.t0 + self.span + 800
        results = []
        for group in self.doc.groups:
            if group.kind not in ("dialogue", "song"):
                continue
            if self.karaoke_mode and group.kind != "song":
                continue
            a, b = self.doc.group_time(group)
            if b >= start and a <= end:
                results.append(self._group_hints(group))
        return results

    def _refresh_hints(self) -> None:
        self._nearby_hints()
        self.timingSuggestionChanged.emit(self.suggested_times() is not None)
        self.update()

    def suggested_times(self) -> Optional[tuple[int | None, int | None]]:
        if not self.karaoke_mode and self.hint_lines and (self._features is not None or self.dialogue_records) and self._is_dialogue(self.current_gid):
            group = self.doc.group(self.current_gid)
            return self._group_hints(group)["suggested"]
        return None

    def _fail(self, e, generation: Optional[int] = None) -> None:
        if generation is not None and generation != self._load_generation:
            return
        reason, details = _audio_error(e)
        self.status_text = "读取音频失败：" + reason
        self.setToolTip(details)
        self.btn_retry.adjustSize()
        self.btn_retry.move(max(0, (self.width() - self.btn_retry.width()) // 2), self.height() // 2 + 20)
        self.btn_retry.show()
        self.update()

    def set_document(self, doc: Document) -> None:
        if self.doc is not doc:
            self.dialogue_records = {}
            self.k_selected = self.k_rest_selected = None
            self.kSelectionChanged.emit()
            self.clear_suggestions()
        self.doc = doc
        self._refresh_hints()
        self.update()

    def clear_suggestions(self) -> None:
        self.k_suggestions = None
        self.suggestionsChanged.emit(False)
        self.update()

    def set_suggestions(self, mid: str, beats: list, unit: str) -> None:
        self.k_suggestions = (mid, beats, unit)
        self.suggestionsChanged.emit(True)
        self.update()

    # ------------------------------------------------------------ view
    def x_of(self, ms: float) -> float:
        return (ms - self.t0) / self.span * self.width()

    def ms_of(self, x: float) -> float:
        return self.t0 + x / max(1, self.width()) * self.span

    def set_playhead(self, ms: float, follow: bool = True) -> None:
        self.playhead = ms
        if follow and self.follow and not (self.t0 + self.span * 0.05 <= ms <= self.t0 + self.span * 0.9):
            self.t0 = max(0.0, ms - self.span * 0.2)
            self.viewChanged.emit()
        self.update()

    def show_range(self, a: float, b: float) -> None:
        if b - a > self.span:
            self.t0 = max(0.0, a - self.span * 0.1)
        elif not (self.t0 <= a and b <= self.t0 + self.span):
            self.t0 = max(0.0, a - (self.span - (b - a)) / 2)
        self.viewChanged.emit()
        self.update()

    def fit_current_group(self) -> None:
        edges = self.current_edges()
        if edges is not None:
            a, b = edges
            self.set_span((b - a) * 1.4, center=(a + b) / 2)

    def set_span(self, span: float, center: Optional[float] = None) -> None:
        c = self.t0 + self.span / 2 if center is None else center
        self.span = float(np.clip(span, self.MIN_SPAN, self.MAX_SPAN))
        self.t0 = max(0.0, c - self.span / 2)
        self.viewChanged.emit()
        self.update()

    def set_amp(self, amp: float) -> None:
        self.amp = float(np.clip(amp, 1.0, 16.0))
        self.viewChanged.emit()
        self.update()

    def wheelEvent(self, e) -> None:  # noqa: N802
        d = e.angleDelta().y() or e.angleDelta().x()
        mods = e.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            c = self.ms_of(e.position().x())
            f = 0.8 if d > 0 else 1.25
            self.span = float(np.clip(self.span * f, self.MIN_SPAN, self.MAX_SPAN))
            self.t0 = max(0.0, c - (e.position().x() / max(1, self.width())) * self.span)
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            self.amp = float(np.clip(self.amp * (1.25 if d > 0 else 0.8), 1.0, 16.0))
        else:
            self.t0 = max(0.0, self.t0 - d / 120 * self.span * 0.15)
        self.viewChanged.emit()
        self.update()

    def current_edges(self) -> Optional[tuple[int, int]]:
        if self.doc is None or not self.current_gid:
            return None
        try:
            return self.doc.group_time(self.doc.group(self.current_gid))
        except KeyError:
            return None

    # ------------------------------------------------------------ paint
    def paintEvent(self, _e) -> None:  # noqa: N802
        self._refresh_detail()
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#101418"))
        wave_top, wave_bot = LANE_H + 4, h - LANE_H - RULER_H
        mid = (wave_top + wave_bot) / 2
        amp = (wave_bot - wave_top) / 2 * self.amp
        edges = self.current_edges()
        if self.display_mode == "spectrogram" and self.spectrogram is not None:
            self._paint_spectrogram(p, h - RULER_H)
            if self.kviews:
                for y0, y1 in self._k_rows().values():
                    p.fillRect(QRectF(0, y0, w, y1 - y0), QColor(0, 0, 0, 102))
        if edges:   # the current group's span, lightly shaded
            p.fillRect(QRectF(self.x_of(edges[0]), wave_top, self.x_of(edges[1]) - self.x_of(edges[0]), wave_bot - wave_top),
                       QColor(61, 139, 253, 28))
        if self.display_mode == "waveform" and self.samples is not None and self.span < DETAIL_SPAN_MS and w > 0:
            lows, highs = pixel_peaks(self.samples, self.t0, self.span, w)
            p.setPen(QPen(QColor("#3fb950"), 1))
            p.setClipRect(QRectF(0, wave_top, w, wave_bot - wave_top))
            previous = None
            for x, (lo, hi) in enumerate(zip(lows, highs)):
                top, bottom = mid - float(hi) * amp, mid - float(lo) * amp
                p.drawLine(QPointF(x, top), QPointF(x, bottom))
                if previous is not None:
                    # Join adjacent envelopes so even a single-sample impulse stays continuous.
                    p.drawLine(QPointF(x - 1, previous[0]), QPointF(x, top))
                    p.drawLine(QPointF(x - 1, previous[1]), QPointF(x, bottom))
                previous = (top, bottom)
            p.setClipping(False)
        elif self.display_mode == "waveform" and self.mins is not None and w > 0:
            b0 = int(self.t0 / BLOCK_MS)
            b1 = int((self.t0 + self.span) / BLOCK_MS) + 1
            b0c, b1c = max(0, b0), min(len(self.mins), b1)
            if b1c > b0c:
                seg_min = self.mins[b0c:b1c]
                seg_max = self.maxs[b0c:b1c]
                cols = np.linspace(0, len(seg_min), num=w + 1).astype(int)
                p.setPen(QPen(QColor("#3fb950"), 1))
                p.setClipRect(QRectF(0, wave_top, w, wave_bot - wave_top))
                for x in range(w):
                    a, b = cols[x], max(cols[x] + 1, cols[x + 1])
                    if a >= len(seg_min):
                        break
                    lo = float(seg_min[a:b].min())
                    hi = float(seg_max[a:b].max())
                    xx = x + (b0c - b0) * w / max(1, (b1 - b0))
                    p.drawLine(QPointF(xx, mid - hi * amp), QPointF(xx, mid - lo * amp))
                p.setClipping(False)
        elif self.mins is None:
            p.setPen(QColor("#8c959f"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.status_text)
        # keyframes (scene cuts): pink lines like Aegisub
        if self.keyframes is not None and len(self.keyframes):
            i0 = int(np.searchsorted(self.keyframes, self.t0))
            i1 = int(np.searchsorted(self.keyframes, self.t0 + self.span))
            p.setPen(QPen(QColor(255, 105, 180, 170), 1))
            for t_ in self.keyframes[i0:i1]:
                x = self.x_of(float(t_))
                p.drawLine(QPointF(x, LANE_H + 2), QPointF(x, h - RULER_H))
        # time ruler
        p.fillRect(QRectF(0, h - RULER_H, w, RULER_H), QColor("#161b22"))
        p.setPen(QColor("#6e7681"))
        p.setFont(QFont("Segoe UI", 8))
        for t, major, label in ruler_ticks(self.t0, self.span, w, p.fontMetrics().horizontalAdvance):
            x = self.x_of(t)
            p.drawLine(QPointF(x, h - RULER_H), QPointF(x, h - RULER_H + (7 if major else 3)))
            if label:
                p.drawText(QPointF(x + 2, h - 4), label)
        # selection
        if self.sel:
            a, b = sorted(self.sel)
            p.fillRect(QRectF(self.x_of(a), wave_top, self.x_of(b) - self.x_of(a), wave_bot - wave_top), QColor(200, 200, 200, 40))
        # group bars
        self._bars = []
        if self.doc is not None:
            lo_t, hi_t = self.t0 - 1000, self.t0 + self.span + 1000
            for g in self.doc.groups:
                if g.kind not in ("dialogue", "song"):
                    continue
                if self.karaoke_mode and g.kind != "song":
                    continue
                for mid_ in g.members:
                    m = self.doc.members[mid_]
                    if m.end < lo_t or m.start > hi_t or m.role == "comment":
                        continue
                    y = 2 if m.lane == "translation" else h - LANE_H - RULER_H
                    r = QRectF(self.x_of(m.start), y, max(2.0, self.x_of(m.end) - self.x_of(m.start)), LANE_H)
                    cur = g.id == self.current_gid
                    col = QColor("#1f6feb" if m.lane == "translation" else "#a371f7")
                    if not cur:
                        col.setAlpha(100)
                    p.fillRect(r, col)
                    if cur:
                        p.setPen(QPen(QColor("#ffffff"), 1.5))
                        p.drawRect(r)
                    self._bars.append((r, mid_, g.id, m.lane))
        self._paint_beats(p, h)
        if self.hint_lines:
            hints = self._nearby_hints()
            for kind, color in (("start", "#ffd33d"), ("end", "#ffd33d"),
                                ("onset", "#40e0d0"), ("offset", "#c9d1d9")):
                for result in hints:
                    col = QColor(color)
                    if result["kind"] == "dialogue" and result.get("gid") != self.current_gid:
                        col.setAlphaF(0.45)
                    p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
                    for t in result.get(kind, ()):
                        if self.t0 <= t <= self.t0 + self.span:
                            x, y = self.x_of(t), LANE_H + 2
                            p.drawLine(QPointF(x, y), QPointF(x, h - RULER_H))
                            if kind in ("start", "end"):
                                direction = 1 if kind == "start" else -1
                                p.setBrush(col)
                                p.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x, y + 8),
                                                         QPointF(x + direction * 6, y + 4)]))
                                p.setBrush(Qt.BrushStyle.NoBrush)
            suggested = self.suggested_times()
            if suggested:
                p.setPen(QPen(QColor("#ffffff"), 1))
                for t, label in zip(suggested, ("建议开始", "建议结束")):
                    if t is None:
                        continue
                    x = self.x_of(t)
                    p.drawLine(QPointF(x, LANE_H + 2), QPointF(x, h - RULER_H))
                    label_w = p.fontMetrics().horizontalAdvance(label)
                    p.drawText(QPointF(max(0, min(w - label_w, x + 3)), LANE_H + 14), label)
        if self.k_suggestions is not None:
            _mid, beats, _unit = self.k_suggestions
            p.setPen(QPen(QColor("#40e0d0"), 2, Qt.PenStyle.SolidLine))
            for t in sorted({t for b in beats for t in (b.t0, b.t1)}):
                x = self.x_of(t)
                p.drawLine(QPointF(x, LANE_H + 2), QPointF(x, h - RULER_H))
        # the current group's start (blue) and end (red) lines, like Aegisub
        if edges:
            for t_, col in ((edges[0], START_COLOR), (edges[1], END_COLOR)):
                x = self.x_of(t_)
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(x, 0), QPointF(x, h - RULER_H))
        # playhead
        p.setPen(QPen(QColor("#ffdf5d"), 1.5))
        x = self.x_of(self.playhead)
        p.drawLine(QPointF(x, 0), QPointF(x, h))
        self._paint_review_borders(p)

    # ------------------------------------------------------------ K 轴
    def _paint_spectrogram(self, p: QPainter, height: int) -> None:
        # Both resolutions use the same palette; low frequencies sit at the bottom.
        fine = self._detail_cache.get(self._visible_detail_key())
        if fine is not None:
            self._draw_spectrum(p, fine, self.t0, len(fine) * DETAIL_HOP_MS, height)
            return
        lo = max(0, int(self.t0 // HOP_MS))
        hi = min(len(self.spectrogram), int(np.ceil((self.t0 + self.span) / HOP_MS)))
        if hi <= lo or height <= 0:
            return
        self._draw_spectrum(p, self.spectrogram[lo:hi], lo * HOP_MS, (hi - lo) * HOP_MS, height)

    def _draw_spectrum(self, p: QPainter, spectrum: np.ndarray, start: float, span: float, height: int) -> None:
        if not len(spectrum) or height <= 0:
            return
        pixels = np.ascontiguousarray(spectrum.T[::-1])
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                       QImage.Format.Format_Indexed8)
        image.setColorTable(spectrum_color_table(self.spectrum_palette, self.spectrum_contrast, self.spectrum_brightness))
        p.drawImage(QRectF(self.x_of(start), 0, self.x_of(start + span) - self.x_of(start), height), image)

    def _k_rows(self) -> dict[str, tuple[float, float]]:
        h = self.height()
        top = LANE_H + 4
        bot = h - LANE_H - RULER_H
        row_h = 40 if self.karaoke_mode else 20
        return {"original": (top, top + row_h), "translation": (bot - row_h, bot)}

    def _paint_beats(self, p: QPainter, h: int) -> None:
        if not self.kviews:
            return
        rows = self._k_rows()
        p.setFont(QFont("Yu Gothic UI", 9))
        for lane, (mid, view) in self.kviews.items():
            y0, y1 = rows[lane]
            from subkit.core.karaoke import rests_of

            for rest in rests_of(view.beats):
                x0, x1 = self.x_of(rest.t0), self.x_of(rest.t1)
                rect = QRectF(x0, y0, x1 - x0, y1 - y0)
                selected = self.k_rest_selected == (mid, view.beats[rest.after].key, view.beats[rest.after + 1].key)
                p.fillRect(rect, QColor(61, 139, 253, 100) if selected else QColor(100, 110, 125, 35))
                p.fillRect(rect, QBrush(QColor("#8b949e"), Qt.BrushStyle.BDiagPattern))
                p.setPen(QPen(QColor("#d2e4ed") if selected else QColor("#8b949e"), 1))
                p.drawRect(rect)
            for i, b in enumerate(view.beats):
                x0, x1 = self.x_of(b.t0), self.x_of(b.t1)
                if x1 < 0 or x0 > self.width():
                    continue
                placeholder = not b.timed
                col = QColor(255, 200, 60, 70 if i % 2 else 45)
                if placeholder:
                    col = QColor(140, 140, 160, 40 if i % 2 else 25)
                if "reading_changed" in b.flags:
                    col = QColor(255, 80, 80, 70)
                if self.k_selected == (mid, b.key):
                    col = QColor(61, 139, 253, 150)
                if self.k_active == (lane, i):
                    col = QColor(255, 140, 0, 150)
                p.fillRect(QRectF(x0, y0, max(1.0, x1 - x0), y1 - y0), col)
                pen = QPen(QColor(255, 210, 80) if not placeholder else QColor(170, 170, 190), 1,
                           Qt.PenStyle.SolidLine if not placeholder else Qt.PenStyle.DashLine)
                full = lane == "original"
                for x in (x0, x1):
                    a = QPointF(x, y0 if not full else LANE_H + 2)
                    z = QPointF(x, y1 if not full else h - RULER_H)
                    p.setPen(QPen(QColor("#080b0e"), 3))
                    p.drawLine(a, z)
                    p.setPen(pen)
                    p.drawLine(a, z)
                p.setPen(QColor("#f0f6fc") if not placeholder else QColor("#8b949e"))
                if b.parts:
                    for j, (part, left, right) in enumerate(self._k_part_cells(b)):
                        if j:
                            p.drawLine(QPointF(left, y0 + 6), QPointF(left, y1 - 6))
                        self._paint_beat_label(p, QRectF(left, y0, max(1.0, right - left), y1 - y0), part.sung)
                else:
                    self._paint_beat_label(p, QRectF(x0, y0, max(1.0, x1 - x0), y1 - y0), b.sung)
            if view.beats:
                last = view.beats[-1]
                p.setPen(QPen(QColor(255, 210, 80), 1))
                x = self.x_of(last.t1)
                p.drawLine(QPointF(x, y0), QPointF(x, y1))

    def _paint_review_borders(self, p: QPainter) -> None:
        """Draw last, so suggestion/edge/playhead lines do not hide review borders."""
        rows = self._k_rows()
        p.setPen(QPen(QColor("#ff9e3d"), 2))
        for lane, (mid, view) in self.kviews.items():
            beats = list(view.beats)
            if self.k_suggestions is not None and self.k_suggestions[0] == mid:
                beats += self.k_suggestions[1]
            y0, y1 = rows[lane]
            for beat in beats:
                if "check" in beat.flags:
                    x0, x1 = self.x_of(beat.t0), self.x_of(beat.t1)
                    if x1 >= 0 and x0 <= self.width():
                        p.drawRect(QRectF(x0, y0, max(1.0, x1 - x0), y1 - y0))

    def _paint_beat_label(self, p: QPainter, cell: QRectF, text: str) -> None:
        if not text:
            return
        p.save()
        p.setClipRect(cell)
        font = p.font()
        font.setBold(True)
        p.setFont(font)
        metrics = p.fontMetrics()
        width = min(cell.width(), metrics.horizontalAdvance(text) + 6)
        height = min(cell.height(), metrics.height() + 6)
        chip = QRectF(cell.center().x() - width / 2, cell.center().y() - height / 2, width, height)
        p.fillRect(chip, QColor(16, 20, 24, 230))
        text_path = QPainterPath()
        text_path.addText(0, 0, font, text)
        bounds = text_path.boundingRect()
        text_path.translate(chip.center().x() - bounds.center().x(), chip.center().y() - bounds.center().y())
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#080b0e"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(QColor("#ffffff"))
        p.drawPath(text_path)
        p.fillPath(text_path, QColor("#ffffff"))
        p.restore()

    def selected_beat(self) -> Optional[tuple[str, int]]:
        for lane, (mid, view) in self.kviews.items():
            for i, b in enumerate(view.beats):
                if self.k_selected == (mid, b.key):
                    return lane, i
        self.k_selected = None
        return None

    def selected_rest(self) -> Optional[tuple[str, int]]:
        from subkit.core.karaoke import rests_of

        for lane, (mid, view) in self.kviews.items():
            for rest in rests_of(view.beats):
                if self.k_rest_selected == (mid, view.beats[rest.after].key, view.beats[rest.after + 1].key):
                    return lane, rest.after
        self.k_rest_selected = None
        return None

    def _k_rest_at(self, lane: str, x: float) -> Optional[int]:
        from subkit.core.karaoke import rests_of

        return next((rest.after for rest in rests_of(self.kviews[lane][1].beats)
                     if self.x_of(rest.t0) < x < self.x_of(rest.t1)), None)

    def _k_select_rest(self, lane: str, after: int) -> None:
        mid, view = self.kviews[lane]
        self.k_selected = None
        self.k_rest_selected = (mid, view.beats[after].key, view.beats[after + 1].key)
        self.kSelectionChanged.emit()
        self.update()

    def _k_part_cells(self, beat) -> list:
        """Use the same character cells for drawing and boundary hit testing."""
        x0, x1 = self.x_of(beat.t0), self.x_of(beat.t1)
        n = len(beat.parts)
        return [(p, x0 + (x1 - x0) * j / n, x0 + (x1 - x0) * (j + 1) / n)
                for j, p in enumerate(beat.parts)]

    def _k_edit_hit(self, pt: QPointF) -> Optional[tuple[str, str, dict]]:
        for lane, (y0, y1) in self._k_rows().items():
            if lane not in self.kviews or not (y0 <= pt.y() <= y1):
                continue
            beats = self.kviews[lane][1].beats
            # Existing beat boundaries take priority over inner character boundaries.
            merges = [(abs(pt.x() - self.x_of(b.t0)), b.parts[0].at)
                      for b in beats[1:] if b.parts]
            if merges:
                distance, at = min(merges, key=lambda item: item[0])
                if distance <= 5:
                    return lane, "merge", at
            splits = [(abs(pt.x() - left), part.at) for b in beats
                      if self.x_of(b.t0) < pt.x() < self.x_of(b.t1)
                      for part, left, _right in self._k_part_cells(b)[1:]]
            if splits:
                distance, at = min(splits, key=lambda item: item[0])
                if distance <= 5:
                    return lane, "split", at
        return None

    def _k_select_play(self, lane: str, i: int, *, to_end: bool = False, play: bool = True) -> None:
        mid, view = self.kviews[lane]
        beat = view.beats[i]
        end = self.doc.members[mid].end if to_end and self.doc else beat.t1
        if not play and self.doc:
            self.groupClicked.emit(self.doc.members[mid].group)
        self.k_rest_selected = None
        self.k_selected = (mid, beat.key)
        self.kSelectionChanged.emit()
        if play:
            self.playRange.emit(int(beat.t0), int(end))
        else:
            self.seekRequested.emit(int(beat.t0))
        self.update()

    def _k_hit(self, pt: QPointF) -> Optional[tuple[str, int, str]]:
        """(lane, beat index, "start" | "end" | "body" | "rest") in a K row."""
        for lane, (y0, y1) in self._k_rows().items():
            if lane not in self.kviews or not (y0 - 2 <= pt.y() <= y1 + 2):
                continue
            beats = self.kviews[lane][1].beats
            edges = [(abs(pt.x() - self.x_of(b.t0)), i, "start") for i, b in enumerate(beats)]
            edges += [(abs(pt.x() - self.x_of(b.t1)), i, "end") for i, b in enumerate(beats)
                      if i + 1 == len(beats) or beats[i + 1].t0 != b.t1]
            if edges:
                distance, i, edge = min(edges, key=lambda item: item[0])
                if distance <= 5:
                    return lane, i, edge
            rest = self._k_rest_at(lane, pt.x())
            if rest is not None:
                return lane, rest, "rest"
            for i, b in enumerate(beats):
                if self.x_of(b.t0) <= pt.x() <= self.x_of(b.t1):
                    return lane, i, "body"
        return None

    def _k_drag_apply(self, d: dict, ms: float, *, alt: bool = False) -> list:
        """New beats after dragging a boundary (Shift: the later beats move along)."""
        import copy as _copy

        lane, i, edge = d["lane"], d["index"], d["edge"]
        from subkit.core.karaoke import beat_link

        link = None
        pointer_ms = ms
        if not alt:
            ms = self.snap_ms(ms, exclude_group=self.current_gid, px=6, beat=True)
        if lane == "translation" and not alt and "original" in self.kviews:
            candidates = [(abs(self.x_of(t) - self.x_of(pointer_ms)), t, b, e)
                          for b in self.kviews["original"][1].beats
                          for t, e in ((b.t0, "start"), (b.t1, "end")) if t is not None]
            if candidates:
                distance, target, original, original_edge = min(candidates, key=lambda item: item[0])
                snapped_distance = abs(self.x_of(ms) - self.x_of(pointer_ms)) if ms != pointer_ms else float("inf")
                if distance <= 6 and distance <= snapped_distance:
                    ms, link = target, beat_link(original, original_edge)
        ms = int(round(ms))
        from subkit.core.karaoke import move_edge

        beats = move_edge(d["orig"], i, edge, ms, shift=d["shift"])
        ms = beats[i].t0 if edge == "start" else beats[i].t1
        if lane == "translation":
            targets = [(i, edge)]
            if edge == "start" and i > 0 and d["orig"][i - 1].t1 == d["orig"][i].t0:
                targets.append((i - 1, "end"))
            # A clamped drag must never claim a snap it could not reach.
            snapped = link is not None and ms == int(target)
            for k, own_edge in targets:
                if snapped:
                    beats[k].link[own_edge] = _copy.deepcopy(link)
                else:
                    beats[k].link.pop(own_edge, None)
            self.setToolTip("已吸附到原声拍（以后跟着原声拍走）" if snapped else "")
        return beats

    # ------------------------------------------------------------ mouse
    def _hit_bar(self, pt: QPointF) -> Optional[tuple[QRectF, str, str, str, str]]:
        for r, mid, gid, lane in reversed(self._bars):
            if r.adjusted(-5, 0, 5, 0).contains(pt):
                if abs(pt.x() - r.left()) <= 5:
                    return r, mid, gid, lane, "start"
                if abs(pt.x() - r.right()) <= 5:
                    return r, mid, gid, lane, "end"
                if r.contains(pt):
                    return r, mid, gid, lane, "body"
        return None

    def _in_lanes(self, pt: QPointF) -> bool:
        return pt.y() <= LANE_H + 2 or self.height() - LANE_H - RULER_H <= pt.y() < self.height() - RULER_H

    def _near_edge(self, pt: QPointF) -> Optional[str]:
        edges = self.current_edges()
        if not edges:
            return None
        if abs(pt.x() - self.x_of(edges[0])) <= 5:
            return "start"
        if abs(pt.x() - self.x_of(edges[1])) <= 5:
            return "end"
        return None

    def _group_at(self, ms: float) -> Optional[str]:
        if self.doc is None:
            return None
        candidates = [(self.doc.group_time(g)[0], g.kind == "dialogue", g.id)
                      for g in self.doc.groups if g.kind in ("dialogue", "song")
                      and self.doc.group_time(g)[0] <= ms < self.doc.group_time(g)[1]]
        return max(candidates, key=lambda item: item[:2])[2] if candidates else None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        pt = e.position()
        btn = e.button()
        if btn == Qt.MouseButton.MiddleButton or (pt.y() >= self.height() - RULER_H and btn == Qt.MouseButton.LeftButton):
            self.seekRequested.emit(int(self.ms_of(pt.x())))
            return
        if self.karaoke_mode and btn == Qt.MouseButton.RightButton:
            self._karaoke_click(pt)
            return
        if self.kviews and btn == Qt.MouseButton.LeftButton:
            if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                edit = self._k_edit_hit(pt)
                if edit is not None:
                    lane, op, at = edit
                    mid = self.kviews[lane][0]
                    if op == "split":
                        self.beatSplit.emit(mid, at, int(round(self.ms_of(pt.x()))))
                    else:
                        self.beatMerge.emit(mid, at)
                if any(y0 <= pt.y() <= y1 for lane, (y0, y1) in self._k_rows().items() if lane in self.kviews):
                    self._drag = None
                    return
            kh = self._k_hit(pt)
            if kh is not None:
                lane, i, part = kh
                mid, view = self.kviews[lane]
                if part == "rest":
                    self._k_select_rest(lane, i)
                    if self.karaoke_mode:
                        self.seekRequested.emit(int(self.ms_of(pt.x())))
                    self._drag = None
                    return
                self._drag = {"mode": "k" if part != "body" else "kbody", "lane": lane, "index": i, "edge": part,
                              "mid": mid, "orig": list(view.beats), "x0": pt.x(), "moved": False,
                              "shift": bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)}
                return
        if self.karaoke_mode:
            if btn == Qt.MouseButton.LeftButton:
                self._karaoke_click(pt, play=False)
            return
        timing_click = self.timing_mode == "select" and bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self._in_lanes(pt) and btn == Qt.MouseButton.LeftButton and not timing_click:
            hit = self._hit_bar(pt)
            if hit:
                _r, mid, gid, _lane, part = hit
                m = self.doc.members[mid]
                self._drag = {"mode": part, "mid": mid, "gid": gid, "x0": pt.x(), "start": m.start, "end": m.end, "moved": False}
                return
        if self.timing_mode == "select" and not timing_click:
            self._drag = {"mode": "select", "gid": self._group_at(self.ms_of(pt.x())),
                          "button": btn, "x0": pt.x(), "moved": False}
            return
        edge = self._near_edge(pt) if btn == Qt.MouseButton.LeftButton and not timing_click else None
        if edge:
            self._drag = {"mode": "edge", "edge": edge, "x0": pt.x(), "ms": self.ms_of(pt.x()), "moved": False}
            return
        self._drag = {"mode": "click", "button": btn, "a": self.ms_of(pt.x()), "x0": pt.x(), "moved": False}

    def _karaoke_click(self, pt: QPointF, *, play: bool = True) -> None:
        """Translation K row uses its own beat; the audio area uses the original beat."""
        self._drag = None
        if not play and self._in_lanes(pt):
            hit = self._hit_bar(pt)
            if hit:
                self.groupClicked.emit(hit[2])
                return
        rows = self._k_rows()
        lane = "translation" if rows["translation"][0] - 2 <= pt.y() <= rows["translation"][1] + 2 else "original"
        if LANE_H + 2 <= pt.y() < self.height() - LANE_H - RULER_H + 2 and lane in self.kviews:
            ms = self.ms_of(pt.x())
            for i, beat in enumerate(self.kviews[lane][1].beats):
                if beat.t0 <= ms < beat.t1:
                    self._k_select_play(lane, i, play=play)
                    return
        if not play:
            self.seekRequested.emit(int(self.ms_of(pt.x())))

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        pt = e.position()
        if self._drag is None:
            edit = self._k_edit_hit(pt) if self.kviews else None
            if edit is not None and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.setToolTip("Ctrl+点击：" + ("拆开" if edit[1] == "split" else "合并"))
                return
            self.setToolTip("")
            kh = self._k_hit(pt) if self.kviews else None
            if kh is not None:
                self.setCursor(Qt.CursorShape.SizeHorCursor if kh[2] in ("start", "end") else Qt.CursorShape.PointingHandCursor)
                return
            hit = self._hit_bar(pt) if self._in_lanes(pt) else None
            if hit:
                text = self.doc.members[hit[1]].text.replace("\\N", " ").replace("\n", " ")
                self.setToolTip(f"第 {self.doc.group_index(hit[2]) + 1} 组：{text[:20]}")
            if hit and hit[4] != "body" or self._near_edge(pt):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif hit:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif pt.y() >= self.height() - RULER_H:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return
        d = self._drag
        threshold = 5 if d["mode"] in ("body", "start", "end", "select") else 3
        if abs(pt.x() - d["x0"]) > threshold:
            d["moved"] = True
        if d["mode"] == "click":
            if d["button"] == Qt.MouseButton.LeftButton and d["moved"]:
                self.sel = (d["a"], self.ms_of(pt.x()))
        elif d["mode"] == "edge":
            d["ms"] = self.ms_of(pt.x())
        elif d["mode"] == "k":
            if d["moved"]:
                lane = d["lane"]
                mid, view = self.kviews[lane]
                view.beats = self._k_drag_apply(d, self.ms_of(pt.x()),
                                                alt=bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))
        elif d["mode"] == "kbody":
            pass
        elif d["mode"] == "select":
            pass
        else:
            if not d["moved"]:
                return
            dms = (pt.x() - d["x0"]) / max(1, self.width()) * self.span
            m = self.doc.members[d["mid"]]
            if d["mode"] == "start":
                m.start = max(0, int(min(d["start"] + dms, m.end - 10)))
            elif d["mode"] == "end":
                m.end = int(max(d["end"] + dms, m.start + 10))
            else:
                d["delta"] = int(dms)
        self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        d, self._drag = self._drag, None
        if d is None:
            return
        modifier = Qt.KeyboardModifier.AltModifier if self.timing_mode == "select" else Qt.KeyboardModifier.ShiftModifier
        free = bool(e.modifiers() & modifier)
        snap = (lambda t, edge, gid=None: t) if free else (
            lambda t, edge, gid=None: self.snap_ms(t, exclude_group=gid or self.current_gid, edge=edge))
        if d["mode"] == "click":
            if d["moved"] and self.sel:
                a, b = sorted(self.sel)
                self.selectionChanged.emit(int(a), int(b))
            elif not d["moved"]:
                self.sel = None
                if d["button"] == Qt.MouseButton.LeftButton:
                    self.timeClicked.emit(int(round(snap(d["a"], "start"))), "start")
                elif d["button"] == Qt.MouseButton.RightButton:
                    self.timeClicked.emit(int(round(snap(d["a"], "end"))), "end")
        elif d["mode"] == "select":
            if not d["moved"] and d["button"] == Qt.MouseButton.LeftButton and d["gid"]:
                self.groupClicked.emit(d["gid"])
        elif d["mode"] == "edge":
            if d["moved"]:
                self.timeClicked.emit(int(round(snap(d["ms"], d["edge"]))), d["edge"])
        elif d["mode"] == "k":
            if d["moved"]:
                beats = self._k_drag_apply(d, self.ms_of(e.position().x()),
                                           alt=bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))
                self.kviews[d["lane"]][1].beats = list(d["orig"])          # the edit goes through the undo stack
                self.beatsEdited.emit(d["mid"], beats)
            else:
                rest = self._k_rest_at(d["lane"], e.position().x())
                if rest is not None:
                    self._k_select_rest(d["lane"], rest)
                else:
                    self._k_select_play(d["lane"], d["index"], play=not self.karaoke_mode)
        elif d["mode"] == "kbody":
            if not d["moved"]:
                self._k_select_play(d["lane"], d["index"], play=not self.karaoke_mode)
        elif d["mode"] in ("start", "end"):
            m = self.doc.members[d["mid"]]
            new = (m.start, m.end)
            m.start, m.end = d["start"], d["end"]          # restore; the edit goes through the undo stack
            if d["moved"]:
                if d["mode"] == "start":
                    new = (min(int(round(snap(new[0], "start", d["gid"]))), new[1] - 10), new[1])
                else:
                    new = (new[0], max(new[0] + 10, int(round(snap(new[1], "end", d["gid"])))))
                self.memberTimeEdited.emit(d["mid"], int(new[0]), int(new[1]))
        elif d["mode"] == "body" and d["moved"] and d.get("delta"):
            self.groupShifted.emit(d["gid"], int(d["delta"]))
        if d["mode"] in ("body", "start", "end") and not d["moved"]:
            self.groupClicked.emit(d["gid"])
        self.update()

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            hit = self._k_hit(e.position())
            if hit is not None:
                self._drag = None
                if not e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    rest = self._k_rest_at(hit[0], e.position().x())
                    if rest is not None:
                        self._k_select_rest(hit[0], rest)
                    else:
                        self._k_select_play(hit[0], hit[1], to_end=not self.karaoke_mode,
                                            play=not self.karaoke_mode)
                return
        super().mouseDoubleClickEvent(e)

    def contextMenuEvent(self, e) -> None:  # noqa: N802
        e.accept()          # right click is "set end", never a menu


def _ruler_label(ms: int, precise: bool) -> str:
    minutes, seconds = divmod(ms // 10, 6000)
    return f"{minutes}:{seconds / 100:05.2f}" if precise else f"{minutes}:{seconds // 100:02d}"


def ruler_ticks(start: float, span: float, width: int, text_width) -> list[tuple[int, bool, str]]:
    """Choose two tick levels using the actual label width; never draw clipped labels."""
    if span <= 0 or width <= 0:
        return []
    for major in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 60000, 120000, 300000, 600000):
        precise = major < 1000
        label_w = max(text_width(_ruler_label(int(t), precise)) for t in (start, start + span))
        if major * width / span >= label_w + 8:
            break
    minor = max(10, major // (10 if major in (100, 200, 1000, 2000, 10000, 20000, 60000, 120000, 600000) else 5))
    ticks = []
    for t in range(math.ceil(start / minor) * minor, math.ceil((start + span) / minor) * minor, minor):
        is_major = t % major == 0
        label = _ruler_label(t, precise) if is_major else ""
        if label and (t - start) * width / span + 2 + text_width(label) > width:
            label = ""
        ticks.append((t, is_major, label))
    return ticks


class _GroupVisibility(QObject):
    """Hide a toolbar group (title included) while every one of its controls is hidden."""

    def __init__(self, group: QWidget, controls) -> None:
        super().__init__(group)
        self.group, self.controls = group, list(controls)
        for control in self.controls:
            control.installEventFilter(self)

    def sync(self) -> None:
        self.group.setVisible(any(not control.isHidden() for control in self.controls))

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            self.sync()
        return False


class FlowLayout(QLayout):
    """Wrap toolbar controls at their full size hints, without shrinking their text."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self._items = []
        self.setSpacing(6)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), False)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.sizeHint())
        left, top, right, bottom = self.getContentsMargins()
        return size + QSize(left + right, top + bottom)

    def _arrange(self, rect, apply) -> int:
        left, top, right, bottom = self.getContentsMargins()
        x0, y = rect.x() + left, rect.y() + top
        x, row_h = x0, 0
        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            if x > x0 and x + size.width() > rect.x() + rect.width() - right:
                x, y, row_h = x0, y + row_h + self.spacing(), 0
            if item.widget() is not None and item.widget().property("flow_right"):
                x = max(x, rect.right() - right - size.width() + 1)
            if apply:
                item.setGeometry(QRect(x, y, size.width(), size.height()))
            x += size.width() + self.spacing()
            row_h = max(row_h, size.height())
        return y + row_h + bottom - rect.y()


TARGETS = (("group", "整组"), ("translation", "只改翻译行"), ("original", "只改原声行"))
SCOPES = (("line", "这一句"), ("selected", "选中的句"), ("song", "整首歌"))


class _CommandButton(QToolButton):
    """A panel command owns the action later adopted by the window registry."""

    def __init__(self, *, text: str) -> None:
        super().__init__()
        self.command_enabled = True
        self.command_tooltip = ""
        self.command_status = ""
        action = QAction(text, self)
        self.setDefaultAction(action)
        # K-scene key labels belong to menus; keep the panel's original short text.
        action.changed.connect(lambda: QToolButton.setText(self, action.text().split("\t")[0]))

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        self.command_enabled = enabled
        self.defaultAction().setEnabled(enabled)

    def setText(self, text: str) -> None:  # noqa: N802
        self.defaultAction().setText(text)

    def setToolTip(self, text: str) -> None:  # noqa: N802
        self.command_tooltip = text
        self.defaultAction().setToolTip(text)

    def setStatusTip(self, text: str) -> None:  # noqa: N802
        self.command_status = text
        self.defaultAction().setStatusTip(text)

    def setVisible(self, visible: bool) -> None:  # noqa: N802
        self.defaultAction().setVisible(visible)
        super().setVisible(visible)


class WaveformPanel(QWidget):
    help_anchor = "timing"
    """Toolbar (zoom, amplitude, timing target, follow, pin) above the waveform; a K 轴 row for song groups."""

    kAction = Signal(str)                 # "start" | "import" | "clear" | "confirm"
    kAutoOption = Signal(str, object)
    kEffect = Signal(str)                 # registry key or project preset
    kParameters = Signal()
    kUnit = Signal(str)                   # syllable | mora
    speedChanged = Signal(float)
    karaokeToggled = Signal(bool)
    adoptTiming = Signal()

    def __init__(self, view: WaveformView, parent=None) -> None:
        super().__init__(parent)
        self.view = view
        self.project = None
        self._episode_padding = None
        self._manual_padding = None
        self._calibration_pending = True
        self.calibration_ms = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("waveBar")
        bar.setStyleSheet("#waveBar { background: #161b22; } QLabel { color: #c9d1d9; } QToolButton { color: #e6edf3; background: #30363d; border: 1px solid #484f58; border-radius: 3px; padding: 1px 7px; } QToolButton:checked { background: #1f6feb; border-color: #388bfd; } QToolButton:hover { border-color: #8b949e; } QToolButton::menu-indicator { image: none; }")
        bl = FlowLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        bl.setSpacing(6)
        self.mode_buttons = QButtonGroup(self)
        self.btn_waveform = QToolButton(text="波形", checkable=True)
        self.btn_spectrum = QToolButton(text="频谱", checkable=True)
        self.mode_buttons.addButton(self.btn_waveform, 0)
        self.mode_buttons.addButton(self.btn_spectrum, 1)
        self.btn_waveform.setChecked(True)
        self.mode_buttons.idClicked.connect(self._display_changed)
        self.btn_fit = QToolButton(text="适合本组")
        self.btn_fit.setToolTip("按本组长度缩放，两边各留 20%；选组时只平移，不改变缩放")
        self.btn_fit.clicked.connect(self.view.fit_current_group)
        self.lbl_span = QLabel()
        self.btn_step = QToolButton(text="逐段播放", checkable=True)
        self.btn_step.help_anchor = "playback"
        self.btn_step.setToolTip("Ctrl+Shift+Space：空格播到本组结尾或下一组完全显示时停下")
        self.speed_label = QLabel("速度（对白）")
        self.speed = QComboBox()
        for v in (1.0, 0.75, 0.5):
            self.speed.addItem(f"{v:g}×", v)
        self.speed.setToolTip("歌曲速度和对白速度分别记住；音高不变，时间按原速记录")
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(0, 100)
        self.zoom.setFixedWidth(120)
        self.zoom.setToolTip("时间缩放（Ctrl+滚轮）")
        self.amp = QSlider(Qt.Orientation.Horizontal)
        self.amp.setRange(0, 100)
        self.amp.setFixedWidth(80)
        self.amp.setToolTip("振幅缩放（Shift+滚轮）")
        self.target = QComboBox()
        self.target_label = QLabel("打轴对象")
        for key, label in TARGETS:
            self.target.addItem(label, key)
        self.target.setToolTip("左键/右键设时间时改哪些行：整组（中外文同一时间）/ 只改翻译 / 只改原声")
        self.btn_follow = QToolButton(text="跟随播放", checkable=True)
        self.btn_follow.setChecked(True)
        self.btn_pin = QToolButton(text="钉住", checkable=True)
        self.btn_pin.setToolTip("观剧模式：继续播放时也不收起波形")
        self.btn_retry = QToolButton(text="重新读取音频")
        self.btn_retry.help_anchor = "audio-read"
        self.btn_retry.clicked.connect(self.view.retry_audio)
        self.btn_snap = QToolButton(text="吸附", checkable=True)
        self.btn_snap.setChecked(True)
        self.btn_snap.setToolTip("点击/拖动时间时吸附到画面切换线（粉色）和相邻行的边界；按住 Shift 临时不吸附")
        self.btn_snap.toggled.connect(lambda v: setattr(self.view, "snap", v))
        self.btn_hint_lines = QToolButton(text="显示提示线", checkable=True)
        self.btn_hint_lines.help_anchor = "timing"
        self.btn_hint_lines.setToolTip("人声开始/结束按声音能量估计，有背景音乐时可能不准；建议时间 = 人声开始前提前一点、人声结束后延后一点，不会和前后组撞车")
        self.btn_hint_lines.toggled.connect(self._hints_changed)
        self.btn_onset = QToolButton(text="吸附到提示线", checkable=True)
        self.btn_onset.help_anchor = "timing"
        self.btn_onset.setToolTip("6 像素内吸附：对白开始/结束贴人声开始/结束，拍边界贴起音/收音，与其他目标取最近；对白按打轴方式用 Shift 或 Alt 临时不吸附，打拍用 Alt")
        self.btn_onset.toggled.connect(self._onset_changed)
        self.btn_adopt_time = QToolButton(text="采用建议时间")
        self.btn_adopt_time.help_anchor = "timing"
        self.btn_adopt_time.setToolTip("C：只改当前对白组，一步撤销；使用所选建议提前量")
        self.btn_adopt_time.setEnabled(False)
        self.btn_adopt_time.clicked.connect(self.adoptTiming.emit)
        self.view.timingSuggestionChanged.connect(self.btn_adopt_time.setEnabled)
        self.padding_label = QLabel("建议提前量")
        self.padding_mode = QComboBox()
        for label, mode in (("按我调过的行（推荐）", "manual"), ("按本集字幕", "episode"),
                            ("同自动调轴设置", "autotime"), ("固定值…", "fixed")):
            self.padding_mode.addItem(label, mode)
        self.padding_mode.activated.connect(self._padding_changed)
        self.btn_recalibrate = QToolButton(text="重新计算")
        self.btn_recalibrate.clicked.connect(self.recalibrate_hint_padding)
        for widget in (self.padding_mode, self.btn_recalibrate):
            widget.help_anchor = "timing"
        hint = QLabel()
        self.hint = hint
        hint.setWordWrap(True)
        self.btn_hint = self._hint_button(hint, "wave/hint_visible", "timing")
        self.set_timing_mode("aegisub")
        hint.setStyleSheet("color: #8b949e;")
        # Keep the existing controls as state/signal owners for menu actions.
        self.menu_controls = QWidget(self)
        self.menu_controls.hide()
        self._submenu_actions = []
        for widget in (self.target_label, self.target, self.padding_label, self.padding_mode):
            widget.setParent(self.menu_controls)
        self.btn_settings = QToolButton(text="设置 ▾")
        self.settings_menu = QMenu(self.btn_settings)
        self.settings_menu.help_anchor = "timing"
        self.btn_settings.setMenu(self.settings_menu)
        self.btn_settings.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        amp_menu = self.amp_menu = self.settings_menu.addMenu("振幅")
        self._submenu_actions.append(amp_menu.menuAction())
        amp_menu.help_anchor = "timing"
        amp_action = QWidgetAction(amp_menu)
        amp_action.setDefaultWidget(self.amp)
        amp_menu.addAction(amp_action)
        palette_menu = self.palette_menu = self.settings_menu.addMenu("频谱配色")
        self._submenu_actions.append(palette_menu.menuAction())
        palette_menu.help_anchor = "timing"
        palette_group = QActionGroup(palette_menu)
        self.palette_actions = {}
        for key, (label, _stops) in SPECTRUM_PALETTES.items():
            action = palette_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == DEFAULT_SPECTRUM_PALETTE)
            palette_group.addAction(action)
            action.triggered.connect(lambda _=False, k=key: self.set_spectrum_palette(k))
            self.palette_actions[key] = action
        palette_menu.addSeparator()
        levels = QWidget()
        form = QFormLayout(levels)
        form.setContentsMargins(12, 4, 12, 4)
        self.spectrum_sliders = {}
        for name, label in (("contrast", "对比度"), ("brightness", "亮度")):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setMinimumWidth(160)
            slider.setToolTip("对比度：压暗背景噪声；亮度：提亮中等强度的声音（人声谐波）")
            slider.valueChanged.connect(lambda _v: self._spectrum_levels_changed())
            form.addRow(label, slider)
            self.spectrum_sliders[name] = slider
        self.spectrum_sliders["contrast"].setValue(DEFAULT_SPECTRUM_CONTRAST)
        self.spectrum_sliders["brightness"].setValue(DEFAULT_SPECTRUM_BRIGHTNESS)
        levels_action = QWidgetAction(palette_menu)
        levels_action.setDefaultWidget(levels)
        palette_menu.addAction(levels_action)
        palette_menu.addAction("恢复默认对比度和亮度").triggered.connect(
            lambda: self.set_spectrum_levels(DEFAULT_SPECTRUM_CONTRAST, DEFAULT_SPECTRUM_BRIGHTNESS))
        self.target_menu = self._combo_menu(self.settings_menu, "打轴对象", self.target, "timing")
        self.menu_actions = {}
        for button in (self.btn_snap, self.btn_onset):
            self.menu_actions[button] = self._button_action(self.settings_menu, button)
        self.padding_menu = self._combo_menu(self.settings_menu, "建议提前量", self.padding_mode, "timing")
        self.padding_menu.addSeparator()
        self.menu_actions[self.btn_recalibrate] = self._button_action(self.padding_menu, self.btn_recalibrate)
        for button in (self.btn_follow, self.btn_pin, self.btn_retry):
            self.menu_actions[button] = self._button_action(self.settings_menu, button)
        self.wave_groups = []
        groups = (("视图", (self.btn_waveform, self.btn_spectrum, QLabel("缩放"), self.zoom,
                           self.lbl_span, self.btn_fit)),
                  ("播放", (self.speed_label, self.speed, self.btn_step)),
                  ("打轴", (self.btn_hint_lines, self.btn_adopt_time)),
                  ("", (self.btn_settings,)))
        for index, (title, widgets) in enumerate(groups):
            group = QWidget()
            row = QHBoxLayout(group)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            if index:
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.VLine)
                separator.setFixedSize(1, 22)
                row.addWidget(separator)
            if title:
                label = QLabel(title)
                label.setStyleSheet("color: #f0c674;")
                row.addWidget(label)
            for widget in widgets:
                row.addWidget(widget)
            bl.addWidget(group)
            self.wave_groups.append(group)
            if title:   # e.g. 打轴 has only dialogue controls; on a song group the title would stand alone
                _GroupVisibility(group, [w for w in widgets if not isinstance(w, QLabel)]).sync()
        bl.addWidget(self.btn_hint)
        self.bar = bar
        lay.addWidget(bar)
        lay.addWidget(hint)
        # ---- K 轴 row (song groups)
        kb = QWidget()
        kb.setObjectName("kBar")
        kb.setStyleSheet("#kBar { background: #1c1f26; } QLabel { color: #f0c674; } QToolButton { color: #e6edf3; background: #30363d; border: 1px solid #484f58; border-radius: 3px; padding: 1px 7px; } QToolButton:checked { background: #1f6feb; border-color: #388bfd; } QToolButton:hover { border-color: #8b949e; } QToolButton::menu-indicator { image: none; }")
        kbox = QVBoxLayout(kb)
        kbox.setContentsMargins(6, 1, 6, 1)
        kbox.setSpacing(2)
        controls = QWidget()
        kl = FlowLayout(controls)
        kl.setContentsMargins(0, 0, 0, 0)
        self.k_state = QLabel("K 轴")
        self.k_state.setWordWrap(True)
        kbox.addWidget(self.k_state)
        kbox.addWidget(controls)
        self.k_controls = controls
        self.btn_karaoke = QToolButton(text="卡拉 OK 模式", checkable=True)
        self.btn_karaoke.help_anchor = "karaoke"
        self.btn_karaoke.setToolTip("F9：卡拉 OK 布局；再按一次回到原布局")
        self.btn_karaoke.toggled.connect(self.karaokeToggled.emit)
        kl.addWidget(self.btn_karaoke)
        self.btn_k = _CommandButton(text="按住 K 跟唱")
        self.btn_k.setToolTip("从选中拍或行首前开始播放；每一拍唱的时候按住 K（按下=开始，松开=结束）；Esc 取消。播放中没有选拍时直接按住 K = 刷剧连打")
        self.k_unit = QComboBox()
        self.k_unit.addItem("音节（っ ー ん 跟前一拍）", "syllable")
        self.k_unit.addItem("拍（mora，逐拍分开）", "mora")
        self.k_fx = QComboBox()
        self.set_fx_presets([])
        self.btn_kparams = _CommandButton(text="参数…")
        self.btn_kparams.help_anchor = "karaoke-fx"
        self.btn_kparams.defaultAction().triggered.connect(self.kParameters.emit)
        self.btn_kmap = _CommandButton(text="对应…")
        self.btn_kmap.help_anchor = "karaoke"
        self.btn_kmap.defaultAction().triggered.connect(lambda: self.kAction.emit("mapping"))
        self.set_mapping_reason("（不是歌曲）")
        self.btn_kenglish = _CommandButton(text="英文按音节拆开")
        self.btn_kenglish.help_anchor = "karaoke"
        self.btn_kenglish.defaultAction().triggered.connect(lambda: self.kAction.emit("english"))
        self.set_english_available(False)
        self.btn_kimport = _CommandButton(text="读入原字幕的 K 轴")
        self.btn_kclear = _CommandButton(text="清除 K 轴")
        self.btn_kconfirm = _CommandButton(text="确认 K 轴")
        self.btn_ksuggest = QToolButton(text="自动 K 轴 ▾")
        self.btn_ksuggest.help_anchor = "karaoke-auto"
        self.kauto_menu = QMenu(self.btn_ksuggest)
        self.kauto_menu.help_anchor = "karaoke-auto"
        self.kauto_menu.setToolTipsVisible(True)
        self.btn_ksuggest.setMenu(self.kauto_menu)
        self.btn_ksuggest.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.kauto_scopes = {}
        for key, text in (("line", "这一句"), ("selected", "选中的句"), ("song", "整首歌")):
            action = self.kauto_menu.addAction(text)
            action.triggered.connect(lambda _=False, key=key: self.kAction.emit("auto:" + key))
            self.kauto_scopes[key] = action
        self.kauto_menu.addSeparator()
        self.kauto_translations = self.kauto_menu.addAction("同时重新分配翻译拍")
        self.kauto_translations.setCheckable(True)
        self.kauto_translations.triggered.connect(lambda v: self.kAutoOption.emit("translations", v))
        self.kauto_last = self.kauto_menu.addMenu("最后一拍")
        self.kauto_offset = self.kauto_menu.addMenu("拍头提前量")
        self.kauto_continuation = self.kauto_menu.addMenu("长音拍")
        self.kauto_radios = {}
        for key, menu, entries in (("last_end", self.kauto_last, (("voice", "唱完就结束（推荐）"), ("line", "延长到这一句结束"))),
                ("offset", self.kauto_offset, (("habit", "按我调过的拍（推荐）"), ("none", "不调整"), ("custom", "自定义…"))),
                ("continuation_mode", self.kauto_continuation, (("even", "平分（推荐）"), ("habit", "按我调过的比例")))):
            menu.help_anchor = "karaoke-auto"
            group = QActionGroup(menu)
            for value, text in entries:
                action = menu.addAction(text)
                action.setCheckable(True)
                group.addAction(action)
                action.setProperty("help_anchor", "karaoke-auto")
                action.triggered.connect(lambda _=False, key=key, value=value: self.kAutoOption.emit(key, value))
                self.kauto_radios[key, value] = action
        self.kauto_continuation.addSeparator()
        self.kauto_follow_notes = self.kauto_continuation.addAction("翻译跟整个长音走（推荐）")
        self.kauto_follow_notes.setCheckable(True)
        self.kauto_follow_notes.setProperty("help_anchor", "karaoke-auto")
        self.kauto_follow_notes.triggered.connect(lambda v: self.kAutoOption.emit("translation_follow_notes", v))
        self.kauto_overwrite = self.kauto_menu.addAction("覆盖手调过的句")
        self.kauto_overwrite.setCheckable(True)
        self.kauto_overwrite.triggered.connect(lambda v: self.kAutoOption.emit("overwrite", v))
        self.kauto_menu.addSeparator()
        old = self.kauto_onset = self.kauto_menu.addAction("按起音（旧方法，只这一句）")
        old.triggered.connect(lambda: self.kAction.emit("suggest"))
        for action in self.kauto_menu.actions():
            action.setProperty("help_anchor", "karaoke-auto")
        self.set_auto_options({})
        self.btn_kadopt = _CommandButton(text="采用建议")
        self.btn_kaddrest = _CommandButton(text="后面加空拍")
        self.btn_kdelrest = _CommandButton(text="删掉空拍")
        for button, action, tip in ((self.btn_kaddrest, "add_rest", "选中一拍后 Shift+K：结尾提前 0.20 秒；不足 0.40 秒时留三分之一，下一拍开始不变"),
                                    (self.btn_kdelrest, "remove_rest", "点斜线格选中空拍后 Delete：上一拍延到下一拍开始")):
            button.help_anchor = "karaoke"
            button.setToolTip(tip)
            button.defaultAction().triggered.connect(lambda _checked=False, action=action: self.kAction.emit(action))
        self.view.kSelectionChanged.connect(self.sync_rest_buttons)
        self.sync_rest_buttons()
        self.btn_kadopt.setEnabled(False)
        self.btn_kadopt.setToolTip("Enter 采用，Esc 隐藏；采用后仍需确认 K 轴")
        self.btn_khintlines = QToolButton(text="显示提示线", checkable=True)
        self.btn_khintsnap = QToolButton(text="吸附到提示线", checkable=True)
        for song, shared in ((self.btn_khintlines, self.btn_hint_lines), (self.btn_khintsnap, self.btn_onset)):
            song.help_anchor = "karaoke"
            song.setToolTip(shared.toolTip())
            song.toggled.connect(shared.setChecked)
            shared.toggled.connect(song.setChecked)
        self.view.suggestionsChanged.connect(self.btn_kadopt.setEnabled)
        khint = QLabel("点拍=选中/播放　双击=播到行尾　Ctrl+点字间/拍界=拆开/合并　K=重打")
        self.khint = khint
        self._khint_base = khint.text()
        self.btn_khint = self._hint_button(khint, "wave/khint_visible", "karaoke")
        self.btn_khint.setToolTip(khint.text())
        self.view.suggestionsChanged.connect(lambda _value: self.sync_review_hint())
        khint.setToolTip("拖拍的边界=调整；按住 Shift，后面的拍一起移动；译文拍在 6 像素内吸附到原声拍，Alt 临时不吸附")
        khint.setStyleSheet("color: #8b949e;")
        self.btn_more = QToolButton(text="更多 ▾")
        self.more_menu = QMenu(self.btn_more)
        self.more_menu.help_anchor = "karaoke"
        self.btn_more.setMenu(self.more_menu)
        self.btn_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        lyrics_menu = self.lyrics_menu = self.more_menu.addMenu("歌词")
        self._submenu_actions.append(lyrics_menu.menuAction())
        lyrics_menu.help_anchor = "karaoke"
        for button in (self.btn_kmap, self.btn_kenglish, self.btn_kimport):
            self.menu_actions[button] = self._button_action(lyrics_menu, button)
        # Lyric commands with a scope: kAction "<kind>:line|selected|song".
        self.scope_menus, self.scope_actions = {}, {}
        for kind, title, dots in (("redistribute", "重新分配翻译拍", ""), ("align_times", "翻译时间对齐原声", ""),
                                  ("fade", "淡入淡出", "…")):
            menu = self.scope_menus[kind] = lyrics_menu.addMenu(title)
            menu.help_anchor = "karaoke"
            menu.menuAction().setProperty("help_anchor", "karaoke")
            actions = self.scope_actions[kind] = {}
            for key, text in SCOPES:
                action = menu.addAction(text + dots)
                action.setProperty("help_anchor", "karaoke")
                action.triggered.connect(lambda _=False, kind=kind, key=key: self.kAction.emit(f"{kind}:{key}"))
                actions[key] = action
        self.scope_menus["align_times"].setToolTipsVisible(True)
        self.scope_menus["align_times"].menuAction().setToolTip("翻译用原声的开始和结束时间，两行一起淡入淡出")
        self.redistribute_menu = self.scope_menus["redistribute"]
        self.redistribute_actions = self.scope_actions["redistribute"]
        self.k_fx.setParent(self.menu_controls)
        fx_menu = self.fx_menu = self._combo_menu(self.more_menu, "特效", self.k_fx, "karaoke-fx")
        fx_menu.addSeparator()
        self.menu_actions[self.btn_kparams] = self._button_action(fx_menu, self.btn_kparams)
        groups = (("打拍", (self.btn_k, QLabel("分拍"), self.k_unit)),
                  ("提示", (self.btn_khintlines, self.btn_khintsnap, self.btn_ksuggest, self.btn_kadopt)),
                  ("空拍", (self.btn_kaddrest, self.btn_kdelrest)),
                  ("", (self.btn_more, self.btn_kclear, self.btn_kconfirm)))
        self.k_groups = []
        for index, (label, widgets) in enumerate(groups):
            group = QWidget()
            row = QHBoxLayout(group)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            if index:
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.VLine)
                separator.setFixedSize(1, 22)
                row.addWidget(separator)
            if label:
                row.addWidget(QLabel(label))
            for widget in widgets:
                row.addWidget(widget)
            kl.addWidget(group)
            self.k_groups.append(group)
            _GroupVisibility(group, [w for w in widgets if not isinstance(w, QLabel)]).sync()
        kl.addWidget(self.btn_khint)
        khint.setWordWrap(True)
        kbox.addWidget(khint)
        self.kbar = kb
        kb.setVisible(False)
        lay.addWidget(kb)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet("background: #161b22; color: #c9d1d9; padding: 3px 6px;")
        self.legend.help_anchor = "timing"
        lay.addWidget(self.legend)
        self.view.featuresLoaded.connect(self._calibrate_if_pending)
        self.view.timingSuggestionChanged.connect(lambda _available: self._update_legend())
        self._update_legend()
        lay.addWidget(view, 1)
        self.btn_kadopt.defaultAction().triggered.connect(lambda: self.kAction.emit("adopt"))
        self.btn_k.defaultAction().triggered.connect(lambda: self.kAction.emit("start"))
        self.btn_kimport.defaultAction().triggered.connect(lambda: self.kAction.emit("import"))
        self.btn_kclear.defaultAction().triggered.connect(lambda: self.kAction.emit("clear"))
        self.btn_kconfirm.defaultAction().triggered.connect(lambda: self.kAction.emit("confirm"))
        self.k_fx.activated.connect(lambda _i: self.kEffect.emit(self.k_fx.currentData()))
        self.k_unit.activated.connect(lambda _i: self.kUnit.emit(self.k_unit.currentData()))
        self.speed.activated.connect(lambda _i: self.speedChanged.emit(float(self.speed.currentData())))
        self.zoom.valueChanged.connect(self._zoom_changed)
        self.amp.valueChanged.connect(lambda v: self.view.set_amp(1.0 * (16.0 ** (v / 100.0))))
        self.btn_follow.toggled.connect(lambda v: setattr(self.view, "follow", v))
        self.view.viewChanged.connect(self._sync)
        self._sync()

    def _hint_button(self, hint: QLabel, key: str, anchor: str) -> QToolButton:
        button = QToolButton(text="?", checkable=True)
        button.help_anchor = anchor
        button.setProperty("flow_right", True)
        button.setChecked(True)

        def toggle(visible):
            hint.setVisible(visible)
            if hasattr(self, "settings"):
                self.settings.setValue(key, visible)

        button.toggled.connect(toggle)
        return button

    def _button_action(self, menu, button):
        button.setParent(self.menu_controls)
        if isinstance(button, _CommandButton):
            action = button.defaultAction()
            action.setProperty("help_anchor", getattr(button, "help_anchor", menu.help_anchor))
            menu.addAction(action)
            return action
        action = menu.addAction(button.text())
        action.setCheckable(button.isCheckable())
        action.setProperty("help_anchor", getattr(button, "help_anchor", menu.help_anchor))
        action.triggered.connect(lambda _checked: button.click())
        button.toggled.connect(action.setChecked)

        def sync():
            action.setText(button.text())
            action.setToolTip(button.toolTip())
            action.setStatusTip(button.toolTip())
            action.setEnabled(button.isEnabled())
            action.setChecked(button.isChecked())
            action.setVisible(not button.isHidden())

        menu.aboutToShow.connect(sync)
        sync()
        return action

    def _combo_menu(self, parent, title, combo, anchor):
        menu = parent.addMenu(title)
        menu.help_anchor = anchor
        menu_action = menu.menuAction()
        self._submenu_actions.append(menu_action)
        menu_action.setProperty("help_anchor", anchor)
        group = QActionGroup(menu)
        actions = []

        def choose(index):
            combo.setCurrentIndex(index)
            combo.activated.emit(index)
            sync()

        def sync():
            # Presets can be added/renamed when the project changes.
            while len(actions) > combo.count():
                action = actions.pop()
                menu.removeAction(action)
                action.deleteLater()
            while len(actions) < combo.count():
                index = len(actions)
                following = next((a for a in menu.actions() if a not in actions), None)
                action = menu.addAction("")
                # Insert new presets before any trailing parameter actions.
                if following is not None:
                    menu.insertAction(following, action)
                action.setCheckable(True)
                action.setProperty("help_anchor", anchor)
                group.addAction(action)
                action.triggered.connect(lambda _checked, i=index: choose(i))
                actions.append(action)
            for index, action in enumerate(actions):
                action.setText(combo.itemText(index))
                action.setChecked(index == combo.currentIndex())
                action.setEnabled(combo.isEnabled())
                action.setToolTip(combo.toolTip())

        menu.aboutToShow.connect(sync)
        combo.currentIndexChanged.connect(lambda _index: sync())
        sync()
        return menu

    def sync_rest_buttons(self) -> None:
        selected = self.view.selected_beat()
        can_add = selected is not None and selected[1] + 1 < len(self.view.kviews[selected[0]][1].beats)
        self.btn_kaddrest.setEnabled(can_add)
        self.btn_kdelrest.setEnabled(self.view.selected_rest() is not None)

    def set_english_available(self, available: bool) -> None:
        reason = "" if available else "（这一行没有能拆的英文词）"
        self.btn_kenglish.setEnabled(available)
        self.btn_kenglish.setText("英文按音节拆开" + reason)
        self.btn_kenglish.setToolTip(reason or "原声和翻译中的多音节英文词拆成占位拍，需重打或拖动；Ctrl+Z 撤销")

    def set_redistribute_scopes(self, reasons):
        self.set_scope_reasons("redistribute", reasons)

    def set_scope_reasons(self, kind, reasons):
        dots = "…" if kind == "fade" else ""
        for key, label in SCOPES:
            action = self.scope_actions[kind][key]
            reason = reasons[key]
            action.setEnabled(not reason)
            action.setText(label + (reason or dots))
            action.setToolTip(reason)

    def sync_review_hint(self):
        beats = [b for _mid, view in self.view.kviews.values() for b in view.beats]
        if self.view.k_suggestions is not None:
            mid, suggested, _unit = self.view.k_suggestions
            if any(mid == row_mid for row_mid, _view in self.view.kviews.values()):
                beats += suggested
        extra = "　橙色框 = 两种识别结果不一致，请试听" if any("check" in b.flags for b in beats) else ""
        self.khint.setText(self._khint_base + extra)
        self.btn_khint.setToolTip(self.khint.text())

    def set_mapping_reason(self, reason: str) -> None:
        self.btn_kmap.setEnabled(not reason)
        self.btn_kmap.setText("对应…" + reason)
        self.btn_kmap.setToolTip(reason or "歌词对应：翻译拍跟着原声拍走")

    def set_timing_mode(self, mode: str) -> None:
        self.view.timing_mode = mode if mode in ("aegisub", "select") else "aegisub"
        if self.view.karaoke_mode:
            text = "左键=选中/跳转  右键=播放这一拍  拖边界=调拍  Alt=不吸附"
        elif self.view.timing_mode == "select":
            text = "左键点块=选中  Shift+左键=开始  Shift+右键=结束  Alt=不吸附"
        else:
            text = "左键=开始  右键=结束  点上下色条=选中那一组"
        self.hint.setText(text if self.view.karaoke_mode else
                          text + "  点刻度=跳转  PgUp/PgDn=上一组/下一组  粉线=画面切换")
        self.hint.setToolTip(self.hint.text())
        self.btn_hint.setToolTip(self.hint.text())
        modifier = "Alt" if self.view.timing_mode == "select" else "Shift"
        self.btn_snap.setToolTip(f"点击/拖动时间时吸附到画面切换线（粉色）和相邻行的边界；按住 {modifier} 临时不吸附")

    def set_karaoke_mode(self, enabled: bool) -> None:
        self.view.karaoke_mode = enabled
        self.view._bars = []
        self.btn_karaoke.blockSignals(True)
        self.btn_karaoke.setChecked(enabled)
        self.btn_karaoke.blockSignals(False)
        self.target_menu.menuAction().setVisible(not enabled)
        self.btn_adopt_time.setVisible(not enabled)
        self.set_timing_mode(self.view.timing_mode)
        self.view.update()

    def set_speed(self, kind: str, speed: float) -> None:
        self.speed_label.setText("速度（歌曲）" if kind == "song" else "速度（对白）")
        self.speed.setCurrentIndex(max(0, self.speed.findData(speed)))

    def set_fx_presets(self, presets):
        from subkit.karaoke_fx import REGISTRY

        current = self.k_fx.currentData()
        self.k_fx.clear()
        for key, fx in REGISTRY.items():
            self.k_fx.addItem(fx.display_name, key)
        for preset in presets:
            self.k_fx.addItem(preset["name"], "preset:" + preset["name"])
        self.k_fx.setCurrentIndex(max(0, self.k_fx.findData(current)))

    def set_settings(self, settings) -> None:
        self.settings = settings
        for key, button, hint in (("wave/hint_visible", self.btn_hint, self.hint),
                                  ("wave/khint_visible", self.btn_khint, self.khint)):
            visible = settings.value(key, True, type=bool)
            button.setChecked(visible)
            hint.setVisible(visible)
        for key, value in zip(("lead_in_ms", "lead_out_ms"), HABIT_INITIAL_MS):
            if not settings.contains("waveform/habit/" + key):
                settings.setValue("waveform/habit/" + key, value)
        self.set_spectrum_palette(settings.value("waveform/spectrum_palette", DEFAULT_SPECTRUM_PALETTE), save=False)
        self.set_spectrum_levels(settings.value("waveform/spectrum_contrast", DEFAULT_SPECTRUM_CONTRAST, type=int),
                                 settings.value("waveform/spectrum_brightness", DEFAULT_SPECTRUM_BRIGHTNESS, type=int),
                                 save=False)
        mode = settings.value("waveform/display_mode", "waveform")
        self._display_changed(1 if mode == "spectrogram" else 0)
        self.btn_onset.setChecked(settings.value("karaoke/snap_onset", False, type=bool))
        self.btn_hint_lines.setChecked(settings.value("waveform/hint_lines", False, type=bool))
        self.sync_hint_padding()

    def set_spectrum_palette(self, key: str, *, save: bool = True) -> None:
        key = key if key in SPECTRUM_PALETTES else DEFAULT_SPECTRUM_PALETTE
        self.view.spectrum_palette = key
        self.palette_actions[key].setChecked(True)
        if save and hasattr(self, "settings"):
            self.settings.setValue("waveform/spectrum_palette", key)
        self.view.update()

    def set_spectrum_levels(self, contrast: int, brightness: int, *, save: bool = True) -> None:
        for name, value in (("contrast", contrast), ("brightness", brightness)):
            slider = self.spectrum_sliders[name]
            slider.blockSignals(True)
            slider.setValue(max(0, min(100, int(value))))
            slider.blockSignals(False)
        self._spectrum_levels_changed(save=save)

    def _spectrum_levels_changed(self, *, save: bool = True) -> None:
        self.view.spectrum_contrast = self.spectrum_sliders["contrast"].value()
        self.view.spectrum_brightness = self.spectrum_sliders["brightness"].value()
        if save and hasattr(self, "settings"):
            self.settings.setValue("waveform/spectrum_contrast", self.view.spectrum_contrast)
            self.settings.setValue("waveform/spectrum_brightness", self.view.spectrum_brightness)
        self.view.update()

    def _display_changed(self, index: int) -> None:
        self.view.display_mode = "spectrogram" if index == 1 else "waveform"
        self.mode_buttons.button(index).setChecked(True)
        if hasattr(self, "settings"):
            self.settings.setValue("waveform/display_mode", self.view.display_mode)
        self.view.update()

    def _onset_changed(self, enabled: bool) -> None:
        self.view.snap_onset = enabled
        if hasattr(self, "settings"):
            self.settings.setValue("karaoke/snap_onset", enabled)
        self.view._refresh_hints()

    def _hints_changed(self, enabled: bool) -> None:
        self.view.hint_lines = enabled
        self._update_legend()
        if hasattr(self, "settings"):
            self.settings.setValue("waveform/hint_lines", enabled)
        self.view._refresh_hints()
        self.view.update()

    def _autotime_padding(self) -> tuple[int, int]:
        if hasattr(self, "settings"):
            return (int(round(self.settings.value("autotime/lead_in", 0.27, type=float) * 1000)),
                    int(round(self.settings.value("autotime/lead_out", 0.0, type=float) * 1000)))
        return (270, 0)

    def _habit_padding(self) -> tuple[int, int]:
        if hasattr(self, "settings"):
            return tuple(self.settings.value("waveform/habit/" + key, value, type=int)
                         for key, value in zip(("lead_in_ms", "lead_out_ms"), HABIT_INITIAL_MS))
        return HABIT_INITIAL_MS

    def _manual_timing_inputs(self):
        yield self.view._features.snap_activity, self.view.doc
        if self.project is None or not hasattr(self.project, "root"):
            return
        from subkit.model.serialize import load_document, read_json

        # Edited episodes already have waveform caches. Read them without opening
        # an Episode (which can migrate/save documents) or decoding other media.
        for entry in self.project.data.get("episodes", []):
            if entry["id"] == self.view.doc.episode:
                continue
            folder = self.project.root / entry["path"]
            try:
                doc = load_document(folder / "document.json")
                if not any(g.kind == "dialogue" and (g.state == "confirmed" or
                           any(doc.members[mid].extra.get("time_manual") for mid in g.members))
                           for g in doc.groups):
                    continue
                media = read_json(folder / "episode.json")["media"]
                cache = _waveform_cache_file(Path(media["video"]["path"]), media.get("audio_track"),
                                             self.project.cache_dir / "waveform")
                if not cache.exists():
                    continue
                with np.load(cache, allow_pickle=False) as z:
                    features = AudioFeatures(*(z[k] for k in ("mins", "maxs", "spectrogram", "onset")))
                yield _snap_activity(features), doc
            except (OSError, ValueError, KeyError):
                continue

    def _padding_settings(self) -> dict:
        return self.project.data.get("settings", {}).get("hint_padding", {}) if self.project else {}

    def set_auto_options(self, options):
        self.kauto_translations.setChecked(options.get("translations", True))
        self.kauto_overwrite.setChecked(options.get("overwrite", False))
        self.kauto_follow_notes.setChecked(options.get("translation_follow_notes", True))
        for (key, value), action in self.kauto_radios.items():
            current = options.get(key, {"last_end": "voice", "offset": "habit", "continuation_mode": "even"}[key])
            action.setChecked(current == value or value == "custom" and type(current) is int)

        current = options.get("offset", "habit")
        self.kauto_radios["offset", "custom"].setText(
            f"自定义…（{current:+d} ms）" if type(current) is int else "自定义…")

    def set_auto_scopes(self, line, selected, song, available):
        tip = "需要安装 torch、torchaudio、transformers、demucs（见说明「自动 K 轴」）"
        for key, enabled in (("line", line), ("selected", selected), ("song", song)):
            action = self.kauto_scopes[key]
            action.setEnabled(bool(enabled and available))
            action.setToolTip("" if available else tip)
        self.btn_ksuggest.setToolTip("" if available else tip)

    def set_hint_context(self, project) -> None:
        self.project = project
        self._episode_padding = None
        self._manual_padding = None
        self._dialogue_habit = None
        self._calibration_pending = True
        saved = self._padding_settings()
        mode = saved.get("mode", "manual")
        # Before X2b only a user activation persisted mode; its presence proves
        # an explicit choice. A marked implicit episode default may migrate.
        if mode == "episode" and saved.get("mode_explicit") is False:
            mode = "manual"
            self._save_padding(mode=mode)

        self.padding_mode.setCurrentIndex(max(0, self.padding_mode.findData(mode)))
        self.sync_hint_padding()

    def _save_padding(self, **values) -> None:
        if self.project is not None:
            self.project.data.setdefault("settings", {}).setdefault("hint_padding", {}).update(values)
            self.project.save()

    def _calibrate_if_pending(self) -> None:
        if self._calibration_pending:
            self.recalibrate_hint_padding()

    def recalibrate_hint_padding(self) -> None:
        self.recalibrate_dialogue_padding()
        features, doc = self.view._features, self.view.doc
        if features is None or features.snap_activity is None or doc is None:
            self._calibration_pending = True
            return
        started = time.perf_counter()
        mode = self.padding_mode.currentData()
        if mode == "manual":
            self._manual_padding = calibrate_manual_hint_padding(self._manual_timing_inputs(), self._habit_padding())
            if self._manual_padding.source == "manual" and hasattr(self, "settings"):
                self.settings.setValue("waveform/habit/lead_in_ms", self._manual_padding.lead_in_ms)
                self.settings.setValue("waveform/habit/lead_out_ms", self._manual_padding.lead_out_ms)
        elif mode == "episode":
            last = self._padding_settings().get("last_episode")
            self._episode_padding = calibrate_hint_padding(
                features.snap_activity, (doc.group_time(g) for g in doc.groups if g.kind == "dialogue"),
                last_project=tuple(last) if last is not None else None, autotime_values=self._autotime_padding())
            if self._episode_padding.source == "episode":
                self._save_padding(last_episode=[self._episode_padding.lead_in_ms, self._episode_padding.lead_out_ms])
        self.calibration_ms = (time.perf_counter() - started) * 1000
        self._calibration_pending = False
        self.sync_hint_padding()

    def _fixed_padding_dialog(self, values) -> tuple[int, int] | None:
        from subkit_app.help import dialog_buttons

        dialog = QDialog(self)
        dialog.help_anchor = "timing"
        dialog.setWindowTitle("固定值…")
        layout = QFormLayout(dialog)
        lead_in, lead_out = QSpinBox(), QSpinBox()
        lead_in.setRange(0, 600)
        lead_out.setRange(-200, 600)
        lead_in.setValue(values[0])
        lead_out.setValue(values[1])
        layout.addRow("提前（毫秒）", lead_in)
        layout.addRow("延后（毫秒）", lead_out)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        dialog_buttons(dialog, buttons)
        try:
            return (lead_in.value(), lead_out.value()) if dialog.exec() == QDialog.DialogCode.Accepted else None
        finally:
            dialog.deleteLater()

    def _padding_changed(self, _index: int) -> None:
        mode = self.padding_mode.currentData()
        if mode == "fixed":
            values = self._fixed_padding_dialog(self._padding_settings().get(
                "fixed", (self.view.lead_in_ms, self.view.lead_out_ms)))
            if values is None:
                previous = self._padding_settings().get("mode", "manual")
                self.padding_mode.setCurrentIndex(max(0, self.padding_mode.findData(previous)))
                return
            self._save_padding(mode=mode, mode_explicit=True, fixed=list(values))
        else:
            self._save_padding(mode=mode, mode_explicit=True)
        if mode in ("manual", "episode"):
            self.recalibrate_hint_padding()
        self.sync_hint_padding()

    def recalibrate_dialogue_padding(self) -> None:
        from subkit.kalign.dialogue import learn_habit
        from ..dialogue_alignment import habit_samples, load_records
        edges, rows = [], []
        if self.view.doc is not None:
            a, b = habit_samples(self.view.doc, self.view.dialogue_records)
            edges.extend(a)
            rows.extend(b)
        if self.project is not None and hasattr(self.project, "root"):
            from subkit.model.serialize import load_document, read_json
            for entry in self.project.data.get("episodes", []):
                if self.view.doc is not None and entry["id"] == self.view.doc.episode:
                    continue
                folder = self.project.root / entry["path"]
                try:
                    media = read_json(folder / "episode.json")["media"]
                    records = load_records(media["video"]["path"], media.get("audio_track"), self.project.cache_dir)
                    if not records:
                        continue          # never analysed: no samples, so do not load its document
                    doc = load_document(folder / "document.json")
                    a, b = habit_samples(doc, records)
                    edges.extend(a)
                    rows.extend(b)
                except (OSError, ValueError, TypeError, KeyError):
                    continue
        settings = self.project.data.get("settings", {}).get("dialogue_align", {}) if self.project else {}
        self._dialogue_habit = learn_habit(edges, rows, stored=settings.get("habit"))
        if self.project is not None and self._dialogue_habit[2] == "habit":
            values = list(self._dialogue_habit[:2])
            if settings.get("habit") != values:
                self.project.data.setdefault("settings", {}).setdefault("dialogue_align", {})["habit"] = values
                self.project.save()
        self.sync_hint_padding()

    def sync_hint_padding(self) -> None:
        mode, settings = self.padding_mode.currentData(), self._padding_settings()
        values, self.padding_source = self._autotime_padding(), "同自动调轴设置"
        if mode == "fixed":
            values, self.padding_source = settings.get("fixed", values), "手动设定"
        elif mode == "manual":
            values, self.padding_source = self._habit_padding(), "按我的习惯"
            if self._manual_padding is not None and self._manual_padding.source == "manual":
                values = (self._manual_padding.lead_in_ms, self._manual_padding.lead_out_ms)
                self.padding_source = "按我调过的行算出"
        elif mode == "episode":
            if self._episode_padding is not None:
                if self._episode_padding.source != "autotime":
                    values = (self._episode_padding.lead_in_ms, self._episode_padding.lead_out_ms)
                    self.padding_source = "按本集字幕算出"
            elif settings.get("last_episode") is not None:
                values, self.padding_source = settings["last_episode"], "按本集字幕算出"
        self.view.lead_in_ms, self.view.lead_out_ms = values
        self.dialogue_padding_source = self.padding_source
        self.view.dialogue_padding = tuple(values)
        if mode == "manual":
            habit = getattr(self, "_dialogue_habit", None) or (165, 250, "default", 0, 0)
            self.view.dialogue_padding = habit[:2]
            self.dialogue_padding_source = {"habit": "按我调过的行算出", "stored": "按我的对齐习惯", "default": "默认值"}[habit[2]]
        self.btn_recalibrate.setEnabled(mode in ("manual", "episode"))
        self._update_legend()
        self.view._refresh_hints()

    def _update_legend(self) -> None:
        if not hasattr(self, "legend"):
            return
        self.legend.setVisible(self.view.hint_lines)
        source = getattr(self, "padding_source", "同自动调轴设置")
        aligned = (self.view._is_dialogue(self.view.current_gid)
                   and self.view.dialogue_edge(self.view.doc.group(self.view.current_gid)) is not None)
        values = self.view.dialogue_padding if aligned else (self.view.lead_in_ms, self.view.lead_out_ms)
        if aligned:
            source = getattr(self, "dialogue_padding_source", "默认值")
        origin = "按台词对齐" if aligned else "按人声能量"
        out = values[1]
        tail = f"结束前 {-out} 毫秒" if out < 0 else f"结束后 {out} 毫秒"
        self.legend.setText(
            '<span style="color:#ffd33d">■ ▷ 人声开始　■ ◁ 人声结束</span>　'
            f'<span style="color:#ffffff">■ │ 建议时间：提前 {values[0]} 毫秒、{tail}（{source}）；建议来源：{origin}</span>　'
            '<span style="color:#40e0d0">■ │ 起音</span>　<span style="color:#c9d1d9">■ │ 收音</span>　'
            '<span style="color:#f778ba">■ 粉线 画面切换</span>')

    def show_k(self, view, *, has_tags: bool) -> None:
        """Show the K row for a song group (view = the original lyric's KView, or None)."""
        self.sync_review_hint()
        self.kbar.setVisible(view is not None or self.view.karaoke_mode)
        self.btn_hint_lines.setVisible(view is None)
        self.btn_adopt_time.setVisible(view is None)
        if view is None:
            return
        state = {"none": "未打", "placeholder": "占位/未完成", "suggested": "建议", "timed": "已打", "confirmed": "已确认"}.get(view.state, view.state)
        n = len(view.beats)
        self.k_state.setText(f"K 轴：{n} 拍 · {state}" + (" · " + "；".join(view.issues) if view.issues else ""))
        for box, val in ((self.k_unit, view.unit), (self.k_fx, view.kind)):
            i = box.findData(val)
            if i >= 0:
                box.setCurrentIndex(i)
        self.btn_kimport.setVisible(has_tags)

    def timing_target(self) -> str:
        return self.target.currentData() or "group"

    def pinned(self) -> bool:
        return self.btn_pin.isChecked()

    def _zoom_changed(self, v: int) -> None:
        lo, hi = math.log(WaveformView.MIN_SPAN), math.log(WaveformView.MAX_SPAN)
        span = math.exp(lo + (hi - lo) * v / 100.0)
        if abs(span - self.view.span) > 1:
            self.view.set_span(span)

    def _sync(self) -> None:
        lo, hi = math.log(WaveformView.MIN_SPAN), math.log(WaveformView.MAX_SPAN)
        v = int(round((math.log(self.view.span) - lo) / (hi - lo) * 100))
        for sl, val in ((self.zoom, v), (self.amp, int(round(math.log(self.view.amp, 16.0) * 100)))):
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        self.lbl_span.setText(f"可见 {self.view.span / 1000:.1f} 秒")
