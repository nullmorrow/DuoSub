"""KR3 synthetic signals, real ffmpeg/cache and Qt offscreen data delivery."""

from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit import audio, proc


def clicks(times: list[int], duration_ms: int = 2000) -> np.ndarray:
    samples = np.zeros(duration_ms * audio.RATE // 1000, np.float32)
    for ms in times:
        samples[ms * audio.RATE // 1000] = 1
    return samples


def starts(times: list) -> list:
    return [SimpleNamespace(t0=t, t1=9999, by="placeholder") for t in times]


class FeatureTests(unittest.TestCase):
    def test_click_timing_on_and_between_hops(self) -> None:
        for times in ([100, 500, 1000, 1500], [103, 507, 1009, 1504], [0, 1999]):
            features = audio.compute_features(clicks(times))
            peaks = audio.onset_peaks(features.onset, 0, 2000)
            self.assertEqual(len(peaks), len(times))
            for peak, expected in zip(peaks, times):
                self.assertLessEqual(abs(int(peak) - expected), 10)

    def test_tone_bursts_timing_and_frequency(self) -> None:
        samples = np.zeros(audio.RATE * 2, np.float32)
        tone = np.sin(2 * np.pi * 1000 * np.arange(1600) / audio.RATE).astype(np.float32)
        times = [203, 706, 1201]
        for t in times:
            i = t * audio.RATE // 1000
            samples[i:i + len(tone)] = tone
        features = audio.compute_features(samples)
        for expected in times:
            i = int(np.argmax(features.onset[expected // 10 - 3:expected // 10 + 4]))
            actual = (i + expected // 10 - 3) * 10
            self.assertLessEqual(abs(actual - expected), 10)
        band = int(features.spectrogram[25].argmax())
        self.assertLessEqual(audio.BAND_EDGES_HZ[band], 1000)
        self.assertGreaterEqual(audio.BAND_EDGES_HZ[band + 1], 1000)
        self.assertEqual(features.spectrogram.dtype, np.uint8)
        self.assertEqual(features.onset.dtype, np.float32)

    def test_empty_silence_short_and_partial_block(self) -> None:
        for length in (0, 1, 159, 160, 161, 513):
            features = audio.compute_features(np.zeros(length, np.float32))
            count = (length + 159) // 160
            self.assertEqual(features.spectrogram.shape, (count, 128))
            self.assertEqual(features.onset.shape, (count,))
            self.assertFalse(np.any(features.spectrogram))
            self.assertFalse(np.any(features.onset))
        features = audio.compute_features(np.r_[np.zeros(160), [0.25, 0.5]])
        np.testing.assert_array_equal(features.mins, [0, 0.25])
        np.testing.assert_array_equal(features.maxs, [0, 0.5])

    def test_nyquist_bin_and_determinism(self) -> None:
        samples = np.tile(np.array([0.5, -0.5], np.float32), 8000)
        one, two = audio.compute_features(samples), audio.compute_features(samples)
        self.assertEqual(int(one.spectrogram[50].argmax()), 127)
        for field in ("mins", "maxs", "spectrogram", "onset"):
            np.testing.assert_array_equal(getattr(one, field), getattr(two, field))

    def test_display_averages_linear_magnitudes_without_changing_flux(self) -> None:
        samples = np.random.default_rng(42).normal(0, 0.1, 3200).astype(np.float32)
        features = audio.compute_features(samples)
        window = np.hanning(512).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(np.pad(samples, (256, 256)), 512)[::160][:20]
        magnitude = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
        magnitude *= 2.0 / window.sum()
        expected_flux = np.maximum(np.diff(magnitude, axis=0, prepend=np.zeros((1, 257), np.float32)), 0).mean(axis=1)
        np.testing.assert_array_equal(features.onset, expected_flux)
        expected = []
        frequencies = np.arange(257) * 31.25
        for lo, hi in zip(audio.BAND_EDGES_HZ[:-1], audio.BAND_EDGES_HZ[1:]):
            bins = np.flatnonzero((frequencies >= lo) & (frequencies < hi))
            if hi >= 7999.99:
                bins = np.flatnonzero(frequencies >= lo)
            if not len(bins):
                bins = [min(256, int(np.ceil(lo / 31.25)))]
            mean = magnitude[:, bins].mean(axis=1, dtype=np.float64)
            expected.append(np.rint(np.clip((20 * np.log10(np.maximum(mean, 1e-4)) + 80) / 80, 0, 1) * 255))
        np.testing.assert_array_equal(features.spectrogram, np.array(expected, np.uint8).T)
        self.assertEqual(audio.CACHE_VERSION, "audio-v4")

    def test_invalid_pcm(self) -> None:
        for samples in (np.zeros((2, 2)), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError):
                audio.compute_features(samples)

    def test_flux_continuity_across_fft_chunks(self) -> None:
        samples = clicks([20475, 20545], 21000)
        features = audio.compute_features(samples)
        for t in (20475, 20545):
            # An independent short analysis must agree across the 2048-frame boundary.
            local = audio.compute_features(samples[(t // 10 - 10) * 160:(t // 10 + 10) * 160])
            np.testing.assert_allclose(features.onset[t // 10 - 5:t // 10 + 5], local.onset[5:15])


class SuggestionTests(unittest.TestCase):
    def test_clicks_recover_boundaries_and_leave_beats_unchanged(self) -> None:
        features = audio.compute_features(clicks([220, 570, 940, 1280]))
        beats = starts([200, 550, 900, 1250])
        before = copy.deepcopy(beats)
        result = audio.suggest_beats(features.onset, 200, 1600, beats)
        self.assertEqual(result, [220, 570, 940, 1280])
        self.assertEqual(beats, before)
        self.assertEqual(audio.suggest_beats(features.onset, 200, 1600, beats), result)

    def test_tone_burst_suggestions(self) -> None:
        samples = np.zeros(32000, np.float32)
        tone = np.sin(2 * np.pi * 700 * np.arange(2400) / 16000)
        for t in (300, 700, 1100):
            samples[t * 16:t * 16 + len(tone)] = tone
        result = audio.suggest_beats(audio.compute_features(samples).onset, 200, 1500,
                                     starts([280, 680, 1080]))
        for actual, expected in zip(result, (300, 700, 1100)):
            self.assertLessEqual(abs(actual - expected), 10)
        self.assertEqual(len(result), 3)

    def test_peak_spacing_plateau_and_line_bounds(self) -> None:
        onset = np.zeros(100, np.float32)
        onset[[0, 10, 14, 20, 99]] = [1, 2, 3, 3, 5]
        onset[30:33] = 3
        self.assertEqual(audio.onset_peaks(onset, 100, 990).tolist(), [140, 300])
        self.assertEqual(audio.onset_peaks(onset, 141, 200).tolist(), [])

    def test_onsets_keep_strongest_with_exact_100ms_boundary(self) -> None:
        onset = np.zeros(100, np.float32)
        onset[[10, 19, 29, 38, 48]] = [2, 4, 3, 1, 3]
        peaks = audio.onset_peaks(onset, 0, 1000)
        self.assertEqual(peaks.tolist(), [190, 290, 480])
        self.assertTrue(np.all(np.diff(peaks) >= 100))

    def test_local_strength_rejects_weak_peaks_but_keeps_quiet_region(self) -> None:
        onset = np.zeros(300, np.float32)
        onset[[10, 30, 50, 210, 230]] = [4, 0.1, 0.2, 0.1, 0.01]
        self.assertEqual(audio.onset_peaks(onset, 0, 3000).tolist(), [100, 2100])
        self.assertEqual(audio.onset_peaks(onset, 200, 600).tolist(), [])

    def test_silence_empty_and_too_few_peaks(self) -> None:
        onset = np.zeros(100, np.float32)
        self.assertEqual(audio.suggest_beats(onset, 100, 900, starts([120, 330, 760])), [120, 330, 760])
        self.assertEqual(audio.suggest_beats(onset, 100, 900, starts([None] * 4)), [100, 300, 500, 700])
        onset[50] = 1
        result = audio.suggest_beats(onset, 100, 900, starts([100, 300, 500, 700]))
        self.assertEqual(len(result), 4)
        self.assertEqual(result, sorted(set(result)))
        self.assertTrue(all(100 <= t < 900 for t in result))
        self.assertEqual(audio.suggest_beats(onset, 100, 101, starts([100, 100])), [])
        self.assertEqual(audio.suggest_beats(onset, 900, 100, starts([100])), [])
        self.assertEqual(audio.suggest_beats(onset, 100, 900, []), [])

    def test_distance_penalty_and_global_order(self) -> None:
        onset = np.zeros(200, np.float32)
        onset[[25, 75, 150]] = [0.8, 0.8, 1]
        self.assertEqual(audio.suggest_beats(onset, 0, 2000, starts([250, 750])), [250, 750])
        # A greedy per-beat choice would repeat 500; DP can use it only once.
        onset = np.zeros(100, np.float32)
        onset[[40, 50, 60]] = [0.8, 1, 0.8]
        result = audio.suggest_beats(onset, 300, 800, starts([490, 510]))
        candidates = sorted(set([400, 500, 600, 490, 510]))
        def score(ts):
            return sum((float(onset[t // 10]) if t in (400, 500, 600) else 0)
                       - 0.35 * ((t - target) / 125) ** 2 for t, target in zip(ts, (490, 510)))
        optimum = max(score((a, b)) for a in candidates for b in candidates if a < b)
        self.assertAlmostEqual(score(result), optimum)

    def test_invalid_onsets_and_clamping(self) -> None:
        result = audio.suggest_beats(np.array([np.nan, -1, np.inf]), 5, 25, starts([-10, 100]))
        self.assertEqual(result, [5, 24])
        with self.assertRaises(ValueError):
            audio.suggest_beats(np.zeros((2, 2)), 0, 100, starts([10]))


class PeakJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        from subkit_app.views.waveform import PeakJob, WaveformView
        cls.PeakJob, cls.WaveformView = PeakJob, WaveformView

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="subkit_audio_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "click.wav"
        with wave.open(str(self.path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(audio.RATE)
            out.writeframes((clicks([200, 600, 1000]) * 30000).astype("<i2").tobytes())

    def run_job(self, track=None):
        result, errors = [], []
        job = self.PeakJob(self.path, track, self.root / "cache")
        job.signals.done.connect(lambda features, _generation: result.append(features))
        job.signals.failed.connect(lambda error, _generation: errors.append(error))
        job.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(result), 1)
        return result[0]

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_real_decode_once_then_cache_reuse(self) -> None:
        begin = time.perf_counter()
        with patch.object(proc.KillableRun, "run", autospec=True, side_effect=proc.KillableRun.run) as run:
            features = self.run_job()
        elapsed = time.perf_counter() - begin
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[1][0], "ffmpeg")
        self.assertEqual(audio.suggest_beats(features.onset, 100, 1400, starts([190, 590, 990])),
                         [200, 600, 1000])
        begin = time.perf_counter()
        with patch("subkit_app.views.waveform.proc.KillableRun.run", side_effect=AssertionError("cache decoded again")):
            cached = self.run_job(0)
        cached_elapsed = time.perf_counter() - begin
        for field in ("mins", "maxs", "spectrogram", "onset"):
            np.testing.assert_array_equal(getattr(features, field), getattr(cached, field))
        cache = next((self.root / "cache").glob("*.npz"))
        size = cache.stat().st_size
        cache.unlink()  # The npz reader must not retain an open Windows file handle.
        print(f"KR3 real ffmpeg: 2.000s/16kHz PCM, decode+features+cache {elapsed:.3f}s; "
              f"cache hit {cached_elapsed:.3f}s; {size} bytes; one ffmpeg run")

    def test_cache_invalidation_file_track_version_and_nanosecond_mtime(self) -> None:
        raw = SimpleNamespace(stdout=(clicks([200]) * 30000).astype("<i2").tobytes())
        with patch("subkit_app.views.waveform.proc.KillableRun.run", return_value=raw) as run:
            self.run_job()
            self.run_job(0)
            self.assertEqual(run.call_count, 1)
            self.run_job(1)
            self.assertEqual(run.call_count, 1)
            self.run_job(2)
            self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("-map") + 1], "0:a:1")
            st = self.path.stat()
            os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000))
            self.run_job()
            with self.path.open("ab") as out:
                out.write(b"x")
            self.run_job()
            with patch("subkit.audio.CACHE_VERSION", "test-new-version"):
                self.run_job()
            self.path = self.root / "other.wav"
            self.path.write_bytes(b"source")
            self.run_job()
        self.assertEqual(run.call_count, 6)

    def test_view_data_reset_and_late_results(self) -> None:
        view = self.WaveformView()
        try:
            features = audio.compute_features(clicks([200]))
            view._loaded(features)
            self.assertIs(view.spectrogram, features.spectrogram)
            self.assertIs(view.onset, features.onset)
            old = view._load_generation
            view.load(None, None, self.root)
            view._loaded(features, old)
            view._keyframes_loaded(np.array([200]), old)
            view._fail("old failure", old)
            self.assertIsNone(view.spectrogram)
            self.assertIsNone(view.onset)
            self.assertIsNone(view.mins)
            self.assertIsNone(view.keyframes)
            self.assertEqual(view.status_text, "（没有视频/音频文件）")
        finally:
            view.close()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg/ffprobe")
    def test_background_decode_delivers_features_on_ui_thread(self) -> None:
        from PySide6.QtCore import QThread, QThreadPool
        threads = []
        class RecordingView(self.WaveformView):
            def _loaded(self, features, generation=None):
                threads.append(QThread.currentThread())
                super()._loaded(features, generation)
        view = RecordingView()
        try:
            view.load(self.path, 0, self.root / "cache")
            deadline = time.monotonic() + 10
            while view.onset is None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertIsNotNone(view.onset, view.status_text)
            self.assertEqual(threads, [self.app.thread()])
            self.assertEqual(view.spectrogram.shape, (200, 128))
        finally:
            self.assertTrue(QThreadPool.globalInstance().waitForDone(10000))
            self.app.processEvents()
            view.close()


if __name__ == "__main__":
    unittest.main()
