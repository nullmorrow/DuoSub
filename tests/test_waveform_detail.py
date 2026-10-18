"""KB reference DSP, real decoding and Qt offscreen job/rendering lifecycle."""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QThread, QThreadPool
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication
from subkit import audio, proc
from subkit_app.views import waveform


class DetailDSP(unittest.TestCase):
    def test_pixel_extrema_against_sample_time_reference(self):
        samples = np.random.default_rng(19).integers(-32768, 32768, 2001, dtype=np.int16)
        times = np.arange(len(samples)) * 1000 / audio.RATE
        for start, span, width in ((0, 125, 603), (12.3125, 73.875, 91), (-20, 160, 800),
                                   (124, 20, 401), (200, 1000, 600), (0, 1, 1000)):
            with self.subTest(start=start, width=width):
                expected_lo, expected_hi = [], []
                for x in range(width):
                    lo, hi = start + x * span / width, start + (x + 1) * span / width
                    column = samples[(times >= lo) & (times < hi)]
                    expected_lo.append(float(column.min()) / 32768 if len(column) else 0)
                    expected_hi.append(float(column.max()) / 32768 if len(column) else 0)
                actual = audio.pixel_peaks(samples, start, span, width)
                np.testing.assert_array_equal(actual[0], np.array(expected_lo, np.float32))
                np.testing.assert_array_equal(actual[1], np.array(expected_hi, np.float32))
        self.assertEqual(len(audio.pixel_peaks(samples, 0, 1, 0)[0]), 0)
        np.testing.assert_array_equal(audio.pixel_peaks(np.empty(0, np.int16), 0, 1, 3)[0], [0, 0, 0])

    def test_stft_against_individual_numpy_windows_and_band_means(self):
        samples = np.random.default_rng(11).integers(-30000, 30000, 2500, dtype=np.int16)
        frequencies = np.fft.rfftfreq(512, 1 / audio.RATE)
        window = np.hanning(512).astype(np.float32)
        for start, span in ((0, 17), (73.03125, 31.5), (154, 15), (-8, 19), (250, 11)):
            rows = []
            for i in range(int(np.ceil(span / 2))):
                center = round(start * 16) + i * 32
                indices = np.arange(center - 256, center + 256)
                frame = np.zeros(512, np.float32)
                valid = (indices >= 0) & (indices < len(samples))
                frame[valid] = samples[indices[valid]].astype(np.float32) / 32768
                magnitude = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
                magnitude *= 2.0 / window.sum()
                bands = []
                for low, high in zip(audio.BAND_EDGES_HZ[:-1], audio.BAND_EDGES_HZ[1:]):
                    bins = np.flatnonzero((frequencies >= low) & (frequencies < high))
                    if high >= 7999.99:
                        bins = np.flatnonzero(frequencies >= low)
                    if not len(bins):
                        bins = [min(256, int(np.ceil(low / 31.25)))]
                    bands.append(magnitude[bins].mean(dtype=np.float64))
                db = 20 * np.log10(np.maximum(bands, 1e-4))
                rows.append(np.rint(np.clip((db + 80) / 80, 0, 1) * 255))
            np.testing.assert_array_equal(audio.window_spectrogram(samples, start, span), np.array(rows, np.uint8))

    def test_fine_frames_match_coarse_scaling_at_shared_centers(self):
        samples = np.random.default_rng(7).integers(-20000, 20000, 32000, dtype=np.int16)
        coarse = audio.compute_features(samples.astype(np.float32) / 32768)
        fine = audio.window_spectrogram(samples, 500, 1000)
        self.assertEqual(fine.shape, (500, 128))
        np.testing.assert_array_equal(fine[::5], coarse.spectrogram[50:150])

    def test_silence_and_chunk_cancellation(self):
        samples = np.zeros(32000, np.int16)
        self.assertFalse(audio.window_spectrogram(samples, 0, 1).any())
        check = Mock(side_effect=[None, None, proc.CancelledError()])
        with self.assertRaises(proc.CancelledError):
            audio.window_spectrogram(samples, 0, 2000, check_cancelled=check)
        self.assertEqual(check.call_count, 3)

    def test_reject_invalid_pcm_and_ranges(self):
        for samples in (np.zeros(10, np.float32), np.zeros((2, 2), np.int16)):
            with self.assertRaises(ValueError):
                audio.pixel_peaks(samples, 0, 1, 2)
            with self.assertRaises(ValueError):
                audio.window_spectrogram(samples, 0, 1)
        for start, span in ((np.nan, 1), (0, 0), (0, np.inf)):
            with self.assertRaises(ValueError):
                audio.window_spectrogram(np.zeros(10, np.int16), start, span)


class RecordingView(waveform.WaveformView):
    def __init__(self):
        super().__init__()
        self.drawn = []
        self.delivery_threads = []

    def _draw_spectrum(self, painter, spectrum, start, span, height):
        self.drawn.append(spectrum.shape)
        super()._draw_spectrum(painter, spectrum, start, span, height)

    def _detail_loaded(self, spectrum, key, generation, request):
        self.delivery_threads.append(QThread.currentThread())
        super()._detail_loaded(spectrum, key, generation, request)


class DetailQt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("KB tests require offscreen Qt")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kb_audio_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.view = RecordingView()
        self.addCleanup(self.dispose)
        self.view.resize(600, 200)
        samples = np.random.default_rng(8).integers(-20000, 20000, 48000, dtype=np.int16)
        features = audio.compute_features(samples.astype(np.float32) / 32768)
        features.samples = samples
        self.view._loaded(features)
        self.view.display_mode = "spectrogram"
        self.view.show()
        self.app.processEvents()

    def dispose(self):
        self.view.close()
        self.assertTrue(QThreadPool.globalInstance().waitForDone(3000))
        self.app.processEvents()
        self.view.deleteLater()
        self.app.processEvents()

    def wait_for(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.002)
        self.assertTrue(condition(), self.view.status_text)

    def slow_spectrum(self, started, release):
        real = audio.window_spectrogram
        def compute(*args, **kwargs):
            started.set()
            while not release.wait(0.002):
                kwargs["check_cancelled"]()
            return real(*args, **kwargs)
        return compute

    def test_zoom_one_job_coarse_until_ready_then_repaint_and_cache(self):
        started, release = threading.Event(), threading.Event()
        with patch.object(waveform, "window_spectrogram", side_effect=self.slow_spectrum(started, release)) as compute:
            self.view.set_span(1000, center=500)
            self.assertTrue(started.wait(3))
            for _ in range(3):
                self.view.repaint()
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(self.view.drawn[-1], (100, 128))
            release.set()
            self.wait_for(lambda: len(self.view._detail_cache) == 1)
            self.view.repaint()
            self.assertEqual(self.view.drawn[-1], (500, 128))
            self.assertEqual(self.view.delivery_threads, [self.app.thread()])
            self.view.set_amp(2)
            self.view.repaint()
            self.assertEqual(compute.call_count, 1)
            self.view.set_span(6000, center=3000)
            self.view.set_span(1000, center=500)
            self.view.repaint()
            self.assertEqual(compute.call_count, 1)

    def test_scroll_cancels_and_rejects_late_old_result(self):
        started, release = threading.Event(), threading.Event()
        with patch.object(waveform, "window_spectrogram", side_effect=self.slow_spectrum(started, release)):
            self.view.set_span(1000, center=500)
            self.assertTrue(started.wait(3))
            old = self.view._detail_job
            self.view.t0 = 500
            self.view.viewChanged.emit()
            self.assertTrue(old.cancelled)
            self.view._detail_loaded(np.full((500, 128), 255, np.uint8), old.key, old.generation, old.request)
            self.assertNotIn(old.key, self.view._detail_cache)
            release.set()
            self.wait_for(lambda: self.view._visible_detail_key() in self.view._detail_cache)
            self.assertEqual(list(self.view._detail_cache), [(500, 1000, 600)])

    def test_close_and_episode_switch_cancel_running_stft_release_memory(self):
        for action in ("close", "switch"):
            with self.subTest(action=action):
                started, release = threading.Event(), threading.Event()
                # A recording Mock retains PCM in call_args, defeating the weakref assertion.
                with patch.object(waveform, "window_spectrogram", new=self.slow_spectrum(started, release)):
                    self.view.set_span(1000, center=500)
                    self.assertTrue(started.wait(3))
                    job = self.view._detail_job
                    samples_ref = weakref.ref(self.view.samples)
                    if action == "close":
                        self.view.close()
                    else:
                        self.view.load(None, None, self.root / "cache")
                    self.assertTrue(job.cancelled)
                    self.assertTrue(job.finished.is_set())
                    self.assertIsNone(job.samples)
                    self.assertIsNone(self.view.samples)
                    self.assertEqual(self.view._detail_cache, {})
                    self.app.processEvents()
                    gc.collect()
                    self.assertIsNone(samples_ref())
                # Restore PCM for the next lifecycle action on the same widget.
                self.view._loaded(audio.AudioFeatures(np.zeros(100), np.zeros(100),
                                  np.zeros((100, 128), np.uint8), np.zeros(100), np.zeros(16000, np.int16)))
                self.view.set_span(12000)

    def test_close_releases_completed_fine_cache(self):
        self.view.set_span(1000, center=500)
        self.wait_for(lambda: bool(self.view._detail_cache))
        fine_ref = weakref.ref(next(iter(self.view._detail_cache.values())))
        self.view.close()
        self.app.processEvents()
        gc.collect()
        self.assertIsNone(fine_ref())

    def test_width_changes_request_new_data_and_cache_is_bounded(self):
        self.view.set_span(1000, center=500)
        self.wait_for(lambda: bool(self.view._detail_cache))
        self.view.resize(640, 200)
        self.wait_for(lambda: (0, 1000, 640) in self.view._detail_cache)
        self.assertIn((0, 1000, 600), self.view._detail_cache)
        for start in (500, 1000, 1500, 2000):
            self.view.t0 = start
            self.view.viewChanged.emit()
            self.wait_for(lambda: self.view._visible_detail_key() in self.view._detail_cache)
        self.assertEqual(len(self.view._detail_cache), 4)
        self.assertNotIn((0, 1000, 600), self.view._detail_cache)

    def test_fine_waveform_uses_pcm_and_envelope_is_continuous(self):
        self.view.display_mode = "waveform"
        self.view.set_span(1000, center=500)
        with patch.object(waveform, "pixel_peaks", wraps=audio.pixel_peaks) as peaks:
            image = self.view.grab().toImage()
        self.assertEqual(peaks.call_count, 1)
        self.assertEqual(peaks.call_args.args[1:], (0.0, 1000.0, 600))
        self.assertIsNone(self.view._detail_job)
        for x in range(1, 599):
            self.assertTrue(any(image.pixelColor(x, y).name() == "#3fb950" for y in range(31, 154)))

    def test_spectrum_low_frequency_bottom_and_palette_shared(self):
        spectrum = np.zeros((500, 128), np.uint8)
        spectrum[:, 0] = 255
        self.view.span = 1000
        image = QImage(600, 200, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        painter = QPainter(image)
        self.view.spectrum_palette = "classic"        # pixel values below are the classic map, unadjusted
        self.view.spectrum_contrast = self.view.spectrum_brightness = 0
        self.view._draw_spectrum(painter, spectrum, 0, 1000, 200)
        painter.end()
        self.assertEqual(image.pixelColor(300, 199).name(), "#ffff18")
        self.assertEqual(image.pixelColor(300, 0).name(), "#0000ff")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg/ffprobe")
    def test_real_cached_features_lazy_decode_once_at_zoom(self):
        path = self.root / "signal.wav"
        samples = np.arange(32000, dtype=np.int16)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(audio.RATE)
            out.writeframes(samples.astype("<i2").tobytes())
        peak = waveform.PeakJob(path, 0, self.root / "cache")
        peak.run()
        self.view.set_span(12000)
        real = proc.KillableRun.run
        with patch.object(proc.KillableRun, "run", autospec=True, side_effect=real) as run:
            self.view.load(path, 0, self.root / "cache")
            self.wait_for(lambda: self.view.audio_features() is not None)
            self.assertIsNone(self.view.samples)
            self.assertFalse(any(call.args[1][0] == "ffmpeg" for call in run.call_args_list))
            self.view.set_span(1000, center=500)
            self.view.repaint()
            self.wait_for(lambda: bool(self.view._detail_cache))
            np.testing.assert_array_equal(self.view.samples, samples)
            self.assertEqual(sum(call.args[1][0] == "ffmpeg" for call in run.call_args_list), 1)
            self.assertIs(self.view.audio_features().samples, self.view.samples)

    def test_lazy_decode_cancel_kills_child_on_close_and_switch(self):
        for action in ("close", "switch"):
            with self.subTest(action=action):
                self.view.cancel_jobs()
                self.view.samples = None
                self.view._audio_source = (self.root / "signal.wav", 0)
                self.view._features = audio.AudioFeatures(np.zeros(100), np.zeros(100),
                                       np.zeros((100, 128), np.uint8), np.zeros(100))
                started = threading.Event()
                children = []
                real = subprocess.Popen
                def spawn(_cmd, **kwargs):
                    child = real([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
                    children.append(child)
                    started.set()
                    return child
                with patch.object(proc.subprocess, "Popen", side_effect=spawn):
                    self.view.set_span(1000, center=500)
                    self.assertTrue(started.wait(3))
                    job = self.view._sample_job
                    if action == "close":
                        self.view.close()
                    else:
                        self.view.load(None, None, self.root / "cache")
                    self.assertTrue(job.cancelled)
                    self.assertTrue(job.finished.is_set())
                    self.assertTrue(all(child.poll() is not None for child in children))
                self.app.processEvents()
                self.assertIsNone(self.view.samples)


if __name__ == "__main__":
    unittest.main()
