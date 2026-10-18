"""Host CPU ONNX smoke test, deterministic segmentation, cache and cancellation."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit import autotime, proc, vad
from subkit.audio import AudioFeatures, RATE, compute_features


class VadTests(unittest.TestCase):
    def result(self, probabilities):
        return vad.VadResult(np.array(probabilities, np.float32), len(probabilities) * 32, "test")

    def test_hysteresis_minimum_gap_and_final_partial_frame(self):
        result = self.result([0, .5, .4, .4, .4, .34, .1, .6, .4, .4, .4, 0])
        np.testing.assert_array_equal(vad.segments(result), [[32, 352]])
        np.testing.assert_array_equal(vad.segments(result, join_ms=32), [[32, 160], [224, 352]])
        self.assertEqual(vad.segments(self.result([.9, .9, .9, 0])).shape, (0, 2))
        partial = vad.VadResult(np.ones(4, np.float32), 100.0625, "test")
        np.testing.assert_array_equal(vad.segments(partial), [[0, 100]])

    def test_exact_join_boundary_and_threshold_validation(self):
        result = self.result([.6] * 4 + [0] * 4 + [.6] * 4)
        np.testing.assert_array_equal(vad.segments(result, join_ms=128), [[0, 384]])
        self.assertEqual(len(vad.segments(result, join_ms=127)), 2)
        with self.assertRaises(ValueError):
            vad.segments(result, end_threshold=.6)

    def test_cache_identity_uses_audio_index_version_and_media_stat(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as scratch:
            video = Path(scratch) / "media"
            video.write_bytes(b"fixture")
            key = vad.cache_file(video, 1, Path(scratch), "v1")
            self.assertEqual(key, vad.cache_file(video, 0, Path(scratch), "v1"))
            self.assertNotEqual(key, vad.cache_file(video, 2, Path(scratch), "v1"))
            self.assertNotEqual(key, vad.cache_file(video, 1, Path(scratch), "v2"))
            video.write_bytes(b"changed fixture")
            self.assertNotEqual(key, vad.cache_file(video, 1, Path(scratch), "v1"))

    def test_cache_roundtrip_version_corruption_and_cancel_publish(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as scratch:
            path = Path(scratch) / "cache.npz"
            result = self.result([0, .6, .6, .6, .6, 0])
            vad.save_cache(path, result, lambda: None)
            np.testing.assert_array_equal(vad.load_cache(path, "test").probabilities, result.probabilities)
            self.assertIsNone(vad.load_cache(path, "other"))
            before = path.read_bytes()
            calls = []
            def cancelled():
                calls.append(1)
                if len(calls) == 2:
                    raise proc.CancelledError()
            with self.assertRaises(proc.CancelledError):
                vad.save_cache(path, result, cancelled)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(scratch).glob("*.tmp")), [])
            path.write_bytes(before[:30])
            self.assertIsNone(vad.load_cache(path, "test"))
            path.write_bytes(b"corrupt")
            self.assertIsNone(vad.load_cache(path, "test"))

    def test_real_cpu_onnx_silence_music_cache_and_no_torch(self):
        before = set(sys.modules)
        t = np.arange(RATE * 2, dtype=np.float32) / RATE
        music = (.1 * np.sin(2 * np.pi * 220 * t) + .04 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as scratch:
            cache = Path(scratch) / "vad.npz"
            result = vad.analyze(music, cache_path=cache)
            self.assertEqual(len(result.probabilities), 63)
            self.assertEqual(result.duration_ms, 2000)
            self.assertEqual(len(vad.segments(result)), 0)
            with patch("onnxruntime.InferenceSession", side_effect=AssertionError("cache miss")):
                again = vad.analyze(music, cache_path=cache)
                with self.assertRaises(ValueError):
                    vad.analyze(np.array([np.nan]), cache_path=cache)
            np.testing.assert_array_equal(result.probabilities, again.probabilities)
        self.assertFalse(any(p == "torch" or p.startswith("torch.") for p in set(sys.modules) - before))

    def test_cancel_during_inference_and_invalid_pcm(self):
        calls = []
        def cancelled():
            calls.append(1)
            if len(calls) == 3:
                raise proc.CancelledError()
        with self.assertRaises(proc.CancelledError):
            vad.analyze(np.zeros(1600, np.float32), check_cancelled=cancelled)
        for samples in (np.zeros(4, np.int16), np.array([np.nan]), np.array([1.01]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                vad.analyze(samples)

    def test_speech_like_over_music_energy_refinement_is_bounded(self):
        t = np.arange(RATE * 3) / RATE
        pcm = (.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        rng = np.random.default_rng(5)
        pcm[RATE:2 * RATE] += rng.normal(0, .15, RATE).astype(np.float32)
        runs = np.array([[1100, 2100]], np.int64)
        refined = vad.refine_edges(runs, compute_features(pcm))
        self.assertTrue(np.all(np.abs(refined - runs) <= 150))
        self.assertLessEqual(abs(refined[0, 0] - 1000), 40)
        self.assertLessEqual(abs(refined[0, 1] - 2000), 40)

    def test_refinement_restored_short_run_cannot_overlap_previous_offset(self):
        energy = np.zeros(200, np.float32)
        energy[100:127] = .8
        energy[134:136] = .1
        features = AudioFeatures(np.zeros(200), np.zeros(200), np.zeros((200, 128), np.uint8), np.zeros(200))
        runs = np.array([[1000, 1120], [1248, 1360]])
        with patch.object(autotime, "_snap_activity", return_value=energy):
            refined = vad.refine_edges(runs, features)
        self.assertGreater(refined[0, 1], runs[0, 1])
        self.assertLessEqual(refined[0, 1], refined[1, 0])
        self.assertTrue(np.all(refined[:, 1] > refined[:, 0]))
        self.assertTrue(np.all(np.abs(refined - runs) <= 150))

    def test_explicit_segments_feed_calibration_and_snap_with_fallback(self):
        pcm = np.zeros(RATE * 65, np.float32)
        features = compute_features(pcm)
        lines = [(i * 3000 + 1000, i * 3000 + 2000) for i in range(21)]
        runs = np.array([(s + 100, e - 50) for s, e in lines], np.int64)
        calibration = autotime.calibrate_hint_padding(np.zeros(6500), lines, vad_segments=runs)
        self.assertEqual((calibration.lead_in_ms, calibration.lead_out_ms), (100, 50))
        self.assertEqual(calibration.source, "episode")
        self.assertEqual(autotime.snap_lines(features, lines, lead_in_ms=0, vad_segments=runs),
                         [(s + 100, e - 50) for s, e in lines])
        self.assertEqual(autotime.snap_lines(features, lines), lines)
        self.assertEqual(autotime.speech_segments(np.ones(10), 0, 1000, vad_segments=np.empty((0, 2))), [])
        self.assertEqual(autotime.speech_segments(np.zeros(10), 1150, 1950, vad_segments=runs[:1]), [(None, None)])


if __name__ == "__main__":
    unittest.main()
