"""Deterministic PCM checks; --benchmark measures without asserting wall time."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
import tempfile
import time
import unittest
import warnings
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit import audio, audio_match as am


def music(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(round(seconds * audio.RATE), np.float32)
    at = 0
    while at < len(out):
        count = min(len(out) - at, int(rng.uniform(0.18, 0.55) * audio.RATE))
        t = np.arange(count) / audio.RATE
        frequency = 110 * 2 ** (int(rng.integers(0, 30)) / 12)
        note = sum(np.sin(2 * np.pi * frequency * h * t + rng.uniform(0, 6)) / h
                   for h in range(1, 5))
        envelope = np.minimum(t / 0.015, 1) * np.minimum((count / audio.RATE - t) / 0.05, 1)
        out[at:at + count] = note * envelope * rng.uniform(0.1, 0.25)
        at += count
    return out


class MatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pcm = music(40, 51)
        cls.features = audio.compute_features(cls.pcm)
        cls.template = am.template_from(cls.features, 0, 40000)

    def embed(self, pcm: np.ndarray, starts: tuple[int, ...] = (7000,), seconds: int = 60):
        episode = music(seconds, 81)
        for start in starts:
            lo = start * 16
            episode[lo:lo + len(pcm)] = pcm
        return am.EpisodeIndex(audio.compute_features(episode).spectrogram)

    def test_exact_embedding_full_coverage(self) -> None:
        result = self.embed(self.pcm).find(self.template)
        self.assertEqual(len(result.matches), 1)
        match = result.matches[0]
        self.assertLessEqual(abs(match.start_ms - 7000), 10)
        self.assertGreaterEqual(match.score, 0.9)
        self.assertEqual(match.covered, (0, 40000))
        self.assertLessEqual(abs(match.drift_ms), 10)
        print(f"exact score={match.score:.6f}, start_ms={match.start_ms}, covered={match.covered}")

    def test_gain_and_ten_db_noise(self) -> None:
        signal = self.pcm * 10 ** (-6 / 20)
        noise = np.random.default_rng(71).normal(0, np.sqrt(np.mean(signal ** 2) / 10), len(signal))
        result = self.embed((signal + noise).astype(np.float32)).find(self.template)
        self.assertEqual(len(result.matches), 1, result)
        self.assertLessEqual(abs(result.matches[0].start_ms - 7000), 20)
        print(f"-6dB / 10dB SNR score={result.matches[0].score:.6f}, start_ms={result.matches[0].start_ms}")

    def test_unrelated_has_clear_margin(self) -> None:
        result = am.EpisodeIndex(audio.compute_features(music(60, 91)).spectrogram).find(self.template)
        self.assertEqual(result.matches, [])
        self.assertLess(result.best_score, am.SEGMENT_THRESHOLD - 0.15)
        print(f"unrelated best_score={result.best_score:.6f}; threshold={am.SEGMENT_THRESHOLD}; margin={am.SEGMENT_THRESHOLD - result.best_score:.6f}")

    def test_middle_half_coverage(self) -> None:
        result = self.embed(self.pcm[10 * 16000:30 * 16000], (17000,)).find(self.template)
        self.assertEqual(len(result.matches), 1)
        match = result.matches[0]
        self.assertLessEqual(abs(match.start_ms - 7000), 10)
        self.assertEqual(match.covered, (10000, 30000))

    def test_rev007_copied_ends_random_middle_preserves_intervals_and_plan(self) -> None:
        from subkit.core.passages import plan
        from subkit.core.ops import Editor
        from subkit.model.document import Document, Group
        from tests.test_passages import member, recorded
        rng = np.random.default_rng(7007)
        spectrum = rng.integers(0, 256, (4000, 128), dtype=np.uint8)
        episode = rng.integers(0, 256, (5000, 128), dtype=np.uint8)
        episode[500:1500] = spectrum[:1000]
        episode[3500:4500] = spectrum[3000:]
        result = am.EpisodeIndex(episode).find(am.Template(spectrum, np.zeros(4000, np.float32)))
        self.assertEqual(len(result.matches), 1)
        match = result.matches[0]
        self.assertEqual(match.start_ms, 5000)
        self.assertEqual(match.covered_intervals, ((0, 10000), (30000, 40000)))
        source = Document("source", groups=[Group(f"g_{i}", "song") for i in range(4)])
        for i in range(4):
            member(source, f"g_{i}", f"m_{i}", f"line{i}", start=i * 10000, end=(i + 1) * 10000)
        doc = Document("target")
        p = plan(doc, recorded(source), match)
        self.assertEqual([l.coverage for l in p.lines], ["direct", "unknown", "unknown", "direct"])
        self.assertEqual([l.actions[0].checked for l in p.lines], [True, False, False, True])
        summary = Editor(doc, {}).apply_passage(p, {a.id for l in p.lines for a in l.actions if a.checked})
        self.assertEqual(summary.inserted_lines, 2)
        print("REV-007", match.covered_intervals, [l.coverage for l in p.lines])

    def test_fit_unions_overlapping_pieces_and_retains_short_gap(self) -> None:
        match = am.EpisodeIndex._fit([(0, 500., 500., 1.), (1, 1000., 500., 1.), (2, 2250., 500., 1.)],
                                    [(0, 1000), (500, 1500), (2000, 2500)], 6.)
        self.assertEqual(match.covered_intervals, ((0, 15000), (20000, 25000)))

    def test_two_occurrences(self) -> None:
        result = self.embed(self.pcm, (7000, 55000), 100).find(self.template)
        self.assertEqual(len(result.matches), 2)
        for match, expected in zip(result.matches, (7000, 55000)):
            self.assertLessEqual(abs(match.start_ms - expected), 10)
            self.assertEqual(match.covered, (0, 40000))

    def test_one_percent_slower(self) -> None:
        # Linear resampling changes pitch too: harder than pitch-preserving stretch.
        stretched = np.interp(np.arange(round(len(self.pcm) * 1.01)) / 1.01,
                              np.arange(len(self.pcm)), self.pcm).astype(np.float32)
        result = self.embed(stretched).find(self.template)
        self.assertEqual(len(result.matches), 1, result)
        match = result.matches[0]
        self.assertGreater(match.drift_ms, 0)
        self.assertAlmostEqual(match.scale, 1.01, delta=0.003)
        self.assertLessEqual(abs(match.start_ms - 7000), 60)
        print(f"1% slower score={match.score:.6f}, start_ms={match.start_ms}, drift_ms={match.drift_ms}, scale={match.scale:.6f}")

    def test_single_segment(self) -> None:
        t = am.template_from(self.features, 0, 8000)
        result = self.embed(self.pcm[:8 * 16000]).find(t)
        self.assertEqual(len(result.matches), 1)
        self.assertLessEqual(abs(result.matches[0].start_ms - 7000), 10)
        self.assertEqual(result.matches[0].scale, 1.0)
        self.assertGreaterEqual(result.matches[0].score, am.SINGLE_THRESHOLD)

    def test_coarse_phase_block_boundary_and_final_window(self) -> None:
        rng = np.random.default_rng(102)
        spectrum = rng.integers(0, 256, (303, 128), dtype=np.uint8)
        template = am.Template(spectrum, np.zeros(303, np.float32))
        for start in (2045, 2046, 2047, 2048, 2397):
            with self.subTest(start=start):
                episode = rng.integers(0, 256, (2700, 128), dtype=np.uint8)
                episode[start:start + 303] = spectrum
                index = am.EpisodeIndex(episode)
                self.assertEqual(index._audio.dtype, np.float32)
                self.assertTrue(all(value.dtype == np.complex64 for value in index._ffts.values()))
                result = index.find(template)
                self.assertEqual(len(result.matches), 1)
                self.assertEqual(result.matches[0].start_ms, start * 10)
                self.assertGreater(result.matches[0].score, 0.999)

    def test_empty_short_and_silent_no_warnings_or_nan(self) -> None:
        previous = np.geterr().copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for count in (0, 100, 5000):
                result = am.EpisodeIndex(np.zeros((count, 128), np.uint8)).find(self.template)
                self.assertEqual(result.matches, [])
                self.assertEqual(result.best_score, 0)
            silent = am.template_from(audio.compute_features(np.zeros(80000, np.float32)), 0, 5000)
            result = self.embed(self.pcm).find(silent)
            self.assertEqual(result.matches, [])
            self.assertEqual(result.best_score, 0)
        self.assertEqual(np.geterr(), previous)

    def test_npz_exact_atomic_round_trip_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "template.npz"
            am.save_template(path, self.template)
            am.save_template(path, self.template)
            t = am.load_template(path)
            np.testing.assert_array_equal(t.spectrogram, self.template.spectrogram)
            np.testing.assert_array_equal(t.onset, self.template.onset)
            self.assertEqual((t.hop_ms, t.features), (10, audio.CACHE_VERSION))
            with path.open("wb") as out:
                np.savez(out, spectrogram=t.spectrogram, onset=t.onset, hop_ms=10, features="audio-v0")
            with self.assertRaisesRegex(ValueError, am.VERSION_MISMATCH):
                am.load_template(path)
            path.unlink()  # No retained npz handles on Windows.

    def test_failed_replace_keeps_old_file_and_closes_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "template.npz"
            am.save_template(path, self.template)
            before = path.read_bytes()
            def failed(source, destination):
                self.assertEqual(Path(source).parent, path.parent)
                self.assertEqual(Path(destination), path)
                raise OSError("simulated replace failure")
            with patch("subkit.audio_match.os.replace", side_effect=failed):
                with self.assertRaisesRegex(OSError, "simulated"):
                    am.save_template(path, self.template)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_too_short_and_frame_rounding(self) -> None:
        with self.assertRaisesRegex(ValueError, "段落太短") as raised:
            am.template_from(self.features, 0, 2999)
        self.assertEqual(str(raised.exception), am.TOO_SHORT)
        t = am.template_from(self.features, 3, 3003)
        np.testing.assert_array_equal(t.spectrogram, self.features.spectrogram[1:301])
        for lo, hi in ((-1, 4000), (5000, 4000), (0, 40001)):
            with self.assertRaises(ValueError):
                am.template_from(self.features, lo, hi)


class AnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/eval/passage_match_real.py"))

    def test_chinese_anchor_skips_extra_lines_and_subtracts_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source.ass", Path(temp) / "target.ass"
            source.write_text("Dialogue: 0,0:00:01.23,0:00:04.00,JPOPED,,0,0,0,,{\\kf20}日本語\n"
                              "Dialogue: 0,0:00:01.23,0:00:04.00,CNOPED,,0,0,0,,{\\kf20}同  句歌詞\n", encoding="utf-8")
            target.write_text("Dialogue: 0,0:00:01.00,0:00:02.00,ED,,0,0,0,,extra\n"
                              "Dialogue: 0,0:00:02.00,0:00:03.00,ED,,0,0,0,,extra again\n"
                              "Dialogue: 0,0:00:05.00,0:00:08.00,ED,,0,0,0,,同　句\\N歌詞\n", encoding="utf-8")
            expected, anchor = self.tool["anchor"](source, target, (1000, 4000), "ED")
            self.assertEqual(expected, 4770)
            self.assertEqual(anchor["source_style"], "CNOPED")
            self.assertEqual(anchor["source_offset_ms"], 230)
            self.assertEqual(anchor["target_start_ms"], 5000)

    def test_japanese_anchor_and_missing_identical_lyric(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source.ass", Path(temp) / "target.ass"
            source.write_text("Dialogue: 0,0:00:01.00,0:00:04.00,JPOPED,,0,0,0,,{\\kf20}同句\n", encoding="utf-8")
            target.write_text("Dialogue: 0,0:00:05.00,0:00:08.00,ED,,0,0,0,,同句\n", encoding="utf-8")
            expected, anchor = self.tool["anchor"](source, target, (1000, 4000), "ED")
            self.assertEqual(expected, 5000)
            self.assertEqual(anchor["source_style"], "JPOPED")
            with self.assertRaisesRegex(ValueError, "no identical lyric anchor"):
                self.tool["anchor"](source, target, (1000, 4000), "OPCH")


def benchmark() -> None:
    # Feature extraction excluded: the GUI already holds the episode features.
    block = audio.compute_features(music(360, 24))
    spectrum = np.tile(block.spectrogram, (4, 1))
    templates = [am.template_from(block, i * 90000, (i + 1) * 90000) for i in range(4)]
    times = []
    builds = []
    searches = []
    for _ in range(3):
        begin = time.perf_counter()
        index = am.EpisodeIndex(spectrum)
        builds.append(time.perf_counter() - begin)
        template_times = []
        for t in templates:
            search_begin = time.perf_counter()
            index.find(t)
            template_times.append(time.perf_counter() - search_begin)
        searches.append(template_times)
        times.append(time.perf_counter() - begin)
    print(json.dumps({"frames": len(spectrum), "templates": 4, "template_seconds": 360,
                      "index_seconds": builds, "per_template_seconds": searches,
                      "runs_seconds": times, "median_seconds": float(np.median(times)),
                      "target_seconds": 1.0}, ensure_ascii=False))


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
