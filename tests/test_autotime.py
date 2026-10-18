"""Deterministic time-map evidence, including speed, cuts and weak audio."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit import audio, autotime as at


def fixture(offset: int = 0, rate: float = 1.0, jump: int = 0, missing: bool = False):
    rng = np.random.default_rng(701)
    intervals = []
    start = 10000
    while start < 1180000:
        end = start + int(rng.integers(700, 2400))
        if end < 585000 or start > 615000:
            intervals.append((start, end))
        start = end + int(rng.integers(700, 4500))
    spectrum = np.full((130000, 128), 30, np.uint8)
    bands = (audio.BAND_EDGES_HZ[:-1] >= 250) & (audio.BAND_EDGES_HZ[1:] <= 3500)
    for start, end in intervals:
        lo = round((start * rate + offset + (jump if start > 600000 else 0)) / 10)
        hi = round((end * rate + offset + (jump if start > 600000 else 0)) / 10)
        spectrum[max(0, lo):max(0, hi), bands] = 180
    features = audio.AudioFeatures(np.zeros(130000), np.zeros(130000), spectrum, np.zeros(130000))
    return features, intervals[::2] if missing else intervals


class TimeMapTests(unittest.TestCase):
    def test_constant_and_search_edges(self):
        for offset in (-59000, 730, 59000):
            with self.subTest(offset=offset):
                features, intervals = fixture(offset)
                result = at.estimate(features, intervals, method="shift", correlation_bias_ms=0)
                self.assertEqual(len(result.segments), 1)
                self.assertLess(abs(result.map(400000) - 400000 - offset), 60)
                self.assertNotEqual(result.segments[0].confidence, "low")

    def test_rates(self):
        for rate in (1.001, 1.0427, 23.976 / 25):
            with self.subTest(rate=rate):
                features, intervals = fixture(500, rate)
                result = at.estimate(features, intervals, correlation_bias_ms=0)
                self.assertEqual(len(result.segments), 1)
                self.assertAlmostEqual(result.segments[0].rate, rate, delta=0.0006)
                for t in (100000, 1000000):
                    self.assertLess(abs(result.map(t) - round(t * rate + 500)), 180)

    def test_jump_at_gap(self):
        features, intervals = fixture(-440, jump=15000)
        result = at.estimate(features, intervals, correlation_bias_ms=0)
        self.assertEqual(len(result.segments), 2)
        self.assertLess(abs(result.segments[1].start_ms - 600000), 5000)
        self.assertLess(abs(result.map(200000) - 199560), 100)
        self.assertLess(abs(result.map(1000000) - 1014560), 100)

    def test_missing_subtitles(self):
        features, intervals = fixture(820, missing=True)
        result = at.estimate(features, intervals, correlation_bias_ms=0)
        self.assertLess(abs(result.map(800000) - 800820), 100)

    def test_music_outliers_and_weak_neighbours(self):
        windows = [at.WindowLag(i * 30000, i * 30000 + 120000,
                                30000 if i in (5, 12) else 720, 0.5, "high") for i in range(25)]
        windows.extend([at.WindowLag(900000, 1020000, -40000, 0.05, "low")])
        result = at.fit(windows)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.map(1000000), 1000720)

    def test_silence_empty_and_constant_background(self):
        features, intervals = fixture()
        features.spectrogram[:] = 90
        result = at.estimate(features, intervals)
        self.assertTrue(all(s.confidence == "low" for s in result.segments))
        self.assertEqual(result.map(1000), 1000)
        empty = audio.compute_features(np.zeros(0, np.float32))
        self.assertEqual(at.estimate(empty, []).map(1000), 1000)
        self.assertEqual(at.speech_activity(empty).dtype, np.float32)

    def test_json_monotonic_and_reference(self):
        samples = [(i * 30000, round(i * 30000 * 1.001 + 440)) for i in range(40)]
        result = at.reference_map(samples, duration_ms=1200000)
        restored = at.TimingMap.from_json(result.to_json())
        self.assertEqual(restored.to_json(), result.to_json())
        mapped = [restored.map(t) for t in range(0, 1200000, 77)]
        self.assertTrue(all(a < b for a, b in zip(mapped, mapped[1:])))
        self.assertLess(abs(restored.map(900000) - 901340), 2)
        piecewise = at.reference_map([(i * 30000, i * 30000 + (500 if i < 20 else 15500))
                                     for i in range(40)], method="piecewise", duration_ms=1200000)
        self.assertEqual(len(piecewise.segments), 2)
        self.assertTrue(all(s.rate == 1 for s in piecewise.segments))

    def test_invalid_inputs_and_rounding(self):
        result = at.TimingMap([at.Segment(0, 100, 0.5)])
        self.assertEqual(result.map(0), 0)
        self.assertEqual(result.map(1), 2)
        for rate in (0, 1.06, float("nan")):
            with self.assertRaises(ValueError):
                at.TimingMap([at.Segment(0, 100, 0, rate)])
        with self.assertRaises(ValueError):
            at.reference_map([(0, float("nan"))])
        with self.assertRaises(ValueError):
            at.window_lags(np.array([float("nan")]), [(0, 1000)])
        with self.assertRaises(ValueError):
            at.fit([], method="bad")

    def test_bias_calibration_is_explicit(self):
        features, intervals = fixture(820)
        raw = at.estimate(features, intervals, correlation_bias_ms=0)
        corrected = at.estimate(features, intervals)
        self.assertEqual(raw.map(400000) - corrected.map(400000), at.CORRELATION_BIAS_MS)

    def test_rate_prior_rejects_spurious_slope_and_short_evidence(self):
        rng = np.random.default_rng(83)
        for span, slope in ((1200000, 0.0044), (240000, 0.001), (240000, 25 / 24 - 1)):
            windows = [at.WindowLag(int(t), int(t), 300 + slope * t + rng.normal(0, 50),
                                    0.2, "medium") for t in np.linspace(0, span, 25)]
            result = at.fit(windows)
            self.assertTrue(all(s.rate == 1.0 for s in result.segments))

    def test_rate_significance_and_physical_families(self):
        x = np.arange(0, 1200001, 30000)
        # A plausible but tiny noisy trend saves less than one saturated outlier.
        y = 500 + 0.0001 * x + np.where(np.arange(len(x)) % 2, 160, -160)
        candidate = y - (np.mean(y - 0.0001 * x) + 0.0001 * x)
        self.assertLess(at._loss(y - np.mean(y)) - at._loss(candidate), at.RATE_PENALTY)
        _, slope, _ = at._line(x, y, np.ones(len(x)), True)
        self.assertEqual(slope, 0.0)
        for rate in (1.001, 25 / 23.976, 23.976 / 25, 25 / 24, 24 / 25):
            with self.subTest(rate=rate):
                result = at.reference_map([(int(t), round(t * rate + 500)) for t in x])
                self.assertEqual(len(result.segments), 1)
                self.assertAlmostEqual(result.segments[0].rate, rate, delta=0.00001)

    def test_weak_segment_inherits_offset_without_speed(self):
        windows = [at.WindowLag(int(t), int(t), 500 + 0.001 * t, 0.5, "high")
                   for t in range(0, 600001, 30000)]
        windows += [at.WindowLag(t, t, 20000, 0.2, "medium") for t in (900000, 930000, 960000, 990000)]
        result = at.fit(windows, duration_ms=1200000)
        self.assertAlmostEqual(result.segments[0].rate, 1.001)
        self.assertEqual(result.segments[-1].rate, 1.0)


def snap_fixture(spans=((2000, 3000), (4500, 5300))):
    spectrum = np.full((800, 128), 30, np.uint8)
    bands = (audio.BAND_EDGES_HZ[:-1] >= 250) & (audio.BAND_EDGES_HZ[1:] <= 3500)
    for start, end in spans:
        spectrum[start // 10:end // 10, bands] = 180
    return audio.AudioFeatures(np.zeros(800), np.zeros(800), spectrum, np.zeros(800))


class SnapTests(unittest.TestCase):
    def test_modes_and_padding(self):
        f = snap_fixture()
        lines = [(1700, 3500), (4200, 5500)]
        for mode in ("start", "both", "keep_duration"):
            with self.subTest(mode=mode):
                result = at.snap_lines(f, lines, mode=mode, lead_in_ms=150, lead_out_ms=300)
                self.assertAlmostEqual(result[0][0], 1850, delta=30)
                if mode == "start":
                    self.assertEqual(result[0][1], 3500)
                elif mode == "both":
                    self.assertAlmostEqual(result[0][1], 3300, delta=30)
                else:
                    self.assertEqual(result[0][1] - result[0][0], 1800)
        self.assertEqual(lines, [(1700, 3500), (4200, 5500)])

    def test_skip_silence_and_search_bound_including_padding(self):
        f = snap_fixture()
        lines = [(1700, 3500), (6000, 6800)]
        self.assertEqual(at.snap_lines(f, lines, skip=lambda line: line[0] == 1700), lines)
        self.assertEqual(at.snap_lines(f, lines, search_ms=0), lines)
        bounded = at.snap_lines(f, lines, lead_in_ms=850, lead_out_ms=900, search_ms=200)
        for before, after in zip(lines, bounded):
            self.assertTrue(all(abs(a - b) <= 200 for a, b in zip(before, after)))
        f.spectrogram[:] = 90
        self.assertEqual(at.snap_lines(f, lines), lines)
        empty = audio.compute_features(np.empty(0, np.float32))
        self.assertEqual(at.snap_lines(empty, lines), lines)

    def test_partial_edges_and_duration_safety(self):
        f = snap_fixture(((2000, 4500),))
        self.assertEqual(at.snap_lines(f, [(1700, 3000)], mode="both")[0][1], 3000)
        result = at.snap_lines(f, [(2400, 2500)], mode="start", lead_in_ms=0)
        self.assertGreater(result[0][1], result[0][0])

    def test_order_overlaps_equal_spans_and_unsorted_input(self):
        f = snap_fixture()
        lines = [(2100, 3500), (1700, 3400), (1700, 3400), (2200, 2500)]
        result = at.snap_lines(f, lines)
        self.assertEqual(len(result), len(lines))
        self.assertEqual(result[1], result[2])
        self.assertTrue(any(a[0] < b[1] and b[0] < a[1]
                            for i, a in enumerate(result) for b in result[i + 1:]))
        for edge in (0, 1):
            for i, a in enumerate(lines):
                for j, b in enumerate(lines):
                    if a[edge] <= b[edge]:
                        self.assertLessEqual(result[i][edge], result[j][edge])

    def test_new_overlap_is_cut_with_unsorted_source_order(self):
        lines = [(2000, 3000), (1000, 1800)]
        with patch.object(at, "_speech_edges", side_effect=[(1900, 3000), (1000, 2100)]):
            self.assertEqual(at.snap_lines(snap_fixture(), lines, lead_in_ms=0),
                             [(1900, 3000), (1000, 1900)])
        self.assertEqual(lines, [(2000, 3000), (1000, 1800)])

    def test_cut_300ms_boundary_and_short_cut_reverts_end(self):
        lines = [(1000, 1400), (1500, 2500)]
        for next_start, expected in ((1300, [(1000, 1300), (1300, 2500)]),
                                     (1299, [(1000, 1400), (1500, 2500)])):
            with self.subTest(next_start=next_start), patch.object(
                    at, "_speech_edges", side_effect=[(1000, 1600), (next_start, 2500)]):
                self.assertEqual(at.snap_lines(snap_fixture(), lines, lead_in_ms=0), expected)

    def test_short_cut_reverts_only_end_if_that_resolves_collision(self):
        with patch.object(at, "_speech_edges", side_effect=[(1300, 1800), (1500, 2500)]):
            self.assertEqual(at.snap_lines(snap_fixture(), [(1000, 1400), (1500, 2500)], lead_in_ms=0),
                             [(1300, 1400), (1500, 2500)])

    def test_excluded_line_stays_put_and_existing_overlap_stays_allowed(self):
        with patch.object(at, "_speech_edges", return_value=(1600, 3000)):
            self.assertEqual(at.snap_lines(snap_fixture(), [(1000, 1800), (2000, 3000)],
                                           lead_in_ms=0, skip=lambda t: t[0] == 1000),
                             [(1000, 1800), (2000, 3000)])
        with patch.object(at, "_speech_edges", side_effect=[(1000, 2200), (1600, 3000)]):
            self.assertEqual(at.snap_lines(snap_fixture(), [(1000, 1800), (1700, 3000)], lead_in_ms=0),
                             [(1000, 2200), (1600, 3000)])

    def test_equal_spans_and_overlap_chain_do_not_hide_disjoint_pair(self):
        lines = [(1000, 1800), (1000, 1800), (1500, 2600), (2000, 3000)]
        with patch.object(at, "_speech_edges", side_effect=[(1000, 2200), (1000, 2200),
                                                          (1500, 2600), (2000, 3000)]):
            result = at.snap_lines(snap_fixture(), lines, lead_in_ms=0)
        self.assertEqual(result, [(1000, 2000), (1000, 2000), (1500, 2600), (2000, 3000)])

    def test_all_modes_resolve_new_overlap(self):
        for mode in ("start", "both", "keep_duration"):
            with self.subTest(mode=mode), patch.object(at, "_speech_edges", side_effect=[(1000, 1800), (1600, 3000)]):
                result = at.snap_lines(snap_fixture(), [(1000, 1800), (2000, 3000)], mode=mode, lead_in_ms=0)
                self.assertLessEqual(result[0][1], result[1][0])

    def test_varied_intervals_preserve_order_exclusions_and_no_new_overlaps(self):
        rng = np.random.default_rng(7361)
        for case in range(100):
            starts = rng.integers(0, 8000, 8)
            lines = [(int(s), int(s + d)) for s, d in zip(starts, rng.integers(1, 1600, 8))]
            edges = [(s + int(a), e + int(b)) for (s, e), (a, b) in
                     zip(lines[1:], rng.integers(-600, 601, (7, 2)))]
            with patch.object(at, "_speech_edges", side_effect=edges):
                result = at.snap_lines(snap_fixture(), lines, lead_in_ms=0, skip=lambda t: t == lines[0])
            self.assertEqual(result[0], lines[0], case)
            self.assertTrue(all(e > s for s, e in result), case)
            for i, a in enumerate(lines):
                for j, b in enumerate(lines):
                    if a[1] <= b[0]:
                        self.assertLessEqual(result[i][1], result[j][0], case)
                    for edge in (0, 1):
                        if a[edge] <= b[edge]:
                            self.assertLessEqual(result[i][edge], result[j][edge], case)

    def test_invalid_settings_and_intervals(self):
        f = snap_fixture()
        for settings in ({"mode": "bad"}, {"search_ms": -1}, {"lead_in_ms": float("nan")},
                         {"lead_out_ms": 0.5}):
            with self.assertRaises(ValueError):
                at.snap_lines(f, [(1000, 2000)], **settings)
        with self.assertRaises(ValueError):
            at.snap_lines(f, [(2000, 1000)])


class HintTimingTests(unittest.TestCase):
    def test_single_sides_clamp_against_untouched_current_edge(self):
        kw = dict(previous=(0, 1000), following=(2000, 3000))
        self.assertEqual(at.clamp_suggested_times((900, None), (1000, 2000), **kw), (1000, None))
        self.assertEqual(at.clamp_suggested_times((None, 2300), (1000, 2000), **kw), (None, 2000))
        for pair in ((2000, None), (None, 1000), (None, None)):
            self.assertIsNone(at.clamp_suggested_times(pair, (1000, 2000)))
        self.assertEqual(at.clamp_suggested_times((900, None), (1000, 2000),
                                                previous=(0, 1100)), (900, None))
        self.assertEqual(at.clamp_suggested_times((None, 2300), (1000, 2000),
                                                following=(1900, 3000)), (None, 2300))

    def test_run_on_end_and_start_use_same_inclusive_energy_valley(self):
        curve = np.zeros(450, np.float32)
        curve[90:370] = .5
        curve[205] = .2  # Still above threshold: no detected pause.
        self.assertEqual(at.hint_speech_edges(curve, 1000, 2000, following=(2200, 3600)), (900, 2050))
        self.assertEqual(at.hint_speech_edges(curve, 2200, 3600, previous=(1000, 2000)), (2050, 3700))
        # Overlapping speakers and absent neighbours do not invent a boundary.
        self.assertEqual(at.hint_speech_edges(curve, 1000, 2000, following=(1900, 3600)), (900, None))
        self.assertEqual(at.hint_speech_edges(curve, 1000, 2000), (900, None))
        curve[170] = .1  # Inclusive lower limit for the ending window.
        self.assertEqual(at.hint_speech_edges(curve, 1000, 2000, following=(2200, 3600))[1], 1700)

    def test_detected_pause_and_flat_audio_do_not_get_run_on_valleys(self):
        curve = np.zeros(450, np.float32)
        curve[90:180], curve[220:370] = .5, .5
        self.assertEqual(at.hint_speech_edges(curve, 1000, 2000, following=(2200, 3600)), (900, 1800))
        self.assertEqual(at.hint_speech_edges(np.ones(450), 1000, 2000,
                                             following=(2200, 3600)), (None, None))

    def test_run_on_with_decoded_synthetic_pcm_features(self):
        samples = np.zeros(audio.RATE * 5, np.float32)
        rng = np.random.default_rng(10)
        samples[900 * 16:3700 * 16] = rng.normal(0, .15, 2800 * 16)
        samples[2040 * 16:2070 * 16] *= .1  # Short dip is merged, not a pause.
        features = audio.compute_features(samples)
        activity = at._snap_activity(features)
        end = at.hint_speech_edges(activity, 1000, 2000, following=(2200, 3600))[1]
        start = at.hint_speech_edges(activity, 2200, 3600, previous=(1000, 2000))[0]
        self.assertEqual(start, end)
        self.assertLessEqual(abs(end - 2050), 30)

    def test_one_sided_samples_do_not_train_fabricated_calibration_edges(self):
        with patch.object(at, "speech_segments", return_value=[(1150, None)]), \
                patch.object(at, "hint_speech_edges", side_effect=AssertionError("synthetic boundary")):
            value = at.calibrate_hint_padding(np.zeros(1000), [(1000, 3000)] * 20,
                                              last_project=(100, 160))
        self.assertEqual((value.onset_samples, value.offset_samples), (20, 0))
        self.assertEqual((value.lead_in_ms, value.lead_out_ms, value.source), (100, 160, "project"))

    def test_snap_independent_edges_stay_independent(self):
        with patch.object(at, "_speech_edges", side_effect=[(1200, None), (None, 3900)]):
            self.assertEqual(at.snap_lines(snap_fixture(), [(1000, 2000), (3000, 4000)],
                                           lead_in_ms=100, lead_out_ms=100), [(1100, 2000), (3000, 4000)])

    def test_manual_learning_counts_groups_once_and_requires_both_boundaries(self):
        from subkit.model.document import Document, Group, Member
        doc = Document("manual")
        for i in range(20):
            gid = str(i)
            mids = [gid + "a", gid + "b"]
            doc.groups.append(Group(gid, "dialogue", mids))
            for mid in mids:
                doc.members[mid] = Member(mid, gid, "ja", "original", "dialogue", 1000, 2000,
                                          extra={"time_manual": True})
        with patch.object(at, "speech_segments", return_value=[(1150, None)]):
            result = at.calibrate_manual_hint_padding([(np.zeros(1), doc)], (100, 160))
            self.assertEqual((result.onset_samples, result.offset_samples), (20, 0))
            self.assertEqual((result.source, result.lead_in_ms, result.lead_out_ms), ("habit", 100, 160))
        with patch.object(at, "speech_segments", return_value=[(1600, 1400)]):
            result = at.calibrate_manual_hint_padding([(np.zeros(1), doc)], (100, 160))
            self.assertEqual((result.onset_samples, result.offset_samples), (20, 20))
            self.assertEqual((result.source, result.lead_in_ms, result.lead_out_ms), ("manual", 600, 600))


    def test_real_activity_median_and_twenty_sample_threshold(self):
        groups = [(i * 3000 + 1000, i * 3000 + 2000) for i in range(20)]
        curve = np.zeros(6200, np.float32)
        for i, (start, end) in enumerate(groups):
            curve[(start + 130 + i % 2 * 10) // 10:(end - 50) // 10] = 0.5
        actual = at.calibrate_hint_padding(curve, groups, last_project=(400, 100))
        self.assertEqual((actual.lead_in_ms, actual.lead_out_ms, actual.source), (135, 50, "episode"))
        self.assertEqual((actual.onset_samples, actual.offset_samples), (20, 20))
        short = at.calibrate_hint_padding(curve, groups[:19], last_project=(400, 100))
        self.assertEqual((short.lead_in_ms, short.lead_out_ms, short.source), (400, 100, "project"))

    def test_hint_calibration_skips_zero_and_reversed_groups(self):
        with patch.object(at, "speech_segments", return_value=[(1140, 2950)]) as segments:
            actual = at.calibrate_hint_padding(np.zeros(1), [(1000, 3000)] * 20 + [(0, 0), (500, 200)])
        self.assertEqual((actual.lead_in_ms, actual.lead_out_ms, actual.source), (140, 50, "episode"))
        self.assertEqual(segments.call_count, 20)

    def test_nearest_edges_radius_tie_and_clamps(self):
        groups = [(1000, 3000)] * 20
        for runs, values in (([(1100, 2900), (1200, 2800)], (100, 100)),
                             ([(900, 2900), (1100, 3100)], (0, 100)),
                             ([(1600, 3600)], (600, -200)),
                             ([(400, 2400)], (0, 600))):
            with self.subTest(values=values), patch.object(at, "speech_segments", return_value=runs):
                actual = at.calibrate_hint_padding(np.zeros(1), groups)
                self.assertEqual((actual.lead_in_ms, actual.lead_out_ms, actual.source), (*values, "episode"))
        with patch.object(at, "speech_segments", return_value=[(1601, 3601)]):
            actual = at.calibrate_hint_padding(np.zeros(1), groups)
            self.assertEqual((actual.onset_samples, actual.offset_samples, actual.source), (0, 0, "autotime"))

    def test_fallback_pair_order_and_insufficient_offsets(self):
        for runs in ([], [(1100, None)]):
            with patch.object(at, "speech_segments", return_value=runs):
                actual = at.calibrate_hint_padding(np.zeros(1), [(1000, 3000)] * 20,
                                                   autotime_values=(210, 40))
                self.assertEqual((actual.lead_in_ms, actual.lead_out_ms, actual.source), (210, 40, "autotime"))
                previous = at.calibrate_hint_padding(np.zeros(1), [(1000, 3000)] * 20,
                                                     last_project=(0, -100), autotime_values=(210, 40))
                self.assertEqual((previous.lead_in_ms, previous.lead_out_ms, previous.source), (0, -100, "project"))

    def test_clamp_sides_touching_and_empty(self):
        kw = {"previous": (100, 1000), "following": (2000, 2500)}
        self.assertEqual(at.clamp_suggested_times((900, 2200), (1000, 2000), **kw), (1000, 2000))
        self.assertEqual(at.clamp_suggested_times((900, 1900), (1000, 2000), **kw), (1000, 1900))
        self.assertEqual(at.clamp_suggested_times((1100, 2200), (1000, 2000), **kw), (1100, 2000))
        self.assertIsNone(at.clamp_suggested_times((900, 1000), (1000, 2000), **kw))
        self.assertIsNone(at.clamp_suggested_times((2000, 2000), (1000, 2000)))
        self.assertEqual(at.clamp_suggested_times((900, 2200), (1000, 2000)), (900, 2200))

    def test_preexisting_overlap_exempts_only_its_side(self):
        self.assertEqual(at.clamp_suggested_times((800, 2300), (1000, 2000),
                                                 previous=(500, 1100), following=(2100, 2500)), (800, 2100))
        self.assertEqual(at.clamp_suggested_times((800, 2300), (1000, 2000),
                                                 previous=(500, 900), following=(1900, 2500)), (900, 2300))
        self.assertEqual(at.clamp_suggested_times((800, 2300), (1000, 2000),
                                                 previous=(500, 1100), following=(1900, 2500)), (800, 2300))

class SpeechUnderMusic(unittest.TestCase):
    @staticmethod
    def signal(gap=None):
        t = np.arange(4 * audio.RATE) / audio.RATE
        music = (0.1 * np.sin(2 * np.pi * 220 * t)
                 + 0.04 * np.sin(2 * np.pi * 880 * t)
                 + 0.03 * np.sin(2 * np.pi * 1320 * t))
        if gap is None:
            return music.astype(np.float32)
        rng = np.random.default_rng(20261006)
        spectrum = np.fft.rfft(rng.normal(size=len(t)))
        hz = np.fft.rfftfreq(len(t), 1 / audio.RATE)
        spectrum[(hz < 350) | (hz > 3000)] = 0
        voice = np.fft.irfft(spectrum, n=len(t))
        voice /= np.std(voice)
        syllables = 0.7 + 0.3 * np.sin(2 * np.pi * 5 * t)
        audible = ((t >= 1) & (t < 1.6)) | ((t >= 1.6 + gap) & (t < 2.4))
        return (music + 0.12 * voice * syllables * audible).astype(np.float32)

    def test_separate_modulated_speech_bursts_over_steady_music(self):
        features = audio.compute_features(self.signal(0.12))
        curve = at._snap_activity(features)
        runs = at.speech_segments(curve, 500, 3000)
        self.assertEqual(len(runs), 2)
        for actual, expected in zip(runs, ((1000, 1600), (1720, 2400))):
            for edge, reference in zip(actual, expected):
                self.assertLessEqual(abs(edge - reference), 30)
        self.assertLessEqual(runs[0][1], runs[1][0])

    def test_short_syllable_pause_still_joins_over_music(self):
        curve = at._snap_activity(audio.compute_features(self.signal(0.06)))
        runs = at.speech_segments(curve, 500, 3000)
        self.assertEqual(len(runs), 1)
        self.assertLessEqual(abs(runs[0][0] - 1000), 30)
        self.assertLessEqual(abs(runs[0][1] - 2400), 30)

    def test_steady_tonal_music_without_speech_abstains(self):
        curve = at._snap_activity(audio.compute_features(self.signal()))
        self.assertEqual(at.speech_segments(curve, 500, 3000), [])

    def test_join_boundary_and_truncated_edges(self):
        for gap, count in ((8, 1), (9, 2)):
            curve = np.zeros(150, np.float32)
            curve[30:60] = 0.5
            curve[60 + gap:100] = 0.5
            with self.subTest(gap_ms=gap * audio.HOP_MS):
                self.assertEqual(len(at.speech_segments(curve, 0, 1500)), count)
                cut_start = at.speech_segments(curve, 400, 1200)
                cut_end = at.speech_segments(curve, 100, 900)
                self.assertIsNone(cut_start[0][0])
                self.assertIsNone(cut_end[-1][1])


if __name__ == "__main__":
    unittest.main()
