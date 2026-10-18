"""Episode-sized reference fits with robust rates and real cut boundaries."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit import autotime as at


def reference_samples(lines=300, rate=1.0, jump=0, second_jump=0):
    rng = np.random.default_rng(517)
    samples, intervals = [], []
    starts = np.r_[np.linspace(1000, 698000, lines // 2, dtype=int),
                   np.linspace(740000, 1437000, lines - lines // 2, dtype=int)]
    for i, start in enumerate(starts):
        end = int(start) + 1800
        intervals.append((int(start), end))
        cut = jump if i >= lines // 2 else 0
        cut += second_jump if i >= lines * 3 // 4 else 0
        for t in (int(start), end):
            samples.append((t, round(t * rate + 500 + cut + rng.normal(0, 8))))
    return samples, intervals


class ReferenceScaleTests(unittest.TestCase):
    def test_episode_single_segment_scale(self):
        for lines in (100, 200, 300):
            samples, intervals = reference_samples(lines)
            started = time.perf_counter()
            mapping = at.reference_map(samples, intervals=intervals, duration_ms=1440000)
            elapsed = time.perf_counter() - started
            print(f"reference_map {lines} lines / {len(samples)} samples: {elapsed:.6f} s", flush=True)
            self.assertLess(elapsed, 10, "episode reference fit regressed")
            self.assertEqual(len(mapping.segments), 1)
            self.assertEqual(mapping.segments[0].confidence, "high")
            self.assertEqual(mapping.segments[0].rate, 1)
            self.assertLess(abs(mapping.map(900000) - 900500), 3)

    def test_dense_rate_with_outliers(self):
        for rate in (1.001, 25 / 23.976, 23.976 / 25):
            with self.subTest(rate=rate):
                samples, intervals = reference_samples(rate=rate)
                samples = [(t, s + (30000 if i % 37 == 5 else 0))
                           for i, (t, s) in enumerate(samples)]
                mapping = at.reference_map(samples, intervals=intervals, duration_ms=1440000)
                self.assertEqual(len(mapping.segments), 1)
                self.assertAlmostEqual(mapping.segments[0].rate, rate, delta=0.00002)
                self.assertEqual(mapping.segments[0].confidence, "high")
                for t in (100000, 1300000):
                    self.assertLess(abs(mapping.map(t) - round(t * rate + 500)), 8)

    def test_dense_rate_does_not_hide_positive_or_negative_cut(self):
        for jump in (200, 300, -300, 1500, -1500, 15000, -15000):
            with self.subTest(jump=jump):
                samples, intervals = reference_samples(rate=1.001, jump=jump)
                mapping = at.reference_map(samples, intervals=intervals, duration_ms=1440000)
                self.assertEqual(len(mapping.segments), 2)
                self.assertLess(abs(mapping.segments[1].start_ms - 720000), 6000)
                for t, cut in ((200000, 0), (1200000, jump)):
                    self.assertLess(abs(mapping.map(t) - round(t * 1.001 + 500 + cut)), 10)
                self.assertTrue(all(s.confidence == "high" for s in mapping.segments))

    def test_two_cuts_and_overlapping_intervals(self):
        samples, intervals = reference_samples(jump=3000, second_jump=-8000)
        intervals += [(a + 200, b - 200) for a, b in intervals]
        mapping = at.reference_map(samples, intervals=intervals, method="piecewise", duration_ms=1440000)
        self.assertEqual(len(mapping.segments), 3)
        self.assertTrue(all(s.rate == 1 for s in mapping.segments))
        for t, offset in ((200000, 500), (900000, 3500), (1300000, -4500)):
            self.assertLess(abs(mapping.map(t) - t - offset), 5)
        for segment in mapping.segments[1:]:
            self.assertFalse(any(a < segment.start_ms < b for a, b in intervals))

    def test_duplicate_endpoints_and_low_confidence(self):
        samples, _ = reference_samples()
        mapping = at.reference_map(samples + samples)
        self.assertEqual(len(mapping.segments), 1)
        weak = at.fit([at.WindowLag(t, t, s - t, 0.1, "low") for t, s in samples])
        self.assertEqual(weak.segments[0].confidence, "low")
        self.assertEqual(weak.map(900000), 900000)
