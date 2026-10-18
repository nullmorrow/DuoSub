"""Banded alignment keeps both inputs across long one-sided runs."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit.align.dp2 import Aligner2, Row, Weights2
from subkit.core.units import Unit


def unit(i, start):
    u = Unit(i, "o1", i, 0, 0, 4, "test", start, start + 500)
    u.wstart, u.wend = u.start, u.end
    return u


class BandTests(unittest.TestCase):
    def assert_kept(self, result, rows, units):
        self.assertEqual([i for b in result.blocks for i in b.rows], list(range(len(rows))))
        self.assertEqual([i for b in result.blocks for i in b.units], list(range(len(units))))

    def test_appendix_a_initial_state(self):
        for count in (39, 40, 41, 60):
            with self.subTest(count=count):
                units = [unit(i, i * 1000) for i in range(count)] + [unit(count, 100000)]
                rows = [Row(0, 100000, 100500, "test", 0)]
                result = Aligner2(rows, units).run()
                self.assert_kept(result, rows, units)
                self.assertEqual(result.blocks[-1].kind, "pair")
                self.assertEqual(result.blocks[-1].units, [count])
                print(f"DP {count} OK {len(result.blocks)} blocks; every unit once", flush=True)

    def test_middle_and_trailing_original_runs(self):
        units = [unit(i, i * 1000) for i in range(181)]
        rows = [Row(i, t, t + 500, "test", i) for i, t in enumerate((0, 90000, 120000))]
        for band in (0, 1, 40):
            with self.subTest(band=band):
                result = Aligner2(rows, units, weights=Weights2(band=band)).run()
                self.assert_kept(result, rows, units)
                pairs = [b for b in result.blocks if b.kind == "pair"]
                self.assertEqual([b.rows for b in pairs], [[0], [1], [2]])
                self.assertEqual([b.units for b in pairs], [[0], [90], [120]])

    def test_translation_only_runs_before_between_and_after_matches(self):
        rows = [Row(i, i * 1000, i * 1000 + 500, "test", i) for i in range(181)]
        units = [unit(i, t) for i, t in enumerate((60000, 120000))]
        result = Aligner2(rows, units, weights=Weights2(band=0)).run()
        self.assert_kept(result, rows, units)
        self.assertEqual([b.rows for b in result.blocks if b.kind == "pair"], [[60], [120]])

    def test_empty_inputs(self):
        for row_count, unit_count in ((0, 0), (0, 100), (100, 0)):
            with self.subTest(rows=row_count, units=unit_count):
                rows = [Row(i, i * 1000, i * 1000 + 500, "test", i) for i in range(row_count)]
                units = [unit(i, i * 1000) for i in range(unit_count)]
                self.assert_kept(Aligner2(rows, units).run(), rows, units)

    def test_no_path_fallback_preserves_source_order_and_visibility(self):
        # No unit-consuming transition: exercise the defensive no-path branch.
        rows = [Row(i, t, t + 500, "test", i) for i, t in enumerate((1000, 500, 3000))]
        units = [unit(i, t) for i, t in enumerate((0, 2000, 1500, 4000))]
        units[0].interj = True
        result = Aligner2(rows, units, weights=Weights2(max_units=0, max_joint_rows=1),
                          filler_mode="compact").run()
        self.assert_kept(result, rows, units)
        self.assertTrue(all(not b.hidden for b in result.blocks))
        self.assertTrue(all("alignment_fallback" in b.doubts for b in result.blocks))

    def test_band_stays_bounded_after_ordinary_matches(self):
        count = 200
        units = [unit(i, i * 10000) for i in range(count)]
        rows = [Row(i, u.start, u.end, "test", i) for i, u in enumerate(units)]
        aligner = Aligner2(rows, units, weights=Weights2(band=2))
        calls = []
        original = aligner.assign_score
        def counted(i, j, b):
            calls.append(1)
            return original(i, j, b)
        aligner.assign_score = counted
        result = aligner.run()
        self.assert_kept(result, rows, units)
        self.assertEqual([b.units for b in result.blocks if b.kind == "pair"],
                         [[i] for i in range(count)])
        self.assertLess(len(calls), count * 20)

    def test_fallback_propagates_cancellation(self):
        rows = [Row(0, 1000, 1500, "test", 0)]
        calls = []
        def cancel():
            calls.append(1)
            if len(calls) == 3:
                raise InterruptedError("cancelled in fallback")
        aligner = Aligner2(rows, [unit(0, 0)], cancel_check=cancel,
                           weights=Weights2(max_units=0, max_joint_rows=1))
        with self.assertRaisesRegex(InterruptedError, "cancelled in fallback"):
            aligner.run()
