"""Synthetic CTC paths, inclusive windows, holds, and prototype parity."""

from __future__ import annotations

import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.kalign.viterbi import align  # noqa: E402


def emission(symbols, vocab_size=4):
    lp = np.full((len(symbols), vocab_size), -20.0)
    lp[np.arange(len(symbols)), symbols] = 0.0
    return lp


class Viterbi(unittest.TestCase):
    def test_per_beat_minimum_empty_multitoken_and_entry_anchor(self):
        lp = emission([1, 2, 0, 0, 0, 3, 0, 0, 0, 0])
        first, last, _ = align(lp, [[1, 2], [3]], [(0, 9)] * 2,
                               min_frames=[5, 5], start_windows=[(0, 0), (5, 5)])
        np.testing.assert_array_equal(first, [0, 5])
        self.assertGreaterEqual(first[1] - first[0], 5)
        with self.assertRaisesRegex(ValueError, "no valid CTC path"):
            align(lp[:-1], [[1, 2], [3]], [(0, 9)] * 2, min_frames=[5, 5])
        first, _, _ = align(emission([0] * 10), [[], []], [(0, 9)] * 2, min_frames=[5, 5])
        self.assertGreaterEqual(first[1] - first[0], 5)

    def test_start_window_restricts_entry_not_token_self_loop(self):
        self.assert_frames(align(emission([0, 1, 1, 1, 0]), [[1]], [(0, 4)],
                                 start_windows=[(1, 1)]), [1], [3])

    def test_optional_minimum_inputs_and_default_parity(self):
        lp = emission([1, 0, 2])
        np.testing.assert_array_equal(align(lp, [[1], [2]], [(0, 2)] * 2)[0],
                                      align(lp, [[1], [2]], [(0, 2)] * 2, min_frames=[0, 0])[0])
        for kwargs in ({"min_frames": [1]}, {"min_frames": [-1, 2]}, {"min_frames": [1.5, 2]},
                       {"start_windows": [(0, 1)]}, {"start_windows": [(1, 0), (0, 2)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                align(lp, [[1], [2]], [(0, 2)] * 2, **kwargs)

    def assert_frames(self, actual, first, last):
        np.testing.assert_array_equal(actual[0], first)
        np.testing.assert_array_equal(actual[1], last)
        self.assertTrue(np.isfinite(actual[2]))

    def test_distinct_tokens_skip_blank(self):
        self.assert_frames(align(emission([1, 2]), [[1], [2]], [(0, 1)] * 2), [0, 1], [0, 1])

    def test_nonzero_blank_and_leading_trailing_blanks(self):
        self.assert_frames(align(emission([3, 0, 0, 3, 1, 3]), [[0], [1]], [(0, 5)] * 2, blank=3),
                           [1, 4], [2, 4])

    def test_repeated_tokens_need_blank_between_beats(self):
        self.assert_frames(align(emission([1, 0, 1]), [[1], [1]], [(0, 2)] * 2), [0, 2], [0, 2])
        with self.assertRaisesRegex(ValueError, "no valid CTC path"):
            align(emission([1, 1]), [[1], [1]], [(0, 1)] * 2)

    def test_repeated_tokens_inside_one_beat(self):
        self.assert_frames(align(emission([1, 0, 1, 2]), [[1, 1, 2]], [(0, 3)]), [0], [3])

    def test_inclusive_window_endpoints(self):
        self.assert_frames(align(emission([0, 1, 0, 2, 0]), [[1], [2]], [(1, 1), (3, 3)]),
                           [1, 3], [1, 3])

    def test_window_cannot_be_ignored_for_better_emission(self):
        self.assert_frames(align(emission([1, 0, 0, 0]), [[1]], [(2, 2)]), [2], [2])

    def test_hold_requires_three_extra_frames_and_excludes_them(self):
        self.assert_frames(align(emission([1, 0, 0, 0, 2, 0, 0, 0]), [[1], [2]],
                                 [(0, 3), (4, 7)], hold=3), [0, 4], [0, 4])
        with self.assertRaisesRegex(ValueError, "no valid CTC path"):
            align(emission([1, 0, 0]), [[1]], [(0, 2)], hold=3)

    def test_hold_emits_token_or_blank_and_following_blank_is_optional(self):
        self.assert_frames(align(emission([1, 1, 0, 1, 1, 0, 1, 0]), [[1], [1]],
                                 [(0, 3), (4, 7)], hold=3), [0, 4], [0, 4])

    def test_hold_window_also_restricts_extra_frames(self):
        with self.assertRaisesRegex(ValueError, "no valid CTC path"):
            align(emission([1, 0, 0, 0, 0]), [[1]], [(0, 2)], hold=3)

    def test_empty_beat_is_one_timed_blank_state_without_hold(self):
        self.assert_frames(align(emission([0]), [[]], [(0, 0)]), [0], [0])
        self.assert_frames(align(emission([1, 0, 0, 0, 2]), [[1], [], [2]],
                                 [(0, 0), (2, 2), (4, 4)]), [0, 2, 4], [0, 2, 4])

    def test_empty_beats_with_hold_and_optional_blanks(self):
        self.assert_frames(align(emission([0] * 8), [[], []], [(0, 3), (4, 7)], hold=3),
                           [0, 4], [0, 4])

    def test_terminal_token_does_not_require_trailing_blank(self):
        self.assert_frames(align(emission([1]), [[1]], [(0, 0)]), [0], [0])

    def test_no_beats_and_zero_frames(self):
        result = align(np.array([[-0.4, -1.0], [-0.2, -2.0]]), [], [])
        self.assert_frames(result, [], [])
        self.assertAlmostEqual(result[2], -0.6)
        self.assertEqual(align(np.empty((0, 2)), [], [])[2], 0.0)
        with self.assertRaisesRegex(ValueError, "no emission frames"):
            align(np.empty((0, 2)), [[1]], [(0, 0)])

    def test_impossible_windows_and_posteriors_raise(self):
        for lp, beats, windows, hold in ((emission([0] * 4), [[1]], [(5, 8)], 0),
                                         (emission([0] * 4), [[1]], [(-5, -1)], 0),
                                         (emission([0] * 4), [[1], [2]], [(3, 3), (0, 0)], 0),
                                         (np.full((5, 3), -np.inf), [[1]], [(0, 4)], 0),
                                         (emission([0] * 2), [[]], [(0, 1)], 3)):
            with self.subTest(windows=windows, hold=hold), self.assertRaisesRegex(ValueError, "no valid CTC path"):
                align(lp, beats, windows, hold=hold)

    def test_negative_infinity_and_very_low_finite_scores(self):
        lp = np.full((3, 3), -np.inf)
        lp[0, 1] = lp[1, 0] = lp[2, 2] = -2e9
        self.assert_frames(align(lp, [[1], [2]], [(0, 2)] * 2), [0, 2], [0, 2])
        self.assertEqual(align(lp, [[1], [2]], [(0, 2)] * 2)[2], -6e9)

    def test_invalid_inputs(self):
        lp = emission([0, 1, 0])
        invalid = [(np.zeros(3), [[1]], [(0, 2)], 0, 0),
                   (np.zeros((3, 3), dtype=int), [[1]], [(0, 2)], 0, 0),
                   (lp, [[1]], [], 0, 0), (lp, [[1]], [(2, 1)], 0, 0),
                   (lp, [[1]], [(0.5, 2)], 0, 0), (lp, [[1]], [(0,)], 0, 0),
                   (lp, [[1]], [(0, 2)], -1, 0), (lp, [[1]], [(0, 2)], 4, 0),
                   (lp, [[1]], [(0, 2)], 0, -1), (lp, [[1]], [(0, 2)], 0, 1.5),
                   (lp, [[0]], [(0, 2)], 0, 0), (lp, [[9]], [(0, 2)], 0, 0),
                   (lp, [[1.5]], [(0, 2)], 0, 0),
                   (np.full((3, 4), np.nan), [[1]], [(0, 2)], 0, 0),
                   (np.full((3, 4), np.inf), [[1]], [(0, 2)], 0, 0)]
        for args in invalid:
            with self.subTest(args=args[1:]), self.assertRaises(ValueError):
                align(*args)

    def test_float32_and_blank_index_permutation(self):
        lp = emission([0, 1, 2, 0, 2, 0]).astype(np.float32)
        original = align(lp, [[1, 2], [2]], [(0, 5)] * 2)
        permuted = align(lp[:, [1, 2, 3, 0]], [[0, 1], [1]], [(0, 5)] * 2, blank=3)
        self.assert_frames(permuted, original[0], original[1])
        self.assertEqual(permuted[2], original[2])

    def test_prototype_final_row_score_convention(self):
        lp = np.zeros((3, 2))
        lp[:, 1] = -5.0
        result = align(lp, [[1]], [(1, 1)])
        self.assert_frames(result, [1], [1])
        self.assertEqual(result[2], 0.0)  # Initial blank state scores above the completed path (-5).

    def test_short_ctc_paths_against_exhaustive_symbol_paths(self):
        rng = np.random.default_rng(24013)
        for tokens in ([1, 2], [1, 1], [2, 1, 2]):
            lp = rng.normal(-2, 0.5, (5, 3))
            candidates = []
            for symbols in itertools.product(range(3), repeat=5):
                collapsed = [symbol for i, symbol in enumerate(symbols)
                             if symbol != 0 and (i == 0 or symbols[i - 1] != symbol)]
                if collapsed == tokens:
                    candidates.append((sum(lp[t, symbol] for t, symbol in enumerate(symbols)), symbols))
            _, best = max(candidates)
            frames = [t for t, symbol in enumerate(best) if symbol != 0]
            self.assert_frames(align(lp, [tokens], [(0, 4)]), [frames[0]], [frames[-1]])

    def test_synthetic_prototype_parity(self):
        spec = importlib.util.spec_from_file_location("kalign_prototype", ROOT / "tools/experiments/kalign/ctc_align.py")
        prototype = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prototype)
        rng = np.random.default_rng(20261013)
        for hold in (0, 1, 3):
            for blank in (0, 3):
                symbols = [i for i in range(4) if i != blank]
                beats = [[symbols[0], symbols[1]], [], [symbols[1]], [symbols[2], symbols[2]]]
                for trial in range(8):
                    lp = np.zeros((48, 4)) if trial == 0 else rng.normal(-2, 1, (48, 4))
                    windows = [(0, 15), (10, 25), (18, 35), (28, 47)]
                    expected = prototype.align(lp, beats, windows, blank, hold)
                    actual = align(lp, beats, windows, blank, hold)
                    with self.subTest(hold=hold, blank=blank, trial=trial):
                        self.assert_frames(actual, expected[0], expected[1])
                        self.assertEqual(actual[2], expected[2])


if __name__ == "__main__":
    unittest.main()
