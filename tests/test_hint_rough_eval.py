"""Real-rough evaluation pairing, denominators and answer isolation."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval import hint_rough_real as ev
from subkit.sources.srt import parse_srt


def row(text, start=1000, end=2000, identity="r"):
    return dict(text=text, start=start, end=end, id=identity)


class RoughPairTests(unittest.TestCase):
    def test_repeated_text_is_monotonic_unique_and_unmatched_is_retained(self):
        source = [row("你好", 1000), row("你好", 3000), row("完全不同的长句", 5000)]
        gold = [row("你好", 1100), row("你好", 3100)]
        self.assertEqual(ev.pair_rows(source, gold, language="zh"), [0, 1, -1])

    def test_fifteen_seconds_is_in_ms_inclusive(self):
        self.assertEqual(ev.pair_rows([row("hello", 1000)], [row("hello", 16000)], language="zh"), [0])
        self.assertEqual(ev.pair_rows([row("hello", 1000)], [row("hello", 16001)], language="zh"), [-1])

    def test_simplified_traditional_and_japanese_metadata(self):
        self.assertEqual(ev.pair_rows([row("这个时间")], [row("這個時間！")], language="zh"), [0])
        self.assertEqual(ev.pair_rows([row("（七実）遠月(とおつき)学園")],
                                     [row("遠月学園")], language="ja"), [0])
        self.assertEqual(ev.pair_rows([row("♪〜"), row("")], [row("")], language="ja"), [-1, -1])

    def test_srt_rounding_matches_product_not_float_seconds(self):
        cue = parse_srt("1\n00:00:01,004 --> 00:00:02,005\nhello\n")[0]
        self.assertEqual((cue.start * 10, cue.end * 10), (1000, 2010))


class RoughScoreTests(unittest.TestCase):
    def test_default_padding_matches_shipped_habit(self):
        from subkit_app.views.waveform import HABIT_INITIAL_MS
        self.assertEqual(ev.DEFAULT_PADDING, HABIT_INITIAL_MS)

    def scored(self, rough, gold, suggestion, gold_suggestion=None):
        adopted = tuple(c if s is None else s for c, s in zip(rough, suggestion)) if suggestion else rough
        return dict(rough_ms=rough, gold_ms=gold, suggested_ms=suggestion,
                    adopted_ms=adopted, gold_suggested_ms=gold_suggestion, idempotent=True)

    def test_abstention_and_unknown_keep_full_denominator(self):
        rows = [self.scored((1000, 2000), (1000, 2000), None),
                self.scored((9000, 10000), None, (9000, None))]
        scores = ev.score_rows(rows)
        self.assertEqual(scores["start"]["denominator"], 2)
        self.assertEqual(scores["start"]["unknown"], 1)
        self.assertEqual(scores["start"]["covered"], 1)
        self.assertEqual(scores["start"]["suggestion_within_200ms"], 0)
        self.assertEqual(scores["start"]["adopted_within_200ms"], 1)
        self.assertEqual(scores["end"]["abstained"], 2)
        self.assertEqual(scores["start"]["gold_kept"], 1)

    def test_thresholds_made_worse_better_severe_and_gold_kept(self):
        rows = [self.scored((1100, 2200), (1000, 2000), (1201, 2000), (1201, 2200)),
                self.scored((2001, 3200), (1000, 2000), (1100, 3200))]
        scores = ev.score_rows(rows)
        self.assertEqual(scores["start"]["made_worse"], 1)
        self.assertEqual(scores["start"]["made_better"], 1)
        self.assertEqual(scores["start"]["baseline_severe_gt_1000ms"], 1)
        self.assertEqual(scores["start"]["suggestion_within_100ms"], 1)
        self.assertEqual(scores["end"]["gold_moved_beyond_200ms"], 0)
        self.assertEqual(scores["start"]["gold_moved_beyond_200ms"], 1)

    def test_candidates_have_no_gold_and_neighbours_remain_rough(self):
        self.assertEqual(list(inspect.signature(ev.generate).parameters), ["activity", "rough", "padding"])
        rough = [row("first", 1000, 2000), row("second", 3000, 4000)]
        gold = [row("first", 1500, 2500), row("second", 3500, 4500)]
        calls = []
        def suggest(activity, start, end, **kwargs):
            calls.append(((start, end), kwargs))
            return None
        with patch.object(ev.autotime, "suggest_group_times", suggest):
            result = ev.evaluate(np.zeros(600), rough, gold, [0, 1], (100, 160))
        self.assertEqual(calls[0][0], (1000, 2000))
        self.assertEqual(calls[2][0], (1500, 2500))
        self.assertEqual(calls[2][1]["following"], (3000, 4000))
        self.assertEqual(result["rows"][0]["gold_id"], "r")

    def test_leave_one_episode_out_excludes_gold_and_fallback(self):
        samples = {6: ([600] * 20, [600] * 20), 7: ([123] * 20, [234] * 20)}
        calibration = ev.leave_one_out(samples, 6)
        self.assertEqual(calibration["training_episodes"], [7])
        self.assertEqual((calibration["lead_in_ms"], calibration["lead_out_ms"]), (123, 234))
        fallback = ev.leave_one_out({6: samples[6]}, 6)
        self.assertEqual((fallback["lead_in_ms"], fallback["lead_out_ms"]), (100, 160))

    def test_idempotence_failure_is_reported_separately(self):
        rough = [row("a")]
        with patch.object(ev.autotime, "suggest_group_times", side_effect=[(1100, None), (1200, None)]):
            candidates = ev.generate(np.zeros(600), rough, (100, 160))
        data = self.scored((1000, 2000), (1000, 2000), (1100, None))
        data.update(candidates[0])
        self.assertEqual(ev.score_rows([data])["idempotence_failed"], 1)


if __name__ == "__main__":
    unittest.main()
