"""Evaluation denominators, inclusive tolerance and the cross-episode gate."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit import autotime
from subkit.model.document import Document, Group, Member
from tools.eval import hint_lines_real as evaluation


class HintEvaluation(unittest.TestCase):
    def test_one_sided_denominators_and_inclusive_error_limits(self):
        style = dict(lead_in_ms=0, lead_out_ms=0, linking={"threshold_ms": None})
        actual = evaluation.evaluate_gold([(1000, 2000), (3000, 4000), (5000, 6000)],
                                          ["a", "b", "c"], [(1100, None), (None, 4200), (None, None)],
                                          style, one_sided=True)
        self.assertEqual(actual["coverage"], 2 / 3)
        self.assertEqual((actual["start_coverage"], actual["end_coverage"]), (1 / 3, 1 / 3))
        self.assertEqual((actual["start_within_100ms"], actual["end_within_200ms"]), (1, 1))
        self.assertEqual((actual["median_abs_start_ms"], actual["median_abs_end_ms"]), (100, 200))
        self.assertEqual(actual["rows"][0]["suggested_ms"], (1100, None))
        self.assertIsNone(actual["rows"][0]["end_error_ms"])

    def test_x10_gate_rejects_either_side_accuracy_loss(self):
        before = dict(start_coverage=.8, end_coverage=.8, new_collisions=0,
                      start_within_100ms=.7, start_within_200ms=.9,
                      end_within_100ms=.5, end_within_200ms=.8)
        after = dict(before, start_coverage=.9, end_coverage=.9, end_within_200ms=.781)
        self.assertTrue(evaluation.x10_gate(before, after))
        after["end_within_200ms"] = .779
        self.assertFalse(evaluation.x10_gate(before, after))

    def test_gold_ass_filters_exact_dialogue_styles_and_scores_jp_once(self):
        import os
        import tempfile
        styles = ("JP", "CN", "Furigana", "JPOPED", "JPTop", "Staff", "Logo", "Website")
        text = "\n".join(f"Dialogue: 0,0:00:01.00,0:00:02.00,{style},,0,0,0,,台詞"
                         for style in styles)
        with tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as scratch:
            path = Path(scratch) / "gold.ass"
            path.write_text(text, encoding="utf-8")
            doc, counts = evaluation.gold_document(path)
        self.assertEqual(len(doc.groups), 1)
        self.assertEqual(doc.members["m0"].text, "台詞")
        self.assertEqual((counts["jp_lines"], counts["cn_lines"], counts["cn_equal_jp_span"]), (1, 1, 1))

    def test_two_show_habit_pools_samples_not_episode_medians(self):
        def episode(name, onsets, offsets):
            return {"name": name, "styles": {"energy": {"lead_in_samples": onsets, "lead_out_samples": offsets}}}
        actual = evaluation.pooled_habit([episode("utena", [150] * 5, [200] * 5),
                                          episode("overlord", [300], [-100])])
        self.assertEqual((actual["lead_in_ms"], actual["lead_out_ms"]), (150, 200))
        self.assertEqual((actual["onset_samples"], actual["offset_samples"]), (6, 6))
        self.assertEqual(actual["training_episodes"], ["utena", "overlord"])

    def test_cross_episode_calibration_excludes_scored_episode_and_chained_offsets(self):
        episodes = []
        for name, lead in (("train", 100), ("heldout", 500)):
            times = [(1000, 2000), (2000, 3000), (5000, 6000)]
            edges = [(1000 + lead, 1700), (2000 + lead, 2960), (5000 + lead, 5960)]
            style = evaluation.timing_style(times, edges)
            episodes.append(dict(name=name, times=times, edges={"energy": edges}, styles={"energy": style}))
        actual = evaluation.cross_episode_style(episodes, "heldout", "energy")
        self.assertEqual((actual["lead_in_ms"], actual["lead_out_ms"]), (100, 40))
        self.assertEqual(actual["training_episodes"], ["train"])
        self.assertEqual(episodes[0]["styles"]["energy"]["touching"], 1)
        self.assertEqual(episodes[0]["styles"]["energy"]["lead_out_samples"], [40, 40])

    def test_linking_requires_data_and_exact_touching_precision(self):
        self.assertFalse(evaluation.fit_linking([(50, True)] * 19)["supported"])
        self.assertFalse(evaluation.fit_linking([(50, True)] * 15 + [(50, False)] * 5)["supported"])
        fitted = evaluation.fit_linking([(50, True)] * 16 + [(50, False)] * 4 + [(100, False)] * 10)
        self.assertEqual(fitted["threshold_ms"], 50)
        self.assertEqual(fitted["precision"], .8)

    def test_gold_metrics_include_100_200ms_and_abstentions_in_overall_rates(self):
        style = dict(lead_in_ms=0, lead_out_ms=0, linking={"threshold_ms": None})
        actual = evaluation.evaluate_gold([(1000, 2000), (3000, 4000), (5000, 6000)],
                                          ["a", "b", "c"], [(1100, 2200), (3200, 4201), (None, None)], style)
        self.assertEqual(actual["coverage"], 2 / 3)
        self.assertEqual(actual["start_within_100ms"], .5)
        self.assertEqual(actual["start_within_200ms"], 1)
        self.assertEqual(actual["start_all_within_200ms"], 2 / 3)
        self.assertEqual(actual["end_within_200ms"], .5)
        self.assertEqual(actual["median_abs_start_ms"], 150)

    def test_gold_linking_respects_overlaps_and_unclamped_control(self):
        style = dict(lead_in_ms=100, lead_out_ms=0, linking={"threshold_ms": 100})
        times = [(1000, 2000), (2000, 3000), (2800, 4000)]
        edges = [(1100, 1900), (2100, 2900), (3100, 3900)]
        actual = evaluation.evaluate_gold(times, ["a", "b", "c"], edges, style, linking=True)
        self.assertEqual(actual["rows"][0]["suggested_ms"], (1000, 2000))
        self.assertEqual(actual["rows"][1]["suggested_ms"], (2000, 2900))
        self.assertEqual(actual["new_collisions"], 0)
        control = evaluation.evaluate_gold([(1000, 2000), (2100, 3100)], ["a", "b"],
                                           [(1100, 2200), (2200, 3000)], style, clamp=False)
        self.assertEqual(control["new_collisions"], 1)

    def test_gold_gate_rejects_precision_gained_by_abstaining_and_episode_regression(self):
        before = dict(coverage=.8, start_all_within_200ms=.6, end_all_within_200ms=.6,
                      median_abs_start_ms=120, median_abs_end_ms=130, new_collisions=0)
        after = dict(coverage=.9, start_all_within_200ms=.7, end_all_within_200ms=.7,
                     median_abs_start_ms=100, median_abs_end_ms=110, new_collisions=0)
        def rows(good):
            return [dict(group=str(i), suggested_ms=(1000, 2000), start_error_ms=0 if i < good else 300,
                         end_error_ms=0 if i < good else 300, new_collisions=0) for i in range(10)]
        before["rows"], after["rows"] = rows(6), rows(7)
        episodes = [dict(variants={"energy": {"ui_hints": dict(before)},
                                   "vad": {"ui_hints": dict(after)}}) for _ in range(6)]
        self.assertTrue(evaluation.gold_gate(episodes, "vad"))
        candidate = episodes[0]["variants"]["vad"]["ui_hints"]
        candidate["coverage"] = .7
        self.assertFalse(evaluation.gold_gate(episodes, "vad"))
        candidate["coverage"] = .9
        candidate["median_abs_end_ms"] = 140
        self.assertFalse(evaluation.gold_gate(episodes, "vad"))
        candidate["median_abs_end_ms"] = 110
        candidate["new_collisions"] = 1
        self.assertFalse(evaluation.gold_gate(episodes, "vad"))

    def test_vad_gate_requires_improvement_on_utena_and_both_pipelines(self):
        before = dict(coverage=.7, start_rate=.8, end_rate=.75, new_collisions=0)
        after = dict(coverage=.8, start_rate=.85, end_rate=.8, new_collisions=0)
        report = {"episodes": [dict(name=name, variants={
            variant: {pipeline: dict(metrics) for pipeline in ("speech_edges", "ui_hints")}
            for variant, metrics in (("baseline", before), ("silero", after))})
            for name in ("soma_e01", "soma_e02", "utena_e17")]}
        self.assertTrue(evaluation.eligible_vad(report, "silero"))
        candidate = report["episodes"][2]["variants"]["silero"]["ui_hints"]
        candidate["coverage"] = .7
        self.assertFalse(evaluation.eligible_vad(report, "silero"))
        candidate["coverage"] = .8
        candidate["end_rate"] = .74
        self.assertFalse(evaluation.eligible_vad(report, "silero"))
        candidate["end_rate"] = .8
        candidate["new_collisions"] = 1
        self.assertFalse(evaluation.eligible_vad(report, "silero"))
        self.assertFalse(evaluation.eligible_vad({"episodes": report["episodes"][:2]}, "silero"))

    def test_six_comparison_plots_contain_waveform_spectrum_and_missing_suggestion(self):
        import os
        import tempfile
        from PySide6.QtGui import QImage
        from subkit.audio import compute_features, RATE
        doc = Document(episode="plot-fixture")
        for i in range(6):
            gid, mid = f"g{i}", f"m{i}"
            doc.groups.append(Group(id=gid, kind="dialogue", members=[mid]))
            doc.members[mid] = Member(id=mid, group=gid, lang="ja", lane="original", role="dialogue",
                                      start=1000 + i * 2000, end=1700 + i * 2000)
        features = compute_features(np.zeros(RATE * 14, np.float32))
        evaluated = evaluation.evaluate(autotime._snap_activity(features), doc)
        with tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as scratch:
            plots = evaluation.save_comparisons(Path(scratch), "fixture", doc, features,
                                                evaluated, evaluated, np.empty((0, 2)))
            self.assertEqual(len({r["group"] for r in plots}), 6)
            for row in plots:
                image = QImage(row["path"])
                self.assertEqual((image.width(), image.height()), (1400, 750))

    def test_rates_use_suggested_groups_and_include_200ms(self):
        doc = Document(episode="fixture")
        for i, (start, end) in enumerate(((1000, 2000), (3000, 4000), (5000, 6000))):
            gid, mid = f"g{i}", f"m{i}"
            doc.groups.append(Group(id=gid, kind="dialogue", members=[mid]))
            doc.members[mid] = Member(id=mid, group=gid, lang="ja", lane="original",
                                      role="dialogue", start=start, end=end)
        calibration = autotime.HintCalibration(100, 0, "episode", 30, 30)
        with patch.object(autotime, "calibrate_hint_padding", return_value=calibration), \
                patch.object(autotime, "_speech_edges", side_effect=[(1300, 2200), (None, None), (5100, 6201)]):
            result = evaluation.evaluate(np.zeros(800), doc)["speech_edges"]
        self.assertEqual((result["groups"], result["suggestions"]), (3, 2))
        self.assertEqual(result["coverage"], 2 / 3)
        self.assertEqual((result["start_rate"], result["end_rate"]), (1, 0.5))
        self.assertEqual(result["new_collisions"], 0)

    def test_gate_requires_all_soma_metrics_and_no_utena_regression(self):
        def metrics(coverage, start, end, collisions=0):
            return dict(coverage=coverage, start_rate=start, end_rate=end, new_collisions=collisions)
        report = {"episodes": [dict(name=name, variants={
            "baseline": {"speech_edges": metrics(0.7, 0.8, 0.75)},
            "candidate": {"speech_edges": metrics(0.8, 0.85, 0.8)}})
            for name in ("soma_e01", "soma_e02", "utena_e17")]}
        self.assertTrue(evaluation.eligible(report, "candidate", "speech_edges"))
        report["episodes"][1]["variants"]["candidate"]["speech_edges"]["end_rate"] = 0.75
        self.assertFalse(evaluation.eligible(report, "candidate", "speech_edges"))
        report["episodes"][1]["variants"]["candidate"]["speech_edges"]["end_rate"] = 0.8
        utena = report["episodes"][2]["variants"]["candidate"]["speech_edges"]
        utena.update(metrics(0.7, 0.8, 0.75))
        self.assertTrue(evaluation.eligible(report, "candidate", "speech_edges"))
        utena["new_collisions"] = 1
        self.assertFalse(evaluation.eligible(report, "candidate", "speech_edges"))
        utena.update(metrics(0.7, 0.79, 0.75))
        self.assertFalse(evaluation.eligible(report, "candidate", "speech_edges"))


if __name__ == "__main__":
    unittest.main()
