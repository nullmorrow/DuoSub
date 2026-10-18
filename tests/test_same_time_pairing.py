"""Same-time bilingual import preserves both source sequences and provenance."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.core.merge import MergeSettings, merge_episode
from subkit.model.serialize import document_from_json, document_to_json
from subkit.model.source import Source, SourceEvent
from subkit.model.validate import validate


def source(lane, events):
    return Source(id=lane, lang="ja" if lane == "original" else "zh-Hans", lane=lane,
                  format="ass", path="synthetic.ass", copy="synthetic.ass", sha256="", bytes=0,
                  encoding="utf-8", newline="crlf",
                  events=[SourceEvent(i, "Dialogue", a, b, text, style="Default")
                          for i, (a, b, text) in enumerate(events)],
                  roles={i: "dialogue" for i in range(len(events))})


class SameTimePairingTests(unittest.TestCase):
    def merge(self, cn_events, jp_events, **settings):
        cn, ja = source("translation", cn_events), source("original", jp_events)
        before = copy.deepcopy((cn, ja))
        result = merge_episode("E1", cn, ja, MergeSettings(pairing="same_time", **settings))
        self.assertEqual((cn, ja), before)
        sources = {cn.id: cn, ja.id: ja}
        for doc in (result.document, document_from_json(document_to_json(result.document))):
            report = validate(doc, sources)
            self.assertTrue(report.ok, report.summary())
        self.assertEqual([i for b in result.alignment.blocks for i in b.rows], list(range(len(result.rows))))
        self.assertEqual([i for b in result.alignment.blocks for i in b.units], list(range(len(result.units))))
        self.assertEqual(result.alignment.offset, 0)
        self.assertEqual(result.alignment.reanchored, [])
        self.assertEqual(result.summary["cuts"], [])
        return result

    def test_exact_pair_keeps_all_phrases_of_source_event_together(self):
        result = self.merge([(1000, 3000, "第一句")], [(1000, 3000, "これは 最初。次です。")])
        self.assertGreater(len(result.units), 1)
        self.assertEqual([(b.kind, b.rows, b.units) for b in result.alignment.blocks],
                         [("pair", [0], list(range(len(result.units))))])
        jp = next(m for m in result.document.members.values() if m.lane == "original")
        self.assertEqual((jp.start, jp.end), (1000, 3000))
        self.assertEqual(result.log, ["按相同时间配对：配对 1 组，只有中文 0 行，只有日文 0 行"])

    def test_tolerance_checks_both_edges_inclusively(self):
        for start, end, paired in ((1010, 1990, True), (990, 2010, True), (1011, 2000, False),
                                   (1000, 2011, False), (989, 2000, False), (1000, 1989, False)):
            with self.subTest(start=start, end=end):
                result = self.merge([(1000, 2000, "中文")], [(start, end, "にほんご")])
                self.assertEqual(result.summary["pairs"], int(paired))
                self.assertEqual(result.summary["original_only"], int(not paired))
                self.assertEqual(result.summary["translation_only"], int(not paired))

    def test_simultaneous_lines_pair_in_source_order(self):
        result = self.merge([(1000, 3000, "中文甲"), (1000, 3000, "中文乙")],
                            [(1000, 3000, "にほんご甲"), (1000, 3000, "にほんご乙")])
        pairs = [(result.rows[b.rows[0]].e, {result.units[i].e for i in b.units})
                 for b in result.alignment.blocks]
        self.assertEqual(pairs, [(0, {0}), (1, {1})])
        self.assertEqual(result.summary["pairs"], 2)

    def test_one_sided_lines_stay_separate_at_original_times(self):
        result = self.merge([(1000, 2000, "中文甲"), (3000, 4000, "中文乙"), (5000, 6000, "中文丙")],
                            [(1000, 2000, "にほんご甲"), (2200, 2700, "にほんご乙"), (5000, 6000, "にほんご丙")])
        self.assertEqual(result.summary["pairs"], 2)
        self.assertEqual((result.summary["translation_only"], result.summary["original_only"]), (1, 1))
        jp = next(m for m in result.document.members.values() if m.lane == "original" and "乙" in m.text)
        self.assertEqual((jp.start, jp.end), (2200, 2700))

    def test_crossing_pair_is_left_unpaired_preserving_both_lanes(self):
        result = self.merge([(1000, 2000, "中文甲"), (3000, 4000, "中文乙"), (5000, 6000, "中文丙")],
                            [(3000, 4000, "にほんご乙"), (1000, 2000, "にほんご甲"), (5000, 6000, "にほんご丙")])
        self.assertEqual([b.kind for b in result.alignment.blocks], ["unpaired", "pair", "empty_row", "pair"])
        for lane in ("translation", "original"):
            positions = [r.e for g in result.document.groups for m in result.document.members_of(g)
                         if m.lane == lane for r in m.runs if r.src == lane]
            self.assertEqual(positions, [0, 1, 2])
        self.assertEqual(result.summary["pairs"], 2)

    def test_no_alignment_models_warp_or_reanchoring(self):
        with patch("subkit.core.merge.time_warp", side_effect=AssertionError("warp")), \
             patch("subkit.core.merge.Aligner2", side_effect=AssertionError("aligner")), \
             patch("subkit.core.merge.realign_detached", side_effect=AssertionError("realign")), \
             patch("subkit.core.merge.Weights2", side_effect=AssertionError("weights")):
            result = self.merge([(1000, 2000, "中文")], [(1000, 2000, "にほんご")], semantic=True)
        self.assertEqual(result.summary["pairs"], 1)

    def test_unpaired_fillers_remain_visible_even_in_compact_mode(self):
        result = self.merge([(1000, 2000, "中文")], [(3000, 4000, "あっ")], filler_mode="compact")
        self.assertEqual(len(result.document.groups), 2)
        self.assertFalse(result.document.comments)
        self.assertFalse(any(b.hidden for b in result.alignment.blocks))

    def test_short_unpaired_event_keeps_source_end(self):
        result = self.merge([], [(1000, 1050, "にほんご")])
        jp = next(iter(result.document.members.values()))
        self.assertEqual((jp.start, jp.end), (1000, 1050))

    def test_empty_or_one_sided_sources(self):
        for cn, ja in (([], []), ([(1000, 2000, "中文")], []), ([], [(1000, 2000, "にほんご")])):
            with self.subTest(cn=bool(cn), ja=bool(ja)):
                result = self.merge(cn, ja)
                self.assertEqual(result.summary["pairs"], 0)
                self.assertEqual(len(result.document.groups), len(cn) + len(ja))

    def test_default_settings_still_use_aligner(self):
        cn = source("translation", [(1000, 2000, "中文")])
        ja = source("original", [(1000, 2000, "にほんご")])
        with patch("subkit.core.merge.Aligner2", side_effect=RuntimeError("default alignment")):
            with self.assertRaisesRegex(RuntimeError, "default alignment"):
                merge_episode("E1", cn, ja, MergeSettings())


if __name__ == "__main__":
    unittest.main()
