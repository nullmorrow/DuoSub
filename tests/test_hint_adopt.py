"""Hint adoption stability and per-edge timing provenance."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit import autotime as at
from subkit.core.autotime_doc import Settings, plan
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json


def document(start=3600, end=5000):
    doc = Document("hint")
    doc.groups.append(Group("g", "dialogue", ["m"]))
    doc.members["m"] = Member("m", "g", "ja", "original", "dialogue", start, end,
                              runs=[Run("line", ins="i", b=4)])
    return doc


class HintAdoptTests(unittest.TestCase):
    def stable(self, curve, current, **kwargs):
        options = dict(lead_in_ms=270, lead_out_ms=160)
        options.update(kwargs)
        suggested = at.suggest_group_times(curve, *current, **options)
        self.assertIsNotNone(suggested)
        doc = document(*current)
        editor = Editor(doc, {})
        editor.set_group_time("g", start=suggested[0], end=suggested[1], source="adopted")
        adopted = doc.group_time(doc.groups[0])
        again = at.suggest_group_times(curve, *adopted, **options)
        if again is not None:
            self.assertTrue(all(s is None or s == t for s, t in zip(again, adopted)),
                            (current, suggested, again))
        return suggested

    def test_breath_before_line(self):
        curve = np.zeros(800, np.float32)
        curve[310:325], curve[355:500] = .5, .8
        self.assertEqual(self.stable(curve, (3600, 5000)), (3280, 5160))

    def test_trailing_noise(self):
        curve = np.zeros(800, np.float32)
        curve[355:500], curve[525:540] = .8, .5
        self.assertEqual(self.stable(curve, (3600, 5100)), (3280, 5160))

    def test_continuous_valley_large_lead_in(self):
        curve = np.zeros(800, np.float32)
        curve[270:500], curve[348] = .5, .2
        suggested = self.stable(curve, (3200, 5000), lead_in_ms=400, previous=(2000, 3000))
        self.assertEqual(suggested, (3080, 5160))

    def test_valley_that_would_expose_a_lower_dip_abstains(self):
        curve = np.zeros(800, np.float32)
        curve[270:520], curve[370], curve[345] = .5, .2, .1
        self.assertEqual(at.hint_speech_edges(curve, 3500, 5500, lead_in_ms=400,
                                            lead_out_ms=160, previous=(2000, 3000))[0], 3700)
        self.assertIsNone(at.suggest_group_times(curve, 3500, 5500, lead_in_ms=400,
                                                lead_out_ms=160, previous=(2000, 3000)))

    def test_one_sided_suggestion(self):
        curve = np.zeros(800, np.float32)
        curve[355:700] = .8
        self.assertEqual(self.stable(curve, (3600, 4000)), (3280, None))

    def test_neighbour_clamp(self):
        curve = np.zeros(800, np.float32)
        curve[355:500] = .8
        self.assertEqual(self.stable(curve, (3600, 5000), previous=(1000, 3400),
                                     following=(5100, 6500)), (3400, 5100))

    def test_search_window_truncation(self):
        curve = np.zeros(800, np.float32)
        curve[:500] = .8
        self.assertEqual(self.stable(curve, (2000, 5500)), (None, 5160))

    def test_gold_time_preservation_is_separate_from_idempotence(self):
        curve = np.zeros(800, np.float32)
        curve[355:500] = .8
        gold = (3280, 5160)
        self.assertEqual(self.stable(curve, gold), gold)

    def test_sources_only_changed_edges_undo_redo_and_serialization(self):
        doc = document()
        editor = Editor(doc, {})
        before = copy.deepcopy(doc.members["m"])
        editor.set_group_time("g", start=3280, end=5000, source="adopted")
        self.assertEqual(doc.members["m"].extra, {"time_manual": True, "time_src": {"start": "adopted"}})
        editor.history.undo()
        self.assertEqual(doc.members["m"], before)
        editor.history.redo()
        editor.set_member_time("m", end=5160)
        self.assertEqual(doc.members["m"].extra["time_src"], {"start": "adopted", "end": "manual"})
        loaded = document_from_json(document_to_json(doc))
        self.assertEqual(loaded.members["m"].extra, doc.members["m"].extra)
        editor.set_group_time("g", start=3280, end=5160)
        self.assertEqual(doc.members["m"].extra["time_src"], {"start": "adopted", "end": "manual"})

    def test_manual_drag_edge_shift_and_crossing_record_actual_changes(self):
        for operation, expected in (
                (lambda e: e.set_group_edge("g", start=3500), {"start": "manual"}),
                (lambda e: e.shift_group("g", 100), {"start": "manual", "end": "manual"}),
                (lambda e: e.set_group_time("g", start=6000), {"start": "manual", "end": "manual"})):
            with self.subTest(expected=expected):
                doc = document()
                operation(Editor(doc, {}))
                self.assertEqual(doc.members["m"].extra["time_src"], expected)

    def test_autotime_records_changed_edges_and_restores_sources(self):
        doc = document()
        doc.members["m"].extra = {"time_manual": True, "time_src": {"start": "adopted"}}
        mapping = at.TimingMap([at.Segment(0, 10000, 500, confidence="high")])
        preview = plan(doc, {}, None, Settings(include_protected=True), mapping_override=mapping)
        before = copy.deepcopy(doc.members["m"])
        editor = Editor(doc, {})
        editor.apply_autotime(preview)
        self.assertEqual(doc.members["m"].extra["time_src"], {"start": "autotime", "end": "autotime"})
        self.assertTrue(doc.members["m"].extra["time_manual"])
        editor.history.undo()
        self.assertEqual(doc.members["m"], before)
        editor.history.redo()
        self.assertEqual(doc.members["m"].extra["time_src"]["end"], "autotime")

    def test_manual_calibration_legacy_confirmed_and_per_edge_sources(self):
        curve = np.zeros(800, np.float32)
        curve[355:500] = .8
        cases = [({}, "draft", (0, 0)),
                 ({"time_manual": True}, "draft", (1, 1)),
                 ({"time_manual": True, "time_src": {"start": "manual"}}, "draft", (1, 0)),
                 ({"time_manual": True, "time_src": {"start": "adopted"}}, "confirmed", (0, 1)),
                 ({"time_src": {"start": "autotime", "end": "adopted"}}, "confirmed", (0, 0)),
                 ({}, "confirmed", (1, 1))]
        for extra, state, counts in cases:
            with self.subTest(extra=extra, state=state):
                doc = document(3280, 5160)
                doc.members["m"].extra = extra
                doc.groups[0].state = state
                result = at.calibrate_manual_hint_padding([(curve, doc)], (100, 160))
                self.assertEqual((result.onset_samples, result.offset_samples), counts)

    def test_twenty_manual_starts_do_not_learn_unedited_ends(self):
        curve = np.zeros(800, np.float32)
        curve[355:500] = .8
        doc = document(3280, 5160)
        doc.members["m"].extra = {"time_manual": True, "time_src": {"start": "manual"}}
        result = at.calibrate_manual_hint_padding([(curve, doc)] * 20, (100, 160))
        self.assertEqual((result.onset_samples, result.offset_samples), (20, 0))
        self.assertEqual((result.lead_in_ms, result.lead_out_ms, result.source), (100, 160, "habit"))
        doc.members["m"].extra["time_src"]["end"] = "manual"
        result = at.calibrate_manual_hint_padding([(curve, doc)] * 20, (100, 160))
        self.assertEqual((result.lead_in_ms, result.lead_out_ms, result.source), (270, 160, "manual"))
        doc.members["m"].extra["time_src"] = {"start": "adopted", "end": "autotime"}
        doc.groups[0].state = "confirmed"
        result = at.calibrate_manual_hint_padding([(curve, doc)] * 20, (100, 160))
        self.assertEqual((result.onset_samples, result.offset_samples), (0, 0))

    def test_legacy_manual_edge_survives_one_sided_adoption(self):
        doc = document()
        doc.members["m"].extra["time_manual"] = True
        Editor(doc, {}).set_group_time("g", start=3280, source="adopted")
        self.assertEqual(doc.members["m"].extra["time_src"], {"start": "adopted", "end": "manual"})

    def test_batch_group_edges_restore_all_members_sources_and_one_step(self):
        doc = document()
        doc.groups.append(Group("g2", "dialogue", ["m2", "comment"]))
        doc.members["m2"] = Member("m2", "g2", "zh-Hans", "translation", "dialogue", 6000, 7000,
                                    extra={"time_manual": True, "retained": [1, 2]})
        doc.members["comment"] = Member("comment", "g2", "ja", "original", "comment", 6000, 7000)
        editor = Editor(doc, {})
        before_members, before_groups = copy.deepcopy(doc.members), copy.deepcopy(doc.groups)
        editor.set_group_times([("g", 3280, None), ("g2", None, 7400)], label="精确对话轴（整集）")
        self.assertEqual(len(editor.history.done), 1)
        self.assertEqual(doc.members["m"].extra["time_src"], {"start": "adopted"})
        self.assertEqual(doc.members["m2"].extra["time_src"], {"start": "manual", "end": "adopted"})
        self.assertEqual(doc.members["comment"], before_members["comment"])
        after_members = copy.deepcopy(doc.members)
        editor.history.undo()
        self.assertEqual(doc.members, before_members)
        self.assertEqual(doc.groups, before_groups)
        editor.history.redo()
        self.assertEqual(doc.members, after_members)
        editor.set_group_times([], label="empty")
        self.assertEqual(len(editor.history.done), 1)

    def test_batch_adoption_creates_no_new_overlap(self):
        def three():
            doc = document(1000, 2000)
            for gid, a, b in (("g2", 2500, 3500), ("g3", 4000, 5000)):
                doc.groups.append(Group(gid, "dialogue", [gid + "m"]))
                doc.members[gid + "m"] = Member(gid + "m", gid, "ja", "original", "dialogue", a, b)
            return doc
        doc = three()
        # the earlier end gives way (40 ms gap) to an adopted later start; a lone end gives way to the old start
        Editor(doc, {}).set_group_times([("g", None, 2800), ("g2", 2600, None), ("g3", None, 5200)],
                                        label="t", avoid_new_overlap=True)
        self.assertEqual([doc.group_time(g) for g in doc.groups], [(1000, 2560), (2600, 3500), (4000, 5200)])
        doc = three()
        Editor(doc, {}).set_group_times([("g2", None, 4300)], label="t", avoid_new_overlap=True)
        self.assertEqual(doc.group_time(doc.groups[1]), (2500, 3960))
        doc = three()
        Editor(doc, {}).set_group_times([("g3", 3000, None)], label="t", avoid_new_overlap=True)
        self.assertEqual(doc.group_time(doc.groups[2]), (3540, 5000))
        doc = three()                                       # an existing overlap is left alone
        doc.members["g2m"].start = 1900
        Editor(doc, {}).set_group_times([("g", None, 2300)], label="t", avoid_new_overlap=True)
        self.assertEqual(doc.group_time(doc.groups[0]), (1000, 2300))
        doc = three()                                       # without the option: unchanged behaviour
        Editor(doc, {}).set_group_times([("g", None, 2800)], label="t")
        self.assertEqual(doc.group_time(doc.groups[0]), (1000, 2800))

    def test_batch_reuses_crossing_semantics_and_rolls_back_failure(self):
        doc = document()
        editor = Editor(doc, {})
        editor.set_group_times([("g", 7000, None)], label="crossing", source="manual")
        self.assertEqual(doc.group_time(doc.groups[0]), (7000, 8400))
        self.assertEqual(doc.members["m"].extra["time_src"], {"start": "manual", "end": "manual"})
        before = copy.deepcopy(doc.members)
        with self.assertRaises((KeyError, AttributeError)):
            editor.set_group_times([("g", 6000, None), ("missing", 1, 2)], label="invalid")
        self.assertEqual(doc.members, before)
        self.assertEqual(len(editor.history.done), 1)
        with self.assertRaises(ValueError):
            editor.set_group_times([("g", 1, 2)], label="source", source="unknown")
        self.assertEqual(doc.members, before)
