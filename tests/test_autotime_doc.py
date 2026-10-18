"""Document auto-timing: selection, human protection and exact transactional undo."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit import autotime as at
from subkit.core.autotime_doc import Settings, estimation_intervals, plan
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from subkit.model.source import Source, SourceEvent
from tests.test_autotime import snap_fixture


def document(count=5):
    doc = Document("episode")
    for i in range(count):
        g = Group(f"g{i}")
        doc.groups.append(g)
        for lane, lang in (("original", "ja"), ("translation", "zh-Hant")):
            mid = f"{lane}{i}"
            g.members.append(mid)
            doc.members[mid] = Member(mid, g.id, lang, lane, "dialogue", 1000 + i * 30000,
                                      1800 + i * 30000, runs=[Run("word", ins=mid, b=4)])
    return doc


def mapping(offset=500, confidence="high"):
    return at.TimingMap([at.Segment(0, 200000, offset, confidence=confidence)])


def preview(doc, result=None, settings=Settings(), features=None, sources=None):
    with patch("subkit.autotime.estimate", return_value=result or mapping()):
        return plan(doc, sources or {}, features or snap_fixture(), settings)


class DocumentTimingTests(unittest.TestCase):
    def test_cut_inside_line_is_excluded_even_when_low_confidence_is_included(self):
        doc = document(2)
        result = at.TimingMap([at.Segment(0, 1400, 500, confidence="high"),
                               at.Segment(1400, 200000, -500, confidence="high")])
        for include in (False, True):
            p = plan(doc, {}, None, Settings(include_low_confidence=include), mapping_override=result)
            self.assertEqual(p.low_confidence_count, 1)
            self.assertEqual([g.group_id for g in p.changes], ["g1"])
            self.assertEqual(doc.members["original0"].start, 1000)

    def test_changes_below_one_frame_leave_groups_untouched(self):
        doc = document(3)
        self.assertFalse(preview(doc, mapping(39)).changes)
        self.assertEqual(len(preview(doc, mapping(40)).changes), 3)
        self.assertEqual(len(preview(doc, mapping(-60)).changes), 3)

    def test_intervals_exclude_roles_kinds_hidden_empty_and_positions(self):
        doc = document(1)
        base = doc.members["original0"]
        for i, role in enumerate(("dialogue", "top", "alt", "song", "sign", "info", "comment")):
            m = copy.deepcopy(base)
            m.id = f"role{i}"
            m.role = role
            doc.members[m.id] = m
            doc.groups[0].members.append(m.id)
        self.assertEqual(len(estimation_intervals(doc, {})), 5)
        for overrides in ({"pos": [1, 2]}, {"move": [1, 2, 3, 4]}, {"an": 8}, {"raw": r"\org(2,3)"}):
            base.overrides = overrides
            self.assertEqual(len(estimation_intervals(doc, {})), 4)
        base.overrides = {}
        base.source_kind = "Comment"
        self.assertEqual(len(estimation_intervals(doc, {})), 4)
        base.source_kind = "Dialogue"
        base.runs = []
        self.assertEqual(len(estimation_intervals(doc, {})), 4)
        doc.groups[0].kind = "song"
        self.assertEqual(estimation_intervals(doc, {}), [])

    def test_position_tags_in_whole_source_even_if_not_in_runs(self):
        doc = document(1)
        source = Source("o", "ja", "original", "ass", "", "", "", 0, "utf-8", "lf",
                        events=[SourceEvent(0, "Dialogue", 1000, 1800, r"{\pos(5,6)}word")])
        doc.members["original0"].runs = [Run("word", src="o", e=0, a=11, b=15)]
        self.assertEqual(estimation_intervals(doc, {"o": source}), [(1000, 1800)])

    def test_estimate_receives_only_speech_and_plan_is_pure(self):
        doc = document(2)
        doc.groups[1].kind = "sign"
        before = copy.deepcopy(doc)
        with patch("subkit.autotime.estimate", return_value=mapping()) as estimate:
            p = plan(doc, {}, snap_fixture())
        self.assertEqual(estimate.call_args.args[1], [(1000, 1800)] * 2)
        self.assertEqual(doc, before)
        self.assertEqual((p.group_count, p.member_count), (2, 4))
        self.assertEqual(p.hint.max_offset_ms, 500)
        self.assertTrue(p.hint.show)

    def test_protection_low_confidence_and_independent_switches(self):
        doc = document(3)
        doc.groups[0].state = "confirmed"
        doc.members["original1"].extra["time_manual"] = True
        p = preview(doc, mapping(confidence="low"))
        self.assertEqual((p.protected_count, p.low_confidence_count, p.group_count), (2, 1, 0))
        self.assertFalse(p.hint.show)
        p = preview(doc, mapping(confidence="low"), Settings(include_low_confidence=True))
        self.assertEqual(p.group_count, 1)
        p = preview(doc, mapping(confidence="low"), Settings(include_protected=True))
        self.assertEqual(p.low_confidence_count, 3)
        p = preview(doc, mapping(confidence="low"), Settings(include_protected=True, include_low_confidence=True))
        self.assertEqual(p.group_count, 3)

    def test_low_segment_at_end_does_not_exclude_previous_group(self):
        doc = document(1)
        result = at.TimingMap([at.Segment(0, 1800, 500, confidence="high"),
                               at.Segment(1800, 200000, 500, confidence="low")])
        p = preview(doc, result)
        self.assertEqual(p.low_confidence_count, 0)

    def test_reference_modes_and_neighbour_map_without_audio(self):
        for lane in ("original", "translation"):
            with self.subTest(lane=lane):
                doc = document(12)
                source = Source("s", "ja", lane, "srt", "", "", "", 0, "utf-8", "lf")
                for i in range(12):
                    if i == 5:
                        continue
                    m = doc.members[f"{lane}{i}"]
                    index = len(source.events)
                    source.events.append(SourceEvent(index, "Dialogue", round(m.start * 1.001 + 500),
                                                     round(m.end * 1.001 + 500), "word"))
                    m.runs = [Run("word", src="s", e=index, b=4)]
                before = copy.deepcopy(source)
                p = plan(doc, {"s": source}, None, Settings(reference=lane))
                self.assertEqual(p.group_count, 12)
                hole = next(g for g in p.changes if g.group_id == "g5")
                for change in hole.members:
                    self.assertAlmostEqual(change.start, round(change.before_start * 1.001 + 500), delta=1)
                self.assertEqual(source, before)

    def test_snap_only_speech_signs_songs_and_empty_follow_map(self):
        doc = document(3)
        doc.groups[1].kind = "song"
        doc.members["translation0"].role = "sign"
        doc.members["translation0"].start += 2000
        doc.members["translation0"].end += 2000
        doc.members["original2"].overrides["pos"] = [10, 20]
        doc.members["translation2"].runs = []
        with patch("subkit.autotime.snap_lines", side_effect=lambda f, lines, **kw: [(s + 100, e) for s, e in lines]) as snap:
            p = preview(doc, settings=Settings(snap="start"))
        self.assertEqual(snap.call_args_list[0].args[1], [(1500, 2300)])
        changed = {m.member_id: m for g in p.changes for m in g.members}
        self.assertEqual(changed["original0"].start, 1600)
        for mid in changed.keys() - {"original0"}:
            self.assertEqual(changed[mid].start, changed[mid].before_start + 500)

    def test_actual_snapping_uses_measured_defaults(self):
        doc = document(1)
        for m in doc.members.values():
            m.start, m.end = 1700, 3500
        p = preview(doc, mapping(0), Settings(snap="both"), snap_fixture())
        for c in p.changes[0].members:
            self.assertEqual((c.start, c.end), (1730, 3000))

    def test_snap_preserves_order_relative_to_skipped_members(self):
        doc = document(1)
        doc.members["translation0"].role = "sign"
        with patch("subkit.autotime.snap_lines", side_effect=lambda f, lines, **kw: [(s + 100, e) for s, e in lines]):
            p = preview(doc, settings=Settings(snap="start"))
        self.assertEqual([c.start for c in p.changes[0].members], [1500, 1500])

    def test_rev006_whole_document_neighbours(self):
        doc = document(2)
        for g, times in zip(doc.groups, ((1000, 2000), (2200, 3200))):
            for m in doc.members_of(g):
                m.start, m.end = times
        before = copy.deepcopy(doc)
        p = preview(doc, mapping(0), Settings(snap="both", lead_in_ms=200),
                    snap_fixture(((1400, 2300), (2400, 3500))))
        self.assertEqual([[(c.start, c.end) for c in g.members] for g in p.changes],
                         [[(1200, 2200)] * 2, [(2200, 3500)] * 2])
        self.assertEqual(doc, before)

    def test_snap_clamps_to_protected_current_and_song_planned_times(self):
        for protected in (False, True):
            for reverse in (False, True):
                doc = document(2)
                for g, times in zip(doc.groups, ((1000, 2000), (2200, 3200))):
                    for m in doc.members_of(g):
                        m.start, m.end = times
                doc.groups[1].kind = "song"
                if protected:
                    doc.groups[1].state = "confirmed"
                if reverse:
                    doc.groups.reverse()
                with patch.object(at, "snap_lines", side_effect=lambda f, lines, **kw: [(s, e + 500) for s, e in lines]):
                    p = preview(doc, mapping(100), Settings(snap="both"))
                changes = {g.group_id: g for g in p.changes}
                boundary = 2200 if protected else 2300
                self.assertEqual([c.end for c in changes["g0"].members], [boundary] * 2)
                if protected:
                    self.assertNotIn("g1", changes)

    def test_snap_preserves_mapped_overlap_and_clamps_early_start(self):
        for second_start, expected in ((1900, 1500), (2200, 2000)):
            doc = document(2)
            for g, times in zip(doc.groups, ((1000, 2000), (second_start, 3200))):
                for m in doc.members_of(g):
                    m.start, m.end = times
            with patch.object(at, "snap_lines", side_effect=lambda f, lines, **kw: [(1500, e) for s, e in lines] if lines[0][0] == second_start else lines):
                p = preview(doc, mapping(0), Settings(snap="start"))
            self.assertEqual([c.start for c in p.changes[0].members], [expected] * 2)

    def test_overlapping_short_group_does_not_hide_next_neighbour(self):
        doc = document(3)
        for g, times in zip(doc.groups, ((1000, 2000), (1500, 1800), (2200, 3200))):
            for m in doc.members_of(g):
                m.start, m.end = times
        with patch.object(at, "snap_lines", side_effect=lambda f, lines, **kw: [(s, e + 500) for s, e in lines]):
            p = preview(doc, mapping(0), Settings(snap="both"))
        changes = {g.group_id: g for g in p.changes}
        self.assertEqual([c.end for c in changes["g0"].members], [2200] * 2)
        self.assertEqual([c.end for c in changes["g1"].members], [2200] * 2)

    def test_plan_cancellation_before_compute_and_between_groups(self):
        from subkit.proc import CancelledError
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 4:
                raise CancelledError()
        doc = document(3)
        before = copy.deepcopy(doc)
        with self.assertRaises(CancelledError):
            plan(doc, {}, None, mapping_override=mapping(), cancel_check=check)
        self.assertEqual(doc, before)

    def test_apply_preserves_existing_manual_flag_when_included(self):
        doc = document(1)
        doc.members["original0"].extra = {"time_manual": True, "other": "kept"}
        p = preview(doc, settings=Settings(include_protected=True))
        editor = Editor(doc, {})
        editor.apply_autotime(p)
        self.assertEqual(doc.members["original0"].extra, {"time_manual": True, "other": "kept",
                                                          "time_src": {"start": "autotime", "end": "autotime"}})
        self.assertNotIn("time_manual", doc.members["translation0"].extra)

    def test_settings_missing_audio_and_silence(self):
        for values in ({"reference": "bad"}, {"method": "bad"}, {"snap": "bad"}, {"lead_in_ms": -1}):
            with self.assertRaises(ValueError):
                Settings(**values)
        with self.assertRaises(ValueError):
            plan(document(1), {}, None)
        f = snap_fixture()
        f.spectrogram[:] = 0
        p = plan(document(1), {}, f)
        self.assertEqual((p.group_count, p.low_confidence_count, p.hint.show), (0, 1, False))

    def test_apply_beats_selected_groups_and_undo_redo_exact(self):
        doc = document(2)
        doc.karaoke["k"] = {"id": "k", "member": "original0", "state": "confirmed", "beats": [
            {"core": [{"ins": "original0", "a": 0, "b": 4}], "t0": 1200, "t1": 1600, "by": "manual"},
            {"core": [{"ins": "original1", "a": 0, "b": 4}], "t0": 31200, "t1": 31600, "by": "manual"}]}
        result = at.TimingMap([at.Segment(0, 200000, 500, 1.001, "high")])
        p = preview(doc, result)
        editor = Editor(doc, {})
        before = copy.deepcopy(doc)
        self.assertEqual(editor.apply_autotime(p, ["g0"]), (1, 2))
        self.assertEqual(doc.karaoke["k"]["beats"][0]["t0"], 1701)
        self.assertEqual(doc.karaoke["k"]["beats"][0]["t1"], 2102)
        self.assertEqual(doc.karaoke["k"]["beats"][1], before.karaoke["k"]["beats"][1])
        self.assertEqual(doc.members["original1"], before.members["original1"])
        self.assertTrue(editor.history.done[0].keeps_text)
        self.assertEqual(editor.history.done[0].label, "自动调轴")
        self.assertTrue(all("time_manual" not in m.extra for m in doc.members.values()))
        after = copy.deepcopy(doc)
        editor.history.undo()
        for field in ("groups", "members", "comments", "annotations", "karaoke", "tombstones", "extra"):
            self.assertEqual(getattr(doc, field), getattr(before, field))
        editor.history.redo()
        self.assertEqual(doc.members, after.members)
        self.assertEqual(doc.karaoke, after.karaoke)

    def test_manual_time_operations_flag_in_same_undo_step(self):
        operations = (lambda e: e.set_member_time("original0", start=900),
                      lambda e: e.set_group_edge("g0", end=2000),
                      lambda e: e.set_group_time("g0", start=900, target="original"),
                      lambda e: e.shift_group("g0", 100))
        for operation in operations:
            doc = document(1)
            editor = Editor(doc, {})
            before = copy.deepcopy(doc.members)
            operation(editor)
            self.assertTrue(doc.members["original0"].extra["time_manual"])
            self.assertEqual(len(editor.history.done), 1)
            editor.history.undo()
            self.assertEqual(doc.members, before)
            editor.history.redo()
            self.assertTrue(doc.members["original0"].extra["time_manual"])

    def test_stale_plan_and_empty_choice_do_not_add_history(self):
        doc = document(1)
        editor = Editor(doc, {})
        p = preview(doc)
        self.assertEqual(editor.apply_autotime(p, []), (0, 0))
        self.assertFalse(editor.history.done)
        editor.set_member_time("original0", start=900)
        with self.assertRaises(ValueError):
            editor.apply_autotime(p)
        self.assertEqual(doc.members["original0"].start, 900)
        self.assertEqual(len(editor.history.done), 1)


class QualityTests(unittest.TestCase):
    def test_thresholds_scatter_and_low_windows(self):
        for mad, grade in ((0, "good"), (80, "good"), (81, "fair"), (200, "fair"), (201, "bad")):
            windows = [at.WindowLag(t, t, 15500 + sign * mad, 0.5, "high")
                       for t, sign in zip((1000, 2000, 3000, 4000), (-1, 1, -1, 1))]
            result = at.quality(at.TimingMap([at.Segment(0, 5000, 15500, confidence="high")], windows=windows))
            self.assertEqual(result.grade, grade)
            self.assertEqual(result.residual_mad_ms, mad)
            self.assertEqual(result.max_offset_ms, 15500)
        windows = [at.WindowLag(i, i, 0, 0.5, "high" if i < 2 else "low") for i in range(4)]
        result = at.quality(at.TimingMap([at.Segment(0, 10, 0)], windows=windows))
        self.assertEqual((result.grade, result.low_fraction), ("bad", 0.5))
        self.assertEqual(at.quality(mapping()).grade, "bad")

    def test_maximum_includes_left_limit_at_jump_and_rate_endpoint(self):
        result = at.quality(at.TimingMap([at.Segment(0, 1000, -100, 0.95), at.Segment(1000, 2000, 20)]))
        self.assertAlmostEqual(result.max_offset_ms, 150)


if __name__ == "__main__":
    unittest.main()
