"""统一两种语言的时间 and batch import 「调轴」 (2026-10-15)."""
from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

from subkit.core.batch import BatchOptions
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from subkit.model.source import Source, SourceEvent
from tests import test_batch_core as batch_core           # module import: its tests are not collected here


def source(sid, lane, times):
    return Source(sid, "ja" if lane == "original" else "zh-Hant", lane, "srt", "x", "x", "h", 1, "utf-8", "lf",
                  events=[SourceEvent(i, "Dialogue", a, b, f"{lane}{i}") for i, (a, b) in enumerate(times)])


def line(doc, mid, gid, lane, src, e, start, end, *, kind="dialogue", role="dialogue"):
    group = next((g for g in doc.groups if g.id == gid), None)
    if group is None:
        group = Group(gid, kind, [])
        doc.groups.append(group)
    text = f"{lane}{e}"
    doc.members[mid] = Member(mid, gid, "ja" if lane == "original" else "zh-Hant", lane, role, start, end,
                              runs=[Run(text, src=src, e=e, a=0, b=len(text))])
    group.members.append(mid)
    return doc.members[mid]


class Unify(unittest.TestCase):
    def setUp(self):
        # The merge put the original lines on the translation's times; the original file runs 3 s later.
        self.jp = source("o1", "original", [(4000, 6500), (9000, 11000), (20000, 22000)])
        self.cn = source("t1", "translation", [(1000, 3500), (6000, 7000), (7000, 8000), (12000, 13000),
                                                (16000, 17000)])
        doc = self.doc = Document("E1")
        line(doc, "j0", "g0", "original", "o1", 0, 1000, 3500)
        line(doc, "c0", "g0", "translation", "t1", 0, 1000, 3500)
        line(doc, "j1", "g1", "original", "o1", 1, 6000, 8000)             # one original, two translation lines
        line(doc, "c1", "g1", "translation", "t1", 1, 6000, 7000)
        line(doc, "c2", "g1", "translation", "t1", 2, 7000, 8000)
        line(doc, "c3", "g2", "translation", "t1", 3, 12000, 13000)         # no partner
        line(doc, "j2", "g3", "original", "o1", 2, 17000, 19000)            # original only, on the translation clock
        line(doc, "c4", "g4", "translation", "t1", 4, 16000, 17000, kind="song", role="song")
        doc.structure_changed()
        self.ed = Editor(doc, {"o1": self.jp, "t1": self.cn})

    def times(self, *mids):
        return [(self.doc.members[m].start, self.doc.members[m].end) for m in mids]

    def test_original_times_with_translation_following(self):
        report = self.ed.unify_lane_times("original")
        self.assertEqual(self.times("j0", "c0"), [(4000, 6500)] * 2)
        self.assertEqual(self.times("j1", "c1", "c2"), [(9000, 11000), (9000, 10000), (10000, 11000)])  # order kept
        self.assertEqual(self.times("j2"), [(20000, 22000)])
        self.assertEqual(self.times("c3"), [(15000, 16000)])                 # the local 3 s difference
        self.assertEqual(self.times("c4"), [(16000, 17000)])                 # songs stay
        self.assertEqual((report["groups"], report["shifted"], report["protected"]), (3, 1, 0))
        self.assertEqual(self.doc.members["c0"].extra["time_src"], {"start": "autotime", "end": "autotime"})
        self.assertFalse(self.doc.members["c0"].extra.get("time_manual"))  # later auto-timing may still move it
        self.ed.history.undo()
        self.assertEqual(self.times("j0", "c3", "j2"), [(1000, 3500), (12000, 13000), (17000, 19000)])

    def test_translation_times_are_the_merge_result_and_protection(self):
        undo = len(self.ed.history.done)
        self.assertEqual(self.ed.unify_lane_times("translation")["lines"], 0)   # the merge already did this
        self.assertEqual(len(self.ed.history.done), undo)                        # no empty undo step
        self.doc.group("g0").state = "confirmed"
        self.doc.members["j1"].extra["time_manual"] = True
        report = self.ed.unify_lane_times("original")
        self.assertEqual(report["protected"], 2)
        self.assertEqual(self.times("j0", "c0", "j1"), [(1000, 3500), (1000, 3500), (6000, 8000)])
        self.assertEqual(self.times("c3"), [(15000, 16000)])                 # protected pairs still measure
        self.assertEqual(self.ed.unify_lane_times("original", include_protected=True)["protected"], 0)
        self.assertEqual(self.times("j0"), [(4000, 6500)])


class BatchTiming(unittest.TestCase):
    put = batch_core.BatchCoreTests.put
    run_fast = batch_core.BatchCoreTests.run_fast

    def setUp(self):
        import os
        import shutil
        import stat

        batch_core.BatchCoreTests.setUp(self)
        self.addCleanup(shutil.rmtree, self.root,
                        onerror=lambda f, p, _e: (os.chmod(p, stat.S_IWRITE | stat.S_IREAD), f(p)))

    def jp_later(self):
        self.put("S01E001.ass", batch_core.ass_text([(100, 300, "Chinese", "你好")]))
        self.put("S01E001.srt", "1\n00:00:01,200 --> 00:00:03,400\nこんにちは\n")
        self.put("S01E001.mkv")

    def dialogue_times(self, result):
        from subkit.core.workspace import Project
        doc = Project.open(result.project).episode("S01E001").document
        group = next(g for g in doc.groups if g.kind == "dialogue")
        return sorted({(m.start, m.end) for m in doc.members_of(group)})

    def test_choices_and_report(self):
        self.assertRaises(ValueError, BatchOptions, mode="5.2", timing="original")
        self.assertRaises(ValueError, BatchOptions, timing="video")
        self.jp_later()
        result = self.run_fast(BatchOptions(timing="original"))
        self.assertEqual(result.episodes[0].status, "完成", result.episodes[0].reason)
        self.assertEqual(self.dialogue_times(result), [(1200, 3400)])
        self.assertIn("调轴：按原声字幕的时间：1 组", result.report.read_text(encoding="utf-8"))

    def test_translation_is_the_old_default(self):
        self.jp_later()
        result = self.run_fast(BatchOptions())
        self.assertEqual(self.dialogue_times(result), [(1000, 3000)])
        self.assertIn("调轴：用翻译字幕的时间（没另外调）", result.report.read_text(encoding="utf-8"))

    def test_audio_reads_the_audio_once_for_timing_and_suggestions(self):
        self.jp_later()
        calls = []

        def provider(video, track, cache):
            calls.append(video)
            return "features"

        with patch("subkit.core.batch._timing_by_audio", return_value="按视频的声音：挪了 1 组、2 行") as timed, \
                patch("subkit.core.batch.adopt_suggestions", return_value=0):
            result = self.run_fast(BatchOptions(timing="audio", adopt_suggested_times=True), feature_provider=provider)
        self.assertEqual(result.episodes[0].status, "完成", result.episodes[0].reason)
        self.assertEqual(len(calls), 1)
        self.assertEqual(timed.call_args.args[1], "features")
        self.assertIn("调轴：按视频的声音：挪了 1 组、2 行", result.report.read_text(encoding="utf-8"))

    def test_audio_without_a_video_leaves_the_timing(self):
        self.jp_later()
        (self.season / "S01E001.mkv").unlink()
        result = self.run_fast(dataclasses.replace(BatchOptions(timing="audio"), require_video=False))
        self.assertEqual(result.episodes[0].status, "完成", result.episodes[0].reason)
        self.assertIn("调轴：按视频的声音：没有视频，没调", result.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
