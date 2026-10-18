"""W4: clamp working times, preserving sources, undo and untimed export."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtWidgets import QApplication
from subkit.autotime import Segment, TimingMap
from subkit.ass import AssDocument
from subkit.core.autotime_doc import GroupTimeChange, MemberTimeChange
from subkit.core.export import ExportOptions, build_ass
from subkit.core.karaoke import Beat, beats_to_track, move_edge
from subkit.core.merge import MergeSettings, build_document
from subkit.core.ops import Editor
from subkit.core.units import build_units
from subkit.core.workspace import Episode, Project
from subkit.model.document import Document, Group, Member, Run
from subkit.model.source import Source, SourceEvent
from subkit.model.serialize import read_json, save_document
from subkit.styles_default import default_styles_ass
from subkit.align.dp2 import AlignOutput, Block
from subkit_app.session import EpisodeSession


def fixture():
    m = Member("m", "g", "ja", "original", "song", 1250, 2200,
               runs=[Run("風", ins="i", b=1)])
    doc = Document("E", groups=[Group("g", "song", ["m"])], members={"m": m})
    doc.karaoke["k"] = {"id": "k", "member": "m", "beats": [
        {"core": [{"ins": "i", "a": 0, "b": 1}], "t0": 1250, "t1": 2200}]}
    return doc, Editor(doc, {})


class NegativeTimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_literal_braces_are_kept_and_reported(self):
        ja = Source("o1", "ja", "original", "srt", "", "", "", 0, "utf-8", "crlf",
                    events=[SourceEvent(0, "Dialogue", 1000, 2000, "xz")], roles={0: "dialogue"})
        cn = Source("t1", "zh-Hans", "translation", "srt", "", "", "", 0, "utf-8", "crlf")
        units = build_units(ja)
        out = AlignOutput(blocks=[Block("unpaired", [], list(range(len(units))))])
        doc = build_document("E", cn, ja, units, [], out, MergeSettings())
        mid = next(iter(doc.members))
        Editor(doc, {}).replace_text(mid, "x{y}z")                      # braces typed in the editor stay text
        with tempfile.TemporaryDirectory() as folder:
            ass, report = build_ass(doc, {"o1": ja, "t1": cn}, AssDocument.parse(default_styles_ass()),
                                    Path(folder), ExportOptions(ruby=False))
        self.assertTrue(any("x{y}z" in e.text for e in ass.events))          # exported as typed
        warning = next(w for w in report.warnings if "大括号" in w)
        self.assertIn("「{」「}」", warning)
        self.assertIn("\\{ \\}", warning)
        self.assertIn("x{y}z", warning)

    def test_negative_alignment_and_zero_comment_export_keep_sources(self):
        ja = Source("o1", "ja", "original", "srt", "", "", "", 0, "utf-8", "crlf",
                    events=[SourceEvent(0, "Dialogue", 1250, 2200, "GA文庫")], roles={0: "dialogue"})
        cn = Source("t1", "zh-Hans", "translation", "srt", "", "", "", 0, "utf-8", "crlf")
        before = copy.deepcopy(ja.events_json())
        units = build_units(ja)
        for unit in units:
            unit.wstart, unit.wend = unit.start - 2900, unit.end - 2900
        out = AlignOutput(blocks=[Block("unpaired", [], list(range(len(units))))])
        doc = build_document("E", cn, ja, units, [], out, MergeSettings())
        self.assertEqual([(m.start, m.end) for m in doc.members.values()], [(0, 0)])
        with tempfile.TemporaryDirectory() as folder:
            ass, report = build_ass(doc, {"o1": ja, "t1": cn}, AssDocument.parse(default_styles_ass()),
                                    Path(folder), ExportOptions(ruby=False))
        line = next(e for e in ass.events if "GA文庫" in e.text)
        self.assertEqual((line.kind, line.start, line.end), ("Comment", 0, 0))
        self.assertTrue(report.not_timed)
        self.assertEqual(ja.events_json(), before)

    def test_single_and_multi_shift_clamp_beats_and_undo(self):
        for method in ("shift_group", "shift_groups"):
            doc, ed = fixture()
            before = copy.deepcopy(doc)
            getattr(ed, method)("g" if method == "shift_group" else ["g", "g"], -2900)
            self.assertEqual((doc.members["m"].start, doc.members["m"].end), (0, 0))
            self.assertEqual([(b["t0"], b["t1"]) for b in doc.karaoke["k"]["beats"]], [(0, 0)])
            self.assertEqual(len(ed.history.done), 1)
            ed.history.undo()
            self.assertEqual(doc.members, before.members)
            self.assertEqual(doc.karaoke, before.karaoke)
            ed.history.redo()
            self.assertEqual(doc.members["m"].end, 0)

    def test_partial_shift_and_manual_edges(self):
        doc, ed = fixture()
        ed.shift_group("g", -1500)
        self.assertEqual((doc.members["m"].start, doc.members["m"].end), (0, 700))
        for method, args in (("set_member_time", {"start": -200, "end": -50}),
                             ("set_group_time", {"start": -200, "end": -50}),
                             ("set_group_edge", {"start": -200})):
            with self.subTest(method=method):
                doc, ed = fixture()
                getattr(ed, method)("m" if method == "set_member_time" else "g", **args)
                self.assertGreaterEqual(doc.members["m"].start, 0)
                self.assertGreaterEqual(doc.members["m"].end, 0)

    def test_autotime_apply_clamps_member_and_karaoke(self):
        doc, ed = fixture()
        mapping = TimingMap([Segment(0, 10000, -2900, confidence="high")])
        plan = SimpleNamespace(episode="E", revision=0, mapping=mapping,
            changes=[GroupTimeChange("g", (MemberTimeChange("m", 1250, 2200, -1650, -700),))])
        self.assertEqual(ed.apply_autotime(plan), (1, 1))
        self.assertEqual((doc.members["m"].start, doc.members["m"].end), (0, 0))
        self.assertEqual([(b["t0"], b["t1"]) for b in doc.karaoke["k"]["beats"]], [(0, 0)])

    def test_old_document_dirty_save_and_no_undo(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Project.create(Path(folder) / "project", "W4")
            ep = project.add_episode("E")
            doc, _ed = fixture()
            doc.members["m"].start, doc.members["m"].end = -1650, -700
            doc.karaoke["k"]["beats"][0].update(t0=-1650, t1=-700)
            save_document(doc, ep.dir / "document.json")
            raw = (ep.dir / "document.json").read_bytes()
            opened = Episode.open(project, ep.dir)
            self.assertEqual((ep.dir / "document.json").read_bytes(), raw)
            self.assertEqual(opened.document.revision, 1)
            with patch.object(project, "template", return_value=AssDocument.parse(default_styles_ass())):
                session = EpisodeSession(project, opened)
            try:
                self.assertTrue(session.dirty)
                self.assertFalse(session.editor.history.done)
                session.save()
                self.assertFalse(session.dirty)
                self.assertFalse(opened.times_changed)
                stored = read_json(ep.dir / "document.json")
                self.assertEqual((stored["members"]["m"]["start"], stored["members"]["m"]["end"]), (0, 0))
                self.assertEqual(stored["karaoke"]["k"]["beats"][0]["t0"], 0)
                self.assertFalse(Episode.open(project, ep.dir).times_changed)
            finally:
                session.close()

    def test_transaction_clamps_other_time_writers_and_karaoke_drag(self):
        doc, ed = fixture()
        def write(tx):
            tx.member("m").start = -500
            tx.track("k")["beats"][0]["t0"] = -500
        ed.history.run("time writer", write)
        self.assertEqual(doc.members["m"].start, 0)
        self.assertEqual(doc.karaoke["k"]["beats"][0]["t0"], 0)
        beat = Beat(a=0, b=1, slot=0, core=[("i", "i", 0)], k=0, sung="か", t0=100, t1=200, by="manual")
        for shift in (False, True):
            changed = move_edge([beat], 0, "start", -200, shift=shift)
            self.assertEqual(changed[0].t0, 0)
        beat.t0, beat.t1 = -100, -20
        stored = beats_to_track(doc.members["m"], [beat])["beats"][0]
        self.assertEqual((stored["t0"], stored["t1"]), (0, 0))
