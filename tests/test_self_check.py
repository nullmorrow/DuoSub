"""Scoped self-reference decisions, occurrence identity and document/file undo."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core.ops import Editor
from subkit.core.people import CharacterBook
from subkit.core.readings import anchor_of, ruby_parts, visible_map
from subkit.core.self_check import check
from subkit.model.document import Annotation, Document, Group, Member, Run


class SelfCheckTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="d1d-self-")
        self.addCleanup(tmp.cleanup)
        self.book = CharacterBook(Path(tmp.name) / "characters.json")
        self.ch = self.book.add_character({"ja": "太郎"}, self_refs=[
            {"surface": "私", "reading": "わたくし", "scope": {}},
            {"surface": "私", "reading": "わたし", "scope": {"line": "scene"}},
            {"surface": "僕", "reading": "ぼく", "scope": {}}])
        self.doc = Document("E1", groups=[Group("g")])
        self.editor = Editor(self.doc, {}, character_book=self.book)

    def member(self, mid, text="私と私", cid="default", lang="ja"):
        cid = self.ch.id if cid == "default" else cid
        m = Member(mid, "g", lang, "original", "dialogue", 1000, 2000,
                   speaker={"character": cid, "label": "太郎"},
                   runs=[Run(text, ins="i_" + mid, b=len(text))])
        self.doc.members[mid] = m
        self.doc.groups[0].members.append(mid)
        return m

    def annotate(self, mid, a, b, reading, status="auto", parts=None):
        vt, ids = visible_map(self.doc.members[mid])
        ann = Annotation(f"ann_{len(self.doc.annotations)}", "ruby", mid, anchor_of(ids[a:b]),
                         vt.text[a:b], reading, status=status,
                         parts=ruby_parts(vt.text[a:b], reading) if parts is None else parts)
        self.doc.annotations[ann.id] = ann
        return ann

    def check(self, episode="E1", season="S1"):
        return check(self.doc, self.book, episode, season)

    def test_scene_readings_occurrences_missing_and_other_self_surface(self):
        self.member("normal", "私と私と僕")
        self.member("scene", "私")
        self.annotate("normal", 0, 1, "わたし")
        self.annotate("normal", 2, 3, "わたくし")
        self.annotate("scene", 0, 1, "わたくし")
        rows = self.check()["mismatches"]
        self.assertEqual([(r["member"], r["at"], r["expected"]) for r in rows],
                         [("normal", [0, 1], "わたくし"), ("normal", [4, 5], "ぼく"),
                          ("scene", [0, 1], "わたし")])
        self.assertIsNone(rows[1]["current"])
        self.assertEqual((rows[0]["group"], rows[0]["speaker"], rows[0]["time"]), ("g", self.ch.id, 1000))

    def test_precedence_line_episodes_season_all_independent_of_list_order(self):
        self.book.update_character(self.ch.id, self_refs=[
            {"surface": "私", "reading": reading, "scope": scope} for reading, scope in
            [("あたし", {"line": "scene"}), ("わし", {"episodes": ["E1", "E2"]}),
             ("わたし", {"season": "S1"}), ("わたくし", {})]])
        self.member("scene", "私")
        self.member("other", "私")
        for episode, season, expected in [("E1", "S1", "わし"), ("E3", "S1", "わたし"),
                                          ("E3", "S2", "わたくし")]:
            rows = self.check(episode, season)["mismatches"]
            self.assertEqual([r["expected"] for r in rows], ["あたし", expected])

    def test_unknown_speakers_separate_crowd_not_person_other_language_skipped(self):
        self.member("unknown", "私と私", cid=None)
        self.member("missing", "私", cid="chr_deleted")
        self.member("crowd", "私", cid="crowd")
        self.member("en", "私", cid=None, lang="en")
        ignored = self.member("ignored", "私", cid=None)
        ignored.speaker["label"] = "効果音・回想"
        self.book.mark_label("効果音", crowd=False)
        result = self.check()
        self.assertEqual(result["mismatches"], [])
        self.assertEqual([r["member"] for r in result["unknown_speakers"]], ["unknown", "unknown", "missing"])

    def test_confirmed_hidden_equivalent_katakana_and_non_dialogue_not_flagged(self):
        self.member("m", "私私私私")
        self.annotate("m", 0, 1, "わたし", "confirmed")
        self.annotate("m", 1, 2, "", "hidden")
        self.annotate("m", 2, 3, "ワタクシ")
        self.annotate("m", 3, 4, "わたくし")
        self.member("sign", "私").role = "sign"
        self.assertEqual(self.check()["mismatches"], [])

    def test_visible_offsets_ass_tags_breaks_and_read_only_check(self):
        self.member("m", r'{\b1}私\N私')
        self.annotate("m", 0, 1, "わたし")
        before = copy.deepcopy(self.doc)
        self.assertEqual([r["at"] for r in self.check()["mismatches"]], [[0, 1], [1, 2]])
        self.assertEqual(self.doc, before)

    def test_fix_only_chosen_occurrence_confirmed_one_undo_redo(self):
        self.member("m")
        self.annotate("m", 0, 1, "わたし")
        other = self.annotate("m", 2, 3, "わたし")
        old = copy.deepcopy(self.doc.annotations)
        row = self.check()["mismatches"][0]
        aid = self.editor.fix_self_reference(row)
        ann = self.doc.annotations[aid]
        self.assertEqual((ann.text, ann.status, ann.origin), ("わたくし", "confirmed", "manual"))
        self.assertEqual(self.doc.annotations[other.id], other)
        self.assertEqual(len(self.editor.history.done), 1)
        self.assertEqual(len(self.check()["mismatches"]), 1)
        self.editor.history.undo()
        self.assertEqual(self.doc.annotations, old)
        self.editor.history.redo()
        self.assertEqual(self.doc.annotations[aid], ann)

    def test_keep_current_or_missing_confirmation_stops_relisting(self):
        for reading in ("わたし", None):
            with self.subTest(reading=reading):
                self.doc.annotations.clear()
                self.doc.members.clear()
                self.doc.groups[0].members.clear()
                self.member("m", "私")
                if reading:
                    self.annotate("m", 0, 1, reading)
                aid = self.editor.keep_self_reference(self.check()["mismatches"][0])
                self.assertEqual(self.doc.annotations[aid].text, reading or "")
                self.assertEqual(self.doc.annotations[aid].status, "confirmed")
                self.assertFalse(self.check()["mismatches"])

    def test_keep_episode_exception_document_memory_file_exact_undo_redo(self):
        self.member("m", "私")
        self.annotate("m", 0, 1, "わたし")
        raw, data = self.book.path.read_bytes(), self.book.to_json()
        old = copy.deepcopy(self.doc.annotations)
        aid = self.editor.keep_self_reference(self.check()["mismatches"][0], add_exception=True)
        saved = self.book.path.read_bytes()
        self.assertEqual(self.book.character(self.ch.id).self_refs[-1],
                         {"surface": "私", "reading": "わたし", "scope": {"episodes": ["E1"]}})
        self.assertEqual(len(self.editor.history.done), 1)
        self.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), raw)
        self.assertEqual(self.book.to_json(), data)
        self.assertEqual(self.doc.annotations, old)
        self.editor.history.redo()
        self.assertEqual(self.book.path.read_bytes(), saved)
        self.assertEqual(self.doc.annotations[aid].status, "confirmed")
        self.doc.annotations[aid].status = "auto"
        self.assertEqual(self.check()["mismatches"], [])

    def test_episode_exception_preserves_line_rule_and_write_failure_rolls_back(self):
        self.member("scene", "私")
        self.annotate("scene", 0, 1, "わたくし")
        row = self.check()["mismatches"][0]
        before = copy.deepcopy(self.doc.annotations)
        raw = self.book.path.read_bytes()
        with patch.object(self.book, "save", side_effect=OSError("synthetic write failure")):
            with self.assertRaises(OSError):
                self.editor.keep_self_reference(row, add_exception=True)
        self.assertEqual(self.doc.annotations, before)
        self.assertEqual(self.book.path.read_bytes(), raw)
        self.assertEqual(len(self.editor.history.done), 0)
        self.editor.keep_self_reference(row, add_exception=True)
        refs = self.book.character(self.ch.id).self_refs
        self.assertEqual(len([r for r in refs if r["scope"] == {"line": "scene"}]), 1)
        self.assertEqual(refs[-1]["scope"], {"episodes": ["E1"]})
        self.assertEqual(refs[-1]["reading"], "わたくし")

    def test_broader_annotation_parts_preserved_and_undone(self):
        self.member("m", "私僕")
        ann = self.annotate("m", 0, 2, "わたしぼく", parts=[[0, 1, "わたし"], [1, 2, "ぼく"]])
        row = self.check()["mismatches"][0]
        self.assertEqual(row["current"], "わたし")
        self.editor.fix_self_reference(row)
        self.assertEqual(sorted((a.base, a.text) for a in self.doc.annotations.values()),
                         [("僕", "ぼく"), ("私", "わたくし")])
        self.editor.history.undo()
        self.assertEqual(self.doc.annotations, {ann.id: ann})

    def test_stale_reading_surface_speaker_and_unsplittable_ruby_rejected(self):
        m = self.member("m", "私僕")
        ann = self.annotate("m", 0, 1, "わたし")
        row = self.check()["mismatches"][0]
        ann.text = "あたし"
        with self.assertRaises(ValueError):
            self.editor.fix_self_reference(row)
        ann.text = "わたし"
        m.speaker["character"] = None
        with self.assertRaises(ValueError):
            self.editor.fix_self_reference(row)
        m.speaker["character"] = self.ch.id
        m.runs[0].text = "僕私"
        with self.assertRaises(ValueError):
            self.editor.fix_self_reference(row)
        m.runs[0].text = "私僕"
        self.doc.annotations.clear()
        self.annotate("m", 0, 2, "わたしぼく", parts=[[0, 2, "わたしぼく"]])
        with self.assertRaises(ValueError):
            self.editor.fix_self_reference(self.check()["mismatches"][0])
        self.assertEqual(len(self.editor.history.done), 0)


if __name__ == "__main__":
    unittest.main()
