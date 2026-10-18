"""Project word boundaries, protected decisions, one-step undo and the Qt input path."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit.core.ops import Editor
from subkit.core.readings import anchor_of, anchor_positions, reading_spans, visible_map
from subkit.core.workspace import Project
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.reading.dictionary import ReadingDict, normalize_word_split
from subkit.reading.engine import Analyzer, ReadingEngine


SURFACE = "世界征服"
PARTS = [{"surface": "世界", "reading": "せかい"}, {"surface": "征服", "reading": "せいふく"}]
ALTERNATE = [{"surface": "世", "reading": "せ"}, {"surface": "界征服", "reading": "かいせいふく"}]


class DictionarySplits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "readings.json"

    def engine(self, data):
        rd = ReadingDict()
        rd.merge_json(data, "project")
        return ReadingEngine(rd)

    def test_real_sudachi_word_proposal_and_separate_entries(self):
        self.assertEqual(Analyzer.get().split_a(SURFACE), [("世界", "せかい"), ("征服", "せいふく")])
        entries = self.engine({"word_splits": {SURFACE: PARTS}}).analyze(SURFACE)
        self.assertEqual([(e.start, e.end, e.surface, e.reading) for e in entries],
                         [(0, 2, "世界", "せかい"), (2, 4, "征服", "せいふく")])
        self.assertTrue(all(e.locked and e.rule_surface == SURFACE for e in entries))

    def test_legacy_groups_and_splits_remain_one_entry(self):
        for data in ({"words": {"世界 征服": "せかい せいふく"}}, {"splits": {SURFACE: PARTS}}):
            entries = self.engine(data).analyze(SURFACE)
            self.assertEqual(len(entries), 1)
            self.assertEqual(len(entries[0].parts), 2)
            self.assertIsNone(entries[0].rule_surface)

    def test_load_does_not_write_or_migrate(self):
        original = b'{"version": 2, "unknown": [true, {"x": 7}]}\r\n'
        self.path.write_bytes(original)
        self.assertEqual(ReadingDict.load(self.path).word_splits, {})
        self.assertEqual(self.path.read_bytes(), original)
        missing = self.path.with_name("missing.json")
        self.assertEqual(ReadingDict.load(missing).word_splits, {})
        self.assertFalse(missing.exists())

    def test_write_preserves_unknown_history_other_words_bom_and_newlines(self):
        data = {"version": 2, "future": {"a": [1, True]}, "typed": {SURFACE: ["たいぷ"]},
                "words": {SURFACE: "old", "世界 征服": "old", "世界": "せかい", "無関係": "むかんけい"},
                "readings": {SURFACE: "old"}, "splits": {SURFACE: PARTS}}
        self.path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8-sig"))
        ReadingDict().set_word_split(self.path, SURFACE, PARTS)
        raw = self.path.read_bytes()
        saved = json.loads(raw.decode("utf-8-sig"))
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))
        self.assertEqual(saved["future"], data["future"])
        self.assertEqual(saved["typed"], data["typed"])
        self.assertEqual(saved["words"], {"世界": "せかい", "無関係": "むかんけい"})
        self.assertEqual(saved["readings"], {})
        self.assertEqual(saved["splits"], {})
        self.assertEqual(saved["word_splits"], {SURFACE: PARTS})
        self.assertEqual(len(ReadingEngine(ReadingDict.load(self.path)).analyze(SURFACE)), 2)

    def test_update_delete_and_later_merge_are_exclusive(self):
        rd = ReadingDict()
        rd.set_word_split(self.path, SURFACE, PARTS)
        rd.set_word_split(self.path, SURFACE, ALTERNATE)
        self.assertEqual([e.surface for e in ReadingEngine(rd).analyze(SURFACE)], ["世", "界征服"])
        self.assertTrue(rd.remove_word_split(self.path, SURFACE))
        self.assertFalse(rd.remove_word_split(self.path, SURFACE))
        rd.set_word_split(self.path, SURFACE, PARTS)
        rd.add_word(self.path, SURFACE, "せかいせいふく")
        self.assertNotIn(SURFACE, rd.word_splits)
        self.assertNotIn(SURFACE, ReadingDict.load(self.path).word_splits)
        self.assertEqual(len(ReadingEngine(ReadingDict.load(self.path)).analyze(SURFACE)), 1)

    def test_later_dictionary_and_episode_words_override_split(self):
        rd = ReadingDict()
        rd.merge_json({"word_splits": {SURFACE: PARTS}}, "project")
        rd.merge_json({"words": {SURFACE: "とくべつ"}}, "episode")
        entries = ReadingEngine(rd).analyze(SURFACE)
        self.assertEqual([(e.surface, e.reading) for e in entries], [(SURFACE, "とくべつ")])

    def test_repeated_matches_offsets_hints_and_longest_match(self):
        engine = self.engine({"word_splits": {SURFACE: PARTS}})
        hint = SimpleNamespace(start=0, end=4, surface=SURFACE, reading="べつ")
        entries = engine.analyze(SURFACE + "と" + SURFACE + "軍", hints=[hint])
        self.assertEqual([(e.start, e.surface) for e in entries if e.rule_surface],
                         [(0, "世界"), (2, "征服"), (5, "世界"), (7, "征服")])
        engine.dict.merge_json({"words": {"大" + SURFACE: "だいせかいせいふく"}}, "project")
        self.assertEqual([e.surface for e in engine.analyze("大" + SURFACE)], ["大" + SURFACE])
        separate = engine.analyze("世界と征服")
        self.assertFalse(any(e.rule_surface for e in separate))

    def test_strict_validation_rejects_before_file_creation(self):
        invalid = [[], PARTS[:1], [{"surface": "世界", "reading": "せかい"}, {"surface": "服", "reading": "ふく"}],
                   [{"surface": " 世界", "reading": "せかい"}, PARTS[1]],
                   [PARTS[0], {"surface": "征服", "reading": ""}],
                   [PARTS[0], {"surface": "征服", "reading": "せい\nふく"}], ["世界", "征服"]]
        for parts in invalid:
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                ReadingDict().set_word_split(self.path, SURFACE, parts)
            self.assertFalse(self.path.exists())
        self.assertEqual(normalize_word_split(SURFACE, PARTS), PARTS)


class SplitOperations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Project.create(self.root / "project", "splits")
        self.episode = self.project.add_episode("E1")
        self.doc = Document("E1", groups=[Group("g", members=[])])
        self.episode.document = self.doc
        self.ed = Editor(self.doc, {}, reading_engine_factory=self.episode.reading_engine)
        self.current = self.member("current", SURFACE)
        self.other = self.member("other", SURFACE)
        self.path = self.project.dict_path("readings")

    def member(self, mid, text):
        member = Member(mid, "g", "ja", "original", "dialogue", 1000, 3000,
                        runs=[Run(text, ins="i_" + mid, b=len(text))])
        self.doc.members[mid] = member
        self.doc.groups[0].members.append(mid)
        return member

    def seed(self, member, a=0, b=None, *, status="auto", origin="analyzer", reading="せかいせいふく"):
        b = len(visible_map(member)[1]) if b is None else b
        text, ids = visible_map(member)
        ann = Annotation("a_" + member.id + "_" + str(a), "ruby", member.id, anchor_of(ids[a:b]), text.text[a:b], reading,
                         status=status, origin=origin, parts=[[0, b-a, reading]])
        self.doc.annotations[ann.id] = ann
        return ann

    def anns(self, member):
        ids = visible_map(self.doc.members[member.id])[1]
        return sorted((a for a in self.doc.annotations.values() if a.member == member.id and a.kind == "ruby"),
                      key=lambda a: anchor_positions(a, ids))

    def test_here_is_two_words_not_reading_spaces_or_legacy_parts(self):
        ann = self.seed(self.current)
        self.ed.set_reading(ann.id, "せかい せいふく")
        self.assertEqual(len(self.anns(self.current)), 1)
        self.assertEqual(len(reading_spans(self.doc, self.current)), 1)
        before = copy.deepcopy(self.doc.annotations)
        count = len(self.ed.history.done)
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS)
        self.assertEqual([a.base for a in self.anns(self.current)], ["世界", "征服"])
        self.assertTrue(all(a.status == "confirmed" and a.origin == "manual" for a in self.anns(self.current)))
        self.assertEqual(len(reading_spans(self.doc, self.current)), 2)
        self.assertEqual(len(self.ed.history.done), count + 1)
        self.assertFalse(self.path.exists())
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations, before)

    def test_project_apply_and_undo_are_one_step_preserving_fixed_and_runs(self):
        self.seed(self.current, status="confirmed", origin="manual")
        self.seed(self.other)
        kept = []
        for k, (status, origin) in enumerate((("confirmed", "analyzer"), ("hidden", "manual"), ("auto", "manual"))):
            member = self.member("fixed" + str(k), SURFACE)
            kept.append(copy.deepcopy(self.seed(member, status=status, origin=origin, reading="てがき")))
        before = copy.deepcopy(self.doc.annotations)
        runs = {mid: copy.deepcopy(m.runs) for mid, m in self.doc.members.items()}
        self.doc.karaoke["untouched"] = {"beats": [{"t0": 1234, "t1": 2345}]}
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        self.assertEqual(len(self.ed.history.done), 1)
        self.assertTrue(self.ed.history.done[-1].reading_rules_changed)
        self.assertTrue(self.ed.history.done[-1].keeps_text)
        self.assertEqual([a.base for a in self.anns(self.other)], ["世界", "征服"])
        self.assertTrue(all(a.rule == "word_split:" + SURFACE for a in self.anns(self.other)))
        for ann in kept:
            self.assertEqual(self.doc.annotations[ann.id], ann)
        self.assertEqual({mid: m.runs for mid, m in self.doc.members.items()}, runs)
        self.assertEqual(self.doc.karaoke["untouched"]["beats"], [{"t0": 1234, "t1": 2345}])
        after = copy.deepcopy(self.doc.annotations)
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations, before)
        self.assertFalse(self.path.exists())
        self.ed.history.redo()
        self.assertEqual(self.doc.annotations, after)
        self.assertIn(SURFACE, ReadingDict.load(self.path).word_splits)

    def test_existing_bytes_update_delete_undo_and_reload(self):
        self.path.parent.mkdir(exist_ok=True)
        original = '{"future": {"x": 1}, "typed": {"他": ["ほか"]}, "words": {"世界征服": "とくべつ"}}\r\n'.encode("utf-8-sig")
        self.path.write_bytes(original)
        self.seed(self.other)
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        saved = self.path.read_bytes()
        self.episode.save()
        reopened = Project.open(self.project.root).episode("E1")
        self.assertEqual([e.surface for e in reopened.reading_engine().analyze(SURFACE)], ["世界", "征服"])
        self.assertEqual([a.base for a in reopened.document.annotations.values() if a.member == self.current.id], ["世界", "征服"])
        self.ed.split_reading_words(self.current.id, 0, 4, ALTERNATE, scope="project", project=self.project)
        self.assertEqual([a.base for a in self.anns(self.other)], ["世", "界征服"])
        self.ed.history.undo()
        self.assertEqual(self.path.read_bytes(), saved)
        before_delete = copy.deepcopy(self.doc.annotations)
        self.ed.remove_word_split_rule(SURFACE, project=self.project)
        self.assertNotIn(SURFACE, ReadingDict.load(self.path).word_splits)
        self.assertEqual([a.base for a in self.anns(self.current)], ["世界", "征服"])
        self.ed.history.undo()
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(self.doc.annotations, before_delete)
        self.ed.history.undo()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(json.loads(original.decode("utf-8-sig"))["future"], {"x": 1})

    def test_episode_explicit_reading_rule_wins_elsewhere(self):
        self.doc.rules = [{"kind": "reading", "surface": SURFACE, "reading": "とくべつ"}]
        self.seed(self.other)
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        self.assertEqual([(a.base, a.text) for a in self.anns(self.other)], [(SURFACE, "とくべつ")])
        self.assertEqual([a.base for a in self.anns(self.current)], ["世界", "征服"])
        self.assertEqual([e.reading for e in self.episode.reading_engine().analyze(SURFACE)], ["とくべつ"])

    def test_preview_all_occurrences_and_preserves_partial_manual(self):
        repeated = self.member("repeat", SURFACE + "と" + SURFACE)
        fixed = copy.deepcopy(self.seed(repeated, 0, 4, status="confirmed", origin="manual", reading="てがき"))
        self.seed(repeated, 5, 9)
        partial = self.member("partial", SURFACE)
        first = copy.deepcopy(self.seed(partial, 0, 2, status="confirmed", origin="manual", reading="せかい"))
        before = document_to_json(self.doc)
        preview = self.ed.preview_word_split_rule(SURFACE, PARTS, project=self.project)
        items = [item for item in preview if item["member"] == repeated.id]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["kept"], [0, 1, 2, 3])
        self.assertFalse(items[0]["changes"])
        self.assertTrue(items[1]["changes"])
        self.assertEqual(document_to_json(self.doc), before)
        self.assertFalse(self.path.exists())
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        self.assertEqual(self.doc.annotations[fixed.id], fixed)
        self.assertEqual(self.doc.annotations[first.id], first)
        self.assertEqual([a.base for a in self.anns(repeated)], [SURFACE, "世界", "征服"])
        self.assertEqual([a.base for a in self.anns(partial)], ["世界", "征服"])

    def test_two_occurrences_do_not_fill_unannotated_gap(self):
        repeated = self.member("gap", SURFACE + " 秘密 " + SURFACE)
        self.seed(repeated, 0, 4)
        self.seed(repeated, 8, 12)
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        self.assertEqual([a.base for a in self.anns(repeated)], ["世界", "征服", "世界", "征服"])

    def test_removed_larger_automatic_word_resuggests_its_residual(self):
        larger = self.member("larger", SURFACE + "軍")
        self.seed(larger, reading="せかいせいふくぐん")
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        self.assertEqual([a.base for a in self.anns(larger)], ["世界", "征服", "軍"])

    def test_regenerate_protects_manual_merge_unless_explicitly_included(self):
        self.seed(self.current, status="confirmed", origin="manual", reading="てがき")
        ReadingDict().set_word_split(self.path, SURFACE, PARTS)
        engine = self.episode.reading_engine()
        self.ed.regenerate_readings(self.current.id, 0, 4, engine)
        self.assertEqual([a.base for a in self.anns(self.current)], [SURFACE])
        self.ed.regenerate_readings(self.current.id, 0, 4, engine, include_confirmed=True)
        self.assertEqual([a.base for a in self.anns(self.current)], ["世界", "征服"])

    def test_overlap_or_invalid_parts_rejected_before_write(self):
        self.seed(self.current, status="confirmed", origin="manual")
        before = document_to_json(self.doc)
        for a, b, parts in ((0, 3, [{"surface": "世界", "reading": "せかい"}, {"surface": "征", "reading": "せい"}]),
                            (0, 4, [{"surface": "世界", "reading": "せかい"}, {"surface": "服", "reading": "ふく"}])):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                self.ed.split_reading_words(self.current.id, a, b, parts, scope="project", project=self.project)
            self.assertEqual(document_to_json(self.doc), before)
            self.assertFalse(self.path.exists())
        self.assertFalse(self.ed.history.done)

    def test_atomic_write_failure_keeps_file_and_annotations(self):
        self.path.parent.mkdir(exist_ok=True)
        original = b'{"future": [1, 2]}\n'
        self.path.write_bytes(original)
        self.seed(self.current)
        before = copy.deepcopy(self.doc.annotations)
        with patch("subkit.reading.dictionary.os.replace", side_effect=OSError("write denied")), self.assertRaises(OSError):
            self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        self.assertEqual(self.doc.annotations, before)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(list(self.path.parent.glob("*.tmp")))
        self.assertFalse(self.ed.history.done)

    def test_project_isolation_and_cached_factory_not_mutated_by_preview(self):
        another = Project.create(self.root / "another", "other")
        global_path = self.root / "global.json"
        global_path.write_text('{"words": {"他": "ほか"}}', encoding="utf-8")
        original = global_path.read_bytes()
        cached = self.episode.reading_engine()
        self.ed.reading_engine_factory = lambda: cached
        self.ed.preview_word_split_rule(SURFACE, PARTS)
        self.assertEqual(cached.dict.word_splits, {})
        self.ed.split_reading_words(self.current.id, 0, 4, PARTS, scope="project", project=self.project)
        self.assertNotIn(SURFACE, another.reading_engine().dict.word_splits)
        self.assertEqual(global_path.read_bytes(), original)


class SplitUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, **kw):
        from subkit_app.views.word_split_dialog import WordSplitDialog
        dialog = WordSplitDialog(SURFACE, **kw)
        self.addCleanup(dialog.close)
        return dialog

    def test_real_analyzer_prefill_scope_and_romaji(self):
        dialog = self.dialog(analyzer=Analyzer.get(), scope="project")
        self.assertEqual(dialog.values(), (PARTS, "project"))
        dialog.reading_edit.setText("sekai／seifuku")
        self.assertEqual(dialog.values(), (PARTS, "project"))
        self.assertEqual(dialog.help_anchor, "readings")

    def test_invalid_segmentation_alignment_and_empty_reading_block_accept(self):
        dialog = self.dialog(parts=PARTS)
        for words, readings in (("世界/服", "せかい/ふく"), ("世界/征服", "せかい"),
                                ("世界/征服", "せかい/"), (SURFACE, "せかいせいふく")):
            dialog.word_edit.setText(words)
            dialog.reading_edit.setText(readings)
            self.assertFalse(dialog.apply_here_button.isEnabled())
            self.assertFalse(dialog.apply_project_button.isEnabled())
            dialog.accept()
            self.assertEqual(dialog.result(), 0)

    def test_ime_accept_and_cancel_and_remove_without_validation(self):
        from PySide6.QtGui import QInputMethodEvent
        from PySide6.QtWidgets import QApplication, QDialog
        dialog = self.dialog(parts=PARTS, existing_rule=True)
        self.assertFalse(dialog.remove_rule)
        QApplication.sendEvent(dialog.reading_edit, QInputMethodEvent("せ", []))
        self.assertFalse(dialog.apply_here_button.isEnabled())
        self.assertFalse(dialog.apply_project_button.isEnabled())
        dialog.accept()
        self.assertEqual(dialog.result(), 0)
        QApplication.sendEvent(dialog.reading_edit, QInputMethodEvent())
        self.assertTrue(dialog.apply_here_button.isEnabled())
        self.assertTrue(dialog.apply_project_button.isEnabled())
        dialog.word_edit.setText("invalid")
        dialog.remove_button.click()
        self.assertTrue(dialog.remove_rule)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        normal = self.dialog(parts=PARTS)
        self.assertTrue(normal.remove_button.isHidden())
        normal.reject()
        self.assertFalse(normal.remove_rule)

    def test_separate_buttons_choose_scope_without_a_combobox_or_generic_ok(self):
        from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox
        for initial, button_name, expected in (("project", "apply_here_button", "here"),
                                                ("here", "apply_project_button", "project")):
            with self.subTest(initial=initial):
                dialog = self.dialog(parts=PARTS, scope=initial)
                self.assertEqual(dialog.findChildren(QComboBox), [])
                self.assertIsNone(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok))
                getattr(dialog, button_name).click()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
                self.assertEqual(dialog.values(), (PARTS, expected))

    def test_enter_keeps_incoming_scope_and_escape_cancels(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QDialog
        for scope in ("here", "project"):
            with self.subTest(scope=scope):
                dialog = self.dialog(parts=PARTS, scope=scope)
                dialog.show()
                dialog.reading_edit.setFocus()
                self.app.processEvents()
                QTest.keyClick(dialog.reading_edit, Qt.Key.Key_Return)
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
                self.assertEqual(dialog.values(), (PARTS, scope))
        dialog = self.dialog(parts=PARTS)
        dialog.show()
        self.app.processEvents()
        QTest.keyClick(dialog.word_edit, Qt.Key.Key_Escape)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)

    def test_popup_signal_with_existing_changed_and_missing_scope(self):
        from subkit_app.views.reading_popup import ReadingPopup
        cases = [(None, 0, 4), ({"text": "ここは" + SURFACE, "a": 3, "b": 7, "spans": []}, 3, 7)]
        for scope, a, b in cases:
            pop = ReadingPopup(SURFACE, "せかいせいふく", [], "manual", "confirmed", scope=scope)
            self.addCleanup(pop.close)
            got = []
            pop.wordSplitRequested.connect(lambda *args: got.append(args))
            pop.scope_group.button(2).setChecked(True)
            pop.btn_word_split.click()
            self.assertEqual(got, [(a, b, 2)])
        scope = {"text": "世界征服", "a": 0, "b": 2,
                 "spans": [{"a": 0, "b": 2, "reading": "せかい"}, {"a": 2, "b": 4, "reading": "せいふく"}]}
        pop = ReadingPopup("世界", "せかい", [], "analyzer", "auto", scope=scope)
        self.addCleanup(pop.close)
        pop._extend(1)
        got = []
        pop.wordSplitRequested.connect(lambda *args: got.append(args))
        self.assertFalse(pop.btn_word_split.isHidden())
        pop.btn_word_split.click()
        self.assertEqual(got, [(0, 4, 0)])


if __name__ == "__main__":
    unittest.main()
