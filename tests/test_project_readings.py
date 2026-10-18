"""Project-only rule management, draft UI and byte-exact external-file undo."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QInputMethodEvent
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMessageBox
from subkit.core.ops import Editor
from subkit.core.project_readings import ProjectReadingsDraft, normalize_rule_input
from subkit.core.readings import anchor_of, visible_map
from subkit.core.workspace import Project
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.reading.dictionary import ReadingDict
from subkit_app.views.project_readings import ProjectReadingRuleDialog, ProjectReadingsDialog


class ProjectReadingsFixtures:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_manager_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Project.create(self.root / "project", "管理测试")
        self.path = self.project.dict_path("readings")

    def write(self, data, *, bom=False):
        raw = (json.dumps(data, ensure_ascii=False, indent=1).replace("\n", "\r\n") + "\r\n").encode("utf-8-sig" if bom else "utf-8")
        self.path.write_bytes(raw)
        return raw

    def rows(self, draft):
        return {(row.table, row.key): row for row in draft.rows()}

    def editor(self):
        ep = self.project.add_episode("E1")
        ep.document = Document("E1", groups=[Group("g")])
        ed = Editor(ep.document, {}, reading_engine_factory=ep.reading_engine)
        for i, (status, origin, surface) in enumerate((("auto", "analyzer", "世界征服"),
                ("confirmed", "analyzer", "世界征服"), ("hidden", "manual", "世界征服"),
                ("auto", "manual", "世界征服"), ("auto", "analyzer", "学校"))):
            mid = "m" + str(i)
            m = Member(mid, "g", "ja", "original", "dialogue", 1000, 3000,
                       runs=[Run(surface, ins="i" + mid, b=len(surface))])
            ep.document.members[mid] = m
            ep.document.groups[0].members.append(mid)
            ids = visible_map(m)[1]
            ep.document.annotations["a" + str(i)] = Annotation("a" + str(i), "ruby", mid,
                anchor_of(ids), surface, "とくべつ", status=status, origin=origin, parts=[[0, len(surface), "とくべつ"]])
        return ed


class ProjectReadingsTests(ProjectReadingsFixtures, unittest.TestCase):
    def test_lists_all_storage_types_and_preserves_unedited_rules_and_unknown_fields(self):
        data = {"words": {"世界征服": "せかいせいふく"}, "readings": {"幸平 創真": "ゆきひら そうま"},
                "splits": {"迷宮都市": [{"surface": "迷宮", "reading": "めいきゅう"}, {"surface": "都市", "reading": "とし"}]},
                "word_splits": {"研究開発": [{"surface": "研究", "reading": "けんきゅう"}, {"surface": "開発", "reading": "かいはつ"}]},
                "prefer": {"辛い": "からい"}, "typed": {"世界": ["せかい"]}, "speakers": {"人": {"私": "わたくし"}},
                "candidates": {"上手": ["じょうず"]}, "unknown": {"keep": [1, True]}}
        original = self.write(data, bom=True)
        draft = ProjectReadingsDraft(self.project)
        self.assertEqual([r.kind for r in draft.rows()], ["word", "segments", "segments", "word_split"])
        draft.set_rule("word", "追加", "ついか")
        result = draft.save()
        saved = json.loads(self.path.read_text(encoding="utf-8-sig"))
        for key, value in data.items():
            if key == "words":
                for surface, reading in value.items():
                    self.assertEqual(saved[key][surface], reading)
            else:
                self.assertEqual(saved[key], value)
        self.assertEqual(result.backup.read_bytes(), original)
        self.assertTrue(self.path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\n", self.path.read_bytes().replace(b"\r\n", b""))

    def test_edit_legacy_segmented_rules_keeps_one_word_and_storage_semantics(self):
        self.write({"readings": {"幸平 創真": "ゆきひら そうま"}, "splits": {"迷宮都市": [
            {"surface": "迷宮", "reading": "めいきゅう"}, {"surface": "都市", "reading": "とし"}]}})
        draft = ProjectReadingsDraft(self.project)
        a = self.rows(draft)[("readings", "幸平 創真")]
        b = self.rows(draft)[("splits", "迷宮都市")]
        draft.set_rule("segments", "幸平/創真", "こうへい/そうま", original=a)
        draft.set_rule("segments", "迷宮/都市", "めいきゅう/まち", original=b)
        draft.save()
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["readings"]["幸平 創真"], "こうへい そうま")
        self.assertEqual(saved["splits"]["迷宮都市"][1]["reading"], "まち")
        rd = ReadingDict.load(self.path)
        self.assertFalse(rd.word_splits)
        self.assertEqual(len(rd.words["迷宮都市"].groups), 2)
        self.assertFalse(rd.words["迷宮都市"].separate_words)

    def test_crud_exclusive_type_conversion_and_persistence(self):
        draft = ProjectReadingsDraft(self.project)
        rule = draft.set_rule("word", "世界征服", "せかいせいふく")
        with self.assertRaisesRegex(ValueError, "已有工程规则"):
            draft.set_rule("word_split", "世界/征服", "せかい/せいふく")
        rule = draft.set_rule("word_split", "世界/征服", "せかい/せいふく", original=rule)
        draft.save()
        self.assertIn("世界征服", ReadingDict.load(self.path).word_splits)
        draft = ProjectReadingsDraft(self.project)
        rule = draft.rows()[0]
        draft.remove_rule(rule)
        draft.save()
        self.assertFalse(ReadingDict.load(self.path).word_splits)

    def test_add_then_delete_is_no_change_and_creates_no_file(self):
        draft = ProjectReadingsDraft(self.project)
        rule = draft.set_rule("word", "世界", "せかい")
        draft.remove_rule(rule)
        self.assertFalse(draft.dirty)
        self.assertEqual(draft.save().changed_surfaces, ())
        self.assertFalse(self.path.exists())

    def test_default_new_empty_project_and_project_isolation(self):
        other = Project.create(self.root / "other", "其他工程")
        other.dict_path("readings").write_text('{"words":{"別":"べつ"}}', encoding="utf-8")
        original = other.dict_path("readings").read_bytes()
        draft = ProjectReadingsDraft(self.project)
        self.assertEqual(draft.rows(), [])
        draft.set_rule("segments", "世界/征服", "せかい/せいふく")
        draft.save()
        self.assertEqual(other.dict_path("readings").read_bytes(), original)
        self.project.data["dictionaries"]["readings"] = str(other.dict_path("readings"))
        with self.assertRaisesRegex(ValueError, "只管理本工程"):
            ProjectReadingsDraft(self.project)

    def test_concurrent_edit_creation_and_deletion_are_rejected(self):
        for initial, external in ((None, b'{}'), (b'{}', b'{"other":1}'), (b'{}', None)):
            with self.subTest(initial=initial, external=external):
                self.path.unlink(missing_ok=True)
                if initial is not None:
                    self.path.write_bytes(initial)
                draft = ProjectReadingsDraft(self.project)
                draft.set_rule("word", "世界", "せかい")
                self.path.unlink(missing_ok=True)
                if external is not None:
                    self.path.write_bytes(external)
                with self.assertRaisesRegex(RuntimeError, "其他操作改过"):
                    draft.save()
                self.assertEqual(self.path.read_bytes() if self.path.exists() else None, external)

    def test_single_transaction_byte_exact_undo_redo_and_manual_protection(self):
        original = self.write({"words": {"世界征服": "せかいせいふく"}, "typed": {"保存": ["ほぞん"]}}, bom=True)
        ed = self.editor()
        before = copy.deepcopy(ed.doc)
        protected = {aid: copy.deepcopy(a) for aid, a in ed.doc.annotations.items() if a.member != "m0"}
        draft = ProjectReadingsDraft(self.project)
        draft.set_rule("word_split", "世界/征服", "せかい/せいふく", original=draft.rows()[0])
        draft.set_rule("word", "追加", "ついか")
        result = draft.save(editor=ed)
        self.assertEqual(len(ed.history.done), 1)
        self.assertEqual(result.changed_surfaces, ("世界征服", "追加"))
        self.assertEqual([a.base for a in ed.doc.annotations.values() if a.member == "m0"], ["世界", "征服"])
        for aid, ann in protected.items():
            self.assertEqual(ed.doc.annotations[aid], ann)
        self.assertEqual(ed.doc.members, before.members)
        saved, after = self.path.read_bytes(), copy.deepcopy(ed.doc.annotations)
        ed.history.undo()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(ed.doc.annotations, before.annotations)
        ed.history.redo()
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(ed.doc.annotations, after)

    def test_missing_file_undo_removes_file_and_refresh_failure_rolls_back(self):
        ed = self.editor()
        before = copy.deepcopy(ed.doc.annotations)
        draft = ProjectReadingsDraft(self.project)
        draft.set_rule("word", "世界征服", "せかいせいふく")
        with patch.object(ed, "_word_split_engine", side_effect=RuntimeError("engine failed")):
            with self.assertRaisesRegex(RuntimeError, "engine failed"):
                draft.save(editor=ed)
        self.assertFalse(self.path.exists())
        self.assertEqual(ed.doc.annotations, before)
        self.assertFalse(ed.history.done)
        draft.save(editor=ed)
        self.assertTrue(self.path.exists())
        ed.history.undo()
        self.assertFalse(self.path.exists())

    def test_validation_and_unedited_invalid_legacy_data_preserved(self):
        self.write({"words": {"古": ["future"]}, "future": 1})
        draft = ProjectReadingsDraft(self.project)
        self.assertTrue(draft.rows()[0].error)
        draft.set_rule("word", "世界", "せかい")
        draft.save()
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["words"]["古"], ["future"])
        for kind, surfaces, readings in (("word", "世界/征服", "せかい/せいふく"),
                ("segments", "世界", "せかい"), ("word_split", "世界/征服", "せかい/"),
                ("word_split", "世界/征服", "せかい"), ("word", "世 界", "せかい")):
            with self.subTest(kind=kind, surfaces=surfaces, readings=readings):
                with self.assertRaises(ValueError):
                    normalize_rule_input(kind, surfaces, readings)


class ProjectReadingsQt(ProjectReadingsFixtures, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, editor=None):
        dialog = ProjectReadingsDialog(self.project, editor=editor)
        self.addCleanup(dialog.deleteLater)
        dialog.show()
        self.app.processEvents()
        return dialog

    def test_qt_search_filter_add_edit_delete_are_drafts_and_cancel_never_writes(self):
        original = self.write({"words": {"世界": "せかい", "迷宮 都市": "めいきゅう とし"}})
        d = self.dialog()
        self.assertEqual(d.table.rowCount(), 2)
        for query in ("迷宮都市", "めいきゅうとし"):
            d.search.setText(query)
            self.assertEqual(d.table.rowCount(), 1)
        d.search.setText("めいきゅう")
        self.assertEqual(d.table.rowCount(), 1)
        d.kind.setCurrentIndex(d.kind.findData("word"))
        self.assertEqual(d.table.rowCount(), 0)
        d.search.clear()
        self.assertEqual(d.table.rowCount(), 1)
        d.table.selectRow(0)
        self.assertTrue(d.edit_button.isEnabled())
        def edit(child):
            child.reading_edit.setText("せけん")
            child.accept()
            return child.result()
        with patch.object(ProjectReadingRuleDialog, "exec", edit):
            d.edit_button.click()
        self.assertTrue(d.save_button.isEnabled())
        self.assertIn("世界", d.status.text())
        self.assertEqual(self.path.read_bytes(), original)
        d.table.selectRow(0)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            d.delete_button.click()
        d.reject()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(d.saved)

    def test_qt_save_all_and_child_preserves_segment_semantics_and_ime(self):
        self.write({"readings": {"幸平 創真": "ゆきひら そうま"}})
        d = self.dialog()
        d.table.selectRow(0)
        def edit(child):
            self.assertEqual(child.kind.currentData(), "segments")
            self.assertEqual(child.word_edit.text(), "幸平/創真")
            child.reading_edit.setText("こうへい/そうま")
            self.app.sendEvent(child.reading_edit, QInputMethodEvent("こう", []))
            child.accept()
            self.assertEqual(child.result(), QDialog.DialogCode.Rejected)
            self.app.sendEvent(child.reading_edit, QInputMethodEvent())
            child.accept()
            return child.result()
        with patch.object(ProjectReadingRuleDialog, "exec", edit):
            d.edit_button.click()
        d.save_button.click()
        self.assertTrue(d.saved)
        self.assertEqual(d.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["readings"]["幸平 創真"], "こうへい そうま")
        self.assertTrue(d.backup.exists())
        self.assertEqual(d.help_anchor, "project-readings")

    def test_qt_blank_project_add_rule_and_concurrency_error_keeps_window(self):
        d = self.dialog()
        def add(child):
            child.kind.setCurrentIndex(child.kind.findData("word_split"))
            child.word_edit.setText("世界/征服")
            child.reading_edit.setText("せかい/せいふく")
            child.accept()
            return child.result()
        with patch.object(ProjectReadingRuleDialog, "exec", add):
            d.add_button.click()
        self.assertEqual(d.table.rowCount(), 1)
        self.assertFalse(self.path.exists())
        self.path.write_bytes(b'{"external":true}')
        with patch.object(QMessageBox, "warning") as warning:
            d.save_button.click()
        warning.assert_called_once()
        self.assertFalse(d.saved)
        self.assertEqual(d.result(), QDialog.DialogCode.Rejected)
        self.assertEqual(self.path.read_bytes(), b'{"external":true}')


if __name__ == "__main__":
    unittest.main()
