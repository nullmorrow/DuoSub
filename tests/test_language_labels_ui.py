"""Project language labels and Japanese-only controls, offscreen Qt / fake player."""

from __future__ import annotations

import shutil
import unittest
from unittest.mock import patch

from tests import test_convert_ui as ui
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QLabel, QGroupBox
from subkit.core.review import build_packet
from subkit.core.workspace import Project
from subkit.languages import JAPANESE_ONLY, LANGUAGE_NAMES, has_japanese, language_name, lane_label
from subkit_app.views.dialogs import AddEpisodeDialog, MergeDialog, NewProjectDialog
from subkit_app.views.group_table import C_CN, C_JP, GroupTableModel
from subkit_app.views.template_editor import TemplateEditor
from subkit_app.watch import InlineTextEditor


class LanguageLabelsUI(unittest.TestCase):
    setUpClass = classmethod(ui.ConversionUI.setUpClass.__func__)
    setUp = ui.ConversionUI.setUp
    _dispose = ui.ConversionUI._dispose
    group = ui.ConversionUI.group
    select = ui.ConversionUI.select
    snapshot = ui.ConversionUI.snapshot
    context_menu = ui.ConversionUI.context_menu

    def labels(self, widget):
        return " ".join(w.text() for w in widget.findChildren(QLabel))

    def test_names_fallback_and_japanese_in_either_lane(self):
        for code, name in LANGUAGE_NAMES.items():
            self.assertEqual(language_name(code), name)
        self.assertEqual(language_name("pt-BR"), "葡萄牙语")
        self.assertEqual(lane_label("original", {"original": "en"}), "原声（英语）")
        self.assertTrue(has_japanese({"original": "en", "translation": "ja-JP"}))
        self.assertFalse(has_japanese({"original": "en", "translation": "zh-Hans"}))

    def test_soma_copy_labels_preserve_document_and_controls(self):
        dest = self.path / "soma"
        shutil.copytree(ui.ROOT / "test_projects/soma", dest, copy_function=shutil.copyfile)
        project = Project.open(dest)
        project.data["languages"]["translation"] = "zh-Hant"
        self.assertEqual(project.languages, {"original": "ja", "translation": "zh-Hant"})
        self.window.project = project
        self.window._apply_language_labels()
        ep = project.episode(project.episode_ids()[0])
        before = ui.document_to_json(ep.document)
        model = GroupTableModel(ep.document, languages=project.languages)
        self.addCleanup(model.deleteLater)
        for column, label in ((C_CN, "翻译（繁中）"), (C_JP, "原声（日语）")):
            self.assertEqual(model.headerData(column, Qt.Orientation.Horizontal), label)
        self.window.editor.set_group(ep.document, ep.document.groups[0], None, None)
        self.assertEqual(self.window.editor.jp_edit.toolTip(), "原声（日语）")
        self.assertTrue(self.window.act_hide_period.isEnabled())
        self.assertTrue(self.window.act_brackets.isEnabled())
        self.assertTrue(self.window.wave_panel.k_unit.isEnabled())
        self.assertEqual(ui.document_to_json(ep.document), before)

    def test_english_project_disabled_controls_delete_lanes_and_undo(self):
        project = Project.create(self.path / "english", "English", original="en", translation="zh-Hans")
        self.window.project = project
        self.window._apply_language_labels()
        model = GroupTableModel(self.doc, languages=project.languages)
        self.addCleanup(model.deleteLater)
        self.assertEqual(model.headerData(C_JP, Qt.Orientation.Horizontal), "原声（英语）")
        self.assertEqual(model.headerData(C_CN, Qt.Orientation.Horizontal), "翻译（简中）")
        for control in (self.window.act_hide_period, self.window.act_brackets, self.window.wave_panel.k_unit):
            self.assertFalse(control.isEnabled())
            self.assertEqual(control.toolTip(), JAPANESE_ONLY)
        trans = self.group("trans", text="你好")
        orig = self.group("orig", text="Hello world")
        orig.lang, orig.lane = "en", "original"
        self.select("orig")
        self.assertTrue(self.window.editor.ruby.isHidden())
        actions = self.context_menu()
        names = [a.text() for a in actions] + [child.text() for a in actions if a.menu() for child in a.menu().actions()]
        self.assertNotIn("只删原声", names)       # one language selected: same as deleting the group
        self.select("orig", "trans")
        actions = self.context_menu()
        names = [a.text() for a in actions] + [child.text() for a in actions if a.menu() for child in a.menu().actions()]
        self.assertIn("只删原声", names)
        self.assertIn("只删翻译", names)
        before = self.snapshot()
        with patch.object(self.window, "_ask", return_value=True):
            self.context_menu("只删原声")
        self.assertNotIn(orig.id, self.doc.members)
        self.assertIn(trans.id, self.doc.members)
        self.session.editor.history.undo()
        self.assertEqual(self.snapshot(), before)
        project.data["languages"]["translation"] = "ja"
        self.window._apply_language_labels()
        self.assertTrue(self.window.act_hide_period.isEnabled())
        self.assertTrue(self.window.act_brackets.isEnabled())
        self.assertTrue(self.window.wave_panel.k_unit.isEnabled())

    def test_dialog_and_template_labels_update_with_new_project_languages(self):
        languages = {"original": "en", "translation": "zh-Hans"}
        for cls in (AddEpisodeDialog, MergeDialog):
            dialog = cls(languages=languages)
            self.addCleanup(dialog.deleteLater)
            labels = self.labels(dialog)
            self.assertIn("原声（英语）", labels)
            self.assertIn("翻译（简中）", labels)
        with patch.object(TemplateEditor, "_schedule"):
            dialog = NewProjectDialog()
            self.addCleanup(dialog.deleteLater)
            dialog.original.setCurrentIndex(dialog.original.findData("en"))
            dialog.translation.setCurrentIndex(dialog.translation.findData("zh-Hans"))
            editor = dialog.editor
            titles = [box.title() for box in editor.findChildren(QGroupBox)]
            self.assertIn("原声（英语）", titles)
            self.assertIn("翻译（简中）", titles)
            self.assertIn("翻译（简中）距底边", self.labels(editor))
            self.assertEqual(editor.w["jp_text"].title, "原声（英语）文字颜色")
            self.assertFalse(editor.w["ruby_size"].isEnabled())
            self.assertEqual(editor.w["ruby_size"].toolTip(), JAPANESE_ONLY)
            dialog.translation.setCurrentIndex(dialog.translation.findData("ja"))
            self.assertTrue(editor.w["ruby_size"].isEnabled())

    def test_inline_hint_and_review_packet_follow_language(self):
        m = self.group("english", text="Hello world")
        m.lang, m.lane = "en", "original"
        self.group("chinese", text="你好")
        editor = InlineTextEditor(self.doc, [m.id], m.id, QRect(0, 0, 600, 100), 20)
        self.addCleanup(editor.close)
        self.addCleanup(editor.deleteLater)
        self.assertIn("原声（英语）", editor.head.text())
        self.assertIn(JAPANESE_ONLY, editor.hint.text())
        self.assertNotIn("Ctrl+G", editor.hint.text())
        self.assertTrue(editor.ruby.isHidden())
        md, _ = build_packet(self.doc, episode="English", orig_lang="en", trans_lang="zh-Hans")
        self.assertIn("原声（英语）", md)
        self.assertIn("翻译（简中）", md)
        self.assertNotIn("中文是否准确翻译了同一组的日文", md)


if __name__ == "__main__":
    unittest.main()
