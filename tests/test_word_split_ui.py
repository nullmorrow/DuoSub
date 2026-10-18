"""Project word rules through real popup signals and MainWindow/session transactions."""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from tests import test_format_scope as format_fixtures
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QDialog, QMessageBox

from subkit.core.readings import annotate_member
from subkit.reading.dictionary import ReadingDict
from subkit.reading.engine import ReadingEngine
from subkit_app.views.reading_popup import ReadingPopup


PARTS = [{"surface": "世界", "reading": "せかい"}, {"surface": "征服", "reading": "せいふく"}]


class WordSplitUI(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = format_fixtures.FormatScope._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        format_fixtures.FormatScope.setUp(self)
        self.dictionary = self.path / "readings.json"
        self.project.dict_path = lambda name: self.dictionary
        self.project.reading_engine = lambda **kwargs: ReadingEngine(ReadingDict.load(self.dictionary))
        self.episode.reading_engine = self.project.reading_engine
        self.session.editor.reading_engine_factory = self.project.reading_engine
        annotate_member(self.doc, self.jp, self.project.reading_engine())

    def close_popups(self):
        for pop in self.window.findChildren(ReadingPopup):
            pop.close()

    def test_project_split_slot_and_session_engine_undo_redo(self):
        old_engine = self.window._engine()

        def choose(dialog):
            dialog.word_edit.setText("世界/征服")
            dialog.reading_edit.setText("せかい/せいふく")
            dialog.apply_project_button.click()
            return dialog.result()

        with patch("subkit_app.views.word_split_dialog.WordSplitDialog.exec", choose), \
             patch.object(self.window, "_help_message", return_value=QMessageBox.StandardButton.Yes):
            self.window._on_word_split(self.jp.id, 0, 4, 2)
        self.assertEqual(json.loads(self.dictionary.read_bytes())["word_splits"]["世界征服"], PARTS)
        self.assertEqual([a.base for a in self.doc.annotations_of(self.jp.id)], ["世界", "征服"])
        self.assertIsNone(self.session.engine)
        self.assertIsNot(self.window._engine(), old_engine)
        self.session.editor.history.undo()
        self.assertFalse(self.dictionary.exists())
        self.assertEqual([a.base for a in self.doc.annotations_of(self.jp.id)], ["世界征服"])
        self.assertIsNone(self.session.engine)
        self.session.editor.history.redo()
        self.assertEqual(json.loads(self.dictionary.read_bytes())["word_splits"]["世界征服"], PARTS)

    def test_current_sentence_button_overrides_popup_project_scope_without_saving_rule(self):
        def choose(dialog):
            dialog.word_edit.setText("世界/征服")
            dialog.reading_edit.setText("せかい/せいふく")
            dialog.apply_here_button.click()
            return dialog.result()

        with patch("subkit_app.views.word_split_dialog.WordSplitDialog.exec", choose), \
             patch.object(self.window, "_help_message") as preview:
            self.window._on_word_split(self.jp.id, 0, 4, 2)
        preview.assert_not_called()
        self.assertFalse(self.dictionary.exists())
        self.assertEqual([a.base for a in self.doc.annotations_of(self.jp.id)], ["世界", "征服"])
        self.session.editor.history.undo()
        self.assertEqual([a.base for a in self.doc.annotations_of(self.jp.id)], ["世界征服"])

    def test_project_button_preview_cancel_keeps_document_and_dictionary(self):
        before = copy.deepcopy(self.doc)

        def choose(dialog):
            dialog.word_edit.setText("世界/征服")
            dialog.reading_edit.setText("せかい/せいふく")
            dialog.apply_project_button.click()
            return dialog.result()

        with patch("subkit_app.views.word_split_dialog.WordSplitDialog.exec", choose), \
             patch.object(self.window, "_help_message", return_value=QMessageBox.StandardButton.No) as preview:
            self.window._on_word_split(self.jp.id, 0, 4, 0)
        preview.assert_called_once()
        self.assertFalse(self.dictionary.exists())
        self.assertEqual(self.doc, before)

    def test_cancel_leaves_document_and_dictionary_unchanged(self):
        before = copy.deepcopy(self.doc)
        with patch("subkit_app.views.word_split_dialog.WordSplitDialog.exec", return_value=QDialog.DialogCode.Rejected):
            self.window._on_word_split(self.jp.id, 0, 4, 2)
        self.assertFalse(self.dictionary.exists())
        self.assertEqual(self.doc, before)

    def test_original_reading_popup_routes_current_range(self):
        ann = self.doc.annotations_of(self.jp.id)[0]
        got = Mock()
        with patch.object(self.window, "_on_word_split", got):
            self.window._on_reading(ann.id, QPoint(30, 30))
            pop = self.window.findChildren(ReadingPopup)[-1]
            pop.btn_word_split.click()
            self.app.processEvents()
        got.assert_called_once_with(self.jp.id, 0, 4, 0)
        self.close_popups()

    def test_drag_selection_popup_routes_scope(self):
        got = Mock()
        with patch.object(self.window, "_on_word_split", got):
            self.window._on_new_reading(self.jp.id, 0, 4, QPoint(30, 30))
            pop = self.window.findChildren(ReadingPopup)[-1]
            pop.scope_group.button(2).setChecked(True)
            pop.btn_word_split.click()
            self.app.processEvents()
        got.assert_called_once_with(self.jp.id, 0, 4, 2)
        self.close_popups()

    def test_popup_queued_action_cannot_target_replaced_session(self):
        got = Mock()
        with patch.object(self.window, "_on_word_split", got):
            self.window._on_new_reading(self.jp.id, 0, 4, QPoint(30, 30))
            pop = self.window.findChildren(ReadingPopup)[-1]
            pop.btn_word_split.click()
            self.window.session = None
            try:
                self.app.processEvents()
            finally:
                self.window.session = self.session
        got.assert_not_called()
        self.close_popups()

    def test_delete_project_split_from_dialog_is_undoable(self):
        ReadingDict().set_word_split(self.dictionary, "世界征服", PARTS)

        def remove(dialog):
            self.assertFalse(dialog.remove_button.isHidden())
            dialog.remove_button.click()
            return dialog.result()

        with patch("subkit_app.views.word_split_dialog.WordSplitDialog.exec", remove):
            self.window._on_word_split(self.jp.id, 0, 4, 2)
        self.assertNotIn("世界征服", json.loads(self.dictionary.read_bytes()).get("word_splits", {}))
        self.session.editor.history.undo()
        self.assertIn("世界征服", json.loads(self.dictionary.read_bytes())["word_splits"])

    def test_manager_menu_works_without_current_episode(self):
        self.window.session = None
        try:
            with patch("subkit_app.views.project_readings.ProjectReadingsDialog") as factory:
                factory.return_value.exec.return_value = QDialog.DialogCode.Rejected
                self.window._project_readings()
            factory.assert_called_once_with(self.project, editor=None, parent=self.window)
        finally:
            self.window.session = self.session


if __name__ == "__main__":
    unittest.main()
