"""Offscreen Qt pinyin UI, real undo/rules and temporary soma project copies; fake player."""

from __future__ import annotations

import copy
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as ui
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QInputMethodEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton
from subkit.core.readings import anchor_positions, visible_map
from subkit.core.workspace import Project
from subkit.core.pinyin_notes import INVALID_PINYIN, INVALID_RULE
from subkit_app import help as help_ui
from subkit_app.views.reading_popup import PinyinSettingsDialog, ReadingPopup


class PinyinUI(unittest.TestCase):
    setUpClass = classmethod(ui.ConversionUI.setUpClass.__func__)
    _dispose = ui.ConversionUI._dispose
    group = ui.ConversionUI.group
    select = ui.ConversionUI.select
    snapshot = ui.ConversionUI.snapshot

    def setUp(self):
        ui.ConversionUI.setUp(self)
        path = self.path / "soma"
        shutil.copytree(ui.ROOT / "test_projects/soma", path, copy_function=shutil.copyfile)
        self.project = Project.open(path)
        self.window.project = self.session.project = self.project
        self.episode.settings = self.project.settings
        self.first = self.group("g", text="行长绿了")
        self.other = self.group("other", text="行长")
        self.session.editor.set_pinyin_mode(self.project, "PY")
        self.select("g")

    def popup(self, a=0, b=1):
        self.window._on_pinyin(self.first.id, a, b, QPoint(30, 30))
        pop = next(p for p in reversed(self.window.findChildren(ReadingPopup)) if p.isVisible())
        self.addCleanup(pop.close)
        return pop

    def anns(self, mid=None):
        member = self.doc.members[mid or self.first.id]
        ids = visible_map(member)[1]
        return sorted((a for a in self.doc.annotations.values() if a.member == member.id and a.kind == "pinyin"),
                      key=lambda a: anchor_positions(a, ids))

    def test_settings_apply_single_undo_restores_data_and_cursor_status(self):
        self.session.editor.history.undo()
        before = self.snapshot()
        self.window.statusBar().showMessage("之前的状态")
        count = len(self.session.editor.history.done)
        seen = []
        def accept(dialog):
            dialog.modes.button(2).setChecked(True)
            seen.extend(b.text() for b in dialog.modes.buttons())
            self.assertEqual(dialog.windowTitle(), "拼音")
            return dialog.DialogCode.Accepted
        real = self.session.editor.set_pinyin_mode
        def apply(project, mode):
            self.assertEqual(QApplication.overrideCursor().shape(), Qt.CursorShape.WaitCursor)
            self.assertEqual(self.window.statusBar().currentMessage(), "正在应用拼音设置……")
            return real(project, mode)
        with patch.object(PinyinSettingsDialog, "exec", accept), patch.object(self.session.editor, "set_pinyin_mode", apply):
            self.window._pinyin_settings()
        self.assertEqual(seen, ["关", "拼音（声调符号）", "注音符号", "威妥玛"])
        self.assertEqual(self.project.pinyin_mode, "ZY")
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(self.window.statusBar().currentMessage(), "之前的状态")
        self.session.editor.history.undo()
        self.assertEqual(self.project.pinyin_mode, "off")
        self.assertEqual(self.snapshot(), before)

    def test_settings_exception_and_cancel_restore_cursor_status(self):
        self.window.statusBar().showMessage("之前的状态")
        with patch.object(PinyinSettingsDialog, "exec", return_value=PinyinSettingsDialog.DialogCode.Accepted), \
                patch.object(self.session.editor, "set_pinyin_mode", side_effect=RuntimeError("test error")), \
                patch.object(QMessageBox, "warning") as warning:
            self.window._pinyin_settings()
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(self.window.statusBar().currentMessage(), "之前的状态")
        self.assertIn("test error", warning.call_args.args[2])
        with patch.object(PinyinSettingsDialog, "exec", return_value=PinyinSettingsDialog.DialogCode.Rejected), \
                patch.object(self.session.editor, "set_pinyin_mode") as setter:
            self.window._pinyin_settings()
        setter.assert_not_called()

    def test_popup_candidate_number_typing_invalid_and_hide(self):
        pop = self.popup()
        self.assertEqual(pop.windowTitle(), "拼音：行")
        self.assertEqual(pop.windowType(), Qt.WindowType.Tool)
        self.assertEqual([b.text() for b in pop.scope_group.buttons()], ["只改这里", "这一集都这样", "整个工程都这样"])
        expected = pop.cands[1]
        QTest.keyClick(pop.edit, Qt.Key.Key_2)
        self.assertEqual(self.anns()[0].text, expected)
        self.assertFalse(pop.isVisible())
        for text, expected in (("hang2", "háng"), ("háng", "háng"), ("lv4", "lǜ"), ("le5", "le")):
            pop = self.popup()
            pop.edit.setText(text)
            pop.btn_ok.click()
            self.assertEqual(self.anns()[0].text, expected)
        pop = self.popup()
        pop.edit.setText("wrong2")
        pop.btn_ok.click()
        self.assertTrue(pop.isVisible())
        self.assertEqual(pop.preview.text(), INVALID_PINYIN)
        pop.edit.clear()
        self.app.sendEvent(pop.edit, QInputMethodEvent("h", []))
        QTest.keyClick(pop.edit, Qt.Key.Key_2)
        self.assertTrue(pop.isVisible())
        self.app.sendEvent(pop.edit, QInputMethodEvent("", []))
        pop.btn_hide.click()
        self.assertEqual(self.anns()[0].status, "hidden")
        saved = copy.deepcopy(self.doc.annotations)
        self.session.editor.reannotate_pinyin(self.doc.members)
        self.assertEqual(self.anns()[0], saved[self.anns()[0].id])

    def test_word_scopes_preview_cancel_protection_and_one_undo(self):
        for scope in (1, 2):
            pop = self.popup(0, 2)
            pop.edit.setText("xing2")
            self.assertEqual(pop.preview.text(), INVALID_RULE)
            pop.edit.setText("xing2 chang2")
            pop.scope_group.button(scope).setChecked(True)
            before, count = self.snapshot(), len(self.session.editor.history.done)
            with patch.object(self.window, "_help_message", return_value=QMessageBox.StandardButton.No):
                pop.btn_ok.click()
            self.assertEqual(self.snapshot(), before)
            pop = self.popup(0, 2)
            pop.edit.setText("xing2 chang2")
            pop.scope_group.button(scope).setChecked(True)
            with patch.object(self.window, "_help_message", return_value=QMessageBox.StandardButton.Yes) as preview:
                pop.btn_ok.click()
            self.assertIn("本集会更新", preview.call_args.args[3])
            self.assertEqual(len(self.session.editor.history.done), count + 1)
            self.assertEqual([a.text for a in self.anns(self.other.id)], ["xíng", "cháng"])
            self.assertEqual(self.project.pinyin_words() if scope == 2 else {r['surface']: r['reading'] for r in self.doc.rules},
                             {"行长": "xíng cháng"})
            self.session.editor.history.undo()
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.project.pinyin_words(), {})
        self.session.editor.set_reading(self.anns()[0].id, "hang2")
        with patch.object(self.window, "_help_message", return_value=QMessageBox.StandardButton.Yes) as preview:
            self.window._apply_pinyin(self.first.id, 0, 2, "xíng cháng", 1)
        self.assertIn("保持不变", preview.call_args.args[3])
        self.assertEqual(self.anns()[0].text, "háng")

    def test_editor_row_click_drag_language_and_off(self):
        self.window.show()
        self.app.processEvents()
        row = self.window.editor.pinyin_rows[0]
        QTest.mouseClick(row, Qt.MouseButton.LeftButton, pos=row._chars[0].center().toPoint())
        self.assertTrue(any(p.isVisible() for p in self.window.findChildren(ReadingPopup)))
        for pop in self.window.findChildren(ReadingPopup):
            pop.close()
        start, end = [row._chars[i].center().toPoint() for i in (0, 1)]
        QTest.mousePress(row, Qt.MouseButton.LeftButton, pos=start)
        self.app.sendEvent(row, QMouseEvent(QEvent.Type.MouseMove, QPointF(end), QPointF(row.mapToGlobal(end)),
                                           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        QTest.mouseRelease(row, Qt.MouseButton.LeftButton, pos=end)
        self.assertTrue(any(p.isVisible() and p.base == "行长" for p in self.window.findChildren(ReadingPopup)))
        for pop in self.window.findChildren(ReadingPopup):
            pop.close()
        self.first.lane = "original"
        self.select("g")
        self.assertEqual(len(self.window.editor.pinyin_rows), 1)
        self.session.editor.set_pinyin_mode(self.project, "off")
        self.select("g")
        self.assertEqual(self.window.editor.pinyin_rows, [])
        self.window._on_pinyin(self.first.id, 0, 1, QPoint())
        self.assertFalse(any(p.isVisible() for p in self.window.findChildren(ReadingPopup)))

    def test_watch_right_click_base_and_pinyin_owner_only_when_on(self):
        watch = self.window.watch
        watch.enabled = True
        watch.ass = Mock()
        watch._ready = Mock(return_value=True)
        watch._global = Mock(return_value=QPoint(20, 20))
        ids = visible_map(self.first)[1]
        for mode in ("off", "PY"):
            self.project.set_pinyin_mode(mode)
            for kind in ("base", "pinyin"):
                line = SimpleNamespace(member=self.first.id, group="g", kind=kind,
                                       ann=self.anns()[0].id if kind == "pinyin" else None)
                watch._hit = Mock(return_value=(SimpleNamespace(line=line, identity=ids[0], char=0), (0, 0)))
                watch._press(QPointF(10, 10), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
                visible = [p for p in self.window.findChildren(ReadingPopup) if p.isVisible()]
                self.assertEqual(bool(visible), mode != "off")
                for pop in visible:
                    self.assertEqual(pop.windowTitle(), "拼音：行")
                    pop.close()

    def test_help_and_escape(self):
        pop = self.popup()
        self.assertEqual(help_ui.resolve_anchor(self.window, active=pop, focus=pop.edit), "pinyin")
        help_button = next(b for b in pop.findChildren(QPushButton) if b.text() == "帮助")
        with patch.object(help_ui, "open_manual") as opener:
            help_button.click()
        opener.assert_called_once_with("pinyin")
        before = self.snapshot()
        QTest.keyClick(pop.edit, Qt.Key.Key_Escape)
        self.assertFalse(pop.isVisible())
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
