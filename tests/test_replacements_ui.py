"""Real Qt replacement drafts and Project entry points; no player/model/render jobs."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox

from subkit.core.replacements import load_rules
from subkit.core.workspace import Project
from subkit.model.serialize import document_to_json
from subkit_app import help as help_ui
from subkit_app.views.replacements import ReplacementsDialog


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class ReplacementsQt(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_replacements_ui_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Project.create(self.root / "project", "本工程", original="en", translation="zh-Hant")
        self.other = Project.create(self.root / "other", "另一工程")
        self.dialogs = []
        self.addCleanup(self.dispose)

    def dispose(self):
        for dialog in self.dialogs:
            dialog.reject()
            dialog.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def dialog(self, project=None, **kwargs):
        dialog = ReplacementsDialog(project or self.project, **kwargs)
        self.dialogs.append(dialog)
        dialog.show()
        dialog.activateWindow()
        self.app.processEvents()
        return dialog

    def add(self, d, source, target="", lane="both", enabled=True):
        d.add_button.click()
        i = d.table.rowCount() - 1
        d.table.cellWidget(i, 1).setText(source)
        d.table.cellWidget(i, 2).setText(target)
        box = d.table.cellWidget(i, 3)
        box.setCurrentIndex(box.findData(lane))
        d.table.cellWidget(i, 0).setChecked(enabled)
        return i

    def test_mouse_and_keyboard_can_edit_a_b_tab_and_save(self):
        d = self.dialog(samples={"translation": "OldName"})
        QTest.mouseClick(d.add_button, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        source, target = d.table.cellWidget(0, 1), d.table.cellWidget(0, 2)
        self.assertIs(self.app.focusWidget(), source)
        QTest.keyClicks(self.app.focusWidget(), "OldName")
        self.assertEqual(source.text(), "OldName")
        QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=target.rect().center())
        self.assertIs(self.app.focusWidget(), target)
        QTest.keyClicks(self.app.focusWidget(), "NewName")
        self.assertEqual(target.text(), "NewName")
        self.assertEqual(d.output.toPlainText(), "NewName")
        QTest.mouseClick(source, Qt.MouseButton.LeftButton, pos=source.rect().center())
        self.assertIs(self.app.focusWidget(), source)
        self.assertEqual((d.table.currentRow(), d.table.currentColumn()), (0, 1))
        QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Tab)
        self.assertIs(self.app.focusWidget(), target)
        self.assertEqual((d.table.currentRow(), d.table.currentColumn()), (0, 2))
        QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Backspace)
        self.assertEqual(target.text(), "")
        self.assertEqual(d.output.toPlainText(), "")
        QTest.mouseClick(d.save_button, Qt.MouseButton.LeftButton)
        self.assertTrue(d.saved)
        self.assertEqual(load_rules(Project.open(self.project.root)), [
            dict(enabled=True, source="OldName", target="", lane="both")])

    def test_actual_focus_accepts_ime_commits_and_tracks_clicked_row(self):
        before = snapshot(self.project.root)
        d = self.dialog(samples={"original": "黒井津さん"})
        QTest.mouseClick(d.add_button, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        source, target = d.table.cellWidget(0, 1), d.table.cellWidget(0, 2)
        self.assertIs(self.app.focusWidget(), source)
        self.app.sendEvent(self.app.focusWidget(), QInputMethodEvent("くろいつ", []))
        self.assertIs(self.app.focusWidget(), source)
        self.assertFalse(d.save_button.isEnabled())
        commit = QInputMethodEvent()
        commit.setCommitString("黒井津")
        self.app.sendEvent(self.app.focusWidget(), commit)
        self.assertEqual(source.text(), "黒井津")
        self.assertTrue(d.save_button.isEnabled())
        QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=target.rect().center())
        self.assertIs(self.app.focusWidget(), target)
        self.app.sendEvent(self.app.focusWidget(), QInputMethodEvent("hei jing jin", []))
        self.assertFalse(d.save_button.isEnabled())
        commit = QInputMethodEvent()
        commit.setCommitString("黑井津")
        self.app.sendEvent(self.app.focusWidget(), commit)
        self.assertEqual(d.output.toPlainText(), "黑井津さん")
        self.assertTrue(d.save_button.isEnabled())
        QTest.mouseClick(d.add_button, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        first_target = d.table.cellWidget(0, 2)
        QTest.mouseClick(first_target, Qt.MouseButton.LeftButton, pos=first_target.rect().center())
        self.assertIs(self.app.focusWidget(), first_target)
        self.assertEqual(d.table.currentRow(), 0)
        self.assertEqual([i.row() for i in d.table.selectionModel().selectedRows()], [0])
        self.assertTrue(d.delete_button.isEnabled())
        QTest.mouseClick(d.delete_button, Qt.MouseButton.LeftButton)
        self.assertEqual(d.table.rowCount(), 1)
        self.assertEqual(d.rules()[0]["source"], "")
        d.reject()
        self.assertEqual(snapshot(self.project.root), before)

    def test_crud_disabled_rows_and_cancel_are_only_drafts(self):
        before, data = snapshot(self.project.root), copy.deepcopy(self.project.data)
        d = self.dialog(samples={"translation": "多余名字"})
        self.assertEqual(d.table.rowCount(), 0)
        self.assertFalse(d.save_button.isEnabled())
        self.add(d, "多余", "", "translation")
        self.add(d, "名字", "姓名", "both")
        d.table.cellWidget(1, 2).setText("人名")
        self.assertEqual(d.output.toPlainText(), "人名")
        d.table.cellWidget(0, 0).setChecked(False)
        self.assertEqual(d.output.toPlainText(), "多余人名")
        self.assertIn("原声（英语）", d.table.cellWidget(0, 3).itemText(0))
        self.assertEqual(d.table.cellWidget(0, 3).itemText(1), "翻译（繁中）")
        d.table.selectRow(1)
        d.delete_button.click()
        self.assertEqual(d.table.rowCount(), 1)
        self.assertFalse(d.rules()[0]["enabled"])
        self.assertEqual(snapshot(self.project.root), before)
        d.reject()
        self.assertFalse(d.saved)
        self.assertEqual(snapshot(self.project.root), before)
        self.assertEqual(self.project.data, data)

    def test_save_once_reopen_project_isolation_and_empty_target_delete(self):
        other_before = snapshot(self.other.root)
        d = self.dialog(samples={"translation": "王某多余旧名"})
        self.add(d, "王某", "王明", "translation")
        self.add(d, "王明", "王强", "both")
        self.add(d, "多余", "", "both")
        self.add(d, "旧名", "新名", "original", enabled=False)
        self.assertEqual(d.output.toPlainText(), "王明旧名")
        with patch.object(self.project, "save", wraps=self.project.save) as save:
            d.save_button.click()
        save.assert_called_once()
        self.assertTrue(d.saved)
        self.assertEqual(d.result(), QDialog.DialogCode.Accepted)
        reopened = Project.open(self.project.root)
        again = self.dialog(reopened, samples={"translation": "王某多余旧名"})
        self.assertEqual(again.rules(), d.rules())
        self.assertEqual(again.output.toPlainText(), "王明旧名")
        self.assertFalse(again.save_button.isEnabled())
        isolated = self.dialog(self.other)
        self.assertEqual(isolated.rules(), [])
        self.assertEqual(snapshot(self.other.root), other_before)

    def test_move_priority_and_literal_preview_does_not_chain(self):
        d = self.dialog(samples={"original": "AB A . a"})
        self.add(d, "A", "X", "original")
        self.add(d, "AB", "Y", "both")
        self.add(d, "X", "Z", "both")
        self.add(d, ".", "!", "both")
        self.assertEqual(d.output.toPlainText(), "XB X ! a")
        d.table.selectRow(1)
        d.up_button.click()
        self.assertEqual(d.table.currentRow(), 0)
        self.assertFalse(d.up_button.isEnabled())
        self.assertEqual(d.output.toPlainText(), "Y X ! a")
        d.down_button.click()
        self.assertEqual(d.output.toPlainText(), "XB X ! a")
        d.table.selectRow(d.table.rowCount() - 1)
        self.assertFalse(d.down_button.isEnabled())

    def test_invalid_structure_is_explained_and_spaces_remain_literal(self):
        d = self.dialog(samples={"translation": " A A "})
        self.add(d, "", "")
        self.assertFalse(d.save_button.isEnabled())
        self.assertIn("词 A", d.status.text())
        source, target = d.table.cellWidget(0, 1), d.table.cellWidget(0, 2)
        source.setText(" A ")
        target.setText(" B ")
        self.assertTrue(d.save_button.isEnabled())
        self.assertEqual(d.output.toPlainText(), " B A ")
        for value in ("{\\b1}", "a\\N", "a\nb", "a\x00b"):
            target.setText(value)
            self.assertFalse(d.save_button.isEnabled())
            self.assertIn("不能", d.status.text())
        target.setText("")
        self.assertTrue(d.save_button.isEnabled())
        self.assertEqual(d.output.toPlainText(), "A ")
        self.assertEqual(d.rules()[0]["source"], " A ")

    def test_save_failure_keeps_dialog_and_project_bytes(self):
        d = self.dialog()
        self.add(d, "旧名", "新名")
        before, data = snapshot(self.project.root), copy.deepcopy(self.project.data)
        with patch.object(self.project, "save", side_effect=OSError("只读工程")), \
                patch.object(QMessageBox, "warning") as warning:
            d.save_button.click()
        self.assertIn("没有保存", warning.call_args.args[2])
        self.assertTrue(d.isVisible())
        self.assertFalse(d.saved)
        self.assertEqual(snapshot(self.project.root), before)
        self.assertEqual(self.project.data, data)

    def test_ime_preedit_and_enter_cannot_save_incomplete_rule(self):
        d = self.dialog()
        self.add(d, "黒井津", "黑井津")
        edit = d.table.cellWidget(0, 1)
        edit.setFocus()
        self.app.sendEvent(edit, QInputMethodEvent("くろいつ", []))
        self.assertFalse(d.save_button.isEnabled())
        self.assertFalse(d.add_button.isEnabled())
        d.accept()
        self.assertFalse(d.saved)
        self.assertEqual(load_rules(self.project), [])
        self.app.sendEvent(edit, QInputMethodEvent())
        self.assertTrue(d.save_button.isEnabled())
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.assertTrue(d.isVisible())
        self.assertFalse(d.saved)
        d.save_button.click()
        self.assertTrue(d.saved)

    def test_f1_and_help_button_open_anchor_without_accepting(self):
        d = self.dialog()
        self.add(d, "旧名", "新名")
        d.sample.setFocus()
        self.app.processEvents()
        with patch.object(help_ui, "open_manual", return_value=True) as opened:
            QTest.keyClick(d.sample, Qt.Key.Key_F1)
            self.app.processEvents()
            opened.assert_called_once_with("display-replacements")
            opened.reset_mock()
            help_button = next(b for b in d.buttons.buttons() if b.text() == "帮助")
            help_button.click()
            opened.assert_called_once_with("display-replacements")
        self.assertTrue(d.isVisible())
        self.assertFalse(d.saved)
        self.assertEqual(load_rules(self.project), [])


class ReplacementsEntry(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.real_project = Project.create(self.path / "project", "本工程")

    def test_menu_entry_uses_current_plain_text_and_refreshes_after_save(self):
        member = fixtures.ConversionUI.group(self, "line", text=r"{\b1}旧名\N多余{\p1}m 0 0 l 1 1{\p0}")
        before = copy.deepcopy(document_to_json(self.doc))
        self.window.project, self.window.current_gid = self.real_project, "line"
        self.window.display_menu.aboutToShow.emit()
        action = self.window.act_replacements
        self.assertIn(action, self.window.display_menu.actions())
        self.assertEqual(action.text(), "批量替换…")
        self.assertEqual(action.property("help_anchor"), "display-replacements")
        self.assertTrue(action.statusTip())
        self.assertTrue(action.isEnabled())
        def choose(d):
            self.assertEqual(d.sample.toPlainText(), "旧名\n多余")
            d.add_rule()
            d.table.cellWidget(0, 1).setText("旧名")
            d.table.cellWidget(0, 2).setText("新名")
            d.save_button.click()
            return d.result()
        with patch.object(ReplacementsDialog, "exec", choose), patch.object(self.window, "_apply_rules") as apply:
            action.trigger()
        apply.assert_called_once()
        self.session.request_preview.assert_called_once()
        self.assertEqual(load_rules(self.real_project)[0]["target"], "新名")
        self.assertEqual(document_to_json(self.doc), before)
        self.assertEqual(member.text, r"{\b1}旧名\N多余{\p1}m 0 0 l 1 1{\p0}")

    def test_no_current_episode_can_save_and_cancel_does_not_refresh(self):
        self.window.project, self.window.session = self.real_project, None
        self.window.display_menu.aboutToShow.emit()
        def choose(d):
            self.assertIn("示例", d.sample_origin.text())
            d.add_rule()
            d.table.cellWidget(0, 1).setText("余字")
            d.table.cellWidget(0, 2).setText("")
            d.accept()
            return d.result()
        with patch.object(ReplacementsDialog, "exec", choose), patch.object(self.window, "_apply_rules") as apply:
            self.window.act_replacements.trigger()
        apply.assert_called_once()
        self.assertEqual(load_rules(Project.open(self.real_project.root))[0]["target"], "")
        before = snapshot(self.real_project.root)
        with patch.object(ReplacementsDialog, "exec", return_value=QDialog.DialogCode.Rejected), \
                patch.object(self.window, "_apply_rules") as apply:
            self.window.act_replacements.trigger()
        apply.assert_not_called()
        self.assertEqual(snapshot(self.real_project.root), before)
        self.window.project = None
        self.window.display_menu.aboutToShow.emit()
        self.assertFalse(self.window.act_replacements.isEnabled())
        with patch.object(ReplacementsDialog, "exec") as shown:
            self.window._open_replacements()
        shown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
