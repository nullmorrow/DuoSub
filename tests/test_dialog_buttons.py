"""U9a button geometry and behavior, using offscreen Qt and isolated settings."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt, qVersion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QHeaderView,
                               QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget)
from subkit_app import help as help_ui
from tests import test_convert_ui as fixtures


def row_buttons(dialog):
    container = dialog._button_row.container
    layout = container.layout() if isinstance(container, QWidget) else container
    return [layout.itemAt(i).widget() for i in range(layout.count())
            if isinstance(layout.itemAt(i).widget(), QPushButton)]


class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        print("U9a: Qt", qVersion(), "offscreen; browser/player/job doubles")

    def test_box_order_text_default_help_and_keyboard(self):
        dialog = QDialog()
        self.addCleanup(dialog.deleteLater)
        dialog.help_anchor = "keys"
        layout = QVBoxLayout(dialog)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        extra = box.addButton("恢复默认", QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(box)
        box.accepted.connect(dialog.accept)
        box.rejected.connect(dialog.reject)
        help_ui.dialog_buttons(dialog, box)
        for width in (420, 650, 420):
            dialog.resize(width, 120)
            dialog.show()
            self.app.processEvents()
            buttons = row_buttons(dialog)
            self.assertEqual([b.text() for b in buttons], ["恢复默认", "确定", "取消", "帮助"])
            self.assertTrue(buttons[1].isDefault())
            self.assertFalse(extra.isDefault())
            with patch.object(help_ui, "open_manual") as opened:
                buttons[-1].click()
                opened.assert_called_once_with("keys")
                self.assertTrue(dialog.isVisible())
        QTest.keyClick(dialog, Qt.Key.Key_Return)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)

    def test_destructive_accept_and_role_are_never_default(self):
        for text, role in (("移除", QDialogButtonBox.ButtonRole.AcceptRole),
                           ("删除", QDialogButtonBox.ButtonRole.DestructiveRole)):
            with self.subTest(text=text):
                dialog = QDialog()
                self.addCleanup(dialog.deleteLater)
                dialog.help_anchor = "project"
                layout = QVBoxLayout(dialog)
                box = QDialogButtonBox()
                remove = box.addButton(text, role)
                cancel = box.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
                remove.setDefault(True)
                layout.addWidget(box)
                help_ui.dialog_buttons(dialog, box)
                dialog.show()
                self.app.processEvents()
                self.assertFalse(remove.isDefault())
                self.assertFalse(remove.autoDefault())
                self.assertTrue(cancel.isDefault())
                self.assertEqual(row_buttons(dialog)[0], remove)
                dialog.hide()

    def test_message_help_same_row_normal_width_and_original_result(self):
        for standards in (QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No):
            box = QMessageBox()
            self.addCleanup(box.deleteLater)
            box.help_anchor = "save"
            box.setText("Message with enough content to leave room beside ordinary buttons.")
            box.setStandardButtons(standards)
            help_button = help_ui.dialog_buttons(box)
            self.assertIs(help_ui.dialog_buttons(box), help_button)
            box.show()
            self.app.processEvents()
            self.assertNotIn(help_button, box.buttons())
            buttons = row_buttons(box)
            self.assertIs(buttons[-1], help_button)
            centers = [b.mapTo(box, b.rect().center()) for b in buttons]
            self.assertEqual(len({p.y() for p in centers}), 1)
            self.assertEqual([p.x() for p in centers], sorted(p.x() for p in centers))
            self.assertLess(help_button.width(), box.width() // 2)
            self.assertFalse(help_button.isDefault())
            with patch.object(help_ui, "open_manual") as opened:
                help_button.click()
                opened.assert_called_once_with("save")
                self.assertTrue(box.isVisible())
                self.assertIsNone(box.clickedButton())
            ordinary = box.buttons()[0]
            ordinary.click()
            self.assertIs(box.clickedButton(), ordinary)
            self.assertEqual(box.result(), box.standardButton(ordinary))

    def test_manual_apply_auxiliary_and_cancel(self):
        dialog = QDialog()
        self.addCleanup(dialog.deleteLater)
        dialog.help_anchor = "autotime"
        layout = QVBoxLayout(dialog)
        row = QHBoxLayout()
        auxiliary, apply, cancel = [QPushButton(t) for t in ("试听", "应用", "取消")]
        for button in (auxiliary, apply, cancel):
            row.addWidget(button)
        help_ui.dialog_buttons(dialog, row, primary=apply, auxiliary=(auxiliary,))
        layout.addLayout(row)
        dialog.show()
        self.app.processEvents()
        self.assertEqual([b.text() for b in row_buttons(dialog)], ["试听", "应用", "取消", "帮助"])
        self.assertTrue(apply.isDefault())
        self.assertFalse(auxiliary.isDefault())

    def test_startup_message_uses_shared_row(self):
        def inspect(box):
            box.show()
            self.app.processEvents()
            self.assertEqual([b.text() for b in row_buttons(box)], ["确定", "帮助"])
            with patch.object(help_ui, "open_manual") as opened:
                row_buttons(box)[-1].click()
                opened.assert_called_once_with("project")
                self.assertTrue(box.isVisible())
            box.hide()
            return QMessageBox.StandardButton.Ok
        with patch.object(QMessageBox, "exec", inspect):
            self.assertEqual(help_ui.critical_message(None, "启动失败", "错误详情"), QMessageBox.StandardButton.Ok)


class DialogTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.ConversionUI.setUpClass.__func__)
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def check_dialog(self, dialog, expected):
        self.addCleanup(dialog.deleteLater)
        dialog.show()
        self.app.processEvents()
        buttons = row_buttons(dialog)
        self.assertEqual([b.text() for b in buttons], expected, type(dialog).__name__)
        points = [b.mapTo(dialog, b.rect().center()) for b in buttons if b.isVisible()]
        self.assertEqual(len({p.y() for p in points}), 1)
        self.assertEqual([p.x() for p in points], sorted(p.x() for p in points))
        self.assertFalse(buttons[-1].isDefault())
        with patch.object(help_ui, "open_manual") as opened:
            buttons[-1].click()
            opened.assert_called_once_with(dialog.help_anchor)
            self.assertTrue(dialog.isVisible())
        print(type(dialog).__name__ + ": " + " | ".join(expected))
        dialog.hide()

    def test_basic_dialog_classes(self):
        from subkit_app.views import dialogs
        from subkit_app.views.template_editor import TemplateEditor
        from subkit.core.merge import MergeSettings
        from subkit_app.views.group_times import GroupTimesDialog
        from subkit_app.views.lyrics_map import LyricsMapDialog
        from subkit_app.views.dialign_apply import DialogueAlignApplyDialog
        from subkit_app.views.dialign_result import DialogueAlignResultDialog
        from subkit_app.views.kalign_result import KalignResultDialog
        from subkit_app.views.fonts import FontCheckDialog
        from subkit.core.fontcheck import FontCheck
        self.group("g", "song")
        with patch.object(TemplateEditor, "_schedule"):
            self.check_dialog(dialogs.NewProjectDialog(self.window), ["建立工程", "取消", "帮助"])
        for cls in (dialogs.AddEpisodeDialog, dialogs.MergeDialog, dialogs.ExportDialog):
            self.check_dialog(cls(self.window), ["确定", "取消", "帮助"])
        self.check_dialog(dialogs.ConvertToSongDialog(self.doc, [], self.window, styles=MergeSettings().styles,
                         template_styles=[], settings=self.settings), ["转换", "取消", "帮助"])
        self.check_dialog(GroupTimesDialog(self.doc, "g", self.window), ["确定", "取消", "帮助"])
        self.check_dialog(LyricsMapDialog(self.doc, "g", {}), ["应用", "取消", "帮助"])
        self.check_dialog(DialogueAlignApplyDialog(self.doc, {}, [], (0, 0)), ["采用勾选的", "取消", "帮助"])
        self.check_dialog(DialogueAlignResultDialog(), ["整集采用…", "关闭", "帮助"])
        self.check_dialog(KalignResultDialog(), ["关闭", "帮助"])
        for exporting in (False, True):
            self.check_dialog(FontCheckDialog(FontCheck(), exporting=exporting),
                              ["仍然导出", "取消", "帮助"] if exporting else ["关闭", "帮助"])

    def test_reports_batch_and_progress(self):
        from subkit_app.views.check_result import CheckResultDialog
        from subkit.model.validate import Report
        from tests.test_review_fixes_f5 import episode, sample
        from subkit_app.views.batch import BatchDialog, BatchReportDialog
        from subkit.core.batch import BatchResult
        from subkit_app.episode_job import EpisodeProgress
        from subkit_app.export_job import ExportProgress
        self.check_dialog(CheckResultDialog(episode(sample()), Report()), ["关闭", "帮助"])
        self.check_dialog(CheckResultDialog(episode(sample()), Report(), merged=True), ["开始编辑", "帮助"])
        self.check_dialog(BatchReportDialog(BatchResult([], self.path / "report.txt", self.path), "report"),
                          ["打开输出文件夹", "关闭", "帮助"])
        self.check_dialog(BatchDialog("5.1", self.settings, lambda: (0, 0), self.path, Mock(), Mock()),
                          ["开始", "取消", "关闭", "帮助"])
        for cls in (EpisodeProgress, ExportProgress):
            dialog = cls(self.window)
            self.check_dialog(dialog, ["取消", "帮助"])
            cancel = Mock()
            dialog.cancelRequested.connect(cancel)
            dialog.show()
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
            cancel.assert_called_once()
            self.assertTrue(dialog.isVisible())
            dialog.hide()

    def test_listening_preserves_no_enter_adoption(self):
        from subkit_app.views.ai_transcribe import ListeningDialog
        from tests.test_ai_transcribe_ui import Request, AudioDouble, PlayerDouble
        with patch("subkit_app.views.ai_transcribe.QAudioOutput", AudioDouble), \
                patch("subkit_app.views.ai_transcribe.QMediaPlayer", PlayerDouble), \
                patch.object(ListeningDialog, "_initial_start"):
            dialog = ListeningDialog(Request(), "译文")
            self.check_dialog(dialog, ["补入原文", "关闭", "帮助"])
            self.assertFalse(any(b.isDefault() for b in dialog.findChildren(QPushButton)))
            dialog.timer.stop()

    def test_shortcuts_columns_and_bottom_actions(self):
        from subkit_app.views.shortcuts_dialog import ShortcutsDialog
        dialog = ShortcutsDialog(self.window.action_hub.registry, self.window)
        self.check_dialog(dialog, ["全部恢复默认", "打开快捷键一览", "确定", "取消", "帮助"])
        dialog.show()
        for width in (860, 1040):
            dialog.resize(width, 620)
            self.app.processEvents()
            for column in (1, 2):
                self.assertEqual(dialog.table.header().sectionResizeMode(column), QHeaderView.ResizeMode.ResizeToContents)
                for item in dialog.rows.values():
                    index = dialog.table.indexFromItem(item, column)
                    rect = dialog.table.visualRect(index)
                    needed = dialog.table.fontMetrics().horizontalAdvance(item.text(column))
                    self.assertGreaterEqual(rect.width() - 6, needed, item.text(column))
                longest = max(dialog.rows.values(), key=lambda item: dialog.table.fontMetrics().horizontalAdvance(item.text(column)))
                print(f"Shortcuts width={width}, column={column}, column_px={dialog.table.columnWidth(column)}, "
                      f"longest_text_px={dialog.table.fontMetrics().horizontalAdvance(longest.text(column))}: {longest.text(column)}")
            self.assertEqual(dialog.table.header().sectionResizeMode(0), QHeaderView.ResizeMode.Stretch)
        dialog.hide()


class AutoTimeDialogs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_autotime_ui import AutoTimeUI
        AutoTimeUI.setUpClass()
        cls.app = AutoTimeUI.app

    def test_three_timing_dialogs(self):
        from tests.test_autotime_ui import AutoTimeUI, wait_preview
        from subkit_app.views import autotime as ui
        from subkit.core.autotime_doc import plan
        fixture = AutoTimeUI()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.app = fixture.app
        self.check_dialog = DialogTests.check_dialog.__get__(self)
        self.check_dialog(ui.EpisodeAutoProcessDialog(500, False, ["OP"], fixture.window),
                          ["套用重复段落…", "自动调轴", "先不处理", "帮助"])
        preview = plan(fixture.doc, {}, fixture.features, mapping_override=fixture.map)
        self.check_dialog(ui.AutoTimeResultDialog(fixture.session, preview, (1, 1), None,
                                                fixture.features, fixture.window),
                          ["撤销", "逐段查看和调整…", "关闭", "帮助"])
        with patch.object(ui, "plan", return_value=preview):
            dialog = ui.AutoTimeDetailsDialog(fixture.session, fixture.features, fixture.window.player,
                                             fixture.settings, fixture.window)
            wait_preview(dialog)
            self.check_dialog(dialog, ["试听", "应用", "取消", "帮助"])
            dialog.reject()
