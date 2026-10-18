"""Release-only cleanup and annotation ordering through real export dialogs and jobs."""
from __future__ import annotations

import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from tests.test_reference_export_ui import document
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox
from subkit.core.export import ExportReport
from subkit.core.fontcheck import FontCheck
from subkit.core.workspace import Episode, Project
from subkit.model.document import Document
from subkit_app import help as app_help
from subkit_app import mainwindow as mw
from subkit_app.export_job import ExportJob, ExportProgress, SeasonExportJob
from subkit_app.views.dialogs import ExportDialog
from subkit_app.views.season_export import SeasonExportDialog, SeasonReportDialog


class _PresentationFixture:
    @classmethod
    def setUpClass(cls):
        # Also runnable under an independently created, hidden Windows QApplication.
        cls.app = QApplication.instance() or QApplication([])

    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        original_progress = ExportProgress.__init__

        def hidden_progress(progress, *args, **kwargs):
            original_progress(progress, *args, **kwargs)
            progress.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)

        progress_patch = patch.object(ExportProgress, "__init__", hidden_progress)
        progress_patch.start()
        self.addCleanup(progress_patch.stop)
        self.project = Project.create(self.path / "presentation", "发布选项")
        self.episode = self.project.add_episode("S01E001")
        self.episode.document = document(self.episode.id, " 这里是軟體 ")
        self.episode.save()
        self.session.episode = self.episode
        self.window.project = self.project
        self.out = self.path / "out"
        self.out.mkdir()

    def hidden_dialog(self, dialog):
        self.addCleanup(dialog.deleteLater)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        dialog.show()
        self.app.processEvents()
        return dialog

    def select_after_parent(self, dialog):
        combo = dialog.annotation_order
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key.Key_End)
        self.assertEqual(combo.currentData(), "after_parent")

    def check_modal_help(self, dialog):
        def press_and_close():
            QTest.keyClick(dialog, Qt.Key.Key_F1)
            dialog.reject()

        # Hidden native windows have no OS active-window identity. Supply the
        # dialog that this test put in the modal loop, while exercising real F1.
        with patch.object(app_help, "open_manual") as opened, \
                patch.object(QApplication, "activeModalWidget", return_value=dialog):
            QTimer.singleShot(0, press_and_close)
            dialog.exec()
        opened.assert_called_once_with("export")

    def capture_build(self, calls):
        original = Episode.build_export

        def build(ep, **options):
            calls.append((ep.id, dict(options), threading.get_ident()))
            return original(ep, **options)

        return build


class SinglePresentationUI(_PresentationFixture, unittest.TestCase):
    def choose(self, dialog, *, both=False, order="after_parent"):
        dialog.path.setText(str(self.out / "release.ass"))
        dialog.zh_both.setChecked(both)
        dialog.annotation_order.setCurrentIndex(dialog.annotation_order.findData(order))
        return QDialog.DialogCode.Accepted

    def export(self, *, both=False, order="after_parent", probe=None):
        calls = []
        with patch.object(ExportDialog, "exec", lambda d: self.choose(d, both=both, order=order)), \
                patch.object(Episode, "build_export", self.capture_build(calls)), \
                patch.object(mw, "check_fonts", side_effect=probe or (lambda ass, runner=None: FontCheck())), \
                patch.object(QMessageBox, "information"), \
                patch.object(QMessageBox, "warning") as warning:
            self.window.export()
        warning.assert_not_called()
        return calls

    def test_default_reset_keyboard_selection_help_and_no_persistence(self):
        before = copy.deepcopy(self.project.data)
        self.settings.setValue("export/annotation_order", "after_parent")
        d = self.hidden_dialog(ExportDialog(self.window, has_chinese=True))
        self.assertEqual(d.annotation_order.currentData(), "end")
        self.assertEqual([d.annotation_order.itemData(i) for i in range(d.annotation_order.count())],
                         ["end", "after_parent"])
        self.assertIn("歌曲", d.annotation_order.toolTip())
        self.assertIn("Pinyin", d.annotation_order.toolTip())
        self.select_after_parent(d)
        self.check_modal_help(d)
        d.accept()
        reopened = self.hidden_dialog(ExportDialog())
        self.assertEqual(reopened.annotation_order.currentData(), "end")
        self.assertEqual(self.settings.value("export/annotation_order"), "after_parent")
        self.assertEqual(self.project.data, before)
        self.assertTrue(any("自动清理每行首尾半角空格" in label.text() for label in d.findChildren(QLabel)))
        self.assertEqual(list(self.out.iterdir()), [])

    def test_single_after_parent_reaches_worker_and_report_without_editing_project(self):
        saved = (self.episode.dir / "document.json").read_bytes()
        before = copy.deepcopy(self.episode.document)
        calls = self.export()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["annotation_order"], "after_parent")
        self.assertNotEqual(calls[0][2], threading.get_ident())
        text = (self.out / "release.ass").read_text(encoding="utf-8-sig")
        self.assertIn("这里是軟體", text)
        self.assertNotIn(" 这里是軟體 ", text)
        report = (self.out / "release.export-report.txt").read_text(encoding="utf-8")
        self.assertIn("注音行顺序：紧跟对应正文行", report)
        self.assertIn("发布清理", report)
        self.assertEqual(self.episode.document, before)
        self.assertEqual((self.episode.dir / "document.json").read_bytes(), saved)
        self.assertFalse(self.settings.contains("export/annotation_order"))

    def test_single_default_tail_is_explicitly_passed(self):
        calls = self.export(order="end")
        self.assertEqual(calls[0][1]["annotation_order"], "end")
        report = (self.out / "release.export-report.txt").read_text(encoding="utf-8")
        self.assertIn("注音行顺序：统一放在文件末尾", report)

    def test_single_dual_scripts_share_order_and_clean_both_outputs(self):
        calls = self.export(both=True)
        self.assertEqual([kw["zh_direction"] for _, kw, _ in calls], ["simplified", "traditional"])
        self.assertTrue(all(kw["annotation_order"] == "after_parent" for _, kw, _ in calls))
        for suffix, body in (("zh-Hans", "这里是软体"), ("zh-Hant", "這裡是軟體")):
            target = self.out / f"release.{suffix}.ass"
            self.assertIn(body, target.read_text(encoding="utf-8-sig"))
            self.assertNotIn(" " + body + " ", target.read_text(encoding="utf-8-sig"))
            self.assertIn("紧跟对应正文行", target.with_suffix(".export-report.txt").read_text(encoding="utf-8"))

    def test_single_cancel_dialog_starts_no_worker_and_writes_nothing(self):
        def reject(dialog):
            dialog.annotation_order.setCurrentIndex(1)
            dialog.reject()
            return dialog.result()

        with patch.object(ExportDialog, "exec", reject), \
                patch.object(mw.QThreadPool.globalInstance(), "start") as start:
            self.window.export()
        start.assert_not_called()
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertFalse(self.settings.contains("export/annotation_order"))

    def test_single_cancel_second_script_writes_no_first_script_or_report(self):
        probes = []

        def cancel(ass, runner=None):
            probes.append(ass)
            if len(probes) == 2:
                self.window._export_job.cancel()
            return FontCheck()

        self.export(both=True, probe=cancel)
        self.assertEqual(len(probes), 2)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(self.window.statusBar().currentMessage(), "已取消导出")

    def test_report_zero_cleanup_is_quiet_and_missing_new_fields_remain_compatible(self):
        report = ExportReport()
        lines = self.window._export_report_lines(report, FontCheck())
        self.assertFalse(any("发布清理" in line for line in lines))
        report.edge_spaces_removed = 3
        report.empty_events_removed = 2
        report.annotation_order = "after_parent"
        lines = self.window._export_report_lines(report, FontCheck())
        self.assertIn("注音行顺序：紧跟对应正文行", lines)
        self.assertTrue(any("3 个首尾半角空格" in line and "2 条空白字幕" in line for line in lines))
        old = SimpleNamespace(**{k: v for k, v in vars(report).items()
                                 if k not in ("edge_spaces_removed", "empty_events_removed", "annotation_order")})
        lines = self.window._export_report_lines(old, FontCheck())
        self.assertIn("注音行顺序：统一放在文件末尾", lines)
        self.assertFalse(any("发布清理" in line for line in lines))


class SeasonPresentationUI(_PresentationFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        second = self.project.add_episode("S01E002")
        second.document = document(second.id, " 这里是軟體 ")
        second.save()
        self.window.session = None

    def choose(self, dialog, *, both=False, order="after_parent"):
        dialog.folder.setText(str(self.out))
        dialog.zh_both.setChecked(both)
        dialog.annotation_order.setCurrentIndex(dialog.annotation_order.findData(order))
        dialog.accept()
        return dialog.result()

    def export(self, *, both=False, order="after_parent", probe=None):
        calls = []
        with patch.object(SeasonExportDialog, "exec", lambda d: self.choose(d, both=both, order=order)), \
                patch.object(SeasonReportDialog, "exec", return_value=QDialog.DialogCode.Accepted), \
                patch.object(Episode, "build_export", self.capture_build(calls)), \
                patch.object(mw, "check_fonts", side_effect=probe or (lambda ass, runner=None: FontCheck())), \
                patch.object(QMessageBox, "warning") as warning:
            self.window.export_season()
        warning.assert_not_called()
        return calls

    def test_season_default_reset_keyboard_help_and_no_saved_option(self):
        before = copy.deepcopy(self.project.data)
        self.settings.setValue("export/annotation_order", "after_parent")
        d = self.hidden_dialog(SeasonExportDialog(self.project, self.settings, parent=self.window))
        d.folder.setText(str(self.out))
        self.assertEqual(d.annotation_order.currentData(), "end")
        self.select_after_parent(d)
        self.check_modal_help(d)
        d.accept()
        self.assertEqual(d.result(), QDialog.DialogCode.Accepted)
        reopened = self.hidden_dialog(SeasonExportDialog(self.project, self.settings))
        self.assertEqual(reopened.annotation_order.currentData(), "end")
        self.assertEqual(self.settings.value("export/annotation_order"), "after_parent")
        self.assertEqual(self.project.data, before)
        self.assertTrue(any("跳过空白字幕" in label.text() for label in d.findChildren(QLabel)))
        self.assertEqual(list(self.out.iterdir()), [])

    def test_season_after_parent_reaches_every_worker_report_and_leaves_documents(self):
        saved = {eid: (self.project.episode(eid).dir / "document.json").read_bytes()
                 for eid in ("S01E001", "S01E002")}
        calls = self.export()
        self.assertEqual([eid for eid, _, _ in calls], ["S01E001", "S01E002"])
        self.assertTrue(all(kw["annotation_order"] == "after_parent" for _, kw, _ in calls))
        self.assertTrue(all(thread != threading.get_ident() for _, _, thread in calls))
        for eid in saved:
            self.assertEqual((self.project.episode(eid).dir / "document.json").read_bytes(), saved[eid])
            body = (self.out / f"{eid}.zh-Hant.ass").read_text(encoding="utf-8-sig")
            self.assertIn("这里是軟體", body)
            self.assertNotIn(" 这里是軟體 ", body)
        report = (self.out / "presentation.zh-Hant.export-report.txt").read_text(encoding="utf-8")
        self.assertEqual(report.count("注音行顺序：紧跟对应正文行"), 2)
        self.assertEqual(report.count("发布清理"), 2)
        self.assertFalse(self.settings.contains("export/annotation_order"))

    def test_season_default_tail_reaches_each_episode(self):
        calls = self.export(order="end")
        self.assertTrue(all(kw["annotation_order"] == "end" for _, kw, _ in calls))
        report = (self.out / "presentation.zh-Hant.export-report.txt").read_text(encoding="utf-8")
        self.assertEqual(report.count("注音行顺序：统一放在文件末尾"), 2)

    def test_season_dual_scripts_share_order_and_cleanup(self):
        calls = self.export(both=True)
        self.assertEqual([kw["zh_direction"] for _, kw, _ in calls], ["simplified", "traditional"] * 2)
        self.assertTrue(all(kw["annotation_order"] == "after_parent" for _, kw, _ in calls))
        self.assertEqual(len(list(self.out.glob("*.ass"))), 4)
        for eid in ("S01E001", "S01E002"):
            for suffix, body in (("zh-Hans", "这里是软体"), ("zh-Hant", "這裡是軟體")):
                text = (self.out / f"{eid}.{suffix}.ass").read_text(encoding="utf-8-sig")
                self.assertIn(body, text)
                self.assertNotIn(" " + body + " ", text)
        report = (self.out / "presentation.zh-Hans+zh-Hant.export-report.txt").read_text(encoding="utf-8")
        self.assertEqual(report.count("注音行顺序：紧跟对应正文行"), 4)

    def test_season_cancel_dialog_starts_no_worker_and_writes_nothing(self):
        def reject(dialog):
            dialog.annotation_order.setCurrentIndex(1)
            dialog.reject()
            return dialog.result()

        with patch.object(SeasonExportDialog, "exec", reject), \
                patch.object(mw.QThreadPool.globalInstance(), "start") as start:
            self.window.export_season()
        start.assert_not_called()
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertFalse(self.settings.contains("export/annotation_order"))

    def test_season_cancel_last_script_writes_no_partial_outputs(self):
        probes = []

        def cancel(ass, runner=None):
            probes.append(ass)
            if len(probes) == 4:
                self.window._export_job.cancel()
            return FontCheck()

        self.export(both=True, probe=cancel)
        self.assertEqual(len(probes), 4)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(self.window.statusBar().currentMessage(), "已取消整季导出，没有写入文件")


class WorkerPresentationOptions(unittest.TestCase):
    def test_single_worker_default_and_selected_order_reach_both_scripts(self):
        for order in (None, "after_parent"):
            with self.subTest(order=order):
                build = Mock(return_value=(None, ExportReport()))
                ep = SimpleNamespace(build_export=build)
                kw = {} if order is None else {"annotation_order": order}
                job = ExportJob(ep, lambda ass, runner=None: FontCheck(), zh_both=True, **kw)
                job.run()
                self.assertIsNone(job.error)
                options = [call.kwargs for call in job.episode.build_export.call_args_list]
                self.assertEqual([item["annotation_order"] for item in options], [order or "end"] * 2)
                self.assertEqual([item["zh_direction"] for item in options], ["simplified", "traditional"])

    def test_season_worker_default_and_selected_order_reach_both_scripts(self):
        for order in (None, "after_parent"):
            with self.subTest(order=order):
                build = Mock(return_value=(None, ExportReport()))
                ep = SimpleNamespace(id="E1", document=Document("E1"), build_export=build,
                                     validate=lambda: SimpleNamespace(ok=True))
                project = SimpleNamespace(episode=lambda eid: ep)
                items = [("E1", "a.ass", "simplified"), ("E1", "b.ass", "traditional")]
                kw = {} if order is None else {"annotation_order": order}
                job = SeasonExportJob(project, items, None, lambda ass, runner=None: FontCheck(), **kw)
                job.run()
                self.assertIsNone(job.error)
                self.assertTrue(all(not result.error for result in job.results))
                options = [call.kwargs for call in build.call_args_list]
                self.assertEqual([item["annotation_order"] for item in options], [order or "end"] * 2)
                self.assertEqual([item["zh_direction"] for item in options], ["simplified", "traditional"])


if __name__ == "__main__":
    unittest.main()
