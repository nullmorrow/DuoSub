"""Offscreen batch wiring, worker lifetime and report actions; no real media or Explorer."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from tests import test_convert_ui as conversion_ui
from subkit.core.batch import BatchResult
from subkit.core.workspace import Project
from subkit_app.session import EpisodeSession
from subkit_app import help as app_help
from subkit_app.views import batch


class BatchUI(conversion_ui.ConversionUI):
    def setUp(self):
        super().setUp()
        self.project = Project.create(self.path / "project", "Open project", translation="zh-Hans")
        self.window.project = self.project
        self.session.project = self.project
        self.window._update_batch_actions()

    def wait_for(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertTrue(condition(), "后台任务没有按时结束")

    def batch_dialog(self, mode="5.1"):
        dialog = batch.BatchDialog(mode, self.settings, lambda: (431, -23), self.project.root,
                                   self.window._prepare_batch, self.window._reload_batch_project, self.window)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def season(self):
        root = self.path / "season"
        root.mkdir()
        for name in ("S01E001.mkv", "S01E001.ass", "S01E001.ja.srt", "S01E002.ass",
                     "S01E003.mkv", "S01E003.srt", "S01E004.mkv", "S01E004.ass",
                     "S01E005.mkv", "S01E005.a.ass", "S01E005.b.ass", "other.zip"):
            (root / name).write_bytes(b"")
        return root

    def ready(self, dialog):
        root = self.season()
        dialog.season.setText(str(root))
        self.wait_for(lambda: not dialog._jobs)
        self.assertTrue(dialog.start_button.isEnabled())
        return root

    def test_song_table_defaults_choices_forwarded_and_folder_reset(self):
        from tests.test_batch_core import BatchCoreTests
        d = self.batch_dialog("5.2")
        root = self.season()
        # Use the fixture constructor without creating another test's project.
        fixture = BatchCoreTests("test_plain_song_roles_and_line_count")
        doc = fixture.song_doc(ruby=True)
        (root / "S01E001.ass").write_text(doc.serialize(), encoding="utf-8")
        d.season.setText(str(root))
        self.wait_for(lambda: not d._jobs)
        self.assertEqual(d.song_table.rowCount(), 2)
        self.assertEqual(d.options().song_choices, {"OP": "transcribe", "ED": "transcribe_vertical"})
        vertical = d.song_table.cellWidget(1, 4)
        self.assertEqual([vertical.itemData(i) for i in range(vertical.count())],
                         ["transcribe", "keep", "transcribe_vertical"])
        self.assertEqual(vertical.currentText(), "转写为竖排并加振假名（右侧）")
        vertical.setCurrentIndex(vertical.findData("keep"))
        combo = d.song_table.cellWidget(0, 4)
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["转写并加振假名", "原样保留"])
        combo.setCurrentIndex(combo.findData("keep"))
        report = root / "report.txt"
        report.write_text("完成", encoding="utf-8")
        with patch.object(batch, "run_season", return_value=BatchResult([], report, self.project.root)) as run:
            d.start()
            self.wait_for(lambda: not d._jobs)
        self.assertEqual(run.call_args.args[2].song_choices, {"OP": "keep", "ED": "keep"})
        d.report_dialog.accept()
        d.scan()
        self.wait_for(lambda: not d._jobs)
        self.assertEqual(d.options().song_choices, {"OP": "transcribe", "ED": "transcribe_vertical"})
        self.assertFalse(self.settings.contains("batch/song_choices"))

    def test_interjection_checkbox_default_persistence_and_running_state(self):
        for mode in ("5.1", "5.2"):
            self.settings.remove("batch/hide_unpaired_interjections")
            d = self.batch_dialog(mode)
            self.assertEqual(d.hide_interjections.text(), "没有中文对照的叫喊和语气词不显示")
            self.assertTrue(d.hide_interjections.isChecked())
            self.assertTrue(d.options().hide_unpaired_interjections)
            d.hide_interjections.setChecked(False)
            self.settings.sync()
            other = self.batch_dialog(mode)
            self.assertFalse(other.hide_interjections.isChecked())
            self.assertFalse(other.options().hide_unpaired_interjections)
            other._set_running(True)
            self.assertFalse(other.hide_interjections.isEnabled())
            other._set_running(False)
            self.assertTrue(other.hide_interjections.isEnabled())
            other.hide_interjections.setChecked(True)
            self.assertTrue(self.batch_dialog(mode).options().hide_unpaired_interjections)

    def test_private_menu_modes_and_single_dialog(self):
        w = self.window
        for enabled in (False, True):
            w.act_private.setChecked(enabled)
            self.assertTrue(all(a.isVisible() == enabled for a in w.batch_actions.values()))
        for mode in ("5.1", "5.2"):
            with patch.object(batch.BatchDialog, "show"):
                w.batch_actions[mode].trigger()
                d = w._batch_dialog
                self.assertEqual(d.mode, mode)
                self.assertTrue(d.isModal())
                self.assertEqual(d.options().lead_in_ms, w.wave.lead_in_ms)
                with patch.object(d, "raise_") as raised:
                    w._batch("5.2")
                    self.assertIs(w._batch_dialog, d)
                    raised.assert_called_once()
                d.reject()
                self.assertIsNone(w._batch_dialog)
        w.act_private.setChecked(False)
        w._batch("5.1")
        self.assertIsNone(w._batch_dialog)

    def test_menu_placement_and_requires_open_project(self):
        w = self.window
        labels = [a.text() for a in w.ep_menu.actions()]
        i = labels.index("添加一集…")
        self.assertEqual(labels[i+1:i+3], list(batch.MODE_LABELS.values()))
        file_menu = w.menuBar().actions()[0].menu()
        self.assertFalse(any(a in file_menu.actions() for a in w.batch_actions.values()))
        for private in (False, True):
            w.act_private.setChecked(private)
            for project in (None, self.project):
                w.project = project
                w._update_batch_actions()
                for action in w.batch_actions.values():
                    self.assertEqual(action.isVisible(), private)
                    self.assertEqual(action.isEnabled(), private and project is not None)
                if project is None:
                    w._batch("5.1")
                    self.assertIsNone(w._batch_dialog)
        w.project = self.project
        w._episode_job = object()
        w._update_batch_actions()
        self.assertFalse(any(a.isEnabled() for a in w.batch_actions.values()))
        w._episode_job = None

    def test_save_before_run_reload_after_run_keeps_current_session(self):
        w = self.window
        self.session._autosave.stop()
        self.session._debounce.stop()
        self.session.deleteLater()
        ep = self.project.add_episode("E01")
        ep.document = self.doc
        ep.save()
        self.session = EpisodeSession(self.project, ep)
        self.session.request_preview = Mock()
        w.session = self.session
        w.ep_box.blockSignals(True)
        w.ep_box.addItem("E01")
        w.ep_box.blockSignals(False)
        self.session.dirty = True
        session, project, document = self.session, self.project, self.session.doc
        d = self.batch_dialog()
        root = self.ready(d)
        report = root / "report.txt"
        report.write_text("完成", encoding="utf-8")
        def run(season, target, options, **kw):
            self.assertFalse(session.dirty)
            worker = Project.open(target)
            self.assertIsNot(worker, project)
            worker.add_episode("S01E001")
            self.assertEqual(project.episode_ids(), ["E01"])
            return BatchResult([], report, target)
        with patch.object(batch, "run_season", side_effect=run), \
                patch.object(w, "_switch_episode") as switch:
            d.start()
            self.wait_for(lambda: not d._jobs)
            switch.assert_not_called()
        self.assertIs(w.project, project)
        self.assertIs(w.session, session)
        self.assertIs(session.project, project)
        self.assertIs(session.doc, document)
        self.assertEqual(project.episode_ids(), ["E01", "S01E001"])
        self.assertEqual([w.ep_box.itemText(i) for i in range(w.ep_box.count())], ["E01", "S01E001"])
        self.assertEqual(w.ep_box.currentText(), "E01")
        d.report_dialog.accept()

    def test_save_failure_does_not_launch_worker(self):
        d = self.batch_dialog()
        self.ready(d)
        with patch.object(self.project, "save", side_effect=OSError("写入失败")), \
                patch.object(batch, "run_season") as run, patch.object(QMessageBox, "warning") as warning:
            d.start()
            run.assert_not_called()
            warning.assert_called_once()
        self.assertFalse(d._jobs)
        self.assertTrue(d.start_button.isEnabled())

    def test_scan_statuses_columns_defaults_and_stale_result(self):
        d = self.batch_dialog()
        self.ready(d)
        self.assertEqual([d.table.item(i, 4).text() for i in range(5)],
                         ["就绪", "缺视频", "缺中文字幕", "缺日文字幕",
                          "有多个候选：S01E005.a.ass、S01E005.b.ass"])
        self.assertFalse(hasattr(d, "create_project"))
        self.assertFalse(d.adopt_times.isChecked())
        self.assertTrue(d.embed_fonts.isChecked())
        self.assertFalse(hasattr(d, "furigana"))
        self.assertEqual(d.assfonts.text(), str(batch.BatchOptions().assfonts))
        self.assertFalse(d.table.isColumnHidden(3))
        d2 = self.batch_dialog("5.2")
        self.assertTrue(d2.table.isColumnHidden(3))
        self.assertEqual(d2.table.horizontalHeaderItem(2).text(), "双语字幕")
        # A completed earlier scan must not overwrite the current folder's table.
        job = batch._BatchJob(d._serial - 1, self.path, d.options(), scan=True)
        d._jobs[job.serial] = job
        d._completed(job.serial, [], "旧扫描失败")
        d._reap()
        self.assertIn(job.serial, d._jobs)
        job.finished.set()
        d._reap()
        self.assertEqual(d.table.rowCount(), 5)

    def test_options_worker_progress_cancel_and_report(self):
        d = self.batch_dialog()
        root = self.ready(d)
        d.adopt_times.setChecked(True)
        d.assfonts.setText(str(root / "assfonts.exe"))
        release, entered = threading.Event(), threading.Event()
        seen = {}
        report = root / "report.txt"
        report.write_text("报告原文", encoding="utf-8")
        def run(season, target, options, *, progress, cancel_check):
            seen.update(season=season, target=target, options=options, thread=threading.get_ident())
            progress(1, 5, "S01E001", "导出")
            entered.set()
            release.wait(5)
            seen["cancelled"] = cancel_check()
            return BatchResult([], report, None, seen["cancelled"])
        with patch.object(batch, "run_season", side_effect=run):
            d.start()
            job = d._run_job
            self.assertFalse(job.autoDelete())
            try:
                self.wait_for(lambda: entered.is_set() and "第 1/5 集 S01E001：导出" == d.progress.text())
                self.assertNotEqual(seen["thread"], threading.get_ident())
                self.assertFalse(d.start_button.isEnabled())
                d.cancel_button.click()
                self.assertTrue(job.cancelled.is_set())
                self.assertIs(d._run_job, job)
                self.assertFalse(job.finished.is_set())
            finally:
                release.set()
            self.wait_for(lambda: not d._jobs)
        opts = seen["options"]
        self.assertEqual(seen["season"], root)
        self.assertEqual((opts.mode, opts.adopt_suggested_times,
                          opts.lead_in_ms, opts.lead_out_ms, opts.assfonts),
                         ("5.1", True, 431, -23, root / "assfonts.exe"))
        self.assertTrue(seen["cancelled"])
        self.assertTrue(job.finished.is_set())
        self.assertEqual(d.report_dialog.text.toPlainText(), "报告原文")
        self.assertTrue(d.report_dialog.text.isReadOnly())
        self.assertFalse(hasattr(d.report_dialog, "project_button"))
        self.assertEqual(seen["target"], self.project.root)
        self.assertTrue(d.start_button.isEnabled())
        d.report_dialog.accept()
        self.assertEqual(self.settings.value("batch/assfonts"), str(root / "assfonts.exe"))
        d.embed_fonts.setChecked(False)
        self.assertIsNone(d.options().assfonts)

    def test_close_running_asks_waits_and_retains_report(self):
        d = self.batch_dialog()
        root = self.ready(d)
        release = threading.Event()
        report = root / "report.txt"
        report.write_text("已取消", encoding="utf-8")
        def run(*args, **kwargs):
            release.wait(5)
            return BatchResult([], report, root / "工程", True)
        with patch.object(batch, "run_season", side_effect=run):
            d.show()
            d.start()
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
                d.reject()
                self.assertFalse(d._run_job.cancelled.is_set())
            with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                d.close()
                self.assertTrue(d._run_job.cancelled.is_set())
                self.assertTrue(d.isVisible())
            release.set()
            self.wait_for(lambda: not d._jobs)
        self.assertTrue(d.report_dialog.isVisible())
        d.report_dialog.accept()
        self.assertFalse(d.isVisible())

    def test_worker_failure_restores_controls_and_can_retry(self):
        d = self.batch_dialog()
        self.ready(d)
        with patch.object(batch, "run_season", side_effect=RuntimeError("实测失败")), \
                patch.object(QMessageBox, "warning") as warning:
            d.start()
            self.wait_for(lambda: not d._jobs)
            self.assertIn("实测失败", warning.call_args.args[2])
        self.assertTrue(d.start_button.isEnabled())
        self.assertTrue(d.season.isEnabled())
        self.assertTrue(d.cancel_button.isHidden())

    def test_report_folder_and_f1(self):
        w = self.window
        report = self.path / "DuoSub" / "report.txt"
        project = self.path / "工程"
        d = batch.BatchReportDialog(BatchResult([], report, project), "原始报告", w)
        self.addCleanup(d.deleteLater)
        with patch.object(batch.subprocess, "Popen") as popen:
            d.folder_button.click()
            popen.assert_called_once_with(["explorer.exe", str(report.parent.resolve())])
        for dialog in (d, self.batch_dialog()):
            dialog.show()
            dialog.activateWindow()
            self.app.processEvents()
            with patch.object(app_help, "open_manual") as help_call:
                QTest.keyClick(dialog, Qt.Key.Key_F1)
                help_call.assert_called_once_with("batch")
            dialog.reject()
        self.assertIn('id="batch"', app_help.manual_path().read_text(encoding="utf-8"))
