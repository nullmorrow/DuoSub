"""Release workers with a real Qt loop, detached snapshots and killable children."""
from __future__ import annotations

import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import test_convert_ui as ui
from tests.test_fontcheck import episode
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from subkit.core.fontcheck import FontCheck, check_fonts
from subkit.proc import CancelledError, KillableRun
from subkit_app import mainwindow as mw
from subkit_app import help as app_help
from subkit_app.export_job import ExportJob
from subkit_app.views.dialogs import ExportDialog


class ExportJobs(unittest.TestCase):
    setUpClass = classmethod(ui.ConversionUI.setUpClass.__func__)
    setUp = ui.ConversionUI.setUp
    _dispose = ui.ConversionUI._dispose

    def accept_export(self, dialog):
        dialog.path.setText(str(self.path / "out.ass"))
        return 1

    def prepare(self, build=None):
        self.episode.validate = Mock(return_value=SimpleNamespace(ok=True, summary=lambda: "通过"))
        self.episode.build_export = build or Mock(return_value=(None, None))
        self.episode.write_export = Mock()

    def when_running(self, action, *, wait_child=False):
        timer = QTimer(self.window)
        timer.setInterval(10)
        def tick():
            job = self.window._export_job
            if job is not None and (not wait_child or job.runner._process is not None):
                timer.stop()
                action(self.window._export_job, self.window._export_progress)
        timer.timeout.connect(tick)
        timer.start()
        self.addCleanup(timer.stop)

    def test_export_worker_snapshot_and_gui_font_dialog(self):
        ep = episode(self.path)
        ep.validate = Mock(return_value=SimpleNamespace(ok=True, summary=lambda: "通过"))
        self.session.episode = ep
        expected, _ = ep.build_export(video_height=1080)
        expected.save(self.path / "direct.ass")
        self.window.act_private.setChecked(False)
        gui = threading.get_ident()
        threads = []
        from subkit.core.workspace import Episode
        original = Episode.build_export
        def build(snapshot, **kw):
            threads.append(threading.get_ident())
            self.assertIsNot(snapshot.document, ep.document)
            self.assertIsNot(snapshot.project, ep.project)
            return original(snapshot, **kw)
        def font_probe(ass, **kw):
            threads.append(threading.get_ident())
            self.assertIsInstance(kw['runner'], KillableRun)
            return FontCheck(ran=True, missing={"Absent": ["ref"]})
        def font_dialog(fc, **kw):
            self.assertEqual(threading.get_ident(), gui)
            self.assertIsNone(self.window._export_job)
            return True
        with patch.object(ExportDialog, "exec", lambda d: self.accept_export(d)), patch.object(Episode, "build_export", build), \
                patch.object(mw, "check_fonts", font_probe), patch.object(self.window, "_show_font_check", side_effect=font_dialog) as shown, \
                patch.object(QMessageBox, "information"):
            self.window.export()
        self.assertEqual(len(threads), 2)
        self.assertTrue(all(t != gui for t in threads))
        shown.assert_called_once()
        self.assertEqual((self.path / "out.ass").read_bytes(), (self.path / "direct.ass").read_bytes())
        self.assertTrue((self.path / "out.export-report.txt").exists())
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(self.window.statusBar().currentMessage(), "")

    def test_two_copy_release_scripts_japanese_and_reports(self):
        from subkit.core.workspace import Project
        from tests.test_pinyin_export import document
        import copy
        project = Project.create(self.path / "both", "两份")
        ep = project.add_episode("E1")
        ep.document = document("这里是軟體")
        japanese = copy.deepcopy(ep.document.members["m_1"])
        japanese.id, japanese.lang, japanese.lane, japanese.style = "m_ja", "ja", "original", "JP"
        ep.document.members[japanese.id] = japanese
        ep.document.groups[0].members.append(japanese.id)
        self.session.episode = ep
        ep.validate = Mock(return_value=SimpleNamespace(ok=True, summary=lambda: "通过"))
        before = copy.deepcopy(ep.settings())
        old = self.path / "out.zh-Hans.ass"
        old.write_bytes(b"keep simplified")
        old_v1 = self.path / "out.zh-Hans.v1.ass"
        old_v1.write_bytes(b"keep v1")
        def choose(d):
            self.assertFalse(d.zh_both.isHidden())
            d.path.setText(str(self.path / "out.zh-Hant.ass"))
            d.zh_both.setChecked(True)
            return 1
        with patch.object(ExportDialog, "exec", choose), patch.object(mw, "check_fonts", return_value=FontCheck()), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No), \
                patch.object(QMessageBox, "information"):
            self.window.export()
        from subkit.ass.document import AssDocument
        bodies = []
        for suffix, expected in (("zh-Hans", "这里是软体"), ("zh-Hant", "這裡是軟體")):
            path = self.path / ("out." + suffix + (".v1.v2" if suffix == "zh-Hans" else "") + ".ass")
            ass = AssDocument.parse(path.read_text(encoding="utf-8-sig"))
            bodies.append([e.text for e in ass.events if e.style == "JP"])
            self.assertTrue(any(expected in e.text for e in ass.events))
            self.assertIn("检查：通过", path.with_suffix(".export-report.txt").read_text(encoding="utf-8"))
        self.assertEqual(bodies[0], bodies[1])
        self.assertTrue(bodies[0])
        self.assertEqual(ep.settings(), before)
        self.assertEqual(old.read_bytes(), b"keep simplified")
        self.assertEqual(old_v1.read_bytes(), b"keep v1")
        self.assertTrue(self.settings.value("export/zh_both", False, type=bool))

    def test_two_copy_protects_second_source_before_first_write(self):
        self.prepare()
        ui.ConversionUI.group(self, "both")
        source = self.path / "out.zh-Hant.ass"
        source.write_bytes(b"original subtitle")
        self.episode.sources = {"source": SimpleNamespace(path=str(source))}
        def choose(d):
            d.path.setText(str(source))
            d.zh_both.setChecked(True)
            return 1
        with patch.object(ExportDialog, "exec", choose), patch.object(mw, "check_fonts", return_value=FontCheck()), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
                patch.object(QMessageBox, "warning") as warning:
            self.window.export()
        self.episode.write_export.assert_not_called()
        self.assertEqual(source.read_bytes(), b"original subtitle")
        self.assertFalse((self.path / "out.zh-Hans.ass").exists())
        self.assertIn("不能覆盖原料字幕", warning.call_args.args[2])

    def test_two_copy_cancel_second_build_writes_neither_copy(self):
        calls = []
        def build(*args, runner, **kw):
            calls.append(kw["zh_direction"])
            if len(calls) == 2:
                self.slow(runner=runner)
            return None, None
        self.prepare(build)
        ui.ConversionUI.group(self, "both")
        self.when_running(lambda job, d: d.reject(), wait_child=True)
        def choose(d):
            self.accept_export(d)
            d.zh_both.setChecked(True)
            return 1
        with patch.object(ExportDialog, "exec", choose), patch.object(mw, "check_fonts", return_value=FontCheck()):
            self.window.export()
        self.assertEqual(calls, ["simplified", "traditional"])
        self.episode.write_export.assert_not_called()
        self.assertFalse(list(self.path.glob("out*.ass")))
        self.assertFalse(list(self.path.glob("out*.export-report.txt")))
        self.assertIsNone(QApplication.overrideCursor())

    @staticmethod
    def slow(*args, runner, **kw):
        runner.run([sys.executable, "-c", "import time; time.sleep(60)"])
        return None, None

    def test_cancel_build_kills_child_and_writes_nothing(self):
        self.prepare(Mock(side_effect=self.slow))
        jobs = []
        def cancel(job, dialog):
            jobs.append(job)
            self.assertFalse(job.autoDelete())
            self.assertEqual(dialog.windowModality(), Qt.WindowModality.WindowModal)
            self.assertEqual(dialog.windowTitle(), "正在导出…")
            dialog.cancel_button.click()
        self.when_running(cancel, wait_child=True)
        start = time.monotonic()
        with patch.object(ExportDialog, "exec", lambda d: self.accept_export(d)), patch.object(mw, "check_fonts") as fonts:
            self.window.export()
        self.assertLess(time.monotonic() - start, 5)
        self.assertTrue(jobs[0].finished.is_set())
        self.assertIsNone(jobs[0].runner._process)
        fonts.assert_not_called()
        self.episode.write_export.assert_not_called()
        self.assertFalse((self.path / "out.ass").exists())
        self.assertFalse((self.path / "out.export-report.txt").exists())
        self.assertEqual(self.window.statusBar().currentMessage(), "已取消导出")

    def test_cancel_font_probe_and_f1(self):
        self.prepare()
        def cancel(job, dialog):
            with patch.object(app_help, "open_manual") as help_call:
                QTest.keyClick(dialog, Qt.Key.Key_F1)
                help_call.assert_called_once_with("fonts")
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.when_running(cancel, wait_child=True)
        with patch.object(mw, "check_fonts", side_effect=self.slow), \
                patch.object(self.window, "_show_font_check") as shown:
            self.window.check_episode_fonts()
        shown.assert_not_called()
        self.assertEqual(self.window.statusBar().currentMessage(), "已取消检查字体")
        self.assertIsNone(QApplication.overrideCursor())

    def test_close_cancels_running_export_before_session_closure(self):
        self.prepare(Mock(side_effect=self.slow))
        jobs = []
        def close(job, dialog):
            jobs.append(job)
            self.window.close()
        self.when_running(close, wait_child=True)
        with patch.object(ExportDialog, "exec", lambda d: self.accept_export(d)), patch.object(mw, "check_fonts"):
            self.window.export()
        self.assertTrue(jobs[0].finished.is_set())
        self.assertTrue(jobs[0].cancelled)
        self.episode.write_export.assert_not_called()
        self.assertIsNone(self.window.session)
        self.assertIsNone(self.window._export_job)

    def test_build_error_restores_status_and_does_not_write(self):
        self.prepare(Mock(side_effect=ValueError("build failed")))
        with patch.object(ExportDialog, "exec", lambda d: self.accept_export(d)), patch.object(QMessageBox, "warning") as warning:
            self.window.export()
        self.assertIn("build failed", warning.call_args.args[2])
        self.assertEqual(self.window.statusBar().currentMessage(), "")
        self.assertIsNone(self.window._export_job)
        self.episode.write_export.assert_not_called()

    def test_cancel_export_during_font_probe_writes_nothing(self):
        self.prepare()
        self.when_running(lambda job, dialog: dialog.reject(), wait_child=True)
        with patch.object(ExportDialog, "exec", lambda d: self.accept_export(d)), \
                patch.object(mw, "check_fonts", side_effect=self.slow):
            self.window.export()
        self.episode.write_export.assert_not_called()
        self.assertFalse((self.path / "out.export-report.txt").exists())
        self.assertEqual(self.window.statusBar().currentMessage(), "已取消导出")

    def test_progress_blocks_transform_shortcut_and_export_f1(self):
        self.prepare(Mock(side_effect=self.slow))
        self.window.watch_mode = True
        self.window.watch.selected = ["g"]
        def cancel(job, dialog):
            with patch.object(self.window, "_start_transform") as transform, \
                    patch.object(app_help, "open_manual") as help_call:
                QTest.keyClick(dialog, Qt.Key.Key_T)
                transform.assert_not_called()
                QTest.keyClick(dialog, Qt.Key.Key_F1)
                help_call.assert_called_once_with("export")
            dialog.reject()
        self.when_running(cancel, wait_child=True)
        with patch.object(ExportDialog, "exec", lambda d: self.accept_export(d)):
            self.window.export()
        self.window.watch_mode = False

    def test_k_axis_prompt_precedes_job_and_reject_starts_no_worker(self):
        member = ui.ConversionUI.group(self, "g", kind="song", text="A")
        self.prepare()
        with patch.object(ExportDialog, "exec", lambda d: self.accept_export(d)), \
                patch.object(mw.kara, "unfinished", return_value=[(member.id, ["未完成"])]), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No), \
                patch.object(mw.QThreadPool.globalInstance(), "start") as start:
            self.window.export()
        start.assert_not_called()
        self.episode.write_export.assert_not_called()

    def test_fontcheck_uses_supplied_runner_and_propagates_cancel(self):
        from tests.test_fontcheck import document
        runner = Mock(spec=KillableRun)
        runner.run.side_effect = CancelledError()
        with self.assertRaises(CancelledError):
            check_fonts(document("A"), runner=runner)
        runner.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
