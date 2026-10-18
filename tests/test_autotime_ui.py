"""Offscreen auto-timing flows over scratch project copies and synthetic audio."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QLineEdit, QMessageBox, QPushButton

from subkit import autotime as at
from subkit.core.autotime_doc import Settings, plan
from subkit_app import mainwindow as mw
from subkit_app import help as help_ui
from subkit_app.views import autotime as ui
import tests.test_passages_ui as fixtures


def wait_preview(dialog):
    deadline = time.monotonic() + 5
    while dialog._debounce.isActive() or dialog._jobs:
        if time.monotonic() > deadline:
            raise AssertionError("timing preview did not finish")
        QTest.qWait(10)


class AutoTimeUI(unittest.TestCase):
    setUpClass = classmethod(fixtures.PassageUI.setUpClass.__func__)
    _dispose = fixtures.PassageUI._dispose
    select = fixtures.PassageUI.select

    def setUp(self):
        fixtures.PassageUI.setUp(self)
        self.doc.groups[0].kind = "dialogue"
        self.episode.video = self.path / "fake-video.mkv"
        self.before = copy.deepcopy(self.doc.members)
        self.map = at.TimingMap([at.Segment(0, 12000, 500, confidence="high")],
                               windows=[at.WindowLag(0, 6000, 500, .8, "high")])
        self.window.statusBar().showMessage("原状态")

    def wait_plan(self, dialog):
        wait_preview(dialog)

    def one_click(self, inspect):
        with patch.object(at, "estimate", return_value=self.map), \
                patch.object(ui.AutoTimeResultDialog, "exec", inspect):
            self.window._autotime()

    def test_one_click_and_ctrl_z_exact_restore(self):
        def inspect(dialog):
            self.assertEqual(self.doc.members["m_jp"].start, self.before["m_jp"].start + 500)
            self.assertEqual(len(self.session.editor.history.done), 1)
            self.assertEqual(self.session.editor.history.done[-1].label, "自动调轴")
            self.assertIsNone(QApplication.overrideCursor())
            self.assertEqual(self.window.statusBar().currentMessage(), "原状态")
            dialog.show()
            dialog.activateWindow()
            dialog.setFocus()
            self.app.processEvents()
            QTimer.singleShot(20, lambda: (QTest.keyClick(dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier), dialog.reject()))
            return QDialog.exec(dialog)
        self.one_click(inspect)
        self.assertEqual(self.doc.members, self.before)
        self.assertIsNone(QApplication.overrideCursor())

    def test_counts_quality_and_low_confidence(self):
        self.doc.groups[0].state = "confirmed"
        self.map = at.TimingMap([at.Segment(0, 7000, 500, confidence="high"),
                                at.Segment(7000, 12000, 1000, confidence="low")])
        def inspect(dialog):
            labels = [label.text() for label in dialog.findChildren(QLabel)]
            self.assertIn("1 组你确认过或手动调过时间", labels)
            self.assertIn("1 组所在的片段把握小", labels)
            self.assertIn(ui.BAD_QUALITY, dialog.quality_label.text())
            self.assertIn(ui.BAD_QUALITY, dialog.quality_label.text())
            return QDialog.DialogCode.Rejected
        self.one_click(inspect)
        self.assertEqual(self.doc.members, self.before)

    def test_aligned_and_no_confident_map_messages(self):
        for offset, confidence, message in ((0, "high", "字幕和视频声音已经对得上（最多差 0.00 秒），没有改动。"),
                                            (500, "low", "没有找到把握大的时间差，没有改动。")):
            self.map = at.TimingMap([at.Segment(0, 12000, offset, confidence=confidence)])
            def inspect(dialog):
                self.assertEqual(dialog.summary.text(), message)
                self.assertTrue(dialog.details_button.isEnabled())
                return QDialog.DialogCode.Rejected
            self.one_click(inspect)
        self.assertFalse(self.session.editor.history.done)

    def test_missing_video_and_loading_messages(self):
        with patch.object(QMessageBox, "information") as info:
            self.episode.video = None
            self.window._autotime()
            self.assertEqual(info.call_args.args[2], "这一集没有视频，不能按音频调轴。")
            self.episode.video = self.path / "fake.mkv"
            with patch.object(self.window.wave, "audio_features", return_value=None):
                self.window._autotime()
            self.assertEqual(info.call_args.args[2], "音频还在读取，读完后再试。")

    def test_details_undo_recompute_edit_offset_and_apply(self):
        def details(dialog):
            self.assertEqual(self.doc.members, self.before)
            self.assertEqual(dialog.lead_in.value(), .27)
            self.assertEqual(dialog.lead_out.value(), 0)
            self.assertEqual(dialog.snap.currentData(), "off")
            dialog.method.setCurrentIndex(0)
            self.wait_plan(dialog)
            dialog.segments.item(0, 2).setText("1.250")
            self.wait_plan(dialog)
            self.assertEqual(dialog.current_plan.changes[0].members[0].start, self.before["m_jp"].start + 1250)
            dialog.apply()
            self.assertEqual(self.doc.members["m_jp"].start, self.before["m_jp"].start + 1250)
            self.assertEqual(len(self.session.editor.history.done), 1)
            return QDialog.DialogCode.Accepted
        def result(dialog):
            with patch.object(ui.AutoTimeDetailsDialog, "exec", details):
                dialog.details()
            return QDialog.DialogCode.Accepted
        self.one_click(result)
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members, self.before)

    def test_cancelling_details_restores_the_one_click_result(self):
        adjusted = {}
        def details(dialog):
            self.assertEqual(self.doc.members, self.before)
            return QDialog.DialogCode.Rejected
        def result(dialog):
            adjusted.update(copy.deepcopy(self.doc.members))
            with patch.object(ui.AutoTimeDetailsDialog, "exec", details):
                dialog.details()
            return QDialog.DialogCode.Accepted
        self.one_click(result)
        self.assertEqual(self.doc.members, adjusted)
        self.assertNotEqual(self.doc.members, self.before)
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members, self.before)

    def test_result_cannot_undo_another_transaction(self):
        def inspect(dialog):
            self.session.editor.set_member_time("m_jp", start=777)
            dialog.undo()
            self.assertEqual(self.doc.members["m_jp"].start, 777)
            self.assertEqual(len(self.session.editor.history.done), 2)
            return QDialog.DialogCode.Rejected
        self.one_click(inspect)

    def test_cursor_and_status_restored_on_exception(self):
        with patch.object(mw, "plan_autotime", side_effect=RuntimeError("synthetic failure")), \
                patch.object(QMessageBox, "warning") as warning:
            self.window._autotime()
        self.assertIn("synthetic failure", warning.call_args.args[2])
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(self.window.statusBar().currentMessage(), "原状态")
        self.assertEqual(self.doc.members, self.before)

    def test_prompt_current_session_combines_and_stale_guard(self):
        self.window._passage_audio_session = self.session
        passage = dict(id="p", name="测试", category="OP", version="")
        result = Mock(matches=[Mock(start_ms=1000)])
        with patch.object(mw.PassageStore, "list", return_value=[passage]), \
                patch.object(self.window, "_search_passages", return_value=[(passage, result)]), \
                patch.object(at, "estimate", return_value=self.map), patch.object(QMessageBox, "information") as info:
            self.window.wave.featuresLoaded.emit()
            pending = self.window._auto_process_pending
            self.assertEqual(pending[2:], (500, False, ["片头曲《测试》"]))
            self.assertEqual(self.window.statusBar().currentMessage(), "原状态")
            info.assert_not_called()
            self.window.statusBar().clearMessage()
            self.window._passage_audio_session = object()
            with patch.object(mw, "plan_autotime") as planner:
                self.window.wave.featuresLoaded.emit()
                planner.assert_not_called()
            self.assertEqual(self.window.statusBar().currentMessage(), "")

    def queue_prompt(self, *, offset=500, confidence="high", names=True, segmented=False):
        self.window._passage_audio_session = self.session
        passage = dict(id="p", name="测试", category="OP", version="TV")
        result = Mock(matches=[Mock(start_ms=1000)])
        segments = [at.Segment(0, 12000, offset, confidence=confidence)]
        if segmented:
            segments = [at.Segment(0, 6000, offset, confidence=confidence),
                        at.Segment(6000, 12000, offset * 2, confidence=confidence)]
        with patch.object(mw.PassageStore, "list", return_value=[passage] if names else []), \
                patch.object(self.window, "_search_passages", return_value=[(passage, result)]), \
                patch.object(at, "estimate", return_value=at.TimingMap(segments)):
            self.window.wave.featuresLoaded.emit()

    def test_prompt_once_per_opening_and_persisted_mute(self):
        before = (self.project.root / "subkit-project.json").read_bytes()
        calls = []
        def choose(dialog):
            calls.append(dialog)
            dialog.mute.setChecked(True)
            dialog.choose(None)
            return QDialog.DialogCode.Accepted
        with patch.object(ui.EpisodeAutoProcessDialog, "exec", choose):
            self.queue_prompt()
            self.window._try_auto_process()
            self.queue_prompt()
            self.window._try_auto_process()
            self.assertEqual(len(calls), 1)
            self.assertFalse(self.window._auto_process_timer.isActive())
            self.window._auto_process_asked.clear()
            self.settings.sync()
            self.queue_prompt()
            self.window._try_auto_process()
            self.assertEqual(len(calls), 1)
        self.assertEqual((self.project.root / "subkit-project.json").read_bytes(), before)
        key = self.window._auto_process_key()
        mute_key = self.window._auto_process_mute_key(key)
        self.assertTrue(self.settings.value(mute_key, False, type=bool))
        self.assertNotEqual(mute_key, self.window._auto_process_mute_key((key[0], "E02")))
        self.assertNotEqual(mute_key, self.window._auto_process_mute_key(("other-project", key[1])))

    def test_prompt_once_without_mute_after_session_reopening(self):
        with patch.object(ui.EpisodeAutoProcessDialog, "exec", return_value=QDialog.DialogCode.Rejected) as shown:
            self.queue_prompt()
            self.window._try_auto_process()
            self.window.close_session()
            self.window.session = self.session
            self.session._closed = False
            self.queue_prompt()
            self.window._try_auto_process()
            self.assertEqual(shown.call_count, 1)
            self.window._auto_process_asked.clear()
            self.queue_prompt()
            self.window._try_auto_process()
            self.assertEqual(shown.call_count, 2)

    def test_prompt_deferred_while_typing_then_timer_shows(self):
        self.window.show()
        self.window.activateWindow()
        field = QLineEdit(self.window)
        field.show()
        field.setFocus()
        self.app.processEvents()
        self.assertIs(QApplication.focusWidget(), field)
        with patch.object(ui.EpisodeAutoProcessDialog, "exec", return_value=QDialog.DialogCode.Rejected) as shown, \
                patch.object(mw.time, "monotonic", return_value=100.0):
            self.window._last_key_time = 99.0           # typed a second ago
            self.queue_prompt()
            QTest.qWait(300)
            shown.assert_not_called()
            self.window._last_key_time = 96.0           # idle for 4 s with the text box still focused
            QTest.qWait(300)
            shown.assert_called_once()
        field.deleteLater()

    def test_prompt_deferred_while_modal_or_tool_is_open(self):
        for modal in (True, False):
            with self.subTest(modal=modal):
                blocker = QDialog(self.window)
                blocker.setModal(modal)
                if not modal:
                    blocker.setWindowFlag(Qt.WindowType.Tool)
                blocker.show()
                self.app.processEvents()
                with patch.object(ui.EpisodeAutoProcessDialog, "exec", return_value=QDialog.DialogCode.Rejected) as shown:
                    self.queue_prompt()
                    self.window._try_auto_process()
                    shown.assert_not_called()
                    blocker.hide()
                    self.window._try_auto_process()
                    shown.assert_called_once()
                blocker.deleteLater()
                self.app.processEvents()
                self.window._auto_process_asked.clear()

    def test_prompt_deferred_while_playing_transforming_or_tapping(self):
        for state in ("playing", "transform", "tapping"):
            with self.subTest(state=state), \
                    patch.object(ui.EpisodeAutoProcessDialog, "exec", return_value=QDialog.DialogCode.Rejected) as shown:
                self.queue_prompt()
                if state == "playing":
                    self.window.player.paused = False
                elif state == "transform":
                    self.window.watch.transform = object()
                else:
                    self.window.tapper.run = {}
                self.window._try_auto_process()
                shown.assert_not_called()
                self.window.player.paused = True
                self.window.watch.transform = None
                self.window.tapper.run = None
                self.window._try_auto_process()
                shown.assert_called_once()
                self.window._auto_process_asked.clear()

    def test_prompt_dropped_on_switch_or_close(self):
        with patch.object(ui.EpisodeAutoProcessDialog, "exec") as shown:
            self.queue_prompt()
            self.window.session = object()
            self.window._try_auto_process()
            self.assertIsNone(self.window._auto_process_pending)
            self.assertFalse(self.window._auto_process_timer.isActive())
            self.window.session = self.session
            self.queue_prompt()
            self.window.close_session()
            self.assertIsNone(self.window._auto_process_pending)
            self.assertFalse(self.window._auto_process_timer.isActive())
            shown.assert_not_called()

    def test_prompt_buttons_and_flow_order(self):
        for button, calls in (("自动调轴", ["timing", "passages"]),
                              ("套用重复段落…", ["passages"]), ("先不处理", [])):
            with self.subTest(button=button):
                order = []
                def choose(dialog):
                    next(b for b in dialog.findChildren(QPushButton) if b.text() == button).click()
                    return QDialog.DialogCode.Accepted
                with patch.object(ui.EpisodeAutoProcessDialog, "exec", choose), \
                        patch.object(self.window, "_autotime", side_effect=lambda: order.append("timing")), \
                        patch.object(self.window, "_find_passages", side_effect=lambda: order.append("passages")):
                    self.queue_prompt()
                    self.window._try_auto_process()
                self.assertEqual(order, calls)
                self.window._auto_process_asked.clear()

    def test_prompt_relevant_buttons_verbatim_help_and_no_findings(self):
        for offset, confidence, names, buttons in (
                (500, "high", False, {"自动调轴", "先不处理", "帮助"}),
                (0, "high", True, {"套用重复段落…", "先不处理", "帮助"}),
                (500, "high", True, {"自动调轴", "套用重复段落…", "先不处理", "帮助"})):
            def inspect(dialog):
                self.assertEqual(dialog.windowTitle(), "这一集可以自动处理")
                self.assertEqual({b.text() for b in dialog.findChildren(QPushButton)}, buttons)
                self.assertEqual(dialog.mute.text(), "这一集不再询问")
                labels = [label.text() for label in dialog.findChildren(QLabel)]
                self.assertIn("要现在处理吗？处理后都可以 Ctrl+Z 撤销。", labels)
                if offset:
                    self.assertIn("字幕比视频声音早约 1.00 秒（片中有一处前后差得不一样）。", labels)
                if names:
                    self.assertIn("找到重复段落：片头曲《测试》（TV）。", labels)
                self.assertEqual(help_ui.resolve_anchor(self.window, active=dialog), "autotime")
                with patch.object(help_ui, "open_manual") as opener:
                    next(b for b in dialog.findChildren(QPushButton) if b.text() == "帮助").click()
                    opener.assert_called_once_with("autotime")
                return QDialog.DialogCode.Rejected
            with patch.object(ui.EpisodeAutoProcessDialog, "exec", inspect):
                self.queue_prompt(offset=offset, confidence=confidence, names=names, segmented=True)
                self.window._try_auto_process()
            self.window._auto_process_asked.clear()
        with patch.object(ui.EpisodeAutoProcessDialog, "exec") as shown:
            for offset, confidence in ((0, "high"), (149, "high"), (500, "low")):
                self.queue_prompt(offset=offset, confidence=confidence, names=False)
                self.window._try_auto_process()
            shown.assert_not_called()

    def test_prompt_watch_mode_remains_fullscreen_and_pinned(self):
        self.window.player._autohide = False
        self.window.player._hide_timer = QTimer(self.window.player)
        self.window.player._hide_timer.setInterval(2000)
        self.window.player.pause = Mock(side_effect=lambda: setattr(self.window.player, "paused", True))
        for pinned in (True, False):
            self.window.set_watch_mode(True, pinned=pinned)
            def choose(dialog):
                self.assertTrue(dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
                self.assertTrue(self.window.isFullScreen())
                dialog.choose("passages")
                return QDialog.DialogCode.Accepted
            with patch.object(ui.EpisodeAutoProcessDialog, "exec", choose), \
                    patch.object(self.window, "_find_passages") as passages:
                self.queue_prompt()
                self.window._try_auto_process()
                passages.assert_called_once()
            self.assertTrue(self.window.watch_mode)
            self.assertTrue(self.window.isFullScreen())
            self.assertEqual(self.window.wave_panel.pinned(), pinned)
            self.window._auto_process_asked.clear()
        # Drain the existing watch-mode welcome singleShot before destroying its widgets.
        QTest.qWait(350)

    def test_prompt_runs_one_click_result_then_passage_dialog(self):
        passage = dict(id="p", name="测试", category="OP", version="")
        result = Mock(matches=[Mock(start_ms=1000)])
        order = []
        def choose(dialog):
            dialog.choose("autotime")
            return QDialog.DialogCode.Accepted
        def timing_result(dialog):
            order.append("result")
            self.assertEqual(self.doc.members["m_jp"].start, self.before["m_jp"].start + 500)
            self.assertEqual(len(self.session.editor.history.done), 1)
            return QDialog.DialogCode.Accepted
        with patch.object(ui.EpisodeAutoProcessDialog, "exec", choose), \
                patch.object(at, "estimate", return_value=self.map), \
                patch.object(ui.AutoTimeResultDialog, "exec", timing_result), \
                patch.object(mw.PassageStore, "list", return_value=[passage]), \
                patch.object(self.window, "_search_passages", return_value=[(passage, result)]), \
                patch.object(mw, "PassagesDialog") as passages:
            passages.return_value.exec.side_effect = lambda: order.append("passages")
            self.queue_prompt()
            self.window._try_auto_process()
            passages.assert_called_once_with(self.session, [(passage, result)], self.window.player, self.window)
        self.assertEqual(order, ["result", "passages"])
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members, self.before)

    def test_details_low_defaults_breakpoint_audition_help_and_cancel(self):
        mapping = at.TimingMap([at.Segment(0, 7000, 500, confidence="high"),
                               at.Segment(7000, 12000, 1500, confidence="low")])
        with patch.object(at, "estimate", return_value=mapping):
            dialog = ui.AutoTimeDetailsDialog(self.session, self.features, self.window.player, self.settings, self.window)
            try:
                self.wait_plan(dialog)
                self.assertEqual(dialog.group_items["g_1"].checkState(), Qt.CheckState.Checked)
                self.assertEqual(dialog.group_items["g_2"].checkState(), Qt.CheckState.Unchecked)
                dialog.segments.setCurrentCell(1, 0)
                dialog.remove_breakpoint()
                self.wait_plan(dialog)
                self.assertEqual(len(dialog.current_plan.segments), 1)
                self.assertEqual(dialog.current_plan.mapping.map(8000), 8500)
                dialog.groups.setCurrentCell(0, 0)
                with patch.object(self.window.player, "seek") as seek, patch.object(self.window.player, "play") as play:
                    dialog.audition()
                    seek.assert_called_once_with(1.513)
                    play.assert_called_once()
                self.assertEqual(help_ui.resolve_anchor(self.window, active=dialog), "autotime")
                dialog.reject()
                self.assertEqual(self.doc.members, self.before)
                self.assertIsNone(QApplication.overrideCursor())
            finally:
                dialog.deleteLater()

    def test_details_protection_remembered_values_and_preview_failure(self):
        self.doc.groups[0].state = "confirmed"
        with patch.object(at, "estimate", return_value=self.map):
            dialog = ui.AutoTimeDetailsDialog(self.session, self.features, self.window.player, self.settings, self.window)
            try:
                self.wait_plan(dialog)
                self.assertNotIn("g_1", dialog.group_items)
                dialog.protected.setChecked(True)
                self.wait_plan(dialog)
                self.assertIn("g_1", dialog.group_items)
                dialog.lead_in.setValue(.125)
                dialog.lead_out.setValue(.375)
                self.wait_plan(dialog)
                self.assertEqual(dialog.current_plan.settings.lead_in_ms, 125)
                self.assertEqual(dialog.current_plan.settings.lead_out_ms, 375)
                self.assertEqual(float(self.settings.value("autotime/lead_in")), .125)
                with patch.object(ui, "plan", side_effect=ValueError("preview failure")):
                    dialog.recompute()
                    self.wait_plan(dialog)
                self.assertIsNone(dialog.current_plan)
                self.assertFalse(dialog.apply_button.isEnabled())
                self.assertIn("preview failure", dialog.note.text())
                self.assertIsNone(QApplication.overrideCursor())
                self.assertEqual(self.window.statusBar().currentMessage(), "原状态")
                with patch.object(help_ui, "open_manual") as opener:
                    button = next(b for b in dialog.findChildren(ui.QPushButton) if b.text() == "帮助")
                    button.click()
                    opener.assert_called_once_with("autotime")
            finally:
                dialog.deleteLater()

    def test_background_debounce_busy_and_detached_snapshot(self):
        threads, revisions = [], []
        real_plan = ui.plan
        def compute(doc, *args, **kwargs):
            threads.append(threading.get_ident())
            revisions.append(doc.revision)
            self.assertIsNot(doc, self.doc)
            return real_plan(doc, *args, **kwargs)
        with patch.object(at, "estimate", return_value=self.map), patch.object(ui, "plan", side_effect=compute) as planner:
            dialog = ui.AutoTimeDetailsDialog(self.session, self.features, self.window.player, self.settings, self.window)
            try:
                dialog.method.setCurrentIndex(0)
                dialog.lead_in.setValue(.2)
                dialog.lead_out.setValue(.1)
                self.assertEqual(dialog.note.text(), "正在计算…")
                self.assertFalse(dialog.apply_button.isEnabled())
                dialog.apply()
                self.assertEqual(self.doc.members, self.before)
                self.wait_plan(dialog)
                self.assertEqual(planner.call_count, 1)
                self.assertNotEqual(threads, [threading.get_ident()])
                self.assertEqual(dialog.current_plan.settings.lead_in_ms, 200)
                self.assertTrue(dialog.apply_button.isEnabled())
                self.assertIsNone(QApplication.overrideCursor())
                self.assertEqual(self.window.statusBar().currentMessage(), "原状态")
            finally:
                dialog.reject()
                dialog.deleteLater()

    def test_stale_result_and_error_ignored_and_close_cancels(self):
        entered, release = threading.Event(), threading.Event()
        real_plan = ui.plan
        def delayed(doc, *args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError("worker was not released")
            return real_plan(doc, *args, **kwargs)
        with patch.object(at, "estimate", return_value=self.map), patch.object(ui, "plan", side_effect=delayed):
            dialog = ui.AutoTimeDetailsDialog(self.session, self.features, self.window.player, self.settings, self.window)
            try:
                dialog._debounce.stop()
                dialog._start_plan()
                self.assertTrue(entered.wait(1))
                heartbeat = []
                QTimer.singleShot(0, lambda: heartbeat.append(True))
                QTest.qWait(20)
                self.assertEqual(heartbeat, [True])
                generation = dialog._generation
                job = dialog._jobs[generation]
                stale = plan(self.doc, {}, self.features, mapping_override=self.map)
                dialog.method.setCurrentIndex(0)
                self.assertTrue(job._cancel.is_set())
                dialog._plan_finished(generation, stale, "")
                dialog._plan_finished(generation, None, "stale failure")
                self.assertIsNone(dialog.current_plan)
                self.assertEqual(dialog.note.text(), "正在计算…")
                dialog.show()
                dialog.close()
                self.assertFalse(dialog._debounce.isActive())
                release.set()
                self.assertTrue(ui.QThreadPool.globalInstance().waitForDone(3000))
                self.app.processEvents()
                self.assertIsNone(dialog.current_plan)
                self.assertEqual(self.doc.members, self.before)
            finally:
                release.set()
                dialog.reject()
                dialog.deleteLater()

    def test_close_before_debounce_starts_no_worker(self):
        with patch.object(ui, "plan") as planner:
            dialog = ui.AutoTimeDetailsDialog(self.session, self.features, self.window.player, self.settings, self.window)
            dialog.reject()
            QTest.qWait(250)
            planner.assert_not_called()
            self.assertFalse(dialog._jobs)
            dialog.deleteLater()

    def test_low_confidence_can_be_explicitly_chosen(self):
        self.map = at.TimingMap([at.Segment(0, 12000, 500, confidence="low")])
        with patch.object(at, "estimate", return_value=self.map):
            dialog = ui.AutoTimeDetailsDialog(self.session, self.features, self.window.player, self.settings, self.window)
            try:
                self.wait_plan(dialog)
                self.assertTrue(all(item.checkState() == Qt.CheckState.Unchecked for item in dialog.group_items.values()))
                dialog.group_items["g_2"].setCheckState(Qt.CheckState.Checked)
                dialog.apply()
                self.assertEqual(self.doc.members["m_other"].start, 8500)
                self.assertEqual(self.doc.members["m_jp"], self.before["m_jp"])
            finally:
                dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
