"""Automatic timing UI with an injected runner: no process, media or models."""
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from PySide6.QtCore import QObject, Signal
from tests import test_karaoke_ui as base
from subkit.core import karaoke
from subkit.kalign.song import SongResult, LineResult


class FakeRunner(QObject):
    progress = Signal(dict)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancel = Mock()
    def start(self, request): self.request = request


class KalignUI(unittest.TestCase):
    setUpClass = classmethod(base.KaraokeUI.setUpClass.__func__)
    dispose = base.KaraokeUI.dispose
    line = base.KaraokeUI.line
    snapshot = base.KaraokeUI.snapshot

    def setUp(self):
        base.KaraokeUI.setUp(self)
        self.project.save = Mock()
        self.episode.video = self.path / "fake.mkv"
        self.episode.data = {"media": {"audio_track": 2}}
        self.window.kalign_runner_factory = FakeRunner
        p = patch("subkit.kalign.available", return_value=(True, []))
        p.start(); self.addCleanup(p.stop)
        self.first = self.line("かな", gid="g1", timed=False)
        self.second = self.line("かに", gid="g2", timed=False)
        for m in (self.first, self.second): m.style = "ED.JP"
        self.window.select_group("g1")

    def start(self, scope="song"):
        self.window._k_action("auto:"+scope)
        self.assertIsNotNone(self.window._kalign_runner)
        return self.window._kalign_runner

    def result(self, runner, *, offset=0, count=0):
        return SongResult([LineResult(l.mid, [(1100+i*400, 1500+i*400, .8, i==0) for i,b in enumerate(l.beats)])
                           for l in runner.request.lines], offset, "habit" if count else "none", count)

    def test_menu_options_persist_and_missing_dependencies_disable_only_alignment(self):
        p = self.window.wave_panel
        p.kauto_menu.aboutToShow.emit()
        self.assertEqual(p.btn_ksuggest.text(), "自动 K 轴 ▾")
        self.assertTrue(p.kauto_scopes["song"].isEnabled())
        self.assertFalse(p.kauto_scopes["selected"].isEnabled())
        p.kauto_translations.trigger()
        p.kauto_overwrite.trigger()
        p.kauto_radios["last_end", "line"].trigger()
        self.assertEqual(self.project.data["settings"]["karaoke_auto"], {"translations": False, "overwrite": True, "last_end": "line"})
        self.assertEqual(self.project.save.call_count, 3)
        with patch("subkit.kalign.available", return_value=(False, ["torch"])):
            p.kauto_menu.aboutToShow.emit()
        self.assertTrue(all(not a.isEnabled() for a in p.kauto_scopes.values()))
        self.assertIn("需要安装 torch", p.kauto_scopes["line"].toolTip())
        self.assertTrue(p.kauto_menu.actions()[-1].isEnabled())
        self.assertEqual(p.btn_ksuggest.help_anchor, "karaoke-auto")

    def test_single_line_is_suggestion_then_adopted_once_with_checks(self):
        before = self.snapshot()
        runner = self.start("line")
        runner.finished.emit(self.result(runner))
        self.assertEqual(before, self.snapshot())
        self.assertIn("check", self.wave.k_suggestions[1][0].flags)
        with patch.object(self.session.editor, "adopt_auto_karaoke", wraps=self.session.editor.adopt_auto_karaoke) as adopt:
            self.window._k_action("adopt")
        adopt.assert_called_once()
        view = karaoke.effective(self.doc, self.first)
        self.assertEqual(view.state, "suggested")
        self.assertTrue(all(b.by == "align" for b in view.beats))
        self.assertIn("check", view.beats[0].flags)

    def test_multi_apply_protects_manual_and_skips_stale(self):
        beats = karaoke.derive(self.doc, self.first)
        for i,b in enumerate(beats): b.t0,b.t1,b.by=1200+i*400,1600+i*400,"tap"
        self.session.editor.set_karaoke(self.first.id, beats)
        runner = self.start()
        runner.finished.emit(self.result(runner))
        report = self.window._last_auto_apply_report
        self.assertEqual((report["updated"],len(report["skipped"])),(1,1))
        runner = self.start()
        self.second.runs[0].text = "かぬ"
        runner.finished.emit(self.result(runner))
        self.assertEqual(self.window._last_auto_apply_report["stale"], [self.second.id])
        self.assertIn("1 句在处理期间改过", self.window.statusBar().currentMessage())

    def test_cancel_and_episode_replacement_discard_late_results(self):
        before = self.snapshot()
        runner = self.start()
        result = self.result(runner)
        self.window._cancel_auto_karaoke()
        runner.cancel.assert_called_once()
        runner.finished.emit(result)
        self.assertEqual(before, self.snapshot())
        runner = self.start()
        session = self.window.session
        self.window.session = object()
        runner.finished.emit(self.result(runner))
        self.window.session = session
        self.assertEqual(before, self.snapshot())

    def test_failure_and_progress_leave_document_unchanged(self):
        before = self.snapshot()
        runner = self.start()
        for stage, label in (("download", "下载模型"), ("model1", "识别（1/2）"), ("align", "对齐")):
            runner.progress.emit({"stage": stage, "done": 0, "total": 1})
            self.assertIn(label, self.window._kalign_dialog.labelText())
        with patch("subkit_app.mainwindow.QMessageBox.warning") as message:
            runner.failed.emit("模型错误")
        self.assertIn("自动 K 轴失败：模型错误", self.window.statusBar().currentMessage())
        message.assert_called_once_with(self.window, "自动 K 轴", "自动 K 轴失败：模型错误")
        self.assertEqual(before,self.snapshot())

    def test_calibration_stored_and_requested_lines_only(self):
        runner = self.start("line")
        runner.finished.emit(self.result(runner, offset=12, count=34))
        self.assertEqual(self.project.data["settings"]["karaoke_auto"]["stored_habit"], 12)
        self.assertIn("拍头提前量：按我调过的拍 +12 ms（34 拍）", self.window.statusBar().currentMessage())
        self.assertFalse(karaoke.tracks_of(self.doc, self.second.id))

    def test_identical_unrequested_twin_is_not_written(self):
        self.second.runs[0].text = "かな"
        self.second.start, self.second.end = self.first.start, self.first.end
        runner = self.start("song")
        self.second.runs[0].text = "かぬ"
        runner.finished.emit(self.result(runner))
        self.assertFalse(karaoke.tracks_of(self.doc, self.second.id))
        self.assertEqual(self.window._last_auto_apply_report["updated"], 1)

    def test_selected_scope_and_session_close_cancel_late_result(self):
        with patch.object(self.window.table, "selected_gids", return_value=["g1", "g2"]):
            self.window.wave_panel.kauto_menu.aboutToShow.emit()
            self.assertTrue(self.window.wave_panel.kauto_scopes["selected"].isEnabled())
            runner = self.start("selected")
        self.assertEqual(self.window._kalign_context["meta"]["requested_ids"], [self.first.id, self.second.id])
        before, result = self.snapshot(), self.result(runner)
        self.window.close_session()
        runner.cancel.assert_called_once()
        runner.finished.emit(result)
        self.assertEqual(before, self.snapshot())

    def test_results_reused_jump_stays_open_and_session_closes(self):
        from PySide6.QtWidgets import QPushButton
        runner = self.start()
        runner.finished.emit(self.result(runner, offset=12, count=34))
        dialog = self.window._kalign_results
        self.assertEqual(dialog.windowTitle(), "自动 K 轴")
        self.assertFalse(dialog.isModal())
        self.assertEqual(dialog.summary.text(), "已更新 2 句，跳过 0 句（已确认或手调过），2 句有拍需要复查")
        self.assertEqual(dialog.offset.text(), "拍头提前量：按我调过的拍 +12 ms（34 拍）")
        self.assertEqual(dialog.rows.topLevelItemCount(), 2)
        row = dialog.rows.topLevelItem(1)
        self.assertEqual([row.text(i) for i in range(3)], ["かに", "更新", "1"])
        dialog.rows.itemDoubleClicked.emit(row, 0)
        self.assertEqual(self.window.current_gid, "g2")
        self.assertTrue(dialog.isVisible())
        next(b for b in dialog.findChildren(QPushButton) if b.text() == "关闭").click()
        self.assertFalse(dialog.isVisible())
        runner = self.start()
        runner.finished.emit(self.result(runner))
        self.assertIs(self.window._kalign_results, dialog)
        self.assertTrue(dialog.isVisible())
        self.window.close_session()
        self.assertIsNone(self.window._kalign_results)
        self.assertFalse(dialog.isVisible())

    def test_result_rows_report_protected_and_stale_reasons(self):
        beats = karaoke.derive(self.doc, self.first)
        for i, b in enumerate(beats): b.t0, b.t1, b.by = 1200+i*400, 1600+i*400, "tap"
        self.session.editor.set_karaoke(self.first.id, beats)
        runner = self.start()
        self.second.runs[0].text = "かぬ"
        runner.finished.emit(self.result(runner))
        dialog = self.window._kalign_results
        self.assertIn("1 句在处理期间改过，没有写入，请重新运行", dialog.summary.text())
        self.assertEqual(dialog.rows.topLevelItem(0).text(1), "已确认或手调过")
        self.assertEqual(dialog.rows.topLevelItem(1).text(1), "处理期间改过")

    def test_custom_offset_label_and_dialog_reuses_value(self):
        from PySide6.QtWidgets import QInputDialog, QDialog
        self.window._auto_option("offset", 20)
        action = self.window.wave_panel.kauto_radios["offset", "custom"]
        self.assertEqual(action.text(), "自定义…（+20 ms）")
        self.assertTrue(action.isChecked())
        def choose(dialog):
            self.assertEqual(dialog.intValue(), 20)
            self.assertEqual((dialog.intMinimum(), dialog.intMaximum()), (-150, 150))
            dialog.setIntValue(-150)
            return QDialog.DialogCode.Accepted
        with patch.object(QInputDialog, "exec", choose):
            action.trigger()
        self.assertEqual(action.text(), "自定义…（-150 ms）")
        with patch.object(QInputDialog, "exec", return_value=QDialog.DialogCode.Rejected):
            action.trigger()
        self.assertEqual(self.window._auto_options()["offset"], -150)

    def test_orange_borders_on_both_rows_suggestions_and_hint_clear(self):
        from PySide6.QtGui import QImage, QPainter, QColor
        translation = self.line("翻译", gid="g1", lane="translation")
        self.window.select_group("g1")
        for lane in ("original", "translation"):
            self.wave.kviews[lane][1].beats[0].flags.append("check")
        self.window.wave_panel.sync_review_hint()
        self.assertIn("橙色框 = 两种识别结果不一致，请试听", self.window.wave_panel.khint.text())
        def paint():
            image = QImage(self.wave.size(), QImage.Format.Format_ARGB32)
            image.fill(QColor("black"))
            painter = QPainter(image)
            try:
                self.wave._paint_beats(painter, self.wave.height())
                self.wave._paint_review_borders(painter)
            finally: painter.end()
            return image
        image = paint()
        for lane in ("original", "translation"):
            beat = self.wave.kviews[lane][1].beats[0]
            x = round(self.wave.x_of((beat.t0 + beat.t1) / 2))
            y = round(self.wave._k_rows()[lane][0])
            self.assertEqual(image.pixelColor(x, y).name(), "#ff9e3d")
            beat.flags.remove("check")
        self.window.wave_panel.sync_review_hint()
        self.assertNotIn("橙色框", self.window.wave_panel.khint.text())
        proposal = copy.deepcopy(self.wave.kviews["translation"][1].beats)
        proposal[0].flags.append("check")
        self.wave.set_suggestions(translation.id, proposal, "syllable")
        image = paint()
        x = round(self.wave.x_of((proposal[0].t0 + proposal[0].t1) / 2))
        y = round(self.wave._k_rows()["translation"][0])
        self.assertEqual(image.pixelColor(x,y).name(), "#ff9e3d")
        self.assertIn("橙色框", self.window.wave_panel.khint.text())
        self.wave.clear_suggestions()
        self.assertNotIn("橙色框", self.window.wave_panel.khint.text())

    def test_redistribution_scopes_explicit_overwrite_and_one_undo(self):
        for gid in ("g1", "g2"):
            self.line("中文", gid=gid, lane="translation")
        self.window.select_group("g1")
        for scope, expected in (("line", 1), ("selected", 2), ("song", 2)):
            before = self.snapshot()
            count = len(self.session.editor.history.done)
            with patch.object(self.window.table, "selected_gids", return_value=["g1", "g2"]):
                self.window.wave_panel.redistribute_menu.aboutToShow.emit()
                self.window.wave_panel.redistribute_actions[scope].trigger()
            self.assertEqual(len(self.session.editor.history.done), count + 1)
            self.assertEqual(self.window.statusBar().currentMessage(), f"已重新分配 {expected} 句的翻译拍（Ctrl+Z 撤销）")
            self.session.editor.history.undo()
            self.assertEqual(before, self.snapshot())

    def test_redistribution_disable_reasons_selection_and_song_boundary(self):
        p = self.window.wave_panel
        self.window._refresh_redistribute_menu()
        self.assertTrue(all(not a.isEnabled() for a in p.redistribute_actions.values()))
        self.assertIn("这一组没有", p.redistribute_actions["line"].toolTip())
        self.line("中文", gid="g1", lane="translation")
        self.window._refresh_redistribute_menu()
        self.assertTrue(p.redistribute_actions["line"].isEnabled())
        self.assertFalse(p.redistribute_actions["selected"].isEnabled())
        self.first.style = "JP"
        self.window._refresh_redistribute_menu()
        self.assertFalse(p.redistribute_actions["song"].isEnabled())
        for a in p.redistribute_actions.values(): self.assertEqual(a.property("help_anchor"), "karaoke")
        self.assertEqual(p.redistribute_menu.help_anchor, "karaoke")

    def test_long_note_menu_defaults_persist_restore_and_request(self):
        p = self.window.wave_panel
        self.assertTrue(p.kauto_radios["continuation_mode", "even"].isChecked())
        self.assertTrue(p.kauto_follow_notes.isChecked())
        actions = p.kauto_menu.actions()
        self.assertEqual(actions.index(p.kauto_continuation.menuAction()),
                         actions.index(p.kauto_offset.menuAction()) + 1)
        self.assertEqual([a.text() for a in p.kauto_continuation.actions()],
                         ["平分（推荐）", "按我调过的比例", "", "翻译跟整个长音走（推荐）"])
        p.kauto_radios["continuation_mode", "habit"].trigger()
        p.kauto_follow_notes.trigger()
        self.assertFalse(p.kauto_radios["continuation_mode", "even"].isChecked())
        self.assertEqual(self.project.data["settings"]["karaoke_auto"],
                         {"continuation_mode": "habit", "translation_follow_notes": False})
        self.assertEqual(self.project.save.call_count, 2)
        p.set_auto_options({})
        self.window._refresh_auto_menu()
        self.assertTrue(p.kauto_radios["continuation_mode", "habit"].isChecked())
        self.assertFalse(p.kauto_follow_notes.isChecked())
        self.window._auto_option("stored_continuation", .62)
        runner = self.start()
        self.assertEqual(runner.request.options["continuation_mode"], "habit")
        self.assertEqual(runner.request.options["stored_continuation"], .62)
        runner.finished.emit(self.result(runner))

    def test_continuation_result_labels_rounding_and_learning(self):
        saved = self.project.data.setdefault("settings", {}).setdefault("karaoke_auto", {})
        for source, ratio, count, text in (
                ("habit", .625, 8, "；长音：按我调过的拍 62%（8 个）"),
                ("stored", .625, 0, "；长音：工程存过的 62%"),
                ("default", .5, 4, "；长音：手调不到 5 个，平分"),
                ("even", .5, 0, "；长音：平分"),
                ("none", None, 0, "")):
            with self.subTest(source=source):
                runner = self.start()
                result = self.result(runner)
                result.continuation_source, result.continuation_ratio, result.continuation_count = source, ratio, count
                runner.finished.emit(result)
                expected = "拍头提前量：不调整 +0 ms" + text
                self.assertEqual(self.window._kalign_results.offset.text(), expected)
                self.assertTrue(self.window.statusBar().currentMessage().endswith(expected))
                self.assertEqual(saved["stored_continuation"], .625)
        self.assertEqual(self.project.save.call_count, 1)

    def test_translation_setting_passed_to_apply_adopt_and_redistribute(self):
        self.line("中文", gid="g1", lane="translation")
        self.window.select_group("g1")
        self.window._auto_option("translation_follow_notes", False)
        ed = self.session.editor
        runner = self.start()
        with patch.object(ed, "apply_auto_karaoke", wraps=ed.apply_auto_karaoke) as apply:
            runner.finished.emit(self.result(runner))
        self.assertIs(apply.call_args.kwargs["follow_notes"], False)
        runner = self.start("line")
        self.window._auto_option("translation_follow_notes", True)
        runner.finished.emit(self.result(runner))
        with patch.object(ed, "adopt_auto_karaoke", wraps=ed.adopt_auto_karaoke) as adopt:
            self.window._k_action("adopt")
        self.assertIs(adopt.call_args.kwargs["follow_notes"], False)
        with patch.object(ed, "redistribute_translation", wraps=ed.redistribute_translation) as distribute:
            self.window._k_redistribute("line")
        self.assertIs(distribute.call_args.kwargs["follow_notes"], True)
        self.window._auto_option("translation_follow_notes", False)
        with patch.object(ed, "redistribute_translation", wraps=ed.redistribute_translation) as distribute:
            self.window._k_redistribute("line")
        self.assertIs(distribute.call_args.kwargs["follow_notes"], False)

    def test_mapping_dialog_receives_project_translation_setting(self):
        from subkit_app.views.lyrics_map import LyricsMapDialog
        from PySide6.QtWidgets import QDialog
        self.line("中文", gid="g1", lane="translation")
        self.window.select_group("g1")
        self.window._auto_option("translation_follow_notes", False)
        seen = []
        def cancel(dialog):
            seen.append(dialog.follow_notes)
            return QDialog.DialogCode.Rejected
        with patch.object(LyricsMapDialog, "exec", cancel):
            self.window._k_action("mapping")
        self.assertEqual(seen, [False])

    def test_clipped_summary_is_multiline_and_clears_on_next_report(self):
        from subkit_app.views.kalign_result import KalignResultDialog
        dialog = KalignResultDialog(self.window)
        self.addCleanup(dialog.deleteLater)
        report = {"updated": 2, "skipped": {}, "lines_with_check": 1, "clipped": 2}
        dialog.set_report(report, [])
        self.assertTrue(dialog.summary.text().endswith("\n2 句放宽时碰到相邻的句，拍已截短（橙色框）"))
        report["clipped"] = 0
        dialog.set_report(report, [])
        self.assertNotIn("截短", dialog.summary.text())

    def test_menu_handles_removed_current_group(self):
        self.window.current_gid = "removed"
        self.window._refresh_auto_menu()
        self.assertFalse(self.window.wave_panel.kauto_scopes["line"].isEnabled())
        self.assertFalse(self.window.wave_panel.kauto_scopes["song"].isEnabled())
