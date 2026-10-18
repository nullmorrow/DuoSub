"""Dialogue analysis wiring offscreen, with fake runner/cache and no models."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QKeySequence, QShortcutEvent

from tests import test_karaoke_ui as base
from tests.test_kalign_ui import FakeRunner
from subkit.audio import ffmpeg_audio_index
from subkit.kalign import cache, dialogue
from subkit_app.dialogue_alignment import habit_samples, records_from_json


class DialogueAlignUI(unittest.TestCase):
    setUpClass = classmethod(base.KaraokeUI.setUpClass.__func__)
    dispose = base.KaraokeUI.dispose
    line = base.KaraokeUI.line
    snapshot = base.KaraokeUI.snapshot

    def setUp(self):
        base.KaraokeUI.setUp(self)
        self.project.save = Mock()
        self.project.episode_ids = Mock(return_value=[])
        self.episode.video = self.path / "fake.mkv"
        self.episode.video.write_bytes(b"fake video identity, never decoded")
        self.episode.data = {"media": {"audio_track": 2}}
        self.window.kalign_runner_factory = FakeRunner
        p = patch("subkit.kalign.available", return_value=(True, []))
        p.start()
        self.addCleanup(p.stop)
        self.first = self.add_line("かな", "g1", 1000, 3000)
        self.second = self.add_line("かに", "g2", 4000, 5000)
        self.window.wave_panel.set_hint_context(self.project)
        self.window.select_group("g1")
        self.wave.hint_lines = True

    def add_line(self, text, gid, start, end):
        member = self.line(text, gid=gid, timed=False)
        member.role = "dialogue"
        member.start, member.end = start, end
        self.doc.group(gid).kind = "dialogue"
        return member

    def rows(self):
        return [dict(mid=m.id, signature=dialogue.spoken_kana(self.doc, m)[1],
                     speech_start=a, speech_end=b, conf=.9, flags=flags)
                for m, a, b, flags in ((self.first, 1300, 2600, []),
                                      (self.second, 4200, 4800, ["disagree"]))]

    def start(self):
        self.window._start_dialogue_align()
        self.assertIsNotNone(self.window._dialign_runner)
        return self.window._dialign_runner

    def finish(self):
        runner = self.start()
        runner.finished.emit(self.rows())
        return self.window._dialign_results

    def test_menu_dependency_video_and_busy_states(self):
        self.window.time_menu.aboutToShow.emit()
        action = self.window.act_dialogue_align
        self.assertEqual(action.text(), "精确对话轴（按台词对齐）…")
        self.assertEqual(action.property("help_anchor"), "dialogue-align")
        self.assertTrue(action.isEnabled())
        with patch("subkit.kalign.available", return_value=(False, ["torch"])):
            self.window.time_menu.aboutToShow.emit()
        self.assertFalse(action.isEnabled())
        self.assertEqual(action.toolTip(), "需要安装 torch、torchaudio、transformers、demucs（见说明「自动 K 轴」）")
        self.start()
        self.window.time_menu.aboutToShow.emit()
        self.assertFalse(action.isEnabled())
        self.window._start_auto_karaoke("line")
        self.assertIsNone(self.window._kalign_runner)
        self.window._cancel_dialogue_align()

    def test_progress_cpu_warning_and_cancel_discard_late_results(self):
        before = self.snapshot()
        runner = self.start()
        self.assertEqual(runner.request.kind, "dialogue")
        self.assertIsNone(runner.request.clip)
        self.assertEqual(len(runner.request.lines), 2)
        for stage, label in (("extract", "读取音频"), ("download", "下载模型"),
                             ("separate", "分离人声"), ("model1", "识别（1/2）"),
                             ("model2", "识别（2/2）"), ("align", "对齐台词（1/2）")):
            runner.progress.emit(dict(stage=stage, done=1, total=2, device="cpu"))
            self.assertIn(label, self.window._dialign_dialog.labelText())
            self.assertIn("没有可用的显卡，CPU 上需要 15–20 分钟", self.window._dialign_dialog.labelText())
        self.window._dialign_dialog.canceled.emit()
        runner.cancel.assert_called_once()
        runner.finished.emit(self.rows())
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.wave.dialogue_records)

    def test_failure_closes_progress_without_editing(self):
        before = self.snapshot()
        runner = self.start()
        with patch("subkit_app.mainwindow.QMessageBox.warning") as warning:
            runner.failed.emit("模型错误")
        warning.assert_called_once_with(self.window, "精确对话轴", "精确对话轴失败：模型错误")
        self.assertIsNone(self.window._dialign_dialog)
        self.assertEqual(before, self.snapshot())

    def test_result_summary_and_enabled_apply_and_close(self):
        before = self.snapshot()
        dialog = self.finish()
        self.assertEqual(dialog.windowTitle(), "精确对话轴")
        self.assertEqual(dialog.help_anchor, "dialogue-align")
        self.assertFalse(dialog.isModal())
        self.assertEqual(dialog.summary.text(), "已对齐 2 句，0 句没有日文或对不上（用能量判断），1 句两种识别不一致")
        self.assertEqual(dialog.offset.text(), "提前 165 ms，延后 250 ms（默认值）")
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(dialog.apply_button.text(), "整集采用…")
        self.assertEqual(before, self.snapshot())
        dialog.close()
        self.window._show_dialogue_results()
        self.assertIs(self.window._dialign_results, dialog)
        self.assertTrue(dialog.isVisible())

    def test_cache_identity_track_and_corrupt_cache(self):
        key = cache.cache_key(self.episode.video, ffmpeg_audio_index(2), None)
        folder = self.path / "kalign" / key
        folder.mkdir(parents=True)
        result = folder / "dialogue.json"
        result.write_text(json.dumps(self.rows(), ensure_ascii=False), encoding="utf-8")
        self.window._load_dialogue_cache()
        self.assertEqual(self.wave.suggested_times(), (1135, 2850))
        self.episode.data["media"]["audio_track"] = 3
        self.window._load_dialogue_cache()
        self.assertFalse(self.wave.dialogue_records)
        self.episode.data["media"]["audio_track"] = 2
        result.write_text("broken", encoding="utf-8")
        self.window._load_dialogue_cache()
        self.assertFalse(self.wave.dialogue_records)

    def test_aligned_hints_legend_and_c_key_adoption_one_undo(self):
        self.finish()
        hints = self.wave._group_hints(self.doc.group("g1"))
        self.assertEqual((hints["start"], hints["end"]), ([1300], [2600]))
        self.assertIn("建议来源：按台词对齐", self.window.wave_panel.legend.text())
        before, undo = self.snapshot(), len(self.session.editor.history.done)
        self.window._typing = Mock(return_value=False)
        shortcut = self.window.action_hub.action("time.adopt")
        QCoreApplication.sendEvent(shortcut, QShortcutEvent(QKeySequence("C"), 0))
        self.assertEqual(self.doc.group_time(self.doc.group("g1")), (1135, 2850))
        self.assertEqual(self.first.extra["time_src"], {"start": "adopted", "end": "adopted"})
        self.assertEqual(len(self.session.editor.history.done), undo + 1)
        self.window.undo()
        self.assertEqual(self.snapshot(), before)
        self.window.redo()
        QCoreApplication.sendEvent(shortcut, QShortcutEvent(QKeySequence("C"), 0))
        self.assertEqual(self.doc.group_time(self.doc.group("g1")), (1135, 2850))

    def test_signature_edit_falls_back_and_reopened_summary_counts_stale(self):
        self.finish()
        self.first.runs[0].text = "かぬ"
        self.wave._refresh_hints()
        self.assertIsNone(self.wave.suggested_times())  # No energy loaded in this test.
        self.assertIn("建议来源：按人声能量", self.window.wave_panel.legend.text())
        self.window._show_dialogue_results()
        self.assertEqual(self.window._dialign_results.stale.text(), "1 句文字改过，需要重新分析")
        self.assertIn("已对齐 1 句，1 句", self.window._dialign_results.summary.text())

    def test_keyframe_arrival_recomputes_end_only_and_current_neighbours(self):
        self.finish()
        self.assertEqual(self.wave.suggested_times(), (1135, 2850))
        self.wave._keyframes_loaded(np.array([1100, 2900]))
        self.assertEqual(self.wave.suggested_times(), (1135, 2900))
        self.first.start, self.first.end = 1000, 4100
        self.window.select_group("g2")
        self.assertEqual(self.wave.suggested_times()[0], 4140)
        self.first.end = 4300
        self.assertEqual(self.wave.suggested_times()[0], 4340)

    def test_fixed_values_and_stored_habit_are_project_specific(self):
        self.project.data["settings"] = {"hint_padding": {"mode": "fixed", "fixed": [100, -100]},
                                         "dialogue_align": {"habit": [200, 300]}}
        self.window.wave_panel.set_hint_context(self.project)
        dialog = self.finish()
        self.assertEqual(self.wave.suggested_times(), (1200, 2500))
        self.assertIn("手动设定", dialog.offset.text())
        panel = self.window.wave_panel
        panel.padding_mode.setCurrentIndex(panel.padding_mode.findData("manual"))
        panel.sync_hint_padding()
        self.assertEqual(self.wave.suggested_times(), (1100, 2900))
        self.assertIn("按我的对齐习惯", panel.legend.text())

    def test_manual_edge_rules_minimum_and_chained_end_exclusion(self):
        for i in range(2, 22):
            self.add_line("かな", f"g{i+1}", 6000 + i*3000, 7000 + i*3000)
        rows = []
        for group in self.doc.groups:
            m = self.doc.members[group.members[0]]
            m.extra["time_src"] = {"start": "manual", "end": "manual"}
            rows.append(dict(mid=m.id, signature=dialogue.spoken_kana(self.doc, m)[1],
                             speech_start=m.start + 120, speech_end=m.end - 180, conf=.9, flags=[]))
        self.wave.set_dialogue_records(records_from_json(rows))
        panel = self.window.wave_panel
        panel.recalibrate_dialogue_padding()
        self.assertEqual(panel._dialogue_habit[:3], (120, 180, "habit"))
        self.assertEqual(self.project.data["settings"]["dialogue_align"]["habit"], [120, 180])
        self.first.extra["time_src"]["start"] = "adopted"
        self.second.start = self.first.end + 40
        edges, manual = habit_samples(self.doc, self.wave.dialogue_records)
        learned = dialogue.learn_habit(edges, manual)
        self.assertEqual(learned[3:], (21, 21))
        self.assertEqual(manual[0]["time_src"]["start"], "other")

    def test_session_close_and_replacement_ignore_late_results(self):
        runner = self.start()
        before, rows = self.snapshot(), self.rows()
        session = self.window.session
        self.window.session = object()
        runner.finished.emit(rows)
        self.window.session = session
        self.assertEqual(before, self.snapshot())
        runner = self.start()
        self.window.close_session()
        runner.cancel.assert_called_once()
        runner.finished.emit(rows)
        self.assertFalse(self.wave.dialogue_records)

    def test_malformed_records_abstain(self):
        row = self.rows()[0]
        for replacement in ({"speech_start": "1300"}, {"speech_end": 1200}, {"flags": "bad"}, {"signature": None}):
            self.assertEqual(records_from_json([dict(row, **replacement)]), {})

    def test_signature_mismatch_uses_existing_energy_result(self):
        from tests.test_waveform_hints import speech_pcm
        self.first.end = 4300
        self.second.start, self.second.end = 6000, 7000
        panel = self.window.wave_panel
        panel.padding_mode.setCurrentIndex(panel.padding_mode.findData("autotime"))
        panel.sync_hint_padding()
        self.wave._loaded(speech_pcm())
        energy = self.wave.suggested_times()
        self.assertIsNotNone(energy)
        self.finish()
        self.assertNotEqual(self.wave.suggested_times(), energy)
        self.first.runs[0].text = "かぬ"
        self.assertEqual(self.wave.suggested_times(), energy)

    def test_pool_other_episode_caches_without_id_collision(self):
        from subkit.model.serialize import save_document
        for i in range(10):
            self.add_line("かな", f"g{i+3}", 6000 + i*3000, 7000 + i*3000)
        records = []
        for group in self.doc.groups:
            m = self.doc.members[group.members[0]]
            m.extra["time_src"] = {"start": "manual", "end": "manual"}
            records.append(dict(mid=m.id, signature=dialogue.spoken_kana(self.doc, m)[1],
                                speech_start=m.start+100, speech_end=m.end-200, conf=.9, flags=[]))
        self.wave.set_dialogue_records(records_from_json(records))
        other = copy.deepcopy(self.doc)
        other.episode = "other"
        folder = self.path / "other"
        folder.mkdir()
        save_document(other, folder / "document.json")
        video = self.path / "other.mkv"
        video.write_bytes(b"other video identity")
        (folder / "episode.json").write_text(json.dumps({"media": {"video": {"path": str(video)}, "audio_track": 2}}), encoding="utf-8")
        directory = self.path / "kalign" / cache.cache_key(video, ffmpeg_audio_index(2), None)
        directory.mkdir(parents=True)
        other_records = [dict(row, speech_start=row["speech_start"]+100, speech_end=row["speech_end"]-100) for row in records]
        (directory / "dialogue.json").write_text(json.dumps(other_records), encoding="utf-8")
        self.project.root = self.path
        self.project.data["episodes"] = [{"id": "other", "path": "other"}]
        self.window.wave_panel.recalibrate_dialogue_padding()
        self.assertEqual(self.window.wave_panel._dialogue_habit, (150, 250, "habit", 24, 24))

    def test_runner_device_protocol_and_windowless_arguments(self):
        from subkit.kalign.request import build_dialogue_request
        from subkit_app.kalign_runner import KalignRunner
        request, _ = build_dialogue_request(self.doc, video=self.episode.video, audio_track=2, cache_dir=self.path)
        runner = KalignRunner(self.window)
        self.addCleanup(runner.deleteLater)
        self.addCleanup(runner._release)
        values = []
        runner.progress.connect(values.append)
        with patch("subkit_app.kalign_runner.QProcess") as process_class:
            process_class.return_value.setCreateProcessArgumentsModifier = None
            runner.start(request)
            process = runner.process
            program, arguments = process.start.call_args.args
            self.assertTrue(program.endswith("pythonw.exe"))
            self.assertIn("-c", arguments)
            self.assertIn("torch.cuda.is_available()", arguments[arguments.index("-c")+1])
            process.readAllStandardOutput.return_value = b'{"dialogue_device":"cpu"}\n{"stage":"align","done":1,"total":2}\n'
            runner._read()
            self.assertEqual(values, [{"stage": "extract", "device": "cpu"}, {"stage": "align", "done": 1, "total": 2}])
            runner._release()

    def apply_dialog(self):
        from subkit_app.views.dialign_apply import DialogueAlignApplyDialog
        self.finish()
        dialog = DialogueAlignApplyDialog(self.doc, self.wave.dialogue_records,
                                          self.wave.keyframes, self.wave.dialogue_padding, self.window)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_episode_proposals_once_current_neighbours_active_padding_keyframes(self):
        from subkit_app.views.dialign_apply import DialogueAlignApplyDialog, suggest_times
        self.first.end = 4300
        self.project.data["settings"] = {"hint_padding": {"mode": "fixed", "fixed": [100, 200]}}
        self.window.wave_panel.set_hint_context(self.project)
        self.finish()
        self.wave._keyframes_loaded(np.array([2900]))
        with patch("subkit_app.views.dialign_apply.suggest_times", wraps=suggest_times) as suggest:
            dialog = DialogueAlignApplyDialog(self.doc, self.wave.dialogue_records,
                self.wave.keyframes, self.wave.dialogue_padding, self.window)
            self.addCleanup(dialog.deleteLater)
            self.assertEqual([r.suggested for r in dialog.rows], [(1200, 2900), (4340, 5000)])
            self.assertEqual(suggest.call_count, 1)
            self.assertEqual(suggest.call_args.kwargs["previous"], "current")
            self.assertEqual(suggest.call_args.args[1], [(1000, 4300), (4000, 5000)])
            changes = dialog.checked_changes()
            self.session.editor.set_group_times(changes, label="精确对话轴（整集）")
            self.assertEqual(self.doc.group_time(self.doc.group("g2")), (4340, 5000))
            self.assertEqual(suggest.call_count, 1)

    def test_apply_manual_edge_grey_selectable_disagreement_and_filter_state(self):
        from PySide6.QtCore import Qt
        self.first.extra["time_src"] = {"start": "manual", "end": "autotime"}
        dialog = self.apply_dialog()
        self.assertEqual(dialog.table.item(0, 2).checkState(), Qt.CheckState.Unchecked)
        self.assertEqual(dialog.table.item(0, 4).checkState(), Qt.CheckState.Checked)
        self.assertTrue(dialog.table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEnabled)
        self.assertEqual(dialog.table.item(0, 3).foreground().color().name(), "#808080")
        self.assertIn("手调过", dialog.table.item(0, 6).text())
        self.assertIn("两种识别不一致", dialog.table.item(1, 6).text())
        dialog.table.item(0, 2).setCheckState(Qt.CheckState.Checked)
        selected = dialog.checked_changes()
        self.assertEqual(dialog.threshold.value(), 100)
        dialog.filter.setChecked(True)
        self.assertFalse(dialog.table.isRowHidden(0))
        self.assertTrue(dialog.table.isRowHidden(1))
        dialog.threshold.setValue(150)
        self.assertTrue(dialog.table.isRowHidden(0))  # Strictly greater, not >=.
        self.assertEqual(dialog.checked_changes(), selected)
        dialog.filter.setChecked(False)
        self.assertFalse(dialog.table.isRowHidden(1))

    def test_apply_manual_legacy_confirmed_and_explicit_adopted_rules(self):
        from PySide6.QtCore import Qt
        self.first.extra["time_manual"] = True
        self.doc.group("g2").state = "confirmed"
        self.second.extra["time_src"] = {"start": "adopted"}
        dialog = self.apply_dialog()
        self.assertEqual(dialog.rows[0].manual, (True, True))
        self.assertEqual(dialog.rows[1].manual, (False, True))
        self.assertEqual(dialog.table.item(1, 2).checkState(), Qt.CheckState.Checked)

    def test_apply_stale_hidden_song_missing_and_collapsed_rows_not_applicable(self):
        from subkit_app.views.dialign_apply import episode_suggestions
        self.finish()
        self.first.runs[0].text = "かぬ"
        rows, stale = episode_suggestions(self.doc, self.wave.dialogue_records, [], (165, 250))
        self.assertEqual([r.gid for r in rows if r.suggested is not None], ["g2"])
        self.assertEqual(stale, 1)
        self.assertEqual(rows[0].suggested, None)
        self.assertIn("文字改过", rows[0].flags)
        from PySide6.QtCore import Qt
        from subkit_app.views.dialign_apply import DialogueAlignApplyDialog
        dialog = DialogueAlignApplyDialog(self.doc, self.wave.dialogue_records, [], (165, 250), self.window)
        self.addCleanup(dialog.deleteLater)
        self.assertIn("文字改过", dialog.table.item(0, 6).text())
        self.assertFalse(dialog.table.item(0, 2).flags() & Qt.ItemFlag.ItemIsUserCheckable)
        self.assertEqual([gid for gid, _, _ in dialog.checked_changes()], ["g2"])
        self.doc.group("g2").flags.append("hidden")
        rows, _ = episode_suggestions(self.doc, self.wave.dialogue_records, [], (165, 250))
        self.assertFalse([r for r in rows if r.suggested is not None])
        self.doc.group("g2").flags.clear()
        self.doc.group("g2").kind = "song"
        rows, _ = episode_suggestions(self.doc, self.wave.dialogue_records, [], (165, 250))
        self.assertFalse([r for r in rows if r.suggested is not None])
        self.doc.group("g2").kind = "dialogue"
        self.first.end = 7000
        rows, _ = episode_suggestions(self.doc, self.wave.dialogue_records, [], (165, 250))
        self.assertFalse([r for r in rows if r.suggested is not None])  # Previous current end makes the second span invalid.
        rows, _ = episode_suggestions(self.doc, {}, [], (165, 250))
        self.assertFalse([r for r in rows if r.suggested is not None])

    def test_apply_wiring_checked_edges_one_undo_redo_and_time_sources(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog
        from subkit_app.views.dialign_apply import DialogueAlignApplyDialog
        self.first.extra["time_src"] = {"start": "manual", "end": "autotime"}
        result = self.finish()
        before, undo = self.snapshot(), len(self.session.editor.history.done)
        def accept(dialog):
            self.assertEqual(dialog.help_anchor, "dialogue-align")
            dialog.table.item(1, 4).setCheckState(Qt.CheckState.Unchecked)
            return QDialog.DialogCode.Accepted
        with patch.object(DialogueAlignApplyDialog, "exec", accept):
            result.apply_button.click()
        self.assertEqual(self.doc.group_time(self.doc.group("g1")), (1000, 2850))
        self.assertEqual(self.doc.group_time(self.doc.group("g2")), (4035, 5000))
        self.assertEqual(self.first.extra["time_src"], {"start": "manual", "end": "adopted"})
        self.assertEqual(self.second.extra["time_src"], {"start": "adopted"})
        self.assertEqual(len(self.session.editor.history.done), undo + 1)
        self.assertEqual(self.session.editor.history.done[-1].label, "精确对话轴（整集）")
        after = self.snapshot()
        self.window.undo()
        self.assertEqual(self.snapshot(), before)
        self.window.redo()
        self.assertEqual(self.snapshot(), after)

    def test_apply_cancel_empty_selection_and_f1_do_not_edit(self):
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QPushButton
        from subkit_app.views.dialign_apply import DialogueAlignApplyDialog
        dialog = self.apply_dialog()
        before, undo = self.snapshot(), len(self.session.editor.history.done)
        with patch("subkit_app.help.open_manual") as opener:
            next(b for b in dialog.findChildren(QPushButton) if b.text() == "帮助").click()
            opener.assert_called_once_with("dialogue-align")
            opener.reset_mock()
            def cancel():
                QTest.keyClick(dialog, Qt.Key.Key_F1)
                QTest.keyClick(dialog, Qt.Key.Key_Escape)
            QTimer.singleShot(0, cancel)
            dialog.exec()
            opener.assert_called_once_with("dialogue-align")
        self.assertEqual(self.snapshot(), before)
        with patch.object(DialogueAlignApplyDialog, "exec", return_value=DialogueAlignApplyDialog.DialogCode.Rejected):
            self.window._apply_dialogue_episode()
        def accept_empty(dialog):
            for i in range(len(dialog.rows)):
                for col in (2, 4):
                    dialog.table.item(i, col).setCheckState(Qt.CheckState.Unchecked)
            return dialog.DialogCode.Accepted
        with patch.object(DialogueAlignApplyDialog, "exec", accept_empty):
            self.window._apply_dialogue_episode()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.session.editor.history.done), undo)


if __name__ == "__main__":
    unittest.main()
