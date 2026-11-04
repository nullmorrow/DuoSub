"""Actual listening widgets with detached fake jobs; no speech models or media decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from subkit_app.listening_job import ListeningJob
from subkit_app.views.ai_transcribe import ListeningDialog
from subkit.listening_progress import read_progress, write_progress


@dataclass
class Request:
    video: str = "never-decoded.mkv"
    audio_track: int | None = 2
    start_ms: int = 806200
    end_ms: int = 807720
    padding_ms: int = 1200
    separate: bool = True
    model: str = "large-v3"
    python: str = ""
    hint: str = ""
    translation: str = ""
    semantic: bool = False


@dataclass
class Candidate:
    text: str
    start_ms: int
    end_ms: int
    sources: list[str]
    audio: str
    warnings: list[str] = field(default_factory=list)


class FakeJob:
    def __init__(self, request, output):
        self.request, self.output = request, output
        self.finished = threading.Event()
        self.cancelled = False
        self.result = self.error = None
        self.message = "正在识别原声…"
        self.cancel = Mock(side_effect=self._cancel)

    def _cancel(self):
        self.cancelled = True

    def progress_text(self):
        return self.message

    def complete(self, result=None, error=None):
        self.result, self.error = result, error
        self.finished.set()


class FakeSemanticJob(FakeJob):
    def __init__(self, request, output, **options):
        super().__init__(request, output)
        self.options = options


class AudioDouble(QObject):
    pass


class PlayerDouble(QObject):
    MediaStatus = QMediaPlayer.MediaStatus
    mediaStatusChanged = Signal(object)
    positionChanged = Signal(int)
    errorOccurred = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.url = QUrl()
        self.setAudioOutput = Mock()
        self.stop = Mock()
        self.play = Mock()
        self.setPosition = Mock()

    def source(self):
        return self.url

    def setSource(self, url):
        self.url = url

    def mediaStatus(self):
        return self.MediaStatus.LoadedMedia


class ListeningWidgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Listening widget tests require QT_QPA_PLATFORM=offscreen")

    def setUp(self):
        self.dialogs = []
        self.pool = SimpleNamespace(start=Mock())
        for name, value in (("QThreadPool.globalInstance", Mock(return_value=self.pool)),
                            ("QMediaPlayer", PlayerDouble), ("QAudioOutput", AudioDouble)):
            p = patch("subkit_app.views.ai_transcribe." + name, value)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.dispose)

    def dispose(self):
        for dialog in self.dialogs:
            if dialog.job is not None:
                dialog.job.complete()
            dialog.reject()
            dialog.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def dialog(self):
        dialog = ListeningDialog(Request(), "要氣球嗎", job_factory=FakeJob)
        self.dialogs.append(dialog)
        dialog.show()
        self.app.processEvents()
        self.assertIsNotNone(dialog.job)
        return dialog

    def result(self, dialog, *, candidates=True):
        output = dialog.job.output
        paths = {}
        for name in ("original", "vocals"):
            path = output / (name + ".wav")
            path.write_bytes(b"fake WAV; never decoded")
            paths[name] = str(path)
        items = [Candidate("風船いかがですか", 806300, 807600, ["原声／large-v3"], paths["original"], ["与前景对白重叠"]),
                 Candidate("風船どうですか", 806150, 807650, ["人声／medium"], paths["vocals"])] if candidates else []
        return SimpleNamespace(candidates=items, clips=paths, window_start_ms=805000,
                               window_end_ms=808920, device="cpu", diagnostics=["候选需试听核对"])

    def finish(self, dialog, *, candidates=True):
        result = self.result(dialog, candidates=candidates)
        dialog.job.complete(result)
        dialog._poll()
        return result

    def test_starts_once_progress_defaults_and_never_auto_accepts(self):
        dialog = self.dialog()
        self.assertEqual(dialog.windowType(), Qt.WindowType.Tool)
        self.assertEqual(dialog.help_anchor, "ai-transcribe")
        self.pool.start.assert_called_once()
        self.assertIs(self.pool.start.call_args.args[0].job, dialog.job)
        self.assertFalse(dialog.adopt_button.isEnabled())
        self.assertEqual(dialog.state_label.text(), "正在听译…")
        self.assertTrue(dialog.progress_bar.isVisible())
        self.assertEqual((dialog.progress_bar.minimum(), dialog.progress_bar.maximum()), (0, 0))
        self.assertFalse(dialog.text.isEnabled())
        self.assertFalse(dialog.candidates.isEnabled())
        self.assertEqual((dialog.padding.minimum(), dialog.padding.maximum()), (0, 5000))
        self.assertTrue(dialog.separate.isChecked())
        self.assertEqual(dialog.hint.text(), "")
        self.assertEqual(dialog.hint.maxLength(), 80)
        self.assertIn("仅填写实际听到的词", dialog.hint.placeholderText())
        dialog._poll()
        self.assertEqual(dialog.status.text(), "正在识别原声…")
        accepted = Mock()
        dialog.accepted.connect(accepted)
        self.finish(dialog)
        self.assertEqual(dialog.candidates.topLevelItemCount(), 2)
        self.assertIn("听译完成", dialog.state_label.text())
        self.assertFalse(dialog.progress_bar.isVisible())
        self.assertTrue(dialog.text.isEnabled())
        self.assertTrue(dialog.candidates.isEnabled())
        self.assertIn("2 个候选", dialog.status.text())
        self.assertEqual(dialog.text.toPlainText(), "風船いかがですか")
        self.assertTrue(dialog.adopt_button.isEnabled())
        self.assertTrue(dialog.isVisible())
        self.assertFalse(any(button.isDefault() or button.autoDefault() for button in dialog.findChildren(QPushButton)))
        QTest.keyClick(dialog.start, Qt.Key.Key_Return)
        accepted.assert_not_called()

    def test_semantic_results_are_separate_filterable_and_keep_provenance_on_selection(self):
        from subkit.listening import Candidate as RealCandidate
        dialog = self.dialog()
        result = self.result(dialog)
        semantic = RealCandidate("風船、どうですか？", dialog.request.start_ms, dialog.request.end_ms,
                                 ["中文语义推测"], result.clips["original"],
                                 ["语义推测，未经音频确认"], kind="semantic")
        result.candidates.append(semantic)
        dialog.job.complete(result)
        dialog._poll()
        self.assertIn("语义推测 1 个", dialog.status.text())
        self.assertEqual(dialog.candidates.topLevelItem(2).text(1), "语义推测（未确认）")
        dialog.kind_filter.setCurrentIndex(dialog.kind_filter.findData("semantic"))
        self.assertTrue(dialog.candidates.topLevelItem(0).isHidden())
        self.assertFalse(dialog.candidates.topLevelItem(2).isHidden())
        self.assertIs(dialog._selected_candidate(), semantic)
        self.assertEqual(dialog.text.toPlainText(), semantic.text)
        self.assertIn("未经音频确认", dialog.warning.text())
        self.assertIs(dialog.selection()[0], semantic)
        dialog.play_candidate()
        self.assertEqual(dialog.player.source(), QUrl.fromLocalFile(result.clips["original"]))
        dialog.kind_filter.setCurrentIndex(0)
        self.assertTrue(all(not dialog.candidates.topLevelItem(i).isHidden() for i in range(3)))

    def test_all_eight_semantic_candidates_are_selectable_with_matching_editable_text(self):
        from subkit.listening_semantics import add_semantic_candidates
        dialog = self.dialog()
        result = self.result(dialog, candidates=False)
        texts = [f"風船の候補{i}" for i in range(8)]
        added = add_semantic_candidates(result, texts, translation="要氣球嗎", start=806200,
                                         end=807720, origin="eight-row UI fixture")
        dialog.job.complete(result)
        dialog._poll()
        accepted = Mock()
        dialog.accepted.connect(accepted)
        dialog.kind_filter.setCurrentIndex(dialog.kind_filter.findData("semantic"))
        self.assertEqual(dialog.candidates.topLevelItemCount(), 8)
        for i, candidate in enumerate(added):
            item = dialog.candidates.topLevelItem(i)
            dialog.candidates.scrollToItem(item)
            dialog.candidates.setCurrentItem(item)
            self.assertFalse(item.isHidden())
            self.assertIs(dialog.selection()[0], candidate)
            self.assertEqual(dialog.text.toPlainText(), texts[i])
            self.assertTrue(dialog.adopt_button.isEnabled())
            self.assertIn("未经音频确认", dialog.warning.text())
        accepted.assert_not_called()

    def test_semantic_option_passes_chinese_separately_only_after_rerun(self):
        dialog = self.dialog()
        first = dialog.job
        self.assertFalse(first.request.semantic)
        dialog.semantic.setChecked(True)
        self.assertFalse(first.request.semantic)
        self.finish(dialog)
        dialog.rerun_button.click()
        self.assertTrue(dialog.job.request.semantic)
        self.assertEqual(dialog.job.request.translation, dialog.cn_text)
        self.assertEqual(dialog.job.request.hint, "")
        self.assertIn("Whisper 只识别音频", dialog.usage_note.text())

    def test_empty_semantic_filter_cannot_adopt_a_hidden_asr_candidate(self):
        dialog = self.dialog()
        self.finish(dialog)
        dialog.kind_filter.setCurrentIndex(dialog.kind_filter.findData("semantic"))
        self.assertIsNone(dialog._selected_candidate())
        self.assertEqual(dialog.text.toPlainText(), "")
        self.assertFalse(dialog.adopt_button.isEnabled())

    def test_close_before_scheduled_start_does_not_launch_after_retirement(self):
        dialog = ListeningDialog(Request(), "译文", job_factory=FakeJob)
        self.dialogs.append(dialog)
        dialog.reject()
        self.app.processEvents()
        self.pool.start.assert_not_called()
        self.assertIsNone(dialog.job)

    def test_elapsed_time_and_actual_stage_survive_stopping_then_reset_on_restart(self):
        with patch("subkit_app.views.ai_transcribe.time.monotonic", return_value=100.):
            dialog = self.dialog()
        dialog.job.message = "正在分离人声与伴奏…"
        with patch("subkit_app.views.ai_transcribe.time.monotonic", return_value=165.):
            dialog._poll()
            self.assertEqual(dialog.elapsed_label.text(), "已用时 1:05")
            self.assertEqual(dialog.status.text(), "正在分离人声与伴奏…")
            dialog.cancel()
            self.assertEqual(dialog.state_label.text(), "正在停止…")
            self.assertTrue(dialog.progress_bar.isVisible())
            self.assertFalse(dialog.text.isEnabled())
            dialog.job.message = "迟到的识别进度"
            dialog._poll()
            self.assertNotIn("迟到", dialog.status.text())
            dialog.job.complete()
            dialog._poll()
        self.assertEqual(dialog.state_label.text(), "听译已停止")
        self.assertFalse(dialog.progress_bar.isVisible())
        with patch("subkit_app.views.ai_transcribe.time.monotonic", return_value=900.):
            dialog._poll()
            self.assertEqual(dialog.elapsed_label.text(), "已用时 1:05")
            dialog.rerun()
            self.assertEqual(dialog.elapsed_label.text(), "已用时 0:00")
            self.assertEqual(dialog.state_label.text(), "正在听译…")

    def test_optional_heard_word_ime_only_used_after_explicit_rerun(self):
        dialog = self.dialog()
        first_job = dialog.job
        accepted = Mock()
        dialog.accepted.connect(accepted)
        event = QInputMethodEvent()
        event.setCommitString(" ふうせん ")
        QApplication.sendEvent(dialog.hint, event)
        self.assertEqual(dialog.hint.text(), " ふうせん ")
        self.assertEqual(first_job.request.hint, "")
        self.pool.start.assert_called_once()
        QTest.keyClick(dialog.hint, Qt.Key.Key_Return)
        self.pool.start.assert_called_once()
        accepted.assert_not_called()
        self.finish(dialog)
        dialog.rerun_button.click()
        self.assertIsNot(dialog.job, first_job)
        self.assertEqual(dialog.job.request.hint, "ふうせん")
        self.assertEqual((dialog.job.request.start_ms, dialog.job.request.end_ms), (806200, 807720))
        self.assertEqual(dialog.hint.text(), " ふうせん ")
        self.assertEqual(self.pool.start.call_count, 2)
        self.assertFalse(dialog.adopt_button.isEnabled())
        accepted.assert_not_called()

    def test_candidate_selection_edit_ime_bounds_and_explicit_adoption(self):
        dialog = self.dialog()
        result = self.finish(dialog)
        dialog.candidates.setCurrentItem(dialog.candidates.topLevelItem(1))
        self.assertEqual(dialog.text.toPlainText(), "風船どうですか")
        dialog.text.clear()
        event = QInputMethodEvent()
        event.setCommitString("風船はいかがですか？")
        QApplication.sendEvent(dialog.text, event)
        dialog.start.setValue(806100)
        dialog.end.setValue(807800)
        self.assertEqual(dialog.selection(), (result.candidates[1], "風船はいかがですか？", 806100, 807800))
        dialog.start.setValue(900000)
        self.assertEqual(dialog.start.value(), result.window_end_ms)
        self.assertFalse(dialog.adopt_button.isEnabled())
        dialog.start.setValue(806100)
        adopted = Mock()
        dialog.accepted.connect(adopted)
        dialog.adopt_button.click()
        adopted.assert_called_once()
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.selection()[1], "風船はいかがですか？")

    def test_no_candidate_allows_manual_correction_without_fake_recognition(self):
        dialog = self.dialog()
        result = self.finish(dialog, candidates=False)
        self.assertIn("没有可靠候选", dialog.status.text())
        self.assertIn("无可靠候选", dialog.state_label.text())
        self.assertFalse(dialog.progress_bar.isVisible())
        self.assertTrue(dialog.original_button.isEnabled())
        self.assertFalse(dialog.adopt_button.isEnabled())
        dialog.text.setPlainText("風船はいかがですか")
        with patch.dict(sys.modules, {"subkit.listening": SimpleNamespace(Candidate=Candidate)}):
            candidate, text, start, end = dialog.selection()
        self.assertEqual(candidate.sources, ["manual"])
        self.assertEqual(candidate.audio, result.clips["original"])
        self.assertEqual((text, start, end), ("風船はいかがですか", 806200, 807720))

    def test_failure_message_and_stopped_job_cannot_adopt(self):
        dialog = self.dialog()
        dialog.job.complete(error=RuntimeError("模型文件不存在"))
        dialog._poll()
        self.assertIn("听译失败：模型文件不存在", dialog.status.text())
        self.assertEqual(dialog.state_label.text(), "听译失败")
        self.assertFalse(dialog.progress_bar.isVisible())
        self.assertFalse(dialog.original_button.isEnabled())
        self.assertFalse(dialog.adopt_button.isEnabled())
        dialog.text.setPlainText("失敗後の手動入力")
        self.assertFalse(dialog.adopt_button.isEnabled())
        self.assertIsNone(dialog.selection())
        dialog.rerun()
        job = dialog.job
        dialog.stop_button.click()
        job.cancel.assert_called_once()
        job.complete(self.result(dialog))
        dialog._poll()
        self.assertIn("听译已停止", dialog.status.text())
        self.assertEqual(dialog.candidates.topLevelItemCount(), 0)
        dialog.text.setPlainText("採用しない")
        self.assertFalse(dialog.adopt_button.isEnabled())

    def test_rerun_waits_for_cancelled_worker_before_replacing_or_cleaning_files(self):
        dialog = self.dialog()
        old = dialog.job
        old_directory = old.output
        pending_file = old_directory / "working.wav"
        pending_file.write_bytes(b"still in use")
        dialog.padding.setValue(3000)
        dialog.model.setCurrentIndex(dialog.model.findData("medium"))
        dialog.separate.setChecked(False)
        dialog.rerun_button.click()
        old.cancel.assert_called_once()
        self.assertIs(dialog.job, old)
        self.assertEqual(dialog.state_label.text(), "正在停止…")
        self.assertTrue(pending_file.exists())
        self.pool.start.assert_called_once()
        old.complete(self.result(dialog))
        dialog._poll()
        self.assertIsNot(dialog.job, old)
        self.assertFalse(old_directory.exists())
        self.assertEqual(self.pool.start.call_count, 2)
        self.assertEqual((dialog.job.request.padding_ms, dialog.job.request.model, dialog.job.request.separate),
                         (3000, "medium", False))
        self.assertEqual(dialog.candidates.topLevelItemCount(), 0)

    def test_close_and_escape_keep_dialog_alive_until_worker_has_stopped(self):
        for close in (lambda d: d.close(), lambda d: QTest.keyClick(d, Qt.Key.Key_Escape)):
            with self.subTest(close=close):
                dialog = self.dialog()
                directory = dialog.job.output
                (directory / "processing.wav").write_bytes(b"in use")
                finished = Mock()
                dialog.finished.connect(finished)
                close(dialog)
                self.assertEqual(dialog.state_label.text(), "正在停止…")
                self.assertTrue(dialog.progress_bar.isVisible())
                self.assertTrue(dialog.isVisible())
                self.assertTrue(directory.exists())
                finished.assert_not_called()
                dialog.job.complete()
                dialog._poll()
                self.assertFalse(dialog.isVisible())
                self.assertFalse(directory.exists())
                finished.assert_called_once_with(QDialog.DialogCode.Rejected)

    def test_invalidate_disables_adoption_and_late_results(self):
        dialog = self.dialog()
        dialog.invalidate("这一组已改动，请重新打开AI听译。")
        dialog.job.cancel.assert_called_once()
        dialog.job.complete(self.result(dialog))
        dialog._poll()
        self.assertIn("这一组已改动", dialog.status.text())
        self.assertEqual(dialog.state_label.text(), "结果已失效")
        self.assertFalse(dialog.progress_bar.isVisible())
        self.assertEqual(dialog.candidates.topLevelItemCount(), 0)
        dialog.text.setPlainText("誤って採用しない")
        dialog.accept()
        self.assertFalse(dialog.adopt_button.isEnabled())
        self.assertFalse(dialog.rerun_button.isEnabled())
        self.assertTrue(dialog.isVisible())

    def test_audition_seeks_correct_variant_interval_stops_and_releases_files(self):
        dialog = self.dialog()
        result = self.finish(dialog)
        output = dialog.job.output
        dialog.candidates.setCurrentItem(dialog.candidates.topLevelItem(1))
        dialog.play_candidate()
        self.assertEqual(dialog.player.source(), QUrl.fromLocalFile(result.clips["vocals"]))
        dialog.player.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.LoadedMedia)
        dialog.player.setPosition.assert_called_once_with(1150)
        dialog.player.play.assert_called_once()
        dialog.player.positionChanged.emit(2650)
        self.assertIsNone(dialog._audio_bounds)
        dialog.play_original()
        dialog.player.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertEqual(dialog.player.setPosition.call_args.args, (0,))
        dialog.reject()
        self.assertTrue(dialog.player.source().isEmpty())
        self.assertFalse(output.exists())

    def test_missing_audio_is_visible_and_does_not_play(self):
        dialog = self.dialog()
        result = self.finish(dialog)
        Path(result.clips["original"]).unlink()
        dialog.play_original()
        self.assertIn("找不到这份试听音频", dialog.warning.text())
        dialog.player.play.assert_not_called()
        dialog.player.errorOccurred.emit(QMediaPlayer.Error.ResourceError, "设备不可用")
        self.assertEqual(dialog.warning.text(), "试听失败：设备不可用")

    def test_cleanup_does_not_follow_unexpected_subdirectory(self):
        dialog = self.dialog()
        self.finish(dialog)
        output = dialog.job.output
        child = output / "unexpected"
        child.mkdir()
        saved = child / "preserve.txt"
        saved.write_text("leave unexpected contents", encoding="utf-8")
        dialog.reject()
        self.assertTrue(saved.is_file())
        self.assertFalse((output / "original.wav").exists())
        # The test owns these exact paths; production deliberately leaves them for inspection.
        saved.unlink()
        child.rmdir()
        output.rmdir()

    def test_pre_adoption_callback_rejection_retains_dialog_audio_and_draft(self):
        dialog = self.dialog()
        self.finish(dialog)
        output = dialog.job.output
        dialog.before_adopt = Mock(return_value=False)
        accepted = Mock()
        dialog.accepted.connect(accepted)
        dialog.adopt_button.click()
        dialog.before_adopt.assert_called_once_with(dialog.selection())
        accepted.assert_not_called()
        self.assertTrue(dialog.isVisible())
        self.assertTrue(output.is_dir())
        self.assertEqual(dialog.text.toPlainText(), "風船いかがですか")
        dialog.before_adopt = Mock(side_effect=ValueError("这一组已修改"))
        dialog.adopt_button.click()
        self.assertIn("补入原文失败：这一组已修改", dialog.status.text())
        self.assertTrue(output.is_dir())
        dialog.before_adopt = Mock(return_value=True)
        dialog.adopt_button.click()
        accepted.assert_called_once()
        self.assertFalse(output.exists())


    def refreshable(self):
        from subkit.listening import Candidate as RealCandidate
        dialog = self.dialog()
        result = self.result(dialog)
        result.candidates = [RealCandidate(**vars(c)) for c in result.candidates]
        result.candidates.append(RealCandidate("風船、いる？", 806200, 807720,
            ["中文语义推测"], result.clips["original"], kind="semantic"))
        result.metadata = {"semantic": {"model": "initial", "round_index": 0}}
        dialog.job.complete(result)
        dialog._poll()
        dialog.semantic_job_factory = FakeSemanticJob
        return dialog, result

    def batch(self, *texts):
        return {"texts": list(texts), "model": "local-test", "round_index": 1, "seed": 7919}

    def test_refresh_appends_distinct_phrasings_preserves_editor_audio_selection_and_provenance(self):
        dialog, result = self.refreshable()
        old_candidates = result.candidates[:]
        dialog.candidates.setCurrentItem(dialog.candidates.topLevelItem(2))
        old_item = dialog.candidates.currentItem()
        dialog.text.setPlainText("私の編集中の文")
        dialog.start.setValue(806300)
        dialog.end.setValue(807600)
        dialog.more_button.click()
        job = dialog.job
        output = job.output
        self.assertEqual(job.options, dict(existing=("風船、いる？",), round_index=1, device="cpu"))
        self.assertEqual(job.request.translation, "要氣球嗎")
        self.assertIs(dialog.result, result)
        self.assertTrue(Path(result.clips["original"]).exists())
        self.assertIn("更多翻译", dialog.state_label.text())
        self.assertFalse(dialog.more_button.isEnabled())
        self.assertFalse(dialog.adopt_button.isEnabled())
        job.complete(self.batch("風船、いる？", "風船、いかがですか？", "風船、どうですか？"))
        dialog._poll()
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(result.candidates[:3], old_candidates)
        self.assertEqual(dialog.candidates.topLevelItemCount(), 5)
        self.assertIs(dialog.candidates.currentItem(), old_item)
        self.assertEqual(dialog.text.toPlainText(), "私の編集中の文")
        self.assertEqual((dialog.start.value(), dialog.end.value()), (806300, 807600))
        self.assertIn("新增 2 条", dialog.status.text())
        self.assertEqual(result.candidates[-1].evidence[0]["generation"]["round_index"], 1)
        self.assertEqual(len(result.metadata["semantic_batches"]), 2)
        self.assertFalse(output.exists())
        self.assertTrue(Path(result.clips["original"]).exists())
        self.assertTrue(dialog.more_button.isEnabled())
        self.assertTrue(dialog.adopt_button.isEnabled())

    def test_refresh_rounds_exclude_all_previous_semantic_results_and_zero_new_is_explicit(self):
        dialog, result = self.refreshable()
        dialog.more_button.click()
        dialog.job.complete(self.batch("風船、いかがですか？"))
        dialog._poll()
        dialog.more_button.click()
        self.assertEqual(dialog.job.options["round_index"], 2)
        self.assertEqual(dialog.job.options["existing"], ("風船、いる？", "風船、いかがですか？"))
        dialog.job.complete(self.batch("風船、いる？"))
        dialog._poll()
        self.assertEqual(len(result.candidates), 4)
        self.assertIn("没有不同的新说法", dialog.status.text())

    def test_refresh_error_or_stop_preserves_adoption_and_retry(self):
        for mode in ("error", "cancel"):
            with self.subTest(mode=mode):
                dialog, result = self.refreshable()
                dialog.text.setPlainText("残す文")
                dialog.more_button.click()
                output = dialog.job.output
                if mode == "cancel":
                    dialog.cancel()
                    self.assertTrue(output.exists())
                    dialog.job.complete()
                else:
                    dialog.job.complete(error=RuntimeError("model busy"))
                dialog._poll()
                self.assertIs(dialog.result, result)
                self.assertEqual(dialog.text.toPlainText(), "残す文")
                self.assertTrue(dialog.adopt_button.isEnabled())
                self.assertTrue(dialog.more_button.isEnabled())
                self.assertFalse(output.exists())
                self.assertTrue(Path(result.clips["original"]).exists())
                dialog.more_button.click()
                self.assertEqual(dialog.job.options["round_index"], 2)

    def test_close_during_refresh_waits_then_cleans_both_audio_and_refresh_directories(self):
        dialog, result = self.refreshable()
        audio_output = Path(result.clips["original"]).parent
        dialog.more_button.click()
        refresh_output = dialog.job.output
        dialog.reject()
        self.assertTrue(audio_output.exists())
        self.assertTrue(refresh_output.exists())
        dialog.job.complete(self.batch("風船、いかがですか？"))
        dialog._poll()
        self.assertFalse(audio_output.exists())
        self.assertFalse(refresh_output.exists())
        self.assertTrue(dialog._retired)
        self.assertEqual(len(result.candidates), 3)

    def test_full_rerun_during_refresh_waits_and_replaces_old_results_once(self):
        dialog, result = self.refreshable()
        old_audio = Path(result.clips["original"]).parent
        dialog.more_button.click()
        refresh_job = dialog.job
        dialog.rerun()
        self.assertIs(dialog.job, refresh_job)
        refresh_job.complete(self.batch("風船、いかがですか？"))
        dialog._poll()
        self.assertIsInstance(dialog.job, FakeJob)
        self.assertNotIsInstance(dialog.job, FakeSemanticJob)
        self.assertIsNone(dialog.result)
        self.assertFalse(old_audio.exists())
        self.assertFalse(refresh_job.output.exists())
        self.assertEqual(len(result.candidates), 3)

    def test_invalidated_refresh_never_publishes_late_suggestions(self):
        dialog, result = self.refreshable()
        dialog.more_button.click()
        dialog.invalidate("译文已修改")
        dialog.job.complete(self.batch("風船、いかがですか？"))
        dialog._poll()
        self.assertEqual(len(result.candidates), 3)
        self.assertFalse(dialog.more_button.isEnabled())
        self.assertFalse(dialog.adopt_button.isEnabled())
        self.assertIn("失效", dialog.state_label.text())

    def test_refresh_does_not_replace_manual_draft_when_filter_has_no_selection(self):
        dialog, result = self.refreshable()
        dialog.candidates.setCurrentItem(None)
        dialog.text.setPlainText("手入力")
        dialog.more_button.click()
        dialog.job.complete(self.batch("風船、いかがですか？"))
        dialog._poll()
        self.assertIsNone(dialog.candidates.currentItem())
        self.assertEqual(dialog.text.toPlainText(), "手入力")
        self.assertEqual(dialog.selection()[0].kind, "manual")


class ListeningWorker(unittest.TestCase):
    def test_atomic_child_progress_superseded_by_parent_and_ignored_after_completion(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            with patch("subkit_app.listening_job.time.time_ns", return_value=10):
                job = ListeningJob(Request(), output)
            with patch("subkit.listening_progress.time.time_ns", return_value=20):
                write_progress(output, "正在加载 Whisper…")
            self.assertEqual(job.progress_text(), "正在加载 Whisper…")
            self.assertFalse((output / "progress.json.tmp").exists())
            with patch("subkit_app.listening_job.time.time_ns", return_value=30):
                job._set_progress("正在整理候选…")
            self.assertEqual(job.progress_text(), "正在整理候选…")
            with patch("subkit.listening_progress.time.time_ns", return_value=40):
                write_progress(output, "迟到的消息")
            job.finished.set()
            self.assertEqual(job.progress_text(), "正在整理候选…")

    def test_missing_malformed_or_oversized_progress_never_breaks_polling(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            self.assertIsNone(read_progress(output, newer_than_ns=0))
            for value in ('{', '[]', json.dumps({"text": "x" * 501, "updated_ns": 10}),
                          json.dumps({"text": "test", "updated_ns": True}),
                          json.dumps({"text": "test", "updated_ns": "10"}),
                          '{"text": "' + 'x' * 5000):
                (output / "progress.json").write_text(value, encoding="utf-8")
                self.assertIsNone(read_progress(output, newer_than_ns=0))
            with patch("subkit.listening_progress.os.replace", side_effect=PermissionError("busy")):
                write_progress(output, "nonfatal")

    def test_success_progress_and_no_worker_qobject_signal_lifetime(self):
        expected = object()
        def generate(request, output, *, runner, progress):
            self.assertIs(request, job.request)
            self.assertIs(runner, job.runner)
            progress("正在分离人声…")
            return expected
        with tempfile.TemporaryDirectory() as root:
            job = ListeningJob(Request(), Path(root))
            with patch.dict(sys.modules, {"subkit.listening": SimpleNamespace(generate=generate)}):
                job.run()
            self.assertTrue(job.finished.is_set())
            self.assertIs(job.result, expected)
            self.assertEqual(job.progress_text(), "正在分离人声…")
            self.assertIsNone(job.error)
            self.assertFalse(job.cancelled)

    def test_cancel_before_start_prevents_generate_and_sets_completion(self):
        generate = Mock()
        with tempfile.TemporaryDirectory() as root:
            job = ListeningJob(Request(), Path(root))
            job.cancel()
            with patch.dict(sys.modules, {"subkit.listening": SimpleNamespace(generate=generate)}):
                job.run()
            generate.assert_not_called()
            self.assertTrue(job.finished.is_set())
            self.assertTrue(job.cancelled)
            self.assertIsNone(job.result)

    def test_cancel_during_generate_discards_result_and_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            for failure in (False, True):
                with self.subTest(failure=failure):
                    job = ListeningJob(Request(), Path(root))
                    def generate(*args, **kwargs):
                        if failure:
                            raise ValueError("bad audio")
                        job.cancel()
                        return object()
                    with patch.dict(sys.modules, {"subkit.listening": SimpleNamespace(generate=generate)}):
                        job.run()
                    self.assertTrue(job.finished.is_set())
                    self.assertIsNone(job.result)
                    self.assertEqual(job.cancelled, not failure)
                    self.assertEqual(str(job.error) if failure else job.error, "bad audio" if failure else None)


if __name__ == "__main__":
    unittest.main()
