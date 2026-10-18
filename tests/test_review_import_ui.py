"""Offscreen review decisions, using crafted answers and disposable soma copies."""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, Qt, Signal
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QWidget

from subkit.core import review_import as review
from subkit.core.workspace import Project
from subkit_app import help as app_help
from subkit_app.session import EpisodeSession
from subkit_app.views import review_answers as ui
from tests import test_review_import as fixtures

answer = fixtures.answer


class SessionDouble(QObject):
    changed = Signal(object)

    def __init__(self, project, episode, fixture):
        super().__init__()
        self.project, self.episode = project, episode
        self.doc, self.editor = fixture.doc, fixture.ed
        self._closed = False
        self.editor.history.listeners.append(lambda *_: self.changed.emit(set()))


class ReviewAnswersUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Review tests require offscreen Qt")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_review_ui_")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        shutil.copytree(ROOT / "test_projects/soma", root / "project", ignore=shutil.ignore_patterns("cache"))
        self.project = Project.open(root / "project")
        self.real_episode = self.project.episode(self.project.episode_ids()[0])
        self.fixture = fixtures.ReviewImport()
        self.fixture.setUp()
        self.doc, self.ed = self.fixture.doc, self.fixture.ed
        self.episode = SimpleNamespace(id="review-ui", dir=self.real_episode.dir, sources={})
        self.session = SessionDouble(self.project, self.episode, self.fixture)
        self.parent = QWidget()
        self.parent.select_group, self.parent.play_group = Mock(), Mock()
        self.dialogs = []
        self.addCleanup(self.dispose)

    def dispose(self):
        for dialog in reversed(self.dialogs):
            dialog.close()
            dialog.deleteLater()
        self.parent.close()
        self.parent.deleteLater()
        self.session.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def dialog(self, items=None, unreadable=()):
        d = ui.ReviewAnswersDialog(self.session, items or [answer()], unreadable, self.parent)
        self.dialogs.append(d)
        return d

    def test_file_and_paste_paths_unreadable_and_no_automatic_edits(self):
        before = copy.deepcopy(self.doc)
        raw = "说明\n```json\n" + json.dumps(vars(answer()), ensure_ascii=False) + '\n```\n{"bad":true}'
        path = Path(self.tmp.name) / "回答.jsonl"
        path.write_text(raw, encoding="utf-8-sig")
        chooser = ui.ImportAnswersDialog(self.parent)
        self.dialogs.append(chooser)
        with patch.object(ui.QFileDialog, "getOpenFileName", return_value=(str(path), "")):
            chooser.file_button.click()
        chooser.accept()
        self.assertEqual(chooser.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(chooser.items, [answer()])
        self.assertEqual(chooser.unreadable, ["说明", '{"bad":true}'])
        pasted = ui.ImportAnswersDialog(self.parent)
        self.dialogs.append(pasted)
        pasted.paste.setPlainText(raw)
        pasted.accept()
        self.assertEqual(pasted.items, chooser.items)
        d = self.dialog(pasted.items, pasted.unreadable)
        self.assertEqual(d.unreadable_button.text(), "没读懂的 2 行")
        self.assertTrue(d.unreadable_box.isHidden())
        d.unreadable_button.click()
        self.assertFalse(d.unreadable_box.isHidden())
        self.assertEqual(d.unreadable_box.toPlainText(), "说明\n{\"bad\":true}")
        self.assertEqual(self.doc, before)
        self.assertFalse((self.episode.dir / "review/answers.json").exists())

    def test_nothing_read_and_read_error_and_file_cancel(self):
        chooser = ui.ImportAnswersDialog(self.parent)
        self.dialogs.append(chooser)
        chooser.paste.setPlainText("说明")
        with patch.object(ui.QMessageBox, "information") as info:
            chooser.accept()
        self.assertEqual(info.call_args.args[2], ui.NOTHING_READ)
        self.assertEqual(chooser.result(), QDialog.DialogCode.Rejected)
        with patch.object(ui.QFileDialog, "getOpenFileName", return_value=("", "")):
            chooser.choose_file()
        self.assertEqual(chooser.paste.toPlainText(), "说明")
        with patch.object(ui.QFileDialog, "getOpenFileName", return_value=("missing-answer-file", "")), \
                patch.object(ui.QMessageBox, "warning") as warning:
            chooser.choose_file()
        self.assertIn("读取文件失败", warning.call_args.args[2])

    def test_filters_counts_and_per_kind_buttons(self):
        items = [answer(kind=k, target="私", old="私" if k != "reading" else "わたし")
                 for k in ui.KIND_NAMES]
        d = self.dialog(items)
        self.assertEqual(d.counts.text(), "6 条，已处理 0 条")
        for kind in ui.KIND_NAMES:
            d.kind.setCurrentIndex(d.kind.findData(kind))
            self.assertEqual(len(d.rows), 1)
            buttons = d.rows[0].buttons
            self.assertIn("不采用", buttons)
            self.assertIn("试听", buttons)
            self.assertEqual("采用" in buttons, kind in ("reading", "text"))
            self.assertEqual("标为已处理" in buttons, kind not in ("reading", "text"))
        d.kind.setCurrentIndex(0)
        d.decide(items[-1], "handled")
        self.assertEqual(d.counts.text(), "6 条，已处理 1 条")
        self.assertEqual(len(d.rows), 5)
        d.pending.setChecked(False)
        self.assertEqual(len(d.rows), 6)

    def test_reading_accept_and_undo_preserve_log_show_current(self):
        d = self.dialog()
        d.rows[0].buttons["采用"].click()
        self.assertEqual(len(self.ed.history.done), 1)
        self.assertEqual(self.fixture.match(answer()).status, "already_done")
        self.assertEqual(len(d.rows), 0)
        d.undo()
        self.assertEqual(self.fixture.match(answer()).status, "ok")
        d.pending.setChecked(False)
        self.assertEqual(d.rows[0].suggestion.status, "ok")
        self.assertEqual(d.rows[0].suggestion.current, "わたし")
        self.assertEqual(d.rows[0].suggestion.decision, "accepted")
        self.assertFalse(d.rows[0].buttons["采用"].isEnabled())
        self.assertEqual(review.read_answers(self.episode.dir)[answer().fingerprint]["decision"], "accepted")
        again = self.dialog()
        self.assertEqual(len(again.rows), 0)

    def test_pinyin_and_text_accept_undo(self):
        self.fixture.annotation("m_zh", 0, 2, "yin2 hang2", kind="pinyin")
        pinyin = answer(lang="zh-Hans", target="银行", old="yin2 hang2", new="yin2 xing2")
        text = answer(kind="text", target="银行", lang="zh-Hans", old="银行", new="银号")
        for item in (pinyin, text):
            before = copy.deepcopy(self.doc)
            d = self.dialog([item])
            d.accept_one(item)
            self.assertEqual(self.fixture.match(item).status, "already_done")
            self.assertEqual(len(self.ed.history.done), 1)
            d.undo()
            self.assertEqual(self.doc.members, before.members)
            self.assertEqual(self.doc.annotations, before.annotations)

    def test_edit_prefilled_ime_commit_cancel_and_accept(self):
        item = answer(kind="text", target="私と私", old="私と私", new="あなた")
        d = self.dialog([item])
        def edit(dialog):
            self.assertEqual(dialog.windowType(), Qt.WindowType.Tool)
            self.assertEqual(dialog.edit.toPlainText(), "あなた")
            dialog.edit.selectAll()
            event = QInputMethodEvent()
            event.setCommitString("输入法修改")
            QCoreApplication.sendEvent(dialog.edit, event)
            return QDialog.DialogCode.Accepted
        with patch.object(ui.EditAnswerDialog, "exec", edit):
            d.edit_accept(item)
        self.assertEqual(self.doc.members["m_1"].text, "输入法修改")
        self.assertEqual(len(self.ed.history.done), 1)
        d.undo()
        other = answer(kind="text", target="银行", lang="zh-Hans", old="银行", new="银号")
        with patch.object(ui.EditAnswerDialog, "exec", return_value=QDialog.DialogCode.Rejected):
            d.edit_accept(other)
        self.assertEqual(self.doc.members["m_zh"].text, "银行软件")

    def test_ambiguous_pick_only_selected_occurrence(self):
        self.doc.annotations["a_1"].text = "わたし"
        d = self.dialog()
        self.assertEqual(d.rows[0].suggestion.status, "ambiguous")
        self.assertFalse(d.rows[0].buttons["采用"].isEnabled())
        d.rows[0].pick.setCurrentIndex(2)
        self.app.processEvents()
        self.assertEqual(d.rows[0].suggestion.location.a, 2)
        d.rows[0].buttons["采用"].click()
        self.assertEqual(sorted(a.text for a in self.doc.annotations.values()), ["わたくし", "わたし"])
        self.assertEqual(len(self.ed.history.done), 1)
        d.undo()
        self.assertEqual([a.text for a in self.doc.annotations.values()], ["わたし", "わたし"])

    def test_statuses_highlighting_and_jump_audition(self):
        items = [answer(group="wrong"), answer(target="不存在"),
                 answer(target="银行", lang="zh-Hans", old="别的", new="银号", kind="text")]
        d = self.dialog(items)
        self.assertEqual([r.suggestion.status for r in d.rows], ["found_by_text", "not_found", "mismatch"])
        self.assertEqual(d.rows[0].status.text(), "按文字找到（组 ID 不对）")
        self.assertIn("现在的写法和意见里的“原来”不一样：银行", d.rows[2].status.text())
        self.assertIn("background-color", d.sentence(d.rows[0].suggestion.location))
        self.assertFalse(d.rows[1].buttons["采用"].isEnabled())
        d.rows[0].buttons["跳到这一组"].click()
        self.parent.select_group.assert_called_with("g_1", seek=True)
        d.rows[0].buttons["试听"].click()
        self.parent.select_group.assert_called_with("g_1", seek=False)
        self.parent.play_group.assert_called_once()

    def test_cross_break_manual_only(self):
        self.fixture.member(r"私\Nです", mid="m_cross")
        item = answer(group="g_2", kind="text", target="私です", old="私です", new="俺です")
        d = self.dialog([item])
        self.assertEqual(d.rows[0].status.text(), ui.CROSS_BREAK)
        self.assertEqual(set(d.rows[0].buttons), {"跳到这一组", "标为已处理", "不采用"})
        d.accept_one(item)
        self.assertEqual(self.doc.members["m_cross"].text, r"私\Nです")
        self.assertEqual(len(self.ed.history.done), 0)

    def test_accept_all_confirmation_cancel_single_undo_and_persist(self):
        items = [answer(), answer(old="あたし", new="わたくし")]
        d = self.dialog(items)
        with patch.object(ui.QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as ask:
            d.all_button.click()
        self.assertEqual(ask.call_args.args[2], "采用 2 条读音意见？")
        self.assertEqual(len(self.ed.history.done), 0)
        with patch.object(ui.QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            d.all_button.click()
        self.assertEqual(len(self.ed.history.done), 1)
        self.assertEqual(len(review.read_answers(self.episode.dir)), 2)
        self.assertEqual(len(d.rows), 0)
        d.undo()
        self.assertEqual(sorted(a.text for a in self.doc.annotations.values()), ["あたし", "わたし"])
        self.assertEqual(len(review.read_answers(self.episode.dir)), 2)

    def test_reject_handled_reimport_help_and_escaping(self):
        item = answer(kind="speaker", target="私と私", reason="<script>unsafe</script>")
        d = self.dialog([item])
        d.rows[0].buttons["不采用"].click()
        reimported = self.dialog([item])
        self.assertEqual(len(reimported.rows), 0)
        reimported.pending.setChecked(False)
        self.assertEqual(reimported.rows[0].status.text(), "不采用　")
        self.assertEqual(app_help._anchor(reimported), "review-answers")
        with patch.object(app_help, "open_manual") as opener:
            next(b for b in reimported.findChildren(ui.QPushButton) if b.text() == "帮助").click()
            opener.assert_called_once_with("review-answers")
        self.assertEqual(len(self.ed.history.done), 0)

    def test_real_soma_session_and_source_preservation(self):
        original = (self.real_episode.dir / "document.json").read_bytes()
        with patch.object(EpisodeSession, "request_preview"):
            session = EpisodeSession(self.project, self.real_episode)
            try:
                member = next(m for m in session.doc.members.values() if m.lang == "ja" and m.role != "comment")
                text = review._shown(member, ui.DisplayRules())[0]
                item = answer(group=member.group, kind="other", target=text, old=text, new="复查")
                d = ui.ReviewAnswersDialog(session, [item], [], self.parent)
                self.dialogs.append(d)
                self.assertEqual(len(d.rows), 1)
                d.decide(item, "handled")
                self.assertEqual(len(d.rows), 0)
                self.assertEqual((self.real_episode.dir / "document.json").read_bytes(), original)
                d.close()
            finally:
                session.close()
                session.deleteLater()

    def test_main_menu_cancel_open_f1_and_session_close(self):
        from subkit_app import mainwindow as mw
        from tests.test_convert_ui import PlayerDouble

        settings = QSettings(str(Path(self.tmp.name) / "settings.ini"), QSettings.Format.IniFormat)
        with patch.object(mw, "MpvWidget", PlayerDouble), patch.object(mw, "QSettings", return_value=settings), \
                patch.object(mw.MainWindow, "_open_last"):
            window = mw.MainWindow()
        try:
            window.session = self.session
            with patch.object(ui.ImportAnswersDialog, "exec", return_value=QDialog.DialogCode.Rejected):
                window.import_review_answers()
            self.assertIsNone(getattr(window, "_review_answers_dialog", None))
            def import_answer(chooser):
                chooser.paste.setPlainText(json.dumps(vars(answer()), ensure_ascii=False))
                chooser.accept()
                return chooser.result()
            with patch.object(ui.ImportAnswersDialog, "exec", import_answer):
                window.import_review_answers()
            d = window._review_answers_dialog
            self.assertTrue(d.isVisible())
            d.activateWindow()
            self.app.processEvents()
            with patch.object(app_help, "open_manual") as opener:
                QTest.keyClick(d, Qt.Key.Key_F1)
                opener.assert_called_once_with("review-answers")
            QTest.keyClick(d, Qt.Key.Key_Escape)
            self.assertFalse(d.isVisible())
            self.assertFalse(d._connected)
            # Switching an episode must dispose of the old review window.
            window.session = None
            window.close_session()
            self.assertIsNone(window._review_answers_dialog)
        finally:
            window.session = None
            window.close()
            window.deleteLater()
            self.app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settings.sync()


if __name__ == "__main__":
    unittest.main()
