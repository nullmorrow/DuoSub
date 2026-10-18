"""Save state and close/switch decisions with real files, offscreen Qt and player doubles."""
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from tests import test_restore_position as fixtures
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox
from subkit.core.workspace import Project
from subkit.model.document import Document
from subkit_app import help as app_help
from subkit_app.mainwindow import APP_NAME, MainWindow
from subkit_app.session import EpisodeSession

ASK_SAVE = MainWindow._ask_save


class SaveStateTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.RestorePosition.setUpClass.__func__)
    setUp = fixtures.RestorePosition.setUp
    dispose = fixtures.RestorePosition.dispose

    def dirty(self):
        s = self.window.session
        s.editor.set_mark_note(['A-0'], '未保存')
        self.assertTrue(s.dirty)
        return s

    def test_title_status_and_autosave_does_not_mean_saved(self):
        w = self.window
        s = w.session
        stamp = (s.episode.dir / 'document.json').stat().st_mtime
        when = time.strftime('%H:%M', time.localtime(stamp))
        self.assertEqual(w.windowTitle(), f'{APP_NAME} — Restore — A')
        self.assertEqual(w.lbl_save_state.text(), f'已保存 {when}')
        self.assertEqual(w.lbl_save_state.toolTip(), '')
        self.dirty()
        self.assertEqual(w.windowTitle(), f'{APP_NAME} — Restore — A ●')
        self.assertEqual(w.lbl_save_state.text(), f'有未保存的改动（上次保存 {when}）')
        original = (s.episode.dir / 'document.json').read_bytes()
        s.autosave()
        self.assertTrue(s.dirty)
        self.assertEqual((s.episode.dir / 'document.json').read_bytes(), original)
        recovery = time.strftime('%H:%M', time.localtime(s.recovery_saved_at))
        self.assertEqual(w.lbl_save_state.toolTip(), f'恢复用副本 {recovery}（程序意外退出时用，不是正式保存）')
        w.save()
        self.assertFalse(s.dirty)
        self.assertFalse(w.windowTitle().endswith(' ●'))
        self.assertEqual(w.lbl_save_state.text(), '已保存 ' + time.strftime('%H:%M', time.localtime(s.last_saved_at)))

    def test_never_saved_and_direct_dirty_changes(self):
        ep = self.project.add_episode('NEW')
        ep.document = Document(episode='NEW')
        w = self.window
        w.close_session()
        w._start_session(ep)
        self.assertEqual(w.lbl_save_state.text(), '还没保存过')
        w.session.dirty = True
        self.assertTrue(w.windowTitle().endswith('NEW ●'))
        self.assertEqual(w.lbl_save_state.text(), '还没保存过')
        w.session.save()
        self.assertTrue(w.lbl_save_state.text().startswith('已保存 '))

    def check_decision(self, answer, operation):
        w = self.window
        s = self.dirty()
        s.autosave()
        document = s.episode.dir / 'document.json'
        auto = s.episode.dir / 'journal' / 'autosave.json'
        before, recovery = document.read_bytes(), auto.read_bytes()
        selection = (w.current_gid, w.table.current_gid(), w.wave.current_gid, w.watch.selected)
        project = w.project
        title, status = w.windowTitle(), w.lbl_save_state.text()
        event = QCloseEvent()
        with patch.object(w, '_ask_save', return_value=answer) as ask:
            if operation == 'close':
                w.closeEvent(event)
            elif operation == 'episode':
                w.ep_box.setCurrentText('B')
            else:
                other = Project.create(Path(self.tmp.name) / 'other', 'Other')
                w.open_project(other.root)
            ask.assert_called_once_with()
        if answer == QMessageBox.StandardButton.Cancel:
            if operation == 'close':
                self.assertFalse(event.isAccepted())
            self.assertIs(w.session, s)
            self.assertIs(w.project, project)
            self.assertEqual(w.ep_box.currentText(), 'A')
            self.assertEqual((w.current_gid, w.table.current_gid(), w.wave.current_gid, w.watch.selected), selection)
            self.assertEqual((w.windowTitle(), w.lbl_save_state.text()), (title, status))
            self.assertTrue(s.dirty)
            self.assertFalse(s._closed)
            self.assertFalse(w._closing)
            self.assertEqual(document.read_bytes(), before)
            self.assertEqual(auto.read_bytes(), recovery)
        else:
            self.assertTrue(s._closed)
            self.assertIsNot(w.session, s)
            if operation == 'close':
                self.assertTrue(event.isAccepted())
            elif operation == 'episode':
                self.assertEqual(w.session.episode.id, 'B')
                self.assertEqual(w.ep_box.currentText(), 'B')
            else:
                self.assertEqual(w.project.name, 'Other')
                self.assertEqual(w.windowTitle(), f'{APP_NAME} — Other')
            if answer == QMessageBox.StandardButton.Save:
                self.assertNotEqual(document.read_bytes(), before)
                self.assertEqual(Project.open(project.root).episode('A').document.group('A-0').extra['mark_note'], '未保存')
            else:
                self.assertEqual(document.read_bytes(), before)
                self.assertFalse(auto.exists())
                self.assertIsNone(Project.open(project.root).episode('A').autosave_recovery())

    def test_close_save(self):
        self.check_decision(QMessageBox.StandardButton.Save, 'close')

    def test_close_discard(self):
        self.check_decision(QMessageBox.StandardButton.Discard, 'close')

    def test_close_cancel(self):
        self.check_decision(QMessageBox.StandardButton.Cancel, 'close')

    def test_episode_save(self):
        self.check_decision(QMessageBox.StandardButton.Save, 'episode')

    def test_episode_discard(self):
        self.check_decision(QMessageBox.StandardButton.Discard, 'episode')

    def test_episode_cancel(self):
        self.check_decision(QMessageBox.StandardButton.Cancel, 'episode')

    def test_project_save(self):
        self.check_decision(QMessageBox.StandardButton.Save, 'project')

    def test_project_discard(self):
        self.check_decision(QMessageBox.StandardButton.Discard, 'project')

    def test_project_cancel(self):
        self.check_decision(QMessageBox.StandardButton.Cancel, 'project')

    def test_cancel_leaves_draft_and_jobs_untouched(self):
        w = self.window
        self.dirty()
        with patch.object(w, '_ask_save', return_value=QMessageBox.StandardButton.Cancel), \
                patch.object(w.watch, 'finish_transform') as transform, \
                patch.object(w.wave, 'cancel_jobs') as jobs:
            self.assertFalse(w.close_session())
            transform.assert_not_called()
            jobs.assert_not_called()

    def test_setting_defaults_on_persists_and_off_silently_saves(self):
        w = self.window
        self.assertTrue(w.act_ask_save.isChecked())
        self.assertEqual(w.act_ask_save.property('help_anchor'), 'save')
        self.assertTrue(w.act_ask_save.statusTip())
        w.act_ask_save.setChecked(False)
        self.settings.sync()
        reopened = QSettings(self.settings.fileName(), QSettings.Format.IniFormat)
        self.assertFalse(reopened.value('save/ask_on_close', True, type=bool))
        self.dirty()
        with patch.object(w, '_ask_save') as ask:
            w.ep_box.setCurrentText('B')
            ask.assert_not_called()
        self.assertEqual(self.project.episode('A').document.group('A-0').extra['mark_note'], '未保存')

    def test_real_prompt_buttons_default_escape_and_help(self):
        w = self.window
        self.dirty()
        for choice in ('保存', '不保存', '取消', 'enter', 'escape'):
            errors = []
            def choose():
                dialog = self.app.activeModalWidget()
                try:
                    self.assertEqual(dialog.text(), '这一集（A）有未保存的改动。保存吗？')
                    self.assertCountEqual([b.text() for b in dialog.buttons()], ['保存', '不保存', '取消'])
                    self.assertEqual(dialog.defaultButton().text(), '保存')
                    self.assertEqual(dialog.escapeButton().text(), '取消')
                    with patch.object(app_help, 'open_manual') as opener:
                        QTest.keyClick(dialog, Qt.Key.Key_F1)
                        opener.assert_called_once_with('save')
                    if choice in ('enter', 'escape'):
                        QTest.keyClick(dialog, Qt.Key.Key_Return if choice == 'enter' else Qt.Key.Key_Escape)
                    else:
                        next(b for b in dialog.buttons() if b.text() == choice).click()
                except BaseException as exc:
                    errors.append(exc)
                    dialog.reject()
            QTimer.singleShot(0, choose)
            answer = ASK_SAVE(w)
            if errors:
                raise errors[0]
            self.assertEqual(answer, {'保存': QMessageBox.StandardButton.Save, 'enter': QMessageBox.StandardButton.Save,
                                    '不保存': QMessageBox.StandardButton.Discard,
                                    '取消': QMessageBox.StandardButton.Cancel, 'escape': QMessageBox.StandardButton.Cancel}[choice])

    def test_save_failure_keeps_session_open(self):
        w = self.window
        s = self.dirty()
        with patch.object(s.episode, 'save', side_effect=OSError('disk full')), \
                patch.object(w, '_help_message') as warning:
            self.assertFalse(w.close_session())
            warning.assert_called_once()
        self.assertIs(w.session, s)
        self.assertTrue(s.dirty)
        self.assertFalse(s._closed)
