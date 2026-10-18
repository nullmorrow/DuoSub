"""Real recovery files and offscreen prompt/session wiring."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox
from subkit.core.workspace import Project, Episode
from subkit.model.document import Document
from subkit.model.serialize import document_to_json, write_json_atomic
from subkit_app.mainwindow import MainWindow
from subkit_app.session import EpisodeSession


class AutosaveRecover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ConversionUI.setUpClass()
        cls.app = fixtures.ConversionUI.app

    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="subkit_recovery_"))
        self.project = Project.create(self.path / "project", "Recovery")
        self.ep = self.project.add_episode("E01")
        self.ep.document = Document(episode="E01")
        self.ep.save()
        self.saved = self.ep.dir / "document.json"
        self.auto = self.ep.dir / "journal" / "autosave.json"
        self.before = self.saved.read_bytes()
        self.stamp = self.saved.stat().st_mtime + 5

    def autosave(self, *, different=True, stamp=None):
        data = document_to_json(self.ep.document)
        if different:
            data["recovery_marker"] = "人工成果"
        write_json_atomic(self.auto, data)
        os.utime(self.auto, (stamp or self.stamp, stamp or self.stamp))
        return data

    def test_only_newer_and_different_prompts(self):
        self.assertIsNone(self.ep.autosave_recovery())
        self.autosave(different=False)
        self.assertIsNone(self.ep.autosave_recovery())
        self.autosave(stamp=self.saved.stat().st_mtime)
        self.assertIsNone(self.ep.autosave_recovery())
        self.autosave(stamp=self.saved.stat().st_mtime - 2)
        self.assertIsNone(self.ep.autosave_recovery())
        expected = self.autosave()
        document, stamp = self.ep.autosave_recovery()
        self.assertEqual(document_to_json(document), expected)
        self.assertEqual(stamp, self.stamp)
        self.assertEqual(self.saved.read_bytes(), self.before)

    def test_json_whitespace_difference_is_not_a_recovery(self):
        self.autosave(different=False)
        self.auto.write_text(self.auto.read_text(encoding="utf-8") + "\n  ", encoding="utf-8")
        os.utime(self.auto, (self.stamp, self.stamp))
        self.assertIsNone(self.ep.autosave_recovery())

    def test_discard_archives_bytes_and_avoids_overwrite(self):
        self.autosave()
        contents = self.auto.read_bytes()
        first = self.ep.discard_autosave(self.stamp)
        self.autosave()
        second = self.ep.discard_autosave(self.stamp)
        self.assertNotEqual(first, second)
        self.assertFalse(self.auto.exists())
        self.assertEqual(first.read_bytes(), contents)
        self.assertEqual(second.read_bytes(), contents)
        self.assertEqual(self.saved.read_bytes(), self.before)

    def prompt(self, label):
        # The real QMessageBox is offscreen; a timer chooses its actual button.
        from PySide6.QtWidgets import QWidget
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        def choose():
            dialog = self.app.activeModalWidget()
            self.assertEqual(dialog.help_anchor, "project")
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.stamp))
            self.assertEqual(dialog.text(), f"上次没有正常关闭。找到 {when} 的自动保存，比已保存的文档新。恢复它吗？")
            next(b for b in dialog.buttons() if b.text() == label).click()
        QTimer.singleShot(0, choose)
        MainWindow._offer_autosave_recovery(parent, self.ep)

    def test_recover_marks_dirty_then_manual_save_persists(self):
        expected = self.autosave()
        self.prompt("恢复")
        self.assertEqual(document_to_json(self.ep.document), expected)
        self.assertEqual(self.saved.read_bytes(), self.before)
        session = EpisodeSession(self.project, self.ep)
        try:
            self.assertTrue(session.dirty)
            session.save()
            self.assertFalse(session.dirty)
            self.assertFalse(self.ep.recovered)
            self.assertEqual(document_to_json(Episode.open(self.project, self.ep.dir).document), expected)
        finally:
            session.close()
            session.deleteLater()

    def test_discard_prompt_keeps_saved_document(self):
        self.autosave()
        self.prompt("丢弃")
        self.assertFalse(self.ep.recovered)
        self.assertFalse(self.auto.exists())
        self.assertEqual(len(list(self.auto.parent.glob("autosave-discarded-*.json"))), 1)
        self.assertEqual(self.saved.read_bytes(), self.before)

    def test_episode_open_calls_recovery_before_session(self):
        self.autosave()
        recovered, _ = self.ep.autosave_recovery()
        calls = []
        def recover(ep):
            calls.append("recovery")
            ep.document, ep.recovered = recovered, True
        box = Mock(); box.currentText.return_value = "E01"
        window = SimpleNamespace(_episode_job=None, session=None, project=self.project,
                                 close_session=Mock(return_value=True), people=Mock(), settings=Mock(), ep_box=box,
                                 _offer_autosave_recovery=recover,
                                 _start_session=lambda ep: calls.append(("session", ep.recovered)))
        MainWindow._switch_episode(window, "E01")
        self.assertEqual(calls, ["recovery", ("session", True)])

    def test_invalid_autosave_is_reported_without_changing_files(self):
        self.autosave()
        self.auto.write_text("{broken", encoding="utf-8")
        os.utime(self.auto, (self.stamp, self.stamp))
        with patch.object(QMessageBox, "warning") as warning:
            MainWindow._offer_autosave_recovery(None, self.ep)
        warning.assert_called_once()
        self.assertEqual(self.saved.read_bytes(), self.before)
        self.assertEqual(self.auto.read_text(encoding="utf-8"), "{broken")
