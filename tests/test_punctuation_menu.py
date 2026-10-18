"""Punctuation display menus, persistence and font help with offscreen Qt."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox
from subkit.core.display import DisplayRules
from subkit.core.workspace import Project
from subkit.languages import lane_label
from subkit.styles_default import default_styles_ass
from subkit_app import help as help_ui, mainwindow as mw
from subkit_app.session import EpisodeSession, PreviewJob
from subkit_app.views.punctuation import PunctuationDialog
from tests.test_convert_ui import PlayerDouble
from tests.test_pinyin_export import document


class PunctuationMenu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Punctuation menu tests require offscreen Qt")

    def setUp(self):
        fonts = patch("subkit_app.views.punctuation.QFontDatabase.families", return_value=["Arial", "Times New Roman"])
        fonts.start()
        self.addCleanup(fonts.stop)
        self.path = Path(tempfile.mkdtemp(prefix="punctuation-menu-", dir=os.environ["TEMP"]))
        self.settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        with patch.object(mw, "MpvWidget", PlayerDouble), patch.object(mw, "QSettings", return_value=self.settings), \
                patch.object(mw.MainWindow, "_open_last", Mock()):
            self.window = mw.MainWindow()
        self.project = Project.create(self.path / "project", "punctuation", template_text=default_styles_ass())
        self.ep = self.project.add_episode("E1")
        self.ep.document = document('甲，乙...!?“丙”', lang="ja", lane="original", style="JP")
        self.session = EpisodeSession(self.project, self.ep)
        self.window.project, self.window.session = self.project, self.session
        self.window._apply_rules()

    def tearDown(self):
        self.session.close()
        self.window.session = None
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settings.sync()

    def test_screen_replaces_old_submenus_and_font_action(self):
        w = self.window
        actions = w.act_punctuation.parent().menuBar().actions()
        def walk(items):
            for action in items:
                yield action
                if action.menu():
                    yield from walk(action.menu().actions())
        all_actions = list(walk(actions))
        self.assertFalse(any(a.text() == "标点字体…" for a in all_actions))
        entries = [a for a in all_actions if a.text() == "标点和空格…"]
        self.assertEqual(len(entries), 1)
        self.assertTrue(all(a.menu() is None for a in entries))
        with patch.object(mw.PunctuationDialog, "exec", return_value=0) as opened:
            w.act_punctuation.trigger()
            opened.assert_called_once()

    def test_legacy_preview_matches_export_without_document_changes(self):
        before = copy.deepcopy(self.ep.document)
        self.window._set_punctuation("original", "ellipsis", "single")
        job = PreviewJob(self.session, 0, self.path / "preview.ass")
        done, errors = [], []
        job.signals.done.connect(lambda *args: done.append(args))
        job.signals.failed.connect(lambda *args: errors.append(args))
        job.run()
        self.assertFalse(errors, errors)
        exported, _ = self.ep.build_export()
        self.assertEqual(done[0][2].events[0].text, exported.events[0].text)
        self.assertEqual(self.ep.document, before)
        self.assertFalse(self.session.dirty)

