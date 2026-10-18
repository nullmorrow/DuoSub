"""R9 layouts at 1080p: real soma copy, offscreen Qt and player double."""

import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout
from subkit.core.workspace import Project
from subkit_app.session import EpisodeSession
from subkit_app.views.group_table import GroupFilter, GroupTableModel


class ReviewR9(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__['setUpClass']
    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        # Qt offscreen's default screen is 800x800 and native restoreGeometry clamps to it.
        # Keep a 1920x1040 client for these layout checks; real screen restoration remains human QA.
        p = patch.object(fixtures.mw.MainWindow, 'restoreGeometry',
                         lambda window, data: window.resize(1920, 1040))
        p.start()
        self.addCleanup(p.stop)
        fixtures.ConversionUI.setUp(self)
        for font in ('msyh.ttc', 'msgothic.ttc'):
            QFontDatabase.addApplicationFont(str(Path(os.environ['WINDIR']) / 'Fonts' / font))
        self.app.setFont(QFont('Microsoft YaHei', 9))
        self.session.close()
        self.session.deleteLater()
        shutil.copytree(fixtures.ROOT / 'test_projects' / 'soma', self.path / 'soma',
                        copy_function=shutil.copyfile, ignore=shutil.ignore_patterns('cache'))
        project = Project.open(self.path / 'soma')
        episode = project.episode(project.episode_ids()[0])
        self.session = EpisodeSession(project, episode)
        self.session.request_preview = Mock()
        w = self.window
        w.project, w.session = project, self.session
        w.model = GroupTableModel(self.session.doc)
        w.proxy = GroupFilter()
        w.proxy.setSourceModel(w.model)
        w.table.setModel(w.proxy)
        w.wave.set_document(self.session.doc)
        w.source.set_data(self.session.doc, episode.sources)
        w.select_group(self.session.doc.groups[40].id)
        w.player.pause = Mock()
        w.player._hide_timer = Mock()
        w.resize(1920, 1040)
        w.show()
        self.app.processEvents()

    def test_layout_geometry_and_screenshots(self):
        w = self.window
        evidence = Path(os.environ['TEMP']) / 'r9-evidence'
        evidence.mkdir(exist_ok=True)
        frame = evidence / 'frame.png'
        if frame.exists():
            label = QLabel(w.player)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('background:black')
            box = QVBoxLayout(w.player)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(label)
        sizes = {}
        for layout in ('video', 'list', 'columns'):
            w.set_edit_layout(layout)
            self.app.processEvents()
            self.assertEqual((w.width(), w.height()), (1920, 1040))
            self.assertTrue(w.wave_panel.isVisible())
            self.assertTrue(w.editor.isVisible())
            self.assertGreaterEqual(w.table.viewport().height(), w.table.rowHeight(0) * 6)
            self.assertGreaterEqual(w.wave.height(), 120)
            self.assertGreaterEqual(w.editor.height(), 240)
            if layout == 'video':
                self.assertLessEqual(abs(w.player.width() - round(w.player.height() * 16 / 9)), 2)
                self.assertIs(w.wave_panel.parentWidget(), w.video_split)
                self.assertFalse(w.dock_source.isVisible())
                self.assertFalse(w.people.isVisible())
            elif layout == 'columns':
                self.assertIs(w.table.parentWidget(), w.top_split)
                self.assertIs(w.dock_source.parentWidget(), w.right_host)
                self.assertIn(w.people, w.right_host.tabifiedDockWidgets(w.dock_source))
            if frame.exists():
                label.setPixmap(QPixmap(str(frame)).scaled(w.player.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))
                self.app.processEvents()
            sizes[layout] = {name: [widget.width(), widget.height()] for name, widget in
                             (('video', w.player), ('wave', w.wave), ('editor', w.editor), ('list', w.table))}
            self.assertTrue(w.grab().save(str(evidence / (layout + '.png'))))
        (evidence / 'geometry.json').write_text(json.dumps(sizes, indent=2), encoding='utf-8')

    def test_splitter_memory_restart_migration_and_toolbar(self):
        w = self.window
        w.top_split.setSizes([1200, 720])
        self.app.processEvents()
        video_sizes = w.top_split.sizes()
        w.set_edit_layout('list')
        w.main_split.setSizes([500, 480])
        list_sizes = w.main_split.sizes()
        w.set_edit_layout('video')
        self.app.processEvents()
        self.assertEqual(w.top_split.sizes(), video_sizes)
        w.set_edit_layout('list')
        self.assertEqual(w.main_split.sizes(), list_sizes)
        w.dock_source.toggleViewAction().trigger()
        self.assertFalse(w.dock_source.isVisible())
        self.assertIn(w.people.toggleViewAction(), w.toolbar.actions())
        w.close()
        restarted = fixtures.mw.MainWindow()
        try:
            restarted.show()
            self.app.processEvents()
            self.assertEqual(restarted.edit_layout, 'list')
            self.assertEqual(restarted.main_split.sizes(), list_sizes)
            self.assertFalse(restarted.dock_source.isVisible())
        finally:
            restarted.close()
            restarted.deleteLater()
        # Simulate a pre-R9 user with visible docks and saved old main-window state.
        self.settings.remove('layouts')
        self.settings.setValue('state', w.saveState())
        migrated = fixtures.mw.MainWindow()
        try:
            migrated.show()
            self.app.processEvents()
            self.assertEqual(migrated.edit_layout, 'video')
            self.assertFalse(migrated.dock_source.isVisible())
            self.assertFalse(migrated.people.isVisible())
        finally:
            migrated.close()
            migrated.deleteLater()

    def test_watch_round_trip_all_layouts(self):
        w = self.window
        for layout in ('video', 'list', 'columns'):
            w.set_edit_layout(layout)
            self.app.processEvents()
            before = {key: split.sizes() for key, split in w._edit_splitters()}
            visible = (w.dock_source.isVisible(), w.people.isVisible())
            for pinned in (True, False, True, False):
                w.set_watch_mode(True, pinned=pinned)
                self.app.processEvents()
                self.assertTrue(w.isFullScreen())
                self.assertIs(w.wave_panel.parentWidget(), w.video_split)
                self.assertEqual(w.wave_panel.isVisible(), pinned)
            w.set_watch_mode(False)
            self.app.processEvents()
            self.assertEqual(w.edit_layout, layout)
            self.assertIs(w.wave_panel.parentWidget(), w.video_split if layout == 'video' else w.right_split)
            self.assertEqual((w.dock_source.isVisible(), w.people.isVisible()), visible)
            self.assertEqual({key: split.sizes() for key, split in w._edit_splitters()}, before)


if __name__ == '__main__':
    unittest.main()
