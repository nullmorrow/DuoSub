"""R7 navigation: offscreen Qt events, real soma copy and player double."""

import shutil
import time
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit
from subkit.core.workspace import Project
from subkit_app.session import EpisodeSession
from subkit_app.views.group_table import GroupFilter, GroupTableModel
from subkit_app.views.reading_popup import ReadingPopup


class ReviewR7(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__['setUpClass']
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        for i in range(8):
            self.group(str(i), text=f'台词{i}', start=i * 3000)
        self.window.select_group('0')
        self.window.show()
        self.window.activateWindow()
        self.window.table.setFocus()
        self.app.processEvents()
        self.window.player.seek.reset_mock()

    def test_click_burst_highlights_and_updates_only_final_group(self):
        w = self.window
        with patch.object(w.editor, 'set_group', wraps=w.editor.set_group) as heavy:
            for i in range(1, 8):
                pos = w.table.visualRect(w.proxy.index(i, 4)).center()
                QTest.mouseClick(w.table.viewport(), Qt.MouseButton.LeftButton, pos=pos)
                self.assertEqual(w.current_gid, str(i))
                self.assertEqual(w.table.current_gid(), str(i))
                self.assertEqual(heavy.call_count, 0)
            QTest.qWait(200)
            self.assertEqual(heavy.call_count, 1)
        self.assertEqual(w.editor.group.id, '7')
        self.assertEqual(w.wave.current_gid, '7')
        self.assertEqual(w.watch.selected, '7')
        seeks = w.player.seek.call_args_list
        self.assertEqual(len(seeks), 8)
        self.assertTrue(all(call.kwargs == {'exact': False} for call in seeks[:-1]))
        self.assertEqual(seeks[-1].args, (21.0,))
        state = (w.editor.title.text(), w.source.toPlainText(), w.wave.current_gid,
                 w.watch.selected, dict(w.wave.kviews))
        w.select_group('7')
        direct = (w.editor.title.text(), w.source.toPlainText(), w.wave.current_gid,
                  w.watch.selected, dict(w.wave.kviews))
        self.assertEqual(state, direct)

    def test_held_arrows_and_page_keys_wait_for_release(self):
        w = self.window
        for key in (Qt.Key.Key_Down, Qt.Key.Key_PageDown, Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            with self.subTest(key=key):
                w.select_group('4')
                w.table.setFocus()
                with patch.object(w.editor, 'set_group', wraps=w.editor.set_group) as heavy:
                    QTest.keyPress(w.table, key)
                    QTest.qWait(200)
                    self.assertEqual(heavy.call_count, 0)
                    for _ in range(2):
                        self.app.sendEvent(w.table, QKeyEvent(QEvent.Type.KeyPress, key,
                                           Qt.KeyboardModifier.NoModifier, '', True, 1))
                        self.app.sendEvent(w.table, QKeyEvent(QEvent.Type.KeyRelease, key,
                                           Qt.KeyboardModifier.NoModifier, '', True, 1))
                    self.assertEqual(heavy.call_count, 0)
                    QTest.keyRelease(w.table, key)
                    QTest.qWait(200)
                    self.assertEqual(heavy.call_count, 1)
                    self.assertEqual(w.editor.group.id, w.current_gid)

    def test_lost_key_release_never_defers_selection_forever(self):
        # Walkthrough 2026-10-18: after a burst of synthetic arrow presses one release was lost and
        # clicking a row no longer updated the panels until the next key press.
        w = self.window
        w.select_group('4')
        w.table.setFocus()
        with patch.object(w.editor, 'set_group', wraps=w.editor.set_group) as heavy:
            QTest.keyPress(w.table, Qt.Key.Key_Down)       # its release never arrives
            QTest.qWait(200)
            self.assertEqual(heavy.call_count, 0)           # still treated as held
            w._group_nav_last -= 2.0                        # no auto-repeat for longer than any repeat delay
            QTest.qWait(200)
            self.assertEqual(heavy.call_count, 1)
            self.assertEqual(w.editor.group.id, '5')
        QTest.keyPress(w.table, Qt.Key.Key_Down)            # another lost release …
        with patch.object(w.editor, 'set_group', wraps=w.editor.set_group) as heavy:
            pos = w.table.visualRect(w.proxy.index(1, 4)).center()
            QTest.mouseClick(w.table.viewport(), Qt.MouseButton.LeftButton, pos=pos)   # … ended by a click
            QTest.qWait(200)
            self.assertEqual(heavy.call_count, 1)
        self.assertEqual(w.editor.group.id, '1')

    def test_typing_popup_inline_transform_and_tapper_never_defer(self):
        w = self.window
        edit = QLineEdit(w)
        edit.show()
        edit.setFocus()
        self.app.processEvents()
        w.select_group('1', defer=True)
        self.assertIsNone(w._pending_group)
        self.assertEqual(w.editor.group.id, '1')
        w.table.setFocus()
        pop = ReadingPopup('私', 'わたし', [], 'manual', '', w)
        pop.show()
        with patch.object(w, '_typing', return_value=False):
            w.select_group('2', defer=True)
        self.assertEqual(w.editor.group.id, '2')
        self.assertIsNone(w._pending_group)
        pop.close()
        pop.deleteLater()
        with patch.object(w, '_inline', Mock()):
            w.select_group('3', defer=True)
            self.assertIsNone(w._pending_group)
        with patch.object(w.watch, 'transform', Mock()), patch.object(w.watch, 'set_selected'):
            w.select_group('4', defer=True)
            self.assertIsNone(w._pending_group)
        w.tapper.run = {'active': True}
        w.select_group('5', defer=True)
        self.assertIsNone(w._pending_group)
        w.tapper.run = None
        edit.close()

    def test_typing_after_navigation_uses_final_group(self):
        w = self.window
        w._on_table_group('2')
        w.editor.cn_edits[0][1].setFocus()
        self.app.processEvents()
        self.assertEqual(w.editor.group.id, '2')
        self.assertIsNone(w._pending_group)
        edit = self.app.focusWidget()
        self.assertIs(edit, w.editor.cn_edits[0][1])
        edit.selectAll()
        QTest.keyClicks(edit, 'changed')
        edit.commit()
        self.assertEqual(self.doc.members['m_2'].text, 'changed')
        self.assertEqual(self.doc.members['m_0'].text, '台词0')

    def test_direct_selection_cancels_old_seek_and_close_cancels_timer(self):
        w = self.window
        w._on_table_group('1')
        w.select_group('2', seek=False)
        w.player.seek.reset_mock()
        QTest.qWait(200)
        w.player.seek.assert_not_called()
        self.assertEqual(w.editor.group.id, '2')
        w._on_table_group('3')
        w.close_session()
        self.assertFalse(w._group_select_timer.isActive())
        self.assertIsNone(w._pending_group)

    def test_k_start_and_transform_flush_navigation_before_targeting(self):
        w = self.window
        member = self.group('song', 'song', text='歌詞')
        w._on_table_group('song')
        w.player.play = Mock()
        w._k_action('start')
        self.assertEqual(w.tapper.run['mid'], member.id)
        self.assertIsNone(w._pending_group)
        w.tapper.cancel()
        w._on_table_group('2')
        w.watch_mode = True
        with patch.object(w.watch, 'enter_transform') as enter:
            w._start_transform()
            enter.assert_called_once_with(only_line=False)
        self.assertEqual(w.watch.selected, '2')
        w.watch_mode = False

    def test_soma_navigation_main_thread_under_16ms(self):
        w = self.window
        shutil.copytree(fixtures.ROOT / 'test_projects/soma', self.path / 'soma',
                        copy_function=shutil.copyfile, ignore=shutil.ignore_patterns('cache'))
        project = Project.open(self.path / 'soma')
        episode = project.episode(project.episode_ids()[0])
        self.session.close()
        self.session.deleteLater()
        self.session = EpisodeSession(project, episode)
        self.session.request_preview = Mock()
        w.project, w.session = project, self.session
        w.model = GroupTableModel(self.session.doc)
        w.proxy = GroupFilter()
        w.proxy.setSourceModel(w.model)
        w.table.setModel(w.proxy)
        w.wave.set_document(self.session.doc)
        w.source.set_data(self.session.doc, episode.sources)
        w.select_group(self.session.doc.groups[0].id)
        self.assertGreaterEqual(len(self.session.doc.groups), 400)
        durations = []
        with patch.object(w.editor, 'set_group', wraps=w.editor.set_group) as heavy:
            for g in self.session.doc.groups[20:60]:
                start = time.perf_counter()
                w._on_table_group(g.id)
                durations.append((time.perf_counter() - start) * 1000)
            self.assertEqual(heavy.call_count, 0)
            self.assertLess(max(durations), 16, durations)
            start = time.perf_counter()
            QTest.qWait(200)
            self.assertEqual(heavy.call_count, 1)
            self.assertLess(time.perf_counter() - start, .3)
            self.assertEqual(w.editor.group.id, self.session.doc.groups[59].id)


if __name__ == '__main__':
    unittest.main()
