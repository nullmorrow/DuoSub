"""R1: real offscreen Qt events, with isolated settings and a player double."""

from unittest.mock import Mock, patch
import unittest

from PySide6.QtCore import QPoint, QRect, Qt, QItemSelectionModel
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu

from tests import test_convert_ui as fixtures
from subkit_app.views.people import SpeakerPicker


class ReviewR1(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.window.player._hide_timer = Mock()
        self.window.player._autohide = False
        self.window.player.pause = Mock()

    def show(self):
        self.window.show()
        self.window.activateWindow()
        self.window.player.setFocus()
        self.app.processEvents()

    def test_watch_keys_hidden_menus_switch_once_and_restore(self):
        self.addCleanup(lambda: QTest.qWait(350))
        self.show()
        self.window.dock_source.show()
        self.window.people.show()
        self.app.processEvents()
        for key, mode, pinned in ((Qt.Key.Key_F10, True, True), (Qt.Key.Key_F11, True, False),
                                  (Qt.Key.Key_F10, True, True), (Qt.Key.Key_F11, True, False),
                                  (Qt.Key.Key_F11, False, False)):
            QTest.keyClick(self.window, key)
            self.app.processEvents()
            w = self.window
            self.assertEqual(w.watch_mode, mode)
            self.assertEqual(w.isFullScreen(), mode)
            self.assertEqual(w.wave_panel.pinned(), pinned)
            self.assertEqual(w.wave_panel.isVisible(), not mode or pinned)
            for widget in (w.menuBar(), w.statusBar(), w.table, w.dock_source, w.people, w.toolbar):
                self.assertEqual(widget.isVisible(), not mode)
            self.assertEqual(w.act_watch.isChecked(), mode and not pinned)
            self.assertEqual(w.act_watch2.isChecked(), mode and pinned)

    def test_follow_song_updates_beats_preserving_multi_selection(self):
        self.group("a", start=0)
        member = self.group("song", "song", start=3000)
        self.group("b", start=6000)
        self.select("a", "b")
        self.window.player.paused = False
        self.window._on_time(3.5)
        self.assertEqual(self.window.current_gid, "song")
        self.assertEqual(self.window.table.current_gid(), "song")
        self.assertEqual(self.window.table.selected_gids(), ["a", "b"])
        self.assertEqual(self.window.wave.kviews["translation"][0], member.id)
        self.assertTrue(self.window.wave.kviews["translation"][1].beats)

    def test_overlap_latest_start_then_dialogue(self):
        self.group("old", start=0)
        self.group("song", "song", start=500)
        self.group("dialogue", start=500)
        self.group("sign", "sign", start=600)
        self.assertEqual(self.window._group_at(1000).id, "dialogue")
        self.doc.members["m_song"].start = 700
        self.assertEqual(self.window._group_at(1000).id, "song")
        self.assertIsNone(self.window._group_at(10000))

    def test_right_click_selected_rows_builds_menu_and_emits_all(self):
        self.group("a").lane = "original"
        self.group("b")
        self.show()
        table = self.window.table
        for label, signal, expected in (("删除这", table.deleteGroupsRequested, (["a", "b"],)),
                                        ("只删原声", table.deleteLaneRequested, (["a", "b"], "original")),
                                        ("只删翻译", table.deleteLaneRequested, (["a", "b"], "translation")),
                                        ("确认这一组", table.confirmGroupsRequested, (["a", "b"], True)),
                                        ("取消确认", table.confirmGroupsRequested, (["a", "b"], False)),
                                        ("歌曲（K 轴）…", table.convertRequested, (["a", "b"], "song")),
                                        ("记为重复", table.recordPassageRequested, (["a", "b"],))):
            for group in self.doc.groups:
                group.state = "confirmed" if label == "取消确认" else "draft"
            self.select("a", "b")
            signal.disconnect()
            received = []
            signal.connect(lambda *args: received.append(args))
            pos = table.visualRect(table.model().index(1, 4)).center()
            choose = [False]

            class MenuDouble(QMenu):
                def exec(self, _pos):
                    return next((a for a in self.actions() + [c for item in self.actions() if item.menu() for c in item.menu().actions()] if label in a.text()), None) if choose[0] else None

            with patch("subkit_app.views.group_table.QMenu", MenuDouble):
                QTest.mouseClick(table.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, pos)
                self.assertEqual(table.selected_gids(), ["a", "b"])
                choose[0] = True
                event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, table.viewport().mapToGlobal(pos))
                self.app.sendEvent(table.viewport(), event)
            self.assertEqual(received, [expected])

    def test_disabled_actions_reasons_and_reenable(self):
        self.group("song", "song")
        self.select("song")
        self.window._update_convert_actions()
        self.assertIn("（已经是歌曲）", self.window.act_convert_song.text())
        self.assertIn("已经是歌曲", self.window.act_convert_song.toolTip())
        self.assertIn("先在观剧模式里点选一行字幕", self.window.act_transform.text())
        self.group("d")
        self.select("d")
        self.window._update_convert_actions()
        self.assertEqual(self.window.act_convert_song.text(), "歌曲（K 轴）…")
        self.assertIn("已经是对白", self.window.act_convert_dialogue.text())

    def test_picker_bottom_small_screen_caps_and_scrolls(self):
        member = self.group("a")
        book = self.session.editor.character_book
        for i in range(40):
            book.add_character({"zh-Hans": f"人物{i}"})
        area = QRect(0, 0, 400, 500)
        with patch("subkit_app.views.placement.screen_area", return_value=area):
            self.window._open_speaker_picker(member.id, QPoint(380, 480))
        picker = self.window._speaker_picker
        self.addCleanup(picker.close)
        self.app.processEvents()
        self.assertLessEqual(picker.height(), area.height() - 40)
        self.assertLess(picker.y(), 480)
        self.assertTrue(area.contains(picker.geometry()))
        self.assertGreater(picker.list.verticalScrollBar().maximum(), 0)
        self.assertEqual(picker.windowType(), Qt.WindowType.Tool)
        picker.close()


if __name__ == "__main__":
    unittest.main()
