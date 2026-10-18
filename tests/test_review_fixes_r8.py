"""R8 mouse and F10 navigation: Qt offscreen, isolated settings, player double."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from tests import test_convert_ui as fixtures
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, qVersion
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from subkit_app import mainwindow as mw
from subkit_app.views.waveform import LANE_H


class ReviewR8(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.group("a", text="第一句", start=1000)
        self.group("b", text="第二句", start=5000)
        self.window.player.pause = Mock()
        self.window.player._hide_timer = Mock()
        self.window.player._autohide = False
        self.window.select_group("a")
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        self.wave = self.window.wave
        self.wave.t0, self.wave.span = 0, 10000
        self.wave.repaint()
        self.window.player.seek.reset_mock()
        self.times, self.shifts = [], []
        self.wave.timeClicked.disconnect()
        self.wave.groupShifted.disconnect()
        self.wave.timeClicked.connect(lambda *args: self.times.append(args))
        self.wave.groupShifted.connect(lambda *args: self.shifts.append(args))

    def point(self, ms, y=None):
        return QPoint(round(self.wave.x_of(ms)), self.wave.height() // 2 if y is None else y)

    def drag(self, point, dx):
        QTest.mousePress(self.wave, Qt.MouseButton.LeftButton, pos=point)
        moved = QPointF(point + QPoint(dx, 0))
        event = QMouseEvent(QEvent.Type.MouseMove, moved, moved,
                            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(self.wave, event)
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, pos=point + QPoint(dx, 0))

    def test_default_plain_and_shift_timing_snap(self):
        self.assertEqual(self.wave.timing_mode, "aegisub")
        p = self.point(4000)
        raw = round(self.wave.ms_of(p.x()))
        self.wave.keyframes = np.array([raw + 10])
        QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, pos=p)
        QTest.mouseClick(self.wave, Qt.MouseButton.RightButton, pos=p)
        QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, pos=p)
        self.assertEqual(self.times, [(raw + 10, "start"), (raw + 10, "end"), (raw, "start")])

    def test_bar_body_and_edges_small_move_select_without_edit(self):
        edits = []
        self.wave.memberTimeEdited.connect(lambda *args: edits.append(args))
        for mode in ("aegisub", "select"):
            self.window._set_timing_mode(mode)
            for ms in (5000, 6000, 7000):
                self.window.select_group("a")
                self.wave.repaint()
                before = [(m.start, m.end) for m in self.doc.members.values()]
                self.drag(self.point(ms, 10), 5)
                self.assertEqual(self.window.current_gid, "b")
                self.assertEqual(before, [(m.start, m.end) for m in self.doc.members.values()])
                self.window.player.pause.assert_called()
                self.window.player.seek.assert_called_with(5.0)
        self.assertFalse(edits or self.times or self.shifts)
        self.assertEqual(LANE_H, 26)
        self.assertTrue(all(r.height() == 26 for r, *_ in self.wave._bars))

    def test_drag_above_threshold_moves_only_once(self):
        self.drag(self.point(6000, 10), 6)
        self.assertEqual(len(self.shifts), 1)
        self.assertEqual(self.shifts[0][0], "b")
        self.assertEqual(self.window.current_gid, "a")

    def test_hover_uses_list_number_including_non_dialogue(self):
        self.doc.group("a").kind = "sign"
        self.wave.repaint()
        QTest.mouseMove(self.wave, self.point(6000, 10))
        self.assertEqual(self.wave.toolTip(), "第 2 组：第二句")
        self.assertEqual(self.window.model.data(self.window.model.index(1, 1)), 2)

    def test_select_mode_plain_range_empty_and_shift_alt(self):
        self.window._set_timing_mode("select")
        QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, pos=self.point(6000))
        self.assertEqual(self.window.current_gid, "b")
        self.window.player.seek.reset_mock()
        QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, pos=self.point(8000))
        QTest.mouseClick(self.wave, Qt.MouseButton.RightButton, pos=self.point(4000))
        self.window.player.seek.assert_not_called()
        self.assertFalse(self.times)
        p = self.point(4000)
        raw = round(self.wave.ms_of(p.x()))
        self.wave.keyframes = np.array([raw + 10])
        shift = Qt.KeyboardModifier.ShiftModifier
        QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, shift, pos=p)
        QTest.mouseClick(self.wave, Qt.MouseButton.RightButton, shift, pos=p)
        QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, shift | Qt.KeyboardModifier.AltModifier, pos=p)
        self.assertEqual(self.times, [(raw + 10, "start"), (raw + 10, "end"), (raw, "start")])

    def test_mode_persistence_and_help(self):
        self.window.timing_actions["select"].trigger()
        self.assertEqual(self.settings.value("wave/timing_mode"), "select")
        other = mw.MainWindow()
        try:
            self.assertEqual(other.wave.timing_mode, "select")
            self.assertTrue(other.timing_actions["select"].isChecked())
            self.assertIn("Alt=不吸附", other.wave_panel.hint.text())
            self.assertEqual(other.timing_menu.help_anchor, "timing")
        finally:
            other.close()
            other.deleteLater()
            self.app.processEvents()

    def test_f10_bar_selection_and_page_keys_both_modes(self):
        self.window.set_watch_mode(True, pinned=True)
        self.app.processEvents()
        for mode in ("aegisub", "select"):
            self.window._set_timing_mode(mode)
            self.window.select_group("a")
            self.wave.t0, self.wave.span = 0, 10000
            self.wave.repaint()
            QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, pos=self.point(6000, 10))
            self.assertEqual(self.window.watch.selected, "b")
            self.window.player.pause.assert_called()
            self.window.player.seek.assert_called_with(5.0)
            QTest.keyClick(self.wave, Qt.Key.Key_PageUp)
            self.assertEqual(self.window.current_gid, "a")
            QTest.keyClick(self.wave, Qt.Key.Key_PageDown)
            self.assertEqual(self.window.current_gid, "b")
            self.assertIn("PgUp/PgDn=上一组/下一组", self.window.wave_panel.hint.text())
            self.assertGreater(self.window.wave_panel.hint.width(), 0)
            self.assertGreater(self.window.wave_panel.hint.height(), 0)
        QTest.qWait(200)
        self.assertEqual(self.window.editor.group.id, "b")
        self.group("song", "song", start=8000)
        QTest.keyClick(self.wave, Qt.Key.Key_PageDown)
        self.assertEqual(self.window.current_gid, "song")
        QTest.keyClick(self.wave, Qt.Key.Key_PageUp)
        self.assertEqual(self.window.current_gid, "b")


if __name__ == "__main__":
    unittest.main()
