"""KE: Qt offscreen layouts and input, isolated settings, no native media player."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tests import test_convert_ui as fixtures
from PySide6.QtCore import QPoint, QEvent, Qt, qVersion
from PySide6.QtGui import QFontDatabase, QMouseEvent
from PySide6.QtTest import QTest
from subkit.core import karaoke as kara
from subkit.model.document import Run


class KaraokeMode(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.window.player.play = Mock()
        self.window.player.pause = Mock()
        self.window.player._hide_timer = Mock()
        self.group("dialogue", start=1000)
        translation = self.group("song", "song", text="甲乙", start=1200)
        original = copy.deepcopy(translation)
        original.id, original.lane, original.lang = "original", "original", "ja"
        original.runs = [Run("あい", "text", ins="original", a=0, b=2)]
        self.doc.members[original.id] = original
        self.doc.group("song").members.append(original.id)
        self.doc.structure_changed()
        for member, ranges in ((original, ((1200, 1800), (2000, 2600))),
                               (translation, ((1300, 1600), (2100, 2800)))):
            beats = kara.derive(self.doc, member)
            for beat, (start, end) in zip(beats, ranges):
                beat.t0, beat.t1, beat.by = start, end, "tap"
            self.session.editor.set_karaoke(member.id, beats, twins=False)
        self.window.select_group("song")
        self.window.resize(1920, 1040)
        self.window.show()
        self.app.processEvents()

    def click(self, ms, y, button):
        wave = self.window.wave
        pt = QPoint(round(wave.x_of(ms)), round(y))
        QTest.mouseClick(wave, button, Qt.KeyboardModifier.NoModifier, pt)
        self.app.processEvents()

    def test_layout_geometry_filter_selection_zoom_and_toggle(self):
        w = self.window
        w.set_edit_layout("columns")
        w.filter_box.setCurrentText("只看对白")
        w.wave.t0, w.wave.span = 1000, 4000
        selected = ("original", w.wave.kviews["original"][1].beats[0].key)
        w.wave.k_selected = selected
        w.wave.sel = (1400, 1500)
        w.toggle_karaoke_mode()
        self.app.processEvents()
        self.assertEqual(w.edit_layout, "karaoke")
        self.assertIs(w.table.parentWidget(), w.top_split)
        self.assertIs(w.wave_panel.parentWidget(), w.right_split)
        self.assertEqual(w.main_split.orientation(), Qt.Orientation.Horizontal)
        self.assertGreater(w.wave_panel.height(), w.editor.height() * 2)
        self.assertGreater(w.wave.height(), 300)
        self.assertEqual(w.wave._k_rows()["original"][1] - w.wave._k_rows()["original"][0], 40)
        self.assertFalse(w.dock_source.isVisible())
        self.assertFalse(w.people.isVisible())
        self.assertFalse(w.editor.fmt.isVisible())
        self.assertFalse(w.editor.position.isVisible())
        self.assertEqual(w.editor.speaker_combos, {})
        self.assertTrue(w.editor.jp_edit.isVisible())
        self.assertTrue(w.editor.cn_edits[0][1].isVisible())
        self.assertTrue(w.editor.ruby.isVisible())
        self.assertEqual(w.filter_box.currentText(), "只看歌曲")
        self.assertEqual(w.proxy.rowCount(), 1)
        self.assertFalse(w.filter_box.isEnabled())
        self.assertEqual(w.current_gid, "song")
        self.assertEqual((w.wave.t0, w.wave.span, w.wave.sel, w.wave.k_selected),
                         (1000, 4000, (1400, 1500), selected))
        w.wave.grab()
        self.assertEqual({bar[2] for bar in w.wave._bars}, {"song"})
        self.assertEqual(w._group_at(1500).id, "song")
        self.assertIsNone(w._group_at(1100))
        w.select_group("dialogue")
        self.assertEqual(w.current_gid, "song")
        evidence = Path(os.environ["TEMP"]) / "ke-evidence"
        evidence.mkdir(exist_ok=True)
        self.assertTrue(w.grab().save(str(evidence / "karaoke.png")))
        (evidence / "geometry.json").write_text(json.dumps({"qt": qVersion(), "platform": self.app.platformName(),
            "sizes": {name: [obj.width(), obj.height()] for name, obj in
                      (("wave", w.wave), ("wave_panel", w.wave_panel), ("editor", w.editor),
                       ("list", w.table), ("video", w.player))}}, indent=2), encoding="utf-8")
        w.wave_panel.btn_karaoke.click()
        self.app.processEvents()
        self.assertEqual(w.edit_layout, "columns")
        self.assertEqual(w.filter_box.currentText(), "只看对白")
        self.assertEqual(w.current_gid, "song")
        self.assertEqual(w.wave.span, 4000)
        self.assertEqual(w.wave.sel, (1400, 1500))
        self.assertEqual(w.wave.k_selected, selected)
        self.assertTrue(w.editor.fmt.isVisible())

    def test_left_selects_and_seeks_right_plays_only_beats(self):
        w = self.window
        w.set_edit_layout("karaoke")
        self.app.processEvents()
        w.wave.t0, w.wave.span = 0, 4000
        timing, edited = [], []
        w.wave.timeClicked.connect(lambda *args: timing.append(args))
        w.wave.memberTimeEdited.connect(lambda *args: edited.append(args))
        for display in ("waveform", "spectrogram"):
            w.wave.display_mode = display
            for button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                rows = w.wave._k_rows()
                for y, expected in ((sum(rows["original"]) / 2, (1.2, 1.8)),
                                    (sum(rows["translation"]) / 2, (1.3, 1.6)),
                                    (w.wave.height() / 2, (1.2, 1.8))):
                    calls = w.player.play.call_count
                    selected = []
                    w.wave.groupClicked.connect(selected.append)
                    self.click(1500, y, button)
                    self.assertAlmostEqual(w.player.seek.call_args.args[0], expected[0])
                    if button == Qt.MouseButton.LeftButton:
                        self.assertEqual(selected, ["song"])
                        self.assertEqual(w.current_gid, "song")
                        self.assertEqual(w.player.play.call_count, calls)
                        self.assertIsNone(w._stop_at)
                    else:
                        self.assertAlmostEqual(w._stop_at, expected[1])
                        w._on_time(expected[1])
                        w.player.pause.assert_called()
                    w.wave.groupClicked.disconnect(selected.append)
                for ms, y in ((1900, w.wave.height() / 2), (3500, w.wave.height() / 2),
                              (3500, 12), (1000, w.wave.height() - 30),
                              (3500, w.wave.height() - 5)):
                    calls = w.player.play.call_count
                    seeks = w.player.seek.call_count
                    self.click(ms, y, button)
                    self.assertEqual(w.player.play.call_count, calls)
                    if button == Qt.MouseButton.LeftButton:
                        self.assertAlmostEqual(w.player.seek.call_args.args[0], ms / 1000, delta=.01)
                    else:
                        self.assertEqual(w.player.seek.call_count, seeks)
        self.assertEqual(timing, [])
        self.assertEqual(edited, [])

    def test_left_bars_select_other_song_group_without_playing(self):
        w = self.window
        translation = self.group("other", "song", start=4000)
        original = copy.deepcopy(translation)
        original.id, original.lane = "other_original", "original"
        self.doc.members[original.id] = original
        self.doc.group("other").members.append(original.id)
        self.doc.structure_changed()
        w.set_edit_layout("karaoke")
        self.app.processEvents()
        for lane_y in (12, w.wave.height() - 30):
            w.select_group("song")
            w.wave.t0, w.wave.span = 0, 8000
            w.wave.grab()
            calls = w.player.play.call_count
            self.click(4500, lane_y, Qt.MouseButton.LeftButton)
            self.assertEqual(w.current_gid, "other")
            self.assertEqual(w.player.play.call_count, calls)
            self.assertAlmostEqual(w.player.seek.call_args.args[0], 4.0)
        self.assertEqual("左键=选中/跳转  右键=播放这一拍  拖边界=调拍  Alt=不吸附", w.wave_panel.hint.text())

    def test_left_double_click_never_plays_in_karaoke_and_normal_layout_still_plays(self):
        w = self.window
        for layout in ("karaoke", "video"):
            w.set_edit_layout(layout)
            self.app.processEvents()
            w.wave.t0, w.wave.span = 0, 4000
            y = sum(w.wave._k_rows()["original"]) / 2
            calls = w.player.play.call_count
            self.click(1500, y, Qt.MouseButton.LeftButton)
            if layout == "karaoke":
                self.assertEqual(w.player.play.call_count, calls)
                pt = QPoint(round(w.wave.x_of(1500)), round(y))
                QTest.mouseDClick(w.wave, Qt.MouseButton.LeftButton, pos=pt)
                self.assertEqual(w.player.play.call_count, calls)
            else:
                self.assertEqual(w.player.play.call_count, calls + 1)
                self.assertAlmostEqual(w._stop_at, 1.8)

    def test_label_chips_contrast_on_bright_spectrum_and_waveform(self):
        # Windows offscreen starts without system fonts; register them for this screenshot only.
        if not QFontDatabase.families() and os.name == "nt":
            for filename in ("YuGothR.ttc", "segoeui.ttf"):
                path = Path(os.environ["WINDIR"]) / "Fonts" / filename
                if path.exists():
                    font_id = QFontDatabase.addApplicationFont(str(path))
                    if font_id >= 0:
                        self.addCleanup(QFontDatabase.removeApplicationFont, font_id)
        w = self.window
        w.set_edit_layout("karaoke")
        self.app.processEvents()
        wave = w.wave
        wave.t0, wave.span = 0, 4000
        wave.spectrogram = np.full((400, 64), 255, dtype=np.uint8)
        wave.mins, wave.maxs = np.zeros(400), np.zeros(400)
        evidence = Path(os.environ["TEMP"]) / "w2-evidence"
        evidence.mkdir(exist_ok=True)
        measurements = []
        for display in ("spectrogram", "waveform"):
            wave.display_mode = display
            with patch.object(wave, "_paint_beat_label", lambda *args: None):
                underneath = wave.grab().toImage()
            rendered = wave.grab().toImage()
            self.assertTrue(rendered.save(str(evidence / (display + "-labels.png"))))
            for lane, (y0, y1) in wave._k_rows().items():
                beat = wave.kviews[lane][1].beats[0]
                x = round(wave.x_of((beat.t0 + beat.t1) / 2))
                y = round((y0 + y1) / 2)
                pixels = [rendered.pixelColor(xx, yy) for xx in range(x - 10, x + 11)
                          for yy in range(y - 9, y + 10)]
                dark = sum(c.red() < 100 and c.green() < 100 and c.blue() < 100 for c in pixels)
                light = sum(min(c.red(), c.green(), c.blue()) > 170 for c in pixels)
                self.assertGreater(dark, 40)
                self.assertGreater(light, 3)
                if display == "spectrogram":
                    changed = sum(rendered.pixelColor(xx, yy).green() < underneath.pixelColor(xx, yy).green() - 70
                                  for xx in range(x - 10, x + 11) for yy in range(y - 9, y + 10))
                    self.assertGreater(changed, 40)
                measurements.append({"display": display, "lane": lane, "dark_pixels": dark, "light_pixels": light})
        (evidence / "label-contrast.json").write_text(json.dumps({"qt": qVersion(),
            "platform": self.app.platformName(), "measurements": measurements}, indent=2), encoding="utf-8")

    def test_beat_boundary_still_drags_and_emits_one_edit(self):
        w = self.window
        w.set_edit_layout("karaoke")
        self.app.processEvents()
        w.wave.t0, w.wave.span, w.wave.snap = 0, 4000, False
        edits, timing = [], []
        w.wave.beatsEdited.connect(lambda *args: edits.append(args))
        w.wave.timeClicked.connect(lambda *args: timing.append(args))
        y = sum(w.wave._k_rows()["original"]) / 2
        a = QPoint(round(w.wave.x_of(1800)), round(y))
        b = QPoint(round(w.wave.x_of(1850)), round(y))
        QTest.mousePress(w.wave, Qt.MouseButton.LeftButton, pos=a)
        event = QMouseEvent(QEvent.Type.MouseMove, b, w.wave.mapToGlobal(b), Qt.MouseButton.NoButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        w.wave.mouseMoveEvent(event)
        QTest.mouseRelease(w.wave, Qt.MouseButton.LeftButton, pos=b)
        self.assertEqual(len(edits), 1)
        self.assertAlmostEqual(edits[0][1][0].t1, 1850, delta=5)
        self.assertEqual(timing, [])

    def test_zoomed_beat_drag_keeps_view_until_group_leaves_it(self):
        w = self.window
        w.set_edit_layout("karaoke")
        self.app.processEvents()
        w.wave.t0, w.wave.span, w.wave.snap = 1500, 1000, False   # the song (1200–2800) is wider than the view
        y = sum(w.wave._k_rows()["original"]) / 2
        a = QPoint(round(w.wave.x_of(1800)), round(y))
        b = QPoint(round(w.wave.x_of(1850)), round(y))
        QTest.mousePress(w.wave, Qt.MouseButton.LeftButton, pos=a)
        event = QMouseEvent(QEvent.Type.MouseMove, b, w.wave.mapToGlobal(b), Qt.MouseButton.NoButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        w.wave.mouseMoveEvent(event)
        QTest.mouseRelease(w.wave, Qt.MouseButton.LeftButton, pos=b)
        self.app.processEvents()
        self.assertAlmostEqual(w.wave.kviews["original"][1].beats[0].t1, 1850, delta=5)
        self.assertEqual((w.wave.t0, w.wave.span), (1500, 1000))
        w.undo()
        self.assertAlmostEqual(w.wave.kviews["original"][1].beats[0].t1, 1800, delta=5)
        self.assertEqual((w.wave.t0, w.wave.span), (1500, 1000))
        w.wave.t0 = 6000                                           # scrolled away: the next edit brings it back
        beats = list(w.wave.kviews["original"][1].beats)
        beats[0].t1 = 1820
        w._k_dragged(w.wave.kviews["original"][0], beats)
        self.assertEqual((w.wave.t0, w.wave.span), (1100, 1000))
        w.select_group("song")
        w.wave.t0 = 1500
        w.select_group("song")                                     # plain reselect still frames the group
        self.assertEqual(w.wave.t0, 1100)

    def test_f9_round_trip_with_f10_f11(self):
        w = self.window
        with patch.object(fixtures.mw.MainWindow, "restoreGeometry", lambda widget, data: widget.resize(1920, 1040)):
            for layout in ("video", "list", "columns"):
                w.set_edit_layout(layout)
                w.wave.span = 4000
                w.activateWindow()
                w.player.setFocus()
                self.app.processEvents()
                QTest.keyClick(w.player, Qt.Key.Key_F9)
                self.app.processEvents()
                self.assertEqual(w.edit_layout, "karaoke")
                for key in (Qt.Key.Key_F10, Qt.Key.Key_F11, Qt.Key.Key_F10):
                    QTest.keyClick(w.player, key)
                    self.app.processEvents()
                    self.assertTrue(w.watch_mode)
                QTest.keyClick(w.player, Qt.Key.Key_Tab)
                self.app.processEvents()
                self.assertFalse(w.watch_mode)
                self.assertEqual(w.edit_layout, "karaoke")
                self.assertTrue(w.wave.karaoke_mode)
                self.assertFalse(w.dock_source.isVisible())
                self.assertEqual(w.wave.span, 4000)
                QTest.keyClick(w.player, Qt.Key.Key_F9)
                self.app.processEvents()
                self.assertEqual(w.edit_layout, layout)

    def test_speed_group_switch_settings_restart_and_file_loaded(self):
        w = self.window
        w.player.mpv = Mock()
        w.player.mpv.get.return_value = None
        w._set_speed(.5)
        self.assertEqual(w.wave_panel.speed_label.text(), "速度（歌曲）")
        self.assertEqual(w.lbl_speed.text(), "0.5× 播放")
        w.select_group("dialogue")
        self.assertEqual(w.wave_panel.speed.currentData(), 1.0)
        self.assertEqual(w.lbl_speed.text(), "")
        w._set_speed(.75)
        w.select_group("song", defer=True)
        self.assertEqual(w.wave_panel.speed.currentData(), .5)
        w.player.mpv.set.assert_called_with("speed", "0.5")
        w._flush_group_selection()
        w.select_group("dialogue")
        self.assertEqual(w.wave_panel.speed_label.text(), "速度（对白）")
        self.assertEqual(w.wave_panel.speed.currentData(), .75)
        w.set_edit_layout("karaoke")
        self.assertEqual(w.wave_panel.speed.currentData(), .5)
        self.assertIs(w.wave_panel.speed.parentWidget(), w.wave_panel.wave_groups[1])
        w.set_edit_layout("video")
        self.settings.sync()
        restarted = fixtures.mw.MainWindow()
        try:
            self.assertEqual(restarted._speeds, {"song": .5, "dialogue": .75})
            restarted.player.mpv = Mock()
            restarted.player.fileLoaded.emit()
            restarted.player.mpv.set.assert_called_with("speed", "0.75")
            self.assertTrue(restarted.wave_panel.speed.isVisibleTo(restarted.wave_panel))
        finally:
            restarted.close()
            restarted.deleteLater()

    def test_enter_from_dialogue_and_no_song(self):
        w = self.window
        w.select_group("dialogue")
        w.set_edit_layout("karaoke")
        self.assertEqual(w.current_gid, "song")
        w.set_edit_layout("video")
        self.doc.groups = [self.doc.group("dialogue")]
        self.doc.structure_changed()
        w.model.reset()
        w.select_group("dialogue")
        w.set_edit_layout("karaoke")
        self.assertIsNone(w.current_gid)
        self.assertEqual(w.wave.kviews, {})
        self.assertEqual(w.proxy.rowCount(), 0)
        self.assertEqual(w._speed_kind(), "song")


if __name__ == "__main__":
    unittest.main()
