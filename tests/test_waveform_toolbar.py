"""KA toolbars, selection zoom and ruler; Qt offscreen, no native player."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
from PySide6.QtWidgets import QApplication, QToolButton
from subkit_app.views.waveform import WaveformPanel, WaveformView, ruler_ticks


class CountingPanel(WaveformPanel):
    def __init__(self, view):
        self.zoom_calls = self.sync_calls = 0
        super().__init__(view)

    def _zoom_changed(self, value):
        self.zoom_calls += 1
        super()._zoom_changed(value)

    def _sync(self):
        self.sync_calls += 1
        super()._sync()


class WaveformToolbar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("KA tests require offscreen Qt")

    def setUp(self):
        self.view = WaveformView()
        self.panel = CountingPanel(self.view)
        self.addCleanup(self.dispose)
        self.kview = SimpleNamespace(state="timed", beats=[1, 2], issues=[], unit="syllable", kind="kf")

    def dispose(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_exclusive_display_buttons_persist_both_modes(self):
        with tempfile.TemporaryDirectory(prefix="ka_settings_") as tmp:
            path = str(Path(tmp) / "settings.ini")
            settings = QSettings(path, QSettings.Format.IniFormat)
            self.panel.set_settings(settings)
            for button, mode in ((self.panel.btn_spectrum, "spectrogram"), (self.panel.btn_waveform, "waveform")):
                button.click()
                self.assertEqual(self.view.display_mode, mode)
                self.assertTrue(button.isChecked())
                self.assertEqual(sum(b.isChecked() for b in self.panel.mode_buttons.buttons()), 1)
                settings.sync()
                restored = WaveformPanel(WaveformView())
                try:
                    restored.set_settings(QSettings(path, QSettings.Format.IniFormat))
                    self.assertEqual(restored.view.display_mode, mode)
                    self.assertTrue(restored.mode_buttons.button(1 if mode == "spectrogram" else 0).isChecked())
                finally:
                    restored.close()
                    restored.deleteLater()
                    self.app.processEvents()

    def test_toolbars_wrap_at_600_without_clipping_any_button(self):
        self.panel.show_k(self.kview, has_tags=True)
        self.panel.resize(600, 900)
        self.panel.show()
        self.app.processEvents()
        self.assertEqual(self.panel.width(), 600)
        for bar in (self.panel.bar, self.panel.k_controls):
            buttons = [b for b in bar.findChildren(QToolButton) if b.isVisible()]
            self.assertGreater(len({b.mapTo(bar, b.rect().topLeft()).y() for b in buttons}), 1)
            for button in buttons:
                with self.subTest(text=button.text()):
                    self.assertGreaterEqual(button.width(), button.sizeHint().width())
                    self.assertGreaterEqual(button.height(), button.sizeHint().height())
                    self.assertGreaterEqual(button.x(), 0)
                    self.assertLessEqual(button.geometry().right(), bar.width())
                    self.assertLessEqual(button.geometry().bottom(), bar.height())
        wide_h = self.panel.k_controls.heightForWidth(1200)
        narrow_h = self.panel.k_controls.heightForWidth(600)
        self.assertGreater(narrow_h, wide_h)
        self.panel.show_k(self.kview, has_tags=False)
        self.app.processEvents()
        self.assertFalse(self.panel.btn_kimport.isVisible())

    def test_repeated_show_k_has_one_zoom_amp_follow_and_sync_connection(self):
        for _ in range(6):
            self.panel.show_k(self.kview, has_tags=True)
            self.panel.show_k(None, has_tags=False)
        self.panel.zoom_calls = self.panel.sync_calls = 0
        self.panel.zoom.setValue(30)
        self.assertEqual(self.panel.zoom_calls, 1)
        self.assertEqual(self.panel.sync_calls, 1)
        self.view.set_amp = Mock(wraps=self.view.set_amp)
        self.panel.amp.setValue(40)
        self.view.set_amp.assert_called_once()
        self.panel.btn_follow.setChecked(False)
        self.assertFalse(self.view.follow)
        self.panel.btn_follow.setChecked(True)
        self.assertTrue(self.view.follow)

    def test_zoom_controls_work_before_first_song_selection(self):
        before = self.view.span
        self.panel.zoom.setValue(20)
        self.assertNotEqual(before, self.view.span)
        self.assertEqual(self.panel.zoom_calls, 1)

    def test_settings_menu_mirrors_controls_and_shared_k_checks(self):
        p = self.panel
        self.assertEqual([a.text() for a in p.settings_menu.actions()],
                         ["振幅", "频谱配色", "打轴对象", "吸附", "吸附到提示线", "建议提前量", "跟随播放", "钉住", "重新读取音频"])
        for button in (p.btn_snap, p.btn_onset, p.btn_follow, p.btn_pin):
            action = p.menu_actions[button]
            before = button.isChecked()
            action.trigger()
            self.assertEqual(button.isChecked(), not before)
            self.assertEqual(action.isChecked(), button.isChecked())
            button.setChecked(before)
            self.assertEqual(action.isChecked(), before)
        self.assertEqual(self.view.snap_onset, p.btn_khintsnap.isChecked())
        p.btn_pin.blockSignals(True)
        p.btn_pin.setChecked(True)
        p.btn_pin.blockSignals(False)
        p.settings_menu.aboutToShow.emit()
        self.assertTrue(p.menu_actions[p.btn_pin].isChecked())
        with patch.object(self.view, "load") as load:
            p.menu_actions[p.btn_retry].trigger()
            # No video loaded: retry has nothing to decode.
            load.assert_not_called()
        self.assertEqual(p.menu_actions[p.btn_retry].property("help_anchor"), "audio-read")

    def test_padding_radio_modes_fixed_cancel_recalculate_and_persistence(self):
        p = self.panel
        p.project = SimpleNamespace(data={}, save=Mock())
        actions = [a for a in p.padding_menu.actions() if not a.isSeparator()]
        self.assertEqual([a.text() for a in actions],
                         ["按我调过的行（推荐）", "按本集字幕", "同自动调轴设置", "固定值…", "重新计算"])
        for index, mode in enumerate(("manual", "episode", "autotime")):
            actions[index].trigger()
            self.assertEqual(p.padding_mode.currentData(), mode)
            self.assertEqual(p.project.data["settings"]["hint_padding"]["mode"], mode)
            self.assertEqual(sum(a.isChecked() for a in actions[:4]), 1)
        with patch.object(p, "_fixed_padding_dialog", return_value=None):
            actions[3].trigger()
        self.assertEqual(p.padding_mode.currentData(), "autotime")
        self.assertTrue(actions[2].isChecked())
        with patch.object(p, "_fixed_padding_dialog", return_value=(123, -45)):
            actions[3].trigger()
        self.assertEqual((self.view.lead_in_ms, self.view.lead_out_ms), (123, -45))
        self.assertEqual(p.project.data["settings"]["hint_padding"]["fixed"], [123, -45])
        p.padding_menu.aboutToShow.emit()
        self.assertFalse(actions[4].isEnabled())
        actions[0].trigger()
        p.padding_menu.aboutToShow.emit()
        self.assertTrue(actions[4].isEnabled())
        recalc = Mock()
        p.btn_recalibrate.clicked.connect(recalc)
        actions[4].trigger()
        recalc.assert_called_once()

    def test_onset_menu_uses_existing_setting_and_restores(self):
        with tempfile.TemporaryDirectory(prefix="x7_settings_") as tmp:
            settings = QSettings(str(Path(tmp) / "settings.ini"), QSettings.Format.IniFormat)
            self.panel.set_settings(settings)
            self.panel.menu_actions[self.panel.btn_onset].trigger()
            self.assertTrue(settings.value("karaoke/snap_onset", type=bool))
            restored = WaveformPanel(WaveformView())
            try:
                restored.set_settings(settings)
                self.assertTrue(restored.menu_actions[restored.btn_onset].isChecked())
                self.assertTrue(restored.btn_khintsnap.isChecked())
            finally:
                restored.deleteLater()
                self.app.processEvents()

    def test_more_menu_updates_disabled_reasons_import_and_project_presets(self):
        p = self.panel
        self.assertEqual([a.text() for a in p.more_menu.actions()], ["歌词", "特效"])
        lyrics = p.more_menu.actions()[0].menu()
        p.set_mapping_reason("（这一组没有原声（日语））")
        p.set_english_available(False)
        p.show_k(self.kview, has_tags=False)
        lyrics.aboutToShow.emit()
        self.assertEqual(lyrics.actions()[0].text(), "对应…（这一组没有原声（日语））")
        self.assertFalse(lyrics.actions()[0].isEnabled())
        self.assertFalse(lyrics.actions()[1].isEnabled())
        self.assertFalse(lyrics.actions()[2].isVisible())
        p.show_k(self.kview, has_tags=True)
        p.set_mapping_reason("")
        lyrics.aboutToShow.emit()
        emit = Mock()
        p.kAction.connect(emit)
        lyrics.actions()[0].trigger()
        lyrics.actions()[2].trigger()
        self.assertEqual([c.args for c in emit.call_args_list], [("mapping",), ("import",)])
        p.set_fx_presets([{"name": "我的预设"}])
        fx_menu = p.more_menu.actions()[1].menu()
        fx_menu.aboutToShow.emit()
        self.assertEqual(fx_menu.actions()[-1].text(), "参数…")
        preset = next(a for a in fx_menu.actions() if a.text() == "我的预设")
        effect = Mock()
        p.kEffect.connect(effect)
        preset.trigger()
        effect.assert_called_once_with("preset:我的预设")
        self.assertEqual(p.menu_actions[p.btn_kparams].property("help_anchor"), "karaoke-fx")
        p.set_fx_presets([])
        fx_menu.aboutToShow.emit()
        self.assertNotIn("我的预设", [a.text() for a in fx_menu.actions()])

    def test_900px_bar_only_contains_common_controls_and_settings(self):
        p = self.panel
        p.resize(900, 500)
        p.show()
        self.app.processEvents()
        buttons = [b for b in p.bar.findChildren(QToolButton) if b.isVisible()]
        self.assertEqual([b.text() for b in buttons],
                         ["波形", "频谱", "适合本组", "逐段播放", "显示提示线", "采用建议时间", "设置 ▾", "?"])
        self.assertLessEqual(len({g.y() for g in p.wave_groups}), 2)
        for button in (p.btn_onset, p.btn_snap, p.btn_follow, p.btn_pin, p.btn_retry):
            self.assertFalse(button.isVisible())
        p.show_k(self.kview, has_tags=True)
        self.app.processEvents()
        for widget in (p.btn_kmap, p.btn_kenglish, p.btn_kimport, p.k_fx, p.btn_kparams):
            self.assertFalse(widget.isVisible())

    def test_beat_chip_dark_backdrop_white_glyphs_and_cell_clipping(self):
        image = QImage(200, 80, QImage.Format.Format_ARGB32)
        image.fill(QColor("#ffff00"))
        painter = QPainter(image)
        painter.setFont(QFont("Yu Gothic UI", 9))
        self.view._paint_beat_label(painter, QRectF(20, 10, 160, 50), "かな")
        painter.end()
        colors = [image.pixelColor(x, y) for x in range(20, 180) for y in range(10, 60)]
        self.assertTrue(any(c.red() > 245 and c.green() > 245 and c.blue() > 245 for c in colors))
        self.assertTrue(any(c.red() < 45 and c.green() < 50 and 15 < c.blue() < 30 for c in colors))
        self.assertEqual(image.pixelColor(19, 30).name(), "#ffff00")

    def test_menu_help_target_selection_and_amplitude_keep_old_entry_points(self):
        from subkit_app import help as help_ui
        p = self.panel
        for menu, action, anchor in ((p.settings_menu, p.menu_actions[p.btn_retry], "audio-read"),
                                     (p.padding_menu, p.padding_menu.actions()[0], "timing"),
                                     (p.lyrics_menu, p.menu_actions[p.btn_kmap], "karaoke"),
                                     (p.more_menu.actions()[1].menu(), p.menu_actions[p.btn_kparams], "karaoke-fx")):
            menu.setActiveAction(action)
            with patch.object(QApplication, "activePopupWidget", return_value=menu):
                self.assertEqual(help_ui.resolve_anchor(None), anchor)
        selected = Mock()
        p.target.activated.connect(selected)
        p.target_menu.actions()[2].trigger()
        selected.assert_called_once_with(2)
        self.assertEqual(p.timing_target(), "original")
        self.view.set_amp = Mock()
        p.amp.setValue(75)
        self.view.set_amp.assert_called_once_with(8.0)

    def test_show_range_preserves_span_for_short_long_and_visible_groups(self):
        for span in (1000, 4000, 12000, 120000):
            for start, length in ((20000, 200), (20000, span * 4), (0, 100), (21000, 200)):
                self.view.t0, self.view.span = 20000, float(span)
                self.view.show_range(start, start + length)
                self.assertEqual(self.view.span, span)
                self.assertLessEqual(self.view.t0, start)
                self.assertLess(start, self.view.t0 + span)
                if length > span:
                    self.assertEqual(self.view.t0, max(0, start - span * 0.1))

    def test_fit_button_uses_group_length_and_twenty_percent_each_side(self):
        self.view.current_edges = Mock(return_value=(20000, 25000))
        self.panel.btn_fit.click()
        self.assertEqual(self.view.span, 7000)
        self.assertEqual(self.view.t0, 19000)
        self.view.current_edges.return_value = None
        self.panel.btn_fit.click()
        self.assertEqual((self.view.t0, self.view.span), (19000, 7000))
        for edges, expected in (((0, 100), 1000), ((1000, 200000), 120000)):
            self.view.current_edges.return_value = edges
            self.panel.btn_fit.click()
            self.assertEqual(self.view.span, expected)
            self.assertGreaterEqual(self.view.t0, 0)


class RulerTicks(unittest.TestCase):
    def test_two_levels_no_overlap_and_at_least_ten_ms(self):
        width_of = lambda label: len(label) * 8
        for start in (0, 75620, 3599500, 12345678):
            for span in (1000, 2200, 6000, 12000, 120000):
                for width in (200, 600, 2000):
                    ticks = ruler_ticks(start, span, width, width_of)
                    self.assertTrue(ticks)
                    self.assertTrue(any(not major for _, major, _ in ticks))
                    self.assertGreaterEqual(min(b[0] - a[0] for a, b in zip(ticks, ticks[1:])), 10)
                    last_end = -8
                    for t, major, label in ticks:
                        self.assertGreaterEqual(t, start)
                        self.assertLess(t, start + span)
                        if label:
                            self.assertTrue(major)
                            x = (t - start) * width / span + 2
                            self.assertGreaterEqual(x, last_end + 6)
                            last_end = x + width_of(label)
                            self.assertLessEqual(last_end, width)

    def test_zoomed_ruler_ten_ms_and_centiseconds(self):
        ticks = ruler_ticks(75620, 1000, 2000, lambda text: len(text) * 7)
        self.assertEqual(ticks[1][0] - ticks[0][0], 10)
        self.assertIn("1:15.65", [label for _, _, label in ticks])
        self.assertEqual(ruler_ticks(0, 1000, 0, len), [])
        self.assertEqual(ruler_ticks(0, 0, 600, len), [])

    def test_real_qt_font_metrics_stay_apart(self):
        app = QApplication.instance() or QApplication([])
        metrics = QFontMetrics(QFont("Segoe UI", 8))
        for span in (1000, 6000, 120000):
            last_end = -8
            for t, _, label in ruler_ticks(3599500, span, 600, metrics.horizontalAdvance):
                if label:
                    x = (t - 3599500) * 600 / span + 2
                    self.assertGreaterEqual(x, last_end + 6)
                    last_end = x + metrics.horizontalAdvance(label)
