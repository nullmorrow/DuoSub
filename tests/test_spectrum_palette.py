"""Spectrum colour maps: calm default, menu choice remembered per machine."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subkit_app.views import waveform as wf


class SpectrumPalette(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def panel(self, settings):
        view = wf.WaveformView()
        panel = wf.WaveformPanel(view)
        self.addCleanup(panel.deleteLater)
        panel.set_settings(settings)
        return panel, view

    def test_default_choice_persistence_and_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder) / "s.ini"), QSettings.Format.IniFormat)
            panel, view = self.panel(settings)
            self.assertEqual(view.spectrum_palette, "deep_sea")
            self.assertTrue(panel.palette_actions["deep_sea"].isChecked())
            self.assertEqual([a.text() for a in panel.palette_menu.actions()][:len(wf.SPECTRUM_PALETTES)],
                             [label for label, _ in wf.SPECTRUM_PALETTES.values()])
            panel.palette_actions["magma"].trigger()
            self.assertEqual((view.spectrum_palette, settings.value("waveform/spectrum_palette")), ("magma", "magma"))
            _, again = self.panel(settings)
            self.assertEqual(again.spectrum_palette, "magma")
            settings.setValue("waveform/spectrum_palette", "turbo-pascal")
            _, fallback = self.panel(settings)
            self.assertEqual(fallback.spectrum_palette, "deep_sea")
            settings.sync()

    def test_tables_dark_low_end_and_classic_kept(self):
        for key in wf.SPECTRUM_PALETTES:
            table = wf.spectrum_color_table(key)
            self.assertEqual(len(table), 256)
        low = wf.spectrum_color_table("deep_sea")[0] & 0xFFFFFF
        self.assertLess(sum((low >> s) & 0xFF for s in (0, 8, 16)), 3 * 40)      # quiet parts stay dark
        self.assertEqual(wf.spectrum_color_table("unknown"), wf.spectrum_color_table("deep_sea"))


    def test_contrast_brightness_sliders_persist_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder) / "s.ini"), QSettings.Format.IniFormat)
            panel, view = self.panel(settings)
            self.assertEqual((view.spectrum_contrast, view.spectrum_brightness), (50, 60))
            panel.spectrum_sliders["contrast"].setValue(70)
            panel.spectrum_sliders["brightness"].setValue(20)
            self.assertEqual((view.spectrum_contrast, view.spectrum_brightness), (70, 20))
            _, again = self.panel(settings)
            self.assertEqual((again.spectrum_contrast, again.spectrum_brightness), (70, 20))
            reset = next(a for a in panel.palette_menu.actions() if a.text() == "恢复默认对比度和亮度")
            reset.trigger()
            self.assertEqual((view.spectrum_contrast, view.spectrum_brightness), (50, 60))
            self.assertEqual(settings.value("waveform/spectrum_contrast", type=int), 50)
            settings.sync()

    def test_levels_hide_noise_floor_and_lift_mid_tones(self):
        flat = wf.spectrum_levels(0, 0)
        self.assertEqual((round(flat[0]), round(flat[128]), round(flat[255])), (0, 128, 255))
        adjusted = wf.spectrum_levels(50, 60)
        self.assertEqual(adjusted[40], 0)                         # weak background becomes the darkest colour
        self.assertGreater(adjusted[180], flat[180])               # speech-strength values get brighter
        self.assertEqual(round(adjusted[255]), 255)

if __name__ == "__main__":
    unittest.main()
