"""Width ceilings, legacy output and offscreen project settings."""

import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication, QDialog, QLabel
from subkit.ass.document import AssDocument, Event, Style
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.workspace import Project
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
from subkit.ruby.layout import Item, layout_line
from subkit.ruby.measure import Box
from subkit_app.mainwindow import MainWindow
from subkit_app.views.ruby_settings import RubySettingsDialog


def job_layout(percent=150, *, scale=None, text="妹", width=96, base_width=48):
    doc = AssDocument()
    doc.styles = [Style.new("Default", Fontsize=48), Style.new("Default.Ruby", Fontsize=24, Outline=4)]
    measurer = SimpleNamespace(playres_x=1920, playres_y=1080, anchor_x=lambda *args: 960)
    gen = RubyGenerator(doc, measurer)
    gen._metrics = {"Default": (60, 5, 50), "Default.Ruby": (30, 2, 25)}
    job = RubyJob(Event(style="Default", text=text), "Default.Ruby",
                  [RubyPiece(0, len(text), "いもうと", scale=scale)], max_width_percent=percent)
    boxes = [Box(960-base_width/2+i*base_width/len(text), 960-base_width/2+(i+1)*base_width/len(text), 5, 50)
             for i in range(len(text))]
    gen._layout_job(job, boxes, [Box(0, width, 0, 24)])
    return job


class WidthLayout(unittest.TestCase):
    def test_single_kanji_four_kana_percentages_and_floor_warning(self):
        for percent, scale in ((100, 60), (120, 60), (150, 75), (200, 100)):
            with self.subTest(percent=percent):
                j = job_layout(percent)
                self.assertAlmostEqual(j.placements[0][2], scale)
                self.assertEqual(j.warnings, ["振假名太长，压到下限仍是汉字占位宽度的 120%：妹（いもうと）"] if percent == 100 else [])
        self.assertIn("\\fscx75", job_layout().out[0].text)

    def test_multi_character_word_does_not_compress(self):
        self.assertEqual(job_layout(150, text="姉妹", base_width=96).placements[0][2], 100)

    def test_manual_scale_is_unchanged_even_below_minimum(self):
        for scale in (42, 125):
            j = job_layout(100, scale=scale)
            self.assertEqual(j.placements[0][2], scale)
            self.assertFalse(any("振假名太长" in w for w in j.warnings))

    def test_collision_only_compresses_further_and_never_below_floor(self):
        items = [Item(0, 48, 96, max_scale=75), Item(48, 96, 96, max_scale=90)]
        layout_line(items)
        self.assertTrue(all(60 <= i.scale <= cap for i, cap in zip(items, (75, 90))))
        self.assertLess(items[0].scale, 75)
        self.assertLessEqual(items[0].x+items[0].w()/2+2, items[1].x-items[1].w()/2+0.5)

    def test_rounding_never_rounds_capped_output_up(self):
        j = job_layout(150, width=95)
        self.assertIn("\\fscx75.78", j.out[0].text)

    def test_disabled_legacy_bytes(self):
        j = job_layout(None)
        self.assertEqual(j.out[0].text.encode("utf-8"), b"{\\an2\\pos(960.0,1018.0)}"+"いもうと".encode("utf-8"))
        self.assertEqual(j.placements[0][2], 100)


class WidthExport(unittest.TestCase):
    def test_old_settings_defaults_and_independent_settings(self):
        project = Project(Path("unused"), {"settings": {"ruby": {"min_scale": 65}}})
        ruby = project.settings()["ruby"]
        self.assertEqual(ruby["max_width"], {"dialogue": {"enabled": True, "percent": 150},
                                            "song": {"enabled": True, "percent": 150}})
        opts = ExportOptions(ruby_max_width={"dialogue": {"enabled": False}, "song": {"percent": 120}})
        self.assertIsNone(opts.ruby_width_percent("dialogue"))
        self.assertEqual(opts.ruby_width_percent("song"), 120)
        self.assertNotEqual(opts.key(), ExportOptions().key())

    def test_real_libass_export_cache_song_fx_and_ktv(self):
        from tests.test_karaoke_fx import fixture, ktv_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for effect in ("kf", "pop", "ktv"):
                doc, template = ktv_fixture(ruby=True) if effect == "ktv" else fixture(effect, ruby=True)
                for ann in doc.annotations.values():
                    ann.text = "いもうと"
                    ann.parts = [[0, 1, "いもうと"]]
                disabled = {"dialogue": {"enabled": False}, "song": {"enabled": False}}
                song = {"dialogue": {"enabled": False}, "song": {"enabled": True, "percent": 100}}
                cache = ExportCache()
                before, _ = build_ass(doc, {}, template, root, ExportOptions(ruby_max_width=disabled),
                                      cache=cache, measure_cache=root/"measure.sqlite3")
                after, report = build_ass(doc, {}, template, root, ExportOptions(ruby_max_width=song),
                                          cache=cache, measure_cache=root/"measure.sqlite3")
                fresh, _ = build_ass(doc, {}, template, root, ExportOptions(ruby_max_width=song),
                                     measure_cache=root/"measure.sqlite3")
                self.assertEqual(after.serialize(), fresh.serialize())
                self.assertNotEqual(before.serialize(), after.serialize(), effect)
                self.assertGreater(report.laid_out, 0)


class WidthDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_labels_defaults_custom_cancel_and_help(self):
        dialog = RubySettingsDialog({"min_scale": 65})
        try:
            self.assertEqual(dialog.windowTitle(), "振假名设置")
            self.assertEqual(dialog.help_anchor, "readings")
            self.assertIn("按所注汉字的排版占位宽度限制振假名，计入字体、字号、字距和横向缩放；超过上限时横向压缩，最多压到 65%。保留居中、相邻振假名避让和人工缩放，F11 与 ASS 导出使用相同规则。",
                          [x.text() for x in dialog.findChildren(QLabel)])
            enabled, combo = dialog.controls["song"]
            self.assertEqual([combo.itemText(i) for i in range(5)], ["100%（不超出汉字）", "120%（紧凑）", "150%（推荐）", "200%（宽松）", "自定义…"])
            combo.setCurrentIndex(4)
            with patch("subkit_app.views.ruby_settings.QInputDialog.getInt", return_value=(175, True)) as prompt:
                dialog._choose_percent(combo, 4)
            self.assertEqual(prompt.call_args.args[-2:], (100, 300))
            self.assertEqual(dialog.values()["song"]["percent"], 175)
            with patch("subkit_app.views.ruby_settings.QInputDialog.getInt", return_value=(0, False)):
                dialog._choose_percent(combo, 4)
            self.assertEqual(dialog.values()["song"]["percent"], 175)
            enabled.setChecked(False)
            self.assertFalse(combo.isEnabled())
            self.assertEqual(dialog.values()["dialogue"], {"enabled": True, "percent": 150})
        finally:
            dialog.close()
            dialog.deleteLater()
            self.app.processEvents()

    def test_ok_saves_and_requests_preview_cancel_does_neither(self):
        project = Mock(data={"settings": {}}, settings=Mock(return_value={"ruby": {}}))
        window = SimpleNamespace(project=project, session=Mock())
        dialog = Mock(DialogCode=QDialog.DialogCode, values=Mock(return_value={"song": {"percent": 120}}))
        with patch("subkit_app.mainwindow.RubySettingsDialog", return_value=dialog):
            dialog.exec.return_value = QDialog.DialogCode.Rejected
            MainWindow._ruby_settings(window)
            project.save.assert_not_called()
            window.session.request_preview.assert_not_called()
            dialog.exec.return_value = QDialog.DialogCode.Accepted
            MainWindow._ruby_settings(window)
            project.save.assert_called_once()
            window.session.request_preview.assert_called_once()
            self.assertEqual(project.data["settings"]["ruby"]["max_width"], dialog.values())


if __name__ == "__main__":
    unittest.main()
