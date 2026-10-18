"""Selective fansub style import, shared detection and offscreen editor wiring."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton
from subkit import templates as T
from subkit.ass import AssDocument
from subkit.ass.document import Event, Style
from subkit.core.batch import source_roles
from subkit_app.views.template_editor import StyleImportDialog, TemplateEditor


def fansub(height=1080):
    doc = AssDocument()
    if height is not None:
        doc.set_info("PlayResX", 1920 if height == 1080 else 640)
        doc.set_info("PlayResY", height)
    doc.styles = [Style.new("Default", Fontname="First CN", Fontsize=99),
                  Style.new("JP", Fontname="First JP", Fontsize=98),
                  Style.new("Screen"), Style.new("Effect"), Style.new("OPJP"),
                  Style.new("Default", Fontname="Last CN", Fontsize=22, Bold=0, Spacing=1.5,
                            Outline=1.5, MarginV=24, PrimaryColour=T.ass_colour("123456"),
                            OutlineColour=T.ass_colour("ABCDEF", 37), BackColour=T.ass_colour("135790", 91)),
                  Style.new("JP", Fontname="Last JP", Fontsize=18, Bold=-1, Outline=1,
                            MarginV=8, PrimaryColour=T.ass_colour("654321"), OutlineColour=T.ass_colour("FEDCBA"))]
    doc.events = ([Event(style="Default", text=r"{\i1}這是主要對白\N下一行") for _ in range(4)] +
                  [Event(style="JP", text="ここでまっています") for _ in range(3)] +
                  [Event(style="Screen", text=r"{\pos(30,30)}看板") for _ in range(12)] +
                  [Event(style="Effect", text=r"{\move(0,0,20,20)}特效") for _ in range(15)] +
                  [Event(style="OPJP", text="そらにむかってうたおう") for _ in range(8)] +
                  [Event(kind="Comment", style="Default", text="Comment excluded")])
    return doc


class DialogueImport(unittest.TestCase):
    def setUp(self):
        self.doc = fansub()
        self.fonts = dataclasses.replace(T.font_set("utena_serif"), ruby_offset=5, pinyin_offset=-3,
                                         cn_mv=103, jp_mv=17, ruby_ratio=.63, song_cn_ratio=1.7)
        self.pal = T.palette("utena")

    def load(self, **kwargs):
        return T.import_dialogue_styles(self.fonts, self.pal, self.doc, "Default", "JP", **kwargs)

    def test_default_parts_last_definitions_and_unselected_settings(self):
        fonts, pal = self.load()
        self.assertEqual((fonts.cn_hans, fonts.cn_hant, fonts.jp), ("Last CN", "Last CN", "Last JP"))
        self.assertEqual((fonts.cn_size, fonts.jp_size, fonts.cn_bold, fonts.jp_bold, fonts.spacing),
                         (22, 18, False, True, 1.5))
        self.assertEqual((pal.width, pal.width * fonts.jp_outline_ratio), (1.5, 1))
        self.assertEqual((pal.text, pal.outline, pal.outline_alpha, pal.jp_fill, pal.jp_edge,
                          pal.shadow, pal.shadow_alpha),
                         ("123456", "ABCDEF", 37, "654321", "FEDCBA", "135790", 91))
        changed_f = {"id", "name", "cn_hans", "cn_hant", "jp", "cn_size", "jp_size", "cn_bold", "jp_bold",
                     "spacing", "jp_outline_ratio", "jp_outline_width"}
        changed_p = {"id", "name", "text", "outline", "outline_alpha", "jp_text", "jp_outline", "shadow",
                     "shadow_alpha", "width"}
        for before, after, allowed in ((self.fonts, fonts, changed_f), (self.pal, pal, changed_p)):
            for field in dataclasses.fields(before):
                if field.name not in allowed:
                    self.assertEqual(getattr(before, field.name), getattr(after, field.name), field.name)

    def test_colours_only_keeps_fonts_margins_and_widths(self):
        fonts, pal = self.load(read_fonts=False)
        self.assertEqual(fonts, self.fonts)
        self.assertEqual(pal.width, self.pal.width)
        self.assertEqual((pal.text, pal.jp_fill), ("123456", "654321"))

    def test_fonts_only_keeps_colours_and_margins(self):
        fonts, pal = self.load(read_colours=False)
        self.assertEqual((fonts.cn_mv, fonts.jp_mv), (103, 17))
        self.assertEqual(pal, dataclasses.replace(self.pal, id="custom", name="自定义", width=1.5))

    def test_margins_only_scales_without_other_changes(self):
        self.doc.set_info("PlayResY", 360)
        fonts, pal = self.load(read_fonts=False, read_colours=False, read_margins=True)
        self.assertEqual(fonts, dataclasses.replace(self.fonts, id="custom", name="自定义", cn_mv=72, jp_mv=24))
        self.assertEqual(pal, self.pal)

    def test_playres_scaling_and_missing_height_default(self):
        for height, scale in ((1080, 1), (360, 3), (None, 1)):
            self.doc = fansub(height)
            fonts, pal = self.load(read_margins=True)
            self.assertEqual((fonts.cn_size, fonts.jp_size, pal.width, fonts.cn_mv, fonts.jp_mv),
                             (22 * scale, 18 * scale, 1.5 * scale, 24 * scale, 8 * scale))
            self.assertEqual(pal.width * fonts.jp_outline_ratio, scale)
        fonts, _ = T.import_dialogue_styles(self.fonts, self.pal, fansub(360), "Default", "JP", playres_y=720)
        self.assertEqual(fonts.cn_size, 44)

    def test_one_side_keeps_other_font_position_colour_and_outline(self):
        for cn, jp in ((None, "JP"), ("Default", None)):
            fonts, pal = T.import_dialogue_styles(self.fonts, self.pal, self.doc, cn, jp, read_margins=True)
            if cn is None:
                self.assertEqual((fonts.cn_hans, fonts.cn_hant, fonts.cn_size, fonts.cn_mv, pal.text, pal.outline,
                                  pal.width, pal.outline_alpha),
                                 (self.fonts.cn_hans, self.fonts.cn_hant, self.fonts.cn_size, self.fonts.cn_mv,
                                  self.pal.text, self.pal.outline, self.pal.width, self.pal.outline_alpha))
            else:
                self.assertEqual((fonts.jp, fonts.jp_size, fonts.jp_mv, pal.jp_fill, pal.jp_edge),
                                 (self.fonts.jp, self.fonts.jp_size, self.fonts.jp_mv, self.pal.jp_fill, self.pal.jp_edge))
                self.assertAlmostEqual(pal.width * fonts.jp_outline_ratio, self.pal.width * self.fonts.jp_outline_ratio)

    def test_empty_selection_is_noop(self):
        self.assertEqual(T.import_dialogue_styles(self.fonts, self.pal, self.doc, None, None), (self.fonts, self.pal))
        self.assertEqual(self.load(read_fonts=False, read_colours=False), (self.fonts, self.pal))

    def test_zero_cn_outline_retains_independent_jp_width_and_scheme(self):
        self.doc.style_map()["Default"].set("Outline", 0)
        fonts, pal = self.load()
        self.assertEqual((pal.width, fonts.jp_outline_width), (0, 1))
        self.assertEqual(T.scheme_from_dict(T.scheme_dict(fonts, pal)), (fonts, pal))
        styles = AssDocument.parse(T.build_styles(fonts, pal)).style_map()
        self.assertEqual((styles["CN"].outline, styles["JP"].outline), (0, 1))


class StyleImportUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        assert cls.app.platformName() == "offscreen"

    def tearDown(self):
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def picker(self, doc=None, **kwargs):
        dialog = StyleImportDialog(doc if doc is not None else fansub(), translation_lang=kwargs.get("translation", "zh-Hant"),
                                   original_lang=kwargs.get("original", "ja"))
        self.addCleanup(dialog.deleteLater)
        return dialog

    def editor(self):
        with patch.object(TemplateEditor, "_schedule"):
            editor = TemplateEditor()
        editor.extra_styles = ["Existing extra style"]
        self.addCleanup(editor.close)
        return editor

    def read_file(self, editor, doc, select=None, accepted=True):
        root = Path(tempfile.mkdtemp(prefix="template_import_", dir=os.environ.get("TEMP")))
        path = root / "fansub.ass"
        doc.save(path)
        original_bytes = path.read_bytes()

        def choose(dialog):
            if select:
                select(dialog)
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

        with patch("subkit_app.views.template_editor.QFileDialog.getOpenFileName", return_value=(str(path), "ASS")), \
                patch.object(StyleImportDialog, "exec", choose), patch.object(editor, "_schedule") as schedule:
            editor._from_file()
        self.assertEqual(path.read_bytes(), original_bytes)
        return schedule

    def test_preselection_counts_samples_unique_names_and_defaults(self):
        dialog = self.picker()
        self.assertEqual((dialog.translation.currentData(), dialog.original.currentData()), ("Default", "JP"))
        self.assertEqual(dialog.translation.count(), 6)
        label = dialog.translation.itemText(dialog.translation.findData("Default"))
        self.assertIn("Default（4 行，例：這是主要對白", label)
        self.assertNotIn(r"\i1", label)
        self.assertTrue(dialog.fonts.isChecked() and dialog.colours.isChecked())
        self.assertFalse(dialog.margins.isChecked())
        self.assertEqual(dialog.windowTitle(), "从 .ass 读入样式")
        self.assertIn("DuoSub 会按振假名自动排开中日两行，一般不用读", [w.text() for w in dialog.findChildren(QLabel)])
        self.assertTrue({"读入", "取消", "帮助"}.issubset({w.text() for w in dialog.findChildren(QPushButton)}))

    def test_kana_fallback_and_positioned_name_not_selected(self):
        doc = fansub()
        for style in doc.styles:
            if style.name == "JP":
                style.name = "Original"
        for ev in doc.events:
            if ev.style == "JP":
                ev.style = "Original"
        doc.events = [ev for ev in doc.events if ev.style != "OPJP"]
        doc.events += [Event(style="JP Effect", text=r"{\pos(0,0)}ここです") for _ in range(30)]
        doc.styles.append(Style.new("JP Effect"))
        dialog = self.picker(doc)
        self.assertEqual((dialog.translation.currentData(), dialog.original.currentData()), ("Default", "Original"))

    def test_missing_lanes_and_empty_document_allow_manual_selection(self):
        for style, expected in (("Default", ("Default", None)), ("JP", (None, "JP"))):
            doc = fansub()
            doc.events = [ev for ev in doc.events if ev.style == style]
            dialog = self.picker(doc)
            self.assertEqual((dialog.translation.currentData(), dialog.original.currentData()), expected)
            with self.assertRaises(ValueError):
                source_roles(doc, "5.2")
        dialog = self.picker(AssDocument())
        self.assertIsNone(dialog.translation.currentData())
        self.assertIsNone(dialog.original.currentData())

    def test_labels_follow_project_languages(self):
        dialog = self.picker(translation="en", original="ko")
        text = " ".join(w.text() for w in dialog.findChildren(QLabel))
        self.assertIn("翻译（英语）", text)
        self.assertIn("原声（韩语）", text)

    def test_editor_import_refresh_info_and_no_extra_styles(self):
        editor = self.editor()
        before = editor.fonts
        changed = []
        editor.changed.connect(lambda: changed.append(True))
        schedule = self.read_file(editor, fansub())
        schedule.assert_called_once()
        self.assertEqual((editor.fonts.cn_hant, editor.fonts.jp), ("Last CN", "Last JP"))
        self.assertEqual((editor.fonts.cn_mv, editor.fonts.jp_mv), (before.cn_mv, before.jp_mv))
        self.assertEqual(editor.extra_styles, ["Existing extra style"])
        self.assertEqual(changed, [True])
        self.assertIn("已从 fansub.ass 读入：翻译（繁中） ← Default，原声（日语） ← JP（字体和字号、颜色和边框）", editor.info.text())

    def test_editor_colours_only_and_skip_original(self):
        editor = self.editor()
        before = editor.fonts
        jp_fill = editor.pal.jp_fill

        def choose(dialog):
            dialog.fonts.setChecked(False)
            dialog.original.setCurrentIndex(0)

        self.read_file(editor, fansub(), choose)
        self.assertEqual(editor.fonts, before)
        self.assertEqual((editor.pal.text, editor.pal.jp_fill), ("123456", jp_fill))
        self.assertNotIn("原声（日语） ←", editor.info.text())

    def test_cancel_or_no_parts_preserves_editor(self):
        editor = self.editor()
        before = (editor.fonts, editor.pal, editor.extra_styles.copy(), editor.info.text())
        for accepted, select in ((False, None), (True, lambda d: (d.fonts.setChecked(False), d.colours.setChecked(False)))):
            schedule = self.read_file(editor, fansub(), select, accepted)
            schedule.assert_not_called()
            self.assertEqual((editor.fonts, editor.pal, editor.extra_styles, editor.info.text()), before)

    def test_duosub_direct_path_keeps_existing_import(self):
        editor = self.editor()
        editor.fonts = dataclasses.replace(editor.fonts, ruby_offset=7, pinyin_offset=-2)
        doc = AssDocument.parse(T._HEAD + T.build_styles(T.font_set("lanting_round"), T.palette("classic")))
        doc.styles.append(Style.new("Handwritten"))
        expected_f, expected_p = T.scheme_from_styles(doc)
        with patch.object(StyleImportDialog, "__init__", side_effect=AssertionError("direct path must not show picker")):
            self.read_file(editor, doc)
        self.assertEqual(editor.fonts, dataclasses.replace(expected_f, ruby_offset=7, pinyin_offset=-2))
        self.assertEqual(editor.pal, expected_p)
        self.assertEqual(len(editor.extra_styles), 1)
        self.assertIn("另外保留 1 个", editor.info.text())


if __name__ == "__main__":
    unittest.main()
