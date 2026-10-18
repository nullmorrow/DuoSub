"""Project style templates: colour schemes × the user's release fonts; switching a project's template.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit import templates as T  # noqa: E402
from subkit.ass import AssDocument  # noqa: E402
from subkit.core.workspace import Project  # noqa: E402
from subkit.styles_default import default_styles_ass  # noqa: E402


# ffmpeg/libass 0.17.0-39, DirectWrite on this host (U+10FFFD, Arial).
UNDRAWABLE_LOG = """[Parsed_ass_0 @ 0000020145964980] fontselect: (Arial, 400, 0) -> ArialMT, 0, ArialMT
[Parsed_ass_0 @ 0000020145964980] Glyph 0x10FFFD not found, selecting one more font for (Arial, 400, 0)
[Parsed_ass_0 @ 0000020145964980] fontselect: failed to find any fallback with glyph 0x10FFFD for font: (Arial, 400, 0)
"""


class FontLogs(unittest.TestCase):
    def test_order_first_selection_and_duplicates(self) -> None:
        log = ("fontselect: (B, 400, 0) -> B-Regular, 0, B-Regular\n"
               "Glyph 0x8131 not found, selecting one more font for (B, 400, 0)\n"
               "fontselect: (A, 400, 0) -> ArialMT, 0, ArialMT\n"
               "fontselect: (B, 700, 0) -> B-Bold, 0, B-Bold\n"
               "Glyph 0x8131 not found, selecting one more font for (B, 700, 0)\n"
               "Glyph 0x80CE not found, selecting one more font for (A, 400, 0)\n"
               "Glyph 0x9AA8 not found, selecting one more font for (B, 400, 0)\n")
        parsed = T.parse_font_log(log)
        self.assertEqual(list(parsed.selected.items()), [("B", "B-Regular"), ("A", "ArialMT")])
        self.assertEqual(list(parsed.gaps.items()), [("B", "脱骨"), ("A", "胎")])
        self.assertEqual(parsed.undrawable, {})

    def test_real_undrawable_log(self) -> None:
        parsed = T.parse_font_log(UNDRAWABLE_LOG + UNDRAWABLE_LOG)
        self.assertEqual(parsed.selected, {"Arial": "ArialMT"})
        self.assertEqual(parsed.gaps, {"Arial": "\U0010fffd"})
        self.assertEqual(parsed.undrawable, {"Arial": "\U0010fffd"})

    def test_empty_log(self) -> None:
        self.assertEqual(T.parse_font_log("unrelated ffmpeg output"), T.FontLog())


class BuildStyles(unittest.TestCase):
    def test_every_scheme_has_the_tool_style_names(self) -> None:
        needed = [s.name for s in AssDocument.parse(default_styles_ass()).styles]
        for pal in T.PALETTES:
            for fs in T.FONT_SETS:
                doc = AssDocument.parse(T.build_styles(fs, pal))
                self.assertEqual([s.name for s in doc.styles], needed, (pal.id, fs.id))

    def test_utena_matches_the_release(self) -> None:
        doc = AssDocument.parse(T.build_styles(T.font_set("utena_serif"), T.palette("utena"), translation_lang="zh-Hant"))
        cn, jp, top, song = doc.style("CN"), doc.style("JP"), doc.style("CN.Top"), doc.style("OP.CN")
        self.assertEqual(cn.fontname, "思源宋體 Heavy")                     # traditional translation → TC font
        self.assertEqual(cn.get("PrimaryColour"), "&H005F1FA8")              # rose text …
        self.assertEqual(cn.get("OutlineColour"), "&H14FFFFFF")              # … white outline (slightly see-through)
        self.assertEqual(jp.get("OutlineColour"), "&H00F0F0F0")
        self.assertEqual(top.get("PrimaryColour"), "&H00FFFFFF")             # second speaker / top: swapped
        self.assertEqual(top.get("OutlineColour"), "&H005F1FA8")
        self.assertEqual(song.get("PrimaryColour"), "&H00FFFFFF")            # songs: white, rose outline,
        self.assertEqual(song.get("OutlineColour"), "&H005F1FA8")
        self.assertEqual(song.get("SecondaryColour"), "&H00000000")          # not yet sung: black
        self.assertEqual((cn.fontsize, jp.fontsize, song.fontsize), (68, 50, 80))
        self.assertEqual((cn.outline, jp.outline, doc.style("JP.Ruby").outline), (3.5, 3, 2))
        # simplified: the very template the tool has shipped as its default (made from the Utena release)
        hans = AssDocument.parse(T.build_styles(T.font_set("utena_serif"), T.palette("utena"), translation_lang="zh-Hans"))
        default = AssDocument.parse(default_styles_ass())
        for st in default.styles:
            self.assertEqual(hans.style(st.name).to_line(default.style_format), st.to_line(default.style_format))

    def test_top_colour_swap_is_separate_from_second_speaker(self) -> None:
        import dataclasses
        fonts, utena = T.font_set("utena_serif"), T.palette("utena")      # preset: both swapped (top follows alt)

        def colours(pal, name):
            st = AssDocument.parse(T.build_styles(fonts, pal)).style(name)
            return st.get("PrimaryColour"), st.get("OutlineColour")
        normal, swapped = ("&H005F1FA8", "&H14FFFFFF"), ("&H00FFFFFF", "&H005F1FA8")
        self.assertEqual((colours(utena, "CN.Alt"), colours(utena, "CN.Top")), (swapped, swapped))
        only_alt = dataclasses.replace(utena, top_inverted=False)
        self.assertEqual((colours(only_alt, "CN.Alt"), colours(only_alt, "CN.Top")), (swapped, normal))
        only_top = dataclasses.replace(utena, alt_inverted=False, top_inverted=True)
        self.assertEqual((colours(only_top, "CN.Alt"), colours(only_top, "CN.Top")), (normal, swapped))
        # read back from the styles, and through the stored scheme
        for pal in (only_alt, only_top):
            _f, read = T.scheme_from_styles(AssDocument.parse(T.build_styles(fonts, pal)))
            self.assertEqual((read.alt_inverted, bool(read.top_inverted)), (pal.alt_inverted, pal.top_inverted))
            self.assertEqual(T.scheme_from_dict(T.scheme_dict(fonts, pal))[1], pal)

    def test_simplified_uses_the_sc_font(self) -> None:
        doc = AssDocument.parse(T.build_styles(T.font_set("nana_black"), T.palette("nana_purple"), translation_lang="zh-Hans"))
        self.assertEqual(doc.style("CN").fontname, "Noto Sans CJK SC Black")
        self.assertEqual(doc.style("JP").fontname, "Noto Sans CJK JP Black")
        self.assertEqual(doc.style("CN").get("OutlineColour"), "&H00760876")

    def test_font_report(self) -> None:
        log = ("[Parsed_ass_0 @ 0] fontselect: (Noto Sans CJK JP Black, 700, 0) -> NotoSansCJKjp-Black, 0, NotoSansCJKjp-Black\n"
               "[Parsed_ass_0 @ 0] fontselect: (没有的字体, 700, 0) -> Arial-BoldMT, 0, Arial-BoldMT\n"
               "[Parsed_ass_0 @ 0] Glyph 0x8131 not found, selecting one more font for (某字体, 700, 0)\n"
               "[Parsed_ass_0 @ 0] fontselect: (某字体, 700, 0) -> YuGothicUI-Semibold, 2, YuGothicUI-Semibold\n")
        missing, gaps = T.font_report(log)
        self.assertEqual(missing, ["没有的字体"])
        self.assertEqual(gaps, {"某字体": "脱"})


class ProjectTemplate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_and_replace(self) -> None:
        styles = T.build_styles(T.font_set("nana_black"), T.palette("nana_purple"), translation_lang="zh-Hant")
        p = Project.create(self.tmp / "p", "t", template_text=styles,
                           template_scheme={"palette": "nana_purple", "fonts": "nana_black"})
        self.assertEqual(p.data["template"]["scheme"], {"palette": "nana_purple", "fonts": "nana_black"})
        self.assertEqual(p.template().style("CN").get("OutlineColour"), "&H00760876")
        # a style of the user's own survives a template change; the old file is kept
        own = "Style: CN手写体,华康手札体A Std W7,72,&H0081514D,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,4,10,10,10,1\n"
        p.template_path().write_text(p.template_path().read_text(encoding="utf-8") + own, encoding="utf-8")
        v1 = p.template_path()
        path = p.replace_template(T.build_styles(T.font_set("nana_black"), T.palette("ocean"), translation_lang="zh-Hant"),
                                  {"palette": "ocean", "fonts": "nana_black"})
        self.assertTrue(v1.exists())
        self.assertEqual(path.name, "v2.ass")
        p2 = Project.open(p.root)
        self.assertEqual(p2.data["template"]["version"], 2)
        self.assertEqual(p2.data["template"]["scheme"]["palette"], "ocean")
        t2 = p2.template()
        self.assertEqual(t2.style("CN").get("OutlineColour"), "&H00894E1D")
        self.assertIsNotNone(t2.style("CN手写体"))
        self.assertEqual(t2.styles[-1].name, "CN手写体")

    def test_default_without_a_template(self) -> None:
        p = Project.create(self.tmp / "q", "t")
        self.assertEqual(p.template_path().read_text(encoding="utf-8"), default_styles_ass())
        self.assertNotIn("scheme", p.data["template"])


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg (libass)")
class Preview(unittest.TestCase):
    def test_renders_and_reports_a_missing_font(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            fake = T.FontSet("x", "x", "SubKit 不存在的字体", "SubKit 不存在的字体", "SubKit 不存在的字体")
            res = T.render_preview(T.build_styles(fake, T.palette("classic")), tmp / "p.png", width=320)
            self.assertTrue(res.png.exists() and res.png.stat().st_size > 1000)
            self.assertIn("SubKit 不存在的字体", res.missing_fonts)
            self.assertIn("没有安装", res.warning())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
