"""Fonts and sizes for a whole line or part of it; measurement fixes found with the 千之妖精 sign.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_stage0 import CN_ASS, JP_SRT, _write  # noqa: E402

from subkit.core.importer import import_source  # noqa: E402
from subkit.core.merge import MergeSettings, merge_episode  # noqa: E402
from subkit.core.ops import Editor  # noqa: E402


class Formatting(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        d = self.tmp / "orig"
        d.mkdir()
        self.ep_dir = self.tmp / "episode"
        cn = import_source(_write(d, "c.ass", CN_ASS), src_id="t1", lang="zh-Hans", lane="translation",
                           episode_dir=self.ep_dir, cn_cfg={"dialogue": ["Default"]})
        ja = import_source(_write(d, "j.srt", JP_SRT), src_id="o1", lang="ja", lane="original", episode_dir=self.ep_dir)
        self.sources = {"t1": cn, "o1": ja}
        self.doc = merge_episode("F", cn, ja, MergeSettings()).document
        self.ed = Editor(self.doc, self.sources)
        self.cn = next(m for m in self.doc.members.values() if m.lane == "translation" and "你好啊" in m.text)
        self.sign = next(m for m in self.doc.members.values() if "招牌" in m.text)

    def tearDown(self) -> None:
        for p in self.ep_dir.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _line(self, needle: str) -> str:
        from subkit.ass import AssDocument
        from subkit.core.export import ExportOptions, build_ass
        from subkit.styles_default import default_styles_ass

        ass, _rep = build_ass(self.doc, self.sources, AssDocument.parse(default_styles_ass()), self.ep_dir,
                              ExportOptions(mode="release", ruby=False))
        return next(e.text for e in ass.events if needle in e.text)

    # a whole line: tags first; the line's own tags of the same names are left out of the output only
    def test_whole_line(self) -> None:
        self.ed.set_format(self.cn.id, {"fs": 80, "b": 1, "c1": "&H0000FF&"})
        self.assertEqual(self._line("你好啊"), "{\\fs80\\b1\\1c&H0000FF&}你好啊")
        before = self.sign.text
        self.ed.set_format(self.sign.id, {"fs": 50})
        line = self._line("招牌")
        self.assertTrue(line.startswith("{\\pos(960,100)") is False and "\\fs50" in line and "\\pos(960,100)" in line, line)
        self.assertEqual(self.doc.members[self.sign.id].text, before)            # the source text is untouched
        self.ed.history.undo()
        self.ed.history.undo()
        self.assertEqual(self._line("你好啊"), "你好啊")

    # part of a line: the span opens with its tags and restores what was in effect before it
    def test_part_of_line(self) -> None:
        from subkit.ass import AssDocument
        from subkit.styles_default import default_styles_ass

        st = AssDocument.parse(default_styles_ass()).style(self.cn.style)
        self.ed.set_format(self.cn.id, {"fn": "Impact", "fs": 90}, 1, 2)
        self.assertEqual(self._line("好"), f"你{{\\fnImpact\\fs90}}好{{\\fn{st.fontname}\\fs{st.fontsize:g}}}啊")
        # a line that sets its own size: restore to that size, not to the style's
        self.ed.set_format(self.sign.id, {"fs": 70}, 0, 1)
        self.assertIn("{\\fs70}招{\\fs40}牌", self._line("牌"))

    # formatting is attached to the characters: typing in front of the span does not move it
    def test_follows_the_text(self) -> None:
        self.ed.set_format(self.cn.id, {"b": 1}, 1, 2)
        self.ed.replace_text(self.cn.id, "喂你好啊")
        line = self._line("喂")
        self.assertIn("你{\\b1}好{\\b", line)
        self.ed.clear_format(self.cn.id)
        self.assertEqual(self._line("喂"), "喂你好啊")

    def test_roundtrip(self) -> None:
        from subkit.model.serialize import document_from_json, document_to_json

        self.ed.set_format(self.cn.id, {"fs": 72}, 0, 2)
        doc2 = document_from_json(document_to_json(self.doc))
        self.assertEqual(doc2.members[self.cn.id].format, self.doc.members[self.cn.id].format)


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg (libass measurement)")
class Measurement(unittest.TestCase):
    """Two measurement fixes: black text on the black probe canvas, and tags written into the probe as text."""

    def setUp(self) -> None:
        from subkit.ass import AssDocument

        self.doc = AssDocument.parse("""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Name,Arial,55,&H00000000,&H000000FF,&H00FAFAFE,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
""")

    def test_black_text_and_font_tags(self) -> None:
        from subkit.ruby.measure import LineRequest, Measurer

        m = Measurer(self.doc)
        boxes = m.line_char_boxes([LineRequest("Name", "{\\fs45}千之妖精", 2, 527.0)])[0]
        self.assertTrue(all(b is not None for b in boxes), boxes)               # black text was invisible
        left, right = boxes[0].x0, boxes[-1].x1
        self.assertLess(abs((left + right) / 2 - 527), 6, (left, right))        # \fs45 used to shift the line
        self.assertLess(right - left, 4 * 45 * 1.1)                              # and the size is the tag's, not 55


if __name__ == "__main__":
    unittest.main()
