"""Round 5: any ASS override tags (and the furigana following them), lines with no time on screen, deleting several
groups / one language, brackets that are not readings (眷属(ファミリア)), the visual template engine.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import dataclasses
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

from test_stage0 import CN_ASS, _write  # noqa: E402

from subkit.ass import AssDocument, Event  # noqa: E402
from subkit.ass.state import check_raw, restore_tags, state_at, strip_inside  # noqa: E402
from subkit.core import cc  # noqa: E402
from subkit.core.importer import import_source  # noqa: E402
from subkit.core.merge import MergeSettings, merge_episode  # noqa: E402
from subkit.core.ops import Editor  # noqa: E402
from subkit.core.readings import visible_map  # noqa: E402
from subkit.model.validate import validate  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg"))

JP_SRT = """1
00:00:01,000 --> 00:00:03,000
こんにちは

2
00:00:03,100 --> 00:00:05,000
黄昏を前に風（うず）を巻け

3
00:00:05,500 --> 00:00:06,500
グオオオ

4
00:00:08,000 --> 00:00:10,000
自らの眷属（ファミリア）を結成し
"""


class Episode(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        d = self.tmp / "orig"
        d.mkdir()
        self.ep_dir = self.tmp / "episode"
        cn = import_source(_write(d, "c.ass", CN_ASS), src_id="t1", lang="zh-Hans", lane="translation",
                           episode_dir=self.ep_dir, cn_cfg={"dialogue": ["Default"]})
        ja = import_source(_write(d, "j.srt", JP_SRT), src_id="o1", lang="ja", lane="original", episode_dir=self.ep_dir)
        self.sources = {"t1": cn, "o1": ja}
        self.doc = merge_episode("R5", cn, ja, MergeSettings()).document
        self.ed = Editor(self.doc, self.sources)

    def tearDown(self) -> None:
        for p in self.ep_dir.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def member(self, needle: str):
        return next(m for m in self.doc.members.values() if needle in visible_map(m)[0].text)

    def export(self):
        from subkit.core.export import ExportOptions, build_ass
        from subkit.styles_default import default_styles_ass

        return build_ass(self.doc, self.sources, AssDocument.parse(default_styles_ass()), self.ep_dir,
                         ExportOptions(mode="release", ruby=False))


class RawTags(Episode):
    def test_check_raw(self) -> None:
        self.assertEqual(check_raw("{\\frz10\\fscx120}", span=False), ("\\frz10\\fscx120", ""))
        self.assertIn("开头", check_raw("frz10", span=False)[1])
        self.assertIn("不认识", check_raw("\\zzz5", span=False)[1])
        self.assertIn("拖动", check_raw("\\pos(1,2)", span=False)[1])
        self.assertIn("K 轴", check_raw("\\k20", span=False)[1])
        self.assertIn("整行", check_raw("\\org(1,2)", span=True)[1])
        self.assertEqual(check_raw("\\t(0,500,\\frz30)", span=True)[1], "")
        self.assertTrue(check_raw("\\t(0,500,\\pos(1,2))", span=False)[1])         # \pos cannot be animated here

    def test_whole_line_raw_replaces_the_same_tags(self) -> None:
        sign = self.member("招牌")
        self.ed.set_format(sign.id, {"raw": "\\frz12\\fscx130"})
        line = next(e.text for e in self.export()[0].events if "招牌" in e.text)
        self.assertTrue(line.startswith("{\\frz12\\fscx130}"), line)
        self.assertIn("\\pos(960,100)", line)                      # the source's own position stays
        with self.assertRaises(ValueError):
            self.ed.set_format(sign.id, {"raw": "\\pos(1,1)"})

    def test_span_raw_is_restored_after_the_span(self) -> None:
        cn = self.member("你好啊")
        self.ed.set_format(cn.id, {"raw": "\\frz15\\fscy140"}, 1, 2)
        line = next(e.text for e in self.export()[0].events if "好" in e.text and "你" in e.text)
        self.assertEqual(line, "你{\\frz15\\fscy140}好{\\frz0\\fscy100}啊")

    def test_state_and_strip(self) -> None:
        t = "a{\\frz5\\1c&H0000FF&}b{\\frz9}cd"
        st, _ts = state_at(t, 2, ("frz", "1c"))
        self.assertEqual([x.raw for x in st.values()], ["\\1c&H0000FF&", "\\frz9"])
        self.assertEqual(restore_tags(t, 3, ["frz", "fscx"], None), "\\frz9\\fscx100")
        self.assertEqual(strip_inside(t, 1, 3, {"frz"}), "a{\\frz5\\1c&H0000FF&}b{}cd".replace("{}", ""))


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg (libass measurement)")
class RubyFollows(unittest.TestCase):
    """Furigana take the rotation / shear / scale / move of their kanji (positions computed unrotated, like libass)."""

    HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: JP,Arial,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,2,1
Style: JP.Ruby,Arial,25,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,0,0,0,1
"""

    def ruby(self, text: str):
        from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
        from subkit.ruby.measure import Measurer

        doc = AssDocument.parse(self.HEAD)
        ev = Event(kind="Dialogue", start=0, end=300, style="JP", text=text)
        doc.events.append(ev)
        job = RubyJob(ev, "JP.Ruby", [RubyPiece(3, 5, "めいきゅう")])
        RubyGenerator(doc, Measurer(doc)).run([job])
        return job

    def test_rotation_shear_scale_move(self) -> None:
        plain = self.ruby("{\\an2\\pos(960,700)}ここが迷宮都市").out[0].text
        rot = self.ruby("{\\an2\\pos(960,700)\\frz20}ここが迷宮都市").out[0].text
        self.assertIn("\\org(960.0,700.0)", rot)
        self.assertIn("\\frz20", rot)
        self.assertEqual(rot.split("\\org")[0], plain.split("}")[0])          # the same unrotated position
        tall = self.ruby("{\\an2\\pos(960,700)\\fscy170}ここが迷宮都市").out[0].text
        y = lambda s: float(s.split("\\pos(")[1].split(")")[0].split(",")[1])   # noqa: E731
        self.assertLess(y(tall), y(plain) - 15)                             # lifted above the taller kanji
        self.assertIn("\\fscy170", tall)
        mv = self.ruby("{\\an2\\move(700,700,1200,800)}ここが迷宮都市").out[0].text
        self.assertIn("\\move(", mv)
        anim = self.ruby("{\\an2\\pos(960,700)\\t(0,1000,\\frz30\\fs80)}ここが迷宮都市")
        self.assertIn("\\t(0,1000,\\frz30)", anim.out[0].text)              # the rotation follows, the size cannot
        self.assertTrue(any("动画" in w for w in anim.warnings))


class NotTimed(Episode):
    def test_zero_length_line_becomes_a_comment(self) -> None:
        m = self.member("こんにちは")
        m.start, m.end = -1650, -700                                        # before the video starts
        ass, rep = self.export()
        ev = next(e for e in ass.events if "こんにちは" in e.text)
        self.assertEqual(ev.kind, "Comment")
        self.assertTrue(rep.not_timed and "こんにちは" in rep.not_timed[0])
        self.assertFalse(any(e.kind == "Dialogue" and e.end <= e.start for e in ass.events))


class Deleting(Episode):
    def test_delete_groups_one_undo(self) -> None:
        gids = [self.member("グオオオ").group, self.member("こんにちは").group]
        n0, t0 = len(self.doc.groups), len(self.doc.tombstones)
        self.ed.delete_groups(gids)
        self.assertEqual(len(self.doc.groups), n0 - 2)
        self.assertGreater(len(self.doc.tombstones), t0)
        self.assertFalse(validate(self.doc, self.sources).errors)
        self.ed.history.undo()
        self.assertEqual(len(self.doc.groups), n0)
        self.assertEqual(len(self.doc.tombstones), t0)

    def test_delete_several_in_list_order(self) -> None:
        """Regression: the group index went stale after the first removal and the next lookup hit a neighbour."""
        keep = [g.id for g in self.doc.groups]
        gone = [keep[0], keep[1], keep[3]]
        expected = [g for g in keep if g not in gone]
        self.ed.delete_groups(gone)
        self.assertEqual([g.id for g in self.doc.groups], expected)
        for g in self.doc.groups:
            self.assertTrue(all(m in self.doc.members for m in g.members))
        self.assertFalse(validate(self.doc, self.sources).errors)
        self.ed.history.undo()
        self.assertEqual([g.id for g in self.doc.groups], keep)
        self.assertFalse(validate(self.doc, self.sources).errors)

    def test_delete_one_language(self) -> None:
        g = self.member("你好啊").group
        n = self.ed.delete_lane([g, self.member("グオオオ").group], "original")
        self.assertEqual(n, 2)
        self.assertTrue(any(m.lane == "translation" for m in self.doc.members_of(self.doc.group(g))))
        self.assertFalse(any(m.lane == "original" for m in self.doc.members_of(self.doc.group(g))))
        self.assertFalse(validate(self.doc, self.sources).errors)

    def test_confirm_many(self) -> None:
        gids = [g.id for g in self.doc.groups[:3]]
        self.ed.confirm_many(gids)
        self.assertTrue(all(self.doc.group(g).state == "confirmed" for g in gids))
        self.ed.history.undo()
        self.assertFalse(any(self.doc.group(g).state == "confirmed" for g in gids))


class Brackets(Episode):
    def test_katakana_bracket_is_text(self) -> None:
        self.assertIn("眷属（ファミリア）を結成し", visible_map(self.member("結成"))[0].text)
        an = cc.analyze_event(0, "風（うず）を巻け", "srt")
        self.assertEqual([s.kind for s in an.spans], ["reading_hint"])     # hiragana is still a reading

    def test_bracket_back_into_the_text(self) -> None:
        m = self.member("黄昏")
        self.assertNotIn("うず", visible_map(m)[0].text)
        from subkit.core.readings import annotate_member
        from subkit.reading.engine import ReadingEngine

        engine = ReadingEngine()
        annotate_member(self.doc, m, engine, source=self.sources["o1"])
        aid = next((a.id for a in self.doc.annotations.values() if a.member == m.id and a.origin == "cc_hint"), None)
        self.assertIsNotNone(aid)
        self.assertEqual(self.ed.bracket_to_text([aid], engine=engine), 1)
        self.assertIn("風（うず）を巻け", visible_map(m)[0].text)
        self.assertNotIn(aid, self.doc.annotations)
        self.assertFalse([e for e in validate(self.doc, self.sources).errors if "ledger" in e.code])
        self.ed.history.undo()                                              # undo restores a snapshot: look it up again
        self.assertNotIn("うず", visible_map(self.doc.members[m.id])[0].text)


class TemplateEngine(unittest.TestCase):
    def test_scheme_round_trip_and_read_back(self) -> None:
        from subkit import templates as T
        from subkit.styles_default import default_styles_ass

        fs, pal = T.font_set("lanting_round"), T.palette("green")
        self.assertEqual(T.scheme_from_dict(T.scheme_dict(fs, pal)), (fs, pal))
        f2, p2 = T.scheme_from_styles(AssDocument.parse(default_styles_ass()))
        self.assertEqual(T.build_styles(f2, p2, translation_lang="zh-Hans").splitlines(), default_styles_ass().splitlines())
        doc = AssDocument.parse(T.build_styles(fs, pal, translation_lang="zh-Hant"))
        cn, jp = doc.style("CN"), doc.style("JP")          # the user's style lines
        self.assertEqual((cn.fontname, cn.fontsize, cn.bold, cn.spacing, cn.get("OutlineColour")),
                         ("方正兰亭圆_GBK_中粗", 66, False, 1, "&H14489925"))
        self.assertEqual((jp.fontname, jp.fontsize, jp.bold, jp.outline, jp.num("MarginV")),
                         ("A-OTF Shin Maru Go Pro DB", 54, True, 3.4, 14))

    @unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg (libass)")
    def test_scene_layers_render_and_fit(self) -> None:
        from subkit import templates as T

        fs, pal = T.font_set("utena_serif"), T.palette("utena")
        styles = T.build_styles(fs, pal, translation_lang="zh-Hant")
        lines = T.scene_events(styles, "dialogue", traditional=True)
        layer = {ln.split(",")[3]: ln.split(",")[0] for ln in lines}
        self.assertEqual((layer["CN"], layer["JP"]), ("Dialogue: 1", "Dialogue: 0"))     # like the export
        tmp = Path(tempfile.mkdtemp())
        try:
            res = T.render_scene(styles, tmp / "s.png", scene="alt", width=480, traditional=True, pinyin=True)
            self.assertTrue(res.png.exists())
            bad = dataclasses.replace(fs, cn_mv=55)
            fixed, c = T.fit_layout(bad, pal, translation_lang="zh-Hant", measure_cache=tmp / "m.sqlite3")
            self.assertLessEqual(abs(c["cn_ruby"] - T.TARGET["cn_ruby"]), 1)
            self.assertGreater(fixed.cn_mv, 55)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
