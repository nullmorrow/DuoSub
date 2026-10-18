"""K 轴 tests (docs/KARAOKE.md): beats follow the old tool's rules and reproduce the user's finished karaoke.

The reference line is the user's Utena E01 OP (Done/Publish Candidate, read-only):
    {\\kf71}潔{\\kf33}く {\\kf21}カッ{\\kf15}コ{\\kf22}よ{\\kf37}く {\\kf24}生{\\kf35}き{\\kf0}て{\\kf33}行{\\kf25}こ{\\kf51}う…{\\kf0}
with ruby lines {\\kf21}い{\\kf14}さ{\\kf19}ぎ{\\kf17}よ / {\\kf199}{\\kf24}い / {\\kf258}{\\kf33}い.
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

from subkit.core import karaoke as kara  # noqa: E402
from subkit.core.importer import import_source  # noqa: E402
from subkit.core.merge import MergeSettings, merge_episode  # noqa: E402
from subkit.core.ops import Editor  # noqa: E402
from subkit.model.document import Annotation, Group, Member, Run  # noqa: E402

LYRIC = "潔く カッコよく 生きて行こう…"
START = 9130
DURS = [21, 14, 19, 17, 33, 21, 15, 22, 37, 24, 35, 0, 33, 25, 51]          # cs, one per beat (user's final)
BASE = "{\\kf71}潔{\\kf33}く {\\kf21}カッ{\\kf15}コ{\\kf22}よ{\\kf37}く {\\kf24}生{\\kf35}き{\\kf0}て{\\kf33}行{\\kf25}こ{\\kf51}う…"


class Karaoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        d = self.tmp / "orig"
        d.mkdir()
        self.ep_dir = self.tmp / "episode"
        cn = import_source(_write(d, "ep.zh.ass", CN_ASS), src_id="t1", lang="zh-Hans", lane="translation",
                           episode_dir=self.ep_dir, cn_cfg={"dialogue": ["Default"]})
        ja = import_source(_write(d, "ep.ja.srt", JP_SRT), src_id="o1", lang="ja", lane="original", episode_dir=self.ep_dir)
        self.sources = {"t1": cn, "o1": ja}
        self.doc = merge_episode("E1", cn, ja, MergeSettings()).document
        self.ed = Editor(self.doc, self.sources)
        g = Group("g_song", "song", ["m_song", "m_cn"])
        self.doc.groups.append(g)
        self.doc.structure_changed()
        self.m = Member("m_song", g.id, "ja", "original", "song", START, 12800, style="Default",
                        runs=[Run(LYRIC, "text", ins="i_k", a=0, b=len(LYRIC))])
        self.cn = Member("m_cn", g.id, "zh-Hant", "translation", "song", START, 12800, style="Default",
                         runs=[Run("{\\kf61}", "tag", ins="i_t", a=0, b=7), Run("果敢地", "text", ins="i_c", a=0, b=3),
                               Run("{\\kf62}", "tag", ins="i_t2", a=0, b=7), Run("活下去", "text", ins="i_c2", a=0, b=3)])
        self.doc.members[self.m.id] = self.m
        self.doc.members[self.cn.id] = self.cn
        for aid, a, b, base, reading, parts in (("a_isagi", 0, 2, "潔く", "いさぎよく", [[0, 1, "いさぎよ"]]),
                                               ("a_iki", 9, 11, "生き", "いき", [[0, 1, "い"]]),
                                               ("a_iko", 12, 15, "行こう", "いこう", [[0, 1, "い"]])):
            self.doc.annotations[aid] = Annotation(aid, "ruby", self.m.id, [{"ins": "i_k", "a": a, "b": b}], base, reading,
                                                   status="confirmed", parts=parts)

    def tearDown(self) -> None:
        for p in self.ep_dir.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _timed(self):
        beats = kara.derive(self.doc, self.m)
        t = START
        for b, d in zip(beats, DURS):
            b.t0, b.t1, b.by = t, t + d * 10, "tap"
            t += d * 10
        return beats

    def _export(self, ruby: bool = False):
        from subkit.ass import AssDocument
        from subkit.core.export import ExportOptions, build_ass
        from subkit.styles_default import default_styles_ass

        return build_ass(self.doc, self.sources, AssDocument.parse(default_styles_ass()), self.ep_dir,
                         ExportOptions(mode="release", ruby=ruby))

    # 分拍 follows the old tool: kanji blocks by reading, ッ joins, punctuation joins, 4 beats on 潔
    def test_beats_follow_old_rules(self) -> None:
        beats = kara.derive(self.doc, self.m)
        self.assertEqual([b.sung for b in beats], ["い", "さ", "ぎ", "よ", "く", "かっ", "こ", "よ", "く", "い", "き", "て", "い", "こ", "う"])
        self.assertEqual(len({b.slot for b in beats[:4]}), 1)                 # 潔 = one slot of four beats
        mora = kara.derive(self.doc, self.m, unit="mora")
        self.assertIn("っ", [b.sung for b in mora])                            # mora unit: ッ is its own beat
        self.assertEqual(kara.kana_beats("ガーデン"), [(0, 2), (2, 4)])          # ガー|デン like the user's 校庭
        self.assertEqual(kara.kana_beats("ガーデン", "mora"), [(0, 1), (1, 2), (2, 3), (3, 4)])

    # the export reproduces the user's finished line and ruby lines exactly
    def test_export_matches_users_final(self) -> None:
        self.ed.set_karaoke(self.m.id, self._timed())
        ass, rep = self._export()
        line = next(e.text for e in ass.events if "潔" in e.text)
        self.assertEqual(line, BASE)
        view = kara.effective(self.doc, self.m)
        start_cs = START // 10
        self.assertEqual(kara.ruby_text(view, "a_isagi", 0, "いさぎよ", start_cs), "{\\kf21}い{\\kf14}さ{\\kf19}ぎ{\\kf17}よ")
        self.assertEqual(kara.ruby_text(view, "a_iki", 0, "い", start_cs), "{\\kf199}{\\kf24}い")
        self.assertEqual(kara.ruby_text(view, "a_iko", 0, "い", start_cs), "{\\kf258}{\\kf33}い")
        self.assertEqual(rep.karaoke_lines, 1)
        self.assertEqual(rep.karaoke_unfinished, [])

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg (libass measurement)")
    def test_export_ruby_lines_carry_karaoke(self) -> None:
        self.ed.set_karaoke(self.m.id, self._timed())
        ass, _rep = self._export(ruby=True)
        ruby = [e.text for e in ass.events if e.style.endswith(".Ruby")]
        self.assertTrue(any(t.endswith("{\\kf199}{\\kf24}い") for t in ruby), ruby)
        self.assertTrue(any(t.endswith("{\\kf21}い{\\kf14}さ{\\kf19}ぎ{\\kf17}よ") for t in ruby), ruby)

    # \k already in a source line: read in, exported identically
    def test_import_from_tags(self) -> None:
        kid = self.ed.import_karaoke(self.cn.id)
        self.assertIsNotNone(kid)
        ass, _rep = self._export()
        line = next(e.text for e in ass.events if "果" in e.text)
        durs = [int(x) for x in __import__("re").findall(r"\\kf(\d+)", line)]
        self.assertEqual(len(durs), 6)                                         # one beat per character
        self.assertEqual((sum(durs[:3]), sum(durs[3:])), (61, 62))           # the source syllables' lengths
        # import split the source's per-word syllables evenly per character, each a timed beat
        view = kara.effective(self.doc, self.cn)
        self.assertTrue(view.complete)
        self.assertEqual(view.beats[0].t0, START)
        self.assertEqual(view.beats[-1].t1, START + 1230)

    # changing a reading keeps the block's time span, splits it, and flags it for re-timing
    def test_reading_change_is_flagged(self) -> None:
        self.ed.set_karaoke(self.m.id, self._timed())
        self.ed.set_reading("a_isagi", "きよく")
        view = kara.effective(self.doc, self.m)
        blk = [b for b in view.beats if b.ann == "a_isagi"]
        self.assertEqual([b.sung for b in blk], ["き", "よ"])
        self.assertEqual((blk[0].t0, blk[-1].t1), (START, START + 710))
        self.assertTrue(all("reading_changed" in b.flags for b in blk))
        _ass, rep = self._export()
        self.assertTrue(rep.karaoke_unfinished)
        self.ed.history.undo()
        self.assertTrue(kara.effective(self.doc, self.m).complete)

    # typing a character keeps every other beat's time; the new one is a flagged placeholder
    def test_text_edit_keeps_timing(self) -> None:
        self.ed.set_karaoke(self.m.id, self._timed())
        self.ed.replace_text(self.m.id, LYRIC + "よ")
        view = kara.effective(self.doc, self.m)
        self.assertEqual([b.by for b in view.beats[:15]], ["tap"] * 15)
        self.assertEqual(view.beats[-1].by, "placeholder")
        self.assertIn("有新字或改过的字还没打拍", view.issues)

    # moving the group moves the beats; the exported tags (relative) stay the same; undo restores
    def test_shift_group_moves_beats(self) -> None:
        self.ed.set_karaoke(self.m.id, self._timed())
        self.ed.shift_group("g_song", 1000)
        self.assertEqual(kara.effective(self.doc, self.m).beats[0].t0, START + 1000)
        ass, _rep = self._export()
        self.assertEqual(next(e.text for e in ass.events if "潔" in e.text), BASE)
        self.ed.history.undo()
        self.assertEqual(kara.effective(self.doc, self.m).beats[0].t0, START)

    # changing the effect never touches the beats (acceptance 14)
    def test_effect_change_keeps_beats(self) -> None:
        self.ed.set_karaoke(self.m.id, self._timed())
        before = [(b.t0, b.t1) for b in kara.effective(self.doc, self.m).beats]
        self.ed.set_karaoke_effect([self.m.id], "ko")
        ass, _rep = self._export()
        self.assertEqual(next(e.text for e in ass.events if "潔" in e.text), BASE.replace("\\kf", "\\ko"))
        self.ed.set_karaoke_effect([self.m.id], "none")
        ass, _rep = self._export()
        self.assertEqual(next(e.text for e in ass.events if "潔" in e.text), LYRIC)
        self.assertEqual([(b.t0, b.t1) for b in kara.effective(self.doc, self.m).beats], before)

    # the project file keeps K data across save/load
    def test_json_roundtrip(self) -> None:
        from subkit.model.serialize import document_from_json, document_to_json

        self.ed.set_karaoke(self.m.id, self._timed())
        doc2 = document_from_json(document_to_json(self.doc))
        self.assertEqual(doc2.karaoke, self.doc.karaoke)


if __name__ == "__main__":
    unittest.main()
