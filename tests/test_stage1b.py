"""Stage-1b tests: preview cache, positions, Aegisub-style timing, readings, review packet, hit testing.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
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
from subkit.model.document import Annotation, Member, Run  # noqa: E402
from subkit.model.validate import validate  # noqa: E402


SONG_STYLES = r"""Style: opjp,Arial,34,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,10,10,18,1
Style: opcn,Arial,38,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,3,10,10,18,1
"""
SONG_EVENTS = r"""[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,你好啊
Dialogue: 1,0:01:00.00,0:01:03.00,opjp,,0,0,0,,{\3c00008C}生まれ変わる
Dialogue: 0,0:01:00.00,0:01:03.00,opjp,,0,0,0,,{\blur2.5\1aFF}生まれ変わる
Dialogue: 1,0:01:00.00,0:01:03.00,opcn,,0,0,0,,{\3c00008C}脱胎换骨
Dialogue: 0,0:01:00.00,0:01:03.00,opcn,,0,0,0,,{\blur2.5\1aFF}脱胎换骨
"""
SONG_SRT = """1
00:00:01,000 --> 00:00:03,000
こんにちは
"""
FORMATTED_SRT = r"""1
00:00:01,000 --> 00:00:03,000
<font size="20">森の先人よ　誇り高き同胞よ</font>

2
00:00:03,100 --> 00:00:05,000
<font size="20">-グオオ
-んっ</font>

3
00:00:08,000 --> 00:00:10,000
{\an8}<i>-30度だ</i>
"""


class Stage1b(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        src_dir = self.tmp / "orig"
        src_dir.mkdir()
        self.ep_dir = self.tmp / "episode"
        cn_path = _write(src_dir, "ep.zh.ass", CN_ASS)
        jp_path = _write(src_dir, "ep.ja.srt", JP_SRT)
        self.cn = import_source(cn_path, src_id="t1", lang="zh-Hans", lane="translation", episode_dir=self.ep_dir,
                                cn_cfg={"dialogue": ["Default"]})
        self.ja = import_source(jp_path, src_id="o1", lang="ja", lane="original", episode_dir=self.ep_dir)
        self.sources = {"t1": self.cn, "o1": self.ja}
        self.doc = merge_episode("E1", self.cn, self.ja, MergeSettings()).document
        self.ed = Editor(self.doc, self.sources)

    def tearDown(self) -> None:
        for p in self.ep_dir.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _template(self):
        from subkit.ass import AssDocument
        from subkit.styles_default import default_styles_ass

        return AssDocument.parse(default_styles_ass())

    def _pair(self):
        for g in self.doc.groups:
            ms = self.doc.members_of(g)
            if g.kind == "dialogue" and {m.lane for m in ms} == {"original", "translation"}:
                return g, next(m for m in ms if m.lane == "original"), next(m for m in ms if m.lane == "translation")
        raise AssertionError("no paired group")

    # the live preview only lays out changed members, and its output is identical to a full export
    def test_export_cache_matches_full_export(self) -> None:
        from subkit.core.export import ExportCache, ExportOptions, build_ass

        tpl = self._template()
        opts = ExportOptions(mode="preview", ruby=False)
        cache = ExportCache()

        def full():
            return build_ass(self.doc, self.sources, tpl, self.ep_dir, opts)[0].serialize()

        a, rep = build_ass(self.doc, self.sources, tpl, self.ep_dir, opts, cache=cache)
        self.assertEqual(a.serialize(), full())
        self.assertEqual(len(rep.owners), len(a.events))
        self.assertTrue(all(o["member"] in self.doc.members for o in rep.owners))
        a, rep = build_ass(self.doc, self.sources, tpl, self.ep_dir, opts, cache=cache)
        self.assertEqual(rep.laid_out, 0)
        g, jp, cn = self._pair()
        self.ed.replace_text(cn.id, self.ed.visible_text(cn.id) + "吧")
        self.ed.set_group_time(g.id, start=jp.start - 200)
        a, rep = build_ass(self.doc, self.sources, tpl, self.ep_dir, opts, cache=cache)
        self.assertEqual(a.serialize(), full())
        self.assertLessEqual(rep.laid_out, len(g.members))

    # a user position wins over the source's \pos in the output; the source text is untouched; undo works
    def test_position_override(self) -> None:
        from subkit.core.export import ExportOptions, build_ass

        sign = next(m for m in self.doc.members.values() if "招牌" in m.text)
        before = sign.text
        self.ed.set_overrides({sign.id: {"an": 7, "pos": [100, 200.5]}})
        ass, rep = build_ass(self.doc, self.sources, self._template(), self.ep_dir, ExportOptions(mode="release", ruby=False))
        line = next(e.text for e in ass.events if "招牌" in e.text)
        self.assertTrue(line.startswith("{\\an7\\pos(100,200.5)}"), line)
        self.assertNotIn("960", line)
        self.assertEqual(rep.positions_overridden, 1)
        self.assertEqual(sign.text, before)
        self.ed.history.undo()
        self.assertEqual(self.doc.members[sign.id].overrides, {})

    # Aegisub-style timing: left = start, right = end; whole group gets one time (as in the user's finals)
    def test_group_time_like_aegisub(self) -> None:
        g, jp, cn = self._pair()
        orig = (jp.start, jp.end)
        self.ed.set_group_time(g.id, start=900)
        self.assertEqual({self.doc.members[m].start for m in g.members}, {900})
        self.ed.set_group_time(g.id, end=3300, target="translation")
        self.assertEqual(self.doc.members[cn.id].end, 3300)
        self.assertNotEqual(self.doc.members[jp.id].end, 3300)
        dur = self.doc.members[jp.id].end - self.doc.members[jp.id].start
        self.ed.set_group_time(g.id, start=self.doc.members[jp.id].end + 5000, target="original")   # past the end
        m = self.doc.members[jp.id]
        self.assertEqual(m.end - m.start, dur)                                # keeps its length, never collapses
        for _ in range(3):
            self.ed.history.undo()
        self.assertEqual((self.doc.members[jp.id].start, self.doc.members[jp.id].end), orig)

    # readings are placed on visible characters, also on lines that contain a line break
    def test_reading_on_line_with_break(self) -> None:
        from subkit.core.readings import anchor_positions, plain_range_to_visible, visible_map

        g, jp, _cn = self._pair()
        m = Member("m_br", g.id, "ja", "original", "dialogue", jp.start, jp.end,
                   runs=[Run("今日は", "text", ins="i_1", a=0, b=3), Run("\\N", "sep"), Run("晴れだ", "text", ins="i_2", a=0, b=3)])
        self.doc.members[m.id] = m
        g.members.append(m.id)
        vt, ids = visible_map(m)
        self.assertEqual(vt.text, "今日は晴れだ")
        self.assertEqual(plain_range_to_visible(m, 5, 6), (3, 4))            # editing text "今日は\N晴れだ": 晴 at 5
        aid = self.ed.add_annotation(m.id, 3, 4, "は")
        self.assertIsNotNone(aid)
        self.assertEqual(anchor_positions(self.doc.annotations[aid], ids), (3, 4))

    # Ctrl+G: fresh suggestions for a range; confirmed readings kept unless asked; one undo step
    def test_regenerate_readings(self) -> None:
        from subkit.core.readings import annotate_document, visible_map
        from subkit.reading.engine import ReadingEngine

        engine = ReadingEngine()
        annotate_document(self.doc, self.sources, engine)
        m = next(self.doc.members[a.member] for a in self.doc.annotations.values())
        a0 = next(a for a in self.doc.annotations.values() if a.member == m.id)
        self.ed.set_reading(a0.id, "てすと")
        n_vis = len(visible_map(m)[0].text)
        self.ed.regenerate_readings(m.id, 0, n_vis, engine, source=self.ja)
        self.assertEqual(self.doc.annotations[a0.id].text, "てすと")             # confirmed: kept
        self.ed.regenerate_readings(m.id, 0, n_vis, engine, source=self.ja, include_confirmed=True)
        self.assertNotIn(a0.id, self.doc.annotations)
        self.assertTrue(any(a.member == m.id and a.base == a0.base for a in self.doc.annotations.values()))
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations[a0.id].text, "てすと")

    # 审议包: readings written on their kanji (okurigana outside), guesses marked, one JSON line per group
    def test_review_packet(self) -> None:
        from subkit.core.review import annotated, build_packet

        g, jp, _cn = self._pair()
        m = Member("m_rv", g.id, "ja", "original", "dialogue", jp.start, jp.end,
                   runs=[Run("合わねぇ 何だ", "text", ins="i_rv", a=0, b=7)])
        self.doc.members[m.id] = m
        g.members.append(m.id)
        self.doc.annotations["a_1"] = Annotation("a_1", "ruby", m.id, [{"ins": "i_rv", "a": 0, "b": 2}], "合わ", "あわ",
                                                 status="confirmed", parts=[[0, 1, "あ"]])
        self.doc.annotations["a_2"] = Annotation("a_2", "ruby", m.id, [{"ins": "i_rv", "a": 5, "b": 6}], "何", "なん",
                                                 status="uncertain", candidates=["なん", "なに"], parts=[[0, 1, "なん"]])
        text, readings = annotated(self.doc, m)
        self.assertEqual(text, "合(あ)わねぇ 何(なん?)だ")
        self.assertEqual([r["status"] for r in readings], ["confirmed", "uncertain"])
        md, jl = build_packet(self.doc, episode="E1")
        recs = [json.loads(x) for x in jl.splitlines()]
        self.assertEqual(len(recs), sum(1 for x in self.doc.groups if x.kind in ("dialogue", "song")))
        self.assertIn("合(あ)わねぇ", md)
        self.assertIn('"group"', md)                                          # the answer format is spelled out

    # song lines keep their source layers: a fansub's glow copy stays under the main line (libass stacks lines of
    # the same layer instead of overlapping them)
    def test_song_lines_keep_source_layers(self) -> None:
        head = CN_ASS.split("[Events]")[0].replace("Style: Sign,", SONG_STYLES + "Style: Sign,")
        d = self.tmp / "songs"
        d.mkdir()
        cn = import_source(_write(d, "s.ass", head + SONG_EVENTS), src_id="t1", lang="zh-Hans", lane="translation",
                           episode_dir=self.tmp / "es", cn_cfg={"dialogue": ["Default"], "song_jp": ["opjp"], "song_cn": ["opcn"]})
        ja = import_source(_write(d, "s.srt", SONG_SRT), src_id="o1", lang="ja", lane="original", episode_dir=self.tmp / "es")
        doc = merge_episode("S", cn, ja, MergeSettings()).document
        songs = [g for g in doc.groups if g.kind == "song"]
        self.assertEqual(len(songs), 2)
        layers = sorted(tuple(sorted(doc.members[m].layer for m in g.members)) for g in songs)
        self.assertEqual(layers, [(0, 0), (1, 1)])                     # main pair on layer 1, glow pair on layer 0

    # SRT formatting (<font>, <i>, {\an8}) and the dashes of a two-speaker cue are metadata, not dialogue;
    # every source character is still accounted for, and a lone leading dash in a one-line cue stays text
    def test_srt_formatting_removed_at_import(self) -> None:
        d = self.tmp / "fmt"
        d.mkdir()
        cn = import_source(_write(d, "f.ass", CN_ASS), src_id="t1", lang="zh-Hans", lane="translation",
                           episode_dir=self.tmp / "ef", cn_cfg={"dialogue": ["Default"]})
        ja = import_source(_write(d, "f.srt", FORMATTED_SRT), src_id="o1", lang="ja", lane="original",
                           episode_dir=self.tmp / "ef")
        doc = merge_episode("F", cn, ja, MergeSettings()).document
        texts = [doc.members[m].text for g in doc.groups for m in g.members if doc.members[m].lane == "original"]
        joined = " | ".join(texts)
        self.assertNotIn("<", joined)
        self.assertNotIn("{", joined)
        self.assertIn("森の先人よ", joined)
        self.assertTrue(any("グオオ" in t and "んっ" in t and "-" not in t for t in texts), texts)
        self.assertTrue(any(t.startswith("-30度") for t in texts), texts)          # one-line cue: not a speaker dash
        kinds = [s.kind for s in ja.spans]
        self.assertEqual(kinds.count("format"), 7)                               # 2 + 2 + {\an8} <i> </i>
        self.assertEqual(sum(1 for s in ja.spans if s.kind == "marker" and (s.data or {}).get("turn")), 2)
        self.assertEqual(next(s.data for s in ja.spans if s.kind == "format"), {"tag": "font", "attrs": {"size": "20"}})
        rep = validate(doc, {"o1": ja, "t1": cn})
        self.assertTrue(rep.ok, rep.summary())
        self.assertEqual((rep.ledger["o1"]["missing"], rep.ledger["o1"]["duplicate"]), (0, 0))

    # clicking: the top-most line wins (ruby is drawn after its base), and the nearest character is found
    def test_hit_test(self) -> None:
        from subkit.core.layout_index import LayoutIndex, LineGeom, Rect

        base = LineGeom(0, "m1", "g1", "base", 0, Rect(100, 900, 220, 960),
                        chars=[Rect(100, 900, 140, 960), None, Rect(180, 900, 220, 960)], ids=[("i", "x", 0), None, ("i", "x", 2)])
        ruby = LineGeom(5, "m1", "g1", "ruby", 0, Rect(100, 870, 140, 895), ann="a_1")
        other = LineGeom(2, "m2", "g2", "base", 1, Rect(90, 880, 300, 940), chars=[Rect(90, 880, 300, 940)], ids=[("i", "y", 0)])
        h = LayoutIndex.hit([base, ruby], 120, 880)
        self.assertEqual((h.line.kind, h.line.ann), ("ruby", "a_1"))
        h = LayoutIndex.hit([base, ruby], 175, 930)
        self.assertEqual((h.line.member, h.char, h.identity), ("m1", 2, ("i", "x", 2)))
        h = LayoutIndex.hit([base, ruby, other], 150, 920)
        self.assertEqual(h.line.member, "m2")                                 # higher layer is on top
        self.assertIsNone(LayoutIndex.hit([base, ruby], 600, 300))


if __name__ == "__main__":
    unittest.main()
