"""Stage-0 acceptance tests (REQUIREMENTS 十九 1–10, 16, 18) on small synthetic sources.

Run from the repository root:  python -m unittest discover -s tests
"""

from __future__ import annotations

import hashlib
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

from subkit.align.dp2 import Aligner2, Row  # noqa: E402
from subkit.core.importer import import_source  # noqa: E402
from subkit.core.merge import MergeSettings, merge_episode  # noqa: E402
from subkit.core.ops import Editor  # noqa: E402
from subkit.core.units import build_units  # noqa: E402
from subkit.model.serialize import document_from_json, document_to_json  # noqa: E402
from subkit.model.validate import validate  # noqa: E402

CN_ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,40,1
Style: Sign,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,8,10,10,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,你好啊
Dialogue: 0,0:00:03.10,0:00:05.00,Default,,0,0,0,,{\\fad(100,0)}今天天气真好 #1 @me
Dialogue: 0,0:00:08.00,0:00:10.00,Default,,0,0,0,,再见
Dialogue: 0,0:00:08.50,0:00:09.50,Sign,,0,0,0,,{\\pos(960,100)}招牌
Comment: 0,0:00:00.00,0:00:00.00,Default,,0,0,0,,字幕组注释
"""

JP_SRT = """1
00:00:01,000 --> 00:00:03,000
(太郎)こんにちは

2
00:00:03,100 --> 00:00:05,000
あっ 今日は いい天気だね

3
00:00:05,500 --> 00:00:06,500
あっ…

4
00:00:06,600 --> 00:00:07,600
(花子)ねえ 聞いてる？

5
00:00:08,000 --> 00:00:10,000
≪さようなら｡
"""


def _write(d: Path, name: str, text: str, encoding: str = "utf-8-sig") -> Path:
    p = d / name
    p.write_bytes(text.replace("\n", "\r\n").encode(encoding))
    return p


class Stage0(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.src_dir = self.tmp / "orig"
        self.src_dir.mkdir()
        self.cn_path = _write(self.src_dir, "ep.zh.ass", CN_ASS)
        self.jp_path = _write(self.src_dir, "ep.ja.srt", JP_SRT)
        self.hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.cn_path, self.jp_path)}
        self.ep_dir = self.tmp / "episode"
        self.cn = import_source(self.cn_path, src_id="t1", lang="zh-Hans", lane="translation", episode_dir=self.ep_dir,
                                cn_cfg={"dialogue": ["Default"]})
        self.ja = import_source(self.jp_path, src_id="o1", lang="ja", lane="original", episode_dir=self.ep_dir)
        self.sources = {"t1": self.cn, "o1": self.ja}

    def tearDown(self) -> None:
        for p in self.ep_dir.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def merged(self, mode: str = "keep"):
        return merge_episode("E1", self.cn, self.ja, MergeSettings(filler_mode=mode)).document

    # 1. sources never change; copies are byte-identical and read-only
    def test_sources_read_only(self) -> None:
        doc = self.merged()
        ed = Editor(doc, self.sources)
        g = next(g for g in doc.groups if g.kind == "dialogue")
        ed.confirm(g.id)
        for p, h in self.hashes.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), h)
        for s in self.sources.values():
            copy = self.ep_dir / s.copy
            self.assertEqual(hashlib.sha256(copy.read_bytes()).hexdigest(), s.sha256)
            self.assertFalse(os.access(copy, os.W_OK))

    # 2. every character has an auditable place; nothing added; both filler modes
    def test_ledger_complete(self) -> None:
        for mode in ("keep", "compact"):
            doc = self.merged(mode)
            rep = validate(doc, self.sources)
            self.assertTrue(rep.ok, (mode, rep.summary(), [i.message for i in rep.errors][:5]))
            for sid, counts in rep.ledger.items():
                self.assertEqual(counts["missing"], 0)
                self.assertEqual(counts["duplicate"], 0)
            ja = rep.ledger["o1"]
            self.assertGreater(ja["metadata"], 0)                 # (太郎) (花子) ≪ ｡ kept as metadata
            if mode == "keep":
                self.assertEqual(ja["comment"], 0)               # every filler displayed
            # labels are not dialogue text but stay findable as speakers
            labels = [s.value for s in self.ja.spans if s.kind == "speaker_label"]
            self.assertEqual(labels, ["太郎", "花子"])

    # 3. JP order with a stand-alone filler and an unpaired line in between
    def test_order_kept_with_fillers_and_orphans(self) -> None:
        doc = self.merged("keep")
        seq = []
        for g in doc.groups:
            for mid in g.members:
                m = doc.members[mid]
                seq += [(r.e, r.a) for r in m.runs if r.src == "o1"]
        self.assertEqual(seq, sorted(seq))
        jp_texts = [doc.members[m].text for g in doc.groups for m in g.members if doc.members[m].lane == "original"]
        self.assertIn("あっ…", jp_texts)                            # stand-alone filler displayed in keep mode
        self.assertTrue(any("聞いてる" in t for t in jp_texts))      # line without CN still present, in order

    # 4. crossed order: never swap the original; a joint block keeps both orders
    def test_crossing_is_not_swapped(self) -> None:
        units = build_units(self.ja)
        for u in units:
            u.wstart, u.wend = u.start, u.end
        rows = [Row(0, 3100, 4000, "天气好", 1), Row(1, 4000, 5000, "今天", 1)]
        sim = lambda cn, jp: 0.9 if (("今日" in jp) == ("今天" in cn)) else 0.1  # noqa: E731
        sel = [u for u in units if u.e == 1]
        out = Aligner2(rows, sel, sim_fn=sim).run()
        order = [k for bl in out.blocks for k in bl.units]
        self.assertEqual(order, sorted(order))

    # 5. two rows at the same time are not merged only because of time
    def test_simultaneous_lines_not_merged(self) -> None:
        doc = self.merged()
        signs = [g for g in doc.groups if g.kind == "sign"]
        self.assertEqual(len(signs), 1)                               # the sign overlapping 再见 keeps its own group
        g_bye = next(g for g in doc.groups if any("再见" in doc.members[m].text for m in g.members))
        self.assertEqual(len(g_bye.members), 2)

    # 6–8. edits keep the document valid, undo is exact, deletions are remembered
    def test_edits_validate_and_undo(self) -> None:
        doc = self.merged("compact")
        before = json.dumps({k: v for k, v in document_to_json(doc).items() if k != "revision"}, sort_keys=True, ensure_ascii=False)
        ed = Editor(doc, self.sources)
        g2 = next(g for g in doc.groups if any("天气" in doc.members[m].text for m in g.members))
        jp = ed.lane_member(g2, "original")
        self.assertIsNotNone(jp)
        ed.move_phrase(g2.id, "original", "tail_to_next")
        ed.move_phrase(g2.id, "original", "take_from_next")
        ed.merge_with_next(g2.id)
        ed.split(g2.id, jp_at=2)
        ed.unlink(g2.id)
        m = ed.lane_member(doc.group(g2.id), "original") or next(doc.members[x] for g in doc.groups for x in g.members if doc.members[x].lane == "original")
        ed.delete_range(m.id, 0, 1)
        for cid in list(doc.comments):
            ed.restore_comment(cid)
        ed.shift_group(g2.id, 120)
        rep = validate(doc, self.sources)
        self.assertTrue(rep.ok, rep.summary())
        self.assertGreater(rep.ledger["o1"]["deleted"], 0)
        self.assertTrue(doc.tombstones)
        while ed.history.undo():
            pass
        after = json.dumps({k: v for k, v in document_to_json(doc).items() if k != "revision"}, sort_keys=True, ensure_ascii=False)
        self.assertEqual(before, after)

    # 9. two 私 in one line are independent annotations
    def test_two_watashi_independent(self) -> None:
        d = self.tmp / "w"
        d.mkdir()
        p = _write(d, "w.srt", "1\n00:00:01,000 --> 00:00:03,000\n私は 私だ\n")
        src = import_source(p, src_id="o1", lang="ja", lane="original", episode_dir=self.tmp / "ew")
        cnp = _write(d, "w.ass", CN_ASS.split("[Events]")[0] + "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,我就是我\n")
        cn = import_source(cnp, src_id="t1", lang="zh-Hans", lane="translation", episode_dir=self.tmp / "ew", cn_cfg={"dialogue": ["Default"]})
        doc = merge_episode("W", cn, src, MergeSettings()).document
        ed = Editor(doc, {"o1": src, "t1": cn})
        g = next(g for g in doc.groups if g.kind == "dialogue")
        m = ed.lane_member(g, "original")
        a1 = ed.add_annotation(m.id, 0, 1, "わたし")
        a2 = ed.add_annotation(m.id, 3, 4, "わたくし")
        self.assertNotEqual(doc.annotations[a1].anchor, doc.annotations[a2].anchor)
        ed.set_reading(a2, "あたし")
        self.assertEqual(doc.annotations[a1].text, "わたし")
        self.assertEqual(doc.annotations[a2].text, "あたし")

    # 16. export: only tool data is left out; ASS tags and @/# in dialogue survive
    def test_export_keeps_dialogue_text(self) -> None:
        from subkit.ass import AssDocument
        from subkit.core.export import ExportOptions, build_ass
        from subkit.styles_default import default_styles_ass

        doc = self.merged()
        tpl = AssDocument.parse(default_styles_ass())
        ass, rep = build_ass(doc, self.sources, tpl, self.ep_dir, ExportOptions(mode="release", ruby=False))
        texts = [e.text for e in ass.events]
        self.assertTrue(any("{\\fad(100,0)}今天天气真好 #1 @me" == t for t in texts), texts)
        self.assertTrue(any("\\pos(" in t and "招牌" in t for t in texts))
        self.assertFalse(any(e.effect.startswith("#") or "g_" in e.effect for e in ass.events))
        self.assertFalse(any("字幕组注释" == t for t in texts))     # source comment lines are optional, default off

    # okurigana: ruby sits on the kanji only (old tool's rule, user's finals: 見せて → み over 見)
    def test_okurigana_ruby_on_kanji_only(self) -> None:
        from subkit.core.readings import placement_text, ruby_parts

        cases = {("合わ", "あわ"): "合(あ)わ", ("合わ", "あ"): "合(あ)わ", ("与える", "あたえる"): "与(あた)える",
                 ("目覚め", "めざめ"): "目覚(めざ)め", ("黄色い", "きいろい"): "黄色(きいろ)い",
                 ("突き落とさ", "つきおとさ"): "突(つ)き落(お)とさ", ("A級", "えーきゅう"): "A級(えーきゅう)",
                 ("見せ", "みせ"): "見(み)せ"}
        for (base, reading), expect in cases.items():
            self.assertEqual(placement_text(base, reading), expect)
        self.assertEqual(ruby_parts("ありがとう", "ありがとう"), [])
        # every automatic annotation of a merged document keeps okurigana bare
        from subkit.core.readings import annotate_document
        from subkit.reading.engine import ReadingEngine

        doc = self.merged()
        annotate_document(doc, self.sources, ReadingEngine())
        self.assertTrue(doc.annotations)
        for a in doc.annotations.values():
            for pa, pb, _t in a.parts:
                last = a.base[pb - 1]
                self.assertTrue("一" <= last <= "鿿" or last.isdigit() or not last.strip() is None,
                                (a.base, a.parts))
                self.assertFalse("ぁ" <= last <= "ゖ", (a.base, a.parts))

    # display rule: sentence-final 。 hidden only when rendering; document unchanged
    def test_hide_final_period_is_display_only(self) -> None:
        from subkit.core.display import DisplayRules, display_runs
        from subkit.model.document import Member, Run

        m = Member("m_x", "g_x", "ja", "original", "dialogue", 0, 1000,
                   runs=[Run("言った。 「行く。」と", "text", ins="i_a", a=0, b=12), Run("  ", "sep"),
                         Run("次｡", "text", ins="i_b", a=0, b=2)])
        before = [r.text for r in m.runs]
        runs, n = display_runs(m, DisplayRules(hide_final_period=True))
        self.assertEqual("".join(r.text for r in runs), "言った 「行く。」と  次")
        self.assertEqual(n, 2)
        self.assertEqual([r.text for r in m.runs], before)              # the member itself is untouched
        runs, n = display_runs(m, DisplayRules(hide_final_period=False))
        self.assertEqual(n, 0)

    # 18. JSON round trip is exact (exchange format)
    def test_json_roundtrip(self) -> None:
        doc = self.merged("compact")
        j = document_to_json(doc)
        j["unknown_extension"] = {"keep": True}
        j["groups"][0]["x_future_field"] = 1
        d2 = document_from_json(json.loads(json.dumps(j, ensure_ascii=False)))
        self.assertEqual(document_to_json(d2), j)


if __name__ == "__main__":
    unittest.main()
