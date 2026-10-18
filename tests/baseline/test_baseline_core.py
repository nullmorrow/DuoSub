"""Unit tests for the pure-Python core (no ffmpeg / embeddings needed).

Run:  python -m unittest discover -s _tool/tests      (from the project root, PYTHONPATH=_tool)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import VisibleText, insert_at_visible, plain_text, private_blocks, remove_private  # noqa: E402
from subkit.ass.timefmt import ass_to_cs, cs_to_ass, srt_to_cs  # noqa: E402
from subkit.encoding import decode_bytes  # noqa: E402
from subkit.reading.kana_align import align  # noqa: E402
from subkit.reading.numbers import counter_reading, number_reading  # noqa: E402
from subkit.reading.records import parse_record, romaji_to_kana  # noqa: E402
from subkit.ruby.layout import Item, layout_line  # noqa: E402
from subkit.sources.jp import JpOptions, clean_cues, clean_line, extract_inline_readings  # noqa: E402
from subkit.sources.cn import classify  # noqa: E402
from subkit.align.aligner import AlignRow, Aligner, realign_detached_blocks  # noqa: E402
from subkit.align.fragments import build_units  # noqa: E402
from subkit.config import Config  # noqa: E402
from subkit.align.fragments import is_interjection, split_phrases  # noqa: E402
from subkit.reading.characters import is_generic_label  # noqa: E402
from subkit.gloss import parse_glosses  # noqa: E402
from subkit.karaoke import parse_k, morae  # noqa: E402


class TextTests(unittest.TestCase):
    def test_visible_and_breaks(self):
        vt = VisibleText(r"{\fad(1,2)}あい\Nう{\b1}え\hお")
        self.assertEqual(vt.text, "あいうえ お")
        self.assertEqual(vt.line_count, 2)
        self.assertEqual(vt.line_of(2), 1)

    def test_insert_and_private(self):
        t = r"{\fad(1,2)}あい{\b1}うえ"
        self.assertEqual(insert_at_visible(t, {2: "|"}), r"{\fad(1,2)}あい{\b1}|うえ")
        self.assertEqual(private_blocks("校庭{@注:校庭=ガーデン}")[0][0], 2)
        self.assertEqual(remove_private("a{@x}b{\\b1}c"), "ab{\\b1}c")

    def test_drawing_is_not_text(self):
        self.assertEqual(plain_text(r"{\p1}m 0 0 l 10 10{\p0}字"), "字")

    def test_times(self):
        self.assertEqual(ass_to_cs("0:01:02.34"), 6234)
        self.assertEqual(cs_to_ass(6234), "0:01:02.34")
        self.assertEqual(srt_to_cs("00:01:02,345"), 6235)


class EncodingTests(unittest.TestCase):
    def test_shift_jis_is_not_gb18030(self):
        s = "（有栖川(ありすがわ)）えっ？ 何だって？ 生徒会の連中は"
        self.assertEqual(decode_bytes(s.encode("cp932")).text, s)

    def test_gbk_chinese(self):
        s = "这是一个测试，我们开始吧。"
        self.assertEqual(decode_bytes(s.encode("gbk")).text, s)


class DocumentTests(unittest.TestCase):
    def test_roundtrip(self):
        src = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n[V4+ Styles]\n"
               "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
               "Style: JP,Arial,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n"
               "[Events]\nFormat: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text\n"
               "Dialogue: 0,0:00:01.00,0:00:02.00,JP,創真,0,0,0,#3,テスト, with comma\n\n[Aegisub Project Garbage]\nVideo File: x.mkv\n")
        d = AssDocument.parse(src)
        self.assertEqual(d.events[0].name, "創真")
        self.assertEqual(d.events[0].text, "テスト, with comma")
        s1 = d.serialize()
        self.assertEqual(AssDocument.parse(s1).serialize(), s1)
        self.assertIn("Video File: x.mkv", s1)


class JapaneseSourceTests(unittest.TestCase):
    def test_netflix_cc(self):
        cues = clean_cues([(0, 100, ["≪(城一郎)創真｡", "ん？"]), (100, 200, ["(電話が切れた音)"]), (200, 300, ["♬～"])], JpOptions())
        a = cues[0].lines[0]
        self.assertEqual((a.text, a.speaker, a.offscreen), ("創真", "城一郎", True))
        self.assertEqual(cues[1].lines[0].kind, "sfx")
        self.assertEqual(cues[2].lines[0].kind, "music")

    def test_label_on_its_own_line(self):
        cues = clean_cues([(0, 100, ["（有栖川(ありすがわ)樹璃(じゅり)）", "奇跡を信じて"])], JpOptions())
        line = cues[0].lines[1]
        self.assertEqual(line.speaker, "有栖川樹璃")
        self.assertIn(("樹璃", "じゅり"), [(h.surface, h.reading) for h in line.hints])

    def test_inline_readings(self):
        text, hints = extract_inline_readings("遠月(とおつき)学園に 苑田(そのだ)茎子(けいこ)")
        self.assertEqual(text, "遠月学園に 苑田茎子")
        self.assertEqual([(h.surface, h.reading) for h in hints], [("遠月", "とおつき"), ("苑田", "そのだ"), ("茎子", "けいこ")])


class AlignmentUnitTests(unittest.TestCase):
    def test_phrases(self):
        self.assertEqual(split_phrases("大体さあ  数学なんて九九が言えりゃ  それでいいのよ"), ["大体さあ", "数学なんて九九が言えりゃ", "それでいいのよ"])
        self.assertEqual(split_phrases("えっ？何！"), ["えっ？", "何！"])

    def test_interjections(self):
        for w in ("あっ", "えっ？", "ウフフフ…", "オーホッホッホ！", "チュッ", "へ〜え"):
            self.assertTrue(is_interjection(w), w)
        for w in ("まさか！", "そんな！", "だが！", "はい", "うん", "あっそ", "料理"):
            self.assertFalse(is_interjection(w), w)

    def test_generic_labels(self):
        self.assertTrue(is_generic_label("女子生徒１"))
        self.assertTrue(is_generic_label("創真･城一郎"))
        self.assertFalse(is_generic_label("城一郎"))


class ReadingUnitTests(unittest.TestCase):
    def test_kana_align(self):
        self.assertEqual(align("行った", "いった"), [(0, 1, "い")])
        self.assertEqual(align("取り扱い", "とりあつかい"), [(0, 1, "と"), (2, 3, "あつか")])
        self.assertEqual(align("目覚め", "めざめ"), [(0, 2, "めざ")])
        self.assertEqual(align("危な〜い", "あぶない"), [(0, 1, "あぶ")])
        self.assertEqual(align("解錠薬", "ステイタスシーフ"), [(0, 3, "ステイタスシーフ")])

    def test_numbers(self):
        self.assertEqual(number_reading(9000), "きゅうせん")
        self.assertEqual(number_reading(300), "さんびゃく")
        self.assertEqual(counter_reading(3, "本"), ("ぼん", False))
        self.assertEqual(counter_reading(1, "杯"), ("ぱい", False))
        self.assertEqual(counter_reading(3, "分"), ("ぷん", False))
        self.assertEqual(counter_reading(2, "人"), ("ふたり", True))

    def test_records(self):
        es = parse_record("?何（なに/なん） !私（わたくし） 生（-） 並行（へいこう,80） 階（kai）+層（そう）")
        self.assertEqual([(e.surface, e.reading) for e in es], [("何", "なに"), ("私", "わたくし"), ("生", ""), ("並行", "へいこう"), ("階層", "かいそう")])
        self.assertTrue(es[0].uncertain and es[1].forced and es[2].hidden)
        self.assertEqual(es[3].scale, 80.0)
        self.assertEqual(romaji_to_kana("gakkou"), "がっこう")
        self.assertEqual(romaji_to_kana("shinnyuu"), "しんにゅう")


class LayoutTests(unittest.TestCase):
    def test_overhang_without_compression(self):
        it = Item(100, 150, 75, left_limit=87.5, right_limit=162.5)
        layout_line([it])
        self.assertAlmostEqual(it.scale, 100.0)
        self.assertAlmostEqual(it.x, 125.0)

    def test_shift_before_compress(self):
        a, b = Item(100, 150, 52), Item(150, 200, 52)
        layout_line([a, b])
        self.assertEqual((a.scale, b.scale), (100.0, 100.0))
        self.assertGreaterEqual(b.x - b.w() / 2, a.x + a.w() / 2 + 1.9)

    def test_widest_gives_way(self):
        long, short = Item(100, 150, 125, left_limit=87.5), Item(150, 200, 25, right_limit=212.5)
        layout_line([long, short])
        self.assertLess(long.scale, 100.0)
        self.assertEqual(short.scale, 100.0)

    def test_crowded_word_compresses_uniformly_and_locally(self):
        # 食の上流階級に: 上流+階級 share the space between の and に; 食 keeps 100%
        items = [Item(520, 550, 43, 470, 562.5), Item(582, 644, 91, 569.5), Item(645, 706, 78, right_limit=718.5)]
        layout_line(items, word_gap=7.5)
        sho, jo, kai = items
        self.assertEqual(sho.scale, 100.0)
        self.assertAlmostEqual(jo.scale, kai.scale, delta=0.5)
        self.assertFalse(any(it.overlap for it in items))
        self.assertGreaterEqual(jo.x - jo.w() / 2, sho.x + sho.w() / 2 + 7.0)   # word gap kept

    def test_overfull_breaks_bounds_not_glyphs(self):
        a, b = Item(100, 130, 91, 100, 130), Item(131, 161, 43, 131)
        layout_line([a, b])
        self.assertTrue(a.overlap)
        self.assertGreaterEqual(b.x - b.w() / 2, a.x + a.w() / 2 + 1.9)   # never collide


class SourceTests(unittest.TestCase):
    def test_flashback_and_descriptions(self):
        o = JpOptions()
        self.assertEqual(clean_line("((恵！))", o).text, "恵！")
        self.assertTrue(clean_line("((恵！))", o).flashback)
        line = clean_line("(郁魅)((ああっ！))", o)
        self.assertEqual((line.text, line.speaker), ("ああっ！", "郁魅"))
        self.assertEqual(clean_line("ほら お守りよ))", o).text, "ほら お守りよ")
        self.assertEqual(clean_line("((ピィー！(警笛)))", o).kind, "sfx")
        self.assertEqual(clean_line("(大御堂(だいみどう)ふみ緒(お))", o).kind, "sfx")   # a label, not ((…))
        self.assertIsNone(clean_line("(棒読みで)よろしく", o).speaker)
        self.assertEqual(clean_line("(城一郎・回想)おあがりよ", o).speaker, "城一郎")
        cues = clean_cues([(0, 100, ["[TEL](ﾏﾅｰﾓｰﾄﾞ)", "あっ｡"]), (100, 200, ["(郁魅の母)", "((おめでとう))"])], o)
        self.assertIsNone(cues[0].lines[1].speaker)
        self.assertEqual(cues[1].lines[1].speaker, "郁魅の母")

    def test_repeated_sound_word_is_one_filler(self):
        self.assertEqual(split_phrases("コツ コツ"), ["コツ コツ"])
        self.assertTrue(is_interjection("ガシャン ガシャン！"))
        self.assertFalse(is_interjection("ラーメン！"))
        self.assertTrue(is_generic_label("研究員"))

    def test_song_style_shared_by_jp_and_cn(self):
        doc = AssDocument()
        from subkit.ass.document import Event
        doc.events = [Event(kind="Dialogue", start=0, end=100, style="shiki zw", text="对白"),
                      Event(kind="Dialogue", start=0, end=100, style="OPED", text="思い描いてる"),
                      Event(kind="Dialogue", start=0, end=100, style="OPED", text="比心中描繪的"),
                      Event(kind="Dialogue", start=0, end=100, style="OPED", text="{\blur3}")]
        src = classify(doc, {"dialogue": ["Shiki ZW"], "song_jp": ["OPED"], "song_cn": ["OPED"]})
        self.assertEqual([src.roles[i] for i in range(4)], ["dialogue", "song_jp", "song_cn", "keep"])
        src = classify(doc, {"dialogue": ["Default"]})
        self.assertEqual(src.roles[0], "dialogue")   # configured style missing → auto
        self.assertTrue(src.notes)

    def test_season_section(self):
        cfg = Config({"cn": {"dialogue": ["Default"], "top": ["8para"], "S02": {"dialogue": ["shiki zw"]}}}, Path("."))
        self.assertEqual(cfg.section("cn", 1), {"dialogue": ["Default"], "top": ["8para"]})
        self.assertEqual(cfg.section("cn", 2), {"dialogue": ["shiki zw"], "top": ["8para"]})


class AlignTests(unittest.TestCase):
    def test_detached_preview_block(self):
        # body aligned at offset 0; the preview sits 90 s later in the JP captions
        cues = clean_cues([(1000, 1300, ["お前は誰だ"]), (1400, 1700, ["料理人だ"]),
                           (12000, 12300, ["次回 食戟のソーマ"]), (12400, 12700, ["お楽しみに"])], JpOptions())
        units = build_units(cues)
        for u in units:
            u.wstart, u.wend = u.start, u.end
        rows = [AlignRow(0, 1000, 1300, "你是谁"), AlignRow(1, 1400, 1700, "厨师"),
                AlignRow(2, 3000, 3300, "下集 食戟之灵"), AlignRow(3, 3400, 3700, "敬请期待")]
        res = Aligner(rows, units).run()
        self.assertEqual(res.rows[2], [])
        moved = realign_detached_blocks(rows, units, res, lambda r, u: Aligner(r, u))
        self.assertEqual(len(moved), 1)
        self.assertAlmostEqual(moved[0][2], -9000, delta=60)
        self.assertEqual([units[k].text for k in res.rows[2]], ["次回", "食戟のソーマ"])
        self.assertEqual([units[k].text for k in res.rows[3]], ["お楽しみに"])
        self.assertEqual(res.skipped, [])


class GlossKaraokeTests(unittest.TestCase):
    def test_gloss(self):
        g = parse_glosses("光差す校庭{@注:校庭=ガーデン} 手を")
        self.assertEqual((g[0].start, g[0].end, g[0].text), (3, 5, "ガーデン"))

    def test_k_parse(self):
        sy = parse_k(r"{\k32}う{\k25}ま{\kf40}れ")
        self.assertEqual([(s.text, s.start, s.end) for s in sy], [("う", 0, 32), ("ま", 32, 57), ("れ", 57, 97)])
        self.assertEqual(morae("きょうっー"), ["きょ", "うっー"])


if __name__ == "__main__":
    unittest.main()
