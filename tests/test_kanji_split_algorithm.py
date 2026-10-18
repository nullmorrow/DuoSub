"""Per-kanji ruby splitting: lexical evidence, uniqueness and bounded work."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.reading import kanji_split
from subkit.reading.kanji_split import KanjiReading, split_kanji_reading


class FakeAnalyzer:
    def __init__(self, readings):
        self.readings = readings
        self.calls = []

    def lookup_readings(self, surface):
        self.calls.append(surface)
        return self.readings.get(surface, [])


def values(result):
    return None if result is None else [(part.start, part.end, part.base, part.reading) for part in result]


class KanjiSplitAlgorithm(unittest.TestCase):
    def test_required_examples(self):
        analyzer = FakeAnalyzer({"世": ["せい", "よ", "せ"], "間": ["かん", "けん"],
                                 "学": ["まなぶ", "がく", "がっ"], "校": ["こう"],
                                 "迷": ["めい"], "宮": ["みや", "きゅう"], "都": ["と"],
                                 "市": ["し"], "日": ["ひ", "にち", "じつ"], "本": ["ほん"],
                                 "今": ["いま", "こん"], "一": ["いち", "ひと"],
                                 "人": ["ひと", "じん", "にん"]})
        for surface, reading, expected in (("世間", "せけん", ["せ", "けん"]),
                                            ("学校", "がっこう", ["がっ", "こう"]),
                                            ("迷宮都市", "めいきゅうとし", ["めい", "きゅう", "と", "し"]),
                                            ("日本", "にっぽん", ["にっ", "ぽん"]),
                                            ("日本", "にほん", ["に", "ほん"])):
            with self.subTest(surface=surface, reading=reading):
                result = split_kanji_reading(surface, reading, analyzer)
                self.assertIsNotNone(result)
                self.assertEqual([part.reading for part in result], expected)
                self.assertEqual([part.base for part in result], list(surface))
        for surface, reading in (("今日", "きょう"), ("一人", "ひとり")):
            with self.subTest(surface=surface):
                self.assertIsNone(split_kanji_reading(surface, reading, analyzer))

    def test_rendaku_and_handakuten(self):
        for source, targets in (("か", "が"), ("き", "ぎ"), ("く", "ぐ"), ("け", "げ"), ("こ", "ご"),
                                 ("さ", "ざ"), ("し", "じ"), ("す", "ず"), ("せ", "ぜ"), ("そ", "ぞ"),
                                 ("た", "だ"), ("ち", "ぢ"), ("つ", "づ"), ("て", "で"), ("と", "ど"),
                                 ("は", "ばぱ"), ("ひ", "びぴ"), ("ふ", "ぶぷ"), ("へ", "べぺ"), ("ほ", "ぼぽ")):
            analyzer = FakeAnalyzer({"甲": ["あ"], "乙": [source + "ん"]})
            for target in targets:
                with self.subTest(source=source, target=target):
                    result = split_kanji_reading("甲乙", "あ" + target + "ん", analyzer)
                    self.assertEqual([part.reading for part in result], ["あ", target + "ん"])

    def test_sokuon_variants_and_duplicate_proofs(self):
        for ending in "つちくき":
            with self.subTest(ending=ending):
                analyzer = FakeAnalyzer({"甲": ["あ" + ending], "乙": ["こう"]})
                self.assertEqual([p.reading for p in split_kanji_reading("甲乙", "あっこう", analyzer)],
                                 ["あっ", "こう"])
        analyzer = FakeAnalyzer({"甲": ["カク", "かく", "かっ", "カッ", "かっ"], "乙": ["こう"]})
        self.assertEqual([p.reading for p in split_kanji_reading("甲乙", "カッコウ", analyzer)], ["かっ", "こう"])

    def test_no_arbitrary_truncation_or_final_sokuon(self):
        analyzer = FakeAnalyzer({"甲": ["あいう"], "乙": ["えつ"]})
        for reading in ("あえつ", "あいうえっ"):
            with self.subTest(reading=reading):
                self.assertIsNone(split_kanji_reading("甲乙", reading, analyzer))

    def test_okurigana_requires_exact_lexicon_evidence(self):
        analyzer = FakeAnalyzer({"学": ["まなぶ", "がく"], "学ぶ": ["まなぶ"], "者": ["もの"]})
        self.assertEqual(values(split_kanji_reading("学ぶ者", "まなぶもの", analyzer)),
                         [(0, 1, "学", "まな"), (2, 3, "者", "もの")])
        # A suffix of an on-reading is not automatically an okurigana stem.
        self.assertIsNone(split_kanji_reading("学く者", "がくもの", analyzer))
        self.assertIsNone(split_kanji_reading("学べ者", "まなべもの", analyzer))

    def test_multiple_written_kana_are_fixed_and_have_no_boxes(self):
        analyzer = FakeAnalyzer({"取": ["とり"], "取り": ["とり"],
                                 "組": ["くみ"], "組み": ["くみ"]})
        self.assertEqual(values(split_kanji_reading("取り組み", "とりくみ", analyzer)),
                         [(0, 1, "取", "と"), (2, 3, "組", "く")])
        self.assertIsNone(split_kanji_reading("取り組み", "とくみ", analyzer))

    def test_literal_anchors_can_eliminate_an_ambiguous_cut(self):
        analyzer = FakeAnalyzer({"甲": ["あ", "ああ"], "乙": ["い", "あい"]})
        self.assertEqual(values(split_kanji_reading("甲あ乙", "ああい", analyzer)),
                         [(0, 1, "甲", "あ"), (2, 3, "乙", "い")])

    def test_distinct_complete_divisions_are_ambiguous(self):
        analyzer = FakeAnalyzer({"甲": ["あ", "あい"], "乙": ["いう", "う"]})
        self.assertIsNone(split_kanji_reading("甲乙", "あいう", analyzer))

    def test_hiragana_katakana_halfwidth_and_combining_kana_normalize(self):
        analyzer = FakeAnalyzer({"世": ["セ"], "間": ["ケン"], "学": ["ガク"], "校": ["コウ"]})
        for reading in ("せけん", "セケン", "ｾｹﾝ"):
            with self.subTest(reading=reading):
                self.assertEqual([p.reading for p in split_kanji_reading("世間", reading, analyzer)], ["せ", "けん"])
        self.assertEqual([p.reading for p in split_kanji_reading("学校", "か\u3099っこう", analyzer)],
                         ["がっ", "こう"])

    def test_normalized_kana_literals_keep_original_surface_offsets(self):
        analyzer = FakeAnalyzer({"甲": ["あ"], "乙": ["い"]})
        self.assertEqual(values(split_kanji_reading("甲ｶﾞ乙", "あがい", analyzer)),
                         [(0, 1, "甲", "あ"), (3, 4, "乙", "い")])

    def test_explicit_nichi_contraction_needs_lexicon_and_compound(self):
        analyzer = FakeAnalyzer({"日": ["ひ"], "本": ["ほん"]})
        self.assertIsNone(split_kanji_reading("日本", "にほん", analyzer))
        analyzer = FakeAnalyzer({"日": ["にち"]})
        self.assertIsNone(split_kanji_reading("日", "に", analyzer))
        self.assertIsNone(split_kanji_reading("日", "にっ", analyzer))

    def test_invalid_candidates_and_unmatched_suffix_are_rejected(self):
        analyzer = FakeAnalyzer({"学": [None, "", "学", "A", "が!", "っ", "がく"], "校": ["こう"]})
        self.assertEqual([p.reading for p in split_kanji_reading("学校", "がっこう", analyzer)], ["がっ", "こう"])
        self.assertIsNone(split_kanji_reading("学校", "がっこうです", analyzer))
        self.assertIsNone(split_kanji_reading("学校", "gakkou", analyzer))

    def test_empty_unknown_and_nonkanji_inputs(self):
        analyzer = FakeAnalyzer({"世": ["せ"]})
        for surface, reading in (("", "せ"), ("世", ""), ("かな", "かな"), ("世間", "せけん")):
            with self.subTest(surface=surface, reading=reading):
                self.assertIsNone(split_kanji_reading(surface, reading, analyzer))

    def test_supplementary_kanji_positions_and_immutable_results(self):
        analyzer = FakeAnalyzer({"𠮷": ["よし"], "田": ["た"]})
        result = split_kanji_reading("𠮷田", "よしだ", analyzer)
        self.assertEqual(result, [KanjiReading(0, 1, "𠮷", "よし"), KanjiReading(1, 2, "田", "だ")])
        with self.assertRaises(AttributeError):
            result[0].reading = "べつ"

    def test_lookup_failure_leaves_the_split_unknown(self):
        class Broken:
            def lookup_readings(self, surface):
                raise RuntimeError("dictionary unavailable")
        self.assertIsNone(split_kanji_reading("学校", "がっこう", Broken()))

    def test_long_unique_input_uses_one_lookup_and_no_recursion(self):
        analyzer = FakeAnalyzer({"甲": ["あ"]})
        count = kanji_split.MAX_SURFACE_CHARS
        result = split_kanji_reading("甲" * count, "あ" * count, analyzer)
        self.assertEqual(len(result), count)
        self.assertEqual(analyzer.calls, ["甲"])
        self.assertEqual(result[-1].start, count - 1)

    def test_ambiguous_long_input_does_not_enumerate_paths(self):
        analyzer = FakeAnalyzer({"甲": ["あ", "ああ"]})
        self.assertIsNone(split_kanji_reading("甲" * 128, "あ" * 192, analyzer))
        self.assertEqual(analyzer.calls, ["甲"])

    def test_input_limits_do_not_start_dictionary_work(self):
        analyzer = FakeAnalyzer({})
        self.assertIsNone(split_kanji_reading("甲" * (kanji_split.MAX_SURFACE_CHARS + 1), "あ", analyzer))
        self.assertIsNone(split_kanji_reading("甲", "あ" * (kanji_split.MAX_READING_CHARS + 1), analyzer))
        self.assertEqual(analyzer.calls, [])

    def test_lookup_limit_does_not_report_a_false_unique_answer(self):
        analyzer = FakeAnalyzer({"甲": ["あ"] * (kanji_split.MAX_LOOKUP_READINGS + 1), "乙": ["い"]})
        self.assertIsNone(split_kanji_reading("甲乙", "あい", analyzer))

    def test_transition_budget_does_not_report_a_partial_answer(self):
        analyzer = FakeAnalyzer({"甲": ["あ", "い", "う", "え", "お"], "乙": ["か"]})
        with patch.object(kanji_split, "MAX_TRANSITIONS", 3):
            self.assertIsNone(split_kanji_reading("甲乙", "あか", analyzer))

    def test_candidate_budget_does_not_discard_a_competing_division(self):
        analyzer = FakeAnalyzer({"甲": ["あ", "あい"], "乙": ["いう", "う"]})
        with patch.object(kanji_split, "MAX_CANDIDATES", 1):
            self.assertIsNone(split_kanji_reading("甲乙", "あいう", analyzer))


class InstalledKanjiDictionary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from subkit.reading.engine import Analyzer

        cls.analyzer = Analyzer.get()

    def test_installed_dictionary_spec_examples(self):
        for surface, reading, expected in (("世間", "せけん", ["せ", "けん"]),
                                            ("学校", "がっこう", ["がっ", "こう"]),
                                            ("迷宮都市", "めいきゅうとし", ["めい", "きゅう", "と", "し"]),
                                            ("日本", "にっぽん", ["にっ", "ぽん"]),
                                            ("日本", "にほん", ["に", "ほん"])):
            with self.subTest(surface=surface, reading=reading):
                result = split_kanji_reading(surface, reading, self.analyzer)
                self.assertIsNotNone(result)
                self.assertEqual([part.reading for part in result], expected)

    def test_installed_dictionary_irregulars_stay_manual(self):
        for surface, reading in (("今日", "きょう"), ("一人", "ひとり")):
            with self.subTest(surface=surface):
                self.assertIsNone(split_kanji_reading(surface, reading, self.analyzer))

    def test_installed_dictionary_okurigana_is_not_part_of_a_box(self):
        self.assertEqual(values(split_kanji_reading("学ぶ人", "まなぶひと", self.analyzer)),
                         [(0, 1, "学", "まな"), (2, 3, "人", "ひと")])


if __name__ == "__main__":
    unittest.main()
