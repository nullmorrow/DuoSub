"""Colloquial elongations keep original text and kanji-only ruby offsets."""

import copy
import json
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch

from subkit.core.readings import annotate_document
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.reading.dictionary import ReadingDict
from subkit.reading.engine import Morph, ReadingEngine, _tokenizer_text


# SPEC-2026-10-15-READINGS, section 一: text, whole reading, kanji, ruby.
EXAMPLES = (
    ("お疲れー", "つかれ", "疲", "つか"),
    ("帰ろー", "かえろ", "帰", "かえ"),
    ("帰るー", "かえる", "帰", "かえ"),
    ("寒ーい", "さむい", "寒", "さむ"),
    ("行こーよ", "いこ", "行", "い"),
    ("好きー", "すき", "好", "す"),
    ("早くー", "はやく", "早", "はや"),
    ("痛いー", "いたい", "痛", "いた"),
    ("怖ーい", "こわい", "怖", "こわ"),
    ("長ーい", "ながい", "長", "なが"),
    ("頑張れー！", "がんばれ", "頑張", "がんば"),
)


def document(text):
    member = Member("m", "g", "ja", "original", "dialogue", 0, 1000,
                    runs=[Run(text, ins="insert", a=0, b=len(text))])
    return Document("E", groups=[Group("g", members=["m"])], members={"m": member})


class LongVowelReadings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ReadingEngine()

    def test_lone_adjective_stems_and_ruby(self):
        for kanji, stem in (("硬", "かた"), ("寒", "さむ"), ("早", "はや"), ("痛", "いた"), ("凄", "すご")):
            for prefix in ("", "あっ ", "あっ"):
                for marks in ("ー", "ｰ", "ーー", "ｰーｰ"):
                    for suffix in ("", " ", "！", "カタカナ"):
                        text = prefix + kanji + marks + suffix
                        with self.subTest(text=text):
                            entry = self.engine.analyze(text)[0]
                            a = len(prefix)
                            self.assertEqual((entry.surface, entry.reading), (kanji, stem))
                            self.assertEqual([(p.start, p.end, p.reading) for p in entry.parts],
                                             [(a, a + 1, stem)])
                            doc = document(text)
                            self.assertEqual(annotate_document(doc, {}, self.engine), 1)
                            ann = next(iter(doc.annotations.values()))
                            self.assertEqual((ann.base, ann.text, ann.parts), (kanji, stem, [[0, 1, stem]]))
                            self.assertEqual(doc.members["m"].text, text)

    def test_adjective_stem_exclusions_keep_normal_readings(self):
        for text, surface, reading, ruby in (("手ー", "手", "て", "て"), ("色ー", "色", "いろ", "いろ"),
                                            ("皆ー", "皆", "みな", "みな"),
                                            ("ナイト様ー", "様", "さま", "さま"),
                                            ("寒ーい", "寒ーい", "さむい", "さむ"),
                                            ("論外ー", "論外", "ろんがい", "ろんがい")):
            for mark in ("ー", "ｰ"):
                original = text.replace("ー", mark)
                with self.subTest(text=original):
                    entry = self.engine.analyze(original)[0]
                    self.assertEqual((entry.surface, entry.reading), (surface.replace("ー", mark), reading))
                    self.assertEqual(entry.parts[0].reading, ruby)

    def test_adjective_stem_uses_first_matching_lexicon_reading(self):
        analyzer = Mock()
        analyzer.lookup_readings.return_value = ["こう", "かたい", "かてい"]
        analyzer.tokenize.return_value = [Morph("硬", 0, 1, "こう", (), "硬")]
        entry = ReadingEngine(analyzer=analyzer).analyze("硬ーー！")[0]
        self.assertEqual(entry.reading, "かた")
        analyzer.lookup_readings.assert_called_once_with("硬い")

    def test_adjective_stem_keeps_project_and_hint_readings(self):
        rd = ReadingDict()
        rd.merge_json({"words": {"硬": "てがき"}}, "project")
        self.assertEqual(ReadingEngine(rd).analyze("硬ー")[0].reading, "てがき")
        preferred = ReadingEngine(ReadingDict(prefer={"硬": "こう"})).analyze("硬ー")[0]
        self.assertEqual((preferred.reading, preferred.source), ("こう", "prefer"))
        hint = Mock(start=0, end=1, surface="硬", reading="ひんと")
        self.assertEqual(self.engine.analyze("硬ー", hints=[hint])[0].reading, "ひんと")

    def test_normalization_boundaries_and_length(self):
        cases = (("", ""), ("ーｰ", "ーｰ"), ("あーあｰ", "あ〜あ〜"),
                 ("漢ー字ｰ", "漢〜字〜"), ("\u3041ー\u309fｰ", "\u3041〜\u309f〜"),
                 ("カーカｰｶｰヶーヵｰ", "カーカｰｶｰヶーヵｰ"),
                 (" ー1ー！ｰー", " ー1ー！ｰー"), ("あーー", "あ〜ー"))
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(_tokenizer_text(text), expected)
                self.assertEqual(len(_tokenizer_text(text)), len(text))

    def test_every_spec_row_analyzes_with_original_surface(self):
        for original, reading, kanji, ruby in EXAMPLES:
            for mark in ("ー", "ｰ"):
                text = "えっ " + original.replace("ー", mark)
                with self.subTest(text=text):
                    entries = self.engine.analyze(text)
                    self.assertEqual(len(entries), 1)
                    entry = entries[0]
                    self.assertEqual(entry.surface, text[entry.start:entry.end])
                    self.assertEqual(entry.reading, reading)
                    a = text.index(kanji)
                    self.assertEqual([(p.start, p.end, p.reading) for p in entry.parts],
                                     [(a, a + len(kanji), ruby)])
                    self.assertNotIn("〜", repr(asdict(entry)))
                    self.assertNotIn("？", entry.reading)

    def test_every_spec_row_through_annotation_path(self):
        for original, reading, kanji, ruby in EXAMPLES:
            for mark in ("ー", "ｰ"):
                text = "えっ " + original.replace("ー", mark)
                with self.subTest(text=text):
                    doc = document(text)
                    runs = copy.deepcopy(doc.members["m"].runs)
                    self.assertEqual(annotate_document(doc, {}, self.engine), 1)
                    ann = next(iter(doc.annotations.values()))
                    entry = self.engine.analyze(text)[0]
                    self.assertEqual(ann.base, text[entry.start:entry.end])
                    self.assertEqual(ann.text, reading)
                    self.assertEqual(ann.parts, [[p.start - entry.start, p.end - entry.start, p.reading]
                                                 for p in entry.parts])
                    self.assertEqual([(ann.base[a:b], r) for a, b, r in ann.parts], [(kanji, ruby)])
                    self.assertEqual(doc.members["m"].text.encode("utf-8"), text.encode("utf-8"))
                    self.assertEqual(doc.members["m"].runs, runs)
                    self.assertNotIn("〜", json.dumps(asdict(ann), ensure_ascii=False))
                    if original in ("寒ーい", "怖ーい", "長ーい"):
                        self.assertIn(mark, ann.base)

    def test_katakana_and_kana_only_do_not_gain_annotations(self):
        for text in ("レフィーヤー", "ハロー", "ソードー", "すごーい", "すごｰい"):
            with self.subTest(text=text):
                self.assertEqual(self.engine.analyze(text), [])
                doc = document(text)
                self.assertEqual(annotate_document(doc, {}, self.engine), 0)
                self.assertEqual(doc.members["m"].text, text)

    def test_unaffected_results_are_byte_identical(self):
        for text in ("取り扱った荷物を迷宮都市へ運ぶ", "3人と二十日と1000万", "ハロー、レフィーヤー様！"):
            def encoded():
                return json.dumps([asdict(e) for e in self.engine.analyze(text)],
                                  ensure_ascii=False, sort_keys=True).encode("utf-8")
            with self.subTest(text=text):
                with patch("subkit.reading.engine._tokenizer_text", side_effect=lambda s: s):
                    before = encoded()
                self.assertEqual(encoded(), before)

    def test_residual_uses_original_offsets_and_keeps_dictionary_reading(self):
        rd = ReadingDict()
        rd.merge_json({"words": {"寒": "かん"}}, "episode")
        analyzer = Mock()
        analyzer.tokenize.side_effect = [[Morph("寒〜い暑〜い", 0, 6, "さむいあつい", (), "")],
                                        [Morph("〜", 0, 1, "きごう", (), "〜"),
                                         Morph("い", 1, 2, "い", (), "い"),
                                         Morph("暑〜い", 2, 5, "あつい", (), "暑い")]]
        entries = ReadingEngine(rd, analyzer).analyze("寒ーい暑ーい")
        self.assertEqual([call.args[0] for call in analyzer.tokenize.call_args_list], ["寒〜い暑〜い", "ーい暑〜い"])
        self.assertEqual([(e.surface, e.reading) for e in entries], [("寒", "かん"), ("暑ーい", "あつい")])
        self.assertEqual([(p.start, p.end, p.reading) for p in entries[1].parts], [(3, 4, "あつ")])

    def test_manual_confirmed_hidden_and_typed_readings_survive_refresh(self):
        for status, origin in (("confirmed", "analyzer"), ("auto", "manual"), ("hidden", "manual")):
            with self.subTest(status=status):
                doc = document("寒ーい")
                ann = Annotation("keep", "ruby", "m", [{"ins": "insert", "a": 0, "b": 3}],
                                 "寒ーい", "てがき", status=status, origin=origin, parts=[[0, 1, "てがき"]])
                doc.annotations[ann.id] = ann
                before = copy.deepcopy(asdict(ann))
                self.assertEqual(annotate_document(doc, {}, self.engine), 0)
                self.assertEqual(asdict(doc.annotations[ann.id]), before)
        rd = ReadingDict()
        rd.merge_json({"words": {"寒ーい": "てがき"}, "typed": {"寒ーい": ["たいぷ"]}}, "project")
        self.assertEqual(ReadingEngine(rd).analyze("寒ーい")[0].reading, "てがき")
        self.assertEqual(rd.typed_readings("寒ーい"), ["たいぷ"])
        self.assertEqual(ReadingEngine(ReadingDict(typed=rd.typed)).analyze("寒ーい")[0].reading, "さむい")

    def test_number_counter_residual_normalizes_and_keeps_original_surface(self):
        analyzer = Mock()
        analyzer.tokenize.return_value = [Morph("寒〜い", 0, 3, "さむい", (), "寒い")]
        engine = ReadingEngine(analyzer=analyzer)
        entry = engine._number_entry("3年寒ーい", Morph("3", 0, 1, "さん", (), "3"),
                                     Morph("年寒ーい", 1, 5, "unknown", (), "年寒い"))
        analyzer.tokenize.assert_called_once_with("寒〜い")
        self.assertEqual(entry.surface, "年寒ーい")
        self.assertNotIn("〜", entry.reading)
        self.assertEqual([(p.start, p.end, p.reading) for p in entry.parts],
                         [(1, 2, "ねん"), (2, 3, "さむ")])
