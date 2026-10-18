"""Inline Sword Oratoria ED fixtures; no project/media/model access."""
from __future__ import annotations

import copy
import json
import unittest

from tests.test_lyrics_map import document
from subkit.core import karaoke as kara, lyrics_auto as la, lyrics_map as lm
from subkit.core.ops import Transaction
from subkit.model.document import Annotation
from subkit.model.serialize import document_from_json, document_to_json


# Original text and hand-timed beats, read once from S01E001 for this fixture.
KNOW = ("「分かってる…　分かってる…」", "「我知道...我都知道啦...」", [
    ("わ", 0, 2, 1348639, 1349106), ("かっ", 2, 4, 1349106, 1349856),
    ("て", 4, 5, 1349856, 1350330), ("る", 5, 8, 1350330, 1351500),
    ("わ", 8, 9, 1352434, 1352904), ("かっ", 9, 11, 1352904, 1353640),
    ("て", 11, 12, 1353640, 1354093), ("る", 12, 15, 1354093, 1355720)])
LOVE = ("いつか　きっと　愛を知るんだ", "總有一天 一定會 明白什麼是愛", [
    ("い", 0, 1, 1367420, 1367778), ("つ", 1, 2, 1367778, 1368011),
    ("か", 2, 4, 1368011, 1368245), ("き", 4, 5, 1368245, 1368802),
    ("っ", 5, 6, 1368802, 1368960), ("と", 6, 8, 1368960, 1370346),
    ("あ", 8, 9, 1370346, 1370675), ("い", 8, 9, 1370675, 1370847),
    ("を", 9, 10, 1370847, 1371178), ("し", 10, 11, 1371178, 1371753),
    ("る", 11, 12, 1371753, 1372300), ("ん", 12, 13, 1372300, 1372780),
    ("だ", 13, 14, 1372780, 1373760)])
TOMORROW = ("明日の君が　今日の君を", "明天的妳 看著今天的妳", [
    ("あ", 0, 2, 1373760, 1374105), ("す", 0, 2, 1374105, 1374620),
    ("の", 2, 3, 1374620, 1374863), ("き", 3, 4, 1374863, 1375368),
    ("み", 3, 4, 1375368, 1375806), ("が", 4, 6, 1375806, 1377560),
    ("きょ", 6, 8, 1377560, 1378193), ("う", 6, 8, 1378193, 1378394),
    ("の", 8, 9, 1378394, 1378641), ("き", 9, 10, 1378641, 1378920),
    ("み", 9, 10, 1378920, 1379552), ("を", 10, 11, 1379552, 1381360)])
# A 150 ms rest inside 窓 (ま｜ど) is a breath inside a word, not a phrase break.
MORNING = ("おはよう  窓開けて", "說聲早安 推開窗戶", [
    ("お", 0, 1, 1326700, 1326950), ("は", 1, 2, 1326950, 1327420),
    ("よ", 2, 3, 1327420, 1327900), ("う", 3, 6, 1327900, 1328140),
    ("ま", 6, 7, 1328140, 1328470), ("ど", 6, 7, 1328620, 1329071),
    ("あ", 7, 8, 1329071, 1329310), ("け", 8, 9, 1329310, 1329550),
    ("て", 9, 10, 1329550, 1330560)])


# S01E005 OP (rows add whether the beat reads a ruby base).  ない｜今 is い｜い: two words, not one long い.
TRUTH = ("瞳には嘘のない  今が映って", "倒映在你眼中 最真誠的當下", [
    ("ひ", 0, 1, 113500, 113660, True), ("と", 0, 1, 113660, 113850, True), ("み", 0, 1, 113850, 114270, True),
    ("に", 1, 2, 114270, 114440, False), ("は", 2, 3, 114440, 114808, False), ("う", 3, 4, 114808, 114908, True),
    ("そ", 3, 4, 114908, 115061, True), ("の", 4, 5, 115061, 115521, False), ("な", 5, 6, 115521, 115722, False),
    ("い", 6, 9, 115722, 115941, False), ("い", 9, 10, 115941, 116160, True), ("ま", 9, 10, 116763, 117050, True),
    ("が", 10, 11, 117223, 117410, False), ("う", 11, 12, 117410, 117838, True), ("つ", 11, 13, 117838, 118020, True),
    ("て", 13, 14, 118224, 118460, False)])
# を｜う(生) is not a long vowel; 生 / 生め are different places, not an anchor.
SYNERGY = ("相乗効果を生め", "產生無窮的可能", [
    ("そ", 0, 2, 146139, 146284, True), ("う", 0, 2, 146284, 146430, True), ("じょ", 0, 2, 146430, 146601, True),
    ("う", 0, 2, 146601, 146772, True), ("こ", 2, 4, 146772, 146932, True), ("う", 2, 4, 146932, 147092, True),
    ("か", 2, 4, 147092, 147490, True), ("を", 4, 5, 147490, 147752, False), ("う", 5, 6, 147752, 148011, True),
    ("め", 6, 7, 148011, 148420, False)])
SHINE = ("照らし合い輝いて", "相互輝映的絢爛光芒", [
    ("て", 0, 1, 138993, 139133, True), ("ら", 1, 2, 139133, 139313, False), ("し", 2, 3, 139313, 139503, False),
    ("あ", 3, 4, 139503, 139633, True), ("い", 4, 5, 139633, 139774, False), ("か", 5, 6, 139774, 139934, True),
    ("が", 5, 6, 139934, 140387, True), ("や", 5, 6, 140387, 140724, True), ("い", 6, 7, 140724, 140864, False),
    ("て", 7, 8, 140864, 141140, False)])


def fixture(values, **kw):
    text, translation, rows = values
    beats = [kara.Beat(a, b, i, sung=sung, t0=t0, t1=t1, by="manual", base_text=text[a:b],
                       ann="a_ruby" if ruby else None)
             for i, (sung, a, b, t0, t1, *flag) in enumerate(rows) for ruby in [bool(flag and flag[0])]]
    return beats, la.redistribute(beats, translation, original_text=text, **kw)


def mapping(values, **kw):
    beats, segments = fixture(values, **kw)
    return "".join(s.text + "（" + "".join(b.sung for b in beats[s.start.beat:s.end.beat + 1]) + "）"
                   for s in segments)


class WordStartsAndDistantAnchors(unittest.TestCase):
    def test_word_start_vowel_is_not_a_long_note_across_a_phrase(self):
        self.assertEqual(mapping(TRUTH, splits=False, follow_notes=True),
                         "倒（ひと）映（み）在（には）你（うそ）眼（の）中 （ない）最（い）真（ま）誠（が）的（う）當（つ）下（て）")

    def test_distant_shared_characters_do_not_crowd_the_translation(self):
        self.assertEqual(mapping(SYNERGY, splits=False, follow_notes=True),
                         "產（そう）生（じょう）無（こう）窮（か）的（を）可（う）能（め）")
        self.assertEqual(mapping(SHINE, splits=False, follow_notes=True),
                         "相（て）互（ら）輝（し）映（あい）的（か）絢（が）爛（や）光（い）芒（て）")


class PhraseRedistribution(unittest.TestCase):
    def test_know_two_phrases_and_la_second_half_of_last_ru(self):
        beats, segments = fixture(KNOW)
        self.assertEqual("".join(s.text for s in segments), KNOW[1])
        self.assertEqual([s.start.beat for s in segments[:3]], [0, 2, 3])
        self.assertEqual(segments[2].end, la.Reference(3, 1))
        self.assertEqual(segments[3].start, la.Reference(4, 0))
        self.assertEqual(segments[-1].start, la.Reference(7, .5))
        self.assertEqual(segments[-1].end, la.Reference(7, 1))
        self.assertEqual(la.phrase_groups(beats), [(0, 4), (4, 8)])

    def test_love_three_phrases_correspond(self):
        _beats, segments = fixture(LOVE)
        for a, z, first, last in ((0, 4, 0, 2), (4, 7, 3, 5), (7, 13, 6, 12)):
            self.assertEqual(segments[a].start, la.Reference(first, 0))
            self.assertEqual(segments[z - 1].end, la.Reference(last, 1))
        self.assertEqual("".join(s.text for s in segments), LOVE[1])

    def test_tomorrow_shared_character_anchors(self):
        _beats, segments = fixture(TOMORROW)
        self.assertEqual(segments[0].start, la.Reference(0, 0))  # 明 on 明日
        today = next(s for s in segments if s.text == "今")
        self.assertIn(today.start.beat, (6, 7))                 # 今 on 今日
        self.assertEqual(segments[3].end, la.Reference(5, 1))
        self.assertEqual(segments[4].start, la.Reference(6, 0))

    def test_rest_inside_a_word_is_not_a_phrase_break(self):
        beats, segments = fixture(MORNING)
        self.assertEqual(la.phrase_groups(beats), [(0, 4), (4, 9)])
        self.assertEqual(segments[3].text.strip(), "安")
        self.assertEqual(segments[3].end, la.Reference(3, 1))
        self.assertEqual(segments[4].start, la.Reference(4, 0))
        mapping = lm.auto_mapping(beats, MORNING[1])
        self.assertIn("早安 （よう）推（ま", mapping)

    def test_rest_between_words_is_a_weak_break(self):
        beats, _ = fixture(KNOW)
        beats[3].base_text = "る"
        beats[4].t0 = beats[3].t1 + 150
        self.assertEqual(la.phrase_breaks(beats), {4: False})
        beats[3].base_text = "る…　"
        self.assertEqual(la.phrase_breaks(beats), {4: True})

    def test_phrase_proposal_has_no_splits_and_keeps_format(self):
        for values in (KNOW, LOVE, TOMORROW):
            with self.subTest(text=values[0]):
                beats, _segments = fixture(values)
                mapping = lm.auto_mapping(beats, values[1])
                parsed = lm.parse(mapping, beats, values[1])
                self.assertEqual("".join(s.text for s in parsed), values[1])
                self.assertEqual(lm.format_mapping(parsed, beats), mapping)
                self.assertTrue(all(s.last > s.first for s in parsed))
                if values == TOMORROW:
                    today = next(s for s in parsed if "今" in s.text)
                    self.assertLess(today.first, 8)
                    self.assertGreater(today.last, 6)

    def test_morae_katakana_small_kana_closure_long_and_n(self):
        self.assertEqual(la.morae("キャッーん"), 3)
        self.assertEqual(la.morae("っ"), .5)
        self.assertEqual(la.morae("きゃ"), 1)

    def test_unit_offsets_preserve_punctuation_words_and_whitespace(self):
        text = " 「你好...」  don't stop! "
        units = la.translation_units(text)
        self.assertEqual([u.text for u in units], ["你", "好", "don't", "stop"])
        self.assertEqual("".join(text[u.a:u.b] for u in units), text)
        self.assertEqual([u.brk for u in units], [False, True, True, True])

    def test_rest_boundary_threshold(self):
        beats, _ = fixture(KNOW)
        beats[3].base_text = "る"
        beats[4].t0 = beats[3].t1 + 149
        self.assertEqual(la.phrase_groups(beats), [(0, 8)])
        beats[4].t0 += 1
        self.assertEqual(la.phrase_groups(beats), [(0, 4), (4, 8)])

    def test_zero_duration_empty_and_excess_units(self):
        self.assertEqual(la.redistribute([], "你好"), [])
        beats = [kara.Beat(0, 1, 0, sung="か", t0=0, t1=0, base_text="か")]
        self.assertEqual(la.redistribute(beats, " ... "), [])
        for text in ("你", "你好啊", "你好世界呀", "one two three four five"):
            segments = la.redistribute(beats, text)
            self.assertEqual("".join(s.text for s in segments), text)
            self.assertLessEqual(len(segments), 3)

    def test_simplified_traditional_and_shinjitai_anchors(self):
        self.assertEqual(la.norm_han("楽気変", True), "乐气变")
        self.assertEqual(la.norm_han("樂氣變"), "乐气变")


class RedistributionStorage(unittest.TestCase):
    def prepared(self):
        doc, ed = document("かな", "你好啊", sung=[("か", 300), ("な", 900)])
        return doc, ed

    def test_split_links_proportional_times_follow_and_undo(self):
        doc, ed = self.prepared()
        before, count = copy.deepcopy(doc.karaoke), len(ed.history.done)
        report = ed.redistribute_translation(["g", "g"])
        self.assertEqual(report["updated"], 1)
        self.assertEqual(report["updated_members"], 1)
        self.assertEqual(len(ed.history.done), count + 1)
        self.assertEqual(ed.history.done[-1].label, "重新分配翻译拍")
        cn = doc.members["cn"]
        beats = kara.effective(doc, cn).beats
        self.assertEqual([(b.t0, b.t1) for b in beats], [(1000, 1300), (1300, 1750), (1750, 2200)])
        self.assertNotIn("end", beats[1].link)
        self.assertNotIn("start", beats[2].link)
        self.assertEqual(beats[1].link["start"], kara.beat_link(kara.effective(doc, doc.members["jp"]).beats[1], "start"))
        stored = copy.deepcopy(doc.karaoke)
        ed.history.undo()
        self.assertEqual(doc.karaoke, before)
        ed.history.redo()
        self.assertEqual(doc.karaoke, stored)
        originals = kara.effective(doc, doc.members["jp"]).beats
        originals[1].t1 = 2600
        ed.set_karaoke("jp", originals)
        self.assertEqual([(b.t0, b.t1) for b in kara.effective(doc, cn).beats],
                         [(1000, 1300), (1300, 1950), (1950, 2600)])
        ed.history.undo()
        self.assertEqual(doc.karaoke, stored)
        restored = document_from_json(json.loads(json.dumps(document_to_json(doc))))
        self.assertEqual(restored.karaoke, doc.karaoke)

    def test_twins_effect_and_unit_preserved(self):
        doc, ed = self.prepared()
        twin = copy.deepcopy(doc.members["cn"])
        twin.id = "glow"
        twin.runs[0].ins = "i_glow"
        doc.members[twin.id] = twin
        doc.groups[0].members.append(twin.id)
        doc.structure_changed()
        ed.set_karaoke("cn", kara.effective(doc, doc.members["cn"]).beats,
                       state="placeholder", unit="mora", effect={"preset": "ko"})
        report = ed.redistribute_translation(["g"])
        self.assertEqual((report["updated"], report["updated_members"]), (1, 2))
        a, b = kara.effective(doc, doc.members["cn"]), kara.effective(doc, twin)
        self.assertEqual([(x.t0, x.t1) for x in a.beats], [(x.t0, x.t1) for x in b.beats])
        self.assertEqual((a.unit, a.kind), ("mora", "ko"))
        self.assertNotEqual(a.beats[0].core, b.beats[0].core)

    def test_automatic_protection_legacy_placeholder_and_taps(self):
        for mode, updated in (("none", 1), ("legacy", 1), ("placeholder", 1), ("custom", 0), ("tap", 0), ("drag", 0)):
            with self.subTest(mode=mode):
                doc, ed = self.prepared()
                originals = kara.effective(doc, doc.members["jp"]).beats
                if mode in ("legacy", "tap", "drag"):
                    ed.apply_lyrics_mapping({"g": lm.proportional_mapping(originals, "你好啊")})
                elif mode == "custom":
                    ed.apply_lyrics_mapping({"g": "你好啊（かな）"})
                elif mode == "placeholder":
                    ed.set_karaoke("cn", kara.effective(doc, doc.members["cn"]).beats, state="placeholder")
                if mode in ("tap", "drag"):
                    beats = kara.effective(doc, doc.members["cn"]).beats
                    beats[0].by = mode
                    ed.set_karaoke("cn", beats)
                before = copy.deepcopy(doc.karaoke)
                report = ed.redistribute_translation(["g"], only_uncustomized=True)
                self.assertEqual(report["updated"], updated)
                if not updated:
                    self.assertEqual(doc.karaoke, before)
                self.assertEqual(ed.redistribute_translation(["g"])["updated"], 1)

    def test_given_transaction_no_nested_history_and_atomic_rollback(self):
        doc, ed = self.prepared()
        before, count = copy.deepcopy(doc.karaoke), len(ed.history.done)
        tx = Transaction(doc, "outer")
        self.assertEqual(la.redistribute_in_transaction(ed, tx, ["g"], only_uncustomized=False)["updated"], 1)
        self.assertEqual(len(ed.history.done), count)
        tx.undo()
        self.assertEqual(doc.karaoke, before)
        with self.assertRaises(KeyError):
            ed.redistribute_translation(["g", "missing"])
        self.assertEqual(doc.karaoke, before)
        self.assertEqual(len(ed.history.done), count)

    def test_auto_alignment_redistributes_in_same_undo_step(self):
        doc, ed = self.prepared()
        before, count = copy.deepcopy(doc.karaoke), len(ed.history.done)
        report = ed.apply_auto_karaoke([{"mid": "jp", "beats": [(1100, 1400, 1, False), (1400, 2600, 1, False)]}],
                                     translations=True, overwrite=True)
        self.assertEqual(report["translation_report"]["updated"], 1)
        self.assertEqual(len(ed.history.done), count + 1)
        ed.history.undo()
        self.assertEqual(doc.karaoke, before)

    def test_placeholder_splits_become_timed_after_original_retap(self):
        doc, ed = document("かな", "你好啊")
        ed.redistribute_translation(["g"])
        self.assertFalse(kara.effective(doc, doc.members["cn"]).complete)
        originals = kara.effective(doc, doc.members["jp"]).beats
        for beat, a, z in zip(originals, (1000, 1300), (1300, 2200)):
            beat.t0, beat.t1, beat.by = a, z, "tap"
        ed.set_karaoke("jp", originals)
        self.assertTrue(kara.effective(doc, doc.members["cn"]).complete)

    def test_derived_base_text_preserves_ruby_and_separators(self):
        doc, _ = document("明日　かな", "明天你好")
        doc.annotations["ruby"] = Annotation("ruby", "ruby", "jp", [{"ins": "i_jp", "a": 0, "b": 2}],
                                             "明日", "あす", parts=[[0, 2, "あす"]])
        beats = kara.effective(doc, doc.members["jp"]).beats
        self.assertEqual([b.base_text for b in beats], ["明日　", "明日　", "か", "な"])
        self.assertEqual(la.phrase_groups(beats), [(0, 2), (2, 4)])
        self.assertTrue(lm.auto_mapping(beats, "明天你好").startswith("明"))


if __name__ == "__main__":
    unittest.main()
