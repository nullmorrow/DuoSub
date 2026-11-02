"""L2a validation, existing edits/links, and read-only finished Utena material."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass import AssDocument
from subkit.ass.text import insert_at_visible
from subkit.core.export import ExportOptions, build_ass
from subkit.core import karaoke as kara, lyrics_map as lm
from subkit.core.ops import Editor
from subkit.core.readings import visible_map
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json
from subkit.styles_default import default_styles_ass


def document(kana="かなよ", translation="你好世界", sung=None):
    doc = Document("mapping", groups=[Group("g", "song", ["jp", "cn"])])
    for mid, text, lane, lang in (("jp", kana, "original", "ja"), ("cn", translation, "translation", "zh-Hant")):
        doc.members[mid] = Member(mid, "g", lang, lane, "song", 1000, 10000, style="OP." + mid.upper(),
                                  runs=[Run(text, ins="i_" + mid, a=0, b=len(text))])
    ed = Editor(doc, {})
    if sung is not None:
        # Legacy kf boundaries can differ from the default syllable partition.
        ends, pos = set(), 0
        nonempty = [(text, duration) for text, duration in sung if text]
        for text, _duration in nonempty[:-1]:
            pos += len(text)
            ends.add(pos)
        parts = [p for b in kara.derive(doc, doc.members["jp"]) for p in b.parts]
        edits = [{"op": "split" if p.a in ends else "merge", "at": p.at} for p in parts[1:]]
        beats = kara.derive(doc, doc.members["jp"], edits=edits)
        assert len(beats) == len(nonempty)
        t = 1000
        iterator = iter(beats)
        for text, duration in sung:
            if not text:
                t += duration
                continue
            b = next(iterator)
            b.t0, b.t1, b.by = t, t + duration, "import"
            t += duration
        ed.set_karaoke("jp", beats, edits=edits)
    return doc, ed


class LyricsMapTests(unittest.TestCase):
    def test_roundtrip_punctuation_whitespace_and_katakana(self):
        doc, _ = document("ガーなよ", " 你，好！ 世界。")
        originals = kara.effective(doc, doc.members["jp"]).beats
        segments = lm.parse("你好(ガー)世界（なよ）", originals, " 你，好！ 世界。")
        self.assertEqual([s.text for s in segments], [" 你，好！ ", "世界。"])
        self.assertEqual(lm.parse(lm.format_mapping(segments, originals), originals, " 你，好！ 世界。"), segments)
        self.assertEqual([(s.first, s.last) for s in segments], [(0, 1), (1, 3)])

    def test_exact_errors_and_no_partial_beats(self):
        doc, _ = document()
        beats = kara.effective(doc, doc.members["jp"]).beats
        for mapping, message in (("你好（か）世界（なよ", "括号不成对"),
                                 ("你好（か)世界（なよ）", "括号不成对"),
                                 ("你好（）世界（かなよ）", "有一段没有对应任何拍"),
                                 ("你好（か）世界", "有一段没有对应任何拍"),
                                 ("你好（か）世界（のよ）", "括号里的假名和原声拍对不上：第 2 拍应该是「な」"),
                                 ("你好（かな）世界（よ）多（よ）", "翻译文字对不上：多了「多」"),
                                 ("你好（かなよ）", "翻译文字对不上：少了「世界」")):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError) as error:
                lm.parse(mapping, beats, "你好世界")
            self.assertEqual(str(error.exception), message)
        doc, _ = document("ガー", "你")
        with self.assertRaisesRegex(ValueError, "第 1 拍应该是「がー」"):
            lm.parse("你（が）", kara.effective(doc, doc.members["jp"]).beats, "你")

    def test_auto_mapping_word_units_and_fewer_original_beats(self):
        for kana, text in (("かなよ", "Just a long way!"), ("かなよ", "你好"), ("か", "你好世界"),
                           ("かなよ", "Hello world"), ("かなよ", "Hello")):
            doc, _ = document(kana, text)
            beats = kara.effective(doc, doc.members["jp"]).beats
            segments = lm.parse(lm.auto_mapping(beats, text), beats, text)
            self.assertTrue(all(s.last > s.first for s in segments))
            self.assertEqual("".join(s.text for s in segments), text)
            if text == "Just a long way!":
                self.assertEqual([s.text for s in segments], ["Just ", "a ", "long way!"])

    def test_auto_mapping_phrase_boundaries_replace_proportional_result(self):
        doc, _ = document("かなよ　あいう", "你好 世界啊啊")
        beats = kara.effective(doc, doc.members["jp"]).beats
        proposal = lm.auto_mapping(beats, "你好 世界啊啊")
        self.assertNotEqual(proposal, lm.proportional_mapping(beats, "你好 世界啊啊"))
        segments = lm.parse(proposal, beats, "你好 世界啊啊")
        second = next(s for s in segments if s.text.startswith("世"))
        self.assertEqual(second.first, 3)
        self.assertEqual(segments[-1].last, 6)

    def test_replaces_edits_taps_links_single_undo_save_and_retap(self):
        doc, ed = document(sung=[("か", 300), ("な", 400), ("よ", 500)])
        translation = doc.members["cn"]
        beats = kara.effective(doc, translation).beats
        ed.set_karaoke("cn", beats)
        ed.merge_beats("cn", kara.boundary_key(doc, translation, 1))
        before, count = copy.deepcopy(doc.karaoke), len(ed.history.done)
        ed.apply_lyrics_mapping({"g": "你好（かな）世界（よ）"})
        self.assertEqual(len(ed.history.done), count + 1)
        self.assertEqual(ed.history.done[-1].label, "歌词对应")
        self.assertEqual([b.sung for b in kara.effective(doc, translation).beats], ["你好", "世界"])
        after = copy.deepcopy(doc.karaoke)
        ed.history.undo()
        self.assertEqual(doc.karaoke, before)
        ed.history.redo()
        self.assertEqual(doc.karaoke, after)
        reopened = document_from_json(json.loads(json.dumps(document_to_json(doc), ensure_ascii=False)))
        self.assertEqual(lm.current_mapping(reopened, reopened.members["jp"], reopened.members["cn"]), "你好（かな）世界（よ）")
        originals = kara.effective(doc, doc.members["jp"]).beats
        for b in originals:
            b.t0 += 123
            b.t1 += 123
            b.by = "tap"
        ed.set_karaoke("jp", originals)
        translated = kara.effective(doc, translation).beats
        self.assertEqual([(b.t0, b.t1) for b in translated], [(1123, 1823), (1823, 2323)])
        self.assertEqual(translation.text, "你好世界")

    def test_multiple_groups_atomic_failure_and_one_undo(self):
        doc, ed = document()
        other, _ = document("あいう", "第二行")
        group = copy.deepcopy(other.groups[0])
        group.id, group.members = "g2", ["jp2", "cn2"]
        doc.groups.append(group)
        for old in other.members.values():
            new = copy.deepcopy(old)
            new.id, new.group = old.id + "2", "g2"
            new.runs[0].ins += "2"
            doc.members[new.id] = new
        doc.structure_changed()
        before = copy.deepcopy(doc.karaoke)
        with self.assertRaises(ValueError):
            ed.apply_lyrics_mapping({"g": "你好（かな）世界（よ）", "g2": "坏"})
        self.assertEqual(doc.karaoke, before)
        count = len(ed.history.done)
        ed.apply_lyrics_mapping({"g": "你好（かな）世界（よ）", "g2": "第二（あい）行（う）"})
        self.assertEqual(len(ed.history.done), count + 1)
        ed.history.undo()
        self.assertEqual(doc.karaoke, before)

    def test_english_several_words_one_beat_and_one_word_several_beats(self):
        doc, ed = document("かなよ", "Hello, my world!")
        ed.apply_lyrics_mapping({"g": "Hello（かな）my world（よ）"})
        self.assertEqual([visible_map(doc.members["cn"])[0].text[b.a:b.b]
                          for b in kara.effective(doc, doc.members["cn"]).beats], ["Hello, ", "my world!"])

    def test_mapping_before_tapping_is_placeholder_until_all_originals_timed(self):
        doc, ed = document()
        ed.apply_lyrics_mapping({"g": "你好世界（かなよ）"})
        self.assertFalse(kara.effective(doc, doc.members["cn"]).complete)
        originals = kara.effective(doc, doc.members["jp"]).beats
        originals[0].by = originals[-1].by = "tap"
        ed.set_karaoke("jp", originals)
        self.assertFalse(kara.effective(doc, doc.members["cn"]).complete)
        originals[1].by = "tap"
        ed.set_karaoke("jp", originals)
        self.assertTrue(kara.effective(doc, doc.members["cn"]).complete)
        ed.history.undo()
        self.assertFalse(kara.effective(doc, doc.members["cn"]).complete)

    def test_real_finished_utena_roundtrip_retap_export_and_unchanged_source(self):
        source = Path('./sample-media/少女革命ウテナ/S02E003 - [LoliHouse] Shoujo Kakumei Utena - 16 [BDRip 1440x1080 HEVC-10bit AACx2].zh-Hant.ass')
        if not source.exists():
            self.skipTest("finished Utena material unavailable")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="l2a-real-") as scratch:
            copied = Path(scratch) / "finished.ass"
            shutil.copyfile(source, copied)
            ass = AssDocument.parse(copied.read_text(encoding="utf-8-sig"))
            selected = [e for e in ass.events if e.kind == "Comment" and "KaraOKMapping:" in e.text
                        and any(marker in e.text for marker in ("KaraOKMapping:果", "KaraOKMapping:在", "臉頰("))]
            self.assertGreaterEqual(len(selected), 3)
            for event in selected:
                mapping = re.search(r"\{KaraOKMapping:(.*?)\}", event.text).group(1)
                translation = re.search(r"\{CNOrigin:(.*?),JPOrigin:", event.text).group(1)
                sung = [(text, int(duration) * 10) for duration, text in re.findall(r"\{\\kf(\d+)\}([^{}]*)", event.text.split(r"\N")[0])]
                doc, ed = document("".join(s for s, _ in sung), translation, sung)
                originals = kara.effective(doc, doc.members["jp"]).beats
                segments = lm.parse(mapping, originals, translation)
                self.assertEqual(lm.parse(lm.format_mapping(segments, originals), originals, translation), segments)
                ed.apply_lyrics_mapping({"g": mapping})
                for b in originals:
                    b.t0 += 170
                    b.t1 += 170
                    b.by = "tap"
                ed.set_karaoke("jp", originals)
                view = kara.effective(doc, doc.members["cn"])
                expected = [(originals[s.first].t0, originals[s.last - 1].t1) for s in segments]
                self.assertEqual([(b.t0, b.t1) for b in view.beats], expected)
                vt, ids = visible_map(doc.members["cn"])
                exported = insert_at_visible(vt.text, kara.base_inserts(view, ids, 100))
                released, report = build_ass(doc, {}, AssDocument.parse(default_styles_ass()), Path(scratch),
                                             ExportOptions(mode="release", ruby=False))
                self.assertEqual(released.events[-1].text, exported)
                spans = [(int(duration), text) for duration, text in re.findall(r"\{\\kf(\d+)\}([^{}]*)", exported) if text]
                self.assertEqual(spans, [((z - a) // 10, s.text) for (a, z), s in zip(expected, segments)])
                print(f"Utena L2a: {translation}; {len(originals)} original beats -> {len(segments)} segments; roundtrip/retap/kf OK")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        print("Utena source SHA256 unchanged:", before)


if __name__ == "__main__":
    unittest.main()
