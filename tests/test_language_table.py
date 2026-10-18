"""Unified language registry and unchanged soma merge/export (Windows host)."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_languages import ROOT, document, member
from tests.ruby_policy import legacy_ruby_layout
from subkit.core.ops import Editor, phrase_breaks
from subkit.core.pinyin_notes import suggest_pinyin
from subkit.core.readings import suggest_readings
from subkit.core.sdh_en import analyzer_for
from subkit.core.units import is_interjection, split_range
from subkit.core.workspace import Project
from subkit.languages import LANGUAGES, get_language, lane_label, normalize_language_code
from subkit.model.serialize import document_to_json


class LanguageTable(unittest.TestCase):
    def test_every_language_has_complete_data_and_name_defaults(self):
        self.assertEqual(set(LANGUAGES), set("ja zh-Hans zh-Hant yue ko en vi fr de es pt it ru".split()))
        for code, row in LANGUAGES.items():
            with self.subTest(code=code):
                self.assertIs(get_language(code), row)
                self.assertTrue(row.display_name)
                self.assertIsInstance(row.word_spaces, bool)
                self.assertTrue(row.sentence_punctuation)
                self.assertTrue(row.phrase_punctuation)
                self.assertTrue(row.filler_words)
                self.assertEqual(analyzer_for(code)[0]["name"], row.analyzer_id)
                family = code in {"ja", "zh-Hans", "zh-Hant", "yue", "ko", "vi"}
                self.assertEqual(row.name_order, "family_first" if family else "given_first")
                self.assertEqual(row.name_separator, "" if family and code != "vi" else " ")
        self.assertEqual(get_language("yue").phrase_punctuation, get_language("zh-Hant").phrase_punctuation)

    def test_code_normalization_and_lane_labels(self):
        for code, expected in {"JA-jp": "ja", "en-US": "en", " ko_KR ": "ko", "pt-BR": "pt",
                               "zh": "zh-Hans", "ZH-cn": "zh-Hans", "zh-SG": "zh-Hans",
                               "zh-tw": "zh-Hant", "zh-HK": "zh-Hant", "zh-MO": "zh-Hant",
                               "ZH-hANT-cn": "zh-Hant", "zh-Hans-HK": "zh-Hans",
                               "YUE-HK": "yue", "xx-lATN-us": "xx-Latn-US"}.items():
            with self.subTest(code=code):
                self.assertEqual(normalize_language_code(code), expected)
                self.assertEqual(normalize_language_code(expected), expected)
        self.assertEqual(lane_label("original", {"original": "KO-kr"}), "原声（韩语）")
        self.assertEqual(lane_label("translation", {"translation": "VI-vn"}), "翻译（越南语）")

    def test_unknown_language_fallback_by_script(self):
        for text, spaces in [("你好 世界", False), ("かな カナ", False), ("ｶﾅ", False),
                             ("한글 자막", False), ("한", False), ("𠀀", False),
                             ("Hello world", True), ("Привет мир", True), ("Nguyễn Văn An", True), ("", True)]:
            with self.subTest(text=text):
                row = get_language("xx", text)
                self.assertEqual(row.word_spaces, spaces)
                self.assertEqual((row.analyzer_id, row.annotation_system, row.filler_words),
                                 ("generic", "none", frozenset()))
                self.assertEqual((row.name_order, row.name_separator), ("given_first", " "))
        for text, expected in [("你好 世界，回家。", ["你好", "世界，", "回家。"]),
                               ("Hello world, go home.", ["Hello world,", "go home."])]:
            self.assertEqual([text[a:b] for a, b, _ in split_range(text, 0, len(text), "xx")], expected)

    def test_korean_splits_punctuation_and_keeps_word_spaces(self):
        text = "안녕하세요 여러분, 집에 갑시다. 정말 좋아요!"
        pieces = split_range(text, 0, len(text), "KO-kr")
        self.assertEqual([text[a:b] for a, b, _ in pieces],
                         ["안녕하세요 여러분,", "집에 갑시다.", "정말 좋아요!"])
        self.assertEqual(phrase_breaks(member(text, "ko")), [a for a, _, _ in pieces[1:]])
        text = "값은 3.14 입니다. 좋아요"
        self.assertEqual([text[a:b] for a, b, _ in split_range(text, 0, len(text), "ko")],
                         ["값은 3.14 입니다.", "좋아요"])

    def test_fillers_include_vietnamese_accents_and_korean(self):
        for code in ("vi", "ko", "fr", "de", "es", "pt", "it", "ru"):
            for word in get_language(code).filler_words:
                with self.subTest(code=code, word=word):
                    self.assertTrue(is_interjection(word.upper() + "?!", code))
                    self.assertTrue(is_interjection(unicodedata.normalize("NFD", word), code))
            self.assertFalse(is_interjection("Hello world", code))
        self.assertEqual(get_language("vi").filler_words, frozenset("à ừ ờ ơ ủa hả nhé".split()))
        self.assertFalse(is_interjection("a", "vi"))
        self.assertFalse(is_interjection("嗯", "xx"))
        self.assertTrue(is_interjection("啊啊！", "yue"))

    def test_annotation_eligibility_follows_registry_in_both_lanes(self):
        for code in LANGUAGES:
            expected = "ruby" if code == "ja" else "pinyin" if code.startswith("zh-") else "none"
            self.assertEqual(get_language(code).annotation_system, expected)
            if expected == "ruby":
                continue
            for lane in ("original", "translation"):
                m = member("银行", code, lane)
                self.assertEqual(suggest_readings(m, None), [])
                self.assertEqual(bool(suggest_pinyin(m)), expected == "pinyin")
                doc = document(m)
                ed = Editor(doc, {})
                self.assertIsNone(ed.add_annotation(m.id, 0, 1, "ぎん", kind="ruby"))
        self.assertEqual(get_language("zh-HK").annotation_system, "pinyin")
        self.assertEqual(get_language("yue-HK").annotation_system, "none")

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires real ffmpeg/libass")
    @legacy_ruby_layout()
    def test_soma_fresh_merge_and_export_identity(self):
        source = ROOT / "test_projects/soma"
        def hashes():
            return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob("*") if p.is_file()}
        before = hashes()
        with tempfile.TemporaryDirectory(prefix="c5-soma-") as tmp:
            path = Path(tmp)
            shutil.copytree(source, path / "soma", copy_function=shutil.copyfile)
            project = Project.open(path / "soma")
            # Historical byte fingerprint uses the legacy ruby layout; X11 defaults to width limits.
            project.data.setdefault("settings", {}).setdefault("ruby", {})["max_width"] = {
                "dialogue": {"enabled": False}, "song": {"enabled": False}}
            ep = project.episode(project.episode_ids()[0])
            ass, _ = ep.build_export()
            ass.save(path / "before.ass")
            data = (path / "before.ass").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),
                             "f4e1d14b93d23e27ef2de064cc075cc0c8dc0b252aa938c157bf455e4b45af8e")  # 0.20.0: song translation lines one layer above the original (was 0.10.8)
            count = itertools.count(1)
            def ids(prefix):
                return prefix + format(next(count), "020x")
            with patch("subkit.core.merge.new_id", side_effect=ids), patch("subkit.core.readings.new_id", side_effect=ids):
                doc = ep.merge()
            book = project.character_book()
            normalized = json.loads(json.dumps(document_to_json(doc)))
            self.assertEqual(normalized.pop("coordinate_playres"), [1920, 1080])
            for m in normalized["members"].values():
                if (m.get("speaker") or {}).get("character"):
                    self.assertEqual(m["speaker"]["character"], book.resolve(m["speaker"]["label"]))
                    # TSV conversion assigns fresh character IDs in every copy.
                    m["speaker"]["character"] = m["speaker"]["label"]
            raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.assertEqual(hashlib.sha256(raw).hexdigest(),
                             "9618e384e2678c9f5c0cd119785aebb62c816784eb86c8c7a078ccd21dc70e2d")
            ass, _ = ep.build_export()
            ass.save(path / "merged.ass")
            merged = (path / "merged.ass").read_bytes()
            self.assertEqual(hashlib.sha256(merged).hexdigest(),
                             "cacacb18e4a6082cafbf913cb477f45151a937c893fd767539af698d317bfcf6")  # 0.20.0: song translation lines one layer above the original (was 0.10.8)
            scratch = Path(os.environ.get("TEMP", tempfile.gettempdir()))
            for name, value in (("c5-soma-before.ass", data), ("c5-soma-merge-before.json", raw),
                                ("c5-soma-merged-before.ass", merged)):
                baseline = scratch / name
                if baseline.exists():
                    expected = baseline.read_bytes()
                    if name.endswith(".json"):
                        record = json.loads(expected)
                        for m in record["members"].values():
                            if (m.get("speaker") or {}).get("character"):
                                m["speaker"]["character"] = m["speaker"]["label"]
                        expected = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    self.assertEqual(value, expected)
            print(f"C5 soma identity: {len(data)} / {len(merged)} export bytes; "
                  f"fresh merge {len(doc.groups)} groups / {len(doc.members)} members, exact baseline match")
        self.assertEqual(hashes(), before)


if __name__ == "__main__":
    unittest.main()
