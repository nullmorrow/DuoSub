"""D1c: structured language names, lossless migration, readings and export."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.ruby_policy import legacy_ruby_layout
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.core.people import Character, CharacterBook, NameForm
from subkit.core.pinyin_notes import annotate_pinyin, suggest_pinyin
from subkit.core.review import build_packet
from subkit.core.workspace import Project
from subkit.model.serialize import document_to_json
from tests.test_people_use import member, document


class PeopleNameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="d1c-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Project.create(self.root / "project", "Names")
        self.book = self.project.character_book()

    def soma_character(self):
        return self.book.add_character({
            "ja": NameForm("幸平", "創真", "ゆきひら", "そうま", short="family"),
            "zh-Hans": NameForm("幸平", "创真", "xìng píng", "chuàng zhēn", short="given"),
            "en": NameForm("Yukihira", "Soma", short="given")})

    def test_documented_name_compositions_and_language_defaults(self):
        ch = self.soma_character()
        self.assertEqual([ch.full_name(l) for l in ("ja", "zh-Hans", "en")],
                         ["幸平創真", "幸平创真", "Soma Yukihira"])
        self.assertEqual([ch.name(l) for l in ("ja", "zh-Hans", "en")], ["幸平", "创真", "Soma"])
        meg = self.book.add_character({"ja": NameForm(given="めぐみん"),
                                       "zh-Hans": NameForm(given="惠惠", given_reading="huì huì"),
                                       "en": NameForm(given="Megumin")})
        self.assertEqual([meg.full_name(l) for l in ("ja", "zh-Hans", "en")], ["めぐみん", "惠惠", "Megumin"])
        sheldon = self.book.add_character({
            "en": NameForm("Cooper", "Sheldon"),
            "ja": NameForm("クーパー", "シェルドン", order="given_first", separator="・"),
            "zh-Hans": NameForm("库珀", "谢尔顿", order="given_first", separator="·")})
        self.assertEqual([sheldon.full_name(l) for l in ("ja", "zh-Hans", "en")],
                         ["シェルドン・クーパー", "谢尔顿·库珀", "Sheldon Cooper"])
        for lang, expected in [("ko", "姓名"), ("vi", "姓 名"), ("fr", "名 姓")]:
            self.assertEqual(Character("chr_test", {lang: NameForm("姓", "名")}).full_name(lang), expected)
        self.assertEqual(ch.name("ja", "family"), "幸平")
        self.assertEqual(ch.name("ja", "given"), "創真")
        self.assertEqual(self.book.resolve("幸平創真"), ch.id)
        self.assertIsNone(self.book.resolve("幸平"))

    def test_chinese_variant_fallback_then_generic_and_raw_label(self):
        ch = self.book.add_character({"zh-Hant": NameForm(given="繁體"), "zh": NameForm(given="通用"),
                                      "ja": NameForm(given="日本語")})
        self.assertEqual(ch.full_name("zh-Hans"), "繁體")
        self.book.update_character(ch.id, names={"zh": NameForm(given="通用")})
        self.assertEqual(ch.full_name("zh-Hant"), "通用")
        self.assertEqual(self.book.display_name(ch.id, "en", "raw"), "通用")

    def test_v1_upgrade_keeps_exact_backup_ids_unknown_fields_and_repeats_safely(self):
        data = {"format": "subkit-characters", "format_version": 1, "future": [1],
                "characters": [{"id": "chr_old", "names": {"ja": "幸平 創真", "ja-kana": "ゆきひら そうま",
                                 "en": "Sheldon Cooper"}, "aliases": ["創真"], "future_character": {"x": 2}}],
                "crowd": ["男1"], "not_person": ["音"], "pending": [{"label": "不明", "count": 1}]}
        self.book.path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(data, ensure_ascii=False, indent=3) + "\r\n").encode("utf-8")
        self.book.path.write_bytes(raw)
        book = CharacterBook.load(self.book.path)
        backup = self.book.path.with_name("characters.v1.bak.json")
        self.assertEqual(backup.read_bytes(), raw)
        self.assertEqual(book.to_json()["format_version"], 2)
        ch = book.characters[0]
        self.assertEqual((ch.id, ch.names["ja"].family, ch.names["ja"].given), ("chr_old", "幸平", "創真"))
        self.assertEqual(ch.names["en"].given, "Sheldon Cooper")
        self.assertEqual(ch.names["en"].family, "")
        self.assertEqual((book.extra, ch.extra), ({"future": [1]}, {"future_character": {"x": 2}}))
        self.assertEqual((book.crowd, book.not_person, book.pending), (data["crowd"], data["not_person"], data["pending"]))
        migrated = book.path.read_bytes()
        self.assertEqual(CharacterBook.load(book.path).to_json(), book.to_json())
        self.assertEqual((book.path.read_bytes(), backup.read_bytes()), (migrated, raw))
        self.assertFalse(migrated.startswith(b"\xef\xbb\xbf"))

    def test_v1_converted_parts_and_v2_unknown_name_fields_round_trip(self):
        ch = Character.from_json({"id": "chr_old", "names": {"ja": "幸平創真", "ja-kana": "ゆきひらそうま"},
                                  "name_parts": ["幸平", "創真"], "kana_parts": ["ゆきひら", "そうま"]})
        self.assertEqual((ch.names["ja"].family, ch.names["ja"].given_reading), ("幸平", "そうま"))
        newer = Character.from_json({"id": "chr_new", "names": {"en": {"given": "Name", "future": True}}})
        self.assertTrue(newer.to_json()["names"]["en"]["future"])

    def test_locked_japanese_parts_full_name_and_chinese_polyphonic_names(self):
        self.soma_character()
        engine = self.project.reading_engine()
        entries = engine.analyze("幸平創真と幸平と創真")
        self.assertEqual([(e.surface, e.reading, e.locked) for e in entries if e.surface in ("幸平創真", "幸平", "創真")],
                         [("幸平創真", "ゆきひらそうま", True), ("幸平", "ゆきひら", True), ("創真", "そうま", True)])
        ch = self.book.add_character({"zh-Hans": NameForm("单", "行", "shan4", "hang2")})
        self.assertEqual((ch.names["zh-Hans"].family_reading, ch.names["zh-Hans"].given_reading), ("shàn", "háng"))
        self.assertEqual(CharacterBook.load(self.book.path).character(ch.id).names["zh-Hans"].family_reading, "shàn")
        m = member("cn", "单行与幸平创真", lang="zh-Hans")
        anns = suggest_pinyin(m, self.project.pinyin_words(m.lang))
        self.assertEqual([a.text for a in anns], ["shàn", "háng", "yǔ", "xìng", "píng", "chuàng", "zhēn"])
        self.assertTrue(all(a.origin == "rule" for a in anns if a.base != "与"))

    def test_pinyin_book_edit_refresh_preserves_manual_hidden_and_undo(self):
        ch = self.book.add_character({"zh-Hans": NameForm(given="单行", given_reading="shàn háng")})
        m = member("cn", "单行单行", lang="zh-Hans")
        doc = document(m)
        annotate_pinyin(doc, m, self.project.pinyin_words(m.lang))
        anns = list(doc.annotations.values())
        anns[0].status, anns[0].text = "confirmed", "dān"
        anns[1].status, anns[1].text = "hidden", "xíng"
        before, raw = copy.deepcopy(document_to_json(doc)), self.book.path.read_bytes()
        ed = Editor(doc, {}, character_book=self.book, pinyin_words_factory=self.project.pinyin_words)
        ed.edit_characters(lambda b: b.update_character(ch.id, names={"zh-Hans": NameForm(given="单行", given_reading="dān xíng")}))
        self.assertEqual([a.text for a in doc.annotations.values()], ["dān", "xíng", "dān", "xíng"])
        self.assertEqual(len(ed.history.done), 1)
        ed.history.undo()
        restored = document_to_json(doc)
        self.assertGreater(restored.pop("revision"), before.pop("revision"))
        self.assertEqual(restored, before)
        self.assertEqual(self.book.path.read_bytes(), raw)

    def test_export_formats_per_language_raw_label_and_cache(self):
        ch = self.soma_character()
        ms = [member(l, "text", lang=l, cid=ch.id, label="RAW") for l in ("ja", "zh-Hans", "en", "fr")]
        ms.append(member("unknown", "text", lang="en", label="UNLINKED"))
        doc, cache = document(*ms), ExportCache()
        expected = {"short": ["幸平", "创真", "Soma", "幸平", "UNLINKED"],
                    "full": ["幸平創真", "幸平创真", "Soma Yukihira", "幸平創真", "UNLINKED"],
                    "family": ["幸平", "幸平", "Yukihira", "幸平", "UNLINKED"],
                    "given": ["創真", "创真", "Soma", "創真", "UNLINKED"],
                    "label": ["RAW"] * 4 + ["UNLINKED"]}
        for fmt, names in expected.items():
            opts = ExportOptions(ruby=False, name_format=fmt)
            out, _ = build_ass(doc, {}, self.project.template(), self.root, opts, cache=cache, character_book=self.book)
            self.assertEqual([e.name for e in out.events], names)
            self.assertEqual(build_ass(doc, {}, self.project.template(), self.root, opts, cache=cache,
                                       character_book=self.book)[1].laid_out, 0)
        self.book.update_character(ch.id, names={"ja": NameForm("幸平", "創真", short="custom", short_custom="ソーマ")})
        out, _ = build_ass(doc, {}, self.project.template(), self.root, ExportOptions(ruby=False),
                           cache=cache, character_book=self.book)
        self.assertEqual(out.events[0].name, "ソーマ")
        md, jl = build_packet(doc, episode="E1", chars=self.book)
        self.assertEqual(json.loads(jl.splitlines()[0])["original"][0]["speaker"]["name"], "幸平創真")
        self.assertIn("幸平創真", md)
        with self.assertRaises(ValueError):
            ExportOptions(name_format="invalid")

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires real ffmpeg/libass")
    @legacy_ruby_layout()
    def test_soma_fresh_merge_label_bytes_and_default_size(self):
        source = ROOT / "test_projects/soma"
        hashes = lambda: {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source.rglob("*") if p.is_file()}
        before = hashes()
        target = self.root / "soma"
        shutil.copytree(source, target, copy_function=shutil.copyfile)
        project = Project.open(target)
        # Keep the historical fingerprint as a byte-exact legacy-layout check.
        project.data.setdefault("settings", {}).setdefault("ruby", {})["max_width"] = {
            "dialogue": {"enabled": False}, "song": {"enabled": False}}
        ep = project.episode(project.episode_ids()[0])
        count = itertools.count(1)
        with patch("subkit.core.merge.new_id", side_effect=lambda p: p + format(next(count), "020x")), \
             patch("subkit.core.readings.new_id", side_effect=lambda p: p + format(next(count), "020x")):
            ep.merge()
        sizes = {}
        for fmt in ("label", "short"):
            ass, _ = ep.build_export(name_format=fmt)
            path = self.root / (fmt + ".ass")
            ass.save(path)
            raw = path.read_bytes()
            sizes[fmt] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.assertEqual(sizes["label"], {"bytes": 140505, "sha256": "9dfc755c010b327f817a47dea1b8818ab4f2f731b73ca7e107df9f9a838abdcd"})  # 0.20.0: song translation lines one layer above the original (was 0.10.8)
        self.assertEqual(hashes(), before)
        print("D1c soma fresh merge: " + json.dumps(sizes) + "; original_unchanged=True")


if __name__ == "__main__":
    unittest.main()
