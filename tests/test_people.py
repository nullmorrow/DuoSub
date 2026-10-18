"""Character-book persistence, exact linking and project/document undo."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core.cc import analyze_event
from subkit.core.merge import merge_episode
from subkit.core.ops import Editor
from subkit.core.people import CharacterBook, NameForm, preserve_user_speakers, suggest_character
from subkit.core.sdh_en import analyze_event as analyze_en
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json
from subkit.model.source import Source, SourceEvent
from subkit.reading.characters import CharacterTable


def document(labels, episode="E1"):
    doc = Document(episode, groups=[Group("g_1")])
    for i, label in enumerate(labels):
        mid = f"m_{i}"
        text = f"例句{i}"
        doc.members[mid] = Member(mid, "g_1", "ja", "original", "dialogue", i * 1000, i * 1000 + 900,
                                 speaker={"character": None, "label": label} if label is not None else None,
                                 runs=[Run(text, src="o1", e=i, a=0, b=len(text))])
        doc.groups[0].members.append(mid)
    return doc


def sources():
    out = []
    for sid, lang, lane, text, analyzer in [
            ("o1", "ja", "original", "(太郎)こんにちは", analyze_event),
            ("t1", "en", "translation", "[TARO] hello", analyze_en)]:
        event = SourceEvent(0, "Dialogue", 1000, 3000, text)
        src = Source(sid, lang, lane, "srt", "", "", "", 0, "utf-8", "lf",
                     events=[event], spans=analyzer(0, text, "srt").spans, roles={0: "dialogue"})
        out.append(src)
    return out[1], out[0]


class PeopleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.book = CharacterBook.load(self.root / "dict" / "characters.json")

    def test_suggest_unique_parts_any_language_alias_and_no_substring_guess(self):
        ch = self.book.add_character({"ja": NameForm(family="幸平", given="創真"),
                                      "en": NameForm(family="Cooper", given="Sheldon")}, aliases=["ソーマ"])
        before = self.book.path.read_bytes()
        for label in ("幸平", "創真", "Cooper", "Sheldon", "ソーマ", "創真・回想"):
            self.assertEqual(suggest_character(self.book, label), ch.id)
        for label in ("", "Unknown", "創", "ソー", "幸平創真"):
            self.assertIsNone(suggest_character(self.book, label))
        self.assertIsNone(self.book.resolve("創真"))
        self.assertEqual(self.book.path.read_bytes(), before)

    def test_suggest_katakana_and_middle_dot_components_ambiguous_none(self):
        ch = self.book.add_character({"ja": NameForm(given="アルベール・ド・モルセール子爵"),
                                      "zh-Hans": NameForm(given="阿尔贝·德·莫尔塞夫")}, aliases=["アルベール"])
        self.assertEqual(suggest_character(self.book, "アルベール"), ch.id)
        self.assertEqual(suggest_character(self.book, "阿尔贝"), ch.id)
        self.assertIsNone(suggest_character(self.book, "モルセール"))
        self.book.add_character({"en": NameForm(given="Albert")}, aliases=["アルベール"])
        self.assertIsNone(suggest_character(self.book, "アルベール"))
        self.book.add_character({"zh-Hans": NameForm(given="阿尔贝")})
        self.assertIsNone(suggest_character(self.book, "阿尔贝"))

    def test_tsv_conversion_once_preserves_all_fields_and_bytes(self):
        tsv = self.root / "characters.tsv"
        tsv.write_bytes(("\ufeff# 表\r\n幸平 創真\tゆきひら そうま\t幸平创真\t創真／ソーマ\tわたし\t少年\r\n"
                         "城一郎\tじょういちろう\t城一郎\t父\tわたくし\t父親\r\n").encode("utf-8"))
        before = tsv.read_bytes()
        self.book = CharacterBook.load(self.book.path, tsv=tsv, translation="zh-Hans")
        ch = self.book.characters[0]
        self.assertEqual(ch.full_name("ja"), "幸平創真")
        self.assertEqual(ch.full_name("zh-Hans"), "幸平创真")
        self.assertEqual((ch.names["ja"].family_reading, ch.names["ja"].given_reading), ("ゆきひら", "そうま"))
        self.assertEqual((ch.name_parts, ch.kana_parts), (["幸平", "創真"], ["ゆきひら", "そうま"]))
        self.assertEqual(ch.aliases, ["創真", "ソーマ"])
        self.assertEqual(ch.self_refs, [{"surface": "私", "reading": "わたし", "scope": {}}])
        self.assertEqual(ch.persona["note_ja"], "少年")
        self.assertEqual(tsv.read_bytes(), before)
        cid = ch.id
        tsv.write_text("別人\tべつじん", encoding="utf-8")
        again = CharacterBook.load(self.book.path, tsv=tsv)
        self.assertEqual(again.characters[0].id, cid)
        self.assertEqual(again.to_json(), self.book.to_json())

    def test_soma_conversion_keeps_name_reading_parts(self):
        original = ROOT / "test_projects" / "soma" / "dict" / "characters.tsv"
        self.assertTrue(original.exists())
        target = self.root / "characters.tsv"
        shutil.copyfile(original, target)
        old = CharacterTable.load(target)
        book = CharacterBook.load(self.book.path, tsv=target)
        self.assertEqual(len(book.characters), len(old.chars))
        for a, b in zip(old.chars, book.characters):
            self.assertEqual((a.name, a.name_parts, a.kana_parts, a.aliases, a.note),
                             (b.full_name("ja"), b.name_parts, b.kana_parts, b.aliases, b.persona["note_ja"]))
            self.assertEqual(a.self_reading, b.self_refs[0]["reading"] if b.self_refs else "")
        self.assertEqual(target.read_bytes(), original.read_bytes())

    def test_utf8_round_trip_scopes_persona_style_and_unknown_fields(self):
        ch = self.book.add_character({"ja": "太郎", "en": "Taro"}, aliases=["タロウ"], style="人物JP",
                                     self_refs=[{"surface": "私", "reading": "わたくし", "scope": s} for s in
                                                [{}, {"season": "S01"}, {"episodes": ["E1"]}, {"line": "m_1"}]],
                                     persona={"self": "俺", "tone": "丁寧", "note_tr": "备注"})
        ch.extra["future"] = {"value": "保留"}
        self.book.extra["future_book"] = True
        self.book.save()
        self.assertFalse(self.book.path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertIn("太郎", self.book.path.read_text(encoding="utf-8"))
        self.assertEqual(CharacterBook.load(self.book.path).to_json(), self.book.to_json())

    def test_exact_match_normalisation_no_substring_spaces_or_guess(self):
        ch = self.book.add_character({"ja": "幸平創真", "zh-Hans": "幸平创真", "en": "Soma"}, aliases=["創真"])
        for label in ["幸平創真", "幸平创真", "Soma", "創真", "創真・心の声", "創真・回想"]:
            self.assertEqual(self.book.resolve(label), ch.id)
        for label in ["幸平", "幸平 創真", "soma", " 創真", "創眞", "創真・城一郎"]:
            self.assertIsNone(self.book.resolve(label))
        self.book.add_character({"ja": "別人"}, aliases=["創真"])
        self.assertIsNone(self.book.resolve("創真"))

    def test_pending_counts_examples_episodes_relink_idempotent(self):
        doc = document(["未知"] * 5 + ["別人"])
        self.book.link_document(doc)
        p = next(p for p in self.book.pending if p["label"] == "未知")
        self.assertEqual((p["count"], p["examples"], p["episodes"]), (5, ["例句0", "例句1", "例句2"], ["E1"]))
        self.book = CharacterBook.load(self.book.path)
        self.book.link_document(doc)
        self.book.link_document(document(["未知"] * 2, "E2"))
        p = next(p for p in self.book.pending if p["label"] == "未知")
        self.assertEqual((p["count"], p["episodes"]), (7, ["E1", "E2"]))
        self.book.link_document(document([], "E1"))
        self.assertEqual([(p["label"], p["count"], p["episodes"]) for p in self.book.pending], [("未知", 2, ["E2"])])

    def test_crowd_not_person_and_pending_resolution_are_explicit(self):
        doc = document(["生徒A", "効果音", "太郎", "生徒A・回想"])
        self.book.link_document(doc)
        self.assertEqual(len(self.book.pending), 4)
        self.book.mark_label("生徒A", crowd=True)
        self.book.mark_label("効果音", crowd=False)
        ch = self.book.add_character({"ja": "山田"})
        self.book.resolve_pending("太郎", ch.id)
        self.book.link_document(doc)
        self.assertEqual([m.speaker["character"] for m in doc.members.values()], ["crowd", None, ch.id, "crowd"])
        self.assertEqual(self.book.pending, [])
        reloaded = CharacterBook.load(self.book.path)
        self.assertEqual((reloaded.crowd, reloaded.not_person), (["生徒A"], ["効果音"]))
        self.assertEqual([m.speaker["label"] for m in doc.members.values()], ["生徒A", "効果音", "太郎", "生徒A・回想"])

    def test_merge_links_both_languages_and_preserves_user_pick(self):
        ch = self.book.add_character({"ja": "太郎", "en": "TARO"})
        other = self.book.add_character({"ja": "花子"})
        cn, ja = sources()
        doc = merge_episode("E1", cn, ja, character_book=self.book).document
        self.assertEqual({m.speaker["character"] for m in doc.members.values()}, {ch.id})
        original = next(m for m in doc.members.values() if m.lane == "original")
        Editor(doc, {}).set_speaker([original.id], other.id, whole_group=False)
        rebuilt = merge_episode("E1", cn, ja, character_book=self.book, previous=doc).document
        m = next(m for m in rebuilt.members.values() if m.lane == "original")
        self.assertEqual(m.speaker, {"character": other.id, "label": "太郎", "by": "user"})
        self.assertNotEqual(m.id, original.id)
        self.book.link_document(rebuilt)
        self.assertEqual(m.speaker["character"], other.id)

    def test_conflicting_or_unmatched_user_picks_block_remerge(self):
        old = document(["甲", "乙"])
        old.members["m_0"].runs = [Run("あ", src="o1", e=0, a=0, b=1)]
        old.members["m_1"].runs = [Run("い", src="o1", e=0, a=1, b=2)]
        ed = Editor(old, {})
        ed.set_speaker(["m_0"], "chr_a", whole_group=False)
        ed.set_speaker(["m_1"], "chr_b", whole_group=False)
        rebuilt = document(["甲"])
        rebuilt.members["m_0"].runs = [Run("あい", src="o1", e=0, a=0, b=2)]
        before = document_to_json(old)
        with self.assertRaisesRegex(RuntimeError, "不同的人工说话人"):
            preserve_user_speakers(old, rebuilt)
        self.assertEqual(document_to_json(old), before)
        old.members["m_1"].runs = [Run("手改", ins="i_1", a=0, b=2)]
        with self.assertRaisesRegex(RuntimeError, "无法保留"):
            preserve_user_speakers(old, rebuilt)

    def test_set_speaker_per_line_whole_group_round_trip_undo_redo(self):
        doc = document(["甲", "乙", None])
        before = copy.deepcopy(document_to_json(doc))
        ed = Editor(doc, {})
        self.assertEqual(ed.set_speaker(["m_0"], "crowd", whole_group=False), 1)
        self.assertIsNone(doc.members["m_1"].speaker["character"])
        ed.history.undo()
        self.assertEqual(document_to_json(doc)["members"], before["members"])
        self.assertEqual(ed.set_speaker(["m_0", "m_0"], None, whole_group=True), 3)
        self.assertEqual(len(ed.history.done), 1)
        self.assertEqual([m.speaker["label"] for m in doc.members.values()], ["甲", "乙", ""])
        restored = document_from_json(document_to_json(doc))
        self.assertTrue(all(m.speaker["by"] == "user" for m in restored.members.values()))
        ed.history.undo()
        self.assertIsNone(doc.members["m_2"].speaker)
        ed.history.redo()
        self.assertEqual(doc.members["m_2"].speaker, {"label": "", "character": None, "by": "user"})

    def test_relink_skips_user_picks_and_undo_restores_pending_and_file(self):
        doc = document(["太郎", "花子", "未知"])
        ed = Editor(doc, {}, character_book=self.book)
        ed.set_speaker(["m_0"], None, whole_group=False)
        ch = self.book.add_character({"ja": "太郎"}, aliases=["花子"])
        raw = self.book.path.read_bytes()
        self.assertEqual(ed.relink_speakers(), 1)
        self.assertIsNone(doc.members["m_0"].speaker["character"])
        self.assertEqual(doc.members["m_1"].speaker["character"], ch.id)
        self.assertEqual([p["label"] for p in self.book.pending], ["未知"])
        ed.history.undo()
        self.assertEqual(self.book.path.read_bytes(), raw)
        self.assertEqual(self.book.pending, [])
        self.assertIsNone(doc.members["m_1"].speaker["character"])

    def test_merge_characters_moves_links_including_user_and_undo_exact_file(self):
        a = self.book.add_character({"ja": "太郎", "en": "Taro"}, aliases=["タロウ"],
                                    persona={"note_ja": "原注"}, self_refs=[{"surface": "俺", "reading": "おれ", "scope": {}}])
        b = self.book.add_character({"ja": "山田"}, aliases=["ヤマダ"])
        doc = document(["太郎", "タロウ", "山田"])
        self.book.link_document(doc)
        ed = Editor(doc, {}, character_book=self.book)
        ed.set_speaker(["m_1"], a.id, whole_group=False)
        self.book.path.write_text(json.dumps(self.book.to_json(), ensure_ascii=False, indent=3) + "\n", encoding="utf-8")
        before_file = self.book.path.read_bytes()
        before_book = self.book.to_json()
        before_members = copy.deepcopy(document_to_json(doc)["members"])
        ed.merge_characters(a.id, b.id)
        self.assertEqual(len(ed.history.done), 2)
        self.assertTrue(all(m.speaker["character"] == b.id for m in doc.members.values()))
        self.assertEqual(doc.members["m_1"].speaker["by"], "user")
        self.assertEqual(self.book.character(b.id).persona["note_ja"], "原注")
        self.assertEqual(self.book.resolve("Taro"), b.id)
        after = self.book.to_json()
        ed.history.undo()
        self.assertEqual(self.book.path.read_bytes(), before_file)
        self.assertEqual(self.book.to_json(), before_book)
        self.assertEqual(document_to_json(doc)["members"], before_members)
        ed.history.redo()
        self.assertEqual(self.book.to_json(), after)
        self.assertTrue(all(m.speaker["character"] == b.id for m in doc.members.values()))

    def test_delete_character_and_undo_preserve_user_label(self):
        ch = self.book.add_character({"ja": "太郎"})
        doc = document(["太郎"])
        ed = Editor(doc, {}, character_book=self.book)
        ed.set_speaker(["m_0"], ch.id, whole_group=False)
        self.book.delete_character(ch.id, editor=ed)
        self.assertEqual(doc.members["m_0"].speaker, {"character": None, "label": "太郎", "by": "user"})
        ed.history.undo()
        self.assertEqual(doc.members["m_0"].speaker["character"], ch.id)
        new = self.book.add_character({"ja": "別人"})
        self.assertNotEqual(new.id, ch.id)

    def test_failed_book_edit_rolls_back_and_missing_file_undo_removes_json(self):
        doc = document(["太郎"])
        ed = Editor(doc, {}, character_book=self.book)
        def fail(book):
            book.add_character({"ja": "太郎"})
            raise ValueError("cancelled")
        with self.assertRaisesRegex(ValueError, "cancelled"):
            ed.edit_characters(fail)
        self.assertFalse(self.book.path.exists())
        self.assertEqual(self.book.characters, [])
        self.assertEqual(ed.history.done, [])
        ed.edit_characters(lambda book: book.add_character({"ja": "太郎"}))
        self.assertTrue(self.book.path.exists())
        ed.history.undo()
        self.assertFalse(self.book.path.exists())
        self.assertIsNone(doc.members["m_0"].speaker["character"])

    def test_project_custom_json_path_conversion_and_reanalysis_editor(self):
        project = Project.create(self.root / "project", "人物", translation="en")
        project.data["dictionaries"]["characters_json"] = "dict/custom.json"
        tsv = project.root / "dict" / "characters.tsv"
        tsv.write_text("太郎\tたろう\tTaro\tタロウ\tわたし\t备注\n", encoding="utf-8")
        project.save()
        reopened = Project.open(project.root)
        book = reopened.character_book()
        self.assertEqual(book.path, project.root / "dict" / "custom.json")
        self.assertEqual(book.characters[0].full_name("en"), "Taro")
        ep = reopened.add_episode("E1")
        cn, ja = sources()
        ep.sources = {cn.id: cn, ja.id: ja}
        ep.document = document(["太郎"])
        ed = Editor(ep.document, ep.sources)
        ep.reanalyze_sources(editor=ed)
        self.assertEqual(ep.document.members["m_0"].speaker["character"], book.characters[0].id)
        self.assertEqual(len(ed.history.done), 1)
        ed.history.undo()
        self.assertIsNone(ep.document.members["m_0"].speaker["character"])

    def test_reject_unknown_format_or_duplicate_ids(self):
        for data in [{"format": "other", "format_version": 1},
                     {"format": "subkit-characters", "format_version": 3},
                     {"format": "subkit-characters", "format_version": 1,
                      "characters": [{"id": "chr_same"}, {"id": "chr_same"}]}]:
            self.book.path.parent.mkdir(parents=True, exist_ok=True)
            self.book.path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                CharacterBook.load(self.book.path)


if __name__ == "__main__":
    unittest.main()
