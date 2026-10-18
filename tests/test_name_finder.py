"""Synthetic name analysis: actual Sudachi, Latin heuristics and paired suggestions."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core.name_finder import find_names
from subkit.core.people import CharacterBook, NameForm
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_to_json


def document(rows):
    doc = Document("E1")
    for i, row in enumerate(rows):
        group = Group(f"g_{i}")
        doc.groups.append(group)
        for j, (lang, text) in enumerate(row):
            mid = f"m_{i}_{j}"
            doc.members[mid] = Member(mid, group.id, lang, "original" if j == 0 else "translation",
                                     "dialogue", i * 1000, i * 1000 + 900,
                                     runs=[Run(text, ins="i_" + mid, b=len(text))])
            group.members.append(mid)
    return doc


class NameFinderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="d1d-names-")
        self.addCleanup(tmp.cleanup)
        self.book = CharacterBook(Path(tmp.name) / "characters.json")

    def find(self, doc, languages=None):
        return find_names(doc, {}, self.book, languages or {"original": "ja", "translation": "zh-Hans"})

    def test_sudachi_surname_given_full_kana_counts_examples_read_only(self):
        doc = document([[('ja', '山田太郎と山田太郎。')]] * 4)
        before = copy.deepcopy(document_to_json(doc))
        out = {row["text"]: row for row in self.find(doc)}
        self.assertEqual((out["山田"]["kind"], out["太郎"]["kind"], out["山田太郎"]["kind"]),
                         ("family", "given", "person"))
        self.assertEqual(out["山田太郎"]["reading"], "やまだたろう")
        self.assertEqual(out["山田太郎"]["count"], 8)
        self.assertEqual(len(out["山田太郎"]["examples"]), 3)
        self.assertEqual(out["山田太郎"]["examples"][0]["group"], "g_0")
        self.assertEqual(out["山田太郎"]["episodes"], ["E1"])
        self.assertEqual(document_to_json(doc), before)
        self.assertFalse(self.book.path.exists())

    def test_book_names_parts_short_alias_and_rejected_labels_excluded(self):
        self.book.add_character({"ja": NameForm(family="山田", given="太郎", short="custom", short_custom="タロウ")},
                                aliases=["花子"])
        self.book.mark_label("佐藤", crowd=True)
        self.book.mark_label("次郎", crowd=False)
        doc = document([[('ja', '山田太郎と花子と佐藤と次郎。')]])
        self.assertFalse(self.find(doc))

    def test_not_name_persisted_idempotent_and_not_a_speaker_decision(self):
        self.book.mark_not_name("太郎")
        self.book.mark_not_name("太郎")
        self.book.extra["future"] = "保留"
        self.book.save()
        loaded = CharacterBook.load(self.book.path)
        self.assertEqual(loaded.not_name, ["太郎"])
        self.assertEqual(loaded.extra["future"], "保留")
        self.assertEqual(loaded.not_person, [])
        self.assertNotIn("太郎", [r["text"] for r in self.find(document([[('ja', '太郎です。')]]))])
        self.assertFalse(self.book.path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_english_mid_sentence_repeat_boundaries_common_words_singletons(self):
        doc = document([[('en-US', 'Alice saw Bob. Then Alice met Bob and Charlie. We said Yes.')],
                        [('en', 'Bob knows Alice! "Alice left." She called Bob and Yes.')]])
        out = {row["text"]: row for row in self.find(doc)}
        self.assertEqual(set(out), {"Bob", "Alice"})
        self.assertEqual((out["Bob"]["count"], out["Alice"]["count"]), (3, 2))
        self.assertIsNone(out["Bob"]["reading"])
        self.assertEqual(out["Bob"]["lang"], "en")

    def test_accented_latin_hyphen_and_unsupported_scripts(self):
        doc = document([[('fr', 'Je vois Élodie et Jean-Luc.')],
                        [('fr', 'Tu connais Élodie et Jean-Luc.')],
                        [('ru', 'Я вижу Борис. Я вижу Борис.')],
                        [('ko', '나는 철수 철수.')]])
        self.assertEqual({r["text"] for r in self.find(doc)}, {"Élodie", "Jean-Luc"})

    def test_paired_chinese_recurrence_rarity_and_no_mutation(self):
        doc = document([[('ja', '太郎は帰る。'), ('zh-Hans', '太郎回家。')],
                        [('ja', '太郎が来た。'), ('zh-Hant', '看見太郎。')],
                        [('ja', 'おはよう。'), ('zh-Hans', '回家再说。')]])
        row = next(r for r in self.find(doc) if r["text"] == "太郎")
        self.assertEqual(row["suggested_translation"], "太郎")

    def test_weak_ambiguous_and_common_chinese_are_omitted(self):
        cases = [
            [[('ja', '太郎。'), ('zh-Hans', '太郎回家。')]],
            [[('ja', '太郎。'), ('zh-Hans', '太郎、花子。')]] * 2,
            [[('ja', '太郎。'), ('zh-Hans', '你好。')]] * 2 +
            [[('ja', 'ありがとう。'), ('zh-Hans', '你好。')]] * 3,
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                self.assertNotIn("suggested_translation", self.find(document(rows))[0])

    def test_ass_breaks_do_not_join_names_or_make_line_initial_latin_names(self):
        doc = document([[('ja', r'{\b1}山田\N太郎')],
                        [('en', r'I saw Alice\NBob saw Alice')],
                        [('en', r'I saw Alice\NBob saw Alice')]])
        out = {r["text"] for r in self.find(doc)}
        self.assertNotIn("山田\n太郎", out)
        self.assertNotIn("Bob", out)
        self.assertIn("Alice", out)

    def test_hidden_comment_sign_and_source_comment_members_not_scanned(self):
        doc = document([[('ja', '太郎。')]] * 3)
        doc.members["m_0_0"].role = "comment"
        doc.members["m_1_0"].role = "sign"
        doc.members["m_2_0"].source_kind = "Comment"
        self.assertEqual(self.find(doc), [])


if __name__ == "__main__":
    unittest.main()
