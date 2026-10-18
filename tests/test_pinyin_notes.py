"""Pinyin data and editing, without rendering or real player interaction."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit.core.ops import Editor
from subkit.core.pinyin_notes import annotate_pinyin, normalize_pinyin, suggest_pinyin
from subkit.core.readings import anchor_positions, visible_map
from subkit.core.review import build_packet
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json


def member(text="行长银行了长大", mid="m_1", *, lang="zh-Hans", lane="translation"):
    return Member(mid, "g_1", lang, lane, "dialogue", 1000, 3000,
                  runs=[Run(text, ins="i_" + mid, b=len(text))])


class PinyinNotes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Project.create(Path(self.tmp.name) / "project", "Pinyin")
        self.m = member()
        self.other = member("行长", "m_2", lane="original")
        self.ja = member("銀行", "m_3", lang="ja", lane="original")
        self.doc = Document("E1", groups=[Group("g_1", members=[self.m.id, self.other.id, self.ja.id])],
                            members={m.id: m for m in (self.m, self.other, self.ja)})
        self.ed = Editor(self.doc, {})

    def anns(self, mid="m_1"):
        ids = visible_map(self.doc.members[mid])[1]
        return sorted((a for a in self.doc.annotations.values() if a.member == mid and a.kind == "pinyin"),
                      key=lambda a: anchor_positions(a, ids)[0])

    def test_sentence_context_and_candidates(self):
        anns = suggest_pinyin(self.m)
        self.assertEqual([a.text for a in anns], ["háng", "zhǎng", "yín", "háng", "le", "zhǎng", "dà"])
        self.assertTrue(all(a.status == "auto" and a.origin == "analyzer" for a in anns))
        self.assertIn("xíng", anns[0].candidates)
        self.assertIn("cháng", anns[1].candidates)
        self.assertIn("liǎo", anns[4].candidates)

    def test_traditional_context(self):
        anns = suggest_pinyin(member("行長銀行了長大"))
        self.assertEqual([a.text for a in anns], ["háng", "zhǎng", "yín", "háng", "le", "zhǎng", "dà"])

    def test_mixed_text_anchors_and_language(self):
        m = member(r"{\b1}银 A12，行！\N长大。")
        anns = suggest_pinyin(m)
        self.assertEqual([a.base for a in anns], list("银行长大"))
        vt, ids = visible_map(m)
        self.assertEqual([vt.text[anchor_positions(a, ids)[0]] for a in anns], list("银行长大"))
        self.assertEqual(suggest_pinyin(self.ja), [])
        self.assertEqual(len(suggest_pinyin(self.other)), 2)

    def test_normalization(self):
        for text, expected in [("hang2", "háng"), ("háng", "háng"), ("lv4", "lǜ"),
                               ("lü4", "lǜ"), ("lǜ", "lǜ"), ("lv5", "lü"), ("lv", "lü"),
                               ("le5", "le"), ("le", "le"), ("LVE4", "lüè"),
                               ("shui3", "shuǐ"), ("liu2", "liú"), ("ha\u0301ng", "háng")]:
            with self.subTest(text=text):
                self.assertEqual(normalize_pinyin(text), expected)

    def test_reject_invalid_before_mutation(self):
        for text in ["", "xyz", "hang0", "hang6", "hang22", "háng2", "hángà", "hello", "han g", "hàng!"]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "拼音无效"):
                normalize_pinyin(text)
        before = document_to_json(self.doc)
        with self.assertRaisesRegex(ValueError, "每个汉字"):
            self.ed.add_annotation(self.m.id, 0, 2, "hang2", kind="pinyin")
        self.assertEqual(document_to_json(self.doc), before)
        self.assertFalse(self.ed.history.done)

    def test_keep_confirmed_hidden_and_manual(self):
        annotate_pinyin(self.doc, self.m)
        anns = self.anns()
        self.ed.set_reading(anns[0].id, "xing2")
        self.ed.hide_annotation(anns[1].id)
        anns[2].origin = "exception"
        saved = copy.deepcopy([self.doc.annotations[a.id] for a in anns[:3]])
        annotate_pinyin(self.doc, self.m, {"行长": "háng cháng"})
        self.assertEqual([self.doc.annotations[a.id] for a in saved], saved)
        self.assertEqual(len(self.anns()), len(self.m.text))

    def test_mode_generation_off_and_undo(self):
        self.assertEqual(self.project.pinyin_mode, "off")
        n = self.ed.set_pinyin_mode(self.project, "PY")
        self.assertEqual(n, 9)
        self.assertEqual(len(self.ed.history.done), 1)
        self.assertTrue(self.ed.history.done[0].keeps_text)
        self.assertEqual(Project.open(self.project.root).pinyin_mode, "PY")
        saved = copy.deepcopy(self.doc.annotations)
        self.ed.history.undo()
        self.assertFalse(self.doc.annotations)
        self.assertEqual(Project.open(self.project.root).pinyin_mode, "off")
        self.ed.history.redo()
        self.assertEqual(self.doc.annotations, saved)
        self.ed.set_reading(self.anns()[0].id, "xing2")
        saved = copy.deepcopy(self.doc.annotations)
        self.ed.set_pinyin_mode(self.project, "off")
        self.assertEqual(self.doc.annotations, saved)
        self.ed.set_pinyin_mode(self.project, "ZY")
        self.assertEqual(self.anns()[0].text, "xíng")

    def test_here_scope_per_character_and_undo(self):
        self.ed.reannotate_pinyin(self.doc.members)
        before = copy.deepcopy(self.doc.annotations)
        count = len(self.ed.history.done)
        self.ed.apply_pinyin_rule(self.m.id, 0, 2, "xing2 chang2")
        self.assertEqual([a.text for a in self.anns()[:2]], ["xíng", "cháng"])
        self.assertTrue(all(a.status == "confirmed" for a in self.anns()[:2]))
        self.assertEqual([a.text for a in self.anns("m_2")], ["háng", "zhǎng"])
        self.assertEqual(len(self.ed.history.done), count + 1)
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations, before)

    def test_episode_rule_preview_and_undo(self):
        self.ed.reannotate_pinyin(self.doc.members)
        self.ed.set_reading(self.anns()[0].id, "hang2")
        before = copy.deepcopy(self.doc.annotations)
        preview = self.ed.preview_pinyin_rule("行长", "xing2 chang2")
        self.assertEqual(len(preview), 2)
        self.assertEqual(preview[0]["kept"], [0])
        self.assertEqual(len(preview[0]["changes"]), 1)
        count = len(self.ed.history.done)
        self.ed.apply_pinyin_rule(self.m.id, 0, 2, "xing2 chang2", scope="episode")
        self.assertEqual(self.doc.rules, [{"kind": "pinyin", "surface": "行长", "reading": "xíng cháng"}])
        self.assertEqual([a.text for a in self.anns()[:2]], ["háng", "cháng"])
        self.assertEqual([a.text for a in self.anns("m_2")], ["xíng", "cháng"])
        self.assertEqual(len(self.ed.history.done), count + 1)
        self.ed.history.undo()
        self.assertEqual(self.doc.rules, [])
        self.assertEqual(self.doc.annotations, before)
        self.ed.history.redo()
        self.assertEqual(self.anns("m_2")[0].text, "xíng")

    def test_project_rule_file_undo_and_existing_bytes(self):
        self.ed.apply_pinyin_rule(self.m.id, 0, 2, "xing2 chang2", scope="project", project=self.project)
        data = json.loads(self.project.pinyin_path.read_text(encoding="utf-8"))
        self.assertEqual(data, {"format": "subkit-pinyin", "format_version": 1,
                                "words": {"行长": "xíng cháng"}})
        self.ed.history.undo()
        self.assertFalse(self.project.pinyin_path.exists())
        self.ed.history.redo()
        before = self.project.pinyin_path.read_bytes()
        anns = copy.deepcopy(self.doc.annotations)
        self.ed.set_pinyin_rule("行长", "hang2 zhang3", project=self.project)
        self.ed.history.undo()
        self.assertEqual(self.project.pinyin_path.read_bytes(), before)
        self.assertEqual(self.doc.annotations, anns)

    def test_episode_overrides_project_longest_word(self):
        self.project.set_pinyin_word("行长", "xing2 chang2")
        self.doc.rules.append({"kind": "pinyin", "surface": "行长", "reading": "háng zhǎng"})
        self.ed.set_pinyin_mode(self.project, "WG")
        self.assertEqual([a.text for a in self.anns()[:2]], ["háng", "zhǎng"])
        self.assertEqual([a.text for a in suggest_pinyin(self.other, {"行": "xíng", "行长": "háng zhǎng"})],
                         ["háng", "zhǎng"])

    def test_review_on_polyphones_and_confirmed_off_unchanged(self):
        self.m.runs = [Run("行你好长", ins="i_review", b=4)]
        before = build_packet(self.doc, episode="E1")
        annotate_pinyin(self.doc, self.m)
        anns = self.anns()
        anns[1].candidates = [anns[1].text]
        anns[2].candidates = [anns[2].text]
        self.ed.set_reading(anns[2].id, "hao3")
        self.ed.hide_annotation(anns[3].id)
        self.assertEqual(build_packet(self.doc, episode="E1"), before)
        md, jl = build_packet(self.doc, episode="E1", pinyin_mode="PY")
        self.assertIn("行(xíng)你好(hǎo)长", md)
        rec = json.loads(jl.splitlines()[0])
        self.assertEqual(rec["translation"][0]["annotated"], "行(xíng)你好(hǎo)长")
        annotate_pinyin(self.doc, self.other)
        rec = json.loads(build_packet(self.doc, episode="E1", pinyin_mode="ZY")[1].splitlines()[0])
        self.assertIn("行(háng)", rec["original"][0]["annotated"])

    def test_serialization_preserves_choices_and_rules(self):
        self.ed.set_pinyin_rule("行长", "xing2 chang2")
        self.ed.set_reading(self.anns()[0].id, "hang2")
        self.ed.hide_annotation(self.anns()[1].id)
        data = document_to_json(self.doc)
        self.assertEqual(document_to_json(document_from_json(data)), data)

    def test_session_changed_chinese_only_refresh_and_undo(self):
        from PySide6.QtWidgets import QApplication
        from subkit_app.session import EpisodeSession

        QApplication.instance() or QApplication([])
        ep = self.project.add_episode("E1")
        ep.document = self.doc
        session = EpisodeSession(self.project, ep)
        self.addCleanup(session.close)
        self.ed = session.editor
        with patch.object(session, "request_preview"), patch.object(session, "_reannotate"):
            self.ed.set_pinyin_mode(self.project, "PY")
            self.ed.set_reading(self.anns()[0].id, "xing2")
            untouched = copy.deepcopy(self.anns("m_2"))
            self.ed.insert_text(self.m.id, 2, "长大")
            self.assertEqual(len(self.anns()), 9)
            self.assertEqual(self.anns()[0].text, "xíng")
            self.assertEqual(self.anns("m_2"), untouched)
            self.ed.history.undo()
            self.assertEqual(len(self.anns()), 7)
            self.assertEqual(self.anns()[0].text, "xíng")
            self.ed.history.redo()
            self.assertEqual(len(self.anns()), 9)
            self.ed.set_pinyin_mode(self.project, "off")
            saved = copy.deepcopy(self.doc.annotations)
            self.ed.insert_text(self.m.id, 0, "你")
            self.assertEqual(self.doc.annotations, saved)


if __name__ == "__main__":
    unittest.main()
