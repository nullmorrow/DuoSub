"""Deleted provenance remains deleted through rebuilding and persistence."""
from __future__ import annotations

import copy
import unittest

from tests import test_stage0 as stage0
from subkit.core.merge import MergeSettings, merge_episode
from subkit.core.ops import Editor
from subkit.model.serialize import document_from_json, document_to_json
from subkit.model.validate import validate


class RemergeTombstones(unittest.TestCase):
    setUp = stage0.Stage0.setUp
    tearDown = stage0.Stage0.tearDown
    merged = stage0.Stage0.merged

    def rebuild(self, previous, mode="keep"):
        before = copy.deepcopy(document_to_json(previous))
        result = merge_episode("E1", self.cn, self.ja, MergeSettings(filler_mode=mode), previous=previous)
        doc = result.document
        self.assertEqual(document_to_json(previous), before)
        self.assertEqual(doc.tombstones, previous.tombstones)
        self.assertIsNot(doc.tombstones, previous.tombstones)
        report = validate(doc, self.sources)
        self.assertTrue(report.ok, [(i.code, i.message) for i in report.errors])
        for stone in doc.tombstones:
            for obj in [*doc.members.values(), *doc.comments.values()]:
                self.assertFalse(any(r.src == stone.src and r.e == stone.e and
                                     max(r.a, stone.a) < min(r.b, stone.b) for r in obj.runs))
        return doc

    def test_partial_deletion_keeps_remaining_characters_and_roundtrip(self):
        doc = self.merged()
        member = next(m for m in doc.members.values() if "こんにちは" in m.text)
        Editor(doc, self.sources).delete_range(member.id, 1, 3)
        rebuilt = self.rebuild(document_from_json(document_to_json(doc)))
        self.assertIn("こちは", [m.text for m in rebuilt.members.values()])
        self.assertEqual(document_to_json(document_from_json(document_to_json(rebuilt))), document_to_json(rebuilt))

    def test_deleted_group_is_not_recreated(self):
        doc = self.merged()
        group = next(g for g in doc.groups if any(m.text == "你好啊" for m in doc.members_of(g)))
        editor = Editor(doc, self.sources)
        for mid in list(group.members):
            editor.delete_member(mid)
        rebuilt = self.rebuild(doc)
        self.assertFalse(any("你好啊" in m.text or "こんにちは" in m.text for m in rebuilt.members.values()))
        self.assertLess(len(rebuilt.groups), len(self.merged().groups))

    def test_sign_and_translation_deletions(self):
        doc = self.merged()
        editor = Editor(doc, self.sources)
        sign = next(m for m in doc.members.values() if "招牌" in m.text)
        editor.delete_member(sign.id)
        cn = next(m for m in doc.members.values() if m.text == "你好啊")
        editor.delete_member(cn.id)
        rebuilt = self.rebuild(doc)
        self.assertFalse(any("招牌" in m.text or "你好啊" in m.text for m in rebuilt.members.values()))
        self.assertTrue(any("こんにちは" in m.text for m in rebuilt.members.values()))

    def test_deleted_filler_does_not_return_as_hidden_comment(self):
        doc = self.merged()
        filler = next(m for m in doc.members.values() if m.text == "あっ…")
        Editor(doc, self.sources).delete_member(filler.id)
        rebuilt = self.rebuild(doc, "compact")
        self.assertFalse(any("あっ…" in c.content for c in rebuilt.comments.values()))

    def test_deleted_source_space_stays_deleted_through_later_edits(self):
        from subkit.model.document import Document, Group, Member, Run
        from subkit.model.source import Source, SourceEvent
        src = Source("t", "zh-Hant", "translation", "ass", "unused", "unused", "hash", 9, "utf-8", "lf",
                     events=[SourceEvent(0, "Dialogue", 100, 1000, "不行 萬萬不可")])
        doc = Document("E1")
        doc.members["m"] = Member("m", "g", "zh-Hant", "translation", "dialogue", 100, 1000,
                                  runs=[Run("不行 萬萬不可", src="t", e=0, b=7)])
        doc.groups.append(Group("g", members=["m"]))
        doc.structure_changed()
        editor = Editor(doc, {"t": src})
        self.assertTrue(editor.replace_editor_text("m", "不行萬萬不可"))   # E005: the space used to come back
        self.assertEqual(doc.members["m"].text, "不行萬萬不可")
        self.assertEqual([(t.a, t.b, t.text) for t in doc.tombstones], [(2, 3, " ")])
        editor.replace_editor_text("m", "不行萬萬不可！")                  # normalising again keeps it deleted
        self.assertEqual(doc.members["m"].text, "不行萬萬不可！")
        editor.history.undo()
        editor.history.undo()
        self.assertEqual(doc.members["m"].text, "不行 萬萬不可")
        self.assertFalse(doc.tombstones)

    def test_undo_deletion_allows_remerge_to_restore_source(self):
        doc = self.merged()
        member = next(m for m in doc.members.values() if "こんにちは" in m.text)
        editor = Editor(doc, self.sources)
        editor.delete_member(member.id)
        editor.history.undo()
        rebuilt = self.rebuild(doc)
        self.assertTrue(any("こんにちは" in m.text for m in rebuilt.members.values()))
