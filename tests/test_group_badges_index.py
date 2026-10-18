"""Reading badges and the needs-attention filter share a revision index."""
from __future__ import annotations

import unittest

from PySide6.QtCore import Qt
from tests import test_convert_ui as fixtures
from subkit.core.ops import Editor
from subkit.model.document import Annotation, Comment, Document, Group, Member, Run
from subkit_app.views.group_table import C_TAGS, GroupFilter, GroupTableModel, group_badges


class CountingAnnotations(dict):
    scans = 0

    def values(self):
        self.scans += 1
        return super().values()


def document(size=4):
    doc = Document("badges-index")
    for i in range(size):
        gid = str(i)
        mids = [gid + lane for lane in ("original", "translation")]
        doc.groups.append(Group(gid, members=mids))
        for mid, lane in zip(mids, ("original", "translation")):
            doc.members[mid] = Member(mid, gid, "ja", lane, "dialogue", 1000, 2000,
                                      runs=[Run("字", ins=mid, a=0, b=1)])
    doc.annotations = CountingAnnotations()
    return doc


def annotation(doc, mid, status="uncertain", kind="ruby"):
    aid = str(len(doc.annotations))
    doc.annotations[aid] = Annotation(aid, kind, mid, [{"ins": mid, "a": 0, "b": 1}],
                                      "字", "じ", status=status)
    return aid


class GroupBadgesIndex(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]

    def setUp(self):
        self.doc = document()
        self.model = GroupTableModel(self.doc)
        self.proxy = GroupFilter()
        self.proxy.setSourceModel(self.model)
        self.addCleanup(self.model.deleteLater)
        self.addCleanup(self.proxy.deleteLater)

    def selected(self):
        self.proxy.set_mode("需要处理")
        return [self.proxy.data(self.proxy.index(i, 0), Qt.ItemDataRole.UserRole)
                for i in range(self.proxy.rowCount())]

    def test_badge_text_order_and_all_annotation_kinds_match_legacy_counts(self):
        g = self.doc.groups[0]
        for kind in ("ruby", "pinyin", "gloss"):
            annotation(self.doc, g.members[0], kind=kind)
        annotation(self.doc, g.members[1])
        for status in ("auto", "confirmed", "hidden"):
            annotation(self.doc, g.members[0], status)
        annotation(self.doc, "orphan")
        annotation(self.doc, self.doc.groups[1].members[0])
        g.origin = {"doubts": ["close_boundary", "reanchored"]}
        g.flags = ["order_exception"]
        self.doc.comments["hidden"] = Comment("hidden", g.id, "hidden_filler")
        g.comments = ["hidden"]
        expected = [("边界拿不准", "warn"), ("预告重对齐", "info"),
                    ("含 1 处未渲染语气词", "warn"), ("读音?×4", "warn"), ("顺序例外", "warn")]
        self.assertEqual(group_badges(self.doc, g), expected)
        self.assertEqual(self.model._row(g)[4], expected)
        self.assertEqual(self.model.data(self.model.index(0, C_TAGS)), "  ".join(b for b, _ in expected))

    def test_filter_scans_annotations_once_per_revision_for_many_groups(self):
        doc = document(200)
        for g in doc.groups:
            for _ in range(5):
                annotation(doc, g.members[0])
        model = GroupTableModel(doc)
        proxy = GroupFilter()
        self.addCleanup(model.deleteLater)
        self.addCleanup(proxy.deleteLater)
        proxy.setSourceModel(model)
        proxy.set_mode("需要处理")
        self.assertEqual(proxy.rowCount(), 200)
        for g in doc.groups:
            self.assertEqual(group_badges(doc, g), [("读音?×5", "warn")])
            self.assertEqual(model._row(g)[4], [("读音?×5", "warn")])
        self.assertEqual(doc.annotations.scans, 1)
        model.reset()
        self.assertEqual(proxy.rowCount(), 200)
        self.assertEqual(doc.annotations.scans, 1)
        doc.revision += 1
        model.reset()
        self.assertEqual(proxy.rowCount(), 200)
        self.assertEqual(doc.annotations.scans, 2)

    def test_confirm_reading_undo_redo_and_filter_refresh(self):
        aid = annotation(self.doc, self.doc.groups[0].members[0])
        editor = Editor(self.doc, {})
        self.assertEqual(self.selected(), ["0"])
        editor.history.run("confirm reading", lambda tx: setattr(tx.annotation(aid), "status", "confirmed"))
        self.model.refresh_groups({"0"})
        self.assertEqual(self.selected(), [])
        editor.history.undo()
        self.model.reset()
        self.assertEqual(self.selected(), ["0"])
        editor.history.redo()
        self.model.reset()
        self.assertEqual(self.selected(), [])

    def test_empty_index_does_not_rescan_and_insert_remove_rebuilds(self):
        self.assertEqual(self.selected(), [])
        for g in self.doc.groups:
            self.model._row(g)
        self.assertEqual(self.doc.annotations.scans, 1)
        aid = annotation(self.doc, self.doc.groups[2].members[0])
        self.doc.revision += 1
        self.model.reset()
        self.assertEqual(self.selected(), ["2"])
        del self.doc.annotations[aid]
        self.doc.revision += 1
        self.model.reset()
        self.assertEqual(self.selected(), [])
        self.assertEqual(self.doc.annotations.scans, 3)

    def test_members_reassigned_and_annotation_moved_use_current_group_members(self):
        mid = self.doc.groups[0].members[0]
        aid = annotation(self.doc, mid)
        self.assertEqual(self.selected(), ["0"])
        self.doc.groups[0].members.remove(mid)
        self.doc.groups[1].members.append(mid)
        self.doc.members[mid].group = "1"
        self.doc.revision += 1
        self.model.reset()
        self.assertNotIn(("读音?×1", "warn"), self.model._row(self.doc.groups[0])[4])
        self.assertIn(("读音?×1", "warn"), self.model._row(self.doc.groups[1])[4])
        self.doc.annotations[aid].member = self.doc.groups[2].members[0]
        self.doc.revision += 1
        self.model.reset()
        self.assertNotIn(("读音?×1", "warn"), self.model._row(self.doc.groups[1])[4])
        self.assertIn(("读音?×1", "warn"), self.model._row(self.doc.groups[2])[4])

    def test_confirmed_and_non_dialogue_groups_still_show_badges_but_are_filtered(self):
        for g in self.doc.groups:
            annotation(self.doc, g.members[0])
        self.doc.groups[1].state = "confirmed"
        self.doc.groups[2].kind = "song"
        self.doc.groups[3].kind = "sign"
        self.assertEqual(self.selected(), ["0"])
        for g in self.doc.groups:
            self.assertIn(("读音?×1", "warn"), self.model._row(g)[4])

    def test_duplicate_member_reference_does_not_double_count_readings(self):
        g = self.doc.groups[0]
        annotation(self.doc, g.members[0])
        g.members.append(g.members[0])
        self.assertEqual(group_badges(self.doc, g), [("读音?×1", "warn")])

    def test_real_split_merge_and_undo_rebuild_member_counts(self):
        g = self.doc.groups[0]
        mid = g.members[0]
        self.doc.members[mid].runs = [Run("字字", ins=mid, a=0, b=2)]
        first = annotation(self.doc, mid)
        second = annotation(self.doc, mid)
        self.doc.annotations[second].anchor = [{"ins": mid, "a": 1, "b": 2}]
        editor = Editor(self.doc, {})
        self.assertIn(("读音?×2", "warn"), self.model._row(g)[4])
        right = editor.split(g.id, jp_at=1)
        self.model.reset()
        for gid in (g.id, right):
            self.assertIn(("读音?×1", "warn"), self.model._row(self.doc.group(gid))[4])
        self.assertNotEqual(self.doc.annotations[first].member, self.doc.annotations[second].member)
        editor.merge_with_next(g.id)
        self.model.reset()
        self.assertIn(("读音?×2", "warn"), self.model._row(self.doc.group(g.id))[4])
        editor.history.undo()
        self.model.reset()
        for gid in (g.id, right):
            self.assertIn(("读音?×1", "warn"), self.model._row(self.doc.group(gid))[4])
        editor.history.undo()
        self.model.reset()
        self.assertIn(("读音?×2", "warn"), self.model._row(self.doc.group(g.id))[4])
