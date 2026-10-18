"""Several translation lines on one original line: one group per line at import (MergeSettings.joint)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.core.merge import MergeSettings, joint_cuts, split_joint_groups
from subkit.core.readings import visible_map
from subkit.model.document import Document, Group, Member, Run
from subkit.model.validate import validate


def joint_doc(jp="で どんな奴や 自分の狙とるその子供っちゅうのは",
              cn=(("然後? 被妳看上的那孩子", 612550, 614530), ("是個怎樣的人", 614610, 616980))):
    doc = Document("E3")
    mids = []
    for i, (text, a, b) in enumerate(cn):
        mid = f"cn{i}"
        doc.members[mid] = Member(mid, "g", "zh-Hant", "translation", "dialogue", a, b, layer=1, style="CN",
                                  runs=[Run(text, src="t1", e=i, a=0, b=len(text), ins="i_" + mid)])
        mids.append(mid)
    doc.members["jp"] = Member("jp", "g", "ja", "original", "dialogue", cn[0][1], cn[-1][2], style="JP",
                               runs=[Run(jp, src="o1", e=0, a=0, b=len(jp), ins="i_jp")])
    doc.groups.append(Group("g", "dialogue", [*mids, "jp"], [], "auto",
                            {"by": "aligner", "method": "subkit-align-v2", "block": "joint", "doubts": ["joint_rows"]}))
    doc.groups.append(Group("next", "dialogue", ["n"]))
    doc.members["n"] = Member("n", "next", "ja", "original", "dialogue", 617260, 619260, style="JP",
                              runs=[Run("強くはないわ", src="o1", e=1, a=0, b=6, ins="i_n")])
    doc.structure_changed()
    return doc


def rows(doc):
    return [[(doc.members[m].lane, visible_map(doc.members[m])[0].text, doc.members[m].start, doc.members[m].end)
             for m in g.members] for g in doc.groups]


class JointSplit(unittest.TestCase):
    def test_default_first_line_keeps_the_whole_original(self):
        self.assertEqual(MergeSettings().joint, "first")
        doc = joint_doc()
        self.assertEqual(split_joint_groups(doc), 1)
        self.assertEqual(rows(doc), [
            [("translation", "然後? 被妳看上的那孩子", 612550, 614530),
             ("original", "で どんな奴や 自分の狙とるその子供っちゅうのは", 612550, 614530)],
            [("translation", "是個怎樣的人", 614610, 616980)],
            [("original", "強くはないわ", 617260, 619260)]])
        self.assertIn("close_boundary", doc.groups[0].origin["doubts"])       # 边界拿不准: the original needs a look
        self.assertNotIn("close_boundary", doc.groups[1].origin["doubts"])    # 只有翻译 already says it
        self.assertEqual(doc.groups[1].origin["by"], "aligner")
        self.assertFalse([i for i in validate(doc, {}).errors if i.code != "run.unknown_source"])  # fixture has no sources

    def test_cut_at_sentence_breaks_follows_line_durations(self):
        doc = joint_doc()
        split_joint_groups(doc, "cut")
        self.assertEqual([[r[1] for r in g] for g in rows(doc)[:2]],
                         [["然後? 被妳看上的那孩子", "で どんな奴や"], ["是個怎樣的人", "自分の狙とるその子供っちゅうのは"]])
        self.assertEqual(rows(doc)[1][1][2:], (614610, 616980))              # the cut follows the second line
        self.assertTrue(all("close_boundary" in g.origin["doubts"] for g in doc.groups[:2]))
        # no break in the original: kept whole with the first line, as in the default
        doc = joint_doc(jp="でどんな奴や自分の狙とるその子供っちゅうのは")
        split_joint_groups(doc, "cut")
        self.assertEqual([len(g.members) for g in doc.groups], [2, 1, 1])

    def test_three_lines_and_keep(self):
        cn = (("一", 0, 1000), ("二", 1000, 2000), ("三", 2000, 3000))
        doc = joint_doc(jp="あいう、かきく、さしす", cn=cn)
        self.assertEqual(split_joint_groups(doc, "cut"), 2)
        self.assertEqual([[r[1] for r in g] for g in rows(doc)[:3]],
                         [["一", "あいう、"], ["二", "かきく、"], ["三", "さしす"]])
        doc = joint_doc(cn=cn)
        self.assertEqual(split_joint_groups(doc, "keep"), 0)
        self.assertEqual(len(doc.groups[0].members), 4)
        with self.assertRaises(ValueError):
            split_joint_groups(doc, "other")

    def test_cut_positions(self):
        self.assertEqual(joint_cuts("あいう かきくけこさしすせそ", [0, 1000], 5000), [4])
        self.assertEqual(joint_cuts("あいう かきくけこさしすせそ", [0, 4000], 5000), [4])   # only one break
        self.assertEqual(joint_cuts("ab cd ef", [0, 1000, 2000], 3000), [3, 6])
        self.assertIsNone(joint_cuts("abcdef", [0, 1000], 3000))
        self.assertIsNone(joint_cuts("ab cd", [0, 1000, 2000], 3000))


if __name__ == "__main__":
    unittest.main()


class JointDialog(unittest.TestCase):
    def test_merge_dialog_offers_the_three_ways_independently_of_fillers(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from subkit_app.views.dialogs import MergeDialog
        app = QApplication.instance() or QApplication([])
        d = MergeDialog(filler_mode="compact")
        self.assertEqual(d.settings(), {"filler_mode": "compact", "semantic": False, "joint": "first"})
        d.joint["cut"].setChecked(True)
        d.keep.setChecked(True)
        self.assertEqual(d.settings()["joint"], "cut")                       # the two radio sets do not interfere
        self.assertEqual(d.settings()["filler_mode"], "keep")
        self.assertEqual(MergeDialog(joint="keep").settings()["joint"], "keep")
        d.deleteLater()
        self.assertIsNotNone(app)
