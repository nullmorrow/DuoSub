"""Whole selection and explicit bilingual ASS tag scopes through both Qt editors."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QRect
from PySide6.QtGui import QTextCursor

from subkit.core.textformat import group_raw_changes
from subkit.model.document import Group, Member, Run
from subkit_app.views.format_bar import format_selection
from subkit_app.watch import InlineTextEditor


class FormatScope(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    group = fixtures.ConversionUI.group

    def _dispose(self):
        if hasattr(self, "session"):
            self.session.dirty = False
        fixtures.ConversionUI._dispose(self)

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.cn = self.group("g", text="整个世界")
        self.jp = self.group("j", text="世界征服")
        self.doc.groups.pop()
        self.jp.group, self.jp.lane, self.jp.lang, self.jp.style = "g", "original", "ja", "JP"
        self.doc.groups[0].members.append(self.jp.id)
        self.doc.structure_changed()
        self.window.select_group("g")
        self.editor = self.window.editor
        self.edit = self.editor.cn_edits[0][1]
        self.editor._set_fmt_target(self.cn.id, self.edit)

    def select_chars(self, a, b):
        c = self.edit.textCursor()
        c.setPosition(a)
        c.setPosition(b, QTextCursor.MoveMode.KeepAnchor)
        self.edit.setTextCursor(c)

    def raw(self, scope="auto", text=r"\fad(0,720)"):
        bar = self.editor.fmt
        bar.raw_scope.setCurrentIndex(bar.raw_scope.findData(scope))
        bar.raw.setText(text)
        bar._apply_raw()

    def test_full_selection_is_whole_line(self):
        self.select_chars(0, 4)
        self.assertEqual(self.editor.fmt.target.text(), "整行")
        self.raw()
        self.assertEqual(self.cn.overrides["raw"], r"\fad(0,720)")
        self.assertEqual(self.cn.format, [])
        self.assertEqual(self.jp.overrides, {})
        self.assertIsNone(format_selection("世界", (0, 2)))

    def test_partial_selection_rejected_until_explicit_line_scope(self):
        self.select_chars(0, 2)
        with patch("PySide6.QtWidgets.QToolTip.showText"):
            self.raw()
        self.assertNotIn("raw", self.cn.overrides)
        self.assertIn("整组", self.editor.fmt.raw.toolTip())
        self.raw("line")
        self.assertEqual(self.cn.overrides["raw"], r"\fad(0,720)")
        self.assertEqual(self.cn.format, [])
        self.assertEqual(self.jp.overrides, {})

    def test_group_preserves_other_tags_and_undoes_once(self):
        self.cn.overrides["raw"] = r"\bord3\fade(255,0,255,0,100,800,1000)"
        self.jp.overrides["raw"] = r"\frz5\fad(0,100)"
        before = copy.deepcopy(self.doc)
        self.select_chars(0, 2)
        self.raw("group")
        self.assertEqual(self.cn.overrides["raw"], r"\bord3\fad(0,720)")
        self.assertEqual(self.jp.overrides["raw"], r"\frz5\fad(0,720)")
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members[self.cn.id].overrides, before.members[self.cn.id].overrides)
        # History restores snapshots, so look up the current member on redo.
        self.assertEqual(self.doc.members[self.jp.id].overrides, before.members[self.jp.id].overrides)
        self.session.editor.history.redo()
        self.assertEqual(self.doc.members[self.cn.id].overrides["raw"], r"\bord3\fad(0,720)")

    def test_group_only_targets_languages_of_current_group(self):
        comment = Member("comment", "g", "ja", "original", "comment", 1000, 2000,
                         runs=[Run("note", "text", ins="note", a=0, b=4)])
        self.doc.members[comment.id] = comment
        self.doc.groups[0].members.append(comment.id)
        other = self.group("other")
        changes = group_raw_changes(self.doc, self.cn.id, r"\fad(0,720)")
        self.assertEqual(set(changes), {self.cn.id, self.jp.id})
        self.assertEqual(other.overrides, {})
        with self.assertRaises(ValueError):
            group_raw_changes(self.doc, self.cn.id, r"\pos(10,10)")

    def test_group_clear_restores_raw_only(self):
        self.cn.overrides.update(raw=r"\fad(0,720)", fs=70, pos=[100, 200])
        self.jp.overrides["raw"] = r"\frz2\fad(0,720)"
        self.raw("group", "")
        self.assertEqual(self.cn.overrides, {"fs": 70, "pos": [100, 200]})
        self.assertEqual(self.jp.overrides, {})

    def test_inherited_fade_is_explained_without_becoming_manual(self):
        self.cn.overrides["raw"] = r"\fad(0,720)"
        self.editor._set_fmt_target(self.jp.id, self.editor.jp_edit)
        self.assertFalse(self.editor.fmt.fade_hint.isHidden())
        self.assertIn(r"\fad(0,720)", self.editor.fmt.fade_hint.text())
        self.assertEqual(self.jp.overrides, {})
        self.jp.overrides["raw"] = r"\fad(0,0)"
        self.editor._show_fmt()
        self.assertTrue(self.editor.fmt.fade_hint.isHidden())

    def test_inline_same_full_selection_and_group_scope(self):
        inline = InlineTextEditor(self.doc, [self.cn.id, self.jp.id], self.jp.id,
                                  QRect(0, 0, 500, 200), 28, parent=self.window)
        self.addCleanup(inline.cancel)
        inline.formatRequested.connect(self.window._on_format)
        inline.groupFormatRequested.connect(self.window._on_group_format)
        inline.edit.selectAll()
        self.assertEqual(inline.fmt.target.text(), "整行")
        inline.fmt.raw.setText(r"\fad(0,720)")
        inline.fmt._apply_raw()
        self.assertEqual(self.jp.overrides["raw"], r"\fad(0,720)")
        self.assertEqual(self.jp.format, [])
        inline.fmt.raw_scope.setCurrentIndex(inline.fmt.raw_scope.findData("group"))
        inline.fmt.raw.setText(r"\fad(0,360)")
        inline.fmt._apply_raw()
        self.assertEqual(self.doc.members[self.cn.id].overrides["raw"], r"\fad(0,360)")
        self.assertEqual(self.doc.members[self.jp.id].overrides["raw"], r"\fad(0,360)")


if __name__ == "__main__":
    unittest.main()
