"""Screen-text edits through real Qt double-click/Enter/close and group editors."""
from __future__ import annotations

import copy
import unittest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from tests import test_watch_inline_qol as fixtures
from subkit.ass.text import line_tag
from subkit.core.export import ExportOptions, build_ass
from subkit.core.readings import visible_map
from subkit.model.document import Run


class SignEditingUI(unittest.TestCase):
    setUpClass = fixtures.WatchInlineQoL.__dict__["setUpClass"]
    _dispose = fixtures.WatchInlineQoL._dispose
    group = fixtures.WatchInlineQoL.group
    close_inline = fixtures.WatchInlineQoL.close_inline
    key = fixtures.WatchInlineQoL.key

    def setUp(self):
        fixtures.WatchInlineQoL.setUp(self)
        self.cn.role = "sign"
        self.doc.group("g").kind = "sign"
        self.prefix = r"{\an7\pos(321.5,123.25)\c&HD0A6D7&\fs28\fad(0,720)\frz3}"
        self.control = Run(self.prefix, "tag", ins="control", b=len(self.prefix))
        self.cn.runs = [self.control, Run("智力", ins="word", b=2)]
        self.cn.layer = 3
        self.cn.margins = [7, 8, 9]
        self.cn.effect = "screen-label"
        self.before = copy.deepcopy(self.cn)

    def double_click(self):
        mapper = self.watch._mapper()
        x, y = mapper.to_osd(150, 125)
        self.watch._double(QPointF(x, y), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.processEvents()
        self.assertIsNotNone(self.window._inline)
        self.assertEqual(self.window._inline.edit.text(), "智力")
        return self.window._inline

    def assert_preserved(self):
        member = self.doc.members[self.cn.id]
        self.assertEqual(member.runs[0], self.control)
        self.assertEqual(visible_map(member)[0].text, "力量")
        for attr in ("style", "role", "layer", "start", "end", "margins", "effect", "overrides"):
            self.assertEqual(getattr(member, attr), getattr(self.before, attr), attr)
        ass, report = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False))
        event = next(e for e, o in zip(ass.events, report.owners) if o["member"] == member.id)
        self.assertEqual(line_tag(event.text, "pos").float_args(), [321.5, 123.25])
        self.assertEqual(line_tag(event.text, "an").num(), 7)
        self.assertEqual(line_tag(event.text, "fad").float_args(), [0, 720])
        self.assertEqual(event.layer, 3)

    def test_double_click_enter_keeps_screen_position_and_properties(self):
        editor = self.double_click()
        editor.edit.setText("力量")
        self.key(Qt.Key.Key_Return, target=editor.edit)
        self.assert_preserved()

    def test_double_click_close_commits_with_screen_controls(self):
        editor = self.double_click()
        editor.edit.setText("力量")
        editor.close()
        self.app.processEvents()
        self.assert_preserved()

    def test_double_click_escape_preserves_original_and_no_history(self):
        editor = self.double_click()
        editor.edit.setText("力量")
        self.key(Qt.Key.Key_Escape, target=editor.edit)
        self.assertEqual(self.doc.members[self.cn.id], self.before)
        self.assertEqual(self.session.editor.history.done, [])

    def test_double_click_change_undo_and_redo_restore_complete_runs(self):
        editor = self.double_click()
        editor.edit.setText("力量")
        self.key(Qt.Key.Key_Return, target=editor.edit)
        self.assert_preserved()
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members[self.cn.id], self.before)
        self.session.editor.history.redo()
        self.assert_preserved()

    def test_normal_group_editor_commits_sign_with_hidden_ass_prefix(self):
        panel = self.window.editor
        panel.set_group(self.doc, self.doc.group("g"), None, None)
        edit = next(e for _member, e in panel.cn_edits if e.property("member_id") == self.cn.id)
        self.assertEqual(edit.toPlainText(), "智力")
        edit.setPlainText("力量")
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.app.processEvents()
        self.assert_preserved()


if __name__ == "__main__":
    unittest.main()
